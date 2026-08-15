"""
Retrieval controller service.

Implements an optional iterative retrieval loop that is guided by an LLM "router":
- retrieve from enabled sources (memory / expertise / graph), applying novelty masking
- ask the router whether to retrieve again (and with what query), reflect, or stop
- assemble a final `Context` once from the accumulated items and (optionally) session history

This module is intentionally inference-agnostic: callers provide an `ILLMProvider` and may choose
their own models for routing vs final answering.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple

from ctxforge.compaction.topology_view import TopologyAwareRenderer
from ctxforge.config.base import EngineConfig, RetrievalControllerConfig
from ctxforge.core.context import Context
from ctxforge.core.expertise import ExpertiseItem
from ctxforge.core.memory import MemoryItem
from ctxforge.core.session import Session
from ctxforge.engine.services.assembly_service import AssemblyService
from ctxforge.engine.services.expertise_service import ExpertiseService
from ctxforge.engine.services.graph_service import GraphService
from ctxforge.engine.services.memory_service import MemoryService
from ctxforge.extraction.utils import extract_json_from_text
from ctxforge.graph.retrieval.types import BridgeConnection, GraphRetrievalResult, ReasoningPath
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.retrieval.fast_path_retriever import FastPathRetriever

Decision = Literal["retrieve", "reflect", "answer"]


@dataclass(frozen=True)
class RetrievalControllerDecision:
    decision: Decision
    evidence: List[str]
    gaps: Optional[str]
    retrieval_query: Optional[str]
    draft_answer: Optional[str]
    reasoning: Optional[str]


@dataclass(frozen=True)
class RetrievalControllerIteration:
    iteration: int
    query: str
    decision: Decision
    new_memory_count: int
    new_graph_edge_count: int
    new_graph_node_count: int
    new_graph_evidence_count: int
    new_expertise_count: int
    router_input_tokens: int
    router_output_tokens: int
    router_latency_ms: float
    elapsed_ms: float
    stop_reason: Optional[str] = None
    sub_queries: Tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalControllerResult:
    context: Context
    iterations: List[RetrievalControllerIteration]
    evidence: List[str]
    gaps: Optional[str]
    used_memory_ids: List[str]
    used_graph_ids: Dict[str, List[str]]
    used_expertise_item_ids: List[str]


_ROUTER_SYSTEM_PROMPT = """You are a retrieval controller for an assistant.
You decide whether to retrieve more evidence, reflect, or answer now.

Return ONLY a strict JSON object with this schema:
- decision: one of ["retrieve","reflect","answer"]
- evidence: array of concise factual bullets grounded in the retrieved materials
- gaps: string describing what is missing to answer confidently, or "None" if nothing is missing
- coverage_percentage: float 0.0-1.0 estimating what fraction of the question's information needs are met

Only include these conditional fields:
- retrieval_query: only when decision == "retrieve" (a short standalone search query, 5-15 tokens)
- gap_queries: only when decision == "reflect" (array of 1-3 targeted search queries to fill specific gaps)
- reasoning: only when decision == "reflect" (think briefly about evidence/gaps and what to retrieve next)
- draft_answer: only when decision == "answer" (answer using current evidence; no speculation)

Rules:
- evidence must NOT mention missing info (gaps handles missing info)
- if you choose retrieve, ensure retrieval_query is different from the previous query when possible
- when reflecting, generate gap_queries that target the specific missing information
- be conservative: choose answer only when coverage_percentage >= 0.7 and gaps is "None"
"""


_PLANNER_SYSTEM_PROMPT = """You are a query planner for a retrieval system.
Given a user question, decompose it into 1-4 focused sub-queries that together cover all information needs.

Rules:
- If the question is simple and targets a single entity/fact, return just the original question.
- If the question involves multiple entities, comparisons, or multi-part requests, decompose into focused sub-queries.
- Each sub-query should be a short, standalone search query (5-15 tokens).
- Return ONLY a JSON object: {"sub_queries": ["query1", "query2", ...]}
"""


def _clip(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3].rstrip() + "..."


def _normalize_gaps(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s.lower() == "none":
            return None
        return s
    s = str(value).strip()
    if not s or s.lower() == "none":
        return None
    return s


def _normalize_evidence(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    s = str(value).strip()
    return [s] if s else []


def _parse_router_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    json_str = extract_json_from_text(raw) or raw
    return json.loads(json_str)


def _format_router_retrieval_block(
    *,
    question: str,
    last_query: Optional[str],
    evidence: List[str],
    gaps: Optional[str],
    memories: Sequence[MemoryItem],
    graph_facts: Sequence[str],
    graph_entities: Sequence[str],
    graph_evidence: Sequence[str],
    expertise_items: Sequence[ExpertiseItem],
    max_snippet_chars: int = 320,
) -> str:
    lines: List[str] = []
    lines.append("# Question")
    lines.append(question)
    lines.append("")

    lines.append("# Prior Query")
    lines.append(last_query or "None")
    lines.append("")

    lines.append("# Evidence")
    if evidence:
        lines.extend([f"- {e}" for e in evidence[:50]])
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("# Gaps")
    lines.append(gaps or "None")
    lines.append("")

    lines.append("# Retrieved Materials (most recent iteration)")
    if memories:
        lines.append("\n## Memories")
        for m in memories[:25]:
            mtype = getattr(getattr(m, "type", None), "value", None) or str(getattr(m, "type", "memory"))
            lines.append(f"- [{mtype}] ({m.memory_id}) {_clip(m.content, max_snippet_chars)}")
    else:
        lines.append("\n## Memories\n- (none)")

    if graph_facts or graph_entities or graph_evidence:
        lines.append("\n## Graph")
        if graph_facts:
            lines.append("<FACTS>")
            for f in graph_facts[:25]:
                lines.append(f"- {_clip(f, max_snippet_chars)}")
            lines.append("</FACTS>")
        if graph_entities:
            lines.append("<ENTITIES>")
            for ent in graph_entities[:25]:
                lines.append(f"- {_clip(ent, max_snippet_chars)}")
            lines.append("</ENTITIES>")
        if graph_evidence:
            lines.append("<EVIDENCE>")
            for ev in graph_evidence[:25]:
                lines.append(f"- {_clip(ev, max_snippet_chars)}")
            lines.append("</EVIDENCE>")
    else:
        lines.append("\n## Graph\n- (none)")

    if expertise_items:
        lines.append("\n## Expertise")
        for item in expertise_items[:25]:
            lines.append(f"- ({item.item_id}) {_clip(item.content, max_snippet_chars)}")
    else:
        lines.append("\n## Expertise\n- (none)")

    return "\n".join(lines).strip() + "\n"


class RetrievalControllerService:
    """
    Optional iterative retrieval controller.

    The controller is inference-agnostic by default; it runs only when explicitly invoked
    (or when a caller routes to it) and requires a provided `ILLMProvider`.
    """

    def __init__(
        self,
        *,
        config: EngineConfig,
        memory_service: MemoryService,
        graph_service: Optional[GraphService],
        expertise_service: ExpertiseService,
        assembly_service: AssemblyService,
        fast_path_retriever: Optional[FastPathRetriever] = None,
    ):
        self._cfg = config
        self._rcfg: RetrievalControllerConfig = config.retrieval_controller
        self._memory = memory_service
        self._graph = graph_service
        self._expertise = expertise_service
        self._assembly = assembly_service
        self._fast_path_retriever = fast_path_retriever

    async def _plan_queries(
        self,
        question: str,
        llm: ILLMProvider,
        model: Optional[str] = None,
    ) -> List[str]:
        """Decompose a complex question into focused sub-queries.

        Returns a list of 1-N sub-queries.  For simple questions the list
        contains only the original question.
        """
        if not self._rcfg.enable_query_planning:
            return [question]

        planner_model = self._rcfg.router_model or model or llm.default_model
        try:
            resp = await llm.chat(
                [
                    ChatMessage(role="system", content=_PLANNER_SYSTEM_PROMPT),
                    ChatMessage(role="user", content=question),
                ],
                model=planner_model,
                temperature=0.0,
                max_tokens=200,
            )
            data = _parse_router_json(resp.content)
            raw = data.get("sub_queries", [])
            if isinstance(raw, list) and raw:
                queries = [str(q).strip() for q in raw if str(q).strip()]
                if queries:
                    return queries[: self._rcfg.max_sub_queries]
        except Exception:
            pass
        return [question]

    async def prepare_context(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        system_instructions: Optional[str] = None,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        expertise_id: Optional[str] = None,
        llm: ILLMProvider,
        model: Optional[str] = None,
    ) -> RetrievalControllerResult:
        """
        Build a `Context` for answering `question` using an iterative, LLM-guided retrieval loop.

        High-level flow:
        - Repeatedly retrieve new candidate items from enabled sources (memory / expertise / graph).
        - Ask a small "router" LLM call whether to retrieve again (and with what query), reflect, or stop.
        - Assemble the final context once from all accumulated items + optional session history.

        Notes / invariants:
        - This function does **not** produce the final answer text; it produces context + a trace.
        - "Novelty masking" (when enabled) prevents feeding the same item IDs back to the router
          across iterations, which helps avoid loops and wasted budget.
        - Time budget and max LLM calls are hard-stops; when exceeded, we return the best context
          we have so far, along with a stop_reason in the iteration trace.
        """
        if not self._rcfg.enabled:
            raise ValueError("Retrieval controller is disabled. Enable config.retrieval_controller.enabled to use it.")

        start = time.time()
        time_budget_ms = int(self._rcfg.time_budget_ms or 0)

        def _time_exceeded() -> bool:
            # Soft real-time budget for the whole controller run (not per-iteration).
            if time_budget_ms <= 0:
                return False
            return (time.time() - start) * 1000.0 >= float(time_budget_ms)

        # Track novelty across iterations. These sets are used to suppress already-seen
        # items (by stable IDs) when `novelty_masking` is enabled.
        seen_memory_ids: Set[str] = set()
        seen_expertise_ids: Set[str] = set()
        seen_graph_edge_ids: Set[str] = set()
        seen_graph_node_ids: Set[str] = set()
        seen_graph_evidence_ids: Set[str] = set()

        # Accumulators used for final context. We append "new" items each iteration;
        # at the end we assemble the context once from these accumulated lists.
        acc_memories: List[MemoryItem] = []
        acc_expertise: List[ExpertiseItem] = []
        acc_graph_facts: List[str] = []
        acc_graph_entities: List[str] = []
        acc_graph_evidence: List[str] = []
        acc_reasoning_paths: List[ReasoningPath] = []
        acc_bridge_connections: List[BridgeConnection] = []
        acc_graph_results: List[GraphRetrievalResult] = []

        # Router "working state" across iterations. The router can update evidence/gaps
        # and propose new retrieval queries to run next.
        evidence: List[str] = []
        gaps: Optional[str] = None
        retrieval_queries: List[str] = []

        iterations: List[RetrievalControllerIteration] = []
        llm_calls = 0

        # Resolve expertise id defaulting: callers can pass an expertise_id, otherwise
        # we fall back to the controller default (if any).
        resolved_expertise_id = expertise_id or self._rcfg.default_expertise_id

        # Query planning: decompose complex questions into sub-queries
        planned_sub_queries = await self._plan_queries(question, llm, model)
        if len(planned_sub_queries) > 1:
            llm_calls += 1

        for i in range(1, int(self._rcfg.max_iterations) + 1):
            if _time_exceeded():
                # Hard stop: return context assembled from whatever we managed to retrieve so far.
                iterations.append(
                    RetrievalControllerIteration(
                        iteration=i,
                        query=retrieval_queries[-1] if retrieval_queries else question,
                        decision="answer",
                        new_memory_count=0,
                        new_graph_edge_count=0,
                        new_graph_node_count=0,
                        new_graph_evidence_count=0,
                        new_expertise_count=0,
                        router_input_tokens=0,
                        router_output_tokens=0,
                        router_latency_ms=0.0,
                        elapsed_ms=(time.time() - start) * 1000.0,
                        stop_reason="time_budget_exceeded",
                    )
                )
                break

            query = retrieval_queries[-1] if retrieval_queries else question

            # On the first iteration, use planned sub-queries for broader coverage.
            iter_queries = planned_sub_queries if (i == 1 and not retrieval_queries) else [query]

            # ------------------------------------------------------------------
            # Retrieve from enabled sources (novelty-filtered)
            # ------------------------------------------------------------------
            new_mems: List[MemoryItem] = []
            fast_path_hit = False
            if self._fast_path_retriever is not None and self._rcfg.sources.memory:
                fp_result = self._fast_path_retriever.try_fast_path(query)
                if fp_result.hit:
                    fast_path_hit = True
                    new_mems = fp_result.memories
                    acc_memories.extend(new_mems)

            if not fast_path_hit and self._rcfg.sources.memory and int(self._rcfg.memory_limit_per_iter) > 0:
                per_q_limit = max(1, int(self._rcfg.memory_limit_per_iter) // len(iter_queries))
                if len(iter_queries) > 1:
                    # Parallel sub-query retrieval (bounded by max_parallel_queries).
                    sem = asyncio.Semaphore(int(self._rcfg.max_parallel_queries))

                    async def _search_one(sq: str, _sem=sem, _limit=per_q_limit) -> List[MemoryItem]:
                        async with _sem:
                            return await self._memory.search(user_id=user_id, query=sq, limit=_limit)

                    all_mem_results = await asyncio.gather(*[_search_one(sq) for sq in iter_queries])
                    for mems in all_mem_results:
                        for m in mems:
                            if self._rcfg.novelty_masking and m.memory_id in seen_memory_ids:
                                continue
                            seen_memory_ids.add(m.memory_id)
                            new_mems.append(m)
                else:
                    mems = await self._memory.search(user_id=user_id, query=iter_queries[0], limit=per_q_limit)
                    for m in mems:
                        if self._rcfg.novelty_masking and m.memory_id in seen_memory_ids:
                            continue
                        seen_memory_ids.add(m.memory_id)
                        new_mems.append(m)
                acc_memories.extend(new_mems)

            new_exp: List[ExpertiseItem] = []
            if (
                self._rcfg.sources.expertise
                and resolved_expertise_id
                and int(self._rcfg.expertise_limit_per_iter) > 0
            ):
                items = await self._expertise.retrieve_expertise_items(
                    expertise_id=resolved_expertise_id,
                    query=query,
                    limit=int(self._rcfg.expertise_limit_per_iter),
                )
                for it in items:
                    if self._rcfg.novelty_masking and it.item_id in seen_expertise_ids:
                        continue
                    seen_expertise_ids.add(it.item_id)
                    new_exp.append(it)
                acc_expertise.extend(new_exp)

            # Graph retrieval returns a richer structure (edges/nodes/evidence). For the router
            # prompt we only include the incremental items from this iteration (see slicing below),
            # but for the final context we keep full accumulated lists across iterations.
            new_graph_edges = 0
            new_graph_nodes = 0
            new_graph_evidence = 0
            if self._rcfg.sources.graph and self._graph is not None and getattr(self._cfg.graph, "enabled", False):
                rr = await self._graph.build_retrieval_result(user_id=user_id, query=query)
                if rr is not None:
                    # Facts from edges.
                    for e in rr.edges:
                        if self._rcfg.novelty_masking and e.edge_id in seen_graph_edge_ids:
                            continue
                        seen_graph_edge_ids.add(e.edge_id)
                        fact = e.attrs.get("fact") or f"{e.relation} ({e.source_id} -> {e.target_id})"
                        acc_graph_facts.append(str(fact))
                        new_graph_edges += 1

                    # Entities from nodes.
                    for n in rr.nodes:
                        if self._rcfg.novelty_masking and n.node_id in seen_graph_node_ids:
                            continue
                        seen_graph_node_ids.add(n.node_id)
                        labels = n.attrs.get("labels", [])
                        labels_s = ", ".join([str(x) for x in labels]) if labels else ""
                        acc_graph_entities.append(f"{n.label} ({labels_s})".strip())
                        new_graph_nodes += 1

                    # Evidence items.
                    for ev in rr.evidence:
                        ev_key = f"{ev.source}:{ev.source_id}"
                        if self._rcfg.novelty_masking and ev_key in seen_graph_evidence_ids:
                            continue
                        seen_graph_evidence_ids.add(ev_key)
                        acc_graph_evidence.append(f"[{ev.source}:{ev.source_id}] {ev.content}".strip())
                        new_graph_evidence += 1

                    # Reasoning paths and bridge connections (Phase 5).
                    if rr.reasoning_paths:
                        acc_reasoning_paths.extend(rr.reasoning_paths)
                    if rr.bridge_connections:
                        acc_bridge_connections.extend(rr.bridge_connections)

                    # Keep raw result for topology-aware rendering (Phase 6).
                    acc_graph_results.append(rr)

            # ------------------------------------------------------------------
            # Router decision
            # ------------------------------------------------------------------
            if llm_calls >= int(self._rcfg.max_llm_calls):
                # Hard stop: do not exceed configured LLM call budget.
                iterations.append(
                    RetrievalControllerIteration(
                        iteration=i,
                        query=query,
                        decision="answer",
                        new_memory_count=len(new_mems),
                        new_graph_edge_count=new_graph_edges,
                        new_graph_node_count=new_graph_nodes,
                        new_graph_evidence_count=new_graph_evidence,
                        new_expertise_count=len(new_exp),
                        router_input_tokens=0,
                        router_output_tokens=0,
                        router_latency_ms=0.0,
                        elapsed_ms=(time.time() - start) * 1000.0,
                        stop_reason="max_llm_calls_reached",
                    )
                )
                break

            # Build the router input. Important detail: we pass the *newly added* graph items for this
            # iteration (via slicing), but we pass the newly retrieved memories/expertise directly.
            # This keeps the router prompt smaller and avoids re-sending the full accumulated graph.
            retrieval_block = _format_router_retrieval_block(
                question=question,
                last_query=retrieval_queries[-1] if retrieval_queries else None,
                evidence=evidence,
                gaps=gaps,
                memories=new_mems,
                graph_facts=acc_graph_facts[-max(new_graph_edges, 0) :] if new_graph_edges else [],
                graph_entities=acc_graph_entities[-max(new_graph_nodes, 0) :] if new_graph_nodes else [],
                graph_evidence=acc_graph_evidence[-max(new_graph_evidence, 0) :] if new_graph_evidence else [],
                expertise_items=new_exp,
            )

            router_messages = [
                ChatMessage(role="system", content=_ROUTER_SYSTEM_PROMPT),
                ChatMessage(role="user", content=retrieval_block),
            ]
            router_model = self._rcfg.router_model or model or llm.default_model
            router_start = time.time()
            router_resp = await llm.chat(
                router_messages,
                model=router_model,
                temperature=float(self._rcfg.router_temperature),
                max_tokens=int(self._rcfg.router_max_tokens),
            )
            llm_calls += 1

            router_latency_ms = (time.time() - router_start) * 1000.0
            try:
                # Router output is expected to be JSON (see `_ROUTER_SYSTEM_PROMPT`).
                data = _parse_router_json(router_resp.content)
            except Exception:
                # Best-effort fallback: if parsing fails, stop and answer with current evidence.
                data = {
                    "decision": "answer",
                    "evidence": evidence,
                    "gaps": gaps or "None",
                    "draft_answer": None,
                }

            decision_raw = str(data.get("decision", "reflect")).lower().strip()
            decision: Decision = "reflect"
            if decision_raw in {"retrieve", "reflect", "answer"}:
                decision = decision_raw  # type: ignore[assignment]

            # Evidence/gaps are treated as the router's running "scratchpad": the router can refine
            # them each iteration based on new retrieval results.
            evidence = _normalize_evidence(data.get("evidence"))
            gaps = _normalize_gaps(data.get("gaps"))

            # Parse coverage percentage for reflection-based stopping.
            coverage_pct = 0.0
            try:
                coverage_pct = float(data.get("coverage_percentage", 0.0))
            except (TypeError, ValueError):
                pass

            retrieval_query = None
            if decision == "retrieve":
                retrieval_query = data.get("retrieval_query")
                if isinstance(retrieval_query, str):
                    retrieval_query = retrieval_query.strip() or None
                else:
                    retrieval_query = None

            # Parse gap-filling queries from reflection.
            gap_queries: List[str] = []
            if decision == "reflect":
                raw_gq = data.get("gap_queries", [])
                if isinstance(raw_gq, list):
                    gap_queries = [str(q).strip() for q in raw_gq if str(q).strip()]

            # Decide next query if retrieving or reflecting.
            if decision == "retrieve":
                if retrieval_query:
                    retrieval_queries.append(retrieval_query)
                else:
                    retrieval_queries.append(f"{question} {gaps}".strip() if gaps else question)
            elif decision == "reflect" and gap_queries:
                # Use the first gap query as the next retrieval query, promote to retrieve.
                retrieval_queries.append(gap_queries[0])
                decision = "retrieve"

            new_total = len(new_mems) + len(new_exp) + new_graph_edges + new_graph_nodes + new_graph_evidence
            stop_reason = None

            # Coverage-based stopping: if coverage exceeds threshold, stop.
            if coverage_pct >= self._rcfg.min_coverage_percentage and gaps is None:
                decision = "answer"
                stop_reason = "coverage_met"

            if self._rcfg.early_stop_on_no_gaps and gaps is None and decision != "retrieve":
                decision = "answer"
                stop_reason = stop_reason or "no_gaps"
            if self._rcfg.min_new_items_to_continue > 0 and new_total < int(self._rcfg.min_new_items_to_continue) and i > 1:
                # Avoid spinning when retrieval is not producing new material.
                decision = "answer"
                stop_reason = stop_reason or "no_new_items"

            iterations.append(
                RetrievalControllerIteration(
                    iteration=i,
                    query=query,
                    decision=decision,
                    new_memory_count=len(new_mems),
                    new_graph_edge_count=new_graph_edges,
                    new_graph_node_count=new_graph_nodes,
                    new_graph_evidence_count=new_graph_evidence,
                    new_expertise_count=len(new_exp),
                    router_input_tokens=int(getattr(router_resp, "input_tokens", 0) or 0),
                    router_output_tokens=int(getattr(router_resp, "output_tokens", 0) or 0),
                    router_latency_ms=float(getattr(router_resp, "latency_ms", 0.0) or 0.0) or float(router_latency_ms),
                    elapsed_ms=(time.time() - start) * 1000.0,
                    stop_reason=stop_reason,
                    sub_queries=tuple(iter_queries) if len(iter_queries) > 1 else (),
                )
            )

            if decision == "answer":
                break

        # ----------------------------------------------------------------------
        # Final answer context (assembled once)
        # ----------------------------------------------------------------------
        # We optionally format a compact "graph_section" (facts/entities/evidence) as plain text
        # which is then fed into the normal context assembly pipeline.
        graph_section = None

        # Phase 6: topology-aware rendering when enabled
        topo_cfg = getattr(getattr(self._cfg.graph, "retrieval", None), "topology_aware", None)
        if topo_cfg is not None and getattr(topo_cfg, "enabled", False) and acc_graph_results:
            renderer = TopologyAwareRenderer(topo_cfg)
            # Merge all iteration results into a single consolidated result for rendering.
            merged = self._merge_graph_results(acc_graph_results)
            graph_section = renderer.render(merged) or None
        elif acc_graph_facts or acc_graph_entities or acc_graph_evidence:
            lines: List[str] = []
            if acc_graph_facts:
                lines.append("<FACTS>")
                for f in acc_graph_facts[:50]:
                    lines.append(f"- {f}".strip())
            if acc_graph_entities:
                if lines:
                    lines.append("")
                lines.append("<ENTITIES>")
                for e in acc_graph_entities[:50]:
                    lines.append(f"- {e}".strip())
            if acc_graph_evidence:
                if lines:
                    lines.append("")
                lines.append("<EVIDENCE>")
                for ev in acc_graph_evidence[:25]:
                    lines.append(f"- {ev}".strip())
            if acc_reasoning_paths:
                if lines:
                    lines.append("")
                lines.append("<REASONING_PATHS>")
                for idx, path in enumerate(acc_reasoning_paths[:10], 1):
                    chain = " -> ".join(path.node_ids)
                    lines.append(f"  {idx}. {chain}")
            if acc_bridge_connections:
                if lines:
                    lines.append("")
                lines.append(f"<BRIDGE_CONNECTIONS: {len(acc_bridge_connections)} inferred links found>")
            graph_section = "\n".join(lines).strip() or None

        # Context assembly is where we merge session history (optional), system prompt instructions,
        # retrieved memories, and (optionally) graph sections into a final `Context` object.
        sys_prompt = system_instructions or self._cfg.prompts.system_template
        history_limit = max_history_events if max_history_events is not None else self._cfg.prompts.max_history_events

        ctx = await self._assembly.assemble(
            session=session,
            current_query=question,
            memories=acc_memories,
            system_instructions=sys_prompt,
            token_budget=self._cfg.compaction.token_threshold,
            include_history=include_history,
            max_history_events=history_limit,
            graph_section=graph_section,
            graph_section_mode=(
                "topology" if (topo_cfg is not None and getattr(topo_cfg, "enabled", False) and acc_graph_results) else "flat"
            ),
        )

        if acc_expertise and resolved_expertise_id:
            # Expertise items are attached to the context separately so downstream callers can
            # inspect exactly which knowledge base contributed.
            ctx.expertise_items = acc_expertise
            ctx.expertise_id = resolved_expertise_id
            ctx.expertise_items_used = [it.item_id for it in acc_expertise]
            ctx.metadata["expertise_id"] = resolved_expertise_id
            ctx.metadata["expertise_item_count"] = len(acc_expertise)

        # Store a structured trace for debugging/observability. This is intentionally verbose and
        # lets callers reconstruct the controller run (iterations, decisions, IDs used, budgets).
        ctx.metadata["retrieval_controller"] = {
            "enabled": True,
            "iterations": [it.__dict__ for it in iterations],
            "evidence": evidence,
            "gaps": gaps,
            "used_memory_ids": [m.memory_id for m in acc_memories],
            "used_graph_edge_ids": sorted(seen_graph_edge_ids),
            "used_graph_node_ids": sorted(seen_graph_node_ids),
            "used_graph_evidence_ids": sorted(seen_graph_evidence_ids),
            "used_expertise_item_ids": sorted(seen_expertise_ids),
            "llm_calls": llm_calls,
            "elapsed_ms": (time.time() - start) * 1000.0,
        }

        used_graph_ids = {
            "edge_ids": sorted(seen_graph_edge_ids),
            "node_ids": sorted(seen_graph_node_ids),
            "evidence_ids": sorted(seen_graph_evidence_ids),
        }

        return RetrievalControllerResult(
            context=ctx,
            iterations=iterations,
            evidence=evidence,
            gaps=gaps,
            used_memory_ids=[m.memory_id for m in acc_memories],
            used_graph_ids=used_graph_ids,
            used_expertise_item_ids=sorted(seen_expertise_ids),
        )

    async def answer(
        self,
        *,
        session: Session,
        user_id: str,
        question: str,
        llm: ILLMProvider,
        system_instructions: Optional[str] = None,
        include_history: bool = True,
        max_history_events: Optional[int] = None,
        expertise_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[str, RetrievalControllerResult]:
        """
        Convenience wrapper: run the controller to build a context, then call the LLM once to answer.

        Returns:
            (answer_text, controller_result) where controller_result contains the assembled Context + trace.
        """
        result = await self.prepare_context(
            session=session,
            user_id=user_id,
            question=question,
            system_instructions=system_instructions,
            include_history=include_history,
            max_history_events=max_history_events,
            expertise_id=expertise_id,
            llm=llm,
            model=model,
        )

        messages = [ChatMessage(role=m["role"], content=m["content"]) for m in result.context.to_messages()]
        final_model = model or llm.default_model
        final_resp = await llm.chat(messages, model=final_model)
        return final_resp.content, result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_graph_results(results: List[GraphRetrievalResult]) -> GraphRetrievalResult:
        """Merge multiple iteration results into a single consolidated result."""
        if len(results) == 1:
            return results[0]

        all_nodes = []
        all_edges = []
        all_evidence = []
        all_paths: List[ReasoningPath] = []
        all_bridges: List[BridgeConnection] = []
        seen_node_ids: Set[str] = set()
        seen_edge_ids: Set[str] = set()

        for rr in results:
            for n in rr.nodes:
                if n.node_id not in seen_node_ids:
                    seen_node_ids.add(n.node_id)
                    all_nodes.append(n)
            for e in rr.edges:
                if e.edge_id not in seen_edge_ids:
                    seen_edge_ids.add(e.edge_id)
                    all_edges.append(e)
            all_evidence.extend(rr.evidence)
            all_paths.extend(rr.reasoning_paths)
            all_bridges.extend(rr.bridge_connections)

        return GraphRetrievalResult(
            plan_mode=results[0].plan_mode,
            plan_reason=results[0].plan_reason,
            nodes=all_nodes,
            edges=all_edges,
            evidence=all_evidence,
            debug={"merged_iterations": len(results)},
            reasoning_paths=all_paths,
            bridge_connections=all_bridges,
        )


