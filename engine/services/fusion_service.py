from __future__ import annotations

"""
Answer fusion service.

Implements a two-step fusion workflow:
- produce a KG-grounded answer (caller provides KG section)
- produce a memory-grounded answer (caller provides normal context)
- synthesize the two into a final answer with conflict handling

This service does not own inference by default; callers pass an `ILLMProvider`.
"""

from dataclasses import dataclass
from typing import Optional

from ctxforge.config.base import FusionConfig
from ctxforge.core.context import Context
from ctxforge.protocols.llm import ChatMessage, ILLMProvider


@dataclass(frozen=True)
class FusionPrompts:
    kg_system_prompt: str
    synthesis_system_prompt: str


@dataclass(frozen=True)
class TwoStepInputs:
    """
    Prepared inputs for the two-step workflow.

    Agent-friendly: callers can inspect `kg_section` and `memory_context` and run their own answer/fusion logic.
    """

    kg_section: Optional[str]
    memory_context: Context
    memory_messages: list[ChatMessage]


class FusionService:
    def __init__(self, *, config: FusionConfig):
        self._cfg = config

    def build_prompts(self, *, query: str, kg_section: str, kg_answer: str, memory_answer: str) -> FusionPrompts:
        kg_system_prompt = (
            "You are a careful assistant.\n\n"
            "Use ONLY the following knowledge graph context to answer the user.\n"
            "If the answer is not supported by the context, say you don't know.\n\n"
            "Knowledge Graph Context:\n"
            f"{kg_section}\n\n"
            "Output rules:\n"
            "- Be specific and grounded\n"
            "- Do not invent facts\n"
        )

        synthesis_system_prompt = (
            "You are a careful assistant.\n\n"
            "You will be given a user query and two candidate answers:\n"
            "- One grounded in knowledge-graph context\n"
            "- One grounded in general memory/context\n\n"
            "Your task:\n"
            "- Merge them into one coherent answer\n"
            "- If they conflict, explicitly call out the conflict and choose the most grounded statement\n"
            "- Do not invent facts\n\n"
            f"User query:\n{query}\n\n"
            "Answer A (KG-grounded):\n"
            f"{kg_answer}\n\n"
            "Answer B (Memory-grounded):\n"
            f"{memory_answer}\n"
        )

        return FusionPrompts(
            kg_system_prompt=kg_system_prompt,
            synthesis_system_prompt=synthesis_system_prompt,
        )

    async def run_two_step(
        self,
        *,
        llm: ILLMProvider,
        query: str,
        kg_section: str,
        memory_messages: list[ChatMessage],
        model: Optional[str] = None,
    ) -> str:
        """
        Execute the full two-step workflow and return the synthesized answer.

        `memory_messages` should already include the system message + history + user query.
        """
        model = model or self._cfg.synthesis_model or llm.default_model

        # Step 1: KG-only answer
        kg_resp = await llm.chat(
            [
                ChatMessage(role="system", content=self.build_prompts(query=query, kg_section=kg_section, kg_answer="", memory_answer="").kg_system_prompt),
                ChatMessage(role="user", content=query),
            ],
            model=model,
            max_tokens=self._cfg.max_tokens,
        )
        kg_answer = kg_resp.content

        # Step 2: memory/context answer
        mem_resp = await llm.chat(
            memory_messages,
            model=model,
            max_tokens=self._cfg.max_tokens,
        )
        memory_answer = mem_resp.content

        # Step 3: synthesis
        prompts = self.build_prompts(query=query, kg_section=kg_section, kg_answer=kg_answer, memory_answer=memory_answer)
        final_resp = await llm.chat(
            [
                ChatMessage(role="system", content=prompts.synthesis_system_prompt),
                ChatMessage(role="user", content="Produce the final merged answer."),
            ],
            model=model,
            max_tokens=self._cfg.max_tokens,
        )
        return final_resp.content


