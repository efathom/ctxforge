"""
Convenience helper functions that combine CtxForge context with LLM inference.

These functions are intentionally separated from the core ``CtxForge`` engine
to honor the engine's design principle of not owning LLM inference.  Callers
provide an ``ILLMProvider`` explicitly.

All helpers use only the public API of ``CtxForge`` — no private attribute access.
"""

from typing import Optional

from ctxforge.engine.services.fusion_service import FusionService
from ctxforge.protocols.llm import ChatMessage, ILLMProvider


async def answer_two_step(
    engine,
    llm: ILLMProvider,
    *,
    session_id: str,
    user_id: str,
    user_input: str,
    system_instructions: Optional[str] = None,
    include_memories: bool = True,
    include_history: bool = True,
    max_history_events: Optional[int] = None,
    max_memories: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """
    Run a two-step answer workflow (KG + memory -> synthesis) and return the
    synthesized answer.

    This is the recommended replacement for ``engine.answer_two_step()``.

    Args:
        engine: A ``CtxForge`` instance.
        llm: The LLM provider to use for inference.
        session_id: The session identifier.
        user_id: The user identifier.
        user_input: The user's input message.
        system_instructions: Optional system prompt override.
        include_memories: Whether to retrieve and include memories.
        include_history: Whether to include conversation history.
        max_history_events: Max history events.
        max_memories: Max memories to retrieve.
        model: Optional model override.

    Returns:
        The synthesized answer string.
    """
    inputs = await engine.prepare_context(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        system_instructions=system_instructions,
        include_memories=include_memories,
        include_history=include_history,
        max_history_events=max_history_events,
        max_memories=max_memories,
        return_two_step_inputs=True,
    )

    if not inputs.kg_section:
        resp = await llm.chat(inputs.memory_messages, model=model or llm.default_model)
        return resp.content

    fusion = FusionService(config=engine.config.fusion)
    return await fusion.run_two_step(
        llm=llm,
        query=user_input,
        kg_section=inputs.kg_section,
        memory_messages=inputs.memory_messages,
        model=model or engine.config.fusion.synthesis_model,
    )


async def answer_with_controller(
    engine,
    llm: ILLMProvider,
    *,
    session_id: str,
    user_id: str,
    user_input: str,
    system_instructions: Optional[str] = None,
    include_history: bool = True,
    max_history_events: Optional[int] = None,
    expertise_id: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Run the iterative retrieval controller and return the final answer.

    This is the recommended replacement for ``engine.answer_with_controller()``.

    Args:
        engine: A ``CtxForge`` instance.
        llm: The LLM provider to use for inference.
        session_id: The session identifier.
        user_id: The user identifier.
        user_input: The user's input message.
        system_instructions: Optional system prompt override.
        include_history: Whether to include conversation history.
        max_history_events: Max history events.
        expertise_id: Optional expertise ID for the controller.
        model: Optional model override.

    Returns:
        The final answer string.
    """
    context = await engine.prepare_context(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        system_instructions=system_instructions,
        include_history=include_history,
        max_history_events=max_history_events,
        use_controller=True,
        llm=llm,
        expertise_id=expertise_id,
        model=model,
    )

    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in context.to_messages()]
    final_model = model or llm.default_model
    resp = await llm.chat(messages, model=final_model)
    return resp.content
