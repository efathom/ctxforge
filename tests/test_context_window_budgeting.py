import pytest

from ctxforge.compaction.assembler import DefaultContextAssembler
from ctxforge.core.context import Context, ContextSection
from ctxforge.core.events import EventFactory
from ctxforge.protocols.llm import ChatMessage


class DummyTokenizer:
    name = "dummy"

    def count_tokens(self, text: str, model=None) -> int:  # noqa: ANN001
        # Very simple: 1 token per word
        return len((text or "").split())

    def count_message_tokens(self, messages: list[ChatMessage], model=None) -> int:  # noqa: ANN001
        return sum(self.count_tokens(m.content, model=model) for m in messages)


@pytest.mark.asyncio
async def test_fit_to_budget_uses_tokenizer_and_records_breakdown():
    tok = DummyTokenizer()
    assembler = DefaultContextAssembler(tokenizer_provider=tok)

    # Build a context where total_input_tokens is guaranteed to exceed available budget.
    ctx = Context(
        session_id="s",
        user_id="u",
        system_instructions="sys",
        sections=[
            # Low priority first removal (priority=1), then medium, keep required if possible.
            {"name": "low", "content": "x " * 200, "priority": 1, "is_required": False},
            {"name": "mid", "content": "y " * 200, "priority": 10, "is_required": False},
            {"name": "high_req", "content": "z " * 50, "priority": 100, "is_required": True},
        ],
        events=[
            EventFactory.user_message("hello " * 50),
            EventFactory.agent_message("world " * 50),
        ],
        current_query="q " * 20,
        total_token_budget=1200,
        reserved_output_tokens=1000,  # available input budget = 200
    )

    new_ctx = await assembler.fit_to_budget(ctx, budget=1200)
    assert new_ctx.metadata.get("budget_trimmed") is True
    assert "token_breakdown" in new_ctx.metadata

    breakdown = new_ctx.metadata["token_breakdown"]
    assert breakdown["available_input_budget"] == 200
    assert breakdown["total_input_tokens"] <= 200


@pytest.mark.asyncio
async def test_fit_to_budget_trims_history_when_sections_insufficient():
    tok = DummyTokenizer()
    assembler = DefaultContextAssembler(tokenizer_provider=tok)

    # Make sections minimal, but history huge.
    ctx = Context(
        session_id="s",
        user_id="u",
        system_instructions="sys",
        sections=[
            {"name": "required", "content": "keep", "priority": 100, "is_required": True},
        ],
        events=[
            EventFactory.user_message("a " * 150),
            EventFactory.agent_message("b " * 150),
            EventFactory.user_message("c " * 150),
        ],
        current_query="q",
        total_token_budget=1200,
        reserved_output_tokens=1000,  # available input budget = 200
    )

    new_ctx = await assembler.fit_to_budget(ctx, budget=1200)
    assert new_ctx.metadata.get("budget_trimmed") is True
    assert new_ctx.metadata.get("budget_trimmed_events_removed", 0) > 0
    breakdown = new_ctx.metadata["token_breakdown"]
    assert breakdown["total_input_tokens"] <= 200


def test_priority_pack_sections():
    """Test Context.priority_pack_sections greedy packing."""
    ctx = Context(
        session_id="s",
        user_id="u",
        sections=[
            ContextSection(name="required", content="a b c", priority=100, is_required=True),
            ContextSection(name="opt_high", content="d e f g", priority=50, is_required=False),
            ContextSection(name="opt_low", content="h i j k l m", priority=10, is_required=False),
        ],
    )

    # Budget of 8 words: required=3, opt_high=4 fits (total=7), opt_low=6 doesn't fit
    packed = ctx.priority_pack_sections(budget=8)
    names = [s.name for s in packed]
    assert "required" in names
    assert "opt_high" in names
    assert "opt_low" not in names


def test_priority_pack_sections_all_required():
    """Required sections are always included even if they exceed budget."""
    ctx = Context(
        session_id="s",
        user_id="u",
        sections=[
            ContextSection(name="r1", content="a b c d e", priority=100, is_required=True),
        ],
    )
    packed = ctx.priority_pack_sections(budget=2)
    assert len(packed) == 1
    assert packed[0].name == "r1"


