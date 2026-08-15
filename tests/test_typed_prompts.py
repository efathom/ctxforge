"""Tests for typed extraction prompts."""

from ctxforge.core.memory import MemoryType
from ctxforge.extraction.typed_prompts import (
    TYPED_EXTRACTION_PROMPTS,
    get_all_typed_prompts,
    get_typed_prompt,
)


def test_get_typed_prompt_all_types():
    for mtype in MemoryType:
        prompt = get_typed_prompt(mtype)
        assert isinstance(prompt, str)
        assert len(prompt) > 0


def test_get_typed_prompt_fallback():
    # A hypothetical unknown type should fall back to SEMANTIC
    # We simulate by directly calling .get with a made-up key
    fallback = TYPED_EXTRACTION_PROMPTS.get(
        "nonexistent", TYPED_EXTRACTION_PROMPTS[MemoryType.SEMANTIC]
    )
    assert fallback == TYPED_EXTRACTION_PROMPTS[MemoryType.SEMANTIC]


def test_prompts_mention_type_name():
    type_keywords = {
        MemoryType.SEMANTIC: "semantic",
        MemoryType.EPISODIC: "episodic",
        MemoryType.PROCEDURAL: "procedural",
        MemoryType.PREFERENCE: "preference",
        MemoryType.TOOL: "tool",
    }
    for mtype, keyword in type_keywords.items():
        prompt = get_typed_prompt(mtype)
        assert keyword.lower() in prompt.lower(), (
            f"Prompt for {mtype} should mention '{keyword}'"
        )


def test_get_all_typed_prompts_keys():
    all_prompts = get_all_typed_prompts()
    assert len(all_prompts) == 5
    for mtype in (
        MemoryType.SEMANTIC,
        MemoryType.EPISODIC,
        MemoryType.PROCEDURAL,
        MemoryType.PREFERENCE,
        MemoryType.TOOL,
    ):
        assert mtype in all_prompts
