"""
Tokenizer provider adapter.

ctxforge has a dedicated `ITokenizerProvider` protocol used for token budgeting.
Most LLM providers in this repo already implement `count_tokens` / `count_message_tokens`,
so this adapter lets us reuse them as tokenizers without adding a new dependency.
"""

from __future__ import annotations

from typing import List, Optional

from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.protocols.tokenizer import ITokenizerProvider


class LLMTokenizerProvider(ITokenizerProvider):
    """Adapts an `ILLMProvider` into an `ITokenizerProvider`."""

    def __init__(self, llm_provider: ILLMProvider):
        self._llm = llm_provider

    @property
    def name(self) -> str:
        # Prefer the underlying provider name for telemetry.
        return getattr(self._llm, "name", "llm_tokenizer")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return int(self._llm.count_tokens(text, model=model))

    def count_message_tokens(self, messages: List[ChatMessage], model: Optional[str] = None) -> int:
        return int(self._llm.count_message_tokens(messages, model=model))


