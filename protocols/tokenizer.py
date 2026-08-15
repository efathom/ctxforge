"""
Tokenizer Provider Protocol Interface.

Provides token counting for strings and chat message lists.
This is pluggable so different apps can use accurate tokenizers
for their target model/provider.
"""

from typing import List, Optional, Protocol, runtime_checkable

from ctxforge.protocols.llm import ChatMessage


@runtime_checkable
class ITokenizerProvider(Protocol):
    """Protocol for token counting."""

    @property
    def name(self) -> str:
        """The name of this tokenizer provider."""
        ...

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """Count tokens in a single string."""
        ...

    def count_message_tokens(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
    ) -> int:
        """Count tokens in a list of chat messages."""
        ...


