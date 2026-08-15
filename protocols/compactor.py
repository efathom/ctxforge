"""
Compactor Protocol Interface.

Defines the contract for context window management strategies.
Compactors handle the challenge of fitting conversation history
into limited context windows.

Protocols defined here:
- IContextAssembler: Assembles final context for LLM
- CompactionConfig: Configuration for compaction
- CompactionResult: Legacy result type (use CondensationResult for new code)

For condensation (the recommended pattern), see:
- ICondenser, CompactionView, CondensationResult in ctxforge.compaction.view
"""

from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable

from ctxforge.core.context import Context
from ctxforge.core.memory import MemoryItem
from ctxforge.core.session import Session


@dataclass
class CompactionConfig:
    """Configuration for compaction/condensation operations."""

    # When to trigger compaction
    event_threshold: int = 10  # Compact when events exceed this
    token_threshold: int = 4000  # Compact when tokens exceed this

    # How many events to keep after compaction
    keep_recent: int = 5

    # Summarization settings
    summarization_model: Optional[str] = None
    max_summary_tokens: int = 500

    # Strategy-specific settings
    include_tool_calls: bool = True
    preserve_system_events: bool = True


@runtime_checkable
class IContextAssembler(Protocol):
    """
    Protocol for assembling the final context.
    
    Context assemblers take session state, memories, and configuration
    to produce the final context that will be sent to the LLM.
    """
    
    @property
    def name(self) -> str:
        """The name of this assembler."""
        ...
    
    async def assemble(
        self,
        session: Session,
        current_query: str,
        memories: List[MemoryItem],
        system_instructions: str = "",
        token_budget: int = 8000,
    ) -> Context:
        """
        Assemble the context for LLM invocation.
        
        This is the "Mise en Place" step - preparing all ingredients
        before sending to the LLM.
        
        Args:
            session: The current session
            current_query: The user's current query
            memories: Retrieved relevant memories
            system_instructions: System prompt
            token_budget: Maximum tokens available
            
        Returns:
            The assembled Context object
        """
        ...
    
    async def fit_to_budget(
        self,
        context: Context,
        budget: int,
    ) -> Context:
        """
        Fit context to a token budget.
        
        Prunes or compacts sections as needed.
        
        Args:
            context: The context to fit
            budget: The target token budget
            
        Returns:
            The fitted context
        """
        ...

