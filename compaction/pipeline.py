"""
Condenser Pipeline - Chain multiple condensers together.

Allows composing multiple condensation strategies into a single condenser
that processes views through each stage in sequence.
"""

from typing import List, Optional, Union

from ctxforge.compaction.view import (
    CompactionView,
    CondensationResult,
    ICondenser,
)
from ctxforge.engine.registry import registry
from ctxforge.protocols.compactor import CompactionConfig


@registry.register_condenser("pipeline")
class CondenserPipeline:
    """
    Combines multiple condensers into a single condenser.

    Useful for creating multi-stage condensation:
    1. ObservationMaskingCondenser - Hide sensitive output
    2. SummarizingCondenser - Summarize old events
    3. SlidingWindowCondenser - Ensure size limits

    The pipeline processes views through each condenser in sequence,
    accumulating forgotten events and metadata.

    Example:
        >>> pipeline = CondenserPipeline(
        ...     SummarizingCondenser(llm=llm),
        ...     SlidingWindowCondenser(),
        ... )
        >>> result = await pipeline.condense(view, config)

        >>> # Or build incrementally
        >>> pipeline = CondenserPipeline()
        >>> pipeline.add_condenser(SummarizingCondenser(llm=llm))
        >>> pipeline.add_condenser(SlidingWindowCondenser())
    """

    def __init__(self, *condensers: ICondenser):
        """
        Initialize with optional condensers.

        Args:
            *condensers: Condensers to include in the pipeline
        """
        self._condensers: List[ICondenser] = list(condensers)

    @property
    def name(self) -> str:
        """Pipeline name composed from condenser names."""
        if not self._condensers:
            return "pipeline(empty)"
        names = [c.name for c in self._condensers]
        return f"pipeline({' -> '.join(names)})"

    @property
    def condensers(self) -> List[ICondenser]:
        """Get the list of condensers in the pipeline."""
        return list(self._condensers)

    def __len__(self) -> int:
        """Number of condensers in the pipeline."""
        return len(self._condensers)

    def should_condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> bool:
        """
        Check if any condenser in the pipeline wants to condense.

        Returns True if at least one condenser indicates condensation is needed.
        """
        if not self._condensers:
            return False
        config = config or CompactionConfig()
        return any(c.should_condense(view, config) for c in self._condensers)

    async def condense(
        self,
        view: CompactionView,
        config: Optional[CompactionConfig] = None,
    ) -> Union[CompactionView, CondensationResult]:
        """
        Run condensers in sequence.

        Each condenser receives the output of the previous one.
        Metadata is accumulated from all stages.

        Args:
            view: The view to condense
            config: Optional compaction configuration

        Returns:
            Final condensation result with accumulated metadata
        """
        config = config or CompactionConfig()

        if not self._condensers:
            return CondensationResult(
                view=view,
                summary_generated=False,
                tokens_saved=0,
                metadata={"strategy": self.name, "action": "no_condensers"},
            )

        current_view = view
        total_tokens_saved = 0
        combined_metadata: dict = {}
        summary_generated = False
        first_forgotten_id: Optional[str] = None
        last_forgotten_id: Optional[str] = None

        for condenser in self._condensers:
            result = await condenser.condense(current_view, config)

            if isinstance(result, CondensationResult):
                # Extract view for next stage
                current_view = result.view
                total_tokens_saved += result.tokens_saved
                combined_metadata[condenser.name] = result.metadata

                if result.summary_generated:
                    summary_generated = True

                # Track forgotten event range
                if result.events_forgotten_start_id and not first_forgotten_id:
                    first_forgotten_id = result.events_forgotten_start_id
                if result.events_forgotten_end_id:
                    last_forgotten_id = result.events_forgotten_end_id
            else:
                # Result is a CompactionView
                current_view = result

        return CondensationResult(
            view=current_view,
            events_forgotten_start_id=first_forgotten_id,
            events_forgotten_end_id=last_forgotten_id,
            summary_generated=summary_generated,
            tokens_saved=total_tokens_saved,
            metadata={
                "strategy": self.name,
                "stages": combined_metadata,
            },
        )

    def add_condenser(self, condenser: ICondenser) -> "CondenserPipeline":
        """
        Add a condenser to the end of the pipeline.

        Args:
            condenser: Condenser to add

        Returns:
            Self for method chaining
        """
        self._condensers.append(condenser)
        return self

    def insert_condenser(
        self,
        index: int,
        condenser: ICondenser,
    ) -> "CondenserPipeline":
        """
        Insert a condenser at a specific position.

        Args:
            index: Position to insert at (0 = first)
            condenser: Condenser to insert

        Returns:
            Self for method chaining
        """
        self._condensers.insert(index, condenser)
        return self

    def remove_condenser(self, index: int) -> "CondenserPipeline":
        """
        Remove a condenser at a specific position.

        Args:
            index: Position to remove from

        Returns:
            Self for method chaining
        """
        if 0 <= index < len(self._condensers):
            self._condensers.pop(index)
        return self

    def clear(self) -> "CondenserPipeline":
        """
        Remove all condensers from the pipeline.

        Returns:
            Self for method chaining
        """
        self._condensers.clear()
        return self
