"""
Protocols for semantic model components.
"""

from typing import List, Optional, Protocol

from ctxforge.core.semantic_model import SemanticModel


class ISemanticModelStore(Protocol):
    """Protocol for semantic model storage."""
    
    async def load(self, model_id: str) -> Optional[SemanticModel]:
        """Load a semantic model by ID."""
        ...
    
    async def save(self, model_id: str, model: SemanticModel) -> None:
        """Save a semantic model."""
        ...
    
    async def list_models(self) -> List[str]:
        """List all available model IDs."""
        ...
