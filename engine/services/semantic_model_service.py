"""
Semantic Model Service.

Manages domain semantic models and their injection into context.
"""

import logging
from typing import List, Optional

from ctxforge.core.context import Context
from ctxforge.core.semantic_model import (
    EntityDefinition,
    SemanticModel,
    SemanticModelStore,
)

logger = logging.getLogger(__name__)


class SemanticModelService:
    """
    Service for managing and applying semantic models to context.
    
    The semantic model provides a "domain map" that:
    - Is always present in system context
    - Guides the agent on what knowledge is available
    - Provides rules and gotchas upfront
    
    Example usage:
    ```python
    service = SemanticModelService(
        store=InMemorySemanticModelStore(),
        default_model=my_default_model,
    )
    
    # Inject into context during prepare_context
    context = await service.inject_into_context(context, model_id="my-domain")
    ```
    """
    
    def __init__(
        self,
        store: Optional[SemanticModelStore] = None,
        default_model: Optional[SemanticModel] = None,
    ):
        """
        Initialize the service.
        
        Args:
            store: Store for loading/saving semantic models
            default_model: Default model to use when no model_id specified
        """
        self._store = store
        self._default_model = default_model
    
    @property
    def default_model(self) -> Optional[SemanticModel]:
        """Get the default semantic model."""
        return self._default_model
    
    @default_model.setter
    def default_model(self, model: Optional[SemanticModel]) -> None:
        """Set the default semantic model."""
        self._default_model = model
    
    async def get_model(self, model_id: Optional[str] = None) -> Optional[SemanticModel]:
        """
        Get a semantic model by ID, or return the default.
        
        Args:
            model_id: Optional model ID to load
            
        Returns:
            The requested model, default model, or None
        """
        if model_id and self._store:
            model = await self._store.load(model_id)
            if model:
                return model
        return self._default_model
    
    async def save_model(self, model_id: str, model: SemanticModel) -> None:
        """
        Save a semantic model.
        
        Args:
            model_id: ID to save the model under
            model: The model to save
        """
        if self._store:
            await self._store.save(model_id, model)
        else:
            logger.warning("No store configured, cannot save semantic model")
    
    async def list_models(self) -> List[str]:
        """
        List all available model IDs.
        
        Returns:
            List of model IDs
        """
        if self._store:
            return await self._store.list_models()
        return []
    
    async def inject_into_context(
        self,
        context: Context,
        model_id: Optional[str] = None,
        compact: bool = True,
    ) -> Context:
        """
        Inject semantic model into the context's system instructions.
        
        Args:
            context: The context to modify
            model_id: Optional specific model to use
            compact: Use compact format (default True)
            
        Returns:
            The modified context
        """
        model = await self.get_model(model_id)
        if not model:
            return context
        
        # Add semantic model as a section
        model_section = model.to_context_string(compact=compact)
        context.add_section("semantic_model", model_section)
        context.metadata["semantic_model_name"] = model.name
        context.metadata["semantic_model_version"] = model.version
        
        logger.debug(f"Injected semantic model '{model.name}' into context")
        return context
    
    def build_from_expertise(
        self,
        name: str,
        expertise_items: list,
        description: str = "",
    ) -> SemanticModel:
        """
        Build a semantic model from expertise items.
        
        Analyzes expertise items to create entity definitions
        and extract rules/gotchas.
        
        Args:
            name: Name for the semantic model
            expertise_items: List of expertise items to analyze
            description: Optional description
            
        Returns:
            A SemanticModel built from the expertise items
        """
        from ctxforge.core.expertise import ExpertiseSection
        
        model = SemanticModel(name=name, description=description)
        
        # Group items by section for entity creation
        sections_seen = set()
        for item in expertise_items:
            if hasattr(item, 'section'):
                sections_seen.add(item.section)
        
        # Create entities from sections
        for section in sections_seen:
            if section == ExpertiseSection.COMMON_MISTAKES:
                # Extract as gotchas
                for item in expertise_items:
                    if hasattr(item, 'section') and item.section == section:
                        model.add_gotcha(item.content[:200])
            else:
                # Create entity
                display_name = section.to_display_name() if hasattr(section, 'to_display_name') else str(section)
                entity = EntityDefinition(
                    name=section.value,
                    description=f"Knowledge about {display_name.lower()}",
                    use_cases=[],
                )
                model.add_entity(entity)
        
        return model
