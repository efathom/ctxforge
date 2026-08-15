"""
Validated Knowledge Service.

Provides a direct path for saving user-validated knowledge items,
bypassing the reflection/curation loop.

This is the "poor man's continuous learning" approach:
- We do NOT update model weights
- We DO update retrieval knowledge when we find a successful result
- Every good result becomes future context
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ctxforge.core.expertise import ExpertiseItem, ExpertiseSection
from ctxforge.core.memory import MemoryItem, MemorySource, MemoryType
from ctxforge.protocols.expertise import IExpertiseStore
from ctxforge.protocols.storage import IMemoryStore

logger = logging.getLogger(__name__)


class ValidatedKnowledgeEntry(BaseModel):
    """
    A user-validated knowledge entry.
    
    Fields:
    - name: Human-readable name
    - question: The original question
    - content: The validated answer/query/pattern
    - summary: Brief explanation
    - notes: Caveats or assumptions
    """
    name: str
    question: str
    content: str
    summary: Optional[str] = None
    notes: Optional[str] = None
    knowledge_type: str = "validated_pattern"
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    validated_by: Optional[str] = None  # user_id who validated


class ValidatedKnowledgeService:
    """
    Service for directly saving user-validated knowledge.
    
    This provides the "save after success" pattern where:
    1. Agent finds a successful result
    2. Agent asks user: "Would you like me to save this?"
    3. User approves (optionally with edits)
    4. This service persists immediately without reflection/curation
    
    The key difference from the Curator flow:
    - Curator: LLM-driven ADD/UPDATE/DELETE decisions
    - This: Direct human-validated saves
    
    Example agent tool:
        @tool
        def save_validated_query(name: str, question: str, query: str, notes: str):
            '''Save a validated SQL query to the knowledge base.'''
            return await validated_knowledge_service.save_expertise_item(...)
    """
    
    def __init__(
        self,
        expertise_store: Optional[IExpertiseStore] = None,
        expertise_indexer: Optional[Any] = None,
        memory_store: Optional[IMemoryStore] = None,
        memory_indexer: Optional[Any] = None,
    ):
        self._expertise_store = expertise_store
        self._expertise_indexer = expertise_indexer
        self._memory_store = memory_store
        self._memory_indexer = memory_indexer
    
    async def save_expertise_item(
        self,
        expertise_id: str,
        section: ExpertiseSection,
        content: str,
        *,
        name: Optional[str] = None,
        source_question: Optional[str] = None,
        summary: Optional[str] = None,
        notes: Optional[str] = None,
        validated_by: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[ExpertiseItem]:
        """
        Directly save a validated item to an expertise knowledge base.
        
        This is the ctxforge equivalent of save_validated_query:
        - Loads the expertise
        - Adds the item with human_validated=True marker
        - Persists and indexes immediately
        - Returns the created item
        
        Args:
            expertise_id: Which expertise to add to
            section: Category for the item
            content: The validated knowledge content
            name: Optional human-readable name
            source_question: Original question that led to this
            summary: Brief explanation
            notes: Caveats or assumptions
            validated_by: user_id who validated
            tags: Optional tags for categorization
            
        Returns:
            The created ExpertiseItem, or None if save failed
        """
        if not self._expertise_store:
            logger.warning("No expertise store configured for validated knowledge save")
            return None
        
        expertise = await self._expertise_store.load(expertise_id)
        if not expertise:
            logger.warning(f"Expertise {expertise_id} not found")
            return None
        
        # Create the item with validation metadata
        item = expertise.add_item(
            section=section,
            content=content,
            source="user_validated",
        )
        
        # Add validation metadata
        item.metadata["human_validated"] = True
        item.metadata["validated_at"] = datetime.now(timezone.utc).isoformat()
        if validated_by:
            item.metadata["validated_by"] = validated_by
        if name:
            item.metadata["name"] = name
        if source_question:
            item.metadata["source_question"] = source_question
        if summary:
            item.metadata["summary"] = summary
        if notes:
            item.metadata["notes"] = notes
        if tags:
            item.metadata["tags"] = tags
        
        # Persist
        await self._expertise_store.save(expertise)
        
        # Index for retrieval
        if self._expertise_indexer:
            try:
                await self._expertise_indexer.index_item(item, expertise_id)
            except Exception as e:
                logger.warning(f"Failed to index expertise item: {e}")
        
        logger.info(f"Saved validated expertise item {item.item_id} to {expertise_id}")
        return item
    
    async def save_validated_memory(
        self,
        user_id: str,
        content: str,
        memory_type: MemoryType = MemoryType.PROCEDURAL,
        *,
        name: Optional[str] = None,
        source_question: Optional[str] = None,
        summary: Optional[str] = None,
        notes: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[MemoryItem]:
        """
        Directly save a validated memory item.
        
        Similar to save_expertise_item but for the memory subsystem.
        Useful for saving validated patterns, procedures, or facts.
        
        Args:
            user_id: User scope for the memory
            content: The validated content
            memory_type: Type of memory (default: PROCEDURAL)
            name: Optional human-readable name
            source_question: Original question
            summary: Brief explanation
            notes: Caveats or assumptions
            tags: Optional tags
            
        Returns:
            The created MemoryItem, or None if save failed
        """
        if not self._memory_store:
            logger.warning("No memory store configured for validated knowledge save")
            return None
        
        # Create memory with validation metadata
        memory = MemoryItem(
            user_id=user_id,
            content=content,
            type=memory_type,
            source=MemorySource.USER_EXPLICIT,
            confidence_score=1.0,  # Human-validated = high confidence
            tags=tags or [],
            metadata={
                "human_validated": True,
                "validated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        
        if name:
            memory.metadata["name"] = name
        if source_question:
            memory.metadata["source_question"] = source_question
        if summary:
            memory.metadata["summary"] = summary
        if notes:
            memory.metadata["notes"] = notes
        
        # Persist
        await self._memory_store.add(memory)
        
        # Index for retrieval
        if self._memory_indexer:
            try:
                await self._memory_indexer.index(memory)
            except Exception as e:
                logger.warning(f"Failed to index memory: {e}")
        
        logger.info(f"Saved validated memory {memory.memory_id} for user {user_id}")
        return memory
    
    async def save_validated_entry(
        self,
        entry: ValidatedKnowledgeEntry,
        expertise_id: Optional[str] = None,
        user_id: Optional[str] = None,
        section: ExpertiseSection = ExpertiseSection.STRATEGIES,
    ) -> Optional[str]:
        """
        Save a validated knowledge entry (convenience wrapper).
        
        Automatically routes to expertise or memory based on context.
        
        Returns:
            The item_id/memory_id of the saved entry, or None
        """
        if expertise_id:
            item = await self.save_expertise_item(
                expertise_id=expertise_id,
                section=section,
                content=entry.content,
                name=entry.name,
                source_question=entry.question,
                summary=entry.summary,
                notes=entry.notes,
                validated_by=entry.validated_by,
                tags=entry.tags,
            )
            return item.item_id if item else None
        elif user_id:
            memory = await self.save_validated_memory(
                user_id=user_id,
                content=entry.content,
                name=entry.name,
                source_question=entry.question,
                summary=entry.summary,
                notes=entry.notes,
                tags=entry.tags,
            )
            return memory.memory_id if memory else None
        else:
            logger.warning("No expertise_id or user_id provided for validated entry")
            return None
