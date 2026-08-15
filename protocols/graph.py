"""
Graph memory protocol models.

This module defines the core data shapes and store/extractor interfaces for the engine's
graph memory layer. Backends (in-memory, Neo4j, etc.) implement `IGraphStore`.

Notes:
- `scope_id` partitions the graph (e.g., per-user scope and an optional global scope).
- Temporal validity is represented via `valid_at` / `invalid_at`. A query can use `as_of`
  to interpret which edges are considered "currently valid".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Protocol

from ctxforge.core.alignment_types import AlignmentStatus, CharSpan

if TYPE_CHECKING:
    from ctxforge.retrieval.enhanced_structures import EnhancedMemoryIndex


GraphSearchScope = Literal["nodes", "edges", "episodes"]


@dataclass
class GraphEpisode:
    """A stored interaction artifact used as input for extraction and retrieval."""

    episode_id: str
    scope_id: str
    content: str
    content_type: str = "text"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphNode:
    """A graph entity node (typed via `labels`) within a `scope_id` partition."""

    node_id: str
    scope_id: str
    name: str
    labels: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    summary: Optional[str] = None
    name_embedding: Optional[List[float]] = None
    
    # Source grounding fields (types from core module - no circular dependency)
    source_episode_ids: List[str] = field(default_factory=list)  # Episodes this was extracted from
    source_spans: Dict[str, CharSpan] = field(default_factory=dict)  # episode_id -> CharSpan
    alignment_status: Optional[AlignmentStatus] = None
    extraction_confidence: float = 1.0


@dataclass
class GraphEdge:
    """A directed relationship between two nodes, optionally expressed as a natural-language fact."""

    edge_id: str
    scope_id: str
    source_node_id: str
    target_node_id: str
    edge_type: str
    fact: Optional[str] = None
    labels: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    
    # Source grounding fields (types from core module - no circular dependency)
    source_episode_ids: List[str] = field(default_factory=list)  # Episodes this was extracted from
    source_spans: Dict[str, CharSpan] = field(default_factory=dict)  # episode_id -> CharSpan
    alignment_status: Optional[AlignmentStatus] = None
    extraction_confidence: float = 1.0


@dataclass
class GraphSearchFilters:
    """Optional constraints applied during graph searches."""

    node_labels: Optional[List[str]] = None
    edge_types: Optional[List[str]] = None
    valid_only: bool = True
    as_of: Optional[datetime] = None
    node_ids: Optional[List[str]] = None
    edge_ids: Optional[List[str]] = None


@dataclass
class GraphSearchResult:
    """Container returned by `IGraphStore.search()` for a specific search scope."""

    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    episodes: List[GraphEpisode] = field(default_factory=list)


@dataclass
class GraphCommunity:
    """
    A derived community (cluster) of entity nodes within a `scope_id`.

    Communities are not extracted from text directly; they are computed from graph structure
    and enriched with a short `name` and `summary`.
    """

    community_id: str
    scope_id: str
    name: str
    summary: str
    member_count: int
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    name_embedding: Optional[List[float]] = None
    summary_embedding: Optional[List[float]] = None
    overlap: Optional[int] = None  # optional retrieval-time signal


class IGraphStore(Protocol):
    """Persistent store for graph memory objects."""

    async def add_episodes(self, scope_id: str, episodes: List[GraphEpisode]) -> int:
        """Persist episodes for a scope. Returns the number of episodes stored."""
        ...

    async def upsert_nodes(self, scope_id: str, nodes: List[GraphNode]) -> int:
        """Insert or update nodes for a scope. Returns the number of nodes upserted."""
        ...

    async def upsert_edges(self, scope_id: str, edges: List[GraphEdge]) -> int:
        """Insert or update edges for a scope. Returns the number of edges upserted."""
        ...

    async def get_edges_by_ids(self, scope_id: str, edge_ids: List[str]) -> List[GraphEdge]:
        """Fetch edges by id within a scope (best-effort order; may be backend-defined)."""
        ...

    async def invalidate_edges(
        self,
        scope_id: str,
        edge_ids: List[str],
        *,
        invalid_at: datetime,
    ) -> int:
        """Mark edges invalid (no longer "current") as of `invalid_at`. Returns count updated."""
        ...

    async def delete_scope(self, scope_id: str) -> int:
        """Delete all graph objects for a scope. Returns a backend-defined count removed."""
        ...

    async def search_nodes_semantic(
        self,
        scope_id: str,
        query_vector: List[float],
        *,
        limit: int = 20,
        filters: Optional[GraphSearchFilters] = None,
    ) -> List[GraphNode]:
        """Return nodes ranked by vector similarity (if supported by the backend)."""
        ...

    async def search(
        self,
        scope_id: str,
        query: str,
        *,
        scope: GraphSearchScope,
        limit: int = 20,
        filters: Optional[GraphSearchFilters] = None,
        center_node_id: Optional[str] = None,
    ) -> GraphSearchResult:
        """Search within a scope and return results for the requested search scope."""
        ...

    # ---------------------------------------------------------------------
    # Community layer (derived artifacts)
    # ---------------------------------------------------------------------

    async def upsert_communities(self, scope_id: str, communities: List[GraphCommunity]) -> int:
        """Insert or update community nodes for a scope. Returns number upserted."""
        ...

    async def upsert_memberships(self, scope_id: str, memberships: List[tuple[str, str]]) -> int:
        """Insert membership edges (community_id -> node_id) for a scope. Returns number upserted."""
        ...

    async def get_communities_for_nodes(
        self,
        scope_id: str,
        node_ids: List[str],
        *,
        limit: int = 10,
    ) -> List[GraphCommunity]:
        """Return communities covering the provided nodes, ranked by overlap."""
        ...

    async def delete_communities(self, scope_id: str) -> int:
        """Delete all community nodes + membership edges for a scope."""
        ...

    # ---------------------------------------------------------------------
    # Enhanced index layer (derived fast-path aggregates)
    # ---------------------------------------------------------------------

    async def save_enhanced_index(self, scope_id: str, index: "EnhancedMemoryIndex") -> None:
        """Persist the enhanced memory index for a scope (overwrites any existing index)."""
        ...

    async def load_enhanced_index(self, scope_id: str) -> Optional["EnhancedMemoryIndex"]:
        """Load the enhanced memory index for a scope. Returns None if not found."""
        ...


class IGraphExtractor(Protocol):
    async def extract(
        self,
        *,
        scope_id: str,
        episodes: List[GraphEpisode],
        ontology: Any,
        model: Optional[str] = None,
        extraction_passes: int = 1,
        enable_alignment: bool = True,
    ) -> tuple[List[GraphNode], List[GraphEdge]]:
        """
        Extract ontology-validated nodes/edges from a list of episodes.
        
        Args:
            scope_id: The scope to extract into
            episodes: Episodes to extract from
            ontology: Graph ontology for validation
            model: LLM model to use
            extraction_passes: Number of extraction passes for improved recall
            enable_alignment: Whether to align extractions to source text
            
        Returns:
            Tuple of (nodes, edges) extracted
        """
        ...


