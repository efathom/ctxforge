"""
Neo4j-backed graph store.

This backend persists graph memory into Neo4j and implements the `IGraphStore` protocol.

Conventions:
- Partitioning: every node/relationship includes a `scope_id` property.
- Entities are stored as `__Entity__` nodes (configurable via `entity_label`).
- Episodes are stored as `__Episode__` nodes.
- Edges are stored as relationships between entity nodes. Temporal validity is stored using
  `valid_at` and `invalid_at` (with a far-future sentinel used internally for "present").

Vector search:
- If configured, nodes store `name_embedding` (float array) and the store can query a Neo4j
  vector index via `db.index.vector.queryNodes`.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.config.base import Neo4jGraphStoreConfig
from ctxforge.protocols.graph import (
    GraphCommunity,
    GraphEdge,
    GraphEpisode,
    GraphNode,
    GraphSearchFilters,
    GraphSearchResult,
    GraphSearchScope,
    IGraphStore,
)
from ctxforge.retrieval.enhanced_structures import EnhancedMemoryIndex


class Neo4jGraphStore(IGraphStore):
    """Neo4j implementation of `IGraphStore` (async driver)."""

    def __init__(self, config: Neo4jGraphStoreConfig):
        try:
            from neo4j import AsyncGraphDatabase  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Neo4j support requires the 'neo4j' package. Install it and ensure Neo4j is running locally."
            ) from e

        self._cfg = config
        self._driver = AsyncGraphDatabase.driver(
            config.url,
            auth=(config.username, config.password),
        )
        self._database: Optional[str] = config.database
        self._initialized = False
        self._entity_label = config.entity_label or "__Entity__"

    async def close(self) -> None:
        """Close the underlying Neo4j driver."""
        await self._driver.close()

    async def initialize(self) -> None:
        """Create required constraints and indexes (best-effort; no-op if disabled)."""

        if self._initialized:
            return
        # Best-effort index/constraint creation (works on Neo4j 5+).
        if not self._cfg.create_indexes:
            self._initialized = True
            return
        try:
            async with self._driver.session(database=self._database) as session:
                # Uniqueness for entities by scope_id + node_id
                await session.run(
                    f"CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (n:`{self._entity_label}`) "
                    "REQUIRE (n.scope_id, n.node_id) IS UNIQUE"
                )
                await session.run(
                    f"CREATE INDEX entity_name IF NOT EXISTS FOR (n:`{self._entity_label}`) ON (n.scope_id, n.name)"
                )
                await session.run(
                    f"CREATE INDEX entity_node_id IF NOT EXISTS FOR (n:`{self._entity_label}`) ON (n.scope_id, n.node_id)"
                )

                # Episodes as nodes
                await session.run(
                    "CREATE CONSTRAINT episode_unique IF NOT EXISTS FOR (e:`__Episode__`) "
                    "REQUIRE (e.scope_id, e.episode_id) IS UNIQUE"
                )
                await session.run(
                    "CREATE INDEX episode_scope IF NOT EXISTS FOR (e:`__Episode__`) ON (e.scope_id)"
                )
                await session.run(
                    "CREATE INDEX episode_scope_created_at IF NOT EXISTS FOR (e:`__Episode__`) ON (e.scope_id, e.created_at)"
                )

                # Fulltext indexes (best-effort): improve keyword retrieval quality beyond CONTAINS.
                # These are optional because some Neo4j deployments disable fulltext procedures.
                await session.run(
                    f"""
                    CREATE FULLTEXT INDEX `{self._cfg.fulltext_entity_index_name}` IF NOT EXISTS
                    FOR (n:`{self._entity_label}`) ON EACH [n.name, n.summary, n.attributes_text]
                    """
                )
                await session.run(
                    f"""
                    CREATE FULLTEXT INDEX `{self._cfg.fulltext_edge_index_name}` IF NOT EXISTS
                    FOR ()-[r]-() ON EACH [r.fact]
                    """
                )

                # Relationship indexes (Neo4j 5+). Best-effort: on some editions/versions,
                # relationship property indexes may not be available.
                await session.run(
                    "CREATE INDEX edge_scope_edge_id IF NOT EXISTS FOR ()-[r]-() ON (r.scope_id, r.edge_id)"
                )
                await session.run(
                    "CREATE INDEX edge_scope_valid_at IF NOT EXISTS FOR ()-[r]-() ON (r.scope_id, r.valid_at)"
                )
                await session.run(
                    "CREATE INDEX edge_scope_invalid_at IF NOT EXISTS FOR ()-[r]-() ON (r.scope_id, r.invalid_at)"
                )

                # Vector index for node semantic search (best-effort; requires Neo4j 5.11+ with vector indexing).
                if self._cfg.vector_dimensions:
                    dims = int(self._cfg.vector_dimensions)
                    await session.run(
                        f"""
                        CREATE VECTOR INDEX `{self._cfg.vector_index_name}` IF NOT EXISTS
                        FOR (n:`{self._entity_label}`) ON (n.name_embedding)
                        OPTIONS {{
                          indexConfig: {{
                            `vector.dimensions`: {dims},
                            `vector.similarity_function`: 'cosine'
                          }}
                        }}
                        """,
                    )

                # Community layer schema (best-effort).
                await session.run(
                    "CREATE CONSTRAINT community_unique IF NOT EXISTS FOR (c:`__Community__`) "
                    "REQUIRE (c.scope_id, c.community_id) IS UNIQUE"
                )
                await session.run(
                    "CREATE INDEX community_scope IF NOT EXISTS FOR (c:`__Community__`) ON (c.scope_id)"
                )
                await session.run(
                    "CREATE INDEX community_member_count IF NOT EXISTS FOR (c:`__Community__`) ON (c.scope_id, c.member_count)"
                )
                await session.run(
                    "CREATE INDEX has_member_scope IF NOT EXISTS FOR ()-[r:HAS_MEMBER]-() ON (r.scope_id)"
                )
        except Exception:
            # If the user runs an older Neo4j or lacks permissions, we still want basic functionality.
            pass
        self._initialized = True

    async def upsert_communities(self, scope_id: str, communities: List[GraphCommunity]) -> int:
        """Insert/update community nodes for a scope."""
        await self.initialize()
        if not communities:
            return 0
        rows: List[Dict[str, Any]] = []
        for c in communities:
            rows.append(
                {
                    "scope_id": scope_id,
                    "community_id": c.community_id,
                    "name": c.name,
                    "summary": c.summary,
                    "member_count": int(c.member_count),
                    "updated_at": self._dt(c.updated_at) or datetime.now(timezone.utc),
                    "name_embedding": (c.name_embedding if c.name_embedding is not None else []),
                    "summary_embedding": (c.summary_embedding if c.summary_embedding is not None else []),
                }
            )
        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                UNWIND $rows AS row
                MERGE (c:`__Community__` {scope_id: row.scope_id, community_id: row.community_id})
                SET c.name = row.name,
                    c.summary = row.summary,
                    c.member_count = row.member_count,
                    c.updated_at = row.updated_at,
                    c.name_embedding = row.name_embedding,
                    c.summary_embedding = row.summary_embedding
                """,
                rows=rows,
            )
        return len(rows)

    async def upsert_memberships(self, scope_id: str, memberships: List[tuple[str, str]]) -> int:
        """Insert membership edges (community -> entity) for a scope."""
        await self.initialize()
        if not memberships:
            return 0
        rows: List[Dict[str, Any]] = []
        now = datetime.now(timezone.utc)
        for community_id, node_id in memberships:
            if not community_id or not node_id:
                continue
            rows.append(
                {
                    "scope_id": scope_id,
                    "community_id": community_id,
                    "node_id": node_id,
                    "created_at": now,
                }
            )
        if not rows:
            return 0
        async with self._driver.session(database=self._database) as session:
            await session.run(
                f"""
                UNWIND $rows AS row
                MATCH (c:`__Community__` {{scope_id: row.scope_id, community_id: row.community_id}})
                MATCH (n:`{self._entity_label}` {{scope_id: row.scope_id, node_id: row.node_id}})
                MERGE (c)-[r:HAS_MEMBER {{scope_id: row.scope_id, community_id: row.community_id, node_id: row.node_id}}]->(n)
                SET r.created_at = coalesce(r.created_at, row.created_at)
                """,
                rows=rows,
            )
        return len(rows)

    async def get_communities_for_nodes(
        self,
        scope_id: str,
        node_ids: List[str],
        *,
        limit: int = 10,
    ) -> List[GraphCommunity]:
        """Return communities covering `node_ids`, ranked by overlap."""
        await self.initialize()
        want = [x for x in (node_ids or []) if x]
        if not want or limit <= 0:
            return []
        async with self._driver.session(database=self._database) as session:
            res = await session.run(
                f"""
                MATCH (c:`__Community__` {{scope_id: $scope_id}})-[:HAS_MEMBER]->(n:`{self._entity_label}` {{scope_id: $scope_id}})
                WHERE n.node_id IN $node_ids
                WITH c, count(DISTINCT n) AS overlap
                RETURN c.community_id AS community_id,
                       c.name AS name,
                       c.summary AS summary,
                       c.member_count AS member_count,
                       c.updated_at AS updated_at,
                       c.name_embedding AS name_embedding,
                       c.summary_embedding AS summary_embedding,
                       overlap
                ORDER BY overlap DESC, c.member_count DESC, c.updated_at DESC
                LIMIT $limit
                """,
                scope_id=scope_id,
                node_ids=want,
                limit=int(limit),
            )
            out: List[GraphCommunity] = []
            async for row in res:
                out.append(
                    GraphCommunity(
                        community_id=row.get("community_id"),
                        scope_id=scope_id,
                        name=row.get("name") or "",
                        summary=row.get("summary") or "",
                        member_count=int(row.get("member_count") or 0),
                        updated_at=row.get("updated_at") or datetime.now(timezone.utc),
                        name_embedding=list(row.get("name_embedding") or []) or None,
                        summary_embedding=list(row.get("summary_embedding") or []) or None,
                        overlap=int(row.get("overlap") or 0),
                    )
                )
            return out

    async def delete_communities(self, scope_id: str) -> int:
        """Delete all community nodes + membership edges for a scope."""
        await self.initialize()
        async with self._driver.session(database=self._database) as session:
            res = await session.run(
                """
                MATCH (c:`__Community__` {scope_id: $scope_id})
                DETACH DELETE c
                RETURN count(*) AS n
                """,
                scope_id=scope_id,
            )
            row = await res.single()
            try:
                return int(row.get("n") if row else 0)
            except Exception:
                return 0

    def _sanitize_label(self, value: str) -> str:
        v = (value or "").strip()
        if not v:
            return ""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", v):
            return ""
        return v

    def _sanitize_rel_type(self, value: str) -> str:
        v = (value or "").strip().upper()
        if not v:
            return ""
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", v):
            return ""
        return v

    def _dt(self, value: Optional[datetime]) -> Optional[datetime]:
        """Normalize datetimes to UTC (Neo4j expects timezone-aware values)."""
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def _is_far_future(self, value: Any) -> bool:
        try:
            if isinstance(value, datetime):
                return (value.year >= 9999)
        except Exception:
            return False
        return False

    def _json_dumps(self, value: Any) -> str:
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps(str(value))

    def _json_loads(self, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                loaded = json.loads(value)
                return loaded if isinstance(loaded, dict) else {}
            except Exception:
                return {}
        return {}

    async def add_episodes(self, scope_id, episodes):
        """Upsert episodes as `__Episode__` nodes keyed by (scope_id, episode_id)."""
        await self.initialize()
        if not episodes:
            return 0

        rows: List[Dict[str, Any]] = []
        for ep in episodes:
            rows.append(
                {
                    "scope_id": scope_id,
                    "episode_id": ep.episode_id,
                    "content": ep.content,
                    "content_type": ep.content_type,
                    "created_at": self._dt(ep.created_at),
                    "metadata_json": self._json_dumps(ep.metadata or {}),
                }
            )

        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                UNWIND $rows AS row
                MERGE (e:`__Episode__` {scope_id: row.scope_id, episode_id: row.episode_id})
                SET e.content = row.content,
                    e.content_type = row.content_type,
                    e.created_at = row.created_at,
                    e.metadata_json = row.metadata_json
                """,
                rows=rows,
            )
        return len(rows)

    async def upsert_nodes(self, scope_id, nodes):
        """Upsert `__Entity__` nodes keyed by (scope_id, node_id)."""
        await self.initialize()
        if not nodes:
            return 0

        # NOTE: Cypher cannot parameterize labels. To batch safely without APOC,
        # we group rows by the *set of labels* so each UNWIND query has a constant label clause.
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        labels_clause_by_key: Dict[str, str] = {}

        for n in nodes:
            labels = [self._sanitize_label(x) for x in (n.labels or [])]
            labels = [x for x in labels if x]
            label_key = "|".join(sorted(labels))
            labels_clause_by_key.setdefault(label_key, "".join([f":`{x}`" for x in labels]))

            attributes_json = self._json_dumps(n.attributes or {})
            grouped[label_key].append(
                {
                    "scope_id": scope_id,
                    "node_id": n.node_id,
                    "name": n.name,
                    "summary": n.summary,
                    "attributes_json": attributes_json,
                    "attributes_text": attributes_json,
                    "name_embedding": (n.name_embedding if n.name_embedding is not None else []),
                }
            )

        async with self._driver.session(database=self._database) as session:
            for label_key, rows in grouped.items():
                labels_clause = labels_clause_by_key.get(label_key, "")
                await session.run(
                    f"""
                    UNWIND $rows AS row
                    MERGE (node:`{self._entity_label}` {{scope_id: row.scope_id, node_id: row.node_id}})
                    SET node.name = row.name,
                        node.summary = coalesce(row.summary, ""),
                        node.attributes_json = row.attributes_json,
                        node.attributes_text = row.attributes_text,
                        node.name_embedding = row.name_embedding,
                        node.updated_at = datetime()
                    SET node{labels_clause}
                    """,
                    rows=rows,
                )

        return len(nodes)

    async def upsert_edges(self, scope_id, edges):
        """Upsert relationships between entity nodes keyed by (scope_id, edge_id)."""
        await self.initialize()
        if not edges:
            return 0

        # NOTE: Cypher cannot parameterize relationship types. We group edges by sanitized type
        # and UNWIND each group with a constant relationship type in the query.
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in edges:
            rel_type = self._sanitize_rel_type(e.edge_type)
            if not rel_type:
                continue
            by_type[rel_type].append(
                {
                    "scope_id": scope_id,
                    "edge_id": e.edge_id,
                    "source_node_id": e.source_node_id,
                    "target_node_id": e.target_node_id,
                    "fact": (e.fact or ""),
                    "labels": (e.labels or []),
                    "attributes_json": self._json_dumps(e.attributes or {}),
                    "valid_at": self._dt(e.valid_at),
                    "invalid_at": self._dt(e.invalid_at),
                }
            )

        async with self._driver.session(database=self._database) as session:
            for rel_type, rows in by_type.items():
                await session.run(
                    f"""
                    UNWIND $rows AS row
                    MATCH (s:`{self._entity_label}` {{scope_id: row.scope_id, node_id: row.source_node_id}})
                    MATCH (t:`{self._entity_label}` {{scope_id: row.scope_id, node_id: row.target_node_id}})
                    MERGE (s)-[r:`{rel_type}` {{scope_id: row.scope_id, edge_id: row.edge_id}}]->(t)
                    SET r.fact = coalesce(row.fact, ""),
                        r.labels = row.labels,
                        r.attributes_json = row.attributes_json,
                        r.valid_at = coalesce(row.valid_at, datetime()),
                        r.invalid_at = coalesce(row.invalid_at, datetime('9999-12-31T23:59:59Z')),
                        r.updated_at = datetime()
                    """,
                    rows=rows,
                )

        return len(edges)

    async def get_edges_by_ids(self, scope_id: str, edge_ids: List[str]) -> List[GraphEdge]:
        """Fetch edges by id within a scope."""
        await self.initialize()
        want = [x for x in (edge_ids or []) if x]
        if not want:
            return []

        async with self._driver.session(database=self._database) as session:
            result = await session.run(
                f"""
                MATCH (s:`{self._entity_label}` {{scope_id: $scope_id}})-[r]->(t:`{self._entity_label}` {{scope_id: $scope_id}})
                WHERE r.edge_id IN $edge_ids
                RETURN r.edge_id AS edge_id,
                       s.node_id AS source_node_id,
                       t.node_id AS target_node_id,
                       type(r) AS edge_type,
                       r.fact AS fact,
                       r.labels AS labels,
                       r.attributes_json AS attributes_json,
                       r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                """,
                scope_id=scope_id,
                edge_ids=want,
            )
            edges: List[GraphEdge] = []
            async for row in result:
                invalid_at = row.get("invalid_at")
                if self._is_far_future(invalid_at):
                    invalid_at = None
                edges.append(
                    GraphEdge(
                        edge_id=row.get("edge_id"),
                        scope_id=scope_id,
                        source_node_id=row.get("source_node_id"),
                        target_node_id=row.get("target_node_id"),
                        edge_type=row.get("edge_type"),
                        fact=(row.get("fact") or None),
                        labels=list(row.get("labels") or []),
                        attributes=self._json_loads(row.get("attributes_json")),
                        valid_at=row.get("valid_at"),
                        invalid_at=invalid_at,
                    )
                )
            return edges

    async def invalidate_edges(
        self,
        scope_id: str,
        edge_ids: List[str],
        *,
        invalid_at: datetime,
    ) -> int:
        """Mark a list of edges invalid by setting `invalid_at`."""

        await self.initialize()
        want = [x for x in (edge_ids or []) if x]
        if not want:
            return 0
        ts = self._dt(invalid_at) or datetime.now(timezone.utc)
        async with self._driver.session(database=self._database) as session:
            res = await session.run(
                f"""
                MATCH (s:`{self._entity_label}` {{scope_id: $scope_id}})-[r]->(t:`{self._entity_label}` {{scope_id: $scope_id}})
                WHERE r.edge_id IN $edge_ids
                SET r.invalid_at = $invalid_at,
                    r.updated_at = datetime()
                RETURN count(r) AS n
                """,
                scope_id=scope_id,
                edge_ids=want,
                invalid_at=ts,
            )
            row = await res.single()
            try:
                return int(row.get("n") if row else 0)
            except Exception:
                return 0

    async def delete_scope(self, scope_id: str) -> int:
        """Delete all graph objects for a scope_id (episodes, entities, and their relationships)."""
        await self.initialize()
        async with self._driver.session(database=self._database) as session:
            # Delete episodes for scope
            res1 = await session.run(
                """
                MATCH (e:`__Episode__` {scope_id: $scope_id})
                DETACH DELETE e
                RETURN count(*) AS n
                """,
                scope_id=scope_id,
            )
            row1 = await res1.single()
            n1 = int(row1.get("n") if row1 else 0)

            # Delete entities for scope (relationships are deleted by DETACH)
            res2 = await session.run(
                f"""
                MATCH (n:`{self._entity_label}` {{scope_id: $scope_id}})
                DETACH DELETE n
                RETURN count(*) AS n
                """,
                scope_id=scope_id,
            )
            row2 = await res2.single()
            n2 = int(row2.get("n") if row2 else 0)

            # Delete enhanced index for scope
            await session.run(
                """
                MATCH (idx:`__EnhancedIndex__` {scope_id: $scope_id})
                DETACH DELETE idx
                """,
                scope_id=scope_id,
            )

            return n1 + n2

    async def save_enhanced_index(self, scope_id: str, index: EnhancedMemoryIndex) -> None:
        """Persist the enhanced memory index as a JSON property on a dedicated node."""
        await self.initialize()
        data_json = json.dumps(index.to_dict())
        async with self._driver.session(database=self._database) as session:
            await session.run(
                """
                MERGE (idx:`__EnhancedIndex__` {scope_id: $scope_id})
                SET idx.data = $data_json, idx.updated_at = datetime()
                """,
                scope_id=scope_id,
                data_json=data_json,
            )

    async def load_enhanced_index(self, scope_id: str) -> Optional[EnhancedMemoryIndex]:
        """Load the enhanced memory index from a dedicated node."""
        await self.initialize()
        async with self._driver.session(database=self._database) as session:
            res = await session.run(
                """
                MATCH (idx:`__EnhancedIndex__` {scope_id: $scope_id})
                RETURN idx.data AS data
                """,
                scope_id=scope_id,
            )
            row = await res.single()
            if not row:
                return None
            data_json = row.get("data")
            if not data_json:
                return None
            try:
                data = json.loads(data_json)
                return EnhancedMemoryIndex.from_dict(data)
            except Exception:
                return None

    async def search_nodes_semantic(
        self,
        scope_id: str,
        query_vector: List[float],
        *,
        limit: int = 20,
        filters: Optional[GraphSearchFilters] = None,
    ) -> List[GraphNode]:
        """Return nodes ranked by vector similarity (falls back to keyword search if unavailable)."""

        await self.initialize()
        filters = filters or GraphSearchFilters()
        limit = max(0, int(limit))

        node_labels = [self._sanitize_label(x) for x in (filters.node_labels or [])]
        node_labels = [x for x in node_labels if x]
        node_ids = [x for x in (filters.node_ids or []) if x]

        if not self._cfg.vector_dimensions:
            # No configured vector index. Fall back to keyword search.
            out = await self.search(
                scope_id,
                "",
                scope="nodes",
                limit=limit,
                filters=filters,
            )
            return out.nodes

        async with self._driver.session(database=self._database) as session:
            async def run_and_collect() -> List[GraphNode]:
                result = await session.run(
                    """
                    CALL db.index.vector.queryNodes($index_name, $k, $vector) YIELD node, score
                    WHERE node.scope_id = $scope_id
                      AND (size($node_labels) = 0 OR any(l IN $node_labels WHERE l IN labels(node)))
                      AND (size($node_ids) = 0 OR node.node_id IN $node_ids)
                    RETURN node.node_id AS node_id,
                           node.name AS name,
                           labels(node) AS labels,
                           node.attributes_json AS attributes_json,
                           node.summary AS summary,
                           node.name_embedding AS name_embedding
                    ORDER BY score DESC
                    LIMIT $k
                    """,
                    index_name=self._cfg.vector_index_name,
                    k=limit,
                    vector=[float(x) for x in (query_vector or [])],
                    scope_id=scope_id,
                    node_labels=node_labels,
                    node_ids=node_ids,
                )
                nodes: List[GraphNode] = []
                async for row in result:
                    nodes.append(
                        GraphNode(
                            node_id=row.get("node_id"),
                            scope_id=scope_id,
                            name=row.get("name") or "",
                            labels=list(row.get("labels") or []),
                            attributes=self._json_loads(row.get("attributes_json")),
                            summary=row.get("summary"),
                            name_embedding=list(row.get("name_embedding") or []) or None,
                        )
                    )
                return nodes

            try:
                return await run_and_collect()
            except Exception as e:
                # If the index doesn't exist (or isn't supported), fall back to keyword search.
                msg = str(e)
                if "no such vector schema index" in msg.lower():
                    try:
                        # Best-effort: attempt to create the vector index and retry once.
                        dims = int(self._cfg.vector_dimensions or 0)
                        if dims > 0:
                            await session.run(
                                f"""
                                CREATE VECTOR INDEX `{self._cfg.vector_index_name}` IF NOT EXISTS
                                FOR (n:`{self._entity_label}`) ON (n.name_embedding)
                                OPTIONS {{
                                  indexConfig: {{
                                    `vector.dimensions`: {dims},
                                    `vector.similarity_function`: 'cosine'
                                  }}
                                }}
                                """
                            )
                            return await run_and_collect()
                        else:
                            raise e
                    except Exception:
                        out = await self.search(
                            scope_id,
                            "",
                            scope="nodes",
                            limit=limit,
                            filters=filters,
                        )
                        return out.nodes
                else:
                    out = await self.search(
                        scope_id,
                        "",
                        scope="nodes",
                        limit=limit,
                        filters=filters,
                    )
                    return out.nodes

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
        """Keyword search over nodes/edges/episodes with optional label/type and temporal filtering."""
        await self.initialize()
        filters = filters or GraphSearchFilters()

        as_of = filters.as_of
        if as_of is None:
            as_of = datetime.now(timezone.utc)
        else:
            as_of = self._dt(as_of) or datetime.now(timezone.utc)

        q = (query or "").strip().lower()
        words = [w for w in re.split(r"[^a-z0-9_]+", q) if len(w) > 2]
        limit = max(0, int(limit))
        node_ids = [x for x in (filters.node_ids or []) if x]
        edge_ids = [x for x in (filters.edge_ids or []) if x]

        async with self._driver.session(database=self._database) as session:
            if scope == "episodes":
                result = await session.run(
                    """
                    MATCH (e:`__Episode__` {scope_id: $scope_id})
                    WHERE
                      $q = "" OR
                      any(w IN $words WHERE toLower(coalesce(e.content, "")) CONTAINS w)
                    RETURN e.episode_id AS episode_id,
                           e.content AS content,
                           e.content_type AS content_type,
                           e.created_at AS created_at,
                           e.metadata_json AS metadata_json
                    ORDER BY e.created_at DESC
                    LIMIT $limit
                    """,
                    scope_id=scope_id,
                    q=q,
                    words=words,
                    limit=limit,
                )
                episodes: List[GraphEpisode] = []
                async for row in result:
                    episodes.append(
                        GraphEpisode(
                            episode_id=row.get("episode_id"),
                            scope_id=scope_id,
                            content=row.get("content") or "",
                            content_type=row.get("content_type") or "text",
                            created_at=row.get("created_at") or datetime.now(timezone.utc),
                            metadata=self._json_loads(row.get("metadata_json")),
                        )
                    )
                return GraphSearchResult(episodes=episodes)

            if scope == "nodes":
                node_labels = [self._sanitize_label(x) for x in (filters.node_labels or [])]
                node_labels = [x for x in node_labels if x]

                # Prefer fulltext search for keyword queries when available. It yields much better
                # results than substring matching, but we keep the fallback for portability.
                if q:
                    try:
                        ft = await session.run(
                            """
                            CALL db.index.fulltext.queryNodes($index_name, $q) YIELD node, score
                            WHERE node.scope_id = $scope_id
                              AND (size($node_labels) = 0 OR any(l IN $node_labels WHERE l IN labels(node)))
                              AND (size($node_ids) = 0 OR node.node_id IN $node_ids)
                            RETURN node.node_id AS node_id,
                                   node.name AS name,
                                   labels(node) AS labels,
                                   node.attributes_json AS attributes_json,
                                   node.summary AS summary,
                                   node.name_embedding AS name_embedding
                            ORDER BY score DESC
                            LIMIT $limit
                            """,
                            index_name=self._cfg.fulltext_entity_index_name,
                            q=q,
                            scope_id=scope_id,
                            node_labels=node_labels,
                            node_ids=node_ids,
                            limit=limit,
                        )
                        nodes: List[GraphNode] = []
                        async for row in ft:
                            nodes.append(
                                GraphNode(
                                    node_id=row.get("node_id"),
                                    scope_id=scope_id,
                                    name=row.get("name") or "",
                                    labels=list(row.get("labels") or []),
                                    attributes=self._json_loads(row.get("attributes_json")),
                                    summary=row.get("summary"),
                                    name_embedding=list(row.get("name_embedding") or []) or None,
                                )
                            )
                        if nodes:
                            return GraphSearchResult(nodes=nodes)
                    except Exception:
                        # Fall back to substring matching below.
                        pass

                result = await session.run(
                    f"""
                    MATCH (n:`{self._entity_label}` {{scope_id: $scope_id}})
                    WHERE
                      ($q = "" OR any(w IN $words WHERE
                        toLower(coalesce(n.name, "")) CONTAINS w OR
                        toLower(coalesce(n.summary, "")) CONTAINS w OR
                        toLower(coalesce(n.attributes_text, "")) CONTAINS w
                      ))
                      AND (size($node_labels) = 0 OR any(l IN $node_labels WHERE l IN labels(n)))
                      AND (size($node_ids) = 0 OR n.node_id IN $node_ids)
                    RETURN n.node_id AS node_id,
                           n.name AS name,
                           labels(n) AS labels,
                           n.attributes_json AS attributes_json,
                           n.summary AS summary,
                           n.name_embedding AS name_embedding
                    LIMIT $limit
                    """,
                    scope_id=scope_id,
                    q=q,
                    words=words,
                    node_labels=node_labels,
                    node_ids=node_ids,
                    limit=limit,
                )
                nodes: List[GraphNode] = []
                async for row in result:
                    nodes.append(
                        GraphNode(
                            node_id=row.get("node_id"),
                            scope_id=scope_id,
                            name=row.get("name") or "",
                            labels=list(row.get("labels") or []),
                            attributes=self._json_loads(row.get("attributes_json")),
                            summary=row.get("summary"),
                            name_embedding=list(row.get("name_embedding") or []) or None,
                        )
                    )
                return GraphSearchResult(nodes=nodes)

            edge_types = [self._sanitize_rel_type(x) for x in (filters.edge_types or [])]
            edge_types = [x for x in edge_types if x]

            # Prefer relationship fulltext search when possible. This is best-effort: some Neo4j
            # deployments disable the procedure or older versions may not support relationship fulltext.
            if q:
                try:
                    ft = await session.run(
                        """
                        CALL db.index.fulltext.queryRelationships($index_name, $q) YIELD relationship, score
                        WITH relationship AS r, score
                        WHERE r.scope_id = $scope_id
                          AND ($center_node_id IS NULL OR startNode(r).node_id = $center_node_id OR endNode(r).node_id = $center_node_id)
                          AND (size($edge_types) = 0 OR type(r) IN $edge_types)
                          AND (size($edge_ids) = 0 OR r.edge_id IN $edge_ids)
                          AND (NOT $valid_only OR r.invalid_at > $as_of)
                        RETURN r.edge_id AS edge_id,
                               startNode(r).node_id AS source_node_id,
                               endNode(r).node_id AS target_node_id,
                               type(r) AS edge_type,
                               r.fact AS fact,
                               r.labels AS labels,
                               r.attributes_json AS attributes_json,
                               r.valid_at AS valid_at,
                               r.invalid_at AS invalid_at
                        ORDER BY score DESC
                        LIMIT $limit
                        """,
                        index_name=self._cfg.fulltext_edge_index_name,
                        q=q,
                        scope_id=scope_id,
                        center_node_id=center_node_id,
                        edge_types=edge_types,
                        edge_ids=edge_ids,
                        valid_only=bool(filters.valid_only),
                        as_of=as_of,
                        limit=limit,
                    )
                    edges: List[GraphEdge] = []
                    async for row in ft:
                        invalid_at = row.get("invalid_at")
                        if self._is_far_future(invalid_at):
                            invalid_at = None
                        edges.append(
                            GraphEdge(
                                edge_id=row.get("edge_id"),
                                scope_id=scope_id,
                                source_node_id=row.get("source_node_id"),
                                target_node_id=row.get("target_node_id"),
                                edge_type=row.get("edge_type"),
                                fact=(row.get("fact") or None),
                                labels=list(row.get("labels") or []),
                                attributes=self._json_loads(row.get("attributes_json")),
                                valid_at=row.get("valid_at"),
                                invalid_at=invalid_at,
                            )
                        )
                    if edges:
                        return GraphSearchResult(edges=edges)
                except Exception:
                    # Fall back to substring matching below.
                    pass

            result = await session.run(
                f"""
                MATCH (s:`{self._entity_label}` {{scope_id: $scope_id}})-[r]->(t:`{self._entity_label}` {{scope_id: $scope_id}})
                WHERE
                  ($center_node_id IS NULL OR s.node_id = $center_node_id OR t.node_id = $center_node_id)
                  AND (size($edge_types) = 0 OR type(r) IN $edge_types)
                  AND (NOT $valid_only OR r.invalid_at > $as_of)
                  AND (size($edge_ids) = 0 OR r.edge_id IN $edge_ids)
                  AND (
                    $q = "" OR
                    any(w IN $words WHERE
                      toLower(coalesce(r.fact, "")) CONTAINS w OR
                      toLower(type(r)) CONTAINS w
                    )
                  )
                RETURN r.edge_id AS edge_id,
                       s.node_id AS source_node_id,
                       t.node_id AS target_node_id,
                       type(r) AS edge_type,
                       r.fact AS fact,
                       r.labels AS labels,
                       r.attributes_json AS attributes_json,
                       r.valid_at AS valid_at,
                       r.invalid_at AS invalid_at
                LIMIT $limit
                """,
                scope_id=scope_id,
                q=q,
                words=words,
                edge_types=edge_types,
                valid_only=bool(filters.valid_only),
                as_of=as_of,
                edge_ids=edge_ids,
                center_node_id=center_node_id,
                limit=limit,
            )
            edges: List[GraphEdge] = []
            async for row in result:
                invalid_at = row.get("invalid_at")
                if self._is_far_future(invalid_at):
                    invalid_at = None
                edges.append(
                    GraphEdge(
                        edge_id=row.get("edge_id"),
                        scope_id=scope_id,
                        source_node_id=row.get("source_node_id"),
                        target_node_id=row.get("target_node_id"),
                        edge_type=row.get("edge_type"),
                        fact=(row.get("fact") or None),
                        labels=list(row.get("labels") or []),
                        attributes=self._json_loads(row.get("attributes_json")),
                        valid_at=row.get("valid_at"),
                        invalid_at=invalid_at,
                    )
                )
            return GraphSearchResult(edges=edges)


