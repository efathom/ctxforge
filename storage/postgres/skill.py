"""
PostgreSQL skill store implementation.

Provides persistent storage for skills with progressive disclosure support.
"""
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ctxforge.core.exceptions import StorageError
from ctxforge.core.skill import (
    Skill,
    SkillMatch,
    SkillMetadata,
    SkillRelationship,
    SkillRelationType,
    SkillScope,
)
from ctxforge.engine.registry import registry
from ctxforge.storage.connection import PostgresConfig, PostgresConnectionManager

# SQL statements for table creation
CREATE_SKILLS_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(64) NOT NULL,
    description VARCHAR(256) NOT NULL,
    scope VARCHAR(20) NOT NULL,
    scope_id VARCHAR(128) NOT NULL,
    content TEXT NOT NULL,
    triggers JSONB DEFAULT '[]',
    prerequisites JSONB DEFAULT '[]',
    allowed_tools JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{{}}',
    version VARCHAR(16) DEFAULT '1.0',
    category VARCHAR(64),
    tags JSONB DEFAULT '[]',
    when_to_use TEXT,
    effectiveness JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(name, scope, scope_id)
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_scope_lookup ON {table_name}(scope, scope_id);
CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}(name);
CREATE INDEX IF NOT EXISTS idx_{table_name}_category ON {table_name}(category);

-- Full-text search on description
CREATE INDEX IF NOT EXISTS idx_{table_name}_description_fts 
ON {table_name} USING GIN(to_tsvector('english', description));
"""

CREATE_SKILL_RELATIONSHIPS_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name}_relationships (
    id SERIAL PRIMARY KEY,
    source VARCHAR(64) NOT NULL,
    target VARCHAR(64) NOT NULL,
    relation_type VARCHAR(32) NOT NULL,
    reason TEXT DEFAULT '',
    confidence FLOAT DEFAULT 0.8,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(source, target, relation_type)
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_rel_source
    ON {table_name}_relationships(source);
CREATE INDEX IF NOT EXISTS idx_{table_name}_rel_target
    ON {table_name}_relationships(target);
"""


@registry.register_skill_store("postgres")
class PostgresSkillStore:
    """
    PostgreSQL-based skill store.

    Features:
    - Unique key constraint per (name, scope, scope_id)
    - JSONB storage for triggers, prerequisites, allowed_tools
    - Full-text search on description
    """

    def __init__(
        self,
        config: Optional[PostgresConfig] = None,
        connection_manager: Optional[PostgresConnectionManager] = None,
        table_name: str = "skills",
    ):
        """
        Initialize the PostgreSQL skill store.

        Args:
            config: PostgreSQL configuration
            connection_manager: Optional pre-existing connection manager
            table_name: Name of the table to use
        """
        self.config = config or PostgresConfig()
        self._manager = connection_manager or PostgresConnectionManager(self.config)
        self._owns_connection = connection_manager is None
        self._table_name = table_name
        self._initialized = False

    async def connect(self) -> None:
        """Connect to PostgreSQL."""
        if not self._manager.is_connected:
            await self._manager.connect()

    async def disconnect(self) -> None:
        """Disconnect from PostgreSQL."""
        if self._owns_connection and self._manager.is_connected:
            await self._manager.disconnect()

    async def close(self) -> None:
        """Lifecycle alias for ctxforge-managed teardown."""
        await self.disconnect()

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return

        await self.connect()

        sql = CREATE_SKILLS_TABLE.format(table_name=self._table_name)
        await self._manager.execute(sql)

        rel_sql = CREATE_SKILL_RELATIONSHIPS_TABLE.format(table_name=self._table_name)
        await self._manager.execute(rel_sql)

        await self._migrate_skill_columns()

        self._initialized = True

    async def _migrate_skill_columns(self) -> None:
        """Add columns introduced after the initial schema."""
        migrations = [
            "ALTER TABLE {t} ADD COLUMN IF NOT EXISTS category VARCHAR(64)",
            "ALTER TABLE {t} ADD COLUMN IF NOT EXISTS tags JSONB DEFAULT '[]'",
            "ALTER TABLE {t} ADD COLUMN IF NOT EXISTS when_to_use TEXT",
            "ALTER TABLE {t} ADD COLUMN IF NOT EXISTS effectiveness JSONB",
        ]
        for stmt in migrations:
            try:
                await self._manager.execute(stmt.format(t=self._table_name))
            except Exception:
                pass

    def _parse_json_field(self, value: Any, default: Any = None) -> Any:
        """Parse a JSON field from database."""
        if value is None:
            return default if default is not None else []
        if isinstance(value, str):
            return json.loads(value)
        return value

    def _deserialize_skill(self, row: Any) -> Skill:
        """Deserialize skill from database row."""
        created_at = row["created_at"]
        updated_at = row["updated_at"]

        return Skill(
            name=row["name"],
            description=row["description"],
            scope=SkillScope(row["scope"]),
            scope_id=row["scope_id"],
            content=row["content"],
            triggers=self._parse_json_field(row["triggers"], []),
            prerequisites=self._parse_json_field(row["prerequisites"], []),
            allowed_tools=self._parse_json_field(row["allowed_tools"], []),
            metadata=self._parse_json_field(row["metadata"], {}),
            version=row["version"] or "1.0",
            category=row.get("category"),
            tags=self._parse_json_field(row.get("tags"), []),
            when_to_use=row.get("when_to_use"),
            effectiveness=self._parse_json_field(row.get("effectiveness")),
            created_at=(created_at if isinstance(created_at, datetime)
                        else datetime.fromisoformat(str(created_at))),
            updated_at=(updated_at if isinstance(updated_at, datetime)
                        else datetime.fromisoformat(str(updated_at))),
        )

    def _deserialize_metadata(self, row: Any) -> SkillMetadata:
        """Deserialize skill metadata from database row."""
        return SkillMetadata(
            name=row["name"],
            description=row["description"],
            scope=SkillScope(row["scope"]),
            scope_id=row["scope_id"],
            triggers=self._parse_json_field(row["triggers"], []),
            version=row["version"] or "1.0",
            category=row.get("category"),
            tags=self._parse_json_field(row.get("tags"), []),
            when_to_use=row.get("when_to_use"),
        )

    async def save(self, skill: Skill) -> None:
        """Save a skill. Updates if name already exists in scope."""
        await self.initialize()

        # Generate ID from name, scope, scope_id
        skill_id = f"{skill.scope.value}:{skill.scope_id}:{skill.name}"

        query = f"""
            INSERT INTO {self._table_name}
            (id, name, description, scope, scope_id, content, triggers,
             prerequisites, allowed_tools, metadata, version,
             category, tags, when_to_use, effectiveness,
             created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                    $12, $13, $14, $15, $16, $17)
            ON CONFLICT (name, scope, scope_id) DO UPDATE SET
                description = EXCLUDED.description,
                content = EXCLUDED.content,
                triggers = EXCLUDED.triggers,
                prerequisites = EXCLUDED.prerequisites,
                allowed_tools = EXCLUDED.allowed_tools,
                metadata = EXCLUDED.metadata,
                version = EXCLUDED.version,
                category = EXCLUDED.category,
                tags = EXCLUDED.tags,
                when_to_use = EXCLUDED.when_to_use,
                effectiveness = EXCLUDED.effectiveness,
                updated_at = EXCLUDED.updated_at
        """

        try:
            await self._manager.execute(
                query,
                skill_id,
                skill.name,
                skill.description,
                skill.scope.value,
                skill.scope_id,
                skill.content,
                json.dumps(skill.triggers),
                json.dumps(skill.prerequisites),
                json.dumps(skill.allowed_tools),
                json.dumps(skill.metadata),
                skill.version,
                skill.category,
                json.dumps(skill.tags),
                skill.when_to_use,
                json.dumps(skill.effectiveness) if skill.effectiveness else None,
                skill.created_at,
                skill.updated_at,
            )
        except Exception as e:
            raise StorageError(f"Failed to save skill: {e}") from e

    async def get(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> Optional[Skill]:
        """Get full skill by name, scope, and scope_id."""
        await self.initialize()

        query = f"""
            SELECT * FROM {self._table_name}
            WHERE name = $1 AND scope = $2 AND scope_id = $3
        """

        row = await self._manager.fetchrow(query, name, scope.value, scope_id)

        if not row:
            return None

        return self._deserialize_skill(row)

    async def get_metadata(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> Optional[SkillMetadata]:
        """Get only skill metadata (for progressive disclosure)."""
        await self.initialize()

        query = f"""
            SELECT name, description, scope, scope_id, triggers, version,
                   category, tags, when_to_use
            FROM {self._table_name}
            WHERE name = $1 AND scope = $2 AND scope_id = $3
        """

        row = await self._manager.fetchrow(query, name, scope.value, scope_id)

        if not row:
            return None

        return self._deserialize_metadata(row)

    async def list_metadata(
        self,
        scope: SkillScope,
        scope_id: str
    ) -> List[SkillMetadata]:
        """List all skill metadata for a given scope."""
        await self.initialize()

        query = f"""
            SELECT name, description, scope, scope_id, triggers, version,
                   category, tags, when_to_use
            FROM {self._table_name}
            WHERE scope = $1 AND scope_id = $2
            ORDER BY name ASC
        """

        rows = await self._manager.fetch(query, scope.value, scope_id)
        return [self._deserialize_metadata(row) for row in rows]

    async def list_all_metadata(
        self,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[SkillMetadata]:
        """
        List metadata for all available skills with scope layering.

        Returns skills from BASE, then USER (if user_id provided),
        then PROJECT (if project_id provided). Later scopes override
        earlier ones by name.
        """
        await self.initialize()

        # Build query to get all applicable skills
        conditions = ["scope = 'base'"]
        params: List[Any] = []
        param_idx = 1

        if user_id:
            conditions.append(f"(scope = 'user' AND scope_id = ${param_idx})")
            params.append(user_id)
            param_idx += 1

        if project_id:
            conditions.append(f"(scope = 'project' AND scope_id = ${param_idx})")
            params.append(project_id)
            param_idx += 1

        where_clause = " OR ".join(conditions)

        query = f"""
            SELECT name, description, scope, scope_id, triggers, version,
                category, tags, when_to_use,
                CASE scope 
                    WHEN 'project' THEN 2 
                    WHEN 'user' THEN 1 
                    ELSE 0 
                END as scope_priority
            FROM {self._table_name}
            WHERE {where_clause}
            ORDER BY name ASC, scope_priority DESC
        """

        rows = await self._manager.fetch(query, *params)

        # Deduplicate by name (higher scope priority wins)
        skills_by_name: Dict[str, SkillMetadata] = {}
        for row in rows:
            name = row["name"]
            if name not in skills_by_name:
                skills_by_name[name] = self._deserialize_metadata(row)

        return sorted(skills_by_name.values(), key=lambda s: s.name)

    async def delete(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str
    ) -> bool:
        """Delete a skill. Returns True if deleted."""
        await self.initialize()

        query = f"""
            DELETE FROM {self._table_name}
            WHERE name = $1 AND scope = $2 AND scope_id = $3
        """

        result = await self._manager.execute(query, name, scope.value, scope_id)
        return "DELETE 0" not in str(result)

    async def search_by_trigger(
        self,
        query_text: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> List[SkillMatch]:
        """Find skills that match a query based on triggers."""
        # Get all available skills with layering
        available = await self.list_all_metadata(user_id, project_id)

        matches: List[SkillMatch] = []
        query_lower = query_text.lower()

        for skill_meta in available:
            matched_trigger = skill_meta.matches_trigger(query_text)
            if matched_trigger:
                # Calculate confidence based on match quality
                trigger_lower = matched_trigger.lower()
                if trigger_lower == query_lower:
                    confidence = 1.0
                elif (query_lower.startswith(trigger_lower) or
                      query_lower.endswith(trigger_lower)):
                    confidence = 0.9
                else:
                    confidence = 0.7

                matches.append(SkillMatch(
                    skill=skill_meta,
                    confidence=confidence,
                    matched_trigger=matched_trigger,
                    match_reason=f"Trigger '{matched_trigger}' matched query",
                ))

        # Sort by confidence (descending)
        matches.sort(key=lambda m: -m.confidence)
        return matches

    async def count(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Count skills, optionally filtered by scope."""
        await self.initialize()

        conditions = []
        params: List[Any] = []
        param_idx = 1

        if scope is not None:
            conditions.append(f"scope = ${param_idx}")
            params.append(scope.value)
            param_idx += 1

        if scope_id is not None:
            conditions.append(f"scope_id = ${param_idx}")
            params.append(scope_id)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"

        query = f"SELECT COUNT(*) FROM {self._table_name} WHERE {where_clause}"
        result = await self._manager.fetchval(query, *params)

        return result or 0

    async def clear(
        self,
        scope: Optional[SkillScope] = None,
        scope_id: Optional[str] = None
    ) -> int:
        """Clear skills, optionally filtered by scope. Returns count deleted."""
        await self.initialize()

        # First get count
        count = await self.count(scope, scope_id)

        conditions = []
        params: List[Any] = []
        param_idx = 1

        if scope is not None:
            conditions.append(f"scope = ${param_idx}")
            params.append(scope.value)
            param_idx += 1

        if scope_id is not None:
            conditions.append(f"scope_id = ${param_idx}")
            params.append(scope_id)
            param_idx += 1

        if conditions:
            where_clause = " AND ".join(conditions)
            query = f"DELETE FROM {self._table_name} WHERE {where_clause}"
            await self._manager.execute(query, *params)
        else:
            query = f"TRUNCATE TABLE {self._table_name}"
            await self._manager.execute(query)

        return count

    # ------------------------------------------------------------------
    # Relationship methods
    # ------------------------------------------------------------------

    async def save_relationships(
        self, relationships: List[SkillRelationship]
    ) -> int:
        """Save skill relationships. Returns count saved."""
        await self.initialize()

        saved = 0
        for rel in relationships:
            query = f"""
                INSERT INTO {self._table_name}_relationships
                (source, target, relation_type, reason, confidence, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source, target, relation_type) DO UPDATE SET
                    reason = EXCLUDED.reason,
                    confidence = EXCLUDED.confidence
            """
            try:
                await self._manager.execute(
                    query,
                    rel.source,
                    rel.target,
                    rel.relation_type.value,
                    rel.reason,
                    rel.confidence,
                    datetime.now(timezone.utc),
                )
                saved += 1
            except Exception:
                pass
        return saved

    async def get_relationships(
        self, skill_name: str
    ) -> List[SkillRelationship]:
        """Get all relationships for a skill (as source or target)."""
        await self.initialize()

        query = f"""
            SELECT source, target, relation_type, reason, confidence
            FROM {self._table_name}_relationships
            WHERE source = $1 OR target = $1
        """
        rows = await self._manager.fetch(query, skill_name)
        return [
            SkillRelationship(
                source=r["source"],
                target=r["target"],
                relation_type=SkillRelationType(r["relation_type"]),
                reason=r["reason"] or "",
                confidence=r["confidence"],
            )
            for r in rows
        ]

    async def get_all_relationships(self) -> List[SkillRelationship]:
        """Get all stored relationships."""
        await self.initialize()

        query = f"""
            SELECT source, target, relation_type, reason, confidence
            FROM {self._table_name}_relationships
            ORDER BY source, target
        """
        rows = await self._manager.fetch(query)
        return [
            SkillRelationship(
                source=r["source"],
                target=r["target"],
                relation_type=SkillRelationType(r["relation_type"]),
                reason=r["reason"] or "",
                confidence=r["confidence"],
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Category / tag / effectiveness methods
    # ------------------------------------------------------------------

    async def search_by_category(
        self,
        category: str,
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Find skills matching a category."""
        all_meta = await self.list_all_metadata(user_id, project_id)
        return [m for m in all_meta if m.category == category]

    async def search_by_tags(
        self,
        tags: List[str],
        user_id: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> List[SkillMetadata]:
        """Find skills matching any of the provided tags."""
        all_meta = await self.list_all_metadata(user_id, project_id)
        tag_set = {t.lower() for t in tags}
        return [
            m for m in all_meta
            if tag_set & {t.lower() for t in m.tags}
        ]

    async def update_effectiveness(
        self,
        name: str,
        scope: SkillScope,
        scope_id: str,
        metrics: Dict[str, Any],
    ) -> bool:
        """Update effectiveness metrics for a skill. Returns True if updated."""
        await self.initialize()

        # Merge with existing effectiveness JSON
        existing = await self.get(name, scope, scope_id)
        if existing is None:
            return False

        merged = existing.effectiveness or {}
        merged.update(metrics)

        query = f"""
            UPDATE {self._table_name}
            SET effectiveness = $1, updated_at = $2
            WHERE name = $3 AND scope = $4 AND scope_id = $5
        """
        result = await self._manager.execute(
            query,
            json.dumps(merged),
            datetime.now(timezone.utc),
            name,
            scope.value,
            scope_id,
        )
        return "UPDATE 0" not in str(result)
