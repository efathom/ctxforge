"""
PostgreSQL semantic model store implementation.

Provides persistent storage for semantic models with JSONB support.
"""

import json
from typing import Any, Dict, List, Optional

from ctxforge.core.semantic_model import (
    EntityDefinition,
    RelationshipDefinition,
    SemanticModel,
    SemanticModelStore,
)
from ctxforge.storage.connection import PostgresConfig, PostgresConnectionManager

CREATE_SEMANTIC_MODELS_TABLE = """
CREATE TABLE IF NOT EXISTS {table_name} (
    model_id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(50) NOT NULL DEFAULT '1.0',
    data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{table_name}_name ON {table_name}(name);
CREATE INDEX IF NOT EXISTS idx_{table_name}_updated_at ON {table_name}(updated_at);
"""


class PostgresSemanticModelStore(SemanticModelStore):
    """
    PostgreSQL-based semantic model store.

    Features:
    - JSONB storage for flexible model data
    - Indexed queries
    - Full ACID transaction support
    """

    def __init__(
        self,
        config: Optional[PostgresConfig] = None,
        connection_manager: Optional[PostgresConnectionManager] = None,
        table_name: str = "semantic_models",
    ):
        """
        Initialize the PostgreSQL semantic model store.

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

        sql = CREATE_SEMANTIC_MODELS_TABLE.format(table_name=self._table_name)
        await self._manager.execute(sql)
        self._initialized = True

    async def load(self, model_id: str) -> Optional[SemanticModel]:
        """Load a semantic model by ID."""
        await self.connect()

        sql = f"SELECT data FROM {self._table_name} WHERE model_id = $1"
        row = await self._manager.fetchrow(sql, model_id)

        if row is None:
            return None

        data = row["data"]
        if isinstance(data, str):
            data = json.loads(data)

        return self._parse_model_data(data)

    async def save(self, model_id: str, model: SemanticModel) -> None:
        """Save a semantic model."""
        await self.connect()

        data = self._serialize_model(model)

        sql = f"""
        INSERT INTO {self._table_name} (model_id, name, description, version, data)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (model_id) DO UPDATE SET
            name = EXCLUDED.name,
            description = EXCLUDED.description,
            version = EXCLUDED.version,
            data = EXCLUDED.data,
            updated_at = NOW()
        """
        await self._manager.execute(
            sql,
            model_id,
            model.name,
            model.description,
            model.version,
            json.dumps(data),
        )

    async def list_models(self) -> List[str]:
        """List all model IDs."""
        await self.connect()

        sql = f"SELECT model_id FROM {self._table_name} ORDER BY name"
        rows = await self._manager.fetch(sql)

        return [row["model_id"] for row in rows]

    async def delete(self, model_id: str) -> bool:
        """Delete a model by ID."""
        await self.connect()

        sql = f"DELETE FROM {self._table_name} WHERE model_id = $1 RETURNING model_id"
        row = await self._manager.fetchrow(sql, model_id)

        return row is not None

    def _serialize_model(self, model: SemanticModel) -> Dict[str, Any]:
        """Serialize a SemanticModel to dict."""
        return {
            "name": model.name,
            "description": model.description,
            "version": model.version,
            "entities": [
                {
                    "name": e.name,
                    "description": e.description,
                    "use_cases": e.use_cases,
                    "attributes": e.attributes,
                    "related_entities": e.related_entities,
                    "retrieval_hint": e.retrieval_hint,
                }
                for e in model.entities
            ],
            "relationships": [
                {
                    "name": r.name,
                    "from_entity": r.from_entity,
                    "to_entity": r.to_entity,
                    "description": r.description,
                    "cardinality": r.cardinality,
                }
                for r in model.relationships
            ],
            "global_rules": model.global_rules,
            "common_gotchas": model.common_gotchas,
            "default_search_entities": model.default_search_entities,
            "metadata": model.metadata,
        }

    def _parse_model_data(self, data: Dict[str, Any]) -> SemanticModel:
        """Parse dict data into a SemanticModel."""
        entities = []
        for e in data.get("entities", []):
            entities.append(EntityDefinition(
                name=e.get("name", ""),
                description=e.get("description", ""),
                use_cases=e.get("use_cases", []),
                attributes=e.get("attributes", []),
                related_entities=e.get("related_entities", []),
                retrieval_hint=e.get("retrieval_hint"),
            ))

        relationships = []
        for r in data.get("relationships", []):
            relationships.append(RelationshipDefinition(
                name=r.get("name", ""),
                from_entity=r.get("from_entity", ""),
                to_entity=r.get("to_entity", ""),
                description=r.get("description", ""),
                cardinality=r.get("cardinality", "many_to_many"),
            ))

        return SemanticModel(
            name=data.get("name", "Unnamed"),
            description=data.get("description", ""),
            version=data.get("version", "1.0"),
            entities=entities,
            relationships=relationships,
            global_rules=data.get("global_rules", []),
            common_gotchas=data.get("common_gotchas", []),
            default_search_entities=data.get("default_search_entities", []),
            metadata=data.get("metadata", {}),
        )
