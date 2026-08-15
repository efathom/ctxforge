"""
Semantic Model - Domain schema for guiding retrieval.

A semantic model is a lightweight, structured description of available
knowledge that helps agents understand what they can query and retrieve.

The semantic model is injected into system context as a "domain map"
that tells the agent:
- What entities/tables/concepts are available
- How they relate to each other
- What use cases each supports
- Where to find detailed information

This reduces cold-start overhead and provides immediate orientation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class EntityDefinition(BaseModel):
    """
    Definition of an entity/concept in the domain.
    
    Equivalent to a "table" in the Text2SQL semantic model.
    """
    name: str
    description: str
    use_cases: List[str] = Field(default_factory=list)
    attributes: List[Dict[str, str]] = Field(default_factory=list)
    related_entities: List[str] = Field(default_factory=list)
    retrieval_hint: Optional[str] = None  # How to query for this entity


class RelationshipDefinition(BaseModel):
    """
    Definition of a relationship between entities.
    
    Helps the agent understand join paths and connections.
    """
    name: str
    from_entity: str
    to_entity: str
    description: str
    cardinality: str = "many_to_many"  # one_to_one, one_to_many, many_to_many


class SemanticModel(BaseModel):
    """
    A semantic model describing a domain's knowledge structure.
    
    This is the ctxforge equivalent of the Text2SQL semantic_model:
    - Provides a high-level overview of available knowledge
    - Guides the agent on what to search for
    - Included in system context for orientation
    
    Example for a customer service domain:
    ```python
    model = SemanticModel(
        name="Customer Service",
        entities=[
            EntityDefinition(
                name="customer_profiles",
                description="Customer information and preferences",
                use_cases=["Lookup customer by ID", "Find customer preferences"],
            ),
            EntityDefinition(
                name="order_history",
                description="Past orders and transactions",
                use_cases=["Check order status", "Find past purchases"],
            ),
        ],
        global_rules=[
            "Always verify customer identity before sharing order details",
            "Escalate refund requests over $500 to supervisor",
        ],
    )
    ```
    """
    name: str
    description: str = ""
    version: str = "1.0"
    
    # Core schema
    entities: List[EntityDefinition] = Field(default_factory=list)
    relationships: List[RelationshipDefinition] = Field(default_factory=list)
    
    # Domain guidance
    global_rules: List[str] = Field(default_factory=list)
    common_gotchas: List[str] = Field(default_factory=list)
    
    # Retrieval hints
    default_search_entities: List[str] = Field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_context_string(self, compact: bool = True) -> str:
        """
        Convert to a string suitable for inclusion in system context.
        
        Args:
            compact: If True, use condensed format; if False, use verbose
            
        Returns:
            Formatted string for LLM consumption
        """
        lines = [f"## Domain: {self.name}"]
        
        if self.description:
            lines.append(self.description)
            lines.append("")
        
        # Entities
        if self.entities:
            lines.append("### Available Knowledge Areas")
            for entity in self.entities:
                lines.append(f"- **{entity.name}**: {entity.description}")
                if entity.use_cases and not compact:
                    for uc in entity.use_cases:
                        lines.append(f"  - {uc}")
            lines.append("")
        
        # Relationships
        if self.relationships and not compact:
            lines.append("### Relationships")
            for rel in self.relationships:
                lines.append(f"- {rel.from_entity} → {rel.to_entity}: {rel.description}")
            lines.append("")
        
        # Global rules
        if self.global_rules:
            lines.append("### Rules")
            for rule in self.global_rules:
                lines.append(f"- {rule}")
            lines.append("")
        
        # Gotchas
        if self.common_gotchas and not compact:
            lines.append("### Common Gotchas")
            for gotcha in self.common_gotchas:
                lines.append(f"- {gotcha}")
            lines.append("")
        
        # Retrieval hint
        lines.append("*Use `search_knowledge` to retrieve detailed information about any area.*")
        
        return "\n".join(lines)
    
    def get_entity(self, name: str) -> Optional[EntityDefinition]:
        """Get an entity definition by name."""
        for entity in self.entities:
            if entity.name == name:
                return entity
        return None
    
    def add_entity(self, entity: EntityDefinition) -> None:
        """Add an entity to the model."""
        self.entities.append(entity)
    
    def add_relationship(self, relationship: RelationshipDefinition) -> None:
        """Add a relationship to the model."""
        self.relationships.append(relationship)
    
    def add_rule(self, rule: str) -> None:
        """Add a global rule."""
        if rule not in self.global_rules:
            self.global_rules.append(rule)
    
    def add_gotcha(self, gotcha: str) -> None:
        """Add a common gotcha."""
        if gotcha not in self.common_gotchas:
            self.common_gotchas.append(gotcha)


class SemanticModelStore:
    """
    Abstract store for semantic models.
    
    Semantic models can be loaded from:
    - JSON/YAML files
    - Database
    - Programmatic construction
    """
    
    async def load(self, model_id: str) -> Optional[SemanticModel]:
        raise NotImplementedError
    
    async def save(self, model_id: str, model: SemanticModel) -> None:
        raise NotImplementedError
    
    async def list_models(self) -> List[str]:
        raise NotImplementedError


class InMemorySemanticModelStore(SemanticModelStore):
    """In-memory implementation for development/testing."""
    
    def __init__(self):
        self._models: Dict[str, SemanticModel] = {}
    
    async def load(self, model_id: str) -> Optional[SemanticModel]:
        return self._models.get(model_id)
    
    async def save(self, model_id: str, model: SemanticModel) -> None:
        self._models[model_id] = model
    
    async def list_models(self) -> List[str]:
        return list(self._models.keys())
    
    async def delete(self, model_id: str) -> bool:
        """Delete a model. Returns True if deleted."""
        if model_id in self._models:
            del self._models[model_id]
            return True
        return False
    
    async def clear(self) -> None:
        """Clear all models."""
        self._models.clear()


class FileBasedSemanticModelStore(SemanticModelStore):
    """
    File-based semantic model store.
    
    Loads semantic models from YAML or JSON files. This is the recommended
    approach for production deployments where semantic models are defined
    as configuration files.
    
    Directory structure:
    ```
    models_dir/
    ├── customer-service.yaml
    ├── inventory.yaml
    └── analytics.json
    ```
    
    Example YAML format:
    ```yaml
    name: Customer Service KB
    description: Knowledge base for customer service
    version: "1.0"
    entities:
      - name: customers
        description: Customer profiles
        use_cases:
          - Find customer by ID
          - Update contact info
    global_rules:
      - Always verify customer identity
    ```
    """
    
    def __init__(self, models_dir: str, auto_reload: bool = False):
        """
        Initialize the file-based store.
        
        Args:
            models_dir: Directory containing model files
            auto_reload: If True, reload files on each load() call
        """
        self._models_dir = models_dir
        self._auto_reload = auto_reload
        self._cache: Dict[str, SemanticModel] = {}
    
    async def load(self, model_id: str) -> Optional[SemanticModel]:
        """Load a model from file."""
        import json
        import os
        
        # Check cache first (unless auto_reload)
        if not self._auto_reload and model_id in self._cache:
            return self._cache[model_id]
        
        # Try YAML first, then JSON
        yaml_path = os.path.join(self._models_dir, f"{model_id}.yaml")
        yml_path = os.path.join(self._models_dir, f"{model_id}.yml")
        json_path = os.path.join(self._models_dir, f"{model_id}.json")
        
        data = None
        for path in [yaml_path, yml_path, json_path]:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        if path.endswith('.json'):
                            data = json.load(f)
                        else:
                            # Try to import yaml
                            try:
                                import yaml
                                data = yaml.safe_load(f)
                            except ImportError:
                                # YAML not available, skip
                                continue
                    break
                except Exception:
                    continue
        
        if data is None:
            return None
        
        # Parse into SemanticModel
        try:
            model = self._parse_model_data(data)
            self._cache[model_id] = model
            return model
        except Exception:
            return None
    
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
    
    async def save(self, model_id: str, model: SemanticModel) -> None:
        """Save a model to file (JSON format)."""
        import json
        import os
        
        os.makedirs(self._models_dir, exist_ok=True)
        path = os.path.join(self._models_dir, f"{model_id}.json")
        
        data = {
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
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        self._cache[model_id] = model
    
    async def list_models(self) -> List[str]:
        """List all model files in the directory."""
        import os
        
        if not os.path.exists(self._models_dir):
            return []
        
        models = set()
        for filename in os.listdir(self._models_dir):
            if filename.endswith(('.yaml', '.yml', '.json')):
                model_id = filename.rsplit('.', 1)[0]
                models.add(model_id)
        
        return sorted(models)
    
    async def delete(self, model_id: str) -> bool:
        """Delete a model file."""
        import os
        
        for ext in ['.yaml', '.yml', '.json']:
            path = os.path.join(self._models_dir, f"{model_id}{ext}")
            if os.path.exists(path):
                os.remove(path)
                self._cache.pop(model_id, None)
                return True
        return False
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        self._cache.clear()
