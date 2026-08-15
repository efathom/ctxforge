from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type

from pydantic import BaseModel


@dataclass
class GraphOntology:
    """
    Ontology container used to constrain extraction and validation.

    - entity_types / edge_types map names to Pydantic models for attribute validation.
    - allowed_edges constrains which (source_type, target_type) pairs are valid for an edge_type.
    """

    entity_types: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    edge_types: Dict[str, Type[BaseModel]] = field(default_factory=dict)
    allowed_edges: Dict[str, List[Tuple[str, str]]] = field(default_factory=dict)

    def normalize_entity_type(self, entity_type: str) -> str:
        return (entity_type or "").strip()

    def normalize_edge_type(self, edge_type: str) -> str:
        return (edge_type or "").strip()

    def is_entity_type_known(self, entity_type: str) -> bool:
        et = self.normalize_entity_type(entity_type)
        return et in self.entity_types

    def is_edge_type_known(self, edge_type: str) -> bool:
        et = self.normalize_edge_type(edge_type)
        return et in self.edge_types

    def is_edge_allowed(self, edge_type: str, source_type: str, target_type: str) -> bool:
        edge_t = self.normalize_edge_type(edge_type)
        st = self.normalize_entity_type(source_type)
        tt = self.normalize_entity_type(target_type)
        allowed = self.allowed_edges.get(edge_t)
        if not allowed:
            return True
        return (st, tt) in allowed

    def validate_entity_attributes(self, entity_type: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        et = self.normalize_entity_type(entity_type)
        Model = self.entity_types.get(et)
        if Model is None:
            return dict(attributes or {})
        obj = Model.model_validate(attributes or {})
        return obj.model_dump()

    def validate_edge_attributes(self, edge_type: str, attributes: Dict[str, Any]) -> Dict[str, Any]:
        et = self.normalize_edge_type(edge_type)
        Model = self.edge_types.get(et)
        if Model is None:
            return dict(attributes or {})
        obj = Model.model_validate(attributes or {})
        return obj.model_dump()


def load_ontology_from_module(
    module_path: str,
    *,
    attr_name: str = "GRAPH_ONTOLOGY",
) -> GraphOntology:
    module = importlib.import_module(module_path)
    obj = getattr(module, attr_name, None)
    if obj is None:
        raise ValueError(f"Ontology module '{module_path}' does not define '{attr_name}'")
    if not isinstance(obj, GraphOntology):
        raise TypeError(f"'{module_path}.{attr_name}' must be a GraphOntology")
    return obj


