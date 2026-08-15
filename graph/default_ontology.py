from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from ctxforge.graph.ontology import GraphOntology


class Person(BaseModel):
    handle: Optional[str] = None
    email: Optional[str] = None


class Organization(BaseModel):
    website: Optional[str] = None


class Location(BaseModel):
    country: Optional[str] = None
    city: Optional[str] = None


class Likes(BaseModel):
    strength: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class WorksFor(BaseModel):
    role: Optional[str] = None


class SameAs(BaseModel):
    similarity: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class Passage(BaseModel):
    source_episode_id: Optional[str] = None
    chunk_index: Optional[int] = None
    token_count: Optional[int] = None


class Fact(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object_value: Optional[str] = None
    confidence: float = 1.0
    temporal_qualifier: Optional[str] = None


class Mentions(BaseModel):
    pass


class Evidences(BaseModel):
    pass


class SubjectOf(BaseModel):
    pass


class ObjectOf(BaseModel):
    pass


GRAPH_ONTOLOGY = GraphOntology(
    entity_types={
        "Person": Person,
        "Organization": Organization,
        "Location": Location,
        "Passage": Passage,
        "Fact": Fact,
    },
    edge_types={
        "LIKES": Likes,
        "WORKS_FOR": WorksFor,
        "SAME_AS": SameAs,
        "MENTIONS": Mentions,
        "EVIDENCES": Evidences,
        "SUBJECT_OF": SubjectOf,
        "OBJECT_OF": ObjectOf,
    },
    allowed_edges={
        "LIKES": [("Person", "Organization"), ("Person", "Location"), ("Person", "Person")],
        "WORKS_FOR": [("Person", "Organization")],
        "SAME_AS": [
            ("Person", "Person"),
            ("Organization", "Organization"),
            ("Location", "Location"),
        ],
        "MENTIONS": [
            ("Passage", "Person"),
            ("Passage", "Organization"),
            ("Passage", "Location"),
        ],
        "EVIDENCES": [("Passage", "Fact")],
        "SUBJECT_OF": [
            ("Person", "Fact"),
            ("Organization", "Fact"),
            ("Location", "Fact"),
        ],
        "OBJECT_OF": [
            ("Person", "Fact"),
            ("Organization", "Fact"),
            ("Location", "Fact"),
        ],
    },
)


