"""
Tests for Skill Relationship Service.
"""
import json
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

from ctxforge.core.skill import (
    SkillMetadata,
    SkillRelationship,
    SkillRelationType,
    SkillScope,
)
from ctxforge.engine.services.skill_relationship_service import (
    CyclicDependencyError,
    SkillRelationshipService,
)
from ctxforge.protocols.llm import ChatMessage, LLMResponse
from ctxforge.storage.memory.skill import InMemorySkillStore


class FakeLLMProvider:
    """Minimal fake LLM provider for testing."""

    def __init__(self, response_content: str = "[]"):
        self._response_content = response_content

    @property
    def name(self) -> str:
        return "fake"

    @property
    def default_model(self) -> str:
        return "fake-model"

    async def generate(
        self, prompt: str, model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None, **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._response_content, model="fake-model")

    async def chat(
        self, messages: List[ChatMessage], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        functions: Optional[List[Dict[str, Any]]] = None, **kwargs: Any,
    ) -> LLMResponse:
        return LLMResponse(content=self._response_content, model="fake-model")

    async def stream(
        self, messages: List[ChatMessage], model: Optional[str] = None,
        temperature: float = 0.7, max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        yield self._response_content

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        return len(text.split())

    def count_message_tokens(
        self, messages: List[ChatMessage], model: Optional[str] = None,
    ) -> int:
        return sum(len(m.content.split()) for m in messages)


def _make_skills() -> List[SkillMetadata]:
    return [
        SkillMetadata(
            name="skill-a", description="Skill A does X",
            scope=SkillScope.BASE, scope_id="system",
        ),
        SkillMetadata(
            name="skill-b", description="Skill B does Y",
            scope=SkillScope.BASE, scope_id="system",
        ),
        SkillMetadata(
            name="skill-c", description="Skill C does Z",
            scope=SkillScope.BASE, scope_id="system",
        ),
    ]


def _valid_response() -> str:
    return json.dumps([
        {
            "source": "skill-a", "target": "skill-b",
            "relation_type": "depend_on",
            "reason": "A needs B", "confidence": 0.9,
        },
        {
            "source": "skill-a", "target": "skill-c",
            "relation_type": "compose_with",
            "reason": "Often used together", "confidence": 0.8,
        },
        {
            "source": "skill-b", "target": "skill-c",
            "relation_type": "similar_to",
            "reason": "Functionally similar", "confidence": 0.7,
        },
    ])


class TestSkillRelationshipService:
    """Tests for SkillRelationshipService."""

    @pytest.fixture
    def store(self):
        return InMemorySkillStore()

    async def test_analyze_relationships_returns_valid_list(self, store):
        llm = FakeLLMProvider(response_content=_valid_response())
        svc = SkillRelationshipService(llm, store)
        result = await svc.analyze_relationships(_make_skills())
        assert len(result) == 3
        assert all(isinstance(r, SkillRelationship) for r in result)

    async def test_analyze_validates_known_names(self, store):
        """Relationships with unknown source/target are filtered out."""
        response = json.dumps([
            {
                "source": "skill-a", "target": "unknown-skill",
                "relation_type": "depend_on",
                "reason": "test", "confidence": 0.9,
            },
            {
                "source": "skill-a", "target": "skill-b",
                "relation_type": "compose_with",
                "reason": "valid", "confidence": 0.8,
            },
        ])
        llm = FakeLLMProvider(response_content=response)
        svc = SkillRelationshipService(llm, store)
        result = await svc.analyze_relationships(_make_skills())
        assert len(result) == 1
        assert result[0].target == "skill-b"

    async def test_analyze_rejects_invalid_relation_types(self, store):
        """Invalid relation types are skipped during parsing."""
        response = json.dumps([
            {
                "source": "skill-a", "target": "skill-b",
                "relation_type": "invalid_type",
                "reason": "test", "confidence": 0.9,
            },
        ])
        llm = FakeLLMProvider(response_content=response)
        svc = SkillRelationshipService(llm, store)
        result = await svc.analyze_relationships(_make_skills())
        assert len(result) == 0

    async def test_get_related_skills_filters_by_type(self, store):
        rels = [
            SkillRelationship(
                source="a", target="b",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
            SkillRelationship(
                source="a", target="c",
                relation_type=SkillRelationType.SIMILAR_TO,
            ),
        ]
        await store.save_relationships(rels)
        llm = FakeLLMProvider()
        svc = SkillRelationshipService(llm, store)

        deps = await svc.get_related_skills("a", SkillRelationType.DEPEND_ON)
        assert len(deps) == 1
        assert deps[0].target == "b"

        sims = await svc.get_related_skills("a", SkillRelationType.SIMILAR_TO)
        assert len(sims) == 1
        assert sims[0].target == "c"

    async def test_resolve_dependency_chain_simple(self, store):
        """Topological sort of a simple dependency chain."""
        rels = [
            SkillRelationship(
                source="skill-a", target="skill-b",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
            SkillRelationship(
                source="skill-b", target="skill-c",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
        ]
        await store.save_relationships(rels)
        llm = FakeLLMProvider()
        svc = SkillRelationshipService(llm, store)

        chain = await svc.resolve_dependency_chain("skill-a")
        assert chain == ["skill-c", "skill-b", "skill-a"]

    async def test_resolve_dependency_chain_detects_cycle(self, store):
        """Cyclic dependencies raise CyclicDependencyError."""
        rels = [
            SkillRelationship(
                source="a", target="b",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
            SkillRelationship(
                source="b", target="a",
                relation_type=SkillRelationType.DEPEND_ON,
            ),
        ]
        await store.save_relationships(rels)
        llm = FakeLLMProvider()
        svc = SkillRelationshipService(llm, store)

        with pytest.raises(CyclicDependencyError):
            await svc.resolve_dependency_chain("a")

    async def test_find_composable_skills(self, store):
        rels = [
            SkillRelationship(
                source="a", target="b",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
            SkillRelationship(
                source="c", target="a",
                relation_type=SkillRelationType.COMPOSE_WITH,
            ),
        ]
        await store.save_relationships(rels)
        llm = FakeLLMProvider()
        svc = SkillRelationshipService(llm, store)

        composable = await svc.find_composable_skills("a")
        assert set(composable) == {"b", "c"}

    async def test_find_alternatives(self, store):
        rels = [
            SkillRelationship(
                source="a", target="b",
                relation_type=SkillRelationType.SIMILAR_TO,
            ),
        ]
        await store.save_relationships(rels)
        llm = FakeLLMProvider()
        svc = SkillRelationshipService(llm, store)

        alts = await svc.find_alternatives("a")
        assert alts == ["b"]

    async def test_parse_malformed_json(self, store):
        """Malformed JSON raises ValueError."""
        llm = FakeLLMProvider(response_content="not json")
        svc = SkillRelationshipService(llm, store)

        with pytest.raises(ValueError, match="invalid JSON"):
            await svc.analyze_relationships(_make_skills())

    async def test_analyze_fewer_than_two_skills(self, store):
        """Analyzing fewer than 2 skills returns empty list."""
        llm = FakeLLMProvider()
        svc = SkillRelationshipService(llm, store)
        result = await svc.analyze_relationships([_make_skills()[0]])
        assert result == []

    async def test_analyze_self_referencing_filtered(self, store):
        """Self-referencing relationships are filtered out."""
        response = json.dumps([
            {
                "source": "skill-a", "target": "skill-a",
                "relation_type": "similar_to",
                "reason": "same", "confidence": 1.0,
            },
        ])
        llm = FakeLLMProvider(response_content=response)
        svc = SkillRelationshipService(llm, store)
        result = await svc.analyze_relationships(_make_skills())
        assert len(result) == 0
