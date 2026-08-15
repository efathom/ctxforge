"""
Skill Relationship Service.

Analyzes and stores typed relationships between skills, enabling
smarter skill recommendation and dependency resolution.
"""
import json
import logging
from typing import Dict, List, Optional, Set

from ctxforge.config.base import SkillRelationshipConfig
from ctxforge.core.skill import (
    SkillMetadata,
    SkillRelationship,
    SkillRelationType,
)
from ctxforge.engine.prompts.skill_relationship import (
    RELATIONSHIP_ANALYSIS_SYSTEM_PROMPT,
    build_relationship_prompt,
)
from ctxforge.protocols.llm import ChatMessage, ILLMProvider
from ctxforge.protocols.skill import ISkillStore

logger = logging.getLogger(__name__)

_VALID_RELATION_TYPES = {rt.value for rt in SkillRelationType}


class CyclicDependencyError(Exception):
    """Raised when a cyclic dependency is detected in the skill graph."""


class SkillRelationshipService:
    """Analyze and query typed relationships between skills."""

    def __init__(
        self,
        llm_provider: ILLMProvider,
        skill_store: ISkillStore,
        config: Optional[SkillRelationshipConfig] = None,
    ):
        self._llm = llm_provider
        self._store = skill_store
        self._config = config or SkillRelationshipConfig()

    async def analyze_relationships(
        self, skills: List[SkillMetadata]
    ) -> List[SkillRelationship]:
        """Use LLM to infer relationships between skills.

        Args:
            skills: List of skill metadata to analyze.

        Returns:
            List of inferred SkillRelationship objects.

        Raises:
            ValueError: If the LLM response cannot be parsed.
        """
        if len(skills) < 2:
            return []

        known_names: Set[str] = {s.name for s in skills}

        skills_info_parts = []
        for s in skills:
            skills_info_parts.append(f"- **{s.name}**: {s.description}")
        skills_info = "\n".join(skills_info_parts)

        user_prompt = build_relationship_prompt(skills_info)

        messages = [
            ChatMessage(
                role="system", content=RELATIONSHIP_ANALYSIS_SYSTEM_PROMPT
            ),
            ChatMessage(role="user", content=user_prompt),
        ]

        response = await self._llm.chat(
            messages=messages,
            temperature=0.2,
            max_tokens=2048,
        )

        raw_relationships = self._parse_response(response.content)

        valid: List[SkillRelationship] = []
        for rel in raw_relationships:
            if rel.source not in known_names:
                logger.debug(
                    "Skipping relationship: unknown source '%s'", rel.source
                )
                continue
            if rel.target not in known_names:
                logger.debug(
                    "Skipping relationship: unknown target '%s'", rel.target
                )
                continue
            if rel.source == rel.target:
                continue
            valid.append(rel)

        if valid:
            await self._store.save_relationships(valid)

        return valid

    async def get_related_skills(
        self,
        skill_name: str,
        relation_type: Optional[SkillRelationType] = None,
    ) -> List[SkillRelationship]:
        """Get relationships for a skill, optionally filtered by type.

        Args:
            skill_name: The skill to query relationships for.
            relation_type: Optional filter by relationship type.

        Returns:
            List of matching SkillRelationship objects.
        """
        rels = await self._store.get_relationships(skill_name)
        if relation_type is not None:
            rels = [r for r in rels if r.relation_type == relation_type]
        return rels

    async def get_skill_graph(self) -> List[SkillRelationship]:
        """Get the full relationship graph.

        Returns:
            All stored relationships.
        """
        return await self._store.get_all_relationships()

    async def resolve_dependency_chain(self, skill_name: str) -> List[str]:
        """Resolve the full dependency chain for a skill via topological sort.

        Args:
            skill_name: The skill to resolve dependencies for.

        Returns:
            Topologically sorted list of skill names (dependencies first,
            the requested skill last).

        Raises:
            CyclicDependencyError: If a cycle is detected.
        """
        all_rels = await self._store.get_all_relationships()

        dep_edges: Dict[str, List[str]] = {}
        for rel in all_rels:
            if rel.relation_type == SkillRelationType.DEPEND_ON:
                dep_edges.setdefault(rel.source, []).append(rel.target)

        order: List[str] = []
        visited: Set[str] = set()
        in_stack: Set[str] = set()

        def _visit(node: str) -> None:
            if node in in_stack:
                raise CyclicDependencyError(
                    f"Cyclic dependency detected involving '{node}'"
                )
            if node in visited:
                return
            in_stack.add(node)
            for dep in dep_edges.get(node, []):
                _visit(dep)
            in_stack.discard(node)
            visited.add(node)
            order.append(node)

        _visit(skill_name)
        return order

    async def find_composable_skills(self, skill_name: str) -> List[str]:
        """Find skills that compose with the given skill.

        Args:
            skill_name: The skill to find companions for.

        Returns:
            List of skill names that have a compose_with relationship.
        """
        rels = await self.get_related_skills(
            skill_name, SkillRelationType.COMPOSE_WITH
        )
        result: List[str] = []
        for r in rels:
            other = r.target if r.source == skill_name else r.source
            if other not in result:
                result.append(other)
        return result

    async def find_alternatives(self, skill_name: str) -> List[str]:
        """Find skills similar to the given skill.

        Args:
            skill_name: The skill to find alternatives for.

        Returns:
            List of skill names that have a similar_to relationship.
        """
        rels = await self.get_related_skills(
            skill_name, SkillRelationType.SIMILAR_TO
        )
        result: List[str] = []
        for r in rels:
            other = r.target if r.source == skill_name else r.source
            if other not in result:
                result.append(other)
        return result

    def _parse_response(self, raw: str) -> List[SkillRelationship]:
        """Parse the LLM JSON response into SkillRelationship objects.

        Args:
            raw: Raw LLM response text (expected to be a JSON array).

        Returns:
            List of SkillRelationship objects.

        Raises:
            ValueError: If the response is not valid JSON.
        """
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            cleaned = "\n".join(lines)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned invalid JSON: {exc}") from exc

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array of relationships")

        results: List[SkillRelationship] = []
        for item in data:
            rt_val = item.get("relation_type", "")
            if rt_val not in _VALID_RELATION_TYPES:
                logger.debug("Skipping invalid relation_type: '%s'", rt_val)
                continue
            results.append(SkillRelationship(
                source=item.get("source", ""),
                target=item.get("target", ""),
                relation_type=SkillRelationType(rt_val),
                reason=item.get("reason", ""),
                confidence=float(item.get("confidence", 0.8)),
            ))
        return results
