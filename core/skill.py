"""
Skill Models for Skills System.

This module defines the core data structures for skills that support
progressive disclosure with base, user, and project scopes.

Includes CSO (Claude Search Optimization) validation for skill descriptions
to ensure descriptions contain only triggering conditions, not workflow
summaries. This prevents LLMs from following the description as a shortcut
and skipping the full skill content.
"""
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Process verbs that indicate a description is summarizing workflow
# rather than specifying triggering conditions. Empirically derived
# from Superpowers testing: when descriptions contain these terms,
# LLMs follow the description summary instead of reading the full
# skill content.
_CSO_PROCESS_VERBS = re.compile(
    r'\b('
    r'dispatch(?:es|ed|ing)?'
    r'|review(?:s|ed|ing)?'
    r'|execut(?:e|es|ed|ing)'
    r'|implement(?:s|ed|ing)?'
    r'|generat(?:e|es|ed|ing)'
    r'|creat(?:e|es|ed|ing)'
    r'|analyz(?:e|es|ed|ing)'
    r'|refactor(?:s|ed|ing)?'
    r'|deploy(?:s|ed|ing)?'
    r'|then|after that|first.*then|followed by'
    r'|step \d'
    r')\b',
    re.IGNORECASE,
)

# Sequence markers that indicate multi-step workflow summaries
_CSO_SEQUENCE_MARKERS = re.compile(
    r'\b(then|next|afterwards|finally|subsequently|followed by)\b',
    re.IGNORECASE,
)


def lint_skill_description(description: str) -> List[str]:
    """Check a skill description for CSO violations.

    CSO (Claude Search Optimization) requires that skill descriptions
    contain ONLY triggering conditions (when to use), never workflow
    summaries (what the skill does). When descriptions summarize the
    workflow, LLMs follow the summary as a shortcut and skip reading
    the full skill content.

    Args:
        description: The skill description to validate.

    Returns:
        List of warning messages. Empty list means the description
        passes CSO validation.
    """
    warnings: List[str] = []

    if not description:
        return warnings

    # Strip the "Use when [verb-phrase]" prefix before checking for
    # process verbs. Verbs immediately after "Use when" are triggering
    # conditions (e.g., "Use when implementing features" is fine —
    # "implementing" describes WHEN to use the skill, not WHAT it does).
    # We strip through the first clause boundary (comma, dash, period,
    # or coordinating conjunction "and"/"or") to preserve the trigger.
    check_text = description
    use_when_match = re.match(
        r'^use\s+when\s+[^,.\-;]+(?:[,.\-;]|$)',
        description,
        re.IGNORECASE,
    )
    if use_when_match:
        # Only check text after the first clause
        check_text = description[use_when_match.end():]

    # Check for process verbs in the non-trigger portion
    process_matches = _CSO_PROCESS_VERBS.findall(check_text)
    if process_matches:
        warnings.append(
            f"Description contains process verbs {process_matches} "
            f"which may cause LLMs to follow the description as a "
            f"shortcut instead of reading the full skill content. "
            f"Use only triggering conditions (e.g., 'Use when ...')."
        )

    # Check for sequence markers (multi-step summaries)
    seq_matches = _CSO_SEQUENCE_MARKERS.findall(check_text)
    if seq_matches:
        warnings.append(
            f"Description contains sequence markers {seq_matches} "
            f"suggesting a workflow summary. Descriptions should "
            f"specify WHEN to use the skill, not HOW it works."
        )

    return warnings


class SkillRelationType(Enum):
    """Types of relationships between skills."""
    SIMILAR_TO = "similar_to"
    BELONG_TO = "belong_to"
    COMPOSE_WITH = "compose_with"
    DEPEND_ON = "depend_on"


class EvaluationLevel(Enum):
    """Quality level for a single evaluation dimension."""
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"


@dataclass
class SkillRelationship:
    """A typed relationship between two skills."""
    source: str
    target: str
    relation_type: SkillRelationType
    reason: str = ""
    confidence: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "relation_type": self.relation_type.value,
            "reason": self.reason,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillRelationship":
        """Create from dictionary representation."""
        return cls(
            source=data["source"],
            target=data["target"],
            relation_type=SkillRelationType(data["relation_type"]),
            reason=data.get("reason", ""),
            confidence=data.get("confidence", 0.8),
        )


EVALUATION_LEVEL_SCORES: Dict[EvaluationLevel, float] = {
    EvaluationLevel.GOOD: 1.0,
    EvaluationLevel.AVERAGE: 0.5,
    EvaluationLevel.POOR: 0.0,
}


@dataclass
class SkillEvaluation:
    """LLM-based quality evaluation of a skill across 5 dimensions."""
    safety: EvaluationLevel
    safety_reason: str
    completeness: EvaluationLevel
    completeness_reason: str
    executability: EvaluationLevel
    executability_reason: str
    maintainability: EvaluationLevel
    maintainability_reason: str
    cost_awareness: EvaluationLevel
    cost_awareness_reason: str
    evaluated_at: datetime
    overall_score: float

    @classmethod
    def compute_overall_score(
        cls,
        safety: EvaluationLevel,
        completeness: EvaluationLevel,
        executability: EvaluationLevel,
        maintainability: EvaluationLevel,
        cost_awareness: EvaluationLevel,
    ) -> float:
        """Compute weighted overall score from dimension levels.

        Weights: Safety 25%, Completeness 25%, Executability 20%,
                 Maintainability 15%, Cost-awareness 15%.
        """
        score = (
            EVALUATION_LEVEL_SCORES[safety] * 0.25
            + EVALUATION_LEVEL_SCORES[completeness] * 0.25
            + EVALUATION_LEVEL_SCORES[executability] * 0.20
            + EVALUATION_LEVEL_SCORES[maintainability] * 0.15
            + EVALUATION_LEVEL_SCORES[cost_awareness] * 0.15
        )
        return round(score, 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "safety": self.safety.value,
            "safety_reason": self.safety_reason,
            "completeness": self.completeness.value,
            "completeness_reason": self.completeness_reason,
            "executability": self.executability.value,
            "executability_reason": self.executability_reason,
            "maintainability": self.maintainability.value,
            "maintainability_reason": self.maintainability_reason,
            "cost_awareness": self.cost_awareness.value,
            "cost_awareness_reason": self.cost_awareness_reason,
            "evaluated_at": self.evaluated_at.isoformat(),
            "overall_score": self.overall_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillEvaluation":
        """Create from dictionary representation."""
        evaluated_at = data.get("evaluated_at")
        if isinstance(evaluated_at, str):
            evaluated_at = datetime.fromisoformat(evaluated_at)
        elif evaluated_at is None:
            evaluated_at = datetime.now()

        return cls(
            safety=EvaluationLevel(data["safety"]),
            safety_reason=data.get("safety_reason", ""),
            completeness=EvaluationLevel(data["completeness"]),
            completeness_reason=data.get("completeness_reason", ""),
            executability=EvaluationLevel(data["executability"]),
            executability_reason=data.get("executability_reason", ""),
            maintainability=EvaluationLevel(data["maintainability"]),
            maintainability_reason=data.get("maintainability_reason", ""),
            cost_awareness=EvaluationLevel(data["cost_awareness"]),
            cost_awareness_reason=data.get("cost_awareness_reason", ""),
            evaluated_at=evaluated_at,
            overall_score=data.get("overall_score", 0.0),
        )


@dataclass
class SkillContent:
    """Structured skill content with progressive disclosure sections."""
    instructions: str
    scripts: Dict[str, str] = field(default_factory=dict)
    references: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "instructions": self.instructions,
            "scripts": dict(self.scripts),
            "references": dict(self.references),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillContent":
        """Create from dictionary representation."""
        return cls(
            instructions=data.get("instructions", ""),
            scripts=data.get("scripts", {}),
            references=data.get("references", {}),
        )


class SkillScope(Enum):
    """Skill visibility scope (layering: base < user < project)."""
    BASE = "base"          # Built-in skills
    USER = "user"          # User-defined skills
    PROJECT = "project"    # Project-specific skills

    @classmethod
    def priority(cls, scope: "SkillScope") -> int:
        """Get priority for scope (higher = overrides lower)."""
        priorities = {
            cls.BASE: 0,
            cls.USER: 1,
            cls.PROJECT: 2,
        }
        return priorities.get(scope, 0)


@dataclass
class SkillMetadata:
    """Lightweight skill info for progressive disclosure."""
    name: str                          # Unique identifier (lowercase, hyphens)
    description: str                   # What the skill does (max 256 chars)
    scope: SkillScope
    scope_id: str                      # user_id or project_id
    triggers: List[str] = field(default_factory=list)  # Keywords/patterns that activate
    version: str = "1.0"
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    when_to_use: Optional[str] = None

    def __post_init__(self):
        """Validate skill metadata."""
        # Validate name format (lowercase, hyphens only)
        if not re.match(r'^[a-z][a-z0-9-]*$', self.name):
            raise ValueError(
                f"Skill name '{self.name}' must be lowercase, start with a letter, "
                "and contain only letters, numbers, and hyphens"
            )
        # Truncate description if too long
        if len(self.description) > 256:
            self.description = self.description[:253] + "..."
        # CSO validation: warn if description contains workflow summaries
        cso_warnings = lint_skill_description(self.description)
        for warning in cso_warnings:
            logger.warning(f"CSO violation in skill '{self.name}': {warning}")

    @property
    def cso_warnings(self) -> List[str]:
        """Check description for CSO violations."""
        return lint_skill_description(self.description)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "triggers": self.triggers,
            "version": self.version,
            "category": self.category,
            "tags": self.tags,
            "when_to_use": self.when_to_use,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillMetadata":
        """Create from dictionary representation."""
        return cls(
            name=data["name"],
            description=data["description"],
            scope=SkillScope(data["scope"]),
            scope_id=data["scope_id"],
            triggers=data.get("triggers", []),
            version=data.get("version", "1.0"),
            category=data.get("category"),
            tags=data.get("tags", []),
            when_to_use=data.get("when_to_use"),
        )

    def matches_trigger(self, query: str) -> Optional[str]:
        """Check if query matches any trigger. Returns matched trigger or None."""
        query_lower = query.lower()
        for trigger in self.triggers:
            if trigger.lower() in query_lower:
                return trigger
        return None


@dataclass
class Skill:
    """Full skill definition with workflow content."""
    # Metadata (always loaded)
    name: str
    description: str
    scope: SkillScope
    scope_id: str

    # Content (loaded on-demand)
    content: str                       # Full markdown workflow

    # Optional fields
    triggers: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)  # Other skills needed
    allowed_tools: List[str] = field(default_factory=list)  # Tools this skill can use
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    category: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    when_to_use: Optional[str] = None
    composed_of: List[str] = field(default_factory=list)
    evaluation: Optional[SkillEvaluation] = None
    effectiveness: Optional[Dict[str, Any]] = None
    structured_content: Optional[SkillContent] = None

    # Provenance tracking for cross-scope inheritance & graduation
    source_scope: Optional[SkillScope] = None
    source_scope_id: Optional[str] = None
    source_context: Optional[str] = None
    promoted_from: Optional[str] = None
    promoted_at: Optional[datetime] = None

    def __post_init__(self):
        """Validate skill."""
        # Validate name format (lowercase, hyphens only)
        if not re.match(r'^[a-z][a-z0-9-]*$', self.name):
            raise ValueError(
                f"Skill name '{self.name}' must be lowercase, start with a letter, "
                "and contain only letters, numbers, and hyphens"
            )
        # Truncate description if too long
        if len(self.description) > 256:
            self.description = self.description[:253] + "..."
        # CSO validation: warn if description contains workflow summaries
        cso_warnings = lint_skill_description(self.description)
        for warning in cso_warnings:
            logger.warning(f"CSO violation in skill '{self.name}': {warning}")

    @property
    def cso_warnings(self) -> List[str]:
        """Check description for CSO violations."""
        return lint_skill_description(self.description)

    @property
    def skill_metadata(self) -> SkillMetadata:
        """Extract metadata from full skill."""
        return SkillMetadata(
            name=self.name,
            description=self.description,
            scope=self.scope,
            scope_id=self.scope_id,
            triggers=self.triggers,
            version=self.version,
            category=self.category,
            tags=list(self.tags),
            when_to_use=self.when_to_use,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result: Dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "scope": self.scope.value,
            "scope_id": self.scope_id,
            "content": self.content,
            "triggers": self.triggers,
            "prerequisites": self.prerequisites,
            "allowed_tools": self.allowed_tools,
            "metadata": self.metadata,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "category": self.category,
            "tags": self.tags,
            "when_to_use": self.when_to_use,
            "composed_of": self.composed_of,
        }
        if self.evaluation is not None:
            result["evaluation"] = self.evaluation.to_dict()
        else:
            result["evaluation"] = None
        result["effectiveness"] = self.effectiveness
        if self.structured_content is not None:
            result["structured_content"] = self.structured_content.to_dict()
        else:
            result["structured_content"] = None

        # Provenance fields
        result["source_scope"] = (
            self.source_scope.value if self.source_scope else None
        )
        result["source_scope_id"] = self.source_scope_id
        result["source_context"] = self.source_context
        result["promoted_from"] = self.promoted_from
        result["promoted_at"] = (
            self.promoted_at.isoformat() if self.promoted_at else None
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Skill":
        """Create from dictionary representation."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()

        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()

        evaluation_data = data.get("evaluation")
        evaluation = None
        if isinstance(evaluation_data, dict):
            evaluation = SkillEvaluation.from_dict(evaluation_data)

        structured_content_data = data.get("structured_content")
        structured_content = None
        if isinstance(structured_content_data, dict):
            structured_content = SkillContent.from_dict(structured_content_data)

        # Provenance fields
        source_scope_raw = data.get("source_scope")
        source_scope = (
            SkillScope(source_scope_raw) if source_scope_raw else None
        )
        promoted_at_raw = data.get("promoted_at")
        promoted_at = None
        if isinstance(promoted_at_raw, str):
            promoted_at = datetime.fromisoformat(promoted_at_raw)
        elif isinstance(promoted_at_raw, datetime):
            promoted_at = promoted_at_raw

        return cls(
            name=data["name"],
            description=data["description"],
            scope=SkillScope(data["scope"]),
            scope_id=data["scope_id"],
            content=data["content"],
            triggers=data.get("triggers", []),
            prerequisites=data.get("prerequisites", []),
            allowed_tools=data.get("allowed_tools", []),
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0"),
            created_at=created_at,
            updated_at=updated_at,
            category=data.get("category"),
            tags=data.get("tags", []),
            when_to_use=data.get("when_to_use"),
            composed_of=data.get("composed_of", []),
            evaluation=evaluation,
            effectiveness=data.get("effectiveness"),
            structured_content=structured_content,
            source_scope=source_scope,
            source_scope_id=data.get("source_scope_id"),
            source_context=data.get("source_context"),
            promoted_from=data.get("promoted_from"),
            promoted_at=promoted_at,
        )


@dataclass
class SkillMatch:
    """Result of skill matching against a query."""
    skill: SkillMetadata
    confidence: float              # 0.0 - 1.0
    matched_trigger: Optional[str] = None
    match_reason: str = ""         # Description of why this matched

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "skill": self.skill.to_dict(),
            "confidence": self.confidence,
            "matched_trigger": self.matched_trigger,
            "match_reason": self.match_reason,
        }


@dataclass
class SkillsIndex:
    """Index of available skills for prompt injection."""
    skills: List[SkillMetadata]
    scope_counts: Dict[SkillScope, int] = field(default_factory=dict)

    @property
    def total_count(self) -> int:
        """Get total number of skills."""
        return len(self.skills)

    def format_for_prompt(self) -> str:
        """Format skills index for prompt injection.

        Includes category badges when skills have a category assigned.
        """
        if not self.skills:
            return ""

        lines = ["## Available Skills", ""]
        lines.append("You have access to the following skills. "
                     "Request a skill by name when needed.")
        lines.append("")

        for skill in self.skills:
            triggers_str = ", ".join(skill.triggers[:3]) if skill.triggers else "N/A"
            badge = f"[{skill.category}] " if skill.category else ""
            lines.append(f"- **{badge}{skill.name}**: {skill.description}")
            lines.append(f"  Triggers: {triggers_str}")

        return "\n".join(lines)

    def format_compact(self) -> str:
        """Format skills as a compact list."""
        if not self.skills:
            return "No skills available."

        return ", ".join(f"`{s.name}`" for s in self.skills)
