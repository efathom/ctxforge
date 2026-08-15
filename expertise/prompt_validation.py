"""
Prompt validation for expertise extraction.

Validates few-shot examples to ensure they're well-formed
before using them in prompts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence

from ctxforge.extraction.alignment import WordAligner

logger = logging.getLogger(__name__)


class ValidationLevel(str, Enum):
    """Validation strictness levels."""
    
    OFF = "off"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """A single validation issue found."""
    
    example_index: int
    issue_type: str  # "error" or "warning"
    message: str
    alignment_status: Optional[str] = None  # Using string for alignment status
    
    def short_msg(self) -> str:
        return f"[example#{self.example_index}] {self.issue_type}: {self.message}"


@dataclass
class ValidationReport:
    """Collection of validation issues."""
    
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def has_errors(self) -> bool:
        return any(i.issue_type == "error" for i in self.issues)
    
    @property
    def has_warnings(self) -> bool:
        return any(i.issue_type == "warning" for i in self.issues)
    
    @property
    def is_valid(self) -> bool:
        return not self.has_errors


@dataclass
class ExpertiseExample:
    """A few-shot example for expertise extraction/reflection."""
    
    turn_input: str
    turn_response: str
    expected_items: List[str] = field(default_factory=list)
    expected_feedback: Dict[str, str] = field(default_factory=dict)  # item_id -> helpful/harmful/neutral


def validate_expertise_examples(
    examples: Sequence[ExpertiseExample],
    aligner: Optional[WordAligner] = None,
) -> ValidationReport:
    """
    Validate few-shot examples for expertise extraction.
    
    Checks:
    - Examples have required fields
    - Expected feedback values are valid
    - Expected items align to source text
    
    Args:
        examples: The examples to validate
        aligner: Optional aligner for text matching
        
    Returns:
        ValidationReport with any issues found
    """
    aligner = aligner or WordAligner()
    issues: List[ValidationIssue] = []
    
    for idx, ex in enumerate(examples):
        # Check required fields
        if not ex.turn_input:
            issues.append(ValidationIssue(
                example_index=idx,
                issue_type="error",
                message="Missing turn_input",
            ))
        
        if not ex.turn_response:
            issues.append(ValidationIssue(
                example_index=idx,
                issue_type="error",
                message="Missing turn_response",
            ))
        
        # Check feedback values
        valid_feedback = {"helpful", "harmful", "neutral"}
        for item_id, feedback in ex.expected_feedback.items():
            if feedback.lower() not in valid_feedback:
                issues.append(ValidationIssue(
                    example_index=idx,
                    issue_type="warning",
                    message=f"Invalid feedback '{feedback}' for item {item_id}",
                ))
        
        # Check expected items are non-empty strings
        for item_idx, item in enumerate(ex.expected_items):
            if not item or not item.strip():
                issues.append(ValidationIssue(
                    example_index=idx,
                    issue_type="warning",
                    message=f"Empty expected_item at index {item_idx}",
                ))
    
    return ValidationReport(issues=issues)


def validate_expertise_example_alignment(
    examples: Sequence[ExpertiseExample],
    aligner: Optional[WordAligner] = None,
) -> ValidationReport:
    """
    Validate that expected items in examples can be aligned to source text.
    
    This is a stricter validation that checks if the expected items
    actually appear in the turn input/response.
    
    Args:
        examples: The examples to validate
        aligner: Optional aligner for text matching
        
    Returns:
        ValidationReport with alignment issues
    """
    aligner = aligner or WordAligner()
    issues: List[ValidationIssue] = []
    
    for idx, ex in enumerate(examples):
        combined_text = f"{ex.turn_input} {ex.turn_response}"
        
        for item in ex.expected_items:
            if not item.strip():
                continue
            
            result = aligner.align(item, combined_text)
            
            if result.status.value == "unaligned":
                issues.append(ValidationIssue(
                    example_index=idx,
                    issue_type="warning",
                    message=f"Expected item '{item}' could not be aligned to source text",
                    alignment_status=result.status.value,
                ))
    
    return ValidationReport(issues=issues)


class PromptAlignmentError(RuntimeError):
    """Raised when prompt alignment validation fails."""
    pass


def handle_validation_report(
    report: ValidationReport,
    level: ValidationLevel,
    strict: bool = False,
) -> None:
    """
    Handle validation report based on level.
    
    Args:
        report: The validation report
        level: Validation strictness level
        strict: If True, treat warnings as errors in ERROR mode
        
    Raises:
        PromptAlignmentError: If validation fails in ERROR mode
    """
    if level == ValidationLevel.OFF:
        return
    
    for issue in report.issues:
        if level == ValidationLevel.WARNING:
            logger.warning("[WARN] %s", issue.short_msg())
        elif level == ValidationLevel.ERROR:
            if issue.issue_type == "error" or (strict and issue.issue_type == "warning"):
                raise PromptAlignmentError(f"Validation failed: {issue.short_msg()}")
            else:
                logger.warning("[WARN] %s", issue.short_msg())


def merge_validation_reports(*reports: ValidationReport) -> ValidationReport:
    """
    Merge multiple validation reports into one.
    
    Args:
        *reports: Reports to merge
        
    Returns:
        Combined ValidationReport
    """
    all_issues: List[ValidationIssue] = []
    for report in reports:
        all_issues.extend(report.issues)
    return ValidationReport(issues=all_issues)

