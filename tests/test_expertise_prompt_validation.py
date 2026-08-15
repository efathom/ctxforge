"""
Tests for Phase 4: Expertise Extraction Enhancements.

Tests the prompt validation module for expertise extraction.
"""

import pytest

from ctxforge.expertise.prompt_validation import (
    ExpertiseExample,
    PromptAlignmentError,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
    handle_validation_report,
    merge_validation_reports,
    validate_expertise_example_alignment,
    validate_expertise_examples,
)


class TestValidationLevel:
    """Tests for ValidationLevel enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert ValidationLevel.OFF.value == "off"
        assert ValidationLevel.WARNING.value == "warning"
        assert ValidationLevel.ERROR.value == "error"


class TestValidationIssue:
    """Tests for ValidationIssue dataclass."""
    
    def test_short_msg(self):
        """Test short message format."""
        issue = ValidationIssue(
            example_index=0,
            issue_type="error",
            message="Something went wrong",
        )
        
        assert "[example#0]" in issue.short_msg()
        assert "error" in issue.short_msg()
        assert "Something went wrong" in issue.short_msg()
    
    def test_with_alignment_status(self):
        """Test issue with alignment status."""
        issue = ValidationIssue(
            example_index=1,
            issue_type="warning",
            message="Alignment issue",
            alignment_status="unaligned",
        )
        
        assert issue.alignment_status == "unaligned"


class TestValidationReport:
    """Tests for ValidationReport dataclass."""
    
    def test_empty_report(self):
        """Test empty report."""
        report = ValidationReport()
        
        assert report.issues == []
        assert report.has_errors is False
        assert report.has_warnings is False
        assert report.is_valid is True
    
    def test_report_with_errors(self):
        """Test report with errors."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "error", "Error message"),
        ])
        
        assert report.has_errors is True
        assert report.is_valid is False
    
    def test_report_with_warnings_only(self):
        """Test report with warnings only."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "warning", "Warning message"),
        ])
        
        assert report.has_errors is False
        assert report.has_warnings is True
        assert report.is_valid is True
    
    def test_report_with_mixed_issues(self):
        """Test report with both errors and warnings."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "error", "Error message"),
            ValidationIssue(1, "warning", "Warning message"),
        ])
        
        assert report.has_errors is True
        assert report.has_warnings is True
        assert report.is_valid is False


class TestExpertiseExample:
    """Tests for ExpertiseExample dataclass."""
    
    def test_basic_example(self):
        """Test basic example creation."""
        example = ExpertiseExample(
            turn_input="Hello, how can I help?",
            turn_response="I can assist with coding questions.",
        )
        
        assert example.turn_input == "Hello, how can I help?"
        assert example.turn_response == "I can assist with coding questions."
        assert example.expected_items == []
        assert example.expected_feedback == {}
    
    def test_example_with_feedback(self):
        """Test example with expected feedback."""
        example = ExpertiseExample(
            turn_input="What is Python?",
            turn_response="Python is a programming language.",
            expected_items=["programming", "language"],
            expected_feedback={"item-1": "helpful", "item-2": "neutral"},
        )
        
        assert len(example.expected_items) == 2
        assert example.expected_feedback["item-1"] == "helpful"


class TestValidateExpertiseExamples:
    """Tests for validate_expertise_examples function."""
    
    def test_valid_examples(self):
        """Test validation of valid examples."""
        examples = [
            ExpertiseExample(
                turn_input="Hello",
                turn_response="Hi there!",
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        assert report.is_valid is True
        assert len(report.issues) == 0
    
    def test_missing_turn_input(self):
        """Test error when turn_input is missing."""
        examples = [
            ExpertiseExample(
                turn_input="",
                turn_response="Hi there!",
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        assert report.is_valid is False
        assert any("turn_input" in i.message for i in report.issues)
    
    def test_missing_turn_response(self):
        """Test error when turn_response is missing."""
        examples = [
            ExpertiseExample(
                turn_input="Hello",
                turn_response="",
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        assert report.is_valid is False
        assert any("turn_response" in i.message for i in report.issues)
    
    def test_invalid_feedback_value(self):
        """Test warning for invalid feedback values."""
        examples = [
            ExpertiseExample(
                turn_input="Hello",
                turn_response="Hi",
                expected_feedback={"item-1": "invalid_value"},
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        assert report.has_warnings is True
        assert any("invalid_value" in i.message for i in report.issues)
    
    def test_valid_feedback_values(self):
        """Test that valid feedback values pass."""
        examples = [
            ExpertiseExample(
                turn_input="Hello",
                turn_response="Hi",
                expected_feedback={
                    "item-1": "helpful",
                    "item-2": "harmful",
                    "item-3": "neutral",
                },
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        # No warnings about feedback values
        assert not any("Invalid feedback" in i.message for i in report.issues)
    
    def test_empty_expected_item(self):
        """Test warning for empty expected items."""
        examples = [
            ExpertiseExample(
                turn_input="Hello",
                turn_response="Hi",
                expected_items=["valid", "", "  "],
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        assert report.has_warnings is True
        # Should have 2 warnings for empty items
        empty_warnings = [i for i in report.issues if "Empty expected_item" in i.message]
        assert len(empty_warnings) == 2
    
    def test_multiple_examples(self):
        """Test validation of multiple examples."""
        examples = [
            ExpertiseExample(
                turn_input="Hello",
                turn_response="Hi",
            ),
            ExpertiseExample(
                turn_input="",  # Error
                turn_response="Response",
            ),
            ExpertiseExample(
                turn_input="Input",
                turn_response="Response",
                expected_feedback={"item": "bad_value"},  # Warning
            ),
        ]
        
        report = validate_expertise_examples(examples)
        
        assert report.has_errors is True
        assert report.has_warnings is True
        assert len(report.issues) >= 2


class TestValidateExpertiseExampleAlignment:
    """Tests for validate_expertise_example_alignment function."""
    
    def test_aligned_items(self):
        """Test that aligned items pass."""
        examples = [
            ExpertiseExample(
                turn_input="I love Python programming",
                turn_response="Python is great!",
                expected_items=["Python", "programming"],
            ),
        ]
        
        report = validate_expertise_example_alignment(examples)
        
        # Should have no issues for aligned items
        assert len(report.issues) == 0
    
    def test_unaligned_items(self):
        """Test warning for unaligned items."""
        examples = [
            ExpertiseExample(
                turn_input="I love Python",
                turn_response="Great!",
                expected_items=["JavaScript"],  # Not in source
            ),
        ]
        
        report = validate_expertise_example_alignment(examples)
        
        assert report.has_warnings is True
        assert any("could not be aligned" in i.message for i in report.issues)


class TestHandleValidationReport:
    """Tests for handle_validation_report function."""
    
    def test_off_level(self):
        """Test that OFF level does nothing."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "error", "Error"),
        ])
        
        # Should not raise
        handle_validation_report(report, ValidationLevel.OFF)
    
    def test_warning_level_no_raise(self):
        """Test that WARNING level doesn't raise."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "error", "Error"),
        ])
        
        # Should not raise (just prints)
        handle_validation_report(report, ValidationLevel.WARNING)
    
    def test_error_level_raises_on_error(self):
        """Test that ERROR level raises on errors."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "error", "Error message"),
        ])
        
        with pytest.raises(PromptAlignmentError) as exc_info:
            handle_validation_report(report, ValidationLevel.ERROR)
        
        assert "Error message" in str(exc_info.value)
    
    def test_error_level_strict_raises_on_warning(self):
        """Test that ERROR level with strict=True raises on warnings."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "warning", "Warning message"),
        ])
        
        with pytest.raises(PromptAlignmentError):
            handle_validation_report(report, ValidationLevel.ERROR, strict=True)
    
    def test_error_level_non_strict_no_raise_on_warning(self):
        """Test that ERROR level without strict doesn't raise on warnings."""
        report = ValidationReport(issues=[
            ValidationIssue(0, "warning", "Warning message"),
        ])
        
        # Should not raise (just prints warning)
        handle_validation_report(report, ValidationLevel.ERROR, strict=False)


class TestMergeValidationReports:
    """Tests for merge_validation_reports function."""
    
    def test_merge_empty_reports(self):
        """Test merging empty reports."""
        report1 = ValidationReport()
        report2 = ValidationReport()
        
        merged = merge_validation_reports(report1, report2)
        
        assert len(merged.issues) == 0
    
    def test_merge_reports_with_issues(self):
        """Test merging reports with issues."""
        report1 = ValidationReport(issues=[
            ValidationIssue(0, "error", "Error 1"),
        ])
        report2 = ValidationReport(issues=[
            ValidationIssue(1, "warning", "Warning 1"),
        ])
        
        merged = merge_validation_reports(report1, report2)
        
        assert len(merged.issues) == 2
        assert merged.has_errors is True
        assert merged.has_warnings is True
    
    def test_merge_multiple_reports(self):
        """Test merging multiple reports."""
        reports = [
            ValidationReport(issues=[ValidationIssue(i, "warning", f"Issue {i}")])
            for i in range(5)
        ]
        
        merged = merge_validation_reports(*reports)
        
        assert len(merged.issues) == 5

