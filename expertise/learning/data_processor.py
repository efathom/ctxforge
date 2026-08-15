"""
Data processor protocol and implementations.

Provides the interface for task-specific data processing,
following the ACE framework's DataProcessor pattern.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ctxforge.expertise.learning.models import TrainingSample


@runtime_checkable
class IDataProcessor(Protocol):
    """
    Protocol for task-specific data processing.
    
    Following ACE's DataProcessor pattern, this interface defines
    how to convert raw data to training samples and evaluate answers.
    """
    
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[TrainingSample]:
        """
        Convert raw data to standardized TrainingSample format.
        
        Args:
            raw_data: List of dictionaries containing raw sample data
            
        Returns:
            List of TrainingSample objects
        """
        ...
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if prediction matches ground truth.
        
        Task-specific logic for comparing answers.
        
        Args:
            predicted: The predicted answer
            ground_truth: The correct answer
            
        Returns:
            True if the prediction is correct
        """
        ...
    
    def evaluate_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        """
        Calculate accuracy across multiple predictions.
        
        Args:
            predictions: List of predicted answers
            ground_truths: List of correct answers
            
        Returns:
            Accuracy score (0.0 to 1.0)
        """
        ...
    
    def extract_answer(self, response: str) -> str:
        """
        Extract the final answer from LLM response.
        
        Task-specific logic for parsing model output.
        
        Args:
            response: The full LLM response text
            
        Returns:
            The extracted answer
        """
        ...


class BaseDataProcessor(ABC):
    """
    Abstract base class for data processors.
    
    Provides common functionality and requires subclasses to implement
    task-specific logic.
    """
    
    @abstractmethod
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[TrainingSample]:
        """Convert raw data to TrainingSample format."""
        ...
    
    @abstractmethod
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction matches ground truth."""
        ...
    
    def evaluate_accuracy(
        self,
        predictions: List[str],
        ground_truths: List[str],
    ) -> float:
        """Calculate accuracy across multiple predictions."""
        if not predictions or len(predictions) != len(ground_truths):
            return 0.0
        
        correct = sum(
            1 for pred, gt in zip(predictions, ground_truths, strict=False)
            if self.answer_is_correct(pred, gt)
        )
        return correct / len(predictions)
    
    @abstractmethod
    def extract_answer(self, response: str) -> str:
        """Extract the final answer from LLM response."""
        ...
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Default implementation: lowercase, strip whitespace.
        """
        return text.lower().strip()


class ExactMatchProcessor(BaseDataProcessor):
    """
    Simple exact match data processor.
    
    Compares answers using exact string matching (after normalization).
    Useful for classification tasks, multiple choice, or short-answer.
    """
    
    def __init__(
        self,
        case_sensitive: bool = False,
        strip_punctuation: bool = True,
        answer_key: str = "target",
        question_key: str = "question",
        context_key: str = "context",
        id_key: str = "id",
    ):
        """
        Initialize the processor.
        
        Args:
            case_sensitive: Whether comparison is case-sensitive
            strip_punctuation: Whether to strip punctuation before comparing
            answer_key: Key for answer in raw data
            question_key: Key for question in raw data
            context_key: Key for context in raw data
            id_key: Key for sample ID in raw data
        """
        self._case_sensitive = case_sensitive
        self._strip_punctuation = strip_punctuation
        self._answer_key = answer_key
        self._question_key = question_key
        self._context_key = context_key
        self._id_key = id_key
    
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[TrainingSample]:
        """Convert raw data dictionaries to TrainingSample objects."""
        samples = []
        for i, item in enumerate(raw_data):
            sample = TrainingSample(
                sample_id=str(item.get(self._id_key, f"sample-{i}")),
                context=str(item.get(self._context_key, "")),
                question=str(item.get(self._question_key, "")),
                target=str(item.get(self._answer_key, "")),
                metadata={k: v for k, v in item.items() if k not in [
                    self._id_key, self._context_key, self._question_key, self._answer_key
                ]},
            )
            samples.append(sample)
        return samples
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if prediction exactly matches ground truth."""
        pred = self._normalize(predicted)
        gt = self._normalize(ground_truth)
        return pred == gt
    
    def extract_answer(self, response: str) -> str:
        """
        Extract answer from LLM response.
        
        Looks for common patterns like "The answer is X" or "Answer: X",
        falls back to the last line.
        """
        # Try common answer patterns
        patterns = [
            r"(?:the\s+)?answer\s+is[:\s]*(.+?)(?:\.|$)",
            r"answer[:\s]*(.+?)(?:\.|$)",
            r"^(.+?)$",  # Last resort: first non-empty line
        ]
        
        response = response.strip()
        
        for pattern in patterns[:-1]:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Fall back to last non-empty line
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[-1] if lines else ""
    
    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.strip()
        
        if not self._case_sensitive:
            text = text.lower()
        
        if self._strip_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()


class NumericMatchProcessor(BaseDataProcessor):
    """
    Data processor for numeric answers.
    
    Compares answers by extracting numeric values and checking
    if they are within a tolerance.
    """
    
    def __init__(
        self,
        tolerance: float = 0.001,
        answer_key: str = "target",
        question_key: str = "question",
        context_key: str = "context",
        id_key: str = "id",
    ):
        """
        Initialize the processor.
        
        Args:
            tolerance: Tolerance for numeric comparison
            answer_key: Key for answer in raw data
            question_key: Key for question in raw data
            context_key: Key for context in raw data
            id_key: Key for sample ID in raw data
        """
        self._tolerance = tolerance
        self._answer_key = answer_key
        self._question_key = question_key
        self._context_key = context_key
        self._id_key = id_key
    
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[TrainingSample]:
        """Convert raw data dictionaries to TrainingSample objects."""
        samples = []
        for i, item in enumerate(raw_data):
            sample = TrainingSample(
                sample_id=str(item.get(self._id_key, f"sample-{i}")),
                context=str(item.get(self._context_key, "")),
                question=str(item.get(self._question_key, "")),
                target=str(item.get(self._answer_key, "")),
                metadata={k: v for k, v in item.items() if k not in [
                    self._id_key, self._context_key, self._question_key, self._answer_key
                ]},
            )
            samples.append(sample)
        return samples
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if numeric values match within tolerance."""
        pred_num = self._extract_number(predicted)
        gt_num = self._extract_number(ground_truth)
        
        if pred_num is None or gt_num is None:
            return False
        
        return abs(pred_num - gt_num) <= self._tolerance
    
    def extract_answer(self, response: str) -> str:
        """Extract numeric answer from LLM response."""
        # Look for numbers in the response
        patterns = [
            r"(?:the\s+)?answer\s+is[:\s]*([+-]?[\d,.]+)",
            r"result[:\s]*([+-]?[\d,.]+)",
            r"=\s*([+-]?[\d,.]+)",
            r"([+-]?[\d,.]+)\s*$",  # Number at end
        ]
        
        response = response.strip()
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Fall back to any number found
        numbers = re.findall(r'[+-]?[\d,.]+', response)
        return numbers[-1] if numbers else ""
    
    def _extract_number(self, text: str) -> Optional[float]:
        """Extract a numeric value from text."""
        # Remove commas from numbers
        text = text.replace(",", "")
        
        # Try to find a number
        match = re.search(r'[+-]?[\d.]+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                return None
        return None


class MultipleChoiceProcessor(BaseDataProcessor):
    """
    Data processor for multiple choice questions.
    
    Extracts and compares letter choices (A, B, C, D, etc.).
    """
    
    def __init__(
        self,
        answer_key: str = "target",
        question_key: str = "question",
        context_key: str = "context",
        id_key: str = "id",
        choices_key: str = "choices",
    ):
        """
        Initialize the processor.
        
        Args:
            answer_key: Key for answer in raw data
            question_key: Key for question in raw data
            context_key: Key for context in raw data
            id_key: Key for sample ID in raw data
            choices_key: Key for choices in raw data
        """
        self._answer_key = answer_key
        self._question_key = question_key
        self._context_key = context_key
        self._id_key = id_key
        self._choices_key = choices_key
    
    def process_raw_data(self, raw_data: List[Dict[str, Any]]) -> List[TrainingSample]:
        """Convert raw data dictionaries to TrainingSample objects."""
        samples = []
        for i, item in enumerate(raw_data):
            # Build question with choices if available
            question = str(item.get(self._question_key, ""))
            choices = item.get(self._choices_key, [])
            if choices:
                choice_text = "\n".join(
                    f"{chr(65 + j)}. {choice}"
                    for j, choice in enumerate(choices)
                )
                question = f"{question}\n\nChoices:\n{choice_text}"
            
            sample = TrainingSample(
                sample_id=str(item.get(self._id_key, f"sample-{i}")),
                context=str(item.get(self._context_key, "")),
                question=question,
                target=str(item.get(self._answer_key, "")),
                metadata={k: v for k, v in item.items() if k not in [
                    self._id_key, self._context_key, self._question_key, self._answer_key
                ]},
            )
            samples.append(sample)
        return samples
    
    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        """Check if choice letters match."""
        pred_choice = self._extract_choice(predicted)
        gt_choice = self._extract_choice(ground_truth)
        
        return pred_choice.upper() == gt_choice.upper() if pred_choice and gt_choice else False
    
    def extract_answer(self, response: str) -> str:
        """Extract choice letter from LLM response."""
        return self._extract_choice(response) or ""
    
    def _extract_choice(self, text: str) -> Optional[str]:
        """Extract a choice letter (A, B, C, D, etc.) from text."""
        # Common patterns for choice answers
        patterns = [
            r"(?:the\s+)?answer\s+is[:\s]*\(?([A-Za-z])\)?",
            r"(?:choice|option)[:\s]*\(?([A-Za-z])\)?",
            r"^\s*\(?([A-Za-z])\)?\s*[.:]",
            r"\(?([A-Za-z])\)?\s*$",  # Choice at end
        ]
        
        text = text.strip()
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1)
        
        # Look for any single letter that could be a choice
        for line in text.split("\n"):
            line = line.strip()
            if len(line) == 1 and line.upper() in "ABCDEFGHIJ":
                return line
        
        return None

