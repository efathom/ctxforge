"""
Data models for the learning orchestration system.

Defines configuration, sample, and result models for offline and online learning.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LearningMode(str, Enum):
    """
    Learning modes for expertise evolution.
    
    - OFFLINE: Batch learning from labeled dataset
    - ONLINE: Incremental learning during production use
    - FROZEN: No learning, fixed expertise
    """
    OFFLINE = "offline"
    ONLINE = "online"
    FROZEN = "frozen"


class TrainingSample(BaseModel):
    """
    A single training sample with ground truth.
    
    Used for both training and evaluation.
    
    Attributes:
        sample_id: Unique identifier for the sample
        context: Background information/context
        question: The question or instruction
        target: Ground truth answer
        metadata: Additional metadata
    """
    sample_id: str
    context: str = ""
    question: str
    target: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], sample_id: Optional[str] = None) -> "TrainingSample":
        """Create a TrainingSample from a dictionary."""
        return cls(
            sample_id=sample_id or data.get("sample_id", data.get("id", "")),
            context=data.get("context", ""),
            question=data.get("question", data.get("input", "")),
            target=data.get("target", data.get("answer", data.get("output", ""))),
            metadata=data.get("metadata", {}),
        )


class LearningConfig(BaseModel):
    """
    Configuration for learning runs.
    
    Attributes:
        mode: Learning mode (OFFLINE, ONLINE, FROZEN)
        num_epochs: Number of passes over training data
        max_reflection_rounds: Max retries after incorrect answer
        curator_frequency: Run curator every N steps
        eval_frequency: Evaluate on validation every N steps
        save_frequency: Save intermediate expertise every N steps
        token_budget: Max tokens for expertise
        parallel_workers: Workers for parallel evaluation
        use_ground_truth: Whether to use ground truth in reflection
        online_window_size: Window size for online learning
        min_confidence: Minimum reflection confidence to act on
        verbose: Whether to print progress
    """
    mode: LearningMode = LearningMode.OFFLINE
    num_epochs: int = 1
    max_reflection_rounds: int = 3
    curator_frequency: int = 1
    eval_frequency: int = 100
    save_frequency: int = 50
    token_budget: int = 80000
    parallel_workers: int = 10
    use_ground_truth: bool = True
    online_window_size: int = 100
    min_confidence: float = 0.5
    verbose: bool = True


class EvaluationResult(BaseModel):
    """
    Result of evaluating expertise on a dataset.
    
    Attributes:
        accuracy: Accuracy score (0.0 to 1.0)
        correct: Number of correct predictions
        total: Total number of samples
        errors: List of error details
        timestamp: When evaluation was performed
    """
    accuracy: float
    correct: int
    total: int
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    
    @property
    def incorrect(self) -> int:
        """Number of incorrect predictions."""
        return self.total - self.correct


class TrainingStepResult(BaseModel):
    """
    Result of a single training step.
    
    Tracks what happened during training on one sample.
    
    Attributes:
        step: Global step number
        epoch: Current epoch
        sample_id: ID of the sample
        pre_train_correct: Whether initial answer was correct
        post_train_correct: Whether final answer was correct
        reflection_rounds: Number of reflection iterations
        items_used: IDs of expertise items used
        curation_applied: Whether curation was applied
        expertise_token_count: Token count of expertise after step
    """
    step: int
    epoch: int
    sample_id: str
    pre_train_correct: bool
    post_train_correct: bool
    reflection_rounds: int = 0
    items_used: List[str] = Field(default_factory=list)
    curation_applied: bool = False
    expertise_token_count: int = 0
    
    @property
    def improved(self) -> bool:
        """Whether the step resulted in improvement."""
        return not self.pre_train_correct and self.post_train_correct


class LearningResult(BaseModel):
    """
    Result of a complete learning run.
    
    Contains summary statistics and detailed results.
    
    Attributes:
        mode: Learning mode used
        final_accuracy: Final accuracy on test set
        best_accuracy: Best accuracy achieved during training
        initial_accuracy: Initial accuracy before training
        epochs_completed: Number of epochs completed
        total_steps: Total training steps
        training_results: Per-step training results
        validation_results: Periodic validation results
        best_expertise_version: Version of best expertise
        duration_seconds: Total training duration
    """
    mode: LearningMode
    final_accuracy: float
    best_accuracy: float
    initial_accuracy: Optional[float] = None
    epochs_completed: int
    total_steps: int
    training_results: List[TrainingStepResult] = Field(default_factory=list)
    validation_results: List[EvaluationResult] = Field(default_factory=list)
    best_expertise_version: int = 1
    duration_seconds: float = 0.0
    
    @property
    def improvement(self) -> Optional[float]:
        """Improvement from initial to final accuracy."""
        if self.initial_accuracy is not None:
            return self.final_accuracy - self.initial_accuracy
        return None
    
    @property
    def pre_train_accuracy(self) -> float:
        """Accuracy based on pre-train predictions."""
        if not self.training_results:
            return 0.0
        correct = sum(1 for r in self.training_results if r.pre_train_correct)
        return correct / len(self.training_results)
    
    @property
    def post_train_accuracy(self) -> float:
        """Accuracy based on post-train predictions."""
        if not self.training_results:
            return 0.0
        correct = sum(1 for r in self.training_results if r.post_train_correct)
        return correct / len(self.training_results)
    
    @property
    def steps_with_improvement(self) -> int:
        """Number of steps that showed improvement."""
        return sum(1 for r in self.training_results if r.improved)


class OnlineLearningState(BaseModel):
    """
    State for online learning.
    
    Tracks recent samples and performance for incremental learning.
    
    Attributes:
        expertise_id: ID of the expertise being learned
        recent_samples: Sliding window of recent samples
        recent_outcomes: Outcomes of recent samples (True = correct)
        total_samples_seen: Total samples processed
        current_accuracy: Accuracy over the window
        last_curation_step: Step when curation was last run
    """
    expertise_id: str
    recent_samples: List[TrainingSample] = Field(default_factory=list)
    recent_outcomes: List[bool] = Field(default_factory=list)
    total_samples_seen: int = 0
    current_accuracy: float = 0.0
    last_curation_step: int = 0
    
    def add_sample(self, sample: TrainingSample, is_correct: bool, window_size: int = 100) -> None:
        """Add a sample to the sliding window."""
        self.recent_samples.append(sample)
        self.recent_outcomes.append(is_correct)
        self.total_samples_seen += 1
        
        # Maintain window size
        if len(self.recent_samples) > window_size:
            self.recent_samples.pop(0)
            self.recent_outcomes.pop(0)
        
        # Update accuracy
        if self.recent_outcomes:
            self.current_accuracy = sum(self.recent_outcomes) / len(self.recent_outcomes)

