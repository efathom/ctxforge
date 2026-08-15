"""
Learning orchestration for the Expertise system.

This module provides offline and online learning capabilities,
inspired by the ACE framework's training loop.

Components:
- LearningMode, LearningConfig: Configuration for learning runs
- TrainingSample, EvaluationResult: Data models
- IDataProcessor: Protocol for task-specific data processing
- ExpertiseEvaluator: Evaluates expertise performance
- ExpertiseTrainer: Orchestrates learning loops

Usage:
    from ctxforge.expertise.learning import (
        ExpertiseTrainer,
        ExpertiseEvaluator,
        LearningConfig,
        LearningMode,
        TrainingSample,
    )
    
    # Configure learning
    config = LearningConfig(
        mode=LearningMode.OFFLINE,
        num_epochs=3,
        max_reflection_rounds=3,
    )
    
    # Create trainer
    trainer = ExpertiseTrainer(
        llm_provider=llm,
        expertise_store=store,
        expertise_retriever=retriever,
        reflector=reflector,
        curator=curator,
        data_processor=processor,
    )
    
    # Run learning
    result = await trainer.run_offline_learning(
        expertise=expertise,
        train_samples=train_data,
        val_samples=val_data,
        config=config,
    )
"""

from ctxforge.expertise.learning.data_processor import (
    BaseDataProcessor,
    ExactMatchProcessor,
    IDataProcessor,
    MultipleChoiceProcessor,
    NumericMatchProcessor,
)
from ctxforge.expertise.learning.evaluator import (
    ExpertiseEvaluator,
)
from ctxforge.expertise.learning.models import (
    EvaluationResult,
    LearningConfig,
    LearningMode,
    LearningResult,
    OnlineLearningState,
    TrainingSample,
    TrainingStepResult,
)
from ctxforge.expertise.learning.trainer import (
    ExpertiseTrainer,
)

__all__ = [
    # Models
    "LearningMode",
    "LearningConfig",
    "TrainingSample",
    "EvaluationResult",
    "TrainingStepResult",
    "LearningResult",
    "OnlineLearningState",
    # Data Processor
    "IDataProcessor",
    "BaseDataProcessor",
    "ExactMatchProcessor",
    "NumericMatchProcessor",
    "MultipleChoiceProcessor",
    # Evaluator
    "ExpertiseEvaluator",
    # Trainer
    "ExpertiseTrainer",
]

