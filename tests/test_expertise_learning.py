"""
Tests for expertise learning orchestration.

Tests the learning models, data processors, evaluator, and trainer.
"""

from typing import Any, Dict, List, Optional, Tuple

import pytest

from ctxforge.core.expertise import (
    CompletedTurn,
    CurationPlan,
    Expertise,
    ExpertiseItem,
    ExpertiseSection,
    ReflectionResult,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.expertise.learning.data_processor import (
    ExactMatchProcessor,
    MultipleChoiceProcessor,
    NumericMatchProcessor,
)
from ctxforge.expertise.learning.evaluator import ExpertiseEvaluator
from ctxforge.expertise.learning.models import (
    EvaluationResult,
    LearningConfig,
    LearningMode,
    LearningResult,
    OnlineLearningState,
    TrainingSample,
    TrainingStepResult,
)
from ctxforge.expertise.learning.trainer import ExpertiseTrainer

# =============================================================================
# Mock Implementations
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing."""
    
    def __init__(self, responses: Optional[List[str]] = None):
        self._responses = responses or ["The answer is 42"]
        self._call_count = 0
    
    async def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        response = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return response


class MockExpertiseStore:
    """Mock expertise store for testing."""
    
    def __init__(self):
        self._store: Dict[str, Expertise] = {}
        self.save_count = 0
    
    async def get(self, expertise_id: str) -> Optional[Expertise]:
        return self._store.get(expertise_id)
    
    async def save(self, expertise: Expertise) -> None:
        self._store[expertise.expertise_id] = expertise
        self.save_count += 1
    
    async def delete(self, expertise_id: str) -> bool:
        if expertise_id in self._store:
            del self._store[expertise_id]
            return True
        return False


class MockExpertiseRetriever:
    """Mock retriever for testing."""
    
    def __init__(self, items: Optional[List[ExpertiseItem]] = None):
        self._items = items or []
    
    async def retrieve(
        self,
        query: str,
        expertise_id: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[ExpertiseItem]:
        return self._items[:top_k]


class MockReflector:
    """Mock reflector for testing."""
    
    def __init__(
        self,
        feedback: Optional[Dict[str, UsageFeedback]] = None,
        confidence: float = 0.8,
    ):
        self._feedback = feedback or {}
        self._confidence = confidence
    
    async def reflect(
        self,
        turn: CompletedTurn,
        items_used: List[ExpertiseItem],
        outcome: TurnOutcome,
    ) -> ReflectionResult:
        feedback = dict(self._feedback)
        if not feedback:
            for item in items_used:
                if outcome == TurnOutcome.SUCCESS:
                    feedback[item.item_id] = UsageFeedback.HELPFUL
                else:
                    feedback[item.item_id] = UsageFeedback.NEUTRAL
        
        return ReflectionResult(
            item_feedback=feedback,
            confidence=self._confidence,
        )


class MockCurator:
    """Mock curator for testing."""
    
    async def curate(
        self,
        expertise: Expertise,
        reflection: ReflectionResult,
        usage_stats: Dict[str, Any],
    ) -> Tuple[Expertise, CurationPlan]:
        return expertise, CurationPlan(reasoning="Mock curation")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_expertise():
    """Create sample expertise."""
    expertise = Expertise(
        expertise_id="test-exp",
        name="Test Expertise",
    )
    expertise.add_item(
        section=ExpertiseSection.STRATEGIES,
        content="Always double-check calculations",
    )
    return expertise


@pytest.fixture
def sample_items(sample_expertise):
    """Get sample expertise items."""
    return sample_expertise.active_items


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    return [
        TrainingSample(
            sample_id="s1",
            question="What is 2 + 2?",
            target="4",
        ),
        TrainingSample(
            sample_id="s2",
            question="What is the capital of France?",
            target="Paris",
        ),
        TrainingSample(
            sample_id="s3",
            context="In mathematics, pi is approximately 3.14159.",
            question="What is pi rounded to 2 decimal places?",
            target="3.14",
        ),
    ]


# =============================================================================
# Learning Models Tests
# =============================================================================


class TestLearningModels:
    """Tests for learning data models."""
    
    def test_learning_mode_values(self):
        """Test LearningMode enum values."""
        assert LearningMode.OFFLINE.value == "offline"
        assert LearningMode.ONLINE.value == "online"
        assert LearningMode.FROZEN.value == "frozen"
    
    def test_training_sample_creation(self):
        """Test TrainingSample creation."""
        sample = TrainingSample(
            sample_id="test-1",
            context="Background info",
            question="What is X?",
            target="42",
            metadata={"difficulty": "easy"},
        )
        
        assert sample.sample_id == "test-1"
        assert sample.context == "Background info"
        assert sample.question == "What is X?"
        assert sample.target == "42"
        assert sample.metadata["difficulty"] == "easy"
    
    def test_training_sample_from_dict(self):
        """Test TrainingSample.from_dict."""
        data = {
            "id": "s1",
            "question": "Test question",
            "answer": "Test answer",
            "context": "Test context",
        }
        
        sample = TrainingSample.from_dict(data)
        
        assert sample.sample_id == "s1"
        assert sample.question == "Test question"
        assert sample.target == "Test answer"
    
    def test_learning_config_defaults(self):
        """Test LearningConfig default values."""
        config = LearningConfig()
        
        assert config.mode == LearningMode.OFFLINE
        assert config.num_epochs == 1
        assert config.max_reflection_rounds == 3
        assert config.curator_frequency == 1
        assert config.eval_frequency == 100
        assert config.use_ground_truth is True
    
    def test_evaluation_result_properties(self):
        """Test EvaluationResult properties."""
        result = EvaluationResult(
            accuracy=0.8,
            correct=8,
            total=10,
        )
        
        assert result.incorrect == 2
    
    def test_training_step_result_improved(self):
        """Test TrainingStepResult.improved property."""
        # Improved case
        result1 = TrainingStepResult(
            step=1, epoch=1, sample_id="s1",
            pre_train_correct=False,
            post_train_correct=True,
        )
        assert result1.improved is True
        
        # Not improved - already correct
        result2 = TrainingStepResult(
            step=1, epoch=1, sample_id="s1",
            pre_train_correct=True,
            post_train_correct=True,
        )
        assert result2.improved is False
        
        # Not improved - still incorrect
        result3 = TrainingStepResult(
            step=1, epoch=1, sample_id="s1",
            pre_train_correct=False,
            post_train_correct=False,
        )
        assert result3.improved is False
    
    def test_learning_result_properties(self):
        """Test LearningResult properties."""
        training = [
            TrainingStepResult(
                step=1, epoch=1, sample_id="s1",
                pre_train_correct=True, post_train_correct=True,
            ),
            TrainingStepResult(
                step=2, epoch=1, sample_id="s2",
                pre_train_correct=False, post_train_correct=True,
            ),
            TrainingStepResult(
                step=3, epoch=1, sample_id="s3",
                pre_train_correct=False, post_train_correct=False,
            ),
        ]
        
        result = LearningResult(
            mode=LearningMode.OFFLINE,
            final_accuracy=0.67,
            best_accuracy=0.67,
            initial_accuracy=0.33,
            epochs_completed=1,
            total_steps=3,
            training_results=training,
        )
        
        assert result.improvement == pytest.approx(0.34, abs=0.01)
        assert result.pre_train_accuracy == pytest.approx(1/3, abs=0.01)
        assert result.post_train_accuracy == pytest.approx(2/3, abs=0.01)
        assert result.steps_with_improvement == 1
    
    def test_online_learning_state(self):
        """Test OnlineLearningState."""
        state = OnlineLearningState(expertise_id="test-exp")
        
        sample = TrainingSample(sample_id="s1", question="Q", target="A")
        
        # Add samples
        state.add_sample(sample, is_correct=True, window_size=3)
        assert state.total_samples_seen == 1
        assert state.current_accuracy == 1.0
        
        state.add_sample(sample, is_correct=False, window_size=3)
        assert state.total_samples_seen == 2
        assert state.current_accuracy == 0.5
        
        # Window slides
        state.add_sample(sample, is_correct=True, window_size=3)
        state.add_sample(sample, is_correct=True, window_size=3)
        assert len(state.recent_samples) == 3  # Capped at window size


# =============================================================================
# Data Processor Tests
# =============================================================================


class TestExactMatchProcessor:
    """Tests for ExactMatchProcessor."""
    
    def test_answer_is_correct_exact(self):
        """Test exact matching."""
        processor = ExactMatchProcessor()
        
        assert processor.answer_is_correct("Paris", "Paris") is True
        assert processor.answer_is_correct("paris", "Paris") is True  # Case insensitive
        assert processor.answer_is_correct("London", "Paris") is False
    
    def test_answer_is_correct_with_punctuation(self):
        """Test matching with punctuation stripping."""
        processor = ExactMatchProcessor(strip_punctuation=True)
        
        assert processor.answer_is_correct("Paris!", "Paris") is True
        assert processor.answer_is_correct("Paris.", "Paris") is True
    
    def test_case_sensitive(self):
        """Test case-sensitive matching."""
        processor = ExactMatchProcessor(case_sensitive=True)
        
        assert processor.answer_is_correct("Paris", "Paris") is True
        assert processor.answer_is_correct("paris", "Paris") is False
    
    def test_extract_answer(self):
        """Test answer extraction."""
        processor = ExactMatchProcessor()
        
        assert processor.extract_answer("The answer is Paris") == "Paris"
        assert processor.extract_answer("Answer: London") == "London"
    
    def test_process_raw_data(self):
        """Test raw data processing."""
        processor = ExactMatchProcessor()
        
        raw_data = [
            {"id": "1", "question": "Q1", "target": "A1", "context": "C1"},
            {"id": "2", "question": "Q2", "target": "A2"},
        ]
        
        samples = processor.process_raw_data(raw_data)
        
        assert len(samples) == 2
        assert samples[0].sample_id == "1"
        assert samples[0].question == "Q1"
        assert samples[0].target == "A1"
        assert samples[0].context == "C1"
    
    def test_evaluate_accuracy(self):
        """Test batch accuracy evaluation."""
        processor = ExactMatchProcessor()
        
        predictions = ["Paris", "London", "Berlin"]
        ground_truths = ["Paris", "Paris", "Berlin"]
        
        accuracy = processor.evaluate_accuracy(predictions, ground_truths)
        assert accuracy == pytest.approx(2/3, abs=0.01)


class TestNumericMatchProcessor:
    """Tests for NumericMatchProcessor."""
    
    def test_answer_is_correct_exact(self):
        """Test exact numeric matching."""
        processor = NumericMatchProcessor()
        
        assert processor.answer_is_correct("42", "42") is True
        assert processor.answer_is_correct("42.0", "42") is True
        assert processor.answer_is_correct("43", "42") is False
    
    def test_answer_is_correct_with_tolerance(self):
        """Test numeric matching with tolerance."""
        processor = NumericMatchProcessor(tolerance=0.1)
        
        assert processor.answer_is_correct("3.14", "3.14159") is True
        assert processor.answer_is_correct("3.0", "3.14159") is False
    
    def test_extract_answer(self):
        """Test numeric answer extraction."""
        processor = NumericMatchProcessor()
        
        assert processor.extract_answer("The answer is 42") == "42"
        assert processor.extract_answer("Result: 3.14") == "3.14"
        assert processor.extract_answer("x = 100") == "100"


class TestMultipleChoiceProcessor:
    """Tests for MultipleChoiceProcessor."""
    
    def test_answer_is_correct(self):
        """Test choice matching."""
        processor = MultipleChoiceProcessor()
        
        assert processor.answer_is_correct("A", "A") is True
        assert processor.answer_is_correct("a", "A") is True
        assert processor.answer_is_correct("B", "A") is False
    
    def test_extract_answer(self):
        """Test choice extraction."""
        processor = MultipleChoiceProcessor()
        
        assert processor.extract_answer("The answer is A") == "A"
        assert processor.extract_answer("Choice: B") == "B"
        assert processor.extract_answer("(C)") == "C"
    
    def test_process_raw_data_with_choices(self):
        """Test processing data with choices."""
        processor = MultipleChoiceProcessor()
        
        raw_data = [{
            "id": "1",
            "question": "What color is the sky?",
            "choices": ["Red", "Blue", "Green"],
            "target": "B",
        }]
        
        samples = processor.process_raw_data(raw_data)
        
        assert len(samples) == 1
        assert "A. Red" in samples[0].question
        assert "B. Blue" in samples[0].question
        assert samples[0].target == "B"


# =============================================================================
# Evaluator Tests
# =============================================================================


class TestExpertiseEvaluator:
    """Tests for ExpertiseEvaluator."""
    
    @pytest.mark.asyncio
    async def test_evaluate_all_correct(self, sample_expertise, sample_items):
        """Test evaluation with all correct answers."""
        llm = MockLLMProvider(responses=["The answer is 4"])
        processor = ExactMatchProcessor()
        retriever = MockExpertiseRetriever(items=sample_items)
        
        evaluator = ExpertiseEvaluator(
            llm_provider=llm,
            data_processor=processor,
            expertise_retriever=retriever,
        )
        
        samples = [
            TrainingSample(sample_id="s1", question="2+2?", target="4"),
        ]
        
        result = await evaluator.evaluate(sample_expertise, samples)
        
        assert result.accuracy == 1.0
        assert result.correct == 1
        assert result.total == 1
        assert len(result.errors) == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_with_errors(self, sample_expertise, sample_items):
        """Test evaluation with some errors."""
        llm = MockLLMProvider(responses=["The answer is Paris"])
        processor = ExactMatchProcessor()
        retriever = MockExpertiseRetriever(items=sample_items)
        
        evaluator = ExpertiseEvaluator(
            llm_provider=llm,
            data_processor=processor,
            expertise_retriever=retriever,
        )
        
        samples = [
            TrainingSample(sample_id="s1", question="Capital of France?", target="Paris"),
            TrainingSample(sample_id="s2", question="Capital of UK?", target="London"),
        ]
        
        result = await evaluator.evaluate(sample_expertise, samples)
        
        assert result.accuracy == 0.5
        assert result.correct == 1
        assert len(result.errors) == 1
        assert result.errors[0]["sample_id"] == "s2"
    
    @pytest.mark.asyncio
    async def test_evaluate_empty_samples(self, sample_expertise):
        """Test evaluation with no samples."""
        llm = MockLLMProvider()
        processor = ExactMatchProcessor()
        retriever = MockExpertiseRetriever()
        
        evaluator = ExpertiseEvaluator(
            llm_provider=llm,
            data_processor=processor,
            expertise_retriever=retriever,
        )
        
        result = await evaluator.evaluate(sample_expertise, [])
        
        assert result.accuracy == 0.0
        assert result.total == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_single_sample(self, sample_expertise, sample_items):
        """Test single sample evaluation."""
        llm = MockLLMProvider(responses=["The answer is 4"])
        processor = ExactMatchProcessor()
        retriever = MockExpertiseRetriever(items=sample_items)
        
        evaluator = ExpertiseEvaluator(
            llm_provider=llm,
            data_processor=processor,
            expertise_retriever=retriever,
        )
        
        sample = TrainingSample(sample_id="s1", question="2+2?", target="4")
        
        is_correct, predicted, items = await evaluator.evaluate_single_sample(
            sample_expertise, sample
        )
        
        assert is_correct is True
        assert predicted == "4"


# =============================================================================
# Trainer Tests
# =============================================================================


class TestExpertiseTrainer:
    """Tests for ExpertiseTrainer."""
    
    @pytest.fixture
    def trainer_components(self, sample_expertise, sample_items):
        """Create trainer components."""
        llm = MockLLMProvider(responses=["The answer is 4"])
        store = MockExpertiseStore()
        retriever = MockExpertiseRetriever(items=sample_items)
        reflector = MockReflector()
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        return {
            "llm": llm,
            "store": store,
            "retriever": retriever,
            "reflector": reflector,
            "curator": curator,
            "processor": processor,
        }
    
    @pytest.mark.asyncio
    async def test_trainer_initialization(self, trainer_components):
        """Test trainer initialization."""
        trainer = ExpertiseTrainer(
            llm_provider=trainer_components["llm"],
            expertise_store=trainer_components["store"],
            expertise_retriever=trainer_components["retriever"],
            reflector=trainer_components["reflector"],
            curator=trainer_components["curator"],
            data_processor=trainer_components["processor"],
        )
        
        assert trainer is not None
    
    @pytest.mark.asyncio
    async def test_train_single_sample_correct(
        self, sample_expertise, trainer_components
    ):
        """Test training on a single correct sample."""
        trainer = ExpertiseTrainer(**{
            "llm_provider": trainer_components["llm"],
            "expertise_store": trainer_components["store"],
            "expertise_retriever": trainer_components["retriever"],
            "reflector": trainer_components["reflector"],
            "curator": trainer_components["curator"],
            "data_processor": trainer_components["processor"],
        })
        
        sample = TrainingSample(sample_id="s1", question="2+2?", target="4")
        config = LearningConfig(verbose=False)
        
        result = await trainer._train_single_sample(
            expertise=sample_expertise,
            sample=sample,
            step=1,
            epoch=1,
            config=config,
        )
        
        assert result.pre_train_correct is True
        assert result.post_train_correct is True
        assert result.sample_id == "s1"
    
    @pytest.mark.asyncio
    async def test_train_single_sample_incorrect_then_correct(
        self, sample_expertise, sample_items
    ):
        """Test training that improves through reflection."""
        # First response wrong, second response correct
        llm = MockLLMProvider(responses=["The answer is 5", "The answer is 4"])
        store = MockExpertiseStore()
        retriever = MockExpertiseRetriever(items=sample_items)
        reflector = MockReflector(confidence=0.9)
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        sample = TrainingSample(sample_id="s1", question="2+2?", target="4")
        config = LearningConfig(verbose=False, max_reflection_rounds=1)
        
        result = await trainer._train_single_sample(
            expertise=sample_expertise,
            sample=sample,
            step=1,
            epoch=1,
            config=config,
        )
        
        assert result.pre_train_correct is False
        assert result.reflection_rounds == 1
    
    @pytest.mark.asyncio
    async def test_offline_learning_basic(self, sample_expertise, sample_items):
        """Test basic offline learning."""
        llm = MockLLMProvider(responses=["The answer is 4"])
        store = MockExpertiseStore()
        retriever = MockExpertiseRetriever(items=sample_items)
        reflector = MockReflector()
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        train_samples = [
            TrainingSample(sample_id="s1", question="2+2?", target="4"),
        ]
        
        config = LearningConfig(
            num_epochs=1,
            verbose=False,
            save_frequency=100,  # Don't save during test
        )
        
        result = await trainer.run_offline_learning(
            expertise=sample_expertise,
            train_samples=train_samples,
            config=config,
        )
        
        assert result.mode == LearningMode.OFFLINE
        assert result.epochs_completed == 1
        assert result.total_steps == 1
        assert len(result.training_results) == 1
    
    @pytest.mark.asyncio
    async def test_offline_learning_with_validation(
        self, sample_expertise, sample_items
    ):
        """Test offline learning with validation."""
        llm = MockLLMProvider(responses=["The answer is 4"])
        store = MockExpertiseStore()
        retriever = MockExpertiseRetriever(items=sample_items)
        reflector = MockReflector()
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        train_samples = [
            TrainingSample(sample_id="s1", question="2+2?", target="4"),
        ]
        val_samples = [
            TrainingSample(sample_id="v1", question="3+1?", target="4"),
        ]
        
        config = LearningConfig(
            num_epochs=1,
            eval_frequency=1,  # Evaluate after every step
            verbose=False,
            save_frequency=100,
        )
        
        result = await trainer.run_offline_learning(
            expertise=sample_expertise,
            train_samples=train_samples,
            val_samples=val_samples,
            config=config,
        )
        
        assert len(result.validation_results) == 1
        assert result.best_accuracy == 1.0
    
    @pytest.mark.asyncio
    async def test_online_learning(self, sample_expertise, sample_items):
        """Test online learning."""
        llm = MockLLMProvider(responses=["The answer is 4"])
        store = MockExpertiseStore()
        retriever = MockExpertiseRetriever(items=sample_items)
        reflector = MockReflector()
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        sample = TrainingSample(sample_id="s1", question="2+2?", target="4")
        config = LearningConfig(
            mode=LearningMode.ONLINE,
            verbose=False,
            save_frequency=100,
        )
        
        is_correct, state = await trainer.run_online_learning(
            expertise=sample_expertise,
            sample=sample,
            config=config,
        )
        
        assert is_correct is True
        assert state.total_samples_seen == 1
        assert state.current_accuracy == 1.0
    
    @pytest.mark.asyncio
    async def test_online_learning_state_persistence(
        self, sample_expertise, sample_items
    ):
        """Test that online learning state persists across calls."""
        # Responses cycle: pre1, post1, pre2, post2
        # pre1=4 (correct for target=4), post1=4, pre2=wrong, post2=wrong
        llm = MockLLMProvider(responses=[
            "The answer is 4",    # pre-train sample 1 (correct)
            "The answer is 4",    # post-train sample 1
            "The answer is wrong",  # pre-train sample 2 (wrong for target=6)
            "The answer is wrong",  # post-train sample 2
        ])
        store = MockExpertiseStore()
        retriever = MockExpertiseRetriever(items=sample_items)
        reflector = MockReflector()
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        config = LearningConfig(
            mode=LearningMode.ONLINE,
            verbose=False,
            save_frequency=100,
            max_reflection_rounds=0,  # No reflection for predictable responses
        )
        
        # First sample - correct (target=4, response="4")
        sample1 = TrainingSample(sample_id="s1", question="2+2?", target="4")
        is_correct1, state = await trainer.run_online_learning(
            expertise=sample_expertise,
            sample=sample1,
            config=config,
        )
        
        assert state.total_samples_seen == 1
        assert is_correct1 is True
        
        # Second sample - incorrect (target=6, response="wrong")
        sample2 = TrainingSample(sample_id="s2", question="3+3?", target="6")
        is_correct2, state = await trainer.run_online_learning(
            expertise=sample_expertise,
            sample=sample2,
            state=state,
            config=config,
        )
        
        assert state.total_samples_seen == 2
        assert is_correct2 is False
        assert state.current_accuracy == 0.5  # 1 correct out of 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestLearningIntegration:
    """Integration tests for the learning system."""
    
    @pytest.mark.asyncio
    async def test_full_learning_pipeline(self):
        """Test a complete learning pipeline."""
        # Create expertise
        expertise = Expertise(expertise_id="math-exp", name="Math Expertise")
        expertise.add_item(
            section=ExpertiseSection.FORMULAS,
            content="Addition: a + b means combining two numbers",
        )
        
        # Create components
        llm = MockLLMProvider(responses=["The answer is 4"])
        store = MockExpertiseStore()
        await store.save(expertise)
        
        retriever = MockExpertiseRetriever(items=expertise.active_items)
        reflector = MockReflector()
        curator = MockCurator()
        processor = ExactMatchProcessor()
        
        # Create trainer
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        # Create dataset
        train_data = [
            TrainingSample(sample_id="t1", question="2+2=?", target="4"),
            TrainingSample(sample_id="t2", question="1+3=?", target="4"),
        ]
        test_data = [
            TrainingSample(sample_id="e1", question="3+1=?", target="4"),
        ]
        
        # Run training
        config = LearningConfig(
            num_epochs=1,
            verbose=False,
            save_frequency=100,
        )
        
        result = await trainer.run_offline_learning(
            expertise=expertise,
            train_samples=train_data,
            test_samples=test_data,
            config=config,
        )
        
        # Verify results
        assert result.mode == LearningMode.OFFLINE
        assert result.epochs_completed == 1
        assert result.total_steps == 2
        assert result.final_accuracy == 1.0
        assert len(result.training_results) == 2

