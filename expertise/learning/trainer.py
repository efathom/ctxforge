"""
Expertise trainer for offline and online learning.

Orchestrates learning loops with reflection and curation,
inspired by ACE's training loop in ace.py.
"""

import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from ctxforge.core.expertise import (
    CompletedTurn,
    Expertise,
    ReflectionResult,
    TurnOutcome,
    UsageFeedback,
)
from ctxforge.expertise.learning.data_processor import IDataProcessor
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
from ctxforge.protocols import ILLMProvider
from ctxforge.protocols.expertise import (
    ICurator,
    IExpertiseRetriever,
    IExpertiseStore,
    IReflector,
)

logger = logging.getLogger(__name__)


# Type alias for progress callback
ProgressCallback = Callable[[int, int, float], Awaitable[None]]


class ExpertiseTrainer:
    """
    Orchestrates expertise learning in offline and online modes.
    
    Inspired by ACE's training loop, this class manages:
    - Offline learning: Batch training on labeled datasets
    - Online learning: Incremental learning during production
    
    Example:
        trainer = ExpertiseTrainer(
            llm_provider=llm,
            expertise_store=store,
            expertise_retriever=retriever,
            reflector=reflector,
            curator=curator,
            data_processor=processor,
        )
        
        # Run offline learning
        result = await trainer.run_offline_learning(
            expertise=expertise,
            train_samples=train_data,
            val_samples=val_data,
            config=LearningConfig(num_epochs=3),
        )
        
        print(f"Final accuracy: {result.final_accuracy:.2%}")
    """
    
    def __init__(
        self,
        llm_provider: ILLMProvider,
        expertise_store: IExpertiseStore,
        expertise_retriever: IExpertiseRetriever,
        reflector: IReflector,
        curator: ICurator,
        data_processor: IDataProcessor,
        evaluator: Optional[ExpertiseEvaluator] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            llm_provider: LLM for generating responses
            expertise_store: Store for persisting expertise
            expertise_retriever: Retriever for finding relevant items
            reflector: Reflector for analyzing turn outcomes
            curator: Curator for evolving expertise
            data_processor: Processor for evaluating answers
            evaluator: Optional evaluator (created if not provided)
        """
        self._llm = llm_provider
        self._store = expertise_store
        self._retriever = expertise_retriever
        self._reflector = reflector
        self._curator = curator
        self._processor = data_processor
        self._evaluator = evaluator or ExpertiseEvaluator(
            llm_provider, data_processor, expertise_retriever
        )
    
    async def run_offline_learning(
        self,
        expertise: Expertise,
        train_samples: List[TrainingSample],
        val_samples: Optional[List[TrainingSample]] = None,
        test_samples: Optional[List[TrainingSample]] = None,
        config: Optional[LearningConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> LearningResult:
        """
        Run offline learning on a labeled dataset.
        
        Workflow (per ACE):
        1. Optional: Run initial test
        2. For each epoch:
            a. For each training sample:
                - Generate answer with current expertise
                - Check correctness against ground truth
                - If incorrect: Reflect and retry (up to max_rounds)
                - Update expertise via curator (at configured frequency)
                - Generate post-train answer
            b. Periodic validation evaluation
            c. Track best expertise
        3. Optional: Run final test with best expertise
        
        Args:
            expertise: The expertise to train
            train_samples: Training data
            val_samples: Optional validation data
            test_samples: Optional test data
            config: Learning configuration
            progress_callback: Optional callback for progress updates
            
        Returns:
            LearningResult with training statistics
        """
        config = config or LearningConfig()
        start_time = time.time()
        
        best_expertise = expertise.model_copy(deep=True)
        best_accuracy = 0.0
        training_results: List[TrainingStepResult] = []
        validation_results: List[EvaluationResult] = []
        
        total_steps = len(train_samples) * config.num_epochs
        
        # Initial test (if test samples provided)
        initial_accuracy = None
        if test_samples:
            if config.verbose:
                logger.info("Running initial test...")
            initial_result = await self._evaluator.evaluate(
                expertise, test_samples, config.parallel_workers, config.verbose
            )
            initial_accuracy = initial_result.accuracy
            if config.verbose:
                logger.info(f"Initial test accuracy: {initial_accuracy:.3f}")
        
        global_step = 0
        
        for epoch in range(1, config.num_epochs + 1):
            if config.verbose:
                logger.info(f"\n=== Epoch {epoch}/{config.num_epochs} ===")
            
            for _step, sample in enumerate(train_samples, 1):
                global_step += 1
                
                # Train on single sample
                step_result = await self._train_single_sample(
                    expertise=expertise,
                    sample=sample,
                    step=global_step,
                    epoch=epoch,
                    config=config,
                )
                training_results.append(step_result)
                
                # Progress callback
                if progress_callback:
                    await progress_callback(
                        global_step,
                        total_steps,
                        1.0 if step_result.post_train_correct else 0.0,
                    )
                
                # Periodic validation
                if val_samples and global_step % config.eval_frequency == 0:
                    if config.verbose:
                        logger.info(f"\n--- Validation at step {global_step} ---")
                    val_result = await self._evaluator.evaluate(
                        expertise, val_samples, config.parallel_workers, config.verbose
                    )
                    validation_results.append(val_result)
                    if config.verbose:
                        logger.info(f"Validation accuracy: {val_result.accuracy:.3f}")
                    
                    # Track best
                    if val_result.accuracy > best_accuracy:
                        best_accuracy = val_result.accuracy
                        best_expertise = expertise.model_copy(deep=True)
                        if config.verbose:
                            logger.info(f"🎉 New best accuracy: {best_accuracy:.3f}")
                
                # Save intermediate expertise
                if global_step % config.save_frequency == 0:
                    await self._store.save(expertise)
        
        # If no validation was done, use the final expertise as best
        if not validation_results:
            best_expertise = expertise.model_copy(deep=True)
        
        # Final test with best expertise
        final_accuracy = best_accuracy
        if test_samples:
            if config.verbose:
                logger.info("\nRunning final test with best expertise...")
            final_result = await self._evaluator.evaluate(
                best_expertise, test_samples, config.parallel_workers, config.verbose
            )
            final_accuracy = final_result.accuracy
            if config.verbose:
                logger.info(f"Final test accuracy: {final_accuracy:.3f}")
        
        # Save best expertise
        await self._store.save(best_expertise)
        
        duration = time.time() - start_time
        
        return LearningResult(
            mode=LearningMode.OFFLINE,
            final_accuracy=final_accuracy,
            best_accuracy=best_accuracy,
            initial_accuracy=initial_accuracy,
            epochs_completed=config.num_epochs,
            total_steps=global_step,
            training_results=training_results,
            validation_results=validation_results,
            best_expertise_version=best_expertise.version,
            duration_seconds=duration,
        )
    
    async def run_online_learning(
        self,
        expertise: Expertise,
        sample: TrainingSample,
        state: Optional[OnlineLearningState] = None,
        config: Optional[LearningConfig] = None,
    ) -> Tuple[bool, OnlineLearningState]:
        """
        Process a single sample for online learning.
        
        Updates expertise incrementally based on each sample,
        maintaining a sliding window for performance tracking.
        
        Args:
            expertise: The expertise to update
            sample: The sample to learn from
            state: Current online learning state
            config: Learning configuration
            
        Returns:
            Tuple of (is_correct, updated_state)
        """
        config = config or LearningConfig(mode=LearningMode.ONLINE)
        
        # Initialize state if needed
        if state is None:
            state = OnlineLearningState(expertise_id=expertise.expertise_id)
        
        # Process the sample
        step_result = await self._train_single_sample(
            expertise=expertise,
            sample=sample,
            step=state.total_samples_seen + 1,
            epoch=1,
            config=config,
        )
        
        # Update state
        state.add_sample(
            sample=sample,
            is_correct=step_result.post_train_correct,
            window_size=config.online_window_size,
        )
        
        # Save expertise periodically
        if state.total_samples_seen % config.save_frequency == 0:
            await self._store.save(expertise)
        
        return step_result.post_train_correct, state
    
    async def _train_single_sample(
        self,
        expertise: Expertise,
        sample: TrainingSample,
        step: int,
        epoch: int,
        config: LearningConfig,
    ) -> TrainingStepResult:
        """
        Train on a single sample with reflection and curation.
        
        Following ACE's _train_single_sample pattern:
        1. Retrieve relevant expertise items
        2. Generate initial answer (pre-train)
        3. If incorrect: Reflect and retry (up to max rounds)
        4. Run curator (at configured frequency)
        5. Generate post-train answer
        
        Args:
            expertise: The expertise to use/update
            sample: The training sample
            step: Current global step
            epoch: Current epoch
            config: Learning configuration
            
        Returns:
            TrainingStepResult with details
        """
        # 1. Retrieve relevant expertise items
        items = await self._retriever.retrieve(
            query=sample.question,
            expertise_id=expertise.expertise_id,
            top_k=10,
        )
        items_used = [item.item_id for item in items if hasattr(item, 'item_id')]
        
        # 2. Generate initial answer (pre-train)
        pre_answer = await self._generate_answer(sample, items, reflection=None)
        pre_correct = self._processor.answer_is_correct(pre_answer, sample.target)
        
        # 3. Reflection loop (if incorrect)
        reflection_rounds = 0
        reflection: Optional[ReflectionResult] = None
        current_answer = pre_answer
        is_correct = pre_correct
        
        if not is_correct:
            for _round_num in range(config.max_reflection_rounds):
                reflection_rounds += 1
                
                # Build completed turn for reflection
                turn = CompletedTurn(
                    user_input=sample.question,
                    assistant_response=current_answer,
                    expected_output=sample.target if config.use_ground_truth else None,
                )
                outcome = TurnOutcome.FAILURE
                
                # Get expertise items as actual ExpertiseItem objects
                expertise_items = [
                    expertise.get_item(item_id)
                    for item_id in items_used
                    if expertise.get_item(item_id)
                ]
                
                # Reflect on error
                reflection = await self._reflector.reflect(turn, expertise_items, outcome)
                
                # Skip if confidence is too low
                if reflection.confidence < config.min_confidence:
                    break
                
                # Update item counts based on feedback
                for item_id, feedback in reflection.item_feedback.items():
                    item = expertise.get_item(item_id)
                    if item:
                        if feedback == UsageFeedback.HELPFUL:
                            item.increment_helpful()
                        elif feedback == UsageFeedback.HARMFUL:
                            item.increment_harmful()
                
                # Regenerate with reflection context
                current_answer = await self._generate_answer(
                    sample, items, reflection=reflection.insights
                )
                
                if self._processor.answer_is_correct(current_answer, sample.target):
                    is_correct = True
                    break
        else:
            # Even if correct, run reflector to tag helpful items
            turn = CompletedTurn(
                user_input=sample.question,
                assistant_response=current_answer,
                expected_output=sample.target if config.use_ground_truth else None,
            )
            
            expertise_items = [
                expertise.get_item(item_id)
                for item_id in items_used
                if expertise.get_item(item_id)
            ]
            
            reflection = await self._reflector.reflect(
                turn, expertise_items, TurnOutcome.SUCCESS
            )
            
            if reflection.confidence >= config.min_confidence:
                for item_id, feedback in reflection.item_feedback.items():
                    item = expertise.get_item(item_id)
                    if item:
                        if feedback == UsageFeedback.HELPFUL:
                            item.increment_helpful()
                        elif feedback == UsageFeedback.HARMFUL:
                            item.increment_harmful()
        
        # 4. Run curator (at configured frequency)
        curation_applied = False
        if step % config.curator_frequency == 0 and reflection:
            try:
                usage_stats = self._get_usage_stats(expertise)
                updated_expertise, plan = await self._curator.curate(
                    expertise, reflection, usage_stats
                )
                
                if plan.has_operations:
                    # Apply changes to expertise
                    expertise.items = updated_expertise.items
                    expertise.increment_version()
                    curation_applied = True
                    
                    if config.verbose:
                        logger.info(f"  Curator applied {plan.operation_count} operations")
            except Exception as e:
                logger.warning(f"Curation failed: {e}")
        
        # 5. Generate post-train answer
        # Reload items (may have changed from curation)
        items = await self._retriever.retrieve(
            query=sample.question,
            expertise_id=expertise.expertise_id,
            top_k=10,
        )
        post_answer = await self._generate_answer(sample, items, reflection=None)
        post_correct = self._processor.answer_is_correct(post_answer, sample.target)
        
        return TrainingStepResult(
            step=step,
            epoch=epoch,
            sample_id=sample.sample_id,
            pre_train_correct=pre_correct,
            post_train_correct=post_correct,
            reflection_rounds=reflection_rounds,
            items_used=items_used,
            curation_applied=curation_applied,
            expertise_token_count=self._estimate_tokens(expertise),
        )
    
    async def _generate_answer(
        self,
        sample: TrainingSample,
        items: List[Any],
        reflection: Optional[str] = None,
    ) -> str:
        """
        Generate answer using LLM with expertise context.
        
        Args:
            sample: The training sample
            items: Retrieved expertise items
            reflection: Optional reflection insights to include
            
        Returns:
            The generated answer text
        """
        # Build prompt
        parts = ["## EXPERTISE"]
        if items:
            expertise_text = self._format_expertise(items)
            parts.append(expertise_text)
        else:
            parts.append("(No relevant expertise available)")
        
        if sample.context:
            parts.append("\n## CONTEXT")
            parts.append(sample.context)
        
        parts.append("\n## QUESTION")
        parts.append(sample.question)
        
        if reflection:
            parts.append("\n## REFLECTION (from previous attempt)")
            parts.append(reflection)
        
        parts.append("\nPlease provide your answer.")
        
        prompt = "\n".join(parts)
        
        # Generate response
        response = await self._llm.generate([
            {"role": "user", "content": prompt}
        ])
        
        return self._processor.extract_answer(response)
    
    def _format_expertise(self, items: List[Any]) -> str:
        """Format expertise items in ACE style."""
        lines = []
        for item in items:
            if hasattr(item, 'item_id') and hasattr(item, 'content'):
                helpful = getattr(item, 'helpful_count', 0)
                harmful = getattr(item, 'harmful_count', 0)
                line = f"[{item.item_id}] helpful={helpful} harmful={harmful} :: {item.content}"
            else:
                line = str(item)
            lines.append(line)
        return "\n".join(lines)
    
    def _estimate_tokens(self, expertise: Expertise) -> int:
        """Estimate token count for expertise."""
        return expertise.estimate_tokens()
    
    def _get_usage_stats(self, expertise: Expertise) -> Dict[str, Any]:
        """Get usage statistics for expertise."""
        total_helpful = sum(item.helpful_count for item in expertise.active_items)
        total_harmful = sum(item.harmful_count for item in expertise.active_items)
        
        return {
            "total_items": expertise.item_count,
            "active_items": expertise.active_item_count,
            "total_helpful": total_helpful,
            "total_harmful": total_harmful,
            "estimated_tokens": expertise.estimate_tokens(),
        }

