"""
Expertise evaluator for measuring performance on datasets.

Provides parallel evaluation of expertise on training/validation/test sets.
Inspired by ACE's evaluate_test_set function.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, List

from ctxforge.core.expertise import Expertise, ExpertiseItem
from ctxforge.expertise.learning.data_processor import IDataProcessor
from ctxforge.expertise.learning.models import (
    EvaluationResult,
    TrainingSample,
)
from ctxforge.protocols import ILLMProvider
from ctxforge.protocols.expertise import IExpertiseRetriever

logger = logging.getLogger(__name__)


class ExpertiseEvaluator:
    """
    Evaluates expertise performance on a dataset.
    
    Uses parallel execution for efficiency when evaluating
    many samples.
    
    Example:
        evaluator = ExpertiseEvaluator(
            llm_provider=llm,
            data_processor=processor,
            expertise_retriever=retriever,
        )
        
        result = await evaluator.evaluate(
            expertise=expertise,
            samples=test_samples,
            max_workers=10,
        )
        print(f"Accuracy: {result.accuracy:.2%}")
    """
    
    def __init__(
        self,
        llm_provider: ILLMProvider,
        data_processor: IDataProcessor,
        expertise_retriever: IExpertiseRetriever,
        top_k: int = 10,
    ):
        """
        Initialize the evaluator.
        
        Args:
            llm_provider: LLM for generating responses
            data_processor: Processor for evaluating answers
            expertise_retriever: Retriever for finding relevant items
            top_k: Number of expertise items to retrieve per query
        """
        self._llm = llm_provider
        self._processor = data_processor
        self._retriever = expertise_retriever
        self._top_k = top_k
    
    async def evaluate(
        self,
        expertise: Expertise,
        samples: List[TrainingSample],
        max_workers: int = 10,
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate expertise on a set of samples.
        
        Uses parallel execution with a semaphore to limit concurrency.
        
        Args:
            expertise: The expertise to evaluate
            samples: List of samples to evaluate on
            max_workers: Maximum concurrent evaluations
            verbose: Whether to log progress
            
        Returns:
            EvaluationResult with accuracy and error details
        """
        if not samples:
            return EvaluationResult(
                accuracy=0.0,
                correct=0,
                total=0,
            )
        
        errors: List[dict] = []
        semaphore = asyncio.Semaphore(max_workers)
        
        async def evaluate_single(sample: TrainingSample) -> bool:
            """Evaluate a single sample."""
            async with semaphore:
                try:
                    # 1. Retrieve relevant expertise items
                    items = await self._retriever.retrieve(
                        query=sample.question,
                        expertise_id=expertise.expertise_id,
                        top_k=self._top_k,
                    )
                    
                    # 2. Build prompt with expertise
                    prompt = self._build_prompt(sample, items)
                    
                    # 3. Generate answer
                    response = await self._llm.generate([
                        {"role": "user", "content": prompt}
                    ])
                    
                    # 4. Extract and check answer
                    predicted = self._processor.extract_answer(response)
                    is_correct = self._processor.answer_is_correct(predicted, sample.target)
                    
                    if not is_correct:
                        errors.append({
                            "sample_id": sample.sample_id,
                            "question": sample.question[:100],
                            "predicted": predicted,
                            "ground_truth": sample.target,
                            "items_used": [item.item_id for item in items[:5]],
                        })
                    
                    return is_correct
                    
                except Exception as e:
                    logger.warning(f"Error evaluating sample {sample.sample_id}: {e}")
                    errors.append({
                        "sample_id": sample.sample_id,
                        "error": str(e),
                    })
                    return False
        
        # Run evaluations in parallel
        if verbose:
            logger.info(f"Evaluating {len(samples)} samples with {max_workers} workers...")
        
        results = await asyncio.gather(*[evaluate_single(s) for s in samples])
        correct = sum(1 for r in results if r)
        
        if verbose:
            logger.info(f"Evaluation complete: {correct}/{len(samples)} correct")
        
        return EvaluationResult(
            accuracy=correct / len(samples),
            correct=correct,
            total=len(samples),
            errors=errors,
            timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        )
    
    async def evaluate_single_sample(
        self,
        expertise: Expertise,
        sample: TrainingSample,
    ) -> tuple[bool, str, List[ExpertiseItem]]:
        """
        Evaluate a single sample and return details.
        
        Args:
            expertise: The expertise to use
            sample: The sample to evaluate
            
        Returns:
            Tuple of (is_correct, predicted_answer, items_used)
        """
        # Retrieve items
        items = await self._retriever.retrieve(
            query=sample.question,
            expertise_id=expertise.expertise_id,
            top_k=self._top_k,
        )
        
        # Generate answer
        prompt = self._build_prompt(sample, items)
        response = await self._llm.generate([
            {"role": "user", "content": prompt}
        ])
        
        # Extract and check
        predicted = self._processor.extract_answer(response)
        is_correct = self._processor.answer_is_correct(predicted, sample.target)
        
        return is_correct, predicted, items
    
    def _build_prompt(
        self,
        sample: TrainingSample,
        items: List[Any],
    ) -> str:
        """
        Build prompt with expertise context.
        
        Args:
            sample: The training sample
            items: Retrieved expertise items
            
        Returns:
            Formatted prompt string
        """
        expertise_text = self._format_expertise(items)
        
        parts = ["## EXPERTISE"]
        if expertise_text:
            parts.append(expertise_text)
        else:
            parts.append("(No relevant expertise available)")
        
        if sample.context:
            parts.append("\n## CONTEXT")
            parts.append(sample.context)
        
        parts.append("\n## QUESTION")
        parts.append(sample.question)
        parts.append("\nPlease provide your answer.")
        
        return "\n".join(parts)
    
    def _format_expertise(self, items: List[Any]) -> str:
        """
        Format expertise items in ACE style.
        
        Args:
            items: List of expertise items
            
        Returns:
            Formatted string with all items
        """
        if not items:
            return ""
        
        lines = []
        for item in items:
            if hasattr(item, 'item_id') and hasattr(item, 'content'):
                # ExpertiseItem
                helpful = getattr(item, 'helpful_count', 0)
                harmful = getattr(item, 'harmful_count', 0)
                line = f"[{item.item_id}] helpful={helpful} harmful={harmful} :: {item.content}"
            else:
                # Generic item
                line = str(item)
            lines.append(line)
        
        return "\n".join(lines)

