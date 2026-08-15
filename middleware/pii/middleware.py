"""
PII Middleware implementation.

Combines detector and redactor into a middleware component.
"""

from typing import Optional, Set

from ctxforge.middleware.base import BaseMiddleware
from ctxforge.middleware.pii.detector import PIIDetector, PIIType
from ctxforge.middleware.pii.redactor import PIIRedactor, RedactionStrategy
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction


class PIIMiddleware(BaseMiddleware):
    """
    Middleware for detecting and optionally redacting PII.
    
    Can be configured to:
    - Just detect and flag (for logging/auditing)
    - Detect and redact (for privacy)
    - Stop the chain if PII is detected
    
    Example:
        # Just detect and flag
        middleware = PIIMiddleware(redact=False)
        
        # Detect and redact
        middleware = PIIMiddleware(
            redact=True,
            strategy=RedactionStrategy.REPLACE,
        )
        
        # Stop on PII detection
        middleware = PIIMiddleware(
            redact=False,
            stop_on_pii=True,
        )
    """
    
    def __init__(
        self,
        detector: Optional[PIIDetector] = None,
        redactor: Optional[PIIRedactor] = None,
        redact: bool = True,
        redact_input: bool = True,
        redact_response: bool = True,
        strategy: RedactionStrategy = RedactionStrategy.REPLACE,
        stop_on_pii: bool = False,
        pii_types: Optional[Set[PIIType]] = None,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            detector: Custom PII detector
            redactor: Custom PII redactor
            redact: Whether to redact detected PII
            redact_input: Whether to redact user input
            redact_response: Whether to redact agent response
            strategy: Redaction strategy
            stop_on_pii: Stop the chain if PII is detected
            pii_types: Types of PII to detect (all by default)
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled)
        
        self._detector = detector or PIIDetector(enabled_types=pii_types)
        self._redactor = redactor or PIIRedactor(strategy=strategy)
        self._redact = redact
        self._redact_input = redact_input
        self._redact_response = redact_response
        self._stop_on_pii = stop_on_pii
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "pii"
    
    @property
    def detector(self) -> PIIDetector:
        """The PII detector."""
        return self._detector
    
    @property
    def redactor(self) -> PIIRedactor:
        """The PII redactor."""
        return self._redactor
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Process the context for PII.
        
        Detects PII in input and response, optionally redacts,
        and sets appropriate flags.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context
        """
        from ctxforge.middleware.base import StopChainException
        
        all_matches = []
        
        # Detect and handle input PII
        if context.processed_input:
            input_matches = self._detector.detect(context.processed_input)
            
            if input_matches:
                all_matches.extend(input_matches)
                context.add_flag("pii_detected_in_input")
                
                # Record what was found
                context.record_modification(self.name, {
                    "source": "input",
                    "matches": [
                        {
                            "type": m.pii_type.value,
                            "start": m.start,
                            "end": m.end,
                            "confidence": m.confidence,
                        }
                        for m in input_matches
                    ],
                })
                
                if self._redact and self._redact_input:
                    context.processed_input = self._redactor.redact(
                        context.processed_input,
                        input_matches,
                    )
        
        # Detect and handle response PII
        if context.processed_response:
            response_matches = self._detector.detect(context.processed_response)
            
            if response_matches:
                all_matches.extend(response_matches)
                context.add_flag("pii_detected_in_response")
                
                context.record_modification(self.name, {
                    "source": "response",
                    "matches": [
                        {
                            "type": m.pii_type.value,
                            "start": m.start,
                            "end": m.end,
                            "confidence": m.confidence,
                        }
                        for m in response_matches
                    ],
                })
                
                if self._redact and self._redact_response:
                    context.processed_response = self._redactor.redact(
                        context.processed_response,
                        response_matches,
                    )
        
        # Set general flag if any PII was found
        if all_matches:
            context.add_flag("pii_detected")
            context.set_metadata("pii_types_found", [
                m.pii_type.value for m in all_matches
            ])
            context.set_metadata("pii_count", len(all_matches))
            
            # Stop chain if configured
            if self._stop_on_pii:
                raise StopChainException(
                    self.name,
                    f"PII detected: {len(all_matches)} matches",
                )
        
        return await next(context)

