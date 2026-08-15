"""
Content Filter Middleware implementation.
"""

from typing import Optional

from ctxforge.middleware.base import BaseMiddleware, StopChainException
from ctxforge.middleware.content.filters import (
    FilterAction,
    FilterResult,
    IContentFilter,
    KeywordFilter,
)
from ctxforge.middleware.protocol import MiddlewareContext, NextFunction


class ContentFilterMiddleware(BaseMiddleware):
    """
    Content filtering middleware.
    
    Checks input and output for inappropriate or harmful content.
    
    Example:
        # Create filter with blocked words
        keyword_filter = KeywordFilter()
        keyword_filter.add_keywords("profanity", ["badword"], action=FilterAction.BLOCK)
        
        middleware = ContentFilterMiddleware(
            filter=keyword_filter,
            filter_input=True,
            filter_response=True,
        )
    """
    
    def __init__(
        self,
        filter: Optional[IContentFilter] = None,
        filter_input: bool = True,
        filter_response: bool = True,
        stop_on_block: bool = True,
        redact_on_match: bool = False,
        enabled: bool = True,
    ):
        """
        Initialize the middleware.
        
        Args:
            filter: Content filter to use
            filter_input: Whether to filter user input
            filter_response: Whether to filter agent response
            stop_on_block: Stop chain on BLOCK action
            redact_on_match: Redact matched content
            enabled: Whether middleware is enabled
        """
        super().__init__(enabled)
        
        self._filter = filter or KeywordFilter()
        self._filter_input = filter_input
        self._filter_response = filter_response
        self._stop_on_block = stop_on_block
        self._redact_on_match = redact_on_match
    
    @property
    def name(self) -> str:
        """Middleware identifier."""
        return "content_filter"
    
    @property
    def filter(self) -> IContentFilter:
        """The content filter."""
        return self._filter
    
    async def _do_process(
        self,
        context: MiddlewareContext,
        next: NextFunction,
    ) -> MiddlewareContext:
        """
        Filter content in the context.
        
        Args:
            context: The middleware context
            next: Next middleware function
            
        Returns:
            Processed context
        """
        combined_result = FilterResult()
        
        # Filter input
        if self._filter_input and context.processed_input:
            input_result = self._filter.filter(context.processed_input)
            
            if input_result.matched:
                context.add_flag("content_filtered_in_input")
                context.record_modification(self.name, {
                    "source": "input",
                    "action": input_result.action.value,
                    "categories": list(input_result.categories),
                    "matches": input_result.matches,
                })
                
                combined_result = combined_result.merge(input_result)
                
                # Redact if configured
                if self._redact_on_match:
                    context.processed_input = self._redact_matches(
                        context.processed_input,
                        input_result.matches,
                    )
        
        # Filter response
        if self._filter_response and context.processed_response:
            response_result = self._filter.filter(context.processed_response)
            
            if response_result.matched:
                context.add_flag("content_filtered_in_response")
                context.record_modification(self.name, {
                    "source": "response",
                    "action": response_result.action.value,
                    "categories": list(response_result.categories),
                    "matches": response_result.matches,
                })
                
                combined_result = combined_result.merge(response_result)
                
                if self._redact_on_match:
                    context.processed_response = self._redact_matches(
                        context.processed_response,
                        response_result.matches,
                    )
        
        # Handle results
        if combined_result.matched:
            context.add_flag("content_filtered")
            context.set_metadata("content_filter_categories", list(combined_result.categories))
            context.set_metadata("content_filter_action", combined_result.action.value)
            
            if combined_result.action == FilterAction.BLOCK and self._stop_on_block:
                raise StopChainException(
                    self.name,
                    combined_result.message or "Content blocked by filter",
                )
        
        return await next(context)
    
    def _redact_matches(self, text: str, matches: list) -> str:
        """
        Redact matched content from text.
        
        Args:
            text: Original text
            matches: Matched strings to redact
            
        Returns:
            Text with matches redacted
        """
        result = text
        for match in matches:
            result = result.replace(match, "[REDACTED]")
        return result

