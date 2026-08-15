"""
Content Filter implementations.

Provides various content filtering strategies.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Pattern, Protocol, Set, runtime_checkable


class FilterAction(Enum):
    """Actions to take when content is filtered."""
    ALLOW = "allow"       # Content is OK
    WARN = "warn"         # Content is flagged but allowed
    BLOCK = "block"       # Content is blocked
    REDACT = "redact"     # Content should be redacted


# Priority ordering for actions: BLOCK > REDACT > WARN > ALLOW
ACTION_PRIORITY: Dict[FilterAction, int] = {
    FilterAction.ALLOW: 0,
    FilterAction.WARN: 1,
    FilterAction.REDACT: 2,
    FilterAction.BLOCK: 3,
}


@dataclass
class FilterResult:
    """
    Result of content filtering.
    
    Attributes:
        action: Recommended action
        matched: Whether any filter matched
        categories: Categories that matched
        matches: Specific matched content
        confidence: Confidence in the match (0.0 to 1.0)
        message: Human-readable message
    """
    action: FilterAction = FilterAction.ALLOW
    matched: bool = False
    categories: Set[str] = field(default_factory=set)
    matches: List[str] = field(default_factory=list)
    confidence: float = 0.0
    message: Optional[str] = None
    
    def merge(self, other: "FilterResult") -> "FilterResult":
        """
        Merge with another result, taking the stricter action.
        
        Args:
            other: Result to merge with
            
        Returns:
            Combined result
        """
        if ACTION_PRIORITY[other.action] > ACTION_PRIORITY[self.action]:
            action = other.action
            message = other.message
        else:
            action = self.action
            message = self.message
        
        return FilterResult(
            action=action,
            matched=self.matched or other.matched,
            categories=self.categories | other.categories,
            matches=self.matches + other.matches,
            confidence=max(self.confidence, other.confidence),
            message=message,
        )


@runtime_checkable
class IContentFilter(Protocol):
    """Protocol for content filters."""
    
    @property
    def name(self) -> str:
        """Filter identifier."""
        ...
    
    def filter(self, text: str) -> FilterResult:
        """
        Filter content.
        
        Args:
            text: Text to filter
            
        Returns:
            FilterResult with action and details
        """
        ...


class KeywordFilter(IContentFilter):
    """
    Keyword-based content filter.
    
    Checks for presence of blocked words/phrases.
    
    Example:
        filter = KeywordFilter()
        filter.add_keywords("profanity", ["badword1", "badword2"])
        filter.add_keywords("violence", ["kill", "attack"], action=FilterAction.BLOCK)
        
        result = filter.filter("This text contains badword1")
    """
    
    def __init__(
        self,
        case_sensitive: bool = False,
        default_action: FilterAction = FilterAction.WARN,
    ):
        """
        Initialize the filter.
        
        Args:
            case_sensitive: Whether matching is case-sensitive
            default_action: Default action for matches
        """
        self._case_sensitive = case_sensitive
        self._default_action = default_action
        
        # Keywords by category
        self._keywords: Dict[str, Set[str]] = {}
        self._actions: Dict[str, FilterAction] = {}
    
    @property
    def name(self) -> str:
        return "keyword"
    
    def add_keywords(
        self,
        category: str,
        keywords: List[str],
        action: Optional[FilterAction] = None,
    ) -> None:
        """
        Add keywords to a category.
        
        Args:
            category: Category name
            keywords: List of keywords
            action: Action for this category
        """
        if category not in self._keywords:
            self._keywords[category] = set()
        
        if self._case_sensitive:
            self._keywords[category].update(keywords)
        else:
            self._keywords[category].update(k.lower() for k in keywords)
        
        if action:
            self._actions[category] = action
    
    def remove_keywords(self, category: str, keywords: List[str]) -> None:
        """Remove keywords from a category."""
        if category not in self._keywords:
            return
        
        if self._case_sensitive:
            self._keywords[category] -= set(keywords)
        else:
            self._keywords[category] -= set(k.lower() for k in keywords)
    
    def filter(self, text: str) -> FilterResult:
        """Filter text for keywords."""
        if not text:
            return FilterResult()
        
        check_text = text if self._case_sensitive else text.lower()
        
        matched_categories: Set[str] = set()
        matched_keywords: List[str] = []
        max_action = FilterAction.ALLOW
        
        for category, keywords in self._keywords.items():
            for keyword in keywords:
                # Check for word boundary match
                pattern = r'\b' + re.escape(keyword) + r'\b'
                if re.search(pattern, check_text, re.IGNORECASE if not self._case_sensitive else 0):
                    matched_categories.add(category)
                    matched_keywords.append(keyword)
                    
                    action = self._actions.get(category, self._default_action)
                    if ACTION_PRIORITY[action] > ACTION_PRIORITY[max_action]:
                        max_action = action
        
        if matched_keywords:
            action_verb = {
                FilterAction.BLOCK: "Blocked",
                FilterAction.WARN: "Flagged",
                FilterAction.REDACT: "Redacted",
                FilterAction.ALLOW: "Matched",
            }.get(max_action, "Detected")
            
            return FilterResult(
                action=max_action,
                matched=True,
                categories=matched_categories,
                matches=matched_keywords,
                confidence=0.95,  # High confidence for exact matches
                message=f"{action_verb} content in categories: {', '.join(matched_categories)}",
            )
        
        return FilterResult()


class RegexFilter(IContentFilter):
    r"""
    Regex-based content filter.
    
    Checks text against regex patterns.
    
    Example:
        filter = RegexFilter()
        filter.add_pattern("spam", r'\b(?:buy now|limited offer)\b', action=FilterAction.WARN)
        filter.add_pattern("injection", r'(?:drop table|select \* from)', action=FilterAction.BLOCK)
    """
    
    def __init__(
        self,
        default_action: FilterAction = FilterAction.WARN,
    ):
        """
        Initialize the filter.
        
        Args:
            default_action: Default action for matches
        """
        self._default_action = default_action
        
        # Patterns by category
        self._patterns: Dict[str, List[Pattern]] = {}
        self._actions: Dict[str, FilterAction] = {}
    
    @property
    def name(self) -> str:
        return "regex"
    
    def add_pattern(
        self,
        category: str,
        pattern: str,
        flags: int = re.IGNORECASE,
        action: Optional[FilterAction] = None,
    ) -> None:
        """
        Add a pattern to a category.
        
        Args:
            category: Category name
            pattern: Regex pattern
            flags: Regex flags
            action: Action for this category
        """
        if category not in self._patterns:
            self._patterns[category] = []
        
        self._patterns[category].append(re.compile(pattern, flags))
        
        if action:
            self._actions[category] = action
    
    def filter(self, text: str) -> FilterResult:
        """Filter text against patterns."""
        if not text:
            return FilterResult()
        
        matched_categories: Set[str] = set()
        matches: List[str] = []
        max_action = FilterAction.ALLOW
        
        for category, patterns in self._patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matched_categories.add(category)
                    matches.append(match.group())
                    
                    action = self._actions.get(category, self._default_action)
                    if ACTION_PRIORITY[action] > ACTION_PRIORITY[max_action]:
                        max_action = action
        
        if matches:
            return FilterResult(
                action=max_action,
                matched=True,
                categories=matched_categories,
                matches=matches,
                confidence=0.90,
                message=f"Pattern matched in categories: {', '.join(matched_categories)}",
            )
        
        return FilterResult()


class CompositeFilter(IContentFilter):
    """
    Combines multiple filters.
    
    Runs all filters and merges results.
    
    Example:
        composite = CompositeFilter()
        composite.add(KeywordFilter())
        composite.add(RegexFilter())
        
        result = composite.filter("Some text to check")
    """
    
    def __init__(self):
        """Initialize the composite filter."""
        self._filters: List[IContentFilter] = []
    
    @property
    def name(self) -> str:
        return "composite"
    
    def add(self, filter: IContentFilter) -> "CompositeFilter":
        """
        Add a filter.
        
        Args:
            filter: Filter to add
            
        Returns:
            Self for chaining
        """
        self._filters.append(filter)
        return self
    
    def remove(self, name: str) -> bool:
        """
        Remove a filter by name.
        
        Args:
            name: Filter name
            
        Returns:
            True if removed
        """
        for i, f in enumerate(self._filters):
            if f.name == name:
                self._filters.pop(i)
                return True
        return False
    
    def filter(self, text: str) -> FilterResult:
        """Run all filters and merge results."""
        if not self._filters:
            return FilterResult()
        
        result = FilterResult()
        
        for f in self._filters:
            filter_result = f.filter(text)
            result = result.merge(filter_result)
        
        return result

