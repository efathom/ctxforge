"""
Utility functions for the extraction pipeline.

Provides shared functionality for text processing, normalization,
and similarity calculations.
"""

import re
from typing import Any, List


def normalize_text(text: str) -> str:
    """
    Normalize text for consistent processing.
    
    - Strips whitespace
    - Collapses multiple spaces
    - Lowercases for comparison
    
    Args:
        text: The text to normalize
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Strip and collapse whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


def extract_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Handles common sentence boundaries.
    
    Args:
        text: The text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Split on sentence-ending punctuation
    # Handles abbreviations like "Mr." by requiring space after period
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Filter empty sentences and strip
    return [s.strip() for s in sentences if s.strip()]


def extract_quoted_text(text: str) -> List[str]:
    """
    Extract text within quotes.
    
    Args:
        text: The text to search
        
    Returns:
        List of quoted strings
    """
    # Match both single and double quotes
    pattern = r'["\']([^"\']+)["\']'
    matches = re.findall(pattern, text)
    return matches


def extract_key_phrases(text: str, min_words: int = 2, max_words: int = 5) -> List[str]:
    """
    Extract potential key phrases from text.
    
    Simple extraction based on noun-phrase-like patterns.
    
    Args:
        text: The text to analyze
        min_words: Minimum words in phrase
        max_words: Maximum words in phrase
        
    Returns:
        List of potential key phrases
    """
    if not text:
        return []
    
    # Simple approach: extract sequences of capitalized words
    # and common noun phrases
    phrases = []
    
    # Find capitalized word sequences (proper nouns, names, etc.)
    cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
    phrases.extend(re.findall(cap_pattern, text))
    
    # Find words after certain markers
    markers = ['my', 'the', 'a', 'an', 'their', 'our']
    for marker in markers:
        pattern = rf'\b{marker}\s+(\w+(?:\s+\w+){{0,{max_words-1}}})\b'
        matches = re.findall(pattern, text.lower())
        phrases.extend(matches)
    
    # Filter by word count
    result = []
    for phrase in phrases:
        word_count = len(phrase.split())
        if min_words <= word_count <= max_words:
            result.append(phrase.strip())
    
    return list(set(result))  # Remove duplicates


def clean_extraction(text: str) -> str:
    """
    Clean extracted memory content.
    
    Removes common artifacts and normalizes.
    
    Args:
        text: The extracted text
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    text = text.strip()
    
    # Remove leading articles for cleaner storage
    text = re.sub(r'^(that\s+)?', '', text, flags=re.IGNORECASE)
    
    # Ensure proper sentence ending
    if text and text[-1] not in '.!?':
        text += '.'
    
    # Capitalize first letter
    if text:
        text = text[0].upper() + text[1:]
    
    return text


def parse_confidence(value: Any, default: float = 0.5) -> float:
    """
    Parse a confidence value from various formats.
    
    Args:
        value: String or numeric representation of confidence
        default: Default value if parsing fails
        
    Returns:
        Float between 0.0 and 1.0
    """
    try:
        # Handle percentage format
        if isinstance(value, str) and '%' in value:
            return max(0.0, min(1.0, float(value.replace('%', '')) / 100.0))
        
        conf = float(value)
        
        # Only treat as percentage if it's a whole number > 1 and <= 100
        # (e.g., "75" -> 0.75, but "1.5" -> clamp to 1.0)
        if conf > 1.0 and conf <= 100.0 and conf == int(conf):
            # Likely a percentage value (integer)
            conf = conf / 100.0
        
        return max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        return default


def merge_tags(tags_list: List[List[str]]) -> List[str]:
    """
    Merge multiple tag lists, removing duplicates.
    
    Args:
        tags_list: List of tag lists
        
    Returns:
        Merged list of unique tags
    """
    all_tags = set()
    for tags in tags_list:
        all_tags.update(tag.lower().strip() for tag in tags if tag)
    return sorted(all_tags)


def extract_json_from_text(text: str) -> str | None:
    """
    Extract JSON from text that may contain markdown or other content.
    
    Handles responses that may have JSON embedded in markdown
    code blocks or surrounded by other text.
    
    Args:
        text: The text that may contain JSON
        
    Returns:
        JSON string if found, None otherwise
    """
    import json
    
    text = text.strip()
    
    # Look for JSON in markdown code blocks FIRST
    json_block = re.search(r'```(?:json)?\s*\n?([\[\{][\s\S]*?[\]\}])\s*\n?```', text)
    if json_block:
        return json_block.group(1).strip()
    
    # Try direct parse if starts with JSON
    if text.startswith('[') or text.startswith('{'):
        # Find the matching closing bracket/brace
        bracket_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char in '[{':
                    bracket_count += 1
                elif char in ']}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        return text[:i+1]
        
        return text  # Return as-is if balanced
    
    # Try to find array in text
    array_match = re.search(r'\[[\s\S]*?\]', text)
    if array_match:
        candidate = array_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    
    # Try to find object in text
    obj_match = re.search(r'\{[\s\S]*?\}', text)
    if obj_match:
        candidate = obj_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    
    return None


def estimate_memory_importance(
    content: str,
    source_text: str,
    is_explicit: bool = False,
) -> float:
    """
    Estimate the importance of a potential memory.
    
    Heuristic-based importance scoring.
    
    Args:
        content: The memory content
        source_text: Original text it was extracted from
        is_explicit: Whether user explicitly stated this
        
    Returns:
        Importance score (0.0 to 1.0)
    """
    score = 0.5  # Base score
    
    # Explicit statements are more important
    if is_explicit:
        score += 0.2
    
    # Longer, more specific content is often more important
    word_count = len(content.split())
    if word_count > 5:
        score += 0.1
    
    # Strong sentiment indicators suggest importance
    strong_words = ['love', 'hate', 'always', 'never', 'favorite', 'best', 'worst']
    if any(word in content.lower() for word in strong_words):
        score += 0.1
    
    # Personal pronouns indicate personal information
    personal_words = ['i', 'my', 'me', 'myself']
    if any(word in source_text.lower().split() for word in personal_words):
        score += 0.1
    
    return min(1.0, score)
