"""
PII Detection module.

Detects personally identifiable information in text using patterns.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Pattern, Set


class PIIType(Enum):
    """Types of PII that can be detected."""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    ADDRESS = "address"
    NAME = "name"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    BANK_ACCOUNT = "bank_account"
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """
    A detected PII match.
    
    Attributes:
        pii_type: Type of PII detected
        value: The matched text
        start: Start position in text
        end: End position in text
        confidence: Confidence score (0.0 to 1.0)
    """
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float = 1.0
    
    def __repr__(self) -> str:
        return f"PIIMatch({self.pii_type.value}: '{self.value[:20]}...' @ {self.start}-{self.end})"


# Default patterns for PII detection
DEFAULT_PATTERNS: Dict[PIIType, List[Pattern]] = {
    PIIType.EMAIL: [
        re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE),
    ],
    PIIType.PHONE: [
        # US phone formats with parentheses
        re.compile(r'\(\d{3}\)\s*\d{3}[-.\s]?\d{4}'),
        # US phone formats without parentheses
        re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),
        re.compile(r'\b\+1[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
        # International format
        re.compile(r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'),
    ],
    PIIType.SSN: [
        re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
    ],
    PIIType.CREDIT_CARD: [
        # Visa, MasterCard, Amex, Discover
        re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9][0-9])[0-9]{12})\b'),
        # With separators
        re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
    ],
    PIIType.IP_ADDRESS: [
        # IPv4
        re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
        # IPv6 (simplified)
        re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),
    ],
    PIIType.DATE_OF_BIRTH: [
        # Common date formats that might be DOB
        re.compile(r'\b(?:born|dob|birthday|birth date)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b', re.IGNORECASE),
        re.compile(r'\b(\d{1,2}[-/]\d{1,2}[-/]\d{4})\b'),  # Lower confidence
    ],
}


class PIIDetector:
    r"""
    Detects PII in text using regex patterns.
    
    Can be configured with custom patterns and sensitivity levels.
    
    Example:
        detector = PIIDetector()
        matches = detector.detect("Contact me at john@example.com or 555-123-4567")
        # Returns matches for email and phone
        
        # Custom patterns
        detector = PIIDetector()
        detector.add_pattern(PIIType.CUSTOM, r'EMPLOYEE-\d{6}')
    """
    
    def __init__(
        self,
        patterns: Optional[Dict[PIIType, List[Pattern]]] = None,
        enabled_types: Optional[Set[PIIType]] = None,
    ):
        """
        Initialize the detector.
        
        Args:
            patterns: Custom patterns (uses defaults if not provided)
            enabled_types: Types to detect (all by default)
        """
        self._patterns = patterns or dict(DEFAULT_PATTERNS)
        self._enabled_types = enabled_types or set(PIIType)
    
    def detect(self, text: str) -> List[PIIMatch]:
        """
        Detect PII in text.
        
        Args:
            text: The text to scan
            
        Returns:
            List of PIIMatch objects
        """
        if not text:
            return []
        
        matches = []
        
        for pii_type, patterns in self._patterns.items():
            if pii_type not in self._enabled_types:
                continue
            
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get the matched value (use group 1 if it exists, else group 0)
                    if match.lastindex:
                        value = match.group(1)
                        start = match.start(1)
                        end = match.end(1)
                    else:
                        value = match.group(0)
                        start = match.start()
                        end = match.end()
                    
                    pii_match = PIIMatch(
                        pii_type=pii_type,
                        value=value,
                        start=start,
                        end=end,
                        confidence=self._calculate_confidence(pii_type, value),
                    )
                    matches.append(pii_match)
        
        # Remove duplicates and overlaps
        matches = self._deduplicate_matches(matches)
        
        # Sort by position
        matches.sort(key=lambda m: m.start)
        
        return matches
    
    def detect_types(self, text: str) -> Set[PIIType]:
        """
        Detect which types of PII are present.
        
        Args:
            text: The text to scan
            
        Returns:
            Set of detected PII types
        """
        matches = self.detect(text)
        return {m.pii_type for m in matches}
    
    def contains_pii(self, text: str) -> bool:
        """
        Check if text contains any PII.
        
        Args:
            text: The text to scan
            
        Returns:
            True if PII is detected
        """
        return len(self.detect(text)) > 0
    
    def add_pattern(
        self,
        pii_type: PIIType,
        pattern: str,
        flags: int = 0,
    ) -> None:
        """
        Add a custom pattern.
        
        Args:
            pii_type: Type of PII the pattern detects
            pattern: Regex pattern string
            flags: Regex flags
        """
        compiled = re.compile(pattern, flags)
        
        if pii_type not in self._patterns:
            self._patterns[pii_type] = []
        
        self._patterns[pii_type].append(compiled)
    
    def enable_type(self, pii_type: PIIType) -> None:
        """Enable detection of a PII type."""
        self._enabled_types.add(pii_type)
    
    def disable_type(self, pii_type: PIIType) -> None:
        """Disable detection of a PII type."""
        self._enabled_types.discard(pii_type)
    
    def _calculate_confidence(self, pii_type: PIIType, value: str) -> float:
        """
        Calculate confidence score for a match.
        
        Args:
            pii_type: Type of PII
            value: The matched value
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence by type
        base_confidence = {
            PIIType.EMAIL: 0.95,
            PIIType.PHONE: 0.85,
            PIIType.SSN: 0.90,
            PIIType.CREDIT_CARD: 0.80,
            PIIType.IP_ADDRESS: 0.90,
            PIIType.DATE_OF_BIRTH: 0.70,
            PIIType.ADDRESS: 0.60,
            PIIType.NAME: 0.50,
            PIIType.PASSPORT: 0.85,
            PIIType.DRIVERS_LICENSE: 0.80,
            PIIType.BANK_ACCOUNT: 0.75,
            PIIType.CUSTOM: 0.80,
        }
        
        confidence = base_confidence.get(pii_type, 0.70)
        
        # Adjust based on value characteristics
        if pii_type == PIIType.CREDIT_CARD:
            # Luhn check for credit cards
            if self._luhn_check(value.replace('-', '').replace(' ', '')):
                confidence = 0.95
            else:
                confidence = 0.50
        
        return confidence
    
    def _luhn_check(self, card_number: str) -> bool:
        """
        Validate credit card number using Luhn algorithm.
        
        Args:
            card_number: Card number (digits only)
            
        Returns:
            True if valid
        """
        if not card_number.isdigit():
            return False
        
        digits = [int(d) for d in card_number]
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(divmod(d * 2, 10))
        
        return checksum % 10 == 0
    
    def _deduplicate_matches(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """
        Remove duplicate and overlapping matches.
        
        Keeps the match with higher confidence for overlaps.
        
        Args:
            matches: List of matches
            
        Returns:
            Deduplicated list
        """
        if not matches:
            return []
        
        # Sort by confidence (descending) then by start position
        sorted_matches = sorted(matches, key=lambda m: (-m.confidence, m.start))
        
        result = []
        for match in sorted_matches:
            # Check if this overlaps with any existing match
            overlaps = False
            for existing in result:
                if (match.start < existing.end and match.end > existing.start):
                    overlaps = True
                    break
            
            if not overlaps:
                result.append(match)
        
        return result

