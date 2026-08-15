"""
PII Redaction module.

Provides strategies for redacting/masking detected PII.
"""

from enum import Enum
from typing import Callable, Dict, List, Optional

from ctxforge.middleware.pii.detector import PIIMatch, PIIType


class RedactionStrategy(Enum):
    """Strategies for redacting PII."""
    MASK = "mask"           # Replace with asterisks: john@example.com -> ****@*******.***
    REPLACE = "replace"     # Replace with placeholder: [EMAIL]
    HASH = "hash"           # Replace with hash: john@example.com -> a1b2c3d4
    PARTIAL = "partial"     # Partial mask: john@example.com -> j***@e******.com
    REMOVE = "remove"       # Remove entirely


class PIIRedactor:
    """
    Redacts PII from text using various strategies.
    
    Example:
        redactor = PIIRedactor(strategy=RedactionStrategy.REPLACE)
        
        text = "Email me at john@example.com"
        matches = detector.detect(text)
        redacted = redactor.redact(text, matches)
        # "Email me at [EMAIL]"
        
        # Custom replacements
        redactor = PIIRedactor(
            strategy=RedactionStrategy.REPLACE,
            placeholders={
                PIIType.EMAIL: "[REDACTED_EMAIL]",
                PIIType.PHONE: "[REDACTED_PHONE]",
            }
        )
    """
    
    # Default placeholders for REPLACE strategy
    DEFAULT_PLACEHOLDERS: Dict[PIIType, str] = {
        PIIType.EMAIL: "[EMAIL]",
        PIIType.PHONE: "[PHONE]",
        PIIType.SSN: "[SSN]",
        PIIType.CREDIT_CARD: "[CREDIT_CARD]",
        PIIType.IP_ADDRESS: "[IP_ADDRESS]",
        PIIType.DATE_OF_BIRTH: "[DOB]",
        PIIType.ADDRESS: "[ADDRESS]",
        PIIType.NAME: "[NAME]",
        PIIType.PASSPORT: "[PASSPORT]",
        PIIType.DRIVERS_LICENSE: "[LICENSE]",
        PIIType.BANK_ACCOUNT: "[BANK_ACCOUNT]",
        PIIType.CUSTOM: "[REDACTED]",
    }
    
    def __init__(
        self,
        strategy: RedactionStrategy = RedactionStrategy.REPLACE,
        placeholders: Optional[Dict[PIIType, str]] = None,
        mask_char: str = "*",
        hash_func: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the redactor.
        
        Args:
            strategy: Redaction strategy to use
            placeholders: Custom placeholders for REPLACE strategy
            mask_char: Character to use for MASK strategy
            hash_func: Custom hash function for HASH strategy
        """
        self._strategy = strategy
        self._placeholders = {**self.DEFAULT_PLACEHOLDERS, **(placeholders or {})}
        self._mask_char = mask_char
        self._hash_func = hash_func or self._default_hash
    
    @property
    def strategy(self) -> RedactionStrategy:
        """Current redaction strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, value: RedactionStrategy) -> None:
        """Set redaction strategy."""
        self._strategy = value
    
    def redact(
        self,
        text: str,
        matches: List[PIIMatch],
        strategy: Optional[RedactionStrategy] = None,
    ) -> str:
        """
        Redact PII from text.
        
        Args:
            text: The original text
            matches: PII matches to redact
            strategy: Override strategy for this call
            
        Returns:
            Redacted text
        """
        if not matches:
            return text
        
        use_strategy = strategy or self._strategy
        
        # Sort matches by position (descending) to replace from end
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        
        result = text
        for match in sorted_matches:
            replacement = self._get_replacement(match, use_strategy)
            result = result[:match.start] + replacement + result[match.end:]
        
        return result
    
    def redact_match(
        self,
        match: PIIMatch,
        strategy: Optional[RedactionStrategy] = None,
    ) -> str:
        """
        Get the redacted version of a single match.
        
        Args:
            match: The PII match
            strategy: Override strategy
            
        Returns:
            The redacted replacement string
        """
        return self._get_replacement(match, strategy or self._strategy)
    
    def _get_replacement(
        self,
        match: PIIMatch,
        strategy: RedactionStrategy,
    ) -> str:
        """
        Get the replacement string for a match.
        
        Args:
            match: The PII match
            strategy: Redaction strategy
            
        Returns:
            Replacement string
        """
        if strategy == RedactionStrategy.MASK:
            return self._mask(match.value)
        elif strategy == RedactionStrategy.REPLACE:
            return self._placeholders.get(match.pii_type, "[REDACTED]")
        elif strategy == RedactionStrategy.HASH:
            return self._hash_func(match.value)
        elif strategy == RedactionStrategy.PARTIAL:
            return self._partial_mask(match.value, match.pii_type)
        elif strategy == RedactionStrategy.REMOVE:
            return ""
        else:
            return self._placeholders.get(match.pii_type, "[REDACTED]")
    
    def _mask(self, value: str) -> str:
        """
        Fully mask a value.
        
        Preserves structure characters like @ and .
        """
        result = []
        for char in value:
            if char.isalnum():
                result.append(self._mask_char)
            else:
                result.append(char)
        return "".join(result)
    
    def _partial_mask(self, value: str, pii_type: PIIType) -> str:
        """
        Partially mask a value, showing some characters.
        
        Strategy varies by PII type.
        """
        if pii_type == PIIType.EMAIL:
            return self._partial_mask_email(value)
        elif pii_type == PIIType.PHONE:
            return self._partial_mask_phone(value)
        elif pii_type == PIIType.CREDIT_CARD:
            return self._partial_mask_credit_card(value)
        else:
            # Default: show first and last char
            if len(value) <= 2:
                return self._mask_char * len(value)
            return value[0] + self._mask_char * (len(value) - 2) + value[-1]
    
    def _partial_mask_email(self, email: str) -> str:
        """Partially mask an email: j***@e******.com"""
        if "@" not in email:
            return self._mask(email)
        
        local, domain = email.rsplit("@", 1)
        
        # Mask local part except first char
        if len(local) > 1:
            local = local[0] + self._mask_char * (len(local) - 1)
        
        # Mask domain except first char and TLD
        if "." in domain:
            domain_name, tld = domain.rsplit(".", 1)
            if len(domain_name) > 1:
                domain_name = domain_name[0] + self._mask_char * (len(domain_name) - 1)
            domain = f"{domain_name}.{tld}"
        
        return f"{local}@{domain}"
    
    def _partial_mask_phone(self, phone: str) -> str:
        """Partially mask a phone: show last 4 digits."""
        digits = "".join(c for c in phone if c.isdigit())
        
        if len(digits) <= 4:
            return self._mask(phone)
        
        # Keep last 4 digits
        result = []
        digit_count = 0
        for char in reversed(phone):
            if char.isdigit():
                digit_count += 1
                if digit_count <= 4:
                    result.append(char)
                else:
                    result.append(self._mask_char)
            else:
                result.append(char)
        
        return "".join(reversed(result))
    
    def _partial_mask_credit_card(self, card: str) -> str:
        """Partially mask credit card: show last 4 digits."""
        return self._partial_mask_phone(card)  # Same logic
    
    def _default_hash(self, value: str) -> str:
        """Default hash function for HASH strategy."""
        import hashlib
        return hashlib.sha256(value.encode()).hexdigest()[:8]

