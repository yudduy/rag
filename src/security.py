"""
Security module for RAG system.

Provides input sanitization, prompt injection protection, and security validation.
"""

import os
import re
import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Security patterns and constants
PROMPT_INJECTION_PATTERNS = [
    # Direct injection attempts
    r'(?i)(ignore\s+(?:previous|above|earlier)\s+(?:instructions?|prompts?|commands?))',
    r'(?i)(forget\s+(?:everything|all|previous|above))',
    r'(?i)(new\s+(?:instructions?|task|role|persona))',
    r'(?i)(you\s+are\s+now\s+(?:a|an)\s+\w+)',
    r'(?i)(system\s*:\s*)',
    r'(?i)(prompt\s*:\s*)',
    r'(?i)(instruction\s*:\s*)',
    
    # Role/persona injection
    r'(?i)(act\s+as\s+(?:if\s+)?(?:you\s+are\s+)?(?:a|an)\s+\w+)',
    r'(?i)(pretend\s+(?:to\s+be\s+)?(?:that\s+)?you\s+are)',
    r'(?i)(rolep lay\s+as)',
    r'(?i)(simulate\s+(?:being\s+)?(?:a|an)\s+\w+)',
    
    # Command injection
    r'(?i)(execute|run)\s+(?:the\s+)?(?:following\s+)?(?:command|code|script)',
    r'(?i)(\|\s*(?:curl|wget|nc|netcat|bash|sh|python|perl|ruby))',
    r'(?i)(eval\s*\()',
    r'(?i)(__import__|exec|compile)\s*\(',
    
    # Template/format injection
    r'\{\{.*?\}\}',  # Template injection patterns
    r'\$\{.*?\}',    # Variable substitution
    r'<%.*?%>',      # Template tags
    
    # SQL injection attempts in prompts
    r'(?i)(union\s+select|drop\s+table|delete\s+from|insert\s+into)',
    r'(?i)(or\s+1\s*=\s*1|or\s+\'1\'\s*=\s*\'1\')',
    
    # Path traversal
    r'(\.{2,}[/\\])+',
    r'(?i)(file://|ftp://|data:)',
    
    # Encoding bypass attempts
    r'(%[0-9a-fA-F]{2}){3,}',  # URL encoding
    r'(&#x[0-9a-fA-F]+;){2,}',  # HTML entity encoding
    
    # Model-specific injection
    r'(?i)(claude|gpt|ai|assistant)[\s,]*:',
    r'(?i)(human|user)[\s,]*:',
]

# Compiled patterns for performance
COMPILED_INJECTION_PATTERNS = [re.compile(pattern) for pattern in PROMPT_INJECTION_PATTERNS]

# Suspicious keywords that may indicate injection attempts
SUSPICIOUS_KEYWORDS = [
    'ignore', 'forget', 'disregard', 'override', 'bypass', 'jailbreak',
    'system', 'admin', 'root', 'sudo', 'chmod', 'exec', 'eval',
    'script', 'payload', 'exploit', 'vulnerability', 'hack',
    'token', 'secret', 'password', 'api_key', 'private',
]

# File extension restrictions for document uploads
ALLOWED_FILE_EXTENSIONS = {
    '.txt', '.md', '.pdf', '.doc', '.docx', '.rtf',
    '.csv', '.json', '.xml', '.yaml', '.yml',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff',
}

DANGEROUS_FILE_EXTENSIONS = {
    '.exe', '.bat', '.cmd', '.com', '.scr', '.pif',
    '.js', '.vbs', '.ps1', '.sh', '.py', '.rb', '.pl',
    '.jar', '.class', '.dll', '.so', '.dylib',
}


class SecurityValidator:
    """Comprehensive security validation for RAG system inputs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize security validator with configuration."""
        self.config = config or {}
        self.max_query_length = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
        self.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        self.strict_mode = os.getenv("SECURITY_STRICT_MODE", "true").lower() == "true"
        self.log_violations = os.getenv("LOG_SECURITY_VIOLATIONS", "true").lower() == "true"
        
        # Initialize violation tracking
        self.violation_counts = {}
        self.blocked_patterns = set()
        
        logger.info(f"SecurityValidator initialized: strict_mode={self.strict_mode}, max_query_length={self.max_query_length}")
    
    def validate_query(self, query: str, user_id: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Comprehensive query validation with security checks.
        
        Args:
            query: User input query to validate
            user_id: Optional user identifier for tracking
            
        Returns:
            Tuple of (is_valid, sanitized_query, security_metadata)
        """
        if not query:
            return False, "", {"error": "Empty query"}
        
        security_metadata = {
            "original_length": len(query),
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "violations": [],
        }
        
        # Basic length validation
        if len(query) > self.max_query_length:
            violation = f"Query exceeds maximum length ({self.max_query_length} characters)"
            security_metadata["violations"].append(violation)
            self._log_security_violation(violation, query, user_id)
            return False, "", security_metadata
        
        # Check for prompt injection attempts
        injection_detected, injection_details = self._detect_prompt_injection(query)
        if injection_detected:
            security_metadata["violations"].extend(injection_details)
            for detail in injection_details:
                self._log_security_violation(f"Prompt injection detected: {detail}", query, user_id)
            
            if self.strict_mode:
                return False, "", security_metadata
        
        # Sanitize the query
        sanitized_query = self._sanitize_input(query)
        security_metadata["sanitized_length"] = len(sanitized_query)
        security_metadata["sanitization_applied"] = len(query) != len(sanitized_query)
        
        # Additional suspicious content detection
        suspicious_score = self._calculate_suspicion_score(sanitized_query)
        security_metadata["suspicion_score"] = suspicious_score
        
        if suspicious_score > 0.8 and self.strict_mode:
            violation = f"High suspicion score: {suspicious_score}"
            security_metadata["violations"].append(violation)
            self._log_security_violation(violation, query, user_id)
            return False, "", security_metadata
        
        # Rate limiting check (if user_id provided)
        if user_id:
            rate_limit_ok, rate_limit_msg = self._check_rate_limiting(user_id)
            if not rate_limit_ok:
                security_metadata["violations"].append(rate_limit_msg)
                self._log_security_violation(rate_limit_msg, query, user_id)
                return False, "", security_metadata
        
        # Final validation
        is_valid = len(security_metadata["violations"]) == 0 or not self.strict_mode
        
        return is_valid, sanitized_query, security_metadata
    
    def validate_file_upload(self, file_path: Union[str, Path], file_content: Optional[bytes] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate file upload security.
        
        Args:
            file_path: Path to the uploaded file
            file_content: Optional file content for validation
            
        Returns:
            Tuple of (is_valid, security_metadata)
        """
        file_path = Path(file_path)
        security_metadata = {
            "filename": file_path.name,
            "file_extension": file_path.suffix.lower(),
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "violations": [],
        }
        
        # File extension validation
        if file_path.suffix.lower() in DANGEROUS_FILE_EXTENSIONS:
            violation = f"Dangerous file extension: {file_path.suffix}"
            security_metadata["violations"].append(violation)
            self._log_security_violation(violation, str(file_path))
            return False, security_metadata
        
        if file_path.suffix.lower() not in ALLOWED_FILE_EXTENSIONS:
            violation = f"File extension not allowed: {file_path.suffix}"
            security_metadata["violations"].append(violation)
            self._log_security_violation(violation, str(file_path))
            return False, security_metadata
        
        # File size validation
        try:
            if file_path.exists():
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                security_metadata["file_size_mb"] = file_size_mb
                
                if file_size_mb > self.max_file_size_mb:
                    violation = f"File size exceeds limit: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB"
                    security_metadata["violations"].append(violation)
                    self._log_security_violation(violation, str(file_path))
                    return False, security_metadata
        except Exception as e:
            violation = f"Could not validate file size: {e}"
            security_metadata["violations"].append(violation)
            return False, security_metadata
        
        # Content-based validation if provided
        if file_content:
            content_valid, content_metadata = self._validate_file_content(file_content, file_path.suffix)
            security_metadata.update(content_metadata)
            if not content_valid:
                return False, security_metadata
        
        # Path traversal validation
        try:
            resolved_path = file_path.resolve()
            if '..' in str(resolved_path) or str(resolved_path) != str(file_path.resolve()):
                violation = "Path traversal attempt detected"
                security_metadata["violations"].append(violation)
                self._log_security_violation(violation, str(file_path))
                return False, security_metadata
        except Exception:
            violation = "Could not resolve file path"
            security_metadata["violations"].append(violation)
            return False, security_metadata
        
        return len(security_metadata["violations"]) == 0, security_metadata
    
    def _detect_prompt_injection(self, text: str) -> Tuple[bool, List[str]]:
        """Detect potential prompt injection attempts."""
        violations = []
        
        # Check against compiled patterns
        for pattern in COMPILED_INJECTION_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                violations.extend([f"Pattern match: {match}" for match in matches])
        
        # Check for suspicious keyword density
        text_lower = text.lower()
        suspicious_count = sum(1 for keyword in SUSPICIOUS_KEYWORDS if keyword in text_lower)
        if suspicious_count >= 3:  # Multiple suspicious keywords
            violations.append(f"High suspicious keyword density: {suspicious_count} keywords")
        
        # Check for role/instruction markers
        role_markers = ['system:', 'user:', 'assistant:', 'human:', 'ai:', 'claude:']
        for marker in role_markers:
            if marker.lower() in text_lower:
                violations.append(f"Role marker detected: {marker}")
        
        # Check for encoding attempts
        if self._detect_encoding_attempts(text):
            violations.append("Encoding bypass attempt detected")
        
        return len(violations) > 0, violations
    
    def _sanitize_input(self, text: str) -> str:
        """Sanitize user input to remove potentially dangerous content."""
        # Remove null bytes and control characters
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s{10,}', ' ' * 5, text)  # Limit consecutive spaces
        text = re.sub(r'\n{5,}', '\n' * 3, text)  # Limit consecutive newlines
        
        # Remove potentially dangerous HTML/XML tags
        text = re.sub(r'<[^>]*>', '', text)
        
        # Remove script-like patterns
        text = re.sub(r'(?i)(javascript|vbscript|onload|onerror|onclick):.*?(?=[\s\n]|$)', '', text)
        
        # Normalize Unicode to prevent bypass attempts
        try:
            import unicodedata
            text = unicodedata.normalize('NFKC', text)
        except ImportError:
            pass
        
        return text.strip()
    
    def _calculate_suspicion_score(self, text: str) -> float:
        """Calculate a suspicion score for the text (0-1)."""
        score = 0.0
        text_lower = text.lower()
        
        # Keyword-based scoring
        keyword_score = sum(0.1 for keyword in SUSPICIOUS_KEYWORDS if keyword in text_lower)
        score += min(keyword_score, 0.5)  # Cap at 0.5
        
        # Pattern-based scoring
        for pattern in COMPILED_INJECTION_PATTERNS:
            if pattern.search(text):
                score += 0.1
        
        # Length-based penalty for very long inputs
        if len(text) > 5000:
            score += 0.1
        
        # Special character density
        special_chars = len(re.findall(r'[^\w\s]', text))
        if special_chars > len(text) * 0.3:  # More than 30% special chars
            score += 0.2
        
        # URL/path-like patterns
        if re.search(r'https?://|file://|ftp://|\.{2,}/', text_lower):
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _detect_encoding_attempts(self, text: str) -> bool:
        """Detect encoding bypass attempts."""
        # Check for URL encoding
        url_encoded = len(re.findall(r'%[0-9a-fA-F]{2}', text))
        if url_encoded > 5:  # Multiple encoded characters
            return True
        
        # Check for HTML entity encoding
        html_entities = len(re.findall(r'&#x?[0-9a-fA-F]+;', text))
        if html_entities > 3:
            return True
        
        # Check for Unicode escape sequences
        unicode_escapes = len(re.findall(r'\\u[0-9a-fA-F]{4}', text))
        if unicode_escapes > 2:
            return True
        
        return False
    
    def _validate_file_content(self, content: bytes, file_extension: str) -> Tuple[bool, Dict[str, Any]]:
        """Validate file content for security issues."""
        metadata = {"content_validation": {}}
        
        try:
            # Check for executable signatures
            if content.startswith((b'MZ', b'\x7fELF', b'\xfe\xed\xfa')):  # PE, ELF, Mach-O
                metadata["content_validation"]["executable_detected"] = True
                return False, metadata
            
            # Check for script content in non-script files
            if file_extension in {'.txt', '.md', '.csv'}:
                try:
                    text_content = content.decode('utf-8', errors='ignore')
                    if any(pattern.search(text_content) for pattern in COMPILED_INJECTION_PATTERNS):
                        metadata["content_validation"]["script_content_detected"] = True
                        return False, metadata
                except UnicodeDecodeError:
                    pass
            
            # Check file size consistency
            content_size = len(content)
            metadata["content_validation"]["content_size_bytes"] = content_size
            
            if content_size == 0:
                metadata["content_validation"]["empty_file"] = True
                return False, metadata
            
            return True, metadata
            
        except Exception as e:
            metadata["content_validation"]["error"] = str(e)
            return False, metadata
    
    def _check_rate_limiting(self, user_id: str) -> Tuple[bool, str]:
        """Check rate limiting for user."""
        # Simple in-memory rate limiting (in production, use Redis or database)
        current_time = datetime.now(timezone.utc)
        
        if user_id not in self.violation_counts:
            self.violation_counts[user_id] = []
        
        # Clean old violations (older than 1 hour)
        cutoff_time = current_time.timestamp() - 3600
        self.violation_counts[user_id] = [
            timestamp for timestamp in self.violation_counts[user_id] 
            if timestamp > cutoff_time
        ]
        
        # Check if user has exceeded rate limit
        max_violations_per_hour = int(os.getenv("MAX_SECURITY_VIOLATIONS_PER_HOUR", "10"))
        if len(self.violation_counts[user_id]) >= max_violations_per_hour:
            return False, f"Rate limit exceeded: {len(self.violation_counts[user_id])} violations in the last hour"
        
        return True, ""
    
    def _log_security_violation(self, violation: str, content: str, user_id: Optional[str] = None):
        """Log security violations for monitoring."""
        if not self.log_violations:
            return
        
        # Hash content for privacy
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "violation": violation,
            "content_hash": content_hash,
            "user_id": user_id,
            "content_length": len(content),
        }
        
        logger.warning(f"Security violation detected: {log_entry}")
        
        # Track for rate limiting
        if user_id:
            if user_id not in self.violation_counts:
                self.violation_counts[user_id] = []
            self.violation_counts[user_id].append(datetime.now(timezone.utc).timestamp())
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security validation statistics."""
        return {
            "total_users_tracked": len(self.violation_counts),
            "total_violations": sum(len(violations) for violations in self.violation_counts.values()),
            "blocked_patterns": list(self.blocked_patterns),
            "config": {
                "max_query_length": self.max_query_length,
                "max_file_size_mb": self.max_file_size_mb,
                "strict_mode": self.strict_mode,
                "log_violations": self.log_violations,
            }
        }


# Global security validator instance
_security_validator: Optional[SecurityValidator] = None


def get_security_validator() -> SecurityValidator:
    """Get the global security validator instance."""
    global _security_validator
    if _security_validator is None:
        _security_validator = SecurityValidator()
    return _security_validator


def validate_user_query(query: str, user_id: Optional[str] = None) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Convenience function to validate user queries.
    
    Args:
        query: User input query
        user_id: Optional user identifier
        
    Returns:
        Tuple of (is_valid, sanitized_query, security_metadata)
    """
    validator = get_security_validator()
    return validator.validate_query(query, user_id)


def validate_file_upload(file_path: Union[str, Path], file_content: Optional[bytes] = None) -> Tuple[bool, Dict[str, Any]]:
    """
    Convenience function to validate file uploads.
    
    Args:
        file_path: Path to the uploaded file
        file_content: Optional file content
        
    Returns:
        Tuple of (is_valid, security_metadata)
    """
    validator = get_security_validator()
    return validator.validate_file_upload(file_path, file_content)


def sanitize_input(text: str) -> str:
    """
    Convenience function to sanitize user input.
    
    Args:
        text: Input text to sanitize
        
    Returns:
        Sanitized text
    """
    validator = get_security_validator()
    return validator._sanitize_input(text)