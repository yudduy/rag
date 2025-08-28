"""
Security Configuration Validation Module.

Provides comprehensive security configuration validation, secure defaults,
and compliance checks for the RAG system.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import json

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration with validation."""
    # API Security
    api_key_validation_enabled: bool = True
    api_rate_limiting_enabled: bool = True
    api_max_requests_per_minute: int = 60
    
    # Input Security
    input_sanitization_enabled: bool = True
    prompt_injection_protection: bool = True
    max_query_length: int = 10000
    strict_security_mode: bool = True
    
    # Connection Security
    redis_ssl_required: bool = False  # Set to True for production
    redis_auth_required: bool = True
    connection_timeout_seconds: int = 30
    max_connections: int = 100
    
    # File Security
    file_upload_validation: bool = True
    max_file_size_mb: int = 50
    allowed_file_extensions: List[str] = field(default_factory=lambda: [
        '.txt', '.md', '.pdf', '.doc', '.docx', '.rtf',
        '.csv', '.json', '.xml', '.yaml', '.yml'
    ])
    
    # Resource Security
    memory_limit_mb: int = 2048
    cpu_limit_percent: float = 80.0
    thread_limit: int = 50
    query_timeout_seconds: int = 30
    
    # Logging and Monitoring
    security_logging_enabled: bool = True
    log_security_violations: bool = True
    alert_on_violations: bool = True
    violation_rate_limit: int = 10  # per hour
    
    # Compliance
    gdpr_compliance_mode: bool = False
    ccpa_compliance_mode: bool = False
    data_retention_days: int = 90
    
    # Environment
    environment: str = "development"
    debug_mode: bool = False


class SecurityConfigValidator:
    """Validates and manages security configuration."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize security config validator."""
        self.config_path = config_path or os.getenv("SECURITY_CONFIG_PATH", ".security_config.json")
        self.config = SecurityConfig()
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
        self._load_config()
        self._validate_environment_config()
    
    def _load_config(self):
        """Load security configuration from environment and files."""
        # Load from environment variables
        self.config.api_key_validation_enabled = os.getenv("API_KEY_VALIDATION_ENABLED", "true").lower() == "true"
        self.config.input_sanitization_enabled = os.getenv("INPUT_SANITIZATION_ENABLED", "true").lower() == "true"
        self.config.prompt_injection_protection = os.getenv("PROMPT_INJECTION_PROTECTION", "true").lower() == "true"
        self.config.strict_security_mode = os.getenv("SECURITY_STRICT_MODE", "true").lower() == "true"
        
        self.config.max_query_length = int(os.getenv("MAX_QUERY_LENGTH", "10000"))
        self.config.max_file_size_mb = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
        self.config.memory_limit_mb = int(os.getenv("MAX_MEMORY_MB", "2048"))
        self.config.cpu_limit_percent = float(os.getenv("MAX_CPU_PERCENT", "80.0"))
        self.config.thread_limit = int(os.getenv("MAX_THREADS", "50"))
        self.config.query_timeout_seconds = int(os.getenv("MAX_QUERY_TIME_SECONDS", "30"))
        
        self.config.redis_ssl_required = os.getenv("REDIS_SSL_REQUIRED", "false").lower() == "true"
        self.config.redis_auth_required = os.getenv("REDIS_AUTH_REQUIRED", "true").lower() == "true"
        self.config.connection_timeout_seconds = int(os.getenv("CONNECTION_TIMEOUT_SECONDS", "30"))
        self.config.max_connections = int(os.getenv("MAX_CONNECTIONS", "100"))
        
        self.config.security_logging_enabled = os.getenv("SECURITY_LOGGING_ENABLED", "true").lower() == "true"
        self.config.log_security_violations = os.getenv("LOG_SECURITY_VIOLATIONS", "true").lower() == "true"
        self.config.alert_on_violations = os.getenv("ALERT_ON_VIOLATIONS", "true").lower() == "true"
        self.config.violation_rate_limit = int(os.getenv("MAX_SECURITY_VIOLATIONS_PER_HOUR", "10"))
        
        self.config.gdpr_compliance_mode = os.getenv("GDPR_COMPLIANCE_MODE", "false").lower() == "true"
        self.config.ccpa_compliance_mode = os.getenv("CCPA_COMPLIANCE_MODE", "false").lower() == "true"
        self.config.data_retention_days = int(os.getenv("DATA_RETENTION_DAYS", "90"))
        
        self.config.environment = os.getenv("ENVIRONMENT", "development").lower()
        self.config.debug_mode = os.getenv("DEBUG", "false").lower() == "true"
        
        # Load from config file if it exists
        if Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    file_config = json.load(f)
                    self._merge_file_config(file_config)
                logger.info(f"Loaded security config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load security config file: {e}")
    
    def _merge_file_config(self, file_config: Dict[str, Any]):
        """Merge file configuration with current config."""
        for key, value in file_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _validate_environment_config(self):
        """Validate environment-specific configuration."""
        self.validation_errors.clear()
        self.validation_warnings.clear()
        
        # Production environment validations
        if self.config.environment in ("production", "prod", "staging"):
            if self.config.debug_mode:
                self.validation_errors.append("DEBUG mode must be disabled in production")
            
            if not self.config.api_key_validation_enabled:
                self.validation_errors.append("API key validation must be enabled in production")
            
            if not self.config.input_sanitization_enabled:
                self.validation_errors.append("Input sanitization must be enabled in production")
            
            if not self.config.prompt_injection_protection:
                self.validation_errors.append("Prompt injection protection must be enabled in production")
            
            if not self.config.strict_security_mode:
                self.validation_warnings.append("Strict security mode should be enabled in production")
            
            # Redis security for production
            redis_url = os.getenv("REDIS_CACHE_URL", "")
            if redis_url:
                parsed = urlparse(redis_url)
                if parsed.hostname not in ("localhost", "127.0.0.1", None):
                    if not self.config.redis_ssl_required and parsed.scheme != "rediss":
                        self.validation_warnings.append("Redis SSL should be required for production")
                    
                    if not parsed.password and self.config.redis_auth_required:
                        self.validation_warnings.append("Redis authentication should be configured for production")
        
        # Resource limit validations
        if self.config.memory_limit_mb < 512:
            self.validation_warnings.append("Memory limit is very low, may impact performance")
        
        if self.config.memory_limit_mb > 8192:
            self.validation_warnings.append("Memory limit is very high, may impact system stability")
        
        if self.config.max_query_length > 20000:
            self.validation_warnings.append("Max query length is very high, may impact performance")
        
        if self.config.max_file_size_mb > 200:
            self.validation_warnings.append("Max file size is very high, may impact system resources")
        
        # Compliance validations
        if self.config.gdpr_compliance_mode or self.config.ccpa_compliance_mode:
            if self.config.data_retention_days > 365:
                self.validation_warnings.append("Data retention period may conflict with privacy regulations")
            
            if not self.config.security_logging_enabled:
                self.validation_warnings.append("Security logging should be enabled for compliance")
        
        # API rate limiting
        if self.config.api_rate_limiting_enabled and self.config.api_max_requests_per_minute > 1000:
            self.validation_warnings.append("API rate limit is very high")
    
    def validate_openai_api_key(self) -> Tuple[bool, str]:
        """Validate OpenAI API key configuration."""
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return False, "OPENAI_API_KEY is not set"
        
        # Check for placeholder values
        placeholder_keys = {
            'your_api_key_here', 'your_openai_api_key_here', 
            'REDACTED_SK_KEY', 'sk-example', 'replace_with_your_key',
            'INSERT_API_KEY_HERE'
        }
        
        if api_key.lower() in {key.lower() for key in placeholder_keys}:
            return False, "API key appears to be a placeholder value"
        
        # Basic format validation
        if not api_key.startswith('sk-') or len(api_key) < 20:
            return False, "API key format appears invalid"
        
        # Check for suspicious patterns
        suspicious_patterns = ['localhost', '127.0.0.1', 'example.com', 'test.com']
        if any(pattern in api_key.lower() for pattern in suspicious_patterns):
            return False, "API key contains suspicious patterns"
        
        return True, "API key validation passed"
    
    def validate_redis_security(self) -> Tuple[bool, List[str]]:
        """Validate Redis security configuration."""
        issues = []
        redis_url = os.getenv("REDIS_CACHE_URL", "")
        
        if not redis_url:
            return True, []  # No Redis configured
        
        try:
            parsed = urlparse(redis_url)
            
            # Check SSL/TLS
            if self.config.redis_ssl_required and parsed.scheme != "rediss":
                issues.append("Redis SSL is required but not configured")
            
            # Check authentication for remote connections
            if parsed.hostname not in ("localhost", "127.0.0.1", None):
                if self.config.redis_auth_required and not parsed.password:
                    issues.append("Redis authentication is required but not configured")
            
            # Check for default ports
            if parsed.port == 6379 and parsed.hostname not in ("localhost", "127.0.0.1", None):
                issues.append("Using default Redis port for remote connection may be insecure")
            
        except Exception as e:
            issues.append(f"Failed to parse Redis URL: {e}")
        
        return len(issues) == 0, issues
    
    def validate_file_security(self) -> Tuple[bool, List[str]]:
        """Validate file upload security configuration."""
        issues = []
        
        if not self.config.file_upload_validation:
            issues.append("File upload validation is disabled")
        
        # Check allowed extensions
        dangerous_extensions = {'.exe', '.bat', '.cmd', '.com', '.scr', '.js', '.vbs', '.ps1', '.sh'}
        allowed_dangerous = set(self.config.allowed_file_extensions) & dangerous_extensions
        
        if allowed_dangerous:
            issues.append(f"Dangerous file extensions are allowed: {', '.join(allowed_dangerous)}")
        
        if self.config.max_file_size_mb > 500:
            issues.append("Maximum file size is very large, may cause DoS")
        
        return len(issues) == 0, issues
    
    def get_security_recommendations(self) -> Dict[str, List[str]]:
        """Get security recommendations based on current configuration."""
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
        }
        
        # High priority recommendations
        if self.config.environment in ("production", "prod") and self.config.debug_mode:
            recommendations["high_priority"].append("Disable debug mode in production")
        
        if not self.config.api_key_validation_enabled:
            recommendations["high_priority"].append("Enable API key validation")
        
        if not self.config.input_sanitization_enabled:
            recommendations["high_priority"].append("Enable input sanitization")
        
        if not self.config.prompt_injection_protection:
            recommendations["high_priority"].append("Enable prompt injection protection")
        
        # Medium priority recommendations
        if not self.config.strict_security_mode:
            recommendations["medium_priority"].append("Consider enabling strict security mode")
        
        if self.config.max_query_length > 15000:
            recommendations["medium_priority"].append("Consider reducing maximum query length")
        
        if not self.config.security_logging_enabled:
            recommendations["medium_priority"].append("Enable security logging for monitoring")
        
        # Low priority recommendations
        if self.config.api_max_requests_per_minute > 200:
            recommendations["low_priority"].append("Consider stricter API rate limiting")
        
        if self.config.data_retention_days > 180:
            recommendations["low_priority"].append("Consider shorter data retention period")
        
        return recommendations
    
    def generate_secure_config(self) -> Dict[str, Any]:
        """Generate a secure configuration template."""
        return {
            "api_security": {
                "api_key_validation_enabled": True,
                "api_rate_limiting_enabled": True,
                "api_max_requests_per_minute": 60,
            },
            "input_security": {
                "input_sanitization_enabled": True,
                "prompt_injection_protection": True,
                "max_query_length": 8000,
                "strict_security_mode": True,
            },
            "connection_security": {
                "redis_ssl_required": True,  # For production
                "redis_auth_required": True,
                "connection_timeout_seconds": 30,
                "max_connections": 50,
            },
            "file_security": {
                "file_upload_validation": True,
                "max_file_size_mb": 25,
                "allowed_file_extensions": [
                    ".txt", ".md", ".pdf", ".doc", ".docx",
                    ".csv", ".json", ".xml", ".yaml"
                ]
            },
            "resource_security": {
                "memory_limit_mb": 1024,
                "cpu_limit_percent": 70.0,
                "thread_limit": 30,
                "query_timeout_seconds": 20,
            },
            "monitoring": {
                "security_logging_enabled": True,
                "log_security_violations": True,
                "alert_on_violations": True,
                "violation_rate_limit": 5,
            },
            "compliance": {
                "gdpr_compliance_mode": False,
                "ccpa_compliance_mode": False,
                "data_retention_days": 30,
            },
            "environment": {
                "environment": "production",
                "debug_mode": False,
            }
        }
    
    def export_config(self, file_path: Optional[str] = None) -> bool:
        """Export current configuration to file."""
        file_path = file_path or self.config_path
        
        try:
            config_dict = {
                "api_key_validation_enabled": self.config.api_key_validation_enabled,
                "input_sanitization_enabled": self.config.input_sanitization_enabled,
                "prompt_injection_protection": self.config.prompt_injection_protection,
                "strict_security_mode": self.config.strict_security_mode,
                "max_query_length": self.config.max_query_length,
                "max_file_size_mb": self.config.max_file_size_mb,
                "memory_limit_mb": self.config.memory_limit_mb,
                "cpu_limit_percent": self.config.cpu_limit_percent,
                "thread_limit": self.config.thread_limit,
                "query_timeout_seconds": self.config.query_timeout_seconds,
                "redis_ssl_required": self.config.redis_ssl_required,
                "redis_auth_required": self.config.redis_auth_required,
                "connection_timeout_seconds": self.config.connection_timeout_seconds,
                "max_connections": self.config.max_connections,
                "security_logging_enabled": self.config.security_logging_enabled,
                "log_security_violations": self.config.log_security_violations,
                "alert_on_violations": self.config.alert_on_violations,
                "violation_rate_limit": self.config.violation_rate_limit,
                "gdpr_compliance_mode": self.config.gdpr_compliance_mode,
                "ccpa_compliance_mode": self.config.ccpa_compliance_mode,
                "data_retention_days": self.config.data_retention_days,
                "environment": self.config.environment,
                "debug_mode": self.config.debug_mode,
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }
            
            with open(file_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            logger.info(f"Security configuration exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export security configuration: {e}")
            return False
    
    def validate_all(self) -> Dict[str, Any]:
        """Perform comprehensive security validation."""
        validation_result = {
            "overall_status": "unknown",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "environment": self.config.environment,
            "validations": {},
            "errors": self.validation_errors.copy(),
            "warnings": self.validation_warnings.copy(),
            "recommendations": self.get_security_recommendations(),
        }
        
        # API key validation
        api_key_valid, api_key_msg = self.validate_openai_api_key()
        validation_result["validations"]["api_key"] = {
            "status": "pass" if api_key_valid else "fail",
            "message": api_key_msg,
        }
        
        # Redis security validation
        redis_valid, redis_issues = self.validate_redis_security()
        validation_result["validations"]["redis_security"] = {
            "status": "pass" if redis_valid else "fail",
            "issues": redis_issues,
        }
        
        # File security validation
        file_valid, file_issues = self.validate_file_security()
        validation_result["validations"]["file_security"] = {
            "status": "pass" if file_valid else "fail",
            "issues": file_issues,
        }
        
        # Overall status
        has_errors = len(self.validation_errors) > 0 or not api_key_valid
        has_failures = not redis_valid or not file_valid
        
        if has_errors:
            validation_result["overall_status"] = "critical"
        elif has_failures or len(self.validation_warnings) > 0:
            validation_result["overall_status"] = "warning"
        else:
            validation_result["overall_status"] = "pass"
        
        return validation_result
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current security configuration."""
        return {
            "security_features": {
                "api_key_validation": self.config.api_key_validation_enabled,
                "input_sanitization": self.config.input_sanitization_enabled,
                "prompt_injection_protection": self.config.prompt_injection_protection,
                "strict_mode": self.config.strict_security_mode,
                "file_validation": self.config.file_upload_validation,
                "security_logging": self.config.security_logging_enabled,
            },
            "limits": {
                "max_query_length": self.config.max_query_length,
                "max_file_size_mb": self.config.max_file_size_mb,
                "memory_limit_mb": self.config.memory_limit_mb,
                "cpu_limit_percent": self.config.cpu_limit_percent,
                "thread_limit": self.config.thread_limit,
                "query_timeout_seconds": self.config.query_timeout_seconds,
            },
            "compliance": {
                "gdpr_mode": self.config.gdpr_compliance_mode,
                "ccpa_mode": self.config.ccpa_compliance_mode,
                "data_retention_days": self.config.data_retention_days,
            },
            "environment": {
                "environment": self.config.environment,
                "debug_mode": self.config.debug_mode,
            }
        }


# Global security config validator
_security_config_validator: Optional[SecurityConfigValidator] = None


def get_security_config_validator() -> SecurityConfigValidator:
    """Get the global security config validator instance."""
    global _security_config_validator
    if _security_config_validator is None:
        _security_config_validator = SecurityConfigValidator()
    return _security_config_validator


def validate_security_config() -> Dict[str, Any]:
    """Convenience function to validate security configuration."""
    validator = get_security_config_validator()
    return validator.validate_all()


def get_secure_defaults() -> Dict[str, Any]:
    """Get secure default configuration."""
    validator = SecurityConfigValidator()
    return validator.generate_secure_config()