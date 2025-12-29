# config/logger_config.py
"""
Logging Configuration for Trading System

STRUCTURE:
==========
1. ORIGINAL SETUP (Lines 1-60):
   - Basic logging configuration
   - Console/File handlers
   - Emoji filtering
   - Third-party library suppression
   
2. SILENT FAILURE DETECTION (Lines 61+):
   - MetricsCollector for tracking issues
   - SafeDict for safe data access
   - Decorators for method wrapping
   - Helper functions for validation

Both sections are INDEPENDENT and can be used separately.
"""

import logging
from logging.handlers import RotatingFileHandler
import os
import re

# ============================================================================
# SECTION 1: ORIGINAL LOGGER SETUP (Your existing code - UNCHANGED)
# ============================================================================

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Regex to strip emojis for console
EMOJI_PATTERN = re.compile(
    "["  
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA70-\U0001FAFF"  # extended symbols
    "]+", flags=re.UNICODE
)

class RemoveEmojiFilter(logging.Filter):
    """Filter to remove emojis from console output."""
    def filter(self, record):
        record.msg = EMOJI_PATTERN.sub("", str(record.msg))
        return True


def setup_logger():
    """
    Initialize the logging system with console and file handlers.
    
    Returns:
        Configured root logger instance
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # ---- Console Handler (emoji removed) ----
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.addFilter(RemoveEmojiFilter())   # <-- IMPORTANT
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    # ---- File Handler (keeps emojis) ----
    fh = RotatingFileHandler(
        LOG_FILE, 
        maxBytes=2_000_000, 
        backupCount=4, 
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    # Suppress noisy third-party libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    return logger


# ============================================================================
# SECTION 2: SILENT FAILURE DETECTION SYSTEM (NEW - OPTIONAL)
# ============================================================================
# These utilities USE the logger configured above via logging.getLogger(__name__)
# They don't create their own logger - they use your existing setup!

import functools
import traceback
from typing import Any, Dict, Optional, Callable
from collections import defaultdict

# Get logger for this module (will use root logger from setup_logger())
logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects metrics about missing data, failed validations, and fallbacks.
    
    This class USES the logger configured by setup_logger() above.
    It doesn't create its own logger.
    """
    def __init__(self):
        self.missing_keys = defaultdict(int)
        self.failed_methods = defaultdict(int)
        self.fallback_usage = defaultdict(int)
        self.none_returns = defaultdict(int)
        self.empty_returns = defaultdict(int)
        self.validation_failures = defaultdict(int)
        self._current_symbol = None
        
    def set_current_symbol(self, symbol: str):
        """Set the current symbol being processed (for context)."""
        self._current_symbol = symbol
    
    def _format_symbol_prefix(self) -> str:
        """Get symbol prefix for logs."""
        return f"[{self._current_symbol}] " if self._current_symbol else ""
    
    def log_missing_key(self, context: str, key: str, source: str = "unknown"):
        """Log when a dictionary key is missing."""
        metric_key = f"{context}.{key}"
        self.missing_keys[metric_key] += 1
        
        # Use WARNING for first occurrence, DEBUG for repeats
        level = logging.WARNING if self.missing_keys[metric_key] == 1 else logging.DEBUG
        
        # This uses YOUR logger (with emoji filter on console, emojis in file)
        logger.log(
            level,
            f"{self._format_symbol_prefix()}🔍 MISSING KEY: '{key}' not found in {context} "
            f"(source: {source}) [occurrence: {self.missing_keys[metric_key]}]"
        )
    
    def log_failed_method(self, method_name: str, error: Exception, symbol: str = ""):
        """Log when a method fails."""
        self.failed_methods[method_name] += 1
        
        error_msg = f"{type(error).__name__}: {str(error)}"
        logger.error(
            f"{self._format_symbol_prefix()}❌ METHOD FAILED: {method_name} - {error_msg} "
            f"[occurrence: {self.failed_methods[method_name]}]"
        )
        
        # Full traceback at DEBUG level
        logger.debug(f"Traceback for {method_name}:\n{traceback.format_exc()}")
    
    def log_fallback(self, context: str, reason: str):
        """Log when falling back to defaults."""
        self.fallback_usage[context] += 1
        logger.info(
            f"{self._format_symbol_prefix()}⚠️  FALLBACK: {context} - {reason} "
            f"[occurrence: {self.fallback_usage[context]}]"
        )
    
    def log_none_return(self, method_name: str, context: str = ""):
        """Log when a method returns None unexpectedly."""
        metric_key = f"{method_name}.{context}" if context else method_name
        self.none_returns[metric_key] += 1
        
        logger.warning(
            f"{self._format_symbol_prefix()}🚨 NONE RETURN: {method_name} "
            f"{f'({context})' if context else ''} returned None "
            f"[occurrence: {self.none_returns[metric_key]}]"
        )
    
    def log_empty_return(self, method_name: str, value_type: str = ""):
        """Log when a method returns empty collection."""
        metric_key = f"{method_name}.{value_type}" if value_type else method_name
        self.empty_returns[metric_key] += 1
        
        level = logging.WARNING if self.empty_returns[metric_key] == 1 else logging.DEBUG
        
        logger.log(
            level,
            f"{self._format_symbol_prefix()}📭 EMPTY RETURN: {method_name} returned empty {value_type} "
            f"[occurrence: {self.empty_returns[metric_key]}]"
        )
    
    def log_validation_failure(self, validator: str, field: str, actual: Any, expected: str):
        """Log when validation fails."""
        key = f"{validator}.{field}"
        self.validation_failures[key] += 1
        
        logger.warning(
            f"{self._format_symbol_prefix()}⛔ VALIDATION FAILED: {validator} - "
            f"Field '{field}' has value {actual} (expected: {expected}) "
            f"[occurrence: {self.validation_failures[key]}]"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        return {
            "symbol": self._current_symbol,
            "missing_keys": dict(self.missing_keys),
            "failed_methods": dict(self.failed_methods),
            "fallback_usage": dict(self.fallback_usage),
            "none_returns": dict(self.none_returns),
            "empty_returns": dict(self.empty_returns),
            "validation_failures": dict(self.validation_failures),
            "total_issues": (
                sum(self.missing_keys.values()) +
                sum(self.failed_methods.values()) +
                sum(self.none_returns.values()) +
                sum(self.empty_returns.values()) +
                sum(self.validation_failures.values())
            )
        }
    
    def reset(self):
        """Reset all metrics (call at start of each stock evaluation)."""
        self.missing_keys.clear()
        self.failed_methods.clear()
        self.fallback_usage.clear()
        self.none_returns.clear()
        self.empty_returns.clear()
        self.validation_failures.clear()
        self._current_symbol = None


# Global metrics collector instance
METRICS = MetricsCollector()


class SafeDict:
    """
    Wrapper around dict that logs missing keys.
    Uses the logger configured by setup_logger().
    """
    
    def __init__(self, data: Dict, context: str = "unknown", source: str = ""):
        self._data = data if data is not None else {}
        self._context = context
        self._source = source or context
        
        if data is None:
            logger.warning(f"🚨 SafeDict received None for context '{context}'")
        elif len(data) == 0:
            logger.debug(f"📭 SafeDict received empty dict for context '{context}'")
    
    def get(self, key: str, default: Any = None, warn_if_missing: bool = True) -> Any:
        """Safe get with logging."""
        if key not in self._data:
            if warn_if_missing:
                METRICS.log_missing_key(self._context, key, self._source)
            return default
        
        value = self._data[key]
        
        if value is None:
            logger.debug(f"⚠️  Key '{key}' in {self._context} exists but is None")
        elif isinstance(value, (list, dict)) and len(value) == 0:
            logger.debug(f"📭 Key '{key}' in {self._context} is empty {type(value).__name__}")
        
        return value
    
    def get_required(self, key: str, field_type: str = "field") -> Any:
        """Get a required field - logs ERROR if missing."""
        if key not in self._data:
            logger.error(
                f"❌ REQUIRED {field_type.upper()} MISSING: '{key}' not found in {self._context}"
            )
            METRICS.log_missing_key(self._context, key, f"REQUIRED_{field_type}")
            return None
        
        return self._data[key]
    
    def __getitem__(self, key: str) -> Any:
        return self.get(key, warn_if_missing=True)
    
    def __contains__(self, key: str) -> bool:
        return key in self._data
    
    def keys(self):
        return self._data.keys()
    
    def items(self):
        return self._data.items()
    
    def values(self):
        return self._data.values()
    
    @property
    def raw(self) -> Dict:
        """Get underlying dict."""
        return self._data


def log_failures(return_on_error: Any = None, critical: bool = False):
    """
    Decorator to catch and log method failures.
    Uses the logger configured by setup_logger().
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            
            symbol = "unknown"
            if len(args) > 0 and hasattr(args[0], '_current_symbol'):
                symbol = args[0]._current_symbol
            
            try:
                result = func(*args, **kwargs)
                
                if result is None:
                    logger.debug(f"⚠️  {method_name} returned None")
                elif isinstance(result, dict) and len(result) == 0:
                    logger.debug(f"📭 {method_name} returned empty dict")
                elif isinstance(result, list) and len(result) == 0:
                    logger.debug(f"📭 {method_name} returned empty list")
                
                return result
                
            except Exception as e:
                METRICS.log_failed_method(method_name, e, symbol)
                
                if critical:
                    logger.critical(f"🔥 CRITICAL FAILURE in {method_name} - re-raising exception")
                    raise
                else:
                    METRICS.log_fallback(method_name, f"Exception: {type(e).__name__}")
                    return return_on_error
        
        return wrapper
    return decorator


def validate_output(expected_type: type = None, not_none: bool = True, not_empty: bool = False):
    """
    Decorator to validate method output.
    Uses the logger configured by setup_logger().
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            result = func(*args, **kwargs)
            
            if expected_type and result is not None and not isinstance(result, expected_type):
                logger.warning(
                    f"⚠️  TYPE MISMATCH: {method_name} returned {type(result).__name__}, "
                    f"expected {expected_type.__name__}"
                )
                METRICS.validation_failures[f"{method_name}.type"] += 1
            
            if not_none and result is None:
                METRICS.log_none_return(method_name)
            
            if not_empty and result is not None:
                if isinstance(result, (dict, list, str)) and len(result) == 0:
                    METRICS.log_empty_return(method_name, type(result).__name__)
            
            return result
        
        return wrapper
    return decorator


def safe_get_nested(data: Dict, keys: list, default: Any = None, context: str = "nested") -> Any:
    """
    Safely navigate nested dictionaries with logging.
    Uses the logger configured by setup_logger().
    """
    current = data
    path = []
    
    for key in keys:
        path.append(key)
        
        if not isinstance(current, dict):
            logger.warning(
                f"⚠️  NESTED ACCESS FAILED: {context} - "
                f"Expected dict at '{'.'.join(path[:-1])}', got {type(current).__name__}"
            )
            METRICS.log_missing_key(context, ".".join(path), "type_error")
            return default
        
        if key not in current:
            logger.debug(f"🔍 NESTED KEY MISSING: {context} - Path '{'.'.join(path)}' not found")
            METRICS.log_missing_key(context, ".".join(path), "nested_missing")
            return default
        
        current = current[key]
    
    return current


def validate_required_keys(data: Dict, required_keys: list, context: str) -> bool:
    """
    Check if all required keys exist in dict.
    Uses the logger configured by setup_logger().
    """
    if not isinstance(data, dict):
        logger.error(f"❌ VALIDATION ERROR: {context} - Expected dict, got {type(data).__name__}")
        return False
    
    missing = [key for key in required_keys if key not in data]
    
    if missing:
        logger.warning(
            f"⛔ REQUIRED KEYS MISSING in {context}: {', '.join(missing)} "
            f"(have {len(data)} keys, need {len(required_keys)})"
        )
        for key in missing:
            METRICS.log_missing_key(context, key, "required_validation")
        return False
    
    return True


def log_data_quality_summary(fundamentals: Dict, indicators: Dict, patterns: Dict, symbol: str = ""):
    """
    Log a summary of data quality for debugging.
    Uses the logger configured by setup_logger().
    """
    logger.info(f"📊 DATA QUALITY SUMMARY {f'for {symbol}' if symbol else ''}")
    logger.info(f"   Fundamentals: {len(fundamentals) if fundamentals else 0} keys")
    logger.info(f"   Indicators: {len(indicators) if indicators else 0} keys")
    logger.info(f"   Patterns: {len(patterns) if patterns else 0} detected")
    
    if not fundamentals or len(fundamentals) < 5:
        logger.warning(f"⚠️  LOW FUNDAMENTAL DATA: Only {len(fundamentals) if fundamentals else 0} keys")
    
    if not indicators or len(indicators) < 10:
        logger.warning(f"⚠️  LOW INDICATOR DATA: Only {len(indicators) if indicators else 0} keys")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Original exports
    'setup_logger',
    'RemoveEmojiFilter',
    'LOG_FILE',
    'LOG_DIR',
    
    # New exports (optional - can be imported separately)
    'METRICS',
    'MetricsCollector',
    'SafeDict',
    'log_failures',
    'validate_output',
    'safe_get_nested',
    'validate_required_keys',
    'log_data_quality_summary',
]