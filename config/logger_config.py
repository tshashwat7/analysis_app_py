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
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import queue
import os
import re
import time
import atexit

# ============================================================================
# SECTION 1: MULTIPROCESSING-SAFE LOGGER SETUP
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


# Global queue and listener for thread/process-safe logging
_log_queue = None
_queue_listener = None


def setup_logger():
    """
    Initialize the logging system with multiprocessing-safe handlers.

    ✅ FIXED: Uses QueueHandler for process-safe logging
    ✅ Eliminates file locking issues on Windows

    Returns:
        Configured root logger instance
    """
    global _log_queue, _queue_listener

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # ---- Create queue for multiprocessing safety ----
    _log_queue = queue.Queue(-1)

    # ---- Console Handler (emoji removed) ----
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.addFilter(RemoveEmojiFilter())
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    # ---- File Handler (keeps emojis, no rotation on main thread) ----
    # ✅ CRITICAL: Use time-based rotation instead of size-based
    # This avoids the Windows file-locking issue during rotation
    from logging.handlers import TimedRotatingFileHandler
    fh = TimedRotatingFileHandler(
        LOG_FILE,
        when="midnight",      # Rotate at midnight
        interval=1,           # Every day
        backupCount=7,        # Keep 7 days of logs
        encoding="utf-8",
        delay=False           # Create file immediately (don't delay)
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    ))

    # ---- Queue Listener (runs in main thread, writes to file) ----
    # This ensures all writes happen from a single thread, avoiding lock contention
    _queue_listener = QueueListener(_log_queue, ch, fh, respect_handler_level=True)
    _queue_listener.start()

    # ---- Add queue handler to logger (all processes write here) ----
    qh = QueueHandler(_log_queue)
    qh.setLevel(logging.DEBUG)

    if not logger.handlers:
        logger.addHandler(qh)  # ← ONLY handler: QueueHandler

    # Suppress noisy third-party libraries
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    # Ensure cleanup on exit
    atexit.register(_cleanup_logger)

    return logger


def _cleanup_logger():
    """Clean up logging queue on shutdown."""
    global _queue_listener
    if _queue_listener:
        _queue_listener.stop()


# ============================================================================
# SECTION 2: SILENT FAILURE DETECTION SYSTEM (NEW - OPTIONAL)
# ============================================================================
# These utilities USE the logger configured above via logging.getLogger(__name__)
# They don't create their own logger - they use your existing setup!

import functools
import traceback
from typing import Any, Dict, List, Optional, Callable
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

        self.gate_checks = defaultdict(list)
        self.condition_evaluations = defaultdict(int)
        self.merge_operations = defaultdict(int)
        self.hierarchy_decisions = defaultdict(list)
        self.score_calculations = defaultdict(list)
        self.pattern_validations = defaultdict(list)
        self.composite_calculations = defaultdict(list)
        self.strategy_fits = defaultdict(list)
        self.performance_timings = defaultdict(list)
        
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
    
    # New Addition
    # ================================================================
    # GATE VALIDATION LOGGING
    # ================================================================
    
    def log_gate_check(
        self, 
        gate_name: str, 
        phase: str,
        passed: bool, 
        actual: Any, 
        required: Any,
        context: str = ""
    ):
        """
        Log individual gate check with context.
        
        Usage in _validate_structural_gates():
            METRICS.log_gate_check(
                gate_name="adx_min",
                phase="structural",
                passed=True,
                actual=28.5,
                required=25,
                context="MOMENTUM_BREAKOUT"
            )
        """
        result = {
            "passed": passed,
            "actual": actual,
            "required": required,
            "phase": phase,
            "context": context,
            "timestamp": time.time()
        }
        self.gate_checks[gate_name].append(result)
        
        # Log ONLY failures and first success (prevent spam)
        if not passed:
            logger.warning(
                f"{self._format_symbol_prefix()}❌ GATE FAILED [{phase}]: "
                f"{gate_name} | actual={actual} vs required={required} | {context}"
            )
        elif len(self.gate_checks[gate_name]) == 1:
            logger.debug(
                f"{self._format_symbol_prefix()}✅ GATE PASSED [{phase}]: "
                f"{gate_name} | {actual} >= {required}"
            )
    
    def log_gate_summary(self, phase: str, total: int, passed: int, failures: List[Dict]):
        """
        Log summary of gate validation phase.
        
        Usage at end of _validate_structural_gates():
            METRICS.log_gate_summary(
                phase="structural",
                total=12,
                passed=10,
                failures=[{"gate": "adx_min", "actual": 18, "required": 25}]
            )
        """
        if failures:
            logger.warning(
                f"{self._format_symbol_prefix()}⚠️ GATE PHASE [{phase}]: "
                f"{passed}/{total} passed | FAILURES: {[f['gate'] for f in failures]}"
            )
        else:
            logger.info(
                f"{self._format_symbol_prefix()}✅ GATE PHASE [{phase}]: "
                f"All {total} gates passed"
            )
    
    # ================================================================
    # CONDITION EVALUATION LOGGING
    # ================================================================
    
    def log_condition_evaluation(
        self,
        condition: str,
        result: bool,
        context: str = "",
        variables: Dict[str, Any] = None
    ):
        """
        Log condition evaluation with minimal overhead.
        
        Usage in ConditionEvaluator.evaluate_condition():
            METRICS.log_condition_evaluation(
                condition="rsi >= 60",
                result=True,
                context="MOMENTUM_BREAKOUT",
                variables={"rsi": 65.2}
            )
        """
        key = f"{context}.{condition[:50]}"  # Truncate long conditions
        self.condition_evaluations[key] += 1
        
        # Log ONLY first evaluation or failures (prevent spam)
        if not result or self.condition_evaluations[key] == 1:
            level = logging.DEBUG if result else logging.WARNING
            var_str = f" | vars={variables}" if variables and not result else ""
            
            logger.log(
                level,
                f"{self._format_symbol_prefix()}{'✓' if result else '✗'} CONDITION: "
                f"{condition} = {result} | {context}{var_str}"
            )
    
    # ================================================================
    # MERGE OPERATIONS LOGGING
    # ================================================================
    
    def log_merge_operation(
        self,
        merge_type: str,
        source1: str,
        source2: str,
        overrides: int,
        context: str = ""
    ):
        """
        Log configuration merge operations.
        
        Usage in _merge_gates():
            METRICS.log_merge_operation(
                merge_type="gates",
                source1="universal",
                source2="horizon_override",
                overrides=3,
                context="MOMENTUM_BREAKOUT"
            )
        """
        key = f"{merge_type}.{context}"
        self.merge_operations[key] += 1
        
        # Log only if overrides occurred (prevent spam for no-op merges)
        if overrides > 0:
            logger.debug(
                f"{self._format_symbol_prefix()}🔄 MERGE: {merge_type} | "
                f"{source1} + {source2} → {overrides} overrides | {context}"
            )
    
    # ================================================================
    # HIERARCHY DECISION LOGGING
    # ================================================================
    
    def log_hierarchy_decision(
        self,
        decision_type: str,
        winner: str,
        losers: List[str],
        reason: str,
        context: str = ""
    ):
        """
        Log hierarchy enforcement decisions.
        
        Usage in get_setup_priority():
            METRICS.log_hierarchy_decision(
                decision_type="priority",
                winner="horizon_override",
                losers=["setup_default", "master_config"],
                reason="Horizon override takes precedence",
                context="MOMENTUM_BREAKOUT"
            )
        """
        decision = {
            "winner": winner,
            "losers": losers,
            "reason": reason,
            "context": context,
            "timestamp": time.time()
        }
        self.hierarchy_decisions[decision_type].append(decision)
        
        # Log ONLY when non-default source wins (prevent spam)
        if winner not in ["default", "master_config"]:
            logger.info(
                f"{self._format_symbol_prefix()}🏆 HIERARCHY [{decision_type}]: "
                f"{winner} wins over {losers} | {reason}"
            )
    
    # ================================================================
    # SCORE CALCULATION LOGGING
    # ================================================================
    
    def log_score_calculation(
    self,
    score_type: str,
    score: float,
    breakdown: Dict[str, float],
    elapsed: float = 0
    ):
        """
        Log composite score calculations.
        
        Usage in _calculate_all_scores():
            METRICS.log_score_calculation(
                score_type="technical",
                score=7.5,
                breakdown={"trend": 8.0, "momentum": 7.0},
                elapsed=0.005
            )
        """
        result = {
            "score": score,
            "breakdown": breakdown,
            "elapsed": elapsed,
            "timestamp": time.time()
        }
        self.score_calculations[score_type].append(result)
        
        # Log only first calculation per stock (prevent spam)
        if len(self.score_calculations[score_type]) == 1:
            timing = f" ({elapsed*1000:.1f}ms)" if elapsed > 0 else ""
            logger.info(
                f"{self._format_symbol_prefix()}🎯 SCORE [{score_type}]: "
                f"{score:.1f}/10{timing}"
            )
            
            # Breakdown at DEBUG level with safe formatting
            if breakdown:
                # Extract numeric values from nested dicts
                breakdown_str = []
                for k, v in breakdown.items():
                    # Handle nested dict structure
                    if isinstance(v, dict):
                        # Try to extract score/value/raw
                        numeric_val = v.get("score") or v.get("value") or v.get("raw")
                        if numeric_val is not None:
                            breakdown_str.append(f"{k}={numeric_val:.1f}")
                        else:
                            breakdown_str.append(f"{k}=<dict>")
                    elif isinstance(v, (int, float)):
                        breakdown_str.append(f"{k}={v:.1f}")
                    else:
                        breakdown_str.append(f"{k}={v}")
                
                logger.debug(
                    f"   └─ Breakdown: " + ", ".join(breakdown_str)
                )
    
    # ================================================================
    # PATTERN VALIDATION LOGGING
    # ================================================================
    
    def log_pattern_validation(
        self,
        pattern_name: str,
        found: bool,
        quality: float = 0,
        invalidated: bool = False,
        reason: str = ""
    ):
        """
        Log pattern detection and validation.
        
        Usage in _validate_patterns():
            METRICS.log_pattern_validation(
                pattern_name="darvasBox",
                found=True,
                quality=8.5,
                invalidated=False,
                reason="All criteria met"
            )
        """
        result = {
            "found": found,
            "quality": quality,
            "invalidated": invalidated,
            "reason": reason,
            "timestamp": time.time()
        }
        self.pattern_validations[pattern_name].append(result)
        
        # Log only if found or invalidated
        if found and not invalidated:
            logger.info(
                f"{self._format_symbol_prefix()}📊 PATTERN: {pattern_name} "
                f"found (quality={quality:.1f})"
            )
        elif invalidated:
            logger.warning(
                f"{self._format_symbol_prefix()}⚠️ PATTERN INVALIDATED: "
                f"{pattern_name} | {reason}"
            )
    
    # ================================================================
    # STRATEGY FIT LOGGING
    # ================================================================
    
    def log_strategy_fit(
        self,
        strategy_name: str,
        fit_score: float,
        weighted_score: float,
        horizon: str,
        multipliers: Dict[str, float] = None,
        elapsed: float = 0
    ):
        """
        Log strategy fit calculation.
        
        Usage in _classify_strategy():
            METRICS.log_strategy_fit(
                strategy_name="momentum",
                fit_score=75.0,
                weighted_score=90.0,
                horizon="short_term",
                multipliers={"horizon": 1.2, "master": 1.0},
                elapsed=0.003
            )
        """
        result = {
            "fit_score": fit_score,
            "weighted_score": weighted_score,
            "horizon": horizon,
            "multipliers": multipliers or {},
            "elapsed": elapsed,
            "timestamp": time.time()
        }
        self.strategy_fits[strategy_name].append(result)
        
        timing = f" ({elapsed*1000:.1f}ms)" if elapsed > 0 else ""
        mult_str = ""
        if multipliers:
            mult_str = f" (mult={multipliers.get('horizon', 1.0):.2f})"
        
        logger.info(
            f"{self._format_symbol_prefix()}🎯 STRATEGY: {strategy_name} | "
            f"fit={fit_score:.1f} → weighted={weighted_score:.1f}{mult_str}{timing}"
        )
    
    # ================================================================
    # PERFORMANCE TIMING
    # ================================================================
    
    def log_performance(self, method_name: str, elapsed: float, threshold_ms: float = 50):
        """
        Log method performance with smart filtering.
        
        Usage with context manager:
            start = time.time()
            # ... do work ...
            METRICS.log_performance("_calculate_all_scores", time.time() - start)
        """
        self.performance_timings[method_name].append(elapsed)
        
        elapsed_ms = elapsed * 1000
        
        # Calculate running statistics
        timings = self.performance_timings[method_name]
        avg_ms = sum(timings) / len(timings) * 1000
        
        # Log ONLY if significantly slow
        if elapsed_ms > threshold_ms or elapsed_ms > avg_ms * 2:
            logger.warning(
                f"{self._format_symbol_prefix()}⏱️ SLOW: {method_name} "
                f"took {elapsed_ms:.1f}ms (avg={avg_ms:.1f}ms)"
            )
    
    # ================================================================
    # ENHANCED SUMMARY
    # ================================================================
    
    def get_summary(self) -> Dict[str, Any]:
        """Enhanced summary with all resolver metrics."""
        summary = {
            "symbol": self._current_symbol,
            "missing_keys": dict(self.missing_keys),
            "failed_methods": dict(self.failed_methods),
            "fallback_usage": dict(self.fallback_usage),
            "none_returns": dict(self.none_returns),
            "empty_returns": dict(self.empty_returns),
            "validation_failures": dict(self.validation_failures),
        }
        
        # Gate checks summary
        if self.gate_checks:
            summary["gates"] = {
                name: {
                    "total": len(checks),
                    "passed": sum(1 for c in checks if c["passed"]),
                    "failed": sum(1 for c in checks if not c["passed"])
                }
                for name, checks in self.gate_checks.items()
            }
        
        # Condition evaluations summary
        if self.condition_evaluations:
            summary["conditions"] = {
                "total_evaluations": sum(self.condition_evaluations.values()),
                "unique_conditions": len(self.condition_evaluations)
            }
        
        # Score calculations summary
        if self.score_calculations:
            summary["scores"] = {
                score_type: {
                    "avg_score": sum(s["score"] for s in calcs) / len(calcs),
                    "avg_time_ms": sum(s["elapsed"] for s in calcs) / len(calcs) * 1000,
                    "count": len(calcs)
                }
                for score_type, calcs in self.score_calculations.items()
            }
        
        # Strategy fits summary
        if self.strategy_fits:
            summary["strategies"] = {
                name: {
                    "avg_fit": sum(s["fit_score"] for s in fits) / len(fits),
                    "avg_weighted": sum(s["weighted_score"] for s in fits) / len(fits),
                    "count": len(fits)
                }
                for name, fits in self.strategy_fits.items()
            }
        
        # Performance summary
        if self.performance_timings:
            summary["performance"] = {
                method: {
                    "avg_ms": sum(timings) / len(timings) * 1000,
                    "max_ms": max(timings) * 1000,
                    "min_ms": min(timings) * 1000,
                    "count": len(timings)
                }
                for method, timings in self.performance_timings.items()
            }
        
        # Total issues count
        summary["total_issues"] = (
            sum(self.missing_keys.values()) +
            sum(self.failed_methods.values()) +
            sum(self.none_returns.values()) +
            sum(self.empty_returns.values()) +
            sum(self.validation_failures.values()) +
            sum(1 for checks in self.gate_checks.values() 
                for c in checks if not c["passed"])
        )
        
        return summary
    
    def reset(self):
        """Enhanced reset for all metrics."""
        # Existing resets
        self.missing_keys.clear()
        self.failed_methods.clear()
        self.fallback_usage.clear()
        self.none_returns.clear()
        self.empty_returns.clear()
        self.validation_failures.clear()
        
        # New resets
        self.gate_checks.clear()
        self.condition_evaluations.clear()
        self.merge_operations.clear()
        self.hierarchy_decisions.clear()
        self.score_calculations.clear()
        self.pattern_validations.clear()
        self.composite_calculations.clear()
        self.strategy_fits.clear()
        self.performance_timings.clear()
        
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

# ====================================================================
    # ✅ ADD METHOD 1: Track Composite Calculations
    # ====================================================================
    def log_composite_calculation(
        self,
        composite_name: str,
        score: float,
        elapsed: float,
        components: Dict[str, Any] = None
    ):
        """Log composite score calculation with timing."""
        self.composite_calculations[composite_name].append({
            "score": score,
            "elapsed": elapsed,
            "components": components,
            "timestamp": time.time()
        })
        
        # Only log first calculation per symbol (prevent spam)
        if len(self.composite_calculations[composite_name]) == 1:
            logger.debug(
                f"{self._format_symbol_prefix()}🧮 COMPOSITE: {composite_name} = {score:.1f}/10 "
                f"({elapsed*1000:.1f}ms)"
            )
    
    # ====================================================================
    # ✅ ADD METHOD 2: Track Strategy Fit
    # ====================================================================
    def log_strategy_fit(
        self,
        strategy_name: str,
        fit_score: float,
        weighted_score: float,
        horizon: str,
        elapsed: float,
        candidates_count: int = 0
    ):
        """Log strategy fit calculation with timing."""
        self.strategy_fits[strategy_name].append({
            "fit_score": fit_score,
            "weighted_score": weighted_score,
            "horizon": horizon,
            "elapsed": elapsed,
            "candidates_count": candidates_count,
            "timestamp": time.time()
        })
        
        logger.debug(
            f"{self._format_symbol_prefix()}🎯 STRATEGY FIT: {strategy_name} "
            f"(fit={fit_score:.1f}, weighted={weighted_score:.1f}, {elapsed*1000:.1f}ms)"
        )
    
    # ====================================================================
    # ✅ ADD METHOD 3: Track Performance Timing
    # ====================================================================
    def log_performance(self, method_name: str, elapsed: float):
        """Log method performance timing."""
        self.performance_timings[method_name].append(elapsed)
        
        # Calculate running average
        avg = sum(self.performance_timings[method_name]) / len(self.performance_timings[method_name])
        
        # Only log if slower than average (detect performance issues)
        if elapsed > avg * 1.5 and len(self.performance_timings[method_name]) > 3:
            logger.warning(
                f"{self._format_symbol_prefix()}⏱️ SLOW: {method_name} took {elapsed*1000:.1f}ms "
                f"(avg={avg*1000:.1f}ms)"
            )
    
    # ====================================================================
    # ✅ UPDATE get_summary() to include new metrics
    # ====================================================================
    def get_summary(self) -> Dict[str, Any]:
        """Enhanced summary with composite and strategy metrics."""
        # Your existing summary code...
        base_summary = {
            "symbol": self._current_symbol,
            "missing_keys": dict(self.missing_keys),
            "failed_methods": dict(self.failed_methods),
            # ... rest of your existing code ...
        }
        
        # ✅ ADD THIS: Include composite and strategy metrics
        if self.composite_calculations:
            base_summary["composites"] = {
                name: {
                    "avg_score": sum(c["score"] for c in calcs) / len(calcs),
                    "avg_time_ms": sum(c["elapsed"] for c in calcs) / len(calcs) * 1000,
                    "count": len(calcs)
                }
                for name, calcs in self.composite_calculations.items()
            }
        
        if self.strategy_fits:
            base_summary["strategies"] = {
                name: {
                    "avg_fit": sum(s["fit_score"] for s in fits) / len(fits),
                    "avg_weighted": sum(s["weighted_score"] for s in fits) / len(fits),
                    "count": len(fits)
                }
                for name, fits in self.strategy_fits.items()
            }
        
        if self.performance_timings:
            base_summary["performance"] = {
                method: {
                    "avg_ms": sum(timings) / len(timings) * 1000,
                    "max_ms": max(timings) * 1000,
                    "count": len(timings)
                }
                for method, timings in self.performance_timings.items()
            }
        
        return base_summary

# ================================================================
# CONTEXT MANAGER FOR PERFORMANCE TRACKING
# ================================================================

import contextlib

@contextlib.contextmanager
def track_performance(method_name: str):
    """
    Context manager for easy performance tracking.
    
    Usage:
        with track_performance("_calculate_all_scores"):
            result = self._calculate_all_scores(ctx)
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        METRICS.log_performance(method_name, elapsed)


# ================================================================
# HELPER: LOG DATA QUALITY AT START
# ================================================================

def log_resolver_context_quality(
    fundamentals: Dict,
    indicators: Dict,
    patterns: Dict,
    symbol: str
):
    """
    Log data quality summary at resolver initialization.
    
    Usage in _build_evaluation_context():
        log_resolver_context_quality(
            fundamentals=fundamentals,
            indicators=indicators,
            patterns=detected_patterns,
            symbol=symbol
        )
    """
    METRICS.set_current_symbol(symbol)
    
    fund_count = len(fundamentals) if fundamentals else 0
    ind_count = len(indicators) if indicators else 0
    pat_count = len(patterns) if patterns else 0
    
    logger.info(
        f"[{symbol}] 📊 INPUT DATA QUALITY: "
        f"Fundamentals={fund_count}, Indicators={ind_count}, Patterns={pat_count}"
    )
    
    # Warn on low quality
    if fund_count < 5:
        logger.warning(f"[{symbol}] ⚠️ LOW FUNDAMENTAL DATA: {fund_count} keys")
    if ind_count < 10:
        logger.warning(f"[{symbol}] ⚠️ LOW INDICATOR DATA: {ind_count} keys")

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
    'log_composite_calculation',
    'log_strategy_fit',
    'log_performance',
    'get_summary',
]
    