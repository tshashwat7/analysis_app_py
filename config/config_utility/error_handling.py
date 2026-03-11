# config/error_handling.py
"""
Custom Error Handling System for Trading Platform

HIERARCHY:
==========
TradingSystemError (Base)
├── DataError
│   ├── InsufficientDataError
│   ├── MissingFundamentalsError
│   └── InvalidDataFormatError
├── AnalysisError
│   ├── PatternDetectionError
│   ├── CompositeCalculationError
│   └── StrategyAnalysisError
├── ConfigurationError
│   ├── InvalidHorizonError
│   ├── MissingMatrixError
│   └── ResolverError
└── ExecutionError
    ├── GateValidationError
    ├── PositionSizingError
    └── TradeGenerationError
"""

import logging
from typing import Optional, Dict, Any
from functools import wraps
import time

logger = logging.getLogger(__name__)


# ==============================================================================
# BASE ERROR CLASS
# ==============================================================================

class TradingSystemError(Exception):
    """
    Base exception for all trading system errors.
    
    Attributes:
        message: Error message
        ticker: Stock symbol (if applicable)
        context: Additional context dict
        retry_count: Number of retries attempted
        is_recoverable: Whether error can be recovered from
    """
    
    def __init__(
        self,
        message: str,
        ticker: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        is_recoverable: bool = False
    ):
        self.message = message
        self.ticker = ticker
        self.context = context or {}
        self.is_recoverable = is_recoverable
        self.retry_count = 0
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with context."""
        parts = []
        
        if self.ticker:
            parts.append(f"[{self.ticker}]")
        
        parts.append(self.message)
        
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"({context_str})")
        
        return " ".join(parts)
    
    def __str__(self) -> str:
        return self.format_message()


# ==============================================================================
# DATA ERRORS
# ==============================================================================

class DataError(TradingSystemError):
    """Base class for data-related errors."""
    pass


class InsufficientDataError(DataError):
    """Raised when insufficient data is available for analysis."""
    
    def __init__(
        self,
        ticker: str,
        data_type: str,
        required: int,
        actual: int
    ):
        context = {
            "data_type": data_type,
            "required": required,
            "actual": actual
        }
        super().__init__(
            f"Insufficient {data_type}: need {required}, got {actual}",
            ticker=ticker,
            context=context,
            is_recoverable=False  # Can't recover without data
        )


class MissingFundamentalsError(DataError):
    """Raised when required fundamental data is missing."""
    
    def __init__(
        self,
        ticker: str,
        missing_fields: list
    ):
        context = {"missing_fields": missing_fields}
        super().__init__(
            f"Missing fundamentals: {', '.join(missing_fields)}",
            ticker=ticker,
            context=context,
            is_recoverable=True  # Can skip fundamental-dependent features
        )


class InvalidDataFormatError(DataError):
    """Raised when data format is invalid."""
    
    def __init__(
        self,
        ticker: str,
        field: str,
        expected_type: str,
        actual_type: str
    ):
        context = {
            "field": field,
            "expected": expected_type,
            "actual": actual_type
        }
        super().__init__(
            f"Invalid format for '{field}': expected {expected_type}, got {actual_type}",
            ticker=ticker,
            context=context,
            is_recoverable=False
        )


# ==============================================================================
# ANALYSIS ERRORS
# ==============================================================================

class AnalysisError(TradingSystemError):
    """Base class for analysis-related errors."""
    pass


class PatternDetectionError(AnalysisError):
    """Raised when pattern detection fails."""
    
    def __init__(
        self,
        ticker: str,
        pattern_name: str,
        reason: str
    ):
        context = {"pattern": pattern_name, "reason": reason}
        super().__init__(
            f"Pattern detection failed for '{pattern_name}': {reason}",
            ticker=ticker,
            context=context,
            is_recoverable=True  # Can continue without this pattern
        )


class CompositeCalculationError(AnalysisError):
    """Raised when composite score calculation fails."""
    
    def __init__(
        self,
        ticker: str,
        composite_name: str,
        reason: str
    ):
        context = {"composite": composite_name, "reason": reason}
        super().__init__(
            f"Composite calculation failed for '{composite_name}': {reason}",
            ticker=ticker,
            context=context,
            is_recoverable=True  # Can use defaults
        )


class StrategyAnalysisError(AnalysisError):
    """Raised when strategy analysis fails."""
    
    def __init__(
        self,
        ticker: str,
        horizon: str,
        reason: str
    ):
        context = {"horizon": horizon, "reason": reason}
        super().__init__(
            f"Strategy analysis failed for '{horizon}': {reason}",
            ticker=ticker,
            context=context,
            is_recoverable=True  # Can fallback to generic
        )


# ==============================================================================
# CONFIGURATION ERRORS
# ==============================================================================

class ConfigurationError(TradingSystemError):
    """Base class for configuration-related errors."""
    pass


class InvalidHorizonError(ConfigurationError):
    """Raised when invalid horizon is specified."""
    
    def __init__(self, horizon: str, valid_horizons: list):
        context = {"horizon": horizon, "valid": valid_horizons}
        super().__init__(
            f"Invalid horizon '{horizon}', must be one of: {', '.join(valid_horizons)}",
            context=context,
            is_recoverable=False  # System error
        )


class MissingMatrixError(ConfigurationError):
    """Raised when required matrix is missing."""
    
    def __init__(self, matrix_name: str, reason: str):
        context = {"matrix": matrix_name, "reason": reason}
        super().__init__(
            f"Matrix '{matrix_name}' unavailable: {reason}",
            context=context,
            is_recoverable=True  # Can use fallbacks
        )


class ResolverError(ConfigurationError):
    """Raised when resolver fails to initialize."""
    
    def __init__(self, horizon: str, reason: str):
        context = {"horizon": horizon, "reason": reason}
        super().__init__(
            f"Resolver initialization failed for '{horizon}': {reason}",
            context=context,
            is_recoverable=False
        )


# ==============================================================================
# EXECUTION ERRORS
# ==============================================================================

class ExecutionError(TradingSystemError):
    """Base class for execution-related errors."""
    pass


class GateValidationError(ExecutionError):
    """Raised when entry gates fail validation."""
    
    def __init__(
        self,
        ticker: str,
        failed_gates: list,
        gate_details: Dict[str, Any]
    ):
        context = {"failed_gates": failed_gates, "details": gate_details}
        super().__init__(
            f"Entry gates failed: {', '.join(failed_gates)}",
            ticker=ticker,
            context=context,
            is_recoverable=False  # Trade blocked
        )


class PositionSizingError(ExecutionError):
    """Raised when position sizing fails."""
    
    def __init__(self, ticker: str, reason: str):
        context = {"reason": reason}
        super().__init__(
            f"Position sizing failed: {reason}",
            ticker=ticker,
            context=context,
            is_recoverable=False
        )


class TradeGenerationError(ExecutionError):
    """Raised when trade plan generation fails."""
    
    def __init__(self, ticker: str, stage: str, reason: str):
        context = {"stage": stage, "reason": reason}
        super().__init__(
            f"Trade generation failed at stage '{stage}': {reason}",
            ticker=ticker,
            context=context,
            is_recoverable=False
        )


# ==============================================================================
# CIRCUIT BREAKER (Prevent Repeated Failures)
# ==============================================================================

class CircuitBreaker:
    """
    Prevents repeated failures on problematic stocks.
    
    After N failures within a time window, auto-skip the stock.
    """
    
    def __init__(
        self,
        failure_threshold: int = 3,
        time_window_seconds: int = 3600,  # 1 hour
        cooldown_seconds: int = 7200  # 2 hours
    ):
        self.failure_threshold = failure_threshold
        self.time_window = time_window_seconds
        self.cooldown = cooldown_seconds
        
        # Track failures: {ticker: [(timestamp, error_type), ...]}
        self._failures: Dict[str, list] = {}
        
        # Track blocked stocks: {ticker: block_timestamp}
        self._blocked: Dict[str, float] = {}
    
    def record_failure(self, ticker: str, error: TradingSystemError):
        """Record a failure for a ticker."""
        now = time.time()
        
        if ticker not in self._failures:
            self._failures[ticker] = []
        
        # Add new failure
        self._failures[ticker].append((now, type(error).__name__))
        
        # Clean old failures outside time window
        self._failures[ticker] = [
            (ts, err) for ts, err in self._failures[ticker]
            if now - ts < self.time_window
        ]
        
        # Check if threshold exceeded
        if len(self._failures[ticker]) >= self.failure_threshold:
            self._blocked[ticker] = now
            logger.warning(
                f"🚨 CIRCUIT BREAKER: {ticker} blocked due to {self.failure_threshold} "
                f"failures in {self.time_window}s. Cooldown: {self.cooldown}s"
            )
    
    def is_blocked(self, ticker: str) -> bool:
        """Check if ticker is currently blocked."""
        if ticker not in self._blocked:
            return False
        
        now = time.time()
        blocked_at = self._blocked[ticker]
        
        # Check if cooldown expired
        if now - blocked_at > self.cooldown:
            # Unblock
            del self._blocked[ticker]
            logger.info(f"✅ CIRCUIT BREAKER: {ticker} unblocked (cooldown expired)")
            return False
        
        return True
    
    def get_failure_count(self, ticker: str) -> int:
        """Get current failure count for ticker."""
        if ticker not in self._failures:
            return 0
        
        now = time.time()
        recent = [
            f for f in self._failures[ticker]
            if now - f[0] < self.time_window
        ]
        return len(recent)
    
    def reset(self, ticker: Optional[str] = None):
        """Reset circuit breaker (for testing)."""
        if ticker:
            self._failures.pop(ticker, None)
            self._blocked.pop(ticker, None)
        else:
            self._failures.clear()
            self._blocked.clear()


# Global circuit breaker instance
CIRCUIT_BREAKER = CircuitBreaker()


# ==============================================================================
# ERROR HANDLING DECORATORS
# ==============================================================================

def handle_analysis_errors(ticker_arg_index: int = 0):
    """
    Decorator to handle analysis errors gracefully.
    
    Args:
        ticker_arg_index: Position of ticker argument in function signature
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract ticker from args
            ticker = "UNKNOWN"
            if len(args) > ticker_arg_index:
                ticker = args[ticker_arg_index]
            elif "ticker" in kwargs:
                ticker = kwargs["ticker"]
            elif "symbol" in kwargs:
                ticker = kwargs["symbol"]
            
            # Check circuit breaker
            if CIRCUIT_BREAKER.is_blocked(ticker):
                logger.warning(f"⛔ {ticker} blocked by circuit breaker, skipping")
                return {
                    "status": "BLOCKED",
                    "ticker": ticker,
                    "reason": "Circuit breaker active (repeated failures)",
                    "error_type": "CircuitBreakerActive"
                }
            
            try:
                return func(*args, **kwargs)
            
            except TradingSystemError as e:
                # Record failure
                CIRCUIT_BREAKER.record_failure(ticker, e)
                
                # Log appropriately
                if e.is_recoverable:
                    logger.warning(f"⚠️ Recoverable error for {ticker}: {e}")
                else:
                    logger.error(f"❌ Fatal error for {ticker}: {e}")
                
                # Return error result
                return {
                    "status": "ERROR",
                    "ticker": ticker,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "is_recoverable": e.is_recoverable,
                    "context": e.context
                }
            
            except Exception as e:
                # Unexpected error - log with full traceback
                logger.error(
                    f"🔥 UNEXPECTED ERROR for {ticker}: {e}",
                    exc_info=True
                )
                
                # Wrap in generic error
                wrapped_error = TradingSystemError(
                    f"Unexpected error: {str(e)}",
                    ticker=ticker,
                    is_recoverable=False
                )
                CIRCUIT_BREAKER.record_failure(ticker, wrapped_error)
                
                return {
                    "status": "ERROR",
                    "ticker": ticker,
                    "error": str(e),
                    "error_type": "UnexpectedError",
                    "is_recoverable": False
                }
        
        return wrapper
    return decorator


# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================

if __name__ == "__main__":
    # Example 1: Raising specific errors
    try:
        raise InsufficientDataError(
            ticker="RELIANCE",
            data_type="indicators",
            required=20,
            actual=8
        )
    except TradingSystemError as e:
        print(f"Error: {e}")
        print(f"Recoverable: {e.is_recoverable}")
        print(f"Context: {e.context}")
    
    # Example 2: Using decorator
    @handle_analysis_errors(ticker_arg_index=0)
    def analyze_stock(ticker: str, data: dict):
        if not data:
            raise InsufficientDataError(
                ticker=ticker,
                data_type="fundamentals",
                required=10,
                actual=0
            )
        return {"ticker": ticker, "result": "success"}
    
    result = analyze_stock("TEST", {})
    print(f"\nResult: {result}")
    
    # Example 3: Circuit breaker
    print("\n=== Testing Circuit Breaker ===")
    for i in range(4):
        error = InsufficientDataError("FAILED_STOCK", "indicators", 20, 5)
        CIRCUIT_BREAKER.record_failure("FAILED_STOCK", error)
        print(f"Failure {i+1}: Blocked = {CIRCUIT_BREAKER.is_blocked('FAILED_STOCK')}")