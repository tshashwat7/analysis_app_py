# services/patterns/pattern_performance_updater.py
"""
Pattern Performance Update System
==================================
Monitors positions and updates pattern velocity tracking when:
- Targets are hit (T1/T2)
- Stop loss triggered
- Pattern invalidated

🎯 PURPOSE: Feed real-world performance data back into timeline estimator
"""

import logging
from typing import Dict, Any, Optional, List
from services.patterns.pattern_velocity_tracking import update_pattern_performance

logger = logging.getLogger(__name__)


# ============================================================
# SCENARIO 1: TARGET HIT (T1 or T2)
# ============================================================

def on_target_hit(
    symbol: str,
    target_level: str,  # "T1" or "T2"
    current_price: float,
    current_bar: int,
    trade_metadata: Dict[str, Any]
) -> bool:
    """
    Called when a target is reached.
    
    Integration points:
    - Order execution system (when limit order fills)
    - Position monitoring (when price crosses target)
    - Manual exit (user takes profit at target)
    
    Args:
        symbol: Stock symbol
        target_level: "T1" or "T2"
        current_price: Current market price
        current_bar: Current bar/candle number
        trade_metadata: Original trade plan metadata
    
    Returns:
        Success status
    
    Example:
        >>> # In your order execution handler:
        >>> if order.type == "LIMIT" and order.price == trade_plan["targets"]["t1"]:
        ...     on_target_hit(
        ...         symbol="RELIANCE.NS",
        ...         target_level="T1",
        ...         current_price=2850.50,
        ...         current_bar=45,
        ...         trade_metadata=trade_plan
        ...     )
    """
    try:
        # Extract tracking info from trade metadata
        velocity_tracking_id = trade_metadata.get("velocity_tracking_id")
        if not trade_metadata.get("primary_pattern") or not trade_metadata.get("horizon"):
            logger.debug(f"[{symbol}] detected velocity tracking ID {velocity_tracking_id}")
            return False
        
        # Extract pattern info
        setup_type = trade_metadata.get("setup_type")
        horizon = trade_metadata.get("horizon")
        
        if not setup_type or not horizon:
            logger.warning(f"[{symbol}] Missing setup_type or horizon in metadata")
            return False
        
        # Determine which target was hit
        t1_reached = (target_level == "T1")
        t2_reached = (target_level == "T2")
        
        # Get pattern name from metadata
        pattern_name = _extract_pattern_name_from_metadata(trade_metadata)
        if not pattern_name:
            logger.warning(f"[{symbol}] Could not determine pattern name")
            return False
        record_id = trade_metadata.get("velocity_tracking_id")
        # Update performance tracking
        success = update_pattern_performance(
            record_id=record_id,
            symbol=symbol,
            pattern_name=pattern_name,
            horizon=horizon,
            current_price=current_price,
            current_bar=current_bar,
            t1_reached=t1_reached,
            t2_reached=t2_reached
        )
        
        if success:
            logger.info(
                f"✅ [{symbol}] Pattern performance updated: {target_level} hit "
                f"for {pattern_name}"
            )
        
        return success
    
    except Exception as e:
        logger.error(f"❌ on_target_hit failed for {symbol}: {e}", exc_info=True)
        return False


# ============================================================
# SCENARIO 2: STOP LOSS TRIGGERED
# ============================================================

def on_stop_loss_triggered(
    symbol: str,
    current_price: float,
    current_bar: int,
    trade_metadata: Dict[str, Any],
    reason: str = "STOP_LOSS_HIT"
) -> bool:
    """
    Called when stop loss is triggered.
    
    Integration points:
    - Stop loss order execution
    - Risk management system
    - Emergency exit handler
    
    Args:
        symbol: Stock symbol
        current_price: Exit price
        current_bar: Current bar number
        trade_metadata: Original trade plan
        reason: Exit reason code
    
    Returns:
        Success status
    
    Example:
        >>> # In your stop loss handler:
        >>> if price <= trade_plan["stop_loss"]:
        ...     on_stop_loss_triggered(
        ...         symbol="TCS.NS",
        ...         current_price=3250.00,
        ...         current_bar=12,
        ...         trade_metadata=trade_plan,
        ...         reason="STOP_LOSS_HIT"
        ...     )
    """
    try:
        velocity_tracking_id = trade_metadata.get("velocity_tracking_id")
        if not velocity_tracking_id:
            return False
        
        setup_type = trade_metadata.get("setup_type")
        horizon = trade_metadata.get("horizon")
        
        if not setup_type or not horizon:
            return False
        
        pattern_name = _extract_pattern_name_from_metadata(trade_metadata)
        if not pattern_name:
            return False
        record_id = trade_metadata.get("velocity_tracking_id")
        # Mark as stopped out
        success = update_pattern_performance(
            record_id=record_id,
            symbol=symbol,
            pattern_name=pattern_name,
            horizon=horizon,
            current_price=current_price,
            current_bar=current_bar,
            stopped_out=True,
            exit_reason=reason
        )
        
        if success:
            logger.info(
                f"🛑 [{symbol}] Pattern stopped out: {pattern_name} | "
                f"Reason: {reason}"
            )
        
        return success
    
    except Exception as e:
        logger.error(f"❌ on_stop_loss_triggered failed: {e}", exc_info=True)
        return False


# ============================================================
# SCENARIO 3: PATTERN INVALIDATION
# ============================================================

def on_pattern_invalidation(
    symbol: str,
    current_price: float,
    current_bar: int,
    trade_metadata: Dict[str, Any],
    invalidation_reason: str
) -> bool:
    """
    Called when pattern breaks down (from trade_enhancer.py).
    
    Integration points:
    - Pattern invalidation monitoring (trade_enhancer.py)
    - Real-time pattern validation
    - Technical breakdown detection
    
    Args:
        symbol: Stock symbol
        current_price: Price at invalidation
        current_bar: Current bar number
        trade_metadata: Original trade plan
        invalidation_reason: Breakdown description
    
    Returns:
        Success status
    
    Example:
        >>> # In trade_enhancer.py check_pattern_invalidation():
        >>> if invalidation["invalidated"]:
        ...     on_pattern_invalidation(
        ...         symbol="INFY.NS",
        ...         current_price=1450.25,
        ...         current_bar=8,
        ...         trade_metadata=active_trade,
        ...         invalidation_reason="Darvas box breakdown"
        ...     )
    """
    try:
        velocity_tracking_id = trade_metadata.get("velocity_tracking_id")
        if not velocity_tracking_id:
            return False
        
        setup_type = trade_metadata.get("setup_type")
        horizon = trade_metadata.get("horizon")
        
        if not setup_type or not horizon:
            return False
        
        pattern_name = _extract_pattern_name_from_metadata(trade_metadata)
        if not pattern_name:
            return False
        record_id = trade_metadata.get("velocity_tracking_id")
        # Mark as invalidated
        success = update_pattern_performance(
            record_id=record_id,
            symbol=symbol,
            pattern_name=pattern_name,
            horizon=horizon,
            current_price=current_price,
            current_bar=current_bar,
            invalidated=True,
            exit_reason=f"INVALIDATED: {invalidation_reason}"
        )
        
        if success:
            logger.info(
                f"❌ [{symbol}] Pattern invalidated: {pattern_name} | "
                f"{invalidation_reason}"
            )
        
        return success
    
    except Exception as e:
        logger.error(f"❌ on_pattern_invalidation failed: {e}", exc_info=True)
        return False


# ============================================================
# SCENARIO 4: BATCH UPDATE (Position Monitoring)
# ============================================================

def update_active_positions(
    active_positions: List[Dict[str, Any]],
    current_prices: Dict[str, float],
    current_bar: int
) -> Dict[str, Any]:
    """
    Batch updates for position monitoring system.
    
    Call this from your periodic position checker (e.g., every candle close).
    
    Args:
        active_positions: List of active trades with metadata
        current_prices: Dict of {symbol: current_price}
        current_bar: Current bar number
    
    Returns:
        Update summary
    
    Example:
        >>> # In your position monitoring loop:
        >>> active_trades = get_active_trades_from_db()
        >>> current_prices = get_current_prices(symbols)
        >>> 
        >>> summary = update_active_positions(
        ...     active_positions=active_trades,
        ...     current_prices=current_prices,
        ...     current_bar=current_bar_number
        ... )
        >>> 
        >>> print(f"Updated: {summary['t1_hits']} T1s, {summary['t2_hits']} T2s")
    """
    summary = {
        "t1_hits": 0,
        "t2_hits": 0,
        "stop_outs": 0,
        "invalidations": 0,
        "errors": 0
    }
    
    for position in active_positions:
        try:
            symbol = position["symbol"]
            current_price = current_prices.get(symbol)
            
            if not current_price:
                continue
            
            trade_metadata = position.get("metadata", {})
            if not trade_metadata.get("velocity_tracking_id"):
                continue
            
            # Check targets
            t1 = position.get("targets", {}).get("t1")
            t2 = position.get("targets", {}).get("t2")
            stop_loss = position.get("stop_loss")
            
            # T1 check
            if t1 and current_price >= t1:
                if on_target_hit(symbol, "T1", current_price, current_bar, trade_metadata):
                    summary["t1_hits"] += 1
            
            # T2 check
            if t2 and current_price >= t2:
                if on_target_hit(symbol, "T2", current_price, current_bar, trade_metadata):
                    summary["t2_hits"] += 1
            
            # Stop loss check
            if stop_loss and current_price <= stop_loss:
                if on_stop_loss_triggered(symbol, current_price, current_bar, trade_metadata):
                    summary["stop_outs"] += 1
        
        except Exception as e:
            logger.error(f"Failed to update position {position.get('symbol')}: {e}")
            summary["errors"] += 1
    
    if summary["t1_hits"] > 0 or summary["t2_hits"] > 0 or summary["stop_outs"] > 0:
        logger.info(
            f"📊 Batch update complete: "
            f"T1={summary['t1_hits']}, T2={summary['t2_hits']}, "
            f"Stops={summary['stop_outs']}, Errors={summary['errors']}"
        )
    
    return summary


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _extract_pattern_name_from_metadata(trade_metadata: Dict[str, Any]) -> Optional[str]:
    """
    Extracts pattern name from trade plan metadata.
    
    Pattern name could be stored in:
    1. metadata.primary_pattern
    2. eval_ctx.patterns (extract primary)
    3. setup_type (fallback to setup name)
    """
    # Method 1: Direct pattern name
    if "primary_pattern" in trade_metadata:
        return trade_metadata["primary_pattern"]
    
    # Method 2: From eval_ctx
    eval_ctx = trade_metadata.get("eval_ctx")
    if eval_ctx:
        patterns = eval_ctx.get("patterns", {})
        for pattern_key, pattern_data in patterns.items():
            if pattern_data.get("found") and pattern_data.get("quality", 0) >= 7.0:
                # Extract base pattern name from key (e.g., "darvasBoxIntraday" -> "darvasBox")
                return _normalize_pattern_name(pattern_key)
    
    # Method 3: Fallback to setup type
    return trade_metadata.get("setup_type")


def _normalize_pattern_name(pattern_key: str) -> str:
    """
    Normalizes horizon-specific pattern key to base name.
    
    Examples:
        "darvasBoxIntraday" -> "darvasBox"
        "bollingerSqueezeShortTerm" -> "bollingerSqueeze"
    """
    # Remove horizon suffixes
    suffixes = ["Intraday", "ShortTerm", "LongTerm", "Multibagger"]
    
    for suffix in suffixes:
        if pattern_key.endswith(suffix):
            return pattern_key[:-len(suffix)]
    
    return pattern_key


# ============================================================
# INTEGRATION EXAMPLE: Position Monitoring Service
# ============================================================

class PatternPerformanceMonitor:
    """
    Real-time position monitoring with pattern velocity tracking.
    
    Usage:
        >>> monitor = PatternPerformanceMonitor()
        >>> 
        >>> # Add new position when trade executed
        >>> monitor.add_position(trade_plan)
        >>> 
        >>> # Update positions periodically (every candle close)
        >>> monitor.update_all(current_prices, current_bar)
        >>> 
        >>> # Remove completed positions
        >>> monitor.remove_position("RELIANCE.NS")
    """
    
    def __init__(self):
        self.active_positions: Dict[str, Dict] = {}
    
    def add_position(self, trade_plan: Dict[str, Any]) -> bool:
        """Add new position for monitoring."""
        symbol = trade_plan.get("symbol")
        
        if not symbol or not trade_plan.get("velocity_tracking_id"):
            return False
        
        self.active_positions[symbol] = trade_plan
        logger.info(f"✅ Monitoring started: {symbol}")
        return True
    
    def remove_position(self, symbol: str) -> bool:
        """Remove position from monitoring."""
        if symbol in self.active_positions:
            del self.active_positions[symbol]
            logger.info(f"🔚 Monitoring stopped: {symbol}")
            return True
        return False
    
    def update_all(
        self, 
        current_prices: Dict[str, float],
        current_bar: int
    ) -> Dict[str, Any]:
        """Update all active positions."""
        positions_list = list(self.active_positions.values())
        
        summary = update_active_positions(
            positions_list, 
            current_prices, 
            current_bar
        )
        
        # Remove completed positions
        for symbol in list(self.active_positions.keys()):
            position = self.active_positions[symbol]
            price = current_prices.get(symbol)
            
            if price:
                # Remove if stopped out or T2 hit
                if (price <= position.get("stop_loss", 0) or 
                    price >= position.get("targets", {}).get("t2", float('inf'))):
                    self.remove_position(symbol)
        
        return summary


# ============================================================
# USAGE DOCUMENTATION
# ============================================================

"""
INTEGRATION GUIDE
=================

1. ORDER EXECUTION SYSTEM
--------------------------
In your order fill handler:

```python
from services.patterns.pattern_performance_updater import on_target_hit

def on_order_filled(order):
    if order.status == "FILLED":
        trade_plan = get_trade_plan_for_order(order)
        
        # Check which target was hit
        if order.price == trade_plan["targets"]["t1"]:
            on_target_hit(
                symbol=order.symbol,
                target_level="T1",
                current_price=order.fill_price,
                current_bar=get_current_bar(),
                trade_metadata=trade_plan
            )
```

2. STOP LOSS MONITORING
------------------------
In your risk management system:

```python
from services.patterns.pattern_performance_updater import on_stop_loss_triggered

def check_stop_losses():
    for position in active_positions:
        if current_price <= position["stop_loss"]:
            on_stop_loss_triggered(
                symbol=position["symbol"],
                current_price=current_price,
                current_bar=current_bar,
                trade_metadata=position["metadata"]
            )
            
            # Close position
            close_position(position)
```

3. PATTERN INVALIDATION
------------------------
In trade_enhancer.py:

```python
from services.patterns.pattern_performance_updater import on_pattern_invalidation

# After checking pattern invalidation
invalidation = check_pattern_invalidation(...)
if invalidation["invalidated"]:
    on_pattern_invalidation(
        symbol=symbol,
        current_price=current_price,
        current_bar=current_bar,
        trade_metadata=active_trade,
        invalidation_reason=invalidation["reason"]
    )
```

4. PERIODIC BATCH UPDATE
-------------------------
In your main monitoring loop:

```python
from services.patterns.pattern_performance_updater import PatternPerformanceMonitor

monitor = PatternPerformanceMonitor()

# When new trade approved
if trade_plan["status"] == "APPROVED":
    monitor.add_position(trade_plan)

# Every candle close
def on_candle_close():
    current_prices = fetch_current_prices()
    monitor.update_all(current_prices, current_bar)
```
"""
