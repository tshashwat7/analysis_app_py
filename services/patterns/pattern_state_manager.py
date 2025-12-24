# services/pattern_state_manager.py
"""
Pattern State Manager - Duration Candle Tracking
================================================

Tracks pattern breakdown duration using SQLite.
Required for multi-candle confirmation in pattern invalidation.

Example:
    - Pattern breaks support on Day 1 → State saved
    - Pattern still broken on Day 2 → Count incremented
    - If duration_candles=2 met → Invalidate
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from sqlalchemy import Column, String, Integer, DateTime, Float, JSON
from services.db import PatternBreakdownState, SessionLocal
from config.market_utils import get_current_utc

logger = logging.getLogger(__name__)


# ============================================================
# STATE MANAGEMENT FUNCTIONS
# ============================================================

def get_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieves active breakdown state for a pattern.
    
    Returns:
        {
            "candle_count": int,
            "started_at": datetime,
            "price_at_breakdown": float,
            "threshold_level": float
        }
        or None if no active breakdown
    """
    try:
        db = SessionLocal()
        symbol_str = symbol.get('value') or ""
        
        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol == symbol_str,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon == horizon
        ).first()
        
        if not state:
            return None
        
        return {
            "candle_count": state.candle_count,
            "started_at": state.started_at,
            "price_at_breakdown": state.price_at_breakdown,
            "threshold_level": state.threshold_level,
            "condition": state.condition
        }
    
    except Exception as e:
        logger.error(f"get_breakdown_state failed: {e}", exc_info=True)
        return None
    
    finally:
        db.close()


def save_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str,
    price: float,
    threshold: float,
    condition: str
) -> bool:
    """
    Saves initial breakdown state (Day 1).
    """
    try:
        db = SessionLocal()
        now = get_current_utc()
        
        # Upsert logic
        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon == horizon
        ).first()
        
        if state:
            # Already exists → increment count
            state.candle_count += 1
            state.last_updated = now
        else:
            # Create new
            state = PatternBreakdownState(
                symbol=symbol,
                pattern_name=pattern_name,
                horizon=horizon,
                started_at=now,
                last_updated=now,
                candle_count=1,
                price_at_breakdown=price,
                threshold_level=threshold,
                condition=condition
            )
            db.add(state)
        
        db.commit()
        logger.debug(f"Breakdown state saved: {pattern_name} on {symbol} (count={state.candle_count})")
        return True
    
    except Exception as e:
        logger.error(f"save_breakdown_state failed: {e}", exc_info=True)
        db.rollback()
        return False
    
    finally:
        db.close()


def update_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str
) -> Optional[int]:
    """
    Increments candle count for active breakdown.
    
    Returns:
        Updated candle_count or None
    """
    try:
        db = SessionLocal()
        now = get_current_utc()
        
        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon == horizon
        ).first()
        
        if not state:
            return None
        
        state.candle_count += 1
        state.last_updated = now
        db.commit()
        
        logger.debug(f"Breakdown state updated: {pattern_name} on {symbol} (count={state.candle_count})")
        return state.candle_count
    
    except Exception as e:
        logger.error(f"update_breakdown_state failed: {e}", exc_info=True)
        db.rollback()
        return None
    
    finally:
        db.close()


def delete_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str
) -> bool:
    """
    Deletes breakdown state (after invalidation confirmed or recovery).
    """
    try:
        db = SessionLocal()
        
        db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon == horizon
        ).delete()
        
        db.commit()
        logger.debug(f"Breakdown state deleted: {pattern_name} on {symbol}")
        return True
    
    except Exception as e:
        logger.error(f"delete_breakdown_state failed: {e}", exc_info=True)
        db.rollback()
        return False
    
    finally:
        db.close()


# services/pattern_state_manager.py



def cleanup_old_breakdown_states(days_old: int = 7) -> int:
    """
    Cleans up stale breakdown states older than N days.
    
    Returns:
        Number of records deleted
    """
    try:
        db = SessionLocal()
        
        # ✅ Direct datetime comparison (cleaner)
        cutoff = get_current_utc() - timedelta(days=days_old)
        
        count = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.last_updated < cutoff
        ).delete()
        
        db.commit()
        
        if count > 0:
            logger.info(f"✅ Cleaned up {count} old breakdown states (older than {days_old} days)")
        else:
            logger.debug(f"ℹ️  No breakdown states older than {days_old} days")
        
        return count
    
    except Exception as e:
        logger.error(f"❌ cleanup_old_breakdown_states failed: {e}", exc_info=True)
        db.rollback()
        return 0
    
    finally:
        db.close()
