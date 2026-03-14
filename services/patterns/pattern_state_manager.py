# services/pattern_state_manager.py
"""
Pattern State Manager - Duration Candle Tracking (Shared DB Edition)
====================================================================

Tracks pattern breakdown duration using SQLAlchemy.
v5.0: Supports shared DB sessions for bulk scans.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional
from contextlib import contextmanager

from sqlalchemy.orm import Session
from services.db import PatternBreakdownState, SessionLocal
from config.config_utility.market_utils import get_current_utc

logger = logging.getLogger(__name__)

@contextmanager
def get_db():
    """Context manager for sharing a DB session across multiple pattern operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def get_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str,
    db: Optional[Session] = None
) -> Optional[Dict[str, Any]]:
    """Retrieves active breakdown state for a pattern."""
    _own_session = db is None
    if _own_session:
        db = SessionLocal()
    try:
        symbol_str = symbol.get("value") or "" if isinstance(symbol, dict) else symbol

        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol_str,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
        ).first()

        if not state:
            return None

        return {
            "candle_count":       state.candle_count,
            "started_at":         state.started_at,
            "price_at_breakdown": state.price_at_breakdown,
            "threshold_level":    state.threshold_level,
            "condition":          state.condition,
        }

    except Exception as e:
        logger.error(f"get_breakdown_state failed: {e}", exc_info=True)
        return None

    finally:
        if _own_session:
            db.close()

def save_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str,
    price: float,
    threshold: float,
    condition: str,
    db: Optional[Session] = None
) -> bool:
    """Saves/Updates initial breakdown state (Day 1)."""
    _own_session = db is None
    if _own_session:
        db = SessionLocal()
    try:
        now = get_current_utc()
        symbol_str = symbol.get("value") or "" if isinstance(symbol, dict) else symbol

        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol_str,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
        ).first()

        if state:
            state.candle_count += 1
            state.last_updated  = now
        else:
            state = PatternBreakdownState(
                symbol=symbol_str,
                pattern_name=pattern_name,
                horizon=horizon,
                started_at=now,
                last_updated=now,
                candle_count=1,
                price_at_breakdown=price,
                threshold_level=threshold,
                condition=condition,
            )
            db.add(state)

        if _own_session:
            db.commit()

        logger.debug(f"Breakdown state saved: {pattern_name} on {symbol_str} (count={state.candle_count})")
        return True

    except Exception as e:
        logger.error(f"save_breakdown_state failed: {e}", exc_info=True)
        if _own_session:
            db.rollback()
        return False

    finally:
        if _own_session:
            db.close()

def update_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str,
    db: Optional[Session] = None
) -> Optional[int]:
    """Increments candle count for active breakdown."""
    _own_session = db is None
    if _own_session:
        db = SessionLocal()
    try:
        now   = get_current_utc()
        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
        ).first()

        if not state:
            return None

        state.candle_count += 1
        state.last_updated  = now

        if _own_session:
            db.commit()

        logger.debug(f"Breakdown state updated: {pattern_name} on {symbol} (count={state.candle_count})")
        return state.candle_count

    except Exception as e:
        logger.error(f"update_breakdown_state failed: {e}", exc_info=True)
        if _own_session:
            db.rollback()
        return None

    finally:
        if _own_session:
            db.close()

def delete_breakdown_state(
    symbol: str,
    pattern_name: str,
    horizon: str,
    db: Optional[Session] = None
) -> bool:
    """Deletes breakdown state."""
    _own_session = db is None
    if _own_session:
        db = SessionLocal()
    try:
        db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
        ).delete()

        if _own_session:
            db.commit()

        logger.debug(f"Breakdown state deleted: {pattern_name} on {symbol}")
        return True

    except Exception as e:
        logger.error(f"delete_breakdown_state failed: {e}", exc_info=True)
        if _own_session:
            db.rollback()
        return False

    finally:
        if _own_session:
            db.close()

def cleanup_old_breakdown_states(days_old: int = 7) -> int:
    """Background cleanup job."""
    try:
        db = SessionLocal()
        cutoff = get_current_utc() - timedelta(days=days_old)
        count = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.last_updated < cutoff
        ).delete()
        db.commit()
        if count > 0:
            logger.info(f"Cleaned up {count} old breakdown states (older than {days_old} days)")
        return count
    except Exception as e:
        logger.error(f"cleanup_old_breakdown_states failed: {e}", exc_info=True)
        db.rollback()
        return 0
    finally:
        db.close()
