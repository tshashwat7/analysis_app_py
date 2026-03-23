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
from services.db import PatternBreakdownState, PatternBreakdownEvent, SessionLocal
from config.config_utility.market_utils import get_current_utc
from services.patterns.horizon_constants import HORIZON_WINDOWS_SECONDS, HORIZON_EXPIRY_DAYS

logger = logging.getLogger(__name__)


# ✅ P2-1: Capacity gate for per-symbol-horizon tracking
MAX_ACTIVE_STATES_PER_SYMBOL = 20


def _log_breakdown_event(
    db: Session,
    symbol: str,
    pattern_name: str,
    horizon: str,
    event_type: str,
    candle_count: Optional[int] = None,
    price_at_breakdown: Optional[float] = None,
    threshold_level: Optional[float] = None,
    condition: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    db.add(
        PatternBreakdownEvent(
            symbol=symbol,
            pattern_name=pattern_name,
            horizon=horizon,
            event_type=event_type,
            event_time=get_current_utc(),
            candle_count=candle_count,
            price_at_breakdown=price_at_breakdown,
            threshold_level=threshold_level,
            condition=condition,
            details=details or {},
        )
    )

@contextmanager
def _managed_db_session():
    """Private context manager for sharing a DB session across multiple pattern operations.

    W56 fix: renamed from get_db to _managed_db_session to avoid shadowing
    services.db:get_db which is the canonical FastAPI dependency generator.
    Use this only within pattern_state_manager for multi-operation atomic batches.
    """
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
            PatternBreakdownState.status       == "active",
        ).first()

        if not state:
            return None

        return {
            "candle_count":       state.candle_count,
            "started_at":         state.started_at,
            "price_at_breakdown": state.price_at_breakdown,
            "threshold_level":    state.threshold_level,
            "condition":          state.condition,
            "status":             state.status,
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

        # ✅ P2-1 FIX: Capacity eviction before starting a new state
        active_count = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol == symbol_str,
            PatternBreakdownState.status == "active"
        ).count()

        if active_count >= MAX_ACTIVE_STATES_PER_SYMBOL:
            # Resolve oldest active state to make room
            oldest = db.query(PatternBreakdownState).filter(
                PatternBreakdownState.symbol == symbol_str,
                PatternBreakdownState.status == "active"
            ).order_by(PatternBreakdownState.started_at).first()
            if oldest:
                oldest.status = "expired"
                oldest.resolved_at = now
                oldest.resolution_reason = "capacity_eviction"
                db.flush()

        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol_str,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
        ).first()

        if state:
            if state.status != "active":
                state.started_at = now
                state.last_updated = now
                state.candle_count = 1
                state.status = "active"
                state.resolved_at = None
                state.resolution_reason = None
                state.price_at_breakdown = price
                state.threshold_level = threshold
                state.condition = condition
                _log_breakdown_event(
                    db,
                    symbol_str,
                    pattern_name,
                    horizon,
                    "reactivated",
                    candle_count=state.candle_count,
                    price_at_breakdown=price,
                    threshold_level=threshold,
                    condition=condition,
                    details={"previous_status": state.status},
                )
            else:
                state.candle_count += 1
                state.last_updated  = now
                _log_breakdown_event(
                    db,
                    symbol_str,
                    pattern_name,
                    horizon,
                    "reconfirmed",
                    candle_count=state.candle_count,
                    price_at_breakdown=state.price_at_breakdown,
                    threshold_level=state.threshold_level,
                    condition=state.condition,
                )
        else:
            state = PatternBreakdownState(
                symbol=symbol_str,
                pattern_name=pattern_name,
                horizon=horizon,
                started_at=now,
                last_updated=now,
                candle_count=1,
                status="active",
                price_at_breakdown=price,
                threshold_level=threshold,
                condition=condition,
            )
            db.add(state)
            _log_breakdown_event(
                db,
                symbol_str,
                pattern_name,
                horizon,
                "created",
                candle_count=1,
                price_at_breakdown=price,
                threshold_level=threshold,
                condition=condition,
            )

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
        if isinstance(symbol, dict): symbol = symbol.get("value")
        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
            PatternBreakdownState.status       == "active",
        ).first()

        if not state:
            return None

        state.candle_count += 1
        state.last_updated  = now
        _log_breakdown_event(
            db,
            symbol,
            pattern_name,
            horizon,
            "updated",
            candle_count=state.candle_count,
            price_at_breakdown=state.price_at_breakdown,
            threshold_level=state.threshold_level,
            condition=state.condition,
        )

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
    resolution_reason: str = "resolved",
    db: Optional[Session] = None
) -> bool:
    """Soft-resolve breakdown state for auditability."""
    _own_session = db is None
    if _own_session:
        db = SessionLocal()
    try:
        if isinstance(symbol, dict): symbol = symbol.get("value")
        
        state = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.symbol       == symbol,
            PatternBreakdownState.pattern_name == pattern_name,
            PatternBreakdownState.horizon      == horizon,
        ).first()

        if not state:
            return True

        state.status = "resolved"
        state.resolved_at = get_current_utc()
        state.resolution_reason = resolution_reason
        state.last_updated = get_current_utc()
        _log_breakdown_event(
            db,
            symbol,
            pattern_name,
            horizon,
            "resolved",
            candle_count=state.candle_count,
            price_at_breakdown=state.price_at_breakdown,
            threshold_level=state.threshold_level,
            condition=state.condition,
            details={"resolution_reason": resolution_reason},
        )

        if _own_session:
            db.commit()

        logger.debug(f"Breakdown state resolved: {pattern_name} on {symbol}")
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
    """
    Two-stage background cleanup (V15.0 hybrid retention model).

    Stage 1 – Expire stale-actives:
        Any row still "active" but not updated in `days_old` days is soft-marked
        as "expired" (so it feeds the ML audit trail).

    Stage 2 – Hard-purge old resolved rows:
        Any row with status "resolved" or "expired" whose resolved_at
        (falling back to started_at when resolved_at is NULL) is older than 90 days
        is permanently deleted per the V15.0 90-day retention SLA.

    C22/C23 fix: db is initialised to None before the try block so that the
    finally clause never raises an UnboundLocalError on SessionLocal() failure.
    """
    db = None  # C22/C23: guard against UnboundLocalError in finally
    try:
        db = SessionLocal()
        now = get_current_utc()
        
        # ✅ P1-5 FIX: Horizon-aware expiry derived from central constants
        total_expired = 0
        for horizon, days in HORIZON_EXPIRY_DAYS.items():
            cutoff_active = now - timedelta(days=days)
            
            stale_rows = db.query(PatternBreakdownState).filter(
                PatternBreakdownState.status == "active",
                PatternBreakdownState.horizon == horizon,
                PatternBreakdownState.last_updated < cutoff_active
            ).all()

            for row in stale_rows:
                row.status            = "expired"
                row.resolved_at       = now
                row.resolution_reason = "cleanup_expired"
                _log_breakdown_event(
                    db,
                    row.symbol,
                    row.pattern_name,
                    row.horizon,
                    "expired",
                    candle_count=row.candle_count,
                    price_at_breakdown=row.price_at_breakdown,
                    threshold_level=row.threshold_level,
                    condition=row.condition,
                    details={"cutoff_days": days},
                )
            total_expired += len(stale_rows)
            db.flush()

        cutoff_purge  = now - timedelta(days=90)

        # ---------------------------------------------------------------
        # Stage 2: Hard-purge rows past the 90-day retention window
        #
        # Use COALESCE(resolved_at, started_at) so rows where resolved_at
        # was never written (legacy data) are still purged once started_at
        # crosses the 90-day boundary.
        # ---------------------------------------------------------------
        purge_count = db.query(PatternBreakdownState).filter(
            PatternBreakdownState.status.in_(["resolved", "expired"]),
            PatternBreakdownState.resolved_at < cutoff_purge
        ).delete(synchronize_session=False)

        # Fallback purge: rows where resolved_at is NULL (legacy) but started_at
        # is beyond the 90-day window.
        purge_count += db.query(PatternBreakdownState).filter(
            PatternBreakdownState.status.in_(["resolved", "expired"]),
            PatternBreakdownState.resolved_at == None,   # noqa: E711
            PatternBreakdownState.started_at  < cutoff_purge
        ).delete(synchronize_session=False)

        db.commit()

        if total_expired > 0 or purge_count > 0:
            logger.info(
                f"Breakdown cleanup: expired {total_expired} active rows (horizon-aware), "
                f"purged {purge_count} resolved rows (>90d old)"
            )
        return purge_count

    except Exception as e:
        logger.error(f"cleanup_old_breakdown_states failed: {e}", exc_info=True)
        if db is not None:
            db.rollback()
        return 0
    finally:
        if db is not None:   # C22/C23: safe even if SessionLocal() itself raised
            db.close()
