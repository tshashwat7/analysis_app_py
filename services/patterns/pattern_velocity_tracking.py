# services/patterns/pattern_velocity_tracker.py
"""
Pattern Velocity Tracking System
=================================
Tracks actual pattern performance to improve timeline estimates.

✅ Streamlined for signal_engine.py integration
✅ Removed duplicate helpers (use existing from signal_engine)
✅ Focus on core tracking & analytics functions
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple
from sqlalchemy import Column, String, Integer, DateTime, Float, JSON, Boolean, event, func
from sqlalchemy.orm import Session
from services.data_fetch import _safe_float
from services.db import PatternPerformanceHistory, SessionLocal, Base, engine, utc_now
from config.config_helpers.market_utils import get_current_utc

logger = logging.getLogger(__name__)

# ============================================================
# CORE TRACKING FUNCTIONS
# ============================================================

def track_pattern_entry(
    symbol: str,
    pattern_name: str,
    horizon: str,
    setup_type: str,
    entry_price: float,
    target_1: float,
    target_2: float,
    stop_loss: float,
    detection_quality: float,
    trend_regime: str,
    adx: Optional[float],
    volatility_regime: str,
    rr_ratio: float,
    pattern_meta: Dict[str, Any]
) -> Optional[int]:
    """
    Records pattern entry for velocity tracking.
    
    Returns:
        Record ID for later updates, or None if failed
    """
    try:
        db = SessionLocal()
        existing = db.query(PatternPerformanceHistory).filter(
            PatternPerformanceHistory.symbol == symbol,
            PatternPerformanceHistory.pattern_name == pattern_name,
            PatternPerformanceHistory.horizon == horizon,
            PatternPerformanceHistory.completed == False
        ).first()

        if existing:
            # Optional: Update specific fields if needed
            return existing.id
        
        record = PatternPerformanceHistory(
            symbol=symbol,
            pattern_name=pattern_name,
            horizon=horizon,
            setup_type=setup_type,
            detected_at=get_current_utc(),
            detection_quality=_safe_float(detection_quality),
            entry_price=_safe_float(entry_price),
            target_1=_safe_float(target_1) if target_1 else None,
            target_2=_safe_float(target_2) if target_2 else None,
            stop_loss=_safe_float(stop_loss) if stop_loss else None,
            trend_regime=trend_regime,
            adx_at_entry=_safe_float(adx) if adx else None,
            volatility_regime=volatility_regime,
            rr_ratio=_safe_float(rr_ratio) if rr_ratio else None,
            pattern_meta=pattern_meta
        )
        
        db.add(record)
        db.commit()
        db.refresh(record)
        
        logger.info(
            f"✅ Pattern tracked: {pattern_name} on {symbol}/{horizon} "
            f"(ID: {record.id}, Quality: {detection_quality:.1f})"
        )
        
        return record.id
    
    except Exception as e:
        logger.error(f"❌ track_pattern_entry failed: {e}", exc_info=True)
        db.rollback()
        return None
    
    finally:
        db.close()


def update_pattern_performance(
    record_id: Optional[int] = None,
    symbol: str = None,
    pattern_name: str = None,
    horizon: str = None,
    current_price: float = None,
    current_bar: int = 0,
    t1_reached: bool = None,
    t2_reached: bool = None,
    stopped_out: bool = None,
    invalidated: bool = None,
    exit_reason: str = None
) -> bool:
    """
    Updates pattern performance as targets are hit.
    
    ✅ Uses symbol/pattern/horizon lookup (no need to store record_id).
    """
    try:
        db = SessionLocal()
        record = None
        
        # Find the most recent incomplete record
        if record_id:
            record = db.query(PatternPerformanceHistory).filter(
                PatternPerformanceHistory.id == record_id,
                PatternPerformanceHistory.completed == False
            ).first()
        else:
            record = db.query(PatternPerformanceHistory).filter(
                PatternPerformanceHistory.symbol == symbol,
                PatternPerformanceHistory.pattern_name == pattern_name,
                PatternPerformanceHistory.horizon == horizon,
                PatternPerformanceHistory.completed == False
            ).order_by(
                PatternPerformanceHistory.detected_at.desc()
            ).first()

        
        
        if not record:
            logger.warning(
                f"⚠️ No active performance record: {pattern_name} on {symbol}/{horizon}"
            )
            return False
        
        now = get_current_utc()
        elapsed_seconds = (now - record.detected_at).total_seconds()
        elapsed_days = elapsed_seconds / 86400

        # Calculate elapsed bars using metadata stored at entry
        entry_bar = record.pattern_meta.get('bar_index') if record.pattern_meta else None
        updated = False
        
        if t1_reached is not None and t1_reached:
            if not record.t1_reached:
                record.t1_reached = True
                record.days_to_t1 = elapsed_days
                record.bars_to_t1 = (current_bar - entry_bar) if entry_bar is not None else current_bar
                updated = True
                logger.info(
                    f"✅ T1 HIT: {pattern_name} on {symbol} "
                    f"after {elapsed_days:.1f} days ({current_bar} bars)"
                )
        
        if t2_reached is not None and t2_reached:
            if not record.t2_reached:
                record.t2_reached = True
                record.days_to_t2 = elapsed_days
                record.bars_to_t2 = current_bar
                updated = True
                logger.info(f"🎯 T2 HIT: {pattern_name} on {symbol}")
        
        if stopped_out is not None and stopped_out:
            record.stopped_out = True
            record.completed = True
            record.exit_price = float(current_price) if current_price else None
            record.exit_reason = exit_reason or "STOPPED_OUT"
            updated = True
            logger.info(f"🛑 STOPPED OUT: {pattern_name} on {symbol}")
        
        if invalidated is not None and invalidated:
            record.invalidated = True
            record.days_to_invalidation = elapsed_days
            record.completed = True
            record.exit_price = current_price
            record.exit_reason = exit_reason or "INVALIDATED"
            updated = True
            logger.info(f"❌ INVALIDATED: {pattern_name} on {symbol}")
        
        if updated:
            record.updated_at = now
            db.commit()
            return True
        else:
            return False
    
    except Exception as e:
        logger.error(f"❌ update_pattern_performance failed: {e}", exc_info=True)
        db.rollback()
        return False
    
    finally:
        db.close()


# ============================================================
# VELOCITY ANALYTICS
# ============================================================

def get_pattern_velocity_stats(
    pattern_name: str,
    horizon: str,
    trend_regime: Optional[str] = None,
    min_samples: int = 10,
    max_age_days: int = 180
) -> Optional[Dict[str, Any]]:
    """
    Calculates historical velocity statistics.
    
    Returns:
        Stats dict with timing metrics and success rates
    """
    try:
        db = SessionLocal()
        
        cutoff_date = get_current_utc() - timedelta(days=max_age_days)
        
        query = db.query(PatternPerformanceHistory).filter(
            PatternPerformanceHistory.pattern_name == pattern_name,
            PatternPerformanceHistory.horizon == horizon,
            PatternPerformanceHistory.completed == True,
            PatternPerformanceHistory.detected_at >= cutoff_date
        )
        
        if trend_regime:
            query = query.filter(
                PatternPerformanceHistory.trend_regime == trend_regime
            )
        
        records = query.all()
        
        if len(records) < min_samples:
            logger.debug(
                f"⚠️ Insufficient samples: {pattern_name}/{horizon} "
                f"({len(records)} < {min_samples})"
            )
            return None
        
        # Extract timing data
        t1_times_days = [r.days_to_t1 for r in records if r.t1_reached and r.days_to_t1]
        t1_times_bars = [r.bars_to_t1 for r in records if r.t1_reached and r.bars_to_t1]
        t2_times_days = [r.days_to_t2 for r in records if r.t2_reached and r.days_to_t2]
        
        if not t1_times_days:
            return None
        
        import numpy as np
        
        stats = {
            # T1 timing
            "avg_days_to_t1": float(np.mean(t1_times_days)),
            "median_days_to_t1": float(np.median(t1_times_days)),
            "p25_days_to_t1": float(np.percentile(t1_times_days, 25)),
            "p75_days_to_t1": float(np.percentile(t1_times_days, 75)),
            "std_days_to_t1": float(np.std(t1_times_days)),
            
            # T1 bars (for intraday)
            "avg_bars_to_t1": float(np.mean(t1_times_bars)) if t1_times_bars else None,
            "median_bars_to_t1": float(np.median(t1_times_bars)) if t1_times_bars else None,
            
            # T2 timing
            "avg_days_to_t2": float(np.mean(t2_times_days)) if t2_times_days else None,
            "median_days_to_t2": float(np.median(t2_times_days)) if t2_times_days else None,
            
            # Success rates
            "success_rate_t1": sum(1 for r in records if r.t1_reached) / len(records),
            "success_rate_t2": sum(1 for r in records if r.t2_reached) / len(records),
            "failure_rate": sum(1 for r in records if r.invalidated) / len(records),
            
            # Metadata
            "sample_size": len(records),
            "pattern": pattern_name,
            "horizon": horizon,
            "trend_regime": trend_regime
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"❌ get_pattern_velocity_stats failed: {e}", exc_info=True)
        return None
    
    finally:
        db.close()


def get_historical_velocity_multiplier(
    pattern_name: str,
    horizon: str,
    trend_regime: str,
    base_estimate_days: float
) -> Tuple[Optional[float], Optional[str]]:
    """
    Gets data-driven velocity multiplier.
    
    Returns:
        (multiplier, data_source)
    """
    stats = get_pattern_velocity_stats(pattern_name, horizon, trend_regime)
    
    if not stats or stats["sample_size"] < 5:
        return None, None
    
    actual_days = stats["median_days_to_t1"]
    
    if base_estimate_days > 0:
        multiplier = actual_days / base_estimate_days
    else:
        multiplier = 1.0
    
    logger.debug(
        f"📊 Historical velocity: {pattern_name}/{horizon}/{trend_regime}: "
        f"{actual_days:.1f} days → {multiplier:.2f}x (n={stats['sample_size']})"
    )
    
    return multiplier, "historical_data"


def get_pattern_confidence_adjustment(
    pattern_name: str,
    horizon: str,
    trend_regime: str
) -> Tuple[float, str]:
    """
    Adjusts confidence ranges based on historical success rates.
    
    Returns:
        (range_multiplier, confidence_level)
    """
    stats = get_pattern_velocity_stats(pattern_name, horizon, trend_regime)
    
    if not stats:
        return 1.0, "medium"
    
    success_rate = stats["success_rate_t1"]
    failure_rate = stats["failure_rate"]
    
    if success_rate >= 0.75 and failure_rate <= 0.15:
        return 0.8, "high"
    elif success_rate <= 0.50 or failure_rate >= 0.30:
        return 1.3, "low"
    else:
        return 1.0, "medium"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def should_track_pattern(
    pattern_data: Dict[str, Any],
    min_quality: float = 7.0
) -> bool:
    """
    Determines if pattern should be tracked.
    """
    if not pattern_data.get("found"):
        return False
    
    quality = pattern_data.get("quality", 0)
    if quality < min_quality:
        return False
    
    meta = pattern_data.get("raw", {}).get("meta", {})
    velocity_tracking = meta.get("velocity_tracking", {})
    
    if not velocity_tracking.get("can_track", True):
        return False
    
    if not velocity_tracking.get("entry_conditions_met", True):
        return False
    
    return True


def classify_volatility(atr_pct: Optional[float]) -> str:
    """Classify volatility regime."""
    if atr_pct is None:
        return "unknown"
    
    if atr_pct > 4.0:
        return "high"
    elif atr_pct > 2.0:
        return "normal"
    else:
        return "low"


# ============================================================
# MAINTENANCE
# ============================================================

def cleanup_old_performance_records(days_old: int = 365) -> int:
    """Archives old performance records."""
    try:
        db = SessionLocal()
        
        cutoff = get_current_utc() - timedelta(days=days_old)
        
        count = db.query(PatternPerformanceHistory).filter(
            PatternPerformanceHistory.detected_at < cutoff,
            PatternPerformanceHistory.completed == True
        ).delete()
        
        db.commit()
        
        if count > 0:
            logger.info(f"✅ Archived {count} old records (>{days_old} days)")
        
        return count
    
    except Exception as e:
        logger.error(f"❌ cleanup failed: {e}", exc_info=True)
        db.rollback()
        return 0
    
    finally:
        db.close()
