# services/db.py

import os
from datetime import datetime as _dt
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import logging
import time
import random
import functools
from typing import Callable, Any
from sqlalchemy.exc import OperationalError
from config.config_utility.market_utils import get_current_utc
logger = logging.getLogger(__name__)

def utc_now():
    """Returns timezone-aware UTC datetime."""
    return get_current_utc()

# 1. Setup SQLite
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_DIR}/trade.db"

# Single, robust engine definition
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 10.0, # ✅ Issue 6: Standardized 10s timeout
    },
    pool_pre_ping=True,
    pool_size=1,
    max_overflow=0,
    echo=False
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    cursor.execute("PRAGMA cache_size=-64000;")
    cursor.execute("PRAGMA temp_store=MEMORY;")
    cursor.execute("PRAGMA busy_timeout=10000;") # ✅ Issue 6: Standardized 10s timeout
    cursor.close()

# 2. Define Tables
class StockMeta(Base):
    __tablename__ = "stock_meta"
    symbol = Column(String, primary_key=True, index=True)
    is_favorite = Column(Boolean, default=False)
    last_scan_time = Column(DateTime, nullable=True)
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    marketCap = Column(Float, nullable=True)

class SignalCache(Base):
    __tablename__ = "signal_cache"
    symbol = Column(String, primary_key=True, index=True)
    best_horizon = Column(String)           # System-determined optimal (e.g., "intraday")
    selected_horizon = Column(String)       # User's choice (e.g., "multibagger")
    score = Column(Float)
    recommendation = Column(String)
    signal_text = Column(String)
    conf_score = Column(Integer)
    rr_ratio = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    direction = Column(String, nullable=True, index=True)
    horizon_scores = Column(JSON)
    
    # Timezone-aware UTC timestamp
    updated_at = Column(
        DateTime,  # ✅ No timezone=True (SQLite doesn't support it)
        default=utc_now,
        onupdate=utc_now,
        index=True
    )

# Force timezone awareness on READ
@event.listens_for(SignalCache, 'load')
def receive_load(target, context):
    """Ensure all datetime fields are timezone-aware (UTC) after loading."""
    if target.updated_at and target.updated_at.tzinfo is None:
        target.updated_at = target.updated_at.replace(tzinfo=timezone.utc)

class TradeLog(Base):
    __tablename__ = "trade_logs"
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    horizon = Column(String)
    entry_time = Column(DateTime, server_default=text("CURRENT_TIMESTAMP"))
    entry_price = Column(Float)
    stop_loss = Column(Float)
    qty = Column(Integer)
    status = Column(String, default="OPEN")
    notes = Column(Text, nullable=True)

class FundamentalCache(Base):
    __tablename__ = "fundamental_cache"
    symbol = Column(String, primary_key=True, index=True)
    data = Column(JSON) 
    updated_at = Column(
        DateTime, 
        default=utc_now,  # Use the helper function you defined
        onupdate=utc_now,
        index=True
    )

# 1. Pattern Breakdown State
class PatternBreakdownState(Base):
    """Tracks pattern breakdown progress for duration candle logic."""
    __tablename__ = "pattern_breakdown_state"
    
    symbol = Column(String, primary_key=True, index=True)
    pattern_name = Column(String, primary_key=True, index=True)
    horizon = Column(String, primary_key=True, index=True)
    
    # ✅ No timezone=True for SQLite
    started_at = Column(DateTime, nullable=False)
    last_updated = Column(DateTime, nullable=False)
    candle_count = Column(Integer, default=1)
    status = Column(String, default="active", index=True)
    resolved_at = Column(DateTime, nullable=True)
    resolution_reason = Column(String, nullable=True)
    
    price_at_breakdown = Column(Float, nullable=True)
    threshold_level = Column(Float, nullable=True)
    condition = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)

class PatternBreakdownEvent(Base):
    """Append-only audit trail for breakdown state transitions."""
    __tablename__ = "pattern_breakdown_event"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, index=True, nullable=False)
    pattern_name = Column(String, index=True, nullable=False)
    horizon = Column(String, index=True, nullable=False)
    event_type = Column(String, index=True, nullable=False)
    event_time = Column(DateTime, default=utc_now, index=True, nullable=False)
    candle_count = Column(Integer, nullable=True)
    price_at_breakdown = Column(Float, nullable=True)
    threshold_level = Column(Float, nullable=True)
    condition = Column(String, nullable=True)
    details = Column(JSON, nullable=True)

# ✅ Add event listener for timezone enforcement
@event.listens_for(PatternBreakdownState, 'load')
def enforce_timezone_on_pattern_state(target, context):
    """Ensure all datetime fields are timezone-aware (UTC) after loading."""
    if target.started_at and target.started_at.tzinfo is None:
        target.started_at = target.started_at.replace(tzinfo=timezone.utc)
    
    if target.last_updated and target.last_updated.tzinfo is None:
        target.last_updated = target.last_updated.replace(tzinfo=timezone.utc)

@event.listens_for(PatternBreakdownEvent, 'load')
def enforce_timezone_on_breakdown_event(target, context):
    """Ensure event_time stays UTC-aware on read."""
    if target.event_time and target.event_time.tzinfo is None:
        target.event_time = target.event_time.replace(tzinfo=timezone.utc)

# 2. New Pattern Performance History
class PatternPerformanceHistory(Base):
    """Tracks actual pattern performance for velocity analytics."""
    __tablename__ = 'pattern_performance_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Pattern Identity
    symbol = Column(String(20), nullable=False, index=True)
    pattern_name = Column(String(50), nullable=False, index=True)
    horizon = Column(String(20), nullable=False, index=True)
    setup_type = Column(String(50), nullable=True, index=True)
    
    # Detection Metadata
    detected_at = Column(DateTime, nullable=False, default=utc_now)
    detection_quality = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=False)
    
    # Targets & Stops
    target_1 = Column(Float, nullable=True)
    target_2 = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    
    # Actual Performance
    t1_reached = Column(Boolean, default=False)
    t2_reached = Column(Boolean, default=False)
    stopped_out = Column(Boolean, default=False)
    invalidated = Column(Boolean, default=False)
    
    # Timing Metrics (CORE DATA)
    days_to_t1 = Column(Float, nullable=True)
    days_to_t2 = Column(Float, nullable=True)
    days_to_invalidation = Column(Float, nullable=True)
    bars_to_t1 = Column(Integer, nullable=True)
    bars_to_t2 = Column(Integer, nullable=True)
    
    # Market Context
    trend_regime = Column(String(20), nullable=True)
    adx_at_entry = Column(Float, nullable=True)
    volatility_regime = Column(String(20), nullable=True)
    rr_ratio = Column(Float, nullable=True)
    
    # Pattern Metadata
    pattern_meta = Column(JSON, nullable=True)
    
    # Tracking
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)
    completed = Column(Boolean, default=False, index=True)
    
    # Exit details
    exit_price = Column(Float, nullable=True)
    exit_reason = Column(String(50), nullable=True)

@event.listens_for(PatternPerformanceHistory, 'load')
def enforce_timezone_on_performance(target, context):
    """Ensure datetime fields are timezone-aware."""
    if target.detected_at and target.detected_at.tzinfo is None:
        target.detected_at = target.detected_at.replace(tzinfo=timezone.utc)
    
    if target.created_at and target.created_at.tzinfo is None:
        target.created_at = target.created_at.replace(tzinfo=timezone.utc)
    
    if target.updated_at and target.updated_at.tzinfo is None:
        target.updated_at = target.updated_at.replace(tzinfo=timezone.utc)

# 3. Paper Trading Model
class PaperTrade(Base):
    __tablename__ = "paper_trades"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), index=True, nullable=False)
    entry_price = Column(Float, nullable=False)
    target_1 = Column(Float, nullable=True)
    target_2 = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    estimated_hold_days = Column(Integer, nullable=True)
    horizon = Column(String(50), nullable=True)    # NEW
    position_size = Column(Integer, nullable=True) # NEW
    status = Column(String(20), default="OPEN")  # OPEN, WIN, LOSS, PARTIAL
    
    created_at = Column(DateTime, default=utc_now, index=True)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

@event.listens_for(PaperTrade, 'load')
def enforce_timezone_on_paper_trade(target, context):
    """Ensure datetime fields are timezone-aware."""
    if target.created_at and target.created_at.tzinfo is None:
        target.created_at = target.created_at.replace(tzinfo=timezone.utc)
    if target.updated_at and target.updated_at.tzinfo is None:
        target.updated_at = target.updated_at.replace(tzinfo=timezone.utc)
# -----------------------------------------------------------------------
# V15.0: Schema Migrations table (Architecture Decision #2)
# Tracks which raw-SQL ALTER migrations have been applied so that
# run_migrations() is idempotent and auditable — no Alembic required.
# -----------------------------------------------------------------------
class SchemaMigration(Base):
    """One row per applied migration. migration_name is the primary key."""
    __tablename__ = "schema_migrations"
    migration_name = Column(String, primary_key=True)
    applied_at = Column(DateTime, default=utc_now, nullable=False)


def backoff_retry_db(retries: int = 5, base_delay: float = 0.1, max_delay: float = 2.0):
    """
    Decorator for database write operations with exponential backoff and jitter.
    Specifically targets SQLite "database is locked" errors.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except OperationalError as e:
                    last_exc = e
                    msg = str(e).lower()
                    if "locked" in msg or "busy" in msg:
                        if attempt == retries:
                            logger.error(f"DB Max retries ({retries}) reached: {e}")
                            raise
                        
                        delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                        jitter = random.uniform(0, 0.1 * delay)
                        final_delay = delay + jitter
                        
                        logger.warning(
                            f"DB locked/busy. Retry {attempt}/{retries} in {final_delay:.2f}s..."
                        )
                        time.sleep(final_delay)
                        continue
                    raise # Non-retryable operational error
                except Exception as e:
                    logger.error(f"Unexpected DB error: {e}")
                    raise
            if last_exc:
                raise last_exc
        return wrapper
    return decorator


# 4. Create Tables
def init_db():
    # Lazy import — avoids circular import (mb_db_model imports Base from db.py)
    from config.multibagger.mb_db_model import MultibaggerCandidate  # noqa: F401
    Base.metadata.create_all(bind=engine)
    run_migrations()


def _migration_applied(conn, name: str) -> bool:
    """Return True if this migration has already been recorded."""
    try:
        row = conn.execute(
            text("SELECT 1 FROM schema_migrations WHERE migration_name = :n"),
            {"n": name}
        ).fetchone()
        return row is not None
    except Exception:
        # schema_migrations table may not exist yet on the very first run;
        # create_all() runs before run_migrations() so this should not happen,
        # but be defensive.
        return False


def _record_migration(conn, name: str) -> None:
    """Insert a row recording that `name` has been successfully applied."""
    conn.execute(
        text("INSERT INTO schema_migrations (migration_name, applied_at) VALUES (:n, :ts)"),
        {"n": name, "ts": utc_now()}
    )


def run_migrations():
    """
    V15.0 idempotent migration runner.
    Each migration is registered by name; already-applied ones are skipped.
    C31 fix: every migration always releases its connection via engine.begin()
    context manager, and the log level is WARNING so PROD deployments see it.
    """
    registry = {
        "add_selected_horizon": migrate_add_selected_horizon,
        "add_direction_column": migrate_add_direction_column,
        "add_pattern_breakdown_lifecycle": migrate_add_pattern_breakdown_lifecycle,
    }
    for name, fn in registry.items():
        fn(name)


def migrate_add_selected_horizon(migration_name: str = "add_selected_horizon"):
    """Add selected_horizon column if it doesn't exist."""
    # C31 fix: engine.begin() is a context manager — connection is always
    # released on block exit (success or exception).
    try:
        with engine.begin() as conn:
            if _migration_applied(conn, migration_name):
                logger.debug(f"Migration already applied, skipping: {migration_name}")
                return
            result = conn.execute(text("PRAGMA table_info(signal_cache)"))
            columns = [row[1] for row in result.fetchall()]
            if "selected_horizon" not in columns:
                logger.warning(f"[MIGRATION] Running: {migration_name}")
                conn.execute(text("ALTER TABLE signal_cache ADD COLUMN selected_horizon VARCHAR"))
                conn.execute(
                    text("UPDATE signal_cache SET selected_horizon = best_horizon WHERE selected_horizon IS NULL")
                )
            _record_migration(conn, migration_name)
            logger.warning(f"[MIGRATION] ✅ Applied: {migration_name}")
    except Exception as e:
        logger.error(f"[MIGRATION] ❌ Failed: {migration_name} — {e}")
        raise


def migrate_add_direction_column(migration_name: str = "add_direction_column"):
    """Add direction column if it doesn't exist and backfill from horizon_scores JSON."""
    try:
        with engine.begin() as conn:
            if _migration_applied(conn, migration_name):
                logger.debug(f"Migration already applied, skipping: {migration_name}")
                return
            result = conn.execute(text("PRAGMA table_info(signal_cache)"))
            columns = [row[1] for row in result.fetchall()]
            if "direction" not in columns:
                logger.warning(f"[MIGRATION] Running: {migration_name}")
                conn.execute(text("ALTER TABLE signal_cache ADD COLUMN direction VARCHAR"))
                conn.execute(text("""
                    UPDATE signal_cache
                    SET direction = COALESCE(json_extract(horizon_scores, '$.direction'), 'neutral')
                    WHERE direction IS NULL
                """))
            else:
                # Backfill only NULLs even if column already exists
                conn.execute(text("""
                    UPDATE signal_cache
                    SET direction = COALESCE(direction, json_extract(horizon_scores, '$.direction'), 'neutral')
                    WHERE direction IS NULL
                """))
            _record_migration(conn, migration_name)
            logger.warning(f"[MIGRATION] ✅ Applied: {migration_name}")
    except Exception as e:
        logger.error(f"[MIGRATION] ❌ Failed: {migration_name} — {e}")
        raise


def migrate_add_pattern_breakdown_lifecycle(migration_name: str = "add_pattern_breakdown_lifecycle"):
    """Add lifecycle columns to pattern_breakdown_state for auditability."""
    try:
        with engine.begin() as conn:
            if _migration_applied(conn, migration_name):
                logger.debug(f"Migration already applied, skipping: {migration_name}")
                return
            result = conn.execute(text("PRAGMA table_info(pattern_breakdown_state)"))
            columns = [row[1] for row in result.fetchall()]
            logger.warning(f"[MIGRATION] Running: {migration_name}")
            if "status" not in columns:
                conn.execute(text("ALTER TABLE pattern_breakdown_state ADD COLUMN status VARCHAR"))
                conn.execute(
                    text("UPDATE pattern_breakdown_state SET status = 'active' WHERE status IS NULL")
                )
            if "resolved_at" not in columns:
                conn.execute(text("ALTER TABLE pattern_breakdown_state ADD COLUMN resolved_at DATETIME"))
            if "resolution_reason" not in columns:
                conn.execute(
                    text("ALTER TABLE pattern_breakdown_state ADD COLUMN resolution_reason VARCHAR")
                )
            _record_migration(conn, migration_name)
            logger.warning(f"[MIGRATION] ✅ Applied: {migration_name}")
    except Exception as e:
        logger.error(f"[MIGRATION] ❌ Failed: {migration_name} — {e}")
        raise

# 4. Helper to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    
@backoff_retry_db(retries=5)
def _write_signal_cache_with_retry(symbol: str, writer_fn: Callable[[Session, SignalCache], None]):
    """
    Retry-safe atomic writer for SignalCache.
    Encapsulates session management and backoff logic.
    """
    from services.db import SessionLocal, SignalCache
    db = SessionLocal()
    try:
        entry = db.query(SignalCache).filter_by(symbol=symbol).first()
        if not entry:
            entry = SignalCache(symbol=symbol)
            db.add(entry)
        
        writer_fn(db, entry)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB Write failed for {symbol}: {e}")
        raise
    finally:
        db.close()
