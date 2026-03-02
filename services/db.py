# services/db.py

import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text
import logging
logger = logging.getLogger(__name__)

def utc_now():
    """Returns timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)

# 1. Setup SQLite
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_DIR}/trade.db"

# Single, robust engine definition
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, 
    connect_args={
        "check_same_thread": False,
        "timeout": 30.0, 
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
    cursor.execute("PRAGMA busy_timeout=30000;")
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
    
    price_at_breakdown = Column(Float, nullable=True)
    threshold_level = Column(Float, nullable=True)
    condition = Column(String, nullable=True)
    meta = Column(JSON, nullable=True)

# ✅ Add event listener for timezone enforcement
@event.listens_for(PatternBreakdownState, 'load')
def enforce_timezone_on_pattern_state(target, context):
    """Ensure all datetime fields are timezone-aware (UTC) after loading."""
    if target.started_at and target.started_at.tzinfo is None:
        target.started_at = target.started_at.replace(tzinfo=timezone.utc)
    
    if target.last_updated and target.last_updated.tzinfo is None:
        target.last_updated = target.last_updated.replace(tzinfo=timezone.utc)

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

# 4. Create Tables
def init_db():
    # [FIX] Use Base.metadata directly. No invalid import.
    Base.metadata.create_all(bind=engine)
    migrate_add_selected_horizon()

def migrate_add_selected_horizon():
    """Add selected_horizon column if it doesn't exist."""
    try:
        conn = engine.connect()
        # Check if column exists
        result = conn.execute(text("PRAGMA table_info(signal_cache)"))
        columns = [row[1] for row in result.fetchall()]
        
        if "selected_horizon" not in columns:
            logger.info("Running migration: Adding selected_horizon column...")
            conn.execute(text("ALTER TABLE signal_cache ADD COLUMN selected_horizon VARCHAR"))
            conn.execute(text("UPDATE signal_cache SET selected_horizon = best_horizon WHERE selected_horizon IS NULL"))
            conn.commit()
            logger.info("✅ Migration complete: selected_horizon column added")
        else:
            logger.info("ℹ️  selected_horizon column already exists")
        
        conn.close()
    except Exception as e:
        logger.info(f"❌ Migration failed: {e}")
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
    migrate_add_selected_horizon()