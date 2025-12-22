# services/db.py

import os
from datetime import datetime, timezone
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import text

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
    market_cap = Column(Float, nullable=True)

class SignalCache(Base):
    __tablename__ = "signal_cache"
    symbol = Column(String, primary_key=True, index=True)
    score = Column(Float)
    recommendation = Column(String)
    best_horizon = Column(String)
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

# services/db.py

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


# 3. Create Tables
def init_db():
    # [FIX] Use Base.metadata directly. No invalid import.
    Base.metadata.create_all(bind=engine)

# 4. Helper to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()