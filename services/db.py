import os
from datetime import datetime
from sqlalchemy import create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Setup SQLite
DB_DIR = "data"
os.makedirs(DB_DIR, exist_ok=True)
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_DIR}/trade.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. Define Tables

class StockMeta(Base):
    """
    Master list of stocks. Tracks favorites and last update times.
    """
    __tablename__ = "stock_meta"
    
    symbol = Column(String, primary_key=True, index=True)
    is_favorite = Column(Boolean, default=False)
    last_scan_time = Column(DateTime, nullable=True)
    
    # Basic cache to avoid re-fetching info every time
    sector = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    market_cap = Column(Float, nullable=True)

class SignalCache(Base):
    """
    Stores the latest analysis result for the Grid View.
    Replaces the in-memory 'SCORE_CACHE'.
    """
    __tablename__ = "signal_cache"
    
    symbol = Column(String, primary_key=True, index=True)
    
    # Scores & Recs (Best Fit)
    score = Column(Float)
    recommendation = Column(String)  # BUY/HOLD/SELL
    best_horizon = Column(String)    # intraday/short_term...
    
    # Visuals
    signal_text = Column(String)     # "🚀 SQUEEZE"
    conf_score = Column(Integer)
    
    # Phase 1 Data
    rr_ratio = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    
    # Detailed breakdown (Stored as JSON string)
    horizon_scores = Column(JSON)    # {"intra": 7.5, "long": 4.0}
    updated_at = Column(DateTime, default=datetime.now)

class TradeLog(Base):
    """
    Tracks trades you marked as 'Active'.
    """
    __tablename__ = "trade_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    horizon = Column(String)         # Which profile triggered this?
    entry_time = Column(DateTime, default=datetime.now)
    
    entry_price = Column(Float)
    stop_loss = Column(Float)
    qty = Column(Integer)
    
    status = Column(String, default="OPEN") # OPEN, CLOSED
    notes = Column(Text, nullable=True)

# 3. Create Tables
def init_db():
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
    print("✅ Database initialized at data/trade.db")