# seed_backtest_data.py
import logging
from services.data_fetch import safe_history
from services.data_layer import ParquetStore

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seeder")

# 1. The stocks you want to backtest
TEST_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
    "TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "BAJFINANCE.NS"
]

# 2. Configurations
PERIOD = "5y"  # We need deep history for backtesting
INTERVALS = ["1d"] # Add "15m" if you want to backtest intraday later

def seed_data():
    print(f"🌱 Seeding Data Lake with {PERIOD} history for {len(TEST_SYMBOLS)} symbols...")
    
    for symbol in TEST_SYMBOLS:
        for interval in INTERVALS:
            try:
                logger.info(f"[{symbol}] Downloading {interval} data...")
                
                # Fetch deep history from Yahoo
                df = safe_history(symbol, period=PERIOD, interval=interval)
                
                if df is not None and not df.empty:
                    # Save to Parquet (L2 Cache)
                    ParquetStore.save_ohlcv(symbol, df, interval)
                    print(f"✅ Saved {symbol} ({interval}): {len(df)} rows")
                else:
                    print(f"❌ Failed to fetch {symbol}")
                    
            except Exception as e:
                print(f"⚠️ Error {symbol}: {e}")

    print("\n✨ Data Lake Seeded! You can now run 'python run_backtest.py'")

if __name__ == "__main__":
    seed_data()