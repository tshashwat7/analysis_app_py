import os
import logging
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger("data_layer")

# Configuration
STORE_DIR = Path("data/store")
STORE_DIR.mkdir(parents=True, exist_ok=True)

class ParquetStore:
    """
    High-Performance Data Lake using Parquet.
    """
    
    @staticmethod
    def _get_path(symbol: str, interval: str) -> Path:
        clean_sym = symbol.replace(".", "_").replace("^", "")
        # Folder per symbol allows adding more files later (features, info)
        sym_dir = STORE_DIR / clean_sym
        sym_dir.mkdir(exist_ok=True)
        return sym_dir / f"{interval}.parquet"

    @staticmethod
    def save_ohlcv(symbol: str, df: pd.DataFrame, interval: str):
        """
        Writes Pandas DataFrame to Parquet via Polars engine (Fast).
        """
        if df is None or df.empty: return
        
        try:
            path = ParquetStore._get_path(symbol, interval)
            
            # Ensure index is reset for Parquet (Time column needs to be explicit)
            # We convert the Pandas Index to a Column named 'Date' or 'Datetime'
            reset_df = df.reset_index()
            
            # Convert to Polars for optimized write
            pl_df = pl.from_pandas(reset_df)
            pl_df.write_parquet(path)
            
            # logger.debug(f"[{symbol}] Saved parquet to {path}")
        except Exception as e:
            logger.error(f"[{symbol}] Parquet Write Error: {e}")

    @staticmethod
    def load_ohlcv(symbol: str, interval: str, max_age_minutes: int = None) -> pd.DataFrame:
        """
        Reads Parquet -> Returns Pandas DataFrame (indexed by Date).
        Returns None if file doesn't exist or is too old.
        """
        path = ParquetStore._get_path(symbol, interval)
        if not path.exists():
            return None
            
        # 1. Stale Check
        if max_age_minutes:
            mtime = path.stat().st_mtime
            age_mins = (datetime.now().timestamp() - mtime) / 60
            if age_mins > max_age_minutes:
                # logger.debug(f"[{symbol}] Cache Stale (Age: {age_mins:.0f}m)")
                return None

        # 2. Fast Read
        try:
            # Scan -> Collect is efficient
            pl_df = pl.read_parquet(path)
            
            # Convert back to Pandas for your existing indicator engine
            pdf = pl_df.to_pandas()
            
            # Re-set Index (Handle both Date and Datetime keys)
            if "Date" in pdf.columns:
                pdf.set_index("Date", inplace=True)
            elif "Datetime" in pdf.columns:
                pdf.set_index("Datetime", inplace=True)
            elif "index" in pdf.columns:
                pdf.set_index("index", inplace=True)
                
            return pdf
        except Exception as e:
            logger.error(f"[{symbol}] Parquet Read Error: {e}")
            return None