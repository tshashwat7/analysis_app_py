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
            # 1. Force Index Name Standard
            df_to_save = df.copy()
            df_to_save.index.name = "Date"  # Enforce standard name            
            
            # Ensure index is reset for Parquet (Time column needs to be explicit)
            # We convert the Pandas Index to a Column named 'Date' or 'Datetime'
            reset_df = df_to_save.reset_index()
            
            # Convert to Polars for optimized write
            pl_df = pl.from_pandas(reset_df)
            pl_df.write_parquet(path)
            
            # logger.debug(f"[{symbol}] Saved parquet to {path}")
        except Exception as e:
            logger.error(f"[{symbol}] Parquet Write Error: {e}")

    @staticmethod
    def load_ohlcv(symbol: str, interval: str, max_age_minutes: int = None, lookback_days: int = None) -> pd.DataFrame:
        """
        Reads Parquet -> Returns Pandas DataFrame.
        OPTIMIZED: Uses Polars LazyFrame to only load the necessary tail of the data.
        """
        path = ParquetStore._get_path(symbol, interval)
        if not path.exists(): return None
            
        # 1. Stale Check
        if max_age_minutes:
            try:
                mtime = path.stat().st_mtime
                age_mins = (datetime.now().timestamp() - mtime) / 60
                if age_mins > max_age_minutes:
                    return None
            except FileNotFoundError:
                return None

        # 2. Optimized Lazy Read
        try:
            lazy_df = pl.scan_parquet(path)
            
            if lookback_days:
                # Optimized multiplier logic
                multiplier = 400 if "m" in interval else 2 
                row_limit = int(lookback_days * multiplier * 1.2)
                lazy_df = lazy_df.tail(row_limit)

            pdf = lazy_df.collect().to_pandas()
            
            # Robust Index Restoration
            if "Date" in pdf.columns:
                pdf.set_index("Date", inplace=True)
            elif "Datetime" in pdf.columns:
                pdf.set_index("Datetime", inplace=True)
            elif "index" in pdf.columns: # Fallback for legacy files
                pdf.set_index("index", inplace=True)
            
            # Ensure DatetimeIndex (Crucial for Pandas-TA)
            if not isinstance(pdf.index, pd.DatetimeIndex):
                pdf.index = pd.to_datetime(pdf.index)

            return pdf
        except Exception as e:
            logger.error(f"[{symbol}] Parquet Read Error: {e}")
            return None