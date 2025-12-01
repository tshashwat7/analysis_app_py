import os
import logging
import time
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime, timezone

logger = logging.getLogger("data_layer")

# Configuration
STORE_DIR = Path("data/store")
STORE_DIR.mkdir(parents=True, exist_ok=True)

class ParquetStore:
    """
    High-Performance Data Lake using Parquet with Atomic Writes.
    """
    
    @staticmethod
    def _get_path(symbol: str, interval: str) -> Path:
        clean_sym = symbol.replace(".", "_").replace("^", "")
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
            # Improvement A: Safer temp path creation
            temp_path = Path(str(path) + ".tmp")
            
            # 1. Prepare DataFrame
            df_to_save = df.copy()
            df_to_save.index.name = "Date"
            reset_df = df_to_save.reset_index()
            
            # 2. Write to TEMP file (ZSTD compression)
            pl_df = pl.from_pandas(reset_df)
            pl_df.write_parquet(temp_path, compression="zstd")
            
            # 3. Atomic rename
            os.replace(temp_path, path)
            
            # logger.debug(f"[{symbol}] Saved {len(df)} rows to {path}")
            
        except Exception as e:
            logger.error(f"[{symbol}] Parquet Write Error: {e}")
            if 'temp_path' in locals() and temp_path.exists():
                try: temp_path.unlink()
                except: pass

    @staticmethod
    def load_ohlcv(symbol: str, interval: str, max_age_minutes: int = None, lookback_days: int = None) -> pd.DataFrame:
        """
        Reads Parquet -> Returns Pandas DataFrame.
        OPTIMIZED: Uses Polars LazyFrame to only load the necessary tail of the data.
        """
        path = ParquetStore._get_path(symbol, interval)
        
        if not path.exists(): 
            return None
            
        # 1. Stale Check
        if max_age_minutes:
            try:
                mtime = path.stat().st_mtime
                # Use UTC for freshness comparison
                age_mins = (datetime.now(timezone.utc).timestamp() - mtime) / 60
                if age_mins > max_age_minutes:
                    return None
            except FileNotFoundError:
                return None

        # 2. Read with Retry
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                lazy_df = pl.scan_parquet(path)
                
                if lookback_days:
                    multiplier = 400 if "m" in interval else 2 
                    row_limit = int(lookback_days * multiplier * 1.2)
                    lazy_df = lazy_df.tail(row_limit)

                pdf = lazy_df.collect().to_pandas()
                
                # Robust Index Restoration
                if "Date" in pdf.columns: pdf.set_index("Date", inplace=True)
                elif "Datetime" in pdf.columns: pdf.set_index("Datetime", inplace=True)
                elif "index" in pdf.columns: pdf.set_index("index", inplace=True)
                
                # Improvement B: Enforce UTC Timezone on Index
                if not isinstance(pdf.index, pd.DatetimeIndex):
                    pdf.index = pd.to_datetime(pdf.index)
                
                # Standardize to UTC to match datetime.now(timezone.utc)
                if pdf.index.tz is None:
                    pdf.index = pdf.index.tz_localize("UTC")
                else:
                    pdf.index = pdf.index.tz_convert("UTC")

                return pdf
                
            except Exception as e:
                error_msg = str(e)
                if "ArrowInvalid" in error_msg or "Parquet magic bytes" in error_msg:
                    logger.warning(f"[{symbol}] Corrupted parquet, deleting: {path}")
                    try: path.unlink(missing_ok=True)
                    except: pass
                    return None
                
                if attempt < max_retries - 1 and ("PermissionError" in error_msg or "locked" in error_msg.lower()):
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                
                return None
        
        return None