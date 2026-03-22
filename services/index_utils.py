import os
import json
import logging
from typing import List, Tuple, Dict

# Late import to avoid circularity if data_fetch imports index_utils
# from services.data_fetch import parse_index_csv

logger = logging.getLogger(__name__)

DATA_DIR = "data"
STOCK_TO_INDEX_MAP: Dict[str, str] = {}

def get_cached_stocks(index_file: str) -> List[Tuple[str, str]]:
    """Helper to load stocks from a JSON file."""
    if os.path.exists(index_file):
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                out = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "symbol" in item:
                            out.append((item["symbol"], item.get("name", item["symbol"])))
                        elif isinstance(item, (list, tuple)) and len(item) >= 2:
                            out.append((item[0], item[1]))
                return out
        except Exception:
            return []
    return []

def load_or_create_index(index_name: str):
    """
    Loads stock list for an index. Resolves P1-7 by moving this to 
    a shared utility to break circularity with main.py.
    """
    from services.data_fetch import parse_index_csv
    
    json_file = os.path.join(DATA_DIR, f"{index_name}.json")
    csv_file = os.path.join(DATA_DIR, f"{index_name}.csv")

    # If JSON exists, load it
    if os.path.exists(json_file):
        stocks = get_cached_stocks(json_file)
        if stocks: return stocks
    
    # If JSON missing but CSV exists → build JSON
    if os.path.exists(csv_file):
        logger.info(f"Parsing CSV for index: {index_name}")
        pairs = parse_index_csv(csv_file)
        if pairs:
            json_data = [{"symbol": s, "name": n} for s, n in pairs]
            try:
                os.makedirs(os.path.dirname(json_file), exist_ok=True)
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2)
            except Exception: pass
            return pairs
    return []

def load_or_create_global_stocks():
    """Loads the main NSEStock universe."""
    return load_or_create_index("NSEStock")

def build_smart_index_map():
    """Builds a map of Ticker -> Primary Index for benchmarking."""
    global STOCK_TO_INDEX_MAP
    priority_files = [
        "NSEStock", "niftyauto", "niftybank", "niftyfmcg", "niftyinfra", 
        "niftyit", "niftypharma", "niftyrealty", "nifty500", "smallcap250", 
        "microcap250", "smallcap100", "midcap150", "niftynext50", "nifty100", "nifty50"
    ]
    for filename in priority_files:
        filepath = os.path.join(DATA_DIR, f"{filename}.json")
        if os.path.exists(filepath):
            stocks = get_cached_stocks(filepath)
            for symbol, _ in stocks:
                STOCK_TO_INDEX_MAP[symbol.strip().upper()] = filename
    logger.info(f"[INIT] Smart Index Map built for {len(STOCK_TO_INDEX_MAP)} symbols.")
