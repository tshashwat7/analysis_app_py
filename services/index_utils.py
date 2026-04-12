import os
import json
import logging
from typing import List, Tuple, Dict, Optional

# Late import to avoid circularity if data_fetch imports index_utils
# from services.data_fetch import parse_index_csv

logger = logging.getLogger(__name__)

DATA_DIR = "data"
STOCK_TO_INDEX_MAP: Dict[str, str] = {}

NSE_SECTOR_MAP: Dict[str, str] = {
    "information technology": "^CNXIT.NS",
    "technology": "^CNXIT.NS",
    "financial services": "^NSEBANK",
    "banks": "^NSEBANK",
    "banking": "^NSEBANK",
    "automobile and auto components": "^CNXAUTO.NS",
    "auto": "^CNXAUTO.NS",
    "automotive": "^CNXAUTO.NS",
    "healthcare": "^CNXPHARMA.NS",
    "pharmaceuticals": "^CNXPHARMA.NS",
    "pharma": "^CNXPHARMA.NS",
    "consumer goods": "^CNXFMCG.NS",
    "fast moving consumer goods": "^CNXFMCG.NS",
    "fmcg": "^CNXFMCG.NS",
    "realty": "^CNXREALTY.NS",
    "real estate": "^CNXREALTY.NS",
    "infrastructure": "^CNXINFRA.NS",
    "capital goods": "^CNXINFRA.NS",
    "metals & mining": "^CNXMETAL.NS",
    "metals and mining": "^CNXMETAL.NS",
    "metals": "^CNXMETAL.NS",
}


def _normalize_sector_name(sector: str) -> str:
    return str(sector or "").strip().lower()


def get_sector_benchmark_symbol(sector: Optional[str]) -> Optional[str]:
    normalized = _normalize_sector_name(sector)
    if not normalized:
        return None

    if normalized in NSE_SECTOR_MAP:
        return NSE_SECTOR_MAP[normalized]

    if "financial" in normalized or "bank" in normalized:
        return "^NSEBANK"
    if "tech" in normalized or "software" in normalized or normalized == "it":
        return "^CNXIT.NS"
    if "auto" in normalized:
        return "^CNXAUTO.NS"
    if "pharma" in normalized or "health" in normalized:
        return "^CNXPHARMA.NS"
    if "fmcg" in normalized or "consumer" in normalized:
        return "^CNXFMCG.NS"
    if "real" in normalized:
        return "^CNXREALTY.NS"
    if "infra" in normalized or "capital goods" in normalized:
        return "^CNXINFRA.NS"
    if "metal" in normalized or "mining" in normalized:
        return "^CNXMETAL.NS"

    return None

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
