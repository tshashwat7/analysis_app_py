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
    "financials": "^NSEBANK",
    "banks": "^NSEBANK",
    "banking": "^NSEBANK",
    "capital markets": "^NSEBANK",
    "insurance": "^NSEBANK",
    "automobile and auto components": "^CNXAUTO",
    "auto": "^CNXAUTO",
    "automotive": "^CNXAUTO",
    "healthcare": "^CNXPHARMA",
    "pharmaceuticals": "^CNXPHARMA",
    "pharma": "^CNXPHARMA",
    "consumer goods": "^CNXFMCG",
    "fast moving consumer goods": "^CNXFMCG",
    "fmcg": "^CNXFMCG",
    "realty": "^CNXREALTY",
    "real estate": "^CNXREALTY",
    "infrastructure": "^CNXINFRA",
    "capital goods": "^CNXINFRA",
    "construction": "^CNXINFRA",
    "engineering & construction": "^CNXINFRA",
    "engineering and construction": "^CNXINFRA",
    "metals & mining": "^CNXMETAL",
    "metals and mining": "^CNXMETAL",
    "metals": "^CNXMETAL",
    "construction materials": "^CNXMETAL",
    "building materials": "^CNXMETAL",
    # ── Energy (broad: oil, gas, power PSUs) ──────────────────────────────
    "energy":                               "^CNXENERGY",
    "utilities":                            "^CNXENERGY",
    "electric utilities":                   "^CNXENERGY",
    "independent power producers":          "^CNXENERGY",
    "power":                                "^CNXENERGY",
    "power generation":                     "^CNXENERGY",
    "thermal coal":                         "^CNXENERGY",
    # ── Oil & Gas (granular) ──────────────────────────────────────────────
    "oil and gas":                          "NIFTY_OIL_AND_GAS.NS",
    "oil & gas":                            "NIFTY_OIL_AND_GAS.NS",
    "oil gas & consumable fuels":           "NIFTY_OIL_AND_GAS.NS",
    "oil gas and consumable fuels":         "NIFTY_OIL_AND_GAS.NS",
    "oil gas integrated":                   "NIFTY_OIL_AND_GAS.NS",
    "oil gas refining & marketing":         "NIFTY_OIL_AND_GAS.NS",
    "oil gas refining and marketing":       "NIFTY_OIL_AND_GAS.NS",
    "oil gas e&p":                          "NIFTY_OIL_AND_GAS.NS",
    "oil gas midstream":                    "NIFTY_OIL_AND_GAS.NS",
    "oil gas drilling":                     "NIFTY_OIL_AND_GAS.NS",
    "petroleum":                            "NIFTY_OIL_AND_GAS.NS",
    "natural gas":                          "NIFTY_OIL_AND_GAS.NS",
    # ── Media & Entertainment / Telecom ───────────────────────────────────
    "media":                                "^CNXMEDIA",
    "media & entertainment":                "^CNXMEDIA",
    "media and entertainment":              "^CNXMEDIA",
    "entertainment":                        "^CNXMEDIA",
    "broadcasting":                         "^CNXMEDIA",
    "publishing":                           "^CNXMEDIA",
    "telecom":                              "^CNXMEDIA",
    "telecom services":                     "^CNXMEDIA",
    "communication services":               "^CNXMEDIA",
    "advertising agencies":                 "^CNXMEDIA",
    # ── Chemicals / Basic Materials ───────────────────────────────────────
    "chemicals":                            "^CNXMETAL",
    "specialty chemicals":                  "^CNXMETAL",
    "basic materials":                      "^CNXMETAL",
    "fertilizers & agricultural chemicals": "^CNXMETAL",
    "fertilisers":                          "^CNXMETAL",
    # ── yfinance GICS labels for existing sectors ─────────────────────────
    "consumer cyclical":                    "^CNXAUTO",
    "consumer defensive":                   "^CNXFMCG",
    "industrials":                          "^CNXINFRA",
    # ── Services ──────────────────────────────────────────────────────────
    "services":                             "^CNXSERVICE",
    "business services":                    "^CNXSERVICE",
    "commercial services":                  "^CNXSERVICE",
    # ── Consumer Durables (verify NIFTY_CONSDUR.NS on YF) ─────────────────
    "consumer durables":                    "NIFTY_CONSDUR.NS",
    "durables":                             "NIFTY_CONSDUR.NS",
    "household appliances":                 "NIFTY_CONSDUR.NS",
    # ── Consumption ───────────────────────────────────────────────────────
    "consumption":                          "^CNXCONSUM",
    "india consumption":                    "^CNXCONSUM",
    "retail":                               "^CNXCONSUM",
    # ── PSU ───────────────────────────────────────────────────────────────
    "psu":                                  "^CNXPSE",
    "pse":                                  "^CNXPSE",
    "public sector":                        "^CNXPSE",
    "public sector enterprises":            "^CNXPSE",
    # ── Defence (verify NIFTY_INDIA_DEFENCE.NS on YF) ─────────────────────
    "defence":                              "NIFTY_INDIA_DEFENCE.NS",
    "defense":                              "NIFTY_INDIA_DEFENCE.NS",
    "aerospace & defence":                  "NIFTY_INDIA_DEFENCE.NS",
    "aerospace and defence":                "NIFTY_INDIA_DEFENCE.NS",
    "aerospace defense":                    "NIFTY_INDIA_DEFENCE.NS",
}


def _normalize_sector_name(sector: str) -> str:
    return str(sector or "").strip().lower()


def get_sector_benchmark_symbol(sector: Optional[str]) -> Optional[str]:
    normalized = _normalize_sector_name(sector)
    if not normalized:
        return None

    if normalized in NSE_SECTOR_MAP:
        return NSE_SECTOR_MAP[normalized]

    # fuzzy fallbacks — specific before broad
    if "oil" in normalized or "gas" in normalized or "petroleum" in normalized:
        return "NIFTY_OIL_AND_GAS.NS"
    if "power" in normalized or "electric" in normalized or "utilities" in normalized:
        return "^CNXENERGY"
    if "energy" in normalized:
        return "^CNXENERGY"
    if "defence" in normalized or "defense" in normalized or "aerospace" in normalized:
        return "NIFTY_INDIA_DEFENCE.NS"
    if "media" in normalized or "entertainment" in normalized or "broadcast" in normalized:
        return "^CNXMEDIA"
    if "telecom" in normalized or "communication" in normalized:
        return "^CNXMEDIA"
    if "chemical" in normalized or "fertiliz" in normalized or "fertilis" in normalized:
        return "^CNXMETAL"
    if "financial" in normalized or "bank" in normalized:
        return "^NSEBANK"
    if "insurance" in normalized or "capital market" in normalized:
        return "^NSEBANK"
    if "tech" in normalized or "software" in normalized or normalized == "it":
        return "^CNXIT.NS"
    if "auto" in normalized:
        return "^CNXAUTO"
    if "pharma" in normalized or "health" in normalized or "drug" in normalized:
        return "^CNXPHARMA"
    if "durable" in normalized:
        return "NIFTY_CONSDUR.NS"
    if "fmcg" in normalized or "consumer good" in normalized or "consumer defensive" in normalized:
        return "^CNXFMCG"
    if "consumer" in normalized:
        return "^CNXCONSUM"
    if "real" in normalized:
        return "^CNXREALTY.NS"
    if "infra" in normalized or "capital good" in normalized:
        return "^CNXINFRA"
    if "metal" in normalized or "mining" in normalized or "steel" in normalized:
        return "^CNXMETAL"
    if "construction material" in normalized or "building material" in normalized or "cement" in normalized:
        return "^CNXMETAL"
    if "service" in normalized:
        return "^CNXSERVICE"
    if "psu" in normalized or "pse" in normalized or "public sector" in normalized:
        return "^CNXPSE"

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
