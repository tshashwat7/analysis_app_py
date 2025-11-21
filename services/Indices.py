# services/indices.py
import json, os, pandas as pd

INDEX_MAP = {
    "nifty50": "https://query1.finance.yahoo.com/v1/finance/quoteType/NSEI/components",
    "nifty500": "https://query1.finance.yahoo.com/v1/finance/quoteType/NSE500/components"
}

def load_index_from_yahoo(index_name: str):
    """Fetch tickers from Yahoo Finance or fallback JSON."""
    index_name = index_name.lower()
    fallback_file = f"data_cache/{index_name}.json"
    try:
        if index_name == "nifty50":
            data = pd.read_json("https://niftyindices.com/Backpage.aspx/getNifty50StockList")
        elif index_name == "nifty500":
            data = pd.read_json("https://niftyindices.com/Backpage.aspx/getNifty500StockList")
        else:
            raise ValueError("Unsupported index")

        tickers = sorted(list(set(data['Symbol'] + '.NS')))
        save_index(index_name, tickers)
        return tickers
    except Exception as e:
        print(f"⚠️ {index_name} fetch failed: {e}")
        if os.path.exists(fallback_file):
            with open(fallback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

def save_index(index_name: str, tickers):
    os.makedirs("data_cache", exist_ok=True)
    with open(f"data_cache/{index_name}.json", "w", encoding="utf-8") as f:
        json.dump(tickers, f, indent=2)

def list_cached_indices():
    os.makedirs("data_cache", exist_ok=True)
    return [f.replace(".json", "") for f in os.listdir("data_cache") if f.endswith(".json")]
