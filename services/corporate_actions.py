# services/corporate_actions.py
import os
import re
import json
import time
import math
import yfinance as yf
import pandas as pd
import logging
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

from services.data_fetch import _fmt_date, _retry, safe_float

logger = logging.getLogger("corporate_actions")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] corporate_actions: %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------
# Simple robust name matching
# ----------------------------------------------------------------------
def simple_name_match(yahoo_name: str, equitymaster_name: str) -> bool:
    """Loose name match ignoring corporate suffixes and minor variations."""
    if not yahoo_name or not equitymaster_name:
        return False

    def normalize(name):
        name = name.lower()
        name = re.sub(r'\b(ltd|limited|plc|inc|corp|co|company|industries|technologies)\b', '', name, flags=re.I)
        return re.sub(r'\s+', ' ', name).strip()

    y, e = normalize(yahoo_name), normalize(equitymaster_name)
    if not y or not e:
        return False

    # Strong overlap on key tokens
    y_tokens, e_tokens = set(y.split()), set(e.split())
    return bool(y in e or e in y or len(y_tokens & e_tokens) > 0)


# ----------------------------------------------------------------------
# Cache + Equitymaster fetch
# ----------------------------------------------------------------------
CACHE_PATH = "cache/equitymaster_actions.json"
CACHE_TTL_HOURS = 24


def _fetch_action(action_type: str, url: str, headers: dict) -> List[Dict[str, Any]]:
    """Fetch one corporate action type from Equitymaster API."""
    actions = []
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if not r.ok:
            logger.warning(f"Equitymaster {action_type} fetch failed {r.status_code}")
            return actions

        data_json = r.json()
        rows = data_json.get("aaData", [])
        for row in rows:
            if len(row) < 4:
                continue

            name = row[0].strip()
            value, ex_date = None, None

            try:
                if action_type == "Dividend":
                    match = re.search(r"([\d.]+)", str(row[2]))
                    value = round(float(match.group(1)), 2) if match else None
                else:
                    value = row[2].strip()
            except Exception:
                logger.debug(f"Failed to parse value for {name} ({action_type}): {row[2]}")

            try:
                ex_date = datetime.strptime(row[3].strip(), "%d-%b-%Y").date()
            except Exception:
                logger.debug(f"Failed to parse ex_date for {name} ({action_type}): {row[3]}")
                continue

            if ex_date:
                actions.append({
                    "name": name,
                    "type": action_type,
                    "value": value,
                    "ex_date": ex_date
                })

    except Exception as e:
        logger.warning(f"Equitymaster {action_type} fetch failed: {e}")
    return actions


def _fetch_equitymaster_data() -> List[Dict[str, Any]]:
    """Fetch Equitymaster data with caching and fallback."""
    # Load cache first
    try:
        if os.path.exists(CACHE_PATH):
            mtime = datetime.fromtimestamp(os.path.getmtime(CACHE_PATH))
            if datetime.now() - mtime < timedelta(hours=CACHE_TTL_HOURS):
                with open(CACHE_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                for item in cached:
                    if isinstance(item.get("ex_date"), str):
                        item["ex_date"] = datetime.strptime(item["ex_date"], "%Y-%m-%d").date()
                logger.info(f"Loaded {len(cached)} records from cache")
                return cached
    except Exception as e:
        logger.warning("Cache load failed: %s", e)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.equitymaster.com/",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
    }

    api_endpoints = {
        "Dividend": "https://www.equitymaster.com/eqtmapi/getDividendList?indexcode=1-71",
        "Bonus": "https://www.equitymaster.com/eqtmapi/getBonusList?indexcode=1-71",
        "Split": "https://www.equitymaster.com/eqtmapi/getSplitList?indexcode=1-71",
    }

    all_data = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(_fetch_action, a_type, url, headers) for a_type, url in api_endpoints.items()]
        for f in futures:
            try:
                all_data.extend(f.result())
            except Exception as e:
                logger.warning("Partial fetch failed: %s", e)

    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, default=str)
        logger.info(f"Cached {len(all_data)} records to {CACHE_PATH}")
    except Exception as e:
        logger.warning("Cache write failed: %s", e)

    logger.info(f"Fetched total {len(all_data)} corporate actions")
    return all_data

# ----------------------------------------------------------------------
# Main API
# ----------------------------------------------------------------------
def get_corporate_actions(tickers: List[str], mode: str = "past", lookback_days: int = 365) -> List[Dict[str, Any]]:
    results = []
    today = datetime.now().date()
    start_date = today - timedelta(days=lookback_days)
    equitymaster_data = _fetch_equitymaster_data()

    for ticker in tickers:
        try:
            stock = _retry(lambda: yf.Ticker(ticker), retries=2, backoff=0.5)
            actions = []

            if mode == "upcoming":
                info = getattr(stock, "info", {}) or {}
                company_name = (info.get("shortName") or info.get("longName") or "").strip()

                div_ex_date = div_amt = div_yield = None
                bonus_ratio = bonus_ex_date = None
                split_ratio = split_ex_date = None

                # Yahoo Dividends
                if info:
                    try:
                        if info.get("exDividendDate"):
                            div_ex_date = datetime.utcfromtimestamp(info["exDividendDate"]).date()
                        if info.get("dividendRate"):
                            div_amt = float(info["dividendRate"])
                    except Exception as e:
                        logger.debug(f"Yahoo info parse failed for {ticker}: {e}")

                # Equitymaster Match
                matched = [item for item in equitymaster_data if simple_name_match(company_name, item["name"])]
                if matched:
                    logger.info(f"Matched {len(matched)} EM actions for {ticker} ({company_name})")

                for item in matched:
                    if item["type"] == "Dividend":
                        if not div_amt or (div_ex_date and item["ex_date"] > div_ex_date):
                            div_ex_date, div_amt = item["ex_date"], item["value"]
                    elif item["type"] == "Bonus" and not bonus_ratio:
                        bonus_ratio, bonus_ex_date = item["value"], item["ex_date"]
                    elif item["type"] == "Split" and not split_ratio:
                        split_ratio, split_ex_date = item["value"], item["ex_date"]

                # Dividend Yield
                if div_ex_date and div_amt:
                    try:
                        price_df = stock.history(period="1d")
                        if not price_df.empty:
                            price = float(price_df["Close"].iloc[-1])
                            if price > 0:
                                div_yield = round((div_amt / price) * 100, 2)
                    except Exception as e:
                        logger.debug(f"Yield calc failed for {ticker}: {e}")

                # Assemble Results
                if div_amt and div_ex_date:
                    actions.append({
                        "type": "Upcoming Dividend",
                        "amount": round(div_amt, 2),
                        "ex_date": _fmt_date(div_ex_date),
                        "yield": div_yield,
                    })
                if bonus_ratio and bonus_ex_date:
                    actions.append({
                        "type": "Upcoming Bonus",
                        "ratio": bonus_ratio,
                        "ex_date": _fmt_date(bonus_ex_date),
                    })
                if split_ratio and split_ex_date:
                    actions.append({
                        "type": "Upcoming Split",
                        "ratio": split_ratio,
                        "ex_date": _fmt_date(split_ex_date),
                    })

            else:
                # Past corporate actions
                try:
                    corp_actions_df = stock.actions
                    if corp_actions_df is not None and not corp_actions_df.empty:
                        corp_actions_df.index = pd.to_datetime(corp_actions_df.index, errors="coerce")
                        corp_actions_df = corp_actions_df[corp_actions_df.index.date >= start_date]
                        for date, row in corp_actions_df.iterrows():
                            date_str = _fmt_date(date.date())
                            if safe_float(row.get("Dividends")):
                                actions.append({
                                    "type": "Dividend",
                                    "amount": round(float(row["Dividends"]), 2),
                                    "ex_date": date_str,
                                })
                            split_val = _safe_float(row.get("Stock Splits"))
                            if split_val and split_val != 0 and not math.isnan(split_val):
                                actions.append({
                                    "type": "Split",
                                    "ratio": str(split_val),
                                    "ex_date": date_str,
                                })
                except Exception as e:
                    logger.debug(f"Past fetch failed for {ticker}: {e}")

            results.append({
                "ticker": ticker,
                "actions": actions,
                "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
            })

        except Exception as e:
            logger.exception(f"Unhandled error for {ticker}: {e}")
            results.append({"ticker": ticker, "actions": []})

    return results


if __name__ == "__main__":
    print(json.dumps(get_corporate_actions(["INFY.NS"], mode="upcoming"), indent=2))
