# services/fundamentals_refactored_v3_3.py
"""
Fundamentals - Refactored v3.3
- Full coverage of metrics (21 metrics)
- Uses FIELD = FUNDAMENTAL_FIELD_CANDIDATES from config.constants
- Uses services.data_fetch helpers: safe_float, safe_div, safe_info, safe_history, _wrap_calc
- All metrics return standardized dicts: {raw, value, score, desc, alias, source}
- Weighted aggregation uses FUNDAMENTAL_WEIGHTS (includes 0 scores); missing values are excluded
- Optional market-factor penalty applied post-aggregation
- Concise docstrings for each calc_ function
"""
import logging
from functools import lru_cache
import math
from typing import Dict, Any, Optional

import pandas as pd
import yfinance as yf
interval = "1d"
period = "2y"

from services.data_fetch import safe_float, safe_div, safe_get, safe_info, safe_history, _wrap_calc , _fmt_pct
from config.constants import FUNDAMENTAL_WEIGHTS, FUNDAMENTAL_ALIAS_MAP, FUNDAMENTAL_FIELD_CANDIDATES as FIELD , SECTOR_PE_AVG

logger = logging.getLogger(__name__)

def _find_row_series(df: pd.DataFrame, keys: list):
    """
    Returns the first matching row (Series) in DataFrame for any candidate key.
    
    Handles:
    - exact match
    - case-insensitive match
    - partial match ("net income" in "total net income applicable…")
    - stripped / normalized keys
    - multi-index rows
    """

    if df is None or df.empty:
        return None

    # Normalize DataFrame index to comparable lowercase keys
    def normalize(s: str) -> str:
        return str(s).strip().lower().replace("_", "").replace(" ", "")

    norm_index = {normalize(k): k for k in df.index}

    # Try all candidate keys
    for key in keys:
        k_norm = normalize(key)

        # 1️⃣ exact normalized match
        if k_norm in norm_index:
            return df.loc[norm_index[k_norm]]

        # 2️⃣ try substring match
        for norm_key, orig_key in norm_index.items():
            if k_norm in norm_key:
                return df.loc[orig_key]

        # 3️⃣ reverse substring match: df label contains key
        for norm_key, orig_key in norm_index.items():
            if norm_key in k_norm:
                return df.loc[orig_key]

    return None


def _find_from_df_like(obj, candidates):
    """
    Return first matching candidate value from dict-like, DataFrame-like, or attributes.
    `candidates` is a list of strings; typically passed from FIELD[...] map.
    """
    if obj is None:
        return None
    try:
        if isinstance(obj, dict):
            for c in candidates:
                if c in obj and obj[c] is not None:
                    return safe_float(obj[c])
        if hasattr(obj, "index") and hasattr(obj, "loc"):
            for c in candidates:
                if c in obj.index:
                    try:
                        v = obj.loc[c]
                        if hasattr(v, "iloc"):
                            v = v.iloc[0]
                        return safe_float(v)
                    except Exception:
                        continue
        for c in candidates:
            if hasattr(obj, c):
                return safe_float(getattr(obj, c))
    except Exception:
        pass
    return None

def _prepare_base_data(t: yf.Ticker) -> Dict[str, Any]:
    """
    Prepares all base financial data and precomputes candidate field values
    using FIELD map (FUNDAMENTAL_FIELD_CANDIDATES).
    """
    symbol = getattr(t, "ticker", None)
    info = safe_info(symbol) or getattr(t, "info", {}) or {}

    financials = getattr(t, "financials", pd.DataFrame())
    balance_sheet = getattr(t, "balance_sheet", pd.DataFrame())
    cashflow = getattr(t, "cashflow", pd.DataFrame())
    quarterly_financials = getattr(t, "quarterly_financials", pd.DataFrame())
    earnings = getattr(t, "earnings", pd.DataFrame())

    # --- Base containers ---
    base_data = {
        "info": info,
        "financials": financials,
        "balance_sheet": balance_sheet,
        "cashflow": cashflow,
        "quarterly_financials": quarterly_financials,
        "earnings": earnings,
        "history": safe_history(symbol, period="2y", interval="1d"),
        "fields": {}
    }

    # --- Universal lookup helper using your _find_from_df_like ---
    def _try_find(key: str):
        """Try to fetch a field value from all data sources in priority order."""
        candidates = FIELD.get(key, [])
        # 1️⃣ Try info first
        val = _find_from_df_like(info, candidates)
        if val is not None:
            return val
        # 2️⃣ Try financials, balance sheet, cashflow
        for df in (financials, balance_sheet, cashflow, quarterly_financials, earnings):
            val = _find_from_df_like(df, candidates)
            if val is not None:
                return val
        return None

    # --- Collect values for all known FIELD entries ---
    for field_key in FIELD.keys():
        try:
            val = _try_find(field_key)
            if val is not None:
                base_data["fields"][field_key] = val
        except Exception:
            continue

    # --- Add a few computed shortcut aliases ---
    # Example: “pe” combines multiple possible keys
    base_data["fields"].update({
        "pe": safe_float(
            info.get("trailingPE") or info.get("forwardPE") or info.get("peRatio") or info.get("priceEarnings")
        ),
        "pb": safe_float(
            info.get("priceToBook") or info.get("priceToBookRatio") or info.get("pb")
        ),
        "beta": safe_float(info.get("beta")),
        "sector": info.get("sector") or info.get("sectorDisp"),
        "market_cap": safe_float(info.get("marketCap") or info.get("market_cap")),
    })

    # Filter out None or NaN
    base_data["fields"] = {k: v for k, v in base_data["fields"].items() if pd.notna(v)}
    return base_data

def _history_df(t: yf.Ticker, period="2y", interval="1d"):
    """Defensive history fetch using safe_history then yfinance fallback."""
    try:
        df = safe_history(getattr(t, "ticker", None), period=period, interval=interval)
    except Exception:
        df = None
    if df is None or getattr(df, "empty", True):
        try:
            df = yf.download(getattr(t, "ticker", None), period=period, interval=interval, auto_adjust=True, progress=False)
        except Exception:
            df = pd.DataFrame()
    return df or pd.DataFrame()

FUNDAMENTAL_LOOKBACKS = {
    "analyst_rating": 60,  # days
    "trend_strength": 120,  # days for EMA
}
# -------------------------
# Metric calculators (docstrings included)
# -------------------------
def _derive_wacc(base_data) -> float:
    """
    Derives company-specific Weighted Average Cost of Capital (WACC)
    using precomputed fundamentals metrics instead of raw info fetches.
    Falls back gracefully to a default value if data is missing.
    """
    try:
        rf = 6.8        # India 10Y G-sec (risk-free)
        mkt_prem = 5.5  # Market risk premium
        tax = 0.25      # Effective corporate tax rate

        beta = base_data["fields"].get("beta")
        de_ratio = base_data["fields"].get("de_ratio")
        icr = base_data["fields"].get("interest_coverage")

        e_weight = 1 / (1 + de_ratio)
        d_weight = de_ratio / (1 + de_ratio)

        cost_equity = rf + beta * mkt_prem

        cost_debt = (100 / icr) if icr and icr > 0 else 8.0

        wacc = (e_weight * cost_equity) + (d_weight * cost_debt * (1 - tax))
        return round(wacc, 2)
    except Exception:
        return 7.2  # fallback


@_wrap_calc("pe_ratio")
def calc_pe_ratio(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """P/E ratio from info (trailing or forward)."""
    pe = base_data["fields"].get("pe")
    if pe is None:
        return None
    score = 10 if pe <= 12 else 7 if pe <= 20 else 4 if pe <= 30 else 1
    return {"raw": pe, "value": round(pe, 2), "score": int(score), "desc": f"P/E {round(pe,2)}"}


@_wrap_calc("pb_ratio")
def calc_pb_ratio(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Price-to-book ratio using common info fields."""
    pb = base_data["fields"].get("pb")
    if pb is None:
        return None
    score = 10 if pb <= 1.5 else 6 if pb <= 3 else 3 if pb <= 5 else 1
    return {"raw": pb, "value": round(pb, 2), "score": int(score), "desc": f"P/B {round(pb,2)}"}

@_wrap_calc("profit_growth_3y")
def calc_profit_growth_3y(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Calculates 3-Year Profit CAGR from yfinance 'financials' data.
    
    This is the "G" in the PEG ratio.
    
    Note: yfinance 'financials' typically provides 4 years of annual data
    (e.g., 2023, 2022, 2021, 2020). This function calculates the CAGR 
    between the most recent (Year 4) and the oldest (Year 1), 
    which is a 3-year period (n=3).
    """
    # Access the annual financials
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    
    try:
        profit_series = pd.Series(dtype=float)
        
        # 1. Target "Net Income" (Profit), not "Diluted EPS"
        if not getattr(fin, "empty", True) and "Net Income" in fin.index:
            # Data is newest-to-oldest, e.g., [2023, 2022, 2021, 2020]
            ser = fin.loc["Net Income"].dropna().astype(float)
            
            # 2. Reverse to get oldest-to-newest for CAGR logic
            profit_series = ser[::-1].reset_index(drop=True)

        valid = profit_series.dropna()
        
        # We need at least 2 data points to calculate growth
        if valid.empty or len(valid) < 2:
            return None

        vals = list(valid)
        
        start = float(vals[0])  # Oldest value (e.g., 2020)
        end = float(vals[-1]) # Newest value (e.g., 2023)

        # 3. Cannot calculate growth from a negative or zero base
        if start <= 0 or end <= 0:
            return None

        # 'n' is the number of periods, which is len(vals) - 1
        # If we have 4 data points, n = 3 (a 3-year CAGR)
        n = len(vals) - 1
        if n <= 0:
            return None

        # 4. Calculate CAGR
        cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0

        # Cache for reuse in other calculations
        if "fields" in base_data:
            base_data["fields"]["profit_cagr_3y"] = cagr 
        
        score = 10 if cagr > 15 else 5 if cagr >= 5 else 0
        return {
            "raw": cagr, 
            "value": round(cagr, 2), 
            "score": int(score), 
            "desc": f"Profit CAGR {round(cagr,2)}% ({n}yr)"
        }
    except Exception as e:
        logger.debug(f"Profit CAGR calc error: {e}")
        return None

@_wrap_calc("peg_ratio")
def calc_peg_ratio(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Calculates PEG Ratio using 3Y Profit Growth (CAGR).
    Falls back to Yahoo 'earningsGrowth' if 3Y data is missing.
    Includes guard clauses for negative or absurd values.
    """
    try:
        info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
        fields = base_data.get("fields", {}) if base_data else {}

        # 1️⃣ PE Ratio (must exist)
        pe = fields.get("pe")
        if not pe or pe <= 0:
            return None

        # 2️⃣ Profit Growth Sources
        profit_g = None
        growth_desc = ""

        # Priority A: Force calculation from our robust function
        # This ensures we get the data even if the cache isn't populated yet.
        try:
            profit_3y_metric = calc_profit_growth_3y(t, base_data=base_data)
            
            # Handle the dict return type of the function
            if profit_3y_metric and profit_3y_metric.get("raw") is not None:
                profit_g = safe_float(profit_3y_metric.get("raw"))
                growth_desc = "3Y Profit"
        except Exception as e:
            # logger.debug(f"PEG: Calling calc_profit_growth_3y failed: {e}")
            pass

        # Priority B: Fallback to Yahoo 'earningsGrowth'
        if profit_g is None:
            yahoo_epsg = safe_float(info.get("earningsGrowth"))
            if yahoo_epsg:
                # Yahoo usually sends decimal (0.15), we need percent (15.0)
                profit_g = yahoo_epsg * 100.0 
                growth_desc = "Yahoo est."

        # Growth must be positive for PEG to make sense
        if profit_g is None or profit_g <= 0:
            return None

        # Normalization check: if growth is 0.20, it likely means 20%
        if profit_g < 1.0: 
             profit_g = profit_g * 100.0

        # 3️⃣ PEG Calculation
        peg = safe_div(pe, profit_g)
        
        if peg is None:
            return None
            
        # --- GUARD CLAUSE ---
        # Discard insane values (negative or > 50) which break the scale
        if peg <= 0 or peg > 50:
            return None
        # --------------------

        # 4️⃣ Scoring Logic
        score = (
            10 if peg <= 1 else
            8 if peg <= 1.5 else
            5 if peg <= 2.5 else
            0
        )

        desc = f"PEG {round(peg, 2)} (PE {round(pe,2)}, G {round(profit_g,2)}% [{growth_desc}])"

        return {
            "raw": peg,
            "value": round(peg, 2),
            "score": int(score),
            "desc": desc,
            "alias": "PEG Ratio",
            "source": "core",
        }

    except Exception as e:
        logger.debug(f"PEG calc error: {e}")
        return None
@_wrap_calc("roe")
def calc_roe(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Return on Equity using info or financials/balance_sheet fallback."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    roe = safe_float(info.get("returnOnEquity") or info.get("return_on_equity"))
    if roe is None:
        # fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
        # bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
        ni = base_data["fields"].get("net_income") #_find_from_df_like(fin, FIELD.get("net_income", []))
        eq = base_data["fields"].get("total_equity") #_find_from_df_like(bs, FIELD.get("total_equity", []))
        if ni is not None and eq not in (None, 0):
            roe = safe_div(ni, eq)
    if roe is None:
        return None
    pct = roe * 100.0 if abs(roe) < 2 else roe
    score = 10 if pct >= 20 else 7 if pct >= 12 else 3 if pct >= 5 else 0
    return {"raw": pct, "value": _fmt_pct(pct), "score": int(score), "desc": f"ROE {round(pct,2)}%"}


@_wrap_calc("roce")
def calc_roce(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Return on Capital Employed using info or derived from financials/balance_sheet."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    roce = safe_float(info.get("returnOnCapitalEmployed") or info.get("roce"))
    if roce is None:
        # fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
        # bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
        ebit = base_data["fields"].get("ebit") #or _find_from_df_like(fin, FIELD.get("ebit", []))
        total_assets = base_data["fields"].get("total_assets") #or _find_from_df_like(bs, FIELD.get("total_assets", []))
        current_liab = base_data["fields"].get("current_liabilities") #or _find_from_df_like(bs, FIELD.get("current_liabilities", []))
        if ebit is not None and total_assets is not None:
            cap = total_assets - current_liab
            roce_calc = safe_div(ebit, cap)
            if roce_calc is None:
                return None
            roce = roce_calc
    if roce is None:
        return None
    pct = roce * 100.0 if abs(roce) < 2 else roce
    score = 10 if pct >= 20 else 7 if pct >= 12 else 3 if pct >= 5 else 0
    return {"raw": pct, "value": _fmt_pct(pct), "score": int(score), "desc": f"ROCE {round(pct,2)}%"}


@_wrap_calc("roic")
def calc_roic(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    ROIC = NOPAT / Invested Capital
    NOPAT = EBIT * (1 - Tax Rate). Uses dynamic tax rate based on financials.
    """

    # ----------------------------------
    # 0. Base Inputs
    # ----------------------------------
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    # Use fields dictionary for centralized data extraction
    fields = base_data.get("fields", {}) if base_data else {} 
    
    # --- Data Requirements ---
    ebit = fields.get("ebit")
    total_debt = fields.get("total_debt")
    cash = fields.get("total_cash")
    tax_expense = fields.get("tax_expense")
    pre_tax_income = fields.get("pre_tax_income")
    # -------------------------

    if ebit is None:
        return None

    # ----------------------------------
    # 1. Invested Capital Calculation
    # ----------------------------------
    # NOTE: We rely on _prepare_base_data to populate all necessary fields, 
    # ensuring data consistency.
    
    # Calculate Equity Value from YF info (Book Value * Shares Outstanding)
    equity_val = safe_float(info.get("bookValue")) * safe_float(info.get("sharesOutstanding"))

    invested_capital = None
    if total_debt and equity_val:
        # Standard Invested Capital = Total Debt + Equity - Cash
        invested_capital = (total_debt + equity_val) - (cash or 0)

        # Low-debt fallback to Enterprise Value (EV)
        if total_debt / equity_val < 0.05:
            invested_capital = safe_float(info.get("enterpriseValue")) or invested_capital

    if not invested_capital:
        return None

    # ----------------------------------
    # 2. Dynamic Tax Rate Calculation
    # ----------------------------------
    tax_rate = 0.25 # Default fallback
    
    if tax_expense is not None and pre_tax_income is not None:
        if pre_tax_income > 0:
            # Calculate rate only if Pretax Income is positive
            tax_rate = safe_div(tax_expense, pre_tax_income, 0.25)
            # Sanitize rate between 0% and 50%
            tax_rate = min(max(tax_rate, 0.0), 0.5) 
        else:
            # If Pretax Income is zero or negative (loss), effective tax rate for NOPAT is 0
            tax_rate = 0.0 
    else:
        tax_rate = 0.25 # Default fallback
    
    # ----------------------------------
    # 3. NOPAT + ROIC
    # ----------------------------------
    nopat = float(ebit) * (1 - tax_rate)
    roic = safe_div(nopat, invested_capital)
    if roic is None:
        return None

    roic_pct = roic * 100 if abs(roic) < 2 else roic

    # ----------------------------------
    # 4. Spread vs WACC & Score
    # ----------------------------------
    wacc = _derive_wacc(base_data)
    spread = roic_pct - wacc

    if roic_pct >= 20:
        score = 10
    elif roic_pct >= 10:
        score = 7
    elif roic_pct >= 5:
        score = 3
    else:
        score = 0

    return {
        "raw": roic_pct,
        "value": f"{roic_pct:.2f}%",
        "score": score,
        # Updated desc to show dynamic tax rate
        "desc": f"ROIC {roic_pct:.2f}% (Tax {tax_rate*100:.1f}%) vs WACC {wacc:.2f}% → Spread {spread:.2f}%",
        "alias": "Return on Invested Capital (ROIC)",
        "meta": {
            "wacc": wacc,
            "spread": round(spread, 2),
            "tax_rate": round(tax_rate * 100, 2),
            "nopat": round(nopat, 2)
        },
        "source": "core"
    }

@_wrap_calc("de_ratio")
def calc_de_ratio(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    de = safe_float(info.get("debtToEquity"))
    if de is not None and de > 3:  # mis-scaled (e.g. 639 instead of 0.64)
        de = de / 100.0

    if de is None:
        debt = base_data["fields"].get("total_debt") or safe_float(info.get("totalDebt"))
        eq = safe_float(info.get("bookValue")) * safe_float(info.get("sharesOutstanding"))
        de = safe_div(debt, eq)

    if de is None:
        bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
        debt = _find_from_df_like(bs, ["Long Term Debt", "Borrowings", "longTermDebt"]) or 0.0
        equity = _find_from_df_like(bs, ["Total Stockholders Equity", "Shareholders Equity", "totalStockholdersEquity"]) or 0.0
        de = safe_div(debt, equity)

    if de is None:
        return None

    score = 10 if de < 0.5 else 7 if de < 1 else 4 if de < 2 else 0
    return {"raw": de, "value": round(de, 3), "score": int(score),
            "desc": f"Debt/Equity {round(de,3)}", "alias": "Debt to Equity", "source": "core"}

@_wrap_calc("interest_coverage")
def calc_interest_coverage(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Interest coverage (EBIT / Interest). Uses large default if interest tiny."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    ebit = base_data["fields"].get("ebit") or _find_from_df_like(fin, FIELD.get("ebit", []))
    interest = _find_from_df_like(fin, FIELD.get("interest_expense", []))
    if ebit is None:
        return None
    if not interest or interest < 1e5:
        ratio = 999.0
    else:
        ratio = safe_div(ebit, interest)
    if ratio is None:
        return None
    score = 10 if ratio > 5 else 5 if ratio >= 3 else 0
    return {"raw": ratio, "value": f"{round(ratio,2)}x", "score": int(score), "desc": f"Interest Coverage {round(ratio,2)}x"}


@_wrap_calc("fcf_yield")
def calc_fcf_yield(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """FCF yield computed as (OCF - CapEx) / MarketCap, with multiple fallbacks."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    market_cap = base_data["fields"].get("market_cap") or safe_float(info.get("marketCap") or info.get("market_cap"))
    cf = base_data.get("cashflow") if base_data else getattr(t, "cashflow", pd.DataFrame())
    fcf = None
    if cf is not None and not getattr(cf, "empty", True):
        fcf = _find_from_df_like(cf, FIELD.get("free_cash_flow", []))
        if fcf is None:
            ocf = _find_from_df_like(cf, FIELD.get("ocf", []))
            capex = _find_from_df_like(cf, FIELD.get("capex", [])) or 0.0
            if ocf is not None:
                fcf = ocf - abs(capex)
    else:
        fcf = _find_from_df_like(info, FIELD.get("free_cash_flow", []))
    if market_cap is None or market_cap <= 0 or fcf is None:
        return None
    fcf_yield = (fcf / market_cap) * 100.0
    score = 10 if fcf_yield >= 10 else 8 if fcf_yield >= 6 else 5 if fcf_yield >= 4 else 2 if fcf_yield > 0 else 0
    return {"raw": fcf_yield, "value": f"{round(fcf_yield,2)}%", "score": int(score), "desc": f"FCF Yield {round(fcf_yield,2)}%"}


@_wrap_calc("fcf_margin")
def calc_fcf_margin(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """FCF margin = FCF / Revenue (support metric; included if present)."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    cf = base_data.get("cashflow") if base_data else getattr(t, "cashflow", pd.DataFrame())
    if getattr(fin, "empty", True) and getattr(cf, "empty", True):
        return None
    revenue = _find_from_df_like(fin, FIELD.get("revenue", []))
    fcf = None
    if cf is not None and not getattr(cf, "empty", True):
        fcf = _find_from_df_like(cf, FIELD.get("free_cash_flow", []))
        if fcf is None:
            ocf = _find_from_df_like(cf, FIELD.get("ocf", []))
            capex = _find_from_df_like(cf, FIELD.get("capex", [])) or 0.0
            if ocf is not None:
                fcf = ocf - abs(capex)
    if revenue is None or fcf is None:
        return None
    fcf_margin = safe_div(fcf, revenue, 0) * 100.0
    score = 10 if fcf_margin >= 15 else 5 if fcf_margin >= 10 else 0
    return {"raw": fcf_margin, "value": f"{round(fcf_margin,2)}%", "score": int(score), "desc": f"FCF Margin {round(fcf_margin,2)}%"}


@_wrap_calc("current_ratio")
def calc_current_ratio(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Current ratio: current assets / current liabilities."""
    bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
    ca = _find_from_df_like(bs, FIELD.get("current_assets", []))
    cl = _find_from_df_like(bs, FIELD.get("current_liabilities", []))
    if ca is None or cl is None or cl == 0:
        return None
    ratio = safe_div(ca, cl)
    score = 10 if ratio >= 2 else 7 if ratio >= 1.5 else 3 if ratio >= 1 else 0
    return {"raw": ratio, "value": round(ratio, 2), "score": int(score), "desc": f"Current Ratio {round(ratio,2)}"}


@_wrap_calc("asset_turnover")
def calc_asset_turnover(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Asset turnover = Revenue / Total Assets."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
    rev = _find_from_df_like(fin, FIELD.get("revenue", []))
    assets = _find_from_df_like(bs, FIELD.get("total_assets", []))
    if rev is None or assets is None or assets == 0:
        return None
    ratio = safe_div(rev, assets)
    score = 10 if ratio > 1 else 5 if ratio >= 0.5 else 0
    return {"raw": ratio, "value": round(ratio, 2), "score": int(score), "desc": f"Asset Turnover {round(ratio,2)}"}


@_wrap_calc("piotroski_f")
def calc_piotroski_f(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Piotroski F-score (9-point) with breakdown metadata."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
    cf = base_data.get("cashflow") if base_data else getattr(t, "cashflow", pd.DataFrame())
    try:
        if getattr(fin, "empty", True) or getattr(bs, "empty", True):
            return None

        def safe_get(df, key, idx=0):
            try:
                if key in df.index:
                    v = df.loc[key]
                    if hasattr(v, "iloc"):
                        return safe_float(v.iloc[idx])
                    return safe_float(v)
            except Exception:
                return None

        net_income = safe_get(fin, FIELD.get("net_income", [])[0] if FIELD.get("net_income") else "Net Income", 0)
        prev_net_income = safe_get(fin, FIELD.get("net_income", [])[0] if FIELD.get("net_income") else "Net Income", 1)
        total_assets = safe_get(bs, FIELD.get("total_assets", [])[0] if FIELD.get("total_assets") else "Total Assets", 0)
        prev_total_assets = safe_get(bs, FIELD.get("total_assets", [])[0] if FIELD.get("total_assets") else "Total Assets", 1)
        ocf = safe_get(cf, FIELD.get("ocf", [])[0] if FIELD.get("ocf") else "Total Cash From Operating Activities", 0)
        total_debt = safe_get(bs, FIELD.get("total_debt", [])[0] if FIELD.get("total_debt") else "Total Debt", 0)
        prev_debt = safe_get(bs, FIELD.get("total_debt", [])[0] if FIELD.get("total_debt") else "Total Debt", 1)
        current_assets = safe_get(bs, FIELD.get("current_assets", [])[0] if FIELD.get("current_assets") else "Total Current Assets", 0)
        current_liabilities = safe_get(bs, FIELD.get("current_liabilities", [])[0] if FIELD.get("current_liabilities") else "Total Current Liabilities", 0)
        prev_current_assets = safe_get(bs, FIELD.get("current_assets", [])[0] if FIELD.get("current_assets") else "Total Current Assets", 1)
        prev_current_liabilities = safe_get(bs, FIELD.get("current_liabilities", [])[0] if FIELD.get("current_liabilities") else "Total Current Liabilities", 1)
        shares = safe_get(fin, FIELD.get("shares_outstanding", [])[0] if FIELD.get("shares_outstanding") else "Basic Average Shares", 0)
        prev_shares = safe_get(fin, FIELD.get("shares_outstanding", [])[0] if FIELD.get("shares_outstanding") else "Basic Average Shares", 1)
        revenue = safe_get(fin, FIELD.get("revenue", [])[0] if FIELD.get("revenue") else "Total Revenue", 0)
        prev_revenue = safe_get(fin, FIELD.get("revenue", [])[0] if FIELD.get("revenue") else "Total Revenue", 1)
        cogs = safe_get(fin, FIELD.get("cogs", [])[0] if FIELD.get("cogs") else "Cost Of Revenue", 0)
        prev_cogs = safe_get(fin, FIELD.get("cogs", [])[0] if FIELD.get("cogs") else "Cost Of Revenue", 1)

        def safe_div0(a, b):
            try:
                if a is None or b is None or b == 0:
                    return None
                return safe_div(a, b)
            except Exception:
                return None

        roa_now = safe_div0(net_income, total_assets)
        roa_prev = safe_div0(prev_net_income, prev_total_assets)
        cr_now = safe_div0(current_assets, current_liabilities)
        cr_prev = safe_div0(prev_current_assets, prev_current_liabilities)
        gm_now = safe_div0((revenue - cogs), revenue)
        gm_prev = safe_div0((prev_revenue - prev_cogs), prev_revenue)
        at_now = safe_div0(revenue, total_assets)
        at_prev = safe_div0(prev_revenue, prev_total_assets)

        points = 0
        breakdown = {}
        # F1: positive net income
        if net_income is not None and net_income > 0:
            points += 1; breakdown["F1_PositiveNetIncome"] = 1
        else:
            breakdown["F1_PositiveNetIncome"] = 0
        # F2: positive OCF
        if ocf is not None and ocf > 0:
            points += 1; breakdown["F2_PositiveOCF"] = 1
        else:
            breakdown["F2_PositiveOCF"] = 0
        # F3: improving ROA
        if roa_now is not None and roa_prev is not None and roa_now > roa_prev:
            points += 1; breakdown["F3_ImprovingROA"] = 1
        else:
            breakdown["F3_ImprovingROA"] = 0
        # F4: CFO > Net Income
        if ocf is not None and net_income is not None and ocf > net_income:
            points += 1; breakdown["F4_CFO_GT_Net"] = 1
        else:
            breakdown["F4_CFO_GT_Net"] = 0
        # F5: lower leverage (debt/assets)
        try:
            if total_debt is not None and prev_debt is not None and total_assets is not None and prev_total_assets is not None:
                dr_now = safe_div(total_debt, total_assets)
                dr_prev = safe_div(prev_debt, prev_total_assets)
                if dr_now is not None and dr_prev is not None and dr_now < dr_prev:
                    points += 1; breakdown["F5_LowerLeverage"] = 1
                else:
                    breakdown["F5_LowerLeverage"] = 0
            else:
                breakdown["F5_LowerLeverage"] = 0
        except Exception:
            breakdown["F5_LowerLeverage"] = 0
        # F6: higher current ratio
        if cr_now is not None and cr_prev is not None and cr_now > cr_prev:
            points += 1; breakdown["F6_HigherCR"] = 1
        else:
            breakdown["F6_HigherCR"] = 0
        # F7: no share dilution
        if shares is not None and prev_shares is not None and shares <= prev_shares:
            points += 1; breakdown["F7_NoDilution"] = 1
        else:
            breakdown["F7_NoDilution"] = 0
        # F8: improving gross margin
        if gm_now is not None and gm_prev is not None and gm_now > gm_prev:
            points += 1; breakdown["F8_ImprovingGM"] = 1
        else:
            breakdown["F8_ImprovingGM"] = 0
        # F9: improving asset turnover
        if at_now is not None and at_prev is not None and at_now > at_prev:
            points += 1; breakdown["F9_ImprovingAT"] = 1
        else:
            breakdown["F9_ImprovingAT"] = 0

        val = points
        scaled = round((val / 9.0) * 10.0, 1)
        desc = "Strong" if val >= 7 else "Moderate" if val >= 4 else "Weak"
        return {"raw": val, "value": f"{val}/9", "score": scaled, "desc": desc, "meta": {"breakdown": breakdown}}
    except Exception:
        return None


@_wrap_calc("r_d_intensity")
def calc_rd_intensity(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """R&D intensity as pct of revenue (if available)."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    rd = _find_from_df_like(info, FIELD.get("rd_expense", []))
    revenue = _find_from_df_like(info, FIELD.get("revenue", [])) or _find_from_df_like(fin, FIELD.get("revenue", []))
    if rd is None and not getattr(fin, "empty", True):
        rd = _find_from_df_like(fin, FIELD.get("rd_expense", []))
    if rd is None or revenue in (None, 0):
        return None
    pct = (rd / revenue) * 100.0 if revenue else None
    if pct is None:
        return None
    score = 10 if pct > 5 else 5 if pct > 2 else 0
    return {"raw": pct, "value": round(pct, 2), "score": int(score), "desc": f"R&D {round(pct,2)}%"}


@_wrap_calc("earnings_stability")
def calc_earnings_stability(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Earnings stability measured by CV of Net Income series."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    if getattr(fin, "empty", True) or FIELD.get("net_income", []) and FIELD["net_income"][0] not in fin.index:
        return None
    try:
        series = fin.loc[FIELD.get("net_income", ["Net Income"])[0]].dropna().astype(float)
    except Exception:
        try:
            if "Net Income" not in fin.index:
                return None
            series = fin.loc["Net Income"].dropna().astype(float)
        except Exception:
            return None
    if len(series) < 3:
        return None
    std = float(series.std(ddof=0))
    mean = float(series.mean())
    if mean == 0:
        return None
    cv = abs(std / mean)
    score = 10 if cv < 0.2 else 6 if cv < 0.5 else 2
    return {"raw": cv, "value": round(cv, 3), "score": int(score), "desc": f"Earnings CV {round(cv,3)}"}


@_wrap_calc("market_cap_cagr")
def calc_market_cap_cagr(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None, years: int = 5):
    """Market-cap CAGR derived from price history (monthly close) over `years`."""
    try:
        df = _history_df(t, period=f"{years+1}y", interval="1mo")
        if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
            return None
        close = df["Close"].dropna().astype(float)
        if len(close) < 2:
            return None
        beginning = float(close.iloc[0]); ending = float(close.iloc[-1])
        if beginning <= 0 or ending <= 0:
            return None
        cagr = (ending / beginning) ** (1.0 / years) - 1.0
        cagr_pct = round(cagr * 100.0, 2)
        score = 10 if cagr_pct >= 25 else 8 if cagr_pct >= 15 else 5 if cagr_pct >= 5 else 0
        return {"raw": cagr_pct, "value": f"{cagr_pct}%", "score": int(score), "desc": f"Market Cap CAGR {cagr_pct}%"}
    except Exception:
        return None


@_wrap_calc("promoter_holding")
def calc_promoter_holding(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Promoter holding percent from info (if present)."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    p = _find_from_df_like(info, FIELD.get("promoter_holding", []))
    if p is None:
        return None
    pct = p * 100.0 if abs(p) < 2 else p
    score = 10 if pct > 40 else 5 if pct > 20 else 0
    return {"raw": pct, "value": round(pct, 2), "score": int(score), "desc": f"Promoter {round(pct,2)}%"}


@_wrap_calc("institutional_ownership")
def calc_institutional_ownership(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Institutional ownership percent from info (if present)."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    v = _find_from_df_like(info, FIELD.get("institutional_ownership", []))
    if v is None:
        return None
    pct = v * 100.0 if abs(v) < 2 else v
    score = 10 if pct > 30 else 5 if pct > 10 else 0
    return {"raw": pct, "value": round(pct, 2), "score": int(score), "desc": f"Inst {round(pct,2)}%"}


@_wrap_calc("beta")
def calc_beta(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Beta from info (market volatility measure)."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    b = safe_float(info.get("beta"))
    if b is None:
        return None
    score = 10 if abs(b) < 0.8 else 7 if abs(b) < 1 else 3 if abs(b) < 1.3 else 0
    return {"raw": b, "value": round(b, 2), "score": int(score), "desc": f"Beta {round(b,2)}"}


@_wrap_calc("52w_position")
def calc_52w_position(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Distance off 52-week high expressed as percent off-high."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    high = safe_float(info.get("fiftyTwoWeekHigh") or info.get("52WeekHigh"))
    price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice") or info.get("previousClose"))
    if price is None or high is None:
        return None
    off_high_pct = ((high - price) / high) * 100.0
    score = 10 if off_high_pct < 10 else 7 if off_high_pct < 20 else 3 if off_high_pct < 40 else 0
    return {"raw": off_high_pct, "value": round(off_high_pct, 2), "score": int(score), "desc": f"{round(off_high_pct,2)}% off-high"}

@_wrap_calc("dividend_yield")
def calc_dividend_yield(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Dividend yield (%), normalized from ratio form."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    dy = safe_float(info.get("dividendYield"))
    if dy is None:
        return None
        
    # Standard logic: Assumes 'dy' is a ratio (e.g., 0.05 -> 5.0%)
    pct = dy * 100 if abs(dy) < 2 else dy

    # --- THIS IS THE FIX ---
    # Patch for yfinance data errors where a % (0.56) is sent as a ratio (0.56)
    # If pct is absurdly high (e.g., 56.0%), assume it was mis-scaled and divide by 100.
    if pct > 25.0: # A 25% yield is a major outlier/data error
        pct = pct / 100.0
    # --- END FIX ---

    score = 10 if pct >= 5 else 7 if pct >= 3 else 4 if pct >= 1 else 0
    desc = f"Div Yield {round(pct,2)}%"
    return {"raw": pct, "value": f"{round(pct,2)}%", "score": int(score), "desc": desc}

@_wrap_calc("quarterly_growth")
def calc_quarterly_growth(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None, lookback_quarters: int = 1):
    """
    Quarterly YoY EPS & Revenue growth — measures near-term business momentum.
    Uses info fields 'earningsQuarterlyGrowth' and 'revenueQuarterlyGrowth' (YoY ratios).
    Fallback: estimates growth from last 2 quarterly financials if available.
    """
    import pandas as pd

    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    eps_growth = None
    rev_growth = None

    # --- Primary: from info fields ---
    for c in FIELD.get("quarterly_growth", ["earningsQuarterlyGrowth", "revenueQuarterlyGrowth"]):
        val = safe_float(info.get(c))
        if val is not None:
            if "earn" in c.lower():
                eps_growth = eps_growth or val
            elif "rev" in c.lower():
                rev_growth = rev_growth or val

    # --- Fallback: estimate from quarterly financials ---
    if eps_growth is None or rev_growth is None:
        qfin = base_data.get("quarterly_financials") if base_data else getattr(t, "quarterly_financials", pd.DataFrame())
        if not getattr(qfin, "empty", True):
            qfin = qfin.T.sort_index(ascending=False)
            if "Net Income" in qfin.columns and len(qfin) >= 2:
                curr, prev = qfin["Net Income"].iloc[0:2]
                if prev not in (None, 0):
                    eps_growth = eps_growth or (curr - prev) / abs(prev)
            if "Total Revenue" in qfin.columns and len(qfin) >= 2:
                curr, prev = qfin["Total Revenue"].iloc[0:2]
                if prev not in (None, 0):
                    rev_growth = rev_growth or (curr - prev) / abs(prev)

    if eps_growth is None and rev_growth is None:
        return None

    eps_g = (eps_growth or 0) * 100
    rev_g = (rev_growth or 0) * 100
    avg_growth = (eps_g + rev_g) / 2 if (eps_growth and rev_growth) else eps_g or rev_g

    # --- Scoring ---
    score = 10 if avg_growth > 20 else 7 if avg_growth > 10 else 5 if avg_growth > 0 else 2
    desc = f"EPS {round(eps_g,2)}%, Rev {round(rev_g,2)}% (avg {round(avg_growth,2)}%)"
    return {
        "raw": {"eps_yoy": eps_g, "rev_yoy": rev_g, "avg": avg_growth},
        "value": desc,
        "score": int(score),
        "desc": desc
    }

@_wrap_calc("short_interest")
def calc_short_interest(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Short interest metrics — days to cover and % of float shorted.
    Provides insight into market sentiment and potential short squeeze setup.
    """
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    short_ratio = None  # days to cover
    short_pct = None    # fraction of float shorted (0.05 = 5%)

    for c in FIELD.get("short_interest", ["shortRatio", "sharesPercentSharesOut", "shortPercentOfFloat"]):
        val = safe_float(info.get(c))
        if val is None:
            continue
        if "ratio" in c.lower():
            short_ratio = val
        elif "percent" in c.lower() or "shares" in c.lower():
            short_pct = val

    if short_ratio is None and short_pct is None:
        return None

    # Normalize % representation
    pct = (short_pct or 0)
    pct = pct * 100 if pct < 1 else pct  # handle 0.12 -> 12%

    # Interpretations:
    # - Low short ratio (<2) = calm
    # - Moderate (2–5) = neutral
    # - High (>5) = potential squeeze zone
    # - Very high (>10) = heavy bearish sentiment
    score = 10 if (short_ratio or 0) < 2 else 7 if (short_ratio or 0) < 5 else 3 if (short_ratio or 0) < 10 else 0
    desc = f"Short Ratio {round(short_ratio or 0,2)}d, Short Float {round(pct,2)}%"
    return {
        "raw": {"ratio": short_ratio, "percent": pct},
        "value": desc,
        "score": int(score),
        "desc": desc
    }

@_wrap_calc("net_profit_margin")
def calc_net_profit_margin(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Net profit margin computed from financials (latest annual)."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    if getattr(fin, "empty", True):
        return None
    # Try to find latest Net Income and Revenue
    ni = _find_from_df_like(fin, FIELD.get("net_income", [])) or _find_from_df_like(fin, ["Net Income", "netIncome", "NetIncome"])
    revenue = _find_from_df_like(fin, FIELD.get("revenue", [])) or _find_from_df_like(fin, ["Total Revenue", "totalRevenue", "Revenue", "Sales"])
    if ni is None or revenue in (None, 0):
        return None
    try:
        pct = safe_div(ni, revenue, None) * 100.0
    except Exception:
        return None
    score = 10 if pct >= 20 else 7 if pct >= 10 else 3 if pct >= 5 else 0
    return {"raw": pct, "value": _fmt_pct(pct), "score": int(score), "desc": f"Net Margin {round(pct,2)}%"}

@_wrap_calc("operating_margin")
def calc_operating_margin(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Operating margin computed from EBIT / Revenue (latest annual)."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    bs = base_data.get("balance_sheet") if base_data else getattr(t, "balance_sheet", pd.DataFrame())
    ebit = _find_from_df_like(fin, FIELD.get("ebit", [])) or _find_from_df_like(fin, ["Operating Income", "EBIT", "Ebit", "operatingIncome"])
    revenue = _find_from_df_like(fin, FIELD.get("revenue", [])) or _find_from_df_like(fin, ["Total Revenue", "totalRevenue", "Revenue", "Sales"])
    if ebit is None or revenue in (None, 0):
        return None
    pct = safe_div(ebit, revenue, None) * 100.0
    score = 10 if pct >= 20 else 7 if pct >= 10 else 3 if pct >= 5 else 0
    return {"raw": pct, "value": _fmt_pct(pct), "score": int(score), "desc": f"Op Margin {round(pct,2)}%"}


@_wrap_calc("ebitda_margin")
def calc_ebitda_margin(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """EBITDA margin from financials or info (if available)."""
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    ebitda = _find_from_df_like(fin, FIELD.get("ebitda", [])) or safe_float(info.get("ebitdaMargins") or info.get("EBITDA"))
    revenue = _find_from_df_like(fin, FIELD.get("revenue", [])) or safe_float(info.get("totalRevenue") or info.get("revenue"))
    if ebitda is None or revenue in (None, 0):
        return None
    pct = safe_div(ebitda, revenue, None) * 100.0
    score = 10 if pct >= 20 else 7 if pct >= 10 else 3 if pct >= 5 else 0
    return {"raw": pct, "value": _fmt_pct(pct), "score": int(score), "desc": f"EBITDA Margin {round(pct,2)}%"}


@_wrap_calc("eps_growth_5y")
def calc_eps_growth_5y(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    EPS 5-year CAGR. Ensure series is chronological (oldest -> newest).
    Uses earnings or financials 'Diluted EPS' if available.
    """
    earnings = base_data.get("earnings") if base_data else getattr(t, "earnings", pd.DataFrame())
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    try:
        eps_series = pd.Series(dtype=float)
        if isinstance(earnings, pd.DataFrame) and not earnings.empty:
            col = next((c for c in earnings.columns if "eps" in c.lower() or "earn" in c.lower()), None)
            if col:
                eps_series = earnings[col].dropna().apply(safe_float)
        if eps_series.dropna().empty and not getattr(fin, "empty", True) and "Diluted EPS" in fin.index:
            # fin.loc["Diluted EPS"] may be series indexed by date; normalize
            ser = fin.loc["Diluted EPS"].dropna().astype(float)
            # convert series to chronological order with oldest first
            eps_series = ser[::-1].reset_index(drop=True).apply(safe_float)
        valid = eps_series.dropna()
        if valid.empty or len(valid) < 2:
            return None
        # ensure chronological oldest->newest
        vals = list(valid)
        print("EPS SERIES:", list(valid))
        print("LENGTH:", len(valid))
        # If the series is length n, compute CAGR between first and last over (n-1) periods
        start = float(vals[0]); end = float(vals[-1])
        if start <= 0 or end <= 0:
            return None
        n = len(vals) - 1
        cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0
        base_data["fields"]["eps_cagr_5y"] = cagr  # cache for reuse
        score = 10 if cagr > 15 else 5 if cagr >= 5 else 0
        return {"raw": cagr, "value": round(cagr, 2), "score": int(score), "desc": f"EPS CAGR {round(cagr,2)}%"}
    except Exception:
        return None


@_wrap_calc("fcf_growth_3y")
def calc_fcf_growth_3y(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    FCF 3-year CAGR calculated from cashflow statement, ensuring chronological order.
    """
    cf = base_data.get("cashflow") if base_data else getattr(t, "cashflow", pd.DataFrame())
    try:
        if getattr(cf, "empty", True):
            return None
        # prefer explicit Free Cash Flow row
        if "Free Cash Flow" in cf.index:
            ser = cf.loc["Free Cash Flow"].dropna().astype(float)
        else:
            ocf = cf.loc["Total Cash From Operating Activities"].dropna() if "Total Cash From Operating Activities" in cf.index else pd.Series(dtype=float)
            capex = cf.loc["Capital Expenditures"].abs().dropna() if "Capital Expenditures" in cf.index else pd.Series(dtype=float)
            if not ocf.empty:
                if not capex.empty:
                    ser = (ocf - capex).dropna().astype(float)
                else:
                    ser = ocf.dropna().astype(float)
            else:
                ser = pd.Series(dtype=float)
        if ser.empty:
            return None
        # ser likely indexed newest->oldest; reverse to oldest->newest
        ser_chron = ser[::-1].reset_index(drop=True)
        # take last up to 4 points (3-year windows depending on available)
        ser_trim = ser_chron.tail(4)
        if len(ser_trim) < 2:
            return None
        start = safe_float(ser_trim.iloc[0]); end = safe_float(ser_trim.iloc[-1])
        if start is None or start <= 0 or end is None:
            return None
        n = len(ser_trim) - 1
        cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0
        score = 10 if cagr > 10 else 5 if cagr >= 5 else 0
        return {"raw": cagr, "value": round(cagr, 2), "score": int(score), "desc": f"FCF CAGR {round(cagr,2)}%"}
    except Exception:
        return None


@_wrap_calc("pe_vs_sector")
def calc_pe_vs_sector(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Compare P/E vs. sector average (uses constants fallback)."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    pe = base_data["fields"].get("pe")
    sector = info.get("sector") or info.get("sectorDisp")
    sector_pe = safe_float(info.get("sectorPE")) or (SECTOR_PE_AVG.get(sector) if sector else None)
    if not (pe and sector_pe):
        return None
    ratio = safe_div(pe, sector_pe)
    score = 10 if ratio < 0.8 else 7 if ratio < 1.0 else 4 if ratio < 1.5 else 1
    return {"raw": ratio, "value": round(ratio, 3), "score": score,
            "desc": f"P/E {round(pe,2)} vs Sector {round(sector_pe,2)} (×{round(ratio,2)})",
            "alias": "P/E vs Sector Avg", "source": "core"}


# 4️⃣ Dividend payout ratio
@_wrap_calc("dividend_payout")
def calc_dividend_payout(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Dividend payout ratio from info (0–100%)."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    payout = safe_float(info.get("payoutRatio"))
    if payout is None:
        return None
    pct = payout * 100 if payout < 1 else payout
    score = 10 if 30 <= pct <= 70 else 7 if 20 <= pct <= 80 else 3 if pct < 20 else 0
    return {"raw": pct, "value": f"{pct:.1f}%", "score": score,
            "desc": f"Payout {pct:.1f}%", "alias": "Dividend Payout (%)", "source": "core"}


# 5️⃣ 5-year avg yield comparison
@_wrap_calc("yield_vs_avg")
def calc_yield_vs_avg(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Compare current dividend yield vs 5-year average."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    y = safe_float(info.get("dividendYield"))
    y5 = safe_float(info.get("fiveYearAvgDividendYield"))
    if not (y and y5):
        return None
    ratio = safe_div(y, y5)
    score = 10 if ratio >= 1.2 else 7 if ratio >= 1.0 else 3 if ratio >= 0.8 else 0
    return {"raw": ratio, "value": round(ratio, 2), "score": score,
            "desc": f"Yield vs 5Y Avg {round(ratio,2)}×", "alias": "Yield vs 5Y Avg", "source": "core"}


# 6️⃣ Analyst Rating with Upside context
@_wrap_calc("analyst_rating")
def calc_analyst_rating(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Analyst rating enriched with price target upside."""
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    key = info.get("recommendationKey")
    mean = safe_float(info.get("recommendationMean"))
    n = safe_float(info.get("numberOfAnalystOpinions"))
    cur = safe_float(info.get("currentPrice"))
    tgt = safe_float(info.get("targetMeanPrice"))
    upside = None
    if cur and tgt:
        upside = (tgt / cur - 1) * 100
    score = 10 if key in ("strong_buy", "buy") else 5 if key == "hold" else 2 if key == "sell" else 0
    desc = f"{key.title() if key else 'N/A'}"
    if upside is not None:
        desc += f" ({upside:+.1f}% tgt)"
    return {"raw": key, "value": key, "score": score, "desc": desc,
            "alias": "Analyst Rating (Momentum)", "source": "core",
            "meta": {"mean": mean, "opinions": n, "upside_pct": round(upside, 2) if upside else None}}

@_wrap_calc("revenue_growth_cagr")
def calc_revenue_growth_cagr(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Revenue CAGR (%) — measures long-term top-line growth.
    Dynamically calculates CAGR based on all available annual data (e.g., 3-5 years).
    """
    try:
        fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
        if getattr(fin, "empty", True):
            raise ValueError("No financials available")

        # 1. Find the first available revenue column
        rev_col = None
        for key in FIELD.get("revenue", ["Total Revenue", "Revenue"]):
            if key in fin.index:
                rev_col = key
                break
        
        if rev_col is None:
            raise ValueError(f"No revenue columns found in financials")

        # 2. Get the series and drop any NaN values
        ser = fin.loc[rev_col].dropna().astype(float)
        
        # 3. Reverse the series to be chronological (oldest -> newest)
        ser_chron = ser[::-1].reset_index(drop=True)
        
        if len(ser_chron) < 2:
            raise ValueError("Insufficient years of revenue data (< 2)")

        # 4. Get start and end values
        start = safe_float(ser_chron.iloc[0])
        end = safe_float(ser_chron.iloc[-1])

        if start is None or end is None or start <= 0:
            raise ValueError("Invalid start/end revenue values or negative base")

        # 5. Dynamically calculate 'n' (number of periods)
        n = len(ser_chron) - 1
        if n <= 0:
            raise ValueError("Not enough periods to calculate growth")

        cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0
        
        score = 10 if cagr >= 20 else 8 if cagr >= 10 else 5 if cagr >= 5 else 2
        
        return {
            "raw": cagr,
            "value": f"{cagr:.2f}%",
            "score": score,
            "desc": f"Revenue CAGR {cagr:.2f}% ({n}yr)",
            "alias": f"Revenue Growth ({n}Y CAGR)",
            "source": "core"
        }
    except Exception as e:
        logger.warning(f"calc_revenue_growth_cagr failed: {e}")
        return None

@_wrap_calc("days_to_earnings")
def calc_days_to_earnings(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """Days until next earnings — used to avoid near-event exposure."""
    from datetime import datetime
    try:
        info = safe_info(getattr(t, "ticker", None)) or getattr(t, "info", {})
        ts = info.get("earningsTimestamp") or info.get("earningsTimestampStart")
        if not ts:
            return None
        days = max((datetime.fromtimestamp(ts) - datetime.now()).days, 0)
        score = 10 if days > 30 else 7 if days > 14 else 5 if days > 7 else 0
        return {
            "raw": days,
            "value": days,
            "score": score,
            "desc": f"{days} days to earnings",
            "alias": "Days to Next Earnings",
            "source": "core"
        }
    except Exception as e:
        logger.warning(f"calc_days_to_earnings failed: {e}")
        return None

@_wrap_calc("ocf_vs_profit")
def calc_ocf_vs_profit(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    OCF vs Net Profit Ratio — measures earnings quality and accrual consistency.
    High values (>1) mean profits are backed by strong cash generation.
    """
    fin = base_data.get("financials") if base_data else getattr(t, "financials", pd.DataFrame())
    cf = base_data.get("cashflow") if base_data else getattr(t, "cashflow", pd.DataFrame())
    if getattr(fin, "empty", True) or getattr(cf, "empty", True):
        return None

    try:
        ocf = _find_from_df_like(cf, FIELD.get("ocf", [])) or _find_from_df_like(cf, ["Total Cash From Operating Activities"])
        net_profit = _find_from_df_like(fin, FIELD.get("net_income", [])) or _find_from_df_like(fin, ["Net Income", "Profit After Tax"])
        if not ocf or not net_profit or net_profit == 0:
            return None

        ratio = safe_div(ocf, net_profit)
        score = 10 if ratio >= 1.2 else 7 if ratio >= 1.0 else 4 if ratio >= 0.8 else 0
        desc = f"OCF/Profit {round(ratio,2)}x"
        return {
            "raw": ratio,
            "value": round(ratio, 2),
            "score": score,
            "desc": desc,
            "alias": "Operating CF vs Net Profit",
            "source": "core"
        }
    except Exception:
        return None

#
# --- ADD THIS NEW FUNCTION to services/fundamentals.py ---
#

@_wrap_calc("promoter_pledge")
def calc_promoter_pledge(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Finds the percentage of promoter shares pledged from ticker.info.
    A high pledge % is a major red flag.
    """
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    
    # yfinance keys can be 'promoterPledge' or 'pledgedPercentage'
    pledge_pct = _find_from_df_like(info, FIELD.get("promoter_pledge", ["pledgedPercentage", "promoterPledge"]))
    
    if pledge_pct is None:
        # Return a neutral score if data is unavailable
        return {"raw": 0.0, "value": "0.0%", "score": 5, "desc": "Pledge data N/A"}

    # Data can be a ratio (0.15) or a percentage (15.0)
    pct = pledge_pct * 100.0 if pledge_pct < 1.0 else pledge_pct

    # Score: Lower is better. 
    # 0% pledge = 10 (Excellent)
    # > 20% pledge = 0 (Very Bad)
    if pct == 0:
        score = 10
    elif pct < 5:
        score = 7
    elif pct < 20:
        score = 3
    else:
        score = 0
        
    desc = f"Promoter Pledge {round(pct, 2)}%"
    return {"raw": pct, "value": f"{round(pct, 2)}%", "score": int(score), "desc": desc}

#
# --- ADD THIS NEW FUNCTION to services/fundamentals.py ---
#

@_wrap_calc("ps_ratio")
def calc_ps_ratio(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Calculates Price-to-Sales (P/S) ratio.
    Uses Market Cap / Total Revenue.
    """
    info = (base_data.get("info") if base_data else safe_info(getattr(t, "ticker", None))) or {}
    fields = base_data.get("fields", {})

    # 1. Get Market Cap
    market_cap = fields.get("market_cap")
    if market_cap is None:
        return None # Cannot calculate without market cap

    # 2. Get Revenue
    revenue = fields.get("revenue")
    if revenue is None or revenue == 0:
        return None # Cannot calculate without revenue
        
    ps_ratio = safe_div(market_cap, revenue)
    if ps_ratio is None:
        return None
        
    # Score: Lower is better.
    # We use different thresholds than P/E, as P/S is often lower.
    if ps_ratio < 1.0:
        score = 10
    elif ps_ratio < 2.5:
        score = 7
    elif ps_ratio < 5.0:
        score = 4
    else:
        score = 0
        
    desc = f"Price-to-Sales: {round(ps_ratio, 2)}x"
    return {"raw": ps_ratio, "value": f"{round(ps_ratio, 2)}x", "score": int(score), "desc": desc}

#
# --- ADD THIS NEW FUNCTION to services/fundamentals.py ---
#

@_wrap_calc("market_cap")
def calc_market_cap(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None):
    """
    Extracts Market Cap. This is a "metric" so it can be used in penalties.
    """
    fields = base_data.get("fields", {})
    market_cap = fields.get("market_cap")
    
    if market_cap is None:
        return None

    # Score (just for display, not really used)
    # We'll score based on size (e.g., > 1 Trillion Cr is 10)
    if market_cap > 1e13:
        score = 10
    elif market_cap > 1e12:
        score = 7
    elif market_cap > 1e11:
        score = 4
    else:
        score = 1
        
    # Format a human-readable value
    if market_cap > 1e12: # Lakh Cr
        val_str = f"{round(market_cap / 1e12, 2)} L Cr"
    elif market_cap > 1e10: # Cr
        val_str = f"{round(market_cap / 1e7, 2)} Cr"
    else:
        val_str = f"{round(market_cap / 1e7, 2)} Cr"

    desc = f"Market Cap: {val_str}"
    # Raw value is the full float, which is what penalties need
    return {"raw": market_cap, "value": val_str, "score": int(score), "desc": desc}

# -------------------------
# Aggregator (compute everything, then weighted aggregation)
# -------------------------
@lru_cache(maxsize=1024)
def compute_fundamentals(symbol: str, apply_market_penalty: bool = True) -> Dict[str, Any]:
    """
    Compute the full fundamentals dict for `symbol`.
    Returns all metrics + base_score (weighted by FUNDAMENTAL_WEIGHTS) + optional market_penalty + final_score.
    """
    logger.info(f"[fundamentals_v3_3] computing for {symbol}")
    t = yf.Ticker(symbol)
    fundamentals: Dict[str, Dict[str, Any]] = {}
        # ✅ 1. Pre-fetch once
    base_data = _prepare_base_data(t)

    METRIC_FUNCTIONS = {
        "pe_ratio": calc_pe_ratio,
        "pb_ratio": calc_pb_ratio,
        "eps_growth_5y": calc_eps_growth_5y,
        "eps_growth_3y": calc_profit_growth_3y,
        "peg_ratio": calc_peg_ratio,
        "roe": calc_roe,
        "roce": calc_roce,
        "roic": calc_roic,
        "de_ratio": calc_de_ratio,
        "interest_coverage": calc_interest_coverage,
        "fcf_yield": calc_fcf_yield,
        "fcf_margin": calc_fcf_margin,
        "current_ratio": calc_current_ratio,
        "asset_turnover": calc_asset_turnover,
        "piotroski_f": calc_piotroski_f,
        "r_d_intensity": calc_rd_intensity,
        "earnings_stability": calc_earnings_stability,
        "fcf_growth_3y": calc_fcf_growth_3y,
        "market_cap_cagr": calc_market_cap_cagr,
        "promoter_holding": calc_promoter_holding,
        "institutional_ownership": calc_institutional_ownership,
        "beta": calc_beta,
        "52w_position": calc_52w_position,
        "dividend_yield": calc_dividend_yield,
        "analyst_rating": calc_analyst_rating,
        "quarterly_growth": calc_quarterly_growth,
        "short_interest": calc_short_interest,
        "net_profit_margin": calc_net_profit_margin,
        "operating_margin": calc_operating_margin,
        "ebitda_margin": calc_ebitda_margin,
        "pe_vs_sector": calc_pe_vs_sector,
        "dividend_payout": calc_dividend_payout,
        "yield_vs_avg": calc_yield_vs_avg,
        "revenue_growth_5y": calc_revenue_growth_cagr,
        "days_to_earnings": calc_days_to_earnings,
        "ocf_vs_profit": calc_ocf_vs_profit,
        "promoter_pledge": calc_promoter_pledge,
        "ps_ratio": calc_ps_ratio,
        "market_cap": calc_market_cap,
    }

    # compute metrics
    for key, func in METRIC_FUNCTIONS.items():
        alias = FUNDAMENTAL_ALIAS_MAP.get(key, key)
        try:
            try:
                res = func(t, base_data=base_data)  # New preferred signature
            except TypeError:
                res = func(t)  # backward compatibility for old functions
            if res is None:
                fundamentals[key] = {"raw": None, "value": "N/A", "score": 0, "desc": f"{key} -> None", "alias": alias, "source": "core"}
            else:
                entry = {
                    "raw": res.get("raw"),
                    "value": res.get("value"),
                    "score": int(res.get("score", 0)),
                    "desc": res.get("desc", ""),
                    "alias": alias,
                    "source": "core"
                }
                if "meta" in res:
                    entry["meta"] = res.get("meta")
                fundamentals[key] = entry
        except Exception as e:
            logger.debug("Metric %s failed for %s: %s", key, symbol, e)
            fundamentals[key] = {"raw": None, "value": "N/A", "score": 0, "desc": "Error", "alias": alias, "source": "core"}

    # Weighted aggregation using FUNDAMENTAL_WEIGHTS.
    total_w = 0.0
    weighted_sum = 0.0
    used_weights = {}
    for k, w in FUNDAMENTAL_WEIGHTS.items():
        m = fundamentals.get(k)
        if not m:
            continue
        val = m.get("value")
        if val in (None, "N/A"):
            # skip truly missing
            continue
        s = safe_float(m.get("score"))
        if s is None:
            continue
        weighted_sum += float(s) * float(w)
        total_w += float(w)
        used_weights[k] = float(w)

    base_score = round(weighted_sum / total_w, 2) if total_w else 0.0

    # compute optional market factor penalty (transparent)
    market_penalty = 0.0
    penalty_reasons = []
    ph = fundamentals.get("promoter_holding", {}).get("raw")
    if ph not in (None, "N/A"):
        try:
            if float(ph) < 10:
                market_penalty += 0.5; penalty_reasons.append("Low promoter holding")
        except Exception:
            pass
    inst = fundamentals.get("institutional_ownership", {}).get("raw")
    if inst not in (None, "N/A"):
        try:
            if float(inst) < 5:
                market_penalty += 0.5; penalty_reasons.append("Low institutional ownership")
        except Exception:
            pass
    b = fundamentals.get("beta", {}).get("raw")
    if b not in (None, "N/A"):
        try:
            if abs(float(b)) > 1.5:
                market_penalty += 0.5; penalty_reasons.append("High beta")
        except Exception:
            pass
    off = fundamentals.get("52w_position", {}).get("raw")
    if off not in (None, "N/A"):
        try:
            if float(off) > 50:
                market_penalty += 1.0; penalty_reasons.append("Deep off-high")
        except Exception:
            pass

    final_score = base_score
    if apply_market_penalty and market_penalty > 0:
        final_score = max(0.0, round(base_score - market_penalty, 2))

    fundamentals["_meta"] = {
        "weights_used": used_weights,
        "total_weight": total_w,
        "metrics_count": len([k for k in fundamentals.keys() if not k.startswith("_")]),
        "penalty_reasons": penalty_reasons,
    }
    fundamentals["base_score"] = base_score
    fundamentals["market_penalty"] = round(market_penalty, 2)
    fundamentals["final_score"] = final_score

    logger.info("[%s] fundamentals computed: base_score=%s final_score=%s", symbol, base_score, final_score)
    return fundamentals


# @_wrap_calc("trend_strength")
# def calc_trend_strength(t: yf.Ticker, base_data: Optional[Dict[str, Any]] = None, lookback_days: int = 180):
#     """
#     Technical-fundamental hybrid: evaluates trend direction and strength.
#     Uses EMA(50) and EMA(200) over recent price history to determine market bias.
#     """
#     import pandas as pd
#     df = safe_history(getattr(t, "ticker", None), period="2y" , interval="1d")
#     if df is None or getattr(df, "empty", True) or "Close" not in df.columns:
#         return None
#     try:
#         close = df["Close"].astype(float)
#         ema50 = close.ewm(span=50).mean().iloc[-1]
#         ema200 = close.ewm(span=200).mean().iloc[-1]
#         price = float(close.iloc[-1])
#         # Compute relative positioning
#         ratio_50 = (price - ema50) / ema50 * 100
#         ratio_200 = (price - ema200) / ema200 * 100
#         if price > ema50 > ema200:
#             signal = "Bullish"
#             score = 10
#         elif price > ema200 and ema50 < ema200:
#             signal = "Reversal"
#             score = 7
#         elif ema50 > ema200 and price < ema50:
#             signal = "Correction"
#             score = 5
#         else:
#             signal = "Bearish"
#             score = 2
#         desc = f"{signal}: P>{'EMA50>EMA200' if ema50>ema200 else 'EMA50<EMA200'}, Δ50={round(ratio_50,1)}%, Δ200={round(ratio_200,1)}%"
#         return {
#             "raw": {"signal": signal, "ema50": ema50, "ema200": ema200, "ratios": {"50": ratio_50, "200": ratio_200}},
#             "value": signal,
#             "score": int(score),
#             "desc": desc
#         }
#     except Exception:
#         return None