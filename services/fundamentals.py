# services/fundamentals.py
import logging
from functools import lru_cache
import math
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional, List, Union
from sqlalchemy.orm import Session
from services.db import SessionLocal, FundamentalCache
import json
import datetime
from services.data_fetch import (
    safe_float, safe_div, safe_info, safe_history, _wrap_calc, 
    _fmt_pct, _retry, safe_get, get_history_for_horizon, safe_info_normalized
)
from config.constants import (
    FUNDAMENTAL_WEIGHTS, 
    FUNDAMENTAL_ALIAS_MAP, 
    FUNDAMENTAL_FIELD_CANDIDATES as FIELD, 
    SECTOR_PE_AVG
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# 🔧 HELPER: SMART YEAR ALIGNMENT & FUZZY PICKING
# ---------------------------------------------------------
def get_smart_aligned_columns(df_primary: pd.DataFrame, df_secondary: pd.DataFrame) -> tuple[Optional[Any], Optional[Any]]:
    """
    Matches columns between two dataframes based on YEAR.
    Returns a tuple (primary_col_key, secondary_col_key).
    
    Solves the issue where Financials has '2023-03-31' but Balance Sheet has '2023-03-30'.
    """
    if df_primary is None or df_secondary is None: return None, None
    if df_primary.empty or df_secondary.empty: return None, None

    # Helper to map years to column keys
    def map_years(df):
        ymap = {}
        for c in df.columns:
            try:
                # Handle pandas Timestamp or string dates
                dt = pd.to_datetime(c)
                ymap[dt.year] = c
            except Exception:
                pass
        return ymap

    ymap_1 = map_years(df_primary)
    ymap_2 = map_years(df_secondary)
    
    # Find common years
    common_years = set(ymap_1.keys()).intersection(set(ymap_2.keys()))
    
    if not common_years:
        return None, None
        
    # Pick the latest common year
    latest_year = max(common_years)
    
    return ymap_1[latest_year], ymap_2[latest_year]

def pick_value_by_key(df: pd.DataFrame, candidates: List[str], col_key: Any) -> Optional[float]:
    """Extracts value for a specific column key with FUZZY row matching."""
    if df is None or df.empty or col_key is None: return None
    if col_key not in df.columns: return None

    # 1. Exact Match
    for name in candidates:
        if name in df.index:
            try: return safe_float(df.loc[name, col_key])
            except: continue

    # 2. Fuzzy Match
    lower_map = {str(i).lower().strip(): i for i in df.index}
    for name in candidates:
        clean_name = str(name).lower().strip()
        if clean_name in lower_map:
            real_idx = lower_map[clean_name]
            try: return safe_float(df.loc[real_idx, col_key])
            except: continue
                
    return None

def pick_latest_column(df: Optional[pd.DataFrame]) -> Optional[int]:
    """Return index of latest full-year values (usually 0)."""
    if df is None or df.empty: return None
    cols = list(df.columns)
    if not cols: return None
    # Check if columns are timestamps to determine order
    if len(cols) > 1:
        try:
            # If col[0] > col[-1], it is Newest->Oldest (Standard)
            if pd.to_datetime(cols[0]) > pd.to_datetime(cols[-1]):
                return 0
            else:
                return len(cols) - 1
        except Exception:
            # Fallback: YFinance standard is usually 0 for latest
            return 0
            
    return 0

def pick_value_from_df(df: Optional[pd.DataFrame], candidates: list, col_index: Optional[int]=None) -> Optional[float]:
    if df is None or df.empty: return None
    if col_index is None: col_index = pick_latest_column(df)
    if col_index is None: return None

    # Attempt to find candidate rows
    for name in candidates:
        if name in df.index:
            try: return safe_float(df.iloc[df.index.get_loc(name), col_index])
            except Exception as e: 
                logger.warning(f"could find candidates: {e}")
                continue
                
    # Fuzzy match (case-insensitive)
    lower_index = {str(i).lower().strip(): i for i in df.index}
    for name in candidates:
        ln = name.lower().strip()
        if ln in lower_index:
            idx = lower_index[ln]
            try:
                val = df.loc[idx].iat[col_index] if hasattr(df.loc[idx], 'iat') else df.loc[idx].iloc[col_index]
                return safe_float(val)
            except Exception as e:
                logger.warning(f"fuzzy match missed: {e}")
                continue
    return None

# ========================================================
# 🚀 DATA UNIFIER CLASS
# ========================================================
class DataUnifier:
    def __init__(self, ticker: yf.Ticker):
        self.ticker = ticker
        self.symbol = getattr(ticker, "ticker", "UNKNOWN")
        self.flat_data = {}
        
        # 1. Use Cached Info Directly
        self.info = getattr(ticker, "info", {}) or {}

        # Improvement D: Log warning if info is empty avoid API hammering in quick score run
        if not self.info:
            logger.warning(f"[{self.symbol}] Fundamentals Warning: 'info' dict is empty (possible rate limit)")

        # 2. Map Dataframes (Zero API calls if main.py initialized ticker correctly)
        self.financials = getattr(ticker, "financials", pd.DataFrame())
        self.balance_sheet = getattr(ticker, "balance_sheet", pd.DataFrame())
        self.cashflow = getattr(ticker, "cashflow", pd.DataFrame())
        self.earnings = getattr(ticker, "earnings", pd.DataFrame())
        self.quarterly_financials = getattr(ticker, "quarterly_financials", pd.DataFrame())

        if self.info:
            self.flat_data.update(self.info)

    def get_raw(self, key: str) -> Any:
        return self.flat_data.get(key)

    def get(self, candidates: Any) -> Optional[float]:
        if isinstance(candidates, str): candidates = [candidates]
        for k in candidates:
            v = self.flat_data.get(k)
            if v is not None:
                f = safe_float(v)
                if f is not None: return f
        return None

    def find_series(self, candidates: List[str], source_df: pd.DataFrame) -> pd.Series:
        # Legacy support for existing functions not yet patched
        if source_df is None or source_df.empty: return pd.Series(dtype=float)
        
        for key in candidates:
            if key in source_df.index: return source_df.loc[key]
        
        # Fuzzy match
        norm_candidates = [str(k).lower().replace(" ", "").replace("_", "") for k in candidates]
        index_map = {str(k).lower().replace(" ", "").replace("_", ""): k for k in source_df.index}
        
        for nc in norm_candidates:
            if nc in index_map: return source_df.loc[index_map[nc]]
            
        return pd.Series(dtype=float)

    def find_value(self, candidates: List[str], source_df: pd.DataFrame) -> Optional[float]:
        return pick_value_from_df(source_df, candidates)

# ========================================================
# METRIC CALCULATORS
# ========================================================
def _get_ebit_robust(unifier: DataUnifier, df_fin: pd.DataFrame, col_key: Any) -> Optional[float]:
    """Try to get EBIT directly, otherwise calculate it: Net Income + Tax + Interest."""
    # 1. Try Direct Fetch
    ebit = pick_value_by_key(df_fin, FIELD.get("ebit", []) + ["Operating Income", "OperatingIncome"], col_key)
    if ebit is not None:
        return ebit
        
    # 2. Manual Calculation: Net Income + Tax + Interest
    ni = pick_value_by_key(df_fin, FIELD.get("net_income", []), col_key)
    tax = pick_value_by_key(df_fin, FIELD.get("tax_expense", []), col_key)
    interest = pick_value_by_key(df_fin, FIELD.get("interest_expense", []), col_key)
    
    if ni is not None:
        val = ni
        if tax is not None: val += tax
        if interest is not None: val += abs(interest) # Interest is often negative expense
        return val
        
    return None


def _derive_wacc(unifier: DataUnifier) -> float:
    try:
        rf = 6.8; mkt_prem = 5.5; tax = 0.25
        beta = safe_float(unifier.get_raw("beta") or unifier.get_raw("beta3Y")) or 1.0
        de_ratio = unifier.get(["debtToEquity"])
        if de_ratio and de_ratio > 50: de_ratio /= 100.0
        
        if de_ratio is None:
            d = unifier.find_value(FIELD.get("total_debt", []), unifier.balance_sheet)
            e = unifier.find_value(FIELD.get("total_equity", []), unifier.balance_sheet)
            de_ratio = d/e if d is not None and e else 0.5

        icr_ebit = unifier.find_value(FIELD.get("ebit", []), unifier.financials)
        icr_int = unifier.find_value(FIELD.get("interest_expense", []), unifier.financials)
        icr = safe_div(icr_ebit, abs(icr_int)) if icr_ebit and icr_int else None

        e_weight = 1 / (1 + de_ratio)
        d_weight = de_ratio / (1 + de_ratio)

        cost_debt = 8.0
        if icr and icr > 0:
            cost_debt = min(max(100.0 / icr, 3.0), 20.0)

        wacc = (e_weight * (rf + beta * mkt_prem)) + (d_weight * cost_debt * (1 - tax))
        return round(wacc, 2)
    except Exception:
        return 7.2

@_wrap_calc("pe_ratio")
def calc_pe_ratio(t, unifier: DataUnifier = None):
    pe = unifier.get(["trailingPE", "forwardPE", "peRatio"])
    
    # Hybrid: Calc from price & latest EPS from financials if Yahoo lacks PE
    if pe is None:
        price = unifier.get(["currentPrice", "regularMarketPrice"])
        # Use pick_value_from_df to get the LATEST EPS only (index 0)
        eps = pick_value_from_df(unifier.financials, ["Basic EPS", "Diluted EPS", "EPS"])
        if price and eps and eps > 0: pe = price / eps

    if pe is None: return None
    score = 10 if pe <= 12 else 7 if pe <= 20 else 4 if pe <= 30 else 1
    return {"raw": pe, "value": round(pe, 2), "score": score, "desc": f"P/E {pe:.2f}"}

@_wrap_calc("pb_ratio")
def calc_pb_ratio(t, unifier: DataUnifier = None):
    pb = unifier.get(["priceToBook", "pbRatio"])
    if pb is None: return None
    score = 10 if pb <= 1.5 else 6 if pb <= 3 else 3 if pb <= 5 else 1
    return {"raw": pb, "value": round(pb, 2), "score": score, "desc": f"P/B {pb:.2f}"}

@_wrap_calc("profit_growth_3y")
def calc_profit_growth_3y(t, unifier: DataUnifier = None):
    fin = unifier.financials
    profit_series = unifier.find_series(FIELD.get("net_income", []), fin)
    
    if profit_series.empty:  return None
    
    valid = profit_series.dropna().astype(float)[::-1] # Oldest -> Newest
    if len(valid) < 2: return None
    
    start, end = valid.iloc[0], valid.iloc[-1]
    if start <= 0 or end <= 0: return None
    
    n = len(valid) - 1
    cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0
    score = 10 if cagr > 15 else 5 if cagr >= 5 else 0
    return {"raw": cagr, "value": round(cagr, 2), "score": score, "desc": f"Profit CAGR {cagr:.1f}% ({n}yr)"}

@_wrap_calc("eps_growth_5y")
def calc_eps_growth_5y(t, unifier: DataUnifier = None):
    fin = unifier.financials
    eps_series = unifier.find_series(["Diluted EPS", "Basic EPS", "EPS (Diluted)"], fin)
    
    if eps_series.empty:
        earnings = unifier.earnings
        if not earnings.empty:
             # Try to find EPS col
             for c in earnings.columns:
                 if "eps" in c.lower():
                     eps_series = earnings[c]; break

    if eps_series.empty: return None
    
    vals = eps_series.dropna().astype(float).tolist()
    # Ensure Oldest -> Newest
    if isinstance(eps_series.index, pd.DatetimeIndex):
        if eps_series.index[0] > eps_series.index[-1]: vals = vals[::-1]
    else:
        if len(vals) < 10: vals = vals[::-1] # Assume Financials are Newest first

    if len(vals) < 2: return None
    start, end = vals[0], vals[-1]
    if start <= 0 or end <= 0: return None
    
    n = len(vals) - 1
    cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0
    unifier.flat_data["eps_cagr_5y"] = cagr
    score = 10 if cagr > 15 else 5 if cagr >= 5 else 0
    return {"raw": cagr, "value": round(cagr, 2), "score": int(score), "desc": f"EPS CAGR {round(cagr,2)}%"}

@_wrap_calc("peg_ratio")
def calc_peg_ratio(t, unifier: DataUnifier = None):
    peg = unifier.get(["pegRatio"])
    if peg is None:
        pe = unifier.get(["trailingPE"])
        if pe and pe > 0:
            growth = unifier.flat_data.get("eps_cagr_5y")
            if growth is None:
                g_res = calc_profit_growth_3y(t, unifier=unifier)
                if g_res and g_res.get("raw"): growth = float(g_res["raw"])
            
            if growth and growth > 0:
                if growth < 1.0: growth *= 100.0
                peg = pe / growth

    if peg is None or peg <= 0 or peg > 50: return None
    score = 10 if peg <= 1 else 8 if peg <= 1.5 else 5 if peg <= 2.5 else 0
    return {"raw": peg, "value": round(peg, 2), "score": score, "desc": f"PEG {peg:.2f}"}

@_wrap_calc("roe")
def calc_roe(t, unifier: DataUnifier = None):
    fin, bs = unifier.financials, unifier.balance_sheet
    col_fin, col_bs = get_smart_aligned_columns(fin, bs)
    ni = pick_value_by_key(fin, FIELD.get("net_income", []), col_fin)
    equity = pick_value_by_key(bs, FIELD.get("total_equity", []), col_bs)

    if equity is None:
        # Fallback to Book Value * Shares
        bv = safe_info_normalized("bookValue", unifier.info)
        shares = safe_info_normalized("sharesOutstanding", unifier.info)
        mcap = safe_info_normalized("marketCap", unifier.info)
        if bv and shares and mcap:
            calc_eq = bv * shares
            # Sanity Check: Equity within 10% - 400% of Market Cap
            if 0.1 < (calc_eq / mcap) < 4.0: equity = calc_eq     
    if ni is None or equity is None or equity == 0:
        roe = unifier.get(["returnOnEquity"])
        if roe: return {"raw": roe*100, "value": f"{roe*100:.2f}%", "score": 5, "desc": "ROE (Yahoo)"}
        return None
        
    pct = (ni / equity) * 100.0
    score = 10 if pct >= 20 else 7 if pct >= 12 else 3
    return {"raw": pct, "value": f"{pct:.2f}%", "score": score, "desc": f"ROE {pct:.2f}%"}

# --- NEW: ROE HISTORY CALCULATOR FOR STABILITY ---
@_wrap_calc("roe_history")
def calc_roe_history(unifier: DataUnifier) -> List[float]:
    """Calculate ROE for available years to feed Signal Engine stability checks."""
    try:
        fin, bs = unifier.financials, unifier.balance_sheet
        if fin.empty or bs.empty: return []
        
        roe_list = []
        # Iterate over columns (Years)
        # Note: Financials cols are usually Timestamps.
        valid_cols = [c for c in fin.columns if c in bs.columns]
        
        for col in valid_cols:
            ni = pick_value_by_key(fin, FIELD.get("net_income", []), col)
            eq = pick_value_by_key(bs, FIELD.get("total_equity", []), col)
            if ni and eq and eq > 0:
                roe_list.append((ni / eq) * 100.0)
                
        # Return sorted by time? No, statistics.pstdev doesn't care about order.
        return {"raw": roe_list, "value": roe_list, "score": None, "desc": "ROE for available years"}
    
    except Exception:
        return []

@_wrap_calc("roce")
def calc_roce(t, unifier: DataUnifier = None):
    fin, bs = unifier.financials, unifier.balance_sheet
    
    # 1. Smart Alignment
    col_fin, col_bs = get_smart_aligned_columns(fin, bs)
    
    # 2. Fallback
    if col_fin is None: col_fin = fin.columns[0] if not fin.empty else None
    if col_bs is None: col_bs = bs.columns[0] if not bs.empty else None

    # 3. Robust Fetch EBIT (Validated by Interest Coverage match)
    ebit = _get_ebit_robust(unifier, fin, col_fin)
    
    # 4. Check Sector
    sector = unifier.get_raw("sector")
    is_financial = sector == "Financial Services"
    
    cap_employed = None
    
    if is_financial:
        # --- BANK FORMULA ---
        # For banks, Capital Employed ≈ Total Assets 
        # (Since Deposits are liabilities used to generate income)
        assets = pick_value_by_key(bs, FIELD.get("total_assets", []), col_bs)
        if assets:
            cap_employed = assets
    else:
        # --- STANDARD FORMULA ---
        assets = pick_value_by_key(bs, FIELD.get("total_assets", []), col_bs)
        cliab = pick_value_by_key(bs, FIELD.get("current_liabilities", []), col_bs)
        cash = pick_value_by_key(bs, FIELD.get("cash_equivalents", []), col_bs) or 0
        
        if assets is not None and cliab is not None:
            cap_employed = (assets - cliab) - cash
        
        # Fallback Method B (Equity + Debt - Cash)
        if cap_employed is None:
            equity = pick_value_by_key(bs, FIELD.get("total_equity", []), col_bs)
            total_debt = pick_value_by_key(bs, FIELD.get("total_debt", []), col_bs)
            if total_debt is None:
                 lt_debt = pick_value_by_key(bs, ["Long Term Debt", "LongTermDebt"], col_bs)
                 total_debt = lt_debt or 0
            if equity is not None:
                cap_employed = (equity + (total_debt or 0)) - cash

    # 5. Final Calculation
    if ebit is not None and cap_employed is not None and cap_employed > 0:
        roce = (ebit / cap_employed) * 100.0
        
        # Scoring: Banks naturally have lower ROCE (5-10% is good), others need 20%
        if is_financial:
            score = 10 if roce >= 1.5 else 7 if roce >= 1.0 else 3 # Adjusted for huge asset base
        else:
            score = 10 if roce >= 20 else 7 if roce >= 15 else 3
            
        return {"raw": roce, "value": f"{roce:.2f}%", "score": score, "desc": f"ROCE {roce:.2f}%"}
            
    # 6. Yahoo Fallback
    roce = unifier.get(["returnOnCapitalEmployed", "roce"])
    if roce: 
        val = roce * 100 if roce < 1 else roce
        return {"raw": val, "value": f"{val:.2f}%", "score": 5, "desc": "ROCE (Yahoo)"}
        
    return None

@_wrap_calc("roic")
def calc_roic(t, unifier: DataUnifier = None):
    fin, bs = unifier.financials, unifier.balance_sheet
    col_fin, col_bs = get_smart_aligned_columns(fin, bs)
    
    if col_fin is None: col_fin = fin.columns[0] if not fin.empty else None
    if col_bs is None: col_bs = bs.columns[0] if not bs.empty else None

    ebit = _get_ebit_robust(unifier, fin, col_fin)
    equity = pick_value_by_key(bs, FIELD.get("total_equity", []), col_bs)
    debt = pick_value_by_key(bs, FIELD.get("total_debt", []), col_bs) or 0
    cash = pick_value_by_key(bs, FIELD.get("cash_equivalents", []), col_bs) or 0

    tax_exp = pick_value_by_key(fin, FIELD.get("tax_expense", []), col_fin)
    pre_tax = pick_value_by_key(fin, FIELD.get("pre_tax_income", []), col_fin)
    
    sector = unifier.get_raw("sector")
    is_financial = sector == "Financial Services"
    
    tax_rate = 0.25
    if tax_exp and pre_tax and pre_tax > 0:
        calc_rate = tax_exp / pre_tax
        if 0.0 < calc_rate < 0.5: tax_rate = calc_rate

    if ebit is None or equity is None: return None
    
    nopat = ebit * (1 - tax_rate)
    
    # SECTOR SWITCH
    if is_financial:
        # Banks: Invested Capital = Equity + Debt (Do not subtract cash)
        invested_capital = equity + debt
    else:
        # Standard: Invested Capital = Equity + Debt - Cash
        invested_capital = (equity + debt) - cash
    
    if invested_capital < 1000: return None

    roic = (nopat / invested_capital) * 100.0
    wacc = _derive_wacc(unifier)
    
    score = 10 if roic > (wacc + 5) else 7 if roic > wacc else 3
    return {"raw": roic, "value": f"{roic:.2f}%", "score": score, "desc": f"ROIC {roic:.1f}% (WACC {wacc:.1f}%)"}

@_wrap_calc("de_ratio")
def calc_de_ratio(t, unifier: DataUnifier = None):
    bs = unifier.balance_sheet
    info = unifier.info
    col_idx = pick_latest_column(bs)
    
    debt = pick_value_from_df(bs, FIELD.get("total_debt", []), col_idx)
    if debt is None:
        lt = pick_value_from_df(bs, FIELD.get("total_debt", []) + ["Long Term Debt"], col_idx)
        st = pick_value_from_df(bs, FIELD.get("pure_borrowings", []) + ["Current Debt"], col_idx)
        if lt is not None or st is not None: debt = (lt or 0) + (st or 0)

    equity = pick_value_from_df(bs, FIELD.get("total_equity", []), col_idx)
    # Fallback A: Assets - Liabilities
    if equity is None:
        assets = pick_value_from_df(bs, FIELD.get("total_assets", []), col_idx)
        liabs = pick_value_from_df(bs, FIELD.get("total_liabilities", []), col_idx)
        if assets is not None and liabs is not None:
            equity = assets - liabs

    # Fallback B: Book Value * Shares (Unit Safe)
    if equity is None:
        bv = safe_info_normalized("bookValue", info)
        shares = safe_info_normalized("sharesOutstanding", info)
        if bv and shares:
            equity = bv * shares
    if equity and equity > 0 and debt is not None:
        de = debt / equity
        
        # SECTOR SCORING ADJUSTMENT
        sector = unifier.get_raw("sector")
        if sector == "Financial Services":
            # Banks run high leverage naturally (D/E 1.27 is actually very safe for a bank)
            # We don't try to match Screener's 11.1 (Deposits), but we score the Borrowings ratio leniently
            score = 10 if de < 2.0 else 7 if de < 4.0 else 4
        else:
            # Standard Companies
            score = 10 if de < 0.5 else 7 if de < 1.0 else 4 if de < 2.0 else 0
            
        return {"raw": de, "value": round(de, 2), "score": score, "desc": f"D/E {de:.2f}"}
    
    yahoo_de = unifier.get(["debtToEquity"])
    if yahoo_de is not None:
        # Yahoo sends 150.5 for 1.505. Normalize if > 50 (known API quirk)
        val = yahoo_de / 100.0 if yahoo_de > 50 else yahoo_de
        score = 10 if val < 0.5 else 7 if val < 1.0 else 4 if val < 2.0 else 0
        return {"raw": val, "value": round(val, 2), "score": score, "desc": f"D/E {val:.2f} (Yahoo)"}

    # 2. If we reach here, we have no calculated debt and no Yahoo debt.
    # Return None so the engine knows data is missing.
    return None

@_wrap_calc("interest_coverage")
def calc_interest_coverage(t, unifier: DataUnifier = None):
    ebit = unifier.find_value(FIELD.get("ebit", []), unifier.financials)
    interest = unifier.find_value(FIELD.get("interest_expense", []), unifier.financials)
    
    if not ebit: return None
    if not interest or interest == 0:
        return {"raw": 999, "value": ">100x", "score": 10, "desc": "Debt-free"}
    icr = ebit / abs(interest)
    score = 10 if icr > 5 else 5 if icr > 2 else 0
    return {"raw": icr, "value": f"{icr:.2f}x", "score": score, "desc": f"Int. Cov {icr:.1f}x"}

@_wrap_calc("fcf_yield")
def calc_fcf_yield(t, unifier: DataUnifier = None):
    fcf = unifier.find_value(FIELD.get("free_cash_flow", []), unifier.cashflow)
    if not fcf:
        ocf = unifier.find_value(FIELD.get("ocf", []), unifier.cashflow)
        capex = unifier.find_value(FIELD.get("capex", []), unifier.cashflow)
        if ocf is not None: fcf = ocf - abs(capex or 0)
            
    mcap = unifier.get(["marketCap"])
    if not fcf or not mcap: return None
    yld = (fcf / mcap) * 100
    score = 10 if yld >= 10 else 6 if yld >= 6 else 4 if yld >= 2 else 0
    return {"raw": yld, "value": f"{yld:.2f}%", "score": score, "desc": f"FCF Yield {yld:.2f}%"}

@_wrap_calc("fcf_margin")
def calc_fcf_margin(t, unifier: DataUnifier = None):
    fin, cf = unifier.financials, unifier.cashflow
    col_fin, col_cf = get_smart_aligned_columns(fin, cf)
    
    rev = pick_value_by_key(fin, FIELD.get("revenue", []), col_fin)
    fcf = pick_value_by_key(cf, FIELD.get("free_cash_flow", []), col_cf)
    
    if fcf is None and col_cf is not None:
        ocf = pick_value_by_key(cf, FIELD.get("ocf", []), col_cf)
        capex = pick_value_by_key(cf, FIELD.get("capex", []), col_cf)
        if ocf is not None: fcf = ocf - abs(capex or 0)
            
    if not rev or not fcf: return None
    margin = (fcf / rev) * 100
    score = 10 if margin >= 15 else 5 if margin >= 10 else 0
    return {"raw": margin, "value": f"{margin:.1f}%", "score": score, "desc": f"FCF Margin {margin:.1f}%"}

@_wrap_calc("current_ratio")
def calc_current_ratio(t, unifier: DataUnifier = None):
    cr = unifier.get(["currentRatio"])
    if not cr:
        ca = unifier.find_value(FIELD.get("current_assets", []), unifier.balance_sheet)
        cl = unifier.find_value(FIELD.get("current_liabilities", []), unifier.balance_sheet)
        if ca and cl: cr = ca / cl
    if not cr: return None
    score = 10 if cr >= 1.5 else 7 if cr >= 1.0 else 2
    return {"raw": cr, "value": round(cr, 2), "score": score, "desc": f"CR {cr:.2f}"}

@_wrap_calc("asset_turnover")
def calc_asset_turnover(t, unifier: DataUnifier = None):
    fin, bs = unifier.financials, unifier.balance_sheet
    col_fin, col_bs = get_smart_aligned_columns(fin, bs)
    rev = pick_value_by_key(fin, FIELD.get("revenue", []), col_fin)
    assets = pick_value_by_key(bs, FIELD.get("total_assets", []), col_bs)
    if not rev or not assets: return None
    at = rev / assets
    score = 10 if at > 1 else 5 if at > 0.5 else 0
    return {"raw": at, "value": round(at, 2), "score": score, "desc": f"Asset Turnover {at:.2f}"}

@_wrap_calc("piotroski_f")
def calc_piotroski_f(t, unifier: DataUnifier = None):
    fin = unifier.financials; bs = unifier.balance_sheet; cf = unifier.cashflow
    if fin.empty or bs.empty: return None
    
    def _get_2y(field_keys, df):
        ser = unifier.find_series(field_keys, df)
        if len(ser) < 2: return None, None
        return safe_float(ser.iloc[0]), safe_float(ser.iloc[1])

    score = 0
    ni, ni_prev = _get_2y(FIELD.get("net_income", []), fin)
    roa = unifier.get(["returnOnAssets"])
    ocf, _ = _get_2y(FIELD.get("ocf", []), cf)
    
    if ni and ni > 0: score += 1
    if ocf and ocf > 0: score += 1
    if roa and roa > 0: score += 1
    if ocf and ni and ocf > ni: score += 1
    
    debt, debt_prev = _get_2y(FIELD.get("total_debt", []), bs)
    if debt is not None and debt_prev is not None and debt < debt_prev: score += 1
    
    ca, ca_prev = _get_2y(FIELD.get("current_assets", []), bs)
    cl, cl_prev = _get_2y(FIELD.get("current_liabilities", []), bs)
    if ca and cl and ca_prev and cl_prev:
        if (ca/cl) > (ca_prev/cl_prev): score += 1
        
    shares, shares_prev = _get_2y(FIELD.get("shares_outstanding", []), fin)
    if shares and shares_prev and shares <= shares_prev: score += 1
    
    rev, rev_prev = _get_2y(FIELD.get("revenue", []), fin)
    cogs, cogs_prev = _get_2y(FIELD.get("cogs", []), fin)
    if rev and cogs and rev_prev and cogs_prev:
        gm = (rev - cogs) / rev
        gm_prev = (rev_prev - cogs_prev) / rev_prev
        if gm > gm_prev: score += 1
    
    assets, assets_prev = _get_2y(FIELD.get("total_assets", []), bs)
    if rev and assets and rev_prev and assets_prev:
        at = rev / assets
        at_prev = rev_prev / assets_prev
        if at > at_prev: score += 1
    
    normalized = round((score / 9) * 10, 1)
    return {"raw": score, "value": f"{score}/9", "score": normalized, "desc": f"F-Score {score}"}

@_wrap_calc("r_d_intensity")
def calc_rd_intensity(t, unifier: DataUnifier = None):
    rd = unifier.find_value(FIELD.get("rd_expense", []), unifier.financials)
    rev = unifier.find_value(FIELD.get("revenue", []), unifier.financials)
    if not rd or not rev: return None
    pct = (rd / rev) * 100.0
    score = 10 if pct > 5 else 5 if pct > 2 else 0
    return {"raw": pct, "value": round(pct, 2), "score": score, "desc": f"R&D {pct:.2f}%"}

@_wrap_calc("earnings_stability")
def calc_earnings_stability(t, unifier: DataUnifier = None):
    fin = unifier.financials
    ser = unifier.find_series(FIELD.get("net_income", []), fin)
    if len(ser) < 3: return None
    vals = ser.dropna().astype(float)
    mean = vals.mean()
    if mean == 0: return None
    cv = abs(vals.std(ddof=0) / mean)
    score = 10 if cv < 0.2 else 6 if cv < 0.5 else 2
    return {"raw": cv, "value": round(cv, 2), "score": score, "desc": f"Earnings CV {cv:.2f}"}

@_wrap_calc("fcf_growth_3y")
def calc_fcf_growth_3y(t, unifier: DataUnifier = None):
    cf = unifier.cashflow
    if cf.empty: return None
    ser = unifier.find_series(FIELD.get("free_cash_flow", []), cf)
    if ser.empty:
        ocf = unifier.find_series(FIELD.get("ocf", []), cf)
        capex = unifier.find_series(FIELD.get("capex", []), cf)
        if not ocf.empty:
            if not capex.empty: ser = ocf - capex.abs()
            else: ser = ocf

    if ser.empty: return None
    valid = ser.dropna().astype(float)[::-1]
    if len(valid) < 2: return None
    start, end = valid.iloc[0], valid.iloc[-1]
    if start <= 0 or end <= 0: return None
    cagr = ((end / start) ** (1.0 / (len(valid) - 1)) - 1.0) * 100.0
    score = 10 if cagr > 10 else 5 if cagr > 5 else 0
    return {"raw": cagr, "value": f"{cagr:.1f}%", "score": int(score), "desc": f"FCF CAGR {cagr:.1f}%"}

@_wrap_calc("market_cap_cagr")
def calc_market_cap_cagr(t, unifier: DataUnifier = None):
    df = get_history_for_horizon(unifier.symbol, "multibagger")
    if df is None or df.empty: return None
    
    close = df["Close"].dropna()
    if len(close) < 24: return None 
    beg = float(close.iloc[0]); end = float(close.iloc[-1])
    if beg <= 0: return None
    years = len(close) / 12
    cagr = ((end / beg) ** (1.0 / years) - 1.0) * 100.0
    score = 10 if cagr >= 25 else 8 if cagr >= 15 else 5 if cagr >= 5 else 0
    return {"raw": cagr, "value": f"{cagr:.1f}%", "score": score, "desc": f"Mkt Cap CAGR {cagr:.1f}%"}

@_wrap_calc("promoter_holding")
def calc_promoter_holding(t, unifier: DataUnifier = None):
    ph = unifier.get(["promoterShare", "heldPercentInsiders"])
    if ph and ph <= 1:
        ph *= 100
    if ph is None: return None
    pct = ph * 100 if ph < 1 else ph
    score = 10 if pct > 40 else 5 if pct > 20 else 0
    return {"raw": pct, "value": round(pct, 2), "score": score, "desc": f"Promoter {pct:.1f}%"}

@_wrap_calc("institutional_ownership")
def calc_institutional_ownership(t, unifier: DataUnifier = None):
    # Yahoo only gives FII — add DIIs via fallback weights
    fii = unifier.get(["heldPercentInstitutions"])
    dii = unifier.get(["diiPercent", "domesticInstitutionalPercent"])
    inst = 0
    if fii:
        inst += (fii * 100 if fii < 1 else fii)
    if dii:
        inst += (dii * 100 if dii < 1 else dii)

    # If only FII present (Yahoo), scale to realistic India values:
    if inst < 10 and fii:
        inst = round((fii * 100) * 1.5, 2)   # heuristic boost for DIIs

    if inst is None: return None
    pct = inst * 100 if inst < 1 else inst
    score = 10 if pct > 30 else 5 if pct > 10 else 0
    return {"raw": pct, "value": round(pct, 2), "score": score, "desc": f"Inst {pct:.1f}%"}

@_wrap_calc("beta")
def calc_beta(t, unifier: DataUnifier = None):
    b = unifier.get(["beta", "beta3Y"])
    if b is None: return None
    score = 10 if abs(b) < 0.8 else 7 if abs(b) < 1 else 3
    return {"raw": b, "value": round(b, 2), "score": score, "desc": f"Beta {b:.2f}"}

@_wrap_calc("52w_position")
def calc_52w_position(t, unifier: DataUnifier = None):
    high = unifier.get(["fiftyTwoWeekHigh"])
    price = unifier.get(["currentPrice"])
    if not high or not price: return None
    off = ((high - price) / high) * 100.0
    score = 10 if off < 10 else 7 if off < 20 else 3
    return {"raw": off, "value": round(off, 2), "score": score, "desc": f"{off:.1f}% off-high"}

@_wrap_calc("dividend_yield")
def calc_dividend_yield(t, unifier: DataUnifier = None):
    dy = unifier.get(["dividendYield"])
    if dy is None: return None
    pct = dy * 100 if dy < 1 else dy
    if pct > 25.0: pct = pct / 100.0
    score = 10 if pct > 3 else 7 if pct > 1 else 0
    return {"raw": pct, "value": f"{pct:.2f}%", "score": score, "desc": f"Yield {pct:.2f}%"}

@_wrap_calc("quarterly_growth")
def calc_quarterly_growth(t, unifier: DataUnifier = None):
    eps_g = unifier.get(["earningsQuarterlyGrowth"])
    rev_g = unifier.get(["revenueQuarterlyGrowth"])
    if not eps_g and not rev_g:
        qfin = unifier.quarterly_financials
        if not qfin.empty:
            ni = unifier.find_series(FIELD.get("net_income", []), qfin)
            rev = unifier.find_series(FIELD.get("revenue", []), qfin)
            if len(ni) >= 2 and ni.iloc[1]: eps_g = (ni.iloc[0] - ni.iloc[1]) / abs(ni.iloc[1])
            if len(rev) >= 2 and rev.iloc[1]: rev_g = (rev.iloc[0] - rev.iloc[1]) / abs(rev.iloc[1])

    eps_pct = (eps_g or 0) * 100; rev_pct = (rev_g or 0) * 100
    avg = (eps_pct + rev_pct) / 2
    score = 10 if avg > 20 else 7 if avg > 10 else 2
    desc = f"Avg {avg:.1f}%"
    return {"raw": avg, "value": desc, "score": score, "desc": desc}

@_wrap_calc("short_interest")
def calc_short_interest(t, unifier: DataUnifier = None):
    ratio = unifier.get(["shortRatio"])
    pct = unifier.get(["shortPercentOfFloat", "sharesPercentSharesOut"])
    if pct: pct = pct * 100 if pct < 1 else pct
    score = 10 if (ratio or 0) < 2 else 5
    desc = f"Short {ratio}d / {round(pct or 0, 1)}%"
    return {"raw": {"ratio": ratio, "percent": pct}, "value": desc, "score": score, "desc": desc}

@_wrap_calc("net_profit_margin")
def calc_net_profit_margin(t, unifier: DataUnifier = None):
    npm = unifier.get(["profitMargins"])
    if not npm:
        fin = unifier.financials
        col_idx = pick_latest_column(fin)
        ni = pick_value_from_df(fin, FIELD.get("net_income", []), col_idx)
        rev = pick_value_from_df(fin, FIELD.get("revenue", []), col_idx)
        if ni and rev: npm = ni / rev
    if npm is None: return None
    pct = npm * 100 if abs(npm) < 5 else npm
    score = 10 if pct > 15 else 7 if pct > 10 else 3
    return {"raw": pct, "value": f"{pct:.1f}%", "score": score, "desc": f"NPM {pct:.1f}%"}

@_wrap_calc("operating_margin")
def calc_operating_margin(t, unifier: DataUnifier = None):
    fin = unifier.financials
    col_idx = pick_latest_column(fin)

    ebit = pick_value_from_df(fin, FIELD.get("ebit", []), col_idx)
    revenue = pick_value_from_df(fin, FIELD.get("revenue", []), col_idx)

    if ebit is not None and revenue is not None and revenue > 0:
        pct = (ebit / revenue) * 100.0
        score = 10 if pct > 15 else 5 if pct > 10 else 3
        return {"raw": pct, "value": f"{pct:.1f}%", "score": score, "desc": f"OPM {pct:.1f}%"}
    
    # Fallback to Yahoo
    opm = unifier.get(["operatingMargins"])
    if opm:
        pct = opm * 100 if abs(opm) < 5 else opm
        return {"raw": pct, "value": f"{pct:.1f}%", "score": 5, "desc": f"OPM {pct:.1f}% (Yahoo)"}
    return None

@_wrap_calc("ebitda_margin")
def calc_ebitda_margin(t, unifier: DataUnifier = None):
    em = unifier.get(["ebitdaMargins"])
    if em:
        pct = em * 100 if abs(em) < 5 else em
    else:
        fin = unifier.financials
        col_idx = pick_latest_column(fin)
        
        ebitda = pick_value_from_df(fin, FIELD.get("ebitda", []), col_idx)
        rev = pick_value_from_df(fin, FIELD.get("revenue", []), col_idx)
        
        if not ebitda or not rev: return None
        pct = (ebitda / rev) * 100
        
    score = 10 if pct > 20 else 7 if pct > 10 else 3
    return {"raw": pct, "value": f"{pct:.1f}%", "score": score, "desc": f"EBITDA Margin {pct:.1f}%"}

@_wrap_calc("pe_vs_sector")
def calc_pe_vs_sector(t, unifier: DataUnifier = None):
    pe = unifier.get(["trailingPE"])
    sector = unifier.get_raw("sector")
    sec_pe = SECTOR_PE_AVG.get(sector)
    if not pe or not sec_pe: return None
    ratio = pe / sec_pe
    score = 10 if ratio < 0.8 else 7 if ratio < 1.0 else 4
    return {"raw": ratio, "value": round(ratio, 2), "score": score, "desc": f"vs Sector {ratio:.2f}x"}

@_wrap_calc("dividend_payout")
def calc_dividend_payout(t, unifier: DataUnifier = None):
    payout = unifier.get(["payoutRatio"])
    if payout is None: return None
    pct = payout * 100 if payout < 1 else payout
    score = 10 if 30 <= pct <= 70 else 7 if 20 <= pct <= 80 else 3
    return {"raw": pct, "value": f"{pct:.1f}%", "score": score, "desc": f"Payout {pct:.1f}%"}

@_wrap_calc("yield_vs_avg")
def calc_yield_vs_avg(t, unifier: DataUnifier = None):
    y = unifier.get(["dividendYield"])
    y5 = unifier.get(["fiveYearAvgDividendYield"])
    if not y or not y5: return None
    ratio = y / y5
    score = 0 if ratio <= 1 else 10
    return {"raw": ratio, "value": round(ratio, 2), "score": score, "desc": "Yield vs 5Y"}

@_wrap_calc("revenue_growth_5y")
def calc_revenue_growth_cagr(t, unifier: DataUnifier = None):
    fin = unifier.financials
    ser = unifier.find_series(FIELD.get("revenue", []), fin)
    if ser.empty: return None
    valid = ser.dropna().astype(float)[::-1]
    if len(valid) < 2: return None
    start, end = valid.iloc[0], valid.iloc[-1]
    n = len(valid) - 1
    cagr = ((end / start) ** (1.0 / n) - 1.0) * 100.0
    if cagr > 10: score = 10
    elif cagr > 8: score = 8
    elif cagr > 5: score = 5
    else: score = 0
    return {"raw": cagr, "value": f"{cagr:.1f}%", "score": score, "desc": f"Rev CAGR {cagr:.1f}%"}

@_wrap_calc("days_to_earnings")
def calc_days_to_earnings(t, unifier: DataUnifier = None):
    # Import class explicitly to avoid module conflict
    from datetime import datetime as dt_class 
    
    ts = unifier.get_raw("earningsTimestamp")
    if not ts: return None
    
    days = max((dt_class.fromtimestamp(ts) - dt_class.now()).days, 0)
    
    score = 10 if days > 30 else 7 if days > 14 else 5
    return {"raw": days, "value": days, "score": score, "desc": f"{days} days to earnings"}

@_wrap_calc("ocf_vs_profit")
def calc_ocf_vs_profit(t, unifier: DataUnifier = None):
    fin, cf = unifier.financials, unifier.cashflow
    col_fin, col_cf = get_smart_aligned_columns(fin, cf)
    ocf = pick_value_by_key(cf, FIELD.get("ocf", []), col_cf)
    ni = pick_value_by_key(fin, FIELD.get("net_income", []), col_fin)
    if not ocf or not ni: return None
    ratio = ocf / ni
    if ratio > 1.5: score = 10
    elif ratio > 1: score = 5
    else: score = 0
    return {"raw": ratio, "value": round(ratio, 2), "score": score, "desc": f"OCF/NI {ratio:.2f}"}

@_wrap_calc("promoter_pledge")
def calc_promoter_pledge(t, unifier: DataUnifier = None):
    pledge = unifier.get(["pledgedPercentage", "promoterPledge"])
    if pledge is None: return {"raw": 0, "value": "0%", "score": 5, "desc": "N/A"}
    pct = pledge * 100 if pledge < 1 else pledge
    score = 10 if pct == 0 else 7 if pct < 5 else 3 if pct < 20 else 0
    return {"raw": pct, "value": f"{pct:.1f}%", "score": score, "desc": f"Pledge {pct:.1f}%"}

@_wrap_calc("ps_ratio")
def calc_ps_ratio(t, unifier: DataUnifier = None):
    ps = unifier.get(["priceToSalesTrailing12Months"])
    if not ps:
        mc = unifier.get(["marketCap"])
        rev = unifier.find_value(FIELD.get("revenue", []), unifier.financials)
        if mc and rev: ps = mc / rev
    if not ps: return None
    score = 10 if ps < 1 else 7 if ps < 2.5 else 4
    return {"raw": ps, "value": f"{round(ps, 2)}x", "score": score, "desc": f"P/S {ps:.2f}x"}

@_wrap_calc("market_cap")
def calc_market_cap(t, unifier: DataUnifier = None):
    mc = unifier.get(["marketCap"])
    if not mc: return None
    score = 10 if mc > 2e12 else 7 if mc > 5e11 else 5 if mc > 5e10 else 1
    if mc > 1e12: val_str = f"{round(mc / 1e12, 2)} L Cr"
    elif mc > 1e7: val_str = f"{round(mc / 1e7, 0)} Cr"
    else: val_str = f"{mc}"
    return {"raw": mc, "value": val_str, "score": score, "desc": val_str}

@_wrap_calc("analyst_rating")
def calc_analyst_rating(t: yf.Ticker, unifier: DataUnifier = None):
    key = unifier.get_raw("recommendationKey")
    score = 10 if key in ("strong_buy", "buy") else 5 if key == "hold" else 2
    return {"raw": key, "value": key, "score": score, "desc": key.title() if key else 'N/A'}

# ========================================================
# AGGREGATOR
# ========================================================

def _compute_fundamentals_core(symbol: str, apply_market_penalty: bool = True) -> Dict[str, Any]:
    logger.info(f"[fundamentals] computing for {symbol}")
    t = yf.Ticker(symbol)
    unifier = DataUnifier(t)
    fundamentals: Dict[str, Dict[str, Any]] = {}

    METRIC_FUNCTIONS = {
        "pe_ratio": calc_pe_ratio, "pb_ratio": calc_pb_ratio, "peg_ratio": calc_peg_ratio,
        "roe": calc_roe, "roce": calc_roce, "roic": calc_roic,
        "de_ratio": calc_de_ratio, "interest_coverage": calc_interest_coverage,
        "fcf_yield": calc_fcf_yield, "current_ratio": calc_current_ratio,
        "piotroski_f": calc_piotroski_f, "promoter_holding": calc_promoter_holding,
        "institutional_ownership": calc_institutional_ownership, "dividend_yield": calc_dividend_yield,
        "market_cap": calc_market_cap, "net_profit_margin": calc_net_profit_margin,
        "operating_margin": calc_operating_margin, "profit_growth_3y": calc_profit_growth_3y,
        "eps_growth_5y": calc_eps_growth_5y, "fcf_growth_3y": calc_fcf_growth_3y,
        "quarterly_growth": calc_quarterly_growth, "ebitda_margin": calc_ebitda_margin,
        "short_interest": calc_short_interest, "pe_vs_sector": calc_pe_vs_sector,
        "dividend_payout": calc_dividend_payout, "yield_vs_avg": calc_yield_vs_avg,
        "revenue_growth_5y": calc_revenue_growth_cagr, "ocf_vs_profit": calc_ocf_vs_profit,
        "promoter_pledge": calc_promoter_pledge, "ps_ratio": calc_ps_ratio,
        "r_d_intensity": calc_rd_intensity, "earnings_stability": calc_earnings_stability,
        "fcf_margin": calc_fcf_margin, "market_cap_cagr": calc_market_cap_cagr,
        "beta": calc_beta, "52w_position": calc_52w_position,
        "analyst_rating": calc_analyst_rating, "days_to_earnings": calc_days_to_earnings,
        "asset_turnover": calc_asset_turnover,
    }

    for key, func in METRIC_FUNCTIONS.items():
        try:
            res = func(t, unifier=unifier)
            alias = FUNDAMENTAL_ALIAS_MAP.get(key, key)
            if not res:
                fundamentals[key] = {"raw": None, "value": "N/A", "score": 0, "desc": f"{alias} -> None", "alias": alias, "source": "core"}
            else:
                if "alias" not in res: res["alias"] = alias
                fundamentals[key] = res
        except Exception: pass
            
    fundamentals["eps_growth_3y"] = fundamentals.get("profit_growth_3y")
    
    # --- ADD MISSING DATA FOR SIGNAL ENGINE ---
    fundamentals["roe_history"] = calc_roe_history(unifier)
    fundamentals["sector"] = unifier.get_raw("sector")
    fundamentals["industry"] = unifier.get_raw("industry")
    fundamentals["website"] = unifier.get_raw("website")
    fundamentals["current_price"] = unifier.get(["currentPrice", "regularMarketPrice"])

    total_w = 0.0; weighted_sum = 0.0; used_weights = {}
    for k, w in FUNDAMENTAL_WEIGHTS.items():
        m = fundamentals.get(k)
        if not m or m.get("value") in (None, "N/A"): continue
        s = safe_float(m.get("score"))
        if s is not None:
            weighted_sum += float(s) * float(w)
            total_w += float(w)
            used_weights[k] = float(w)

    base_score = round(weighted_sum / total_w, 2) if total_w else 0.0
    market_penalty = 0.0
    penalty_reasons = []
    
    ph_entry = fundamentals.get("promoter_holding")
    # Only penalize if we are SURE it's low (e.g., > 0 but < 10). 
    # If it is exactly 0 or None, assume data missing to avoid false negatives.
    raw_ph = ph_entry.get("raw", 0) if ph_entry else 0
    
    if raw_ph and 0 < raw_ph < 10:  # Changed condition
        market_penalty += 0.5
        penalty_reasons.append("Low Promoter Holding")
        
    inst_entry = fundamentals.get("institutional_ownership")
    if inst_entry and inst_entry.get("raw", 0) < 5:
        market_penalty += 0.5
        penalty_reasons.append("Low Inst. Holding")

    final_score = base_score
    if apply_market_penalty:
        final_score = max(0.0, round(base_score - market_penalty, 2))

    fundamentals["_meta"] = {
        "weights_used": used_weights,
        "total_weight": total_w,
        "penalty_reasons": penalty_reasons
    }
    fundamentals["base_score"] = base_score
    fundamentals["market_penalty"] = market_penalty
    fundamentals["final_score"] = final_score
    fundamentals["symbol"] = symbol

    return fundamentals

def compute_fundamentals(symbol: str, apply_market_penalty: bool = True) -> Dict[str, Any]:
    """
    DB-Cached wrapper. Uses SQLite (trade.db) instead of JSON files.
    TTL: 24 Hours.
    """
    symbol = symbol.strip().upper()
    db: Session = SessionLocal()
    try:
        # 1. READ CACHE FROM DB
        entry = db.query(FundamentalCache).filter(FundamentalCache.symbol == symbol).first()
        if entry:
            age = (datetime.datetime.now() - entry.updated_at).total_seconds()
            if age < (24 * 3600):
                # Valid Cache Hit
                # SQLAlchemy automatically converts the JSON column back to a Dict
                return entry.data

        # 2. FETCH FRESH (If cache missing or stale)
        data = _compute_fundamentals_core(symbol, apply_market_penalty)
        if data and len(data) > 5:
            if entry:
                # Update existing
                entry.data = data
                entry.updated_at = datetime.datetime.now()
            else:
                # Insert new
                entry = FundamentalCache(
                    symbol=symbol,
                    data=data,
                    updated_at=datetime.datetime.now()
                )
                db.add(entry)
            db.commit()
        return data
    except Exception as e:
        logger.error(f"DB Cache Error for {symbol}: {e}")
        db.rollback()
        # Fallback: Just calculate and return without saving if DB fails
        return _compute_fundamentals_core(symbol, apply_market_penalty)
    finally:
        db.close()