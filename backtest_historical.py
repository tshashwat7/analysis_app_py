"""
backtest_historical.py
═══════════════════════════════════════════════════════════════════════════════
Pro Stock Analyzer v15.2 — Historical Signal Replay Backtester
═══════════════════════════════════════════════════════════════════════════════

WHAT THIS FILE DOES
-------------------
Replays past OHLCV data bar-by-bar through your REAL pipeline and measures:

  1. FORWARD OUTCOME    — After BUY/SELL, did price reach T1, T2, or SL?
  2. SIGNAL CONSISTENCY — Same bar → same signal across multiple runs?
  3. SIGNAL TRANSITIONS — Every BUY↔WATCH/BLOCKED flip with exact cause.
  4. HORIZON AGREEMENT  — Do intraday + short_term + long_term agree per day?

ALL DICT PATHS VERIFIED AGAINST ACTUAL SOURCE CODE
---------------------------------------------------
config_helpers.py
  build_evaluation_context(ticker=, indicators=, fundamentals=, horizon=, patterns=)
  build_execution_context(eval_ctx=, capital=)
  get_setup_from_context(eval_ctx)        → (setup_type, priority, meta)
  get_confidence_from_context(eval_ctx)   → (clamped_int, meta_dict)
  check_gates_from_context(eval_ctx, conf)→ {passed, failed_gates, gate_details}
  get_strategy_from_context(eval_ctx)     → {primary_strategy, fit_score, ...}

config_resolver._build_evaluation_context returns ctx with keys:
  ctx["setup"]["type"]
  ctx["confidence"]["clamped"]           — the number to use
  ctx["confidence"]["base"]
  ctx["confidence"]["final"]             — pre-clamp
  ctx["confidence"]["adjustments"]["breakdown"]  — list of strings
  ctx["trend"]["classification"]["direction"]    — "bullish"/"bearish"/"neutral"
  ctx["strategy"]["primary"]
  ctx["structural_gates"]["overall"]["passed"]   — bool
  ctx["structural_gates"]["overall"]["failed_gates"]  — [{"gate":name,"reason":..}]
  ctx["execution_rules"]["overall"]["passed"]
  ctx["opportunity_gates"]["overall"]["passed"]

config_resolver._build_execution_context returns:
  exec_ctx["can_execute"]["can_execute"]      — bool (NOT a bare bool)
  exec_ctx["can_execute"]["is_hard_blocked"]  — bool
  exec_ctx["risk"]["entry_price"]
  exec_ctx["risk"]["stop_loss"]
  exec_ctx["risk"]["targets"]                 — list [t1, t2]
  exec_ctx["risk"]["rrRatio"]
  exec_ctx["market_adjusted_targets"]["execution_entry"]
  exec_ctx["market_adjusted_targets"]["execution_sl"]
  exec_ctx["market_adjusted_targets"]["execution_t1"]
  exec_ctx["market_adjusted_targets"]["execution_t2"]
  exec_ctx["market_adjusted_targets"]["execution_rr_t1"]
  exec_ctx["market_adjusted_targets"]["execution_rr_t2"]
  exec_ctx["market_adjusted_targets"]["adjusted"]  — bool

signal_engine.compute_all_profiles(ticker=, fundamentals=, indicators_by_horizon=,
                                   patterns_by_horizon=, requested_horizons=)
  → {"ticker", "best_fit", "profiles": {horizon: profile_dict}}
  profile_dict["profile_signal"]          — "STRONG"/"MODERATE"/"WEAK"/"AVOID"
  profile_dict["eval_ctx"]                — full eval_ctx
  profile_dict["status"]                  — "SUCCESS" or "FAILED"
  profile_dict["final_score"]

signal_engine.generate_trade_plan(symbol=, winner_profile=, indicators=,
                                  fundamentals=, horizon=, capital=)
  plan["trade_signal"]                    — "BUY"/"SELL"/"WATCH"/"BLOCKED"
  plan["entry"]                           — NOT "entry_price"
  plan["stop_loss"]
  plan["targets"]["t1"]                   — NOT "target_1"
  plan["targets"]["t2"]                   — NOT "target_2"
  plan["final_confidence"]                — NOT "confidence"
  plan["rr_ratio"]
  plan["setup_type"]
  plan["can_trade"]                       — bool
  plan["execution_blocked"]               — bool
  plan["metadata"]["direction"]           — "bullish"/"bearish"/"neutral"
  plan["confidence_breakdown"]["base_floor"]
  plan["confidence_breakdown"]["final_unclamped"]
  plan["confidence_breakdown"]["clamped"]

USAGE
-----
  # Step 1 — download & cache (one-time, needs network)
  python backtest_historical.py --fetch

  # Step 2 — run backtest (fully offline from here)
  python backtest_historical.py

  # Specific symbols / horizons
  python backtest_historical.py --symbols RELIANCE.NS INFY.NS
  python backtest_historical.py --horizons short_term long_term

  # All 20 Nifty 50 symbols
  python backtest_historical.py --all-nifty50

  # Quick smoke test (10 bars per symbol × horizon)
  python backtest_historical.py --quick

  # Print every BUY/SELL bar as it fires
  python backtest_historical.py --verbose

  # Re-print last summary without re-running
  python backtest_historical.py --report

NOTES
-----
  • Intraday data = 5d × 5m from yfinance. Re-run --fetch weekly to refresh.
  • short_term = 2y × 1d, long_term = 5y × 1wk — stable for months.
  • Fundamentals cached as JSON in backtest_cache/fundamentals/.
  • Pattern analysis is disabled by default (ENABLE_PATTERN_ANALYSIS = False).
    Enable it for richer signals at ~10× slower per-bar throughput.
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import sys
import time
import traceback
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT      = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

CACHE_DIR   = ROOT / "backtest_cache"
OHLCV_DIR   = CACHE_DIR / "ohlcv"
FUND_DIR    = CACHE_DIR / "fundamentals"
RESULTS_DIR = ROOT / "backtest_results"
for _d in [CACHE_DIR, OHLCV_DIR, FUND_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("backtest_historical")

ENABLE_PATTERN_ANALYSIS = False   # Set True for real pattern detection (~10× slower)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — UNIVERSE & HORIZON CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

NIFTY50_SYMBOLS = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","INFY.NS","HINDUNILVR.NS",
    "ICICIBANK.NS","KOTAKBANK.NS","BHARTIARTL.NS","ITC.NS","AXISBANK.NS",
    "LT.NS","BAJFINANCE.NS","ASIANPAINT.NS","MARUTI.NS","TITAN.NS",
    "WIPRO.NS","ULTRACEMCO.NS","SUNPHARMA.NS","NESTLEIND.NS","TECHM.NS",
]
NIFTY_NEXT50_SYMBOLS = [
    "ADANIPORTS.NS","BAJAJFINSV.NS","GODREJCP.NS","HAVELLS.NS",
    "PIDILITIND.NS","SIEMENS.NS","TATAPOWER.NS","TORNTPHARM.NS","VEDL.NS",
]

HORIZON_CONFIG: Dict[str, Dict] = {
    "intraday":   {"period":"5d",  "interval":"5m",  "warmup_bars":100, "forward_window":78,
                   "ema_fast":9,   "ema_slow":21,  "atr_period":7},
    "short_term": {"period":"2y",  "interval":"1d",  "warmup_bars":60,  "forward_window":20,
                   "ema_fast":20,  "ema_slow":50,  "atr_period":14},
    "long_term":  {"period":"5y",  "interval":"1wk", "warmup_bars":40,  "forward_window":12,
                   "ema_fast":50,  "ema_slow":200, "atr_period":21},
}

SIGNAL_STATES   = {"BUY","STRONG_BUY","SELL","WATCH","BLOCKED","AVOID","HOLD"}
OUTCOME_T1      = "T1_HIT"
OUTCOME_T2      = "T2_HIT"
OUTCOME_SL      = "SL_HIT"
OUTCOME_EXPIRED = "EXPIRED"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_and_cache_ohlcv(symbol: str, horizon: str) -> Optional[Path]:
    cfg  = HORIZON_CONFIG[horizon]
    slug = symbol.replace(".", "_")
    path = OHLCV_DIR / f"{slug}_{horizon}.parquet"
    if path.exists():
        return path
    try:
        import yfinance as yf
        df = yf.Ticker(symbol).history(period=cfg["period"], interval=cfg["interval"])
        if df is None or len(df) == 0:
            print(f"  [WARN] No data for {symbol} {horizon}"); return None
        df.to_parquet(path)
        print(f"  Cached {symbol} {horizon}: {len(df)} bars")
        return path
    except Exception as e:
        print(f"  [ERROR] {symbol} {horizon}: {e}"); return None


def fetch_and_cache_fundamentals(symbol: str) -> Dict[str, Any]:
    slug = symbol.replace(".", "_")
    path = FUND_DIR / f"{slug}.json"
    if path.exists():
        with open(path) as f: return json.load(f)
    try:
        import yfinance as yf
        info = yf.Ticker(symbol).info or {}
        fund = {
            "trailingPE":              info.get("trailingPE") or 20.0,
            "priceToBook":             info.get("priceToBook") or 2.5,
            "returnOnEquity":          (info.get("returnOnEquity") or 0.15) * 100,
            "returnOnCapitalEmployed": (info.get("returnOnAssets") or 0.10) * 150,
            "debtToEquity":            (info.get("debtToEquity") or 50.0) / 100,
            "promoterHolding": 55.0, "piotroskiF": 6.0,
            "epsGrowth5y":     (info.get("earningsGrowth") or 0.12) * 100,
            "marketCap":       info.get("marketCap") or 5_000_000_000,
            "marketCapCr":     (info.get("marketCap") or 5_000_000_000) / 1e7,
            "revenueGrowth":   (info.get("revenueGrowth") or 0.10) * 100,
            "sector":          info.get("sector") or "Technology",
            "industry":        info.get("industry") or "Technology",
            "symbol": symbol, "high52w": info.get("fiftyTwoWeekHigh") or 0,
            "position52w": 50.0,
        }
        with open(path, "w") as f: json.dump(fund, f, indent=2)
        return fund
    except Exception as e:
        log.warning(f"Fundamentals failed {symbol}: {e}")
        return {"trailingPE":20,"priceToBook":2.5,"returnOnEquity":15,
                "returnOnCapitalEmployed":14,"debtToEquity":0.5,
                "promoterHolding":55,"piotroskiF":6,"epsGrowth5y":12,
                "marketCap":5_000_000_000,"marketCapCr":500,"revenueGrowth":10,
                "sector":"Technology","industry":"Technology","symbol":symbol,
                "high52w":0,"position52w":50.0}


def load_ohlcv(symbol: str, horizon: str) -> Optional[Any]:
    slug = symbol.replace(".", "_")
    path = OHLCV_DIR / f"{slug}_{horizon}.parquet"
    if not path.exists(): return None
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        df.columns = [c.strip().title() for c in df.columns]
        return df
    except Exception as e:
        log.error(f"Load failed {path}: {e}"); return None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — INDICATOR COMPUTATION
# Mirrors services/indicators.py polymorphic lookback table exactly.
# Returns nested {"value":x,"score":y,"raw":x} dicts compatible with
# config_helpers.flatten_market_data_mixed() and data_fetch._get_val().
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators_from_ohlcv(df: Any, horizon: str) -> Dict[str, Any]:
    cfg   = HORIZON_CONFIG[horizon]
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)
    n     = len(close)

    def _L(s) -> float:
        try:
            v = s.iloc[-1]; return float(v) if v==v else 0.0
        except: return 0.0

    # EMAs
    ef = close.ewm(span=min(cfg["ema_fast"],n-1), adjust=False).mean()
    es = close.ewm(span=min(cfg["ema_slow"],n-1), adjust=False).mean()
    mm = close.rolling(min(30,n-1)).mean()
    ef_v, es_v, px = _L(ef), _L(es), _L(close)

    # ATR
    pc  = close.shift(1)
    tr  = (high-low).abs().combine((high-pc).abs(),max).combine((low-pc).abs(),max)
    atr = tr.rolling(min(cfg["atr_period"],n-1)).mean()
    atr_v = max(_L(atr), 0.001)
    atr_pct = (atr_v/px*100) if px>0 else 3.0

    # RSI
    d    = close.diff()
    g    = d.clip(lower=0).rolling(14).mean()
    l_   = (-d.clip(upper=0)).rolling(14).mean()
    rs   = g / l_.replace(0.0, float("nan"))
    rsi_ = 100.0 - (100.0/(1.0+rs))
    rsi_v= _L(rsi_)
    if rsi_v!=rsi_v: rsi_v=50.0

    # RSI slope (used by _build_momentum_context as "rsislope")
    rsi_slope = float(rsi_.diff(5).iloc[-1]) if len(rsi_)>5 else 0.0
    if rsi_slope!=rsi_slope: rsi_slope=0.0

    # MACD
    e12 = close.ewm(span=min(12,n-1),adjust=False).mean()
    e26 = close.ewm(span=min(26,n-1),adjust=False).mean()
    ml  = e12-e26
    ms  = ml.ewm(span=9,adjust=False).mean()
    mh  = ml-ms

    # Bollinger
    bm = close.rolling(min(20,n-1)).mean()
    bs_ = close.rolling(min(20,n-1)).std()
    bu = bm+2*bs_; bd=bm-2*bs_
    bw = (bu-bd)/bm.replace(0.0,float("nan"))
    bp = (close-bd)/(bu-bd).replace(0.0,float("nan"))
    bp_v = max(0.0, min(1.0, _L(bp)))

    # Volume
    vm30 = vol.rolling(min(30,n-1)).mean()
    vv, vm_v = _L(vol), _L(vm30)
    rvol = vv/max(vm_v,1.0)
    if rvol>=3.0:   vs="climax"
    elif rvol>=1.8: vs="surge"
    elif rvol<=0.6: vs="drought"
    else:           vs="normal"

    # ADX
    adx_v = _compute_adx(high, low, close, period=min(14,n-2))

    # regSlope (linear regression slope normalised to % per bar)
    try:
        import numpy as np
        y = close.values[-min(20,n):]
        x = np.arange(len(y))
        reg_slope = float(np.polyfit(x,y,1)[0]) / max(float(y[-1]),1) * 100 if len(y)>=3 else 0.0
    except: reg_slope=0.0

    # Composite quality scores
    ts = _score_trend(ef_v, es_v, adx_v, rsi_v)
    ms_ = _score_momentum(rsi_v, _L(mh), rvol)
    vq = _score_volatility(atr_pct, _L(bw), rvol)

    def _w(v,s): return {"value":v,"score":s,"raw":v}

    return {
        # Trend
        "ema_fast":          _w(ef_v,  ts),
        "ema_slow":          _w(es_v,  ts),
        "mma":               _w(_L(mm),ts),
        "wma":               _w(ef_v,  ts),
        "adx":               _w(adx_v, min(adx_v/5,10)),
        "trendStrength":     _w(ts,    ts),
        "trend_direction":   {"value":"BULLISH" if ef_v>es_v else "BEARISH",
                              "raw":"BULLISH" if ef_v>es_v else "BEARISH"},
        # regSlope is what _build_trend_context reads to determine direction
        "regSlope":          _w(reg_slope, 5.0),
        "rsislope":          _w(rsi_slope, 5.0),
        # Momentum
        "rsi":               _w(rsi_v,  rsi_v/10),
        "macd":              _w(_L(ml), ms_),
        "macd_signal":       _w(_L(ms), ms_),
        "macdhistogram":     _w(_L(mh), ms_),
        "momentumStrength":  _w(ms_,   ms_),
        # Volatility
        "atr":               _w(atr_v, vq),
        "atrDynamic":        _w(atr_v, vq),   # _finalize_risk_model reads atrDynamic
        "atrPct":            _w(atr_pct,vq),
        "bbwidth":           _w(_L(bw), vq),
        "bbHigh":            _w(_L(bu), vq),
        "bbMid":             _w(_L(bm), vq),
        "bbLow":             _w(_L(bd), vq),
        "bbpercentb":        _w(bp_v,  bp_v*10),
        "volatilityQuality": _w(vq,    vq),
        # Volume
        "volume":            _w(vv,   min(rvol*3,10)),
        "rvol":              _w(rvol, min(rvol*3,10)),
        "avg_volume_30Days": _w(vm_v, 5.0),
        "volume_signature":  {"value":vs,"raw":vs},
        # Price — _extract_price_data reads "price" and "prev_close"
        "price":             _w(px,           5.0),
        "prev_close":        _w(_L(close.shift(1)), 5.0),
        "close":             _w(px,           5.0),
        "open":              _w(_L(df["Open"].astype(float)), 5.0),
        "high":              _w(_L(high),     5.0),
        "low":               _w(_L(low),      5.0),
        # Meta
        "marketTrendScore":  _w(ts,  ts),
        "technicalScore":    _w(ts,  ts),
        # Support / resistance (ATR-based approximations for gate context)
        "resistance1":       _w(px+atr_v*1.5, 5.0),
        "resistance2":       _w(px+atr_v*3.0, 5.0),
        "support1":          _w(px-atr_v*1.5, 5.0),
        "support2":          _w(px-atr_v*3.0, 5.0),
        "atrDynamic":        _w(atr_v, vq),
    }


def _compute_adx(high,low,close,period=14) -> float:
    try:
        import numpy as np
        h,l,c = high.values.astype(float),low.values.astype(float),close.values.astype(float)
        n=len(c)
        if n<period+2: return 20.0
        tr,pdm,ndm=np.zeros(n),np.zeros(n),np.zeros(n)
        for i in range(1,n):
            tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
            up,dn=h[i]-h[i-1],l[i-1]-l[i]
            pdm[i]=up if up>dn and up>0 else 0.0
            ndm[i]=dn if dn>up and dn>0 else 0.0
        def W(a):
            o=np.zeros(n);o[period]=a[1:period+1].sum()
            for i in range(period+1,n):o[i]=o[i-1]-o[i-1]/period+a[i]
            return o
        a14,p14,nm14=W(tr),W(pdm),W(ndm)
        with np.errstate(divide="ignore",invalid="ignore"):
            pdi=np.where(a14>0,100*p14/a14,0.0)
            ndi=np.where(a14>0,100*nm14/a14,0.0)
            dx=np.where((pdi+ndi)>0,100*np.abs(pdi-ndi)/(pdi+ndi),0.0)
        adx=W(dx)
        return max(0.0,min(100.0,float(adx[-1])))
    except: return 20.0

def _score_trend(ef,es,adx,rsi):
    s=5.0
    if ef>es: s+=1.5+min((ef-es)/max(es,1)*30,1.5)
    else:     s-=1.5
    if adx>=35:   s+=2.0
    elif adx>=25: s+=1.0
    elif adx<15:  s-=1.5
    if rsi>60:    s+=0.5
    elif rsi<40:  s-=0.5
    return max(0.0,min(10.0,s))

def _score_momentum(rsi,mh,rvol):
    s=5.0
    if 50<rsi<=70:  s+=1.5
    elif rsi>70:    s+=0.5
    elif rsi<30:    s-=1.5
    elif rsi<50:    s-=0.5
    if mh>0:        s+=1.0
    else:           s-=1.0
    if rvol>=1.8:   s+=1.0
    elif rvol<0.7:  s-=1.0
    return max(0.0,min(10.0,s))

def _score_volatility(atr_pct,bw,rvol):
    s=7.0
    if atr_pct>6.0:  s-=2.5
    elif atr_pct>4:  s-=1.0
    if bw>0.15:      s-=1.0
    elif bw<0.04:    s-=0.5
    if rvol>3.0:     s-=1.0
    return max(0.0,min(10.0,s))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — BAR SIGNAL DATACLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class BarSignal:
    symbol:str; horizon:str; bar_date:str; bar_idx:int
    # Signal
    signal:str; profile_signal:str; setup_type:str; strategy:str
    direction:str; can_execute:bool; is_hard_blocked:bool
    # Confidence — verified keys from eval_ctx["confidence"]
    confidence:float          # clamped
    conf_base:float           # base
    conf_pre_clamp:float      # final (pre-clamp)
    conf_breakdown:List[str]  # adjustments.breakdown list
    # Targets — plan["entry"], plan["stop_loss"], plan["targets"]["t1"/"t2"]
    entry:Optional[float]; stop_loss:Optional[float]
    target_1:Optional[float]; target_2:Optional[float]
    rr_ratio:Optional[float]; sl_atr_mult:Optional[float]
    # Gate audit — from eval_ctx["structural_gates"]["overall"]
    struct_gates_passed:bool
    struct_gates_failed:List[str]   # gate names only
    exec_rules_passed:bool
    opp_gates_passed:bool
    # Meta
    close_price:float; atr:float; est_time_str:str; pipeline_mode:str; error:Optional[str]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — PIPELINE ADAPTER
# All extraction paths verified against real source files.
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineAdapter:

    def __init__(self, capital:float=100_000.0):
        self.capital = capital
        self._helpers = None
        self._signal_engine = None
        self.mode = "isolated"
        self._try_import()

    def _try_import(self):
        try:
            from config import config_helpers
            self._helpers = config_helpers
            self.mode = "config_helpers"
        except ImportError as e:
            log.debug(f"config_helpers: {e}")
        try:
            from services import signal_engine
            self._signal_engine = signal_engine
            self.mode = "full"
        except ImportError as e:
            log.debug(f"signal_engine: {e}")

    def run_bar(self, symbol, horizon, indicators, fundamentals, patterns,
                bar_idx, bar_date) -> BarSignal:
        price = _nv(indicators,"price",150.0)
        atr   = _nv(indicators,"atrDynamic",_nv(indicators,"atr",5.0))
        try:
            if self.mode=="full":
                return self._run_full(symbol,horizon,indicators,fundamentals,
                                      patterns,bar_idx,bar_date,price,atr)
            elif self.mode=="config_helpers":
                return self._run_helpers(symbol,horizon,indicators,fundamentals,
                                         patterns,bar_idx,bar_date,price,atr)
            else:
                return self._run_isolated(symbol,horizon,indicators,fundamentals,
                                          bar_idx,bar_date,price,atr)
        except Exception:
            return self._err(symbol,horizon,bar_idx,bar_date,price,atr,traceback.format_exc())

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def _run_full(self,symbol,horizon,indicators,fundamentals,patterns,
                  bar_idx,bar_date,price,atr) -> BarSignal:
        se = self._signal_engine
        # Verified signature: compute_all_profiles(ticker=, fundamentals=,
        #   indicators_by_horizon=, patterns_by_horizon=, requested_horizons=)
        full = se.compute_all_profiles(
            ticker=symbol,
            fundamentals=fundamentals,
            indicators_by_horizon={horizon:indicators,"long_term":indicators},
            patterns_by_horizon={horizon:patterns,"long_term":{}},
            requested_horizons=[horizon],
        )
        profiles  = full.get("profiles",{})
        h_prof    = profiles.get(horizon,{})
        eval_ctx  = h_prof.get("eval_ctx",{})
        winner    = h_prof if h_prof.get("status")=="SUCCESS" else None

        # Verified signature: generate_trade_plan(symbol=, winner_profile=,
        #   indicators=, fundamentals=, horizon=, capital=)
        plan = se.generate_trade_plan(
            symbol=symbol, winner_profile=winner,
            indicators=indicators, fundamentals=fundamentals,
            horizon=horizon, capital=self.capital,
        )
        return self._from_plan(symbol,horizon,bar_idx,bar_date,price,atr,
                               plan,eval_ctx,h_prof,"full")

    # ── config_helpers only ───────────────────────────────────────────────────

    def _run_helpers(self,symbol,horizon,indicators,fundamentals,patterns,
                     bar_idx,bar_date,price,atr) -> BarSignal:
        h = self._helpers
        # Verified: build_evaluation_context(ticker=,indicators=,fundamentals=,horizon=,patterns=)
        eval_ctx = h.build_evaluation_context(
            ticker=symbol, indicators=indicators,
            fundamentals=fundamentals, horizon=horizon, patterns=patterns,
        )
        # Verified: build_execution_context(eval_ctx=, capital=)
        exec_ctx = h.build_execution_context(eval_ctx=eval_ctx, capital=self.capital)

        # Verified accessors
        setup_type,_,_ = h.get_setup_from_context(eval_ctx)
        conf,conf_meta = h.get_confidence_from_context(eval_ctx)
        gate_result    = h.check_gates_from_context(eval_ctx, conf)
        strat_info     = h.get_strategy_from_context(eval_ctx)

        return self._from_ctx(symbol,horizon,bar_idx,bar_date,price,atr,
                              eval_ctx,exec_ctx,setup_type,conf,
                              gate_result,strat_info,"config_helpers")

    # ── Isolated fallback ─────────────────────────────────────────────────────

    def _run_isolated(self,symbol,horizon,indicators,fundamentals,
                      bar_idx,bar_date,price,atr) -> BarSignal:
        adx     = _nv(indicators,"adx",20.0)
        rvol    = _nv(indicators,"rvol",1.0)
        rsi     = _nv(indicators,"rsi",50.0)
        vs      = str(indicators.get("volume_signature",{}).get("raw","normal")
                      if isinstance(indicators.get("volume_signature"),dict)
                      else indicators.get("volume_signature","normal"))
        ef_v    = _nv(indicators,"ema_fast",price)
        es_v    = _nv(indicators,"ema_slow",price*0.98)
        reg_s   = _nv(indicators,"regSlope",0.0)
        direction = ("bullish" if reg_s>0 else "bearish" if reg_s<0
                     else "bullish" if ef_v>es_v else "bearish")

        adx_min = 20.0 if horizon=="intraday" else 18.0
        failed,passed = [],[]
        if adx<adx_min: failed.append("adx")
        else:           passed.append("adx")
        if rvol<1.0:    failed.append("rvol")
        else:           passed.append("rvol")

        if vs=="climax" and not failed: signal="BLOCKED"
        elif failed:                    signal="WATCH"
        elif direction=="bullish":      signal="BUY"
        elif direction=="bearish":      signal="SELL"
        else:                           signal="WATCH"

        conf=55.0
        if adx>=40:       conf+=20.0
        elif adx>=30:     conf+=10.0
        elif adx<18:      conf-=10.0
        if vs=="surge":   conf+=10.0
        elif vs=="drought":conf-=15.0
        elif vs=="climax": conf-=20.0
        if rvol<=2.0 and signal=="BUY": conf=min(conf,90.0)
        conf=max(40.0,min(95.0,conf))

        if signal=="BUY":
            entry,sl=price,price-atr*1.5
            t1,t2=price+atr*2,price+atr*4
            rr=(t1-entry)/max(entry-sl,0.001)
        elif signal=="SELL":
            entry,sl=price,price+atr*1.5
            t1,t2=price-atr*2,price-atr*4
            rr=(entry-t1)/max(sl-entry,0.001)
        else:
            entry=sl=t1=t2=rr=None

        sl_atr=abs(entry-sl)/max(atr,0.001) if entry and sl else None
        roe=fundamentals.get("returnOnEquity",10.0)
        de=fundamentals.get("debtToEquity",0.8)
        profile=("STRONG" if conf>=75 and roe>=15 and de<=1 else
                 "MODERATE" if conf>=60 else
                 "WEAK" if conf>=45 else "AVOID")

        return BarSignal(
            symbol=symbol,horizon=horizon,bar_date=bar_date,bar_idx=bar_idx,
            signal=signal,profile_signal=profile,setup_type="GENERIC",
            strategy="generic",direction=direction,
            can_execute=(signal in ("BUY","SELL")),is_hard_blocked=bool(failed),
            confidence=conf,conf_base=55.0,conf_pre_clamp=conf,conf_breakdown=[],
            entry=_f(entry),stop_loss=_f(sl),target_1=_f(t1),target_2=_f(t2),
            rr_ratio=_f(rr),sl_atr_mult=_f(sl_atr),
            struct_gates_passed=not bool(failed),struct_gates_failed=failed,
            exec_rules_passed=True,opp_gates_passed=True,
            close_price=price,atr=atr,est_time_str="NA",pipeline_mode="isolated",error=None,
        )

    # ── Extraction from generate_trade_plan output ────────────────────────────

    def _from_plan(self,symbol,horizon,bar_idx,bar_date,price,atr,
                   plan,eval_ctx,h_prof,mode) -> BarSignal:
        """
        Verified plan keys from signal_engine.generate_trade_plan():
          plan["trade_signal"]             — the actual trade signal
          plan["entry"]                    — NOT "entry_price"
          plan["stop_loss"]
          plan["targets"]["t1"]            — NOT "target_1"
          plan["targets"]["t2"]
          plan["final_confidence"]         — NOT "confidence"
          plan["rr_ratio"]
          plan["setup_type"]
          plan["can_trade"]
          plan["execution_blocked"]
          plan["metadata"]["direction"]
          plan["confidence_breakdown"]["base_floor"]
          plan["confidence_breakdown"]["final_unclamped"]
          plan["confidence_breakdown"]["clamped"]
        """
        signal    = _s(plan.get("trade_signal","WATCH"))
        entry     = _f(plan.get("entry"))
        sl        = _f(plan.get("stop_loss"))
        tgts      = plan.get("targets") or {}
        t1        = _f(tgts.get("t1"))
        t2        = _f(tgts.get("t2"))
        rr        = _f(plan.get("rr_ratio"))
        conf      = float(plan.get("final_confidence") or 0)
        setup     = plan.get("setup_type") or eval_ctx.get("setup",{}).get("type","GENERIC")
        direction = (plan.get("metadata") or {}).get("direction","neutral")
        can_exec  = bool(plan.get("can_trade",False))
        is_hard   = bool(plan.get("execution_blocked",False))

        # Confidence breakdown from plan["confidence_breakdown"]
        cb        = plan.get("confidence_breakdown") or {}
        conf_base = float(cb.get("base_floor") or 0)
        conf_pre  = float(cb.get("final_unclamped") or conf)

        # Detailed breakdown strings from eval_ctx["confidence"]["adjustments"]["breakdown"]
        adj = eval_ctx.get("confidence",{}).get("adjustments",{})
        breakdown = adj.get("breakdown",[]) if isinstance(adj,dict) else []

        # Gates from eval_ctx (same keys used in both full and helpers paths)
        sg = eval_ctx.get("structural_gates",{}).get("overall",{})
        sg_passed = bool(sg.get("passed",True))
        sg_failed = [f["gate"] for f in sg.get("failed_gates",[]) if isinstance(f,dict) and "gate" in f]
        er_passed = bool(eval_ctx.get("execution_rules",{}).get("overall",{}).get("passed",True))
        op_passed = bool(eval_ctx.get("opportunity_gates",{}).get("overall",{}).get("passed",True))

        profile   = h_prof.get("profile_signal","MODERATE")
        strat     = eval_ctx.get("strategy",{}).get("primary","generic")
        sl_atr    = abs(entry-sl)/max(atr,0.001) if (entry and sl) else None

        return BarSignal(
            symbol=symbol,horizon=horizon,bar_date=bar_date,bar_idx=bar_idx,
            signal=signal,profile_signal=profile,setup_type=setup,
            strategy=strat,direction=str(direction).lower(),
            can_execute=can_exec,is_hard_blocked=is_hard,
            confidence=conf,conf_base=conf_base,conf_pre_clamp=conf_pre,
            conf_breakdown=[str(x) for x in breakdown],
            entry=entry,stop_loss=sl,target_1=t1,target_2=t2,
            rr_ratio=rr,sl_atr_mult=_f(sl_atr),
            struct_gates_passed=sg_passed,struct_gates_failed=sg_failed,
            exec_rules_passed=er_passed,opp_gates_passed=op_passed,
            close_price=price,atr=atr,est_time_str=plan.get("est_time_str", "NA"),
            pipeline_mode=mode,error=None,
        )

    # ── Extraction from config_helpers context objects ────────────────────────

    def _from_ctx(self,symbol,horizon,bar_idx,bar_date,price,atr,
                  eval_ctx,exec_ctx,setup_type,conf_from_accessor,
                  gate_result,strat_info,mode) -> BarSignal:
        """
        Verified eval_ctx keys from config_resolver._build_evaluation_context:
          eval_ctx["confidence"]["clamped"]
          eval_ctx["confidence"]["base"]
          eval_ctx["confidence"]["final"]
          eval_ctx["confidence"]["adjustments"]["breakdown"]
          eval_ctx["trend"]["classification"]["direction"]
          eval_ctx["strategy"]["primary"]
          eval_ctx["structural_gates"]["overall"]["passed"]
          eval_ctx["structural_gates"]["overall"]["failed_gates"] → [{"gate":name}]
          eval_ctx["execution_rules"]["overall"]["passed"]
          eval_ctx["opportunity_gates"]["overall"]["passed"]

        Verified exec_ctx keys from config_resolver._build_execution_context:
          exec_ctx["can_execute"]["can_execute"]     — bool inside dict
          exec_ctx["can_execute"]["is_hard_blocked"] — bool inside dict
          exec_ctx["risk"]["entry_price"]
          exec_ctx["risk"]["stop_loss"]
          exec_ctx["risk"]["targets"]                — list
          exec_ctx["risk"]["rrRatio"]
          exec_ctx["market_adjusted_targets"]["adjusted"]
          exec_ctx["market_adjusted_targets"]["execution_entry/sl/t1/t2/rr_t1/rr_t2"]
        """
        # Confidence
        c_data   = eval_ctx.get("confidence",{})
        conf     = float(c_data.get("clamped", conf_from_accessor or 0))
        conf_b   = float(c_data.get("base",0))
        conf_pre = float(c_data.get("final",conf))
        adj      = c_data.get("adjustments",{})
        breakdown= adj.get("breakdown",[]) if isinstance(adj,dict) else []

        # Direction from eval_ctx["trend"]["classification"]["direction"]
        direction = eval_ctx.get("trend",{}).get("classification",{}).get("direction","neutral")

        # Strategy from eval_ctx["strategy"]["primary"]
        strat = eval_ctx.get("strategy",{}).get("primary","generic")

        # Gates
        sg_ov     = eval_ctx.get("structural_gates",{}).get("overall",{})
        sg_passed = bool(sg_ov.get("passed",True))
        sg_failed = [f["gate"] for f in sg_ov.get("failed_gates",[])
                     if isinstance(f,dict) and "gate" in f]
        er_passed = bool(eval_ctx.get("execution_rules",{}).get("overall",{}).get("passed",True))
        op_passed = bool(eval_ctx.get("opportunity_gates",{}).get("overall",{}).get("passed",True))

        # can_execute — it's a dict {"can_execute":bool, "is_hard_blocked":bool, ...}
        ce_dict   = exec_ctx.get("can_execute",{})
        can_exec  = bool(ce_dict.get("can_execute",False)) if isinstance(ce_dict,dict) else bool(ce_dict)
        is_hard   = bool(ce_dict.get("is_hard_blocked",False)) if isinstance(ce_dict,dict) else False

        # Targets — prefer market_adjusted_targets (after trade_enhancer),
        # fall back to exec_ctx["risk"] (raw resolver output)
        mat  = exec_ctx.get("market_adjusted_targets",{})
        risk = exec_ctx.get("risk",{})
        if mat.get("adjusted"):
            entry = _f(mat.get("execution_entry"))
            sl    = _f(mat.get("execution_sl"))
            t1    = _f(mat.get("execution_t1"))
            t2    = _f(mat.get("execution_t2"))
            rr    = _f(mat.get("execution_rr_t2") or mat.get("execution_rr_t1"))
        else:
            entry = _f(risk.get("entry_price"))
            sl    = _f(risk.get("stop_loss"))
            tgts  = risk.get("targets",[])
            t1    = _f(tgts[0]) if tgts else None
            t2    = _f(tgts[1]) if len(tgts)>1 else None
            rr    = _f(risk.get("rrRatio"))

        sl_atr = abs(entry-sl)/max(atr,0.001) if (entry and sl) else None

        # Derive signal from the gate/confidence state
        vol_t = eval_ctx.get("volume_signature",{}).get("type","normal")
        if vol_t=="climax" and can_exec:              signal="BLOCKED"
        elif not sg_passed or not op_passed:          signal="WATCH"
        elif not can_exec:                            signal="BLOCKED"
        elif conf<60:                                 signal="WATCH"
        elif direction=="bullish":                    signal="BUY"
        elif direction=="bearish":                    signal="SELL"
        else:                                         signal="WATCH"

        profile=("STRONG" if conf>=75 and sg_passed and op_passed else
                 "MODERATE" if conf>=60 else
                 "WEAK" if conf>=45 else "AVOID")

        return BarSignal(
            symbol=symbol,horizon=horizon,bar_date=bar_date,bar_idx=bar_idx,
            signal=_s(signal),profile_signal=profile,setup_type=setup_type,
            strategy=strat,direction=str(direction).lower(),
            can_execute=can_exec,is_hard_blocked=is_hard,
            confidence=conf,conf_base=conf_b,conf_pre_clamp=conf_pre,
            conf_breakdown=[str(x) for x in breakdown],
            entry=entry,stop_loss=sl,target_1=t1,target_2=t2,
            rr_ratio=rr,sl_atr_mult=_f(sl_atr),
            struct_gates_passed=sg_passed,struct_gates_failed=sg_failed,
            exec_rules_passed=er_passed,opp_gates_passed=op_passed,
            close_price=price,atr=atr,est_time_str=exec_ctx.get("timeline", {}).get("t1_estimate", "NA"),
            pipeline_mode=mode,error=None,
        )

    def _err(self,symbol,horizon,bar_idx,bar_date,price,atr,err) -> BarSignal:
        return BarSignal(
            symbol=symbol,horizon=horizon,bar_date=bar_date,bar_idx=bar_idx,
            signal="WATCH",profile_signal="MODERATE",setup_type="GENERIC",
            strategy="generic",direction="neutral",
            can_execute=False,is_hard_blocked=True,
            confidence=0.0,conf_base=0.0,conf_pre_clamp=0.0,conf_breakdown=[],
            entry=None,stop_loss=None,target_1=None,target_2=None,
            rr_ratio=None,sl_atr_mult=None,
            struct_gates_passed=False,struct_gates_failed=["pipeline_error"],
            exec_rules_passed=False,opp_gates_passed=False,
            close_price=price,atr=atr,est_time_str="ERROR",pipeline_mode="error",
            error=err[:500] if err else None,
        )


# Helpers
def _nv(d,k,default=0.0):
    v=d.get(k,default) if isinstance(d,dict) else default
    if isinstance(v,dict): v=v.get("value") or v.get("raw") or default
    try: return float(v)
    except: return float(default)

def _f(v):
    try: return float(v) if v is not None else None
    except: return None

def _s(v):
    s=str(v).upper().strip() if v else "WATCH"
    return s if s in SIGNAL_STATES else "WATCH"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FOUR MEASUREMENT ENGINES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ForwardOutcome:
    symbol:str; horizon:str; bar_date:str; signal:str
    entry:float; target_1:float; target_2:float; stop_loss:float
    rr_ratio:float; outcome:str
    bars_to_outcome:Optional[int]; pct_return:Optional[float]

@dataclasses.dataclass
class SignalTransition:
    symbol:str; horizon:str; from_date:str; to_date:str
    from_signal:str; to_signal:str
    from_conf:float; to_conf:float; conf_delta:float
    new_gates_failed:List[str]; recovered_gates:List[str]; trigger_reason:str

@dataclasses.dataclass
class HorizonAgreement:
    symbol:str; bar_date:str
    intraday:Optional[str]; short_term:Optional[str]; long_term:Optional[str]
    agreement:str; confluence_score:float


class ForwardOutcomeEngine:
    def measure(self,bar_signals,df,horizon,fwd_window):
        outcomes=[]
        closes=df["Close"].astype(float).values
        highs=df["High"].astype(float).values
        lows=df["Low"].astype(float).values
        for bs in bar_signals:
            if bs.signal not in ("BUY","STRONG_BUY","SELL"): continue
            if bs.entry is None or bs.stop_loss is None or bs.target_1 is None: continue
            entry,sl,t1=bs.entry,bs.stop_loss,bs.target_1
            t2=bs.target_2 or (entry+(entry-sl)*3)
            is_long=bs.signal in ("BUY","STRONG_BUY")
            outcome,bars_to,pct=OUTCOME_EXPIRED,None,None
            for fi in range(bs.bar_idx+1,min(bs.bar_idx+fwd_window+1,len(closes))):
                bh,bl=highs[fi],lows[fi]
                if is_long:
                    if bl<=sl:   outcome,bars_to,pct=OUTCOME_SL,fi-bs.bar_idx,(sl-entry)/max(entry,0.01)*100;break
                    if bh>=t2:   outcome,bars_to,pct=OUTCOME_T2,fi-bs.bar_idx,(t2-entry)/max(entry,0.01)*100;break
                    if bh>=t1:   outcome,bars_to,pct=OUTCOME_T1,fi-bs.bar_idx,(t1-entry)/max(entry,0.01)*100
                else:
                    if bh>=sl:   outcome,bars_to,pct=OUTCOME_SL,fi-bs.bar_idx,(entry-sl)/max(entry,0.01)*-100;break
                    if bl<=t2:   outcome,bars_to,pct=OUTCOME_T2,fi-bs.bar_idx,(entry-t2)/max(entry,0.01)*100;break
                    if bl<=t1:   outcome,bars_to,pct=OUTCOME_T1,fi-bs.bar_idx,(entry-t1)/max(entry,0.01)*100
            outcomes.append(ForwardOutcome(
                symbol=bs.symbol,horizon=horizon,bar_date=bs.bar_date,signal=bs.signal,
                entry=entry,target_1=t1,target_2=t2,stop_loss=sl,rr_ratio=bs.rr_ratio or 0,
                outcome=outcome,bars_to_outcome=bars_to,pct_return=pct,
            ))
        return outcomes


class TransitionEngine:
    def extract(self,bar_signals:List[BarSignal])->List[SignalTransition]:
        out=[]
        for i in range(1,len(bar_signals)):
            p,c=bar_signals[i-1],bar_signals[i]
            if p.signal==c.signal: continue
            pf,cf=set(p.struct_gates_failed),set(c.struct_gates_failed)
            new_f,rec=sorted(cf-pf),sorted(pf-cf)
            delta=c.confidence-p.confidence
            parts=[]
            if new_f:       parts.append(f"gates failed: {', '.join(new_f)}")
            if rec:         parts.append(f"gates recovered: {', '.join(rec)}")
            if abs(delta)>=5: parts.append(f"conf {'up' if delta>0 else 'down'} {abs(delta):.1f}pt")
            if p.setup_type!=c.setup_type: parts.append(f"setup {p.setup_type}→{c.setup_type}")
            if p.exec_rules_passed and not c.exec_rules_passed: parts.append("exec_rules failed")
            if not p.exec_rules_passed and c.exec_rules_passed: parts.append("exec_rules recovered")
            out.append(SignalTransition(
                symbol=p.symbol,horizon=p.horizon,from_date=p.bar_date,to_date=c.bar_date,
                from_signal=p.signal,to_signal=c.signal,from_conf=p.confidence,
                to_conf=c.confidence,conf_delta=delta,
                new_gates_failed=new_f,recovered_gates=rec,
                trigger_reason="; ".join(parts) or "minor indicator shift",
            ))
        return out


class HorizonAgreementEngine:
    PRIO={"STRONG_BUY":6,"BUY":5,"SELL":5,"BLOCKED":3,"WATCH":2,"HOLD":1,"AVOID":0}

    def compute(self,sigs_by_hz:Dict[str,List[BarSignal]])->List[HorizonAgreement]:
        by_date:Dict[str,Dict[str,str]]=defaultdict(dict)
        sym=""
        for hz,bars in sigs_by_hz.items():
            if bars and not sym: sym=bars[0].symbol
            if hz=="intraday":
                gd:Dict[str,List[BarSignal]]=defaultdict(list)
                for bs in bars: gd[bs.bar_date].append(bs)
                for date,day in gd.items():
                    by_date[date]["intraday"]=max((bs.signal for bs in day),
                                                  key=lambda s:self.PRIO.get(s,0))
            else:
                for bs in bars: by_date[bs.bar_date][hz]=bs.signal
        out=[]
        for date in sorted(by_date):
            d=by_date[date]
            intra,short,long_=d.get("intraday"),d.get("short_term"),d.get("long_term")
            av=[s for s in [intra,short,long_] if s]
            buys=sum(1 for s in av if s in ("BUY","STRONG_BUY"))
            sells=sum(1 for s in av if s=="SELL")
            n=len(av)
            if n==0: continue
            if buys==n:              ag,sc="FULL_BULL",1.0
            elif sells==n:           ag,sc="FULL_BEAR",1.0
            elif buys>=n-1 and n>=2: ag,sc="PARTIAL_BULL",0.7
            elif sells>=n-1 and n>=2:ag,sc="PARTIAL_BEAR",0.7
            elif buys==0 and sells==0:ag,sc="ALL_NEUTRAL",0.5
            else:                    ag,sc="MIXED",0.2
            out.append(HorizonAgreement(symbol=sym,bar_date=date,
                intraday=intra,short_term=short,long_term=long_,
                agreement=ag,confluence_score=sc))
        return out


class ConsistencyEngine:
    def __init__(self,adapter:PipelineAdapter): self.adapter=adapter
    def check(self,symbol,horizon,indicators,fundamentals,patterns,
              bar_idx,bar_date,runs=2)->Tuple[bool,str]:
        res=[self.adapter.run_bar(symbol,horizon,indicators,fundamentals,
                                  patterns,bar_idx,bar_date) for _ in range(runs)]
        sigs=[r.signal for r in res]
        cons=[round(r.confidence,1) for r in res]
        if len(set(sigs))>1:  return False,f"signal diverged: {sigs}"
        if len(set(cons))>1:  return False,f"confidence diverged: {cons}"
        return True,"consistent"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — REPLAY ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class ReplayResult:
    symbol:str; horizon:str; bars_total:int; bars_processed:int
    errors:int; pipeline_mode:str
    bar_signals:List[BarSignal]; forward_outcomes:List[ForwardOutcome]
    transitions:List[SignalTransition]; consistency_failures:List[str]
    win_rate:Optional[float]=None; t1_rate:Optional[float]=None
    t2_rate:Optional[float]=None; sl_rate:Optional[float]=None
    avg_rr:Optional[float]=None; avg_conf_buy:Optional[float]=None
    avg_conf_watch:Optional[float]=None
    signal_dist:Dict[str,int]=dataclasses.field(default_factory=dict)
    transition_count:int=0


def replay_symbol_horizon(symbol,horizon,adapter,fwd_engine,trans_engine,
                           cons_engine,quick=False,verbose=False,cons_rate=20):
    df=load_ohlcv(symbol,horizon)
    if df is None:
        print(f"    [SKIP] No cache — run --fetch first"); return None

    fund=fetch_and_cache_fundamentals(symbol)
    cfg=HORIZON_CONFIG[horizon]
    warmup,fwd_w,n=cfg["warmup_bars"],cfg["forward_window"],len(df)
    bar_sigs:List[BarSignal]=[]; cons_fails:List[str]=[]; errors=0

    indices=(list(range(warmup,min(warmup+50,n),5))[:10] if quick
             else range(warmup,n))

    for bar_idx in indices:
        bar_df=df.iloc[:bar_idx+1]
        bar_date=str(df.index[bar_idx].date())
        try:
            ind=compute_indicators_from_ohlcv(bar_df,horizon)
        except Exception as e:
            errors+=1
            if verbose: print(f"      [IND_ERR] {bar_date}: {e}")
            continue

        patterns:Dict[str,Any]={}
        if ENABLE_PATTERN_ANALYSIS:
            try:
                from services.analyzers.pattern_analyzer import run_pattern_analysis
                patterns=run_pattern_analysis(bar_df,ind,horizon)
            except Exception: pass

        bs=adapter.run_bar(symbol,horizon,ind,fund,patterns,bar_idx,bar_date)
        if bs.error:
            errors+=1
            if verbose: print(f"      [PIPE_ERR] {bar_date}: {bs.error[:100]}")
        else:
            bar_sigs.append(bs)

        if bar_idx%cons_rate==0:
            ok,diff=cons_engine.check(symbol,horizon,ind,fund,patterns,bar_idx,bar_date)
            if not ok: cons_fails.append(f"{bar_date}: {diff}")

        if verbose: # Print all bars for debugging
            print(f"      {bar_date} | {bs.signal:<8} | conf={bs.confidence:.0f}% "
                  f"| setup={bs.setup_type:<24} | rr={bs.rr_ratio or 0:.2f} | timeline={bs.est_time_str}"
                  f"{' | GATES_FAILED='+str(bs.struct_gates_failed) if bs.struct_gates_failed else ''}")

    outcomes=fwd_engine.measure(bar_sigs,df,horizon,fwd_w)
    transitions=trans_engine.extract(bar_sigs)
    r=ReplayResult(symbol=symbol,horizon=horizon,bars_total=n,
                   bars_processed=len(bar_sigs),errors=errors,
                   pipeline_mode=adapter.mode,bar_signals=bar_sigs,
                   forward_outcomes=outcomes,transitions=transitions,
                   consistency_failures=cons_fails,
                   transition_count=len(transitions))
    _stats(r); return r


def _stats(r:ReplayResult):
    dist:Dict[str,int]=defaultdict(int)
    bc,wc=[],[]
    for bs in r.bar_signals:
        dist[bs.signal]+=1
        if bs.signal in ("BUY","STRONG_BUY"): bc.append(bs.confidence)
        elif bs.signal=="WATCH":              wc.append(bs.confidence)
    r.signal_dist=dict(dist); r.avg_conf_buy=_avg(bc); r.avg_conf_watch=_avg(wc)
    tr=[o for o in r.forward_outcomes if o.outcome!="NO_SIGNAL"]
    if tr:
        r.t1_rate=sum(1 for o in tr if o.outcome in (OUTCOME_T1,OUTCOME_T2))/len(tr)
        r.t2_rate=sum(1 for o in tr if o.outcome==OUTCOME_T2)/len(tr)
        r.sl_rate=sum(1 for o in tr if o.outcome==OUTCOME_SL)/len(tr)
        r.win_rate=r.t1_rate
        r.avg_rr=_avg([o.pct_return for o in tr if o.pct_return is not None])

def _avg(lst): return sum(lst)/len(lst) if lst else None


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(v,d=2): return f"{v:.{d}f}" if v is not None else ""
def _pct(v):     return f"{v*100:.1f}%" if v is not None else "  — "
def _fc(v):      return f"{v:.1f}%" if v is not None else "—"
def _mc(trs):
    if not trs: return "—"
    c:Dict[str,int]=defaultdict(int)
    for t in trs: c[t.trigger_reason.split(";")[0].strip()]+=1
    return max(c,key=c.get)


def write_signal_log(results):
    p=RESULTS_DIR/"signal_log.csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["symbol","horizon","date","bar_idx","signal","profile_signal",
                    "setup_type","strategy","direction","confidence","conf_base",
                    "conf_pre_clamp","entry","stop_loss","target_1","target_2",
                    "rr_ratio","sl_atr_mult","can_execute","is_hard_blocked",
                    "struct_gates_passed","struct_gates_failed",
                    "exec_rules_passed","opp_gates_passed",
                    "close_price","atr","est_time_str","pipeline_mode","error"])
        for r in results:
            for bs in r.bar_signals:
                w.writerow([bs.symbol,bs.horizon,bs.bar_date,bs.bar_idx,
                            bs.signal,bs.profile_signal,bs.setup_type,bs.strategy,
                            bs.direction,f"{bs.confidence:.1f}",f"{bs.conf_base:.1f}",
                            f"{bs.conf_pre_clamp:.1f}",
                            _fmt(bs.entry),_fmt(bs.stop_loss),
                            _fmt(bs.target_1),_fmt(bs.target_2),
                            _fmt(bs.rr_ratio),_fmt(bs.sl_atr_mult),
                            bs.can_execute,bs.is_hard_blocked,
                            bs.struct_gates_passed,"|".join(bs.struct_gates_failed),
                            bs.exec_rules_passed,bs.opp_gates_passed,
                            _fmt(bs.close_price),_fmt(bs.atr),
                            bs.est_time_str,bs.pipeline_mode,bs.error or ""])
    return p


def write_forward_outcomes(results):
    p=RESULTS_DIR/"forward_outcomes.csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["symbol","horizon","date","signal","entry","target_1","target_2",
                    "stop_loss","rr_ratio","outcome","bars_to_outcome","pct_return"])
        for r in results:
            for o in r.forward_outcomes:
                w.writerow([o.symbol,o.horizon,o.bar_date,o.signal,
                            _fmt(o.entry),_fmt(o.target_1),_fmt(o.target_2),
                            _fmt(o.stop_loss),_fmt(o.rr_ratio),
                            o.outcome,o.bars_to_outcome or "",
                            f"{o.pct_return:.2f}" if o.pct_return is not None else ""])
    return p


def write_transitions_log(results):
    p=RESULTS_DIR/"transitions_log.csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["symbol","horizon","from_date","to_date","from_signal","to_signal",
                    "from_conf","to_conf","conf_delta",
                    "new_gates_failed","recovered_gates","trigger_reason"])
        for r in results:
            for t in r.transitions:
                w.writerow([t.symbol,t.horizon,t.from_date,t.to_date,
                            t.from_signal,t.to_signal,
                            f"{t.from_conf:.1f}",f"{t.to_conf:.1f}",f"{t.conf_delta:+.1f}",
                            "|".join(t.new_gates_failed),"|".join(t.recovered_gates),
                            t.trigger_reason])
    return p


def write_confidence_audit(results):
    """
    Per-bar confidence audit matching eval_ctx["confidence"] keys:
      base     → conf_base
      final    → conf_pre_clamp  (pre-clamp)
      clamped  → confidence
      adjustments.breakdown → first 3 items
    """
    p=RESULTS_DIR/"confidence_audit.csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["symbol","horizon","date","signal",
                    "conf_base","conf_pre_clamp","conf_clamped",
                    "breakdown_1","breakdown_2","breakdown_3"])
        for r in results:
            for bs in r.bar_signals:
                bd=bs.conf_breakdown
                w.writerow([bs.symbol,bs.horizon,bs.bar_date,bs.signal,
                            f"{bs.conf_base:.1f}",f"{bs.conf_pre_clamp:.1f}",
                            f"{bs.confidence:.1f}",
                            bd[0] if len(bd)>0 else "",
                            bd[1] if len(bd)>1 else "",
                            bd[2] if len(bd)>2 else ""])
    return p


def write_horizon_agreement(ags_by_sym):
    p=RESULTS_DIR/"horizon_agreement.csv"
    with open(p,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["symbol","date","intraday","short_term","long_term",
                    "agreement","confluence_score"])
        for sym,ags in ags_by_sym.items():
            for ag in ags:
                w.writerow([ag.symbol,ag.bar_date,
                            ag.intraday or "",ag.short_term or "",ag.long_term or "",
                            ag.agreement,f"{ag.confluence_score:.2f}"])
    return p


def write_summary_report(results,ags_by_sym,elapsed):
    p=RESULTS_DIR/"summary_report.txt"
    L=[];A=L.append
    A("═"*72)
    A("  Pro Stock Analyzer v15.2 — Historical Backtest Summary")
    A(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    A(f"  Duration  : {elapsed:.1f}s")
    A(f"  Mode      : {', '.join({r.pipeline_mode for r in results})}")
    A("═"*72)
    A(f"\n  Symbols : {len({r.symbol for r in results})}  "
      f"Horizons: {', '.join(sorted({r.horizon for r in results}))}  "
      f"Bars: {sum(r.bars_processed for r in results):,}  "
      f"Errors: {sum(r.errors for r in results)}")

    A("\n"+"─"*72)
    A("  FORWARD OUTCOME  (signal quality: did price reach T1/T2?)")
    A("─"*72)
    A(f"  {'Symbol':<16} {'Horizon':<12} {'Sigs':>5} {'Win%':>6} {'T1%':>6} {'T2%':>6} {'SL%':>6} {'AvgRet':>8}")
    for r in sorted(results,key=lambda x:(x.symbol,x.horizon)):
        n=len(r.forward_outcomes)
        if n==0: A(f"  {r.symbol:<16} {r.horizon:<12} {'—':>5}"); continue
        A(f"  {r.symbol:<16} {r.horizon:<12} {n:>5} "
          f"{_pct(r.win_rate):>6} {_pct(r.t1_rate):>6} {_pct(r.t2_rate):>6} "
          f"{_pct(r.sl_rate):>6} {(_fmt(r.avg_rr)+'%' if r.avg_rr else '—'):>8}")

    A("\n"+"─"*72)
    A("  SIGNAL DISTRIBUTION")
    A("─"*72)
    A(f"  {'Symbol':<16} {'Horizon':<12} {'BUY':>5} {'SELL':>5} {'WATCH':>6} {'BLOCK':>6}")
    for r in sorted(results,key=lambda x:(x.symbol,x.horizon)):
        d=r.signal_dist
        A(f"  {r.symbol:<16} {r.horizon:<12} "
          f"{d.get('BUY',0)+d.get('STRONG_BUY',0):>5} {d.get('SELL',0):>5} "
          f"{d.get('WATCH',0):>6} {d.get('BLOCKED',0):>6}")

    A("\n"+"─"*72)
    A("  CONFIDENCE AVERAGES (BUY bars vs WATCH bars)")
    A("─"*72)
    A(f"  {'Symbol':<16} {'Horizon':<12} {'Avg BUY%':>10} {'Avg WATCH%':>12}")
    for r in sorted(results,key=lambda x:(x.symbol,x.horizon)):
        A(f"  {r.symbol:<16} {r.horizon:<12} {_fc(r.avg_conf_buy):>10} {_fc(r.avg_conf_watch):>12}")

    A("\n"+"─"*72)
    A("  SIGNAL TRANSITIONS")
    A("─"*72)
    A(f"  {'Symbol':<16} {'Horizon':<12} {'Count':>6}  Top trigger")
    for r in sorted(results,key=lambda x:(x.symbol,x.horizon)):
        A(f"  {r.symbol:<16} {r.horizon:<12} {r.transition_count:>6}  {_mc(r.transitions)[:52]}")

    A("\n"+"─"*72)
    A("  CONSISTENCY CHECK")
    A("─"*72)
    total_fails=sum(len(r.consistency_failures) for r in results)
    if total_fails==0:
        A("  ✓ All consistency checks passed")
    else:
        A(f"  ✗ {total_fails} failure(s) detected:")
        for r in results:
            for f in r.consistency_failures[:3]:
                A(f"    {r.symbol} {r.horizon}: {f}")

    A("\n"+"─"*72)
    A("  HORIZON CONFLUENCE")
    A("─"*72)
    A(f"  {'Symbol':<16} {'Full Bull':>10} {'Full Bear':>10} {'Partial':>9} {'Mixed':>7}")
    for sym,ags in sorted(ags_by_sym.items()):
        n=max(len(ags),1)
        A(f"  {sym:<16} "
          f"{sum(1 for a in ags if a.agreement=='FULL_BULL')/n*100:>9.1f}% "
          f"{sum(1 for a in ags if a.agreement=='FULL_BEAR')/n*100:>9.1f}% "
          f"{sum(1 for a in ags if 'PARTIAL' in a.agreement)/n*100:>8.1f}% "
          f"{sum(1 for a in ags if a.agreement=='MIXED')/n*100:>6.1f}%")

    A("\n"+"─"*72)
    A("  OUTPUT FILES")
    A("─"*72)
    for fn in ["summary_report.txt","signal_log.csv","forward_outcomes.csv",
               "transitions_log.csv","confidence_audit.csv","horizon_agreement.csv"]:
        A(f"    backtest_results/{fn}")
    A("\n"+"═"*72)
    text="\n".join(L); p.write_text(text); return p


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — CLI
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_SYMBOLS  = NIFTY50_SYMBOLS[:5]
DEFAULT_HORIZONS = ["intraday","short_term","long_term"]


def main()->int:
    parser=argparse.ArgumentParser(
        description="Pro Stock Analyzer v15.2 — Historical Signal Replay Backtester")
    parser.add_argument("--fetch",       action="store_true",
        help="Download & cache OHLCV+fundamentals. Intraday=5d×5m (re-fetch weekly).")
    parser.add_argument("--symbols",     nargs="+",default=None)
    parser.add_argument("--horizons",    nargs="+",
        choices=["intraday","short_term","long_term"],default=None)
    parser.add_argument("--all-nifty50", action="store_true")
    parser.add_argument("--mid-cap",     action="store_true")
    parser.add_argument("--quick",       action="store_true",
        help="10 bars per symbol (fast smoke test)")
    parser.add_argument("--verbose",     action="store_true")
    parser.add_argument("--report",      action="store_true",
        help="Print last summary without re-running")
    parser.add_argument("--capital",     type=float,default=100_000.0)
    parser.add_argument("--cons-rate",   type=int,default=20,
        help="Consistency check every N bars")
    args=parser.parse_args()

    if args.report:
        rp=RESULTS_DIR/"summary_report.txt"
        print(rp.read_text() if rp.exists() else "No report — run without --report first.")
        return 0

    symbols=args.symbols or []
    if args.all_nifty50:  symbols=NIFTY50_SYMBOLS
    elif args.mid_cap:    symbols=NIFTY50_SYMBOLS[:10]+NIFTY_NEXT50_SYMBOLS
    elif not symbols:     symbols=DEFAULT_SYMBOLS
    horizons=args.horizons or DEFAULT_HORIZONS

    if args.fetch:
        print("\n  Fetching OHLCV...")
        for sym in symbols:
            for hz in horizons: fetch_and_cache_ohlcv(sym,hz)
        print("\n  Fetching fundamentals...")
        for sym in symbols: fetch_and_cache_fundamentals(sym)
        print("\n  Cache summary:")
        print("    intraday   → 5d × 5m  (re-fetch weekly to stay fresh)")
        print("    short_term → 2y × 1d")
        print("    long_term  → 5y × 1wk")
        print("\n  Run without --fetch to start the backtest.\n")
        return 0

    print("\n"+"═"*72)
    print(f"  Pro Stock Analyzer v15.2 — Historical Replay")
    print(f"  Symbols: {len(symbols)}   Horizons: {', '.join(horizons)}")
    print(f"  Mode: {'QUICK (10 bars)' if args.quick else 'FULL'}")
    print("═"*72)

    adapter=PipelineAdapter(capital=args.capital)
    fwd=ForwardOutcomeEngine(); trans=TransitionEngine()
    cons=ConsistencyEngine(adapter); ag=HorizonAgreementEngine()
    print(f"\n  Pipeline mode: {adapter.mode.upper()}")

    all_results:List[ReplayResult]=[]
    sigs_by_sym:Dict[str,Dict[str,List[BarSignal]]]=defaultdict(dict)
    t0=time.perf_counter()

    for sym in symbols:
        print(f"\n  ▶ {sym}")
        for hz in horizons:
            print(f"    {hz:<14}",end=" ",flush=True)
            r=replay_symbol_horizon(sym,hz,adapter,fwd,trans,cons,
                                    quick=args.quick,verbose=args.verbose,
                                    cons_rate=args.cons_rate)
            if r is None: print("SKIPPED"); continue
            all_results.append(r)
            sigs_by_sym[sym][hz]=r.bar_signals
            buys=sum(r.signal_dist.get(s,0) for s in ("BUY","STRONG_BUY","SELL"))
            wins=len([o for o in r.forward_outcomes if o.outcome in (OUTCOME_T1,OUTCOME_T2)])
            tot=len(r.forward_outcomes)
            ok=not r.consistency_failures
            print(f"✓  bars={r.bars_processed:>4}  signals={buys:>3}  "
                  f"win={wins}/{tot}  trans={r.transition_count:>3}  "
                  f"consistency={'✓' if ok else f'✗({len(r.consistency_failures)})'}")

    elapsed=time.perf_counter()-t0
    ags_by_sym={sym:ag.compute(by_hz)
                for sym,by_hz in sigs_by_sym.items() if len(by_hz)>=2}

    print(f"\n  Writing results to {RESULTS_DIR}/")
    if all_results:
        write_signal_log(all_results)
        write_forward_outcomes(all_results)
        write_transitions_log(all_results)
        write_confidence_audit(all_results)
        write_horizon_agreement(ags_by_sym)
        rp=write_summary_report(all_results,ags_by_sym,elapsed)
        print(); print(rp.read_text())
    else:
        print("\n  No results. Did you run --fetch first?")

    return 1 if any(r.errors>0 or r.consistency_failures for r in all_results) else 0


if __name__=="__main__":
    sys.exit(main())

# Practical recommendation:
# Run it in two modes. First, --quick --horizons short_term on 5 symbols to see if the pipeline runs end-to-end without errors. Then --all-nifty50 --horizons short_term long_term for a real signal quality baseline. The confidence_audit.csv is particularly useful — compare conf_base vs conf_pre_clamp vs conf_clamped across BUY bars to see if your ADX bands and modifiers are actually moving the needle.
# What it won't tell you is whether a specific pattern like darvasBox is triggering correctly — for that you'd need ENABLE_PATTERN_ANALYSIS = True, which the file warns is ~10× slower per bar. Worth one run on a small set to verify the pattern detection pipeline is wired up correctly in the historical context.

# # Step 1 — download and cache data (one-time, needs network)
# python backtest_historical.py --fetch

# # Step 2 — run the full backtest (offline from here)
# python backtest_historical.py

# # Specific symbols
# python backtest_historical.py --symbols RELIANCE.NS INFY.NS TCS.NS

# # Skip intraday for speed (short_term + long_term only)
# python backtest_historical.py --horizons short_term long_term

# # All 20 Nifty50 symbols
# python backtest_historical.py --all-nifty50

# # Quick smoke test — 10 bars per symbol
# python backtest_historical.py --quick

# # Print each BUY/SELL bar as it fires
# python backtest_historical.py --verbose

# # Reprint last report without re-running
# python backtest_historical.py --report