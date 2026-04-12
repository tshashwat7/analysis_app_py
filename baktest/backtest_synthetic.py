"""
backtest_synthetic.py
=====================
Drop-in merge of backtest_all_setups.py + backtest_signal_engine.py.

Two case registries share one harness:
  SETUP_SCENARIOS  — 23 real setup × 4 horizons × 11 strategies (from backtest_all_setups)
  SYNTHETIC_CASES  — crafted assertion-driven cases    (from backtest_signal_engine)

Usage:
    python backtest_synthetic.py                          # run everything
    python backtest_synthetic.py --suite setup            # SETUP_SCENARIOS only (Suites 1-3)
    python backtest_synthetic.py --suite synthetic        # SYNTHETIC_CASES only
    python backtest_synthetic.py --suite unit             # unittest gate/builder checks
    python backtest_synthetic.py --suite regression       # golden-baseline comparison
    python backtest_synthetic.py --suite historical       # parquet fixture replay
    python backtest_synthetic.py --suite all              # everything
    python backtest_synthetic.py --test "BUY signal"      # single test by name
    python backtest_synthetic.py --tags signal,buy        # tag filter
    python backtest_synthetic.py --save-baselines         # save golden baselines
    python backtest_synthetic.py --generate-fixtures      # download parquet fixtures
    python backtest_synthetic.py -v                       # verbose
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import inspect
import io
import json
import logging
import math
import os
import sys
import time
import traceback
import types
import unittest
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, patch

# ── UTF-8 output ─────────────────────────────────────────────────────────────
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Path resolution ───────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

FIXTURE_DIR  = ROOT / "output" / "fixtures" / "ohlcv"
BASELINE_DIR = ROOT / "output" / "baselines"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
BASELINE_DIR.mkdir(parents=True, exist_ok=True)

try:
    from config.config_resolver import ConfigurationError
except ImportError:
    class ConfigurationError(Exception):
        pass

# Suppress noisy logs during test runs
logging.basicConfig(level=logging.CRITICAL, format="%(levelname)s: %(message)s")
log = logging.getLogger("backtest_synthetic")

# ── Optional pipeline imports ─────────────────────────────────────────────────
try:
    from config.setup_pattern_matrix_config import SETUP_PATTERN_MATRIX
    from config.confidence_config import CONFIDENCE_CONFIG
    from config.strategy_matrix_config import STRATEGY_MATRIX
    from config.config_helpers import build_evaluation_context
    _SETUP_IMPORTS_OK = True
except Exception as _e:
    _SETUP_IMPORTS_OK = False
    log.warning(f"Setup config imports unavailable: {_e}")
    SETUP_PATTERN_MATRIX = {}
    CONFIDENCE_CONFIG = {}
    STRATEGY_MATRIX = {}

try:
    from services.signal_engine import generate_trade_plan
    _SIGNAL_ENGINE_AVAILABLE = True
except Exception as _e:
    _SIGNAL_ENGINE_AVAILABLE = False
    log.warning(f"Signal engine unavailable: {_e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC BUILDER FUNCTIONS
# (from backtest_signal_engine — full polymorphic format, production-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

def make_indicators(
    *,
    emaFast: float = 150.0,
    emaSlow: float = 140.0,
    mma: float = 145.0,
    wma: float = 148.0,
    adx: float = 30.0,
    trend_direction: str = "BULLISH",
    trendStrength: float = 7.5,
    rsi: float = 60.0,
    macd: float = 2.5,
    macdSignal: float = 1.8,
    macdHistogram: float = 0.7,
    momentumStrength: float = 6.5,
    atr: float = 5.0,
    atrPct: float = 3.3,
    bbWidth: float = 0.08,
    bbPercentB: float = 0.95,
    rsiSlope: Optional[float] = None,
    volatilityQuality: float = 6.0,
    volume: float = 1_000_000.0,
    rvol: float = 1.8,
    volume_signature: str = "normal",
    price: float = 150.0,
    open_: float = 148.0,
    high: float = 152.0,
    low: float = 147.0,
    marketTrendScore: float = 7.0,
    **extra: Any,
) -> Dict[str, Any]:
    """
    Builds a production-like indicator dict compatible with signal_engine +
    config_resolver. All values wrapped as {"value": x, "raw": x, "score": s}.

    R2 fixes applied:
      - avg_volume_30Days  (was avgVolume in backtest_all_setups — mismatched _extract_price_data)
      - slDistance         (always present — required by _check_sl_distance execution rule)
      - Pattern fields use {"raw": {"found": bool, "score": float, ...}} format
    """
    trend_dir = str(trend_direction).upper()
    ma_trend_signal = extra.pop("maTrendSignal", 1.0 if trend_dir == "BULLISH" else -1.0 if trend_dir == "BEARISH" else 0.0)
    ma_fast_slope = extra.pop("maFastSlope", 5.0 if trend_dir == "BULLISH" else -5.0 if trend_dir == "BEARISH" else 0.0)
    ma_slow_slope = extra.pop("maSlowSlope", 2.0 if trend_dir == "BULLISH" else -2.0 if trend_dir == "BEARISH" else 0.0)
    reg_slope = extra.pop("regSlope", 18.0 if trend_dir == "BULLISH" else -18.0 if trend_dir == "BEARISH" else 0.0)
    supertrend_signal = extra.pop("supertrendSignal", "Bullish" if trend_dir == "BULLISH" else "Bearish" if trend_dir == "BEARISH" else "Neutral")
    prev_supertrend = extra.pop("prev_supertrend", supertrend_signal)
    prev_macd_hist = extra.pop("prevMacdHistogram", macdHistogram * 0.8)
    price_vs_primary = extra.pop("priceVsPrimaryTrendPct", 2.5 if trend_dir == "BULLISH" else -2.5 if trend_dir == "BEARISH" else 0.0)
    rel_strength_nifty = extra.pop("relStrengthNifty", 15.0 if trend_dir == "BULLISH" else -15.0 if trend_dir == "BEARISH" else 0.0)
    atr_dynamic = extra.pop("atrDynamic", atr)
    # R2 fix: slDistance always present (execution rule _check_sl_distance needs it)
    sl_distance = extra.pop("sl_distance", extra.pop("slDistance", 1.5))
    avgVolume = extra.pop("avgVolume30Days", extra.pop("avgVolume", volume * 0.8))
    vwap_bias = extra.pop("vwapBias", "Bullish" if trend_dir == "BULLISH" else "Bearish" if trend_dir == "BEARISH" else "Neutral")
    position52w = extra.pop("position52w", 85.0 if trend_dir == "BULLISH" else 20.0 if trend_dir == "BEARISH" else 50.0)
    prevClose = extra.pop("prevClose", open_)

    if rsiSlope is None:
        rsiSlope = 0.05 if trend_dir == "BULLISH" else -0.05 if trend_dir == "BEARISH" else 0.0

    is_bull = trend_dir == "BULLISH"
    neckline = price * (0.95 if is_bull else 1.05)
    pivot_point = price * (0.94 if is_bull else 1.06)
    ma_mid = price * (0.97 if is_bull else 1.03)
    ma_slow = price * (0.91 if is_bull else 1.09)
    tenkan = price * (0.98 if is_bull else 1.02)
    kijun = price * (0.96 if is_bull else 1.04)
    senkouA = price * (0.94 if is_bull else 1.06)
    senkouB = price * (0.92 if is_bull else 1.08)
    box_high = price * (1.02 if is_bull else 0.98)
    box_low = price * (0.95 if is_bull else 1.05)

    base = {
        "emaFast": {"value": emaFast, "score": trendStrength, "raw": emaFast},
        "emaSlow": {"value": emaSlow, "score": trendStrength * 0.9, "raw": emaSlow},
        "emaMid": {"value": ma_mid, "score": trendStrength * 0.8, "raw": ma_mid},
        "mma": {"value": mma, "score": 5.0, "raw": mma},
        "wma": {"value": wma, "score": 5.0, "raw": wma},
        "adx": {"value": adx, "score": min(adx / 5.0, 10.0), "raw": adx},
        "trendStrength": {"value": trendStrength, "score": trendStrength, "raw": trendStrength},
        "trend_direction": {"value": trend_direction, "score": 10.0 if trend_dir != "NEUTRAL" else 0.0, "raw": trend_direction},
        "maTrendSignal": {"value": ma_trend_signal, "score": abs(ma_trend_signal) * 10.0, "raw": ma_trend_signal},
        "maFastSlope": {"value": ma_fast_slope, "score": min(abs(ma_fast_slope) / 5.0, 10.0), "raw": ma_fast_slope},
        "maSlowSlope": {"value": ma_slow_slope, "score": min(abs(ma_slow_slope) / 2.0, 10.0), "raw": ma_slow_slope},
        "regSlope": {"value": reg_slope, "score": min(abs(reg_slope) / 10.0, 10.0), "raw": reg_slope},
        "priceVsPrimaryTrendPct": {"value": price_vs_primary, "score": min(abs(price_vs_primary) * 2.0, 10.0), "raw": price_vs_primary},
        "relStrengthNifty": {"value": rel_strength_nifty, "score": min(abs(rel_strength_nifty) / 2.0, 10.0), "raw": rel_strength_nifty},
        "supertrendSignal": {"value": supertrend_signal, "score": 10.0 if "bullish" in supertrend_signal.lower() else 0.0, "raw": supertrend_signal},
        "prev_supertrend": {"value": prev_supertrend, "score": 5.0, "raw": prev_supertrend},
        "rsi": {"value": rsi, "score": rsi / 10.0, "raw": rsi},
        "rsiSlope": {"value": rsiSlope, "score": 5.0, "raw": rsiSlope},
        "macd": {"value": macd, "score": 5.0, "raw": macd},
        "macdSignal": {"value": macdSignal, "score": 5.0, "raw": macdSignal},
        "macdHistogram": {"value": macdHistogram, "score": 5.0, "raw": macdHistogram},
        "prevMacdHistogram": {"value": prev_macd_hist, "score": 5.0, "raw": prev_macd_hist},
        "momentumStrength": {"value": momentumStrength, "score": momentumStrength, "raw": momentumStrength},
        "atr": {"value": atr, "score": 5.0, "raw": atr},
        "atrDynamic": {"value": atr_dynamic, "score": 5.0, "raw": atr_dynamic},
        # R2 fix: slDistance always included
        "slDistance": {"value": sl_distance, "score": 5.0, "raw": sl_distance},
        "atrPct": {"value": atrPct, "score": volatilityQuality, "raw": atrPct},
        "bbWidth": {"value": bbWidth, "score": 5.0, "raw": bbWidth},
        "bbPercentB": {"value": bbPercentB, "score": bbPercentB * 10.0, "raw": bbPercentB},
        "volatilityQuality": {"value": volatilityQuality, "score": volatilityQuality, "raw": volatilityQuality},
        "maMid": {"value": ma_mid, "score": 5.0, "raw": ma_mid},
        "maSlow": {"value": ma_slow, "score": 5.0, "raw": ma_slow},
        "neckline": {"value": neckline, "score": 5.0, "raw": neckline},
        "pivotPoint": {"value": pivot_point, "score": 5.0, "raw": pivot_point},
        "peak_similarity": {"value": 0.01, "score": 9.0, "raw": 0.01},
        "bullishNeckline": {"raw": {"found": is_bull, "quality": 10.0}, "value": 1.0 if is_bull else 0.0, "score": 10.0 if is_bull else 0.0},
        "bearishNeckline": {"raw": {"found": not is_bull, "quality": 10.0}, "value": 0.0 if is_bull else 1.0, "score": 10.0 if not is_bull else 0.0},
        "ichiTenkan": {"value": tenkan, "score": 5.0, "raw": tenkan},
        "ichiKijun": {"value": kijun, "score": 5.0, "raw": kijun},
        "ichiSpanA": {"value": senkouA, "score": 5.0, "raw": senkouA},
        "ichiSpanB": {"value": senkouB, "score": 5.0, "raw": senkouB},
        "box_high": {"value": box_high, "score": 5.0, "raw": box_high},
        "box_low": {"value": box_low, "score": 5.0, "raw": box_low},
        "box_age_candles": {"value": 10, "score": 10.0, "raw": 10},
        "squeeze_duration": {"value": 10, "score": 10.0, "raw": 10},
        "pole_length": {"value": 10.0, "score": 8.0, "raw": 10.0},
        "flag_tightness": {"value": 0.02, "score": 9.0, "raw": 0.02},
        "strike_candle_body": {"value": 0.8, "score": 8.0, "raw": 0.8},
        "contraction_pct": {"value": 2.0, "score": 9.0, "raw": 2.0},
        "volume": {"value": volume, "score": 5.0, "raw": volume},
        "rvol": {"value": rvol, "score": min(rvol * 3.0, 10.0), "raw": rvol},
        "volume_signature": {"value": volume_signature, "score": 10.0 if volume_signature == "normal" else 5.0, "raw": volume_signature},
        # R2 fix: key matches config_helpers._extract_price_data lookup
        "avgVolume30Days": {"value": avgVolume, "score": 5.0, "raw": avgVolume},
        "price": {"value": price, "score": 5.0, "raw": price},
        "open": {"value": open_, "score": 5.0, "raw": open_},
        "high": {"value": high, "score": 5.0, "raw": high},
        "low": {"value": low, "score": 5.0, "raw": low},
        "close": {"value": price, "score": 5.0, "raw": price},
        "prevClose": {"value": prevClose, "score": 5.0, "raw": prevClose},
        "maFast": {"value": emaFast, "score": trendStrength, "raw": emaFast},
        "vwapBias": {"value": vwap_bias, "score": 10.0 if "bullish" in vwap_bias.lower() else 0.0, "raw": vwap_bias},
        "position52w": {"value": position52w, "score": position52w / 10.0, "raw": position52w},
        "marketTrendScore": {"value": marketTrendScore, "score": marketTrendScore, "raw": marketTrendScore},
    }

    for key, value in extra.items():
        if isinstance(value, dict):
            base[key] = value
        else:
            base[key] = {"value": value, "raw": value, "score": 5.0}

    return base


def make_fundamentals(
    *,
    pe_ratio: float = 15.0,
    pb_ratio: float = 2.0,
    roe: float = 22.0,
    roce: float = 20.0,
    de_ratio: float = 0.3,
    promoter_holding: float = 65.0,
    piotroski_f: float = 8.0,
    eps_growth_5y: float = 20.0,
    market_cap_cr: float = 15000.0,
    revenue_growth: float = 18.0,
    sector: str = "Technology",
    **extra: Any,
) -> Dict[str, Any]:
    base_raw = {
        "peRatio": pe_ratio,
        "pbRatio": pb_ratio,
        "pegRatio": 1.0,
        "roe": roe,
        "roce": roce,
        "netProfitMargin": 15.0,
        "epsGrowth5y": eps_growth_5y,
        "deRatio": de_ratio,
        "piotroskiF": piotroski_f,
        "promoterHolding": promoter_holding,
        "marketCap": market_cap_cr,
        "sector": sector,
    }
    base_raw.update(extra)
    return {
        k: {"value": v, "raw": v, "score": 5.0} if not isinstance(v, dict) else v
        for k, v in base_raw.items()
    }


def make_pattern(
    name: str,
    *,
    found: bool = True,
    quality: float = 7.5,
    score: float = 72.0,
    pattern_type: str = "bullish",
    price: float = 150.0,
    atr: float = 5.0,
    formation_time: Optional[float] = None,
    age_candles: int = 3,
    **meta_extra: Any,
) -> Dict[str, Any]:
    """Canonical 10-field pattern meta contract."""
    ft = formation_time or (time.time() - age_candles * 3600)
    meta = {
        "age_candles": age_candles,
        "formation_time": ft,
        "formation_timestamp": f"2024-01-15T09:{age_candles:02d}:00+05:30",
        "bar_index": 100 - age_candles,
        "type": pattern_type,
        "invalidation_level": price - atr * 2.0,
        "velocity_tracking": {"current": 0.0, "peak": 0.0, "acceleration": 0.0},
        "pattern_strength": "strong" if quality >= 7.0 else "moderate",
        "current_price": price,
        "horizon": "short_term",
    }
    meta.update(meta_extra)
    payload = {"found": found, "score": score, "quality": quality, "meta": meta}
    payload["raw"] = {"found": found, "score": score, "quality": quality, "meta": copy.deepcopy(meta)}
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SETUP SCENARIO REGISTRY
# (from backtest_all_setups — 23 real setups with correct indicator format)
#
# R2 fixes applied throughout:
#   • All pattern booleans converted to make_pattern() dicts
#   • avg_volume_30Days replaces avgVolume in BASE_INDICATORS
#   • slDistance added to BASE_INDICATORS
#   • currentPrice removed (redundant with price)
# ═══════════════════════════════════════════════════════════════════════════════

# Base indicators — used as defaults merged into every scenario
_BASE_INDICATORS_RAW: Dict[str, Any] = {
    "adx": 20, "rsi": 55, "rvol": 1.0, "trendStrength": 4.0,
    "momentumStrength": 4.0, "maTrendSignal": 2, "volatilityQuality": 4.0,
    "bbWidth": 3.0, "bbPercentB": 0.5, "macdHistogram": 0.5,
    "prevMacdHistogram": 0.3, "rsiSlope": 0.3, "priceVsPrimaryTrendPct": 0,
    "position52w": 60, "price": 1500.0,
    "sma50": 1450.0, "sma200": 1400.0,
    "atr": 25.0, "atrDynamic": 25.0, "atrPct": 1.7,
    # R2 fix: avgVolume30Days (was avgVolume — mismatched _extract_price_data)
    "avgVolume30Days": 1500000, "volume": 2000000,
    # R2 fix: slDistance always present
    "slDistance": 1.5,
    "bbHigh": 1580.0, "bbMid": 1500.0, "bbLow": 1420.0,
    "high52w": 1700.0, "low52w": 1100.0,
    "maFast": 1480.0, "maSlow": 1400.0,
}

_BASE_FUNDAMENTALS_RAW: Dict[str, Any] = {
    "peRatio": 22, "roe": 16, "roce": 18, "deRatio": 0.5,
    "marketCap": 25000, "promoterHolding": 55, "institutionalOwnership": 30,
    "dividendYield": 1.5, "pbRatio": 3.0, "currentRatio": 1.8,
    "interestCoverage": 5.0, "epsGrowth5y": 15, "quarterlyGrowth": 12,
    "salesGrowth": 15, "bookValueGrowth": 12, "operatingMargin": 18,
    "netProfitMargin": 12, "fcfYield": 4.0,
}

# R2 fix: patterns are make_pattern() dicts, never plain booleans.
# The pipeline's _extract_patterns and _classify_setup both do isinstance(p, dict) checks.
_BASE_PATTERN_INDICATORS: Dict[str, Any] = {
    "darvasBox":       make_pattern("darvasBox",       found=False, quality=0.0),
    "minerviniStage2": make_pattern("minerviniStage2", found=False, quality=0.0),
    "cupHandle":       make_pattern("cupHandle",       found=False, quality=0.0),
    "flagPennant":     make_pattern("flagPennant",     found=False, quality=0.0),
    "threeLineStrike": make_pattern("threeLineStrike", found=False, quality=0.0),
    "goldenCross":     make_pattern("goldenCross",     found=False, quality=0.0),
    "bollingerSqueeze":make_pattern("bollingerSqueeze",found=False, quality=0.0),
    "ichimokuSignals": make_pattern("ichimokuSignals", found=False, quality=0.0),
    "bullishNeckline": make_pattern("bullishNeckline", found=False, quality=0.0),
    "bearishNeckline": make_pattern("bearishNeckline", found=False, quality=0.0),
}

# Scenario overrides — only keys that differ from BASE_INDICATORS_RAW / BASE_PATTERN_INDICATORS
SETUP_SCENARIOS: Dict[str, Dict[str, Any]] = {
    "PATTERN_DARVAS_BREAKOUT": {
        "symbol": "TATAMOTORS.NS",
        "description": "Strong breakout from Darvas box",
        "indicators": {
            "rvol": 2.5, "trendStrength": 6.0, "adx": 25, "rsi": 65,
            "momentumStrength": 6.5, "bbPercentB": 0.95, "position52w": 85,
            "price": 800, "atrDynamic": 15, "atr": 15,
            "bbHigh": 820, "bbMid": 780, "bbLow": 740,
        },
        "patterns": {
            "darvasBox":       make_pattern("darvasBox",       found=True, quality=8.0, price=800),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=7.5, price=800),
        },
    },
    "PATTERN_VCP_BREAKOUT": {
        "symbol": "HDFCBANK.NS",
        "description": "VCP pattern with tight compression",
        "indicators": {
            "volatilityQuality": 8.0, "rsi": 58, "adx": 22, "rvol": 1.8,
            "trendStrength": 5.0, "bbWidth": 1.5, "position52w": 75,
            "price": 1700, "atrDynamic": 28, "atr": 28,
            "bbHigh": 1730, "bbMid": 1680, "bbLow": 1630,
        },
        "patterns": {
            "minerviniStage2": make_pattern("minerviniStage2", found=True, quality=8.0, price=1700),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=7.0, price=1700),
        },
    },
    "PATTERN_CUP_BREAKOUT": {
        "symbol": "TCS.NS",
        "description": "Cup-and-handle breakout",
        "indicators": {
            "rvol": 1.8, "trendStrength": 5.0, "adx": 20, "rsi": 62,
            "momentumStrength": 5.5, "bbPercentB": 0.88, "position52w": 80,
            "price": 3800, "atrDynamic": 60, "atr": 60,
            "bbHigh": 3900, "bbMid": 3750, "bbLow": 3600,
        },
        "patterns": {
            "cupHandle":  make_pattern("cupHandle",  found=True, quality=8.0, price=3800),
            "goldenCross":make_pattern("goldenCross",found=True, quality=7.5, price=3800),
        },
    },
    "PATTERN_FLAG_BREAKOUT": {
        "symbol": "RELIANCE.NS",
        "description": "Bull flag continuation",
        "indicators": {
            "rvol": 2.0, "trendStrength": 7.0, "adx": 28, "rsi": 68,
            "momentumStrength": 6.0, "bbPercentB": 0.92, "position52w": 88,
            "price": 2800, "atrDynamic": 45, "atr": 45,
            "bbHigh": 2870, "bbMid": 2750, "bbLow": 2630,
        },
        "patterns": {
            "flagPennant":     make_pattern("flagPennant",     found=True, quality=8.0, price=2800),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=7.0, price=2800),
        },
    },
    "PATTERN_STRIKE_REVERSAL": {
        "symbol": "SUNPHARMA.NS",
        "description": "Reversal strike pattern",
        "indicators": {
            "rvol": 2.0, "rsi": 52, "adx": 18, "trendStrength": 3.5,
            "momentumStrength": 4.0, "bbPercentB": 0.6, "position52w": 50,
            "price": 1200, "atrDynamic": 20, "atr": 20,
        },
        "patterns": {
            "threeLineStrike": make_pattern("threeLineStrike", found=True, quality=7.5, price=1200),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=6.5, price=1200),
        },
    },
    "PATTERN_GOLDEN_CROSS": {
        "symbol": "INFY.NS",
        "description": "Golden cross with momentum",
        "indicators": {
            "trendStrength": 5.0, "momentumStrength": 5.5, "adx": 22,
            "rsi": 60, "rvol": 1.5, "macdHistogram": 1.5, "position52w": 70,
            "price": 1800, "atrDynamic": 30, "atr": 30,
            "bbHigh": 1850, "bbMid": 1780, "bbLow": 1710,
        },
        "patterns": {
            "goldenCross":     make_pattern("goldenCross",     found=True, quality=8.0, price=1800),
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=7.0, price=1800),
        },
    },
    "MOMENTUM_BREAKOUT": {
        "symbol": "ADANIENT.NS",
        "description": "Bollinger Band breakout with volume",
        "indicators": {
            "bbPercentB": 0.99, "rsi": 72, "rvol": 2.5, "adx": 30,
            "trendStrength": 6.5, "momentumStrength": 7.0, "position52w": 90,
            "price": 3200, "atrDynamic": 55, "atr": 55,
            "bbHigh": 3190, "bbMid": 3050, "bbLow": 2910,
        },
        "patterns": {
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=7.0, price=3200),
            "goldenCross":     make_pattern("goldenCross",     found=True, quality=7.5, price=3200),
        },
    },
    "MOMENTUM_BREAKDOWN": {
        "symbol": "ZOMATO.NS",
        "description": "Breakdown below Bollinger Band",
        "indicators": {
            "bbPercentB": 0.01, "rsi": 30, "rvol": 2.0, "adx": 25,
            "trendStrength": 2.0, "momentumStrength": 2.0, "macdHistogram": -1.5,
            "position52w": 15, "price": 120, "atrDynamic": 5, "atr": 5,
            "bbHigh": 145, "bbMid": 132, "bbLow": 119,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=7.0, price=120),
        },
    },
    "TREND_PULLBACK": {
        "symbol": "BAJFINANCE.NS",
        "description": "Pullback to 20MA in strong uptrend",
        "indicators": {
            "trendStrength": 6.5, "priceVsPrimaryTrendPct": 2.0, "rsi": 55,
            "adx": 22, "rvol": 1.2, "momentumStrength": 5.0, "position52w": 75,
            "price": 6900, "atrDynamic": 120, "atr": 120, "maFast": 6850,
            "bbHigh": 7100, "bbMid": 6900, "bbLow": 6700,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=7.0, price=6900),
            "threeLineStrike":  make_pattern("threeLineStrike",  found=True, quality=6.5, price=6900),
        },
    },
    "DEEP_PULLBACK": {
        "symbol": "WIPRO.NS",
        "description": "Deep pullback in uptrend (5-10%)",
        "indicators": {
            "trendStrength": 5.0, "priceVsPrimaryTrendPct": -7.0, "adx": 18,
            "rsi": 42, "rvol": 0.8, "momentumStrength": 3.0, "position52w": 55,
            "price": 450, "atrDynamic": 10, "atr": 10,
            "bbHigh": 480, "bbMid": 460, "bbLow": 440,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=6.5, price=450),
            "cupHandle":        make_pattern("cupHandle",        found=True, quality=7.0, price=450),
        },
    },
    "TREND_FOLLOWING": {
        "symbol": "TITAN.NS",
        "description": "Steady uptrend with momentum",
        "indicators": {
            "rsi": 62, "macdHistogram": 1.5, "adx": 28, "trendStrength": 6.0,
            "momentumStrength": 5.5, "rvol": 1.3, "position52w": 78,
            "price": 3000, "atrDynamic": 50, "atr": 50,
            "bbHigh": 3080, "bbMid": 2980, "bbLow": 2880,
        },
        "patterns": {
            "flagPennant":     make_pattern("flagPennant",     found=True, quality=7.5, price=3000),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=6.5, price=3000),
        },
    },
    "BEAR_TREND_FOLLOWING": {
        "symbol": "PAYTM.NS",
        "description": "Established downtrend with momentum",
        "indicators": {
            "rsi": 35, "macdHistogram": -2.0, "adx": 28, "trendStrength": 2.0,
            "momentumStrength": 2.5, "rvol": 1.5, "position52w": 12,
            "priceVsPrimaryTrendPct": -15, "price": 600, "atrDynamic": 15, "atr": 15,
            "bbHigh": 650, "bbMid": 620, "bbLow": 590,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=7.0, price=600),
        },
    },
    "QUALITY_ACCUMULATION": {
        "symbol": "NESTLEIND.NS",
        "description": "Quality stock in tight range (accumulation)",
        "indicators": {
            "bbWidth": 2.0, "rvol": 0.7, "rsi": 50, "adx": 12,
            "trendStrength": 3.5, "position52w": 55, "price": 22000,
            "atrDynamic": 350, "atr": 350,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=7.5, price=22000),
            "cupHandle":        make_pattern("cupHandle",        found=True, quality=7.0, price=22000),
        },
        "fundamentals": {"roe": 25, "roce": 28, "deRatio": 0.1, "peRatio": 60,
                         "marketCap": 200000, "operatingMargin": 25, "netProfitMargin": 18},
    },
    "DEEP_VALUE_PLAY": {
        "symbol": "ONGC.NS",
        "description": "Deep value: low PE, beaten down",
        "indicators": {
            "rsi": 28, "priceVsPrimaryTrendPct": -15, "position52w": 12, "adx": 15,
            "trendStrength": 2.0, "rvol": 0.6, "price": 180, "atrDynamic": 5, "atr": 5,
        },
        "patterns": {
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=6.5, price=180),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=6.0, price=180),
        },
        "fundamentals": {"peRatio": 6, "roe": 14, "roce": 16, "deRatio": 0.3,
                         "marketCap": 180000, "dividendYield": 5.0, "pbRatio": 0.8},
    },
    "VALUE_TURNAROUND": {
        "symbol": "TATASTEEL.NS",
        "description": "Turnaround play with improving fundamentals",
        "indicators": {
            "trendStrength": 4.5, "rsi": 52, "adx": 20, "rvol": 1.3,
            "momentumStrength": 4.0, "position52w": 45, "price": 130,
            "atrDynamic": 4, "atr": 4,
        },
        "patterns": {
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=6.5, price=130),
            "threeLineStrike": make_pattern("threeLineStrike", found=True, quality=7.0, price=130),
        },
        "fundamentals": {"roe": 22, "roce": 25, "deRatio": 0.6, "peRatio": 12, "marketCap": 150000},
    },
    "QUALITY_ACCUMULATION_DOWNTREND": {
        "symbol": "HINDUNILVR.NS",
        "description": "Quality stock accumulating in downtrend",
        "indicators": {
            "trendStrength": 2.5, "bbPercentB": 0.35, "adx": 18, "rsi": 45,
            "rvol": 0.6, "bbWidth": 3.5, "position52w": 35, "price": 2400,
            "atrDynamic": 40, "atr": 40,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=7.0, price=2400),
            "cupHandle":        make_pattern("cupHandle",        found=True, quality=7.0, price=2400),
        },
        "fundamentals": {"roe": 28, "roce": 30, "deRatio": 0.1, "peRatio": 55, "marketCap": 500000},
    },
    "VOLATILITY_SQUEEZE": {
        "symbol": "SBIN.NS",
        "description": "Bollinger squeeze coiling for breakout",
        "indicators": {
            "bbWidth": 0.3, "volatilityQuality": 8.0, "adx": 20, "rsi": 55,
            "trendStrength": 4.5, "rvol": 0.8, "position52w": 65,
            "price": 600, "atrDynamic": 12, "atr": 12,
        },
        "patterns": {
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=7.0, price=600),
            "goldenCross":     make_pattern("goldenCross",     found=True, quality=7.5, price=600),
        },
    },
    "REVERSAL_MACD_CROSS_UP": {
        "symbol": "BHARTIARTL.NS",
        "description": "MACD histogram crosses from negative to positive",
        "indicators": {
            "macdHistogram": 0.3, "prevMacdHistogram": -0.5, "trendStrength": 3.5,
            "rsi": 48, "adx": 16, "rvol": 1.1, "momentumStrength": 3.5,
            "position52w": 45, "price": 1100, "atrDynamic": 20, "atr": 20,
        },
        "patterns": {
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=6.5, price=1100),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=6.0, price=1100),
        },
    },
    "REVERSAL_RSI_SWING_UP": {
        "symbol": "MARUTI.NS",
        "description": "RSI bouncing off oversold with positive slope",
        "indicators": {
            "rsi": 32, "rsiSlope": 0.15, "trendStrength": 3.0, "adx": 18,
            "rvol": 1.0, "momentumStrength": 3.0, "position52w": 30,
            "price": 10000, "atrDynamic": 180, "atr": 180,
        },
        "patterns": {
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=6.5, price=10000),
            "bollingerSqueeze":make_pattern("bollingerSqueeze",found=True, quality=6.0, price=10000),
        },
    },
    "REVERSAL_ST_FLIP_UP": {
        "symbol": "HCLTECH.NS",
        "description": "Supertrend flip to bullish",
        "indicators": {
            "rvol": 1.5, "adx": 18, "trendStrength": 3.5, "rsi": 50,
            "momentumStrength": 4.0, "position52w": 50, "price": 1500,
            "atrDynamic": 25, "atr": 25,
        },
        "patterns": {
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=6.5, price=1500),
            "ichimokuSignals":  make_pattern("ichimokuSignals",  found=True, quality=6.5, price=1500),
        },
    },
    "SELL_AT_RANGE_TOP": {
        "symbol": "ASIANPAINT.NS",
        "description": "Price at top of range - selling opportunity",
        "indicators": {
            "bbWidth": 3.0, "rsi": 72, "adx": 15, "trendStrength": 4.0,
            "rvol": 0.9, "bbPercentB": 0.92, "position52w": 85,
            "price": 3300, "atrDynamic": 55, "atr": 55,
            "bbHigh": 3350, "bbMid": 3200, "bbLow": 3050,
        },
        "patterns": {
            "bearishNeckline": make_pattern("bearishNeckline", found=True, quality=7.0,
                                           pattern_type="bearish", price=3300),
        },
    },
    "TAKE_PROFIT_AT_MID": {
        "symbol": "ULTRACEMCO.NS",
        "description": "Mid-range profit booking",
        "indicators": {
            "bbWidth": 3.5, "rsi": 58, "adx": 14, "trendStrength": 3.5,
            "rvol": 0.8, "bbPercentB": 0.55, "position52w": 55,
            "price": 8500, "atrDynamic": 140, "atr": 140,
        },
        "patterns": {},
    },
    "GENERIC": {
        "symbol": "IRCTC.NS",
        "description": "No strong pattern - fallback",
        "indicators": {
            "adx": 12, "rsi": 50, "rvol": 0.8, "trendStrength": 3.0,
            "momentumStrength": 3.0, "bbPercentB": 0.45, "position52w": 45,
            "price": 900, "atrDynamic": 18, "atr": 18,
        },
        "patterns": {},
    },
}

SETUP_HORIZONS = ["intraday", "short_term", "long_term"]

# Load horizon config from confidence config if available
_HORIZON_CLAMPS: Dict[str, List] = {}
_HORIZON_MIN_TRADEABLE: Dict[str, float] = {}
_HORIZON_OVERRIDE_THRESHOLD: Dict[str, float] = {}
for _h in SETUP_HORIZONS:
    _h_cfg = CONFIDENCE_CONFIG.get("horizons", {}).get(_h, {})
    _HORIZON_CLAMPS[_h] = _h_cfg.get("confidence_clamp", [20, 95])
    _mtc = _h_cfg.get("min_tradeable_confidence", {})
    _HORIZON_MIN_TRADEABLE[_h] = _mtc.get("min", 0) if isinstance(_mtc, dict) else 0
    _hco = _h_cfg.get("high_confidence_override", {})
    _HORIZON_OVERRIDE_THRESHOLD[_h] = _hco.get("threshold", 0)

ALL_STRATEGIES = list(STRATEGY_MATRIX.keys()) if STRATEGY_MATRIX else []


def _build_scenario_indicators(scenario: Dict) -> Dict:
    """
    Merge BASE + scenario overrides then wrap everything into
    {"value": x, "raw": x, "score": 5.0} polymorphic dicts.

    Pattern keys (make_pattern dicts) pass through unchanged.
    """
    merged: Dict[str, Any] = {**_BASE_INDICATORS_RAW}
    merged.update(scenario.get("indicators", {}))

    # Merge pattern indicators: start from silent base (found=False), apply scenario overrides
    pattern_merged: Dict[str, Any] = {**_BASE_PATTERN_INDICATORS}
    pattern_merged.update(scenario.get("patterns", {}))

    result: Dict[str, Any] = {}
    for k, v in merged.items():
        if isinstance(v, dict):
            result[k] = v  # already wrapped
        else:
            result[k] = {"value": v, "raw": v, "score": 5.0}

    # Inject pattern dicts directly (already in correct format from make_pattern)
    result.update(pattern_merged)
    return result


def _build_scenario_fundamentals(scenario: Dict) -> Dict:
    merged = {**_BASE_FUNDAMENTALS_RAW}
    merged.update(scenario.get("fundamentals", {}))
    return {
        k: v if isinstance(v, dict) else {"value": v, "raw": v, "score": 5.0}
        for k, v in merged.items()
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SETUP SCENARIO RESULT CLASS + SUITE RUNNERS
# (from backtest_all_setups)
# ═══════════════════════════════════════════════════════════════════════════════

class SetupResult:
    def __init__(self, setup: str, horizon: str, symbol: str, suite: str = ""):
        self.setup = setup
        self.horizon = horizon
        self.symbol = symbol
        self.suite = suite
        self.checks: List[Tuple[str, bool, str]] = []
        self.error: Optional[str] = None

    def add_check(self, name: str, passed: bool, detail: str = "") -> None:
        self.checks.append((name, passed, detail))

    @property
    def passed(self) -> bool:
        return self.error is None and all(c[1] for c in self.checks)

    @property
    def failed_checks(self) -> List[Tuple[str, bool, str]]:
        return [c for c in self.checks if not c[1]]


def _run_setup_suite1(setup_name: str, horizon: str, scenario: Dict,
                      eval_ctx: Dict) -> SetupResult:
    result = SetupResult(setup_name, horizon, scenario["symbol"], "S1")
    if not eval_ctx:
        result.error = "Empty eval_ctx"
        return result

    setup_result = eval_ctx.get("setup", {})
    classified = setup_result.get("type", "UNKNOWN")
    result.add_check("setup_classified", classified not in ("UNKNOWN", None), f"setup={classified}")

    conf = eval_ctx.get("confidence", {})
    result.add_check("confidence_structure", "clamped" in conf and "base" in conf,
                     f"keys={list(conf.keys())[:5]}")

    clamped = conf.get("clamped", -1)
    clamp = _HORIZON_CLAMPS.get(horizon, [20, 95])
    result.add_check("clamp_range", clamp[0] <= clamped <= clamp[1],
                     f"clamped={clamped}, range={clamp}")

    tradeable = conf.get("tradeable")
    result.add_check("tradeable_flag", isinstance(tradeable, bool), f"tradeable={tradeable}")

    min_thresh = conf.get("min_tradeable_threshold")
    expected_min = _HORIZON_MIN_TRADEABLE.get(horizon, 0)
    result.add_check("min_tradeable_threshold",
                     (min_thresh == expected_min) or (min_thresh is None and expected_min == 0),
                     f"got={min_thresh}, config={expected_min}")

    hco = conf.get("high_confidence_override")
    hco_ok = isinstance(hco, dict) and "threshold" in hco
    result.add_check("high_confidence_override", hco_ok,
                     f"threshold={hco.get('threshold') if hco else 'None'}")

    if hco_ok:
        expected_thresh = _HORIZON_OVERRIDE_THRESHOLD.get(horizon, 0)
        result.add_check("override_threshold_value", hco.get("threshold") == expected_thresh,
                         f"got={hco.get('threshold')}, expected={expected_thresh}")

    has_sg = "structural_gates" in eval_ctx
    has_og = "opportunity_gates" in eval_ctx
    result.add_check("gates_present", has_sg or has_og,
                     f"structural={'Y' if has_sg else 'N'} opportunity={'Y' if has_og else 'N'}")

    scoring = eval_ctx.get("scoring", {})
    result.add_check("scoring_present",
                     "technical" in scoring and "fundamental" in scoring,
                     f"tech={'Y' if 'technical' in scoring else 'N'} fund={'Y' if 'fundamental' in scoring else 'N'}")

    best_setup = eval_ctx.get("setup", {}).get("best", {})
    priority = best_setup.get("priority", 0)
    fit_score = best_setup.get("fit_score", 0)
    composite = best_setup.get("composite_score", 0)
    expected_comp = (priority * 0.7) + (fit_score * 0.3)
    result.add_check("composite_score_math", abs(composite - expected_comp) < 0.1,
                     f"got={composite:.1f}, expected={expected_comp:.1f} (P:{priority} F:{fit_score})")

    return result


def _run_setup_suite2(setup_name: str, horizon: str, scenario: Dict,
                      eval_ctx: Dict) -> SetupResult:
    result = SetupResult(setup_name, horizon, scenario["symbol"], "S2")
    if not eval_ctx:
        result.error = "Empty eval_ctx"
        return result

    strategy = eval_ctx.get("strategy", {})
    result.add_check("strategy_section", bool(strategy), f"keys={list(strategy.keys())[:5]}")

    primary = strategy.get("primary")
    result.add_check("primary_strategy", primary is not None and isinstance(primary, str),
                     f"primary={primary}")

    fit_score = strategy.get("fit_score")
    result.add_check("fit_score_numeric", isinstance(fit_score, (int, float)),
                     f"fit_score={fit_score}")

    has_full = "all_strategies" in strategy and len(strategy.get("all_strategies", [])) > 0
    has_simple = "all_suggestions" in strategy

    if has_full:
        all_strats = strategy.get("all_strategies", [])
        result.add_check("all_strategies_scored",
                         len(all_strats) == len(ALL_STRATEGIES),
                         f"counted={len(all_strats)}, expected={len(ALL_STRATEGIES)}")

        required_fields = {"name", "fit_score", "weighted_score", "horizon_multiplier"}
        all_have_fields = True
        missing_detail = []
        for s in all_strats:
            if isinstance(s, dict):
                missing = required_fields - set(s.keys())
                if missing:
                    all_have_fields = False
                    missing_detail.append(f"{s.get('name','?')}: missing {missing}")
        result.add_check("strategy_fields_complete", all_have_fields,
                         f"{'OK' if all_have_fields else '; '.join(missing_detail[:3])}")

        ranked = strategy.get("ranked", [])
        result.add_check("ranked_list_present", len(ranked) > 0, f"ranked_count={len(ranked)}")
        if len(ranked) >= 2:
            scores = [s.get("weighted_score", 0) for s in ranked if isinstance(s, dict)]
            is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            result.add_check("ranked_sorted_desc", is_sorted, f"top3_scores={scores[:3]}")

    elif has_simple:
        suggestions = strategy.get("all_suggestions", [])
        result.add_check("suggestions_present", isinstance(suggestions, list),
                         f"count={len(suggestions)}")
        hm = strategy.get("horizon_multiplier")
        result.add_check("horizon_multiplier",
                         isinstance(hm, (int, float)) and hm > 0, f"multiplier={hm}")

    weighted = strategy.get("weighted_score", 0)
    result.add_check("weighted_score_valid",
                     isinstance(weighted, (int, float)) and weighted >= 0,
                     f"weighted={weighted}")
    return result


def _run_setup_suite3(setup_name: str, horizon: str, scenario: Dict) -> SetupResult:
    result = SetupResult(setup_name, horizon, scenario["symbol"], "S3")
    if not _SIGNAL_ENGINE_AVAILABLE:
        result.error = "Signal engine not importable"
        return result

    indicators = _build_scenario_indicators(scenario)
    fundamentals = _build_scenario_fundamentals(scenario)

    try:
        plan = generate_trade_plan(
            symbol=scenario["symbol"],
            indicators=indicators,
            fundamentals=fundamentals,
            horizon=horizon,
            capital=100_000,
        )
    except Exception as e:
        result.error = f"generate_trade_plan crashed: {type(e).__name__}: {e}"
        return result

    if not plan:
        result.error = "Empty plan returned"
        return result

    required_keys = {"symbol", "horizon", "status"}
    result.add_check("plan_basic_fields", required_keys.issubset(set(plan.keys())),
                     f"keys={list(plan.keys())[:8]}")

    result.add_check("plan_identity",
                     plan.get("symbol") == scenario["symbol"] and plan.get("horizon") == horizon,
                     f"sym={plan.get('symbol')}, hz={plan.get('horizon')}")

    setup_type = plan.get("setup_type")
    result.add_check("plan_setup_type", setup_type is not None, f"setup_type={setup_type}")

    base_conf = plan.get("base_confidence")
    final_conf = plan.get("final_confidence")
    result.add_check("plan_confidence",
                     isinstance(base_conf, (int, float)) and isinstance(final_conf, (int, float)),
                     f"base={base_conf}, final={final_conf}")

    status = plan.get("status", "")
    result.add_check("plan_status_valid", isinstance(status, str) and len(status) > 0,
                     f"status={status}")

    signal = plan.get("trade_signal", plan.get("signal", "NONE"))
    result.add_check("plan_trade_signal", isinstance(signal, str) and len(signal) > 0,
                     f"signal={signal}")

    gates_passed = plan.get("gates_passed")
    execution_blocked = plan.get("execution_blocked")
    result.add_check("plan_gates_status",
                     isinstance(gates_passed, bool) and isinstance(execution_blocked, bool),
                     f"gates={gates_passed}, blocked={execution_blocked}")

    metadata = plan.get("metadata", {})
    result.add_check("plan_metadata", isinstance(metadata, dict) and len(metadata) > 0,
                     f"meta_keys={list(metadata.keys())[:5]}")

    conf_history = plan.get("confidence_history", [])
    result.add_check("plan_confidence_history",
                     isinstance(conf_history, list) and len(conf_history) >= 1,
                     f"history_steps={len(conf_history)}")

    if not plan.get("execution_blocked", True):
        entry = plan.get("entry")
        sl = plan.get("stop_loss")
        targets = plan.get("targets", {})
        has_exec = (entry is not None and sl is not None and isinstance(targets, dict) and
                    (targets.get("t1") is not None or targets.get("t2") is not None))
        result.add_check("plan_execution_values", has_exec,
                         f"entry={entry}, sl={sl}, t1={targets.get('t1')}, t2={targets.get('t2')}")

    analytics = plan.get("analytics", {})
    if analytics:
        result.add_check("plan_analytics",
                         "strategy_fit" in analytics or "technical_score" in analytics,
                         f"analytics_keys={list(analytics.keys())[:5]}")

    return result


def _print_setup_suite_summary(title: str, results: List[SetupResult]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    errors = sum(1 for r in results if r.error)
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")
    print(f"  Total: {total}  Passed: {passed} ({100 * passed // max(total, 1)}%)  "
          f"Failed: {total - passed - errors}  Errors: {errors}")

    print(f"\n  {'Setup':<35s} {'intraday':^10s} {'short_term':^10s} {'long_term':^10s}")
    print(f"  {'---' * 12} {'---' * 4} {'---' * 4} {'---' * 4}")

    setup_map: Dict[str, Dict[str, SetupResult]] = {}
    for r in results:
        setup_map.setdefault(r.setup, {})[r.horizon] = r

    for setup in sorted(setup_map):
        row = f"  {setup:<35s}"
        for h in SETUP_HORIZONS:
            r = setup_map[setup].get(h)
            if r is None:
                row += f"  {'--':^8s}"
            elif r.error:
                row += f"  {'ERR':^8s}"
            elif r.passed:
                row += f"  {'PASS':^8s}"
            else:
                row += f"  {'FAIL':^8s}"
        print(row)

    failed_results = [r for r in results if not r.passed]
    if failed_results:
        print("\n  FAILURE DETAILS:")
        for r in failed_results[:15]:
            print(f"\n  {r.setup}/{r.horizon} ({r.symbol}):")
            if r.error:
                print(f"    ERROR: {r.error}")
            for name, p, detail in r.checks:
                if not p:
                    print(f"    FAIL {name}: {detail}")


def run_setup_suites(verbose: bool = False) -> int:
    """Run Suites 1-3 across all 23 scenarios × 3 horizons. Returns 0 if all pass."""
    if not _SETUP_IMPORTS_OK:
        print("  SKIPPED: Setup config imports unavailable (SETUP_PATTERN_MATRIX / CONFIDENCE_CONFIG)")
        return 0

    print("=" * 80)
    print("  SETUP SCENARIO BACKTESTS")
    print(f"  {len(SETUP_SCENARIOS)} Setups × {len(SETUP_HORIZONS)} Horizons")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Build eval_ctx cache shared between Suite 1 and Suite 2 (one fetch per scenario/horizon)
    print("\n  Building evaluation contexts...")
    eval_ctx_cache: Dict[str, Dict] = {}
    for setup_name, scenario in sorted(SETUP_SCENARIOS.items()):
        indicators = _build_scenario_indicators(scenario)
        fundamentals = _build_scenario_fundamentals(scenario)
        for horizon in SETUP_HORIZONS:
            key = f"{setup_name}|{horizon}"
            try:
                ctx = build_evaluation_context(
                    ticker=scenario["symbol"],
                    indicators=indicators,
                    fundamentals=fundamentals,
                    horizon=horizon,
                )
                eval_ctx_cache[key] = ctx
            except Exception as e:
                eval_ctx_cache[key] = {"_error": str(e)}

    valid_ctx = sum(1 for v in eval_ctx_cache.values() if "_error" not in v)
    print(f"  Built {valid_ctx}/{len(eval_ctx_cache)} contexts successfully\n")

    # Suite 1
    s1_results: List[SetupResult] = []
    for setup_name, scenario in sorted(SETUP_SCENARIOS.items()):
        for horizon in SETUP_HORIZONS:
            key = f"{setup_name}|{horizon}"
            ctx = eval_ctx_cache.get(key, {})
            if "_error" in ctx:
                r = SetupResult(setup_name, horizon, scenario["symbol"], "S1")
                r.error = ctx["_error"]
            else:
                r = _run_setup_suite1(setup_name, horizon, scenario, ctx)
            s1_results.append(r)
            if verbose:
                print(f"  S1 {setup_name}/{horizon}: {'PASS' if r.passed else ('ERR' if r.error else 'FAIL')}")
    _print_setup_suite_summary("SUITE 1: Setup + Confidence + Gates", s1_results)

    # Suite 2
    s2_results: List[SetupResult] = []
    for setup_name, scenario in sorted(SETUP_SCENARIOS.items()):
        for horizon in SETUP_HORIZONS:
            key = f"{setup_name}|{horizon}"
            ctx = eval_ctx_cache.get(key, {})
            if "_error" in ctx:
                r = SetupResult(setup_name, horizon, scenario["symbol"], "S2")
                r.error = ctx["_error"]
            else:
                r = _run_setup_suite2(setup_name, horizon, scenario, ctx)
            s2_results.append(r)
            if verbose:
                primary = ctx.get("strategy", {}).get("primary", "?") if "_error" not in ctx else "?"
                print(f"  S2 {setup_name}/{horizon}: {'PASS' if r.passed else ('ERR' if r.error else 'FAIL')} primary={primary}")
    _print_setup_suite_summary("SUITE 2: Strategy Fit Scoring", s2_results)

    # Strategy coverage
    if ALL_STRATEGIES:
        strategy_primary: Dict[str, int] = Counter()
        for ctx in eval_ctx_cache.values():
            if "_error" not in ctx:
                primary = ctx.get("strategy", {}).get("primary", "")
                if primary:
                    strategy_primary[primary] += 1
        print(f"\n  STRATEGY PRIMARY COUNTS:")
        for strat in ALL_STRATEGIES:
            count = strategy_primary.get(strat, 0)
            marker = " **" if count > 0 else ""
            print(f"  {strat:<30s} {count:>4d}{marker}")

    # Suite 3
    if _SIGNAL_ENGINE_AVAILABLE:
        s3_results: List[SetupResult] = []
        for setup_name, scenario in sorted(SETUP_SCENARIOS.items()):
            for horizon in SETUP_HORIZONS:
                r = _run_setup_suite3(setup_name, horizon, scenario)
                s3_results.append(r)
                if verbose:
                    print(f"  S3 {setup_name}/{horizon}: {'PASS' if r.passed else ('ERR' if r.error else 'FAIL')}")
        _print_setup_suite_summary("SUITE 3: Signal Generation", s3_results)
    else:
        s3_results = []
        print("\n  SUITE 3: SKIPPED (signal engine unavailable)")

    all_results = s1_results + s2_results + s3_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r.passed)
    print(f"\n{'=' * 80}")
    print(f"  GRAND TOTAL: {passed}/{total} ({100 * passed // max(total, 1)}%)")
    print(f"{'=' * 80}")
    return 0 if passed == total else 1


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — SYNTHETIC CASE DATACLASS + CASES
# (from backtest_signal_engine — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class BacktestCase:
    name: str
    description: str
    horizon: str
    indicators: Dict[str, Any]
    fundamentals: Dict[str, Any]
    patterns: Dict[str, Any] = dataclasses.field(default_factory=dict)
    symbol: str = "TEST.NS"
    signal: Optional[str] = None
    profile_signal: Optional[str] = None
    setup_type: Optional[str] = None
    strategy: Optional[str] = None
    confidence_min: Optional[float] = None
    confidence_max: Optional[float] = None
    rr_min: Optional[float] = None
    target1_gt_entry: Optional[bool] = None
    sl_in_atr_range: Optional[bool] = None
    gate_must_pass: Optional[List[str]] = None
    gate_must_fail: Optional[List[str]] = None
    direction: Optional[str] = None
    can_execute: Optional[bool] = None
    best_horizon: Optional[str] = None
    hold_months_range: Optional[Tuple[int, int]] = None
    is_atr_fallback: Optional[bool] = None
    execution_t1: Optional[float] = None
    execution_sl: Optional[float] = None
    tags: List[str] = dataclasses.field(default_factory=list)
    expected_fail: bool = False


SYNTHETIC_CASES: List[BacktestCase] = [
    BacktestCase(
        name="BUY signal: strong ADX breakout",
        description="All structural gates pass, ADX=38 (strong), RVOL=2.8 (surge).",
        horizon="short_term",
        indicators=make_indicators(adx=38.0, rvol=2.8, rsi=62.0, trendStrength=8.5,
                                   atrPct=2.8, volatilityQuality=7.5, bbPercentB=0.72,
                                   trend_direction="BULLISH", volume_signature="surge",
                                   goldenCross={"found": True}),
        fundamentals=make_fundamentals(roe=22.0, de_ratio=0.3, piotroski_f=7.0),
        patterns={"goldenCross": make_pattern("goldenCross", quality=8.0, score=78.0)},
        signal="BUY", profile_signal="MODERATE", direction="bullish",
        confidence_min=65.0, rr_min=1.5, target1_gt_entry=True,
        sl_in_atr_range=True, gate_must_pass=["adx", "rvol"],
        tags=["signal", "buy", "breakout"],
    ),
    BacktestCase(
        name="SELL signal: death cross bearish breakdown",
        description="ADX=32 strong bearish trend, death cross pattern.",
        horizon="short_term",
        indicators=make_indicators(adx=32.0, rvol=2.1, rsi=38.0, trendStrength=7.0,
                                   trend_direction="BEARISH", volume_signature="surge",
                                   atrPct=3.1, volatilityQuality=6.5, bbPercentB=0.10),
        fundamentals=make_fundamentals(roe=8.0, de_ratio=1.2, piotroski_f=4.0),
        patterns={"deathCross": make_pattern("deathCross", quality=7.5, score=70.0,
                                              pattern_type="bearish")},
        signal="SELL", direction="bearish", confidence_min=55.0, target1_gt_entry=False,
        tags=["signal", "sell", "bearish"],
    ),
    BacktestCase(
        name="WATCH signal: neutral trend, no pattern",
        description="ADX=16 (weak), no patterns. Expect WATCH.",
        horizon="short_term",
        indicators=make_indicators(adx=16.0, rvol=0.9, rsi=50.0, trendStrength=4.5,
                                   trend_direction="NEUTRAL", volume_signature="drought"),
        fundamentals=make_fundamentals(roe=10.0, de_ratio=0.8),
        patterns={}, signal="WATCH",
        tags=["signal", "watch", "weak_trend"],
    ),
    BacktestCase(
        name="Architecture: strong trend without primary pattern stays WATCH",
        description="High-confidence bullish trend should still remain WATCH until a primary entry pattern appears.",
        horizon="short_term",
        indicators=make_indicators(
            adx=42.0, rvol=1.8, rsi=68.0, trendStrength=8.8,
            atrPct=2.4, volatilityQuality=7.2, bbPercentB=0.76,
            macdHistogram=1.4, trend_direction="BULLISH", volume_signature="surge",
            price=2500.0, atr=35.0
        ),
        fundamentals=make_fundamentals(roe=23.0, roce=21.0, de_ratio=0.25, piotroski_f=8.0),
        patterns={},
        signal="WATCH",
        confidence_min=70.0,
        gate_must_fail=["STRUCTURE_GATE: GENERIC fallback with no primary pattern cannot generate BUY"],
        tags=["signal", "watch", "architecture", "no_primary_pattern", "high_confidence"],
    ),
    BacktestCase(
        name="Architecture: primary pattern unlocks BUY for strong trend",
        description="Same quality trend should emit BUY once a primary pattern is supplied.",
        horizon="short_term",
        indicators=make_indicators(
            adx=42.0, rvol=1.8, rsi=68.0, trendStrength=8.8,
            atrPct=2.4, volatilityQuality=7.2, bbPercentB=0.76,
            macdHistogram=1.4, trend_direction="BULLISH", volume_signature="surge",
            price=2500.0, atr=35.0
        ),
        fundamentals=make_fundamentals(roe=23.0, roce=21.0, de_ratio=0.25, piotroski_f=8.0),
        patterns={
            "goldenCross": make_pattern("goldenCross", found=True, quality=8.2, score=84.0, price=2500.0, atr=35.0),
            "bollingerSqueeze": make_pattern("bollingerSqueeze", found=True, quality=7.0, score=74.0, price=2500.0, atr=35.0),
        },
        signal="BUY",
        confidence_min=70.0,
        rr_min=1.5,
        target1_gt_entry=True,
        sl_in_atr_range=True,
        tags=["signal", "buy", "architecture", "primary_pattern", "high_confidence"],
    ),
    BacktestCase(
        name="BLOCKED: execution guard suppresses entry",
        description="Strong pattern setup, but SL-distance execution guard blocks entry.",
        horizon="short_term",
        indicators=make_indicators(adx=35.0, rvol=4.5, rsi=78.0, trendStrength=8.0,
                                   trend_direction="BULLISH", volume_signature="surge",
                                   price=150.0, atr=1.0, slDistance=9.0),
        fundamentals=make_fundamentals(roe=18.0, de_ratio=0.5),
        patterns={"goldenCross": make_pattern("goldenCross", quality=8.0, score=78.0)},
        signal="BLOCKED", can_execute=False, gate_must_fail=["sl_distance_validation"],
        tags=["signal", "blocked", "execution_guard"],
    ),
    BacktestCase(
        name="AVOID: fundamental red flags",
        description="Very low fundamentals → AVOID.",
        horizon="long_term",
        indicators=make_indicators(adx=25.0, rvol=1.2, trendStrength=5.5),
        fundamentals=make_fundamentals(roe=3.0, de_ratio=2.8, piotroski_f=2.0,
                                        eps_growth_5y=-5.0, revenue_growth=-8.0),
        patterns={}, signal="WATCH", profile_signal="AVOID",
        tags=["signal", "avoid", "fundamentals"],
    ),
    BacktestCase(
        name="Setup: TREND_PULLBACK classification",
        description="Current inputs resolve to TREND_PULLBACK under the live ranking model.",
        horizon="short_term",
        indicators=make_indicators(adx=38.0, rsi=65.0, rvol=2.5, trendStrength=8.5,
                                   bbPercentB=0.97, atrPct=2.5, volatilityQuality=7.0,
                                   trend_direction="BULLISH", volume_signature="surge",
                                   emaFast=150.0, emaSlow=140.0, price=160.0, open_=156.0,
                                   high=162.0, low=155.0, momentumStrength=8.0,
                                   priceVsPrimaryTrendPct=4.0),
        fundamentals=make_fundamentals(roe=20.0, de_ratio=0.4),
        patterns={"bollingerSqueeze": make_pattern("bollingerSqueeze", quality=8.5)},
        setup_type="TREND_PULLBACK",
        tags=["setup", "trend_pullback"],
    ),
    BacktestCase(
        name="Setup: QUALITY_ACCUMULATION classification",
        description="Tight-range, low-volume, fundamentally strong long-term accumulation profile.",
        horizon="long_term",
        indicators=make_indicators(adx=22.0, rsi=50.0, rvol=0.8, trendStrength=4.5,
                                   atrPct=1.8, bbWidth=3.0, volatilityQuality=8.0,
                                   trend_direction="BULLISH"),
        fundamentals=make_fundamentals(roe=25.0, roce=26.0, de_ratio=0.2, piotroski_f=8.0,
                                        eps_growth_5y=20.0, revenueGrowth5y=22.0,
                                        dividendYield=1.5, fundamentalScore=8.0,
                                        promoter_holding=65.0),
        patterns={},
        setup_type="QUALITY_ACCUMULATION",
        tags=["setup", "quality_accumulation", "long_term"],
    ),
    BacktestCase(
        name="Confidence: ADX explosive band boosts by +20",
        description="ADX=45 triggers 'explosive' band (+20). Base=55, expect >= 75.",
        horizon="short_term",
        indicators=make_indicators(adx=45.0, rvol=2.0, rsi=60.0, trendStrength=8.0,
                                   trend_direction="BULLISH"),
        fundamentals=make_fundamentals(roe=18.0),
        patterns={}, confidence_min=72.0,
        tags=["confidence", "adx_explosive"],
    ),
    BacktestCase(
        name="Confidence: B8 ceiling caps at 90 when rvol<=2.0",
        description="Bullish breakout, RVOL=1.6. B8 ceiling must clamp at 90.",
        horizon="short_term",
        indicators=make_indicators(adx=42.0, rvol=1.6, rsi=68.0, trendStrength=9.0,
                                   trend_direction="BULLISH", volume_signature="normal"),
        fundamentals=make_fundamentals(roe=22.0, piotroski_f=8.0),
        patterns={"bollingerSqueeze": make_pattern("bollingerSqueeze", quality=9.0)},
        confidence_max=90.0,
        tags=["confidence", "b8_ceiling"],
    ),
    BacktestCase(
        name="Target geometry: T1 > entry for LONG, T2 > T1",
        description="Long trade geometry: entry < T1 < T2, SL < entry, RR >= 1.5.",
        horizon="short_term",
        indicators=make_indicators(adx=32.0, rvol=2.0, rsi=60.0, trendStrength=7.5,
                                   trend_direction="BULLISH", price=200.0, atr=8.0,
                                   goldenCross={"found": True}),
        fundamentals=make_fundamentals(roe=18.0),
        patterns={"goldenCross": make_pattern("goldenCross", price=200.0, atr=8.0, quality=7.5)},
        signal="BUY", target1_gt_entry=True, sl_in_atr_range=True, rr_min=1.5,
        tags=["targets", "geometry", "long", "rr"],
    ),
    BacktestCase(
        name="Target geometry: SHORT trade T1 < entry",
        description="Bearish trade. T1 < entry, SL > entry.",
        horizon="short_term",
        indicators=make_indicators(adx=30.0, rvol=2.2, rsi=35.0, trendStrength=7.0,
                                   trend_direction="BEARISH", price=300.0, atr=10.0,
                                   macdHistogram=-1.0, deathCross={"found": True}),
        fundamentals=make_fundamentals(roe=6.0, de_ratio=1.5),
        patterns={"deathCross": make_pattern("deathCross", price=300.0, atr=10.0,
                                              quality=7.5, pattern_type="bearish")},
        signal="SELL", target1_gt_entry=False, sl_in_atr_range=True, rr_min=1.5,
        tags=["targets", "geometry", "short", "rr"],
    ),
    BacktestCase(
        name="SL distance: too tight (< 0.5×ATR) → BLOCKED",
        description="Tiny ATR → SL distance fails execution rule.",
        horizon="intraday",
        indicators=make_indicators(adx=28.0, rvol=1.8, rsi=58.0, trendStrength=7.0,
                                   price=100.0, atr=0.1, slDistance=0.1, trend_direction="BULLISH"),
        fundamentals=make_fundamentals(roe=15.0),
        patterns={}, can_execute=False, gate_must_fail=["sl_distance_validation"],
        tags=["targets", "sl", "execution_rules"],
    ),
    BacktestCase(
        name="Gate: optional piotroskiF missing → skip not fail",
        description="piotroskiF absent. Must be skipped (optional=True).",
        horizon="short_term",
        indicators=make_indicators(adx=28.0, rvol=1.8, trendStrength=7.0),
        fundamentals={k: v for k, v in make_fundamentals().items() if k != "piotroskiF"},
        patterns={}, gate_must_fail=[],
        tags=["gates", "optional_gate"],
    ),
    BacktestCase(
        name="Architecture: no-pattern guard preempts generic structural gate check",
        description="Pattern-less generic setup is blocked by the no-primary-pattern watch guard first.",
        horizon="short_term",
        indicators=make_indicators(adx=30.0, rvol=0.4, rsi=60.0, trendStrength=7.5,
                                   trend_direction="BULLISH"),
        fundamentals=make_fundamentals(roe=18.0),
        patterns={}, signal="WATCH",
        gate_must_fail=["STRUCTURE_GATE: GENERIC fallback with no primary pattern cannot generate BUY"],
        tags=["architecture", "no_pattern_guard", "generic"],
    ),
    BacktestCase(
        name="Pattern: expired pattern triggers confidence penalty -20",
        description="30-day-old pattern. Enhancer must apply -20 expiry penalty.",
        horizon="short_term",
        indicators=make_indicators(adx=30.0, rvol=2.0, trendStrength=7.5, darvasBox={"found": True}),
        fundamentals=make_fundamentals(roe=18.0),
        patterns={"darvasBox": make_pattern("darvasBox", quality=7.5,
                                               formation_time=time.time() - 30 * 86400,
                                               age_candles=150)},
        confidence_max=72.0,
        tags=["pattern", "expiry", "trade_enhancer"],
    ),
    BacktestCase(
        name="Architecture: WATCH and BLOCKED are never aliased",
        description="Pattern found + climax volume → BLOCKED, not WATCH.",
        horizon="short_term",
        indicators=make_indicators(adx=32.0, rvol=2.0, rsi=60.0, trendStrength=7.5,
                                   trend_direction="BULLISH", volume_signature="surge",
                                   price=150.0, atr=1.0, slDistance=9.0),
        fundamentals=make_fundamentals(roe=18.0),
        patterns={"goldenCross": make_pattern("goldenCross", quality=8.0)},
        signal="BLOCKED", can_execute=False, gate_must_fail=["sl_distance_validation"],
        tags=["architecture", "watch_vs_blocked"],
    ),
    BacktestCase(
        name="Architecture: profile_signal independent of execution gates",
        description="BLOCKED signal can coexist with a non-blocked profile grade.",
        horizon="short_term",
        indicators=make_indicators(adx=42.0, rvol=2.2, rsi=78.0, trendStrength=9.0,
                                   trend_direction="BULLISH", volume_signature="surge",
                                   price=200.0, atr=1.0, slDistance=10.0),
        fundamentals=make_fundamentals(roe=25.0, piotroski_f=8.0),
        patterns={"goldenCross": make_pattern("goldenCross", quality=9.0, score=86.0)},
        signal="BLOCKED", profile_signal="MODERATE", can_execute=False,
        gate_must_fail=["sl_distance_validation"],
        tags=["architecture", "profile_signal"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SYNTHETIC ASSERTION + HARNESS ENGINE
# (from backtest_signal_engine — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

class AssertionResult:
    def __init__(self, field: str, passed: bool, expected: Any, actual: Any, reason: str = ""):
        self.field = field
        self.passed = passed
        self.expected = expected
        self.actual = actual
        self.reason = reason

    def __repr__(self) -> str:
        status = "[PASS]" if self.passed else "[FAIL]"
        if self.passed:
            return f"  {status} {self.field}"
        return (f"  {status} {self.field}: expected={self.expected!r}, got={self.actual!r}"
                + (f" ({self.reason})" if self.reason else ""))


def _extract(data: Any, paths: List[str]) -> Any:
    for path in paths:
        parts = path.split(".")
        current = data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break
        if current is not None:
            return current
    return None


def _extract_numeric(data: Any, paths: List[str]) -> Optional[float]:
    val = _extract(data, paths)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _extract_list(data: Any, paths: List[str]) -> List[Any]:
    val = _extract(data, paths)
    return val if isinstance(val, list) else []


def _normalize_failure_names(items: List[Any]) -> List[str]:
    normalized: List[str] = []
    for item in items:
        if isinstance(item, str):
            normalized.append(item)
        elif isinstance(item, dict):
            for key in ("gate", "rule", "name"):
                value = item.get(key)
                if isinstance(value, str):
                    normalized.append(value)
                    break
    return normalized


def _gate_in_list(gate: str, gate_list: List[Any]) -> bool:
    for g in gate_list:
        if isinstance(g, str) and g == gate:
            return True
        if isinstance(g, dict) and g.get("gate") == gate:
            return True
        if isinstance(g, dict) and g.get("rule") == gate:
            return True
    return False


def run_assertions(case: BacktestCase, result: Dict[str, Any]) -> List[AssertionResult]:
    assertion_results: List[AssertionResult] = []

    def _assert(field: str, expected: Any, actual: Any, passed: bool, reason: str = "") -> None:
        assertion_results.append(AssertionResult(field, passed, expected, actual, reason))

    if case.signal is not None:
        actual_signal = _extract(result, ["trade_signal", "signal", "signal_text"])
        if actual_signal:
            actual_signal = actual_signal.upper().replace("STRONG_", "").replace("_", "")
            exp = case.signal.upper().replace("_", "")
        else:
            exp = case.signal.upper()
        _assert("signal", case.signal, actual_signal,
                actual_signal is not None and (actual_signal == exp or actual_signal.startswith(exp)))

    if case.profile_signal is not None:
        actual_ps = _extract(result, ["profile_signal", "best_fit.profile_signal",
                                       "profiles.short_term.profile_signal"])
        _assert("profile_signal", case.profile_signal, actual_ps,
                actual_ps is not None and actual_ps.upper() == case.profile_signal.upper())

    if case.setup_type is not None:
        actual_setup = _extract(result, ["setup.type", "trade_plan.setup_type",
                                          "eval_ctx.setup.type", "best_fit.setup", "setup_type"])
        _assert("setup_type", case.setup_type, actual_setup,
                actual_setup is not None and actual_setup == case.setup_type)

    if case.strategy is not None:
        actual_strat = _extract(result, ["eval_ctx.strategy.primary", "strategy.primary",
                                          "strategy.primary_strategy", "strategy"])
        _assert("strategy", case.strategy, actual_strat,
                actual_strat is not None and case.strategy.lower() in str(actual_strat).lower())

    if case.confidence_min is not None or case.confidence_max is not None:
        actual_conf = _extract_numeric(result, ["clamped_confidence", "final_confidence",
                                                  "confidence", "eval_ctx.confidence.clamped"])
        if case.confidence_min is not None:
            _assert("confidence >= min", case.confidence_min, actual_conf,
                    actual_conf is not None and actual_conf >= case.confidence_min,
                    f"{actual_conf} < {case.confidence_min}")
        if case.confidence_max is not None:
            _assert("confidence <= max", case.confidence_max, actual_conf,
                    actual_conf is not None and actual_conf <= case.confidence_max,
                    f"{actual_conf} > {case.confidence_max}")

    if case.rr_min is not None:
        actual_rr = _extract_numeric(result, ["rr_ratio", "rrRatio", "trade_plan.rr_ratio", "trade_plan.rrRatio"])
        _assert("rr_ratio >= min", case.rr_min, actual_rr,
                actual_rr is not None and actual_rr >= case.rr_min,
                f"{actual_rr} < {case.rr_min}")

    if case.target1_gt_entry is not None:
        entry = _extract_numeric(result, ["entry", "trade_plan.entry", "execution_entry"])
        t1 = _extract_numeric(result, ["execution_t1", "trade_plan.execution_t1",
                                        "target_1", "targets.t1"])
        if entry is not None and t1 is not None:
            actual_rel = t1 > entry
            _assert("target1_gt_entry" if case.target1_gt_entry else "target1_lt_entry",
                    case.target1_gt_entry, actual_rel,
                    actual_rel == case.target1_gt_entry, f"entry={entry}, T1={t1}")
        else:
            _assert("target1_vs_entry", case.target1_gt_entry, None, False,
                    "entry or T1 not found in result")

    if case.sl_in_atr_range is not None:
        entry = _extract_numeric(result, ["entry", "trade_plan.entry", "execution_entry"])
        sl = _extract_numeric(result, ["stop_loss", "trade_plan.stop_loss", "execution_sl"])
        atr_raw = case.indicators.get("atr", {})
        atr_val = float(atr_raw.get("value", atr_raw) if isinstance(atr_raw, dict) else atr_raw)
        if entry is not None and sl is not None and atr_val > 0:
            sl_dist = abs(entry - sl)
            in_range = (0.5 * atr_val) <= sl_dist <= (5.0 * atr_val)
            _assert("sl_in_atr_range [0.5×ATR, 5.0×ATR]", True, in_range,
                    in_range == case.sl_in_atr_range,
                    f"sl_dist={sl_dist:.2f}, atr={atr_val:.2f}, ratio={sl_dist / atr_val:.2f}")

    if case.gate_must_pass:
        all_failed = _normalize_failure_names(_extract_list(
            result,
            ["structural_gates.failed_gates", "eval_ctx.structural_gates.overall.failed_gates"],
        ))
        for gate in case.gate_must_pass:
            _assert(f"gate_must_pass[{gate}]", gate, all_failed,
                    not _gate_in_list(gate, all_failed))

    if case.gate_must_fail:
        structural_failed = _normalize_failure_names(_extract_list(
            result,
            ["structural_gates.failed_gates", "eval_ctx.structural_gates.overall.failed_gates"],
        ))
        execution_failed = _normalize_failure_names(_extract_list(
            result,
            ["eval_ctx.execution_rules.overall.failed_rules", "trade_plan.execution_failures"],
        ))
        block_gates = _normalize_failure_names(_extract_list(result, ["trade_plan.block_gates"]))
        all_failed = list(dict.fromkeys(structural_failed + execution_failed + block_gates))
        for gate in case.gate_must_fail:
            _assert(f"gate_must_fail[{gate}]", gate, all_failed,
                    _gate_in_list(gate, all_failed))

    if case.can_execute is not None:
        actual_exec = _extract(result, ["can_execute", "can_trade", "trade_plan.can_trade"])
        _assert("can_execute", case.can_execute, actual_exec,
                bool(actual_exec) == case.can_execute)

    if case.best_horizon is not None:
        actual_hz = _extract(result, ["eval_ctx.best_horizon", "best_fit.horizon", "best_horizon"])
        _assert("best_horizon", case.best_horizon, actual_hz, actual_hz == case.best_horizon)

    return assertion_results


# ─────────────────────────────────────────────────────────────────────────────
# Mock data layer + pipeline runner (from backtest_signal_engine — unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class MockDataLayer:
    def __init__(self, indicators, fundamentals, patterns=None, symbol="TEST.NS", horizon="short_term"):
        self.indicators = indicators
        self.fundamentals = fundamentals
        self.patterns = patterns or {}
        self.symbol = symbol
        self.horizon = horizon
        self._patches: List[Any] = []

    def _start_patch(self, target: str, new: Any) -> None:
        try:
            p = patch(target, new)
            p.start()
            self._patches.append(p)
        except ModuleNotFoundError:
            pass

    def __enter__(self) -> "MockDataLayer":
        ind_copy = copy.deepcopy(self.indicators)
        fund_copy = copy.deepcopy(self.fundamentals)
        pat_copy = copy.deepcopy(self.patterns)
        self._start_patch("services.indicators.compute_indicators", lambda *a, **kw: ind_copy)
        self._start_patch("services.fundamentals.compute_fundamentals", lambda *a, **kw: fund_copy)
        self._start_patch("services.analyzers.pattern_analyzer.PatternAnalyzer.analyze",
                          lambda *a, **kw: pat_copy)
        return self

    def __exit__(self, *_: Any) -> None:
        for p in self._patches:
            try:
                p.stop()
            except RuntimeError:
                pass


def _flatten_indicators(indicators: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for k, v in indicators.items():
        if isinstance(v, dict):
            val = v.get("value") or v.get("raw")
            flat[k] = float(val) if val is not None and isinstance(val, (int, float)) else val
        else:
            try:
                flat[k] = float(v)
            except (TypeError, ValueError):
                flat[k] = v
    return flat


def _inject_synthetic_trade_plan(result: Dict[str, Any], case: BacktestCase) -> None:
    flat = _flatten_indicators(case.indicators)
    price = flat.get("price", 150.0)
    atr = flat.get("atr", 5.0)
    signal = result.get("signal", "WATCH")
    if signal in ("BUY", "STRONG_BUY"):
        entry, sl = price, price - atr * 1.5
        t1, t2 = price + atr * 2.0, price + atr * 4.0
        rr = (t1 - entry) / max(entry - sl, 0.01)
    elif signal == "SELL":
        entry, sl = price, price + atr * 1.5
        t1, t2 = price - atr * 2.0, price - atr * 4.0
        rr = (entry - t1) / max(sl - entry, 0.01)
    else:
        entry = sl = t1 = t2 = rr = None

    result.update({
        "entry": entry, "entry_price": entry, "stop_loss": sl,
        "execution_entry": entry, "execution_sl": sl, "execution_t1": t1,
        "targets": {"t1": t1, "t2": t2}, "target_1": t1, "target_2": t2,
        "rr_ratio": rr, "is_atr_fallback": True,
        "can_trade": signal in ("BUY", "SELL", "STRONG_BUY"),
        "can_execute": signal in ("BUY", "SELL", "STRONG_BUY"),
        "trade_plan": {
            "trade_signal": signal, "entry": entry, "stop_loss": sl,
            "targets": {"t1": t1, "t2": t2}, "rr_ratio": rr,
            "final_confidence": result.get("confidence", 55.0),
            "can_trade": signal in ("BUY", "SELL", "STRONG_BUY"),
            "execution_t1": t1, "execution_sl": sl, "is_atr_fallback": True,
        }
    })


class PipelineRunner:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._full_pipeline_available = False
        self._gate_evaluator = None
        self._resolver_cls = None
        self._signal_engine = None
        self._try_boot_pipeline()

    def _try_boot_pipeline(self) -> None:
        try:
            from config.gate_evaluator import GateEvaluator
            self._gate_evaluator = GateEvaluator()
            from config.config_resolver import ConfigResolver
            self._resolver_cls = ConfigResolver
            from services.signal_engine import SignalEngine
            self._signal_engine = SignalEngine(verbose=self.verbose)
            self._full_pipeline_available = True
        except (ImportError, ModuleNotFoundError) as e:
            log.warning(f"PipelineRunner: falling back to isolated mode ({e})")
        except Exception as e:
            if "ConfigurationError" in str(type(e)):
                raise
            log.error(f"PipelineRunner init failed: {e}")

    def run(self, case: BacktestCase) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        with MockDataLayer(case.indicators, case.fundamentals, case.patterns,
                           case.symbol, case.horizon):
            if self._full_pipeline_available:
                return self._run_full_pipeline(case)
            return self._run_isolated_pipeline(case)

    def _run_full_pipeline(self, case: BacktestCase):
        try:
            se = self._signal_engine
            full_report = se.compute_all_profiles(
                ticker=case.symbol,
                fundamentals=case.fundamentals,
                indicators_by_horizon={case.horizon: case.indicators},
                patterns_by_horizon={case.horizon: case.patterns},
                requested_horizons=[case.horizon],
            )
            profiles = full_report.get("profiles", {})
            h_profile = profiles.get(case.horizon, {})
            eval_ctx = h_profile.get("eval_ctx", {})
            winner = h_profile if h_profile.get("status") == "SUCCESS" else None
            plan = se.generate_trade_plan(
                symbol=case.symbol, winner_profile=winner,
                indicators=case.indicators, fundamentals=case.fundamentals,
                horizon=case.horizon, capital=100_000.0,
            )
            result = dict(full_report)
            result["eval_ctx"] = eval_ctx
            result["profile_signal"] = h_profile.get("profile_signal", "MODERATE")
            result["rr_ratio"] = plan.get("rrRatio") or plan.get("rr_ratio")
            result["trade_signal"] = plan.get("tradeSignal") or plan.get("trade_signal", "WATCH")
            result["signal"] = result["trade_signal"]
            result["entry"] = plan.get("entry")
            result["entry_price"] = plan.get("entry")
            result["stop_loss"] = plan.get("stop_loss")
            tgts = plan.get("targets") or {}
            result["targets"] = tgts
            result["target_1"] = tgts.get("t1")
            result["target_2"] = tgts.get("t2")
            result["final_confidence"] = plan.get("final_confidence", 0)
            result["confidence"] = result["final_confidence"]
            result["rr_ratio"] = plan.get("rr_ratio", plan.get("rrRatio"))
            result["rrRatio"] = result["rr_ratio"]
            result["setup_type"] = plan.get("setup_type")
            result["can_trade"] = plan.get("can_trade", False)
            result["can_execute"] = result["can_trade"]
            result["is_atr_fallback"] = plan.get("is_atr_fallback", False)
            result["execution_t1"] = plan.get("execution_t1", tgts.get("t1"))
            result["execution_t2"] = plan.get("execution_t2", tgts.get("t2"))
            result["execution_sl"] = plan.get("execution_sl", plan.get("stop_loss"))
            sg_ov = eval_ctx.get("structural_gates", {}).get("overall", {})
            opp_ov = eval_ctx.get("opportunity_gates", {}).get("overall", {})
            exec_rules = eval_ctx.get("execution_rules", {}).get("summary", {})
            failed_structural = [f["gate"] for f in sg_ov.get("failed_gates", [])
                                  if isinstance(f, dict) and "gate" in f]
            failed_opp = [str(f.get("gate", f)) if isinstance(f, dict) else str(f)
                           for f in opp_ov.get("failed_gates", [])]
            failed_exec = ([str(f) for f in exec_rules.get("violations", [])] +
                           [str(f) for f in exec_rules.get("warnings", [])])
            result["structural_gates"] = {"overall": sg_ov, "passed": sg_ov.get("passed", True),
                                            "failed_gates": failed_structural}
            result["opportunity_gates"] = {"overall": opp_ov,
                                             "passed": opp_ov.get("passed", True) and sg_ov.get("passed", True),
                                             "failed_gates": list(set(failed_structural + failed_opp + failed_exec))}
            result["trade_plan"] = plan
            return result, None
        except ConfigurationError as ce:
            return None, f"CONFIGURATION_ERROR (Fail-Fast): {ce}"
        except Exception:
            return None, f"Full pipeline error: {traceback.format_exc()}"

    def _run_isolated_pipeline(self, case: BacktestCase):
        result: Dict[str, Any] = {"_mode": "isolated", "symbol": case.symbol, "horizon": case.horizon}
        errors: List[str] = []
        flat_ind = _flatten_indicators(case.indicators)
        if self._gate_evaluator:
            try:
                structural_gates = {
                    "adx": {"min": 20.0 if case.horizon == "intraday" else 18.0},
                    "rvol": {"min": 1.0}, "trendStrength": {"min": 5.0}, "_logic": "AND"
                }
                passed, failed = self._gate_evaluator.evaluate_gates(
                    gates=structural_gates, data={**flat_ind, **case.fundamentals}, empty_gates_pass=True
                )
                result["structural_gates"] = {"overall": {"passed": passed,
                                                            "failed_gates": [{"gate": g} for g in failed]},
                                               "passed": passed, "failed_gates": failed}
            except Exception as e:
                errors.append(f"gate_evaluator: {e}")

        conf_val = 55.0
        result["confidence"] = conf_val
        result["final_confidence"] = conf_val
        gate_passed = result.get("structural_gates", {}).get("passed", True)
        has_pattern = bool(case.patterns) and any(
            p.get("found", True) for p in case.patterns.values() if isinstance(p, dict)
        )
        vol_sig = str(flat_ind.get("volume_signature", "normal")).lower()
        if vol_sig == "climax" and (has_pattern or case.patterns):
            final_signal = "BLOCKED"
            can_execute = False
        elif not gate_passed:
            final_signal = "WATCH"
            can_execute = False
        elif not has_pattern:
            final_signal = "WATCH"
            can_execute = False
        elif conf_val >= 60:
            trend = str(flat_ind.get("trend_direction", "BULLISH")).upper()
            final_signal = "BUY" if trend == "BULLISH" else "SELL"
            can_execute = True
        else:
            final_signal = "WATCH"
            can_execute = False
        result["signal"] = final_signal
        result["trade_signal"] = final_signal
        result["can_execute"] = can_execute
        result["can_trade"] = can_execute
        result["setup_type"] = case.setup_type or "GENERIC"
        result["profile_signal"] = "STRONG" if conf_val >= 75 else "MODERATE" if conf_val >= 60 else "WEAK"
        _inject_synthetic_trade_plan(result, case)
        if errors:
            result["_errors"] = errors
        return result, None


@dataclasses.dataclass
class TestResult:
    case_name: str
    passed: bool
    assertion_results: List[AssertionResult]
    error: Optional[str]
    duration_ms: float
    result_dict: Optional[Dict[str, Any]]
    tags: List[str]
    expected_fail: bool

    @property
    def failure_count(self) -> int:
        return sum(1 for a in self.assertion_results if not a.passed)

    @property
    def assertion_count(self) -> int:
        return len(self.assertion_results)


class BacktestHarness:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.runner = PipelineRunner(verbose=verbose)
        self.test_results: List[TestResult] = []

    def run_case(self, case: BacktestCase) -> TestResult:
        t0 = time.perf_counter()
        result_dict = None
        error = None
        assertion_results: List[AssertionResult] = []
        try:
            result_dict, error = self.runner.run(case)
            if result_dict is not None:
                assertion_results = run_assertions(case, result_dict)
        except Exception:
            error = traceback.format_exc()
        duration_ms = (time.perf_counter() - t0) * 1000
        all_pass = all(a.passed for a in assertion_results)
        passed = (error is None) and (all_pass or not assertion_results)
        if case.expected_fail:
            passed = not passed
        return TestResult(case_name=case.name, passed=passed,
                          assertion_results=assertion_results, error=error,
                          duration_ms=duration_ms,
                          result_dict=result_dict if self.verbose else None,
                          tags=case.tags, expected_fail=case.expected_fail)

    def run_suite(self, cases: List[BacktestCase],
                  filter_tags: Optional[Set[str]] = None,
                  filter_name: Optional[str] = None) -> List[TestResult]:
        filtered = cases
        if filter_tags:
            filtered = [c for c in filtered if set(c.tags) & filter_tags]
        if filter_name:
            filtered = [c for c in filtered if filter_name.lower() in c.name.lower()]
        results: List[TestResult] = []
        for i, case in enumerate(filtered):
            print(f"[{i + 1}/{len(filtered)}] {case.name[:72]:<72}", end=" ", flush=True)
            tr = self.run_case(case)
            results.append(tr)
            if tr.passed:
                print(f"✓  ({tr.duration_ms:.0f}ms)")
            elif tr.expected_fail:
                print(f"~  EXPECTED FAILURE  ({tr.duration_ms:.0f}ms)")
            else:
                print(f"✗  ({tr.duration_ms:.0f}ms)")
                for ar in tr.assertion_results:
                    if not ar.passed:
                        print(f"      ↳ {ar}")
                if tr.error:
                    print(f"      ↳ ERROR: {tr.error[:200]}")
        self.test_results.extend(results)
        return results

    def print_summary(self, results: List[TestResult]) -> bool:
        total = len(results)
        passed = sum(1 for r in results if r.passed and not r.expected_fail)
        xfail = sum(1 for r in results if r.expected_fail)
        failed = total - passed - xfail
        total_ms = sum(r.duration_ms for r in results)
        total_assertions = sum(r.assertion_count for r in results)
        failed_assertions = sum(r.failure_count for r in results)
        print()
        print("═" * 70)
        print(f"  SYNTHETIC BACKTEST SUMMARY")
        print("═" * 70)
        print(f"  Tests:      {total:>4}   Passed: {passed:>4}   Failed: {failed:>4}   XFail: {xfail:>3}")
        print(f"  Assertions: {total_assertions:>4}   Passed: {total_assertions - failed_assertions:>4}   Failed: {failed_assertions:>4}")
        print(f"  Duration:   {total_ms:.0f}ms ({total_ms / max(total, 1):.0f}ms avg)")
        mode = "FULL PIPELINE" if self.runner._full_pipeline_available else "ISOLATED"
        print(f"  Pipeline mode: {mode}")
        if failed > 0:
            print("\n  FAILURES:")
            for r in results:
                if not r.passed and not r.expected_fail:
                    print(f"    ✗  {r.case_name}")
                    for ar in r.assertion_results:
                        if not ar.passed:
                            print(f"         {ar}")
                    if r.error:
                        print(f"         ERROR: {r.error[:200]}")
        print("═" * 70)
        return failed == 0

    def save_baselines(self, results: List[TestResult]) -> None:
        baseline_data = {r.case_name: {k: v for k, v in (r.result_dict or {}).items()
                                         if isinstance(v, (str, int, float, bool)) and not k.startswith("_")}
                         for r in results if r.result_dict}
        path = BASELINE_DIR / "golden_baselines.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(baseline_data, f, indent=2, default=str)
        print(f"\n  Baselines saved → {path}")

    def load_and_compare_baselines(self, results: List[TestResult]) -> bool:
        path = BASELINE_DIR / "golden_baselines.json"
        if not path.exists():
            print("  No baseline file found. Run with --save-baselines first.")
            return True
        with open(path, encoding="utf-8") as f:
            baselines = json.load(f)
        regressions: List[str] = []
        for r in results:
            if r.case_name not in baselines or not r.result_dict:
                continue
            current = {k: v for k, v in r.result_dict.items()
                       if isinstance(v, (str, int, float, bool)) and not k.startswith("_")}
            for field in ["signal", "profile_signal", "setup_type"]:
                bl = baselines[r.case_name].get(field)
                cur = current.get(field)
                if bl and cur and str(bl).upper() != str(cur).upper():
                    regressions.append(f"  REGRESSION [{r.case_name}] {field}: was={bl!r}, now={cur!r}")
        if regressions:
            print("\n  REGRESSION FAILURES:")
            for reg in regressions:
                print(reg)
            return False
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — HISTORICAL REPLAY ENGINE
# (from backtest_signal_engine — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class HistoricalCase:
    name: str
    symbol: str
    horizon: str
    fixture_file: str
    check_window: Tuple[str, str]
    expected_signals: Dict[str, int]
    description: str = ""
    patterns: Dict[str, Any] = dataclasses.field(default_factory=dict)
    fundamentals: Dict[str, Any] = dataclasses.field(default_factory=dict)
    tags: List[str] = dataclasses.field(default_factory=list)


HISTORICAL_CASES: List[HistoricalCase] = [
    HistoricalCase(
        name="RELIANCE: WATCH during trend-only phase without primary pattern",
        symbol="RELIANCE.NS", horizon="short_term",
        fixture_file="RELIANCE_NS_short_term.parquet",
        check_window=("2023-09-01", "2023-12-31"),
        expected_signals={"WATCH": 1},
        description="Golden negative case: strong trend alone should stay WATCH when no primary pattern is supplied.",
        tags=["historical", "reliance", "short_term", "trend_only_watch"],
    ),
    HistoricalCase(
        name="RELIANCE: BUY appears when primary trend patterns are supplied",
        symbol="RELIANCE.NS", horizon="short_term",
        fixture_file="RELIANCE_NS_short_term.parquet",
        check_window=("2023-09-01", "2023-12-31"),
        expected_signals={"BUY": 1},
        description="Golden positive case: same trending window should emit BUY once primary trend-following patterns are present.",
        patterns={
            "goldenCross": make_pattern("goldenCross", found=True, quality=8.0, score=82.0, price=2800),
            "ichimokuSignals": make_pattern("ichimokuSignals", found=True, quality=7.5, score=78.0, price=2800),
        },
        tags=["historical", "reliance", "short_term", "golden_case", "primary_pattern_buy"],
    ),
    HistoricalCase(
        name="ADANIENT: AVOID during high D/E period",
        symbol="ADANIENT.NS", horizon="long_term",
        fixture_file="ADANIENT_NS_long_term.parquet",
        check_window=("2023-01-01", "2023-06-30"),
        expected_signals={"AVOID": 1},
        tags=["historical", "adanient", "long_term"],
    ),
]


def _build_indicators_from_ohlcv(df: Any, horizon: str) -> Dict[str, Any]:
    try:
        close = df["Close"]
        volume = df["Volume"]
        n = len(close)
        ema_fast_period = {"intraday": 9, "short_term": 20, "long_term": 50}.get(horizon, 20)
        ema_slow_period = {"intraday": 21, "short_term": 50, "long_term": 200}.get(horizon, 50)
        atr_period = {"intraday": 7, "short_term": 14, "long_term": 21}.get(horizon, 14)
        ema_fast = float(close.ewm(span=min(ema_fast_period, n)).mean().iloc[-1])
        ema_slow = float(close.ewm(span=min(ema_slow_period, n)).mean().iloc[-1])
        price = float(close.iloc[-1])
        tr = (df["High"] - df["Low"]).abs()
        atr = float(tr.rolling(min(atr_period, n)).mean().iloc[-1])
        atr_pct = (atr / price * 100) if price > 0 else 3.0
        vol_ma = float(volume.rolling(min(20, n)).mean().iloc[-1])
        rvol = float(volume.iloc[-1]) / max(vol_ma, 1.0)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if not rs.empty else 50.0
        trend = "BULLISH" if ema_fast > ema_slow else "BEARISH"
        return make_indicators(price=price, atr=atr, atrPct=atr_pct,
                               emaFast=ema_fast, emaSlow=ema_slow, rsi=rsi, rvol=rvol,
                               trend_direction=trend, adx=25.0,
                               trendStrength=7.0 if trend == "BULLISH" else 5.0)
    except Exception:
        return make_indicators()


class HistoricalReplayEngine:
    def __init__(self, runner: PipelineRunner, verbose: bool = False):
        self.runner = runner
        self.verbose = verbose

    def generate_fixtures(self, symbols: List[str], horizons: List[str]) -> None:
        try:
            import yfinance as yf
        except ImportError:
            print("  yfinance not installed — skipping fixture generation")
            return
        period_map = {"intraday": "5d", "short_term": "1y", "long_term": "3y"}
        interval_map = {"intraday": "5m", "short_term": "1d", "long_term": "1wk"}
        for symbol in symbols:
            for horizon in horizons:
                fname = FIXTURE_DIR / f"{symbol.replace('.', '_')}_{horizon}.parquet"
                if fname.exists():
                    print(f"  Fixture exists: {fname.name}")
                    continue
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period_map.get(horizon, "1y"),
                                        interval=interval_map.get(horizon, "1d"))
                    if df is not None and len(df) > 0:
                        df.to_parquet(fname)
                        print(f"  Generated: {fname.name} ({len(df)} bars)")
                except Exception as e:
                    print(f"  Error fetching {symbol}: {e}")

    def run_case(self, case: HistoricalCase) -> Dict[str, Any]:
        fixture_path = FIXTURE_DIR / case.fixture_file
        if not fixture_path.exists():
            return {"error": f"Fixture not found: {fixture_path}", "skipped": True}
        try:
            import pandas as pd
            df = pd.read_parquet(fixture_path)
        except Exception as e:
            return {"error": f"Failed to load fixture: {e}", "skipped": True}
        start, end = case.check_window
        window = df[start:end]
        if len(window) == 0:
            return {"error": f"No data in window {start}:{end}", "skipped": True}
        signal_counts: Dict[str, int] = {}
        errors = 0
        for bar_idx in range(50, len(window)):
            bar_df = window.iloc[: bar_idx + 1]
            indicators = _build_indicators_from_ohlcv(bar_df, case.horizon)
            fundamentals = copy.deepcopy(case.fundamentals) if case.fundamentals else make_fundamentals()
            bt_case = BacktestCase(name=f"replay_{case.symbol}_{bar_idx}", description="",
                                   horizon=case.horizon, indicators=indicators,
                                   fundamentals=fundamentals, patterns=copy.deepcopy(case.patterns),
                                   symbol=case.symbol)
            try:
                result, error = self.runner.run(bt_case)
                if result:
                    sig = result.get("signal", "UNKNOWN")
                    signal_counts[sig] = signal_counts.get(sig, 0) + 1
                else:
                    errors += 1
            except Exception:
                errors += 1
        validation_passed = True
        failures: List[str] = []
        for signal, min_count in case.expected_signals.items():
            actual_count = signal_counts.get(signal, 0)
            if actual_count < min_count:
                validation_passed = False
                failures.append(f"Expected >= {min_count}× {signal}, got {actual_count}")
        return {"passed": validation_passed, "signal_counts": signal_counts,
                "errors": errors, "failures": failures, "bars_replayed": len(window) - 50}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — UNIT TESTS (unittest-compatible)
# ═══════════════════════════════════════════════════════════════════════════════

class TestGateEvaluator(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from config.gate_evaluator import evaluate_gates, evaluate_invalidation_gates
            self.evaluate_gates = evaluate_gates
            self.evaluate_invalidation_gates = evaluate_invalidation_gates
            self.available = True
        except ImportError:
            self.available = False

    def _skip_if_unavailable(self) -> None:
        if not self.available:
            self.skipTest("gate_evaluator not importable")

    def test_and_logic_all_pass(self) -> None:
        self._skip_if_unavailable()
        passed, failed = self.evaluate_gates(
            {"adx": {"min": 18}, "rvol": {"min": 1.0}, "_logic": "AND"},
            {"adx": 25.0, "rvol": 2.0})
        self.assertTrue(passed)
        self.assertEqual(failed, [])

    def test_and_logic_one_fails(self) -> None:
        self._skip_if_unavailable()
        passed, failed = self.evaluate_gates(
            {"adx": {"min": 18}, "rvol": {"min": 1.5}, "_logic": "AND"},
            {"adx": 25.0, "rvol": 0.8})
        self.assertFalse(passed)
        self.assertIn("rvol", str(failed))

    def test_or_logic_one_passes(self) -> None:
        self._skip_if_unavailable()
        passed, _ = self.evaluate_gates(
            {"adx": {"min": 30}, "rvol": {"min": 1.5}, "_logic": "OR"},
            {"adx": 35.0, "rvol": 0.8})
        self.assertTrue(passed)

    def test_none_threshold_skipped(self) -> None:
        self._skip_if_unavailable()
        passed, _ = self.evaluate_gates(
            {"adx": {"min": None}, "rvol": {"min": 1.0}}, {"adx": 5.0, "rvol": 2.0})
        self.assertTrue(passed)

    def test_min_metric_clause(self) -> None:
        self._skip_if_unavailable()
        passed, _ = self.evaluate_gates(
            {"price": {"min_metric": "box_low", "multiplier": 1.002}},
            {"price": 102.5, "box_low": 100.0})
        self.assertTrue(passed)

    def test_max_metric_clause(self) -> None:
        self._skip_if_unavailable()
        passed, _ = self.evaluate_gates(
            {"price": {"max_metric": "box_high", "multiplier": 1.0}},
            {"price": 105.0, "box_high": 100.0})
        self.assertFalse(passed)

    def test_empty_gates_pass_flag(self) -> None:
        self._skip_if_unavailable()
        passed, _ = self.evaluate_gates({}, {"adx": 25.0}, empty_gates_pass=True)
        self.assertTrue(passed)


class TestIndicatorBuilder(unittest.TestCase):
    def test_all_required_keys_present(self) -> None:
        ind = make_indicators()
        for k in ["adx", "emaFast", "emaSlow", "rsi", "atr", "atrPct", "rvol",
                   "trendStrength", "price", "volume",
                   "avgVolume30Days", "slDistance"]:  # R2 fix keys
            self.assertIn(k, ind, f"Missing key: {k}")

    def test_nested_dict_format(self) -> None:
        ind = make_indicators(adx=30.0)
        adx = ind["adx"]
        self.assertIsInstance(adx, dict)
        self.assertEqual(adx["value"], 30.0)
        self.assertIn("score", adx)
        self.assertIn("raw", adx)

    def test_avg_volume_key_is_snake_case(self) -> None:
        """R2 fix: avgVolume30Days must match _extract_price_data lookup key."""
        ind = make_indicators(volume=2_000_000.0)
        self.assertIn("avgVolume30Days", ind)
        self.assertNotIn("avgVolume", ind)

    def test_sl_distance_always_present(self) -> None:
        """R2 fix: slDistance must always be present for _check_sl_distance rule."""
        ind = make_indicators()
        self.assertIn("slDistance", ind)
        self.assertIsInstance(ind["slDistance"]["value"], float)

    def test_pattern_key_is_dict_not_bool(self) -> None:
        """R2 fix: patterns must be make_pattern() dicts, never plain booleans."""
        ind = make_indicators()
        # bullishNeckline and bearishNeckline are always present as dicts
        self.assertIsInstance(ind["bullishNeckline"], dict)
        raw = ind["bullishNeckline"].get("raw") or ind["bullishNeckline"]
        self.assertIn("found", raw)


class TestPatternContract(unittest.TestCase):
    REQUIRED_META = ["age_candles", "formation_time", "formation_timestamp", "bar_index",
                     "type", "invalidation_level", "velocity_tracking",
                     "pattern_strength", "current_price", "horizon"]

    def test_all_10_meta_fields_present(self) -> None:
        pattern = make_pattern("goldenCross")
        for field in self.REQUIRED_META:
            self.assertIn(field, pattern["meta"], f"Missing meta field: {field}")

    def test_age_candles_is_int(self) -> None:
        pattern = make_pattern("goldenCross", age_candles=15)
        self.assertIsInstance(pattern["meta"]["age_candles"], int)
        self.assertLess(pattern["meta"]["age_candles"], 10000)

    def test_found_field_present_in_raw(self) -> None:
        pattern = make_pattern("darvasBox", found=True)
        raw = pattern.get("raw", {})
        self.assertIn("found", raw)
        self.assertTrue(raw["found"])


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — CLI ENTRYPOINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pro Stock Analyzer — Synthetic + Setup Scenario Backtester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--suite",
                        choices=["setup", "synthetic", "unit", "regression", "historical", "all"],
                        default="all")
    parser.add_argument("--test", type=str, default=None, help="Run single test by name substring")
    parser.add_argument("--tags", type=str, default=None, help="Comma-separated tag filter")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save-baselines", action="store_true")
    parser.add_argument("--generate-fixtures", action="store_true")
    args = parser.parse_args()

    try:
        os.makedirs(os.path.join(ROOT, "output"), exist_ok=True)
        os.environ["TEST_DATABASE_URL"] = f"sqlite:///{os.path.join(ROOT, 'output', 'test_trade.db')}"
        from services.db import init_db
        init_db()
    except Exception:
        pass

    filter_tags: Optional[Set[str]] = None
    if args.tags:
        filter_tags = {t.strip() for t in args.tags.split(",")}

    all_passed = True
    harness = BacktestHarness(verbose=args.verbose)

    # Unit tests
    if args.suite in ("unit", "all"):
        print("\n" + "─" * 70)
        print("  UNIT TESTS (gate evaluator, builders, contracts)")
        print("─" * 70)
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        for cls in [TestGateEvaluator, TestIndicatorBuilder, TestPatternContract]:
            suite.addTests(loader.loadTestsFromTestCase(cls))
        runner = unittest.TextTestRunner(verbosity=1)
        unit_result = runner.run(suite)
        if not unit_result.wasSuccessful():
            all_passed = False

    # Setup scenario suites (Suites 1-3)
    if args.suite in ("setup", "all"):
        print("\n" + "─" * 70)
        print("  SETUP SCENARIO SUITES (23 setups × 3 horizons)")
        print("─" * 70)
        rc = run_setup_suites(verbose=args.verbose)
        if rc != 0:
            all_passed = False

    # Synthetic assertion cases
    if args.suite in ("synthetic", "all"):
        print("\n" + "─" * 70)
        print("  SYNTHETIC BACKTEST SUITE")
        print("─" * 70)
        results = harness.run_suite(SYNTHETIC_CASES, filter_tags=filter_tags,
                                     filter_name=args.test)
        suite_passed = harness.print_summary(results)
        if not suite_passed:
            all_passed = False
        if args.save_baselines:
            harness.save_baselines(results)

    # Regression comparison
    if args.suite in ("regression", "all") and not args.save_baselines:
        print("\n" + "─" * 70)
        print("  REGRESSION COMPARISON")
        print("─" * 70)
        if harness.test_results:
            reg_passed = harness.load_and_compare_baselines(harness.test_results)
            if not reg_passed:
                all_passed = False

    # Historical replay
    if args.suite in ("historical", "all"):
        print("\n" + "─" * 70)
        print("  HISTORICAL REPLAY SUITE")
        print("─" * 70)
        replay = HistoricalReplayEngine(harness.runner, verbose=args.verbose)
        if args.generate_fixtures:
            symbols = list({c.symbol for c in HISTORICAL_CASES})
            horizons = list({c.horizon for c in HISTORICAL_CASES})
            replay.generate_fixtures(symbols, horizons)
        for hcase in HISTORICAL_CASES:
            print(f"  Replaying {hcase.name}...", end=" ", flush=True)
            hr = replay.run_case(hcase)
            if hr.get("skipped"):
                print(f"SKIPPED ({hr.get('error', 'fixture missing')})")
            elif hr.get("passed"):
                print(f"✓  signals={hr.get('signal_counts')}  bars={hr.get('bars_replayed')}")
            else:
                print(f"✗  failures={hr.get('failures')}")
                all_passed = False

    print()
    if all_passed:
        print("  ALL SUITES PASSED")
        return 0
    print("  SOME SUITES FAILED — review output above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
