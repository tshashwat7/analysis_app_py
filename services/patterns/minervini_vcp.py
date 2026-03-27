"""
services/patterns/minervini_vcp.py

Downstream meta consumers
─────────────────────────
trade_enhancer (via raw.meta path):
  • meta["age_candles"]       – check_pattern_expiration + extract_pattern_execution_metadata
  • meta["formation_time"]    – real-time age recalculation

setup_pattern_matrix PATTERN_METADATA (minerviniStage2) invalidation:
  • meta["contraction_pct"]   – metadata_keys analytics + adaptive target scaling
  • meta["volatility_quality"] – metadata_keys analytics (flat scalar for namespace eval)

# Target calculation now handled by Stage 2 (TradeEnhancer) via Market-Adaptive Adjustment.
# Resolver provides structural baseline; Enhancer optimizes based on volatility/regime.

Alias fix applied
─────────────────
  alias = "minerviniStage2"  (was "minerviniVCP" — broke every matrix / resolver lookup)  ✅
  Point 5: self._get_val() used throughout — no direct data_fetch import                   ✅
  _guard() used                                                                             ✅
  formation_time key used (matches trade_enhancer L374/946)                                 ✅
"""

import pandas as pd
from typing import Dict, Any
from services.patterns.base import BasePattern
from services.patterns.utils import _build_formation_context
from services.patterns.horizon_constants import HORIZON_WINDOWS_BARS


class MinerviniVCPPattern(BasePattern):
    """
    Minervini-style Volatility Contraction Pattern (VCP / Stage 2).
    Alias: "minerviniStage2" — matches SETUP_PATTERN_MATRIX and config_resolver keys.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.alias              = "minerviniStage2"
        self.horizons_supported = ["short_term", "long_term", "multibagger"]

    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if not self._is_horizon_supported(horizon):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)

        hw_cfg = HORIZON_WINDOWS_BARS.get(horizon, HORIZON_WINDOWS_BARS["short_term"])
        empty = self._guard(df, hw_cfg["min_history"])
        if empty is not None:
            return empty

        maMid  = self._get_val(indicators, "maMid")
        maSlow = self._get_val(indicators, "maSlow")
        close  = df["Close"].iloc[-1]

        # Stage 2 requirement: price > 50 MA > 200 MA
        if not (maMid and maSlow and close > maMid and maMid > maSlow):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        # ATR volatility gate — VCP requires tight daily action
        atrPct = self._get_val(indicators, "atrPct")
        if atrPct and atrPct > 3.5:
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        # VCP contraction: compare recent 5 bars vs prior 10 bars
        range_recent = (df["High"].iloc[-5:].max()    - df["Low"].iloc[-5:].min())    / close
        range_prev   = (df["High"].iloc[-15:-5].max() - df["Low"].iloc[-15:-5].min()) / close

        is_contracting = range_recent < (range_prev * 0.7)
        is_tight       = range_recent < 0.05

        if not (is_contracting and is_tight):
            return {"found": False, "score": 0, "quality": 0, "meta": {}}

        qual = 7.0
        vol_recent = df["Volume"].iloc[-5:].mean()
        vol_avg    = df["Volume"].iloc[-50:].mean()
        vol_dry    = bool(vol_recent < vol_avg)
        if vol_dry:
            qual += 2.0

        result = {
            "found":   True,
            "score":   self._normalize_score(qual * 10),
            "quality": min(qual, 10.0),
            "desc":    "Minervini VCP (Tight)",
        }

        formation_index = len(df) - 15
        position52w     = self._get_val(indicators, "position52w")
        pivot_point     = self._get_val(indicators, "pivotPoint")
        # Point 5 fix: self._get_val() — no direct data_fetch import
        volatility_quality = self._get_val(indicators, "volatilityQuality")

        invalidation_level = maMid * 0.95 if maMid else None
        entry_conditions_met = is_tight and vol_dry

        stage_quality = (
            "stage2_confirmed" if position52w and position52w >= 85 else
            "transitioning"    if position52w and position52w >= 70 else
            "weak"
        )
        contraction_strength = "tight"    if range_recent < 0.02 else "moderate"
        pattern_strength = (
            "strong"   if result["quality"] >= 8.5 else
            "moderate" if result["quality"] >= 6.5 else
            "weak"
        )

        result["meta"] = {
            # ── Canonical contract field (required by all detectors) ──────────
            "type": "bullish",
            # ── Fields read by trade_enhancer ────────────────────────────────
            "age_candles":    len(df) - formation_index,
            "formation_time": float(df.index[formation_index].timestamp()),
            # ── Fields read by Stage 2 (TradeEnhancer) Target Scaling ──────────
            "contraction_pct": round(range_recent * 100, 2),
            # ── Fields in PATTERN_METADATA invalidation metadata_keys analytics ─
            "volatility_quality": volatility_quality,           # flat scalar for namespace eval
            # ── Structural / UI fields ────────────────────────────────────────
            "bar_index":          len(df),
            "tightness":          f"{range_recent * 100:.1f}%",
            "vol_dry":            vol_dry,
            "formation_timestamp": df.index[formation_index].isoformat(),
            "invalidation_level":   round(invalidation_level, 2) if invalidation_level else None,
            "pivot_point":          round(pivot_point, 2)         if pivot_point         else None,
            "maFast":               round(maMid,  2)              if maMid               else None,
            "maSlow":               round(maSlow, 2)              if maSlow              else None,
            "position52w":          position52w,
            "stage_quality":        stage_quality,
            "contraction_strength": contraction_strength,
            "horizon":              horizon,
            "pattern_strength":     pattern_strength,
            "current_price":        round(float(close), 2),
            "velocity_tracking": {
                "can_track":            result["quality"] >= 7.0 and entry_conditions_met,
                "entry_conditions_met": entry_conditions_met,
                "quality_sufficient":   result["quality"] >= 7.0,
                "contraction_tight":    is_tight,
                "volume_dry":           vol_dry,
            },
            "formation_context": _build_formation_context(indicators),
        }

        return result