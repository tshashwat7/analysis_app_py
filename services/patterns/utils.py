from typing import Any, Dict, Optional

import pandas as pd

def _classify_volatility(atr_pct: Optional[float]) -> str:
    """Classify volatility regime."""
    if atr_pct is None:
        return "unknown"
    if atr_pct > 5.0:
        return 'extreme'
    if atr_pct > 3.0:
        return "high"
    elif atr_pct > 1.5:
        return "normal"
    else:
        return "low"

def _build_velocity_tracking_flags(
    quality: float,
    indicators: Dict[str, Any],
    description: str
) -> Dict[str, Any]:
    """
    Determines if this pattern should be tracked for velocity analytics.
    
    Args:
        quality: Pattern quality score
        indicators: Technical indicators
        description: Pattern description (e.g., "Darvas Box Breakout")
    
    Returns:
        {
            "can_track": bool,           # Is pattern suitable for tracking?
            "entry_conditions_met": bool, # Are entry conditions satisfied?
            "quality_sufficient": bool    # Is quality high enough?
        }
    """
    # Quality threshold
    quality_sufficient = quality >= 7.0
    
    # Entry conditions check
    # Different patterns have different entry criteria
    entry_conditions_met = True  # Default
    
    # For breakout patterns, check if breakout confirmed
    if "breakout" in description.lower():
        rvol = indicators.get("rvol", {})
        rvol_val = rvol.get("value") if isinstance(rvol, dict) else rvol
        
        # Breakout needs volume
        entry_conditions_met = (rvol_val and rvol_val >= 1.3)
    
    # For consolidation patterns, check if tight enough
    elif "squeeze" in description.lower() or "consolidat" in description.lower():
        bb_width = indicators.get("bbWidth", {})
        width_val = bb_width.get("value") if isinstance(bb_width, dict) else bb_width
        
        # Consolidation needs tight BB
        entry_conditions_met = (width_val and width_val <= 5.0)
    
    # Pattern can be tracked if all conditions met
    can_track = quality_sufficient and entry_conditions_met
    
    return {
        "can_track": can_track,
        "entry_conditions_met": entry_conditions_met,
        "quality_sufficient": quality_sufficient
    }


# ============================================================
# HELPER: Formation Context
# ============================================================

def _build_formation_context(indicators: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts market context at pattern formation.
    
    This data is used to filter historical velocity stats
    (e.g., "show me darvasBox performance in strong trends only")
    
    Returns:
        {
            "adx": float,
            "trend_strength": float,
            "volatility_regime": str,  # "high", "normal", "low"
            "momentum_strength": float
        }
    """
    # Safe extraction helper
    def _get_val(key):
        val = indicators.get(key, {})
        if isinstance(val, dict):
            return val.get("value")
        return val
    
    # Extract values
    adx = _get_val("adx")
    trend_strength = _get_val("trendStrength")
    momentum_strength = _get_val("momentumStrength")
    atr_pct = _get_val("atrPct")
    
    # Classify volatility regime
    if atr_pct is None:
        volatility_regime = "unknown"
    elif atr_pct > 4.0:
        volatility_regime = "high"
    elif atr_pct > 2.0:
        volatility_regime = "normal"
    else:
        volatility_regime = "low"
    
    return {
        "adx": adx,
        "trend_strength": trend_strength,
        "volatility_regime": volatility_regime,
        "momentum_strength": momentum_strength
    }
    
def ensure_numeric_df(self, df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
    """
    Defensive coercion helper: ensures requested columns are numeric.
    - strips commas and percent signs, then uses pd.to_numeric(errors='coerce')
    - returns the modified DataFrame (shallow copy semantics)
    Usage: df = self.ensure_numeric_df(df)
    """
    if df is None:
        return df

    if cols is None:
        cols = self.numeric_cols or []

    # operate on a copy to avoid surprising caller-side mutation
    out = df.copy()
    try:
        for c in cols:
            if c not in out.columns:
                continue

            # convert object-like columns (strings) into numeric safely
            # convert to string first so we can strip commas, percent symbols etc.
            if out[c].dtype == "object" or self.coerce_numeric:
                cleaned = out[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
                out[c] = pd.to_numeric(cleaned, errors="coerce")
    except Exception as e:
        # Do not raise — log and return what we have
        self.log_debug(f"ensure_numeric_df error: {e}")
    return out

__all__ = ["_classify_volatility", "ensure_numeric_df", "_build_velocity_tracking_flags", "_build_formation_context"]