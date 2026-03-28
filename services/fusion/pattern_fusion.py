import pandas as pd
from typing import Dict, Any

def merge_pattern_into_indicators(indicators: Dict[str, Any], pattern_results: Dict[str, Any], horizon: str = None, df: pd.DataFrame = None):
    """
    Injects pattern results into the main indicators dictionary.
    UI Standard Format: { "value": X, "raw": Y, "score": Z, "desc": "..." }
    
    NOTE: This function mutates the 'indicators' dictionary in-place.
    """
    for alias, result in pattern_results.items():
        if not result.get("found", False):
            continue
            
        # P3-1: Quality-check to prevent overwriting better patterns
        existing = indicators.get(alias)
        if existing and existing.get("value", 0) > result.get("quality", 0):
            continue

        # Fix 6.3-1: Timestamp Safety (handles non-DatetimeIndex)
        try:
            ts = float(df.index[-1].timestamp()) if df is not None and not df.empty else None
        except (AttributeError, OSError, ValueError):
            ts = None

        # Fix 6.4-1: CamelCase Fallback Labeling
        def _split_camel(s):
            import re
            return re.sub(r'([a-z])([A-Z])', r'\1 \2', s).title()

        indicators[alias] = {
            "value": result.get("quality", 0),
            "found": True,
            "ts": ts,
            "raw": result,                      # Full debug data
            "score": result.get("score", 0),    # For weighting
            "desc": result.get("desc", f"Pattern {_split_camel(alias)} Detected"),
            "alias": alias,
            "source": "Pattern"
        }

        # ✅ Mirror neckline patterns to the aggregate 'doubleTopBottom' key for Strategy/UI parity
        if alias in ["bullishNecklinePattern", "bearishNecklinePattern"]:
            indicators["doubleTopBottom"] = indicators[alias].copy()
            indicators["doubleTopBottom"]["alias"] = "doubleTopBottom"