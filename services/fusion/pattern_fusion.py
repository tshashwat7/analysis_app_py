import pandas as pd
from typing import Dict, Any

def merge_pattern_into_indicators(indicators: Dict[str, Any], pattern_results: Dict[str, Any], horizon: str = None, df: pd.DataFrame = None):
    """
    Injects pattern results into the main indicators dictionary.
    UI Standard Format: { "value": X, "raw": Y, "score": Z, "desc": "..." }
    
    NOTE: This function mutates the 'indicators' dictionary in-place.
    """
    # Fix 6.4-1: CamelCase Fallback Labeling (hoisted out of loop)
    def _split_camel(s):
        import re
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', s).title()

    # Fix 6.3-1: Timestamp Safety (hoisted — same for all patterns in this call)
    try:
        ts = float(df.index[-1].timestamp()) if df is not None and not df.empty else None
    except (AttributeError, OSError, ValueError):
        ts = None

    for alias, result in pattern_results.items():
        if not result.get("found", False):
            # ✅ v15.7.5 FIX: Always inject a stub so _extract_patterns sees the key.
            # Without this, the key is entirely absent from indicators, and the
            # config_resolver logs "pattern_detection_failed" with no way to tell
            # whether the detector ran (and found nothing) or never ran at all.
            if alias not in indicators:
                indicators[alias] = {
                    "value": 0,
                    "found": False,
                    "raw": result,
                    "score": 0,
                    "desc": f"Pattern {_split_camel(alias)} Not Detected",
                    "alias": alias,
                    "source": "Pattern"
                }
            continue
            
        # P3-1: Quality-check to prevent overwriting better patterns
        existing = indicators.get(alias)
        if existing and existing.get("value", 0) > result.get("quality", 0):
            continue

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