import pandas as pd
from typing import Dict, Any

def merge_pattern_into_indicators(indicators: Dict[str, Any], pattern_results: Dict[str, Any], horizon: str = None, df: pd.DataFrame = None):
    """
    Injects pattern results into the main indicators dictionary.
    UI Standard Format: { "value": X, "raw": Y, "score": Z, "desc": "..." }
    
    NOTE: This function mutates the 'indicators' dictionary in-place.
    """
    # FIX: Loop unpacks only (alias, result), not (name, alias, result)
    for alias, result in pattern_results.items():
        if not result.get("found", False):
            continue
            
        # Create a unique key if horizon is provided
        unique_key = f"{alias}_{horizon}" if horizon else alias
        
        # ✅ P3-1: Quality-check to prevent overwriting better patterns
        existing = indicators.get(unique_key)
        if existing and existing.get("value", 0) > result.get("quality", 0):
            continue

        indicators[unique_key] = {
            "value": result.get("quality", 0),
            "found": True,
            "ts": float(df.index[-1].timestamp()) if df is not None and not df.empty else None,
            "raw": result,                      # Full debug data
            "score": result.get("score", 0),    # For weighting
            "desc": result.get("desc", f"Pattern {alias.replace('_', ' ').title()} Detected"),
            "alias": alias,
            "source": "Pattern"
        }