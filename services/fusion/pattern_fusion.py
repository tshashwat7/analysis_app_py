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
        key = f"{alias}_{horizon}" if horizon else alias
        
        # Map pattern output to UI-friendly structure
        indicators[alias] = {
            "value": result.get("quality", 0),  # Display value (0-10)
            "found":result.get("found", False),
            "ts": float(df.index[-1].timestamp()) if df is not None and not df.empty else None,
            "raw": result,                      # Full debug data
            "score": result.get("score", 0),    # For weighting
            "desc": result.get("desc", f"Pattern {alias.replace('_', ' ').title()} Detected"),
            "alias": key,
            "source": "Pattern"
        }