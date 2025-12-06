from typing import Dict, Any

def merge_pattern_into_indicators(indicators: Dict[str, Any], pattern_results: Dict[str, Any]):
    """
    Injects pattern results into the main indicators dictionary.
    UI Standard Format: { "value": X, "raw": Y, "score": Z, "desc": "..." }
    """
    for name, result in pattern_results.items():
        if not result.get("found", False):
            continue
            
        # Map pattern output to UI-friendly structure
        indicators[name] = {
            "value": result.get("quality", 0),  # Display value (0-10)
            "raw": result,                      # Full debug data
            "score": result.get("score", 0),    # For weighting
            "desc": f"Pattern {name.replace('_', ' ').title()} Detected",
            "alias": name,
            "source":"Pattern"
        }