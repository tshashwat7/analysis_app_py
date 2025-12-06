# services/tradeplan/trade_enhancer.py
import copy
from typing import Dict, Any, Optional, Tuple
import math

def _safe_float(x) -> Optional[float]:
    """Coerce numeric-like values (including numpy types) to python float."""
    try:
        if x is None or isinstance(x, bool):
            return None
        f = float(x)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        return None

def enhance_plan_with_patterns(
    plan: Dict[str, Any],
    indicators: Dict[str, Any],
    logger: Optional[Any] = None
) -> Dict[str, Any]:
    """
    Refines the generic Trade Plan using specific Pattern Geometry.
    - Adjusts Targets (T1, T2) based on Pattern Height/Depth.
    - Adjusts Stop Loss based on Pattern Structure.
    - Boosts Confidence if high-quality patterns are present.
    """
    
    # 1. Configuration: Keys must match 'self.alias' in pattern files
    pattern_keys = (
        "darvas_box", 
        "cup_handle", 
        "bollinger_squeeze", 
        "flag_pennant", 
        "minervini_stage2", 
        "three_line_strike",
        "ichimoku_signals",
        "golden_cross",
        "double_top_bottom"
    )

    def _log(msg: str):
        if logger:
            try: logger.debug(msg)
            except: pass

    # 2. Defensive Copy & Prep
    # out = dict(plan) if plan is not None else {}
    out = copy.deepcopy(plan) if plan is not None else {}
    
    # Validate targets and execution_hints shapes before setting
    out.setdefault("targets", {})
    if not isinstance(out["targets"], dict):
        _log("coercing 'targets' to dict")
        out["targets"] = {}

    out.setdefault("execution_hints", {})
    if not isinstance(out["execution_hints"], dict):
        _log("coercing 'execution_hints' to dict")
        out["execution_hints"] = {}

    out.setdefault("analytics", {})
    if not isinstance(out["analytics"], dict):
        _log("coercing 'analytics' to dict")
        out["analytics"] = {}


    # Extract Base Metrics
    entry = _safe_float(out.get("entry"))
    current_sl = _safe_float(out.get("stop_loss"))
    
    # Detect Trade Direction (Important for Stop Loss validation)
    # If SL > Entry, it's a SHORT trade. If SL < Entry, it's LONG.
    is_short_trade = False
    if entry and current_sl and current_sl > entry:
        is_short_trade = True

    # 3. Find Best Pattern
    valid_patterns = []
    for k in pattern_keys:
        p = indicators.get(k)
        if not p or not isinstance(p, dict): continue
        
        found = p.get("found", False)
        score = _safe_float(p.get("score"))
        
        # Threshold: Pattern must be found and have score > 60
        if found and score is not None and score > 60:
            valid_patterns.append((k, p))

    if not valid_patterns:
        return out

    # Sort by score descending (Highest confidence wins)
    valid_patterns.sort(key=lambda x: _safe_float(x[1].get("score")) or 0.0, reverse=True)
    best_name, best_pat = valid_patterns[0]
    meta = best_pat.get("meta", {}) or {}
    if not isinstance(meta, dict):
        _log(f"pattern {best_name} has non-dict meta: {type(meta)}; coercing to empty dict")
        meta = {}

    # Helper to extract meta values safely
    def meta_num(key): return _safe_float(meta.get(key))

    # 4. Pattern-Specific Logic
    
    # --- A. DARVAS BOX ---
    if best_name == "darvas_box":
        box_top = meta_num("box_high")
        box_low = meta_num("box_low")
        
        if box_top and box_low and box_top > box_low:
            height = box_top - box_low
            # Darvas is inherently Bullish breakout
            out["targets"]["t1"] = round(box_top + height, 2)
            out["targets"]["t2"] = round(box_top + (height * 2), 2)
            out["stop_loss"] = round(box_low * 0.995, 2)
            out["execution_hints"]["pattern_note"] = f"Darvas: Target via Box Height ({height:.1f})"

    # --- B. CUP & HANDLE ---
    elif best_name == "cup_handle":
        rim = meta_num("rim_level")
        depth_pct = meta_num("depth_pct")
        
        if rim and depth_pct:
            depth_val = rim * (depth_pct / 100.0)
            # Bullish Pattern
            out["targets"]["t1"] = round(rim + (depth_val * 0.618), 2)
            out["targets"]["t2"] = round(rim + depth_val, 2)
            
            # Smart Stop: Handle Low is ideal, but let's check ATR
            # We don't overwrite SL blindly here, usually ATR is safer for Cup
            out["execution_hints"]["pattern_note"] = f"Cup: Measured Move (+{depth_pct:.1f}%)"

    # --- C. BULL FLAG ---
    elif best_name == "flag_pennant":
        pole_pct = meta_num("pole_gain_pct")
        if pole_pct and entry:
            move = entry * (pole_pct / 100.0)
            out["targets"]["t1"] = round(entry + (move * 0.5), 2)
            out["targets"]["t2"] = round(entry + move, 2)
            out["execution_hints"]["pattern_note"] = "Flag: Target via Pole Height"

    # --- D. MINERVINI VCP ---
    elif best_name == "minervini_stage2":
        # Boost confidence significantly
        conf = _safe_float(out.get("setup_confidence")) or 0
        out["setup_confidence"] = min(100, int(conf + 15))
        out["execution_hints"]["entry_mode"] = "Aggressive (VCP Confirmed)"

    # --- E. BOLLINGER SQUEEZE ---
    elif best_name == "bollinger_squeeze":
        # Boost confidence
        conf = _safe_float(out.get("setup_confidence")) or 0
        out["setup_confidence"] = min(100, int(conf + 10))
        out["execution_hints"]["pattern_note"] = "Volatility Squeeze: Expect expansion"

    # --- F. THREE LINE STRIKE ---
    elif best_name == "three_line_strike":
        ptype = str(meta.get("type", "")).lower()
        if entry:
            if "bullish" in ptype:
                out["targets"]["t1"] = round(entry * 1.03, 2)
                out["targets"]["t2"] = round(entry * 1.06, 2)
            elif "bearish" in ptype:
                out["targets"]["t1"] = round(entry * 0.97, 2)
                out["targets"]["t2"] = round(entry * 0.94, 2)
            out["execution_hints"]["pattern_note"] = f"3-Line Strike ({ptype})"
            
    # --- G. DOUBLE TOP / BOTTOM ---
    elif best_name == "double_top_bottom":
        target = meta_num("target")
        neckline = meta_num("neckline")
        ptype = str(meta.get("type", "")).lower()
        
        if target and neckline:
            out["targets"]["t1"] = round(target, 2)
            # T2 is usually 1.5x the height
            height = abs(neckline - target)
            if "bear" in ptype:
                out["targets"]["t2"] = round(target - (height * 0.5), 2)
            else:
                out["targets"]["t2"] = round(target + (height * 0.5), 2)
            
            out["execution_hints"]["pattern_note"] = "Double Top/Bottom: Measured Move Target"

    # --- H. GOLDEN CROSS ---
    elif best_name == "golden_cross":
        conf = _safe_float(out.get("setup_confidence")) or 0
        ptype = str(meta.get("type", "")).lower()
        
        if "bull" in ptype:
            out["setup_confidence"] = min(100, int(conf + 20))
            out["execution_hints"]["pattern_note"] = "Golden Cross: Major Trend Confirmation"
        else:
            out["setup_confidence"] = min(100, int(conf + 20))
            out["execution_hints"]["pattern_note"] = "Death Cross: Major Downtrend Confirmation"

    # 5. Final Safety Guard
    # Ensure the new pattern-based Stop Loss is on the correct side of Entry
    
    new_sl = _safe_float(out.get("stop_loss"))
    
    if new_sl is not None and entry is not None and current_sl is not None:
        if is_short_trade:
            # SHORT: SL must be > Entry
            if new_sl <= entry:
                _log(f"Reverting Short SL: Pattern SL {new_sl} <= Entry {entry}")
                out["stop_loss"] = current_sl
        else:
            # LONG: SL must be < Entry
            if new_sl >= entry:
                _log(f"Reverting Long SL: Pattern SL {new_sl} >= Entry {entry}")
                out["stop_loss"] = current_sl

    # 6. Analytics Tagging
    out["analytics"]["pattern_driver"] = best_name
    out["analytics"]["pattern_score"] = _safe_float(best_pat.get("score"))

    return out