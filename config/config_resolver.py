# config/config_resolver.py
"""
Fixed ConfigResolver with Correct Path Resolution
Now properly handles horizon-first lookup with global fallback.
"""

from typing import Any, Dict, Optional, List, Tuple, Union
from copy import deepcopy
import logging
import ast
import operator
import re

from services.data_fetch import _get_val, _safe_get_raw_float
logger = logging.getLogger(__name__)

from config.master_config import MASTER_CONFIG

class ConfigResolver:
    """Smart configuration resolver with horizon-aware inheritance."""
    
    def __init__(self, horizon: str = "short_term", master_config: Dict = None, indicators: Dict = None, fundamentals: Dict = None):
        self.horizon = horizon
        self.indicators = indicators or {}
        self.fundamentals = fundamentals or {}
        self.config = master_config or MASTER_CONFIG
        self._cache = {}
        
        valid_horizons = ["intraday", "short_term", "long_term", "multibagger"]
        if horizon not in valid_horizons:
            logger.warning(f"Invalid horizon '{horizon}', falling back to 'short_term'")
            self.horizon = "short_term"
        
        self._horizon_config = self.config.get("horizons", {}).get(self.horizon, {})
        self._global_config = self.config.get("global", {})
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value with smart horizon-first inheritance.
        
        Resolution Order:
        1. horizons.{horizon}.{path}
        2. global.{path}
        3. default
        """
        cache_key = f"{self.horizon}:{path}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # 1. Try horizon-specific first
        value = self._get_nested(self._horizon_config, path)
        
        # 2. Fallback to global
        if value is None:
            value = self._get_nested(self._global_config, path)
        
        # 3. Use default
        if value is None:
            value = default
        
        self._cache[cache_key] = value
        return value
    
    def get_section(self, section: str, merge_global: bool = True) -> Dict:
        """
        Get entire config section with optional global merge.
        Auto-unwraps standard containers like 'rules', 'penalties', etc.
        """
        if merge_global:
            # Start with global, then overlay horizon-specific
            result = deepcopy(self._global_config.get(section, {}))
            horizon_section = self._horizon_config.get(section, {})
            result.update(horizon_section)
        else:
            result = self._horizon_config.get(section, {})
        
        # Auto-unwrap standard containers
        wrappers = ['rules', 'penalties', 'enhancements', 'gates', 'bonuses', 'conditions']
        for key in wrappers:
            if isinstance(result, dict) and key in result and isinstance(result[key], dict):
                logger.debug(f"[{self.horizon}] get_section({section}): auto-unwrapped '{key}'")
                return result[key]
        
        return result

    def get_setup_metadata(self, setup_name: str) -> Dict[str, Any]:
        """
        Fetches setup-specific parameters (like RSI ranges or volume requirements).
        Standardizes lookups to ensure 'QUALITY_ACCUMULATION' matches 'accumulation'.
        """
        all_metadata = self.get("calculation_engine.setup_classification", {})
        
        # Try exact match, then try lowercase (to handle legacy keys like 'accumulation')
        metadata = all_metadata.get(setup_name) or all_metadata.get(setup_name.lower(), {})
        
        return metadata

    
    def get_many(self, paths: List[str], defaults: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get multiple config values at once.
        Performance helper for batch retrieval.
        """
        defaults = defaults or {}
        return {
            path: self.get(path, default=defaults.get(path))
            for path in paths
        }
    
    # ============================================================
    # FIXED: Calculation Engine Accessors
    # ============================================================
    
    def get_setup_rules(self) -> Dict[str, Any]:
        """Get setup rules with horizon-specific priority overrides applied."""
        
        # Fetch base rules
        rules = deepcopy(self.get("calculation_engine.setup_classification.rules", {}))
        
        # ✅ Ensure all rules have a priority (default 50)
        for setup_name, rule in rules.items():
            if "priority" not in rule:
                rule["priority"] = 50  # Default priority
        
        # Apply horizon-specific priority overrides
        overrides = self.get(
            f"calculation_engine.horizon_priority_overrides.{self.horizon}", 
            {}
        )
        
        if overrides:
            applied_count = 0
            
            for setup, priority in overrides.items():
                if setup in rules:
                    rules[setup]["priority"] = priority
                    applied_count += 1
                else:
                    logger.warning(
                        f"[{self.horizon}] Priority override for '{setup}' ignored "
                        f"(setup not defined in rules)"
                    )
            
            logger.debug(
                f"[{self.horizon}] Applied {applied_count}/{len(overrides)} "
                f"priority overrides"
            )
        
        return rules
    
    def get_pattern_priority(self) -> List[Tuple[str, str, int]]:
        """
        Get pattern priority list from calculation engine.
        Path: global.calculation_engine.setup_classification.pattern_priority
        """
        priority = self.get("calculation_engine.setup_classification.pattern_priority", [])
        return [(p["pattern"], p["setup_name"], p["min_score"]) for p in priority]
    
    def evaluate_setup_condition(self, setup_type: str, indicators: Dict, fundamentals: Dict = None) -> bool:
        """
        Evaluates setup conditions from calculation engine rules.
        
        Args:
            setup_type: Setup name (e.g., "MOMENTUM_BREAKOUT")
            indicators: Technical indicators
            fundamentals: Fundamental data (optional)
        
        Returns:
            True if ALL conditions pass
        """
        rules = self.get_setup_rules()
        setup_rule = rules.get(setup_type)
        
        if not setup_rule:
            logger.warning(f"[{self.horizon}] No rule found for setup: {setup_type}")
            return False
        
        conditions = setup_rule.get("conditions", [])
        
        # handle empty conditions correctly
        if not conditions:
            # Empty conditions = always match (e.g., GENERIC setup)
            logger.debug(f"[{self.horizon}] Setup {setup_type} has no conditions (auto-pass)")
            return True  # ✅ Changed from False to True
        
        # ✅ OPTIMIZATION: Build context once, reuse for all conditions
        eval_context = self._build_eval_context(indicators, fundamentals)
        
        # Evaluate all conditions (ALL must pass)
        for cond in conditions:
            if not self._evaluate_condition_string(
                cond, 
                indicators, 
                fundamentals,
                context=eval_context  # ✅ Pass pre-built context
            ):
                logger.debug(f"[{self.horizon}] Setup {setup_type} failed: '{cond}'")
                return False
        
        logger.debug(f"[{self.horizon}] Setup {setup_type} passed all {len(conditions)} conditions")
        return True

    def get_blocked_setups(self) -> List[str]:
        """Get list of setups blocked by strategy preferences."""
        path = f"strategy_preferences.horizon_strategy_config.{self.horizon}.blocked_setups"
        return self.get(path, [])    
    
    # ============================================================
    # FIXED: Dynamic Confidence Floor
    # ============================================================
    
    def calculate_dynamic_confidence_floor(self, adx: float, setup_type: str) -> float:
        """
        Horizon-aware dynamic confidence floor.
        Uses horizon-specific base_floors, then applies ADX adjustment.
        """
        # 1. Get horizon-specific base floor (with global fallback)
        base_floors = self.get("confidence.base_floors", {})
        base = base_floors.get(setup_type, 55)
        
        # 2. Get ADX normalization params (global only)
        norm_cfg = self.get("confidence.adx_normalization", {})
        adx_min = norm_cfg.get("min", 10)
        adx_max = norm_cfg.get("max", 40)
        adj_factor = norm_cfg.get("adjustment_factor", 12)
        
        # 3. Normalize ADX
        adx_val = float(adx or 0)
        adx_norm = max(0.0, min(1.0, (adx_val - adx_min) / (adx_max - adx_min)))
        
        # 4. Apply adjustment
        adjustment = adx_norm * adj_factor
        
        # 5. Apply horizon discount (horizon-specific)
        horizon_discount = self.get("confidence.horizon_discount", 0)
        
        final_floor = base - adjustment - horizon_discount
        return max(35.0, min(75.0, final_floor))
    # ============================================================
    # FIXED: Composite Score Calculation
    # ============================================================
    
    def compute_composite_score(self, composite_name: str, indicators: Dict) -> Dict[str, Any]:
        """
        Calculates composite scores using calculation engine weights.
        Path: global.calculation_engine.composite_weights.{composite_name}
        """
        weights_cfg = self.get(f"calculation_engine.composite_weights.{composite_name}", {})
        
        if composite_name == "trendstrength":
            return self._compute_trend_strength(indicators, weights_cfg)
        elif composite_name == "momentumstrength":
            return self._compute_momentum_strength(indicators, weights_cfg)
        elif composite_name == "volatilityquality":
            return self._compute_volatility_quality(indicators, weights_cfg)
        
        return {"raw": 0, "value": 0, "score": 0, "desc": "Unknown Composite"}
    
    def _compute_trend_strength(self, indicators: Dict, cfg: Dict) -> Dict:
        """Trend Strength composite calculation."""
        adx = self._get_val(indicators, "adx")
        slope = abs(self._get_val(indicators, "ma_fast_slope"))
        di_spread = self._get_val(indicators, "di_plus") - self._get_val(indicators, "di_minus")
        st_signal = str(self._get_nested(indicators, "supertrend_value.value") or "")
        
        # Score components
        adx_t = cfg.get("adx", {}).get("thresholds", {})
        adx_s = 10 if adx >= adx_t.get("strong", 25) else 7 if adx >= adx_t.get("moderate", 20) else 2
        
        slope_t = cfg.get("ema_slope", {}).get("thresholds", {})
        slope_s = 10 if slope >= slope_t.get("strong", 20) else 7 if slope >= slope_t.get("moderate", 5) else 2
        
        di_s = 10 if di_spread >= 15 else 5
        st_s = 10 if "Bull" in st_signal else 0
        
        # Apply weights
        wa = cfg.get("adx", {}).get("weight", 0.4)
        wb = cfg.get("ema_slope", {}).get("weight", 0.3)
        wc = cfg.get("di_spread", {}).get("weight", 0.2)
        wd = cfg.get("supertrend", {}).get("weight", 0.1) if st_signal else 0
        
        raw = (adx_s * wa) + (slope_s * wb) + (di_s * wc) + (st_s * wd)
        score = min(10, round(raw, 2))
        
        return {"raw": raw, "value": score, "score": int(score), "desc": "Trend Strength"}
    
    def _compute_momentum_strength(self, indicators: Dict, cfg: Dict) -> Dict:
        """Momentum Strength composite calculation."""
        rsi = self._get_val(indicators, "rsi")
        slope = self._get_val(indicators, "rsislope")
        macd = self._get_val(indicators, "macdhistogram")
        
        rsi_t = cfg.get("rsi_value", {}).get("thresholds", {})
        rsi_s = 8 if rsi >= rsi_t.get("strong", 60) else 5 if rsi >= rsi_t.get("neutral", 50) else 2
        
        slope_t = cfg.get("rsislope", {}).get("thresholds", {})
        slope_s = 8 if slope >= slope_t.get("strong", 1.0) else 4
        
        macd_t = cfg.get("macd_hist", {}).get("thresholds", {})
        macd_s = 8 if macd >= macd_t.get("strong", 0.5) else 5 if macd > 0 else 2
        
        w_rsi = cfg.get("rsi_value", {}).get("weight", 0.25)
        w_slope = cfg.get("rsislope", {}).get("weight", 0.25)
        w_macd = cfg.get("macd_hist", {}).get("weight", 0.30)
        
        raw = (rsi_s * w_rsi) + (slope_s * w_slope) + (macd_s * w_macd)
        score = min(10, round(raw, 2))
        
        return {"raw": raw, "value": score, "score": int(score), "desc": "Momentum Strength"}
    
    def _compute_volatility_quality(self, indicators: Dict, cfg: Dict) -> Dict:
        """Volatility Quality composite calculation."""
        atr_pct = self._get_val(indicators, "atr_pct")
        bbwidth = self._get_val(indicators, "bbwidth")
        
        atr_t = cfg.get("atr_pct", {}).get("thresholds", {})
        atr_s = 10 if atr_pct <= atr_t.get("low", 1.5) else 6 if atr_pct <= atr_t.get("high", 5.0) else 2
        
        bb_t = cfg.get("bbwidth", {}).get("thresholds", {})
        bb_s = 10 if bbwidth <= bb_t.get("tight", 0.01) else 6
        
        w_atr = cfg.get("atr_pct", {}).get("weight", 0.3)
        w_bb = cfg.get("bbwidth", {}).get("weight", 0.25)
        
        raw = (atr_s * w_atr) + (bb_s * w_bb) + 3.0
        score = min(10, round(raw, 2))
        
        return {"raw": raw, "value": score, "score": int(score), "desc": "Vol Quality"}
    
    def _normalize_volatility_bands(self) -> Dict[str, float]:
        """
        FIXED: Converts volatility_bands to dict format regardless of source.
        Handles both array [min, max] and dict {min, ideal, max} formats.
        """
        bands = self.get("gates.volatility_bands_atr_pct", {"min": 1.0, "ideal": 2.5, "max": 12.0})
        
        # If it's an array, convert to dict
        if isinstance(bands, list) and len(bands) >= 2:
            return {
                "min": bands[0],
                "ideal": (bands[0] + bands[1]) / 2,  # Calculate midpoint
                "max": bands[1]
            }
        
        # If it's already a dict, ensure all keys exist
        if isinstance(bands, dict):
            return {
                "min": bands.get("min", 1.0),
                "ideal": bands.get("ideal", (bands.get("min", 1.0) + bands.get("max", 12.0)) / 2),
                "max": bands.get("max", 12.0)
            }
        
        # Fallback
        return {"min": 1.0, "ideal": 2.5, "max": 12.0}
    
    # ============================================================
    # FIXED: Signal Context Builder
    # ============================================================
    
    def get_signal_context(self) -> Dict[str, Any]:
        """
        Consolidated context for Signal Engine.
        FIXED: All paths now correctly resolve horizon-first with global fallback.
        """
        return {
            "horizon": self.horizon,
            "horizon_info": self.get_horizon_info(),
            
            # Primary Sections (horizon-specific)
            "execution": self.get_execution_params(),
            "risk": self.get_risk_params(),
            "gates": self.get_gate_checks(),
            
            # Scoring (horizon-specific with global penalties)
            "scoring": self.get_section("scoring", merge_global=False),
            
            # Setup Classification (from calculation engine)
            "setup_classification": {
                "rules": self.get_setup_rules(),
                "pattern_priority": self.get_pattern_priority()
            },
            
            # Setup Confidence (horizon-specific with global fallback)
            "setup_confidence": self.get_section("setup_confidence", merge_global=False),
            
            # Enhancements (horizon-specific)
            "enhancements": self.get_section("enhancements", merge_global=False),
            
            # Indicators (horizon-specific with global fallback)
            "indicators": self.get_all_indicators(),
            "moving_averages": self.get_section("moving_averages", merge_global=True),
            
            # Volatility (horizon-specific)
            "volatility": {
                "bands": self._normalize_volatility_bands(),
                "scoring_thresholds": self.get("volatility.scoring_thresholds", {}),
                "quality_mins": self.get("volatility.quality_mins", {})
            },
            
            # Patterns (global calculation engine rules)
            "patterns": {
                "entry_rules": self._get_all_pattern_entry_rules(),
                "invalidation_rules": self._get_all_pattern_invalidation_rules()
            },
            
            # Composite Weights (global calculation engine)
            "composite_weights": {
                "trendstrength": self.get("calculation_engine.composite_weights.trendstrength", {}),
                "momentumstrength": self.get("calculation_engine.composite_weights.momentumstrength", {}),
                "volatilityquality": self.get("calculation_engine.composite_weights.volatilityquality", {})
            },
            
            # Thresholds (horizon-specific with global fallback)
            "thresholds": {
                "volume": {
                    "surge": self.get("volume_analysis.rvol_surge_threshold", 2.5),
                    "drought": self.get("volume_analysis.rvol_drought_threshold", 0.7)
                },
                "momentum": self.get_section("momentum_thresholds", merge_global=True),
                "trend": self.get_section("trend_thresholds", merge_global=True)
            },
            
            # Confidence (horizon-specific)
            "confidence": {
                "floors": self.get("confidence.floors", {}),
                "base_floors": self.get("confidence.base_floors", {}),
                "adx_based_floors": self.get("confidence.adx_based_floors", {}),
                "horizon_discount": self.get("confidence.horizon_discount", 0),
                "volume_penalty": self.get("confidence.volume_penalty", {}),
                "setup_type_overrides": self.get("confidence.setup_type_overrides", {})
            },
            
            # Time Estimation (horizon-specific)
            "time_estimation": self.get_time_estimation_params()
        }
    
    # ============================================================
    # Helper Methods
    # ============================================================
    
    def get_all_indicators(self) -> Dict[str, Any]:
        """Get all indicator configs (horizon-specific with global fallback)."""
        return self.get_section("indicators", merge_global=True)

    

    
    def get_time_estimation_params(self) -> Dict[str, Any]:
        """Get time estimation parameters (horizon-specific)."""
        return {
            "candles_per_unit": self.get("time_estimation.candles_per_unit", 1),
            "base_friction": self.get("time_estimation.base_friction", 0.8),
            "velocity_factors": self.get("time_estimation.velocity_factors", {})
        }
    
    def get_horizon_info(self) -> Dict[str, str]:
        """Get metadata about current horizon."""
        return {
            "horizon": self.horizon,
            "timeframe": self.get("timeframe", "1d"),
            "description": self.get("description", ""),
            "ma_type": self.get("moving_averages.type", "EMA")
        }
    
    def get_rr_multipliers(self, adx: float) -> Dict[str, float]:
        """Get R:R multipliers based on ADX regime (horizon-specific)."""
        adjustments = self.get("risk_management.rr_regime_adjustments", {})
        
        strong = adjustments.get("strong_trend", {})
        normal = adjustments.get("normal_trend", {})
        weak = adjustments.get("weak_trend", {})
        
        if adx >= strong.get("adx_min", 35):
            return {"t1_mult": strong.get("t1_mult", 2.0), "t2_mult": strong.get("t2_mult", 4.0)}
        elif adx >= normal.get("adx_min", 20):
            return {"t1_mult": normal.get("t1_mult", 1.5), "t2_mult": normal.get("t2_mult", 3.0)}
        else:
            return {"t1_mult": weak.get("t1_mult", 1.2), "t2_mult": weak.get("t2_mult", 2.5)}
    
    def _get_all_pattern_entry_rules(self) -> Dict[str, Dict]:
        """Get all pattern entry rules from global calculation engine."""
        patterns = [
            "bollinger_squeeze", "darvas_box", "cup_handle",
            "minervini_stage2", "flag_pennant", "three_line_strike",
            "ichimoku_signals", "golden_cross", "double_top_bottom"
        ]
        
        rules = {}
        for pattern in patterns:
            path = f"pattern_entry_rules.{pattern}.horizons.{self.horizon}"
            rule = self.get(path)
            if rule:
                rules[pattern] = rule
        return rules
    
    def _get_all_pattern_invalidation_rules(self) -> Dict[str, Dict]:
        """Get all pattern invalidation rules from global calculation engine."""
        patterns = [
            "bollinger_squeeze", "darvas_box", "cup_handle",
            "minervini_stage2", "flag_pennant", "three_line_strike",
            "ichimoku_signals", "golden_cross", "double_top_bottom"
        ]
        
        rules = {}
        for pattern in patterns:
            path = f"pattern_invalidation.{pattern}"
            rule = self.get(path)
            if rule:
                breakdown = rule.get("breakdown_threshold", {}).get(self.horizon, {})
                action = rule.get("action", {}).get(self.horizon, "EXIT_ON_CLOSE")
                if breakdown or action:
                    rules[pattern] = {
                        "condition": breakdown.get("condition"),
                        "duration_candles": breakdown.get("duration_candles"),
                        "action": action,
                        "or_condition": breakdown.get("or_condition")
                    }
        return rules
    
    def _get_val(self, data: Dict, key: str, default: float = 0.0) -> float:
        if not data or key not in data:
            logger.debug(f"[{self.horizon}] _get_val: missing key '{key}'")
            return default
        val = _safe_get_raw_float(data.get(key))
        if val is None:
            logger.debug(f"[{self.horizon}] _get_val: missing key '{key}'")
            return default
        return val
    
    def _get_nested(self, data: Dict, path: str) -> Any:
        """Get value from nested dict using dot-separated path."""
        keys = path.split('.')
        current = data
        
        for key in keys:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        
        return current
    
    def _build_eval_context(self, indicators: Dict, fundamentals: Dict = None) -> Dict:
        """
        Context builder with blacklist approach + _get_val reuse.
        """
        ctx = {}
        
        # ====================================================================
        # STAGE 1: Flatten Raw Indicators
        # ====================================================================
        for k in indicators.keys():
            ctx[k] = self._get_val(indicators, k, 0)
        
        # ====================================================================
        # STAGE 2: Pattern Counting
        # ====================================================================
        pattern_keys = [
            "bollinger_squeeze", "cup_handle", "darvas_box", "flag_pennant", 
            "minervini_stage2", "golden_cross", "double_top_bottom", 
            "three_line_strike", "ichimoku_signals"
        ]
        
        pattern_count = sum(
            1 for k in pattern_keys 
            if isinstance(indicators.get(k), dict) and indicators[k].get("found")
        )
        ctx["pattern_count"] = pattern_count

        for k in pattern_keys:
            pattern_data = indicators.get(k, {})
            if isinstance(pattern_data, dict):
                # Remove underscores from key names
                clean_name = k.replace('_', '')
                
                # Add flattened keys
                ctx[f'{clean_name}found'] = pattern_data.get('found', False)
                ctx[f'{clean_name}quality'] = pattern_data.get('quality', 0)
                ctx[f'{clean_name}score'] = pattern_data.get('score', 0)
                
                # Flatten meta
                if 'meta' in pattern_data:
                    for meta_key, meta_val in pattern_data['meta'].items():
                        ctx[f'{clean_name}{meta_key}'] = meta_val
        
        # ====================================================================
        # STAGE 3: Semantic Booleans + Safety Guards
        # ====================================================================
        ctx["price"] = ctx.get("price") or ctx.get("close") or 0.0
        price = ctx["price"]
        
        # Division-safe MA fallbacks (ONLY generic keys)
        ctx["ma_fast"] = ctx.get("ma_fast") or price
        ctx["ma_mid"] = ctx.get("ma_mid") or price
        ctx["ma_slow"] = ctx.get("ma_slow") or price
        
        # Consolidation detection
        bbwidth = ctx.get("bbwidth", 999)
        vol_qual = ctx.get("volatilityquality", 0)
        
        ctx["bb_width_pct"] = (bbwidth / price * 100) if price > 0 else 999
        
        consolidation_threshold = self.get("gates.consolidation_bb_width_threshold", 0.06)
        ctx["is_consolidating"] = bbwidth < consolidation_threshold and vol_qual > 4.0
        
        # Squeeze detection
        ttm_squeeze = ctx.get("ttm_squeeze", "")
        ctx["is_squeeze"] = "on" in str(ttm_squeeze).lower() or bool(ttm_squeeze)
        
        # Trend direction (generic key)
        trend_sig = ctx.get("ma_trend_signal") or 0
        ctx["trend_dir"] = "up" if trend_sig > 0 else "down" if trend_sig < 0 else "neutral"
        
        # Market conditions
        adx = ctx.get("adx", 0)
        rsi = ctx.get("rsi", 50)
        atr_pct = ctx.get("atr_pct", 0)
        
        ctx["is_trending"] = adx > 20
        ctx["is_volatile"] = atr_pct > 5.0
        ctx["is_oversold"] = rsi < 30
        ctx["is_overbought"] = rsi > 70
        
        # Supertrend
        st_val = ctx.get("supertrendsignal")
        if isinstance(st_val, dict):
            st_val = st_val.get("value", "")
        ctx["is_bullish_st"] = "bull" in str(st_val).lower() if st_val else False
        
        # ====================================================================
        # STAGE 4: Flatten Fundamentals (Blacklist + _get_val)
        # ====================================================================
        if fundamentals:
            # ✅ Blacklist: metadata and duplicates
            excluded_keys = {'symbol', 'name', 'website', '_meta','current_price', 'base_score', 'market_penalty', 'final_score','days_to_earnings', 'analyst_rating', 'roe_5y', 'roe_history'}
            # Log excluded keys (debug)
            excluded_found = [k for k in fundamentals.keys() if k in excluded_keys]
            if excluded_found:
                logger.debug(f"[{self.horizon}] Excluding: {excluded_found}")
            
            # ✅ Flatten ALL fundamentals (numeric conversion via _get_val)
            flattened_count = 0
            for key in fundamentals.keys():
                if key not in excluded_keys:
                    ctx[key] = _get_val(fundamentals, key)
                    flattened_count += 1            
            logger.info(
                f"[{self.horizon}] Flattened {flattened_count} fundamentals "
                f"({len(excluded_keys)} excluded)"
            )
        
        # ====================================================================
        # STAGE 5: Strategy Metrics from Indicators
        # ====================================================================
        # ✅ Relative Strength is in indicators (not fundamentals)
        ctx['rel_strength_nifty'] = _get_val(indicators, 'rel_strength_nifty')
        
        # ====================================================================
        # STAGE 6: Support/Resistance Levels
        # ====================================================================
        ctx["resistance_1"] = indicators.get("resistance_1", price * 1.05)
        ctx["resistance_2"] = indicators.get("resistance_2", price * 1.10)
        ctx["support_1"] = indicators.get("support_1", price * 0.95)
        ctx["pivot_point"] = indicators.get("pivot_point", price)
        
        # ====================================================================
        # STAGE 7: Metadata
        # ====================================================================
        ctx["setup_type"] = "UNKNOWN"
        ctx["horizon"] = self.horizon
        
        logger.debug(
            f"[{self.horizon}] Context: {len(ctx)} keys "
            f"(patterns={pattern_count}, price={price:.2f}, ma_fast={ctx['ma_fast']:.2f})"
        )
        logger.debug(f"{self.horizon} Full context keys: {sorted(ctx.keys())}")

        return ctx

    def _evaluate_condition_string(
    self, 
    condition: str, 
    indicators: Dict, 
    fundamentals: Dict = None, 
    context: Dict = None
    ) -> bool:
        """
        Safe condition evaluator with robust whitespace and string handling.
        
        ARCHITECTURE:
        - Stage 0: Extract quoted strings
        - Stage 1: Build evaluation context (if not provided)
        - Stage 2: Fast path for simple comparisons (e.g., "rsi >= 60")
        - Stage 3: Complex path using AST parser (e.g., "(price - ma) / ma <= 0.05")
        
        Args:
            condition: Condition string to evaluate
            indicators: Technical indicators
            fundamentals: Fundamental data (optional)
            context: Pre-built context (optional, for performance)
        
        Returns:
            True if condition evaluates to truthy value
        """
        try:
            # ================================================================
            # STAGE 0: Pre-process quoted strings
            # ================================================================
            clean_condition, extracted_strings = self._extract_quoted_strings(condition)
            
            # ================================================================
            # STAGE 1: Build Evaluation Context (if not provided)
            # ================================================================
            if context is not None:
                eval_context = context  # ✅ Reuse provided context
            else:
                eval_context = self._build_eval_context(indicators, fundamentals)
            
            # Semantic booleans already in _build_eval_context
            """
            bbwidth = eval_context.get("bbwidth", 999)
            vol_qual = eval_context.get("volatilityquality", 0)
            eval_context["is_consolidating"] = bbwidth < 0.06 and vol_qual > 4.0
            eval_context["is_squeeze"] = bool(eval_context.get("ttm_squeeze", False))
            
            trend_sig = eval_context.get("ma_trend_signal", 0)
            eval_context["trend_dir"] = "up" if trend_sig > 0 else "down" if trend_sig < 0 else "neutral"
            """
            
            # ================================================================
            # STAGE 2: FAST PATH - Simple Comparisons
            # ================================================================
            
            # ✅ Use regex split for robust whitespace handling
            parts = re.split(r'\s+', clean_condition.strip())
            
            # Check for simple 3-part condition
            if len(parts) == 3 and not any(c in clean_condition for c in "()+-*/"):
                metric, op, target_raw = parts[0], parts[1], parts[2]
                
                # Restore original string if it was a placeholder
                if target_raw.startswith("__STR_"):
                    idx = int(target_raw.replace("__STR_", "").replace("__", ""))
                    target_raw = extracted_strings[idx]
                
                val = eval_context.get(metric)
                
                if val is not None:
                    target_str = target_raw.strip("'").strip('"').lower()
                    
                    # Try numeric comparison
                    try:
                        v_f, t_f = float(val), float(target_str)
                        ops_map = {
                            ">": operator.gt,
                            "<": operator.lt,
                            ">=": operator.ge,
                            "<=": operator.le,
                            "==": operator.eq,
                            "!=": operator.ne
                        }
                        result = ops_map.get(op, lambda a, b: False)(v_f, t_f)
                        logger.debug(f"[{self.horizon}] Simple: '{condition}' = {result}")
                        return result
                    
                    except (ValueError, TypeError):
                        # Boolean comparison
                        if target_str in ("true", "false"):
                            bool_val = bool(val) if not isinstance(val, str) else val.lower() == "true"
                            target_bool = target_str == "true"
                            result = bool_val == target_bool if op == "==" else bool_val != target_bool
                            logger.debug(f"[{self.horizon}] Boolean: '{condition}' = {result}")
                            return result
                        
                        # String comparison
                        if op == "==":
                            result = str(val).lower() == target_str
                            logger.debug(f"[{self.horizon}] String: '{condition}' = {result}")
                            return result
                        
                        if op == "!=":
                            result = str(val).lower() != target_str
                            logger.debug(f"[{self.horizon}] String: '{condition}' = {result}")
                            return result
            
            # ================================================================
            # STAGE 3: SAFE COMPLEX PATH (AST-based evaluation)
            # ================================================================
            
            result = self._safe_eval_math_expression(clean_condition, eval_context)
            logger.debug(f"[{self.horizon}] Complex: '{condition}' = {result}")
            return result
        
        except Exception as e:
            logger.debug(f"[{self.horizon}] Condition '{condition}' failed: {e}")
            return False


    def _extract_quoted_strings(self, condition: str):
        """
        Extract quoted strings and replace with placeholders.
        
        Example:
            Input: "status == 'Going Concern'"
            Output: ("status == __STR_0__", ['Going Concern'])
        """
        strings = []
        
        # Match both single and double quoted strings
        pattern = r"(['\"])([^'\"]*)\1"
        
        def replacer(match):
            strings.append(match.group(2))
            return f"__STR_{len(strings)-1}__"
        
        clean_condition = re.sub(pattern, replacer, condition)
        return clean_condition, strings


    def _safe_eval_math_expression(self, expression: str, context: Dict) -> bool:
        """
        Safely evaluates mathematical expressions using AST parsing.
        
        Supported:
        - Arithmetic: +, -, *, /, //, %, **
        - Functions: abs(), min(), max()
        - Comparisons: <, >, <=, >=, ==, !=
        - Logical: and, or, not
        
        NOT supported (security):
        - Imports: __import__, exec, eval
        - Attributes: obj.attr
        - Function calls except whitelist
        """
        try:
            # Parse expression into AST
            tree = ast.parse(expression, mode='eval')
            
            # Evaluate AST with whitelist
            result = self._eval_ast_node(tree.body, context)
            
            logger.debug(f"[{self.horizon}] Complex eval: '{expression}' = {result}")
            return bool(result)
        
        except Exception as e:
            logger.debug(f"[{self.horizon}] Complex eval failed: {e}")
            return False

    def _eval_ast_node(self, node, context: Dict):
        """
        Hardened AST evaluator with:
        - dict.get() method call support
        - Zero division guards
        - Dict literal support ({})
        - Python 3.8+ ast.Constant support
        """
        
        # ========================================================================
        # 1. CONSTANT VALUES (Numbers, Strings, Booleans)
        # ========================================================================
        
        if isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        
        elif isinstance(node, ast.Num):  # Legacy
            return node.n
        
        elif isinstance(node, ast.Str):  # Legacy
            return node.s
        
        # ========================================================================
        # 2. VARIABLES (Context Lookup)
        # ========================================================================
        
        if isinstance(node, ast.Name):
            return context.get(node.id, 0.0)
        
        # ========================================================================
        # 3. DICT LITERALS ({}, {'key': 'value'})
        # ========================================================================
        
        # ✅ CRITICAL: Allows {} as default value in .get() calls
        if isinstance(node, ast.Dict):
            # Recursively evaluate keys and values
            return {
                self._eval_ast_node(k, context): self._eval_ast_node(v, context)
                for k, v in zip(node.keys, node.values)
            }
        
        # ========================================================================
        # 4. BINARY OPERATIONS (+, -, *, /, //, %, **)
        # ========================================================================
        
        if isinstance(node, ast.BinOp):
            left = self._eval_ast_node(node.left, context)
            right = self._eval_ast_node(node.right, context)
            
            # Zero division guard
            if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if right == 0:
                    logger.debug(
                        f"[{self.horizon}] AST: Zero division prevented. Returning 0.0"
                    )
                    return 0.0
            
            ops_map = {
                ast.Add: operator.add,
                ast.Sub: operator.sub,
                ast.Mult: operator.mul,
                ast.Div: operator.truediv,
                ast.FloorDiv: operator.floordiv,
                ast.Mod: operator.mod,
                ast.Pow: operator.pow
            }
            
            op_func = ops_map.get(type(node.op))
            if op_func:
                return op_func(left, right)
            
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")
        
        # ========================================================================
        # 5. UNARY OPERATIONS (-, +, not)
        # ========================================================================
        
        if isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_node(node.operand, context)
            
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            if isinstance(node.op, ast.Not):
                return not operand
            
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")
        
        # ========================================================================
        # 6. COMPARISONS (<, >, <=, >=, ==, !=)
        # ========================================================================
        
        if isinstance(node, ast.Compare):
            left = self._eval_ast_node(node.left, context)
            
            ops_map = {
                ast.Lt: operator.lt,
                ast.LtE: operator.le,
                ast.Gt: operator.gt,
                ast.GtE: operator.ge,
                ast.Eq: operator.eq,
                ast.NotEq: operator.ne,
                ast.Is: operator.is_,    
                ast.IsNot: operator.is_not
            }
            
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_ast_node(comparator, context)
                op_func = ops_map.get(type(op))
                
                if not op_func:
                    raise ValueError(f"Unsupported comparison: {type(op)}")
                
                if not op_func(left, right):
                    return False
                
                left = right
            
            return True
        
        # ========================================================================
        # 7. BOOLEAN OPERATIONS (and, or)
        # ========================================================================
        
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._eval_ast_node(val, context) for val in node.values)
            if isinstance(node.op, ast.Or):
                return any(self._eval_ast_node(val, context) for val in node.values)
        
        # ========================================================================
        # 8. FUNCTION & METHOD CALLS
        # ========================================================================
        
        if isinstance(node, ast.Call):
            # A. Direct function calls: abs(x), min(a, b)
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                
                allowed_funcs = {
                    "abs": abs,
                    "min": min,
                    "max": max,
                    "round": round,
                    "int": int,
                    "float": float,
                    "len": len,
                    "bool": bool,
                    "str": str
                }
                
                if func_name not in allowed_funcs:
                    raise ValueError(f"Function '{func_name}' not allowed")
                
                args = [self._eval_ast_node(arg, context) for arg in node.args]
                return allowed_funcs[func_name](*args)
            
            # B. Method calls: dict.get('key', default)
            elif isinstance(node.func, ast.Attribute):
                obj = self._eval_ast_node(node.func.value, context)
                method_name = node.func.attr
                
                if method_name == "get" and isinstance(obj, dict):
                    args = [self._eval_ast_node(arg, context) for arg in node.args]
                    return obj.get(*args)
                
                raise ValueError(f"Method '{method_name}' not allowed on {type(obj)}")
            
            raise ValueError(f"Unsupported call type: {type(node.func)}")
        
        # ========================================================================
        # 9. SUBSCRIPT ACCESS (dict['key'], list[0])
        # ========================================================================
        
        if isinstance(node, ast.Subscript):
            obj = self._eval_ast_node(node.value, context)
            
            # Handle different Python versions
            if isinstance(node.slice, ast.Index):  # Python 3.8 and below
                key = self._eval_ast_node(node.slice.value, context)
            else:  # Python 3.9+
                key = self._eval_ast_node(node.slice, context)
            
            if isinstance(obj, (dict, list, tuple)):
                try:
                    return obj[key]
                except (KeyError, IndexError, TypeError):
                    return None
            
            raise ValueError(f"Subscript access not allowed on {type(obj)}")
        
        # ========================================================================
        # UNSUPPORTED NODE
        # ========================================================================
        
        raise ValueError(f"Unsupported AST node: {type(node).__name__}")
    
    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()
    
    # ============================================================
    # CRITICAL MISSING METHODS (Category A - Actively Used)
    # ============================================================
    
    def clamp_sl_distance(self, risk: float, price: float) -> float:
        """
        Enforces min/max SL distance limits (horizon-specific).
        Used by: calculate_execution_plan()
        """
        limits = self.get("risk_management.atr_sl_limits", {})
        
        max_pct = limits.get("max_percent", 0.05)
        min_pct = limits.get("min_percent", 0.01)
        
        max_risk = price * max_pct
        min_risk = price * min_pct
        
        return max(min_risk, min(risk, max_risk))
    
    def detect_divergence(self, indicators: Dict) -> Dict:
        """
        Config-driven divergence detection.
        Used by: detect_divergence_via_slopes()
        """
        rsislope = self._get_val(indicators, "rsislope")
        price = self._get_val(indicators, "price")
        prev_price = self._get_val(indicators, "prev_close", price)
        
        # Get thresholds from momentum config (horizon-specific with global fallback)
        thresh = self.get("momentum_thresholds.rsislope.deceleration_ceiling", -0.05)
        
        # Get penalty from calculation engine (global)
        penalty = self.get("calculation_engine.divergence_detection.confidence_penalties.bearish_divergence", 0.70)
        
        # Bearish divergence: Price rising but RSI falling
        if price > prev_price and rsislope < thresh:
            return {
                'divergence_type': 'bearish',
                'confidence_factor': penalty,
                'warning': f"Bearish Divergence: RSI_slope={rsislope:.2f}",
                'severity': 'moderate'
            }
        
        return {
            'divergence_type': 'none',
            'confidence_factor': 1.0,
            'warning': None,
            'severity': None
        }
    
    def detect_volume_signature(self, indicators: Dict) -> Dict:
        """
        Config-driven volume signature detection.
        Used by: detect_volume_signature_legacy()
        """
        rvol = self._get_val(indicators, "rvol")
        
        # Get thresholds (horizon-specific with global fallback)
        surge_thresh = self.get("volume_analysis.rvol_surge_threshold", 3.0)
        drought_thresh = self.get("volume_analysis.rvol_drought_threshold", 0.7)
        
        # Get adjustments from calculation engine (global)
        surge_adj = self.get("calculation_engine.volume_signatures.surge.confidence_adjustment", 15)
        drought_adj = self.get("calculation_engine.volume_signatures.drought.confidence_adjustment", -25)
        
        if rvol >= surge_thresh:
            return {
                'type': 'surge',
                'adjustment': surge_adj,
                'warning': f'Volume surge (RVOL={rvol:.2f})'
            }
        
        if rvol <= drought_thresh:
            return {
                'type': 'drought',
                'adjustment': drought_adj,
                'warning': f'Volume drought (RVOL={rvol:.2f})'
            }
        
        return {'type': 'normal', 'adjustment': 0, 'warning': None}
    
    # def get_confidence_floor(self, setup_type: str = None) -> int:
    #     """
    #     Get confidence floor for setup type (horizon-specific).
        
    #     Resolution order:
    #     1. Horizon-specific base_floors
    #     2. Global base_floors
    #     3. Default 55
    #     """
    #     if setup_type:
    #         # Try horizon first
    #         base_floors = self.get("confidence.base_floors", {})
    #         floor = base_floors.get(setup_type)
            
    #         if floor is not None:
    #             return floor
            
    #         # Fallback to global if not in horizon
    #         global_floors = self.global_config.get("confidence", {}).get("base_floors", {})
    #         floor = global_floors.get(setup_type)
            
    #         if floor is not None:
    #             logger.debug(
    #                 f"{self.horizon}: Using global base_floor for {setup_type}: {floor}"
    #             )
    #             return floor
            
    #         # Ultimate fallback
    #         logger.warning(
    #             f"{self.horizon}: No base_floor found for {setup_type}, using 55"
    #         )
    #         return 55
        
    #     # If no setup_type provided, return buy floor
    #     floors = self.get("confidence.floors", {})
    #     return floors.get("buy", 55)

    
    def get_spread_adjustment(self, market_cap: float) -> float:
        """
        Returns spread % based on market cap (global).
        Used by: calculate_execution_plan()
        """
        brackets = self.get("calculation_engine.spread_adjustment.market_cap_brackets", {})
        
        large = brackets.get("large_cap", {})
        mid = brackets.get("mid_cap", {})
        small = brackets.get("small_cap", {})
        
        if market_cap >= large.get("min", 100000):
            return large.get("spread_pct", 0.001)
        elif market_cap >= mid.get("min", 10000):
            return mid.get("spread_pct", 0.002)
        else:
            return small.get("spread_pct", 0.005)

    # ============================================================
    # HELPER UTILITIES (Category B - Should Have)
    # ============================================================
    
    def get_indicator(self, name: str, fallback: Any = None) -> Any:
        """
        Get indicator config with legacy key mapping.
        Safe accessor for indicator parameters.
        """
        # Check if it's a period config
        if name in ["adx_period", "atr_period", "rsi_period"]:
            return self.get(f"indicators.{name}", fallback)
        
        # Check if it's a stochastic config
        if name.startswith("stoch_"):
            stoch_key = name.replace("stoch_", "")
            return self.get(f"indicators.stochastic.{stoch_key}", fallback)
        
        # Legacy MA key mapping
        ma_type = self.get("moving_averages.type", "EMA")
        
        ma_mapping = {
            "ma_fast": {
                "EMA": self.get("moving_averages.fast", 20),
                "WMA": self.get("moving_averages.fast", 10),
                "MMA": self.get("moving_averages.fast", 6)
            },
            "ma_mid": {
                "EMA": self.get("moving_averages.mid", 50),
                "WMA": self.get("moving_averages.mid", 40),
                "MMA": self.get("moving_averages.mid", 12)
            },
            "ma_slow": {
                "EMA": self.get("moving_averages.slow", 200),
                "WMA": self.get("moving_averages.slow", 50),
                "MMA": self.get("moving_averages.slow", 12)
            }
        }
        
        if name in ma_mapping:
            return ma_mapping[name].get(ma_type, fallback)
        
        return self.get(f"indicators.{name}", fallback)
    
    def get_many(self, paths: List[str], defaults: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Get multiple config values at once.
        Performance helper for batch retrieval.
        """
        defaults = defaults or {}
        return {
            path: self.get(path, default=defaults.get(path))
            for path in paths
        }
    
    def get_pattern_entry_rule(self, pattern: str, rule: str) -> Any:
        """
        Get pattern-specific entry rules.
        Path: global.pattern_entry_rules.{pattern}.horizons.{horizon}.{rule}
        """
        path = f"pattern_entry_rules.{pattern}.horizons.{self.horizon}.{rule}"
        return self.get(path)
    
    def get_pattern_invalidation(self, pattern: str) -> Dict:
        """
        Get pattern invalidation rules (horizon-specific).
        Path: global.pattern_invalidation.{pattern}
        """
        path = f"pattern_invalidation.{pattern}"
        rules = self.get(path, {})
        
        breakdown = rules.get("breakdown_threshold", {}).get(self.horizon, {})
        action = rules.get("action", {}).get(self.horizon, "EXIT_ON_CLOSE")
        
        return {
            "condition": breakdown.get("condition"),
            "duration_candles": breakdown.get("duration_candles"),
            "action": action,
            "or_condition": breakdown.get("or_condition")
        }
    
    # ============================================================
    # CONVENIENCE WRAPPERS (Category C - Optional but Useful)
    # ============================================================
    
    def get_setup_multiplier(self, setup_type: str) -> float:
        """
        Get position size multiplier for setup type (global).
        Path: global.positionsizing.globalsetupmultipliers.{setup_type}
        Args:
            setup_type: Setup classification (e.g., "VALUE_TURNAROUND")
        Returns:
            Setup multiplier (default 1.0 if not found)
        Example:
            >>> config.get_setup_multiplier("VALUE_TURNAROUND")
            1.4
        """
        # ✅ FIXED - Use correct path with smart inheritance
        multipliers = self.get("global.positionsizing.globalsetupmultipliers", {})
        mult = multipliers.get(setup_type, 1.0)
        
        logger.debug( f"[{self.horizon}] Setup multiplier for {setup_type}: {mult:.2f}" )
        return mult


    def get_position_sizing_config(self) -> Dict:
        """
        Get complete position sizing configuration for this horizon.
        
        Uses smart inheritance:
        - base_risk_pct: Horizon-specific with global fallback
        - max_position_pct: Horizon-specific with global fallback  
        - setup_multipliers: Global only
        - volatility_adjustments: Global only
        
        Returns:
            Dictionary with all position sizing parameters
        
        Example:
            >>> config = get_config("long_term")
            >>> ps = config.get_position_sizing_config()
            >>> print(ps["base_risk_pct"])  # 0.015 (long_term override)
            >>> print(ps["setup_multipliers"]["VALUE_TURNAROUND"])  # 1.4 (global)
        """
        # ✅ FIXED - Use correct paths with smart inheritance
        return {
            # Horizon-specific with global fallback
            "base_risk_pct": self.get("global.positionsizing.baseriskpct", 0.01),
            
            # Global only (not horizon-specific)
            "setup_multipliers": self.get("global.positionsizing.globalsetupmultipliers", {}),
            "volatility_adjustments": self.get("global.positionsizing.volatilityadjustments", {}),
            
            # Horizon-specific with global fallback
            "max_position_pct": self.get("riskmanagement.maxpositionpct", 0.02)
        }

    def get_volatility_multiplier(self, vol_quality: float) -> float:
        """
        Get position size multiplier based on volatility quality.
        Path: global.positionsizing.volatilityadjustments
        Args:
            vol_quality: Volatility quality score (0-10)
        Returns:
            Volatility multiplier (1.2 for high, 1.0 for medium, 0.9 for low)
        Example:
            >>> config.get_volatility_multiplier(7.5)
            1.2  # High quality
        """
        adjustments = self.get("global.positionsizing.volatilityadjustments", {})
        
        # Check high quality first
        high_cfg = adjustments.get("high_quality", {})
        if "vol_qual_min" in high_cfg and vol_quality >= high_cfg["vol_qual_min"]:
            mult = high_cfg.get("multiplier", 1.2)
            logger.debug(f"[{self.horizon}] Vol regime: HIGH (qual={vol_quality:.1f}, mult={mult:.2f})")
            return mult
        
        # Check low quality
        low_cfg = adjustments.get("low_quality", {})
        if "vol_qual_max" in low_cfg and vol_quality <= low_cfg["vol_qual_max"]:
            mult = low_cfg.get("multiplier", 0.9)
            logger.debug(f"[{self.horizon}] Vol regime: LOW (qual={vol_quality:.1f}, mult={mult:.2f})")
            return mult
        
        # Default to medium
        medium_cfg = adjustments.get("medium_quality", {})
        mult = medium_cfg.get("multiplier", 1.0)
        logger.debug(f"[{self.horizon}] Vol regime: MEDIUM (qual={vol_quality:.1f}, mult={mult:.2f})")
        return mult

    def get_proximity_rejection(self) -> Dict[str, float]:
        """
        Get S/R proximity rejection multipliers.
        Part of execution params.
        """
        return self.get("execution.proximity_rejection", {
            "resistance_mult": 1.005,
            "support_mult": 0.995
        })
    
    def get_technical_weight(self, indicator: str) -> float:
        """
        Get technical indicator weight with horizon overrides.
        Path: global.technical_weights.{indicator}.weight (with horizon overrides)
        """
        base_weight = self.get(f"technical_weights.{indicator}.weight", 1.0)
        override = self.get(f"technical_weight_overrides.{indicator}")
        
        if override is not None:
            return base_weight * override
        return base_weight

    def get_fundamental_mix(self) -> float:
        """
        Get fundamental weight for this horizon's scoring mix.
        Returns tech/fund split ratio (0.0 = pure technical, 1.0 = pure fundamental).
        
        Path: horizons.{horizon}.scoring.fundamental_weight (with global fallback)
        """
        return self.get("scoring.fundamental_weight", 0.0)

    def get_execution_params(self) -> Dict[str, Any]:
        """Get execution parameters (horizon-specific)."""
        return {
            "stop_loss_atr_mult": self.get("execution.stop_loss_atr_mult", 2.0),
            "target_atr_mult": self.get("execution.target_atr_mult", 3.0),
            "max_hold_candles": self.get("execution.max_hold_candles", 20),
            "risk_reward_min": self.get("execution.risk_reward_min", 2.0),
            "base_hold_days": self.get("execution.base_hold_days", 10),
            "proximity_rejection": self.get("execution.proximity_rejection", {
                "resistance_mult": 1.005,
                "support_mult": 0.995
            }),
            "min_profit_pct": self.get("execution.min_profit_pct", 0.5)
        }
    def get_risk_params(self) -> Dict[str, Any]:
        """Get risk management parameters (horizon-specific)."""
        return {
            "max_position_pct": self.get("risk_management.max_position_pct", 0.02),
            "min_rr_ratio": self.get("risk_management.min_rr_ratio", 1.5),
            "horizon_t2_cap": self.get("risk_management.horizon_t2_cap", 0.10),
            "atr_sl_limits": self.get("risk_management.atr_sl_limits", {}),
            "setup_size_multipliers": self.get("risk_management.setup_size_multipliers", {}),
            "rr_regime_adjustments": self.get("risk_management.rr_regime_adjustments", {})
        }

    # ============================================================
    # FIXED: Gates Validation
    # ============================================================
    # ========== NEW METHODS - Add to ConfigResolver class ==========

    def get_strategy_preferences(self) -> Dict[str, Any]:
        """
        Get strategy preferences for current horizon.
        
        Path: global.strategy_preferences.horizon_strategy_config.{horizon}
        
        Returns:
            Dict with:
            - preferred_setups: List[str]
            - blocked_setups: List[str]
            - sizing_multipliers: Dict[str, float]
            - min_fundamental_score: Optional[float]
            - filters: Dict (long_term/multibagger only)
        
        Example:
            >>> config = get_config("intraday")
            >>> prefs = config.get_strategy_preferences()
            >>> prefs["blocked_setups"]
            ['DEEP_VALUE_PLAY', 'QUALITY_ACCUMULATION']
        """
        strategy_config = self.get(
            f"strategy_preferences.horizon_strategy_config.{self.horizon}",
            {}
        )     
        return {
            "preferred_setups": strategy_config.get("preferred_setups", []),
            "blocked_setups": strategy_config.get("blocked_setups", []),
            "sizing_multipliers": strategy_config.get("sizing_multipliers", {}),
            "min_fundamental_score": strategy_config.get("min_fundamental_score"),
            "filters": strategy_config.get("filters", {})
        }


    def is_setup_allowed(self, setup_type: str) -> bool:
        """
        Check if setup is allowed for current horizon strategy.
        
        Args:
            setup_type: Setup classification (e.g., "MOMENTUM_BREAKOUT")
        
        Returns:
            True if setup is not in blocked list
        
        Example:
            >>> config = get_config("intraday")
            >>> config.is_setup_allowed("DEEP_VALUE_PLAY")
            False  # Blocked for intraday
            >>> config.is_setup_allowed("MOMENTUM_BREAKOUT")
            True   # Allowed for intraday
        """
        prefs = self.get_strategy_preferences()
        blocked = prefs.get("blocked_setups", [])
        
        if setup_type in blocked:
            logger.info(f"{self.horizon}: Setup {setup_type} BLOCKED by strategy preferences")
            return False
        
        return True


    def get_strategy_sizing_multiplier(self, setup_type: str) -> float:
        """
        Get strategy-specific sizing multiplier.
        
        These are SEPARATE from setup multipliers - they represent
        horizon strategy preferences (e.g., 'we prefer accumulation for long-term').
        
        Args:
            setup_type: Setup classification
        
        Returns:
            Multiplier from strategy preferences, or 1.0 if not specified
        
        Example:
            >>> config = get_config("multibagger")
            >>> config.get_strategy_sizing_multiplier("VALUE_TURNAROUND")
            1.8  # Multibagger prefers value setups
            >>> config.get_strategy_sizing_multiplier("MOMENTUM_BREAKOUT")
            0.5  # Multibagger reduces momentum exposure
        """
        prefs = self.get_strategy_preferences()
        multiplier = prefs.get("sizing_multipliers", {}).get(setup_type, 1.0)
        
        if multiplier != 1.0:
            logger.debug(
                f"{self.horizon}: Strategy sizing multiplier for {setup_type}: {multiplier:.2f}"
            )
        
        return multiplier


    def apply_fundamental_filters(
        self, 
        fundamentals: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Apply horizon-specific fundamental filters.
        Path: global.strategy_preferences.horizon_strategy_config.{horizon}.filters
        Only applies to horizons with min_fundamental_score > 0.        
        Args:
            fundamentals: Fundamental data dict
        Returns:
            (passes: bool, failures: List[str])
        Example:
            >>> config = get_config("multibagger")
            >>> fundamentals = {"roe": 10, "roce": 12, "deratio": 0.3}
            >>> passes, failures = config.apply_fundamental_filters(fundamentals)
            >>> passes
            False
            >>> failures
            ['ROE 10.0% < 20%', 'ROCE 12.0% < 25%']
        """
        prefs = self.get_strategy_preferences()
        
        min_score = prefs.get("min_fundamental_score")
        if min_score is None or min_score == 0:
            return (True, [])  # No fundamental requirements for this horizon
        
        filters = prefs.get("filters", {})
        if not filters:
            return (True, [])  # No filters defined
        
        failures = []
        
        # Check require_low_debt
        if filters.get("require_low_debt"):
            max_de = filters.get("max_de_ratio", 0.5)
            deratio = _get_val(fundamentals, "deratio", 0)
            if deratio > max_de:
                failures.append(f"DE ratio {deratio:.2f} > {max_de}")
        
        # Check min_roe
        if "min_roe" in filters:
            min_roe = filters["min_roe"]
            roe = _get_val(fundamentals, "roe", 0)
            if roe < min_roe:
                failures.append(f"ROE {roe:.1f}% < {min_roe}%")
        
        # Check min_roce
        if "min_roce" in filters:
            min_roce = filters["min_roce"]
            roce = _get_val(fundamentals, "roce", 0)
            if roce < min_roce:
                failures.append(f"ROCE {roce:.1f}% < {min_roce}%")
        
        # Check min_piotroski_f (multibagger only)
        if "min_piotroski_f" in filters:
            min_f = filters["min_piotroski_f"]
            f_score = _get_val(fundamentals, "piotroski_f", 0)
            if f_score < min_f:
                failures.append(f"Piotroski F-Score {f_score} < {min_f}")
        
        # Check max_pe_ratio
        if "max_pe_ratio" in filters:
            max_pe = filters["max_pe_ratio"]
            peratio = _get_val(fundamentals, "peratio", 999)
            if peratio > max_pe:
                failures.append(f"P/E ratio {peratio:.1f} > {max_pe}")
        
        passes = len(failures) == 0
        
        if not passes:
            logger.info(
                f"{self.horizon}: Fundamental filters FAILED: {', '.join(failures)}"
            )
        else:
            logger.info(f"{self.horizon}: Fundamental filters PASSED ✓")
        
        return (passes, failures)


    def validate_volatility_regime(
        self, 
        indicators: Dict, 
        setup_type: str
    ) -> Tuple[bool, str]:
        """
        Enhanced volatility regime validation using guards.
        
        Now supports setup-specific quality requirements from gates.
        
        Args:
            indicators: Technical indicators
            setup_type: Setup classification
        
        Returns:
            (should_trade: bool, reason: str)
        
        Example:
            >>> config = get_config("intraday")
            >>> can_trade, reason = config.validate_volatility_regime(
            ...     indicators={"volatilityquality": 3.0, "atr_pct": 2.5},
            ...     setup_type="MOMENTUM_BREAKOUT"
            ... )
            >>> can_trade
            True
            >>> reason
            "Volatility expansion allowed for breakout"
        """
        vol_qual = _get_val(indicators, "volatilityquality")
        atr_pct = _get_val(indicators, "atr_pct")
        
        if vol_qual is None or atr_pct is None:
            return (True, "Missing vol data, proceed cautiously")
        
        # Get guards from horizon gates
        guards = self.get_volatility_guards()
        bands = self._normalize_volatility_bands()
        
        # 1. Extreme volatility check
        extreme_buffer = guards.get("extreme_vol_buffer", 2.0)
        max_band = bands.get("max", 12.0)
        extreme_threshold = max_band * extreme_buffer
        
        if atr_pct > extreme_threshold:
            return (False, f"Extreme volatility {atr_pct:.1f}%, avoid all entries")
        
        # 2. Setup-specific quality check
        if "BREAKOUT" in setup_type or "BREAKDOWN" in setup_type:
            min_qual = guards.get("min_quality_breakout", 2.0)
            if vol_qual < min_qual:
                return (False, f"Vol quality {vol_qual:.1f} < {min_qual} for breakout")
            return (True, "Volatility expansion allowed for breakout")
        
        # 3. Normal setup quality check
        min_qual = guards.get("min_quality_normal", 4.0)
        if vol_qual < min_qual:
            return (False, f"Low vol quality {vol_qual:.1f}, potential chop")
        
        return (True, "Volatility regime favorable")


    def get_confidence_floor(self, setup_type: str = None) -> int:
        """
        Get confidence floor for setup type (horizon-specific).
        
        ENHANCED version with better fallback for new setups.
        
        Resolution order:
        1. Horizon-specific base_floors
        2. Global base_floors
        3. Default 55
        
        Args:
            setup_type: Setup classification (optional)
        
        Returns:
            Confidence floor (35-75)
        
        Example:
            >>> config = get_config("short_term")
            >>> config.get_confidence_floor("MOMENTUM_BREAKOUT")
            55
            >>> config.get_confidence_floor("PATTERN_STRIKE_REVERSAL")
            50  # Falls back to global if not in short_term
        """
        if setup_type:
            # Try horizon first
            base_floors = self.get("confidence.base_floors", {})
            floor = base_floors.get(setup_type)
            
            if floor is not None:
                return floor
            
            # Fallback to global if not in horizon
            global_floors = self._global_config.get("confidence", {}).get("base_floors", {})
            floor = global_floors.get(setup_type)
            
            if floor is not None:
                logger.debug(
                    f"{self.horizon}: Using global base_floor for {setup_type}: {floor}"
                )
                return floor
            
            # Ultimate fallback
            logger.warning(
                f"{self.horizon}: No base_floor found for {setup_type}, using 55"
            )
            return 55
        
        # If no setup_type provided, return buy floor
        floors = self.get("confidence.floors", {})
        return floors.get("buy", 55)


# ========== END NEW METHODS ==========


    def get_gate_checks(self) -> Dict[str, Any]:
        """
        ✅ UPDATED: Support for the new 'setup_gate_specifications' structure.
        """
        gates = self.get_section("gates", merge_global=True)
        
        # 1. Fetch Universal Gates (The "Nature" of the setup)
        # This is the new path from your patch!
        universal_specs = self._global_config.get("setup_gate_specifications", {})
        
        # 2. Fetch Horizon Overrides (Confidence and Volatility)
        horizon_overrides = self._horizon_config.get("gates", {}).get("setup_gate_overrides", {})
        
        # 3. Smart Merge Logic
        final_overrides = {}
        all_setup_names = set(universal_specs.keys()) | set(horizon_overrides.keys())
        
        for setup in all_setup_names:
            # Get nature/universal gates first
            spec = universal_specs.get(setup, {}).get("universal_gates", {})
            # Get horizon specific overrides (vol/conf)
            h_ovr = horizon_overrides.get(setup, {})
            
            # Horizon overrides ALWAYS win (e.g., your adx_min: 8 for long-term)
            final_overrides[setup] = {**spec, **h_ovr}
        
        gates["setup_gate_overrides"] = final_overrides
        logger.debug(f"[{self.horizon}] Gate resolution:")
        logger.debug(f"  Global defaults: {len(universal_specs)} setups")
        logger.debug(f"  Horizon overrides: {len(horizon_overrides)} setups")
        logger.debug(f"  Final merged: {len(final_overrides)} setups")
        return gates
    
    def get_trend_threshold(self, metric: str) -> float:
        """
        Get trend slope thresholds (horizon-specific).
        """
        return self.get(f"trend_thresholds.slope.{metric}", 10.0)
    
    def get_volatility_bands(self) -> Dict[str, float]:
        """
        Get volatility bands (min, ideal, max).
        Alias for _normalize_volatility_bands() for backward compatibility.
        """
        return self._normalize_volatility_bands()
    
    def get_volatility_guards(self) -> Dict[str, float]:
        """
        Get volatility guard thresholds - horizon-specific.
        
        Path: horizons.{horizon}.gates.volatility_guards
        Fallback: global defaults
        
        Returns:
            Dict with:
            - extreme_vol_buffer: float (how much above max triggers extreme)
            - min_quality_breakout: float (min vol quality for breakout setups)
            - min_quality_normal: float (min vol quality for normal setups)
        """
        guards = self.get("gates.volatility_guards", {})
        
        # Ensure all required keys exist with proper defaults
        return {
            "extreme_vol_buffer": guards.get("extreme_vol_buffer", 2.0),
            "min_quality_breakout": guards.get("min_quality_breakout", 3.0),
            "min_quality_normal": guards.get("min_quality_normal", 4.0)
        }

    
    def get_volume_threshold(self, metric: str) -> float:
        """
        Get volume analysis thresholds (rvol_surge, rvol_drought).
        Horizon-specific with global fallback.
        """
        key_map = {
            "rvol_surge": "volume_analysis.rvol_surge_threshold",
            "rvol_drought": "volume_analysis.rvol_drought_threshold"
        }
        path = key_map.get(metric, f"volume_analysis.{metric}")
        return self.get(path, 1.0)
    
    def should_trade_volatility(self, indicators: Dict, setup_type: str) -> Tuple[bool, str]:
        """
        Validates volatility regime for trading.
        Returns: (should_trade: bool, reason: str)
        """
        vol_qual = self._get_val(indicators, "volatilityquality")
        atr_pct = self._get_val(indicators, "atr_pct")
        
        if vol_qual is None or atr_pct is None:
            return True, "Missing vol data, proceed cautiously"
        
        # Extreme volatility check
        guards = self.get("gates.volatility_guards", {})
        extreme_buffer = guards.get("extreme_vol_buffer", 2.0)
        bands = self._normalize_volatility_bands()
        
        if atr_pct > bands.get("max", 12.0) + extreme_buffer:
            return False, f"Extreme volatility ({atr_pct:.1f}%), avoid all entries"
        
        # Breakout exception
        if "BREAKOUT" in setup_type or "BREAKDOWN" in setup_type:
            min_qual = guards.get("min_quality_breakout", 2.0)
            if vol_qual < min_qual:
                return False, f"Vol quality {vol_qual:.1f} < {min_qual} for breakout"
            return True, "Volatility expansion allowed for breakout"
        
        # Standard quality check
        min_qual = guards.get("min_quality_normal", 4.0)
        if vol_qual < min_qual:
            return False, f"Low vol quality ({vol_qual:.1f}), potential chop"
        
        return True, "Volatility regime favorable"


# ============================================================
# Module-level API
# ============================================================

_cache: Dict[str, ConfigResolver] = {}

# BOTTOM OF config_resolver.py - REPLACE existing get_config()
# config_resolver.py - FIXED get_config() with caching
# Global _cache: horizon → ConfigResolver(indicators, fundamentals)

def get_config(horizon: str = "short_term", 
               indicators: Dict = None, 
               fundamentals: Dict = None,
               master_config: Dict = None) -> 'ConfigResolver':
    """🔥 CACHED: Pass indicators/fundamentals ONCE per horizon → Reused everywhere!"""

    # if horizon not in _cache:
    #     logger.info(f"📄 [config] NEW ConfigResolver({horizon})")
    # else:
    #     logger.debug(f"🔄 [config] CACHE HIT {horizon}")
    
    if horizon not in _cache:
        _cache[horizon] = ConfigResolver(
            horizon=horizon, 
            indicators=indicators, 
            fundamentals=fundamentals,
            master_config=master_config
        )
        # logger.info(f"🔄 [config] NEW ConfigResolver({horizon}): indicators={len(indicators or {})} keys")
    else:
        # UPDATE existing _cache with new data (sequential flow)
        cached = _cache[horizon]
        if indicators and cached.indicators != indicators:
            cached.indicators = indicators
            cached.clear_cache()
            # logger.debug(f"🔄 [config] UPDATED {horizon} indicators={len(indicators)}")
        if fundamentals and cached.fundamentals != fundamentals:
            cached.fundamentals = fundamentals
            cached.clear_cache()
            # logger.debug(f"🔄 [config] UPDATED {horizon} fundamentals={len(fundamentals)}")
    return _cache[horizon]


def clear_config_cache():
    """Clear all cached ConfigResolver instances."""
    _cache.clear()