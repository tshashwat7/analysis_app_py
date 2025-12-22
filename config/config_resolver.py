# config/config_resolver.py
"""
Smart Configuration resolver with Inheritance & Dynamic Key Mapping
Handles horizon-aware config resolution with fallback chains.

Usage:
    from config.config_resolver import ConfigResolver
    
    config = ConfigResolver(horizon="intraday")
    
    # Get consolidated context for Signal Engine
    ctx = config.get_signal_context()
    
    # Get specific value
    sl_mult = config.get("execution.stop_loss_atr_mult")
"""

from typing import Any, Dict, Optional, List, Tuple, Union
from copy import deepcopy
import logging
import operator
logger = logging.getLogger(__name__)

# Import your master config
from config.master_config import MASTER_CONFIG


class ConfigResolver:
    """
    Smart configuration resolver with horizon-aware inheritance.
    
    Resolution Order:
    1. horizons.[horizon].[section].[key]
    2. global.[section].[key]
    3. default value
    """
    
    def __init__(self, horizon: str = "short_term", master_config: Dict = None):
        """
        Initialize config Resolver for a specific horizon.
        
        Args:
            horizon: One of ['intraday', 'short_term', 'long_term', 'multibagger']
            master_config: Optional custom config dict (defaults to MASTER_CONFIG)
        """
        self.horizon = horizon
        self.config = master_config or MASTER_CONFIG
        self._cache = {}
        
        # Validate horizon
        valid_horizons = ["intraday", "short_term", "long_term", "multibagger"]
        if horizon not in valid_horizons:
            logger.warning(f"Invalid horizon '{horizon}', falling back to 'short_term'")
            self.horizon = "short_term"
        
        # Pre-cache common lookups
        self._horizon_config = self.config.get("horizons", {}).get(self.horizon, {})
        self._global_config = self.config.get("global", {})
    
    # ============================================================
    # CORE GET METHODS
    # ============================================================
    
    def get(self, path: str, default: Any = None) -> Any:
        """
        Get config value with smart inheritance.
        
        Args:
            path: Dot-separated path (e.g., "indicators.adx_period")
            default: Fallback value if not found
            
        Returns:
            Config value with horizon override or global fallback
        """
        # Check cache first
        cache_key = f"{self.horizon}:{path}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Try horizon-specific first
        value = self._get_nested(self._horizon_config, path)
        
        # Fallback to global
        if value is None:
            value = self._get_nested(self._global_config, path)
        
        # Fallback to top-level global (for paths like "global.system.cache")
        if value is None and path.startswith("global."):
            global_path = path[7:]  # Remove "global." prefix
            value = self._get_nested(self._global_config, global_path)
        
        # Use default if still not found
        if value is None:
            value = default
        
        # Cache result
        self._cache[cache_key] = value
        return value
    
    def get_section(self, section: str, merge_global: bool = True) -> Dict:
        """Get config section + auto-unwrap common containers like 'rules', 'penalties'."""
        if merge_global:
            result = deepcopy(self._global_config.get(section, {}))
            horizon_section = self._horizon_config.get(section, {})
            result.update(horizon_section)
        else:
            result = self._horizon_config.get(section, {})
        
        # 🔥 MAGIC: Auto-unwrap standard containers
        wrappers = ['rules', 'penalties', 'enhancements', 'gates', 'bonuses', 'conditions']
        for key in wrappers:
            if isinstance(result, dict) and key in result and isinstance(result[key], dict):
                logger.debug(f"[{self.horizon}] get_section({section}): auto-unwrapped '{key}'")
                return result[key]
        
        return result


    
    def get_many(self, paths: List[str], defaults: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get multiple config values at once."""
        defaults = defaults or {}
        return {
            path: self.get(path, default=defaults.get(path))
            for path in paths
        }
    
    # ============================================================
    # 1. CORE GETTERS (Inheritance Logic)
    # ============================================================

    # ============================================================
    # 2. SETUP CLASSIFICATION ENGINE (Gap 1)
    # ============================================================

    def get_setup_rules(self) -> Dict[str, Any]:
        """Get all setup classification rules for this horizon."""
        return self.get_section("setup_classification", merge_global=True)

    def get_pattern_priority(self) -> List[Tuple[str, str, int]]:
        """Returns ordered list: [(pattern_key, setup_name, min_score), ...]"""
        priority = self.get("global.calculation_engine.setup_classification.pattern_priority", [])
        return [(p["pattern"], p["setup_name"], p["min_score"]) for p in priority]

    def evaluate_setup_condition(self, setup_type: str, indicators: Dict, fundamentals: Dict = None) -> bool:
        """
        Evaluates if indicators/fundamentals meet config-defined conditions.
        Example: "conditions": ["rsi >= 60", "roe >= 20"]
        """
        rules = self.get("global.calculation_engine.setup_classification.rules") or {}
        setup_rule = rules.get(setup_type)
        
        if not setup_rule: 
            return False
            
        conditions = setup_rule.get("conditions", [])
        
        for cond in conditions:
            # ✅ Pass BOTH indicators AND fundamentals
            if not self._evaluate_condition_string(cond, indicators, fundamentals):
                return False
        return True
    # ============================================================
    # 3. DYNAMIC CONFIDENCE ENGINE (Gap 2)
    # ============================================================

    def calculate_dynamic_confidence_floor(self, adx: float, setup_type: str) -> float:
        """
        Calculates horizon-aware dynamic confidence floor based on ADX.
        Formula: Base Floor - (ADX_Normalized * Adjustment_Factor)
        """
        # Get config values
        base_floors = self.get("confidence.base_floors", {})
        norm_cfg = self.get("confidence.adx_normalization", {})
        
        # 1. Get Base Floor for Setup
        base = base_floors.get(setup_type, 55)
        
        # 2. Normalize ADX
        adx_min = norm_cfg.get("min", 10)
        adx_max = norm_cfg.get("max", 40)
        adj_factor = norm_cfg.get("adjustment_factor", 12)
        
        adx_val = float(adx or 0)
        adx_norm = max(0.0, min(1.0, (adx_val - adx_min) / (adx_max - adx_min)))
        
        # 3. Calculate Adjustment
        adjustment = adx_norm * adj_factor
        
        # 4. Apply Horizon Discount (if any)
        horizon_discount = self.get("confidence.horizon_discount", 0)
        
        final_floor = base - adjustment - horizon_discount
        return max(35.0, min(75.0, final_floor))

    # ============================================================
    # 4. COMPOSITE SCORING ENGINE (Gap 3)
    # ============================================================

    def compute_composite_score(self, composite_name: str, indicators: Dict) -> Dict[str, Any]:
        """
        Calculates composite scores (Trend, Momentum, Volatility) using config weights.
        """
        weights_cfg = self.get(f"global.calculation_engine.composite_weights.{composite_name}", {})
        
        if composite_name == "trend_strength":
            return self._compute_trend_strength(indicators, weights_cfg)
        elif composite_name == "momentum_strength":
            return self._compute_momentum_strength(indicators, weights_cfg)
        elif composite_name == "volatility_quality":
            return self._compute_volatility_quality(indicators, weights_cfg)
        
        return {"raw": 0, "value": 0, "score": 0, "desc": "Unknown Composite"}

    def _compute_trend_strength(self, indicators: Dict, cfg: Dict) -> Dict:
        """Trend Strength: ADX + Slope + DI Spread + Supertrend"""
        # 1. Get Values
        adx = self._get_val(indicators, "adx")
        slope = abs(self._get_val(indicators, "ma_fast_slope"))
        di_spread = self._get_val(indicators, "di_plus") - self._get_val(indicators, "di_minus")
        st_signal = str(self._get_nested(indicators, "supertrend_signal.value") or "")
        
        # 2. Score Components using Config Thresholds
        adx_t = cfg.get("adx", {}).get("thresholds", {})
        adx_s = 10 if adx >= adx_t.get("strong", 25) else 7 if adx >= adx_t.get("moderate", 20) else 2
        
        # Slope logic requires config mapping for horizon-specific thresholds? 
        # Actually Master Config has a hardcoded 'thresholds' in calc engine, let's use that.
        slope_t = cfg.get("ema_slope", {}).get("thresholds", {})
        slope_s = 10 if slope >= slope_t.get("strong", 20) else 7 if slope >= slope_t.get("moderate", 5) else 2
        
        di_s = 10 if di_spread >= 15 else 5
        st_s = 10 if "Bull" in st_signal else 0
        
        # 3. Apply Weights
        w = cfg.get("adaptive_weights_no_supertrend") if not st_signal else cfg
        
        # Fallback to standard weights if adaptive missing
        wa = cfg.get("adx", {}).get("weight", 0.4)
        wb = cfg.get("ema_slope", {}).get("weight", 0.3)
        wc = cfg.get("di_spread", {}).get("weight", 0.2)
        wd = cfg.get("supertrend", {}).get("weight", 0.1) if st_signal else 0
        
        raw = (adx_s * wa) + (slope_s * wb) + (di_s * wc) + (st_s * wd)
        score = min(10, round(raw, 2))
        
        return {"raw": raw, "value": score, "score": int(score), "desc": "Trend Strength"}

    def _compute_momentum_strength(self, indicators: Dict, cfg: Dict) -> Dict:
        """Momentum Strength: RSI + Slope + MACD"""
        rsi = self._get_val(indicators, "rsi")
        slope = self._get_val(indicators, "rsi_slope")
        macd = self._get_val(indicators, "macd_histogram")
        
        # Config Thresholds
        rsi_t = cfg.get("rsi_value", {}).get("thresholds", {})
        rsi_s = 8 if rsi >= rsi_t.get("strong", 60) else 5 if rsi >= rsi_t.get("neutral", 50) else 2
        
        slope_t = cfg.get("rsi_slope", {}).get("thresholds", {})
        slope_s = 8 if slope >= slope_t.get("strong", 1.0) else 4
        
        macd_t = cfg.get("macd_hist", {}).get("thresholds", {})
        macd_s = 8 if macd >= macd_t.get("strong", 0.5) else 5 if macd > 0 else 2
        
        # Weights
        w_rsi = cfg.get("rsi_value", {}).get("weight", 0.25)
        w_slope = cfg.get("rsi_slope", {}).get("weight", 0.25)
        w_macd = cfg.get("macd_hist", {}).get("weight", 0.30)
        
        raw = (rsi_s * w_rsi) + (slope_s * w_slope) + (macd_s * w_macd)
        score = min(10, round(raw, 2))
        
        return {"raw": raw, "value": score, "score": int(score), "desc": "Momentum Strength"}

    def _compute_volatility_quality(self, indicators: Dict, cfg: Dict) -> Dict:
        """Volatility Quality: ATR% + BB Width + Stability"""
        atr_pct = self._get_val(indicators, "atr_pct")
        bb_width = self._get_val(indicators, "bb_width")
        
        # Config Thresholds (Note: Higher Vol = Lower Score usually)
        atr_t = cfg.get("atr_pct", {}).get("thresholds", {})
        atr_s = 10 if atr_pct <= atr_t.get("low", 1.5) else 6 if atr_pct <= atr_t.get("high", 5.0) else 2
        
        bb_t = cfg.get("bb_width", {}).get("thresholds", {})
        bb_s = 10 if bb_width <= bb_t.get("tight", 0.01) else 6
        
        # Weights
        w_atr = cfg.get("atr_pct", {}).get("weight", 0.3)
        w_bb = cfg.get("bb_width", {}).get("weight", 0.25)
        
        raw = (atr_s * w_atr) + (bb_s * w_bb)
        # Add baseline 3.0 for missing metrics to prevent 0 score
        raw += 3.0 
        
        score = min(10, round(raw, 2))
        return {"raw": raw, "value": score, "score": int(score), "desc": "Vol Quality"}

    # ============================================================
    # 5. SIGNAL CONTEXT BUILDER (Consolidated)
    # ============================================================

    def get_signal_context(self) -> Dict[str, Any]:
        """
        Consolidated context for Signal Engine.
        Now includes 'computed' thresholds from the new engines above.
        """
        return {
            "horizon": self.horizon,
            "horizon_info": self.get_horizon_info(),
            "execution": self.get_execution_params(),
            "risk": self.get_risk_params(),
            "gates": self.get_gate_checks(),
            "scoring": self.get_section("scoring", merge_global=True),
            
            # Expanded Engine Configs
            "setup_classification": {
                "rules": self.get("global.calculation_engine.setup_classification.rules"),
                "pattern_priority": self.get_pattern_priority()
            },
            "setup_confidence": self.get("setup_confidence", {}),
            "enhancements": self.get("enhancements", {}),
            
            # Calculation Params
            "indicators": self.get_all_indicators(),
            "moving_averages": self.get_section("moving_averages", merge_global=True),
            
            # Volatility
            "volatility": {
                "bands": self.get_volatility_bands(),
                "scoring_thresholds": self.get("volatility.scoring_thresholds", {}),
                "quality_mins": self.get("volatility.quality_mins", {})
            },
            
            # Pattern Rules
            "patterns": {
                "entry_rules": self._get_all_pattern_entry_rules(),
                "invalidation_rules": self._get_all_pattern_invalidation_rules()
            },
            
            # Time & Confidence
            "confidence": self.get_section("confidence", merge_global=True),
            "time_estimation": self.get_time_estimation_params()
        }

    # ============================================================
    # HELPERS
    # ============================================================
    
    def _evaluate_condition_string(self, condition: str, indicators: Dict, fundamentals: Dict = None) -> bool:
        """
        Parses conditions like 'rsi >= 60', 'roe > 15', or 'is_consolidating == True'.
        Checks indicators first, then fundamentals.
        """
        try:
            parts = condition.split(" ")
            # Skip setup_type conditions (handled elsewhere)
            if "setup_type" in condition: return False

            # Handle boolean comparisons
            if len(parts) >= 3 and parts[1] == "==":
                metric = parts[0]
                target_str = parts[2].lower()
                
                val = self._get_val(indicators, metric, None)
                if val is None and fundamentals: val = self._get_val(fundamentals, metric, None)
                
                if val is None: return False
                
                # Boolean check
                if target_str in ["true", "false"]: return bool(val) == (target_str == "true")
            
            # Handle numeric comparisons
            if len(parts) >= 3:
                metric, op, target_str = parts[0], parts[1], parts[2]
                
                val = self._get_val(indicators, metric, None)
                if val is None and fundamentals:
                    val = self._get_val(fundamentals, metric, None)
                
                if val is None:
                    return False
                
                target = float(target_str)
                
                ops = {
                    ">": operator.gt, "<": operator.lt,
                    ">=": operator.ge, "<=": operator.le,
                    "==": operator.eq
                }
                
                return ops.get(op, lambda a, b: False)(val, target)
            
            return False
            
        except Exception as e:
            logger.debug(f"Condition '{condition}' evaluation failed: {e}")
            return False

    def _get_val(self, data: Dict, key: str, default: float = 0.0) -> float:
        """Helper to extract numeric value from indicator dict."""
        if not data: return default
        val = data.get(key)
        if isinstance(val, (int, float)): return float(val)
        if isinstance(val, dict):
            v = val.get("value") or val.get("raw") or val.get("score")
            return float(v) if v is not None else default
        return default

    # ... (Keep existing getters: get_execution_params, get_risk_params, etc.) ...

    # ============================================================
    # SPECIALIZED GETTERS (High-Level APIs)
    # ============================================================
    
    def get_indicator(self, name: str, fallback: Any = None) -> Any:
        """Get indicator config with legacy key mapping."""
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
    
    def get_volume_threshold(self, metric: str) -> float:
        """Get volume analysis thresholds (rvol_surge, rvol_drought)."""
        key_map = {
            "rvol_surge": "volume_analysis.rvol_surge_threshold",
            "rvol_drought": "volume_analysis.rvol_drought_threshold"
        }
        path = key_map.get(metric, f"volume_analysis.{metric}")
        return self.get(path, 1.0)
    
    def get_volatility_bands(self) -> Dict[str, float]:
        """Get volatility bands (min, ideal, max)."""
        return self.get("gates.volatility_bands_atr_pct", {"min": 1.0, "ideal": 2.5, "max": 12.0})
    
    def get_proximity_rejection(self) -> Dict[str, float]:
        """Get S/R proximity rejection multipliers."""
        return self.get("execution.proximity_rejection", {
            "resistance_mult": 1.005,
            "support_mult": 0.995
        })
    
    def get_volatility_guards(self) -> Dict[str, float]:
        """Get volatility guard thresholds."""
        return self.get("gates.volatility_guards", {
            "extreme_vol_buffer": 2.0,
            "min_quality_breakout": 3.0,
            "min_quality_normal": 4.0
        })
    
    def get_trend_threshold(self, metric: str) -> float:
        """Get trend slope thresholds."""
        return self.get(f"trend_thresholds.slope.{metric}", 10.0)
    
    def get_confidence_floor(self, setup_type: str = None) -> int:
        """Get confidence floor for setup type."""
        if setup_type:
            return self.get(f"confidence.base_floors.{setup_type}", 55)
        return self.get("confidence.floors.buy", 55)
    
    def get_pattern_entry_rule(self, pattern: str, rule: str) -> Any:
        """Get pattern-specific entry rules."""
        path = f"global.pattern_entry_rules.{pattern}.horizons.{self.horizon}.{rule}"
        return self.get(path)
    
    def get_pattern_invalidation(self, pattern: str) -> Dict:
        """Get pattern invalidation rules."""
        path = f"global.pattern_invalidation.{pattern}"
        rules = self.get(path, {})
        breakdown = rules.get("breakdown_threshold", {}).get(self.horizon, {})
        action = rules.get("action", {}).get(self.horizon, "EXIT_ON_CLOSE")
        return {
            "condition": breakdown.get("condition"),
            "duration_candles": breakdown.get("duration_candles"),
            "action": action,
            "or_condition": breakdown.get("or_condition")
        }
    
    def get_rr_regime_adjustment(self, adx_value: float) -> Dict[str, float]:
        """Get RR multipliers based on ADX regime."""
        adjustments = self.get("risk_management.rr_regime_adjustments", {})
        strong = adjustments.get("strong_trend", {})
        normal = adjustments.get("normal_trend", {})
        weak = adjustments.get("weak_trend", {})
        
        if adx_value >= strong.get("adx_min", 35):
            return {"t1_mult": strong.get("t1_mult", 2.0), "t2_mult": strong.get("t2_mult", 4.0)}
        elif adx_value >= normal.get("adx_min", 20):
            return {"t1_mult": normal.get("t1_mult", 1.5), "t2_mult": normal.get("t2_mult", 3.0)}
        else:
            return {"t1_mult": weak.get("t1_mult", 1.2), "t2_mult": weak.get("t2_mult", 2.5)}
    
    def get_setup_multiplier(self, setup_type: str) -> float:
        """Get position size multiplier for setup type."""
        multipliers = self.get("risk_management.setup_size_multipliers", {})
        return multipliers.get(setup_type, 1.0)
    
    def get_time_estimation_params(self) -> Dict[str, Any]:
        """Get time estimation parameters."""
        return {
            "candles_per_unit": self.get("time_estimation.candles_per_unit", 1),
            "base_friction": self.get("global.time_estimation.base_friction", 0.8),
            "velocity_factors": self.get("global.time_estimation.velocity_factors", {})
        }
    
    def get_technical_weight(self, indicator: str) -> float:
        """Get technical indicator weight with horizon overrides."""
        base_weight = self.get(f"global.technical_weights.{indicator}.weight", 1.0)
        override = self.get(f"technical_weight_overrides.{indicator}")
        if override is not None:
            return base_weight * override
        return base_weight
    
    def get_fundamental_mix(self) -> float:
        """Get fundamental vs technical mix for horizon."""
        return self.get("scoring.fundamental_weight", 0.3)

    # ============================================================
    # COMPOSITE GETTERS (Return Multiple Related Values)
    # ============================================================
    
    def get_execution_params(self) -> Dict[str, Any]:
        """Get all execution-related parameters."""
        return {
            "stop_loss_atr_mult": self.get("execution.stop_loss_atr_mult", 2.0),
            "target_atr_mult": self.get("execution.target_atr_mult", 3.0),
            "max_hold_candles": self.get("execution.max_hold_candles", 20),
            "risk_reward_min": self.get("execution.risk_reward_min", 2.0),
            "base_hold_days": self.get("execution.base_hold_days", 10),
            "proximity_rejection": self.get_proximity_rejection(),
            "min_profit_pct": self.get("execution.min_profit_pct", 0.5),
            "spread_adjustments": self.get("execution.spread_adjustments", {})
        }
    
    def get_risk_params(self) -> Dict[str, Any]:
        """Get all risk management parameters."""
        return {
            "max_position_pct": self.get("risk_management.max_position_pct", 0.02),
            "min_rr_ratio": self.get("risk_management.min_rr_ratio", 1.5),
            "horizon_t2_cap": self.get("risk_management.horizon_t2_cap", 0.10),
            "atr_sl_limits": self.get("risk_management.atr_sl_limits", {}),
            "setup_size_multipliers": self.get("risk_management.setup_size_multipliers", {}),
            "rr_regime_adjustments": self.get("risk_management.rr_regime_adjustments", {})
        }
    
    def get_gate_checks(self) -> Dict[str, Any]:
        """Get all entry gate requirements."""
        return {
            "min_trend_strength": self.get("gates.min_trend_strength", 3.0),
            "adx_min": self.get("gates.adx_min", 18),
            "volatility_quality_min": self.get("gates.volatility_quality_min", 4.0),
            "volatility_bands": self.get_volatility_bands(),
            "volatility_guards": self.get_volatility_guards(),
            "allowed_supertrend_counter": self.get("gates.allowed_supertrend_counter", False),
            "volatility_quality_mins": self.get("gates.volatility_quality_mins", {})
        }
    
    def get_all_indicators(self) -> Dict[str, Any]:
        """Get all indicator configurations for the horizon."""
        return self.get_section("indicators", merge_global=True)
    
    # Add these methods to your ConfigResolver class:

    # def evaluate_setup_rules(self, indicators: Dict, fundamentals: Dict) -> Tuple[str, int]:
    #     """
    #     Evaluates setup classification rules from config.
    #     Returns: (setup_type, priority_score)
    #     """
    #     rules = self.get("setup_classification.rules", {})
    #     candidates = []
        
    #     for setup_name, rule_cfg in rules.items():
    #         conditions = rule_cfg.get("conditions", [])
    #         priority = rule_cfg.get("priority", 50)
            
    #         # Evaluate ALL conditions
    #         all_met = True
    #         for condition in conditions:
    #             if not self._evaluate_condition_string(condition, indicators, fundamentals):
    #                 all_met = False
    #                 break
            
    #         if all_met:
    #             candidates.append((setup_name, priority))
        
    #     # Return highest priority match
    #     if candidates:
    #         candidates.sort(key=lambda x: x[1], reverse=True)
    #         return candidates[0]
        
    #     return ("GENERIC", 0)

    def classify_setup(indicators: Dict, fundamentals: Dict, horizon: str,) -> Tuple[str, int]:
        """
        Business logic for setup classification.
        Uses ConfigResolver for data, applies classification rules.
        """
        config = get_config(horizon)
        rules = config.get_setup_rules()
        candidates = []
        
        for setup_name, rule_cfg in rules.items():
            if config.evaluate_setup_condition(setup_name, indicators):
                priority = rule_cfg.get("priority", 50)
                candidates.append((setup_name, priority))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0]
        
        return ("GENERIC", 0)
    
    def detect_volume_signature(self, indicators: Dict) -> Dict:
        """Config-driven volume signature detection"""
        rvol = self._get_val(indicators, "rvol")
        
        surge_thresh = self.get("volume_analysis.rvol_surge_threshold", 3.0)
        drought_thresh = self.get("volume_analysis.rvol_drought_threshold", 0.7)
        
        if rvol >= surge_thresh:
            adjustment = self.get("global.calculation_engine.volume_signatures.surge.confidence_adjustment", 15)
            return {
                'type': 'surge',
                'adjustment': adjustment,
                'warning': f'Volume surge (RVOL={rvol:.2f})'
            }
        
        if rvol <= drought_thresh:
            adjustment = self.get("global.calculation_engine.volume_signatures.drought.confidence_adjustment", -25)
            return {
                'type': 'drought',
                'adjustment': adjustment,
                'warning': f'Volume drought (RVOL={rvol:.2f})'
            }
        
        return {'type': 'normal', 'adjustment': 0, 'warning': None}

    def detect_divergence(self, indicators: Dict) -> Dict:
        """Config-driven divergence detection"""
        rsi_slope = self._get_val(indicators, "rsi_slope")
        price = self._get_val(indicators, "price")
        prev_price = self._get_val(indicators, "prev_close", price)
        
        thresh = self.get(f"momentum_thresholds.rsi_slope.deceleration_ceiling", -0.05)
        
        # Bearish divergence: Price rising but RSI falling
        if price > prev_price and rsi_slope < thresh:
            penalty = self.get("global.calculation_engine.divergence_detection.confidence_penalties.bearish_divergence", 0.70)
            return {
                'divergence_type': 'bearish',
                'confidence_factor': penalty,
                'warning': f"Bearish Divergence: RSI_slope={rsi_slope:.2f}",
                'severity': 'moderate'
            }
        
        return {
            'divergence_type': 'none',
            'confidence_factor': 1.0,
            'warning': None,
            'severity': None
        }

    def get_rr_multipliers(self, adx: float) -> Dict[str, float]:
        """Returns R:R multipliers based on ADX regime"""
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

    def get_spread_adjustment(self, market_cap: float) -> float:
        """Returns spread % based on market cap"""
        brackets = self.get("global.calculation_engine.spread_adjustment.market_cap_brackets", {})
        
        large = brackets.get("large_cap", {})
        mid = brackets.get("mid_cap", {})
        small = brackets.get("small_cap", {})
        
        if market_cap >= large.get("min", 100000):
            return large.get("spread_pct", 0.001)
        elif market_cap >= mid.get("min", 10000):
            return mid.get("spread_pct", 0.002)
        else:
            return small.get("spread_pct", 0.005)

    def clamp_sl_distance(self, risk: float, price: float) -> float:
        """Enforces min/max SL distance limits"""
        limits = self.get("risk_management.atr_sl_limits", {})
        
        max_pct = limits.get("max_percent", 0.05)
        min_pct = limits.get("min_percent", 0.01)
        
        max_risk = price * max_pct
        min_risk = price * min_pct
        
        return max(min_risk, min(risk, max_risk))

    # def calculate_position_size(self, indicators: Dict, confidence: float, setup_type: str) -> float:
    #     """Config-driven position sizing"""
    #     base_risk = self.get("global.position_sizing.base_risk_pct", 0.01)
        
    #     # Setup multiplier
    #     multipliers = self.get("global.position_sizing.global_setup_multipliers", {})
    #     setup_mult = multipliers.get(setup_type, 1.0)
        
    #     # Volatility adjustment
    #     vol_qual = self._get_val(indicators, "volatility_quality", 5.0)
    #     vol_adjustments = self.get("global.position_sizing.volatility_adjustments", {})
        
    #     vol_mult = 1.0
    #     for regime, cfg in vol_adjustments.items():
    #         if "min" in cfg and vol_qual >= cfg["min"]:
    #             vol_mult = cfg.get("multiplier", 1.0)
    #         elif "max" in cfg and vol_qual <= cfg["max"]:
    #             vol_mult = cfg.get("multiplier", 1.0)
        
    #     # Calculate
    #     conf_factor = confidence / 100.0
    #     position = base_risk * conf_factor * setup_mult * vol_mult
        
    #     # Cap at horizon max
    #     max_pos = self.get("risk_management.max_position_pct", 0.02)
    #     return round(min(position, max_pos), 4)
    def get_position_sizing_config(self) -> Dict:
        """Get position sizing parameters."""
        return {
            "base_risk_pct": self.get("global.position_sizing.base_risk_pct", 0.01),
            "setup_multipliers": self.get("global.position_sizing.global_setup_multipliers", {}),
            "volatility_adjustments": self.get("global.position_sizing.volatility_adjustments", {}),
            "max_position_pct": self.get("risk_management.max_position_pct", 0.02)
        }

    def should_trade_volatility(self, indicators: Dict, setup_type: str) -> Tuple[bool, str]:
        """Validates volatility regime for trading"""
        vol_qual = self._get_val(indicators, "volatility_quality")
        atr_pct = self._get_val(indicators, "atr_pct")
        
        if vol_qual is None or atr_pct is None:
            return True, "Missing vol data, proceed cautiously"
        
        # Extreme volatility check
        guards = self.get("gates.volatility_guards", {})
        extreme_buffer = guards.get("extreme_vol_buffer", 2.0)
        bands = self.get("gates.volatility_bands_atr_pct", {})
        
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
        # CONTEXT BUILDERS (For Signal Engine)
        # ============================================================

    def get_signal_context(self) -> Dict[str, Any]:
        """
        Returns a consolidated configuration context specifically for the Signal Engine.
        This provides single-point access for all horizon-aware parameters needed
        during signal generation, replacing multiple config lookups.
        
        Returns:
            Dict containing execution, risk, gates, scoring, thresholds, confidence,
            indicators, patterns, and time estimation configs.
            
        Example:
            >>> config = ConfigResolver("intraday")
            >>> ctx = config.get_signal_context()
            >>> sl_mult = ctx['execution']['stop_loss_atr_mult']
            >>> surge = ctx['thresholds']['volume']['surge']
        """
        return {
            "horizon": self.horizon,
            "horizon_info": self.get_horizon_info(),
            
            # --- Primary Sections ---
            "execution": self.get_execution_params(),
            "risk": self.get_risk_params(),
            "gates": self.get_gate_checks(),
            
            # --- Engine-Specific Sections ---
            "scoring": self.get_section("scoring", merge_global=True),
            "setup_classification": self.get("setup_classification", {}),
            "setup_confidence": self.get("setup_confidence", {}),
            "enhancements": self.get("enhancements", {}),
            
            # --- Calculation Configs ---
            "indicators": self.get_all_indicators(),
            "moving_averages": self.get_section("moving_averages", merge_global=True),
            
            # --- Volatility Context ---
            "volatility": {
                "bands": self.get_volatility_bands(),
                "scoring_thresholds": {
                    "atr_pct": self.get("volatility.scoring_thresholds.atr_pct", {}),
                    "bb_width": self.get("volatility.scoring_thresholds.bb_width", {})
                },
                "quality_mins": self.get("volatility.quality_mins", {})
            },
            
            # --- Pattern Rules (Consolidated) ---
            "patterns": {
                "entry_rules": self._get_all_pattern_entry_rules(),
                "invalidation_rules": self._get_all_pattern_invalidation_rules()
            },
            
            # --- Composite Weights ---
            "composite_weights": {
                "trend_strength": self.get("global.calculation_engine.composite_weights.trend_strength", {}),
                "momentum_strength": self.get("global.calculation_engine.composite_weights.momentum_strength", {}),
                "volatility_quality": self.get("global.calculation_engine.composite_weights.volatility_quality", {})
            },
            
            # --- Thresholds ---
            "thresholds": {
                "volume": {
                    "surge": self.get_volume_threshold("rvol_surge"),
                    "drought": self.get_volume_threshold("rvol_drought")
                },
                "momentum": self.get_section("momentum_thresholds", merge_global=True),
                "trend": self.get_section("trend_thresholds", merge_global=True)
            },
            
            # --- Confidence Logic ---
            "confidence": {
                "floors": self.get("confidence.floors", {}),
                "base_floors": self.get("confidence.base_floors", {}),
                "adx_based_floors": self.get("confidence.adx_based_floors", {}),
                "horizon_discount": self.get("confidence.horizon_discount", 0),
                "volume_penalty": self.get("confidence.volume_penalty", {}),
                "setup_type_overrides": self.get("confidence.setup_type_overrides", {})
            },
            
            # --- Time Estimation ---
            "time_estimation": self.get_time_estimation_params()
        }
    
    # ============================================================
    # Resolver METHODS (Private)
    # ============================================================
    
    def _get_all_pattern_entry_rules(self) -> Dict[str, Dict]:
        """Get all pattern entry rules for this horizon."""
        patterns = [
            "bollinger_squeeze", "darvas_box", "cup_handle",
            "minervini_stage2", "flag_pennant", "three_line_strike",
            "ichimoku_signals", "golden_cross", "double_top_bottom"
        ]
        
        rules = {}
        for pattern in patterns:
            # We construct path manually to check existence safely
            path = f"global.pattern_entry_rules.{pattern}.horizons.{self.horizon}"
            rule = self.get(path)
            if rule:
                rules[pattern] = rule
        return rules

    def _get_all_pattern_invalidation_rules(self) -> Dict[str, Dict]:
        """Get all pattern invalidation rules for this horizon."""
        patterns = [
            "bollinger_squeeze", "darvas_box", "cup_handle",
            "minervini_stage2", "flag_pennant", "three_line_strike",
            "ichimoku_signals", "golden_cross", "double_top_bottom"
        ]
        
        rules = {}
        for pattern in patterns:
            # Re-use existing method for consistency
            rule = self.get_pattern_invalidation(pattern)
            # Only include if it has valid logic (not just empty defaults)
            if rule and (rule.get("condition") or rule.get("action")):
                rules[pattern] = rule
        return rules
    
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
    
    def clear_cache(self):
        """Clear the resolution cache."""
        self._cache.clear()
    
    def get_horizon_info(self) -> Dict[str, str]:
        """Get metadata about current horizon."""
        return {
            "horizon": self.horizon,
            "timeframe": self.get("timeframe", "1d"),
            "description": self.get("description", ""),
            "ma_type": self.get("moving_averages.type", "EMA")
        }


# ============================================================
# CONVENIENCE FUNCTIONS (Module-level API)
# ============================================================

_cache: Dict[str, ConfigResolver] = {}

def get_config(horizon: str = "short_term") -> ConfigResolver:
    """Get or create a ConfigResolver instance (cached)."""
    if horizon not in _cache:
        _cache[horizon] = ConfigResolver(horizon)
    return _cache[horizon]

def clear_config_cache():
    """Clear all cached ConfigResolver instances."""
    _cache.clear()