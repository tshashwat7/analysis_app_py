# config/query_optimized_extractor_v4.py
"""
Query-Optimized Config Extractor v3.0
=====================================
Designed for resolver to query configs efficiently.

Key Features:
✅ Fast gate resolution (merged global + horizon + setup)
✅ Pattern metadata access (physics, entry rules, invalidation)
✅ Strategy validation (market cap, fit indicators)
✅ Threshold merging with clear hierarchy
✅ Caching for repeated queries

Author: Quantitative Trading System
Version: 3.0
"""

import hashlib
import re
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass
from functools import lru_cache
from collections import OrderedDict  # ✅ P1-4 FIX
import logging
import json
import config.gate_evaluator as _gate_evaluator

from config.fundamental_score_config import compute_fundamental_score
from config.technical_score_config import calculate_dynamic_score, compute_technical_score
from config.config_extractor import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class ResolvedGate:
    """Resolved gate with full context."""
    metric: str
    threshold: Dict[str, float]  # {"min": X, "max": Y}
    source: str  # "global", "horizon", "setup_override"
    required: bool


@dataclass
class PatternContext:
    """Complete pattern context for a horizon."""
    physics: Dict[str, Any]
    typical_duration: Dict[str, Any]
    entry_rules: Dict[str, Any]
    invalidation: Dict[str, Any]
    scoring_thresholds: Dict[str, float]


class QueryOptimizedExtractor:
    """
    Extractor optimized for resolver queries.
    
    Design Philosophy:
    - Resolver asks questions, extractor answers
    - All config hierarchy resolved here (not in resolver)
    - Caching for repeated queries
    - Clear precedence: Setup > Horizon > Global
    """
    
    def __init__(self, master_config: Dict, horizon: str, logger=None, base_extractor=None):
        from config.config_extractor import ConfigExtractor
        
        # Use provided extractor or create a new one
        self.base_extractor = base_extractor or ConfigExtractor(master_config, horizon, logger)
        self.horizon = horizon
        self.logger = logger or logging.getLogger(__name__)
        
        # Query cache
        self._config_version = self._compute_config_hash(master_config)
        self._gate_cache: OrderedDict[str, Tuple[str, ResolvedGate]] = OrderedDict()  # ✅ P1-4 FIX
        self._pattern_cache: OrderedDict[str, Tuple[str, PatternContext]] = OrderedDict()  # ✅ P1-4 FIX
        
    def get(self, key: str, default: Any = None) -> Any:
        """Proxy to base_extractor.get() to protect layer boundary."""
        return self.base_extractor.get(key, default)

    def get_strict(self, key: str) -> Any:
        """Proxy to base_extractor.get_strict() to protect layer boundary."""
        return self.base_extractor.get_strict(key)
        

    def _compute_config_hash(self, config: Dict) -> str:
        """
        Compute hash of config for version tracking.
        
        Only hashes the parts that affect cached data:
        - entry_gates (for gate_cache)
        - pattern metadata (for pattern_cache)
        
        This is more efficient than hashing entire config.
        """
        # Extract only the config sections we cache
        cache_relevant = {
            "horizon": self.horizon,  # ✅ P0-2 FIX: Include horizon in hash
            "horizons": config.get("horizons", {}),
            "global_gates": config.get("global", {}).get("entry_gates", {}),
            # Add pattern-related sections if they exist
        }
        
        # ✅ FIX: Include confidence config in hash input to prevent stale cache (Issue B)
        if hasattr(self, 'base_extractor') and getattr(self.base_extractor, 'has_confidence_config', False):
            cache_relevant["confidence"] = self.base_extractor.confidence_config
            
        config_str = json.dumps(cache_relevant, sort_keys=True)
        # ✅ P0-2 FIX: Use SHA256 and 16 characters for version tracking
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    # ========================================================================
    # ✅ NEW: CONFIDENCE CONFIG QUERIES
    # ========================================================================

    def get_confidence_range(self) -> Dict[str, Any]:
        """
        Get confidence range (min, max, default clamp).
        
        Returns:
            {
                "absolute_min": 0,
                "absolute_max": 100,
                "default_clamp": [30, 95]
            }
        """

        return self.base_extractor.get("confidence_range", {
            "absolute_min": 0,
            "absolute_max": 100,
            "default_clamp": [30, 95]
        })

    def get_confidence_clamp(self) -> List[int]:
        """
        Get horizon-specific confidence clamp [min, max].
        
        Returns:
            [min_confidence, max_confidence] for this horizon
        
        Raises:
            ConfigurationError: if horizon_confidence_clamp is missing
        """
        horizon_clamp = self.base_extractor.get_strict("horizon_confidence_clamp")
        if isinstance(horizon_clamp, list) and len(horizon_clamp) == 2:
            return horizon_clamp
        
        # Structurally present but malformed — still fail loudly
        from config.config_extractor import ConfigurationError
        raise ConfigurationError(
            f"CRITICAL: horizon_confidence_clamp has invalid format: {horizon_clamp}"
        )


    def get_setup_baseline_floor(self, setup_name: str) -> Optional[float]:
        """
        Get baseline confidence floor for a setup.
        
        Hierarchy:
        1. Horizon override (setup_floor_overrides)
        2. Global baseline (setup_baseline_floors)
        3. Default (40)
        """
        # Check horizon override first
        horizon_overrides = self.base_extractor.get("horizon_setup_floor_overrides", {})
        if setup_name in horizon_overrides:
            override = horizon_overrides[setup_name]
            if override is None:
                # Explicitly blocked for this horizon
                return None
            return override
        
        # ✅ FIX: Check global baseline floors (hierarchy step 2)
        global_floors = self.base_extractor.get("setup_baseline_floors", {})
        if setup_name in global_floors:
            return global_floors[setup_name]

        # ✅ Phase 3 P1-1 FIX: Fail-Fast for unknown setup types.
        # Fallback to 40 is dangerous in production.
        raise ConfigurationError(
            f"ARCHITECTURAL VIOLATION: Horizon '{self.horizon}' or Global config "
            f"is missing baseline confidence floor for setup '{setup_name}'."
        )


    def get_base_confidence_adjustment(self) -> float:
        """
        Get horizon-specific base confidence adjustment (discount/premium).
        
        Returns:
            Adjustment value (e.g., -10 for intraday, 0 for long_term)
        """
        return self.base_extractor.get("horizon_base_confidence_adjustment", 0)

    def get_volume_modifiers(self) -> Dict[str, Dict]:
        """
        Get volume-based confidence modifiers (surge, drought, climax).
        
        Returns:
            {
                "surge_bonus": {
                    "gates": {"rvol": {"min": 3.0}},
                    "confidence_boost": 10,
                    "exclude_setups": [...]
                },
                "drought_penalty": {...},
                "climax_warning": {...}
            }
        
        Raises:
            ConfigurationError: if volume_modifiers section is missing
        """
        return self.base_extractor.get_strict("volume_modifiers")

    def get_universal_adjustments(self) -> Dict[str, Dict]:
        """
        Get universal confidence adjustments (divergence, trend strength).
        
        Returns:
            {
                "divergence_penalties": {
                    "severe": {...},
                    "moderate": {...},
                    "minor": {...}
                },
                "trend_strength_bands": {
                    "explosive": {...},
                    "strong": {...},
                    ...
                }
            }
        
        Raises:
            ConfigurationError: if universal_adjustments section is missing
        """
        return self.base_extractor.get_strict("universal_adjustments")

    def get_conditional_adjustments(self) -> Dict[str, Dict]:
        """
        Get horizon-specific conditional adjustments (penalties/bonuses).
        
        Returns:
            {
                "penalties": {
                    "weak_intraday_trend": {...},
                    "low_liquidity": {...},
                    ...
                },
                "bonuses": {
                    "clean_breakout": {...},
                    "explosive_trend": {...},
                    ...
                }
            }
        """
        horizon_adjustments = self.base_extractor.get("horizon_conditional_adjustments", {})
        return {
            "penalties": horizon_adjustments.get("penalties", {}),
            "bonuses": horizon_adjustments.get("bonuses", {})
        }

    def get_adx_confidence_bands(self) -> Dict[str, Dict]:
        """
        Get horizon-specific ADX confidence bands.
        
        Returns:
            {
                "explosive": {"gates": {"adx": {"min": 35}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 28}}, "confidence_boost": 15},
                "moderate": {"gates": {"adx": {"min": 20}}, "confidence_boost": 10},
            }
        """
        bands = self.base_extractor.get("horizon_adx_confidence_bands", {})
        # Filter out any entries with penalties (they shouldn't be here anyway)
        return {
            name: config for name, config in bands.items()
            if "confidence_boost" in config
        }
    
    def get_adx_confidence_penalties(self) -> Dict[str, Dict]:
        """
        Get horizon-specific ADX confidence PENALTIES (negative adjustments).
        
        Returns:
            {
                "weak": {"gates": {...}, "confidence_penalty": -15}
            }
        """
        return self.base_extractor.get("horizon_adx_confidence_penalties", {})

    def get_divergence_physics(self) -> Dict[str, Any]:
        """
        ✅ NEW: Get unified detection math.
        Returns: {"lookback": 10, "slope_diff_min": -0.05, ...}
        """
        return self.base_extractor.get("divergence_physics", {
            "lookback": 10,
            "slope_diff_min": -0.05,
            "bullish_slope_min": 0.05
        })

    def get_min_tradeable_confidence(self) -> float:
        """
        Horizon-level confidence floor to even consider trades.
        
        Raises:
            ConfigurationError: if min_tradeable_confidence is missing
        """
        cfg = self.base_extractor.get_strict("min_tradeable_confidence")
        # support simple formats: {"min": 55} or just 55
        if isinstance(cfg, dict):
            return float(cfg.get("min", 0))
        try:
            return float(cfg)
        except (TypeError, ValueError):
            from config.config_extractor import ConfigurationError
            raise ConfigurationError(
                f"CRITICAL: min_tradeable_confidence has invalid format: {cfg}"
            )


    def get_high_confidence_override(self) -> Dict[str, Any]:
        """
        Get high-confidence override config (e.g., behavior when confidence >= X).
        
        Raises:
            ConfigurationError: if high_confidence_override is missing
        """
        return self.base_extractor.get_strict("high_confidence_override")
    # ========================================================================
    # CONFIDENCE GATE EVALUATION ENGINE
    # (Pure logic lives in config.gate_evaluator — these are thin wrappers)
    # ========================================================================

    def evaluate_confidence_gates(
        self,
        gates: Dict[str, Any],
        data: Dict[str, Any],
        empty_gates_pass: bool = True,
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate if market data meets gate conditions.

        Delegates to :func:`config.gate_evaluator.evaluate_gates`.
        See that function for full argument and return-value documentation.
        """
        return _gate_evaluator.evaluate_gates(gates, data, empty_gates_pass)

    def evaluate_invalidation_gates(
        self,
        gates: Dict[str, Any],
        data: Dict[str, Any],
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        Evaluate invalidation / breakdown gate conditions.

        Delegates to :func:`config.gate_evaluator.evaluate_invalidation_gates`.
        See that function for full argument and return-value documentation.
        """
        return _gate_evaluator.evaluate_invalidation_gates(gates, data)

    def evaluate_confidence_modifier(
        self,
        modifier_config: Dict[str, Any],
        data: Dict[str, Any],
        setup_type: Optional[str] = None
    ) -> Tuple[bool, Optional[float], str, bool]:
        """
        Evaluate a confidence modifier (penalty/bonus) and return adjustment.
        
        ✅ ENHANCED: Now strictly supports structured gates.
        
        Handles:
        - Gate evaluation (from confidence_config.py)
        - Setup-specific application/exclusion
        - Confidence adjustments (boost/penalty/multiplier/amount)
        
        Args:
            modifier_config: Modifier config
                Example (gates): {
                    "gates": {"adx": {"min": 20}},
                    "confidence_boost": 10,
                    "reason": "Strong trend"
                }
                Example (penalty): {
                    "gates": {"rsi": {"max": 40}},
                    "confidence_penalty": -8,
                    "reason": "Weak momentum"
                }
            data: Market data dictionary
            setup_type: Current setup type (for filtering)
        
        Returns:
            Tuple of (applies: bool, adjustment: float, reason: str, block_entry: bool)
        """
        # 1. Setup Filtering
        apply_to = modifier_config.get("apply_to_setups")
        exclude_from = modifier_config.get("exclude_setups")
        
        if apply_to and setup_type not in apply_to:
            return False, None, "Setup not in apply_to list", False
        
        if exclude_from and setup_type in exclude_from:
            return False, None, "Setup in exclude list", False
        
        # 2. Evaluation Logic (gates)
        gates = modifier_config.get("gates")
        
        if gates:
            # Path A: Structured dictionary evaluation
            applies, failures = self.evaluate_confidence_gates(gates, data)
            failure_reason = f"Gates failed: {'; '.join(failures)}" if not applies else ""
        else:
            # No evaluation criteria
            return False, None, "No evaluation criteria (gates) found", False
        
        if not applies:
            return False, None, failure_reason, False
        
        # 3. Extract Adjustment Value — use explicit key presence check so that
        #    confidence_boost: 0 (an intentional no-op) is not treated as falsy.
        adjustment = None
        for key in ("confidence_multiplier", "confidence_boost", "confidence_penalty"):
            if key in modifier_config:
                adjustment = modifier_config[key]
                if key == "confidence_penalty" and adjustment is not None and float(adjustment) > 0:
                    raise ConfigurationError(
                        f"CRITICAL CONFIG ERROR: confidence_penalty={adjustment} is positive. "
                        f"Penalties must be negative. Check: {modifier_config.get('reason', 'unknown')}"
                    )
                break
        
        block_entry = modifier_config.get("block_entry", False)
        reason = modifier_config.get("reason", "Criteria met")
        
        return True, adjustment, reason, block_entry

    def evaluate_all_confidence_modifiers(
        self,
        data: Dict[str, Any],
        setup_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate ALL confidence modifiers for current horizon and setup.
        
        This is the main method the resolver should call. ADX bands removed from universal modifiers.
        ADX is ONLY applied in calculate_dynamic_confidence_floor.
        
        Returns:
            {
                "volume_modifiers": {
                    "surge_bonus": {"applies": True, "adjustment": 10, "reason": "..."},
                    "drought_penalty": {"applies": False, ...}
                },
                "divergence_penalties": {
                    "severe": {"applies": False, ...}
                },
                "trend_strength_bands": {
                    "strong": {"applies": True, "adjustment": 15, "reason": "..."}
                },
                "conditional_adjustments": {
                    "penalties": {...},
                    "bonuses": {...}
                },
                "adx_bands": {
                    "explosive": {"applies": True, "adjustment": 20, "reason": "..."}
                }
            }
        """
        results = {}
        
        # 0. Prepare flat namespace for nested indicator/fundamental checks
        # This handles fund_valuation_bucket, technicalScore, etc.
        eval_data = self._prepare_evaluation_namespace(data)
        
        # 1. Volume Modifiers
        volume_mods = self.get_volume_modifiers()
        results["volume_modifiers"] = {}
        
        for mod_name, mod_config in volume_mods.items():
            applies, adjustment, reason, block_entry = self.evaluate_confidence_modifier(
                mod_config, eval_data, setup_type
            )
            results["volume_modifiers"][mod_name] = {
                "applies": applies,
                "adjustment": adjustment,
                "reason": reason,
                "block_entry": block_entry
            }
        
        # 2. Universal Adjustments - Divergence
        universal = self.get_universal_adjustments()
        divergence = universal.get("divergence_penalties", {})
        results["divergence_penalties"] = {}
        
        # ✅ PATCH A: Honour the pre-computed divergence_type from detect_divergence().
        # If the upstream resolver already determined divergence_type='none', skip all
        # penalty gate evaluation — prevents rsislope gate from double-counting momentum
        # deceleration as a structural divergence.
        precomputed_divergence_type = eval_data.get("divergence_type", None)
        skip_divergence_eval = (precomputed_divergence_type == "none")

        for severity, div_config in divergence.items():
            if skip_divergence_eval:
                results["divergence_penalties"][severity] = {
                    "applies": False,
                    "adjustment": None,
                    "reason": "Divergence type pre-evaluated as none by detect_divergence()",
                    "block_entry": False
                }
                continue
            applies, adjustment, reason, block_entry = self.evaluate_confidence_modifier(
                div_config, eval_data, setup_type
            )
            results["divergence_penalties"][severity] = {
                "applies": applies,
                "adjustment": adjustment,
                "reason": reason,
                "block_entry": block_entry
            }
        
        # 3. Universal Adjustments - Trend Strength
        trend_bands = universal.get("trend_strength_bands", {})
        results["trend_strength_bands"] = {}

        # Sort bands by threshold (descending) for priority
        band_list = []
        for band_name, band_config in trend_bands.items():
            gates = band_config.get("gates", {})
            # Extract threshold for sorting (assuming trendStrength metric)
            trend_threshold = gates.get("trendStrength", {}).get("min", 0)
            band_list.append((trend_threshold, band_name, band_config))

        # Sort descending - highest threshold first
        band_list.sort(reverse=True, key=lambda x: x[0])

        # Apply ONLY the first matching band
        band_applied = False
        for threshold, band_name, band_config in band_list:
            if band_applied:
                # Mark remaining bands as not applied
                results["trend_strength_bands"][band_name] = {
                    "applies": False,
                    "adjustment": None,
                    "reason": "Higher priority band already applied"
                }
                continue
            
            applies, adjustment, reason, block_entry = self.evaluate_confidence_modifier(
                band_config, eval_data, setup_type
            )
            results["trend_strength_bands"][band_name] = {
                "applies": applies,
                "adjustment": adjustment,
                "reason": reason,
                "block_entry": block_entry
            }
            
            if applies:
                band_applied = True
                self.logger.debug(
                    f"Trend band '{band_name}' applied: {adjustment:+.1f}"
                )
        
        # 4. Conditional Adjustments (Horizon-Specific)
        conditional = self.get_conditional_adjustments()
        results["conditional_adjustments"] = {
            "penalties": {},
            "bonuses": {}
        }
        

        # Penalties
        for penalty_name, penalty_config in conditional.get("penalties", {}).items():
            applies, adjustment, reason, block_entry = self.evaluate_confidence_modifier(
                penalty_config, eval_data, setup_type
            )
            results["conditional_adjustments"]["penalties"][penalty_name] = {
                "applies": applies,
                "adjustment": adjustment,
                "reason": reason,
                "block_entry": block_entry
            }
        
        # Bonuses
        for bonus_name, bonus_config in conditional.get("bonuses", {}).items():
            applies, adjustment, reason, block_entry = self.evaluate_confidence_modifier(
                bonus_config, eval_data, setup_type
            )
            results["conditional_adjustments"]["bonuses"][bonus_name] = {
                "applies": applies,
                "adjustment": adjustment,
                "reason": reason,
                "block_entry": block_entry
            }
        
        # Combine block_entry flags (OR logic)
        final_block = False
        for cat in ["volume_modifiers", "divergence_penalties", "trend_strength_bands"]:
            for res in results.get(cat, {}).values():
                if res.get("applies") and res.get("block_entry"):
                    final_block = True
                    break
            if final_block: break
        
        if not final_block:
            for cat in ["penalties", "bonuses"]:
                for res in results.get("conditional_adjustments", {}).get(cat, {}).values():
                    if res.get("applies") and res.get("block_entry"):
                        final_block = True
                        break
                if final_block: break
                    
        results["block_entry"] = final_block
        return results

    def calculate_total_confidence_adjustment(
        self,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate total confidence adjustment from evaluation results.
        Handle ADX boosts and penalties separately
        
        Handles:
        - Additive adjustments (penalties/bonuses)
        - Multiplicative adjustments (divergence)
        - Adjustment breakdown for debugging
        
        Args:
            evaluation_results: Output from evaluate_all_confidence_modifiers()
        
        Returns:
            Dict with keys:
                - "adjustment" (float): Total confidence delta (already scaled)
                - "breakdown" (List[str]): Human-readable audit trail
                - "multiplier" (float): Divergence multiplier applied (1.0 if none)
                - "block_entry" (bool): Whether any applied modifier requested an entry block

        Example:
            >>> results = extractor.evaluate_all_confidence_modifiers(data, "MOMENTUM_BREAKOUT")
            >>> adj_data = extractor.calculate_total_confidence_adjustment(results)
            >>> print(adj_data["adjustment"])   # 17.5
            >>> print(adj_data["multiplier"])   # 0.7 (moderate divergence)
            >>> print(adj_data["block_entry"])  # True
        """
        total_additive = 0.0
        multipliers = []
        breakdown = []
        block_entry = evaluation_results.get("block_entry", False)
        
        # Process all modifier categories
        for category, modifiers in evaluation_results.items():
            # ❌ REMOVED: Dead ADX bands logic
            # if category == "adx_bands":  # This never executes!
            #     ...
            
            # ❌ REMOVED: Dead ADX penalties logic  
            # elif category == "adx_penalties":  # This never executes!
            #     ...
                        
            if category in ["volume_modifiers", "trend_strength_bands"]:
                # These are additive
                for mod_name, result in modifiers.items():
                    if result["applies"] and result["adjustment"] is not None:
                        total_additive += result["adjustment"]
                        self.logger.info(f"[CONF_DIAG] Added {category}.{mod_name}: {result['adjustment']}, total_additive now: {total_additive}")
                        breakdown.append(
                            f"{category}.{mod_name}: {result['adjustment']:+.1f} ({result['reason']})"
                        )
            
            elif category == "divergence_penalties":
                # These are multiplicative. ONLY APPLY THE FIRST MATCHING BAND.
                # ✅ FIX: Priority explicitly fixed to [severe, moderate, minor] (Issue C)
                div_applied = False
                for severity in ["severe", "moderate", "minor"]:
                    if severity in modifiers:
                        result = modifiers[severity]
                        if not div_applied and result["applies"] and result["adjustment"] is not None:
                            multipliers.append(result["adjustment"])
                            self.logger.info(f"[CONF_DIAG] Applied divergence.{severity}: ×{result['adjustment']}")
                            breakdown.append(
                                f"divergence.{severity}: ×{result['adjustment']} ({result['reason']})"
                            )
                            div_applied = True
                        elif div_applied:
                            # Log skipped bands for visibility
                            pass
            
            elif category == "conditional_adjustments":
                # Penalties and bonuses are additive
                for penalty_name, result in modifiers.get("penalties", {}).items():
                    if result["applies"] and result["adjustment"] is not None:
                        total_additive += result["adjustment"]
                        self.logger.info(f"[CONF_DIAG] Added penalty.{penalty_name}: {result['adjustment']}, total_additive now: {total_additive}")
                        breakdown.append(
                            f"penalty.{penalty_name}: {result['adjustment']:+.1f} ({result['reason']})"
                        )
                
                for bonus_name, result in modifiers.get("bonuses", {}).items():
                    if result["applies"] and result["adjustment"] is not None:
                        total_additive += result["adjustment"]
                        self.logger.info(f"[CONF_DIAG] Added bonus.{bonus_name}: {result['adjustment']}, total_additive now: {total_additive}")
                        breakdown.append(
                            f"bonus.{bonus_name}: {result['adjustment']:+.1f} ({result['reason']})"
                        )

        # Apply multipliers (if any) ONLY to structural adjustments (non-conditional bonuses)
        if multipliers:
            # Use most severe multiplier (lowest value)
            final_multiplier = min(multipliers)
            
            # Recalculate to only scale volume and trend
            scaled_structural = 0.0
            unscaled_conditional = 0.0
            
            for category, modifiers in evaluation_results.items():
                if category in ["volume_modifiers", "trend_strength_bands"]:
                    for mod_name, result in modifiers.items():
                        if result["applies"] and result["adjustment"] is not None:
                            scaled_structural += result["adjustment"]
                            self.logger.info(f"[CONF_DIAG] Recalc Added {category}.{mod_name}: {result['adjustment']}, scaled_structural now: {scaled_structural}")
                elif category == "conditional_adjustments":
                    for name, result in modifiers.get("penalties", {}).items():
                        if result["applies"] and result["adjustment"] is not None:
                            unscaled_conditional += result["adjustment"]
                            self.logger.info(f"[CONF_DIAG] Recalc Added penalty.{name}: {result['adjustment']}, unscaled_conditional now: {unscaled_conditional}")
                    for name, result in modifiers.get("bonuses", {}).items():
                        if result["applies"] and result["adjustment"] is not None:
                            unscaled_conditional += result["adjustment"]
                            self.logger.info(f"[CONF_DIAG] Recalc Added bonus.{name}: {result['adjustment']}, unscaled_conditional now: {unscaled_conditional}")
            
            # Note: if scaled_structural = 0, this results in final_multiplier doing nothing, 
            # and only conditional bonuses/penalties being applied. This is expected behavior (Issue D).
            final_scaled_total = (scaled_structural * final_multiplier) + unscaled_conditional
            breakdown.append(f"Final multiplier (Structural only): ×{final_multiplier}")
            
            return {
                "adjustment": final_scaled_total,
                "breakdown": breakdown,
                "multiplier": final_multiplier,
                "block_entry": block_entry
            }
        else:
            self.logger.info(f"[CONF_DIAG] Returning total_additive: {total_additive}")
            return {
                "adjustment": total_additive,
                "breakdown": breakdown,
                "multiplier": 1.0,
                "block_entry": block_entry
            }
   
    # ========================================================================
    # GATE QUERIES (Most Important for Resolver)
    # ========================================================================

    def _merge_gates_with_priority(self, *gate_layers) -> Dict:
        """Merge gates with correct priority (last wins)."""
        result = {}
        for layer in gate_layers:
            for gate_name, threshold in layer.items():
                if threshold is None:
                    # Explicit None = disable gate
                    if gate_name in result:
                        del result[gate_name]
                else:
                    result[gate_name] = self.normalize_threshold(threshold)
        return result
    
    def get_resolved_gates(
        self,
        phase: str,
        setup_type: Optional[str] = None
    ) -> Dict[str, 'ResolvedGate']:
        """
        Get resolved gates from NEW architecture.

        Merge Priority (later wins):
        1. Global gates       (master_config.global.entry_gates)
        2. Horizon gates      (master_config.horizons.X.entry_gates)
        3. Setup gates        (setup_pattern_matrix base + horizon override,
                            already merged per-metric by get_setup_context_requirements)

        Note: Setup base and setup horizon override are merged upstream in
        get_setup_context_requirements(), so they arrive here as a single
        combined layer. The source label "setup_merged" reflects this.
        """
        # ✅ P2-3 FIX: Use null-byte separator to avoid key collision
        cache_key = f"{phase}\x00{setup_type or 'none'}\x00{self.horizon}"
        
        # ✅ Check cache with version validation
        if cache_key in self._gate_cache:
            cached_version, cached_data = self._gate_cache[cache_key]
            if cached_version == self._config_version:
                self.logger.debug(f"Cache HIT (v{cached_version}): {cache_key}")
                # Move to end (most recently used)
                self._gate_cache.move_to_end(cache_key)
                return cached_data
            else:
                # Config changed - invalidate this entry
                self.logger.debug(
                    f"Cache STALE (v{cached_version} != v{self._config_version}): {cache_key}"
                )
                del self._gate_cache[cache_key]
        
        # ✅ STEP 1: Get global gates (master_config)
        global_section = self.base_extractor.get(f"{phase}_gates", {})
        global_gates = global_section if isinstance(global_section, dict) else {}
        
        # ✅ STEP 2: Get horizon gates (master_config)
        horizon_section = self.base_extractor.get(f"horizon_{phase}_gates", {})
        horizon_gates = horizon_section if isinstance(horizon_section, dict) else {}
        
        # STEP 3 & 4 COMBINED: get_setup_context_requirements already merges base + horizon override
        setup_gates = {}
        if setup_type:
            merged_context = self.get_setup_context_requirements(setup_type)
            if phase == "structural":
                setup_gates = merged_context.get("technical", {})
            elif phase == "opportunity":
                # ✅ FIX R2-1: Force empty setup opportunity gates since they shouldn't exist
                setup_gates = {}

        resolved = {}
        
        # Merge: global → horizon → setup (base + horizon override already merged inside)
        for source_name, source_gates in [
            ("global", global_gates),
            ("horizon", horizon_gates),
            ("setup_merged", setup_gates),   # ← renamed, drop dead setup_horizon entry
        ]:
            for metric, threshold in source_gates.items():
                if threshold is not None:
                    normalized = self.normalize_threshold(threshold)
                    resolved[metric] = ResolvedGate(
                        metric=metric,
                        threshold=normalized,
                        source=source_name,
                        required=True
                    )
                elif metric in resolved:
                    # Explicit None = disable gate
                    del resolved[metric]
        
        # ✅ P1-4 FIX: Proper OrderedDict LRU eviction
        if len(self._gate_cache) >= 1000:
            # popitem(last=False) pops the OLDEST item from OrderedDict
            self._gate_cache.popitem(last=False)

        # ✅ Cache with version
        self._gate_cache[cache_key] = (self._config_version, resolved)
        self.logger.debug(f"Cache MISS (v{self._config_version}): {cache_key}")
        
        return resolved

    def get_execution_rules(self, setup_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get execution rules with horizon overrides.
        
        Returns rules like volatility_guards, structure_validation, etc.
        """
        # Global rules
        global_rules = self.base_extractor.get("execution_rules", {})
        # ✅ FIXED: Read from correct section
        horizon_rules = self.base_extractor.get("horizon_execution_rules", {})
        
        # Merge
        merged = {}
        for rule_name, rule_config in global_rules.items():
            if isinstance(rule_config, dict):
                merged[rule_name] = {**rule_config}
        
        # Apply horizon overrides
        for rule_name, rule_config in horizon_rules.items():
            if isinstance(rule_config, dict):
                if rule_name in merged:
                    merged[rule_name].update(rule_config)
                else:
                    merged[rule_name] = rule_config
        
        return merged
    

    def is_gate_enabled(
        self,
        metric: str,
        phase: str,
        setup_type: Optional[str] = None
    ) -> bool:
        """
        Check if a specific gate is enabled.
        
        Returns False if gate is explicitly set to None at any level.
        """
        gates = self.get_resolved_gates(phase, setup_type)
        return metric in gates
    
    def get_gate_threshold(
        self,
        metric: str,
        phase: str,
        setup_type: Optional[str] = None
    ) -> Optional[Dict[str, float]]:
        """
        Get threshold for a specific gate.
        
        Returns:
            {"min": X, "max": Y} or None if disabled
        """
        gates = self.get_resolved_gates(phase, setup_type)
        gate = gates.get(metric)
        return gate.threshold if gate else None
    
    # ========================================================================
    # PATTERN QUERIES
    # ========================================================================

    def get_pattern_context(self, pattern_name: str) -> Optional[PatternContext]:
        """
        Get complete pattern context for this horizon.
        
        Includes physics, entry rules, and invalidation logic.
        """
        # ✅ P2-3 FIX: Use null-byte separator
        cache_key = f"{pattern_name}\x00{self.horizon}"
        
        # ✅ Check cache with version validation
        if cache_key in self._pattern_cache:
            cached_version, cached_data = self._pattern_cache[cache_key]
            if cached_version == self._config_version:
                self.logger.debug(f"Pattern cache HIT (v{cached_version}): {cache_key}")
                # Move to end (most recently used)
                self._pattern_cache.move_to_end(cache_key)
                return cached_data
            else:
                # Config changed - invalidate
                self.logger.debug(
                    f"Pattern cache STALE (v{cached_version} != v{self._config_version}): {cache_key}"
                )
                del self._pattern_cache[cache_key]
        
        # Check horizon support FIRST
        if not self.is_pattern_supported_for_horizon(pattern_name):
            self.logger.debug(
                f"Pattern '{pattern_name}' not supported for horizon '{self.horizon}'"
            )
            return None
        
        pattern_meta = self.base_extractor.get(f"pattern_{pattern_name}")
        if not pattern_meta:
            # Try alternative key format
            pattern_meta = self.base_extractor.sections.get(f"pattern_{pattern_name}")
            if pattern_meta:
                pattern_meta = pattern_meta.data
        
        if not pattern_meta:
            self.logger.warning(f"Pattern metadata not found: {pattern_name}")
            return None
        
        # Extract components
        physics = pattern_meta.get("physics", {})
        typical_duration = pattern_meta.get("typical_duration", {})
        entry_rules_all = pattern_meta.get("entry_rules", {})
        invalidation = pattern_meta.get("invalidation", {})
        
        # Get horizon-specific entry rules
        entry_rules = entry_rules_all.get(self.horizon, {})
        
        # Get horizon-specific invalidation
        # Get horizon-specific invalidation
        breakdown = invalidation.get("breakdown_threshold", {})
        horizon_invalidation = breakdown.get(self.horizon, {})

        # Resolve action: parent-level "action" may be a horizon-keyed dict or a plain string
        raw_action = invalidation.get("action", "EXIT_ON_CLOSE")
        if isinstance(raw_action, dict):
            resolved_action = raw_action.get(self.horizon, "EXIT_ON_CLOSE")
        else:
            resolved_action = raw_action

        # Inject resolved action into horizon_invalidation so trade_enhancer gets a string
        if horizon_invalidation:
            horizon_invalidation = dict(horizon_invalidation)  # copy — don't mutate cache source
            horizon_invalidation["action"] = resolved_action

        # Get scoring thresholds
        scoring_thresholds = self.base_extractor.get("pattern_scoring_thresholds", {})
        
        context = PatternContext(
            physics=physics,
            typical_duration=typical_duration,
            entry_rules=entry_rules,
            invalidation=horizon_invalidation,
            scoring_thresholds=scoring_thresholds
        )
        
        # ✅ LRU Limit - OrderedDict
        if len(self._pattern_cache) >= 1000:
            self._pattern_cache.popitem(last=False)
            
        self._pattern_cache[cache_key] = (self._config_version, context)
        self.logger.debug(f"Pattern cache MISS (v{self._config_version}): {cache_key}")
        
        return context

    def get_gate_registry(self) -> Dict[str, Any]:
        """
        ✅ NEW: Unified gate registry access.
        Exposes GATE_METRIC_REGISTRY via extractor.
        """
        # ✅ P0-1 FIX: Unified gate registry access via base_extractor
        return self.base_extractor.get("gate_metric_registry", {})

    def get_setup_patterns(self, setup_name: str) -> Dict[str, List[str]]:
        """
        Get pattern mappings for a setup, filtered by horizon support.
        
        Returns only patterns that support the current horizon.
        
        Args:
            setup_name: Setup name (e.g., 'MOMENTUMBREAKOUT')
        
        Returns:
            {
                "PRIMARY": [...],
                "CONFIRMING": [...],  # Filtered by horizon
                "CONFLICTING": [...]
            }
        
        Example:
            For horizon='intraday' and MOMENTUMBREAKOUT:
                PRIMARY: ['darvasBox', 'flagPennant']
                CONFIRMING: ['bollingerSqueeze']  # 'minerviniStage2' filtered out!
        """
        setup_config = self.base_extractor.get(f"setup_{setup_name}")   #self.base_extractor.get(f"setup:{setup_name}")
        if not setup_config:
            return {"PRIMARY": [], "CONFIRMING": [], "CONFLICTING": []}
        
        raw_patterns = setup_config.get("patterns", {
            "PRIMARY": [],
            "CONFIRMING": [],
            "CONFLICTING": []
        })
        
        # Filter patterns based on horizon support
        filtered_patterns = {}
        for category, pattern_list in raw_patterns.items():
            filtered = [
                pattern for pattern in pattern_list
                if self.is_pattern_supported_for_horizon(pattern)
            ]
            filtered_patterns[category] = filtered
            
            # Log filtered patterns for debugging
            removed = set(pattern_list) - set(filtered)
            if removed:
                self.logger.debug(
                    f"Setup '{setup_name}' - {category}: Filtered out patterns "
                    f"{removed} for horizon '{self.horizon}'"
                )
        
        return filtered_patterns

    def get_setup_validation_modifiers(self, setup_name: str) -> Dict[str, Dict]:
        """
        Get validation modifiers (penalties/bonuses) for a setup.
        
        Returns:
            {
                "penalties": {penalty_name: config},
                "bonuses": {bonus_name: config}
            }
        """
        validation = self.base_extractor.get(f"setup_validation_{setup_name}")
        if not validation:
            return {"penalties": {}, "bonuses": {}}
        
        return {
            "penalties": validation.get("penalties", {}),
            "bonuses": validation.get("bonuses", {})
        }

    # ========================================================================
    # STRATEGY QUERIES
    # ========================================================================

    def get_strategy_fit_indicators(
        self,
        strategy_name: str
    ) -> Dict[str, Dict[str, Any]]:
        """
        ✅ ENHANCED: Get fit indicators with normalization.
        
        Changes from original:
        - Normalizes all fit indicators to consistent format
        - Adds validation for required fields
        
        Returns:
            Dict of {metric_name: normalized_fit_config}
            
        Example:
            >>> fit = extractor.get_strategy_fit_indicators('swing_breakout')
            >>> print(fit['adx'])
            {
                'min': 20,
                'max': None,
                'weight': 0.3,
                'direction': 'normal'
            }
        """
        raw_fit = self.base_extractor.get(f"strategy_fit_{strategy_name}", {})
        
        # ✅ Normalize all fit indicators
        normalized = {}
        for metric, config in raw_fit.items():
            normalized[metric] = self.normalize_fit_indicator(config)
        
        return normalized

    def get_strategy_scoring_rules(
        self,
        strategy_name: str
    ) -> Dict[str, Dict]:
        """
        Get scoring rules for a strategy.

        Returns:
            {
                "rule_name": {
                    "gates": {"roe": {"min": 20}, "roce": {"min": 25}},
                    "points": 30,
                    "reason": "Strong capital efficiency"
                }
            }
        """
        return self.base_extractor.get(f"strategy_scoring_{strategy_name}", {})

    def get_strategy_market_cap_requirements(
        self,
        strategy_name: str
    ) -> Dict[str, Dict]:
        """
        Get market cap requirements for a strategy.
        
        Returns bracket-based requirements (micro_cap, small_cap, etc.)
        """
        return self.base_extractor.get(f"strategy_market_cap_{strategy_name}", {})

    def is_strategy_blocked_for_horizon(self, strategy_name: str) -> bool:
        """
        Check if strategy is blocked for this horizon.
        
        ✅ NOW USES: strategy_matrix_config.py
        A strategy is blocked if its fit multiplier for the horizon is 0.0.
        """
        strategy_config = self.base_extractor.get(f"strategy_{strategy_name}", {})
        fit_multipliers = strategy_config.get("horizon_fit_multipliers", {})
        return fit_multipliers.get(self.horizon, 1.0) == 0.0
    
    def get_setup_context_requirements(
        self, 
        setup_name: str,
        include_horizon_override: bool = True
    ) -> Dict[str, Any]:
        """
        ✅ FIXED: Get context requirements with PROPER merge (per-metric, not section-level).
        
        CRITICAL: Merges individual metrics, not entire sections.
        This preserves inheritance: if base has {adx: {min: 18}, rsi: {min: 50}}
        and override only has {adx: {min: 20}}, result is {adx: {min: 20}, rsi: {min: 50}}.
        
        Args:
            setup_name: Setup name
            include_horizon_override: If True, merge horizon override per-metric
        
        Returns:
            Complete context requirements (technical + fundamental + opportunity)
        
        Example:
            >>> # Base has: technical={adx: {min: 18}, trendStrength: {min: 5.0}}
            >>> # Override has: technical={adx: {min: 20}}
            >>> reqs = extractor.get_setup_context_requirements("MOMENTUM_BREAKOUT")
            >>> print(reqs["technical"])
            {
                'adx': {'min': 20},           # ← Overridden
                'trendStrength': {'min': 5.0}  # ← Preserved from base
            }
        """
        # ✅ Finding 1.5-B FIX: Fail-Fast if setup is unknown
        if f"setup_{setup_name}" not in self.base_extractor.sections:
            from config.config_extractor import ConfigurationError
            msg = f"ARCHITECTURAL VIOLATION: Requested context requirements for UNKNOWN setup '{setup_name}'"
            self.logger.error(msg)
            raise ConfigurationError(msg)

        # Get base context requirements
        base_context = self.base_extractor.get(f"setup_context_{setup_name}", {})
        
        if not include_horizon_override:
            return base_context
        
        # Get horizon override
        horizon_override = self.base_extractor.get(
            f"setup_{setup_name}_override_{self.horizon}", {}
        )
        
        # Merge context_requirements
        override_context = horizon_override.get("context_requirements", {})
        if "opportunity" in override_context:
            from config.config_extractor import ConfigurationError
            msg = (
                f"CRITICAL CONFIG ERROR: Setup '{setup_name}' has 'opportunity' "
                f"nested inside 'context_requirements' in horizon override. "
                f"Opportunity gates MUST be at the top level of the override."
            )
            self.logger.error(msg)
            raise ConfigurationError(msg)
        
        merged = {}
        
        # ✅ FIX: Merge per-metric, not per-section
        for section in ["technical", "fundamental"]:
            base_section = base_context.get(section, {})
            override_section = override_context.get(section, {})
            
            # Start with base metrics
            merged_section = {}
            for metric, threshold in base_section.items():
                merged_section[metric] = threshold
            
            # Apply overrides per-metric (only replaces specified metrics)
            for metric, threshold in override_section.items():
                if threshold is None:
                    # Explicit None = remove gate
                    if metric in merged_section:
                        del merged_section[metric]
                else:
                    # Override this specific metric
                    merged_section[metric] = threshold
            
            merged[section] = merged_section
        
        # Merge opportunity gates (same per-metric logic)
        base_opp = base_context.get("opportunity", {})
        override_opp = horizon_override.get("opportunity", {})
        
        merged_opp = {}
        for metric, threshold in base_opp.items():
            merged_opp[metric] = threshold
        
        for metric, threshold in override_opp.items():
            if threshold is None:
                if metric in merged_opp:
                    del merged_opp[metric]
            else:
                merged_opp[metric] = threshold
        
        merged["opportunity"] = merged_opp
        
        return merged
    
    def get_setup_horizon_override(
        self,
        setup_name: str,
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        ✅ NEW: Get raw horizon override for a setup.
        
        ⚠️ USE CASES:
        1. **Debugging/Inspection**: See what the override actually contains
        2. **Advanced Validation**: Check if horizon has specific overrides
        3. **Config Diagnostics**: Generate reports on horizon customization
        
        ❌ DON'T USE FOR:
        - Normal gate resolution (use get_resolved_gates() instead)
        - Context requirements (use get_setup_context_requirements() instead)
        
        Args:
            setup_name: Setup name
            section: Optional specific section ("context_requirements", "opportunity")
        
        Returns:
            Raw horizon override dict (NOT merged with base)
        
        Example:
            >>> # Debugging: Check what intraday changes for MOMENTUM_BREAKOUT
            >>> override = extractor.get_setup_horizon_override("MOMENTUM_BREAKOUT")
            >>> print(override.keys())
            dict_keys(['context_requirements', 'opportunity'])
            
            >>> # Validation: Does this setup have horizon-specific opportunity gates?
            >>> opp_override = extractor.get_setup_horizon_override(
            ...     "MOMENTUM_BREAKOUT", 
            ...     "opportunity"
            ... )
            >>> if opp_override:
            ...     print("This setup has horizon-specific opportunity gates")
            
            >>> # Diagnostics: Generate config report
            >>> all_setups = extractor.get_all_setup_names()
            >>> report = {}
            >>> for setup in all_setups:
            ...     override = extractor.get_setup_horizon_override(setup)
            ...     if override:
            ...         report[setup] = list(override.keys())
            >>> print(f"Setups with horizon overrides: {len(report)}")
        """
        # Use a safe lookup to avoid "Config section not found" debug noise
        # Only query if we know the section exists in the configuration.
        key = f"setup_{setup_name}_override_{self.horizon}"
        if hasattr(self.base_extractor, "sections") and key in self.base_extractor.sections:
            override = self.base_extractor.get(key, {})
        else:
            override = {}
            
        if section:
            return override.get(section, {})
        
        return override
    
    def get_horizon_pillar_weights(self) -> Dict[str, float]:
        """
        Get horizon-specific pillar weights for final arbitration.
        
        Returns:
            {"tech": 0.70, "fund": 0.00, "hybrid": 0.30}
        """
        weights = self.base_extractor.get("horizon_pillar_weights", {})
        if weights:
            return weights
            
        # Fallback only if extraction failed completely
        return {
            "tech": 0.5, 
            "fund": 0.3, 
            "hybrid": 0.2
        }
    
    def get_hybrid_metric_registry(self) -> Dict:
        """
        Get hybrid metric registry for scoring.
        
        Returns:
            HYBRID_METRIC_REGISTRY dict
        """
        # ✅ P0-1 FIX: Unified hybrid registry access via base_extractor
        return self.base_extractor.get("hybrid_metric_registry", {})
    
    def get_hybrid_pillar_composition(self) -> Dict:
        """
        Get horizon-specific hybrid pillar composition weights.
        
        Returns:
            Weights dict for current horizon
        """
        # ✅ P0-1 FIX: Unified hybrid composition access via base_extractor
        return self.base_extractor.get("hybrid_pillar_composition", {})
    # ========================================================================
    # SETUP QUERIES
    # ========================================================================

    def get_setup_priority(self, setup_name: str) -> float:
        """
        Get resolved priority for a setup.
        
        Hierarchy:
        1. Horizon override (if exists)
        2. Setup default (from pattern matrix)
        3. Fallback to 0
        """
        # Check horizon override first
        overrides = self.base_extractor.horizon_priority_overrides
        if setup_name in overrides:
            return overrides[setup_name]
        
        # Check pattern matrix default
        setup_config = self.base_extractor.get(f"setup_{setup_name}")
        if setup_config:
            return setup_config.get("default_priority", 0)
        
        return 0

    def get_setup_confidence_floor(self, setup_name: str) -> float:
        """
        Get resolved confidence floor for a setup.
        
        ✅ NOW USES: confidence_config.py via get_setup_baseline_floor()
        """
        return self.get_setup_baseline_floor(setup_name)

    def is_setup_blocked_for_horizon(self, setup_name: str) -> bool:
        """
        Check if setup is blocked for this horizon.
        
        A setup is blocked if its baseline confidence floor is explicitly set to None 
        in the horizon's setup_floor_overrides within confidence_config.py.
        """
        return self.get_setup_baseline_floor(setup_name) is None

    def get_setup_classification_rules(
            self,
            setup_name: str
        ) -> Dict[str, Any]:
            """
            Get classification rules for a setup.
            
            Returns conditions and requirements from pattern matrix.
            """
            setup_config = self.base_extractor.get(f"setup_{setup_name}")
            if not setup_config:
                return {}
            
            return setup_config.get("classification_rules", {})
    # ========================================================================
    # RISK MANAGEMENT QUERIES (Unchanged)
    # ========================================================================


    def get_rr_gates(self) -> Dict[str, Any]:
        """
        Return merged RR gates from within risk_management.
        Hierarchy: Global.Risk.RR -> Horizon.Risk.RR
        """
        # 1. Get the fully resolved risk config first
        risk_config = self.get_risk_management_config()
        
        # 2. Extract the subsection
        return risk_config.get("rr_gates", {})


    def get_risk_management_config(self) -> Dict[str, Any]:
        """Get merged risk management config (global + horizon)."""
        return self.base_extractor.get_merged(
            "risk_management",
            "horizon_risk_management"
        )
        

    def is_pattern_supported_for_horizon(self, pattern_name: str) -> bool:
        """Check if a pattern supports the current horizon."""
        pattern_meta = self.base_extractor.get(f"pattern_{pattern_name}")
        
        if not pattern_meta:
            section = self.base_extractor.sections.get(f"pattern_{pattern_name}")
            if section and section.is_valid:
                pattern_meta = section.data
        
        if not pattern_meta:
            self.logger.warning(f"Pattern metadata not found: {pattern_name}")
            return False
        
        physics = pattern_meta.get("physics", {})
        horizons_supported = physics.get("horizons_supported", [])
        
        if not horizons_supported:
            return True
        
        current = self.horizon.lower().replace("_", "")
        supported = [h.lower().replace("_", "") for h in horizons_supported]
        
        is_supported = current in supported

        if not is_supported:
            self.logger.debug(
                f"Pattern '{pattern_name}' not supported for horizon '{self.horizon}'. "
                f"Supported horizons: {horizons_supported}"
            )
        
        return is_supported
    
    def get_global_setup_multiplier(self, setup_name: str) -> float:
        """
        Get global setup multiplier (cross-horizon baseline).
        
        This is the FIRST multiplier applied before horizon-specific adjustments.
        It represents the inherent risk/volatility of the setup type itself.
        
        Args:
            setup_name: Setup name (e.g., 'MOMENTUM_BREAKOUT')
        
        Returns:
            Global multiplier (default: 1.0)
        """
        # Get global position sizing config
        global_sizing = self.base_extractor.get("position_sizing", {})
        global_setup_mults = global_sizing.get("global_setup_multipliers", {})
        
        # Return setup-specific multiplier or default to 1.0
        multiplier = global_setup_mults.get(setup_name, 1.0)
        
        self.logger.debug(
            f"[{setup_name}] Global setup multiplier: {multiplier:.2f}"
        )
        
        return multiplier

    def get_combined_position_sizing_multipliers(
        self, 
        setup_name: str
    ) -> Dict[str, float]:
        """
        Get ALL position sizing multipliers for a setup with transparent breakdown.
        
        This provides complete visibility into all three multiplier layers:
        1. **Global setup multiplier** - Inherent setup risk (cross-horizon)
        2. **Horizon setup multiplier** - Horizon-specific adjustment for this setup
        3. **Horizon base multiplier** - Universal horizon adjustment (all setups)
        
        The strategy multiplier is applied separately by the resolver.
        
        Args:
            setup_name: Setup name (e.g., 'MOMENTUM_BREAKOUT')
        
        Returns:
            {
                "global_setup": 1.2,       # From master_config.position_sizing.global_setup_multipliers
                "horizon_setup": 1.1,      # From horizon strategy_config.sizing_multipliers
                "horizon_base": 1.0,       # From horizon base multiplier
                "combined": 1.32           # global × horizon_setup × horizon_base
            }    
        Note:
            This replaces the older get_position_sizing_multiplier() which
            didn't provide transparent breakdown of all layers.
        """
        # 1. Global setup multiplier (cross-horizon baseline)
        global_mult = self.get_global_setup_multiplier(setup_name)
        
        # 2. Horizon-specific setup multiplier
        # This comes from horizon strategy config's sizing_multipliers
        merged_mults = self.base_extractor.get("sizing_multipliers", {})
        horizon_setup_mult = merged_mults.get(setup_name, 1.0)
        
        # 3. Horizon base multiplier (applies to ALL setups in this horizon)
        # Example: intraday might be 0.8x, short_term 1.0x, long_term 1.2x
        horizon_base_mult = self.base_extractor.get("horizon_base_multiplier", 1.0)
        
        # 4. Combined multiplier (strategy mult applied separately by resolver)
        combined = global_mult * horizon_setup_mult * horizon_base_mult
        
        self.logger.debug(
            f"[{setup_name}] Position sizing breakdown: "
            f"global={global_mult:.2f}, "
            f"horizon_setup={horizon_setup_mult:.2f}, "
            f"horizon_base={horizon_base_mult:.2f}, "
            f"combined={combined:.2f}"
        )
        
        return {
            "global_setup": global_mult,
            "horizon_setup": horizon_setup_mult,
            "horizon_base": horizon_base_mult,
            "combined": combined
        }

    
    # ========================================================================
    # ✅ CATEGORY 1: SETUP QUERIES (ADD THESE)
    # ========================================================================
    
    def get_all_setup_names(self) -> List[str]:
        """
        Get list of all available setup names.
        
        Returns:
            List of setup names from setup_pattern_matrix
        
        Example:
            >>> setup_names = extractor.get_all_setup_names()
            >>> print(setup_names[:3])
            ['MOMENTUM_BREAKOUT', 'PATTERN_DARVAS_BREAKOUT', 'QUALITY_ACCUMULATION']
        """
        setup_matrix = self.base_extractor.get("setup_pattern_matrix", {})
        return list(setup_matrix.keys())
    

    # ========================================================================
    # ✅ CATEGORY 2: STRATEGY QUERIES (ADD THESE)
    # ========================================================================
    
    def get_all_strategy_names(self) -> List[str]:
        """
        Get all strategy names from strategy matrix.
        
        Returns:
            List of strategy names
        
        Example:
            >>> strategies = extractor.get_all_strategy_names()
            >>> print(strategies[:3])
            ['swing_breakout', 'day_trading', 'minervini_growth']
        """
        strategy_matrix = self.base_extractor.get("strategy_matrix", {})
        return list(strategy_matrix.keys())
    
    def get_strategy_enabled_status(self, strategy_name: str) -> bool:
        """
        Check if strategy is globally enabled.
        
        Args:
            strategy_name: Strategy name (e.g., 'minervini_growth')
        
        Returns:
            True if enabled, False otherwise
        
        Example:
            >>> extractor.get_strategy_enabled_status('minervini_growth')
            True
        """
        strategy_config = self.base_extractor.get(f"strategy_{strategy_name}")
        return strategy_config.get("enabled", False) if strategy_config else False
    
    def get_strategy_preferred_setups(self, strategy_name: str) -> List[str]:
        """
        Get list of setups preferred by this strategy.
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            List of preferred setup names
        """
        strategy_config = self.base_extractor.get(f"strategy_{strategy_name}")
        return strategy_config.get("preferred_setups", []) if strategy_config else []
    
    def get_strategy_avoided_setups(self, strategy_name: str) -> List[str]:
        """
        Get list of setups avoided by this strategy.
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            List of avoided setup names
        """
        strategy_config = self.base_extractor.get(f"strategy_{strategy_name}")
        return strategy_config.get("avoid_setups", []) if strategy_config else []
    
    # ========================================================================
    # ✅ CATEGORY 3: EXECUTION RULES (ADD THESE)
    # ========================================================================
    
    def is_execution_rule_enabled(self, rule_name: str) -> bool:
        """
        Check if execution rule is enabled for this horizon.
        
        Args:
            rule_name: Rule name (e.g., 'volatility_guards')
        
        Returns:
            True if enabled (or no 'enabled' flag), False if explicitly disabled
        
        Example:
            >>> extractor.is_execution_rule_enabled('volatility_guards')
            True
        """
        exec_rules = self.get_execution_rules()
        rule_config = exec_rules.get(rule_name, {})
        return rule_config.get("enabled", True)  # Default to enabled
    
    def get_volatility_guards_config(self) -> Dict[str, Any]:
        """
        Get volatility guards configuration.
        
        Returns:
            Volatility guards config dict
        
        Example:
            >>> vol_guards = extractor.get_volatility_guards_config()
            >>> print(vol_guards.get('extreme_vol_buffer'))
            2.0
        """
        exec_rules = self.get_execution_rules()
        return exec_rules.get("volatility_guards", {})
    
    def get_structure_validation_config(self) -> Dict[str, Any]:
        """
        Get structure validation configuration.
        
        Returns:
            Structure validation config dict
        """
        exec_rules = self.get_execution_rules()
        return exec_rules.get("structure_validation", {})
    
    def get_sl_distance_validation_config(self) -> Dict[str, Any]:
        """
        Get SL distance validation configuration.
        
        Returns:
            {
                "enabled": bool,
                "min_sl_distance_pct": float,
                "max_sl_distance_pct": float
            }
        """
        exec_rules = self.get_execution_rules()
        return exec_rules.get("sl_distance_validation", {})

    def get_target_proximity_rejection_config(self) -> Dict[str, Any]:
        """
        Get target proximity rejection configuration.
        
        Returns:
            {
                "enabled": bool,
                "min_target_distance_pct": float,
                "max_target_distance_pct": float
            }
        """
        exec_rules = self.get_execution_rules()
        return exec_rules.get("target_proximity_rejection", {})
    # ========================================================================
    # ✅ CATEGORY 4: MARKET CONSTRAINTS (ADD THESE)
    # ========================================================================
    
    def get_market_constraints_config(self) -> Dict[str, Any]:
        """
        Market-level execution constraints (e.g., Indian intraday rules).
        Pure config access — no evaluation.
        """

        if self.horizon in ["intraday", "short_term", "long_term", "multibagger"]:
            exec_rules = self.get_execution_rules()
            gates = exec_rules.get("indian_market_gates", {})

            return {
                "enabled": bool(gates),
                "gates": gates,
                "market": "IN",
                "horizon": self.horizon,
            }

        return {
            "enabled": False,
            "gates": {},
            "market": None,
            "horizon": self.horizon,
        }

    
    def get_time_filters_config(self) -> Dict[str, Any]:
        """
        Get time-based filters for execution.
        
        Returns:
            Time filters config (e.g., avoid first 15 min, lunch hour sizing)
        """
        market_constraints = self.get_market_constraints_config()
        return market_constraints.get("time_filters", {})
    
    # ========================================================================
    # ✅ CATEGORY 5: PATTERN VALIDATION (ADDITIONAL HELPERS)
    # ========================================================================
    

    # ========================================================================
    # ✅ CATEGORY 6: CONFIDENCE CALCULATION (ADVANCED)
    # ========================================================================

    def calculate_dynamic_confidence_floor(
        self,
        setup_type: str,
        adx_data: any
    ) -> float:
        """
        Calculate dynamic confidence floor adjusted by ADX regime.
        
        ✅ FIXED: Now handles both BOOSTS (Strong Trend) and PENALTIES (Weak Trend).
        """
        # Get base floor
        base_floor = self.get_setup_baseline_floor(setup_type)
        
        # Get horizon base adjustment
        # ✅ Phase 3 P2-4 FIX: Horizon floor overrides are absolute. 
        # If a setup-specific floor exists for this horizon, we bypass the baseline -10 penalty.
        # This prevents "leakage" where a 60 floor became 50.
        horizon_overrides = self.base_extractor.get("horizon_setup_floor_overrides", {})
        has_horizon_override = setup_type in horizon_overrides
        
        base_adj = 0.0 if has_horizon_override else self.get_base_confidence_adjustment()
        
        # Apply base adjustment
        adjusted_floor = base_floor + base_adj
        
        # Extract ADX value
        adx_value = adx_data
        
        # ---------------------------------------------------------
        # 1. Apply Boosts (Higher is better)
        # ---------------------------------------------------------
        adx_bands = self.get_adx_confidence_bands()
        
        # Sort bands by ADX threshold (descending) to match highest quality first
        band_list = []
        for band_name, band_config in adx_bands.items():
            gates = band_config.get("gates", {})
            adx_min = gates.get("adx", {}).get("min", 0)
            boost = band_config.get("confidence_boost", 0)
            band_list.append((adx_min, boost, band_name))
        
        band_list.sort(reverse=True, key=lambda x: x[0])
        
        boost_applied = False
        for threshold, boost, band_name in band_list:
            try:
                if adx_value >= threshold:
                    adjusted_floor += boost
                    boost_applied = True
                    self.logger.debug(
                        f"[{setup_type}] ADX {adx_value:.1f} >= {threshold}, "
                        f"applying {band_name} boost: +{boost}"
                    )
                    break
            except Exception as e:
                self.logger.error(
                    f"Error applying ADX boost for band '{band_name}': {e}",
                    exc_info=True
                )
                
        # ---------------------------------------------------------
        # 2. Apply Penalties (Lower is worse) - ✅ NEW
        # ---------------------------------------------------------
        if not boost_applied:
            adx_penalties = self.get_adx_confidence_penalties()
            
            # ✅ P3-2 FIX: Sort penalties by max_val ascending for deterministic priority
            sorted_penalties = sorted(
                adx_penalties.items(),
                key=lambda x: x[1].get("gates", {}).get("adx", {}).get("max", 999)
            )
            
            for name, config in sorted_penalties:
                gates = config.get("gates", {})
                max_val = gates.get("adx", {}).get("max")
                
                if max_val is not None and adx_value <= max_val:
                    penalty_val = config.get("confidence_penalty", 0)
        
                    # ✅ Phase 3 P2-4 FIX: SIGN GUARD - Penalties must be negative (prevent unintended boosts)
                    # Upgrade to ConfigurationError to enforce fail-fast architecture.
                    if penalty_val > 0:
                        raise ConfigurationError(
                            f"ARCHITECTURAL VIOLATION: Positive penalty {penalty_val} found in ADX penalty band "
                            f"'{name}' for setup '{setup_type}'. Penalties must be negative."
                        )
                        
                    adjusted_floor += penalty_val # Penalty is negative
                    self.logger.debug(
                        f"[{setup_type}] ADX {adx_value:.1f} <= {max_val}, "
                        f"applying {name} penalty: {penalty_val}"
                    )
                    break  # Apply only the first matching penalty
        
        # Clamp to valid range
        clamp = self.get_confidence_clamp()
        return max(clamp[0], min(clamp[1], adjusted_floor))
    
    # ========================================================================
    # ✅ CATEGORY 7: UTILITY METHODS
    # ========================================================================

    def _prepare_evaluation_namespace(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare flat namespace for condition evaluation.
        
        Handles nested indicator dicts:
        {"adx": {"value": 25, "raw": 24.8}} → {"adx": 25}
        
        Args:
            data: Raw market data (may have nested dicts)
        
        Returns:
            Flattened namespace for safe eval

        NEW: Exposes fundamental category scores as buckets
        """
        namespace = {}
        
        for key, value in data.items():
            if isinstance(value, dict):
                # Extract scalar from nested dict
                # Priority: value > raw > score > dict itself
                # ✅ Phase 3 P3-4 FIX: Ensure 'value' is NOT None before shadowing.
                # If 'value' is None but 'raw' exists, 'None' would shadow the valid 'raw' decimal.
                namespace[key] = (
                    value.get("value") if value.get("value") is not None and not isinstance(value.get("value"), str) else
                    value.get("raw") if "raw" in value else
                    value.get("score") if "score" in value else
                    value  # Keep dict if no scalar found
                )
            else:
                namespace[key] = value

        # ✅ FIXED: Flatten 'indicators' and 'price_data' dicts for condition checks
        for nested_key in ["indicators", "price_data"]:
            nested = data.get(nested_key, {})
            if isinstance(nested, dict):
                for k, v in nested.items():
                    if k not in namespace:  # don't override top-level
                        namespace[k] = (
                            v.get("value") if isinstance(v, dict) and v.get("value") is not None and not isinstance(v.get("value"), str) else
                            v.get("raw") if isinstance(v, dict) and "raw" in v else
                            v.get("score") if isinstance(v, dict) and "score" in v else
                            v
                        )
        
        # Expose fundamental category scores as buckets
        fundamental_data = data.get("fundamentals", data.get("fundamental", {}))
        if isinstance(fundamental_data, dict):
            if "score" in fundamental_data:
                namespace["fundamentalScore"] = fundamental_data.get("score", 0.0)
            
            category_scores = fundamental_data.get("category_scores", {})
            if category_scores:
                namespace["fund_valuation_bucket"] = category_scores.get("valuation", {}).get("score")
                namespace["fund_profitability_bucket"] = category_scores.get("profitability", {}).get("score")
                namespace["fund_growth_bucket"] = category_scores.get("growth", {}).get("score")
                namespace["fund_health_bucket"] = category_scores.get("financial_health", {}).get("score")
                namespace["fund_quality_bucket"] = category_scores.get("quality", {}).get("score")
                namespace["fund_ownership_bucket"] = category_scores.get("ownership", {}).get("score")
                namespace["fund_dividend_bucket"] = category_scores.get("dividend", {}).get("score")
                namespace["fund_market_bucket"] = category_scores.get("market", {}).get("score")
        
        technical_data = data.get("technical", {})
        if isinstance(technical_data, dict):
            if "score" in technical_data:
                namespace["technicalScore"] = technical_data.get("score", 0.0)
        
        return namespace
    def get_technical_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proxy for technical score calculation.
        Injects the current horizon automatically.
        """
        # Lazy import to prevent circular dependency at module level
        return compute_technical_score(indicators, self.horizon)

    def get_fundamental_score(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proxy for fundamental score calculation.
        Injects the current horizon automatically.
        """
        return compute_fundamental_score(fundamentals, self.horizon)

    def calculate_dynamic_metric_score(
        self, 
        metric_name: str, 
        value: float, 
        indicators: Dict[str, Any] = None
    ) -> float:
        """
        Proxy for dynamic single-metric scoring (used in Hybrid).
        Automatically fetches the registry.
        """
        registry = self.get_hybrid_metric_registry()
        return calculate_dynamic_score(metric_name, value, indicators, registry)
    
    def check_metric_threshold(
        self,
        metric_name: str,
        actual_value: float,
        required_config: Dict[str, Any],
        evaluation_type: str = "gate"
    ) -> Tuple[bool, str]:
        """
        ✅ NEW: Universal metric threshold checker.
        
        Handles both gate-style and fit-indicator-style configs.
        
        Args:
            metric_name: Metric name (for error messages)
            actual_value: Actual metric value
            required_config: Either gate config or fit indicator config
            evaluation_type: "gate" or "fit_indicator"
        
        Returns:
            Tuple of (passed: bool, reason: str)
        
        Example:
            >>> # Gate-style check
            >>> gate = {"min": 20, "max": None}
            >>> extractor.check_metric_threshold('adx', 25, gate, 'gate')
            (True, "adx=25 >= min(20)")
            
            >>> # Fit indicator check
            >>> fit = {"min": 25, "weight": 0.3, "direction": "normal"}
            >>> extractor.check_metric_threshold('adx', 30, fit, 'fit_indicator')
            (True, "adx=30 meets threshold for scoring")
        """
        # Normalize to canonical threshold format
        if evaluation_type == "fit_indicator":
            threshold = self.normalize_fit_indicator(required_config)
        else:
            threshold = self.normalize_threshold(required_config)
        
        # Use universal evaluator
        return self.evaluate_threshold(actual_value, threshold, metric_name)
    
    

    def get_trend_thresholds(self) -> Dict[str, Any]:
        """
        Get horizon-specific trend classification thresholds.
        
        Uses smart inheritance: horizon overrides global if present.
        
        Returns:
            {
                "slope": {
                    "strong": 15.0,    # Varies by horizon (intraday: 15, long_term: 5)
                    "moderate": 5.0
                }
            }
        
        Example:
            >>> thresholds = extractor.get_trend_thresholds()
            >>> strong_threshold = thresholds["slope"]["strong"]
            >>> if ma_slope >= strong_threshold:
            ...     trend_classification = "strong"
            >>> elif ma_slope >= thresholds["slope"]["moderate"]:
            ...     trend_classification = "moderate"
        """
        global_thresholds = self.base_extractor.get("trend_thresholds", {})
        horizon_thresholds = self.base_extractor.get("horizon_trend_thresholds", {})
        
        # Deep merge: horizon overrides global per-key
        merged = {**global_thresholds}
        for section, vals in horizon_thresholds.items():
            if isinstance(vals, dict) and section in merged:
                merged[section] = {**merged[section], **vals}
            else:
                merged[section] = vals
        return merged
    
    def get_momentum_thresholds(self) -> Dict[str, Any]:
        """
        Get horizon-specific momentum classification thresholds.
        
        Uses smart inheritance: horizon overrides global if present.
        
        Returns:
            {
                "rsislope": {
                    "acceleration_floor": 0.10,     # Varies by horizon
                    "deceleration_ceiling": -0.10
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            }
        
        Example:
            >>> thresholds = extractor.get_momentum_thresholds()
            >>> rsi_accel_floor = thresholds["rsislope"]["acceleration_floor"]
            >>> if rsi_slope >= rsi_accel_floor:
            ...     momentum_state = "accelerating"
            >>> elif rsi_slope <= thresholds["rsislope"]["deceleration_ceiling"]:
            ...     momentum_state = "decelerating"
        """
        global_thresholds = self.base_extractor.get("momentum_thresholds", {})
        horizon_thresholds = self.base_extractor.get("horizon_momentum_thresholds", {})
        
        # Deep merge: horizon overrides global per-key
        merged = {**global_thresholds}
        for section, vals in horizon_thresholds.items():
            if isinstance(vals, dict) and section in merged:
                merged[section] = {**merged[section], **vals}
            else:
                merged[section] = vals
        return merged
    

    def get_strategy_horizon_multiplier(self, strategy_name: str) -> float:
        """
        Get horizon fit multiplier for a strategy.
        
        Args:
            strategy_name: Strategy name
        
        Returns:
            Multiplier for current horizon (1.0 = neutral, 0.0 = blocked)
        
        Example:
            >>> # For short_term horizon
            >>> mult = extractor.get_strategy_horizon_multiplier('swing_breakout')
            >>> print(mult)
            1.2  # Swing trading is boosted for short_term
        """
        strategy_config = self.base_extractor.get(f"strategy_{strategy_name}")
        if not strategy_config:
            return 1.0
        
        multipliers = strategy_config.get("horizon_fit_multipliers", {})
        return multipliers.get(self.horizon, 1.0)

    
    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================


    # ------------------------------------------------------------------------
    # 🆕 NORMALIZATION: Unified Threshold Format
    # ------------------------------------------------------------------------
    
    def normalize_threshold(self, raw_threshold: Any) -> Dict[str, Optional[float]]:
        """
        ✅ NEW: Normalize any threshold format to canonical structure.
        
        This is the "Rosetta Stone" - converts all formats to standard form.
        
        Handles:
        - Dict with min/max: {"min": 20, "max": 40} → as-is
        - Dict with only min: {"min": 20} → {"min": 20, "max": None}
        - Single value: 20 → {"min": 20, "max": None}
        - None: None → {"min": None, "max": None}
        
        Args:
            raw_threshold: Any threshold format
        
        Returns:
            Canonical dict: {"min": float|None, "max": float|None}
        
        Example:
            >>> extractor.normalize_threshold({"min": 20})
            {'min': 20, 'max': None}
            
            >>> extractor.normalize_threshold(20)
            {'min': 20, 'max': None}
            
            >>> extractor.normalize_threshold({"min": 20, "max": 40})
            {'min': 20, 'max': 40}
        """
        if raw_threshold is None:
            return {"min": None, "max": None}
        
        if isinstance(raw_threshold, dict):
            # ✅ CLEANER FIX: Preserve only relevant gate keys (avoiding dict pollution)
            result = {
                "min": raw_threshold.get("min"),
                "max": raw_threshold.get("max"),
            }
            # Preserve cross-metric and equality clauses
            for key in ("equals", "min_metric", "max_metric", "multiplier", "duration"):
                if key in raw_threshold:
                    result[key] = raw_threshold[key]
            return result
        
        if isinstance(raw_threshold, (int, float)):
            return {"min": float(raw_threshold), "max": None}
        
        # Unknown format
        logger.warning(f"Unknown threshold format: {raw_threshold}")
        return {"min": None, "max": None}
    
    # ------------------------------------------------------------------------
    # 🆕 NORMALIZATION: Unified Fit Indicator Format
    # ------------------------------------------------------------------------
    
    def normalize_fit_indicator(self, raw_fit: Dict) -> Dict[str, Any]:
        """
        ✅ NEW: Normalize fit indicator to canonical structure.
        
        Ensures all fit indicators have consistent structure regardless
        of how they're defined in strategy_matrix.
        
        Args:
            raw_fit: Raw fit indicator config
        
        Returns:
            Canonical dict with:
            - min: Minimum threshold (or None)
            - max: Maximum threshold (or None)
            - weight: Relative importance (default: 0.1)
            - direction: "normal" or "invert" (default: "normal")
        
        Example:
            >>> raw = {"min": 25, "weight": 0.3}
            >>> extractor.normalize_fit_indicator(raw)
            {
                'min': 25,
                'max': None,
                'weight': 0.3,
                'direction': 'normal'
            }
        """
        return {
            "min": raw_fit.get("min"),
            "max": raw_fit.get("max"),
            "weight": raw_fit.get("weight", 0.1),
            "direction": raw_fit.get("direction", "normal")
        }
    
    # ------------------------------------------------------------------------
    # 🆕 UNIFIED EVALUATION: Universal Threshold Check
    # ------------------------------------------------------------------------
    
    def evaluate_threshold(
        self,
        actual: float,
        threshold: Dict[str, Optional[float]],
        metric_name: str = ""
    ) -> Tuple[bool, str]:
        """
        ✅ NEW: Universal threshold evaluation.
        
        Works for ANY threshold format after normalization.
        Resolver never needs to implement threshold logic.
        
        Args:
            actual: Actual metric value
            threshold: Normalized threshold dict {"min": X, "max": Y}
            metric_name: Metric name (for error messages)
        
        Returns:
            Tuple of (passed: bool, reason: str)
        
        Example:
            >>> threshold = {"min": 20, "max": 40}
            >>> extractor.evaluate_threshold(25, threshold, "adx")
            (True, "adx=25 within range [20, 40]")
            
            >>> extractor.evaluate_threshold(15, threshold, "adx")
            (False, "adx=15 below min(20)")
        """
        if actual is None:
            return False, f"{metric_name} value is None"
        
        min_val = threshold.get("min")
        max_val = threshold.get("max")
        
        # No constraints = always pass
        if min_val is None and max_val is None:
            return True, f"{metric_name} has no constraints"
        
        # Check minimum
        if min_val is not None and actual < min_val:
            return False, f"{metric_name}={actual:.2f} below min({min_val})"
        
        # Check maximum
        if max_val is not None and actual > max_val:
            return False, f"{metric_name}={actual:.2f} above max({max_val})"
        
        # Build success message
        if min_val is not None and max_val is not None:
            reason = f"{metric_name}={actual:.2f} within range [{min_val}, {max_val}]"
        elif min_val is not None:
            reason = f"{metric_name}={actual:.2f} >= min({min_val})"
        else:
            reason = f"{metric_name}={actual:.2f} <= max({max_val})"
        
        return True, reason
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def clear_cache(self):
        """Clear all cached queries."""
        self._gate_cache.clear()
        self._pattern_cache.clear()
        # ❌ REMOVED: self._confidence_cache.clear()
        self.logger.info(f"Cache cleared (was v{self._config_version})")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics with version info."""
        return {
            "config_version": self._config_version,
            "gate_cache_size": len(self._gate_cache),
            "pattern_cache_size": len(self._pattern_cache),
            # ❌ REMOVED: "confidence_cache_size": len(self._confidence_cache)
            "gate_cache_items": list(self._gate_cache.keys()),
            "pattern_cache_items": list(self._pattern_cache.keys()),
        }
    
    # ------------------------------------------------------------------------
    # 🆕 DIAGNOSTIC: Validate extractor state
    # ------------------------------------------------------------------------
    
    def validate_extractor_state(self) -> Dict[str, Any]:
        """
        Validate extractor configuration and state.
        Returns:
            Validation report with errors and warnings
        """
        errors : List[str] = []
        warnings : List[str] = []
        
        # Check base_extractor has required attributes
        if not hasattr(self.base_extractor, 'master_config'):
            errors.append("base_extractor missing 'master_config' attribute")
        else:
            # Check horizon config exists
            master_dict = self.base_extractor.master_config
            if self.horizon not in master_dict.get("horizons", {}):
                errors.append(
                    f"No config found for horizon '{self.horizon}' in master_config"
                )
        
        # Check confidence config loaded
        if not self.is_confidence_config_loaded():
            warnings.append("Confidence config not loaded - using fallback")
        
        # Check critical sections
        critical_sections = ["confidence_range", "setup_baseline_floors", "horizon_confidence_clamp"]
        for section in critical_sections:
            if not self.base_extractor.get(section):
                error_msg = f"Missing section {section}"
                errors.append(error_msg)

        return {
            "horizon": self.horizon,
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "cache_stats": self.get_cache_stats(),
            "has_confidence_config": self.is_confidence_config_loaded(),
        }
    
    def is_confidence_config_loaded(self) -> bool:
        """
        Check if confidence_config.py is loaded.
        
        Delegates to base_extractor to avoid AttributeError.
        """
        return getattr(self.base_extractor, 'has_confidence_config', False)

    # ------------------------------------------------------------------------
    # PROPERTY PROXIES (Delegated to base_extractor)
    # ------------------------------------------------------------------------

    @property
    def blocked_setups(self) -> Set[str]:
        """Get blocked setups for this horizon."""
        return self.base_extractor.blocked_setups

    @property
    def preferred_setups(self) -> List[str]:
        """Get preferred setups for this horizon."""
        return self.base_extractor.preferred_setups

    @property
    def blocked_strategies(self) -> Set[str]:
        """Get blocked strategies for this horizon."""
        return self.base_extractor.blocked_strategies

    @property
    def strategy_multipliers(self) -> Dict[str, float]:
        """Get strategy priority multipliers for this horizon."""
        return self.base_extractor.strategy_multipliers
    



# ==============================================================================
# DIAGNOSTIC: Validate extractor state
# ==============================================================================
