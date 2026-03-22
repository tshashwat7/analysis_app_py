# config/config_extractor_v2.py
"""
Config Extractor v2.0 - Cleaned
================================
Centralized config extraction system.

Key Features:
✅ Extracts from master_config, setup_pattern_matrix, strategy_matrix
✅ Horizon-aware extraction with overrides
✅ Clean query interface with get(), get_required(), get_merged()
✅ Comprehensive validation
✅ Property shortcuts for common queries
"""

from typing import Dict, Any, Optional, Set, List
from dataclasses import dataclass
import copy
import logging

from config.master_config import HORIZON_PILLAR_WEIGHTS, HYBRID_METRIC_REGISTRY, HYBRID_PILLAR_COMPOSITION

# Initialize logger
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════

class ConfigurationError(Exception):
    """Raised when a required config path is missing or invalid.
    
    This replaces silent fallbacks for critical configuration values.
    Any path essential for trade validation (confidence clamping,
    tradeable thresholds, volume modifiers, etc.) will raise this
    instead of silently returning a default.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ConfigSection:
    """Holds a config section with metadata."""
    data: Any
    source: str  # e.g., "confidence_config.global.setup_baseline_floors"
    is_valid: bool = True
    error: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# MAIN CONFIG EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════

class ConfigExtractor:
    """
    Centralized config extraction with validation and error handling.
    Confidence configs come from confidence_config.py
    Design Principles:
    - Extract once, use many times (performance)
    - Validate on extraction (fail fast)
    - Clear error messages (debuggability)
    - Type safety where possible
    """

    def __init__(
        self,
        master_config: Dict,
        horizon: str,
        logger: Optional[logging.Logger] = None
    ):
        self.master_config = master_config
        self.horizon = horizon
        self.logger = logger or logging.getLogger(__name__)

        # ✅ NEW: Import confidence config
        try:
            from config.confidence_config import CONFIDENCE_CONFIG
            self.confidence_config = CONFIDENCE_CONFIG
            self.has_confidence_config = True
            self.logger.info("✅ Loaded confidence_config.py")
        except ImportError as e:
            self.confidence_config = {}
            self.has_confidence_config = False
            self.logger.warning(f"⚠️ Could not import confidence_config: {e}")

        # Cached sections
        self.sections: Dict[str, ConfigSection] = {}

        # Extract all sections
        self.extract_all()

    def extract_all(self):
        """Extract all config sections at initialization."""
        self.logger.info(f"Extracting configs for horizon: {self.horizon}")

        try:
            self.extract_global_sections()
            self.extract_horizon_sections()
            self.extract_risk_sections()
            self.extract_gate_sections()
            self.extract_matrix_sections()
            self.extract_strategy_sections()
            
            # ✅ NEW: Extract confidence sections
            self.extract_confidence_sections()

            # Validate after extraction
            self.validate_extracted_configs()

            self.logger.info(f"✅ Config extraction complete. {len(self.sections)} sections loaded.")

        except Exception as e:
            self.logger.error(f"❌ Config extraction failed: {e}")
            raise

    # ═══════════════════════════════════════════════════════════════════════
    # ✅ NEW: CONFIDENCE CONFIG EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_required(self, source_dict: Dict, key: str, source_path: str) -> Any:
        """Extract a config key that MUST exist.
        
        Logs CRITICAL and raises ConfigurationError if key is missing.
        This prevents silent fallbacks for essential trade-validation paths.
        """
        value = source_dict.get(key)
        if value is None:
            msg = f"CRITICAL: Required config path '{source_path}' not found"
            self.logger.critical(msg)
            raise ConfigurationError(msg)
        return value

    def extract_confidence_sections(self):
        """Extract confidence configs from confidence_config.py.
        
        Required keys raise ConfigurationError if missing.
        Optional keys use safe defaults (0, {}, "").
        """
        if not self.has_confidence_config:
            self.logger.warning("⚠️ Confidence config unavailable, using master_config fallback")
            self._extract_confidence_from_master_config()
            return

        try:
            # ═══════════════════════════════════════════════════════
            # GLOBAL: REQUIRED sections (raise on missing)
            # ═══════════════════════════════════════════════════════
            global_conf = self.confidence_config.get("global", {})
            if not global_conf:
                raise ConfigurationError(
                    "CRITICAL: confidence_config.py missing 'global' section"
                )

            self.sections["confidence_range"] = ConfigSection(
                data=self._extract_required(
                    global_conf, "confidence_range",
                    "confidence_config.global.confidence_range"
                ),
                source="confidence_config.global.confidence_range"
            )

            self.sections["adx_normalization"] = ConfigSection(
                data=self._extract_required(
                    global_conf, "adx_normalization",
                    "confidence_config.global.adx_normalization"
                ),
                source="confidence_config.global.adx_normalization"
            )

            self.sections["volume_modifiers"] = ConfigSection(
                data=self._extract_required(
                    global_conf, "volume_modifiers",
                    "confidence_config.global.volume_modifiers"
                ),
                source="confidence_config.global.volume_modifiers"
            )

            self.sections["universal_adjustments"] = ConfigSection(
                data=self._extract_required(
                    global_conf, "universal_adjustments",
                    "confidence_config.global.universal_adjustments"
                ),
                source="confidence_config.global.universal_adjustments"
            )
            
            # ✅ Phase 3 P3-2 FIX: Validate penalty signs at startup
            uni_adj = self.sections["universal_adjustments"].data
            for cat in ["divergence_penalties", "trend_strength_bands"]:
                 for name, cfg in uni_adj.get(cat, {}).items():
                      penalty = cfg.get("confidence_penalty")
                      if penalty is not None and float(penalty) > 0:
                           self.logger.warning(
                               f"Startup Validation: {cat}.{name} penalty {penalty} is positive. "
                               f"Should be negative. Auto-correction will apply at runtime."
                           )

            # Also check volume modifiers
            vol_mods = self.sections["volume_modifiers"].data
            for cat in ["dry_volume", "climax_volume"]:
                 penalty = vol_mods.get(cat, {}).get("confidence_penalty")
                 if penalty is not None and float(penalty) > 0:
                      self.logger.warning(
                          f"Startup Validation: volume_modifiers.{cat} penalty {penalty} is positive. "
                          f"Should be negative."
                      )

            self.sections["setup_baseline_floors"] = ConfigSection(
                data=self._extract_required(
                    global_conf, "setup_baseline_floors",
                    "confidence_config.global.setup_baseline_floors"
                ),
                source="confidence_config.global.setup_baseline_floors"
            )

            # ═══════════════════════════════════════════════════════
            # GLOBAL: OPTIONAL sections (safe defaults)
            # ═══════════════════════════════════════════════════════
            self.sections["divergence_physics"] = ConfigSection(
                data=global_conf.get("divergence_physics", {}),
                source="confidence_config.global.divergence_physics"
            )

            # ═══════════════════════════════════════════════════════
            # HORIZON: Validate horizon section exists
            # ═══════════════════════════════════════════════════════
            horizons_conf = self.confidence_config.get("horizons", {})
            if not horizons_conf:
                raise ConfigurationError(
                    "CRITICAL: confidence_config.py missing 'horizons' section"
                )

            horizon_conf = horizons_conf.get(self.horizon)
            if horizon_conf is None:
                raise ConfigurationError(
                    f"CRITICAL: confidence_config.py missing horizon '{self.horizon}'"
                )

            # ═══════════════════════════════════════════════════════
            # HORIZON: REQUIRED sections
            # ═══════════════════════════════════════════════════════
            self.sections["horizon_confidence_clamp"] = ConfigSection(
                data=self._extract_required(
                    horizon_conf, "confidence_clamp",
                    f"confidence_config.horizons.{self.horizon}.confidence_clamp"
                ),
                source=f"confidence_config.horizons.{self.horizon}.confidence_clamp"
            )

            self.sections["min_tradeable_confidence"] = ConfigSection(
                data=self._extract_required(
                    horizon_conf, "min_tradeable_confidence",
                    f"confidence_config.horizons.{self.horizon}.min_tradeable_confidence"
                ),
                source=f"confidence_config.horizons.{self.horizon}.min_tradeable_confidence",
            )

            self.sections["high_confidence_override"] = ConfigSection(
                data=self._extract_required(
                    horizon_conf, "high_confidence_override",
                    f"confidence_config.horizons.{self.horizon}.high_confidence_override"
                ),
                source=f"confidence_config.horizons.{self.horizon}.high_confidence_override",
            )

            # ═══════════════════════════════════════════════════════
            # HORIZON: OPTIONAL sections (safe defaults)
            # ═══════════════════════════════════════════════════════
            self.sections["horizon_confidence_philosophy"] = ConfigSection(
                data=horizon_conf.get("confidence_philosophy", ""),
                source=f"confidence_config.horizons.{self.horizon}.confidence_philosophy"
            )

            self.sections["horizon_base_confidence_adjustment"] = ConfigSection(
                data=horizon_conf.get("base_confidence_adjustment", 0),
                source=f"confidence_config.horizons.{self.horizon}.base_confidence_adjustment"
            )

            self.sections["horizon_setup_floor_overrides"] = ConfigSection(
                data=horizon_conf.get("setup_floor_overrides", {}),
                source=f"confidence_config.horizons.{self.horizon}.setup_floor_overrides"
            )

            self.sections["horizon_conditional_adjustments"] = ConfigSection(
                data=horizon_conf.get("conditional_adjustments", {}),
                source=f"confidence_config.horizons.{self.horizon}.conditional_adjustments"
            )

            self.sections["horizon_adx_confidence_bands"] = ConfigSection(
                data=horizon_conf.get("adx_confidence_bands", {}),
                source=f"confidence_config.horizons.{self.horizon}.adx_confidence_bands"
            )

            self.sections["horizon_adx_confidence_penalties"] = ConfigSection(
                data=horizon_conf.get("adx_confidence_penalties", {}),
                source=f"confidence_config.horizons.{self.horizon}.adx_confidence_penalties"
            )

            self.logger.info("✅ Confidence configs extracted from confidence_config.py")

        except ConfigurationError:
            # Re-raise ConfigurationError — these are critical
            raise
        except Exception as e:
            self.logger.error(f"❌ Failed to extract confidence configs: {e}")
            self._extract_confidence_from_master_config()

    def _extract_confidence_from_master_config(self):
        """Fallback: Extract confidence from master_config (DEPRECATED).
        
        WARNING: This fallback does NOT fully support the modern confidence
        pipeline. Sections like min_tradeable_confidence and
        high_confidence_override are populated with empty/default values
        and marked is_valid=False. Any accessor using get_strict() on
        these sections will surface the deprecation immediately.
        """
        self.logger.warning(
            "⚠️ Using DEPRECATED master_config for confidence — "
            "This fallback will be REMOVED in the next version. "
            "Migrate all confidence settings to confidence_config.py."
        )
        
        global_cfg = self.master_config.get("global", {})
        confidence = global_cfg.get("confidence", {})
        
        # Global sections — best-effort from master_config
        self.sections["confidence_range"] = ConfigSection(
            data={"absolute_min": 0, "absolute_max": 100, "default_clamp": [30, 95]},
            source="master_config.global.confidence (DEPRECATED)"
        )
        
        self.sections["adx_normalization"] = ConfigSection(
            data=confidence.get("adx_normalization", {}),
            source="master_config.global.confidence.adx_normalization (DEPRECATED)"
        )
        
        self.sections["setup_baseline_floors"] = ConfigSection(
            data=confidence.get("base_floors", {}),
            source="master_config.global.confidence.base_floors (DEPRECATED)"
        )
        
        self.sections["volume_modifiers"] = ConfigSection(
            data={},
            source="master_config (DEPRECATED — no volume_modifiers)",
            is_valid=False,
            error="volume_modifiers not available in master_config fallback"
        )
        
        self.sections["universal_adjustments"] = ConfigSection(
            data={},
            source="master_config (DEPRECATED — no universal_adjustments)",
            is_valid=False,
            error="universal_adjustments not available in master_config fallback"
        )
        
        self.sections["divergence_physics"] = ConfigSection(
            data={},
            source="master_config (DEPRECATED — no divergence_physics)"
        )
        
        # Horizon sections — best-effort
        horizon_cfg = self.master_config.get("horizons", {}).get(self.horizon, {})
        horizon_conf = horizon_cfg.get("confidence", {})
        
        self.sections["horizon_confidence_clamp"] = ConfigSection(
            data=[30, 95],  # Default
            source=f"master_config.horizons.{self.horizon}.confidence (DEPRECATED)"
        )
        
        self.sections["horizon_base_confidence_adjustment"] = ConfigSection(
            data=horizon_conf.get("horizon_discount", 0),
            source=f"master_config.horizons.{self.horizon}.confidence.horizon_discount (DEPRECATED)"
        )
        
        # Sections NOT available in master_config — marked invalid
        # get_strict() calls on these will raise ConfigurationError
        self.sections["min_tradeable_confidence"] = ConfigSection(
            data={},
            source=f"master_config (DEPRECATED — no min_tradeable_confidence)",
            is_valid=False,
            error="min_tradeable_confidence requires confidence_config.py"
        )
        
        self.sections["high_confidence_override"] = ConfigSection(
            data={},
            source=f"master_config (DEPRECATED — no high_confidence_override)",
            is_valid=False,
            error="high_confidence_override requires confidence_config.py"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # EXISTING EXTRACTION METHODS (Unchanged)
    # ═══════════════════════════════════════════════════════════════════════

    def extract_global_sections(self):
        """Extract global config sections."""
        global_cfg = self.master_config.get("global", {})

        self.sections["hybrid_metric_registry"] = ConfigSection(
            data=HYBRID_METRIC_REGISTRY,
            source="master_config.HYBRID_METRIC_REGISTRY"
        )
        
        self.sections["hybrid_pillar_composition"] = ConfigSection(
            data=HYBRID_PILLAR_COMPOSITION,
            source="master_config.HYBRID_PILLAR_COMPOSITION"
        )
        
        # ✅ NEW: Extract pillar weights (Pre-sliced by horizon)
        self.sections["horizon_pillar_weights"] = ConfigSection(
            data=HORIZON_PILLAR_WEIGHTS.get(self.horizon, {}),
            source=f"master_config.HORIZON_PILLAR_WEIGHTS.{self.horizon}"
        )



        # Calculation Engine
        calc_engine = global_cfg.get("calculation_engine", {})
        self.sections["spread_adjustment"] = ConfigSection(
            data=calc_engine.get("spread_adjustment", {}),
            source="global.calculation_engine.spread_adjustment"
        )

        # Core Sections
        self.sections["position_sizing"] = ConfigSection(
            data=global_cfg.get("position_sizing", {}),
            source="global.position_sizing"
        )
        self.sections["risk_management"] = ConfigSection(
            data=global_cfg.get("risk_management", {}),
            source="global.risk_management"
        )
        self.sections["indicators"] = ConfigSection(
            data=global_cfg.get("indicators", {}),
            source="global.indicators"
        )
        # ✅ NEW: Extract execution config
        self.sections["execution"] = ConfigSection(
            data=global_cfg.get("execution", {}),
            source="global.execution"
        )
        
        # ✅ NEW: Extract trend weights
        self.sections["trend_weights"] = ConfigSection(
            data=global_cfg.get("trend_weights", {}),
            source="global.trend_weights"
        )
        self.sections["moving_averages"] = ConfigSection(
            data=global_cfg.get("moving_averages", {}),
            source="global.moving_averages"
        )
        self.sections["volatility"] = ConfigSection(
            data=global_cfg.get("volatility", {}),
            source="global.volatility"
        )
        self.sections["boosts"] = ConfigSection(
            data=global_cfg.get("boosts", {}),
            source="global.boosts"
        )
        self.sections["targets"] = ConfigSection(
            data=global_cfg.get("targets", {}),
            source="global.targets"
        )
        self.sections["time_estimation"] = ConfigSection(
            data=global_cfg.get("time_estimation", {}),
            source="global.time_estimation"
        )

        # Global trend/momentum thresholds (fallback for get_trend_thresholds / get_momentum_thresholds)
        self.sections["momentum_thresholds"] = ConfigSection(
            data=global_cfg.get("momentum_thresholds", {}),
            source="global.momentum_thresholds"
        )
        self.sections["trend_thresholds"] = ConfigSection(
            data=global_cfg.get("trend_thresholds", {}),
            source="global.trend_thresholds"
        )

    def extract_horizon_sections(self):
        """Extract horizon-specific config sections."""
        horizon_cfg = self.master_config.get("horizons", {}).get(self.horizon, {})

        if not horizon_cfg:
            self.logger.warning(f"No config found for horizon: {self.horizon}")
            return

        # Horizon-specific overrides
        self.sections["horizon_risk_management"] = ConfigSection(
            data=horizon_cfg.get("risk_management", {}),
            source=f"horizons.{self.horizon}.risk_management"
        )
        self.sections["horizon_execution"] = ConfigSection(
            data=horizon_cfg.get("execution", {}),
            source=f"horizons.{self.horizon}.execution"
        )
        self.sections["horizon_indicators"] = ConfigSection(
            data=horizon_cfg.get("indicators", {}),
            source=f"horizons.{self.horizon}.indicators"
        )
        self.sections["horizon_moving_averages"] = ConfigSection(
            data=horizon_cfg.get("moving_averages", {}),
            source=f"horizons.{self.horizon}.moving_averages"
        )
        self.sections["horizon_volatility"] = ConfigSection(
            data=horizon_cfg.get("volatility", {}),
            source=f"horizons.{self.horizon}.volatility"
        )
        self.sections["horizon_volume_analysis"] = ConfigSection(
            data=horizon_cfg.get("volume_analysis", {}),
            source=f"horizons.{self.horizon}.volume_analysis"
        )
        self.sections["horizon_time_estimation"] = ConfigSection(
            data=horizon_cfg.get("time_estimation", {}),
            source=f"horizons.{self.horizon}.time_estimation"
        )
        # Extract trend thresholds
        self.sections["horizon_trend_thresholds"] = ConfigSection(
            data=horizon_cfg.get("trend_thresholds", {}),
            source=f"horizons.{self.horizon}.trend_thresholds"
        )
        
        # Extract momentum thresholds
        self.sections["horizon_momentum_thresholds"] = ConfigSection(
            data=horizon_cfg.get("momentum_thresholds", {}),
            source=f"horizons.{self.horizon}.momentum_thresholds"
        )

    def extract_risk_sections(self):
        """Extract risk management config sections."""
        global_cfg = self.master_config.get("global", {})

        # Global risk management
        risk_mgmt = global_cfg.get("risk_management", {})
        self.sections["atr_sl_limits"] = ConfigSection(
            data=risk_mgmt.get("atr_sl_limits", {}),
            source="global.risk_management.atr_sl_limits"
        )
        self.sections["rr_regime_adjustments"] = ConfigSection(
            data=risk_mgmt.get("rr_regime_adjustments", {}),
            source="global.risk_management.rr_regime_adjustments"
        )
        self.sections["global_rr_gates"] = ConfigSection(
            data=risk_mgmt.get("rr_gates", {}),
            source="global.risk_management.rr_gates"
        )

        # Horizon risk overrides
        horizon_cfg = self.master_config.get("horizons", {}).get(self.horizon, {})
        horizon_risk = horizon_cfg.get("risk_management", {})
        self.sections["horizon_atr_sl_limits"] = ConfigSection(
            data=horizon_risk.get("atr_sl_limits", {}),
            source=f"horizons.{self.horizon}.risk_management.atr_sl_limits"
        )
        self.sections["horizon_rr_regime_adjustments"] = ConfigSection(
            data=horizon_risk.get("rr_regime_adjustments", {}),
            source=f"horizons.{self.horizon}.risk_management.rr_regime_adjustments"
        )
        self.sections["horizon_rr_gates"] = ConfigSection(
            data=horizon_risk.get("rr_gates", {}),
            source=f"horizons.{self.horizon}.risk_management.rr_gates"
        )

        # Deep merge (horizon overrides global)
        global_rr = risk_mgmt.get("rr_gates", {})
        horizon_rr = horizon_risk.get("rr_gates", {})

        merged_rr = {**global_rr, **horizon_rr}

        self.sections["rr_gates"] = ConfigSection(
            data=merged_rr,
            source=f"merged(global + {self.horizon}) rr_gates"
        )

        # Deep merge regime adjustments (W13)
        global_regime = risk_mgmt.get("rr_regime_adjustments", {})
        horizon_regime = horizon_risk.get("rr_regime_adjustments", {})
        
        merged_regime = {}
        for regime in set(global_regime.keys()).union(horizon_regime.keys()):
            merged_regime[regime] = {
                **global_regime.get(regime, {}),
                **horizon_regime.get(regime, {})
            }
            
        self.sections["rr_regime_adjustments"] = ConfigSection(
            data=merged_regime,
            source=f"merged(global + {self.horizon}) rr_regime_adjustments"
        )

        # ✅ FIXED: Multipliers and Overrides consolidated here
        # Priority overrides (moved from extract_setup_sections)
        calc_engine = global_cfg.get("calculation_engine", {})
        priority_overrides = calc_engine.get("horizon_priority_overrides", {})
        self.sections["horizon_priority_overrides"] = ConfigSection(
            data=priority_overrides.get(self.horizon, {}),
            source=f"global.calculation_engine.horizon_priority_overrides.{self.horizon}"
        )

        # Position Sizing Multipliers (moved from extract_strategy_sections)
        # ✅ FIXED: Only extract horizon-specific multipliers here.
        # QueryOptimizedExtractor handles global multipliers separately.
        horizon_mults = horizon_risk.get("setup_size_multipliers", {})
        
        self.sections["sizing_multipliers"] = ConfigSection(
            data=horizon_mults,
            source=f"horizons.{self.horizon}.risk_management.setup_size_multipliers"
        )
        
        # Horizon base multiplier (defaults to 1.0 if not found)
        self.sections["horizon_base_multiplier"] = ConfigSection(
            data=horizon_risk.get("base_multiplier", 1.0),
            source=f"horizons.{self.horizon}.risk_management.base_multiplier"
        )

    def extract_gate_sections(self):
        """
        ✅ FIXED: Extract gates with NEW architecture.
        
        Gate Hierarchy (priority order):
        1. Global gates (master_config.global.entry_gates)
        2. Horizon gates (master_config.horizons.X.entry_gates)
        3. Setup gates (setup_pattern_matrix.SETUP.context_requirements)
        4. Setup horizon overrides (setup_pattern_matrix.SETUP.horizon_overrides.X)
        """
        global_cfg = self.master_config.get("global", {})

        # ✅ STEP 1: Universal gates (master_config only)
        entry_gates = global_cfg.get("entry_gates", {})
        
        self.sections["structural_gates"] = ConfigSection(
            data=entry_gates.get("structural", {}).get("gates", {}),
            source="global.entry_gates.structural.gates"
        )
        
        self.sections["execution_rules"] = ConfigSection(
            data=entry_gates.get("execution_rules", {}),
            source="global.entry_gates.execution_rules"
        )
        
        self.sections["opportunity_gates"] = ConfigSection(
            data=entry_gates.get("opportunity", {}).get("gates", {}),
            source="global.entry_gates.opportunity.gates"
        )
        #check setupmatrix
        # self.sections["setup_gate_specifications"] = ConfigSection(
        #     data=entry_gates.get("setup_gate_specifications", {}),
        #     source="global.entry_gates.setup_gate_specifications"
        # )

        # ✅ STEP 2: Horizon gate overrides (master_config only)
        horizon_cfg = self.master_config.get("horizons", {}).get(self.horizon, {})
        horizon_gates = horizon_cfg.get("entry_gates", {})
        
        self.sections["horizon_structural_gates"] = ConfigSection(
            data=horizon_gates.get("structural", {}),
            source=f"horizons.{self.horizon}.entry_gates.structural"
        )
        
        self.sections["horizon_opportunity_gates"] = ConfigSection(
            data=horizon_gates.get("opportunity", {}),
            source=f"horizons.{self.horizon}.entry_gates.opportunity"
        )
        # self.sections["horizon_gate_overrides"] = ConfigSection(
        #     data=horizon_gates.get("setup_gate_overrides", {}),
        #     source=f"horizons.{self.horizon}.entry_gates.setup_gate_overrides"
        # )
        
        self.sections["horizon_execution_rules"] = ConfigSection(
            data=horizon_gates.get("execution_rules", {}),
            source=f"horizons.{self.horizon}.entry_gates.execution_rules"
        )

        # ❌ REMOVED: setup_gate_specifications extraction
        # This is now in setup_pattern_matrix.py, extracted in extract_matrix_sections()
        
        self.logger.info(
            f"✅ Gate sections extracted (universal + horizon only, "
            f"setup gates in matrix)"
        )
    
    def extract_matrix_sections(self):
        """
        Extract configs from setup_pattern_matrix and strategy_matrix.
        Comprehensive pattern and strategy data extraction.
        """
        try:
            from config.setup_pattern_matrix_config import (
                SETUP_PATTERN_MATRIX,
                PATTERN_METADATA,
                DEFAULT_PHYSICS,
                PATTERN_SCORING_THRESHOLDS
            )

            # Full setup-pattern matrix
            self.sections["setup_pattern_matrix"] = ConfigSection(
                data=SETUP_PATTERN_MATRIX,
                source="setup_pattern_matrix.SETUP_PATTERN_MATRIX"
            )

            # Pattern metadata (physics, entry rules, invalidation)
            self.sections["pattern_metadata"] = ConfigSection(
                data=PATTERN_METADATA,
                source="setup_pattern_matrix.PATTERN_METADATA"
            )

            # Pattern scoring thresholds
            self.sections["pattern_scoring_thresholds"] = ConfigSection(
                data=PATTERN_SCORING_THRESHOLDS,
                source="setup_pattern_matrix.PATTERN_SCORING_THRESHOLDS"
            )

            # Pattern indicator mappings (horizon-aware)
            self.sections["pattern_indicator_mappings"] = ConfigSection(
                data=locals().get('PATTERN_INDICATOR_MAPPINGS', {}),
                source="setup_pattern_matrix.PATTERN_INDICATOR_MAPPINGS"
            )

            # Default physics
            self.sections["default_physics"] = ConfigSection(
                data=DEFAULT_PHYSICS,
                source="setup_pattern_matrix.DEFAULT_PHYSICS"
            )

            # Extract individual setup configs for quick access
            for setup_name, setup_config in SETUP_PATTERN_MATRIX.items():
                self.sections[f"setup_{setup_name}"] = ConfigSection(
                    data=setup_config,
                    source=f"setup_pattern_matrix.{setup_name}"
                )

                # Extract validation modifiers separately
                validation_mods = setup_config.get("validation_modifiers", {})
                if validation_mods:
                    self.sections[f"setup_validation_{setup_name}"] = ConfigSection(
                        data=validation_mods,
                        source=f"setup_pattern_matrix.{setup_name}.validation_modifiers"
                    )

            # Extract individual pattern metadata for quick access
            for pattern_name, pattern_meta in PATTERN_METADATA.items():
                self.sections[f"pattern_{pattern_name}"] = ConfigSection(
                    data=pattern_meta,
                    source=f"pattern_metadata.{pattern_name}"
                )

                # Extract pattern physics
                physics = pattern_meta.get("physics", {})
                if physics:
                    self.sections[f"pattern_physics_{pattern_name}"] = ConfigSection(
                        data=physics,
                        source=f"pattern_metadata.{pattern_name}.physics"
                    )

                # Extract entry rules per horizon
                entry_rules = pattern_meta.get("entry_rules", {})
                if entry_rules:
                    self.sections[f"pattern_entry_{pattern_name}"] = ConfigSection(
                        data=entry_rules,
                        source=f"pattern_metadata.{pattern_name}.entry_rules"
                    )

                # Extract invalidation logic per horizon
                invalidation = pattern_meta.get("invalidation", {})
                if invalidation:
                    self.sections[f"pattern_invalidation_{pattern_name}"] = ConfigSection(
                        data=invalidation,
                        source=f"pattern_metadata.{pattern_name}.invalidation"
                )

            # ✅ NEW: Extract setup-specific gate configurations
            for setup_name, setup_config in SETUP_PATTERN_MATRIX.items():
                # Extract context requirements (base setup gates)
                context_reqs = setup_config.get("context_requirements", {})
                if context_reqs:
                    self.sections[f"setup_context_{setup_name}"] = ConfigSection(
                        data=context_reqs,
                        source=f"setup_pattern_matrix.{setup_name}.context_requirements"
                    )

                # ✅ CRITICAL: Extract horizon_overrides for setup
                horizon_overrides = setup_config.get("horizon_overrides", {})
                if horizon_overrides:
                    self.sections[f"setup_horizon_overrides_{setup_name}"] = ConfigSection(
                        data=horizon_overrides,
                        source=f"setup_pattern_matrix.{setup_name}.horizon_overrides"
                    )
                    
                    # Extract specific horizon override for this instance
                    horizon_override = horizon_overrides.get(self.horizon, {})
                    if horizon_override:
                        self.sections[f"setup_{setup_name}_override_{self.horizon}"] = ConfigSection(
                            data=horizon_override,
                            source=f"setup_pattern_matrix.{setup_name}.horizon_overrides.{self.horizon}"
                        )

            self.logger.info("✅ Setup gates and horizon overrides extracted from matrix")

        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import setup_pattern_matrix: {e}")
            self.sections["setup_pattern_matrix"] = ConfigSection(
                data={},
                source="setup_pattern_matrix.SETUP_PATTERN_MATRIX",
                is_valid=False,
                error=str(e)
            )

        # Extract strategy matrix
        try:
            from config.strategy_matrix_config import STRATEGY_MATRIX

            # Full strategy matrix
            self.sections["strategy_matrix"] = ConfigSection(
                data=STRATEGY_MATRIX,
                source="strategy_matrix.STRATEGY_MATRIX"
            )

            # Extract individual strategy configs
            for strategy_name, strategy_config in STRATEGY_MATRIX.items():
                self.sections[f"strategy_{strategy_name}"] = ConfigSection(
                    data=strategy_config,
                    source=f"strategy_matrix.{strategy_name}"
                )

                # Extract fit indicators
                fit_indicators = strategy_config.get("fit_indicators", {})
                if fit_indicators:
                    self.sections[f"strategy_fit_{strategy_name}"] = ConfigSection(
                        data=fit_indicators,
                        source=f"strategy_matrix.{strategy_name}.fit_indicators"
                    )

                # Extract scoring rules
                scoring_rules = strategy_config.get("scoring_rules", {})
                if scoring_rules:
                    self.sections[f"strategy_scoring_{strategy_name}"] = ConfigSection(
                        data=scoring_rules,
                        source=f"strategy_matrix.{strategy_name}.scoring_rules"
                    )

                # Extract market cap requirements
                market_cap_reqs = strategy_config.get("market_cap_requirements", {})
                if market_cap_reqs:
                    self.sections[f"strategy_market_cap_{strategy_name}"] = ConfigSection(
                        data=market_cap_reqs,
                        source=f"strategy_matrix.{strategy_name}.market_cap_requirements"
                    )

            self.logger.info("✅ Strategy matrix extracted successfully")

        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import strategy_matrix: {e}")
            self.sections["strategy_matrix"] = ConfigSection(
                data={},
                source="strategy_matrix.STRATEGY_MATRIX",
                is_valid=False,
                error=str(e)
            )

    def extract_strategy_sections(self):
        """Extract strategy configurations.

        C8 FIX: This method used to unconditionally overwrite self.sections["strategy_matrix"]
        with a stripped-down version missing fit_indicators / scoring_rules / market_cap_requirements.
        extract_matrix_sections() already populates the full rich version. Skip the overwrite
        if the section was already populated by that call.
        """
        if "strategy_matrix" not in self.sections or not self.sections["strategy_matrix"].is_valid:
            # Only load if extract_matrix_sections() did NOT already populate it
            try:
                from config.strategy_matrix_config import STRATEGY_MATRIX

                self.sections["strategy_matrix"] = ConfigSection(
                    data=STRATEGY_MATRIX,
                    source="strategy_matrix.STRATEGY_MATRIX"
                )

                for strat_name, strat_config in STRATEGY_MATRIX.items():
                    self.sections[f"strategy_{strat_name}"] = ConfigSection(
                        data=strat_config,
                        source=f"strategy_matrix.{strat_name}"
                    )

            except (ImportError, Exception) as e:
                self.logger.warning(f"Could not load strategy_matrix_config: {e}")
        else:
            self.logger.debug(
                "extract_strategy_sections: strategy_matrix already loaded by "
                "extract_matrix_sections() — skipping overwrite (C8 fix)"
            )

        # Horizon strategy preferences
        horizon_cfg = self.master_config.get("horizons", {}).get(self.horizon, {})
        prefs = horizon_cfg.get("strategy_preferences", {})
        
        self.sections["strategy_preferences"] = ConfigSection(
            data=prefs,
            source=f"horizons.{self.horizon}.strategy_preferences"
        )
        
        # Populate property-linked sections
        self.sections["blocked_setups"] = ConfigSection(
            data=set(prefs.get("blocked_setups", [])),
            source=f"horizons.{self.horizon}.strategy_preferences.blocked_setups"
        )
        
        self.sections["preferred_setups"] = ConfigSection(
            data=prefs.get("preferred_setups", []),
            source=f"horizons.{self.horizon}.strategy_preferences.preferred_setups"
        )
        
        self.sections["blocked_strategies"] = ConfigSection(
            data=set(prefs.get("blocked_strategies", [])),
            source=f"horizons.{self.horizon}.strategy_preferences.blocked_strategies"
        )
        
        self.sections["strategy_multipliers"] = ConfigSection(
            data=prefs.get("strategy_multipliers", {}),
            source=f"horizons.{self.horizon}.strategy_preferences.strategy_multipliers"
        )

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def validate_extracted_configs(self):
        """Validate critical configs are present and valid."""
        critical_sections = [
            "position_sizing",
            "risk_management",
            "structural_gates",
            "opportunity_gates",
            "setup_pattern_matrix",  # ✅ P1-3 FIX
            "strategy_matrix"        # ✅ P1-3 FIX
        ]

        missing = []
        for section_name in critical_sections:
            section = self.sections.get(section_name)
            if not section or not section.data:
                missing.append(section_name)

        if missing:
            self.logger.error(f"Missing critical config sections: {missing}")
            raise ValueError(f"Missing critical config sections: {missing}")

        # Run specific validations
        self.validate_gate_structure()
        self.validate_risk_management()
        self.validate_threshold_consistency()  # ✅ P3-1 FIX: Enable consistency check
        
        # ✅ NEW: Validate confidence config
        if self.has_confidence_config:
            self.validate_confidence_structure()

    def validate_confidence_structure(self):
        """Validate confidence config structure."""
        required_global = [
            "confidence_range",
            "adx_normalization",
            "volume_modifiers",
            "universal_adjustments",
            "setup_baseline_floors"
        ]
        
        # ✅ P1-5 & P3-4 FIX: check is_valid instead of truthiness
        missing_global = [
            k for k in required_global 
            if not self.sections.get(k) or not self.sections[k].is_valid
        ]
        
        if missing_global:
            self.logger.error(f"Missing confidence config sections: {missing_global}")
            raise ValueError(f"Missing confidence config sections: {missing_global}")
            
        # ✅ Phase 3 P1-3 FIX: Validate execution confidence adjustments
        exec_cfg = self.sections.get("execution")
        if exec_cfg and exec_cfg.is_valid:
            conf_adj = exec_cfg.data.get("confidence_adjustments", {})
            required_adj_keys = ["warning_penalty", "violation_penalty", "risk_score_thresholds"]
            missing_adj = [k for k in required_adj_keys if k not in conf_adj]
            if missing_adj:
                 self.logger.error(f"execution.confidence_adjustments missing keys: {missing_adj}")
                 raise ValueError(f"execution.confidence_adjustments missing keys: {missing_adj}")
        
        # Validate horizon-specific (REQUIRED sections)
        required_horizon = [
            "horizon_confidence_clamp",
            "horizon_base_confidence_adjustment",
            "horizon_setup_floor_overrides",
            "horizon_conditional_adjustments",
            "min_tradeable_confidence",
            "high_confidence_override"
        ]
        
        # ✅ P1-5 & P3-4 FIX: check is_valid instead of None
        missing_horizon = [
            k for k in required_horizon 
            if not self.sections.get(k) or not self.sections[k].is_valid or self.sections[k].data is None
        ]
        
        if missing_horizon:
            msg = (
                f"CRITICAL: Missing required horizon confidence sections "
                f"for '{self.horizon}': {missing_horizon}"
            )
            self.logger.critical(msg)
            raise ConfigurationError(msg)

    def validate_gate_structure(self):
        """Validate gate structure is correct."""
        structural = self.sections.get("structural_gates")
        if structural and structural.data:
            # Just check that we have some gates defined
            if not structural.data:
                self.logger.warning("Structural gates are empty")

    def validate_risk_management(self):
        """Validate risk management configs."""
        risk = self.sections.get("risk_management")
        if risk and risk.data:
            required_keys = ["max_position_pct"]
            missing = [k for k in required_keys if k not in risk.data]

            if missing:
                self.logger.error(
                    f"Risk management missing keys: {missing} "
                    f"(source: {risk.source})"
                )
                raise ValueError(f"Risk management missing keys: {missing}")

    def validate_threshold_consistency(self):
        """
        Validate min/max threshold pairs are logical.
        Comprehensive threshold validation.
        """
        errors = []

        # Check setup-pattern matrix thresholds
        setup_matrix = self.sections.get("setup_pattern_matrix")
        if setup_matrix and setup_matrix.data:
            for setup_name, setup_config in setup_matrix.data.items():
                context_reqs = setup_config.get("context_requirements", {})

                # Validate technical thresholds
                for metric, constraints in context_reqs.get("technical", {}).items():
                    if not isinstance(constraints, dict):
                        continue

                    min_val = constraints.get("min")
                    max_val = constraints.get("max")

                    if min_val is not None and max_val is not None:
                        if min_val > max_val:
                            errors.append(
                                f"{setup_name}.technical.{metric}: "
                                f"min ({min_val}) > max ({max_val})"
                            )

                # Validate fundamental thresholds
                for metric, constraints in context_reqs.get("fundamental", {}).items():
                    if not isinstance(constraints, dict):
                        continue

                    min_val = constraints.get("min")
                    max_val = constraints.get("max")

                    if min_val is not None and max_val is not None:
                        if min_val > max_val:
                            errors.append(
                                f"{setup_name}.fundamental.{metric}: "
                                f"min ({min_val}) > max ({max_val})"
                            )

        # Check structural gates
        structural = self.sections.get("structural_gates")
        if structural and structural.data:
            for metric, constraints in structural.data.items():
                if not isinstance(constraints, dict):
                    continue

                min_val = constraints.get("min")
                max_val = constraints.get("max")

                if min_val is not None and max_val is not None:
                    if min_val > max_val:
                        errors.append(
                            f"structural_gates.{metric}: "
                            f"min ({min_val}) > max ({max_val})"
                        )

        if errors:
            self.logger.error(f"Threshold validation errors: {errors}")
            raise ValueError(f"Invalid threshold configurations: {errors}")

        self.logger.info(f"✅ Threshold validation passed")

    # ═══════════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════════

    def get(self, section_name: str, default: Any = None) -> Any:
        """
        Get a config section safely.
        Returns a DEEP COPY of the data if valid (P0-3 FIX), otherwise returns default.
        Logs warnings if section is missing or invalid.
        
        For required config paths, use get_strict() instead.
        For performance-critical READ-ONLY access, use get_readonly().
        """
        section = self.sections.get(section_name)

        if not section:
            # ✅ PATCH D: Suppress noise for known optional sections
            is_optional = any(
                section_name.startswith(prefix) 
                for prefix in ("strategy_market_cap_", "setup_")
            )
            if not is_optional:
                self.logger.debug(f"Config section not found: {section_name}")
            return default

        if not section.is_valid:
            self.logger.warning(
                f"Config section invalid: {section_name} - {section.error}"
            )
            return default

        # ✅ P0-3 FIX: Always return a deepcopy to prevent cache corruption
        return copy.deepcopy(section.data)

    def get_readonly(self, section_name: str, default: Any = None) -> Any:
        """
        Performance-optimized getter that returns a LIVE REFERENCE.
        
        WARNING: Callers MUST NOT mutate the returned object, as it will
        corrupt the extractor's internal cache for all future queries.
        """
        section = self.sections.get(section_name)
        if not section or not section.is_valid:
            return default
        return section.data

    def get_strict(self, section_name: str) -> Any:
        """
        Get a config section with STRICT validation.
        
        Raises ConfigurationError if section is missing or invalid.
        Use this for config paths essential to trade validation.
        
        Unlike get_required() which raises ValueError, this raises
        ConfigurationError with a CRITICAL log — signaling a config
        integrity failure that must be fixed before trading.
        """
        section = self.sections.get(section_name)

        if not section:
            msg = f"CRITICAL: Required config section '{section_name}' not found"
            self.logger.critical(msg)
            raise ConfigurationError(msg)

        if not section.is_valid:
            msg = (
                f"CRITICAL: Config section '{section_name}' is invalid: "
                f"{section.error} (source: {section.source})"
            )
            self.logger.critical(msg)
            raise ConfigurationError(msg)

        return section.data

    def get_required(self, section_name: str) -> Any:
        """
        Get a required config section.
        Raises ValueError if section is missing or invalid.
        """
        section = self.sections.get(section_name)

        if not section:
            raise ValueError(f"Required config section not found: {section_name}")

        if not section.is_valid:
            raise ValueError(
                f"Required config section invalid: {section_name} - {section.error}"
            )

        return section.data

    def get_merged(self, global_key: str, horizon_key: str) -> Dict[str, Any]:
        """
        Get merged config (global + horizon override) with RECURSIVE deep merge (P2-4 FIX).

        Example:
            get_merged("risk_management", "horizon_risk_management")
            get_merged("execution_rules", "horizon_execution_rules")
        """
        global_cfg = self.get(global_key, {})
        horizon_cfg = self.get(horizon_key, {})

        return self._deep_merge(global_cfg, horizon_cfg)

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base. Override wins on conflicts (P2-4)."""
        result = copy.deepcopy(base)
        for key, value in override.items():
            if (key in result and isinstance(result[key], dict) 
                    and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def get_config_value(
        self,
        section_name: str,
        key_path: str,
        default: Any = None,
        value_type: str = "auto"
    ) -> Any:
        """
        Universal config value getter with type coercion.

        Args:
            section_name: Config section name
            key_path: Dot-separated path (e.g., "technical.rvol.min")
            default: Default value if not found
            value_type: "auto", "float", "int", "bool", "str"

        Example:
            extractor.get_config_value("structural_gates", "adx.min", 20.0, "float")
        """
        section = self.sections.get(section_name)
        if not section or not section.is_valid:
            return default

        # Navigate nested path
        current = section.data
        for part in key_path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            else:
                return default

            if current is None:
                return default

        # Type coercion
        if value_type == "float":
            try:
                return float(current) if current is not None else default
            except (ValueError, TypeError):
                return default
        elif value_type == "int":
            try:
                return int(current) if current is not None else default
            except (ValueError, TypeError):
                return default
        elif value_type == "bool":
            return bool(current)
        elif value_type == "str":
            return str(current) if current is not None else default
        else:
            # Auto - return as-is
            return current

    def is_rule_enabled(self, section_name: str, rule_name: str = None) -> bool:
        """
        Check if a config rule/section is enabled.

        Args:
            section_name: Name of the config section
            rule_name: Optional specific rule within section

        Returns:
            Boolean enabled status (defaults to True if not specified)

        Example:
            extractor.is_rule_enabled("strategy_matrix", "minervini_growth")
        """
        section = self.sections.get(section_name)
        if not section or not section.is_valid:
            return False

        data = section.data

        if rule_name:
            # Check specific rule within section
            if isinstance(data, dict):
                rule = data.get(rule_name, {})
                if isinstance(rule, dict):
                    return rule.get("enabled", True)  # Default to enabled
                else:
                    return True  # If rule exists but no enabled flag, assume enabled
            return False
        else:
            # Check entire section
            if isinstance(data, dict):
                return data.get("enabled", True)
            return True

    def list_sections(self) -> List[str]:
        """List all extracted section names."""
        return list(self.sections.keys())

    def get_source(self, section_name: str) -> Optional[str]:
        """Get the source path of a config section."""
        section = self.sections.get(section_name)
        return section.source if section else None

    # ═══════════════════════════════════════════════════════════════════════
    # PROPERTY SHORTCUTS
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def blocked_setups(self) -> Set[str]:
        """Get blocked setups for this horizon."""
        return self.get("blocked_setups", set())

    @property
    def preferred_setups(self) -> List[str]:
        """Get preferred setups for this horizon."""
        return self.get("preferred_setups", [])

    @property
    def blocked_strategies(self) -> Set[str]:
        """Get blocked strategies for this horizon."""
        return self.get("blocked_strategies", set())

    @property
    def strategy_multipliers(self) -> Dict[str, float]:
        """Get strategy priority multipliers for this horizon."""
        return self.get("strategy_multipliers", {})

    @property
    def sizing_multipliers(self) -> Dict[str, float]:
        """Get sizing multipliers for this horizon."""
        return self.get("sizing_multipliers", {})

    @property
    def horizon_priority_overrides(self) -> Dict[str, float]:
        """Get horizon-specific setup priority overrides."""
        return self.get("horizon_priority_overrides", {})