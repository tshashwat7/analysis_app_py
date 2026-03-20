# config/multibagger/multibagger_master_config.py
"""
Multibagger Master Configuration
=================================
Passed directly to MBConfigExtractor(MB_MASTER_CONFIG, "multibagger").

ARCHITECTURE NOTES:
- This is a full master_config-shaped dict. Every key that
  ConfigExtractor.extract_global_sections() and extract_horizon_sections()
  call .get() on must be present here (empty dict is fine — no KeyError).
- setup_pattern_matrix and strategy_matrix are at the ROOT LEVEL so that
  MBConfigExtractor.extract_matrix_sections() can read them via
  self.master_config.get("setup_pattern_matrix") without any file import.
- MB_HORIZON_PILLAR_WEIGHTS and MB_HYBRID_PILLAR_COMPOSITION are module-level
  constants imported directly by MBConfigExtractor to overwrite the sections
  that ConfigExtractor bakes in from the main master_config module-level import.
- confidence_config is handled separately via MBConfigExtractor.__init__ override.

TIMEFRAME: Weekly (1wk).
MA KEYS: maFast=MMA(6w), maMid=MMA(12w), maSlow=MMA(24w).
"""

from config.master_config import HYBRID_METRIC_REGISTRY, GATE_METRIC_REGISTRY

# ============================================================================
# MODULE-LEVEL CONSTANTS
# Imported by MBConfigExtractor to overwrite baked-in module-level imports.
# ============================================================================

MB_HYBRID_PILLAR_COMPOSITION = {
    "multibagger": {
        # Fundamentals dominate the hybrid pillar for long-hold thesis
        "volatilityAdjustedRoe":    0.05,  # ROE quality vs technical price risk
        "fundamentalMomentum":      0.25,  # Growth + EPS composite
        "earningsConsistencyIndex": 0.25,  # ROE + NPM quality composite
        "priceToIntrinsicValue":    0.20,  # IV-based valuation
        "fcfYieldVsVolatility":     0.15,  # FCF quality vs price risk
        "trendConsistency":         0.05,  # Minimal — entry timing only
        "priceVsPrimaryTrendPct":   0.05,  # Distance from primary trend
    }
}

MB_HORIZON_PILLAR_WEIGHTS = {
    "multibagger": {
        "tech":   0.10,  # Minimal — technicals are entry timing only
        "fund":   0.60,  # Primary — business quality over 5 years
        "hybrid": 0.30,  # FCF, IV, earnings consistency cross-pillar
    }
}

# ============================================================================
# MASTER CONFIG DICT
# ============================================================================

MB_MASTER_CONFIG = {

    # ========================================================================
    # GLOBAL — mirrors the keys extract_global_sections() reads
    # ========================================================================
    "global": {

        "time_estimation": {
            "candles_per_unit": 1,
            "base_friction": 0.8,
            "velocity_factors": {
                "strong_trend": {"min_strength": 6.0, "factor": 1.2},
                "normal_trend": {"min_strength": 4.0, "factor": 1.0},
                "weak_trend":   {"max_strength": 4.0, "factor": 0.8},
            },
        },

        "entry_gates": {
            "structural": {
                # Weekly structural gates — very loose, fundamentals drive rejection
                "gates": {
                    "trendStrength": {"min": 3.0},
                    "atrPct":        {"min": 1.0},
                    # All other structural gates disabled for weekly holds
                    "adx":                 {"min": None},
                    "volatilityQuality":   {"min": None},
                }
            },
            "execution_rules": {
                # Execution rules disabled — MB is a thesis-driven buy, not a day trade
                "volatility_guards":         {"enabled": False},
                "sl_distance_validation":    {"enabled": False},
                "structure_validation":      {"enabled": False},
                "target_proximity_rejection": {"enabled": False},
            },
            "opportunity": {
                "gates": {
                    "confidence":       {"min": 60},
                    "rrRatio":          {"min": 2.0},
                    "fundamentalScore": {"min": 8.0},
                    "hybridScore":      {"min": None},
                }
            },
        },

        "calculation_engine": {
            "spread_adjustment": {},
            "horizon_priority_overrides": {
                "multibagger": {}
            },
        },

        "position_sizing": {
            "base_risk_pct":    0.02,
            "max_position_pct": 0.05,
        },

        "risk_management": {
            "base_risk_pct":     0.02,
            "max_position_pct":  0.05,
            "rr_gates": {
                "min_t1":         2.5,
                "min_t2":         4.0,
                "min_structural": 5.0,
                "execution_floor": 1.5,
            },
            "atr_sl_limits": {
                "max_percent": 0.10,
                "min_percent": 0.02,
            },
            "rr_regime_adjustments": {
                "strong_trend": {"adx": {"min": 25}, "t1_mult": 3.0, "t2_mult": 10.0},
                "normal_trend": {"adx": {"min": 15}, "t1_mult": 2.5, "t2_mult": 8.0},
                "weak_trend":   {"adx": {"max": 15}, "t1_mult": 2.0, "t2_mult": 6.0},
            },
        },

        "execution": {
            "stop_loss_atr_mult": 3.0,
            "target_atr_mult":    10.0,
            "max_hold_candles":   60,
            "dip_buy_reference":  "maSlow",
            "risk_reward_min":    2.0,
            "base_hold_days":     180,
            "min_profit_pct":     2.0,
        },

        "targets": {
            "t1_atr_mult": 3.0,
            "t2_atr_mult": 8.0,
        },

        # Keys extract_global_sections reads — empty dicts prevent KeyError
        "indicators":          {},
        "trend_weights":       {},
        "boosts":              {"volatility": {}},
        "moving_averages":     {},
        "volatility":          {},
        "momentum_thresholds": {},
        "trend_thresholds":    {},
    },

    # ========================================================================
    # HORIZONS — only "multibagger" needed
    # ========================================================================
    "horizons": {
        "multibagger": {
            "timeframe":   "1wk",
            "description": "Deep Value & Compounders — Weekly Chart",

            "time_estimation": {"candles_per_unit": 1},

            "moving_averages": {
                # These are documentation only — actual MA keys (maFast/maMid/maSlow)
                # are set by indicators.py internal logic for the "multibagger" horizon.
                # MMA(6w) = maFast, MMA(12w) = maMid, MMA(24w) = maSlow
                "type": "MMA",
                "fast": 6, "mid": 12, "slow": 24,
            },

            "indicators": {
                "rsi_period": 14,
                "adx_period": 14,
                "atr_period": 20,
            },

            "volume_analysis": {
                "rvol": {"surge_threshold": 1.5, "drought_threshold": 0.7}
            },

            "trend_thresholds": {
                "slope": {"strong": 20.0, "moderate": 8.0}
            },

            "momentum_thresholds": {
                "rsislope": {
                    "acceleration_floor":  0.02,
                    "deceleration_ceiling": -0.02,
                }
            },

            "volatility": {
                "scoring_thresholds": {
                    "atrPct": {"excellent": 12.0, "good": 20.0, "fair": 30.0, "poor": 40.0}
                }
            },

            "position_sizing": {
                "base_risk_pct":        0.02,
                "setup_size_multipliers": {
                    "QUALITY_ACCUMULATION": 1.5,
                    "DEEP_VALUE_PLAY":      1.2,
                },
                "base_multiplier": 1.0,
            },

            "risk_management": {
                "max_position_pct": 0.05,
                "setup_size_multipliers": {
                    "QUALITY_ACCUMULATION": 1.5,
                    "DEEP_VALUE_PLAY":      1.2,
                },
                "base_multiplier": 1.0,
                "atr_sl_limits": {"max_percent": 0.10, "min_percent": 0.02},
                "rr_gates": {
                    "min_t1": 2.5, "min_t2": 4.0,
                    "min_structural": 5.0, "execution_floor": 1.5,
                },
                "rr_regime_adjustments": {
                    "strong_trend": {"adx": {"min": 25}, "t1_mult": 3.0, "t2_mult": 10.0},
                    "normal_trend": {"adx": {"min": 15}, "t1_mult": 2.5, "t2_mult": 8.0},
                    "weak_trend":   {"adx": {"max": 15}, "t1_mult": 2.0, "t2_mult": 6.0},
                },
                "horizon_t2_cap": 1.00,
            },

            "execution": {
                "stop_loss_atr_mult": 3.0,
                "target_atr_mult":    10.0,
                "max_hold_candles":   60,
                "dip_buy_reference":  "maSlow",
                "risk_reward_min":    2.0,
                "base_hold_days":     180,
                "min_profit_pct":     2.0,
                "proximity_rejection": {
                    "resistance_mult": 1.02,
                    "support_mult":    0.98,
                },
            },

            "lookback": {"python_data": 3000},

            "entry_gates": {
                # Flat gate dict (no nested "gates" subkey) — matches
                # how extract_gate_sections reads horizon entry_gates.structural
                "structural": {
                    "trendStrength": {"min": 3.0},
                    "atrPct":        {"min": 1.0},
                },
                "execution_rules": {
                    "volatility_guards":         {"enabled": False},
                    "sl_distance_validation":    {"enabled": False},
                    "structure_validation":      {"enabled": False},
                    "target_proximity_rejection": {"enabled": False},
                },
                "opportunity": {
                    "confidence":       {"min": 60},
                    "rrRatio":          {"min": 2.0},
                    "fundamentalScore": {"min": 8.0},
                },
            },
        }
    },

    # ========================================================================
    # SETUP PATTERN MATRIX (root level — read by MBConfigExtractor)
    # ========================================================================
    "setup_pattern_matrix": {
        "QUALITY_ACCUMULATION": {
            "enabled":          True,
            "default_priority": 95,
            "patterns": {
                "PRIMARY":    ["minerviniStage2", "cupHandle"],
                "CONFIRMING": ["bollingerSqueeze", "goldenCross"],
                "CONFLICTING": ["headShoulders", "deathCross"],
            },
            "classification_rules": {
                "pattern_detection":  {},  # No hard pattern requirement for weekly
                "technical_gates": {
                    "trendStrength": {"min": 3.0},
                },
                "fundamental_gates": {
                    "roe":        {"min": 15},
                    "epsGrowth5y": {"min": 10},
                },
                "require_fundamentals": True,
            },
            "context_requirements": {
                "technical": {
                    "trendStrength": {"min": 3.0},
                    "rvol":          {"min": 0.8},
                },
                "fundamental": {"required": True},
            },
            "validation_modifiers": {
                "penalties": {
                    "weak_weekly_trend": {
                        "gates": {"trendStrength": {"max": 2.0}},
                        "confidence_penalty": 15,
                        "reason": "Trend too weak for accumulation thesis",
                    }
                },
                "bonuses": {
                    "strong_weekly_rvol": {
                        "gates": {"rvol": {"min": 1.5}},
                        "confidence_boost": 10,
                        "reason": "Institutional accumulation volume",
                    }
                },
            },
            "min_pattern_quality": 0,    # No hard pattern required weekly
            "min_setup_score":     60,
            "setup_type":          "fundamental_driven",
            "horizon_overrides":   {},
        },

        "DEEP_VALUE_PLAY": {
            "enabled":          True,
            "default_priority": 80,
            "patterns": {
                "PRIMARY":    ["goldenCross"],
                "CONFIRMING": ["bollingerSqueeze"],
                "CONFLICTING": ["headShoulders"],
            },
            "classification_rules": {
                "pattern_detection":  {},
                "technical_gates": {
                    "trendStrength": {"min": 2.0},
                },
                "fundamental_gates": {
                    "piotroskiF": {"min": 6},
                    "deRatio":    {"max": 1.5},
                },
                "require_fundamentals": True,
            },
            "context_requirements": {
                "technical":    {"trendStrength": {"min": 2.0}},
                "fundamental":  {"required": True},
            },
            "validation_modifiers": {"penalties": {}, "bonuses": {}},
            "min_pattern_quality": 0,
            "min_setup_score":     55,
            "setup_type":          "fundamental_driven",
            "horizon_overrides":   {},
        },

        "VALUE_TURNAROUND": {
            "enabled":          True,
            "default_priority": 75,
            "patterns": {
                "PRIMARY":    ["goldenCross", "reversal"],
                "CONFIRMING": ["bollingerSqueeze"],
                "CONFLICTING": [],
            },
            "classification_rules": {
                "pattern_detection": {},
                "technical_gates": {
                    "trendStrength": {"min": 2.5},
                    "maTrendSignal": {"min": 0},  # At least neutral
                },
                "fundamental_gates": {
                    "roe":  {"min": 12},
                    "roce": {"min": 12},
                },
                "require_fundamentals": True,
            },
            "context_requirements": {
                "technical":   {"trendStrength": {"min": 2.5}},
                "fundamental": {"required": True},
            },
            "validation_modifiers": {"penalties": {}, "bonuses": {}},
            "min_pattern_quality": 0,
            "min_setup_score":     55,
            "setup_type":          "fundamental_driven",
            "horizon_overrides":   {},
        },
    },

    # ========================================================================
    # STRATEGY MATRIX (root level — read by MBConfigExtractor)
    # ========================================================================
    "strategy_matrix": {
        "quality_compounder": {
            "enabled":     True,
            "description": "High-quality compounding businesses with sustained growth.",
            "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
            "avoid_setups":     ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
            "horizon_fit_multipliers": {
                "intraday":    0.0,
                "short_term":  0.3,
                "long_term":   0.8,
                "multibagger": 1.5,
            },
            "fit_threshold":       50,
            "estimated_hold_months": 24,
            "scoring_rules_max_bonus": 130,
            "fit_indicators": {
                "roe":         {"min": 15, "weight": 0.25},
                "roce":        {"min": 15, "weight": 0.25},
                "epsGrowth5y": {"min": 15, "weight": 0.30},
                "deRatio":     {"max": 1.0, "weight": 0.20},
            },
            "scoring_rules": {
                "exceptional_capital_efficiency": {
                    "gates": {"roe": {"min": 25}, "roce": {"min": 20}},
                    "points": 35,
                    "reason": "Exceptional capital efficiency",
                },
                "strong_growth_track_record": {
                    "gates": {"epsGrowth5y": {"min": 20}, "profitGrowth3y": {"min": 20}},
                    "points": 30,
                    "reason": "Sustained multi-year earnings growth",
                },
                "fortress_balance_sheet": {
                    "gates": {"deRatio": {"max": 0.3}, "interestCoverage": {"min": 10}},
                    "points": 30,
                    "reason": "Fortress balance sheet",
                },
                "high_piotroski": {
                    "gates": {"piotroskiF": {"min": 8}},
                    "points": 20,
                    "reason": "Excellent Piotroski F-Score",
                },
                "promoter_conviction": {
                    "gates": {"promoterHolding": {"min": 50}},
                    "points": 15,
                    "reason": "High promoter conviction",
                },
            },
        },

        "value_investor": {
            "enabled":     True,
            "description": "Deep value with margin of safety and turnaround potential.",
            "preferred_setups": ["DEEP_VALUE_PLAY", "VALUE_TURNAROUND"],
            "avoid_setups":     ["MOMENTUM_BREAKOUT"],
            "horizon_fit_multipliers": {
                "intraday":    0.0,
                "short_term":  0.4,
                "long_term":   1.0,
                "multibagger": 1.3,
            },
            "fit_threshold":         50,
            "estimated_hold_months": 18,
            "scoring_rules_max_bonus": 100,
            "fit_indicators": {
                "piotroskiF": {"min": 6,    "weight": 0.30},
                "deRatio":    {"max": 1.0,  "weight": 0.30},
                "fcfYield":   {"min": 5.0,  "weight": 0.20},
                "roe":        {"min": 12,   "weight": 0.20},
            },
            "scoring_rules": {
                "deep_value_opportunity": {
                    "gates": {"priceToIntrinsicValue": {"max": 0.8}},
                    "points": 40,
                    "reason": "Trading at discount to intrinsic value",
                },
                "strong_fcf": {
                    "gates": {"fcfYield": {"min": 8}, "ocfVsProfit": {"min": 1.2}},
                    "points": 35,
                    "reason": "Excellent free cash flow generation",
                },
                "low_debt": {
                    "gates": {"deRatio": {"max": 0.5}},
                    "points": 25,
                    "reason": "Low debt improves margin of safety",
                },
            },
        },
    },
}
