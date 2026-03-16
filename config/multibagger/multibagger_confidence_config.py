# config/multibagger/multibagger_confidence_config.py
"""
Multibagger Confidence Configuration
======================================
Standalone confidence config for the MB evaluator.

Injected by MBConfigExtractor.__init__ via:
    self.confidence_config = MB_CONFIDENCE_CONFIG
    self.extract_confidence_sections()

REQUIRED sections (raises ConfigurationError if missing):
    global.confidence_range
    global.adx_normalization
    global.volume_modifiers
    global.universal_adjustments
    global.setup_baseline_floors
    horizons.multibagger.confidence_clamp
    horizons.multibagger.min_tradeable_confidence
    horizons.multibagger.high_confidence_override

OPTIONAL sections (safe defaults used if missing):
    global.divergence_physics
    horizons.multibagger.confidence_philosophy
    horizons.multibagger.base_confidence_adjustment
    horizons.multibagger.setup_floor_overrides
    horizons.multibagger.conditional_adjustments
    horizons.multibagger.adx_confidence_bands
    horizons.multibagger.adx_confidence_penalties
"""

MB_CONFIDENCE_CONFIG = {

    # =========================================================================
    # GLOBAL — shared across all setups evaluated by this config
    # =========================================================================
    "global": {

        "confidence_range": {
            "absolute_min":  0,
            "absolute_max":  100,
            "default_clamp": [50, 95],  # Tighter than trading — MB needs conviction
        },

        # Weekly ADX bands — calibrated lower than daily
        "adx_normalization": {
            "floor_boundary":    8,
            "ceiling_boundary":  30,
            "adjustment_factor": 10,
            "formula": "((adx - floor) / (ceiling - floor)) * adjustment_factor",
        },

        # Volume modifiers — weekly rvol > 1.5 is meaningful institutional flow
        "volume_modifiers": {
            "surge_bonus": {
                "gates": {"rvol": {"min": 1.5}},
                "confidence_boost": 10,
                "reason": "Weekly institutional accumulation volume",
            },
            "drought_penalty": {
                "gates": {"rvol": {"max": 0.6}},
                "confidence_penalty": -10,
                "exclude_setups": ["QUALITY_ACCUMULATION"],  # Long-term holders don't need daily vol
                "reason": "Very low weekly volume",
            },
        },

        "divergence_physics": {
            "lookback":          10,
            "slope_diff_min":    -0.02,   # Weekly — tighter than daily
            "bullish_slope_min":  0.02,
        },

        # Universal adjustments — applied regardless of setup
        "universal_adjustments": {
            "divergence_penalties": {
                "severe": {
                    "gates":       {"rsislope": {"max": -0.05}},
                    "block_entry": True,
                    "reason":      "Severe weekly momentum breakdown",
                },
                "moderate": {
                    "gates":                {"rsislope": {"max": -0.02, "min": -0.05}},
                    "confidence_multiplier": 0.75,
                    "block_entry":           False,
                    "reason":               "Moderate weekly divergence",
                },
                "minor": {
                    "gates":                {"rsislope": {"max": 0.0, "min": -0.02}},
                    "confidence_multiplier": 0.90,
                    "block_entry":           False,
                    "reason":               "Minor weekly divergence",
                },
            },

            # Weekly trend bands — thresholds lower than daily equivalents
            "trend_strength_bands": {
                "explosive": {
                    "gates":           {"trendStrength": {"min": 8.0}},
                    "confidence_boost": 20,
                    "reason":          "Explosive multi-month trend",
                },
                "strong": {
                    "gates":           {"trendStrength": {"min": 6.0, "max": 8.0}},
                    "confidence_boost": 12,
                    "reason":          "Strong sustained trend",
                },
                "moderate": {
                    "gates":           {"trendStrength": {"min": 4.0, "max": 6.0}},
                    "confidence_boost": 5,
                    "reason":          "Moderate trend support",
                },
                "weak": {
                    "gates":            {"trendStrength": {"max": 3.9}},
                    "confidence_penalty": -10,
                    "reason":           "Weak trend — sideways risk",
                },
            },
        },

        # Setup baseline floors — MB favours quality and value setups
        "setup_baseline_floors": {
            "QUALITY_ACCUMULATION": 60,
            "DEEP_VALUE_PLAY":      55,
            "VALUE_TURNAROUND":     55,
            "GENERIC":              40,
        },
    },

    # =========================================================================
    # HORIZONS — only "multibagger" is used by the MB module
    # =========================================================================
    "horizons": {
        "multibagger": {
            "confidence_philosophy": (
                "Ultra-selective, fundamentals-obsessed. "
                "Perfect quality + growth profile required."
            ),

            # Clamp [50, 95] — high floor because MB is a high-conviction signal
            "confidence_clamp": [50, 95],

            "base_confidence_adjustment": 0,

            "setup_floor_overrides": {
                "QUALITY_ACCUMULATION": 65,  # +5 vs global floor
                "DEEP_VALUE_PLAY":      60,  # +5 vs global floor
                "VALUE_TURNAROUND":     60,  # +5 vs global floor
                # Block short-term setups
                "MOMENTUM_BREAKOUT":    None,
                "VOLATILITY_SQUEEZE":   None,
            },

            # Horizon-specific penalties and bonuses
            "conditional_adjustments": {
                "penalties": {
                    "insufficient_growth": {
                        "gates": {
                            "epsGrowth5y":      {"max": 15},
                            "revenueGrowth5y":  {"max": 15},
                            "_logic": "OR",
                        },
                        "confidence_penalty": -15,
                        "reason": "Growth insufficient for multibagger thesis",
                    },
                    "weak_quality": {
                        "gates": {
                            "roe":  {"max": 15},
                            "roce": {"max": 18},
                            "_logic": "OR",
                        },
                        "confidence_penalty": -25,
                        "reason": "Quality too low for multi-year hold",
                    },
                    "high_leverage": {
                        "gates": {"deRatio": {"min": 1.2}},
                        "confidence_penalty": -25,
                        "reason": "Debt risk over long horizon",
                    },
                    "bearish_structure": {
                        "gates": {
                            "trendStrength":    {"max": 3.0},
                            "momentumStrength": {"max": 3.0},
                        },
                        "confidence_penalty": -30,
                        "reason": "Bearish structure — wait for reversal",
                    },
                    "overvalued": {
                        "gates": {"priceToIntrinsicValue": {"min": 1.3}},
                        "confidence_penalty": -20,
                        "reason": "Overvalued vs intrinsic value",
                    },
                },

                "bonuses": {
                    "mega_trend_quality": {
                        "gates": {
                            "trendStrength": {"min": 8.5},
                            "roe":           {"min": 25},
                            "roce":          {"min": 30},
                        },
                        "confidence_boost": 35,
                        "reason": "Mega-trend with exceptional fundamentals",
                    },
                    "early_leader": {
                        "gates": {
                            "relStrengthNifty": {"min": 1.2},
                            "marketCapCagr":    {"min": 25},
                        },
                        "confidence_boost": 20,
                        "reason": "Early-stage market leader",
                    },
                    "deep_value": {
                        "gates": {
                            "priceToIntrinsicValue": {"max": 0.7},
                            "roe":                   {"min": 20},
                        },
                        "confidence_boost": 25,
                        "reason": "Deep value with quality",
                    },
                    "quality_emerging_trend": {
                        "gates": {
                            "trendStrength": {"min": 5.0, "max": 7.0},
                            "roe":           {"min": 25},
                            "epsGrowth5y":   {"min": 15},
                        },
                        "confidence_boost": 20,
                        "reason": "Quality company with emerging trend",
                    },
                    "moat_indicators": {
                        "gates": {
                            "roic":    {"min": 20},
                            "roe":     {"min": 25},
                            "deRatio": {"max": 0.3},
                        },
                        "confidence_boost": 15,
                        "reason": "Economic moat indicators present",
                    },
                    "growth_combo": {
                        "gates": {
                            "epsGrowth5y":   {"min": 25},
                            "marketCapCagr": {"min": 25},
                        },
                        "confidence_boost": 18,
                        "reason": "Consistent compounding + price performance",
                    },
                },
            },

            # Weekly ADX bands — lower thresholds than daily
            "adx_confidence_bands": {
                "explosive": {
                    "gates": {"adx": {"min": 25}},
                    "confidence_boost": 20,
                },
                "strong": {
                    "gates": {"adx": {"min": 15, "max": 25}},
                    "confidence_boost": 12,
                },
                "moderate": {
                    "gates": {"adx": {"min": 8, "max": 15}},
                    "confidence_boost": 0,
                },
            },

            "adx_confidence_penalties": {
                "weak": {
                    "gates":            {"adx": {"max": 7}},
                    "confidence_penalty": -10,
                    "reason":           "Trend too weak for multibagger thesis",
                }
            },

            # Below this, the evaluator marks signal as untradeable
            "min_tradeable_confidence": {
                "min":    60,
                "reason": "Multibagger requires very high conviction",
            },

            "high_confidence_override": {
                "threshold": 90,
                "can_override": {
                    "execution_warnings":  False,  # No shortcuts for MB
                    "structural_gates":    False,
                    "fundamental_gates":   False,
                },
                "max_override_count": 0,
                "log_overrides":      True,
            },
        }
    },
}
