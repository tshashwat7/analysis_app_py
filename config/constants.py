# config/constants.py

import os

#todo remove
STOCH_FAST = {"k_period": 5, "d_period": 3, "smooth": 3}
STOCH_SLOW = {"k_period": 14, "d_period": 3, "smooth": 3}
STOCH_THRESHOLDS = {"overbought": 80, "oversold": 20}

# config/constants.py
ENABLE_CACHE = True
ENABLE_CACHE_WARMER = os.getenv("ENABLE_CACHE_WARMER", "false").lower() == "true"
ENABLE_JSON_ENRICHMENT = os.getenv("ENABLE_JSON_ENRICHMENT", "true").lower() == "true"
ENABLE_VOLATILITY_QUALITY = True


# ============================================================================
# MASTER CONFIG V10 (UPDATED WITH DETAILED PATTERN PARAMETERS & CALC ENGINE)
# ============================================================================

class ConfigGuard:
    """
    A smart wrapper for configuration dictionaries.
    It tracks the 'path' of keys accessed (e.g. 'horizons -> intraday -> metrics')
    and raises descriptive errors if a key is missing.
    """
    def __init__(self, data, path="MASTER_CONFIG"):
        self._data = data
        self._path = path

    def __getitem__(self, key):
        # 1. Check if key exists
        if key not in self._data:
            raise KeyError(
                f"❌ CONFIG ERROR: Missing key '{key}' at path '{self._path}'. "
                f"The code expected this key, but it is not in constants.py."
            )
        
        # 2. Retrieve value
        val = self._data[key]

        # 3. If value is a dict, wrap it in a new Guard (Recursive safety)
        if isinstance(val, dict):
            return ConfigGuard(val, path=f"{self._path}.{key}")
        
        # 4. If value is a list/string/number, return raw value
        return val

    def get(self, key, default=None):
        """Safe get method that still wraps dict returns"""
        if key not in self._data:
            return default
        
        val = self._data[key]
        if isinstance(val, dict):
            return ConfigGuard(val, path=f"{self._path}.{key}")
        return val

    def __contains__(self, key):
        return key in self._data

    def items(self):
        """Allows looping, but wraps values if they are dicts"""
        for k, v in self._data.items():
            if isinstance(v, dict):
                yield k, ConfigGuard(v, path=f"{self._path}.{k}")
            else:
                yield k, v
    
    def keys(self):
        return self._data.keys()

    # Allow access to raw dict if absolutely needed
    def unwrap(self):
        return self._data
    
   
MASTER_CONFIG = {

    # ============================================================
    # SECTION 0: GLOBAL CONSTANTS (Universal Physics & Logic)
    # ============================================================
    "global": {

        # 0. time_estimation
        "time_estimation": {
            "base_friction": 0.8,
            "velocity_factors": {
                "strong_trend": {"min_strength": 7.0, "factor": 1.2},
                "normal_trend": {"min_strength": 5.0, "factor": 1.0},
                "weak_trend": {"max_strength": 5.0, "factor": 0.8}
            },
            "formatting": {
                "intraday_candle_minutes": 15,
                "days_per_week": 5
            }
        },
        # 1. PATTERN PHYSICS
        "pattern_physics": {
            "cup_handle": {
                "target_ratio": 0.618, "duration_multiplier": 1.2, "max_stop_pct": 8.0,
                "min_cup_len": 20, "max_cup_depth": 0.50, "handle_len": 5, "require_volume": False,
                "horizons_supported": ["short_term", "long_term"]
            },
            "darvas_box": {
                "target_ratio": 1.0, "duration_multiplier": 1.3, "max_stop_pct": 5.0,
                "lookback": 50, "box_length": 5, "horizons_supported": ["intraday", "short_term"]
            },
            "flag_pennant": {
                "target_ratio": 0.5, "duration_multiplier": 0.8, "max_stop_pct": 6.0,
                "horizons_supported": ["intraday", "short_term"]
            },
            "minervini_vcp": {
                "target_ratio": 1.0, "duration_multiplier": 1.8, "max_stop_pct": 7.0,
                "min_contraction_pct": 1.5, "horizons_supported": ["short_term", "long_term"]
            },
            "volatility_squeeze": {
                "target_ratio": 1.0, "duration_multiplier": 0.5, "max_stop_pct": 4.0,
                "horizons_supported": ["intraday", "short_term"]
            },
            "golden_cross": {
                "target_ratio": None, "duration_multiplier": 2.0, "max_stop_pct": None,
                "horizons_supported": ["short_term", "long_term", "multibagger"]
            },
            "default": {
                "target_ratio": 1.0, "duration_multiplier": 1.0, "max_stop_pct": 10.0
            }
        },

        # 2. PATTERN ENTRY RULES
        "pattern_entry_rules": {
            "bollinger_squeeze": {
                "horizons": {
                    "intraday": {
                        "rsi_min": 50,
                        "macd_hist_min": 0,
                        "squeeze_duration_min": 5,  # Must be squeezed for 75 min
                        "rvol_on_breakout": 1.5
                    },
                    "short_term": {
                        "rsi_min": 50,
                        "macd_hist_min": 0,
                        "squeeze_duration_min": 3,  # 3 days
                        "rvol_on_breakout": 1.2
                    },
                    "long_term": {
                        "rsi_min": 45,  # More lenient
                        "macd_hist_min": -0.2,
                        "squeeze_duration_min": 4,  # 4 weeks
                        "rvol_on_breakout": 1.0
                    }
                }
            },
            "darvas_box": {
                "horizons": {
                    "intraday": {
                        "box_clearance": 1.002,  # Must be 0.2% above box high
                        "volume_surge_required": 1.5,
                        "max_box_age_candles": 50
                    },
                    "short_term": {
                        "box_clearance": 1.005,  # 0.5% above
                        "volume_surge_required": 1.3,
                        "max_box_age_candles": 30
                    }
                }
            },
            "cup_handle": {
                "horizons": {
                    "short_term": {
                        "rim_clearance": 0.995,  # Can enter 0.5% below rim
                        "rvol_min": 1.2,
                        "rvol_bonus_threshold": 2.0
                    },
                    "long_term": {
                        "rim_clearance": 0.99,  # More lenient (1% below)
                        "rvol_min": 1.1,
                        "rvol_bonus_threshold": 1.8
                    }
                }
            },
            "minervini_stage2": {
                "horizons": {
                    "short_term": {
                        "contraction_max": 1.5,
                        "pivot_clearance": 1.01,  # Must be 1% above pivot
                        "rs_rating_min": 80
                    },
                    "long_term": {
                        "contraction_max": 2.0,  # Allow wider VCP
                        "pivot_clearance": 1.005,  # Only 0.5% above
                        "rs_rating_min": 70
                    }
                }
            },
            "flag_pennant": {
                "horizons": {
                    "intraday": {
                        "pole_length_min": 8,  # Min 2 hours of prior move
                        "flag_tightness": 0.03,  # 3% range
                        "breakout_clearance": 1.005
                    },
                    "short_term": {
                        "pole_length_min": 5,  # Min 5 days
                        "flag_tightness": 0.05,
                        "breakout_clearance": 1.01
                    }
                }
            },
            "three_line_strike": {
                "horizons": {
                    "intraday": {
                        "strike_candle_body_min": 0.6,  # 4th candle must engulf 60%
                        "rvol_min": 1.3
                    },
                    "short_term": {
                        "strike_candle_body_min": 0.7,  # 70% engulfment
                        "rvol_min": 1.2
                    }
                }
            },
            "ichimoku_signals": {
                "horizons": {
                    "short_term": {
                        "cloud_thickness_min": 0.01,  # 1% thick cloud
                        "tenkan_kijun_spread_min": 0.005
                    },
                    "long_term": {
                        "cloud_thickness_min": 0.02,  # 2% thick cloud
                        "tenkan_kijun_spread_min": 0.01
                    }
                }
            },
            "golden_cross": {
                "horizons": {
                    "short_term": {
                        "cross_clearance": 0.002,  # 50 EMA must be 0.2% above 200 EMA
                        "volume_confirmation": 1.1
                    },
                    "long_term": {
                        "cross_clearance": 0.005,  # 0.5% clearance
                        "volume_confirmation": 1.0  # Not required
                    }
                }
            },
            "double_top_bottom": {
                "horizons": {
                    "short_term": {
                        "peak_similarity_tolerance": 0.02,  # Peaks within 2%
                        "neckline_clearance": 1.01,
                        "volume_decline_on_second_peak": True
                    },
                    "long_term": {
                        "peak_similarity_tolerance": 0.03,  # 3% tolerance
                        "neckline_clearance": 1.005,
                        "volume_decline_on_second_peak": False  # Not required
                    }
                }
            }
        },

        # 3. PATTERN INVALIDATION
        "pattern_invalidation": {
            # ========================================================
            # 1. BOLLINGER SQUEEZE (High Failure Risk)
            # ========================================================
            "bollinger_squeeze": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < bb_low",  # Changed from bb_mid * 0.995
                        "duration_candles": 2,
                        "or_condition": "bb_width > 8.0"  # NEW: Squeeze released
                    },
                    "short_term": {
                        "condition": "price < bb_low * 0.99",  # Changed from bb_low * 1.01
                        "duration_candles": 1,
                        "or_condition": "bb_width > 10.0"
                    },
                    "long_term": {
                        "condition": "price < bb_low",
                        "duration_candles": 2,
                        "or_condition": "bb_width > 12.0"
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_ON_CLOSE",
                    "long_term": "TIGHTEN_STOP"
                }
            },
            # ========================================================
            # 2. DARVAS BOX (Strict Boundaries)
            # ========================================================
            "darvas_box": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < box_low * 0.998",  # 0.2% buffer
                        "duration_candles": 1  # Single 15-min break = exit
                    },
                    "short_term": {
                        "condition": "price < box_low * 0.995",  # 0.5% buffer
                        "duration_candles": 1  # Daily close below
                    },
                    "long_term": {
                        "condition": "price < box_low * 0.99",  # 1% buffer
                        "duration_candles": 2  # 2-week confirmation
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_IMMEDIATELY",  # Darvas is strict
                    "long_term": "EXIT_ON_CLOSE"
                },
                "notes": "Darvas box breakdown is binary - no 'monitor' mode"
            },
            # ========================================================
            # 3. FLAG/PENNANT (Tight Structure)
            # ========================================================
            "flag_pennant": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < flag_low * 0.998",
                        "duration_candles": 1
                    },
                    "short_term": {
                        "condition": "price < flag_low * 0.995",
                        "duration_candles": 1
                    },
                    "long_term": {
                        "condition": "price < flag_low * 0.99",
                        "duration_candles": 1  # Even long-term is strict (flags are momentum)
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_IMMEDIATELY",
                    "long_term": "EXIT_ON_CLOSE"
                },
                "expiration": {
                    "max_duration_candles": {
                        "intraday": 20,      # Flags expire after ~5 hours
                        "short_term": 10,    # Max 10 days
                        "long_term": 8       # Max 8 weeks
                    },
                    "action_on_expire": "DOWNGRADE_TO_CONSOLIDATION"
                }
            },
            # ========================================================
            # 4. MINERVINI VCP (Gradual Tolerance)
            # ========================================================
            "minervini_stage2": {  # Your alias is "minervini_stage2"
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < pivot * 0.98",  # 2% below pivot
                        "duration_candles": 2  # 30-min confirmation
                    },
                    "short_term": {
                        "condition": "price < pivot * 0.95",  # 5% below pivot
                        "duration_candles": 2  # 2-day confirmation
                    },
                    "long_term": {
                        "condition": "price < pivot * 0.92",  # 8% below pivot (original)
                        "duration_candles": 3  # 3-week confirmation
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_ON_CLOSE",
                    "long_term": "TIGHTEN_STOP"
                },
                "stage_reversion": {
                    "stage_1_threshold": "price < 10wma and volume declining",
                    "action": "EXIT_ON_CLOSE"
                }
            },
            # ========================================================
            # 5. CUP & HANDLE (Progressive Stops)
            # ========================================================
            "cup_handle": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < handle_low * 0.99",  # 1% below handle
                        "duration_candles": 2
                    },
                    "short_term": {
                        "condition": "price < handle_low * 0.97",  # 3% below handle
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < handle_low * 0.95",  # 5% below handle (original)
                        "duration_candles": 3
                    }
                },
                "action": {
                    "intraday": "EXIT_ON_CLOSE",      # Give it one candle
                    "short_term": "TIGHTEN_STOP",     # Move stop to handle mid
                    "long_term": "MONITOR"            # Just watch for now
                },
                "handle_failure": {
                    "max_handle_depth": 0.15,  # Handle can't drop >15% of cup depth
                    "action": "INVALIDATE_PATTERN"
                }
            },
            # ========================================================
            # 6. THREE-LINE STRIKE (Fast Reversal)
            # ========================================================
            "three_line_strike": {
                "expiration": {
                    "max_hold_candles": {
                        "intraday": 10,
                        "short_term": 8,  # Changed from 5
                        "long_term": 6  # Changed from 4
                    }
                }
            },
            # ========================================================
            # 7. ICHIMOKU SIGNALS (Cloud Support)
            # ========================================================
            "ichimoku_signals": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < cloud_bottom * 0.998",
                        "duration_candles": 2
                    },
                    "short_term": {
                        "condition": "price < cloud_bottom * 0.995",
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < cloud_bottom",  # Full cloud break
                        "duration_candles": 3
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_ON_CLOSE",
                    "long_term": "TIGHTEN_STOP"
                },
                "lagging_span_confirmation": {
                    "enabled": True,
                    "condition": "lagging_span < price_26_candles_ago",
                    "action": "INCREASE_STOP_URGENCY"
                }
            },
            # ========================================================
            # 8. GOLDEN/DEATH CROSS (Slow Reversal)
            # ========================================================
            "golden_cross": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "ema_50 < ema_200",
                        "duration_candles": 3
                    },
                    "short_term": {
                        "condition": "ema_50 < ema_200",
                        "duration_candles": 3
                    },
                    "long_term": {
                        "condition": "ema_50 < ema_200",
                        "duration_candles": 4
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",  # Changed from EXIT_ON_CLOSE
                    "short_term": "EXIT_ON_CLOSE",  # Changed from TIGHTEN_STOP
                    "long_term": "EXIT_ON_CLOSE"  # Changed from MONITOR
                }
            },
            # ========================================================
            # 9. DOUBLE TOP/BOTTOM (Neckline Defense)
            # ========================================================
            "double_top_bottom": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < neckline * 0.998",  # 0.2% below neckline
                        "duration_candles": 1
                    },
                    "short_term": {
                        "condition": "price < neckline * 0.995",  # 0.5% below neckline
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < neckline * 0.99",  # 1% below neckline
                        "duration_candles": 3
                    }
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_ON_CLOSE",
                    "long_term": "TIGHTEN_STOP"
                },
                "target_failure": {
                    "condition": "price fails to reach target within max_duration",
                    "max_duration_candles": {
                        "intraday": 15,
                        "short_term": 10,
                        "long_term": 12
                    },
                    "action": "EXIT_ON_CLOSE"
                }
            },
            },
        
        # 4. ENTRY GATES
        "entry_gates": {
            "confidence_requirements": {
                "breakout_base": 70, "trend_discount": -15, "accumulation_discount": -25
            },
            "adx_requirements": {
                "trend_setups_min": 18, "breakout_setups_min": 15
            },
            "execution_constraints": {
                "structure_validation": {
                    "breakout_clearance": 0.001, "breakdown_clearance": 0.001
                },
                "sl_distance_validation": {
                    "min_atr_multiplier": 0.5, "max_atr_multiplier": 5.0
                },
                "target_proximity_rejection": {
                    "min_t1_distance": 0.005, "min_t2_distance": 0.01
                }
            }
        },

        # 6. CALCULATION ENGINE
        "calculation_engine": {
            "composite_weights": {
                "trend_strength": {
                    "adx": {"weight": 0.4, "scoring": [{"min": 25, "score": 10}, {"min": 20, "score": 8}, {"default": 2}]},
                    "ema_slope": {"weight": 0.3, "scoring": [{"min": 20, "score": 10}, {"min": 5, "score": 7}, {"default": 2}]},
                    "di_spread": {"weight": 0.2, "scoring": [{"min": 15, "score": 10}, {"min": 10, "score": 7}, {"default": 5}]},
                    "supertrend": {"weight": 0.1, "scoring": {"bullish": 10, "bearish": 0}},
                    "adaptive_weights_no_supertrend": {"adx": 0.45, "ema_slope": 0.35, "di_spread": 0.20}
                },
                "momentum_strength": {
                    "rsi_value": {"weight": 0.25, "scoring": [{"min": 70, "score": 8}, {"min": 60, "score": 7}, {"default": 2}]},
                    "rsi_slope": {"weight": 0.25, "scoring": [{"min": 1.0, "score": 8}, {"min": 0, "score": 4}, {"default": 2}]},
                    "macd_hist": {"weight": 0.3, "scoring": [{"min": 0.5, "score": 8}, {"min": 0, "score": 5}, {"default": 2}]},
                    "stoch_cross": {"weight": 0.2, "scoring": {"bullish_strong": {"condition": "k>d and k>=50", "score": 8}, "default": 3}}
                },
                "volatility_quality": {
                    "atr_pct": {"weight": 0.3},
                    "bb_width": {"weight": 0.2, "scoring": [{"max": 0.01, "score": 10}, {"max": 0.02, "score": 8}, {"default": 2}]},
                    "true_range_consistency": {"weight": 0.2, "scoring": [{"max": 0.5, "score": 10}, {"default": 4}]},
                    "hv_trend": {"weight": 0.15, "scoring": {"declining": {"condition": "hv10 < hv20", "score": 8}, "default": 4}},
                    "atr_sma_ratio": {"weight": 0.15, "scoring": [{"max": 0.02, "score": 10}, {"default": 3}]}
                }
            },
            "setup_classification": {
                "breakout": {"bb_pos_min": 0.98, "rsi_min": 60, "wick_ratio_max": 2.5, "rvol_min": 1.5},
                "breakdown": {"bb_pos_max": 0.02, "rsi_max": 40, "rvol_min": 1.5},
                "pullback": {"ma_dist_max": 0.05, "rsi_min": 50},
                "bear_pullback": {"ma_dist_max": 0.05, "rsi_max": 50},
                "trend_following": {
                    "classic": {"rsi_min": 55, "macd_hist_min": 0},
                    "strong_drift": {"trend_strength_min": 7.0}
                },
                "accumulation": {"consolidation_required": True, "rsi_range": [40, 60]},
                "quality_accumulation_downtrend": {
                    "fundamental_requirements": {"roe_min": 20, "roce_min": 25, "de_ratio_max": 0.5},
                    "bb_percent_b_range": [0.2, 0.5]
                },
                "divergence": {
                    "lookback": 10, "slope_diff_min": -0.05,
                    "confidence_penalties": {"bearish_divergence": 0.70, "bullish_divergence": 0.70}
                }
            },
            "spread_adjustment": {
                "market_cap_brackets": {
                    "large_cap": {"min": 100000, "spread_pct": 0.001},
                    "mid_cap": {"min": 10000, "max": 100000, "spread_pct": 0.002},
                    "small_cap": {"max": 10000, "spread_pct": 0.005}
                }
            },
            "volume_signatures": {
                "surge": {"threshold": 3.0, "confidence_adjustment": 15},
                "drought": {"threshold": 0.7, "confidence_adjustment": -25},
                "climax": {"threshold": 2.0, "rsi_condition_min": 70}
            },
            "wick_rejection": {
                "max_ratio": 2.5,
                "calculation": "abs(high - close) / abs(close - open)"
            },
            "divergence_detection": {
            "lookback": 10,
            "slope_diff_min": -0.05,
            "confidence_penalties": {
                "bearish_divergence": 0.70,
                "bullish_divergence": 0.70
            },
            # ✅ ADD THIS BLOCK:
            "severity_bands": {
                "minor": {
                    "rsi_slope_min": -0.03,
                    "confidence_penalty": 0.90,
                    "allow_entry": True
                },
                "moderate": {
                    "rsi_slope_min": -0.08,
                    "confidence_penalty": 0.70,
                    "allow_entry": True
                },
                "severe": {
                    "rsi_slope_min": -999,
                    "confidence_penalty": 0.50,
                    "allow_entry": False
                }
            }
        },
        },

        # 7. POSITION SIZING
        "position_sizing": {
            "base_risk_pct": 0.01,
            "global_setup_multipliers": {
                "DEEP_PULLBACK": 1.5, "VOLATILITY_SQUEEZE": 1.3, "MOMENTUM_BREAKOUT": 0.8
            },
            "volatility_adjustments": {
                "high_quality": {"vol_qual_min": 7.0, "multiplier": 1.2},
                "low_quality": {"vol_qual_max": 5.0, "multiplier": 0.9}
            }
        },

        # 8. WEIGHTS
        "trend_weights": {"primary": 0.50, "secondary": 0.30, "acceleration": 0.20},
        "fundamental_weights": {
            "pe_ratio": 0.05, "pb_ratio": 0.04, "peg_ratio": 0.03, "fcf_yield": 0.05,
            "roe": 0.10, "roce": 0.07, "roic": 0.08, "de_ratio": 0.05, 
            "interest_coverage": 0.05, "current_ratio": 0.03, "ocf_vs_profit": 0.02,
            "asset_turnover": 0.04, "piotroski_f": 0.07, "r_d_intensity": 0.04, 
            "earnings_stability": 0.05, "eps_growth_5y": 0.06, "fcf_growth_3y": 0.05,
            "market_cap_cagr": 0.04, "promoter_holding": 0.015, "institutional_ownership": 0.015,
            "beta": 0.01, "52w_position": 0.01, "dividend_payout": 0.03, "yield_vs_avg": 0.02,
            "promoter_pledge": 0.02, "quarterly_growth": 0.03, "revenue_growth_5y": 0.05
        },

        # 9. STRATEGY CONFIG
        "strategy_classification": {
            "swing": {"fit_thresh": 50}, "day_trading": {"fit_thresh": 60},
            "trend_following": {"fit_thresh": 60}, "momentum": {"fit_thresh": 60},
            "minervini": {"fit_thresh": 70}, "canslim": {"fit_thresh": 65},
            "value": {"fit_thresh": 50}
        },
        "strategy_time_multipliers": {
            "momentum": 0.7, "day_trading": 0.5, "swing": 1.0, "trend_following": 1.2,
            "position_trading": 1.5, "value": 1.5, "income": 2.0, "unknown": 1.0
        },

        # 10. BOOSTS
        "boosts": {
            "pattern": {"single": 0.8, "confluence": 1.5},
            "momentum": {"score": 1.2, "trigger_rvol": 2.0},
            "volatility": {"squeeze": {"score": 1.0, "min_quality": 7.0}, "expansion": {"score": 0.6, "min_quality": 4.0}},
            "relative_strength": {"outperforming_weak_market": 1.5, "outperforming_strong_market": 0.8},
            "proximity": {"support": 0.6, "resistance": 0.8},
            "max_cap": 2.5
        },

        # 11. INFRASTRUCTURE
        "system": {
            "process_pool_workers": 4,
            "cache": {"ttl_seconds": 3600, "timezone": "Asia/Kolkata"},
            "cache_warmer": {
                "batch_size": 5, "market_interval_sec": 900, "top_n_during_market": 50,
                "lru_target": 500, "deep_hour": 2, "batch_sleep_market": 5.0
            },
            "fetch": {
                "max_retries": 3, "timeout": 10,
                "period_map": {"intraday": "1mo", "short_term": "2y", "long_term": "5y", "multibagger": "10y"},
                "interval_map": {"intraday": "15m", "short_term": "1d", "long_term": "1wk", "multibagger": "1mo"}
            },
            "corporate_actions": {
                "lookback_days": {"past": 365, "upcoming": 7},
                "display_priority": ["Dividend", "Bonus", "Split", "Rights"]
            }
        }
    },

    # ============================================================
    # SECTION 1: HORIZON PROFILES (Fully Patched)
    # ============================================================
    "horizons": {

        # ========================================================
        # INTRADAY
        # ========================================================
        "intraday": {
            "enhancements": {
                "volume_surge": {
                    "condition": "rvol >= 2.5",  # Higher bar for intraday
                    "amount": 12.0,
                    "reason": "Strong intraday volume surge",
                    "max_boost": 15.0 
                },
                "squeeze_release": {
                    "condition": "bb_width < 3.0 and rvol >= 2.0",
                    "amount": 15.0,
                    "reason": "Tight squeeze breaking with volume",
                    "max_boost": 18.0 
                },
                "momentum_spike": {
                    "condition": "momentum_strength >= 8.0",
                    "amount": 10.0,
                    "reason": "Explosive momentum for scalp",
                    "max_boost": 12.0 
                }
            },
            "timeframe": "15m",
            "description": "Quick scalps and day trades",
            "scoring": {
                "metrics": {
                    "ma_fast_slope": 0.20, "rsi_slope": 0.20, "price_action": 0.15,
                    "vwap_bias": 0.15, "vol_spike_ratio": 0.10, "momentum_strength": 0.10,
                    "volatility_quality": 0.05, "ma_trend_signal": 0.05
                },
                "penalties": {
                    "gap_percent": {"op": "<", "val": 0.1, "pen": 0.1},
                    "atr_pct": {"op": "<", "val": 0.4, "pen": 0.3},
                    "ma_fast_slope": {"op": "<", "val": -2, "pen": 0.3}
                },
                "thresholds": {"buy": 6.5, "hold": 5.0, "sell": 3.8}
            },
            "gates": {
                "min_trend_strength": 2.0, "allowed_supertrend_counter": True,
                "volatility_bands_atr_pct": [0.3, 5.0],
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 2.5,
                    "VOLATILITY_SQUEEZE": 4.0,
                    "TREND_PULLBACK": 3.0,
                    "default": 2.5
                }
            },
            # ✅ CONFIDENCE (Added overrides & range_bound)
            "confidence": {
                "horizon_discount": 10, "floors": {"buy": 55, "wait": 30},
                "volume_penalty": {"rvol_drought_penalty": -20, "ignore_for_squeeze": True},
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 60}, 
                    "moderate": {"adx_min": 25, "floor": 50}, 
                    "weak": {"adx_min": 20, "floor": 40},
                    "range_bound": {"adx_max": 20, "floor": 30} # 
                },
                "setup_type_overrides": { 
                    "MOMENTUM_BREAKOUT": 5, "VOLATILITY_SQUEEZE": 10, "QUALITY_ACCUMULATION": -5
                }
            },
            "moving_averages": {
                "type": "EMA", "fast": 20, "mid": 50, "slow": 200, 
                "keys": ["ema_20", "ema_50", "ema_200"], "dip_buy_reference": "ema_20"
            },
            "momentum_thresholds": {
                "rsi_slope": {"acceleration_floor": 0.10, "deceleration_ceiling": -0.10},
                "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
            },
            "volatility": {
                "scoring_thresholds": {"atr_pct": {"excellent": 1.5,"good": 3.0,"fair": 4.0,"poor": 5.0}, "bb_width": {"tight": 2.0, "normal": 5.0, "wide": 10.0}}
            },
            "risk_management": {
                "max_position_pct": 0.01,
                "setup_size_multipliers": {"VOLATILITY_SQUEEZE": 1.3, "MOMENTUM_BREAKOUT": 0.8},
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "min_rr_ratio": 1.2,
                "horizon_t2_cap": 0.04,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 1.5, "target_atr_mult": 2.5, "max_hold_candles": 25,
                "risk_reward_min": 1.5
            },
            "indicators": {
                "rsi_period": 9, "adx_period": 14, "atr_period": 14,
                "supertrend": {"period": 7, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "fibonacci"
            },
            "lookback": {"python_data": 500}
        },

        # ========================================================
        # SHORT TERM
        # ========================================================
        "short_term": {
            "enhancements": {
                "pattern_confluence": {
                    "condition": "pattern_count >= 2",
                    "amount": 12.0,
                    "reason": "Multiple patterns confirm swing setup",
                    "max_boost": 15.0 
                },
                "trend_momentum_sync": {
                    "condition": "trend_strength >= 7.0 and momentum_strength >= 7.0",
                    "amount": 10.0,
                    "reason": "Strong trend + momentum for swing",
                    "max_boost": 15.0 
                },
                "quality_pullback": {
                    "condition": "setup_type == 'TREND_PULLBACK' and volatility_quality >= 6.0",
                    "amount": 8.0,
                    "reason": "Clean pullback setup",
                    "max_boost": 12.0 
                }
            },
            "timeframe": "1d",
            "description": "Swing trading (Days to Weeks)",
            "scoring": {
                "metrics": {
                    "trend_strength": 0.15, "ma_fast_slope": 0.15, "rsi_slope": 0.10,
                    "momentum_strength": 0.12, "volatility_quality": 0.10, "supertrend_signal": 0.10, 
                    "ma_trend_signal": 0.10, "price_vs_primary_trend_pct": 0.08, "rvol": 0.10
                },
                "penalties": {
                    "days_to_earnings": {"op": "<", "val": 3, "pen": 0.5},
                    "rvol": {"op": "<", "val": 0.5, "pen": 0.2},
                    "atr_pct": {"op": "<", "val": 1.0, "pen": 0.2},
                    "ma_fast_slope": {"op": "<", "val": -5, "pen": 0.3}
                },
                "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 4.0}
            },
            "gates": {
                "min_trend_strength": 2.0, "allowed_supertrend_counter": False,
                "volatility_bands_atr_pct": [0.8, 12.0],
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 3.0,
                    "VOLATILITY_SQUEEZE": 5.0,
                    "TREND_PULLBACK": 3.5,
                    "default": 3
                }
            },
            "confidence": {
                "horizon_discount": 5, "floors": {"buy": 55, "wait": 30},
                "volume_penalty": {"rvol_drought_penalty": -20, "ignore_for_squeeze": True},
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 60}, 
                    "moderate": {"adx_min": 25, "floor": 50}, 
                    "weak": {"adx_min": 20, "floor": 40},
                    "range_bound": {"adx_max": 20, "floor": 30} # 
                },
                "setup_type_overrides": { # 
                    "MOMENTUM_BREAKOUT": 5, "VOLATILITY_SQUEEZE": 10, "QUALITY_ACCUMULATION": -5
                }
            },
            "moving_averages": {
                "type": "EMA", "fast": 20, "mid": 50, "slow": 200, 
                "keys": ["ema_20", "ema_50", "ema_200"], "dip_buy_reference": "ema_50"
            },
            "momentum_thresholds": {
                "rsi_slope": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
                "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
            },
            "volatility": {
                "scoring_thresholds": {"atr_pct": {"excellent": 2.5, "good": 3.0, "fair": 4.5, "poor": 5.5}, "bb_width": {"tight": 3.0, "normal": 6.0, "wide": 12.0}}
            },
            "risk_management": {
                "max_position_pct": 0.02,
                "setup_size_multipliers": {"DEEP_PULLBACK": 1.5, "MOMENTUM_BREAKOUT": 1.0},
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "min_rr_ratio": 1.4,
                "horizon_t2_cap": 0.10,
                 "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                },
            },
            "execution": {
                "stop_loss_atr_mult": 2.0, "target_atr_mult": 3.0, "max_hold_candles": 15,
                "dip_buy_reference": "ema_50", "risk_reward_min": 2.0
            },
            "indicators": {
                "rsi_period": 14, "adx_period": 14, "atr_period": 14,
                "supertrend": {"period": 10, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "classic"
            },
            "lookback": {"python_data": 600}
        },

        # ========================================================
        # LONG TERM
        # ========================================================
        "long_term": {
            "enhancements": {
                "quality_fundamentals": {
                    "condition": "roe >= 20 and roce >= 25",
                    "amount": 15.0,
                    "reason": "High-quality company for trend trade",
                    "max_boost": 20.0 
                },
                "earnings_acceleration": {
                    "condition": "quarterly_growth >= 15 and eps_growth_5y >= 15",
                    "amount": 12.0,
                    "reason": "Consistent earnings growth",
                    "max_boost": 15.0 
                },
                "institutional_interest": {
                    "condition": "institutional_ownership >= 25 and institutional_ownership <= 75",
                    "amount": 8.0,
                    "reason": "Smart money accumulating",
                    "max_boost": 10.0 
                }
            },
            "timeframe": "1wk",
            "description": "Trend Following & Investing",
            "scoring": {
                "metrics": {
                    "ma_trend_signal": 0.15, "price_vs_primary_trend_pct": 0.10, "roe": 0.10,
                    "roce": 0.08, "roic": 0.08, "earnings_stability": 0.08, "fcf_yield": 0.08,
                    "eps_growth_5y": 0.06, "piotroski_f": 0.05, "rel_strength_nifty": 0.04,
                    "ma_fast_slope": 0.05, "promoter_holding": 0.05
                },
                "penalties": {
                    "price_vs_primary_trend_pct": {"op": "<", "val": 0, "pen": 0.5},
                    "roe": {"op": "<", "val": 10, "pen": 0.3},
                    "fcf_yield": {"op": "<", "val": 2, "pen": 0.3},
                    "promoter_pledge": {"op": ">", "val": 15.0, "pen": 0.2},
                    "ocf_vs_profit": {"op": "<", "val": 0.6, "pen": 0.5}
                },
                "thresholds": {"buy": 7.5, "hold": 6.0, "sell": 4.0}
            },
            "gates": {
                "min_trend_strength": 3.0, "allowed_supertrend_counter": False,
                "volatility_bands_atr_pct": [1.0, 15.0],
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 4.0,
                    "VOLATILITY_SQUEEZE": 6.0,
                    "TREND_PULLBACK": 4.5,
                    "default": 4
                }
            },
            "confidence": {
                "horizon_discount": 0, "floors": {"buy": 60, "wait": 40},
                "volume_penalty": {"rvol_drought_penalty": -10, "ignore_for_squeeze": True},
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 65}, 
                    "moderate": {"adx_min": 25, "floor": 55}, 
                    "weak": {"adx_min": 20, "floor": 45},
                    "range_bound": {"adx_max": 20, "floor": 35} # 
                },
                "setup_type_overrides": { # 
                    "MOMENTUM_BREAKOUT": 3, "VOLATILITY_SQUEEZE": 5, "QUALITY_ACCUMULATION": 10
                }
            },
            "moving_averages": {
                "type": "WMA", "fast": 10, "mid": 40, "slow": 50, 
                "keys": ["wma_10", "wma_40", "wma_50"], "dip_buy_reference": "wma_40"
            },
            "momentum_thresholds": {
                "rsi_slope": {"acceleration_floor": 0.03, "deceleration_ceiling": -0.03},
                "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
            },
            "volatility": {
                "scoring_thresholds": {"atr_pct": {"excellent": 5.5, "good": 9.0, "fair": 13.0, "poor": 18.0}, "bb_width": {"tight": 4.0, "normal": 8.0, "wide": 15.0}}
            },
            "risk_management": {
                "max_position_pct": 0.03,
                "setup_size_multipliers": {"default": 1.0},
                "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
                "min_rr_ratio": 1.5,
                "horizon_t2_cap": 0.20,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 35, "t1_mult": 2.5, "t2_mult": 5.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 2.0, "t2_mult": 4.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.5, "t2_mult": 3.0}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 2.5, "target_atr_mult": 5.0, "max_hold_candles": 52,
                "dip_buy_reference": "wma_40", "risk_reward_min": 2.5
            },
            "indicators": {
                "rsi_period": 14, "adx_period": 14, "atr_period": 14,
                "supertrend": {"period": 10, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "classic"
            },
            "lookback": {"python_data": 800}
        },

        # ========================================================
        # MULTIBAGGER
        # ========================================================
        "multibagger": {
            "enhancements": {
                "quality_technical_setup": {
                    "condition": "setup_type == 'QUALITY_ACCUMULATION' and trend_strength >= 4.0",
                    "amount": 20.0,
                    "reason": "Quality stock in early accumulation",
                    "max_boost": 25.0 
                },
                "growth_combo": {
                    "condition": "eps_growth_5y >= 25 and market_cap_cagr >= 25",
                    "amount": 18.0,
                    "reason": "Consistent compounding + price performance",
                    "max_boost": 20.0 
                },
                "moat_indicators": {
                    "condition": "roic >= 20 and roe >= 25 and de_ratio < 0.3",
                    "amount": 15.0,
                    "reason": "Economic moat indicators present",
                    "max_boost": 20.0 
                },
                "undiscovered_gem": {
                    "condition": "institutional_ownership < 15 and market_cap < 10000",
                    "amount": 12.0,
                    "reason": "Under-the-radar quality stock",
                    "max_boost": 15.0 
                }
            },
            "timeframe": "1mo",
            "description": "Deep Value & Compounders",
            "scoring": {
                "metrics": {
                    "eps_growth_5y": 0.10, "revenue_growth_5y": 0.10, "quarterly_growth": 0.05,
                    "market_cap_cagr": 0.08, "roic": 0.10, "roe": 0.08, "peg_ratio": 0.08,
                    "r_d_intensity": 0.05, "ocf_vs_profit": 0.06, "rel_strength_nifty": 0.05,
                    "institutional_ownership": 0.03, "promoter_holding": 0.05
                },
                "penalties": {
                    "peg_ratio": {"op": ">", "val": 3.0, "pen": 0.3},
                    "market_cap": {"op": ">", "val": 1000000000000, "pen": 0.5},
                    "de_ratio": {"op": ">", "val": 1.0, "pen": 0.2},
                    "roe": {"op": "<", "val": 12, "pen": 0.2},
                    "institutional_ownership": {"op": ">", "val": 85, "pen": 0.3},
                    "promoter_pledge": {"op": ">", "val": 10.0, "pen": 0.4}
                },
                "thresholds": {"buy": 8.0, "hold": 6.5, "sell": 4.5}
            },
            "gates": {
                "min_trend_strength": None, "allowed_supertrend_counter": False,
                "volatility_bands_atr_pct": [1.5, 25.0],
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 5.0,
                    "VOLATILITY_SQUEEZE": 7.0,
                    "TREND_PULLBACK": 5.0,
                    "default": 4.5
                }
            },
            "confidence": {
                "horizon_discount": 0, "floors": {"buy": 65, "wait": 50},
                "volume_penalty": {"rvol_drought_penalty": -5, "ignore_for_squeeze": True},
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 70}, 
                    "moderate": {"adx_min": 25, "floor": 60}, 
                    "weak": {"adx_min": 20, "floor": 50},
                    "range_bound": {"adx_max": 20, "floor": 40} # 
                },
                "setup_type_overrides": { # 
                    "MOMENTUM_BREAKOUT": 0, "VOLATILITY_SQUEEZE": 0, "QUALITY_ACCUMULATION": 15
                }
            },
            "moving_averages": {
                "type": "MMA", "fast": 6, "mid": 12, "slow": 12, 
                "keys": ["mma_6", "mma_12", "mma_12"], "dip_buy_reference": "mma_12"
            },
            "momentum_thresholds": {
                "rsi_slope": {"acceleration_floor": 0.02, "deceleration_ceiling": -0.02},
                "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
            },
            "volatility": {
                "scoring_thresholds": {"atr_pct": {"excellent": 11.5, "good": 18.0, "fair": 27.0, "poor": 36.0}, "bb_width": {"tight": 6.0, "normal": 12.0, "wide": 20.0}}
            },
            "risk_management": {
                "max_position_pct": 0.05,
                "setup_size_multipliers": {"QUALITY_ACCUMULATION": 1.5},
                "atr_sl_limits": {"max_percent": 0.10, "min_percent": 0.02},
                "min_rr_ratio": 2.0,
                "horizon_t2_cap": 1.00,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 30, "t1_mult": 3.0, "t2_mult": 10.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 2.5, "t2_mult": 8.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 2.0, "t2_mult": 6.0}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 3.0, "target_atr_mult": 10.0, "max_hold_candles": 60,
                "dip_buy_reference": "mma_12", "risk_reward_min": 3.0
            },
            "indicators": {
                "rsi_period": 14, "adx_period": 20, "atr_period": 12,
                "supertrend": {"period": 10, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "classic"
            },
            "lookback": {"python_data": 3000}
        }
    }
}

# ============================================================
# 5. CONFIG INTEGRITY CHECK (Self-Validation)
# ============================================================
def validate_master_config(cfg):
    """
    Validates the structure of MASTER_CONFIG_V10 at startup.
    Raises RuntimeError if critical keys are missing.
    """
    required_globals = {"pattern_physics", "calculation_engine", "boosts", "system"}
    required_horizons = {"intraday", "short_term", "long_term", "multibagger"}
    required_subkeys = {"scoring", "execution", "indicators", "gates", "risk_management"}
    
    errors = []

    # 1. Check Global Keys
    if "global" not in cfg:
        errors.append("Missing top-level 'global' section.")
    else:
        missing_globals = required_globals - cfg["global"].keys()
        if missing_globals:
            errors.append(f"Missing global sections: {missing_globals}")

    # 2. Check Horizon Keys
    if "horizons" not in cfg:
        errors.append("Missing top-level 'horizons' section.")
    else:
        missing_horizons = required_horizons - cfg["horizons"].keys()
        if missing_horizons:
            errors.append(f"Missing horizons: {missing_horizons}")
        
        # 3. Deep Check per Horizon
        for h, hc in cfg["horizons"].items():
            # Check L1 keys (scoring, execution, etc.)
            missing_subs = required_subkeys - hc.keys()
            if missing_subs:
                errors.append(f"Horizon '{h}' missing sections: {missing_subs}")
            
            # Check L2 scoring metrics existence
            if "scoring" in hc and "metrics" not in hc["scoring"]:
                errors.append(f"Horizon '{h}' scoring missing 'metrics'.")

    if errors:
        error_msg = "\n❌ CRITICAL CONFIG ERROR:\n" + "\n".join(f"- {e}" for e in errors)
        raise RuntimeError(error_msg)
    
    print("✅ MASTER_CONFIG_V10 integrity check passed.")

# Run immediately on import
try:
    validate_master_config(MASTER_CONFIG)
except RuntimeError as e:
    import sys
    print(e)
    sys.exit(1)

# 2. Wrap it in the Guard
MASTER_CONFIG = ConfigGuard(MASTER_CONFIG)

# ==========================OLD==================================
ADX_HORIZON_CONFIG = {
    "intraday": 10,     # Fast
    "short_term": 14,   # Standard
    "long_term": 14,
    "multibagger": 20   # Slow/Smooth
}

STOCH_HORIZON_CONFIG = {
    "intraday": {"k": 8, "d": 3, "smooth": 3},   # Faster
    "short_term": {"k": 14, "d": 3, "smooth": 3}, # Standard
    "long_term": {"k": 14, "d": 3, "smooth": 3},
    "multibagger": {"k": 21, "d": 5, "smooth": 5} # Very smooth
}

# ATR-based stoploss/target multipliers
ATR_MULTIPLIERS = {
    "short_term": {"tp": 3.0, "sl": 2.0},   # ← NEW (wider stops!)
    "long_term": {"tp": 3.5, "sl": 2.0},
    "multibagger": {"tp": 4.0, "sl": 2.5},
    "intraday": {"tp": 2.0, "sl": 1.5}
}

ATR_HORIZON_CONFIG = {
    "intraday": 10,     # Faster reaction for scalping
    "short_term": 14,   # Standard Swing default
    "long_term": 20,    # Smoother for weekly charts
    "multibagger": 12   # Monthly (1 Year Rolling)
}

flowchart_mapping = {
    # 1️⃣ TECHNICAL INDICATORS
    "RSI": ("quick_score", None, "RSI"),
    "MACD": ("quick_score", None, "MACD"),
    "EMA Crossover": ("quick_score", None, "EMA Crossover Trend"),
    "20 EMA": ("quick_score", None, "20 EMA"),
    "50 EMA": ("quick_score", None, "50 EMA"),
    "Bollinger Bands": ("quick_score", None, "BB Signal"),
    "BB High": ("quick_score", None, "BB High"),
    "BB Low": ("quick_score", None, "BB Low"),
    "ATR (Volatility)": ("quick_score", None, "ATR (14)"),
    "Volume Spike": ("quick_score", None, "Volume Spike Signal"),
    "Ichimoku Cloud": ("quick_score", None, "Ichimoku Cloud"),
    "Stochastic Oscillator": ("quick_score", None, "Stoch %K"),
    "Relative Volume (RVOL)": ("quick_score", None, "Relative Volume (RVOL)"),
    "OBV Divergence": ("quick_score", None, "OBV Divergence"),
    "Pivot Points / Fibonacci Levels": ("quick_score", None, "Entry Price (Confirm)"),
    # 2️⃣ FUNDAMENTAL METRICS
    "P/E Ratio": ("fundamentals", None, "Valuation (P/E)"),
    "PEG Ratio": ("fundamentals", None, "PEG Ratio"),
    "ROE": ("fundamentals", None, "Return on Equity (ROE)"),
    "Debt-to-Equity": ("fundamentals", None, "Debt to Equity"),
    "Free Cash Flow Growth": (
        "fundamentals",
        None,
        "FCF Yield (%)",
    ),  # you don't have CAGR yet
    "Dividend Yield": ("fundamentals", None, "Dividend Yield"),
    "Management Quality": ("extended", None, "Promoter Holding (%)"),
    "EPS Growth Consistency": ("extended", None, "EPS Growth Consistency (5Y CAGR)"),
    "Interest Coverage Ratio": ("fundamentals", None, "Interest Coverage"),
    "Operating Cash Flow vs Net Profit": (
        "extended",
        None,
        "Operating CF vs Net Profit",
    ),
    "R&D Intensity": ("extended", None, "R&D Intensity (%)"),
    "Asset Turnover Ratio": ("fundamentals", None, "Asset Turnover Ratio"),
    "Book Value Growth": ("fundamentals", None, "Price to Book (P/B)"),
    # 3️⃣ MULTIBAGGER IDENTIFICATION
    "ROIC": ("fundamentals", None, "ROIC (%)"),
    "Earnings Growth": ("fundamentals", None, "Net Profit Qtr Growth YoY %"),
    "Promoter Holding": ("extended", None, "Promoter Holding (%)"),
    "Market Cap CAGR": ("extended", None, "Market Cap CAGR"),
    "TAM Growth": ("extended", None, "TAM Growth"),
    "Debt/Equity": ("fundamentals", None, "Debt to Equity"),
    "PEG (Forward)": (
        "fundamentals",
        None,
        "PEG Ratio",
    ),  # fallback since no Forward P/E
    "Promoter Pledge": ("extended", None, "Promoter Pledge"),
    "Institutional Ownership Trend": ("extended", None, "Institutional Ownership (%)"),
    "Innovation/R&D Pipeline": ("extended", None, "R&D Intensity (%)"),
    "Sector Leadership": ("fundamentals", None, "Sector"),
    # 4️⃣ SENTIMENT & BEHAVIORAL FACTORS
    "VIX": ("extended", None, "VIX (Volatility Index)"),
    "Analyst Ratings": ("extended", None, "Analyst Ratings"),
    "Fear & Greed Index": ("extended", None, "Fear & Greed Index"),
    "Retail Sentiment": ("extended", None, "Retail Sentiment"),
    "Put-Call Ratio (PCR)": ("extended", None, "Put-Call Ratio (PCR)"),
    "Insider Trading Activity": ("extended", None, "Insider Trading Activity"),
    "Advance-Decline Line (A/D)": ("extended", None, "Advance-Decline Line (A/D)"),
    "News/Google Trends": ("extended", None, "News/Google Trends"),
    # 5️⃣ MACRO-ECONOMIC CONTEXT
    "GDP Growth": ("extended", None, "GDP Growth (%)"),
    "Inflation Rate": ("extended", None, "Inflation Rate (%)"),
    "Interest Rate Trend": ("extended", None, "Repo Rate (%)"),  # from macro_sentiment
    "Crude Oil Prices": ("extended", None, "Crude Oil ($)"),
    "Bond Yield vs Equity Spread": (
        "extended",
        None,
        "10Y Bond Yield (%)",
    ),  # macro proxy
    "Currency Trend": ("extended", None, "Currency Trend (USD/INR)"),
    "PMI": ("extended", None, "PMI"),
    "Sector Rotation": ("payload", None, "index"),
    # 6️⃣ RISK MANAGEMENT
    "Stop-Loss": ("quick_score", None, "Suggested SL (2xATR)"),
    "Position Sizing": ("payload", None, None),
    "Max Drawdown": ("extended", None, "Max Drawdown (%)"),
    "Beta": ("fundamentals", None, "Beta"),
    "Sharpe Ratio": ("extended", None, "Sharpe Ratio"),
    "Sortino Ratio": ("extended", None, "Sortino Ratio"),
    "Correlation Matrix": ("extended", None, "Correlation Matrix"),
    "Drawdown Recovery Period": ("extended", None, "Drawdown Recovery Period"),
}
#todo remove
TECHNICAL_WEIGHTS = {
    "rsi": 1.0,
    "macd_cross": 1.0,
    "macd_hist_z": 0.8,
    "price_vs_200dma_pct": 1.0,
    "adx": 1.0,
    "vwap_bias": 0.8,
    "vol_trend": 0.6,
    "rvol": 0.6,
    "stoch_k": 0.6,
    "bb_low": 0.4,
    "bb_width": 0.3,
    "entry_confirm": 0.5,
    "ema_20_50_cross": 0.8,
    "dma_200_slope": 0.8,
    "ichi_cloud": 1.0,
    "obv_div": 0.6,
    "atr_14": 0.8,
    "vol_spike_ratio": 0.5,
    "rel_strength_nifty": 0.6,
    "price_action": 0.7,
    "supertrend_signal": 1.0,
    "cci": 0.6,
    "bb_percent_b": 0.4,
    "cmf_signal": 0.6,
    "donchian_signal": 0.8,
    "reg_slope": 0.8,
}

TECHNICAL_METRIC_MAP = {
    # Price / meta
    "price": "Current Price",
    "prev_close": "Previous Close",

    # Trend / Moving averages “DMA” should be considered Daily Moving Average (SMA),But your dynamic logic never generates dma_XX anymore we're using:EMA for daily (intraday/short_term) WMA-label for weekly (long_term) MMA-label for monthly (multibagger)
    "dma_20": "20 DMA",
    "dma_50": "50 DMA",
    "dma_200": "200 DMA",
    "dma_10": "10 DMA",
    "dma_40": "40 DMA",
    "wma_50": "50WMA",
    "price_vs_50wma_pct": "Price vs 50WMA (%)",
    "price_vs_200dma_pct": "Price vs 200 DMA (%)",
    "dma_200_slope": "200 DMA Slope",
    # === Dynamic MA Mapping (Fully Horizon-Aware) ===
    # Intraday / Short Term (EMA-based)
    "ema_20": "20 EMA (Short-Term Trend)",
    "ema_50": "50 EMA (Medium-Term Trend)",
    "ema_200": "200 EMA (Long-Term Trend)",
    # EMA Crossovers (Intraday / Short-Term)
    "ema_20_50_cross": "EMA 20/50 Crossover",
    "ema_20_200_cross": "EMA 20/200 Crossover",
    "ema_50_200_cross": "EMA 50/200 Crossover",
    # EMA Trend Stacking
    "ema_20_50_200_trend": "EMA Trend Alignment (20 > 50 > 200)",
    # Long-Term Horizon (Weekly MAs) — WMA prefix, SMA math
    "wma_10": "10-Week MA",
    "wma_40": "40-Week MA",
    "wma_50": "50-Week MA",
    # Weekly Crossover
    "wma_10_40_cross": "Weekly MA Crossover (10/40)",
    # Weekly Trend Stacking
    "wma_10_40_50_trend": "Weekly Trend Alignment (10 > 40 > 50)",
    # Multibagger Horizon (Monthly MAs) — MMA prefix, SMA math
    "mma_6": "6-Month MA",
    "mma_12": "12-Month MA",
    # Monthly Crossover
    "mma_6_12_cross": "Monthly MA Crossover (6/12)",
    # Monthly Trend Stacking
    "mma_6_12_12_trend": "Monthly Trend Alignment (6 > 12 > 12)",
    # Generic Crossover Trend Key (Used by dynamic MA Trend)
    "ma_cross_trend": "Composite MA Trend Signal",
    "ema_20_slope": "20 EMA Slope",
    "ema_50_slope": "50 EMA Slope",
    "wma_50_slope": "50 WMA Slope",
    "mma_12_slope": "12-Month MA Slope",
    "ma_cross_setup": "MA Crossover Setup",
    "ma_trend_setup": "MA Trend Setup",

    # Momentum
    "rsi": "RSI",
    "rsi_slope": "RSI Slope",
    "dma_20_50_cross": "20/50 DMA Cross",
    "dma_10_40_cross": "10/40 DMA Cross",
    "short_ma_cross": "Short MA Cross",
    "macd": "MACD",
    "macd_cross": "MACD Cross",
    "macd_hist_z": "MACD Hist Z-Score",
    "macd_histogram": "MACD Histogram (Raw)",
    "mfi": "MFI",
    "stoch_k": "Stoch %K",
    "stoch_d": "Stoch %D",
    "stoch_cross": "Stoch Crossover",
    "cci": "CCI",
    "adx": "ADX",
    "adx_signal": "ADX Signal",
    "di_plus": "DI+",
    "di_minus": "DI-",

    # Volatility / volume
    "atr_14": "ATR (14)",
    "atr_pct": "ATR %",
    "true_range": "True Range (Raw)",
    "true_range_pct": "True Range % of Price",
    "hv_10": "Historical Volatility (10D)",
    "hv_20": "Historical Volatility (20D)",
    "rvol": "Relative Volume (RVOL)",
    "vol_spike_ratio": "Volume Spike Ratio",
    "vol_spike_signal": "Volume Spike Signal",
    "vol_trend": "Volume Trend",
    "vpt": "VPT",
    "cmf_signal": "Chaikin Money Flow (CMF)",
    "obv_div": "OBV Divergence",

    # Bands / Channel
    "bb_high": "BB High",
    "bb_mid": "BB Mid",
    "bb_low": "BB Low",
    "bb_width": "BB Width",
    "bb_percent_b": "Bollinger %B",
    "ttm_squeeze": "TTM Squeeze Signal",
    "kc_upper": "Keltner Upper",
    "kc_lower": "Keltner Lower",
    "donchian_signal": "Donchian Channel Breakout",
    "ichi_cloud": "Ichimoku Cloud",
    "ichi_span_a": "Ichimoku Span A",
    "ichi_span_b": "Ichimoku Span B",
    "ichi_tenkan": "Tenkan-sen",
    "ichi_kijun": "Kijun-sen",

    # Levels / pivots
    "pivot_point": "Pivot Point (Daily)",
    "resistance_1": "Resistance 1 (Fib)",
    "resistance_2": "Resistance 2 (Fib)",
    "resistance_3": "Resistance 3 (Fib)",
    "support_1": "Support 1 (Fib)",
    "support_2": "Support 2 (Fib)",
    "support_3": "Support 3 (Fib)",
    "entry_confirm": "Entry Price (Confirm)",
    "gap_percent": "Gap %",

    # Misc / signals
    "psar_trend": "Parabolic SAR Trend",
    "psar_level": "PSAR Level",
    "supertrend_signal": "SuperTrend Signal",
    "supertrend_value": "Supertrend Value",
    "price_action": "Price Action",
    "vwap": "VWAP",
    "vwap_bias": "VWAP Bias",

    # Relative / benchmark
    "rel_strength_nifty": "Relative Strength vs NIFTY (%)",
    "nifty_trend_score": "NIFTY Trend Score",

    # Composite placeholders (some are computed in signal_engine but include them so profile keys don't break)
    "fundamental_momentum": "Fundamental Momentum",
    "price_vs_avg": "Price vs Average",

    # Utility / reporting
    "sl_2x_atr": "Suggested SL (2xATR)",
    "technical_score": "Technical Score",
    "Horizon": "Horizon",
    "wick_rejection": "Wick Rejection",
    "atr_dynamic": "Dynamic ATR",
    "sl_atr_dynamic": "Stop Loss (Dynamic ATR)",
    "risk_per_share_pct": "Risk Per Share (%)",
    "atr_sma_ratio": "ATR/SMA Ratio",

    #pattern Key
    "darvas_box": "Darvas Box Pattern",
    "cup_handle": "Cup & Handle Pattern",
    "flag_pennant": "Flag/Pennant Pattern",
    "bollinger_squeeze": "Bollinger Squeeze",
    "golden_cross": "Golden/Death Cross",
    "double_top_bottom": "Double Top/Bottom",
    "three_line_strike": "Three-Line Strike",
    "minervini_stage2": "Minervini VCP / Stage 2",
    "ichimoku_signals": "Ichimoku Signals",
    "trend_strength": "Trend Strength (Composite ADX+Slope+DI)",
    "momentum_strength": "Momentum Strength (Composite RSI+MACD+Stoch)",
    "volatility_quality": "Volatility Quality (0-10 Score)",
    
    # Universal MA Keys
    "ma_fast_slope": "Primary MA Slope (Horizon-Aware)",
    "ma_mid_slope": "Secondary MA Slope",
    "ma_slow_slope": "Tertiary MA Slope",




}

CORE_TECHNICAL_SETUP_METRICS = [
        "rsi", 
        "ema_20", 
        "ema_200", 
        "bb_high", 
        "bb_low", 
        "ttm_squeeze", 
        "atr_14",          # Needed for Stop Loss
        "price_action",    # Good context
        "volatility_quality" # Needed for Confidence Score
    ]
# -------------------------
# Updated fundamental weights (short keys)
# -------------------------
FUNDAMENTAL_WEIGHTS = {
    # --- Valuation (20%) ---
    "pe_ratio": 0.05,
    "pb_ratio": 0.04,
    "peg_ratio": 0.03,
    "fcf_yield": 0.05,
    "dividend_yield": 0.03,
    # --- Profitability / Returns (25%) ---
    "roe": 0.10,
    "roce": 0.07,
    "roic": 0.08,
    # --- Leverage / Liquidity (15%) ---
    "de_ratio": 0.05,
    "interest_coverage": 0.05,
    "current_ratio": 0.03,
    "ocf_vs_profit": 0.02,
    # --- Efficiency / Quality (20%) ---
    "asset_turnover": 0.04,
    "piotroski_f": 0.07,
    "r_d_intensity": 0.04,
    "earnings_stability": 0.05,
    # --- Growth (15%) ---
    "eps_growth_5y": 0.06,
    "fcf_growth_3y": 0.05,
    "market_cap_cagr": 0.04,
    # --- Ownership / Market Sentiment (5%) ---
    "promoter_holding": 0.015,
    "institutional_ownership": 0.015,
    "beta": 0.01,
    "52w_position": 0.01,
    "dividend_payout": 0.03,
    "yield_vs_avg": 0.02,
}

# Safety normalization (ensures sum = 1.0)
_total = sum(FUNDAMENTAL_WEIGHTS.values())
if abs(_total - 1.0) > 1e-3:
    FUNDAMENTAL_WEIGHTS = {k: v / _total for k, v in FUNDAMENTAL_WEIGHTS.items()}

FUNDAMENTAL_ALIAS_MAP = {
    "pe_ratio": "P/E Ratio",
    "pb_ratio": "Price to Book (P/B)",
    "peg_ratio": "PEG Ratio",
    "ps_ratio": "Price-to-Sales (P/S)",
    "pe_vs_sector": "P/E vs Sector",
    "fcf_yield": "FCF Yield (%)",
    "dividend_yield": "Dividend Yield (%)",
    "dividend_payout": "Dividend Payout (%)",
    "market_cap": "Market Cap",
    "market_cap_cagr": "Market Cap CAGR (%)",
    # Profitability / returns
    "roe_history": "ROE History",
    "roe": "Return on Equity (ROE)",
    "roce": "Return on Capital Employed (ROCE)",
    "roic": "Return on Invested Capital (ROIC)",
    "net_profit_margin": "Net Profit Margin (%)",
    "operating_margin": "Operating Margin (%)",
    "ebitda_margin": "EBITDA Margin (%)",
    "fcf_margin": "FCF Margin (%)",
    # Growth
    "revenue_growth_5y": "Revenue Growth (5Y CAGR)",
    "profit_growth_3y": "Profit Growth (3Y CAGR)",
    "eps_growth_5y": "EPS Growth (5Y CAGR)",
    "eps_growth_3y": "EPS Growth (3Y CAGR)",
    "fcf_growth_3y": "FCF Growth (3Y CAGR)",
    "quarterly_growth": "Quarterly Growth (EPS/Rev)",
    # Health / liquidity
    "de_ratio": "Debt to Equity",
    "interest_coverage": "Interest Coverage Ratio",
    "current_ratio": "Current Ratio",
    "ocf_vs_profit": "Operating CF vs Net Profit",
    # Quality / efficiency
    "piotroski_f": "Piotroski F-Score",
    "asset_turnover": "Asset Turnover Ratio",
    "r_d_intensity": "R&D Intensity (%)",
    "earnings_stability": "Earnings Stability",
    # Ownership / market
    "promoter_holding": "Promoter Holding (%)",
    "promoter_pledge": "Promoter Pledge (%)",
    "institutional_ownership": "Institutional Ownership (%)",
    "short_interest": "Short Interest",
    "analyst_rating": "Analyst Rating (Momentum)",
    "52w_position": "52W Position (off-high %)",
    "beta": "Beta",
    "days_to_earnings": "Days to Next Earnings",
    "ps_ratio": "Price-to-Sales (P/S)",
    # reporting/meta
    "base_score": "Base Fundamental Score",
    "final_score": "Final Fundamental Score",
    "_meta": "Meta",
    "52w_high": "52 week high",
    "52w_low": "52 week low",
    "volatility_adjusted_roe": "ROE/Volatility Ratio",
    "price_vs_intrinsic_value": "Price vs Intrinsic Value",
    "fcf_yield_vs_volatility": "FCF Yield vs Volatility",
    "earnings_consistency_index": "Earnings Consistency Index",
    "roe_stability": "ROE Stability (StdDev)"

}
FUNDAMENTAL_FIELD_CANDIDATES = {
    # Income Statement
    "revenue": [
        "Total Revenue",
        "Revenue",
        "totalRevenue",
        "Sales",
        "Net Sales",
        "Operating Revenue",
        "Total Sales",
        "Gross Sales",
    ],
    "net_income": [
        "Net Income",
        "netIncome",
        "NetIncome",
        "Profit After Tax",
        "Net Profit",
        "Profit",
        "PAT",
        "Net Loss",
        "Net Income Common Stockholders",
    ],
    "operating_income": [
        "Operating Income",
        "EBIT",
        "Ebit",
        "Operating Profit",
        "OperatingProfit",
        "Profit from Operations",
    ],
    "ebit": [
        "EBIT",
        "Ebit",
        "Operating Income",
        "Operating Profit",
        "Profit from Operations",
    ],
    "ebitda": [
        "EBITDA",
        "Ebitda",
        "Operating Profit Before Depreciation",
        "Normalized EBITDA",
    ],
    "cogs": [
        "Cost Of Revenue",
        "Cost of Goods Sold",
        "COGS",
        "Total Expenses",
        "Operating Expense",
    ],
    "interest_expense": [
        "Interest Expense",
        "Interest And Debt Expense",
        "Finance Cost",
        "Interest",
        "Total Interest Expense",
    ],
    "tax_expense": ["Income Tax Expense", "Tax Provision", "Total Tax Expense", "Tax"],
    "pre_tax_income": [
        "Pretax Income",
        "Income Before Tax",
        "Income Before Tax Expense",
        "PretaxProfit",
    ],
    # Balance Sheet
    "total_assets": ["Total Assets", "totalAssets", "Assets"],
    "current_assets": [
        "Total Current Assets",
        "totalCurrentAssets",
        "Current Assets",
        "currentAssets",
    ],
    "current_liabilities": [
        "Total Current Liabilities",
        "totalCurrentLiabilities",
        "Current Liabilities",
        "currentLiabilities",
    ],
    "cash_equivalents": [
        "Cash And Cash Equivalents",
        "cashAndCashEquivalents",
        "Cash",
        "Cash Balance",
        "Cash & Equivalents",
        "Cash & Bank Balances",
    ],
    "total_liabilities": [
        "Total Liabilities",
        "Total Current Liabilities",
        "Liabilities",
        "Total Liab",
    ],
    "total_equity": [
        "Total Equity",
        "Total Stockholders Equity",
        "totalStockholdersEquity",
        "Shareholders Equity",
        "Equity",
        "Stockholders Equity",
        "Shareholder Equity",
        "Total Common Equity",
        "Total Stockholder Equity",
        "Shareholder's funds",
        "Total shareholders' funds",
    ],
    "total_debt": [
        "Total Debt",
        "totalDebt",
        "Long Term Debt",
        "Short Long Term Debt",
        "Long Term Borrowings",
        "Short Term Borrowings",
        "Debt",
        "Borrowings",
    ],
    "pure_borrowings": [
        "Short Term Borrowings",
        "ShortTermBorrowings",
        "Short Term Debt",
        "ShortTermDebt",
        "Long Term Borrowings",
        "LongTermBorrowings",
        "Long Term Debt",
        "LongTermDebt",
        "Borrowings",
        "borrowings",
    ],
    # Cash Flow Statement
    "ocf": [
        "Total Cash From Operating Activities",
        "totalCashFromOperatingActivities",
        "Operating Cash Flow",
        "Cash Flow From Operating Activities",
    ],
    "capex": [
        "Capital Expenditures",
        "capitalExpenditures",
        "CapEx",
        "Purchase Of Fixed Assets",
    ],
    "free_cash_flow": [
        "Free Cash Flow",
        "freeCashflow",
        "freeCashFlow",
        "FCF",
        "Free Cash Flow (FCF)",
    ],
    # Other metrics / ratios
    "rd_expense": [
        "Research And Development",
        "Research Development",
        "Research and Development Expense",
        "R&D",
        "Rnd",
    ],
    "eps": ["EPS", "Diluted EPS", "Basic EPS", "Earnings Per Sare", "eps"],
    "shares_outstanding": [
        "Basic Average Shares",
        "Shares Outstanding",
        "Weighted Average Shares",
        "Shares",
    ],
    "gross_profit": ["Gross Profit", "GrossIncome"],
    "market_cap": ["marketCap", "Market Capitalization", "market_cap", "Market Cap"],
    "book_value": ["bookValue", "Book Value", "Book value per share"],
    "dividend": ["Dividends Paid", "dividendRate", "Cash Dividends Paid"],
    "fcf_yield": ["Free Cash Flow Yield", "fcfYield"],
    "promoter_holding": [
        "heldPercentInsiders",
        "insiderPercent",
        "insidersPercent",
        "Insider Ownership",
    ],
    "institutional_ownership": [
        "heldPercentInstitutions",
        "institutionPercent",
        "institutionsPercent",
        "Institutional Ownership",
    ],
    "dividend_yield": ["dividendYield"],
    "analyst_rating": ["recommendations", "recommendationKey", "recommendationMean"],
    "quarterly_growth": ["earningsQuarterlyGrowth", "revenueQuarterlyGrowth"],
    "short_interest": ["shortRatio", "sharesPercentSharesOut", "shortPercentOfFloat"],
    "trend_strength": [],
}

SECTOR_PE_AVG = {
    "Technology": 58.7,
    "Financial Services": 26.4,
    "Healthcare": 38.5,
    "Consumer Defensive": 40.1,
    "Consumer Cyclical": 33.8,
    "Energy": 18.9,
    "Industrials": 29.4,
}

# ============================================================================
# PRODUCTION-READY HORIZON_PROFILE_MAP
# Uses Standardized Dynamic Keys + Fixes from Analysis
# ============================================================================

HORIZON_PROFILE_MAP = {
    "intraday": {
        "metrics": {
            "ma_fast_slope": 0.20,          # ✅ Boosted (was 0.15)
            "rsi_slope": 0.20,              # ✅ Boosted (was 0.15)
            "price_action": 0.15,           # ✅ Kept
            "vwap_bias": 0.15,
            "vol_spike_ratio": 0.10,
            "volatility_quality": 0.05,     # ✅ Reduced (let engine handle)
            "ma_trend_signal": 0.05,
            "momentum_strength": 0.10,
        },
        "penalties": {
            # ✅ REMOVED: bb_width (Let squeeze score high!)
            # ✅ REMOVED: nifty_trend_score (Stock-specific plays allowed)
            "gap_percent": {"operator": "<", "value": 0.1, "penalty": 0.1},  # ✅ Relaxed
            "ma_fast_slope": {"operator": "<", "value": -2, "penalty": 0.3},  # ✅ Relaxed (was 0)
            "atr_pct": {"operator": "<", "value": 0.4, "penalty": 0.3},      # ✅ FIXED (was 0.75)
        },
        "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 3.5},  # Was: 6.5, 5.0, 3.5
    },

    "short_term": {
        "metrics": {
            "trend_strength": 0.15,         # ✅ Key metric
            "ma_trend_signal": 0.10,
            "price_vs_primary_trend_pct": 0.08,  # ✅ Reduced (pullbacks OK)
            "ma_fast_slope": 0.05,
            "supertrend_signal": 0.10,
            "momentum_strength": 0.12,      # ✅ Boosted (was 0.10)
            "rsi_slope": 0.08,              # ✅ Boosted (was 0.05)
            "macd_cross": 0.05,
            "cmf_signal": 0.05,
            "obv_div": 0.05,
            "volatility_quality": 0.05,
            "rvol": 0.05,
            "quarterly_growth": 0.03,
            "analyst_rating": 0.02,
            "nifty_trend_score": 0.02,      # ✅ Reduced (was 0.05)
        },
        "penalties": {
            "days_to_earnings": {"operator": "<", "value": 3, "penalty": 0.5},  # ✅ FIXED (was 7 days, -1.0)
            # ✅ REMOVED: price_vs_primary_trend_pct (Allow dip buys)
            "ma_fast_slope": {"operator": "<", "value": -5, "penalty": 0.3},    # ✅ Kept (severe downtrend)
            "rvol": {"operator": "<", "value": 0.5, "penalty": 0.2}            # ✅ Relaxed (was 0.8)
        },
        "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 3.8},  # Was: 6.5, 5.0, 4.0
    },

    "long_term": {
        # ✅ No changes needed - already balanced
        "metrics": {
            "ma_trend_signal": 0.15,
            "ma_fast_slope": 0.10,
            "price_vs_primary_trend_pct": 0.10,
            "roe": 0.10,
            "roce": 0.08,
            "roic": 0.08,
            "earnings_stability": 0.08,
            "fcf_yield": 0.08,
            "eps_growth_5y": 0.06,
            "piotroski_f": 0.05,
            "de_ratio": 0.03,
            "promoter_holding": 0.05,
            "rel_strength_nifty": 0.04,
            # 🟢 SYNC WITH STRATEGY ANALYZER: Valuation Check
            "peg_ratio": 0.05 
        },
        "penalties": {
            "price_vs_primary_trend_pct": {"operator": "<", "value": 0, "penalty": 0.5},
            "roe": {"operator": "<", "value": 10, "penalty": 0.3},
            "fcf_yield": {"operator": "<", "value": 2, "penalty": 0.3},
            "promoter_pledge": {"operator": ">", "value": 15.0, "penalty": 0.2},
            # 🟢 FRAUD CHECK: High Profit but No Cash Flow
            "ocf_vs_profit": {"operator": "<", "value": 0.6, "penalty": 0.5} 
        },
        "thresholds": {"buy": 7.5, "hold": 6.0, "sell": 4.0},
    },

    "multibagger": {
        # ✅ No changes needed
        "metrics": {
            "ma_trend_signal": 0.10,
            "ma_fast_slope": 0.10,
            "price_vs_primary_trend_pct": 0.05,
            "eps_growth_5y": 0.10,
            "revenue_growth_5y": 0.10,
            # 🟢 SYNC WITH CANSLIM STRATEGY: Recent Growth
            "quarterly_growth": 0.05, 
            "market_cap_cagr": 0.08,
            "roic": 0.10,
            "roe": 0.08,
            "peg_ratio": 0.08,
            "r_d_intensity": 0.05,
            "promoter_holding": 0.05,
            "institutional_ownership": 0.03,
            "ocf_vs_profit": 0.06,
            "rel_strength_nifty": 0.05,
        },
        "penalties": {
            "ma_fast_slope": {"operator": "<", "value": 0, "penalty": 0.5},
            "peg_ratio": {"operator": ">", "value": 3.0, "penalty": 0.3},
            "market_cap": {"operator": ">", "value": 1e12, "penalty": 0.5},
            "de_ratio": {"operator": ">", "value": 1.0, "penalty": 0.2},
            "roe": {"operator": "<", "value": 12, "penalty": 0.2},
            "institutional_ownership": {"operator": ">", "value": 85, "penalty": 0.3},
            "promoter_pledge": {"operator": ">", "value": 10.0, "penalty": 0.4}
        },
        "thresholds": {"buy": 8.0, "hold": 6.5, "sell": 4.5},
    }
}

HORIZON_FETCH_CONFIG = {
    "intraday": {
        "period": "1mo",   # CHANGED from '5d'. Gives ~500 bars (25 * 20 days). Plenty for EMA200 + Warmup.
        "interval": "15m", 
        "label": "Intraday"
    },
    "short_term": {
        "period": "5y",    # CHANGED from '3mo' to 2y. Gives ~250 candles. Enough for 200 DMA.
        "interval": "1d", 
        "label": "Short Term"
    },
    "long_term": {
        "period": "5y",    # changed. from 2y to 5y ~104 weekly bars. (Note: WMA 200 needs ~4 years, but you use WMA 50 here, so 2y is fine)
        "interval": "1wk", 
        "label": "Long Term"
    },
    "multibagger": {
        "period": "10y",   # CHANGED from '5y' to be safe, though 5y (60 months) is usually enough for MMA 12.
        "interval": "1mo", 
        "label": "Multibagger"
    },
}

#todo : remove

QUALITY_WEIGHTS = {
    # Higher is better
    "roe": {"weight": 1.0, "direction": "normal"},
    "roce": {"weight": 1.0, "direction": "normal"},
    "roic": {"weight": 1.0, "direction": "normal"},
    "piotroski_f": {"weight": 1.0, "direction": "normal"},
    "ocf_vs_profit": {"weight": 1.0, "direction": "normal"},
    "interest_coverage": {"weight": 1.0, "direction": "normal"},
    "earnings_stability": {"weight": 1.0, "direction": "normal"},
    "net_profit_margin": {"weight": 1.0, "direction": "normal"},
    # Lower is better
    "de_ratio": {"weight": 1.0, "direction": "invert"},
    "promoter_pledge": {"weight": 1.0, "direction": "normal"},
    "roe_stability": {
        "weight": 0.10,
        "direction": "invert",
    },  # Lower standard deviation = higher score (invert)
    "volatility_quality": {"weight": 0.10, "direction": "normal"},
}
#todo : remove
GROWTH_WEIGHTS = {
    "eps_growth_3y": {"weight": 1.0, "direction": "normal"},
    "revenue_growth_5y": {"weight": 1.0, "direction": "normal"},
    "quarterly_growth": {"weight": 1.0, "direction": "normal"},
    "fcf_growth_3y": {"weight": 1.0, "direction": "normal"},
    "market_cap_cagr": {"weight": 1.0, "direction": "normal"},
}
#todo : remove
VALUE_WEIGHTS = {
    # Lower is better
    "pe_ratio": {"weight": 1.0, "direction": "invert"},
    "pb_ratio": {"weight": 1.0, "direction": "invert"},
    "peg_ratio": {"weight": 1.0, "direction": "invert"},
    "pe_vs_sector": {"weight": 1.0, "direction": "invert"},
    # Higher is better
    "fcf_yield": {"weight": 1.0, "direction": "normal"},
    "dividend_yield": {"weight": 1.0, "direction": "normal"},
}

#todo : remove
MOMENTUM_WEIGHTS = {  # 🆕 CORE COMPOSITES (PRIORITY)
    "momentum_strength": {
        "weight": 0.30,
        "direction": "normal",
    },  # RSI, MACD, Stoch Bundle
    "trend_strength": {
        "weight": 0.40,
        "direction": "normal",
    },  # ADX, EMA Slope, ST Bundle
    "volatility_quality": {
        "weight": 0.10,
        "direction": "normal",
    },  # New Volatility Setup Score
    # CONTEXTUAL/HYBRID MOMENTUM (MUST KEEP)
    "vwap_bias": {"weight": 0.05, "direction": "normal"},
    "price_action": {"weight": 0.05, "direction": "normal"},
    "nifty_trend_score": {"weight": 0.05, "direction": "normal"},  # Macro Context
    "52w_position": {"weight": 0.05, "direction": "normal"},  # Hybrid/Sentiment Context
    # ⚠️ CONTEXTUAL/VOLUME (MUST KEEP)
    "rvol": {
        "weight": 0.00,
        "direction": "normal",
    },  # Weight mass moved to composite, keep key for context
    "obv_div": {
        "weight": 0.00,
        "direction": "normal",
    },  # Weight mass moved to composite, keep key for context
    "psar_trend": {"weight": 0.00, "direction": "normal"},
    "ttm_squeeze": {"weight": 0.00, "direction": "normal"},
}

INDEX_TICKERS = {
    # Broad Market Indices (Comprehensive Coverage)
    "nifty50": "^NSEI",  # Nifty 50 Index (Primary Benchmark)
    "nifty100": "^CNX100",  # Top 100 stocks
    "niftynext50": "^NSMIDCP",  # Top 51-100 stocks
    "nifty500": "^CRSLDX",  # Top 500 stocks (Wide Coverage)
    "midcap150": "NIFTYMIDCAP150.NS",  # Mid-cap segment
    "smallcap100": "^CNXSC",  # Small-cap segment
    "smallcap250": "NIFTYSMLCAP250.NS",  # Broader small-cap coverage
    "microcap250": "NIFTY_MICROCAP250.NS",  # Micro-cap segment
    # Sectoral Indices (Industry-specific Insight)
    "niftybank": "^NSEBANK",  # Banking sector
    "niftyit": "^CNXIT.NS",  # Information Technology sector
    "niftypharma": "^CNXPHARMA.NS",  # Pharmaceutical sector
    "niftyfmcg": "^CNXFMCG.NS",  # Fast Moving Consumer Goods sector
    "niftyauto": "^CNXAUTO.NS",  # Automotive sector
    "niftyrealty": "^CNXREALTY.NS",  # Realty sector
    "niftyinfra": "^CNXINFRA.NS",  # Infrastructure sector
    # Bombay Stock Exchange (BSE) Indices
    "sensex": "^BSESN",  # BSE Sensex (BSE Benchmark)
    "bsemidcap": "^BSEMC.BO",  # BSE Mid-cap segment
    "bsesmallcap": "^BSESC.BO",  # BSE Small-cap segment
    # Default fallback
    "default": "^NSEI",
}

# In constants.py - ADD THIS:
VOL_BANDS_HORIZON_MULTIPLIERS = {
    "intraday": 1.0,      # Use base thresholds (4% = risky)
    "short_term": 1.0,    # Daily data, use base
    "long_term": 2.5,     # Weekly: 4% × 2.5 = 10% ceiling
    "multibagger": 4.0    # Monthly: 4% × 4 = 16% ceiling ✅
}

# 3. Momentum Slopes Thresholds
RSI_SLOPE_THRESH = {
    # Default fallback
    "default": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
    
    # Intraday: Requires sharper moves to filter noise
    "intraday": {"acceleration_floor": 0.10, "deceleration_ceiling": -0.10},
    
    # Short Term: Standard Swing
    "short_term": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
    
    # Long Term: Slower moves are significant
    "long_term": {"acceleration_floor": 0.03, "deceleration_ceiling": -0.03},
    
    # Multibagger: Monthly charts move very slowly
    "multibagger": {"acceleration_floor": 0.02, "deceleration_ceiling": -0.02},
}

MACD_MOMENTUM_THRESH = {
    "acceleration_floor": 0.5,  # MACD Histogram Z-Score > 0.5
    "deceleration_ceiling": -0.5,  # MACD Histogram Z-Score < -0.5
}

# ============================================================================
# VOLATILITY QUALITY MINIMUM THRESHOLDS (UNCHANGED - These are already correct)
# ============================================================================
VOL_QUAL_MINS = {
    "intraday": {
        "MOMENTUM_BREAKOUT": 2.5,
        "VOLATILITY_SQUEEZE": 4.0,
        "TREND_PULLBACK": 3.0,
        "default": 2.5
    },
    "short_term": {
        "MOMENTUM_BREAKOUT": 3.0,
        "VOLATILITY_SQUEEZE": 5.0,
        "TREND_PULLBACK": 3.5,
        "default": 3.0
    },
    "long_term": {
        "MOMENTUM_BREAKOUT": 4.0,
        "VOLATILITY_SQUEEZE": 6.0,
        "TREND_PULLBACK": 4.5,
        "default": 4.0
    },
    "multibagger": {
        "MOMENTUM_BREAKOUT": 5.0,
        "VOLATILITY_SQUEEZE": 7.0,
        "TREND_PULLBACK": 5.0,
        "default": 4.5
    },
    "default": 3.0
}

# ============================================================================
# VOLATILITY REGIME BANDS (✅ CORRECTLY SCALED FOR TIMEFRAMES)
# ============================================================================
VOL_BANDS = {
    "intraday": {
        "min": 1.5,      # Dead money below 1.5% daily range
        "ideal": 3.0,    # Sweet spot for scalping
        "max": 6.0       # Panic selling/buying above 6%
    },
    "short_term": {      # DAILY CANDLES (Baseline)
        "min": 1.0,      # Need some swing
        "ideal": 2.5,    # Healthy volatility
        "max": 5.0       # Too choppy above 5%
    },
    "long_term": {       # ✅ WEEKLY CANDLES (Scale ~2.2x from daily)
        "min": 2.0,      # <2% weekly = dead stock (0.4% daily equivalent)
        "ideal": 5.5,    # Healthy weekly trend (1.1% daily equivalent)
        "max": 12.0      # >12% weekly = unstable (2.4% daily equivalent)
    },
    "multibagger": {     # ✅ MONTHLY CANDLES (Scale ~4.6x from daily)
        "min": 3.0,      # <3% monthly = bond-like (0.65% daily equivalent)
        "ideal": 8.0,    # Steady compounder (1.75% daily equivalent)
        "max": 25.0      # >25% monthly = extreme risk (5.5% daily equivalent)
    }
}

# ============================================================================
# VOLATILITY QUALITY SCORING (✅ SCALED FOR TIMEFRAMES)
# ============================================================================
# Used in compute_volatility_quality() to score ATR%
VOL_SCORING_THRESHOLDS = {
    "intraday": {
        "excellent": 4.0,   # ATR% <= 4% = excellent for scalping
        "good": 6.0,        # ATR% <= 6% = acceptable
        "fair": 8.0,        # ATR% <= 8% = risky but tradeable
        "poor": 10.0        # ATR% > 10% = panic/news event
    },
    "short_term": {         # DAILY CANDLES (Baseline)
        "excellent": 2.5,   # ATR% <= 2.5% = clean swing
        "good": 4.0,        # ATR% <= 4% = normal swing volatility
        "fair": 6.0,        # ATR% <= 6% = choppy
        "poor": 8.0         # ATR% > 8% = avoid
    },
    "long_term": {          # ✅ WEEKLY CANDLES (Scale ~2.2x)
        "excellent": 5.5,   # ATR% <= 5.5% weekly (2.5% daily equiv)
        "good": 9.0,        # ATR% <= 9% weekly (4% daily equiv)
        "fair": 13.0,       # ATR% <= 13% weekly (6% daily equiv) ← HAL is here
        "poor": 18.0        # ATR% > 18% weekly = breakdown
    },
    "multibagger": {        # ✅ MONTHLY CANDLES (Scale ~4.6x)
        "excellent": 11.5,  # ATR% <= 11.5% monthly (2.5% daily equiv)
        "good": 18.0,       # ATR% <= 18% monthly (4% daily equiv) ← HAL (13.76%) is here
        "fair": 27.0,       # ATR% <= 27% monthly (6% daily equiv)
        "poor": 36.0        # ATR% > 36% monthly = extreme
    }
}

# ============================================================================
# SCALING REFERENCE TABLE (For Documentation)
# ============================================================================
# Timeframe   | Days | √Days | Daily 2.5% → Scaled | Daily 5% → Scaled
# ------------|------|-------|---------------------|-------------------
# Intraday    | 1    | 1.0x  | 2.5%                | 5%
# Short-Term  | 1    | 1.0x  | 2.5%                | 5%
# Long-Term   | 5    | 2.24x | 5.6%                | 11.2%
# Multibagger | 21   | 4.58x | 11.5%               | 22.9%


# ============================================================================
# TREND STRENGTH THRESHOLDS (UNCHANGED)
# ============================================================================
TREND_THRESH = {
    "weak_floor": 20.0,
    "moderate_floor": 25.0,
    "strong_floor": 40.0,
    "di_spread_strong": 20.0,
}

RVOL_SURGE_THRESHOLD, RVOL_DROUGHT_THRESHOLD, VOLUME_CLIMAX_SPIKE = 3.0, 0.7, 2.0
ATR_SL_MAX_PERCENT, ATR_SL_MIN_PERCENT = 0.03, 0.01
STRATEGY_TIME_MULTIPLIERS = {'momentum': 0.7, 'day_trading': 0.5, 'swing': 1.0,
                             'trend_following': 1.2, 'position_trading': 1.5, 'value': 1.5,
                             'income': 2.0, 'unknown': 1.0}
TREND_WEIGHTS = {'primary': 0.50, 'secondary': 0.30, 'acceleration': 0.20}
RR_REGIME_ADJUSTMENTS = {'strong_trend': {'t1_mult': 2.0, 't2_mult': 4.0},
                         'normal_trend': {'t1_mult': 1.5, 't2_mult': 3.0},
                         'weak_trend': {'t1_mult': 1.2, 't2_mult': 2.5}}
HORIZON_T2_CAPS = {
    "intraday": 0.04,     # Max 4% expansion
    "short_term": 0.10,   # Max 10% expansion
    "long_term": 0.20,    # Max 20% expansion
    "multibagger": 1.00   # Uncapped (100%)
}

SIGNAL_ENGINE = {
    # ------------------------------------------------------------------
    # Risk / Reward rules (UPDATED)
    # ------------------------------------------------------------------
    "RR_RULES": {
        "min_rr_t1": 1.5,
        "min_rr_by_horizon": {  # ✅ ADD THIS
            "intraday": 1.1,
            "short_term": 1.3,
            "long_term": 1.5,
            "multibagger": 1.5
        },
        "default_multipliers": {  # ✅ ADD THIS
            "t1_mult": 1.5,
            "t2_mult": 3.0
        }
    },

    # ------------------------------------------------------------------
    # Volume Analysis (NEW)
    # ------------------------------------------------------------------\
    "VOLUME_ANALYSIS": {
        "intraday": {
            "rvol_surge_threshold": 3.0,
            "rvol_drought_threshold": 0.7
        },
        "short_term": {
            "rvol_surge_threshold": 2.5,  # Lower bar for daily
            "rvol_drought_threshold": 0.7
        },
        "long_term": {
            "rvol_surge_threshold": 2.0,  # Weekly volume is less volatile
            "rvol_drought_threshold": 0.8
        },
        "multibagger": {
            "rvol_surge_threshold": 1.8,  # Monthly volume even more stable
            "rvol_drought_threshold": 0.8
        }
    },

    # ------------------------------------------------------------------
    # Stop Loss Configuration (NEW)
    # ------------------------------------------------------------------
    "STOP_LOSS_MULTIPLIERS": {  # ✅ ADD THIS ENTIRE SECTION
        "volatility_based": {
            "high_quality": {"threshold": 8.0, "mult": 1.5},
            "low_quality": {"threshold": 4.0, "mult": 3.0},
            "default": 2.0
        }
    },
    
    "STOP_LOSS_VALIDATION": {  # ✅ ADD THIS
        "max_atr_multiplier": 5.0
    },

    # ------------------------------------------------------------------
    # Pattern Priority (NEW)
    # ------------------------------------------------------------------
    "PATTERN_PRIORITY": [  # ✅ ADD THIS ENTIRE SECTION
        {"pattern": "darvas_box", "setup_name": "PATTERN_DARVAS_BREAKOUT", "min_score": 85},
        {"pattern": "minervini_stage2", "setup_name": "PATTERN_VCP_BREAKOUT", "min_score": 85},
        {"pattern": "cup_handle", "setup_name": "PATTERN_CUP_BREAKOUT", "min_score": 80},
        {"pattern": "three_line_strike", "setup_name": "PATTERN_STRIKE_REVERSAL", "min_score": 80},
        {"pattern": "golden_cross", "setup_name": "PATTERN_GOLDEN_CROSS", "min_score": 75},
        {"pattern": "flag_pennant", "setup_name": "PATTERN_FLAG_BREAKOUT", "min_score": 80}
    ],

    # ------------------------------------------------------------------
    # Divergence Adjustments (NEW)
    # ------------------------------------------------------------------
    "DIVERGENCE_ADJUSTMENTS": {  # ✅ ADD THIS
        "opposing_divergence_sl_mult": 0.8
    },

    # ------------------------------------------------------------------
    # Consolidation Detection (NEW)
    # ------------------------------------------------------------------
    "CONSOLIDATION_DETECTION": {  # ✅ ADD THIS
        "bb_atr_ratio_threshold": 0.5
    },

    # ------------------------------------------------------------------
    # Proximity Rejection (NEW)
    # ------------------------------------------------------------------
    "PROXIMITY_REJECTION": {
        "intraday": {"resistance_mult": 1.003, "support_mult": 0.997},
        "short_term": {"resistance_mult": 1.005, "support_mult": 0.995},
        "long_term": {"resistance_mult": 1.010, "support_mult": 0.990},
        "multibagger": {"resistance_mult": 1.020, "support_mult": 0.980}
    },

    # ------------------------------------------------------------------
    # Dynamic confidence floors
    # ------------------------------------------------------------------
    "BASE_FLOORS": {
        "MOMENTUM_BREAKOUT": {
            "intraday": 50,
            "short_term": 55,
            "long_term": 60,
            "multibagger": 65
        },
        "TREND_PULLBACK": {
            "intraday": 48,
            "short_term": 53,
            "long_term": 58,
            "multibagger": 60
        }
    },

    "CONFIDENCE_HORIZON_DISCOUNT": {
        "intraday": 10
    },

    "CONFIDENCE_ADX_NORMALIZATION": {
        "adx_min": 10,
        "adx_range": 30,
        "adx_scale": 12,
        "min_floor": 35,
        "max_floor": 75
    },

    # ------------------------------------------------------------------
    # Horizon-specific target expansion caps
    # ------------------------------------------------------------------
    "HORIZON_T2_CAPS": {
        "intraday": 0.04,
        "short_term": 0.10,
        "long_term": 0.20,
        "multibagger": 1.00
    },

    # ------------------------------------------------------------------
    # Target / resistance calculation constants
    # ------------------------------------------------------------------
    "TARGET_BUFFERS": {
        "min_clearance_mult": 1.002,      # entry * 1.002
        "min_profit_pct": {
            "intraday": 0.3,
            "short_term": 0.5,
            "long_term": 1.0,
            "multibagger": 2.0
        },
        "t1_resistance_mult": 0.96,
        "t2_resistance_mult": 0.98,
        "next_resistance_mult": 1.03,
        "future_resistance_mult": 1.05,
        "t2_fallback_1": 1.15,
        "t2_fallback_2": 1.12,
        "t2_fallback_3": 1.20
    },

    "MAX_PRICE_MOVE": {
        "short_term": 1.10
    },

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------
    "POSITION_SIZING": {
        "base_risk": 0.01,
        "max_pct": {
            "intraday": 0.01,
            "default": 0.02
        },
        "setup_multipliers": {
            "DEEP_PULLBACK": 1.5,
            "VOLATILITY_SQUEEZE": 1.3,
            "MOMENTUM_BREAKOUT": 0.8
        },
        "volatility_multipliers": {
            "high_quality": 1.2,   # vol_qual > 7
            "low_quality": 0.9,    # vol_qual < 5
            "neutral": 1.0
        }
    },

    # ------------------------------------------------------------------
    # Volatility trade permission guards
    # ------------------------------------------------------------------
    "VOLATILITY_GUARDS": {
        "intraday": {
            "extreme_vol_buffer": 2.0,
            "min_quality_breakout": 2.5,
            "min_quality_normal": 4.0
        },
        "short_term": {
            "extreme_vol_buffer": 2.0,
            "min_quality_breakout": 3.0,
            "min_quality_normal": 4.0
        },
        "long_term": {
            "extreme_vol_buffer": 3.0,
            "min_quality_breakout": 4.0,
            "min_quality_normal": 5.0
        },
        "multibagger": {
            "extreme_vol_buffer": 4.0,
            "min_quality_breakout": 5.0,
            "min_quality_normal": 6.0
        }
    },

    # ------------------------------------------------------------------
    # Entry permission logic
    # ------------------------------------------------------------------
    "ENTRY_PERMISSION": {
        "confidence_discounts": {
            "trend": 15,
            "value_reversal": 25
        },
        "required_trend_strength": {
            "intraday": 2.0,
            "short_term": 3.5,
            "long_term": 5.0,
            "multibagger": 6.0
        }
    },

    # ------------------------------------------------------------------
    # Trend strength composite thresholds
    # ------------------------------------------------------------------
    "TREND_STRENGTH_THRESHOLDS": {
        "adx": {
            "strong": 25,
            "medium": 20,
            "weak": 15
        },
        "slope": {
            "intraday": {"strong": 15.0, "moderate": 5.0},
            "short_term": {"strong": 10.0, "moderate": 3.0},
            "long_term": {"strong": 5.0, "moderate": 2.0},
            "multibagger": {"strong": 30.0, "moderate": 10.0}  # Monthly slopes are huge!
        },
        "di_diff": {
            "strong": 15,
            "moderate": 10
        }
    },

    # ------------------------------------------------------------------
    # Momentum composite thresholds
    # ------------------------------------------------------------------
    "MOMENTUM_THRESHOLDS": {
        "rsi": {
            "strong": 70,
            "medium": 60,
            "neutral": 50,
            "weak": 40
        },
        "rsi_slope": {
            "intraday": {"positive": 0.10, "neutral": 0.0},
            "short_term": {"positive": 0.05, "neutral": 0.0},
            "long_term": {"positive": 0.03, "neutral": 0.0},
            "multibagger": {"positive": 0.02, "neutral": 0.0}
        },
        "macd_hist": {
            "positive": 0.5,
            "neutral": 0.0
        }
    },

    # ------------------------------------------------------------------
    # Volatility quality scoring thresholds
    # ------------------------------------------------------------------
    "VOLATILITY_QUALITY_THRESHOLDS": {
        "intraday": {
            "atr_pct": {"very_low": 1.5, "low": 3.0, "high": 6.0},
            "bb_width": {"very_tight": 0.01, "tight": 0.02, "wide": 0.04}
        },
        "short_term": {
            "atr_pct": {"very_low": 1.0, "low": 2.5, "high": 5.0},
            "bb_width": {"very_tight": 0.01, "tight": 0.02, "wide": 0.04}
        },
        "long_term": {
            "atr_pct": {"very_low": 2.0, "low": 5.5, "high": 12.0},
            "bb_width": {"very_tight": 0.02, "tight": 0.04, "wide": 0.08}
        },
        "multibagger": {
            "atr_pct": {"very_low": 3.0, "low": 8.0, "high": 25.0},
            "bb_width": {"very_tight": 0.03, "tight": 0.06, "wide": 0.12}
        }
    },

    # ------------------------------------------------------------------
    # Reversal / divergence
    # ------------------------------------------------------------------
    "REVERSAL_THRESHOLDS": {
        "rsi_oversold": 30
    },

    # ------------------------------------------------------------------
    # Spread adjustment
    # ------------------------------------------------------------------
    "SPREAD_ADJUSTMENT": {
        "market_cap_thresholds": {
            "large": 100000,
            "mid": 10000
        },
        "spread_values": {
            "large": 0.001,
            "mid": 0.002,
            "small": 0.005
        }
    },

    # ------------------------------------------------------------------
    # Pattern confluence bonuses
    # ------------------------------------------------------------------
    "PATTERN_BONUSES": {
        "additional_pattern": 5,
        "minervini_flag": 10,
        "squeeze_combo": 8,
        "golden_cross_cup": 7
    },

    # ------------------------------------------------------------------
    # Misc execution constants
    # ------------------------------------------------------------------
    "MISC": {
        "bear_market_dampener": 0.85,
        "min_trend_score": 0.35,
        "pullback_ma_pct": 0.05
    }
}

STRATEGY_ANALYZER = {

    "DEFAULT_INDICATOR_VALUES": {
        "rsi": 50,
        "rvol": 1.0,
        "atr_pct": 1.0
    },
    # ------------------------------------------------------------------
    # Swing Trading
    # ------------------------------------------------------------------
    "swing": {
        "fit_thresh": 50,
        "bb_proximity_mult": 1.02,          # Price <= BB Low * 1.02
        "bb_proximity_score": 35,
        "rsi_dip_threshold": 45,
        "rsi_dip_score": 25,
        "pattern_score": 40,                # Double Bottom bonus
        "squeeze_bonus": 10
    },

    # ------------------------------------------------------------------
    # Day Trading
    # ------------------------------------------------------------------
    "day_trading": {
        "fit_thresh": 50,
        "rvol_threshold": 1.5,
        "rvol_score": 25,
        "atr_pct_threshold": 1.5,
        "atr_pct_score": 15,
        "pattern_score": 40                 # 3-Line Strike bonus
    },

    # ------------------------------------------------------------------
    # Trend Following
    # ------------------------------------------------------------------
    "trend_following": {
        "fit_thresh": 50,
        "ma_alignment_score": 30,
        "adx_threshold": 25,
        "adx_score": 20,
        "ichimoku_score": 25,
        "golden_cross_score": 25
    },

    # ------------------------------------------------------------------
    # Momentum
    # ------------------------------------------------------------------
    "momentum": {
        "fit_thresh": 50,
        "rsi_threshold": 60,
        "rsi_score": 20,
        "darvas_score": 30,
        "flag_score": 30,
        "squeeze_score": 20
    },

    # ------------------------------------------------------------------
    # Minervini VCP
    # ------------------------------------------------------------------
    "minervini": {
        "fit_thresh": 50,
        "vcp_pattern_score": 50,
        "stage2_fallback_score": 20,
        "stage2_fail_penalty": -50,
        "relative_strength_score": 20,
        "pos_52w_high_threshold": 85,
        "pos_52w_high_score": 20,
        "pos_52w_low_threshold": 50,
        "pos_52w_low_penalty": -20
    },

    # ------------------------------------------------------------------
    # CANSLIM
    # ------------------------------------------------------------------
    "canslim": {
        "fit_thresh": 50,
        "quarterly_growth_threshold": 20,    # C: Current Earnings
        "quarterly_growth_score": 20,
        "annual_growth_threshold": 15,       # A: Annual Earnings
        "annual_growth_score": 15,
        "cup_pattern_score": 30,             # N: New Pattern
        "new_high_threshold": 90,            # N: 52W Position
        "new_high_score": 15,
        "volume_threshold": 1.2,             # S: Supply/Demand
        "volume_score": 10,
        "relative_strength_threshold": 5,    # L: Leader
        "relative_strength_score": 15
    },

    # ------------------------------------------------------------------
    # Value Investing
    # ------------------------------------------------------------------
    "value": {
        "fit_thresh": 50,
        "pe_threshold": 15,
        "pe_score": 35,
        "pb_threshold": 1.5,
        "pb_score": 25
    },

    # ------------------------------------------------------------------
    # Income / Dividend
    # ------------------------------------------------------------------
    "income": {
        "fit_thresh": 50,
        "dividend_yield_threshold": 3.0,
        "dividend_yield_score": 40
    },

    # ------------------------------------------------------------------
    # Position Trading
    # ------------------------------------------------------------------
    "position_trading": {
        "fit_thresh": 50,
        "eps_growth_threshold": 10,
        "eps_growth_score": 30,
        "uptrend_score": 40,
        "golden_cross_bonus": 20
    },

    # ------------------------------------------------------------------
    # Fallback Configs (Legacy)
    # ------------------------------------------------------------------
    "breakout": {
        "high_vol_threshold": 1.5,
        "low_vol_threshold": 0.8,
        "min_confidence": 60
    },
    
    "accumulation": {
        "consolidation_days": 5,
        "volume_ratio": 0.8,
        "min_confidence": 50
    },
    
    "pullback": {
        "max_retracement": 0.5,
        "trend_confirmation": True,
        "min_confidence": 55
    },

    # 1️⃣ DEFAULT FIT THRESHOLD ✅
    "DEFAULT_FIT_THRESH": 50,

    # 2️⃣ STRATEGY EXECUTION ORDER ✅
    "STRATEGY_EXECUTION_ORDER": [
        "swing",
        "day_trading",
        "trend_following",
        "position_trading",
        "momentum",
        "value",
        "income",
        "minervini",
        "canslim"
    ],

    # 3️⃣ PATTERN DETECTION ✅ (CORRECTED)
    "PATTERN_MIN_SCORE": 0,  # Pattern score must be > 0 (exclusive)

    # 4️⃣ SQUEEZE DETECTION ✅ (ENHANCED)
    "SQUEEZE_DETECTION": {
        "indicator_key": "ttm_squeeze",
        "active_token": "on",
        "check_method": "contains"
    },

    # 5️⃣ RELATIVE STRENGTH THRESHOLDS ✅ (CORRECTED)
    "RELATIVE_STRENGTH_THRESHOLDS": {
        "minervini": 0,      # Must be > 0
        "canslim": 5,        # Must be > 5
        "operator": ">"
    },

    # 6️⃣ STAGE 2 TEMPLATE ✅
    "STAGE2_TEMPLATE": {
        "enabled": True,
        "ma_keys": ["ma_mid", "ma_slow"],  # Price > MA50 > MA200
        "price_position": "above_all"
    },

    # 7️⃣ TREND STRUCTURE (OPTIONAL) 🟡
    "TREND_STRUCTURE": {
        "bullish_alignment": {
            "enabled": True,
            "order": ["price", "ma_fast", "ma_mid", "ma_slow"],
            "operator": ">"
        }
    }
}

TRADE_ENHANCER = {

    # ---------------------------------------------------------
    # RR regime adjustments (fallbacks if config missing)
    # ---------------------------------------------------------
    "RR_REGIME_DEFAULTS": {
        "strong_trend": {
            "adx_min": 40,
            "t1_mult": 2.0,
            "t2_mult": 4.0
        },
        "normal_trend": {
            "adx_min": 20,
            "adx_max": 40,
            "t1_mult": 1.5,
            "t2_mult": 3.0
        },
        "weak_trend": {
            "adx_max": 20,
            "t1_mult": 1.2,
            "t2_mult": 2.5
        }
    },

    # ---------------------------------------------------------
    # Pattern expiration – applicable patterns (hardcoded list)
    # ---------------------------------------------------------
    "EXPIRING_PATTERNS": [
        "flag_pennant",
        "three_line_strike"
    ],

    # ---------------------------------------------------------
    # Pattern entry validation defaults
    # ---------------------------------------------------------
    "PATTERN_ENTRY_DEFAULTS": {

        "cup_handle": {
            "rim_clearance": 0.99,
            "rvol_min": 1.2,
            "rvol_bonus_threshold": 2.0,
            "volume_surge_bonus": 10
        },

        "darvas_box": {
            "box_clearance": 1.005
        },

        "minervini_stage2": {
            "contraction_max": 1.5,
            "contraction_warning_penalty": -5
        },

        "bollinger_squeeze": {
            "rsi_min": 50
        },

        "flag_pennant": {
            "pole_length_min": 5,
            "pole_short_penalty": -5
        },

        "three_line_strike": {
            "strike_candle_body_min": 0.6
        }
    },

    # ---------------------------------------------------------
    # Divergence detection (local severity logic) DUPLICATE of constants.py's RSI_SLOPE_THRESH
    # ---------------------------------------------------------
    "DIVERGENCE_THRESHOLDS": {
        "rsi_slope": {
            "severe": -0.08,
            "moderate": -0.03,
            "minor": 0.0
        }
    },

    # ---------------------------------------------------------
    # Pattern selection thresholds
    # ---------------------------------------------------------
    "PATTERN_SELECTION": {
        "min_score": 60
    },

    # ---------------------------------------------------------
    # Supported pattern scan order
    # ---------------------------------------------------------
    "PATTERN_KEYS": [
        "darvas_box",
        "cup_handle",
        "bollinger_squeeze",
        "flag_pennant",
        "minervini_stage2",
        "three_line_strike",
        "ichimoku_signals",
        "golden_cross",
        "double_top_bottom"
    ],

    # ---------------------------------------------------------
    # Pattern priority (implicit via score sort, but thresholded)
    # ---------------------------------------------------------
    "PATTERN_SCORE_THRESHOLDS": {
        "high_quality": 80
    },

    # ---------------------------------------------------------
    # Pattern physics fallbacks
    # ---------------------------------------------------------
    "PATTERN_PHYSICS_DEFAULTS": {
        "target_ratio": 1.0,
        "t2_multiplier": 2
    },

    # ---------------------------------------------------------
    # Stop-loss defaults (when pattern SL missing)
    # ---------------------------------------------------------
    "STOP_LOSS_DEFAULTS": {
        "atr_fallback_mult": 2.0
    },

    # ---------------------------------------------------------
    # Directional sanity rules
    # ---------------------------------------------------------
    "STOP_LOSS_SANITY": {
        "long": {
            "must_be_below_entry": True
        },
        "short": {
            "must_be_above_entry": True
        }
    },

    # ---------------------------------------------------------
    # Execution quality scoring
    # ---------------------------------------------------------
    "EXECUTION_QUALITY_SCORES": {
        "has_geometry": 30,
        "has_stop_loss": 20,
        "has_target": 25,
        "high_pattern_score": 25
    },

    # ---------------------------------------------------------
    # Pattern role classification
    # ---------------------------------------------------------
    "PATTERN_ROLES": {
        "momentum_confirmation": [
            "bollinger_squeeze",
            "three_line_strike"
        ],
        "trend_continuation": [
            "minervini_stage2"
        ],
        "regime_confirmation": [
            "ichimoku_signals",
            "golden_cross"
        ]
    },

    # ---------------------------------------------------------
    # Confidence adjustments
    # ---------------------------------------------------------
    "CONFIDENCE_ADJUSTMENTS": {
        "pattern_expired_penalty": -20
    },

    # ---------------------------------------------------------
    # ATR extraction fallback order (implicit priority)
    # ---------------------------------------------------------
    "ATR_FALLBACK_KEYS": [
        "atr_dynamic",
        "atr_14",
        "atr",
        "atr14"
    ],
    
    # ---------------------------------------------------------
    # Pattern reference levels for invalidation checks (NEW)
    # ---------------------------------------------------------
    "PATTERN_REFERENCE_LEVELS": {  # ✅ ADD THIS
        "darvas_box": "box_low",
        "cup_handle": "handle_low",
        "flag_pennant": "flag_low",
        "minervini_stage2": "pivot_point",
        "bollinger_squeeze": "bb_low",
        "three_line_strike": "entry",
        "ichimoku_signals": "cloud_bottom",
        "double_top_bottom": "neckline"
    },

    # ---------------------------------------------------------
    # MA key fallback paths (for legacy support) (NEW)
    # ---------------------------------------------------------
    "MA_KEY_FALLBACKS": {
        "fast": ["ma_fast", "ema_20", "mafast", "ema20"],
        "mid": ["ma_mid", "ema_50", "mamid", "ema50"],
        "slow": ["ma_slow", "ema_200", "maslow", "ema200"]
    },

    # ---------------------------------------------------------
    # Confidence score bounds (NEW)
    # ---------------------------------------------------------
    "CONFIDENCE_BOUNDS": {  # ✅ ADD THIS
        "min": 0,
        "max": 100,
        "default": 50
    },

    "DIVERGENCE_CONFIDENCE_PENALTIES": {
        "severe": 1.0,      # From MASTER_CONFIG (no entry allowed)
        "moderate": 0.85,   # 15% penalty
        "minor": 0.95       # 5% penalty
    },

    # ---------------------------------------------------------
    # Pattern-specific stop loss adjustments (NEW)
    # ---------------------------------------------------------
    "PATTERN_STOP_LOSS_MULTIPLIERS": {  # ✅ ADD THIS
        "darvas_box": 0.995  # 0.5% below box_low
    },
}

PATTERNS = {
    
    # ==================================================================
    # BOLLINGER SQUEEZE
    # ==================================================================
    "bollinger_squeeze": {
        "squeeze_threshold": 0.10,              # BB Width < 10% = squeeze
        "breakout_confirmation": 0.02,          # 2% above band
        "squeeze_score": 75,
        "breakout_score": 95,
        "estimated_age_candles": 7,
        "squeeze_quality": 8.0,
        "breakout_quality": 10.0
    },

    # ==================================================================
    # CUP & HANDLE
    # ==================================================================
    "cup_handle": {
        "min_cup_len": 20,
        "max_cup_depth": 0.50,
        "min_cup_depth": 0.10,
        "require_volume": False,
        "handle_len": 5,
        "window_size": 60,
        "search_split_ratio": 0.5,
        "rim_alignment_tolerance": 0.15,
        "handle_upper_half_only": True,
        "forming_threshold": 0.90,
        "volume_dry_threshold": 0.9,
        "volume_bonus_quality": 2.0,
        "base_quality": 6.0,
        "breakout_bonus": 2.0,
        "min_history_buffer": 5
    },

    # ==================================================================
    # DOUBLE TOP / BOTTOM
    # ==================================================================
    "double_top_bottom": {
        "peak_window": 5,
        "min_history": 60,
        "window_size": 60,
        "price_level_tolerance": 0.03,
        "pattern_score": 80,
        "pattern_quality": 8.5
    },

    # ==================================================================
    # GOLDEN / DEATH CROSS
    # ==================================================================
    "golden_cross": {
        "min_history": 200,
        "golden_cross_score": 90,
        "death_cross_score": 90,
        "pattern_quality": 9.0
    },

    # ==================================================================
    # MINERVINI VCP / STAGE 2
    # ==================================================================
    "minervini_vcp": {
        "min_history": 50,
        "max_atr_pct": 3.5,
        "recent_window": 5,
        "prev_window": 10,
        "contraction_threshold": 0.7,
        "tightness_threshold": 0.05,
        "vol_recent_window": 5,
        "vol_avg_window": 50,
        "formation_estimate_offset": 15,
        "base_quality": 7.0,
        "vol_dry_bonus": 2.0
    },

    # ==================================================================
    # DARVAS BOX
    # ==================================================================
    "darvas_box": {
        "lookback": 50,
        "box_length": 5,
        "box_lookback_multiplier": 2,
        "consolidation_tolerance_high": 1.01,
        "consolidation_tolerance_low": 0.99,
        "volume_threshold": 1.5,
        "base_quality": 5.0,
        "volume_bonus": 3.0,
        "trend_bonus": 2.0
    },

    # ==================================================================
    # THREE LINE STRIKE
    # ==================================================================
    "three_line_strike": {
        "min_history": 5,
        "pattern_score": 90,
        "pattern_quality": 9.0
    },

    # ==================================================================
    # FLAG / PENNANT
    # ==================================================================
    "flag_pennant": {
        "pole_days": 15,
        "flag_days": 5,
        "strong_pole_threshold": 0.05,
        "flag_drift_min": -0.03,
        "flag_drift_max": 0.01,
        "base_quality": 6.0,
        "volume_dry_bonus": 2.0,
        "breakout_threshold": 1.01,
        "breakout_bonus": 2.0
    },

    # ==================================================================
    # ICHIMOKU SIGNALS
    # ==================================================================
    "ichimoku_signals": {
        "tenkan_window": 9,
        "kijun_window": 26,
        "min_history_buffer": 2,
        
        "quality_scores": {
            "strong_bull_cross": 9.0,
            "weak_bull_cross": 5.0,
            "neutral_bull_cross": 7.0,
            "strong_bear_cross": 9.0,
            "weak_bear_cross": 5.0,
            "neutral_bear_cross": 7.0,
            "price_above_cloud": 5.0
        },
        
        "cross_bonus": 10,
        "fresh_cross_age": 1,
        "established_signal_age": 5
    },

    # ==================================================================
    # BASE PATTERN (Shared defaults)
    # ==================================================================
    "base": {
        "horizons_supported": ["intraday", "short_term", "swing", "long_term"],
        "debug": False,
        "coerce_numeric": True,
        "numeric_cols": ["Open", "High", "Low", "Close", "Volume"]
    },

    # ==================================================================
    # PATTERN DETECTION GLOBAL SETTINGS
    # ==================================================================
    "global": {
        "min_score_threshold": 60,
        "score_normalization": {
            "min": 0.0,
            "max": 100.0
        },
        "age_tracking_enabled": True,
        "require_formation_timestamp": True
    }
}

TIME_ESTIMATOR = {
    
    # ==================================================================
    # Velocity Factors (Trend Regime Adjustments)
    # ==================================================================
    "velocity_factors": {
        "strong_trend": {
            "min_strength": 7.0,
            "factor": 1.2
        },
        "normal_trend": {
            "min_strength": 5.0,
            "factor": 1.0
        },
        "weak_trend": {
            "max_strength": 5.0,
            "factor": 0.8
        }
    },
    
    # ==================================================================
    # Base Calculation Parameters
    # ==================================================================
    "base_friction": 0.8,                   # Global drag factor
    "default_trend_strength": 5.0,
    "default_velocity_factor": 1.0,
    "min_bars_per_atr": 2,
    "min_price_ratio": 0.001,               # 0.1% of price
    "min_atr_absolute": 0.01,
    "min_bars_result": 1,                   # Ensure at least 1 bar
    
    # ==================================================================
    # 🔴 HORIZON-SPECIFIC: Candles per Unit Time
    # ==================================================================
    "candles_per_unit": {
        "intraday": 4,                      # 4 x 15m candles = 1 hour
        "short_term": 1,                    # 1 daily candle = 1 day
        "long_term": 0.2,                   # 1 weekly candle = 5 days
        "multibagger": 0.05                 # 1 monthly candle = 20 days
    },
    
    # ==================================================================
    # Time Formatting Thresholds
    # ==================================================================
    "format_thresholds": {
        "hours_to_days": 1,                 # < 1 day = show hours
        "days_to_weeks": 30,                # < 30 days = show days
        "weeks_to_years": 365               # < 365 days = show weeks
    }
}


"""
Here is the definitive list of what you can DELETE and what you must KEEP to ensure a smooth transition.

✂️ DELETE THESE (Now Redundant)
These are fully handled by MASTER_CONFIG.

Horizon & Profile Logic:

❌ HORIZON_PROFILE_MAP (Replaced by MASTER_CONFIG["horizons"])

❌ HORIZON_FETCH_CONFIG (Replaced by MASTER_CONFIG["global"]["system"]["fetch"])

❌ ADX_HORIZON_CONFIG (Replaced by MASTER_CONFIG["horizons"][h]["indicators"]["adx_period"])

❌ STOCH_HORIZON_CONFIG (Replaced by MASTER_CONFIG["horizons"][h]["indicators"])

❌ ATR_HORIZON_CONFIG (Replaced by MASTER_CONFIG["horizons"][h]["indicators"]["atr_period"])

Volatility & Bands:

❌ VOL_BANDS (Replaced by MASTER_CONFIG["horizons"][h]["gates"]["volatility_bands_atr_pct"])

❌ VOL_BANDS_HORIZON_MULTIPLIERS (Implicit in MASTER_CONFIG's horizon-specific values)

❌ VOL_SCORING_THRESHOLDS (Replaced by MASTER_CONFIG["horizons"][h]["volatility"]["scoring_thresholds"])

❌ VOL_QUAL_MINS (Replaced by MASTER_CONFIG["global"]["boosts"]["volatility"])

Thresholds & Limits:

❌ RSI_SLOPE_THRESH (Replaced by MASTER_CONFIG["horizons"][h]["momentum_thresholds"])

❌ MACD_MOMENTUM_THRESH (Replaced by MASTER_CONFIG["horizons"][h]["momentum_thresholds"])

❌ TREND_THRESH (Replaced by MASTER_CONFIG["global"]["calculation_engine"]["composite_weights"])

❌ ATR_MULTIPLIERS (Replaced by MASTER_CONFIG["horizons"][h]["execution"]["stop_loss_atr_mult"])

❌ STOCH_FAST, STOCH_SLOW, STOCH_THRESHOLDS (Replaced by MASTER_CONFIG["global"]["calculation_engine"])

Weights (Merged into Global):

❌ FUNDAMENTAL_WEIGHTS (Replaced by MASTER_CONFIG["global"]["fundamental_weights"])

❌ QUALITY_WEIGHTS, GROWTH_WEIGHTS, VALUE_WEIGHTS, MOMENTUM_WEIGHTS (Replaced by MASTER_CONFIG["global"]["fundamental_weights"] and MASTER_CONFIG["global"]["calculation_engine"])

🛡️ KEEP THESE (Infrastructure & Reference)
These are NOT in MASTER_CONFIG and are required for the app to run.

Environment & App Settings:

✅ ENABLE_CACHE, ENABLE_CACHE_WARMER, ENABLE_JSON_ENRICHMENT

✅ ENABLE_VOLATILITY_QUALITY (Used as a feature flag)

Mappings & Labels (UI/Data Fetching):

✅ INDEX_TICKERS (Maps "nifty50" to "^NSEI")

✅ TECHNICAL_METRIC_MAP (Human-readable labels for UI)

✅ FUNDAMENTAL_ALIAS_MAP (Human-readable labels for UI)

✅ FUNDAMENTAL_FIELD_CANDIDATES (Critical for yfinance data parsing)

✅ SECTOR_PE_AVG (Reference data for valuation)

✅ flowchart_mapping (Used for the flowchart UI)

Legacy Compatibility (Optional but Recommended):

⚠️ TECHNICAL_WEIGHTS: Your indicators.py uses this in compute_technical_score. Keep this unless you refactor indicators.py to use MASTER_CONFIG scoring.

⚠️ CORE_TECHNICAL_SETUP_METRICS: Used in indicators.py to force specific calculations. Keep.
"""