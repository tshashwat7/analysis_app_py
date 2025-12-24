# config/master_config.py (CLEANED VERSION - PART 1/5)
"""
Master Configuration - Global Section
Smart inheritance: Horizons inherit from global, only override what's different
"""

MASTER_CONFIG = {
    # ============================================================================
    # GLOBAL CONSTANTS (Universal Physics & Logic)
    # ============================================================================
    "global": {
        
        # ========================================================================
        # FUNDAMENTAL WEIGHTS (Universal - No Horizon Overrides)
        # ========================================================================
        "fundamental_weights": {
            "value": {
                "pe_ratio": {"weight": 0.05, "direction": "invert", "ideal_range": [10, 20], "penalty_threshold": 50},
                "pb_ratio": {"weight": 0.04, "direction": "invert", "ideal_range": [1, 3], "penalty_threshold": 5},
                "peg_ratio": {"weight": 0.03, "direction": "invert", "ideal_range": [0.5, 1.5], "penalty_threshold": 3},
                "ps_ratio": {"weight": 0.02, "direction": "invert", "ideal_range": [1, 3]},
                "fcf_yield": {"weight": 0.05, "direction": "normal", "min_threshold": 3.0},
                "dividend_yield": {"weight": 0.03, "direction": "normal", "min_threshold": 2.0},
                "pe_vs_sector": {"weight": 0.03, "direction": "invert"}
            },
            "growth": {
                "eps_growth_5y": {"weight": 0.06, "direction": "normal", "min_threshold": 15, "ideal_range": [20, 40]},
                "eps_growth_3y": {"weight": 0.05, "direction": "normal", "min_threshold": 15},
                "revenue_growth_5y": {"weight": 0.05, "direction": "normal", "min_threshold": 10},
                "profit_growth_3y": {"weight": 0.04, "direction": "normal", "min_threshold": 15},
                "fcf_growth_3y": {"weight": 0.05, "direction": "normal", "min_threshold": 10},
                "market_cap_cagr": {"weight": 0.04, "direction": "normal", "min_threshold": 15},
                "quarterly_growth": {"weight": 0.03, "direction": "normal", "min_threshold": 20}
            },
            "quality": {
                "roe": {"weight": 0.10, "direction": "normal", "min_threshold": 15, "ideal_range": [20, 40]},
                "roce": {"weight": 0.07, "direction": "normal", "min_threshold": 15, "ideal_range": [20, 35]},
                "roic": {"weight": 0.08, "direction": "normal", "min_threshold": 12, "ideal_range": [18, 35]},
                "net_profit_margin": {"weight": 0.04, "direction": "normal", "min_threshold": 10},
                "operating_margin": {"weight": 0.03, "direction": "normal", "min_threshold": 15},
                "ebitda_margin": {"weight": 0.03, "direction": "normal"},
                "fcf_margin": {"weight": 0.04, "direction": "normal", "min_threshold": 8},
                "de_ratio": {"weight": 0.05, "direction": "invert", "max_threshold": 1.0, "ideal_range": [0, 0.5]},
                "interest_coverage": {"weight": 0.05, "direction": "normal", "min_threshold": 3.0, "ideal_range": [5, 20]},
                "current_ratio": {"weight": 0.03, "direction": "normal", "min_threshold": 1.5},
                "ocf_vs_profit": {"weight": 0.02, "direction": "normal", "min_threshold": 0.8},
                "piotroski_f": {"weight": 0.07, "direction": "normal", "min_threshold": 7},
                "asset_turnover": {"weight": 0.04, "direction": "normal", "min_threshold": 0.5},
                "r_d_intensity": {"weight": 0.04, "direction": "normal"},
                "earnings_stability": {"weight": 0.05, "direction": "normal", "min_threshold": 7.0}
            },
            "momentum": {
                "momentum_strength": {"weight": 0.30, "direction": "normal", "min_threshold": 6.0},
                "trend_strength": {"weight": 0.40, "direction": "normal", "min_threshold": 5.0},
                "volatility_quality": {"weight": 0.10, "direction": "normal", "min_threshold": 4.0},
                "52w_position": {"weight": 0.01, "direction": "normal", "ideal_range": [80, 95]},
                "rel_strength_nifty": {"weight": 0.04, "direction": "normal", "min_threshold": 0},
                "promoter_holding": {"weight": 0.015, "direction": "normal", "ideal_range": [50, 75]},
                "institutional_ownership": {"weight": 0.015, "direction": "normal", "ideal_range": [20, 50]},
                "beta": {"weight": 0.01, "direction": "normal", "ideal_range": [0.8, 1.2]},
                "dividend_payout": {"weight": 0.03, "direction": "normal"},
                "yield_vs_avg": {"weight": 0.02, "direction": "normal"}
            }
        },
        
        # ========================================================================
        # TECHNICAL WEIGHTS (Base Weights - Horizons Apply Multipliers)
        # ========================================================================
        "technical_weights": {
            # Momentum Indicators
            "rsi": {"weight": 1.0, "category": "momentum", "speed": "fast"},
            "rsi_slope": {"weight": 0.8, "category": "momentum", "speed": "fast"},
            "macd_cross": {"weight": 1.0, "category": "momentum", "speed": "medium"},
            "macd_hist_z": {"weight": 0.8, "category": "momentum", "speed": "medium"},
            "macd_histogram": {"weight": 0.6, "category": "momentum", "speed": "medium"},
            "stoch_k": {"weight": 0.6, "category": "momentum", "speed": "fast"},
            "stoch_cross": {"weight": 0.8, "category": "momentum", "speed": "fast"},
            "mfi": {"weight": 0.7, "category": "momentum", "speed": "fast"},
            "cci": {"weight": 0.6, "category": "momentum", "speed": "fast"},
            
            # Trend Indicators
            "adx": {"weight": 1.0, "category": "trend", "speed": "slow"},
            "adx_signal": {"weight": 0.6, "category": "trend", "speed": "slow"},
            "ma_cross_signal": {"weight": 0.8, "category": "trend", "speed": "medium"},
            "ma_trend_signal": {"weight": 1.0, "category": "trend", "speed": "slow"},
            "ma_fast_slope": {"weight": 0.8, "category": "trend", "speed": "medium"},
            "ema_20_50_cross": {"weight": 0.8, "category": "trend", "speed": "medium"},
            "price_vs_primary_trend_pct": {"weight": 1.0, "category": "trend", "speed": "slow"},
            "price_vs_200dma_pct": {"weight": 1.0, "category": "trend", "speed": "slow"},
            "dma_200_slope": {"weight": 0.8, "category": "trend", "speed": "slow"},
            "supertrend_signal": {"weight": 1.0, "category": "trend", "speed": "medium"},
            "psar_trend": {"weight": 0.8, "category": "trend", "speed": "medium"},
            "ichi_cloud": {"weight": 1.0, "category": "trend", "speed": "slow"},
            
            # Volatility Indicators
            "atr_14": {"weight": 0.8, "category": "volatility", "speed": "medium"},
            "atr_pct": {"weight": 0.6, "category": "volatility", "speed": "medium"},
            "bb_width": {"weight": 0.3, "category": "volatility", "speed": "medium"},
            "bb_percent_b": {"weight": 0.4, "category": "volatility", "speed": "fast"},
            "bb_low": {"weight": 0.4, "category": "volatility", "speed": "fast"},
            "ttm_squeeze": {"weight": 1.0, "category": "volatility", "speed": "medium"},
            "true_range": {"weight": 0.5, "category": "volatility", "speed": "fast"},
            "hv_10": {"weight": 0.6, "category": "volatility", "speed": "fast"},
            "hv_20": {"weight": 0.6, "category": "volatility", "speed": "medium"},
            
            # Volume Indicators
            "rvol": {"weight": 0.6, "category": "volume", "speed": "fast"},
            "vol_spike_ratio": {"weight": 0.5, "category": "volume", "speed": "fast"},
            "vol_spike_signal": {"weight": 1.0, "category": "volume", "speed": "fast"},
            "vol_trend": {"weight": 0.6, "category": "volume", "speed": "medium"},
            "obv_div": {"weight": 0.6, "category": "volume", "speed": "medium"},
            "vpt": {"weight": 0.5, "category": "volume", "speed": "medium"},
            "cmf_signal": {"weight": 0.6, "category": "volume", "speed": "medium"},
            
            # Support/Resistance
            "entry_confirm": {"weight": 0.5, "category": "level", "speed": "static"},
            "pivot_point": {"weight": 0.4, "category": "level", "speed": "static"},
            "donchian_signal": {"weight": 0.8, "category": "level", "speed": "medium"},
            
            # Pattern Signals
            "wick_rejection": {"weight": 0.7, "category": "pattern", "speed": "fast"},
            
            # Composite Scores
            "trend_strength": {"weight": 1.5, "category": "composite", "speed": "medium"},
            "momentum_strength": {"weight": 1.5, "category": "composite", "speed": "medium"},
            "volatility_quality": {"weight": 1.0, "category": "composite", "speed": "medium"},
            
            # Relative Metrics
            "rel_strength_nifty": {"weight": 0.6, "category": "relative", "speed": "slow"},
            "nifty_trend_score": {"weight": 0.5, "category": "relative", "speed": "slow"},
            
            # Price Action
            "vwap_bias": {"weight": 0.8, "category": "price_action", "speed": "fast"},
            "price_action": {"weight": 0.7, "category": "price_action", "speed": "fast"},
            "gap_percent": {"weight": 0.5, "category": "price_action", "speed": "fast"},
            "reg_slope": {"weight": 0.8, "category": "price_action", "speed": "medium"}
        },
        
        # ========================================================================
        # PATTERN PHYSICS (Universal Pattern Behavior)
        # ========================================================================
        "pattern_physics": {
            "cup_handle": {
                "target_ratio": 0.618,
                "duration_multiplier": 1.2,
                "max_stop_pct": 8.0,
                "min_cup_len": 20,
                "max_cup_depth": 0.50,
                "handle_len": 5,
                "require_volume": False,
                "horizons_supported": ["short_term", "long_term"]
            },
            "darvas_box": {
                "target_ratio": 1.0,
                "duration_multiplier": 1.3,
                "max_stop_pct": 5.0,
                "lookback": 50,
                "box_length": 5,
                "horizons_supported": ["intraday", "short_term"]
            },
            "flag_pennant": {
                "target_ratio": 0.5,
                "duration_multiplier": 0.8,
                "max_stop_pct": 6.0,
                "horizons_supported": ["intraday", "short_term"]
            },
            "minervini_vcp": {
                "target_ratio": 1.0,
                "duration_multiplier": 1.8,
                "max_stop_pct": 7.0,
                "min_contraction_pct": 1.5,
                "horizons_supported": ["short_term", "long_term"]
            },
            "volatility_squeeze": {
                "target_ratio": 1.0,
                "duration_multiplier": 0.5,
                "max_stop_pct": 4.0,
                "horizons_supported": ["intraday", "short_term"]
            },
            "golden_cross": {
                "target_ratio": None,
                "duration_multiplier": 2.0,
                "max_stop_pct": None,
                "horizons_supported": ["short_term", "long_term", "multibagger"]
            },
            "scoring_thresholds": {
                "high_quality": 60,
                "medium_quality": 40,
                "poor_quality": 20
            },
            "default": {
                "target_ratio": 1.0,
                "duration_multiplier": 1.0,
                "max_stop_pct": 10.0
            }
        },
        
        # ========================================================================
        # PATTERN ENTRY RULES (Global - Horizon-Specific in Nested Structure)
        # ========================================================================
        "pattern_entry_rules": {
            "bollinger_squeeze": {
                "horizons": {
                    "intraday": {
                        "rsi_min": 50,
                        "macd_hist_min": 0,
                        "squeeze_duration_min": 5,
                        "rvol_on_breakout": 1.5
                    },
                    "short_term": {
                        "rsi_min": 50,
                        "macd_hist_min": 0,
                        "squeeze_duration_min": 3,
                        "rvol_on_breakout": 1.2
                    },
                    "long_term": {
                        "rsi_min": 45,
                        "macd_hist_min": -0.2,
                        "squeeze_duration_min": 4,
                        "rvol_on_breakout": 1.0
                    }
                }
            },
            "darvas_box": {
                "horizons": {
                    "intraday": {
                        "box_clearance": 1.002,
                        "volume_surge_required": 1.5,
                        "max_box_age_candles": 50
                    },
                    "short_term": {
                        "box_clearance": 1.005,
                        "volume_surge_required": 1.3,
                        "max_box_age_candles": 30
                    }
                }
            },
            "cup_handle": {
                "horizons": {
                    "short_term": {
                        "rim_clearance": 0.995,
                        "rvol_min": 1.2,
                        "rvol_bonus_threshold": 2.0
                    },
                    "long_term": {
                        "rim_clearance": 0.99,
                        "rvol_min": 1.1,
                        "rvol_bonus_threshold": 1.8
                    }
                }
            },
            "minervini_stage2": {
                "horizons": {
                    "short_term": {
                        "contraction_max": 1.5,
                        "pivot_clearance": 1.01,
                        "rs_rating_min": 80
                    },
                    "long_term": {
                        "contraction_max": 2.0,
                        "pivot_clearance": 1.005,
                        "rs_rating_min": 70
                    }
                }
            },
            "flag_pennant": {
                "horizons": {
                    "intraday": {
                        "pole_length_min": 8,
                        "flag_tightness": 0.03,
                        "breakout_clearance": 1.005
                    },
                    "short_term": {
                        "pole_length_min": 5,
                        "flag_tightness": 0.05,
                        "breakout_clearance": 1.01
                    }
                }
            },
            "three_line_strike": {
                "horizons": {
                    "intraday": {
                        "strike_candle_body_min": 0.6,
                        "rvol_min": 1.3
                    },
                    "short_term": {
                        "strike_candle_body_min": 0.7,
                        "rvol_min": 1.2
                    }
                }
            },
            "ichimoku_signals": {
                "horizons": {
                    "short_term": {
                        "cloud_thickness_min": 0.01,
                        "tenkan_kijun_spread_min": 0.005
                    },
                    "long_term": {
                        "cloud_thickness_min": 0.02,
                        "tenkan_kijun_spread_min": 0.01
                    }
                }
            },
            "golden_cross": {
                "horizons": {
                    "short_term": {
                        "cross_clearance": 0.002,
                        "volume_confirmation": 1.1
                    },
                    "long_term": {
                        "cross_clearance": 0.005,
                        "volume_confirmation": 1.0
                    }
                }
            },
            "double_top_bottom": {
                "horizons": {
                    "short_term": {
                        "peak_similarity_tolerance": 0.02,
                        "neckline_clearance": 1.01,
                        "volume_decline_on_second_peak": True
                    },
                    "long_term": {
                        "peak_similarity_tolerance": 0.03,
                        "neckline_clearance": 1.005,
                        "volume_decline_on_second_peak": False
                    }
                }
            }
        },

        # ========================================================================
        # PATTERN INVALIDATION (Global - Horizon-Specific Rules)
        # ========================================================================
        "pattern_invalidation": {
            "bollinger_squeeze": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < bb_low",
                        "duration_candles": 2,
                        "or_condition": "bb_width > 8.0"
                    },
                    "short_term": {
                        "condition": "price < bb_low * 0.99",
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
            "darvas_box": {
                "breakdown_threshold": {
                    "intraday": {"condition": "price < box_low * 0.998", "duration_candles": 1},
                    "short_term": {"condition": "price < box_low * 0.995", "duration_candles": 1},
                    "long_term": {"condition": "price < box_low * 0.99", "duration_candles": 2}
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_IMMEDIATELY",
                    "long_term": "EXIT_ON_CLOSE"
                },
                "notes": "Darvas box breakdown is binary - no 'monitor' mode"
            },
            "flag_pennant": {
                "breakdown_threshold": {
                    "intraday": {"condition": "price < flag_low * 0.998", "duration_candles": 1},
                    "short_term": {"condition": "price < flag_low * 0.995", "duration_candles": 1},
                    "long_term": {"condition": "price < flag_low * 0.99", "duration_candles": 1}
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_IMMEDIATELY",
                    "long_term": "EXIT_ON_CLOSE"
                },
                "expiration": {
                    "max_duration_candles": {
                        "intraday": 20,
                        "short_term": 10,
                        "long_term": 8
                    },
                    "action_on_expire": "DOWNGRADE_TO_CONSOLIDATION"
                }
            },
            "minervini_stage2": {
                "breakdown_threshold": {
                    "intraday": {"condition": "price < pivot * 0.98", "duration_candles": 2},
                    "short_term": {"condition": "price < pivot * 0.95", "duration_candles": 2},
                    "long_term": {"condition": "price < pivot * 0.92", "duration_candles": 3}
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
            "cup_handle": {
                "breakdown_threshold": {
                    "intraday": {"condition": "price < handle_low * 0.99", "duration_candles": 2},
                    "short_term": {"condition": "price < handle_low * 0.97", "duration_candles": 2},
                    "long_term": {"condition": "price < handle_low * 0.95", "duration_candles": 3}
                },
                "action": {
                    "intraday": "EXIT_ON_CLOSE",
                    "short_term": "TIGHTEN_STOP",
                    "long_term": "MONITOR"
                },
                "handle_failure": {
                    "max_handle_depth": 0.15,
                    "action": "INVALIDATE_PATTERN"
                }
            },
            "three_line_strike": {
                "expiration": {
                    "max_hold_candles": {
                        "intraday": 10,
                        "short_term": 8,
                        "long_term": 6
                    }
                }
            },
            "ichimoku_signals": {
                "breakdown_threshold": {
                    "intraday": {"condition": "price < cloud_bottom * 0.998", "duration_candles": 2},
                    "short_term": {"condition": "price < cloud_bottom * 0.995", "duration_candles": 2},
                    "long_term": {"condition": "price < cloud_bottom", "duration_candles": 3}
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
            "golden_cross": {
                "breakdown_threshold": {
                    "intraday": {"condition": "ema_50 < ema_200", "duration_candles": 3},
                    "short_term": {"condition": "ema_50 < ema_200", "duration_candles": 3},
                    "long_term": {"condition": "ema_50 < ema_200", "duration_candles": 4}
                },
                "action": {
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_ON_CLOSE",
                    "long_term": "EXIT_ON_CLOSE"
                }
            },
            "double_top_bottom": {
                "breakdown_threshold": {
                    "intraday": {"condition": "price < neckline * 0.998", "duration_candles": 1},
                    "short_term": {"condition": "price < neckline * 0.995", "duration_candles": 2},
                    "long_term": {"condition": "price < neckline * 0.99", "duration_candles": 3}
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
            }
        },
        
        # ========================================================================
        # TIME ESTIMATION (Global Parameters)
        # ========================================================================
        "time_estimation": {
            "candles_per_unit": 1,
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
        
        # ========================================================================
        # ENTRY GATES (Global Baseline)
        # ========================================================================
        "entry_gates": {
            "confidence_requirements": {
                "breakout_base": 70,
                "trend_discount": -15,
                "accumulation_discount": -25
            },
            "adx_requirements": {
                "trend_setups_min": 18,
                "breakout_setups_min": 15
            },
            "execution_constraints": {
                "structure_validation": {
                    "breakout_clearance": 0.001,
                    "breakdown_clearance": 0.001
                },
                "sl_distance_validation": {
                    "min_atr_multiplier": 0.5,
                    "max_atr_multiplier": 5.0
                },
                "target_proximity_rejection": {
                    "min_t1_distance": 0.005,
                    "min_t2_distance": 0.01
                }
            }
        },
        
        # ========================================================================
        # CALCULATION ENGINE (Global Rules & Logic)
        # ========================================================================
        "calculation_engine": {
            "horizon_priority_overrides": {
                "intraday": {
                    "MOMENTUM_BREAKOUT": 95,
                    "VOLATILITY_SQUEEZE": 90,
                    "REVERSAL_MACD_CROSS_UP": 85,
                    "TREND_PULLBACK": 88,
                    "VALUE_TURNAROUND": 50,
                    "DEEP_VALUE_PLAY": 40,
                    "QUALITY_ACCUMULATION": 45
                },
                "short_term": {
                    "MOMENTUM_BREAKOUT": 90,
                    "VOLATILITY_SQUEEZE": 85,
                    "TREND_PULLBACK": 88,
                    "REVERSAL_MACD_CROSS_UP": 82,
                    "VALUE_TURNAROUND": 65,
                    "DEEP_VALUE_PLAY": 55,
                    "QUALITY_ACCUMULATION": 60
                },
                "long_term": {
                    "VALUE_TURNAROUND": 90,       # Highest priority for long-term
                    "DEEP_VALUE_PLAY": 88,
                    "QUALITY_ACCUMULATION": 85,
                    "TREND_PULLBACK": 80,
                    "MOMENTUM_BREAKOUT": 75,      # Lowered to avoid "Priority Trap"
                    "REVERSAL_MACD_CROSS_UP": 70,
                    "VOLATILITY_SQUEEZE": 65
                },
                "multibagger": {
                    "VALUE_TURNAROUND": 95,       # Absolute priority for multi-year holds
                    "DEEP_VALUE_PLAY": 92,
                    "QUALITY_ACCUMULATION": 90,
                    "TREND_PULLBACK": 70,
                    "REVERSAL_MACD_CROSS_UP": 65,
                    "MOMENTUM_BREAKOUT": 60
                }
            },
            # Composite Score Calculation
            "composite_weights": {
                "trend_strength": {
                    "adx": {
                        "weight": 0.4,
                        "thresholds": {"strong": 25, "moderate": 20, "weak": 15},
                        "scoring": [
                            {"min": 25, "score": 10},
                            {"min": 20, "score": 8},
                            {"min": 15, "score": 4},
                            {"default": 2}
                        ]
                    },
                    "ema_slope": {
                        "weight": 0.3,
                        "thresholds": {"strong": 20.0, "moderate": 5.0},
                        "scoring": [
                            {"min": 20, "score": 10},
                            {"min": 5, "score": 7},
                            {"default": 2}
                        ]
                    },
                    "di_spread": {
                        "weight": 0.2,
                        "scoring": [
                            {"min": 15, "score": 10},
                            {"min": 10, "score": 7},
                            {"default": 5}
                        ]
                    },
                    "supertrend": {
                        "weight": 0.1,
                        "scoring": {"bullish": 10, "bearish": 0}
                    },
                    "adaptive_weights_no_supertrend": {
                        "adx": 0.45,
                        "ema_slope": 0.35,
                        "di_spread": 0.20
                    }
                },
                "momentum_strength": {
                    "rsi_value": {
                        "weight": 0.25,
                        "thresholds": {"overbought": 70, "strong": 60, "neutral": 50, "weak": 40},
                        "scoring": [
                            {"min": 70, "score": 8},
                            {"min": 60, "score": 7},
                            {"min": 50, "score": 5},
                            {"default": 2}
                        ]
                    },
                    "rsi_slope": {
                        "weight": 0.25,
                        "thresholds": {"strong": 1.0, "neutral": 0.0},
                        "scoring": [
                            {"min": 1.0, "score": 8},
                            {"min": 0, "score": 4},
                            {"default": 2}
                        ]
                    },
                    "macd_hist": {
                        "weight": 0.3,
                        "thresholds": {"strong": 0.5, "neutral": 0.0},
                        "scoring": [
                            {"min": 0.5, "score": 8},
                            {"min": 0, "score": 5},
                            {"default": 2}
                        ]
                    },
                    "stoch_cross": {
                        "weight": 0.2,
                        "thresholds": {"overbought": 80, "neutral": 50},
                        "scoring": {
                            "bullish_strong": {"condition": "k>d and k>=50", "score": 8},
                            "default": 3
                        }
                    }
                },
                "volatility_quality": {
                    "atr_pct": {
                        "weight": 0.3,
                        "thresholds": {"low": 1.5, "moderate": 3.0, "high": 5.0},
                        "scoring": [
                            {"max": 1.5, "score": 10},
                            {"max": 3.0, "score": 8},
                            {"max": 5.0, "score": 6},
                            {"default": 2}
                        ]
                    },
                    "bb_width": {
                        "weight": 0.25,
                        "thresholds": {"tight": 0.01, "moderate": 0.02, "wide": 0.04},
                        "scoring": [
                            {"max": 0.01, "score": 10},
                            {"max": 0.02, "score": 8},
                            {"default": 2}
                        ]
                    },
                    "true_range_consistency": {
                        "weight": 0.20,
                        "scoring": [{"max": 0.5, "score": 10}, {"default": 4}]
                    },
                    "hv_trend": {
                        "weight": 0.15,
                        "scoring": {"declining": {"condition": "hv_10 < hv_20", "score": 8}, "default": 4}
                    },
                    "atr_sma_ratio": {
                        "weight": 0.15,
                        "thresholds": {"stable": 0.02, "moderate": 0.035, "volatile": 0.05},
                        "scoring": [
                            {"max": 0.02, "score": 10},
                            {"max": 0.035, "score": 7},
                            {"default": 3}
                        ]
                    }
                }
            },
            
            # Setup Classification (SINGLE SOURCE - NOT in horizons)
            "setup_classification": {
                "pattern_priority": [
                    {"pattern": "darvas_box", "setup_name": "PATTERN_DARVAS_BREAKOUT", "min_score": 85},
                    {"pattern": "minervini_stage2", "setup_name": "PATTERN_VCP_BREAKOUT", "min_score": 85},
                    {"pattern": "cup_handle", "setup_name": "PATTERN_CUP_BREAKOUT", "min_score": 80},
                    {"pattern": "three_line_strike", "setup_name": "PATTERN_STRIKE_REVERSAL", "min_score": 80},
                    {"pattern": "golden_cross", "setup_name": "PATTERN_GOLDEN_CROSS", "min_score": 75},
                    {"pattern": "flag_pennant", "setup_name": "PATTERN_FLAG_BREAKOUT", "min_score": 80}
                ],
                "consolidation": {
                    "bb_width_threshold": 0.5,
                    "volume_ratio_max": 0.8
                },
                "MOMENTUM_BREAKOUT": {
                    "bb_percent_b_min": 0.98,
                    "rsi_min": 60,
                    "wick_ratio_max": 2.5,
                    "rvol_min": 1.5
                },
                "MOMENTUM_BREAKDOWN": {
                    "bb_percentb_max": 0.02,
                    "rsi_max": 40,
                    "rvol_min": 1.5
                },
                "TREND_PULLBACK": {
                    "ma_dist_max": 0.05,
                    "rsi_min": 50
                },
                "bear_pullback": {
                    "ma_dist_max": 0.05,
                    "rsi_max": 50
                },
                "trend_following": {
                    "classic": {
                        "rsi_min": 55,
                        "macd_hist_min": 0
                    },
                    "strong_drift": {
                        "trend_strength_min": 7.0
                    }
                },
                "QUALITY_ACCUMULATION": {
                    "consolidation_required": True,
                    "rsi_range": [40, 60]
                },
                "QUALITY_ACCUMULATION_DOWNTREND": {
                    "fundamental_requirements": {
                        "roe_min": 20,
                        "roce_min": 25,
                        "de_ratio_max": 0.5
                    },
                    "bb_percent_b_range": [0.2, 0.5]
                },
                "divergence": {
                    "lookback": 10,
                    "slope_diff_min": -0.05,
                    "confidence_penalties": {
                        "bearish_divergence": 0.70,
                        "bullish_divergence": 0.70
                    }
                },
                "rules": {
                    "MOMENTUM_BREAKOUT": {
                        "conditions": [
                            "bb_percent_b >= 0.98",
                            "rsi >= 60"
                        ],
                        "priority": 90
                    },
                    "MOMENTUM_BREAKDOWN": {
                        "conditions": [
                            "bb_percent_b <= 0.02",
                            "rsi <= 40"
                        ],
                        "priority": 90
                    },
                    "TREND_PULLBACK": {
                        "conditions": [
                            "trend_dir == 'up'",
                            "abs(price - ma_fast) / ma_fast <= 0.05",
                            "rsi >= 50"
                        ],
                        "priority": 75
                    },
                    "VOLATILITY_SQUEEZE": {
                        "conditions": [
                            "is_squeeze == True",
                            "volatility_quality >= 7.0"
                        ],
                        "priority": 80
                    },
                    "QUALITY_ACCUMULATION": {
                        "conditions": [
                            "is_consolidating == True",
                            "roe >= 20",
                            "roce >= 25",
                            "de_ratio <= 0.5"
                        ],
                        "priority": 65
                    },
                    "REVERSAL_MACD_CROSS_UP": {
                        "conditions": ["macd_histogram > 0", "prev_macd_histogram < 0"],
                        "priority": 80
                    },
                    "REVERSAL_RSI_SWING_UP": {
                        "conditions": ["rsi < 30", "rsi_slope > 0.05"],
                        "priority": 75
                    },
                    "REVERSAL_ST_FLIP_UP": {
                        "conditions": ["supertrend_signal == 'Bullish'", "prev_supertrend_signal == 'Bearish'"],
                        "priority": 70
                    },
                    "SELL_AT_RANGE_TOP": {
                        "conditions": ["is_consolidating == True", "price >= bb_high * 0.98", "rsi > 60"],
                        "priority": 75
                    },
                    "TAKE_PROFIT_AT_MID": {
                        "conditions": ["is_consolidating == True", "price >= bb_mid * 0.98", "price < bb_high * 0.98", "rsi > 55"],
                        "priority": 60
                    },
                    "QUALITY_ACCUMULATION_DOWNTREND": {
                        "conditions": [
                            "trend_strength < 3.0",           # In downtrend/sideways
                            "roe >= 20",                      # High quality
                            "roce >= 25",
                            "de_ratio <= 0.5",                # Low debt
                            "pe_ratio < 15"                   # Cheap valuation
                        ],
                        "priority": 85
                    },
                    "DEEP_VALUE_PLAY": {
                        "conditions": [
                            "pe_ratio < 10",                  # Very cheap
                            "fcf_yield > 5.0",                # High cash generation
                            "roe >= 15",                      # Still profitable
                            "dividend_yield > 4.0"            # High income
                        ],
                        "priority": 80
                    },
                    "VALUE_TURNAROUND": {
                        "conditions": [
                            "trend_strength >= 3.0",          # Starting to turn
                            "trend_strength < 5.0",
                            "roe >= 18",
                            "pe_ratio < 12",
                            "momentum_strength >= 4.0"
                        ],
                        "priority": 82
                    },
                }
            },

            # Spread Adjustment
            "spread_adjustment": {
                "market_cap_brackets": {
                    "large_cap": {"min": 100000, "spread_pct": 0.001},
                    "mid_cap": {"min": 10000, "max": 100000, "spread_pct": 0.002},
                    "small_cap": {"max": 10000, "spread_pct": 0.005}
                }
            },
            
            # Volume Signatures
            "volume_signatures": {
                "surge": {"threshold": 3.0, "confidence_adjustment": 15},
                "drought": {"threshold": 0.7, "confidence_adjustment": -25},
                "climax": {"threshold": 2.0, "rsi_condition_min": 70}
            },
            
            # Wick Rejection
            "wick_rejection": {
                "max_ratio": 2.5,
                "calculation": "abs(high - close) / abs(close - open)"
            },
            
            # Divergence Detection
            "divergence_detection": {
                "lookback": 10,
                "slope_diff_min": -0.05,
                "confidence_penalties": {
                    "bearish_divergence": 0.70,
                    "bullish_divergence": 0.70
                },
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
            
            # Setup Confidence (REMOVED - Will only be in horizons)
        },
        
        # ========================================================================
        # POSITION SIZING (Global Parameters)
        # ========================================================================
        "position_sizing": {
            "base_risk_pct": 0.01,
            "global_setup_multipliers": {
                "DEEP_PULLBACK": 1.5,
                "VOLATILITY_SQUEEZE": 1.3,
                "MOMENTUM_BREAKOUT": 0.8
            },
            "volatility_adjustments": {
                "high_quality": {"vol_qual_min": 7.0, "multiplier": 1.2},
                "low_quality": {"vol_qual_max": 5.0, "multiplier": 0.9}
            }
        },
        
        # ========================================================================
        # WEIGHTS (Various)
        # ========================================================================
        "trend_weights": {
            "primary": 0.50,
            "secondary": 0.30,
            "acceleration": 0.20
        },
        
        # ========================================================================
        # STRATEGY CONFIG
        # ========================================================================
        "strategy_classification": {
            "swing": {"fit_thresh": 50},
            "day_trading": {"fit_thresh": 60},
            "trend_following": {"fit_thresh": 60},
            "momentum": {"fit_thresh": 60},
            "minervini": {"fit_thresh": 70},
            "canslim": {"fit_thresh": 65},
            "value": {"fit_thresh": 50}
        },
        "strategy_time_multipliers": {
            "momentum": 0.7,
            "day_trading": 0.5,
            "swing": 1.0,
            "trend_following": 1.2,
            "position_trading": 1.5,
            "value": 1.5,
            "income": 2.0,
            "unknown": 1.0
        },
        "strategy_weights": {
            "swing": {
                "momentum": 0.30,
                "trend": 0.35,
                "volatility": 0.35
            },
            "day_trading": {
                "momentum": 0.40,
                "trend": 0.30,
                "volatility": 0.30
            },
            "trend_following": {
                "momentum": 0.25,
                "trend": 0.50,
                "volatility": 0.25
            },
            "momentum": {
                "momentum": 0.50,
                "trend": 0.30,
                "volatility": 0.20
            },
            "position_trading": {
                "momentum": 0.20,
                "trend": 0.50,
                "volatility": 0.30
            }
        },
        
        # ========================================================================
        # BOOSTS
        # ========================================================================
        "boosts": {
            "pattern": {"single": 0.8, "confluence": 1.5},
            "momentum": {"score": 1.2, "trigger_rvol": 2.0},
            "volatility": {
                "squeeze": {"score": 1.0, "min_quality": 7.0},
                "expansion": {"score": 0.6, "min_quality": 4.0}
            },
            "relative_strength": {
                "outperforming_weak_market": 1.5,
                "outperforming_strong_market": 0.8
            },
            "proximity": {"support": 0.6, "resistance": 0.8},
            "max_cap": 2.5
        },
        
        # ========================================================================
        # SYSTEM/INFRASTRUCTURE
        # ========================================================================
        "system": {
            "process_pool_workers": 4,
            "cache": {"ttl_seconds": 3600, "timezone": "Asia/Kolkata"},
            "cache_warmer": {
                "batch_size": 5,
                "market_interval_sec": 900,
                "top_n_during_market": 50,
                "lru_target": 500,
                "deep_hour": 2,
                "batch_sleep_market": 5.0
            },
            "fetch": {
                "max_retries": 3,
                "timeout": 10,
                "period_map": {
                    "intraday": "1mo",
                    "short_term": "2y",
                    "long_term": "5y",
                    "multibagger": "10y"
                },
                "interval_map": {
                    "intraday": "15m",
                    "short_term": "1d",
                    "long_term": "1wk",
                    "multibagger": "1mo"
                }
            },
            "corporate_actions": {
                "lookback_days": {"past": 365, "upcoming": 7},
                "display_priority": ["Dividend", "Bonus", "Split", "Rights"]
            }
        },
        
        # ========================================================================
        # DEFAULT CONFIGS (Used as fallback for all horizons)
        # ========================================================================
        "moving_averages": {
            "type": "EMA",
            "fast": 20,
            "mid": 50,
            "slow": 200,
            "keys": ["ema_20", "ema_50", "ema_200"],
            "dip_buy_reference": "ema_50"
        },
        
        "indicators": {
            "rsi_period": 14,
            "adx_period": 14,
            "atr_period": 14,
            "supertrend": {"period": 10, "multiplier": 3},
            "bollinger": {"period": 20, "std_dev": 2.0},
            "keltner": {"period": 20, "atr_mult": 1.5},
            "pivot_type": "classic",
            "stochastic": {
                "k": 14,
                "d": 3,
                "smooth": 3,
                "overbought": 80,
                "oversold": 20
            }
        },
        
        "momentum_thresholds": {
            "rsi_slope": {"acceleration_floor": 0.05, "deceleration_ceiling": -0.05},
            "macd": {"acceleration_floor": 0.5, "deceleration_ceiling": -0.5}
        },
        
        "volatility": {
            "scoring_thresholds": {
                "atr_pct": {"excellent": 2.5, "good": 3.0, "fair": 4.5, "poor": 5.5},
                "bb_width": {"tight": 3.0, "normal": 6.0, "wide": 12.0}
            },
            "extreme_threshold": 7.0,
            "min_quality": 4.0,
            "squeeze_quality_min": 7.0,
            "breakout_min_quality": 2.0
        },
        
        "risk_management": {
            "max_position_pct": 0.02,
            "setup_size_multipliers": {"default": 1.0},
            "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
            "min_rr_ratio": 1.5,
            "horizon_t2_cap": 0.10,
            "rr_regime_adjustments": {
                "strong_trend": {"adx_min": 35, "t1_mult": 2.0, "t2_mult": 4.0},
                "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
            }
        },
        
        "execution": {
            "stop_loss_atr_mult": 2.0,
            "target_atr_mult": 3.0,
            "max_hold_candles": 20,
            "dip_buy_reference": "ema_50",
            "risk_reward_min": 2.0,
            "stop_loss": {
                "vol_qual_high_mult": 1.5,
                "vol_qual_normal_mult": 2.0,
                "vol_qual_low_mult": 3.0,
                "min_distance_mult": 0.5
            },
            "spread_adjustments": {
                "large_cap": {"min_mcap": 100000, "spread": 0.001},
                "mid_cap": {"min_mcap": 10000, "spread": 0.002},
                "small_cap": {"spread": 0.005}
            },
            "structure_validation": {
                "breakout_tolerance": 1.001,
                "breakdown_tolerance": 0.999
            }
        },
        
        "lookback": {"python_data": 600},
        
        "scoring": {
            "metrics": {
                "trend_strength": 0.20,
                "momentum_strength": 0.20,
                "volatility_quality": 0.15,
                "ma_trend_signal": 0.15,
                "supertrend_signal": 0.10,
                "rvol": 0.10,
                "price_action": 0.10
            },
            "penalties": {
                "rvol": {"op": "<", "val": 0.5, "pen": 0.2},
                "atr_pct": {"op": "<", "val": 1.0, "pen": 0.2}
            },
            "thresholds": {"buy": 6.0, "hold": 5.0, "sell": 4.0}
        },
        
        "gates": {
            "min_trend_strength": 3.0,
            "allowed_supertrend_counter": False,
            "volatility_bands_atr_pct": {
                "min": 1.0,
                "ideal": 2.5,
                "max": 12.0
            },
            "volatility_quality_mins": {"default": 3.0},
            "default_setup_gate_overrides": {
            # Universal overrides that apply to ALL horizons These define the NATURE of each setup type
                "QUALITY_ACCUMULATION": { "adx_min": None, "min_trend_strength": None},              # Always skip - consolidation by nature      # Always skip - sideways by definition
                "QUALITY_ACCUMULATION": { "adx_min": None, "min_trend_strength": None},
                "QUALITY_ACCUMULATION_DOWNTREND": { "adx_min": None, "min_trend_strength": None},
                "DEEP_VALUE_PLAY": { "adx_min": None, "min_trend_strength": None},
                "VALUE_TURNAROUND": { "adx_min": 15, "min_trend_strength": 3.0 },              # Universal minimum (turning point) 
                "REVERSAL_MACD_CROSS_UP": { "adx_min": 10, "min_trend_strength": None},
                "REVERSAL_RSI_SWING_UP": { "adx_min": 12, "min_trend_strength": None},
                "REVERSAL_ST_FLIP_UP": { "adx_min": 14, "min_trend_strength": 2.5},
                "TREND_PULLBACK": { "adx_min": 16, "min_trend_strength": 4.0},
                "DEEP_PULLBACK": { "adx_min": 14, "min_trend_strength": 3.5},
                "VOLATILITY_SQUEEZE": { "adx_min": None, "min_trend_strength": None},             # Always skip - pre-trend by nature 
                "MOMENTUM_BREAKOUT": { "adx_min": 18, "min_trend_strength": 5.0},
                "PATTERN_DARVAS_BREAKOUT": { "adx_min": 16, "min_trend_strength": 4.0},
                "PATTERN_VCP_BREAKOUT": { "adx_min": 15, "min_trend_strength": 4.5},
                "PATTERN_CUP_BREAKOUT": { "adx_min": 14, "min_trend_strength": 3.5},
                "PATTERN_FLAG_BREAKOUT": { "adx_min": 18, "min_trend_strength": 5.5},
                "PATTERN_GOLDEN_CROSS": { "adx_min": 15, "min_trend_strength": 3.0},
                "GENERIC": { "adx_min": None, "min_trend_strength": None                } 
            }
            # NOTE: Only ADX/trend overrides in global
            # Horizon-specific values (confidence, volatility) go in horizons
        },
        
        "confidence": {
            "horizon_discount": 5,
            "floors": {"buy": 55, "wait": 35},
            "volume_penalty": {"rvol_drought_penalty": -15, "ignore_for_squeeze": True},
            "adx_based_floors": {
                "strong": {"adx_min": 30, "floor": 60},
                "moderate": {"adx_min": 25, "floor": 50},
                "weak": {"adx_min": 20, "floor": 40},
                "range_bound": {"adx_max": 20, "floor": 30}
            },
            "setup_type_overrides": {
                "MOMENTUM_BREAKOUT": 5,
                "VOLATILITY_SQUEEZE": 10,
                "QUALITY_ACCUMULATION": 0
            },
            "base_floors": {
                "MOMENTUM_BREAKOUT": 55,
                "MOMENTUM_BREAKDOWN": 55,
                "TREND_PULLBACK": 53,
                "DEEP_PULLBACK": 50,
                "QUALITY_ACCUMULATION": 45,
                "VOLATILITY_SQUEEZE": 50,
                "TREND_FOLLOWING": 50,
                "BEAR_TREND_FOLLOWING": 50,
                "REVERSAL_MACD_CROSS_UP": 50,
                "REVERSAL_RSI_SWING_UP": 50,
                "REVERSAL_ST_FLIP_UP": 55,
                "VALUE_TURNAROUND": 50,  # Starts lower as it's a turning point
                "DEEP_VALUE_PLAY": 45,   # Low base floor, relies heavily on fundamental quality
                "QUALITY_ACCUMULATION_DOWNTREND": 45  # Value play in a negative trend
            },
            "adx_normalization": {
                "min": 10,
                "max": 40,
                "adjustment_factor": 12
            }
        },
        
        "composites": {
            "trend_strength": {
                "adx_thresholds": {"strong": 25, "moderate": 20, "weak": 15},
                "slope_thresholds": {"strong": 20.0, "moderate": 5.0},
                "weights": {
                    "adx": 0.4,
                    "slope": 0.3,
                    "di": 0.2,
                    "st": 0.1,
                    "adx_adaptive": 0.45,
                    "slope_adaptive": 0.35,
                    "di_adaptive": 0.20
                }
            },
            "momentum_strength": {
                "rsi_thresholds": {"overbought": 70, "strong": 60, "neutral": 50, "weak": 40},
                "rsi_slope_thresholds": {"strong": 1.0, "neutral": 0.0},
                "macd_thresholds": {"strong": 0.5, "neutral": 0.0},
                "stoch_thresholds": {"overbought": 80, "neutral": 50},
                "weights": {"rsi": 0.25, "slope": 0.25, "macd": 0.30, "stoch": 0.20}
            },
            "volatility_quality": {
                "atr_pct_thresholds": {"low": 1.5, "moderate": 3.0, "high": 5.0},
                "bb_width_thresholds": {"tight": 0.01, "moderate": 0.02, "wide": 0.04},
                "atr_sma_ratio_thresholds": {"stable": 0.02, "moderate": 0.035, "volatile": 0.05},
                "weights": {"atr": 0.25, "bb_width": 0.25, "true_range": 0.20, "hv": 0.15, "ratio": 0.15}
            }
        },
        
        "targets": {
            "resistance_cushion": 0.96,
            "support_cushion": 1.005,
            "min_distance_pct": 0.5,
            "support_buffer": 0.998,
            "cover_cushion": 1.005
        },
        
        "enhancements": {},
        
        "divergence": {
            "rsi_slope_deceleration_ceiling": -0.08,
            "bearish_penalty": 0.70,
            "bullish_penalty": 0.70,
            "severity_bands": {
                "severe": {
                    "rsi_slope_threshold": -0.08,
                    "allow_entry": False,
                    "confidence_penalty": 1.0
                },
                "moderate": {
                    "rsi_slope_threshold": -0.03,
                    "allow_entry": True,
                    "confidence_penalty": 0.7
                },
                "minor": {
                    "rsi_slope_threshold": 0.0,
                    "allow_entry": True,
                    "confidence_penalty": 0.9
                }
            }
        }
    },


    "horizons": {
        
        # ========================================================================
        # INTRADAY (15-minute candles, scalping & day trading)
        # ========================================================================
        "intraday": {
            "volume_analysis": {
                "rvol_surge_threshold": 3.0,
                "rvol_drought_threshold": 0.7
            },
            "time_estimation": {
                "candles_per_unit": 4
            },
            "timeframe": "15m",
            "description": "Quick scalps and day trades",
            
            # Override MA settings
            "moving_averages": {
                "type": "EMA",
                "fast": 20,
                "mid": 50,
                "slow": 200,
                "keys": ["ema_20", "ema_50", "ema_200"],
                "dip_buy_reference": "ema_20"
            },
            
            # Override indicators for faster response
            "indicators": {
                "rsi_period": 9,
                "adx_period": 10,
                "atr_period": 10,
                "supertrend": {"period": 7, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "fibonacci",
                "stochastic": {
                    "k": 8,
                    "d": 3,
                    "smooth": 3,
                    "overbought": 80,
                    "oversold": 20
                }
            },
            
            # Tighter momentum thresholds
            "momentum_thresholds": {
                "rsi_slope": {
                    "acceleration_floor": 0.10,
                    "deceleration_ceiling": -0.10
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            
            # Tighter volatility scoring
            "volatility": {
                "scoring_thresholds": {
                    "atr_pct": {
                        "excellent": 1.5,
                        "good": 3.0,
                        "fair": 4.0,
                        "poor": 5.0
                    },
                    "bb_width": {
                        "tight": 2.0,
                        "normal": 5.0,
                        "wide": 10.0
                    }
                }
            },
            
            # Tighter risk management
            "risk_management": {
                "max_position_pct": 0.01,
                "setup_size_multipliers": {
                    "VOLATILITY_SQUEEZE": 1.3,
                    "MOMENTUM_BREAKOUT": 0.8
                },
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "min_rr_ratio": 1.2,
                "horizon_t2_cap": 0.04,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
            },
            
            # Execution params
            "execution": {
                "stop_loss_atr_mult": 1.5,
                "target_atr_mult": 2.5,
                "max_hold_candles": 25,
                "risk_reward_min": 1.5,
                "base_hold_days": 1,
                "proximity_rejection": {
                    "resistance_mult": 1.003,
                    "support_mult": 0.997
                },
                "min_profit_pct": 0.3
            },
            
            "lookback": {"python_data": 500},
            
            # Scoring - focus on momentum & price action
            "scoring": {
                "fundamental_weight": 0.0,
                "metrics": {
                    "ma_fast_slope": 0.20,
                    "rsi_slope": 0.20,
                    "price_action": 0.15,
                    "vwap_bias": 0.15,
                    "vol_spike_ratio": 0.10,
                    "momentum_strength": 0.10,
                    "volatility_quality": 0.05,
                    "ma_trend_signal": 0.05
                },
                "penalties": {
                    "atr_pct": {"op": "<", "val": 0.4, "pen": 0.3},
                    "ma_fast_slope": {"op": "<", "val": -2, "pen": 0.3}
                },
                "thresholds": {"buy": 6.5, "hold": 5.0, "sell": 3.8},
                "metric_weights": {
                    "momentum_strength": 0.30,
                    "trend_strength": 0.25,
                    "volatility_quality": 0.15,
                    "vwap_bias": 0.10,
                    "rvol": 0.10,
                    "price_action": 0.10
                }
            },
            
            # Technical weight overrides - boost fast indicators
            "technical_weight_overrides": {
                "rsi": 1.3,
                "rsi_slope": 1.2,
                "stoch_k": 1.2,
                "macd_cross": 1.2,
                "vol_spike_signal": 1.5,
                "rvol": 1.3,
                "vol_spike_ratio": 1.3,
                "vwap_bias": 1.4,
                "price_action": 1.3,
                "gap_percent": 1.2,
                "bb_percent_b": 1.2,
                "wick_rejection": 1.2,
                "momentum_strength": 1.3,
                "price_vs_200dma_pct": 0.4,
                "price_vs_primary_trend_pct": 0.5,
                "ma_trend_signal": 0.6,
                "adx": 0.7,
                "dma_200_slope": 0.4,
                "ichi_cloud": 0.5,
                "rel_strength_nifty": 0.5,
                "nifty_trend_score": 0.4,
                "trend_strength": 0.8,
                "volatility_quality": 1.1
            },
            # "gates": {
            #     "min_trend_strength": 2.0,
            #     "allowed_supertrend_counter": True,
            #     "volatility_bands_atr_pct": {
            #         "min": 0.3,
            #         "ideal": 3.0,
            #         "max": 5.0
            #     },
            #     "volatility_quality_mins": {
            #         "MOMENTUM_BREAKOUT": 2.5,
            #         "VOLATILITY_SQUEEZE": 4.0,
            #         "TREND_PULLBACK": 3.0,
            #         "default": 2.5
            #     },
            #     "volatility_guards": {
            #         "extreme_vol_buffer": 2.0,
            #         "min_quality_breakout": 2.5,
            #         "min_quality_normal": 4.0
            #     },
            #     "confidence_min": 55,
            #     "adx_min": 15,
            #     "volatility_quality_min": 3.0,
            #     "rr_ratio_min": 1.1,
            #     "trend_strength_min": 2.0
            # },
            # Gates - more lenient
            "gates": {
                # Base gates for intraday
                "adx_min": 18,
                "min_trend_strength": 5.0,
                "confidence_min": 65,
                "volatility_quality_min": 5.0,
                "volatility_bands_atr_pct": {
                    "min": 0.3,
                    "ideal": 3.0,
                    "max": 5.0
                },
                "volatility_guards": {
                    "extreme_vol_buffer": 2.0,      
                    "min_quality_breakout": 2.5,
                    "min_quality_normal": 4.0
                },
                # Only override horizon-specific values
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"volatility_quality_min": 4.0,"confidence_min": 55},
                    "DEEP_VALUE_PLAY": {"volatility_quality_min": 2.5,"confidence_min": 45},
                    "VOLATILITY_SQUEEZE": {"volatility_quality_min": 7.0, "confidence_min": 60},  # Intraday = tighter squeeze
                    "GENERIC": {"volatility_quality_min": 3.0,"confidence_min": 45}
                    # Don't repeat adx_min/min_trend_strength - inherited from global
                }
            },
            "trend_thresholds": {
                "slope": {
                    "strong": 15.0,
                    "moderate": 5.0
                }
            },
            
            # Confidence - higher discount for noise
            "confidence": {
                "horizon_discount": 10,
                "floors": {"buy": 55, "wait": 30},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 50,
                    "MOMENTUM_BREAKDOWN": 50,
                    "TREND_PULLBACK": 48,
                    "DEEP_PULLBACK": 45,
                    "QUALITY_ACCUMULATION": 40,
                    "VOLATILITY_SQUEEZE": 50,
                    "TREND_FOLLOWING": 48,
                    "BEAR_TREND_FOLLOWING": 48,
                    "REVERSAL_MACD_CROSS_UP": 45,
                    "REVERSAL_RSI_SWING_UP": 45,
                    "REVERSAL_ST_FLIP_UP": 50
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -20,
                    "ignore_for_squeeze": True
                }
            },
            
            # Setup Confidence - intraday-specific penalties/bonuses
            "setup_confidence": {
                "confidence_clamp": [30, 90],
                "penalties": {
                    "weak_trend": {
                        "condition": "adx < 20",
                        "amount": 10,
                        "reason": "Intraday trend too weak"
                    },
                    "low_liquidity": {
                        "condition": "rvol < 0.8",
                        "amount": 10,
                        "reason": "Thin intraday liquidity"
                    },
                    "wide_spread": {
                        "condition": "spread_pct > 0.005",
                        "amount": 8,
                        "reason": "Spread too wide for intraday"
                    },
                    "choppy_market": {
                        "condition": "trend_strength < 3.0",
                        "amount": 20,
                        "reason": "Choppy intraday - high whipsaw risk"
                    }
                },
                "bonuses": {
                    "momentum_volume_confirmed": {
                        "condition": "setup_type == 'MOMENTUM_BREAKOUT' and rvol >= 3.0",
                        "amount": 10,
                        "reason": "Breakout with strong volume"
                    },
                    "clean_flag": {
                        "condition": "setup_type == 'PATTERN_FLAG_BREAKOUT' and volatility_quality >= 6.0",
                        "amount": 8,
                        "reason": "Clean intraday flag breakout"
                    },
                    "explosive_trend": {
                        "condition": "trend_strength >= 8.0",
                        "amount": 20,
                        "reason": "Explosive intraday trend"
                    },
                    "strong_trend": {
                        "condition": "trend_strength >= 5.5",
                        "amount": 15,
                        "reason": "Strong intraday momentum"
                    },
                    "decent_trend": {
                        "condition": "trend_strength >= 4.0",
                        "amount": 8,
                        "reason": "Decent trend support"
                    }
                }
            },
            
            # Enhancements - intraday-specific
            "enhancements": {
                "volume_surge": {
                    "condition": "rvol >= 2.5",
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
            }
        },
        
        # ========================================================================
        # SHORT_TERM (Daily candles, swing trading)
        # ========================================================================
        "short_term": {
            "volume_analysis": {
                "rvol_surge_threshold": 2.5,
                "rvol_drought_threshold": 0.7
            },
            "time_estimation": {
                "candles_per_unit": 1
            },
            "timeframe": "1d",
            "description": "Swing trading (Days to Weeks)",
            
            # Uses global MA settings (EMA 20/50/200)
            
            # Risk management
            "risk_management": {
                "max_position_pct": 0.02,
                "setup_size_multipliers": {
                    "DEEP_PULLBACK": 1.5,
                    "MOMENTUM_BREAKOUT": 1.0
                },
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "min_rr_ratio": 1.4,
                "horizon_t2_cap": 0.10,
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
            },
            
            # Execution
            "execution": {
                "stop_loss_atr_mult": 2.0,
                "target_atr_mult": 3.0,
                "max_hold_candles": 15,
                "dip_buy_reference": "ema_50",
                "risk_reward_min": 2.0,
                "base_hold_days": 10,
                "proximity_rejection": {
                    "resistance_mult": 1.005,
                    "support_mult": 0.995
                },
                "min_profit_pct": 0.5
            },
            
            "lookback": {"python_data": 600},
            
            # Scoring - balanced tech/fund
            "scoring": {
                "fundamental_weight": 0.3,
                "metrics": {
                    "trend_strength": 0.15,
                    "ma_fast_slope": 0.15,
                    "rsi_slope": 0.10,
                    "momentum_strength": 0.12,
                    "volatility_quality": 0.10,
                    "supertrend_signal": 0.10,
                    "ma_trend_signal": 0.10,
                    "price_vs_primary_trend_pct": 0.08,
                    "rvol": 0.10
                },
                "penalties": {
                    "ma_fast_slope": {"op": "<", "val": -5, "pen": 0.3}
                },
                "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 4.0},
                "metric_weights": {
                    "pe_ratio": 0.10,
                    "peg_ratio": 0.10,
                    "roe": 0.10,
                    "roce": 0.10,
                    "de_ratio": 0.05,
                    "eps_growth_5y": 0.15,
                    "revenue_growth_5y": 0.10,
                    "quarterly_growth": 0.05,
                    "trend_strength": 0.10,
                    "momentum_strength": 0.10,
                    "volatility_quality": 0.05
                }
            },
            
            # Technical weights - balanced
            "technical_weight_overrides": {
                "momentum_strength": 1.1,
                "trend_strength": 1.1,
                "macd_cross": 1.1,
                "supertrend_signal": 1.1,
                "ma_cross_signal": 1.1,
                "vol_spike_signal": 1.0,
                "rvol": 1.0,
                "adx": 1.0,
                "price_vs_200dma_pct": 1.0,
                "rsi_slope": 1.0,
                "stoch_k": 1.0,
                "volatility_quality": 1.0,
                "gap_percent": 0.9,
                "vwap_bias": 0.8
            },
            
            # Gates
            # "gates": {
                # "min_trend_strength": 2.0,
                # "allowed_supertrend_counter": False,
                # "volatility_bands_atr_pct": {
                #     "min": 0.8,
                #     "ideal": 2.5,
                #     "max": 12.0
                # },
                # "volatility_quality_mins": {
                #     "MOMENTUM_BREAKOUT": 3.0,
                #     "VOLATILITY_SQUEEZE": 5.0,
                #     "TREND_PULLBACK": 3.5,
                #     "default": 3
                # },
                # "confidence_min": 60,
                # "adx_min": 18,
                # "volatility_quality_min": 4.0,
                # "trend_strength_min": 3.5,
                # "rr_ratio_min": 1.3,
                # "volatility_guards": {
                #     "extreme_vol_buffer": 2.0,
                #     "min_quality_breakout": 3.0,
                #     "min_quality_normal": 4.0
                # },
                # "setup_gate_overrides": {
                #     "QUALITY_ACCUMULATION": {
                #         "min_trend_strength": None,  # Skip trend check
                #         "adx_min": None,
                #         "volatility_quality_min": 3.0
                #     },
                #     "QUALITY_ACCUMULATION_DOWNTREND": {
                #         "min_trend_strength": None,
                #         "adx_min": None,
                #         "volatility_quality_min": 3.0
                #     },
                #     "DEEP_VALUE_PLAY": {
                #         "min_trend_strength": None,
                #         "adx_min": None,
                #         "volatility_quality_min": 2.5
                #     }
                # }
                # },
            "gates": {
                "adx_min": 15,
                "min_trend_strength": 4.0,
                "confidence_min": 60,
                "volatility_quality_min": 4.0,
                "volatility_bands_atr_pct": {
                    "min": 0.8,
                    "ideal": 2.5,
                    "max": 12.0
                },
                 "volatility_guards": {
                    "extreme_vol_buffer": 2.0,      # Daily candles
                    "min_quality_breakout": 3.0,
                    "min_quality_normal": 4.0
                },
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"volatility_quality_min": 2.5, "confidence_min": 45},   # Swing more lenient
                    "DEEP_VALUE_PLAY": {"volatility_quality_min": 2.0,"confidence_min": 40},
                    "VOLATILITY_SQUEEZE": {"volatility_quality_min": 6.0,  "confidence_min": 55}, # Slightly looser than intraday
                    "GENERIC": {"volatility_quality_min": 2.5,"confidence_min": 40                    }
                }
            },
            
            "trend_thresholds": {
                "slope": {
                    "strong": 10.0,
                    "moderate": 3.0
                }
            },
            
            # Confidence
            "confidence": {
                "horizon_discount": 5,
                "floors": {"buy": 55, "wait": 30},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 55,
                    "MOMENTUM_BREAKDOWN": 55,
                    "TREND_PULLBACK": 53,
                    "DEEP_PULLBACK": 50,
                    "QUALITY_ACCUMULATION": 45,
                    "VOLATILITY_SQUEEZE": 50,
                    "TREND_FOLLOWING": 50,
                    "BEAR_TREND_FOLLOWING": 50,
                    "REVERSAL_MACD_CROSS_UP": 50,
                    "REVERSAL_RSI_SWING_UP": 50,
                    "REVERSAL_ST_FLIP_UP": 55
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -20,
                    "ignore_for_squeeze": True
                }
            },
            
            # Setup Confidence
            "setup_confidence": {
                "confidence_clamp": [35, 95],
                "penalties": {
                    "moderate_divergence": {
                        "condition": "rsi_slope < -0.03",
                        "amount": 10,
                        "reason": "Moderate bearish RSI divergence"
                    },
                    "low_breakout_volume": {
                        "condition": "setup_type == 'MOMENTUM_BREAKOUT' and rvol < 1.5",
                        "amount": 8,
                        "reason": "Breakout with insufficient volume"
                    },
                    "sideways_trend": {
                        "condition": "trend_strength < 3.5",
                        "amount": 15,
                        "reason": "Weak trend - sideways risk"
                    },
                    "trend_setup_weak_trend": {
                        "condition": "setup_type in ['TREND_PULLBACK', 'TREND_FOLLOWING'] and trend_strength < 4.0",
                        "amount": 20,
                        "reason": "Trend-following setup without strong trend"
                    }
                },
                "bonuses": {
                    "pattern_confluence": {
                        "condition": "pattern_count >= 2",
                        "amount": 10,
                        "reason": "Multiple bullish patterns aligned"
                    },
                    "exceptional_combo": {
                        "condition": "trend_strength >= 7.0 and momentum_strength >= 7.0",
                        "amount": 25,
                        "reason": "Exceptional trend + momentum synergy"
                    },
                    "very_strong_trend": {
                        "condition": "trend_strength >= 7.5 and momentum_strength < 7.0",
                        "amount": 25,
                        "reason": "Very strong trend (standalone)"
                    },
                    "strong_trend": {
                        "condition": "trend_strength >= 6.0 and trend_strength < 7.5",
                        "amount": 20,
                        "reason": "Strong trend confirmation"
                    },
                    "moderate_trend": {
                        "condition": "trend_strength >= 4.5 and trend_strength < 6.0",
                        "amount": 10,
                        "reason": "Moderate trend support"
                    }
                }
            },
            
            # Enhancements
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
            }
        },
        
        # ========================================================================
        # LONG_TERM (Weekly candles, trend following & investing)
        # ========================================================================
        "long_term": {
            "trend_thresholds": {
                "slope": {
                    "strong": 5.0,
                    "moderate": 2.0
                }
            },
            "volume_analysis": {
                "rvol_surge_threshold": 2.0,
                "rvol_drought_threshold": 0.8
            },
            "time_estimation": {
                "candles_per_unit": 0.2
            },
            "timeframe": "1wk",
            "description": "Trend Following & Investing",
            
            # Override to WMA
            "moving_averages": {
                "type": "WMA",
                "fast": 10,
                "mid": 40,
                "slow": 50,
                "keys": ["wma_10", "wma_40", "wma_50"],
                "dip_buy_reference": "wma_40"
            },
            
            # Momentum thresholds
            "momentum_thresholds": {
                "rsi_slope": {
                    "acceleration_floor": 0.03,
                    "deceleration_ceiling": -0.03
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            
            # Volatility scoring - wider bands
            "volatility": {
                "scoring_thresholds": {
                    "atr_pct": {
                        "excellent": 5.5,
                        "good": 9.0,
                        "fair": 13.0,
                        "poor": 18.0
                    },
                    "bb_width": {
                        "tight": 4.0,
                        "normal": 8.0,
                        "wide": 15.0
                    }
                }
            },
            
            # Risk management
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
            
            # Execution
            "execution": {
                "stop_loss_atr_mult": 2.5,
                "target_atr_mult": 5.0,
                "max_hold_candles": 52,
                "dip_buy_reference": "wma_40",
                "risk_reward_min": 2.5,
                "base_hold_days": 60,
                "proximity_rejection": {
                    "resistance_mult": 1.01,
                    "support_mult": 0.99
                },
                "min_profit_pct": 1.0
            },
            
            "lookback": {"python_data": 800},
            
            # Scoring - 50/50 tech/fund
            "scoring": {
                "fundamental_weight": 0.5,
                "metrics": {
                    "ma_trend_signal": 0.15,
                    "price_vs_primary_trend_pct": 0.10,
                    "roe": 0.10,
                    "roce": 0.08,
                    "roic": 0.08,
                    "earnings_stability": 0.08,
                    "fcf_yield": 0.08,
                    "eps_growth_5y": 0.06,
                    "piotroski_f": 0.05,
                    "rel_strength_nifty": 0.04,
                    "ma_fast_slope": 0.05,
                    "promoter_holding": 0.05
                },
                "penalties": {
                    "price_vs_primary_trend_pct": {"op": "<", "val": 0, "pen": 0.5},
                    "roe": {"op": "<", "val": 10, "pen": 0.3},
                    "fcf_yield": {"op": "<", "val": 2, "pen": 0.3},
                    "promoter_pledge": {"op": ">", "val": 15.0, "pen": 0.2},
                    "ocf_vs_profit": {"op": "<", "val": 0.6, "pen": 0.5}
                },
                "thresholds": {"buy": 7.5, "hold": 6.0, "sell": 4.0},
                "metric_weights": {
                    "roe": 0.12,
                    "roce": 0.12,
                    "roic": 0.10,
                    "de_ratio": 0.05,
                    "interest_coverage": 0.05,
                    "eps_growth_5y": 0.12,
                    "revenue_growth_5y": 0.10,
                    "market_cap_cagr": 0.08,
                    "earnings_stability": 0.06,
                    "fcf_growth_3y": 0.05,
                    "promoter_holding": 0.05,
                    "institutional_ownership": 0.05,
                    "ocf_vs_profit": 0.05
                }
            },
            
            # Technical weights - boost slow indicators
            "technical_weight_overrides": {
                "trend_strength": 1.4,
                "adx": 1.3,
                "ma_trend_signal": 1.3,
                "price_vs_200dma_pct": 1.4,
                "price_vs_primary_trend_pct": 1.3,
                "dma_200_slope": 1.3,
                "ma_cross_signal": 1.2,
                "ichi_cloud": 1.2,
                "rel_strength_nifty": 1.2,
                "supertrend_signal": 1.1,
                "psar_trend": 1.1,
                "rsi": 0.8,
                "rsi_slope": 0.7,
                "stoch_k": 0.7,
                "macd_hist_z": 0.8,
                "vol_spike_signal": 0.6,
                "rvol": 0.7,
                "vol_spike_ratio": 0.6,
                "vwap_bias": 0.5,
                "gap_percent": 0.6,
                "wick_rejection": 0.7,
                "bb_percent_b": 0.7,
                "momentum_strength": 0.9,
                "volatility_quality": 1.1
            },
            
            # Gates # RR ratio is validated during trade plan construction, not gate phase
            # "gates": {
            #     "min_trend_strength": 3.0,
            #     "allowed_supertrend_counter": False,
            #     "volatility_bands_atr_pct": {
            #         "min": 1.0,
            #         "ideal": 5.5,
            #         "max": 15.0
            #     },
            #     "volatility_quality_mins": {
            #         "MOMENTUM_BREAKOUT": 4.0,
            #         "VOLATILITY_SQUEEZE": 6.0,
            #         "TREND_PULLBACK": 4.5,
            #         "default": 4
            #     },
            #     "confidence_min": 65,
            #     "adx_min": 20,
            #     "volatility_quality_min": 5.0,
            #     "trend_strength_min": 5.0,
            #     "rr_ratio_min": 1.5,
            #     "volatility_guards": {
            #         "extreme_vol_buffer": 3.0,
            #         "min_quality_breakout": 4.0,
            #         "min_quality_normal": 5.0
            #     },
            #     "setup_gate_overrides": {
            #         "QUALITY_ACCUMULATION": {
            #             "min_trend_strength": None,
            #             "adx_min": None,
            #             "volatility_quality_min": 3.0
            #         },
            #         "QUALITY_ACCUMULATION_DOWNTREND": {
            #             "min_trend_strength": None,
            #             "adx_min": None,
            #             "volatility_quality_min": 3.0
            #         },
            #         "DEEP_VALUE_PLAY": {
            #             "min_trend_strength": None,
            #             "adx_min": None,
            #             "volatility_quality_min": 2.5
            #         },
            #         "VALUE_TURNAROUND": {
            #             "min_trend_strength": 3.0,  # Only require weak trend
            #             "adx_min": 15
            #         }
            #     },
            # },
            "gates": {
                "adx_min": None,                          # Already relaxed
                "min_trend_strength": 3.0,
                "confidence_min": 55,
                "volatility_quality_min": None,           # Already skipped
                "volatility_bands_atr_pct": {
                    "min": 1.0,
                    "ideal": 5.5,
                    "max": 15.0
                },
                 "volatility_guards": {
                    "extreme_vol_buffer": 3.0,      # Weekly candles = very tolerant
                    "min_quality_breakout": 4.0,     # Still want quality for breakouts
                    "min_quality_normal": 5.0        # Higher bar for long-term
                },
                "setup_gate_overrides": {
                    # Minimal overrides - base gates already relaxed
                    "VALUE_TURNAROUND": { "adx_min": 8, "min_trend_strength": 3.0},  # ← Lower for value (was 15)   # Already good
                    "DEEP_VALUE_PLAY": { "adx_min": None,"min_trend_strength": None },  # ← No ADX requirement for deep value  # No trend requirement
                    "QUALITY_ACCUMULATION": {"confidence_min": 40},              # Position building over time
                    "DEEP_VALUE_PLAY": {"confidence_min": 35},              # Long-term = very patient
                    "MOMENTUM_BREAKOUT": { "min_trend_strength": 4.5},                           # Relax for long-term
                    "GENERIC": {"confidence_min": 35 }
                    # ADX/trend/vol already relaxed at horizon level, no overrides needed
                },
            },
            # Confidence
            "confidence": {
                "horizon_discount": 0,
                "floors": {"buy": 60, "wait": 40},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 60,
                    "MOMENTUM_BREAKDOWN": 60,
                    "TREND_PULLBACK": 58,
                    "DEEP_PULLBACK": 55,
                    "QUALITY_ACCUMULATION": 55,
                    "VOLATILITY_SQUEEZE": 55,
                    "TREND_FOLLOWING": 55,
                    "BEAR_TREND_FOLLOWING": 55,
                    "REVERSAL_MACD_CROSS_UP": 55,
                    "REVERSAL_RSI_SWING_UP": 55,
                    "REVERSAL_ST_FLIP_UP": 60
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -10,
                    "ignore_for_squeeze": True
                }
            },
            
            # Setup Confidence
            "setup_confidence": {
                "confidence_clamp": [40, 98],
                "penalties": {
                    "poor_fundamentals": {
                        "condition": "roe < 15 or roce < 15",
                        "amount": 15,
                        "reason": "Weak fundamentals for long-term hold"
                    },
                    "high_debt": {
                        "condition": "de_ratio > 1.0",
                        "amount": 12,
                        "reason": "High leverage risk"
                    },
                    "moderate_weak_trend": {
                        "condition": "trend_strength >= 4.5 and trend_strength < 5.5",
                        "amount": 10,
                        "reason": "Moderate trend weakness for long-term"
                    },
                    "very_weak_trend": {
                        "condition": "trend_strength < 4.5",
                        "amount": 20,
                        "reason": "Very weak trend - unstable for long-term hold"
                    }
                },
                "bonuses": {
                    "high_quality_compounder": {
                        "condition": "roe >= 20 and roce >= 25 and eps_growth_5y >= 15",
                        "amount": 15,
                        "reason": "High-quality long-term compounder"
                    },
                    "stable_growth": {
                        "condition": "earnings_stability >= 7.0 and revenue_growth_5y >= 10",
                        "amount": 10,
                        "reason": "Stable earnings and revenue growth"
                    },
                    "exceptional_trend": {
                        "condition": "trend_strength >= 8.0",
                        "amount": 25,
                        "reason": "Exceptional sustained trend"
                    },
                    "strong_trend": {
                        "condition": "trend_strength >= 6.5",
                        "amount": 20,
                        "reason": "Strong sustained trend"
                    }
                }
            },
            
            # Enhancements
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
            }
        },
        
        # ========================================================================
        # MULTIBAGGER (Monthly candles, deep value & compounders)
        # ========================================================================
        "multibagger": {
            "trend_thresholds": {
                "slope": {
                    "strong": 30.0,
                    "moderate": 10.0
                }
            },
            "volume_analysis": {
                "rvol_surge_threshold": 1.8,
                "rvol_drought_threshold": 0.8
            },
            "time_estimation": {
                "candles_per_unit": 0.05
            },
            "timeframe": "1mo",
            "description": "Deep Value & Compounders",
            
            # Override to MMA
            "moving_averages": {
                "type": "MMA",
                "fast": 6,
                "mid": 12,
                "slow": 12,
                "keys": ["mma_6", "mma_12", "mma_12"],
                "dip_buy_reference": "mma_12"
            },
            
            # Indicators
            "indicators": {
                "rsi_period": 14,
                "adx_period": 14,
                "atr_period": 20,
                "supertrend": {"period": 10, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "classic",
                "stochastic": {
                    "k": 21,
                    "d": 5,
                    "smooth": 5,
                    "overbought": 85,
                    "oversold": 15
                }
            },
            
            # Momentum thresholds
            "momentum_thresholds": {
                "rsi_slope": {
                    "acceleration_floor": 0.02,
                    "deceleration_ceiling": -0.02
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            
            # Volatility scoring - very wide bands
            "volatility": {
                "scoring_thresholds": {
                    "atr_pct": {
                        "excellent": 11.5,
                        "good": 18.0,
                        "fair": 27.0,
                        "poor": 36.0
                    },
                    "bb_width": {
                        "tight": 6.0,
                        "normal": 12.0,
                        "wide": 20.0
                    }
                }
            },
            
            # Risk management
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
            
            # Execution
            "execution": {
                "stop_loss_atr_mult": 3.0,
                "target_atr_mult": 10.0,
                "max_hold_candles": 60,
                "dip_buy_reference": "mma_12",
                "risk_reward_min": 3.0,
                "base_hold_days": 180,
                "proximity_rejection": {
                    "resistance_mult": 1.02,
                    "support_mult": 0.98
                },
                "min_profit_pct": 2.0
            },
            
            "lookback": {"python_data": 3000},
            
            # Technical weights - heavily favor slow indicators
            "technical_weight_overrides": {
                "trend_strength": 1.5,
                "adx": 1.5,
                "ma_trend_signal": 1.6,
                "price_vs_200dma_pct": 1.6,
                "price_vs_primary_trend_pct": 1.5,
                "dma_200_slope": 1.5,
                "ma_cross_signal": 1.4,
                "ichi_cloud": 1.3,
                "rel_strength_nifty": 1.4,
                "nifty_trend_score": 1.2,
                "rsi": 0.5,
                "rsi_slope": 0.4,
                "stoch_k": 0.4,
                "macd_cross": 0.6,
                "macd_hist_z": 0.5,
                "vol_spike_signal": 0.4,
                "rvol": 0.5,
                "vol_spike_ratio": 0.4,
                "vol_trend": 0.5,
                "vwap_bias": 0.3,
                "gap_percent": 0.4,
                "price_action": 0.5,
                "wick_rejection": 0.5,
                "bb_percent_b": 0.5,
                "ttm_squeeze": 0.6,
                "momentum_strength": 0.7,
                "volatility_quality": 1.2
            },
            
            # Scoring - 70% fundamental
            "scoring": {
                "fundamental_weight": 0.7,
                "metrics": {
                    "eps_growth_5y": 0.10,
                    "revenue_growth_5y": 0.10,
                    "quarterly_growth": 0.05,
                    "market_cap_cagr": 0.08,
                    "roic": 0.10,
                    "roe": 0.08,
                    "peg_ratio": 0.08,
                    "r_d_intensity": 0.05,
                    "ocf_vs_profit": 0.06,
                    "rel_strength_nifty": 0.05,
                    "institutional_ownership": 0.03,
                    "promoter_holding": 0.05
                },
                "penalties": {
                    "peg_ratio": {"op": ">", "val": 3.0, "pen": 0.3},
                    "market_cap": {"op": ">", "val": 1000000000000, "pen": 0.5},
                    "de_ratio": {"op": ">", "val": 1.0, "pen": 0.2},
                    "roe": {"op": "<", "val": 12, "pen": 0.2},
                    "institutional_ownership": {"op": ">", "val": 85, "pen": 0.3},
                    "promoter_pledge": {"op": ">", "val": 10.0, "pen": 0.4}
                },
                "thresholds": {"buy": 8.0, "hold": 6.5, "sell": 4.5},
                "metric_weights": {
                    "eps_growth_5y": 0.15,
                    "revenue_growth_5y": 0.15,
                    "market_cap_cagr": 0.10,
                    "quarterly_growth": 0.05,
                    "roic": 0.10,
                    "roe": 0.10,
                    "roce": 0.08,
                    "de_ratio": 0.05,
                    "ocf_vs_profit": 0.05,
                    "earnings_stability": 0.05,
                    "promoter_holding": 0.04,
                    "institutional_ownership": 0.04,
                    "r_d_intensity": 0.04
                }
            },
            
            # Gates - focus on quality
            # "gates": {
            #     "min_trend_strength": None,
            #     "allowed_supertrend_counter": False,
            #     "volatility_bands_atr_pct": {
            #         "min": 1.5,
            #         "ideal": 8.0,
            #         "max": 25.0
            #     },
            #     "volatility_quality_mins": {
            #         "MOMENTUM_BREAKOUT": 5.0,
            #         "VOLATILITY_SQUEEZE": 7.0,
            #         "TREND_PULLBACK": 5.0,
            #         "default": 4.5
            #     },
            #     "confidence_min": 70,
            #     "adx_min": 25,
            #     "volatility_quality_min": 6.0,
            #     "trend_strength_min": 6.0,
            #     "rr_ratio_min": 1.5,
            #     "volatility_guards": {
            #         "extreme_vol_buffer": 4.0,
            #         "min_quality_breakout": 5.0,
            #         "min_quality_normal": 6.0
            #     },
            #     "setup_gate_overrides": {
            #         "QUALITY_ACCUMULATION": {
            #             "volatility_quality_min": 2.5
            #         },
            #         "QUALITY_ACCUMULATION_DOWNTREND": {
            #             "volatility_quality_min": 2.5
            #         },
            #         "DEEP_VALUE_PLAY": {
            #             "volatility_quality_min": 2.0
            #         }
            #     },
            # },
            "gates": {
                "adx_min": None,                    # ❌ ADX is meaningless early for multibaggers
                "min_trend_strength": 3.0,          # Weak trend is acceptable
                "confidence_min": 65,               # High conviction required
                "volatility_quality_min": None,     # ❌ Volatility not a blocker for investing
                "volatility_bands_atr_pct": {
                    "min": 1.5,
                    "ideal": 8.0,
                    "max": 25.0
                },
                "volatility_guards": {
                    "extreme_vol_buffer": 4.0,      # Monthly candles = maximum tolerance
                    "min_quality_breakout": 5.0,
                    "min_quality_normal": 6.0
                },
                "setup_gate_overrides": {
                    "QUALITY_ACCUMULATION": {"confidence_min": 60},
                    "QUALITY_ACCUMULATION_DOWNTREND": {"confidence_min": 55},
                    "DEEP_VALUE_PLAY": {"confidence_min": 50},
                    "VALUE_TURNAROUND": {"min_trend_strength": 2.5, "confidence_min": 55}, #  # Allow very early turn
                    "TREND_PULLBACK": {"min_trend_strength": 4.0,"confidence_min": 65},
                    "TREND_FOLLOWING": {"min_trend_strength": 4.5,"confidence_min": 68},
                    "MOMENTUM_BREAKOUT": {"min_trend_strength": 4.0,"confidence_min": 70},
                    "PATTERN_CUP_BREAKOUT": {"min_trend_strength": 3.5,"confidence_min": 65},
                    "PATTERN_GOLDEN_CROSS": {"min_trend_strength": 4.0,"confidence_min": 68},
                    "REVERSAL_MACD_CROSS_UP": {"min_trend_strength": 3.0,"confidence_min": 60},
                    "REVERSAL_RSI_SWING_UP": {"min_trend_strength": 3.0,"confidence_min": 60},
                    "GENERIC": {"confidence_min": 55 }
                }
            },

            # Confidence
            "confidence": {
                "horizon_discount": 0,
                "floors": {"buy": 65, "wait": 50},
                "base_floors": {
                    "MOMENTUM_BREAKOUT": 65,
                    "MOMENTUM_BREAKDOWN": 65,
                    "TREND_PULLBACK": 60,
                    "DEEP_PULLBACK": 58,
                    "QUALITY_ACCUMULATION": 60,
                    "VOLATILITY_SQUEEZE": 60,
                    "TREND_FOLLOWING": 60,
                    "BEAR_TREND_FOLLOWING": 60,
                    "REVERSAL_MACD_CROSS_UP": 60,
                    "REVERSAL_RSI_SWING_UP": 60,
                    "REVERSAL_ST_FLIP_UP": 65
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -5,
                    "ignore_for_squeeze": True
                }
            },
            
            # Setup Confidence
            "setup_confidence": {
                "confidence_clamp": [45, 99],
                "penalties": {
                    "insufficient_growth": {
                        "condition": "eps_growth_5y < 15 or revenue_growth_5y < 15",
                        "amount": 20,
                        "reason": "Growth too low for multibagger profile"
                    },
                    "weak_quality": {
                        "condition": "roe < 15 or roce < 18",
                        "amount": 30,
                        "reason": "Insufficient quality for multibagger thesis"
                    },
                    "high_leverage": {
                        "condition": "de_ratio > 1.2",
                        "amount": 20,
                        "reason": "High debt risk for long-term hold"
                    },
                    "bearish_trend": {
                        "condition": "trend_strength < 3.0 and momentum_strength < 3.0",
                        "amount": 25,
                        "reason": "Bearish trend - wait for reversal"
                    }
                },
                "bonuses": {
                    "exceptional_quality_growth": {
                        "condition": "roe >= 25 and roce >= 30 and eps_growth_5y >= 20 and revenue_growth_5y >= 20",
                        "amount": 20,
                        "reason": "Exceptional quality + growth combo"
                    },
                    "early_stage_leader": {
                        "condition": "rel_strength_nifty >= 1.2 and market_cap_cagr >= 25",
                        "amount": 15,
                        "reason": "Outperforming market with strong cap growth"
                    },
                    "mega_trend": {
                        "condition": "trend_strength >= 8.5 and roe >= 25",
                        "amount": 30,
                        "reason": "Mega trend with quality fundamentals"
                    },
                    "strong_trend": {
                        "condition": "trend_strength >= 7.0 and trend_strength < 8.5",
                        "amount": 25,
                        "reason": "Strong multi-year trend"
                    },
                    "quality_emerging_trend": {
                        "condition": "trend_strength >= 5.0 and trend_strength < 7.0 and roe >= 25 and eps_growth_5y >= 15",
                        "amount": 20,
                        "reason": "Quality company with emerging trend - ideal multibagger entry"
                    }
                }
            },
            
            # Enhancements
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
            }
        }
    }
}