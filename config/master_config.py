MASTER_CONFIG = {
    # GLOBAL CONSTANTS (Universal Physics & Logic)
    "global": {
        "fundamental_weights": {
            # ============================================================
            # USAGE: These weights are used by signal_engine.py in compute_profile_score() to calculate fundamental scores.
            # Each horizon can override the "mix" (how much fundamental vs technical matters) but uses these base weights.
            # ============================================================
            "value": {
                # Lower is better (P/E, P/B, PEG)
                "pe_ratio": {"weight": 0.05,"direction": "invert","ideal_range": [10, 20],"penalty_threshold": 50},
                "pb_ratio": {"weight": 0.04,"direction": "invert","ideal_range": [1, 3],"penalty_threshold": 5},
                "peg_ratio": {"weight": 0.03,"direction": "invert","ideal_range": [0.5, 1.5],"penalty_threshold": 3},
                "ps_ratio": {"weight": 0.02,"direction": "invert","ideal_range": [1, 3]},
                # Higher is better (FCF Yield, Dividend Yield)
                "fcf_yield": {"weight": 0.05,"direction": "normal","min_threshold": 3.0},
                "dividend_yield": {"weight": 0.03,"direction": "normal","min_threshold": 2.0},
                # Valuation vs Sector
                "pe_vs_sector": {"weight": 0.03,"direction": "invert" }
            },
            "growth": {
                # All growth metrics: Higher is better
                "eps_growth_5y": {"weight": 0.06,"direction": "normal","min_threshold": 15,"ideal_range": [20, 40]},
                "eps_growth_3y": {"weight": 0.05,"direction": "normal","min_threshold": 15},
                "revenue_growth_5y": {"weight": 0.05,"direction": "normal","min_threshold": 10},
                "profit_growth_3y": {"weight": 0.04,"direction": "normal","min_threshold": 15},
                "fcf_growth_3y": {"weight": 0.05,"direction": "normal","min_threshold": 10},
                "market_cap_cagr": {"weight": 0.04,"direction": "normal","min_threshold": 15},
                "quarterly_growth": {"weight": 0.03,"direction": "normal","min_threshold": 20 }
            },
            "quality": {
                # Profitability & Returns (Higher is better)
                "roe": {"weight": 0.10,"direction": "normal","min_threshold": 15,"ideal_range": [20, 40]},
                "roce": {"weight": 0.07,"direction": "normal","min_threshold": 15,"ideal_range": [20, 35]},
                "roic": {"weight": 0.08,"direction": "normal","min_threshold": 12,"ideal_range": [18, 35]},
                "net_profit_margin": {"weight": 0.04,"direction": "normal","min_threshold": 10},
                "operating_margin": {"weight": 0.03,"direction": "normal","min_threshold": 15},
                "ebitda_margin": {"weight": 0.03,"direction": "normal"},
                "fcf_margin": {"weight": 0.04,"direction": "normal","min_threshold": 8},
                # Leverage & Liquidity (Lower is better for debt)
                "de_ratio": {"weight": 0.05,"direction": "invert","max_threshold": 1.0,"ideal_range": [0, 0.5]},
                "interest_coverage": {"weight": 0.05,"direction": "normal","min_threshold": 3.0,"ideal_range": [5, 20]},
                "current_ratio": {"weight": 0.03,"direction": "normal","min_threshold": 1.5},
                "ocf_vs_profit": {"weight": 0.02,"direction": "normal","min_threshold": 0.8},
                # Efficiency & Quality Scores
                "piotroski_f": {"weight": 0.07,"direction": "normal","min_threshold": 7},
                "asset_turnover": {"weight": 0.04,"direction": "normal","min_threshold": 0.5},
                "r_d_intensity": {"weight": 0.04,"direction": "normal"},
                "earnings_stability": {"weight": 0.05,"direction": "normal","min_threshold": 7.0 }
            },
            "momentum": {
                # Technical-Fundamental Hybrid Metrics
                "momentum_strength": {"weight": 0.30,"direction": "normal","min_threshold": 6.0},
                "trend_strength": {"weight": 0.40,"direction": "normal","min_threshold": 5.0},
                "volatility_quality": {"weight": 0.10,"direction": "normal","min_threshold": 4.0},
                
                # Market Position
                "52w_position": {"weight": 0.01,"direction": "normal","ideal_range": [80, 95]},
                "rel_strength_nifty": {"weight": 0.04,"direction": "normal","min_threshold": 0},
                
                # Ownership Momentum
                "promoter_holding": {"weight": 0.015,"direction": "normal","ideal_range": [50, 75]},
                "institutional_ownership": {"weight": 0.015,"direction": "normal","ideal_range": [20, 50]},
                
                # Market Metrics
                "beta": {"weight": 0.01,"direction": "normal","ideal_range": [0.8, 1.2]},
                
                # Valuation Hybrid
                "dividend_payout": {"weight": 0.03,"direction": "normal"},
                "yield_vs_avg": {"weight": 0.02,"direction": "normal" }
            },
            # ============================================================
            # METADATA: How to use these weights
            # ============================================================
            "_usage_notes": {
                "normalization": "All weights within a category should sum to ~1.0",
                "direction": {"normal": "Higher value = Higher score (e.g., ROE)", "invert": "Lower value = Higher score (e.g., P/E)"},
                "horizon_mixing": "Use 'fundamental_weight' in each horizon's scoring section to control tech/fund mix"
            }
        },
        "technical_weights": {
            # ============================================================
            # USAGE: These weights are used by indicators.py in  compute_technical_score() to calculate weighted technical scores.
            # Each horizon can override specific weights in their "technical_weight_overrides" section.

            # ============================================================
            
            # ===== MOMENTUM INDICATORS =====
            "rsi": {"weight": 1.0,"category": "momentum","speed": "fast"},
            "rsi_slope": {"weight": 0.8,"category": "momentum","speed": "fast"},
            "macd_cross": {"weight": 1.0,"category": "momentum","speed": "medium"},
            "macd_hist_z": {"weight": 0.8,"category": "momentum","speed": "medium"},
            "macd_histogram": {"weight": 0.6,"category": "momentum","speed": "medium"},
            "stoch_k": {"weight": 0.6,"category": "momentum","speed": "fast"},
            "stoch_cross": {"weight": 0.8,"category": "momentum","speed": "fast"},
            "mfi": {"weight": 0.7,"category": "momentum","speed": "fast"},
            "cci": {"weight": 0.6,"category": "momentum","speed": "fast"},
            
            # ===== TREND INDICATORS =====
            "adx": {"weight": 1.0,"category": "trend","speed": "slow"},
            "adx_signal": {"weight": 0.6,"category": "trend","speed": "slow"},
            "ma_cross_signal": {"weight": 0.8,"category": "trend","speed": "medium"},
            "ma_trend_signal": {"weight": 1.0,"category": "trend","speed": "slow"},
            "ma_fast_slope": {"weight": 0.8,"category": "trend","speed": "medium"},
            "ema_20_50_cross": {"weight": 0.8,"category": "trend","speed": "medium"},
            "price_vs_primary_trend_pct": {"weight": 1.0,"category": "trend","speed": "slow"},
            "price_vs_200dma_pct": {"weight": 1.0,"category": "trend","speed": "slow"},
            "dma_200_slope": {"weight": 0.8,"category": "trend","speed": "slow"},
            "supertrend_signal": {"weight": 1.0,"category": "trend","speed": "medium"},
            "psar_trend": {"weight": 0.8,"category": "trend","speed": "medium"},
            "ichi_cloud": {"weight": 1.0,"category": "trend","speed": "slow"},
            
            # ===== VOLATILITY INDICATORS =====
            "atr_14": {"weight": 0.8,"category": "volatility","speed": "medium"},
            "atr_pct": {"weight": 0.6,"category": "volatility","speed": "medium"},
            "bb_width": {"weight": 0.3,"category": "volatility","speed": "medium"},
            "bb_percent_b": {"weight": 0.4,"category": "volatility","speed": "fast"},
            "bb_low": {"weight": 0.4,"category": "volatility","speed": "fast"},
            "ttm_squeeze": {"weight": 1.0,"category": "volatility","speed": "medium"},
            "true_range": {"weight": 0.5,"category": "volatility","speed": "fast"},
            "hv_10": {"weight": 0.6,"category": "volatility","speed": "fast"},
            "hv_20": {"weight": 0.6,"category": "volatility","speed": "medium"},
            
            # ===== VOLUME INDICATORS =====
            "rvol": {"weight": 0.6,"category": "volume","speed": "fast"},
            "vol_spike_ratio": {"weight": 0.5,"category": "volume","speed": "fast"},
            "vol_spike_signal": {"weight": 1.0,"category": "volume","speed": "fast"},
            "vol_trend": {"weight": 0.6,"category": "volume","speed": "medium"},
            "obv_div": {"weight": 0.6,"category": "volume","speed": "medium"},
            "vpt": {"weight": 0.5,"category": "volume","speed": "medium"},
            "cmf_signal": {"weight": 0.6,"category": "volume","speed": "medium"},
            
            # ===== SUPPORT/RESISTANCE =====
            "entry_confirm": {"weight": 0.5,"category": "level","speed": "static"},
            "pivot_point": {"weight": 0.4,"category": "level","speed": "static"},
            "donchian_signal": {"weight": 0.8,"category": "level","speed": "medium"},
            
            # ===== PATTERN SIGNALS =====
            "wick_rejection": {"weight": 0.7,"category": "pattern","speed": "fast"},
            
            # ===== COMPOSITE SCORES =====
            "trend_strength": {"weight": 1.5,"category": "composite","speed": "medium"},
            "momentum_strength": {"weight": 1.5,"category": "composite","speed": "medium"},
            "volatility_quality": {"weight": 1.0,"category": "composite","speed": "medium"},
            
            # ===== RELATIVE METRICS =====
            "rel_strength_nifty": {"weight": 0.6,"category": "relative","speed": "slow"},
            "nifty_trend_score": {"weight": 0.5,"category": "relative","speed": "slow"},
            
            # ===== PRICE ACTION =====
            "vwap_bias": {"weight": 0.8,"category": "price_action","speed": "fast"},
            "price_action": {"weight": 0.7,"category": "price_action","speed": "fast"},
            "gap_percent": {"weight": 0.5,"category": "price_action","speed": "fast"},
            "reg_slope": {"weight": 0.8,"category": "price_action","speed": "medium"},
            
            # ============================================================
            # METADATA: Categorization for smart overrides
            # ============================================================
            "_categories": {
                "fast": ["rsi", "rsi_slope", "stoch_k", "vol_spike_signal", "rvol", "bb_percent_b", "vwap_bias", "price_action"],
                "medium": ["macd_cross", "adx", "supertrend_signal", "atr_14", "ttm_squeeze"],
                "slow": ["price_vs_200dma_pct", "ma_trend_signal", "ichi_cloud", "rel_strength_nifty"]
            },
            "_usage_notes": {
                "normalization": "Weights are relative within categories. Total score is normalized to 0-100.",
                "horizon_overrides": "Fast horizons boost 'fast' indicators, slow horizons boost 'slow' indicators",
                "metadata": "category/speed fields are for documentation and smart filtering, not used in calculations"
            }

        },

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
        # SECTION 2: PATTERN ENTRY RULES
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
        # SECTION 3: PATTERN INVALIDATION
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
                    "intraday": {
                        "condition": "price < box_low * 0.998",
                        "duration_candles": 1
                    },
                    "short_term": {
                        "condition": "price < box_low * 0.995",
                        "duration_candles": 1
                    },
                    "long_term": {
                        "condition": "price < box_low * 0.99",
                        "duration_candles": 2
                    }
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
                        "duration_candles": 1
                    }
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
                    "intraday": {
                        "condition": "price < pivot * 0.98",
                        "duration_candles": 2
                    },
                    "short_term": {
                        "condition": "price < pivot * 0.95",
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < pivot * 0.92",
                        "duration_candles": 3
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
            "cup_handle": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < handle_low * 0.99",
                        "duration_candles": 2
                    },
                    "short_term": {
                        "condition": "price < handle_low * 0.97",
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < handle_low * 0.95",
                        "duration_candles": 3
                    }
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
                    "intraday": {
                        "condition": "price < cloud_bottom * 0.998",
                        "duration_candles": 2
                    },
                    "short_term": {
                        "condition": "price < cloud_bottom * 0.995",
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < cloud_bottom",
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
                    "intraday": "EXIT_IMMEDIATELY",
                    "short_term": "EXIT_ON_CLOSE",
                    "long_term": "EXIT_ON_CLOSE"
                }
            },
            "double_top_bottom": {
                "breakdown_threshold": {
                    "intraday": {
                        "condition": "price < neckline * 0.998",
                        "duration_candles": 1
                    },
                    "short_term": {
                        "condition": "price < neckline * 0.995",
                        "duration_candles": 2
                    },
                    "long_term": {
                        "condition": "price < neckline * 0.99",
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
            }
        },
        # SECTION 4: TIME ESTIMATION
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
        # SECTION 5: ENTRY GATES
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
        # SECTION 6: CALCULATION ENGINE
        "calculation_engine": {
            "composite_weights": {
                "trend_strength": {
                    "adx": {
                        "weight": 0.4,
                        "thresholds": {"strong": 25,"moderate": 20,"weak": 15},
                        "scoring": [{"min": 25, "score": 10},{"min": 20, "score": 8},{"min": 15, "score": 4},{"default": 2} ]
                    },
                    "ema_slope": {
                        "weight": 0.3,
                        "thresholds": {"strong": 20.0,"moderate": 5.0},
                        "scoring": [{"min": 20, "score": 10},{"min": 5, "score": 7},{"default": 2} ]
                    },
                    "di_spread": {
                        "weight": 0.2,
                        "scoring": [{"min": 15, "score": 10},{"min": 10, "score": 7},{"default": 5} ]
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
                        "thresholds": {"overbought": 70,"strong": 60,"neutral": 50,"weak": 40},
                        "scoring": [{"min": 70, "score": 8},{"min": 60, "score": 7},{"min": 50, "score": 5},{"default": 2} ]
                    },
                    "rsi_slope": {
                        "weight": 0.25,
                        "thresholds": {"strong": 1.0,"neutral": 0.0},
                        "scoring": [{"min": 1.0, "score": 8},{"min": 0, "score": 4},{"default": 2} ]
                    },
                    "macd_hist": {
                        "weight": 0.3,
                        "thresholds": {"strong": 0.5,"neutral": 0.0},
                        "scoring": [{"min": 0.5, "score": 8},{"min": 0, "score": 5},{"default": 2} ]
                    },
                    "stoch_cross": {
                        "weight": 0.2,
                        "thresholds": {"overbought": 80,"neutral": 50},
                        "scoring": {"bullish_strong": {    "condition": "k>d and k>=50",    "score": 8},"default": 3 }
                    }
                },
                "volatility_quality": {
                    "atr_pct": {
                        "weight": 0.3,
                        "thresholds": {"low": 1.5,"moderate": 3.0,"high": 5.0},
                        "scoring": [{"max": 1.5, "score": 10},{"max": 3.0, "score": 8},{"max": 5.0, "score": 6},{"default": 2} ]
                    },
                    "bb_width": {
                        "weight": 0.25,
                        "thresholds": {"tight": 0.01,"moderate": 0.02,"wide": 0.04},
                        "scoring": [{"max": 0.01, "score": 10},{"max": 0.02, "score": 8},{"default": 2} ]
                    },
                    "true_range_consistency": {
                        "weight": 0.20,
                        "scoring": [{"max": 0.5, "score": 10},{"default": 4} ]
                    },
                    "hv_trend": {
                        "weight": 0.15,
                        "scoring": {"declining": {    "condition": "hv_10 < hv_20",    "score": 8},"default": 4 }
                    },
                    "atr_sma_ratio": {
                        "weight": 0.15,
                        "thresholds": {"stable": 0.02,"moderate": 0.035,"volatile": 0.05},
                        "scoring": [{"max": 0.02, "score": 10},{"max": 0.035, "score": 7},{"default": 3} ]
                    }
                }
            },
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
                "breakout": {
                    "bb_percent_b_min": 0.98,      
                    "rsi_min": 60,
                    "wick_ratio_max": 2.5,
                    "rvol_min": 1.5
                },
                "breakdown": {
                    "bb_percentb_max": 0.02,
                    "rsi_max": 40,
                    "rvol_min": 1.5
                },
                "pullback": {
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
                "accumulation": {
                    "consolidation_required": True,
                    "rsi_range": [40, 60]
                },
                "quality_accumulation_downtrend": {
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
                        "conditions": ["macd_histogram > 0","prev_macd_histogram < 0"],
                        "priority": 80
                    },
                    "REVERSAL_RSI_SWING_UP": {
                        "conditions": ["rsi < 30","rsi_slope > 0.05"],  # Uses momentum_thresholds.rsi_slope.acceleration_floor
                        "priority": 75
                    },
                    "REVERSAL_ST_FLIP_UP": {
                        "conditions": ["supertrend_signal == 'Bullish'","prev_supertrend_signal == 'Bearish'"],
                        "priority": 70
                    },
                    "SELL_AT_RANGE_TOP": {
                        "conditions": ["is_consolidating == True","price >= bb_high * 0.98","rsi > 60"],
                        "priority": 75
                    },
                    "TAKE_PROFIT_AT_MID": {
                        "conditions": ["is_consolidating == True","price >= bb_mid * 0.98","price < bb_high * 0.98","rsi > 55"],
                        "priority": 60
                    },
                },

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
            "setup_confidence": {
                "confidence_clamp": [0, 100],
                "penalties": {
                    "weak_trend": {
                        "condition": "adx < 25",
                        "amount": 10,
                        "reason": "Weak trend (ADX below threshold)"
                    },
                    "breakout_weak_volume": {
                        "condition": "setup_type == 'MOMENTUM_BREAKOUT' and rvol < 1.5",
                        "amount": 10,
                        "reason": "Breakout with weak volume"
                    }
                },
                "bonuses": {
                    "very_strong_trend": {
                        "condition": "adx >= 35",
                        "amount": 5,
                        "reason": "Very strong trend"
                    }
                }
            }
        },
        # SECTION 7: POSITION SIZING
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
        # SECTION 8: WEIGHTS
        "trend_weights": {
            "primary": 0.50,
            "secondary": 0.30,
            "acceleration": 0.20
        },
        # SECTION 9: STRATEGY CONFIG
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
        # SECTION 10: BOOSTS
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
        # SECTION 11: SYSTEM/INFRASTRUCTURE
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
        # SECTION 12: MOVING AVERAGES (DEFAULT)
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
                "k": 14,  # FLAT STRUCTURE
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
            },
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
            "volatility_bands_atr_pct": [1.0, 12.0],
            "volatility_quality_mins": {"default": 3.0}
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
                "REVERSAL_ST_FLIP_UP": 55
            },
            "adx_normalization": {
                "min": 10,
                "max": 40,
                "adjustment_factor": 12
            },
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
        },
    },
    #horizon Specific configs using smart inheritance overrides 
    "horizons": {
        "intraday": {
            "volume_analysis": {
                "rvol_surge_threshold": 3.0,
                "rvol_drought_threshold": 0.7
            },
            "time_estimation": {
                "candles_per_unit": 4  # 15min candles: 4 per hour
            },
            "timeframe": "15m",
            "description": "Quick scalps and day trades",
            "moving_averages": {
                "type": "EMA",
                "fast": 20,
                "mid": 50,
                "slow": 200,
                "keys": ["ema_20", "ema_50", "ema_200"],
                "dip_buy_reference": "ema_20"  # Changed from ema_50
            },
            "indicators": {
                "rsi_period": 9,  # Changed from 14
                "adx_period": 10,      # CHANGE from 14 - Fast trend detection
                "atr_period": 10,      # CHANGE from 14 - Tighter stops
                "supertrend": {"period": 7, "multiplier": 3},  # Changed from 10
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "fibonacci",  # Changed from classic
                "stochastic": {
                    "k": 8,           # Faster for intraday
                    "d": 3,
                    "smooth": 3,
                    "overbought": 80,
                    "oversold": 20
                }
            },
            "momentum_thresholds": {
                "rsi_slope": {
                    "acceleration_floor": 0.10,  # Changed from 0.05
                    "deceleration_ceiling": -0.10  # Changed from -0.05
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            "volatility": {
                "scoring_thresholds": {
                    "atr_pct": {
                        "excellent": 1.5,  # Changed from 2.5
                        "good": 3.0,
                        "fair": 4.0,  # Changed from 4.5
                        "poor": 5.0  # Changed from 5.5
                    },
                    "bb_width": {
                        "tight": 2.0,  # Changed from 3.0
                        "normal": 5.0,  # Changed from 6.0
                        "wide": 10.0  # Changed from 12.0
                    }
                }
            },
            "risk_management": {
                "max_position_pct": 0.01,  # Changed from 0.02
                "setup_size_multipliers": {
                    "VOLATILITY_SQUEEZE": 1.3,
                    "MOMENTUM_BREAKOUT": 0.8
                },
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "min_rr_ratio": 1.2,  # Changed from 1.5
                "horizon_t2_cap": 0.04,  # Changed from 0.10
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},  # adx_min changed
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 1.5,  # Changed from 2.0
                "target_atr_mult": 2.5,  # Changed from 3.0
                "max_hold_candles": 25,  # Changed from 20
                "risk_reward_min": 1.5,  # Changed from 2.0
                "base_hold_days": 1,
                "proximity_rejection": {  # NEW
                    "resistance_mult": 1.003,
                    "support_mult": 0.997
                },
                "min_profit_pct": 0.3  # NEW
            },
            "lookback": {"python_data": 500},  # Changed from 600
            "scoring": {
                "fundamental_weight": 0.0,  # No fundamental weight for intraday
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
                    "gap_percent": {"op": "<", "val": 0.1, "pen": 0.1},
                    "atr_pct": {"op": "<", "val": 0.4, "pen": 0.3},
                    "ma_fast_slope": {"op": "<", "val": -2, "pen": 0.3}
                },
                "thresholds": {"buy": 6.5, "hold": 5.0, "sell": 3.8},
                "metric_weights": {
                    "momentum_strength":      0.30,
                    "trend_strength":         0.25,
                    "volatility_quality":     0.15,
                    "vwap_bias":              0.10,
                    "rvol":                   0.10,
                    "price_action":           0.10
                },

            },
            "technical_weight_overrides": {
                    # ----- BOOST FAST INDICATORS -----
                    "rsi": 1.3,                      # Fast momentum
                    "rsi_slope": 1.2,
                    "stoch_k": 1.2,
                    "macd_cross": 1.2,
                    "vol_spike_signal": 1.5,         # Volume critical
                    "rvol": 1.3,
                    "vol_spike_ratio": 1.3,
                    "vwap_bias": 1.4,                # Intraday anchor
                    "price_action": 1.3,
                    "gap_percent": 1.2,
                    "bb_percent_b": 1.2,             # Fast volatility
                    "wick_rejection": 1.2,
                    "momentum_strength": 1.3,        # Fast composite
                    # ----- REDUCE SLOW INDICATORS -----
                    "price_vs_200dma_pct": 0.4,      # Long-term trend less relevant
                    "price_vs_primary_trend_pct": 0.5,
                    "ma_trend_signal": 0.6,
                    "adx": 0.7,                      # Slower trend confirmation
                    "dma_200_slope": 0.4,
                    "ichi_cloud": 0.5,
                    "rel_strength_nifty": 0.5,
                    "nifty_trend_score": 0.4,
                    
                    "trend_strength": 0.8,           # Trend less important
                    "volatility_quality": 1.1        # Volatility more important
                },
            "gates": {
                "min_trend_strength": 2.0,  # Changed from 3.0
                "allowed_supertrend_counter": True,  # Changed from False
                "volatility_bands_atr_pct": {  # CHANGE from array to object
                    "min": 0.3,
                    "ideal": 3.0,  # ADD this field
                    "max": 5.0
                },
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 2.5,
                    "VOLATILITY_SQUEEZE": 4.0,
                    "TREND_PULLBACK": 3.0,
                    "default": 2.5
                },
                "volatility_guards": {  # NEW
                    "extreme_vol_buffer": 2.0,
                    "min_quality_breakout": 2.5,
                    "min_quality_normal": 4.0
                },
                "confidence_min": 55,
                "adx_min": 15,
                "volatility_quality_min": 3.0,
                "rr_ratio_min": 1.1,
                "trend_strength_min": 2.0
            },
            "trend_thresholds": {  # NEW section
                "slope": {
                    "strong": 15.0,
                    "moderate": 5.0
                }
            },
            "confidence": {
                "horizon_discount": 10,  # Changed from 5
                "floors": {"buy": 55, "wait": 30},  # wait changed from 35
                "base_floors": {  # ADD ALL OF THESE
                    "MOMENTUM_BREAKOUT": 50,
                    "MOMENTUM_BREAKDOWN": 50,
                    "TREND_PULLBACK": 48,
                    "DEEP_PULLBACK": 45,
                    "QUALITY_ACCUMULATION": 40,
                    "VOLATILITY_SQUEEZE": 50,
                    "TREND_FOLLOWING": 48,
                    "BEAR_TREND_FOLLOWING": 48,
                    "REVERSAL_MACD_CROSSUP": 45,
                    "REVERSAL_RSI_SWINGUP": 45,
                    "REVERSAL_ST_FLIPUP": 50,
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -20,  # Changed from -15
                    "ignore_for_squeeze": True
                },
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 60},
                    "moderate": {"adx_min": 25, "floor": 50},
                    "weak": {"adx_min": 20, "floor": 40},
                    "range_bound": {"adx_max": 20, "floor": 30}
                },
                "setup_type_overrides": {
                    "MOMENTUM_BREAKOUT": 5,
                    "VOLATILITY_SQUEEZE": 10,
                    "QUALITY_ACCUMULATION": -5  # Changed from 0
                }
            },
            "setup_confidence": {
                "confidence_clamp": [30, 90],  # Intraday confidence never too low or absurdly high
                "penalties": {
                    "weak_trend": {
                    "condition": "adx < 20",
                    "amount": 12,
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
                    }
                }
            },
            "setup_classification": {
                "rules": {
                    "MOMENTUM_BREAKOUT": {
                    "conditions": [
                        "bb_percent_b >= 0.98",
                        "rsi >= 60",
                        "rvol >= 2.0"               # stricter volume for intraday
                    ],
                    "priority": 95
                    },
                    "MOMENTUM_BREAKDOWN": {
                    "conditions": [
                        "bb_percent_b <= 0.02",
                        "rsi <= 40",
                        "rvol >= 2.0"
                    ],
                    "priority": 95
                    },
                    "TREND_PULLBACK": {
                    "conditions": [
                        "trend_dir == 'up'",
                        "abs(price - ma_fast) / ma_fast <= 0.03",  # tighter pullback
                        "rsi >= 50"
                    ],
                    "priority": 80
                    },
                    "VOLATILITY_SQUEEZE": {
                    "conditions": [
                        "is_squeeze == True",
                        "volatility_quality >= 7.0",
                        "rvol >= 1.5"
                    ],
                    "priority": 85
                    },
                    "QUALITY_ACCUMULATION": {
                    "conditions": [
                        "is_consolidating == True",
                        "volatility_quality <= 4.0"
                    ],
                    "priority": 60
                    }
                }
            },
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
        # SHORT TERM
        "short_term": {
            "volume_analysis": {
                "rvol_surge_threshold": 2.5,
                "rvol_drought_threshold": 0.7
            },
            "time_estimation": {
                "candles_per_unit": 1  # Only this!
            },
            "timeframe": "1d",
            "description": "Swing trading (Days to Weeks)",
            # moving_averages: SAME AS GLOBAL (no override needed)
            # indicators: SAME AS GLOBAL (no override needed)
            # momentum_thresholds: SAME AS GLOBAL (no override needed)
            # volatility: SAME AS GLOBAL (no override needed)
            "risk_management": {
                "max_position_pct": 0.02,  # Same as global but explicit
                "setup_size_multipliers": {
                    "DEEP_PULLBACK": 1.5,  # Different setups
                    "MOMENTUM_BREAKOUT": 1.0
                },
                "atr_sl_limits": {"max_percent": 0.03, "min_percent": 0.01},
                "min_rr_ratio": 1.4,  # Changed from 1.5
                "horizon_t2_cap": 0.10,  # Same as global
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 40, "t1_mult": 2.0, "t2_mult": 4.0},
                    "normal_trend": {"adx_min": 20, "t1_mult": 1.5, "t2_mult": 3.0},
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.2, "t2_mult": 2.5}
                }
            },
            "execution": {
                "stop_loss_atr_mult": 2.0,  # Same as global
                "target_atr_mult": 3.0,  # Same as global
                "max_hold_candles": 15,  # Changed from 20
                "dip_buy_reference": "ema_50",  # Same as global
                "risk_reward_min": 2.0,  # Same as global
                "base_hold_days": 10,
                "proximity_rejection": {  # NEW
                    "resistance_mult": 1.005,
                    "support_mult": 0.995
                },
                "min_profit_pct": 0.5  # NEW
            },
            "lookback": {"python_data": 600},  # Same as global but explicit
            "scoring": {
                "fundamental_weight": 0.3,  # # 30% fundamental, 70% technical
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
                    "days_to_earnings": {"op": "<", "val": 3, "pen": 0.5},
                    "rvol": {"op": "<", "val": 0.5, "pen": 0.2},
                    "atr_pct": {"op": "<", "val": 1.0, "pen": 0.2},
                    "ma_fast_slope": {"op": "<", "val": -5, "pen": 0.3}
                },
                "thresholds": {"buy": 6.0, "hold": 4.8, "sell": 4.0},
                "metric_weights": {
                    "pe_ratio":               0.10,
                    "peg_ratio":              0.10,
                    "roe":                    0.10,
                    "roce":                   0.10,
                    "de_ratio":               0.05,
                    "eps_growth_5y":          0.15,
                    "revenue_growth_5y":      0.10,
                    "quarterly_growth":       0.05,
                    "trend_strength":         0.10,
                    "momentum_strength":      0.10,
                    "volatility_quality":     0.05,
                },
            },
            "technical_weight_overrides": {
                    # ----- BALANCED APPROACH (mostly use global defaults) -----
                    # Only override what matters for swing trading
                    
                    "momentum_strength": 1.1,        # Slight boost to momentum
                    "trend_strength": 1.1,           # Slight boost to trend
                    
                    "macd_cross": 1.1,               # Key swing signal
                    "supertrend_signal": 1.1,
                    "ma_cross_signal": 1.1,
                    
                    "vol_spike_signal": 1.0,         # Normal volume importance
                    "rvol": 1.0,
                    
                    "adx": 1.0,                      # Standard trend strength
                    "price_vs_200dma_pct": 1.0,
                    
                    "rsi_slope": 1.0,                # Balanced momentum
                    "stoch_k": 1.0,
                    
                    "volatility_quality": 1.0,       # Normal volatility check
                    
                    # Slight reduction for ultra-fast indicators
                    "gap_percent": 0.9,
                    "vwap_bias": 0.8                 # Less relevant for multi-day holds
                },
            "gates": {
                "min_trend_strength": 2.0,  # Changed from 3.0
                "allowed_supertrend_counter": False,  # Same as global
                "volatility_bands_atr_pct": {  # CHANGE from array to object
                    "min": 0.8,
                    "ideal": 2.5,  # ADD this field
                    "max": 12.0
                },
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 3.0,
                    "VOLATILITY_SQUEEZE": 5.0,
                    "TREND_PULLBACK": 3.5,
                    "default": 3
                },
                "confidence_min": 60,
                "adx_min": 18,
                "volatility_quality_min": 4.0,
                "trend_strength_min": 3.5,
                "rr_ratio_min": 1.3,
                "volatility_guards": {  # NEW
                    "extreme_vol_buffer": 2.0,
                    "min_quality_breakout": 3.0,
                    "min_quality_normal": 4.0
                },
            },
            "trend_thresholds": {  # NEW section
                "slope": {
                    "strong": 10.0,
                    "moderate": 3.0
                }
            },
            "confidence": {
                "horizon_discount": 5,  # Same as global
                "floors": {"buy": 55, "wait": 30},  # wait changed from 35
                "base_floors": {  # ADD ALL OF THESE
                    "MOMENTUM_BREAKOUT": 55,
                    "MOMENTUM_BREAKDOWN": 55,
                    "TREND_PULLBACK": 53,
                    "DEEP_PULLBACK": 50,
                    "QUALITY_ACCUMULATION": 45,
                    "VOLATILITY_SQUEEZE": 50,
                    "TREND_FOLLOWING": 50,
                    "BEAR_TREND_FOLLOWING": 50,
                    "REVERSAL_MACD_CROSSUP": 50,
                    "REVERSAL_RSI_SWINGUP": 50,
                    "REVERSAL_ST_FLIPUP": 55,
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -20,  # Changed from -15
                    "ignore_for_squeeze": True
                },
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 60},
                    "moderate": {"adx_min": 25, "floor": 50},
                    "weak": {"adx_min": 20, "floor": 40},
                    "range_bound": {"adx_max": 20, "floor": 30}
                },
                "setup_type_overrides": {
                    "MOMENTUM_BREAKOUT": 5,
                    "VOLATILITY_SQUEEZE": 10,
                    "QUALITY_ACCUMULATION": -5  # Changed from 0
                }
            },
            "setup_confidence": {
                "confidence_clamp": [35, 95],
                "penalties": {
                    "weak_trend": {
                    "condition": "setup_type in ['TRENDPULLBACK', 'TRENDFOLLOWING'] and trend_strength < 3.5",
                    "amount": 15,
                    "reason": "Trend setup with weak trend_strength"
                    },
                    "moderate_divergence": {
                    "condition": "rsi_slope < -0.03",
                    "amount": 10,
                    "reason": "Moderate bearish RSI divergence"
                    },
                    "low_breakout_volume": {
                    "condition": "setup_type == 'MOMENTUM_BREAKOUT' and rvol < 1.5",
                    "amount": 8,
                    "reason": "Breakout with insufficient volume"
                    }
                },
                "bonuses": {
                    "pattern_confluence": {
                    "condition": "pattern_count >= 2",
                    "amount": 10,
                    "reason": "Multiple bullish patterns aligned"
                    },
                    "strong_trend_combo": {
                    "condition": "trend_strength >= 7.0 and momentum_strength >= 7.0",
                    "amount": 12,
                    "reason": "Strong trend + momentum alignment"
                    }
                }
                },
            "setup_classification": {
                "rules": {
                    "MOMENTUM_BREAKOUT": {
                    "conditions": [
                        "bb_percent_b >= 0.98",
                        "rsi >= 60",
                        "rvol >= 1.5"
                    ],
                    "priority": 90
                    },
                    "MOMENTUM_BREAKDOWN": {
                    "conditions": [
                        "bb_percent_b <= 0.02",
                        "rsi <= 40",
                        "rvol >= 1.5"
                    ],
                    "priority": 90
                    },
                    "TREND_PULLBACK": {
                    "conditions": [
                        "trend_dir == 'up'",
                        "abs(price - ma_fast) / ma_fast <= 0.05",
                        "rsi >= 50"
                    ],
                    "priority": 80
                    },
                    "TRENDFOLLOWING": {
                    "conditions": [
                        "trend_strength >= 3.5",
                        "rsi >= 55",
                        "macd_hist >= 0"
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
                    "priority": 70
                    }
                }
            },
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
        # LONG TERM
        "long_term": {
            "trend_thresholds": {  # NEW section
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
                "candles_per_unit": 0.2  # Only this!
            },
            "timeframe": "1wk",
            "description": "Trend Following & Investing",
            "moving_averages": {
                "type": "WMA",  # # âœ… "WMA" = Weekly MA (SMA on weekly interval)
                "fast": 10,  # Changed from 20
                "mid": 40,  # Changed from 50
                "slow": 50,  # Changed from 200
                "keys": ["wma_10", "wma_40", "wma_50"],
                "dip_buy_reference": "wma_40"
            },
            # indicators will be overridden using global (smart inheritance)
            "momentum_thresholds": {
                "rsi_slope": {
                    "acceleration_floor": 0.03,  # Changed from 0.05
                    "deceleration_ceiling": -0.03  # Changed from -0.05
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            "volatility": {
                "scoring_thresholds": {
                    "atr_pct": {
                        "excellent": 5.5,  # Changed from 2.5
                        "good": 9.0,  # Changed from 3.0
                        "fair": 13.0,  # Changed from 4.5
                        "poor": 18.0  # Changed from 5.5
                    },
                    "bb_width": {
                        "tight": 4.0,  # Changed from 3.0
                        "normal": 8.0,  # Changed from 6.0
                        "wide": 15.0  # Changed from 12.0
                    }
                }
            },
            "risk_management": {
                "max_position_pct": 0.03,  # Changed from 0.02
                "setup_size_multipliers": {"default": 1.0},
                "atr_sl_limits": {"max_percent": 0.05, "min_percent": 0.01},
                "min_rr_ratio": 1.5,  # Same as global
                "horizon_t2_cap": 0.20,  # Changed from 0.10
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 35, "t1_mult": 2.5, "t2_mult": 5.0},  # Changed
                    "normal_trend": {"adx_min": 20, "t1_mult": 2.0, "t2_mult": 4.0},  # Changed
                    "weak_trend": {"adx_max": 20, "t1_mult": 1.5, "t2_mult": 3.0}  # Changed
                }
            },
            "execution": {
                "stop_loss_atr_mult": 2.5,  # Changed from 2.0
                "target_atr_mult": 5.0,  # Changed from 3.0
                "max_hold_candles": 52,  # Changed from 20
                "dip_buy_reference": "wma_40",  # Changed from ema_50
                "risk_reward_min": 2.5,  # Changed from 2.0
                "base_hold_days": 60,
                "proximity_rejection": {  # NEW
                    "resistance_mult": 1.01,
                    "support_mult": 0.99
                },
                "min_profit_pct": 1.0  # NEW
            },
            "lookback": {"python_data": 800},  # Changed from 600
            "scoring": {
                "fundamental_weight": 0.5,  # 50-50 split
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
                    # Quality & profitability
                    "roe":                    0.12,
                    "roce":                   0.12,
                    "roic":                   0.10,
                    "de_ratio":               0.05,
                    "interest_coverage":      0.05,

                    # Growth & durability
                    "eps_growth_5y":          0.12,
                    "revenue_growth_5y":      0.10,
                    "market_cap_cagr":        0.08,
                    "earnings_stability":     0.06,
                    "fcf_growth_3y":          0.05,

                    # Ownership / quality of float
                    "promoter_holding":       0.05,
                    "institutional_ownership":0.05,
                    "ocf_vs_profit":          0.05
                },
            },
            "technical_weight_overrides": {
                    # ----- BOOST SLOW/TREND INDICATORS -----
                    "trend_strength": 1.4,           # Trend critical for long holds
                    "adx": 1.3,
                    "ma_trend_signal": 1.3,
                    "price_vs_200dma_pct": 1.4,      # Long-term trend position
                    "price_vs_primary_trend_pct": 1.3,
                    "dma_200_slope": 1.3,
                    "ma_cross_signal": 1.2,
                    
                    "ichi_cloud": 1.2,               # Multi-timeframe trend
                    "rel_strength_nifty": 1.2,       # Relative performance matters
                    
                    "supertrend_signal": 1.1,
                    "psar_trend": 1.1,
                    
                    # ----- REDUCE FAST INDICATORS -----
                    "rsi": 0.8,                      # Less important for long holds
                    "rsi_slope": 0.7,
                    "stoch_k": 0.7,
                    "macd_hist_z": 0.8,
                    
                    "vol_spike_signal": 0.6,         # Volume less critical
                    "rvol": 0.7,
                    "vol_spike_ratio": 0.6,
                    
                    "vwap_bias": 0.5,                # Intraday metric not relevant
                    "gap_percent": 0.6,
                    "wick_rejection": 0.7,
                    "bb_percent_b": 0.7,
                    
                    # ----- COMPOSITES -----
                    "momentum_strength": 0.9,        # Trend > Momentum
                    "volatility_quality": 1.1        # Quality volatility matters
                },
                
            "gates": {
                "min_trend_strength": 3.0,  # Same as global
                "allowed_supertrend_counter": False,  # Same as global
                "volatility_bands_atr_pct": {  # CHANGE from array to object
                    "min": 1.0,
                    "ideal": 5.5,  # ADD this field
                    "max": 15.0
                },
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 4.0,
                    "VOLATILITY_SQUEEZE": 6.0,
                    "TREND_PULLBACK": 4.5,
                    "default": 4
                },
                "confidence_min": 65,
                "adx_min": 20,
                "volatility_quality_min": 5.0,
                "trend_strength_min": 5.0,
                "rr_ratio_min": 1.5,
                "volatility_guards": {  # NEW
                    "extreme_vol_buffer": 3.0,
                    "min_quality_breakout": 4.0,
                    "min_quality_normal": 5.0
                },
            },
            "confidence": {
                "horizon_discount": 0,  # Changed from 5
                "floors": {"buy": 60, "wait": 40},  # Changed from 55/35
                 "base_floors": {  # ADD ALL OF THESE
                    "MOMENTUM_BREAKOUT": 60,
                    "MOMENTUM_BREAKDOWN": 60,
                    "TREND_PULLBACK": 58,
                    "DEEP_PULLBACK": 55,
                    "QUALITY_ACCUMULATION": 55,
                    "VOLATILITY_SQUEEZE": 55,
                    "TREND_FOLLOWING": 55,
                    "BEAR_TREND_FOLLOWING": 55,
                    "REVERSAL_MACD_CROSSUP": 55,
                    "REVERSAL_RSI_SWINGUP": 55,
                    "REVERSAL_ST_FLIPUP": 60,
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -10,  # Changed from -15
                    "ignore_for_squeeze": True
                },
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 65},  # Changed from 60
                    "moderate": {"adx_min": 25, "floor": 55},  # Changed from 50
                    "weak": {"adx_min": 20, "floor": 45},  # Changed from 40
                    "range_bound": {"adx_max": 20, "floor": 35}  # Changed from 30
                },
                "setup_type_overrides": {
                    "MOMENTUM_BREAKOUT": 3,  # Changed from 5
                    "VOLATILITY_SQUEEZE": 5,  # Changed from 10
                    "QUALITY_ACCUMULATION": 10  # Changed from 0
                }
            },
            "setup_confidence": {
                "confidence_clamp": [40, 98],
                "penalties": {
                    "weak_trend": {
                    "condition": "trend_strength < 5.0",
                    "amount": 10,
                    "reason": "Insufficient trend strength for long-term trend setup"
                    },
                    "poor_fundamentals": {
                    "condition": "roe < 15 or roce < 15",
                    "amount": 15,
                    "reason": "Weak fundamentals for long-term hold"
                    },
                    "high_debt": {
                    "condition": "de_ratio > 1.0",
                    "amount": 12,
                    "reason": "High leverage risk"
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
                    }
                }
            },
            "setup_classification": {
                "rules": {
                    "MOMENTUM_BREAKOUT": {
                    "conditions": [
                        "bb_percent_b >= 0.98",
                        "rsi >= 60"
                    ],
                    "priority": 70   # less emphasis vs quality
                    },
                    "TREND_PULLBACK": {
                    "conditions": [
                        "trend_dir == 'up'",
                        "abs(price - ma_mid) / ma_mid <= 0.08",  # allow deeper pullbacks
                        "rsi >= 45"
                    ],
                    "priority": 80
                    },
                    "TRENDFOLLOWING": {
                    "conditions": [
                        "trend_strength >= 5.0",
                        "rsi >= 55"
                    ],
                    "priority": 85
                    },
                    "QUALITY_ACCUMULATION": {
                    "conditions": [
                        "is_consolidating == True",
                        "roe >= 18",
                        "roce >= 20",
                        "de_ratio <= 0.7"
                    ],
                    "priority": 90
                    }
                }
            },
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
        # MULTIBAGGER
        "multibagger": {
            "trend_thresholds": {  # NEW section
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
                "candles_per_unit": 0.05  # Only this!
            },
            "timeframe": "1mo",
            "description": "Deep Value & Compounders",
            "moving_averages": {
                "type": "MMA",   # âœ… "MMA" = Monthly MA (SMA on monthly interval)
                "fast": 6,  # Changed from 20
                "mid": 12,  # Changed from 50
                "slow": 12,  # Changed from 200
                "keys": ["mma_6", "mma_12", "mma_12"],
                "dip_buy_reference": "mma_12"
            },
            "indicators": {
                "rsi_period": 14,
                "adx_period": 14,  # Changed from 14
                "atr_period": 20,  # Changed from 14
                "supertrend": {"period": 10, "multiplier": 3},
                "bollinger": {"period": 20, "std_dev": 2.0},
                "keltner": {"period": 20, "atr_mult": 1.5},
                "pivot_type": "classic",
                "stochastic": {
                    "k": 21,
                    "d": 5,
                    "smooth": 5,
                    "overbought": 85,     # Stricter
                    "oversold": 15        # Stricter
                }
            },
            "momentum_thresholds": {
                "rsi_slope": {
                    "acceleration_floor": 0.02,  # Changed from 0.05
                    "deceleration_ceiling": -0.02  # Changed from -0.05
                },
                "macd": {
                    "acceleration_floor": 0.5,
                    "deceleration_ceiling": -0.5
                }
            },
            "volatility": {
                "scoring_thresholds": {
                    "atr_pct": {
                        "excellent": 11.5,  # Changed from 2.5
                        "good": 18.0,  # Changed from 3.0
                        "fair": 27.0,  # Changed from 4.5
                        "poor": 36.0  # Changed from 5.5
                    },
                    "bb_width": {
                        "tight": 6.0,  # Changed from 3.0
                        "normal": 12.0,  # Changed from 6.0
                        "wide": 20.0  # Changed from 12.0
                    }
                }
            },
            "risk_management": {
                "max_position_pct": 0.05,  # Changed from 0.02
                "setup_size_multipliers": {"QUALITY_ACCUMULATION": 1.5},
                "atr_sl_limits": {"max_percent": 0.10, "min_percent": 0.02},
                "min_rr_ratio": 2.0,  # Changed from 1.5
                "horizon_t2_cap": 1.00,  # Changed from 0.10
                "rr_regime_adjustments": {
                    "strong_trend": {"adx_min": 30, "t1_mult": 3.0, "t2_mult": 10.0},  # Changed
                    "normal_trend": {"adx_min": 20, "t1_mult": 2.5, "t2_mult": 8.0},  # Changed
                    "weak_trend": {"adx_max": 20, "t1_mult": 2.0, "t2_mult": 6.0}  # Changed
                }
            },
            "execution": {
                "stop_loss_atr_mult": 3.0,  # Changed from 2.0
                "target_atr_mult": 10.0,  # Changed from 3.0
                "max_hold_candles": 60,  # Changed from 20
                "dip_buy_reference": "mma_12",  # Changed from ema_50
                "risk_reward_min": 3.0,  # Changed from 2.0
                "base_hold_days": 180,
                "proximity_rejection": {  # NEW
                    "resistance_mult": 1.02,
                    "support_mult": 0.98
                },
                "min_profit_pct": 2.0  # NEW
            },
            "lookback": {"python_data": 3000},  # Changed from 600
            "technical_weight_overrides": {
                    # ----- ULTRA-BOOST SLOW INDICATORS -----
                    "trend_strength": 1.5,           # Primary filter
                    "adx": 1.5,
                    "ma_trend_signal": 1.6,
                    "price_vs_200dma_pct": 1.6,      # Must be in long-term uptrend
                    "price_vs_primary_trend_pct": 1.5,
                    "dma_200_slope": 1.5,
                    
                    "ma_cross_signal": 1.4,          # Major regime shifts
                    "ichi_cloud": 1.3,
                    "rel_strength_nifty": 1.4,       # Must outperform market
                    "nifty_trend_score": 1.2,
                    
                    # ----- HEAVILY REDUCE FAST INDICATORS -----
                    "rsi": 0.5,                      # Noise for long-term
                    "rsi_slope": 0.4,
                    "stoch_k": 0.4,
                    "macd_cross": 0.6,
                    "macd_hist_z": 0.5,
                    
                    "vol_spike_signal": 0.4,         # Volume spikes irrelevant
                    "rvol": 0.5,
                    "vol_spike_ratio": 0.4,
                    "vol_trend": 0.5,
                    
                    "vwap_bias": 0.3,                # Completely irrelevant
                    "gap_percent": 0.4,
                    "price_action": 0.5,
                    "wick_rejection": 0.5,
                    "bb_percent_b": 0.5,
                    "ttm_squeeze": 0.6,
                    
                    # ----- COMPOSITES -----
                    "momentum_strength": 0.7,        # Trend >> Momentum
                    "volatility_quality": 1.2        # Stable volatility preferred
                },
            "scoring": {
                "fundamental_weight": 0.7,  # 70% fundamental, 30% technical
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
                    # Core growth engine
                    "eps_growth_5y":          0.15,
                    "revenue_growth_5y":      0.15,
                    "market_cap_cagr":        0.10,
                    "quarterly_growth":       0.05,
                    # Moat / quality
                    "roic":                   0.10,
                    "roe":                    0.10,
                    "roce":                   0.08,
                    "de_ratio":               0.05,
                    "ocf_vs_profit":          0.05,
                    "earnings_stability":     0.05,
                    # Ownership / discovery
                    "promoter_holding":       0.04,
                    "institutional_ownership":0.04,
                    "r_d_intensity":          0.04
                },
            },
           "gates": {
                "min_trend_strength": None,  # Changed from 3.0
                "allowed_supertrend_counter": False,
                "volatility_bands_atr_pct": {  # CHANGE from array to object
                    "min": 1.5,
                    "ideal": 8.0,  # ADD this field
                    "max": 25.0
                },
                "volatility_quality_mins": {
                    "MOMENTUM_BREAKOUT": 5.0,
                    "VOLATILITY_SQUEEZE": 7.0,
                    "TREND_PULLBACK": 5.0,
                    "default": 4.5
                },
                "confidence_min": 70,
                "adx_min": 25,
                "volatility_quality_min": 6.0,
                "trend_strength_min": 6.0,
                "rr_ratio_min": 1.5,
                "volatility_guards": {  # NEW
                    "extreme_vol_buffer": 4.0,
                    "min_quality_breakout": 5.0,
                    "min_quality_normal": 6.0
                },
            },
            "confidence": {
                "horizon_discount": 0,  # Changed from 5
                "floors": {"buy": 65, "wait": 50},  # Changed from 55/35
                "base_floors": {  # ADD ALL OF THESE
                    "MOMENTUM_BREAKOUT": 65,
                    "MOMENTUM_BREAKDOWN": 65,
                    "TREND_PULLBACK": 60,
                    "DEEP_PULLBACK": 58,
                    "QUALITY_ACCUMULATION": 65,
                    "VOLATILITY_SQUEEZE": 60,
                    "TREND_FOLLOWING": 60,
                    "BEAR_TREND_FOLLOWING": 60,
                    "REVERSAL_MACD_CROSSUP": 60,
                    "REVERSAL_RSI_SWINGUP": 60,
                    "REVERSAL_ST_FLIPUP": 65,
                },
                "volume_penalty": {
                    "rvol_drought_penalty": -5,  # Changed from -15
                    "ignore_for_squeeze": True
                },
                "adx_based_floors": {
                    "strong": {"adx_min": 30, "floor": 70},  # Changed from 60
                    "moderate": {"adx_min": 25, "floor": 60},  # Changed from 50
                    "weak": {"adx_min": 20, "floor": 50},  # Changed from 40
                    "range_bound": {"adx_max": 20, "floor": 40}  # Changed from 30
                },
                "setup_type_overrides": {
                    "MOMENTUM_BREAKOUT": 0,  # Changed from 5
                    "VOLATILITY_SQUEEZE": 0,  # Changed from 10
                    "QUALITY_ACCUMULATION": 15  # Changed from 0
                }
            },
            "setup_confidence": {
                "confidence_clamp": [45, 99],

                "penalties": {
                    "insufficient_growth": {
                    "condition": "eps_growth_5y < 15 or revenue_growth_5y < 15",
                    "amount": 20,
                    "reason": "Growth too low for multibagger profile"
                    },
                    "high_debt": {
                    "condition": "de_ratio > 0.5",
                    "amount": 20,
                    "reason": "Leverage too high for multibagger candidate"
                    },
                    "weak_trend": {
                    "condition": "trend_strength < 6.0",
                    "amount": 10,
                    "reason": "Trend not established"
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
                    }
                }
            },
            "setup_classification": {
                "rules": {
                    "TRENDFOLLOWING": {
                    "conditions": [
                        "trend_strength >= 6.0",
                        "rsi >= 55"
                    ],
                    "priority": 90
                    },
                    "QUALITY_ACCUMULATION": {
                    "conditions": [
                        "is_consolidating == True",
                        "roe >= 20",
                        "roce >= 25",
                        "de_ratio <= 0.5",
                        "eps_growth_5y >= 20",
                        "revenue_growth_5y >= 20"
                    ],
                    "priority": 95
                    },
                    "MOMENTUM_BREAKOUT": {
                    "conditions": [
                        "bb_percent_b >= 0.98",
                        "rsi >= 60",
                        "rel_strength_nifty >= 1.1"
                    ],
                    "priority": 80
                    }
                }
            },
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