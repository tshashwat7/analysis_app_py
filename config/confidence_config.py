# config/confidence_config.py (DEDUPLICATED VERSION)
"""
Confidence Calculation Configuration
Separated from master_config for clarity and maintainability

DESIGN PHILOSOPHY:
- Buckets usage scales with time horizon
- Intraday: Pure technical (no fundamental filtering)
- Short term: Light filtering (only extreme cases)
- Long term: Heavy filtering (profile matters)
- Multibagger: Maximum selectivity (perfect profiles only)
- Uses structured gate format (matching entry_gates pattern)
- Single source of truth for all confidence logic
- Clear calculation pipeline order
- Horizon-specific overrides inherit from global baseline

LAYER ARCHITECTURE:
1. UNIVERSAL: global.volume_modifiers, global.universal_adjustments
2. SETUP-SPECIFIC: horizons.*.conditional_adjustments.bonuses (with apply_to_setups)
3. HORIZON-UNIVERSAL: master_config.py -> horizons.*.enhancements (applies to ALL setups)

USAGE:
    from config.confidence_config import CONFIDENCE_CONFIG
    
    # Get base floor for a setup
    floor = CONFIDENCE_CONFIG['global']['setup_baseline_floors']['MOMENTUM_BREAKOUT']
    
    # Get horizon-specific adjustments
    intraday_adjustments = CONFIDENCE_CONFIG['horizons']['intraday']['conditional_adjustments']
"""
ADX_USAGE_CLARIFICATION = """
TWO SEPARATE ADX SYSTEMS:

1. Global ADX Normalization (Step 4a - COMPARISON ONLY):
   - Formula: ((adx - 10) / (40 - 10)) * 12
   - Purpose: Normalize ADX for cross-setup comparison
   - NOT used in confidence calculation pipeline

2. Horizon ADX Bands (Step 4b - CONFIDENCE ADJUSTMENT):
   - Explosive: ADX ≥ 35 → +20 confidence
   - Strong: ADX 25-35 → +10 confidence
   - Purpose: Actual confidence boost/penalty
   - USED in pipeline step 4

Only adx_confidence_bands affect confidence score.
The normalization formula is for analytics/comparison only.
"""

CONFIDENCE_CONFIG = {
    "global": {
        # ============================================================================
        # CORE CONFIDENCE PARAMETERS (Universal across all horizons)
        # ============================================================================
        "confidence_range": {
            "absolute_min": 0,
            "absolute_max": 100,
            "default_clamp": [30, 95]  # Most horizons use this unless overridden
        },
        
        # ADX-based confidence normalization (affects all horizons unless overridden)
        "adx_normalization": {
            "floor_boundary": 10,
            "ceiling_boundary": 40,
            "adjustment_factor": 12,
            "formula": "((adx - floor) / (ceiling - floor)) * adjustment_factor"
        },
        
        # Volume impact on confidence (universal penalties/bonuses)
        "volume_modifiers": {
            "surge_bonus": {
                "gates": {
                    "rvol": {"min": 3.0}
                },
                "confidence_boost": 10,
                "exclude_setups": ["VOLATILITY_SQUEEZE"],
                "reason": "Strong volume confirmation"
            },
            "drought_penalty": {
                "gates": {
                    "rvol": {"max": 0.7}
                },
                "confidence_penalty": -15,
                "exclude_setups": ["VOLATILITY_SQUEEZE", "QUALITY_ACCUMULATION"],
                "reason": "Insufficient volume support"
            },
            "climax_warning": {
                "gates": {
                    "rvol": {"min": 5.0},
                    "rsi": {"min": 70}
                },
                "confidence_penalty": -20,
                "reason": "Volume climax - potential exhaustion"
            }
        },
        "divergence_physics": {
            "lookback": 10,
            "slope_diff_min": -0.05,        # Bearish trigger
            "bullish_slope_min": 0.05,     # Bullish trigger
            "description": "Mathematical parameters for finding RSI/Price divergence"
        },
        # Setup-agnostic confidence adjustments
        "universal_adjustments": {
            "divergence_penalties": {
                "severe": {
                    "gates": {
                        "rsislope": {"max": -0.08}
                    },
                    "confidence_multiplier": 0.50,
                    "block_entry": True,
                    "reason": "Severe bearish divergence"
                },
                "moderate": {
                    "gates": {
                        "rsislope": {"max": -0.03, "min": -0.08}
                    },
                    "confidence_multiplier": 0.70,
                    "block_entry": False,
                    "reason": "Moderate bearish divergence"
                },
                "minor": {
                    "gates": {
                        "rsislope": {"max": 0.0, "min": -0.03}
                    },
                    "confidence_multiplier": 0.90,
                    "block_entry": False,
                    "reason": "Minor bearish divergence"
                }
            },
            
            "trend_strength_bands": {
                "explosive": {
                    "gates": {
                        "trendStrength": {"min": 8.0}
                    },
                    "confidence_boost": 25,
                    "reason": "Explosive trend momentum"
                },
                "strong": {
                    "gates": {
                        "trendStrength": {"min": 6.0, "max": 8.0}
                    },
                    "confidence_boost": 15,
                    "reason": "Strong sustained trend"
                },
                "moderate": {
                    "gates": {
                        "trendStrength": {"min": 4.0, "max": 6.0}
                    },
                    "confidence_boost": 5,
                    "reason": "Moderate trend support"
                },
                "weak": {
                    "gates": {
                        "trendStrength": {"max": 3.9}  # Fixed: was 4.0, caused double-hit with moderate min:4.0
                    },
                    "confidence_penalty": -10,
                    "reason": "Weak trend - sideways risk"
                }
            }
        },
        
        # Setup-specific baseline confidence floors (inherited unless overridden)
        "setup_baseline_floors": {
            "MOMENTUM_BREAKOUT": 55,
            "MOMENTUM_BREAKDOWN": 55,
            "VOLATILITY_SQUEEZE": 50,
            "QUALITY_ACCUMULATION": 45,
            "QUALITY_ACCUMULATION_DOWNTREND": 40,
            "DEEP_VALUE_PLAY": 40,
            "VALUE_TURNAROUND": 50,
            "TREND_PULLBACK": 50,
            "DEEP_PULLBACK": 48,
            "TREND_FOLLOWING": 50,
            "BEAR_TREND_FOLLOWING": 50,
            "REVERSAL_MACD_CROSS_UP": 48,
            "REVERSAL_RSI_SWING_UP": 45,
            "REVERSAL_ST_FLIP_UP": 52,
            "PATTERN_DARVAS_BREAKOUT": 60,
            "PATTERN_VCP_BREAKOUT": 60,
            "PATTERN_CUP_BREAKOUT": 55,
            "PATTERN_FLAG_BREAKOUT": 60,
            "PATTERN_GOLDEN_CROSS": 55,
            "PATTERN_STRIKE_REVERSAL": 50,
            "SELL_AT_RANGE_TOP": 55,
            "TAKE_PROFIT_AT_MID": 50,
            "GENERIC": 40
        }
    },
    
    # ============================================================================
    # HORIZON-SPECIFIC CONFIDENCE TUNING
    # ============================================================================
    "horizons": {
        "intraday": {
            "confidence_philosophy": "High floor, tight range - need conviction for fast trades. TECHNICAL ONLY.",
            
            "confidence_clamp": [35, 90],
            
            "base_confidence_adjustment": -10,
            
            "setup_floor_overrides": {
                "MOMENTUM_BREAKOUT": 60,
                "VOLATILITY_SQUEEZE": 55,
                "QUALITY_ACCUMULATION": 50,
                "DEEP_VALUE_PLAY": None,  # Block deep value for intraday
                "VALUE_TURNAROUND": None   # Block turnarounds for intraday
            },
            
            "conditional_adjustments": {
                "penalties": {
                    # ========================================================
                    # ONLY TECHNICAL PENALTIES (No fundamental filtering)
                    # ========================================================
                    "weak_intraday_trend": {
                        "gates": {"adx": {"max": 20}},
                        "confidence_penalty": -25,
                        "reason": "Weak trend - choppy intraday action"
                    },
                    "high_volatility_noise": {
                        "gates": {"atrPct": {"min": 8.0}},
                        "confidence_penalty": -15,
                        "reason": "Excessive volatility for intraday"
                    },
                    "wide_spread": {
                        "gates": {"bbWidth": {"min": 5.0}},
                        "confidence_penalty": -10,
                        "reason": "Wide bands suggest uncertainty"
                    },
                    "low_volume_breakout": {
                        "gates": {"rvol": {"max": 1.8}},
                        "apply_to_setups": ["MOMENTUM_BREAKOUT"],
                        "confidence_penalty": -18,
                        "reason": "Breakout without volume support"
                    },
                    "choppy_action": {
                        "gates": {
                            "trendStrength": {"max": 3.0},
                            "volatilityQuality": {"max": 4.0}
                        },
                        "confidence_penalty": -22,
                        "reason": "Choppy intraday - high whipsaw risk"
                    }
                    
                    # ❌ NO FUNDAMENTAL PENALTIES
                    # Rationale: Intraday = technical game. Don't block good 
                    # technical setups on weak fundamental names.
                },
                
                "bonuses": {
                    # ✅ SETUP-SPECIFIC: Only for MOMENTUM_BREAKOUT
                    "clean_breakout": {
                        "gates": {
                            "rvol": {"min": 3.0},
                            "wickRejection": {"max": 1.5}
                        },
                        "apply_to_setups": ["MOMENTUM_BREAKOUT"],
                        "confidence_boost": 15,
                        "reason": "Clean volume-confirmed breakout"
                    },
                    # ✅ UNIQUE: Different from enhancements (trend+momentum vs rvol)
                    "explosive_intraday": {
                        "gates": {
                            "trendStrength": {"min": 8.0},
                            "momentumStrength": {"min": 8.0}
                        },
                        "confidence_boost": 25,
                        "reason": "Explosive intraday momentum"
                    },
                    
                    # ============================================================
                    # MIGRATED FROM master_config.py
                    # ============================================================
                    "volume_surge": {
                        "gates": {"rvol": {"min": 2.5}},
                        "confidence_boost": 12.0,
                        "reason": "Strong intraday volume surge"
                    },
                    "squeeze_release": {
                        "gates": {
                            "bbWidth": {"max": 3.0},
                            "rvol": {"min": 2.0}
                        },
                        "confidence_boost": 15.0,
                        "reason": "Tight squeeze breaking with volume"
                    },
                    "momentum_spike": {
                        "gates": {"momentumStrength": {"min": 8.0}},
                        "confidence_boost": 10.0,
                        "reason": "Explosive momentum for scalp"
                    },
                    
                    # ========================================================
                    # OPTIONAL: Light fundamental bonus (not penalty!)
                    # ========================================================
                    "quality_name_tailwind": {
                        "gates": {"fundamentalScore": {"min": 7.0}},
                        "confidence_boost": 3,  # Very small
                        "reason": "Quality name - slight confidence tailwind"
                    }
                    
                    # Note: We reward quality, but don't penalize junk
                    # Allows capturing volatility in weak names
                }
            },
            
            "adx_confidence_bands": {
                "explosive": {"gates": {"adx": {"min": 35}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 25, "max": 35}}, "confidence_boost": 10},
                "moderate": {"gates": {"adx": {"min": 18, "max": 25}}, "confidence_boost": 0}
            },
            "adx_confidence_penalties": {
                "weak": {
                    "gates": {"adx": {"max": 17}},  # Fixed: was 18, caused double-hit with moderate min:18
                    "confidence_penalty": -15,
                    "reason": "Very weak trend - avoid intraday"
                }
            },
            
            # Horizon-level tradeable floor: below this, signal is untradeable
            "min_tradeable_confidence": {
                "min": 40,
                "reason": "Intraday requires minimum conviction for fast execution"
            },
            
            # High-confidence override: metadata for downstream consumers
            "high_confidence_override": {
                "threshold": 80,
                "can_override": {
                    "execution_warnings": True,   # Can bypass execution warnings only
                    "structural_gates": False,    # Cannot bypass structural gates
                    "fundamental_gates": False    # No fundamental layer for intraday
                },
                "max_override_count": 2,
                "log_overrides": True
            }
        },
        
        "short_term": {
            "confidence_philosophy": "Balanced - technical + fundamentals blend. Allow speculative momentum.",
            
            "confidence_clamp": [35, 95],
            
            "base_confidence_adjustment": -5,
            
            "setup_floor_overrides": {
                "VALUE_TURNAROUND": 55,
                "QUALITY_ACCUMULATION": 50
            },
            
            "conditional_adjustments": {
                "penalties": {
                    "weak_swing_trend": {
                        "gates": {"trendStrength": {"max": 3.5}},
                        "apply_to_setups": ["TREND_PULLBACK", "TREND_FOLLOWING"],
                        "confidence_penalty": -20,
                        "reason": "Trend-following setup without trend"
                    },
                    "breakout_low_volume": {
                        "gates": {"rvol": {"max": 1.4}},
                        "apply_to_setups": ["MOMENTUM_BREAKOUT"],
                        "confidence_penalty": -12,
                        "reason": "Breakout lacks volume conviction"
                    },
                    # ❌ REMOVED: moderate_divergence — duplicated global.universal_adjustments.moderate
                    # (same rsislope range, was causing double-penalty: 0.70× multiplier + -15 additive)
                    
                    # ========================================================
                    # BUCKET-BASED PENALTIES LIGHT FUNDAMENTAL PENALTY (Only Extreme Garbage) 
                    # ========================================================
                    "complete_garbage_swing": {
                        "gates": {
                            "fundamentalScore": {"max": 3.0},  # Extreme threshold
                            "fund_quality_bucket": {"max": 3.5},
                            "fund_health_bucket": {"max": 3.5}
                        },
                        "confidence_penalty": -12,  # Moderate penalty
                        "apply_to_setups": [
                            "TREND_PULLBACK",
                            "TREND_FOLLOWING",
                            "QUALITY_ACCUMULATION"
                        ],
                        "reason": "Extremely poor fundamentals - risky even for swing"
                    }
                    
                    # ❌ NO "speculative growth" penalty
                    # Rationale: Swing trading IS about riding speculative momentum
                    # Don't filter high-growth, low-quality names
                },
                
                "bonuses": {
                    # ========================================================
                    # TECHNICAL BONUSES
                    # ========================================================
                    "quality_swing": {
                        "gates": {
                            "fundamentalScore": {"min": 7.0},
                            "technicalScore": {"min": 7.0}
                        },
                        "confidence_boost": 15,
                        "reason": "High-quality technical + fundamental setup"
                    },
                    
                    # ============================================================
                    # ✅ MIGRATED FROM master_config.py (Short Term)
                    # ============================================================
                    "pattern_confluence": {
                        "gates": {"pattern_count": {"min": 2}},
                        "confidence_boost": 12.0,
                        "reason": "Multiple patterns confirm swing setup"
                    },
                    "trend_momentum_sync": {
                        "gates": {
                            "trendStrength": {"min": 7.0},
                            "momentumStrength": {"min": 7.0}
                        },
                        "confidence_boost": 10.0,
                        "reason": "Strong trend + momentum for swing"
                    },
                    "quality_pullback": {
                        "gates": {"volatilityQuality": {"min": 6.0}},
                        "confidence_boost": 8.0,
                        "apply_to_setups": ["TREND_PULLBACK"],
                        "reason": "Clean pullback setup"
                    },
                    
                    # ========================================================
                    # BUCKET-BASED BONUSES LIGHT FUNDAMENTAL BONUS
                    # ========================================================
                    "quality_swing_profile": {
                        "gates": {
                            "fund_quality_bucket": {"min": 7.0},
                            "fund_health_bucket": {"min": 6.5}
                        },
                        "confidence_boost": 8,  # Moderate bonus
                        "apply_to_setups": [
                            "TREND_PULLBACK",
                            "TREND_FOLLOWING",
                            "QUALITY_ACCUMULATION"
                        ],
                        "reason": "Quality business for swing trade"
                    }
                }
            },
            
            "adx_confidence_bands": {
                "explosive": {"gates": {"adx": {"min": 30}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 22, "max": 30}}, "confidence_boost": 12},
                "moderate": {"gates": {"adx": {"min": 15, "max": 22}}, "confidence_boost": 0}
            },
            "adx_confidence_penalties": {
                "weak": {
                    "gates": {"adx": {"max": 14}},  # Fixed: was 15, caused double-hit with moderate min:15
                    "confidence_penalty": -12,
                    "reason": "Weak trend - reduce swing conviction"
                }
            },
            
            # Horizon-level tradeable floor: below this, signal is untradeable
            "min_tradeable_confidence": {
                "min": 35,
                "reason": "Short-term swing needs a reasonable conviction floor"
            },
            
            # High-confidence override: metadata for downstream consumers
            "high_confidence_override": {
                "threshold": 82,
                "can_override": {
                    "execution_warnings": True,
                    "structural_gates": False,
                    "fundamental_gates": False
                },
                "max_override_count": 2,
                "log_overrides": True
            }
        },
        
        "long_term": {
            "confidence_philosophy": "Fundamentals-first, patient capital. Quality + growth + health matter.",
            
            "confidence_clamp": [40, 98],
            
            "base_confidence_adjustment": 0,
            
            "setup_floor_overrides": {
                "VALUE_TURNAROUND": 55,  # Premium for long-term value
                "DEEP_VALUE_PLAY": 50,  # +10 vs global
                "QUALITY_ACCUMULATION": 55,  # +10 vs global
                "MOMENTUM_BREAKOUT": 50  # -5 vs global (less relevant)
            },
            
            "conditional_adjustments": {
                "penalties": {
                    "poor_fundamentals": {
                        "gates": {
                            "roe": {"max": 15},
                            "roce": {"max": 15},
                            "_logic": "OR"
                        },
                        "confidence_penalty": -20,
                        "reason": "Insufficient quality for long-term hold"
                    },
                    "high_debt": {
                        "gates": {"deRatio": {"min": 1.0}},
                        "confidence_penalty": -15,
                        "reason": "High leverage risk over time"
                    },
                    "weak_long_trend": {
                        "gates": {"trendStrength": {"max": 4.5}},
                        "confidence_penalty": -15,
                        "reason": "Trend too weak for position trade"
                    },
                    "low_growth": {
                        "gates": {
                            "epsGrowth5y": {"max": 10},
                            "revenueGrowth5y": {"max": 10},
                            "_logic": "OR"  # Fixed: was AND — matches multibagger.insufficient_growth intent
                        },
                        "confidence_penalty": -18,
                        "reason": "Stagnant growth profile"
                    },
                    
                    # ========================================================
                    # BUCKET-BASED STRONG BUCKET PENALTIES (Profile detection)
                    # ========================================================
                    "speculative_growth_longterm": {
                        "gates": {
                            "fund_growth_bucket": {"min": 8.5},  # High growth
                            "fund_quality_bucket": {"max": 6.0}  # Low quality
                        },
                        "confidence_penalty": -25,  # Strong penalty
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "DEEP_VALUE_PLAY",
                            "VALUE_TURNAROUND",
                            "TREND_FOLLOWING"
                        ],
                        "reason": "Speculative growth without quality foundation - incompatible with long-term hold"
                    },
                    "fragile_balance_sheet_longterm": {
                        "gates": {
                            "fundamentalScore": {"min": 6.5},  # Acceptable overall
                            "fund_health_bucket": {"max": 5.5}  # Weak health
                        },
                        "confidence_penalty": -20,  # Strong penalty
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "DEEP_VALUE_PLAY",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "Balance sheet too weak for long-term thesis despite acceptable score"
                    }
                },
                
                "bonuses": {
                    # ========================================================
                    # RAW METRIC BONUSES (Kept for specific combinations)
                    # ========================================================
                    "high_quality_compounder": {
                        "gates": {
                            "roe": {"min": 20},
                            "roce": {"min": 25},
                            "epsGrowth5y": {"min": 15}
                        },
                        "confidence_boost": 20,
                        "reason": "High-quality long-term compounder"
                    },
                    # ✅ UNIQUE: ROIC + earnings stability combo
                    "stable_moat": {
                        "gates": {
                            "roic": {"min": 20},
                            "earningsStability": {"min": 7.0}
                        },
                        "confidence_boost": 15,
                        "reason": "Stable moat indicators"
                    },
                    # ✅ UNIQUE: Pure trend filter
                    "sustained_trend": {
                        "gates": {"trendStrength": {"min": 8.0}},
                        "confidence_boost": 25,
                        "reason": "Exceptional multi-year trend"
                    },
                    
                    # ============================================================
                    # ✅ MIGRATED FROM master_config.py (Long Term)
                    # ============================================================
                    # ❌ REMOVED: quality_fundamentals — strict subset of high_quality_compounder
                    # (roe>=20, roce>=25 is a subset of roe>=20, roce>=25, epsGrowth5y>=15)
                    # Both firing stacked +35 for one condition. Removed to prevent double-counting.
                    "earnings_acceleration": {
                        "gates": {
                            "quarterlyGrowth": {"min": 15},
                            "epsGrowth5y": {"min": 15}
                        },
                        "confidence_boost": 12.0,
                        "reason": "Consistent earnings growth"
                    },
                    "institutional_interest": {
                        "gates": {"institutionalOwnership": {"min": 25, "max": 75}},
                        "confidence_boost": 8.0,
                        "reason": "Smart money accumulating"
                    },
                    
                    # ========================================================
                    # BUCKET-BASED STRONG BUCKET BONUSES (Profile detection)
                    # ========================================================
                    "quality_compounder_profile": {
                        "gates": {
                            "fund_quality_bucket": {"min": 7.5},
                            "fund_growth_bucket": {"min": 7.0},
                            "fund_health_bucket": {"min": 7.0}
                        },
                        "confidence_boost": 25,  # Strong bonus
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "DEEP_VALUE_PLAY",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "Complete quality compounder profile: quality + growth + solid balance sheet"
                    },
                    "value_compounder_profile": {
                        "gates": {
                            "fund_quality_bucket": {"min": 7.5},
                            "fund_valuation_bucket": {"min": 7.0}  # Attractive valuation
                        },
                        "confidence_boost": 18,
                        "apply_to_setups": [
                            "DEEP_VALUE_PLAY",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "High-quality business at attractive valuation - value compounder"
                    }
                }
            },
            
            "adx_confidence_bands": {
                "explosive": {"gates": {"adx": {"min": 28}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 18, "max": 28}}, "confidence_boost": 12},
                "moderate": {"gates": {"adx": {"min": 12, "max": 18}}, "confidence_boost": 0}
            },
            "adx_confidence_penalties": {
                "weak": {
                    "gates": {"adx": {"max": 11}},  # Fixed: was 12, caused double-hit with moderate min:12
                    "confidence_penalty": -8,
                    "reason": "Very weak trend - acceptable for value plays"
                }
            },
            
            # Horizon-level tradeable floor: below this, signal is untradeable
            "min_tradeable_confidence": {
                "min": 40,
                "reason": "Long-term positions need a solid confidence floor"
            },
            
            # High-confidence override: metadata for downstream consumers
            "high_confidence_override": {
                "threshold": 85,
                "can_override": {
                    "execution_warnings": True,
                    "structural_gates": False,
                    "fundamental_gates": False
                },
                "max_override_count": 2,
                "log_overrides": True
            }
        },
        
        "multibagger": {
            "confidence_philosophy": "Ultra-selective, fundamentals-obsessed. Perfect profile required.",
            
            "confidence_clamp": [45, 99],
            
            "base_confidence_adjustment": 0,
            
            "setup_floor_overrides": {
                "VALUE_TURNAROUND": 60,  # +10 vs global
                "DEEP_VALUE_PLAY": 55,  # +15 vs global
                "QUALITY_ACCUMULATION": 60,  # +15 vs global
                "MOMENTUM_BREAKOUT": None,  # Block for multibagger
                "VOLATILITY_SQUEEZE": None  # Block for multibagger
            },
            
            "conditional_adjustments": {
                "penalties": {
                    "insufficient_growth": {
                        "gates": {
                            "epsGrowth5y": {"max": 15},
                            "revenueGrowth5y": {"max": 15},
                            "_logic": "OR"
                        },
                        "confidence_penalty": -15,
                        "reason": "Growth insufficient for multibagger thesis"
                    },
                    "weak_quality": {
                        "gates": {
                            "roe": {"max": 15},
                            "roce": {"max": 18},
                            "_logic": "OR"
                        },
                        "confidence_penalty": -25,
                        "reason": "Quality too low for multi-year hold"
                    },
                    "high_leverage": {
                        "gates": {"deRatio": {"min": 1.2}},
                        "confidence_penalty": -25,
                        "reason": "Debt risk over long horizon"
                    },
                    "bearish_structure": {
                        "gates": {
                            "trendStrength": {"max": 3.0},
                            "momentumStrength": {"max": 3.0}
                        },
                        "confidence_penalty": -30,
                        "reason": "Bearish trend - wait for reversal"
                    },
                    "overvalued": {
                        "gates": {"priceToIntrinsicValue": {"min": 1.3}},
                        "confidence_penalty": -20,
                        "reason": "Overvalued vs intrinsic value"
                    },
                    
                    # ========================================================
                    # BUCKET-BASED : MAXIMUM BUCKET PENALTIES
                    # ========================================================
                    "speculative_growth_multibagger": {
                        "gates": {
                            "fund_growth_bucket": {"min": 8.5},  # High growth
                            "fund_quality_bucket": {"max": 6.0}  # Low quality
                        },
                        "confidence_penalty": -30,  # Maximum penalty
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "DEEP_VALUE_PLAY",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "Speculative growth profile - not a multibagger candidate"
                    },
                    "fragile_balance_sheet_multibagger": {
                        "gates": {"fund_health_bucket": {"max": 6.0}},
                        "confidence_penalty": -30,  # Maximum penalty
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "DEEP_VALUE_PLAY",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "Fragile balance sheet incompatible with multi-year compounding thesis"
                    }
                },
                
                "bonuses": {
                    # ========================================================
                    # RAW METRIC BONUSES (Kept)
                    # ========================================================
                    "mega_trend_quality": {
                        "gates": {
                            "trendStrength": {"min": 8.5},
                            "roe": {"min": 25},
                            "roce": {"min": 30}
                        },
                        "confidence_boost": 35,
                        "reason": "Mega-trend with quality fundamentals"
                    },
                    # ✅ UNIQUE: Relative strength + cap growth
                    "early_leader": {
                        "gates": {
                            "relStrengthNifty": {"min": 1.2},
                            "marketCapCagr": {"min": 25}
                        },
                        "confidence_boost": 20,
                        "reason": "Early stage market leader"
                    },
                    # ✅ UNIQUE: Deep value filter
                    "deep_value": {
                        "gates": {
                            "priceToIntrinsicValue": {"max": 0.7},
                            "roe": {"min": 20}
                        },
                        "confidence_boost": 25,
                        "reason": "Deep value with quality"
                    },
                    # ✅ UNIQUE: Emerging trend filter (5.0-7.0 range)
                    "quality_emerging_trend": {
                        "gates": {
                            "trendStrength": {"min": 5.0, "max": 7.0},
                            "roe": {"min": 25},
                            "epsGrowth5y": {"min": 15}
                        },
                        "confidence_boost": 20,
                        "reason": "Quality company with emerging trend"
                    },
                    
                    # ============================================================
                    # ✅ MIGRATED FROM master_config.py (Multibagger)
                    # ============================================================
                    "quality_technical_setup": {
                        "gates": {"trendStrength": {"min": 4.0}},
                        "confidence_boost": 20.0,
                        "apply_to_setups": ["QUALITY_ACCUMULATION"],
                        "reason": "Quality stock in early accumulation"
                    },
                    "growth_combo": {
                        "gates": {
                            "epsGrowth5y": {"min": 25},
                            "marketCapCagr": {"min": 25}
                        },
                        "confidence_boost": 18.0,
                        "reason": "Consistent compounding + price performance"
                    },
                    "moat_indicators": {
                        "gates": {
                            "roic": {"min": 20},
                            "roe": {"min": 25},
                            "deRatio": {"max": 0.3}
                        },
                        "confidence_boost": 15.0,
                        "reason": "Economic moat indicators present"
                    },
                    "undiscovered_gem": {
                        "gates": {
                            "institutionalOwnership": {"max": 15},
                            "marketCap": {"max": 10000}
                        },
                        "confidence_boost": 12.0,
                        "reason": "Under-the-radar quality stock"
                    },
                    
                    # ========================================================
                    # MAXIMUM BUCKET BONUSES
                    # ========================================================
                    "perfect_compounder_multibagger": {
                        "gates": {
                            "fund_quality_bucket": {"min": 8.0},  # Exceptional quality
                            "fund_growth_bucket": {"min": 7.5},   # Strong growth
                            "fund_health_bucket": {"min": 7.5}    # Solid balance sheet
                        },
                        "confidence_boost": 30,  # Maximum bonus
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "VALUE_TURNAROUND",
                            "DEEP_VALUE_PLAY"
                        ],
                        "reason": "Perfect multibagger profile: exceptional quality + growth + financial health"
                    },
                    "megatrend_compounder": {
                        "gates": {
                            "trendStrength": {"min": 8.0},
                            "fund_quality_bucket": {"min": 8.0},
                            "fund_growth_bucket": {"min": 8.0},
                            "fund_health_bucket": {"min": 7.5}
                        },
                        "confidence_boost": 35,  # Maximum bonus
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "Strong multi-year trend + perfect fundamental profile"
                    }
                }
            },
            
            "adx_confidence_bands": {
                "explosive": {"gates": {"adx": {"min": 25}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 15, "max": 25}}, "confidence_boost": 12},
                "moderate": {"gates": {"adx": {"min": 8, "max": 15}}, "confidence_boost": 0}
            },
            "adx_confidence_penalties": {
                "weak": {
                    "gates": {"adx": {"max": 7}},  # Fixed: was 12, overlapped with moderate min:8 (ADX 8-12 got both)
                    "confidence_penalty": -10,
                    "reason": "Trend too weak for multibagger thesis"
                }
            },
            
            # Horizon-level tradeable floor: below this, signal is untradeable
            "min_tradeable_confidence": {
                "min": 50,
                "reason": "Multibagger ideas require very high conviction"
            },
            
            # High-confidence override: metadata for downstream consumers
            "high_confidence_override": {
                "threshold": 90,
                "can_override": {
                    "execution_warnings": False,  # No shortcuts for multibagger
                    "structural_gates": False,
                    "fundamental_gates": False
                },
                "max_override_count": 0,
                "log_overrides": True
            }
        }
    }
}

# ============================================================================
# CONFIDENCE CALCULATION ORDER (for documentation)
# ============================================================================
CONFIDENCE_CALCULATION_PIPELINE = """
1. Start with setup's baseline floor from global.setup_baseline_floors
2. Apply horizon.base_confidence_adjustment
3. Override with horizon.setup_floor_overrides if defined
4. Apply ADX normalization using horizon.adx_confidence_bands
5. Apply global.volume_modifiers (surge/drought/climax)
6. Apply global.universal_adjustments (divergence, trend strength)
7. Apply horizon.conditional_adjustments (penalties then bonuses)
8. Apply master_config.horizons.*.enhancements (horizon-universal bonuses)
9. Clamp result to horizon.confidence_clamp range
10. Final validation against entry_gates.opportunity.confidence.min

Notes:
- Penalties for divergence are MULTIPLICATIVE (severe = 0.5x)
- All other adjustments are ADDITIVE
- Bonuses/penalties are capped to prevent extreme values
- Some setups can be blocked (floor = None) per horizon
- Gates use structured format matching entry_gates
- Logic defaults to AND; use "_logic": "OR" inside gates when needed
- enhancements (master_config) apply AFTER conditional_adjustments (step 8 after step 7)
"""


# ============================================================================
# GRADUATED BUCKET USAGE SUMMARY
# ============================================================================
BUCKET_USAGE_SUMMARY = """
┌──────────────┬─────────────────┬──────────────────┬────────────────────────┐
│ Horizon      │ Penalties       │ Bonuses          │ Rationale              │
├──────────────┼─────────────────┼──────────────────┼────────────────────────┤
│ Intraday     │ 0 bucket        │ 1 light bonus    │ Technical only         │
│              │ (technical only)│ (+3 for quality) │ Don't filter on fund   │
├──────────────┼─────────────────┼──────────────────┼────────────────────────┤
│ Short Term   │ 1 extreme only  │ 1 light bonus    │ Block garbage only     │
│              │ (-12 if score<3)│ (+8 for quality) │ Allow speculation      │
├──────────────┼─────────────────┼──────────────────┼────────────────────────┤
│ Long Term    │ 2 strong        │ 2 strong         │ Profile matters        │
│              │ (-20 to -25)    │ (+18 to +25)     │ Quality + growth       │
├──────────────┼─────────────────┼──────────────────┼────────────────────────┤
│ Multibagger  │ 2 maximum       │ 2 maximum        │ Perfect only           │
│              │ (-30)           │ (+30 to +35)     │ Ultra-selective        │
└──────────────┴─────────────────┴──────────────────┴────────────────────────┘

KEY INSIGHT:
Bucket usage intensity scales with investment time horizon.
Don't apply long-term filters to short-term trades.
"""


# ============================================================================
# GATE EVALUATION GUIDE (for documentation)
# ============================================================================
GATE_EVALUATION_GUIDE = """
Structured Gate Format:

1. Simple Threshold:
   "gates": {
       "adx": {"min": 20}  # adx >= 20
   }

2. Range Check:
   "gates": {
       "trendStrength": {"min": 4.0, "max": 6.0}  # 4.0 <= trendStrength <= 6.0
   }

3. Multiple Conditions (AND - Default):
   "gates": {
       "rvol": {"min": 3.0},
       "wickRejection": {"max": 1.5}
       # Both must be true (default AND logic)
   }

4. Multiple Conditions (OR):
   "gates": {
       "roe": {"max": 15},
       "roce": {"max": 15},
       "_logic": "OR"  # Either being true triggers penalty
   }

5. Setup-Specific Application:
   "apply_to_setups": ["MOMENTUM_BREAKOUT"],  # Only applies to these setups
   "exclude_setups": ["VOLATILITY_SQUEEZE"]   # Doesn't apply to these

Evaluation Logic:
- If "_logic" not specified in gates, default to "AND"
- All thresholds in gates dict must pass (AND) or any must pass (OR)
- If "apply_to_setups" specified, only apply to those setups
- If "exclude_setups" specified, skip those setups
- "_logic" key is INSIDE the gates dict (prefixed with underscore)
"""