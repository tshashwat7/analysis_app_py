# config/confidence_config.py (DEDUPLICATED VERSION)
"""
Confidence Calculation Configuration
Separated from master_config for clarity and maintainability

DESIGN PHILOSOPHY:
- Buckets usage scales with time horizon
- Intraday: Pure technical (no fundamental filtering)
- Short term: Light filtering (only extreme cases)
- Long term: Heavy filtering (profile matters)
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
                        "rsiSlope": {"max": -0.08}
                    },
                    "block_entry": True,
                    "reason": "Severe bearish divergence"
                },
                "moderate": {
                    "gates": {
                        "rsiSlope": {"max": -0.03, "min": -0.08}
                    },
                    "confidence_multiplier": 0.70,
                    "block_entry": False,
                    "reason": "Moderate bearish divergence"
                },
                "minor": {
                    "gates": {
                        "rsiSlope": {"max": 0.0, "min": -0.03}
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
                    "exclude_setups": ["MOMENTUM_BREAKDOWN", "BEAR_TREND_FOLLOWING"],
                    "reason": "Explosive trend momentum"
                },
                "strong": {
                    "gates": {
                        "trendStrength": {"min": 6.0, "max": 7.99}
                    },
                    "confidence_boost": 15,
                    "exclude_setups": ["MOMENTUM_BREAKDOWN", "BEAR_TREND_FOLLOWING"],
                    "reason": "Strong sustained trend"
                },
                "moderate": {
                    "gates": {
                        "trendStrength": {"min": 4.0, "max": 5.99}
                    },
                    "confidence_boost": 5,
                    "reason": "Moderate trend support"
                },
                "weak": {
                    "gates": {
                        "trendStrength": {"max": 3.99}  # Fixed W3: closed dead zone
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
            "MOMENTUM_FLOW_BREAKDOWN": 55,
            "MOMENTUM_FLOW_CONTINUATION": 52,
            "VOLATILITY_SQUEEZE": 50,
            "QUALITY_ACCUMULATION": 45,
            "QUALITY_ACCUMULATION_DOWNTREND": 40,
            "DEEP_VALUE_PLAY": 40,
            "VALUE_TURNAROUND": 50,
            "TREND_PULLBACK": 50,
            "DEEP_PULLBACK": 48,
            "TREND_FOLLOWING": 50,
            "BEAR_TREND_FOLLOWING": 48,
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
                        "gates": {"adx": {"max": 19}},  # ✅ Corrected from 20; aligns with moderate band min:20
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
                "strong": {"gates": {"adx": {"min": 25, "max": 34.9}}, "confidence_boost": 10},
                "moderate": {"gates": {"adx": {"min": 20, "max": 24.9}}, "confidence_boost": 0}  # ✅ Raised from 18 to match structural gate
            },
            "adx_confidence_penalties": {
                "weak": {
                    "gates": {"adx": {"max": 19}},  # ✅ Raised: closes dead zone 18-19; pairs with moderate min:20
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
                        "apply_to_setups": ["TREND_PULLBACK", "TREND_FOLLOWING", "BEAR_TREND_FOLLOWING"],
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
                    # (same rsiSlope range, was causing double-penalty: 0.70× multiplier + -15 additive)
                    
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
                    },
                    "sector_headwind_continuation": {
                        "gates": {"sectorTrendScore": {"max": -5.0}},
                        "apply_to_setups": [
                            "TREND_PULLBACK",
                            "TREND_FOLLOWING",
                            "MOMENTUM_BREAKOUT",
                            "MOMENTUM_FLOW_CONTINUATION",
                            "PATTERN_DARVAS_BREAKOUT",
                            "PATTERN_VCP_BREAKOUT",
                            "PATTERN_FLAG_BREAKOUT",
                            "PATTERN_CUP_BREAKOUT"
                        ],
                        "confidence_penalty": -12,
                        "reason": "Sector trend is fighting bullish continuation setup"
                    },
                    "sector_laggard_breakout": {
                        "gates": {
                            "sectorTrendScore": {"min": 4.0},
                            "rsVsSectorFast": {"max": -2.0}
                        },
                        "apply_to_setups": [
                            "MOMENTUM_BREAKOUT",
                            "MOMENTUM_FLOW_CONTINUATION",
                            "PATTERN_DARVAS_BREAKOUT",
                            "PATTERN_VCP_BREAKOUT",
                            "PATTERN_FLAG_BREAKOUT",
                            "PATTERN_CUP_BREAKOUT"
                        ],
                        "confidence_penalty": -8,
                        "reason": "Sector is healthy but stock is lagging its peers"
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
                        "gates": {"patternCount": {"min": 2}},
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
                    },
                    "bearish_conviction": {
                        "gates": {
                            "deathCross": {"min": 6.0},
                            "bearishNecklinePattern": {"min": 6.0},
                            "_logic": "OR"
                        },
                        "confidence_boost": 10,
                        "apply_to_setups": ["MOMENTUM_BREAKDOWN", "MOMENTUM_FLOW_BREAKDOWN", "BEAR_TREND_FOLLOWING"],
                        "reason": "Strong bearish pattern confirmation"
                    },
                    "leading_stock_in_leading_sector": {
                        "gates": {
                            "sectorTrendScore": {"min": 7.0},
                            "rsVsSectorFast": {"min": 2.0}
                        },
                        "confidence_boost": 10,
                        "apply_to_setups": [
                            "TREND_PULLBACK",
                            "TREND_FOLLOWING",
                            "MOMENTUM_BREAKOUT",
                            "MOMENTUM_FLOW_CONTINUATION",
                            "PATTERN_DARVAS_BREAKOUT",
                            "PATTERN_VCP_BREAKOUT",
                            "PATTERN_FLAG_BREAKOUT",
                            "PATTERN_CUP_BREAKOUT"
                        ],
                        "reason": "Leading stock in a leading sector"
                    },
                    "persistent_sector_leadership": {
                        "gates": {
                            "rsVsSectorFast": {"min": 1.5},
                            "rsVsSectorSlow": {"min": 1.0}
                        },
                        "confidence_boost": 6,
                        "apply_to_setups": [
                            "TREND_FOLLOWING",
                            "TREND_PULLBACK",
                            "QUALITY_ACCUMULATION",
                            "MOMENTUM_FLOW_CONTINUATION"
                        ],
                        "reason": "Stock is consistently outperforming its sector"
                    }
                }
            },
            
            "adx_confidence_bands": {
                "explosive": {"gates": {"adx": {"min": 30}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 22, "max": 29.9}}, "confidence_boost": 12},
                "moderate": {"gates": {"adx": {"min": 15, "max": 21.9}}, "confidence_boost": 0}
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
                            "_logic": "OR"  # Fixed: was AND — matches intent for insufficient_growth
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
                    },
                    "sector_headwind_position_trade": {
                        "gates": {"sectorTrendScore": {"max": -4.0}},
                        "confidence_penalty": -10,
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "VALUE_TURNAROUND",
                            "TREND_FOLLOWING",
                            "DEEP_VALUE_PLAY"
                        ],
                        "reason": "Sector trend is a headwind for a patient long thesis"
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
                    },
                    "sector_compounder_tailwind": {
                        "gates": {
                            "sectorTrendScore": {"min": 6.0},
                            "rsVsSectorSlow": {"min": 2.0}
                        },
                        "confidence_boost": 10,
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "TREND_FOLLOWING",
                            "VALUE_TURNAROUND"
                        ],
                        "reason": "Long-term thesis supported by a favorable sector trend"
                    },
                    "durable_sector_leadership": {
                        "gates": {
                            "rsVsSectorFast": {"min": 1.0},
                            "rsVsSectorSlow": {"min": 2.5},
                            "fund_quality_bucket": {"min": 7.0}
                        },
                        "confidence_boost": 8,
                        "apply_to_setups": [
                            "QUALITY_ACCUMULATION",
                            "TREND_FOLLOWING",
                            "DEEP_VALUE_PLAY"
                        ],
                        "reason": "Quality business is sustainably outperforming its sector"
                    }
                }
            },
            
            "adx_confidence_bands": {
                "explosive": {"gates": {"adx": {"min": 28}}, "confidence_boost": 20},
                "strong": {"gates": {"adx": {"min": 18, "max": 27.9}}, "confidence_boost": 12},
                "moderate": {"gates": {"adx": {"min": 12, "max": 17.9}}, "confidence_boost": 0}
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
            "confidence_philosophy": "Isolated multibagger pipeline support.",
            "confidence_clamp": [30, 95],
            "base_confidence_adjustment": 0,
            "setup_floor_overrides": {},
            "conditional_adjustments": {"penalties": {}, "bonuses": {}},
            "adx_confidence_bands": {},
            "adx_confidence_penalties": {},
            "min_tradeable_confidence": {"min": 30},
            "high_confidence_override": {"threshold": 90, "can_override": {}}
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
└──────────────┴─────────────────┴──────────────────┴────────────────────────┘

KEY INSIGHT:
Bucket usage intensity scales with investment time horizon.
Don't apply long-term filters to short-term trades.
"""
