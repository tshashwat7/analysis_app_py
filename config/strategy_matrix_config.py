# config/strategy_matrix.py
"""
Strategy Classification Matrix - PRODUCTION VERSION v3.0

DESIGN PHILOSOPHY:
✓ Strategy DNA: Fit indicators, scoring rules, market cap filters
✓ Setup preferences: Which setups align with this strategy
✓ Horizon fit multipliers: How well this strategy fits each horizon
✗ DOES NOT DEFINE: Horizon blocking (that's master_config's job)

ARCHITECTURE PRINCIPLE:
Horizon > Strategy > Setup (dependency hierarchy)
- Horizons block strategies/setups (master_config)
- Strategies prefer setups (strategy_matrix)
- Setups require patterns (setup_pattern_matrix)

Author: Quantitative Trading System
Version: 3.0 (Production-Ready)
"""

from typing import Dict, List, Optional, Any

# ═══════════════════════════════════════════════════════════════════════
# STRATEGY CLASSIFICATION MATRIX
# ═══════════════════════════════════════════════════════════════════════

STRATEGY_MATRIX = {
    "swing_trading": {
        "description": "Captures multi-day price swings (3-10 days)",
        "enabled": True,
        "fit_threshold": 50,
        
        "fit_indicators": {
            "trendStrength": {"min": 4.0, "weight": 0.3},
            "volatilityQuality": {"min": 5.0, "weight": 0.3},
            "adx": {"min": 20, "weight": 0.2}
        },
        
        "scoring_rules": {
            "price_near_bb_low": {
                "gates": {"bbpercentb": {"max": 0.05}},
                "points": 35,
                "reason": "Price near Buy Zone (BB Low)"
            },
            "rsi_dip": {
                "gates": {"rsi": {"max": 45}},
                "points": 25,
                "reason": "RSI Oversold/Dip"
            },
            "double_bottom_reversal": {
                "gates": {
                    "double_top_bottom_found": {"equals": True},
                    "double_top_bottom_type": {"equals": "bullish"}
                },
                "points": 40,
                "reason": "Double Bottom Reversal Pattern"
            },
            "squeeze_compression": {
                "gates": {"ttmSqueeze": {"equals": "Squeeze On"}},
                "points": 10,
                "reason": "Volatility Squeeze Active"
            }
        },
        
        # ✅ Strategy preferences (NOT horizon-specific)
        "preferred_setups": ["TREND_PULLBACK", "DEEP_PULLBACK", "PATTERN_FLAG_BREAKOUT"],
        "avoid_setups": ["QUALITY_ACCUMULATION"],
        
        # ✅ Horizon fit multipliers (how well strategy fits each horizon)
        "horizon_fit_multipliers": {
            "intraday": 0.7,        # Not ideal for swing
            "short_term": 1.2,      # Best fit
            "long_term": 1.1        # Good fit
        },
        
        # Max positive bonus points (sum of all positive scoring_rules points)
        # Used by resolver to normalize bonus to 0-100 before blending
        "scoring_rules_max_bonus": 110,  # 35+25+40+10

        "notes": "Best for 3-10 day holds with clear trend structure"
    },

    "day_trading": {
        "description": "Intraday scalping and momentum plays",
        "enabled": True,
        "fit_threshold": 70,
        
        "fit_indicators": {
            "momentumStrength": {"min": 6.0, "weight": 0.4},
            "rvol": {"min": 2.0, "weight": 0.3},
            "volatilityQuality": {"min": 6.0, "weight": 0.3}
        },
        
        "scoring_rules": {
            "volume_surge": {
                "gates": {"rvol": {"min": 1.5}},
                "points": 25,
                "reason": "Strong volume surge"
            },
            "intraday_volatility": {
                "gates": {"atrPct": {"min": 1.5}},
                "points": 15,
                "reason": "Sufficient intraday range"
            },
            "three_line_strike": {
                "gates": {"three_line_strike_found": {"equals": True}},
                "points": 40,
                "reason": "3-Line Strike Reversal"
            }
        },
        "market_cap_requirements": {
            "small_cap": {
                "min_market_cap": 5000,  # ₹5000cr minimum for intraday
                "fit_threshold_override": 80,
                "reason": "Liquidity risk in intraday scalping"
            },
            "mid_large_cap": {
                "min_market_cap": 10000,
                "fit_threshold": 70
            }
        },

        "preferred_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE", "PATTERN_DARVAS_BREAKOUT"],
        "avoid_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
        
        "horizon_fit_multipliers": {
            "intraday": 1.3,        # Perfect fit
            "short_term": 0.8,
            "long_term": 0.4
        },
        
        "scoring_rules_max_bonus": 80,  # 25+15+40

        "notes": "Requires high volume and fast momentum - fundamentals ignored"
    },

    "trend_following": {
        "description": "Classic trend alignment with MA, ADX, and Momentum",
        "enabled": True,
        "fit_threshold": 70,  # ⬆️ Increased from 60 to prevent acting as generic catch-all
        
        "fit_indicators": {
            "adx": {"min": 30, "weight": 0.25},  # ⬆️ Increased from 25
            "trendStrength": {"min": 6.5, "weight": 0.25}, # ⬆️ Increased from 6.0
            "rsi": {"min": 50, "weight": 0.25}, # 🟢 Added to ensure some momentum exists
            "priceVsMaSlowPct": {"min": 0, "weight": 0.25}
        },
        
        "scoring_rules": {
            "ma_alignment": {
                "gates": {
                    "price": {"min_metric": "maFast"},
                    "maFast": {"min_metric": "maMid"},
                    "maMid": {"min_metric": "maSlow"}
                },
                "points": 30,
                "reason": "Bullish MA Alignment"
            },
            "strong_adx": {
                "gates": {"adx": {"min": 25}},
                "points": 20,
                "reason": "Strong Trend (ADX)"
            },
            "ichimoku_signal": {
                "gates": {"ichimoku_signals_found": {"equals": True}},
                "points": 25,
                "reason": "Ichimoku Cloud Signal"
            },
            "golden_cross_bullish": {
                "gates": {
                    "golden_cross_found": {"equals": True},
                    "golden_cross_type": {"equals": "bullish"}
                },
                "points": 25,
                "reason": "Golden Cross (Major Trend)"
            }
        },
        
        "preferred_setups": ["TREND_PULLBACK", "PATTERN_GOLDEN_CROSS", "MOMENTUM_BREAKOUT"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.7,
            "short_term": 1.1,
            "long_term": 1.0
        },
        
        "scoring_rules_max_bonus": 100,  # 30+20+25+25

        "notes": "Follows established trends with technical confirmation"
    },

    "momentum": {
        "description": "Trend velocity and breakout continuation",
        "enabled": True,
        "fit_threshold": 65,
        
        "fit_indicators": {
            "rsi": {"min": 60, "weight": 0.4},
            "maFastSlope": {"min": 10, "weight": 0.3},
            "adx": {"min": 25, "weight": 0.3}
        },
        
        "scoring_rules": {
            "high_velocity": {
                "gates": {"maFastSlope": {"min": 20}},
                "points": 40,
                "reason": "High-velocity trend"
            },
            "rsi_momentum_zone": {
                "gates": {"rsi": {"min": 65, "max": 80}},
                "points": 30,
                "reason": "Bullish momentum zone"
            },
            "strong_trend": {
                "gates": {"adx": {"min": 30}},
                "points": 30,
                "reason": "Established strong trend"
            }
        },
        
        "preferred_setups": ["MOMENTUM_BREAKOUT", "PATTERN_DARVAS_BREAKOUT", "PATTERN_FLAG_BREAKOUT", "VOLATILITY_SQUEEZE"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 1.2,
            "short_term": 1.15,
            "long_term": 0.7
        },
        
        "scoring_rules_max_bonus": 100,  # 40+30+30

        "notes": "Fast-moving markets with established velocity"
    },

    "minervini_growth": {
        "description": "Mark Minervini's growth stock method (Stage 2 + VCP)",
        "enabled": True,
        "fit_threshold": 65,
        
        "fit_indicators": {
            "epsGrowth5y": {"min": 15, "weight": 0.3},       # 25→15: Indian large-caps rarely sustain 25%+ EPS growth
            "relStrengthNifty": {"min": 0, "weight": 0.3},    # 1.2→0: Outperforming Nifty at all (alpha > 0) is meaningful
            "trendStrength": {"min": 6.0, "weight": 0.2},
            "volatilityQuality": {"min": 5.0, "weight": 0.2}
        },
        
        # ✅ Market cap requirements (Indian context)
        "market_cap_requirements": {
            "micro_cap": {
                "max_market_cap": 1000,
                "fit_threshold_override": 80,
                "reason": "High manipulation risk in micro-caps",
                "additional_gates": {
                    "min_delivery_pct": 60,
                    "min_institutional_ownership": 10
                }
            },
            "small_cap": {
                "min_market_cap": 1000,
                "max_market_cap": 10000,
                "fit_threshold_override": 70,
                "reason": "Moderate risk - higher bar"
            },
            "mid_large_cap": {
                "min_market_cap": 10000,
                "fit_threshold": 65,
                "reason": "Lower manipulation risk"
            }
        },
        
        "scoring_rules": {
            "vcp_confirmed": {
                "gates": {"minervini_stage2_found": {"equals": True}},
                "points": 50,
                "reason": "VCP Pattern Confirmed"
            },
            "stage2_alignment": {
                "gates": {
                    "price": {"min_metric": "maMid"},
                    "maMid": {"min_metric": "maSlow"}
                },
                "points": 20,
                "reason": "Stage 2 Trend Alignment"
            },
            "relative_strength": {
                "gates": {"relStrengthNifty": {"min": 0.001}},
                "points": 20,
                "reason": "Outperforming Market"
            },
            "near_52w_high": {
                "gates": {"position52w": {"min": 85}},
                "points": 20,
                "reason": "Near 52W Highs"
            },
            "deep_in_base_penalty": {
                "gates": {"position52w": {"max": 49.99}},
                "points": -20,
                "reason": "Deep in base (wait for breakout)"
            }
        },
        
        "preferred_setups": ["PATTERN_VCP_BREAKOUT", "PATTERN_CUP_BREAKOUT", "TREND_PULLBACK"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.5,
            "short_term": 1.1,
            "long_term": 1.1
        },
        
        "scoring_rules_max_bonus": 110,  # 50+20+20+20 (excludes -20 penalty)

        "notes": "Requires Stage 2 confirmation + growth fundamentals. Market cap filter prevents manipulation."
    },

    "canslim": {
        "description": "William O'Neil Growth Method",
        "enabled": True,
        "fit_threshold": 65,
        
        "fit_indicators": {
            "quarterlyGrowth": {"min": 15, "weight": 0.3},            # 25→15: More realistic for Indian quarterly cycles
            "epsGrowth5y": {"min": 15, "weight": 0.3},               # 20→15: Aligned with minervini_growth
            "relStrengthNifty": {"min": 0, "weight": 0.2},           # 1.0→0: Any positive alpha is meaningful
            "institutionalOwnership": {"min": 10, "weight": 0.2}     # 20→10: Indian FII+DII combined often 10-25%
        },
        
        "scoring_rules": {
            "earnings_acceleration": {
                "gates": {"quarterlyGrowth": {"min_metric": "epsGrowth5y"}},
                "points": 30,
                "reason": "Accelerating Earnings (C+A)"
            },
            "near_52w_highs": {
                "gates": {"position52w": {"min": 90}},
                "points": 25,
                "reason": "Breaking to New Highs (N)"
            },
            "cup_handle_pattern": {
                "gates": {"cup_handle_found": {"equals": True}},
                "points": 30,
                "reason": "Cup & Handle Pattern (N)"
            },
            "market_leader": {
                "gates": {"relStrengthNifty": {"min": 1.2}},
                "points": 25,
                "reason": "Market Leader (L)"
            },
            "institutional_base": {
                "gates": {"institutionalOwnership": {"min": 20, "max": 60}},
                "points": 20,
                "reason": "Stable institutional sponsorship (I)"
            }
        },
        
        "preferred_setups": ["PATTERN_CUP_BREAKOUT", "MOMENTUM_BREAKOUT", "PATTERN_FLAG_BREAKOUT"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.4,
            "short_term": 1.05,
            "long_term": 1.05
        },
        
        "scoring_rules_max_bonus": 130,  # 30+25+30+25+20

        "notes": "4-5 aligned factors is sufficient - perfect 7/7 is rare"
    },

    "value_investing": {
        "description": "Long-term value accumulation with sector context",
        "enabled": True,
        "fit_threshold": 50,
        
        "fit_indicators": {
            "peVsSector": {"max": 0.90, "weight": 0.3, "direction": "invert"},  # 0.85→0.90: 10% sector discount is still value
            "roe": {"min": 12, "weight": 0.3},                                  # 15→12: Value stocks often have lower ROE
            "deRatio": {"max": 0.7, "weight": 0.2, "direction": "invert"},      # 0.5→0.7: Indian capital-heavy sectors need room
            "fcfYield": {"min": 3.0, "weight": 0.2}                              # 5.0→3.0: FCF yield ≥ 3% is meaningful in India
        },
        "market_cap_requirements": {
            "small_cap": {
                "min_market_cap": 1000,    # 1000cr minimum
                "max_market_cap": 10000,
                "fit_threshold_override": 60,
                "reason": "Value investing needs liquidity"
            },
            "mid_large_cap": {
                "min_market_cap": 10000,
                "fit_threshold": 50,
                "reason": "Standard value investing"
            }
        },
        "scoring_rules": {
            "cheap_vs_sector": {
                "gates": {"peVsSector": {"min": 0.001, "max": 0.8}, "peRatio": {"min": 0.001}},
                "points": 35,
                "reason": "Trading below sector average (cheap)"
            },
            "cheap_pb": {
                "gates": {"pbRatio": {"min": 0.001, "max": 1.5}},
                "points": 25,
                "reason": "Low Price to Book"
            },
            "strong_roe": {
                "gates": {"roe": {"min": 15}},
                "points": 20,
                "reason": "Strong Return on Equity"
            },
            "low_debt": {
                "gates": {"deRatio": {"max": 0.5}},
                "points": 20,
                "reason": "Conservative Debt Levels"
            },
            "sector_discount": {
                "gates": {"peVsSector": {"min": 0.001, "max": 0.7}},
                "points": 10,
                "reason": "Deep discount vs sector (30%+ cheaper)"
            }
        },
        
        "preferred_setups": ["QUALITY_ACCUMULATION", "DEEP_VALUE_PLAY", "VALUE_TURNAROUND"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,        # Blocked
            "short_term": 0.85,
            "long_term": 1.3
        },
        
        "scoring_rules_max_bonus": 110,  # 35+25+20+20+10

        "notes": "Uses sector-relative PE instead of absolute bands"
    },

    "income_investing": {
        "description": "Dividend yield and cash flow focus",
        "enabled": True,
        "fit_threshold": 60,
        
        "fit_indicators": {
            "dividendyield": {"min": 1.5, "weight": 0.5},    # 3.0→1.5: Indian growth market; 1.5%+ is solid income
            "fcfYield": {"min": 3.0, "weight": 0.3},         # 5.0→3.0: Aligned with value_investing
            "deRatio": {"max": 0.7, "weight": 0.2, "direction": "invert"}
        },
        
        "scoring_rules": {
            "high_yield": {
                "gates": {"dividendyield": {"min": 3.0}},
                "points": 40,
                "reason": "Attractive Dividend Yield"
            },
            "fcf_strength": {
                "gates": {"fcfYield": {"min": 5.0}},
                "points": 30,
                "reason": "Strong Free Cash Flow"
            },
            "safe_payout": {
                "gates": {"dividendPayout": {"min": 0.001, "max": 60}},
                "points": 30,
                "reason": "Sustainable Payout Ratio"
            }
        },
        
        "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 0.7,
            "long_term": 1.2
        },
        
        "scoring_rules_max_bonus": 100,  # 40+30+30

        "notes": "Focus on stable cash generation and dividend safety"
    },

    "position_trading": {
        "description": "Long-term trend following for major cyclic moves",
        "enabled": True,
        "fit_threshold": 50,
        
        "fit_indicators": {
            "trendStrength": {"min": 6.0, "weight": 0.4},
            "priceVsPrimaryTrendPct": {"min": 0, "weight": 0.4},
            "volatilityQuality": {"min": 6.0, "weight": 0.2}
        },
        
        "scoring_rules": {
            "golden_cross": {
                "gates": {
                    "golden_cross_found": {"equals": True},
                    "golden_cross_type": {"equals": "bullish"}
                },
                "points": 50,
                "reason": "Primary Trend Reversal (Golden Cross)"
            },
            "stable_volatility": {
                "gates": {"volatilityQuality": {"min": 7.0}},
                "points": 25,
                "reason": "Stable, high-quality trend"
            },
            "ma_alignment": {
                "gates": {
                    "price": {"min_metric": "maMid"},
                    "maMid": {"min_metric": "maSlow"}
                },
                "points": 25,
                "reason": "Triple MA Alignment"
            }
        },
        
        "preferred_setups": ["PATTERN_GOLDEN_CROSS", "TREND_PULLBACK", "QUALITY_ACCUMULATION"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 0.8,
            "long_term": 1.25
        },
        
        "scoring_rules_max_bonus": 100,  # 50+25+25

        "notes": "Patient capital seeking major trend cycles"
    },

    "quality_growth": {
        "description": "Quality compounders with sustainable growth at reasonable valuations",
        "enabled": True,
        "fit_threshold": 55,
        
        "fit_indicators": {
            "roe": {"min": 15, "weight": 0.20},                                    # 18→15: Wider quality net
            "roce": {"min": 18, "weight": 0.20},                                   # 20→18: Slightly relaxed
            "epsGrowth5y": {"min": 12, "weight": 0.15},                             # 15→12: Compounders often grow 12-18%
            "revenueGrowth5y": {"min": 10, "weight": 0.10},                         # 12→10: Mature quality companies
            "deRatio": {"max": 0.7, "weight": 0.10, "direction": "invert"},
            "trendStrength": {"min": 4.0, "weight": 0.10},
            "pegRatio": {"max": 2.5, "weight": 0.10, "direction": "invert"},       # 2.0→2.5: Quality premium is real in India
            "earningsStability": {"min": 6.0, "weight": 0.05}                       # 6.5→6.0: Slight relaxation
        },
        
        "scoring_rules": {
            "consistent_quality": {
                "gates": {"roe": {"min": 18}, "roce": {"min": 20}, "roe3yAvg": {"min": 18}},
                "points": 35,
                "reason": "Consistent quality metrics over 3 years"
            },
            "excellent_garp": {
                "gates": {"pegRatio": {"min": 0.001, "max": 1.5}},
                "points": 35,
                "reason": "Excellent GARP (PEG ≤ 1.5)"
            },
            "acceptable_garp": {
                "gates": {"pegRatio": {"min": 1.501, "max": 2.0}},
                "points": 20,
                "reason": "Acceptable GARP (PEG 1.5-2.0)"
            },
            "optimal_entry_zone": {
                "gates": {"trendStrength": {"min": 4.0, "max": 7.0}, "rsi": {"min": 50, "max": 65}},
                "points": 10,
                "reason": "In optimal entry zone - not overheated"
            },
            "balance_sheet_quality": {
                "gates": {"deRatio": {"max": 0.5}, "interestCoverage": {"min": 5}},
                "points": 20,
                "reason": "Conservative balance sheet"
            },
            "earnings_consistency": {
                "gates": {"earningsStability": {"min": 7.0}},
                "points": 15,
                "reason": "Stable, predictable earnings"
            }
        },
        
        "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND", "TREND_PULLBACK", "DEEP_PULLBACK"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 1.1,
            "long_term": 1.35
        },
        
        "scoring_rules_max_bonus": 135,  # 35+35+20+10+20+15 (excludes -10 penalty)

        "notes": "Sweet spot between value and momentum. Adjusted for quality compounders staying overbought."
    },

    "reversal_trading": {
        "description": "Bottom fishing in quality stocks after corrections",
        "enabled": True,
        "fit_threshold": 55,
        
        "fit_indicators": {
            "trendStrength": {"max": 4.0, "weight": 0.25, "direction": "invert"},
            "rsi": {"max": 40, "weight": 0.25, "direction": "invert"},
            "roe": {"min": 15, "weight": 0.20},
            "deRatio": {"max": 0.8, "weight": 0.15, "direction": "invert"},
            "priceVs52wHighPct": {"max": 80, "weight": 0.15, "direction": "invert"}
        },
        
        "scoring_rules": {
            "quality_in_distress": {
                "gates": {"roe": {"min": 18}, "roce": {"min": 20}, "priceVs52wHighPct": {"max": 75}},
                "points": 40,
                "reason": "Quality stock 25%+ off highs"
            },
            "oversold_rsi": {
                "gates": {"rsi": {"max": 35}},
                "points": 25,
                "reason": "Deeply oversold"
            },
            "bullish_divergence": {
                "gates": {"rsislope": {"min": 0.05}, "price_slope": {"max": -0.01}},
                "points": 30,
                "reason": "Bullish divergence (Price falling but RSI rising)"
            },
            "consolidation_base": {
                "gates": {"bbWidth": {"max": 5.0}, "volatilityQuality": {"min": 5.0}},
                "points": 20,
                "reason": "Tight consolidation after fall"
            },
            "reversal_confirmation": {
                "gates": {"macdhistogram": {"min": 0.001}, "supertrendSignal": {"equals": "Bullish"}},
                "points": 20,
                "reason": "Technical reversal confirmed"
            },
            "still_falling_penalty": {
                "gates": {"trendStrength": {"max": 2.0}, "momentumStrength": {"max": 2.0}},
                "points": -25,
                "reason": "Still in freefall - wait for stabilization"
            },
            "poor_quality_penalty": {
                "gates": {"_logic": "OR", "roe": {"max": 11.99}, "deRatio": {"min": 1.001}},
                "points": -20,
                "reason": "Quality too low for reversal play"
            }
        },
        
        "preferred_setups": ["REVERSAL_MACD_CROSS_UP", "REVERSAL_RSI_SWING_UP", "REVERSAL_ST_FLIP_UP", "DEEP_PULLBACK"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "TREND_PULLBACK"],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 1.1,
            "long_term": 1.2
        },
        
        "scoring_rules_max_bonus": 135,  # 40+25+30+20+20 (excludes -25 -20 penalties)

        "notes": "Requires patience and quality floor (ROE ≥ 15, low debt)"
    }
}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS (Same as before)
# ═══════════════════════════════════════════════════════════════════════

def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """Get complete configuration for a strategy."""
    return STRATEGY_MATRIX.get(strategy_name, {})


def get_all_enabled_strategies() -> List[str]:
    """Get list of all enabled strategy names."""
    return [
        name for name, config in STRATEGY_MATRIX.items()
        if config.get("enabled", False)
    ]


def get_strategy_horizon_multiplier(strategy_name: str, horizon: str) -> float:
    """
    Get horizon fit multiplier for a strategy.
    
    Returns:
        Multiplier value (1.0 = neutral, >1.0 = boosted, 0.0 = blocked)
    """
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    multipliers = strategy.get("horizon_fit_multipliers", {})
    return multipliers.get(horizon, 1.0)


def calculate_strategy_fit_score(
    strategy_name: str,
    indicators: Dict[str, float],
    fundamentals: Dict[str, float]
) -> float:
    """
    Calculate fit score for a strategy based on indicators & fundamentals.
    
    Returns:
        Fit score (0-100)
    """
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    if not strategy or not strategy.get("enabled", False):
        return 0.0

    fit_indicators = strategy.get("fit_indicators", {})
    if not fit_indicators:
        return 0.0

    total_weight = 0.0
    weighted_score = 0.0

    for indicator, params in fit_indicators.items():
        weight = params.get("weight", 0.1)
        min_val = params.get("min")
        max_val = params.get("max")
        direction = params.get("direction", "normal")

        # Get actual value
        actual = indicators.get(indicator) or fundamentals.get(indicator)
        if actual is None:
            total_weight += weight  # ✅ Missing data = FAILED, not skipped
            continue

        total_weight += weight

        # Check threshold
        threshold_met = True
        if direction == "invert":
            if max_val is not None and actual > max_val:
                threshold_met = False
        else:
            if min_val is not None and actual < min_val:
                threshold_met = False
            if max_val is not None and actual > max_val:
                threshold_met = False

        if threshold_met:
            weighted_score += weight

    if total_weight == 0:
        return 0.0

    return (weighted_score / total_weight) * 100


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy_matrix() -> Dict[str, Any]:
    """Validate strategy matrix for consistency."""
    errors = []
    warnings = []

    for strategy_name, config in STRATEGY_MATRIX.items():
        if "description" not in config:
            errors.append(f"{strategy_name}: Missing 'description'")

        if "fit_threshold" not in config:
            warnings.append(f"{strategy_name}: Missing 'fit_threshold'")

        if "fit_indicators" not in config or not config["fit_indicators"]:
            errors.append(f"{strategy_name}: Missing or empty 'fit_indicators'")

        # Check horizon multipliers
        if "horizon_fit_multipliers" in config:
            multipliers = config["horizon_fit_multipliers"]
            for horizon in ["intraday", "short_term", "long_term"]:
                if horizon not in multipliers:
                    warnings.append(f"{strategy_name}: Missing multiplier for '{horizon}'")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


if __name__ == "__main__":
    import json
    
    print("="*80)
    print("STRATEGY MATRIX V3.0 VALIDATION")
    print("="*80)
    
    validation = validate_strategy_matrix()
    