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
                "condition": "bbLow > 0 and price <= bbLow * 1.02",
                "points": 35,
                "reason": "Price near Buy Zone (BB Low)"
            },
            "rsi_dip": {
                "condition": "rsi <= 45",
                "points": 25,
                "reason": "RSI Oversold/Dip"
            },
            "double_bottom_reversal": {
                "condition": "double_top_bottom_found == True and double_top_bottom_type == 'bullish'",
                "points": 40,
                "reason": "Double Bottom Reversal Pattern"
            },
            "squeeze_compression": {
                "condition": "ttmSqueeze == 'Squeeze On'",
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
            "long_term": 1.1,       # Good fit
            "multibagger": 0.5      # Poor fit
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
                "condition": "rvol >= 1.5",
                "points": 25,
                "reason": "Strong volume surge"
            },
            "intraday_volatility": {
                "condition": "atrPct >= 1.5",
                "points": 15,
                "reason": "Sufficient intraday range"
            },
            "three_line_strike": {
                "condition": "three_line_strike_found == True",
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
        # ✅ Indian market-specific gates (stays in strategy)
        "indian_market_gates": {
            "min_avg_volume": 500000,
            "max_spread_pct": 0.003,
            "min_delivery_pct": 40,
            "avoid_gsm": True,
            "time_filters": {
                "avoid_first_15_min": True,
                "avoid_last_15_min": True,
                "reduce_size_lunch": 0.5,
                "optimal_windows": [
                    {"start": "09:45", "end": "11:30", "multiplier": 1.0},
                    {"start": "13:30", "end": "15:00", "multiplier": 1.0}
                ]
            },
            "risk_controls": {
                "max_position_pct": 0.01,
                "mandatory_stop_loss": True,
                "max_trades_per_day": 3
            }
        },
        
        "preferred_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE", "PATTERN_DARVAS_BREAKOUT"],
        "avoid_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
        
        "horizon_fit_multipliers": {
            "intraday": 1.3,        # Perfect fit
            "short_term": 0.8,
            "long_term": 0.4,
            "multibagger": 0.0      # Incompatible
        },
        
        "scoring_rules_max_bonus": 80,  # 25+15+40

        "notes": "Requires high volume and fast momentum - fundamentals ignored"
    },

    "trend_following": {
        "description": "Classic trend alignment with MA and ADX",
        "enabled": True,
        "fit_threshold": 60,
        
        "fit_indicators": {
            "adx": {"min": 25, "weight": 0.4},
            "trendStrength": {"min": 6.0, "weight": 0.3},
            "priceVsMaSlowPct": {"min": 0, "weight": 0.3}
        },
        
        "scoring_rules": {
            "ma_alignment": {
                "condition": "price > maFast and maFast > maMid and maMid > maSlow",
                "points": 30,
                "reason": "Bullish MA Alignment"
            },
            "strong_adx": {
                "condition": "adx >= 25",
                "points": 20,
                "reason": "Strong Trend (ADX)"
            },
            "ichimoku_signal": {
                "condition": "ichimoku_signals_found == True",
                "points": 25,
                "reason": "Ichimoku Cloud Signal"
            },
            "golden_cross_bullish": {
                "condition": "golden_cross_found == True and golden_cross_type == 'bullish'",
                "points": 25,
                "reason": "Golden Cross (Major Trend)"
            }
        },
        
        "preferred_setups": ["TREND_PULLBACK", "PATTERN_GOLDEN_CROSS", "MOMENTUM_BREAKOUT"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.7,
            "short_term": 1.1,
            "long_term": 1.0,
            "multibagger": 0.8
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
                "condition": "maFastSlope >= 20",
                "points": 40,
                "reason": "High-velocity trend"
            },
            "rsi_momentum_zone": {
                "condition": "rsi >= 65 and rsi <= 80",
                "points": 30,
                "reason": "Bullish momentum zone"
            },
            "strong_trend": {
                "condition": "adx >= 30",
                "points": 30,
                "reason": "Established strong trend"
            }
        },
        
        "preferred_setups": ["MOMENTUM_BREAKOUT", "PATTERN_DARVAS_BREAKOUT", "PATTERN_FLAG_BREAKOUT", "VOLATILITY_SQUEEZE"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 1.2,
            "short_term": 1.15,
            "long_term": 0.7,
            "multibagger": 0.5
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
                "condition": "minervini_stage2_found == True",
                "points": 50,
                "reason": "VCP Pattern Confirmed"
            },
            "stage2_alignment": {
                "condition": "price > maMid and maMid > maSlow",
                "points": 20,
                "reason": "Stage 2 Trend Alignment"
            },
            "relative_strength": {
                "condition": "relStrengthNifty > 0",
                "points": 20,
                "reason": "Outperforming Market"
            },
            "near_52w_high": {
                "condition": "position52w >= 85",
                "points": 20,
                "reason": "Near 52W Highs"
            },
            "deep_in_base_penalty": {
                "condition": "position52w < 50",
                "points": -20,
                "reason": "Deep in base (wait for breakout)"
            }
        },
        
        "preferred_setups": ["PATTERN_VCP_BREAKOUT", "PATTERN_CUP_BREAKOUT", "TREND_PULLBACK"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.5,
            "short_term": 1.1,
            "long_term": 1.1,
            "multibagger": 1.35
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
                "condition": "quarterlyGrowth > epsGrowth5y",
                "points": 30,
                "reason": "Accelerating Earnings (C+A)"
            },
            "near_52w_highs": {
                "condition": "position52w >= 90",
                "points": 25,
                "reason": "Breaking to New Highs (N)"
            },
            "cup_handle_pattern": {
                "condition": "cup_handle_found == True",
                "points": 30,
                "reason": "Cup & Handle Pattern (N)"
            },
            "market_leader": {
                "condition": "relStrengthNifty >= 1.2",
                "points": 25,
                "reason": "Market Leader (L)"
            },
            "institutional_base": {
                "condition": "institutionalOwnership >= 20 and institutionalOwnership <= 60",
                "points": 20,
                "reason": "Stable institutional sponsorship (I)"
            }
        },
        
        "preferred_setups": ["PATTERN_CUP_BREAKOUT", "MOMENTUM_BREAKOUT", "PATTERN_FLAG_BREAKOUT"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.4,
            "short_term": 1.05,
            "long_term": 1.05,
            "multibagger": 1.25
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
                "condition": "peVsSector > 0 and peRatio > 0 and peVsSector < 0.8",
                "points": 35,
                "reason": "Trading below sector average (cheap)"
            },
            "cheap_pb": {
                "condition": "pbRatio > 0 and pbRatio < 1.5",
                "points": 25,
                "reason": "Low Price to Book"
            },
            "strong_roe": {
                "condition": "roe >= 15",
                "points": 20,
                "reason": "Strong Return on Equity"
            },
            "low_debt": {
                "condition": "deRatio <= 0.5",
                "points": 20,
                "reason": "Conservative Debt Levels"
            },
            "sector_discount": {
                "condition": "peVsSector > 0 and peVsSector < 0.7",
                "points": 10,
                "reason": "Deep discount vs sector (30%+ cheaper)"
            }
        },
        
        "preferred_setups": ["QUALITY_ACCUMULATION", "DEEP_VALUE_PLAY", "VALUE_TURNAROUND"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,        # Blocked
            "short_term": 0.85,
            "long_term": 1.3,
            "multibagger": 1.45
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
                "condition": "dividendyield >= 3.0",
                "points": 40,
                "reason": "Attractive Dividend Yield"
            },
            "fcf_strength": {
                "condition": "fcfYield >= 5.0",
                "points": 30,
                "reason": "Strong Free Cash Flow"
            },
            "safe_payout": {
                "condition": "dividendPayout > 0 and dividendPayout <= 60",
                "points": 30,
                "reason": "Sustainable Payout Ratio"
            }
        },
        
        "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 0.7,
            "long_term": 1.2,
            "multibagger": 1.1
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
                "condition": "golden_cross_found == True and golden_cross_type == 'bullish'",
                "points": 50,
                "reason": "Primary Trend Reversal (Golden Cross)"
            },
            "stable_volatility": {
                "condition": "volatilityQuality >= 7.0",
                "points": 25,
                "reason": "Stable, high-quality trend"
            },
            "ma_alignment": {
                "condition": "price > maMid and maMid > maSlow",
                "points": 25,
                "reason": "Triple MA Alignment"
            }
        },
        
        "preferred_setups": ["PATTERN_GOLDEN_CROSS", "TREND_PULLBACK", "QUALITY_ACCUMULATION"],
        "avoid_setups": [],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 0.8,
            "long_term": 1.25,
            "multibagger": 1.2
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
                "condition": "roe >= 18 and roce >= 20 and roe3yAvg >= 18",
                "points": 35,
                "reason": "Consistent quality metrics over 3 years"
            },
            "excellent_garp": {
                "condition": "pegRatio > 0 and pegRatio <= 1.5",
                "points": 35,
                "reason": "Excellent GARP (PEG ≤ 1.5)"
            },
            "acceptable_garp": {
                "condition": "pegRatio > 1.5 and pegRatio <= 2.0",
                "points": 20,
                "reason": "Acceptable GARP (PEG 1.5-2.0)"
            },
            "optimal_entry_zone": {
                "condition": "trendStrength >= 4.0 and trendStrength <= 7.0 and rsi >= 50 and rsi <= 65",
                "points": 10,
                "reason": "In optimal entry zone - not overheated"
            },
            "balance_sheet_quality": {
                "condition": "deRatio <= 0.5 and interestCoverage >= 5",
                "points": 20,
                "reason": "Conservative balance sheet"
            },
            "earnings_consistency": {
                "condition": "earningsStability >= 7.0",
                "points": 15,
                "reason": "Stable, predictable earnings"
            },
            "overheated_penalty": {
                "condition": "trendStrength >= 9.0 or rsi >= 80",
                "points": -10,
                "reason": "Extremely overextended - wait for pullback"
            },
            "multibagger_zone": {
                "condition": "roe >= 22 and epsGrowth5y >= 20 and pegRatio <= 1.5 and peRatio <= 30",
                "points": 20,
                "reason": "In multibagger zone (HDFC/Asian Paints profile)"
            }
        },
        
        "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND", "TREND_PULLBACK", "DEEP_PULLBACK"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 1.1,
            "long_term": 1.35,
            "multibagger": 1.5
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
            "price_vs_52w_high_pct": {"max": 80, "weight": 0.15, "direction": "invert"}
        },
        
        "scoring_rules": {
            "quality_in_distress": {
                "condition": "roe >= 18 and roce >= 20 and price_vs_52w_high_pct <= 75",
                "points": 40,
                "reason": "Quality stock 25%+ off highs"
            },
            "oversold_rsi": {
                "condition": "rsi <= 35",
                "points": 25,
                "reason": "Deeply oversold"
            },
            "bullish_divergence": {
                "condition": "rsislope >= 0.05 and price_slope <= -0.01",
                "points": 30,
                "reason": "Bullish divergence (Price falling but RSI rising)"
            },
            "consolidation_base": {
                "condition": "bbWidth <= 5.0 and volatilityQuality >= 5.0",
                "points": 20,
                "reason": "Tight consolidation after fall"
            },
            "reversal_confirmation": {
                "condition": "macdhistogram > 0 and supertrendSignal == 'Bullish'",
                "points": 20,
                "reason": "Technical reversal confirmed"
            },
            "still_falling_penalty": {
                "condition": "trendStrength <= 2.0 and momentumStrength <= 2.0",
                "points": -25,
                "reason": "Still in freefall - wait for stabilization"
            },
            "poor_quality_penalty": {
                "condition": "roe < 12 or deRatio > 1.0",
                "points": -20,
                "reason": "Quality too low for reversal play"
            }
        },
        
        "preferred_setups": ["REVERSAL_MACD_CROSS_UP", "REVERSAL_RSI_SWING_UP", "REVERSAL_ST_FLIP_UP", "DEEP_PULLBACK"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "TREND_PULLBACK"],
        
        "horizon_fit_multipliers": {
            "intraday": 0.0,
            "short_term": 1.1,
            "long_term": 1.2,
            "multibagger": 1.05
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
            for horizon in ["intraday", "short_term", "long_term", "multibagger"]:
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
    