# config/strategy_matrix.py
"""
Strategy Classification Matrix

DESIGN PHILOSOPHY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Parallel to setup_pattern_matrix.py, this defines STRATEGY DNA:
- Fit indicators & thresholds for each strategy
- Scoring rules with points & reasoning
- Preferred/avoided setups
- Best horizons
- Market cap filters (for Indian context)

After refactor, remove global.strategyClassification from master_config.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Author: Quantitative Trading System
Version: 1.0
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
            "trendstrength": {"min": 4.0, "weight": 0.3},
            "volatilityquality": {"min": 5.0, "weight": 0.3},
            "adx": {"min": 20, "weight": 0.2}
        },
        "scoring_rules": {
            "price_near_bb_low": {
                "condition": "bblow > 0 and price < bblow * 1.02",
                "points": 35,
                "reason": "Price near Buy Zone (BB Low)"
            },
            "rsi_dip": {
                "condition": "rsi < 45",
                "points": 25,
                "reason": "RSI Oversold/Dip"
            },
            "double_bottom_reversal": {
                "condition": "doubletopbottomfound == True and doubletopbottomtype == 'bullish'",
                "points": 40,
                "reason": "Double Bottom Reversal Pattern"
            },
            "squeeze_compression": {
                "condition": "ttmsqueeze == 'on'",
                "points": 10,
                "reason": "Volatility Squeeze Active"
            }
        },
        "preferred_setups": ["TREND_PULLBACK", "DEEP_PULLBACK", "PATTERNFLAGBREAKOUT"],
        "avoid_setups": ["QUALITY_ACCUMULATION"],
        "best_horizons": ["short_term", "long_term"],
        "notes": "Best for 3-10 day holds with clear trend structure"
    },

    "day_trading": {
        "description": "Intraday scalping and momentum plays",
        "enabled": True,
        "fit_threshold": 70,  # High bar
        "fit_indicators": {
            "momentumstrength": {"min": 6.0, "weight": 0.4},
            "rvol": {"min": 2.0, "weight": 0.3},
            "volatilityquality": {"min": 6.0, "weight": 0.3}
        },
        "scoring_rules": {
            "volume_surge": {
                "condition": "rvol >= 1.5",
                "points": 25,
                "reason": "Strong volume surge"
            },
            "intraday_volatility": {
                "condition": "atrpct >= 1.5",
                "points": 15,
                "reason": "Sufficient intraday range"
            },
            "three_line_strike": {
                "condition": "threelinestrikefound == True",
                "points": 40,
                "reason": "3-Line Strike Reversal"
            }
        },
        "indian_market_gates": {
            "min_avg_volume": 500000,  # 5 lakh shares/day
            "max_spread_pct": 0.003,  # 0.3% max bid-ask spread
            "min_delivery_pct": 40,  # Avoid operator stocks
            "avoid_gsm": True,  # Block GSM/ASM stocks
            "time_filters": {
                "avoid_first_15min": True,  # 9:15-9:30 AM volatility
                "avoid_last_15min": True,  # 3:15-3:30 PM square-off
                "reduce_size_lunch": 0.5,  # 12:00-1:00 PM low liquidity
                "optimal_windows": [
                    {"start": "09:45", "end": "11:30", "multiplier": 1.0},
                    {"start": "13:30", "end": "15:00", "multiplier": 1.0}
                ]
            },
            "risk_controls": {
                "max_position_pct": 0.01,  # 1% of capital max
                "mandatory_stoploss": True,
                "max_trades_per_day": 3
            }
        },
        "preferred_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE", "PATTERNDARVASBREAKOUT"],
        "avoid_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
        "best_horizons": ["intraday"],
        "notes": "Requires high volume and fast momentum - fundamentals ignored"
    },

    "trend_following": {
        "description": "Classic trend alignment with MA and ADX",
        "enabled": True,
        "fit_threshold": 60,
        "fit_indicators": {
            "adx": {"min": 25, "weight": 0.4},
            "trendstrength": {"min": 6.0, "weight": 0.3},
            "pricevs200dmapct": {"min": 0, "weight": 0.3}
        },
        "scoring_rules": {
            "ma_alignment": {
                "condition": "price > ma_fast and ma_fast > ma_mid and ma_mid > ma_slow",
                "points": 30,
                "reason": "Bullish MA Alignment"
            },
            "strong_adx": {
                "condition": "adx >= 25",
                "points": 20,
                "reason": "Strong Trend (ADX)"
            },
            "ichimoku_signal": {
                "condition": "ichimokusignalsfound == True",
                "points": 25,
                "reason": "Ichimoku Cloud Signal"
            },
            "golden_cross_bullish": {
                "condition": "goldencrossfound == True and goldencrosstype == 'bullish'",
                "points": 25,
                "reason": "Golden Cross Major Trend"
            }
        },
        "preferred_setups": ["TREND_PULLBACK", "PATTERNGOLDENCROSS", "MOMENTUM_BREAKOUT"],
        "avoid_setups": [],
        "best_horizons": ["short_term", "long_term"],
        "notes": "Follows established trends with technical confirmation"
    },

    "momentum": {
        "description": "Trend velocity and breakout continuation",
        "enabled": True,
        "fit_threshold": 65,
        "fit_indicators": {
            "rsi": {"min": 60, "weight": 0.4},
            "mafastslope": {"min": 10, "weight": 0.3},
            "adx": {"min": 25, "weight": 0.3}
        },
        "scoring_rules": {
            "high_velocity": {
                "condition": "mafastslope >= 20",
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
        "preferred_setups": ["MOMENTUM_BREAKOUT", "PATTERNDARVASBREAKOUT", "PATTERNFLAGBREAKOUT", "VOLATILITY_SQUEEZE"],
        "avoid_setups": [],
        "best_horizons": ["intraday", "short_term"],
        "notes": "Fast-moving markets with established velocity"
    },

    "minervini_growth": {
        "description": "Mark Minervini's growth stock method (Stage 2 + VCP)",
        "enabled": True,
        "fit_threshold": 65,
        "fit_indicators": {
            "epsgrowth5y": {"min": 25, "weight": 0.3},
            "relstrengthnifty": {"min": 1.2, "weight": 0.3},
            "trendstrength": {"min": 6.0, "weight": 0.2},
            "volatilityquality": {"min": 5.0, "weight": 0.2}
        },
        "market_cap_requirements": {
            "micro_cap": {
                "max_market_cap": 1000,  # ₹1000 cr
                "fit_threshold_override": 80,
                "reason": "High manipulation risk in micro-caps",
                "additional_gates": {
                    "min_delivery_pct": 60,
                    "min_institutional_ownership": 10
                }
            },
            "small_cap": {
                "min_market_cap": 1000,
                "max_market_cap": 10000,  # ₹10,000 cr
                "fit_threshold_override": 70,
                "reason": "Moderate risk - higher bar"
            },
            "mid_large_cap": {
                "min_market_cap": 10000,
                "fit_threshold": 65,  # Standard
                "reason": "Lower manipulation risk"
            }
        },
        "scoring_rules": {
            "vcp_confirmed": {
                "condition": "minervinistage2found == True",
                "points": 50,
                "reason": "VCP Pattern Confirmed"
            },
            "stage2_alignment": {
                "condition": "price > ma_mid and ma_mid > ma_slow",
                "points": 20,
                "reason": "Stage 2 Trend Alignment"
            },
            "relative_strength": {
                "condition": "relstrengthnifty > 0",
                "points": 20,
                "reason": "Outperforming Market"
            },
            "near_52w_high": {
                "condition": "Position52w >= 85",
                "points": 20,
                "reason": "Near 52W Highs"
            },
            "deep_in_base_penalty": {
                "condition": "Position52w < 50",
                "points": -20,
                "reason": "Deep in base - wait for breakout"
            }
        },
        "required_patterns": ["minervini_stage2"],
        "preferred_setups": ["PATTERNVCPBREAKOUT", "PATTERNCUPBREAKOUT", "TREND_PULLBACK"],
        "avoid_setups": [],
        "best_horizons": ["short_term", "long_term"],
        "notes": "Requires Stage 2 confirmation + growth fundamentals. Market cap filter prevents manipulation."
    },

    "canslim": {
        "description": "William O'Neil Growth Method",
        "enabled": True,
        "fit_threshold": 65,  # Lowered from 70
        "fit_indicators": {
            "quarterlygrowth": {"min": 25, "weight": 0.3},
            "epsgrowth5y": {"min": 20, "weight": 0.3},
            "relstrengthnifty": {"min": 1.0, "weight": 0.2},
            "institutionalownership": {"min": 20, "weight": 0.2}
        },
        "scoring_rules": {
            "earnings_acceleration": {
                "condition": "quarterlygrowth > epsgrowth5y",
                "points": 30,
                "reason": "Accelerating Earnings (C + A)"
            },
            "near_52w_highs": {
                "condition": "Position52w >= 90",
                "points": 25,
                "reason": "Breaking to New Highs (N)"
            },
            "cup_handle_pattern": {
                "condition": "cuphandlefound == True",
                "points": 30,
                "reason": "Cup & Handle Pattern (N)"
            },
            "market_leader": {
                "condition": "relstrengthnifty >= 1.2",
                "points": 25,
                "reason": "Market Leader (L)"
            },
            "institutional_base": {
                "condition": "institutionalownership >= 20 and institutionalownership <= 60",
                "points": 20,
                "reason": "Stable institutional sponsorship (I)"
            },
            "minimum_canslim_factors": {
                "condition": "(quarterlygrowth > epsgrowth5y) + (Position52w >= 90) + (relstrengthnifty >= 1.2) + (institutionalownership >= 20) >= 4",
                "points": 0,  # No bonus, but validates approach
                "reason": "At least 4 of 7 CANSLIM factors present"
            }
        },
        "preferred_setups": ["PATTERNCUPBREAKOUT", "MOMENTUM_BREAKOUT", "PATTERNFLAGBREAKOUT"],
        "avoid_setups": [],
        "best_horizons": ["short_term", "long_term"],
        "notes": "4-5 aligned factors is sufficient - perfect 7/7 is rare (O'Neil's own guidance)"
    },

    "value_investing": {
        "description": "Long-term value accumulation with sector context",
        "enabled": True,
        "fit_threshold": 50,
        "fit_indicators": {
            "pevssector": {"max": 0.85, "weight": 0.3, "direction": "invert"},  # Relative PE
            "roe": {"min": 15, "weight": 0.3},
            "deratio": {"max": 0.5, "weight": 0.2, "direction": "invert"},
            "fcfyield": {"min": 5.0, "weight": 0.2}
        },
        "scoring_rules": {
            "cheap_vs_sector": {
                "condition": "pevssector is not None and peratio > 0 and pevssector < 0.8",
                "points": 35,
                "reason": "Trading below sector average (cheap)"
            },
            "cheap_pb": {
                "condition": "pbratio > 0 and pbratio < 1.5",
                "points": 25,
                "reason": "Low Price to Book"
            },
            "strong_roe": {
                "condition": "roe >= 15",
                "points": 20,
                "reason": "Strong Return on Equity"
            },
            "low_debt": {
                "condition": "deratio <= 0.5",
                "points": 20,
                "reason": "Conservative Debt Levels"
            },
            "sector_discount": {
                "condition": "pevssector is not None and pevssector < 0.7",
                "points": 10,
                "reason": "Deep discount vs sector (30%+ cheaper)"
            }
        },
        "preferred_setups": ["QUALITY_ACCUMULATION", "DEEP_VALUE_PLAY", "VALUE_TURNAROUND"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
        "best_horizons": ["long_term", "multibagger"],
        "notes": "Uses sector-relative PE instead of absolute bands - adapts to market cycles"
    },

    "income_investing": {
        "description": "Dividend yield and cash flow focus",
        "enabled": True,
        "fit_threshold": 60,
        "fit_indicators": {
            "dividendyield": {"min": 3.0, "weight": 0.5},
            "fcfyield": {"min": 5.0, "weight": 0.3},
            "deratio": {"max": 0.7, "weight": 0.2, "direction": "invert"}
        },
        "scoring_rules": {
            "high_yield": {
                "condition": "dividendyield >= 3.0",
                "points": 40,
                "reason": "Attractive Dividend Yield"
            },
            "fcf_strength": {
                "condition": "fcfyield >= 5.0",
                "points": 30,
                "reason": "Strong Free Cash Flow"
            },
            "safe_payout": {
                "condition": "dividendpayout > 0 and dividendpayout <= 60",
                "points": 30,
                "reason": "Sustainable Payout Ratio"
            }
        },
        "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND"],
        "avoid_setups": [],
        "best_horizons": ["long_term", "multibagger"],
        "notes": "Focus on stable cash generation and dividend safety"
    },

    "position_trading": {
        "description": "Long-term trend following for major cyclic moves",
        "enabled": True,
        "fit_threshold": 50,
        "fit_indicators": {
            "trendstrength": {"min": 6.0, "weight": 0.4},
            "pricevsprimarytrendpct": {"min": 0, "weight": 0.4},
            "volatilityquality": {"min": 6.0, "weight": 0.2}
        },
        "scoring_rules": {
            "golden_cross": {
                "condition": "goldencrossfound == True and goldencrosstype == 'bullish'",
                "points": 50,
                "reason": "Primary Trend Reversal (Golden Cross)"
            },
            "stable_volatility": {
                "condition": "volatilityquality >= 7.0",
                "points": 25,
                "reason": "Stable, high-quality trend"
            },
            "ma_alignment": {
                "condition": "price > ma_mid and ma_mid > ma_slow",
                "points": 25,
                "reason": "Triple MA Alignment"
            }
        },
        "preferred_setups": ["PATTERNGOLDENCROSS", "TREND_PULLBACK", "QUALITY_ACCUMULATION"],
        "avoid_setups": [],
        "best_horizons": ["long_term", "multibagger"],
        "notes": "Patient capital seeking major trend cycles"
    },

    "quality_growth": {
        "description": "Quality compounders with sustainable growth at reasonable valuations",
        "enabled": True,
        "fit_threshold": 55,
        "fit_indicators": {
            "roe": {"min": 18, "weight": 0.20},
            "roce": {"min": 20, "weight": 0.20},
            "epsgrowth5y": {"min": 15, "weight": 0.15},
            "revenuegrowth5y": {"min": 12, "weight": 0.10},
            "deratio": {"max": 0.7, "weight": 0.10, "direction": "invert"},
            "trendstrength": {"min": 4.0, "weight": 0.10},
            "pegratio": {"max": 2.0, "weight": 0.10, "direction": "invert"},  # Fixed from 2.5
            "earningsstability": {"min": 6.5, "weight": 0.05}
        },
        "scoring_rules": {
            "consistent_quality": {
                "condition": "roe >= 18 and roce >= 20 and roe3yavg >= 18",
                "points": 35,
                "reason": "Consistent quality metrics over 3 years"
            },
            "excellent_garp": {
                "condition": "pegratio > 0 and pegratio <= 1.5",
                "points": 35,  # Increased from 30
                "reason": "Excellent GARP (PEG ≤ 1.5)"
            },
            "acceptable_garp": {
                "condition": "pegratio > 1.5 and pegratio <= 2.0",
                "points": 20,
                "reason": "Acceptable GARP (PEG 1.5-2.0)"
            },
            "optimal_entry_zone": {
                "condition": "trendstrength >= 4.0 and trendstrength <= 7.0 and rsi >= 50 and rsi <= 65",
                "points": 10,
                "reason": "In optimal entry zone - not overheated"
            },
            "balance_sheet_quality": {
                "condition": "deratio <= 0.5 and interestcoverage >= 5",
                "points": 20,
                "reason": "Conservative balance sheet"
            },
            "earnings_consistency": {
                "condition": "earningsstability >= 7.0",
                "points": 15,
                "reason": "Stable, predictable earnings"
            },
            "overheated_penalty": {
                "condition": "trendstrength >= 9.0 or rsi >= 80",
                "points": -10,  # Reduced from -15
                "reason": "Extremely overextended - wait for pullback"
            },
            "moderately_extended": {
                "condition": "(trendstrength >= 8.0 and trendstrength < 9.0) or (rsi >= 70 and rsi < 80)",
                "points": 0,  # Flag only, no penalty
                "reason": "Strong trend but sustainable - monitor position sizing"
            },
            "multibagger_zone": {
                "condition": "roe >= 22 and epsgrowth5y >= 20 and pegratio <= 1.5 and peratio <= 30",
                "points": 20,
                "reason": "In multibagger zone (HDFC/Asian Paints profile)"
            }
        },
        "preferred_setups": ["QUALITY_ACCUMULATION", "VALUE_TURNAROUND", "TREND_PULLBACK", "DEEP_PULLBACK"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "VOLATILITY_SQUEEZE"],
        "best_horizons": ["long_term", "multibagger"],
        "notes": "Sweet spot between value and momentum. Adjusted for quality compounders staying overbought (Asian Paints 2017-2018 lesson)."
    },

    "reversal_trading": {
        "description": "Bottom fishing in quality stocks after corrections",
        "enabled": True,
        "fit_threshold": 55,
        "fit_indicators": {
            "trendstrength": {"max": 4.0, "weight": 0.25, "direction": "invert"},
            "rsi": {"max": 40, "weight": 0.25, "direction": "invert"},
            "roe": {"min": 15, "weight": 0.20},
            "deratio": {"max": 0.8, "weight": 0.15, "direction": "invert"},
            "pricevs52whighpct": {"max": 80, "weight": 0.15, "direction": "invert"}
        },
        "scoring_rules": {
            "quality_in_distress": {
                "condition": "roe >= 18 and roce >= 20 and pricevs52whighpct <= 75",
                "points": 40,
                "reason": "Quality stock 25%+ off highs"
            },
            "oversold_rsi": {
                "condition": "rsi <= 35",
                "points": 25,
                "reason": "Deeply oversold"
            },
            "bullish_divergence": {
                "condition": "rsislope >= 0.05 and priceslope <= -0.01",
                "points": 30,
                "reason": "Bullish divergence (Price falling but RSI rising)"
            },
            "consolidation_base": {
                "condition": "bbwidth <= 5.0 and volatilityquality >= 5.0",
                "points": 20,
                "reason": "Tight consolidation after fall"
            },
            "reversal_confirmation": {
                "condition": "macdhistogram > 0 and supertrendsignal == 'Bullish'",
                "points": 20,
                "reason": "Technical reversal confirmation"
            },
            "still_falling_penalty": {
                "condition": "trendstrength <= 2.0 and momentumstrength <= 2.0",
                "points": -25,
                "reason": "Still in freefall - wait for stabilization"
            },
            "poor_quality_penalty": {
                "condition": "roe < 12 or deratio > 1.0",
                "points": -20,
                "reason": "Quality too low for reversal play"
            }
        },
        "preferred_setups": ["REVERSALMACDCROSSUP", "REVERSALRSISWINGUP", "REVERSALSTFLIPUP", "DEEP_PULLBACK"],
        "avoid_setups": ["MOMENTUM_BREAKOUT", "TREND_PULLBACK"],
        "best_horizons": ["short_term", "long_term"],
        "notes": "Requires patience and quality floor (ROE ≥ 15, low debt). Uses priceslope for proper divergence detection."
    }
}


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_strategy_config(strategy_name: str) -> Dict[str, Any]:
    """
    Get complete configuration for a strategy.

    Args:
        strategy_name: Name of strategy (e.g., "quality_growth")

    Returns:
        Strategy configuration dict or empty dict if not found
    """
    return STRATEGY_MATRIX.get(strategy_name, {})


def get_all_enabled_strategies() -> List[str]:
    """Get list of all enabled strategy names."""
    return [
        name for name, config in STRATEGY_MATRIX.items()
        if config.get("enabled", False)
    ]


def get_strategy_fit_indicators(strategy_name: str) -> Dict[str, Dict]:
    """Get fit indicators for a strategy."""
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    return strategy.get("fit_indicators", {})


def get_strategy_scoring_rules(strategy_name: str) -> Dict[str, Dict]:
    """Get scoring rules for a strategy."""
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    return strategy.get("scoring_rules", {})


def get_strategy_preferred_setups(strategy_name: str) -> List[str]:
    """Get preferred setups for a strategy."""
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    return strategy.get("preferred_setups", [])


def get_strategy_best_horizons(strategy_name: str) -> List[str]:
    """Get best horizons for a strategy."""
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    return strategy.get("best_horizons", [])


def calculate_strategy_fit_score(
    strategy_name: str,
    indicators: Dict[str, float],
    fundamentals: Dict[str, float]
) -> float:
    """
    Calculate fit score for a strategy based on indicators & fundamentals.

    Args:
        strategy_name: Name of strategy
        indicators: Technical indicator values
        fundamentals: Fundamental metric values

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

        # Get actual value from indicators or fundamentals
        actual = indicators.get(indicator) or fundamentals.get(indicator)
        if actual is None:
            continue

        total_weight += weight

        # Check threshold
        threshold_met = True
        if direction == "invert":
            # For inverted metrics (lower is better)
            if max_val is not None and actual > max_val:
                threshold_met = False
        else:
            # Normal metrics (higher is better)
            if min_val is not None and actual < min_val:
                threshold_met = False
            if max_val is not None and actual > max_val:
                threshold_met = False

        if threshold_met:
            weighted_score += weight

    if total_weight == 0:
        return 0.0

    return (weighted_score / total_weight) * 100


def validate_strategy_setup_compatibility(
    strategy_name: str,
    setup_type: str
) -> Dict[str, Any]:
    """
    Check if a setup is compatible with a strategy.

    Args:
        strategy_name: Name of strategy
        setup_type: Setup type

    Returns:
        {
            "compatible": bool,
            "preference": "preferred" | "neutral" | "avoid",
            "reason": str
        }
    """
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    if not strategy:
        return {
            "compatible": True,
            "preference": "neutral",
            "reason": "Strategy not found"
        }

    preferred = strategy.get("preferred_setups", [])
    avoided = strategy.get("avoid_setups", [])

    if setup_type in preferred:
        return {
            "compatible": True,
            "preference": "preferred",
            "reason": f"Setup aligns with {strategy_name} strategy"
        }
    elif setup_type in avoided:
        return {
            "compatible": False,
            "preference": "avoid",
            "reason": f"Setup conflicts with {strategy_name} approach"
        }
    else:
        return {
            "compatible": True,
            "preference": "neutral",
            "reason": "No specific preference"
        }


def get_strategy_market_cap_requirements(
    strategy_name: str,
    market_cap: Optional[float] = None
) -> Dict[str, Any]:
    """
    Get market cap requirements for a strategy (Indian market context).

    Args:
        strategy_name: Name of strategy
        market_cap: Market cap in crores (optional)

    Returns:
        Market cap requirements dict
    """
    strategy = STRATEGY_MATRIX.get(strategy_name, {})
    mc_requirements = strategy.get("market_cap_requirements", {})

    if not mc_requirements or market_cap is None:
        return mc_requirements

    # Determine which category applies
    if market_cap <= 1000:
        return mc_requirements.get("micro_cap", {})
    elif market_cap <= 10000:
        return mc_requirements.get("small_cap", {})
    else:
        return mc_requirements.get("mid_large_cap", {})


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_strategy_matrix() -> Dict[str, Any]:
    """
    Validate strategy matrix for consistency.

    Returns:
        {
            "valid": bool,
            "errors": List[str],
            "warnings": List[str]
        }
    """
    errors = []
    warnings = []

    for strategy_name, config in STRATEGY_MATRIX.items():
        # Check required fields
        if "description" not in config:
            errors.append(f"{strategy_name}: Missing 'description'")

        if "fit_threshold" not in config:
            warnings.append(f"{strategy_name}: Missing 'fit_threshold'")

        if "fit_indicators" not in config or not config["fit_indicators"]:
            errors.append(f"{strategy_name}: Missing or empty 'fit_indicators'")

        # Check fit indicator weights sum
        if "fit_indicators" in config:
            total_weight = sum(
                ind.get("weight", 0)
                for ind in config["fit_indicators"].values()
            )
            if abs(total_weight - 1.0) > 0.01:
                warnings.append(
                    f"{strategy_name}: Fit indicator weights sum to {total_weight:.2f}, not 1.0"
                )

        # Check scoring rules
        if "scoring_rules" in config:
            for rule_name, rule in config["scoring_rules"].items():
                if "condition" not in rule:
                    errors.append(f"{strategy_name}.{rule_name}: Missing 'condition'")
                if "points" not in rule:
                    errors.append(f"{strategy_name}.{rule_name}: Missing 'points'")
                if "reason" not in rule:
                    warnings.append(f"{strategy_name}.{rule_name}: Missing 'reason'")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("="*80)
    print("STRATEGY MATRIX VALIDATION")
    print("="*80)

    validation = validate_strategy_matrix()

    if validation["valid"]:
        print("✓ Strategy matrix is valid")
    else:
        print("✗ Strategy matrix has errors:")
        for error in validation["errors"]:
            print(f"  - {error}")

    if validation["warnings"]:
        print("\nWarnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    print(f"\nTotal strategies: {len(STRATEGY_MATRIX)}")
    print(f"Enabled strategies: {len(get_all_enabled_strategies())}")

    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80)

    # Example: Calculate fit score
    mock_indicators = {
        "roe": 24.2,
        "roce": 28.5,
        "epsgrowth5y": 18.5,
        "revenuegrowth5y": 15.2,
        "deratio": 0.35,
        "trendstrength": 7.2,
        "pegratio": 1.4,
        "earningsstability": 7.8
    }

    fit_score = calculate_strategy_fit_score(
        "quality_growth",
        indicators={},
        fundamentals=mock_indicators
    )

    print(f"Quality Growth fit score: {fit_score:.1f}")

    # Example: Check setup compatibility
    compat = validate_strategy_setup_compatibility(
        "quality_growth",
        "QUALITY_ACCUMULATION"
    )

    print(f"\nQuality Growth + Quality Accumulation:")
    print(f"  Compatible: {compat['compatible']}")
    print(f"  Preference: {compat['preference']}")
    print(f"  Reason: {compat['reason']}")
