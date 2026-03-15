# config/setup_pattern_matrix.py
"""
Setup-Pattern Mapping Matrix - PRODUCTION VERSION v3.0

Complete pattern genome with:
✓ Setup classification rules (moved from master_config)
✓ Default priority scores (horizons override)
# (Removed dead code reference to default_confidence_base_floor)
✓ Universal context requirements (horizons can relax)
✓ Pattern detection flags and conditions
✓ Validation penalties/bonuses

ARCHITECTURE:
- THIS FILE: Setup DNA - what makes a setup valid, universal requirements
- MASTER_CONFIG: Horizon-specific overrides (priorities, gates, confidence adjustments)
- STRATEGY_MATRIX: Strategy DNA - fit indicators, scoring rules
- RESOLVER: Merges all three at runtime

Author: Quantitative Trading System
Version: 3.0 (Production-Ready)
"""

from typing import Dict, List, Literal, Tuple, Any, Optional

PatternRole = Literal["PRIMARY", "CONFIRMING", "CONFLICTING"]

# ============================================================
# GLOBAL PATTERN PHYSICS
# ============================================================

PATTERN_SCORING_THRESHOLDS = {
    "high_quality": 60,
    "medium_quality": 40,
    "poor_quality": 20
}

DEFAULT_PHYSICS = {
    "target_ratio": 1.0,
    "duration_multiplier": 1.0,
    "max_stop_pct": 10.0,
    "horizons_supported": ["intraday", "short_term", "long_term"]
}

# ============================================================
# SETUP-PATTERN AFFINITY MATRIX + CLASSIFICATION RULES
# ============================================================

SETUP_PATTERN_MATRIX: Dict[str, Dict[str, Any]] = {

    # ========================================================
    # PATTERN-BASED SETUPS
    # ========================================================

    "PATTERN_DARVAS_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["darvasBox"],
            "CONFIRMING": ["bollingerSqueeze", "goldenCross", "ichimokuSignals"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross"]
        },
        "classification_rules": {
            "pattern_detection": {
                "darvasBox": True  # Boolean flag check
            },
            "technical_gates": {
                "rvol": {"min": 1.5},
                "trendStrength": {"min": 4.0}
            },
            "require_fundamentals": False
        },
        "default_priority": 98,
        "context_requirements": {
            "technical": {
                "rvol": {"min": 1.5},
                "trendStrength": {"min": 4.0},
                "adx": {"min": 16},
                "volatilityQuality": {"min": 4.0}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "weak_volume": {
                    "gates": {"rvol": {"max": 1.2990000000000002}},
                    "confidence_penalty": 15,
                    "reason": "Darvas breakout needs strong volume"
                }
            },
            "bonuses": {
                "explosive_volume": {
                    "gates": {"rvol": {"min": 3.0}},
                    "confidence_boost": 10,
                    "reason": "Exceptional volume confirmation"
                }
            }
        },
        "min_pattern_quality": 8.5,
        "min_setup_score": 85,
        "setup_type": "pattern_driven",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": 5.0},
                        "rvol": {"min": 2.0}  # Higher for intraday
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 4.0},
                        "rvol": {"min": 1.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": 1.2}
                    }
                }
            },
        },
        "description": "Darvas box breakout with volume confirmation"
    },

    "PATTERN_VCP_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["minerviniStage2"],
            "CONFIRMING": ["bollingerSqueeze", "cupHandle", "darvasBox"],
            "CONFLICTING": ["flagPennant", "threeLineStrike"]
        },
        "classification_rules": {
            "pattern_detection": {
                "minerviniStage2": True
            },
            "technical_gates": {
                "volatilityQuality": {"min": 6.0},
                "rsi": {"min": 50}
            },
            "require_fundamentals": False
        },
        "default_priority": 97,
        "context_requirements": {
            "technical": {
                "volatilityQuality": {"min": 6.0},
                "rsi": {"min": 50},
                "adx": {"min": 15},
                "trendStrength": {"min": 4.5},
                "position52w": {"min": 85}
            },
            "fundamental": {
                "required": False
            }
        },
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 17},
                        "trendStrength": {"min": 5.5},
                        "volatilityQuality": {"min": 7.0}
                    }
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": 6.0}
                    }
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 5.0}
                    }
                }
            },
        },
        "description": "Minervini-style volatility contraction pattern"
    },

    "PATTERN_CUP_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["cupHandle"],
            "CONFIRMING": ["goldenCross", "bollingerSqueeze", "ichimokuSignals"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {
                "cupHandle": True
            },
            "technical_gates": {
                "rvol": {"min": 1.2},
                "trendStrength": {"min": 3.5}
            },
            "require_fundamentals": False
        },
        "default_priority": 96,
        "context_requirements": {
            "technical": {
                "rvol": {"min": 1.2},
                "trendStrength": {"min": 3.5},
                "adx": {"min": 14},
                "volatilityQuality": {"min": 3.5}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "weak_breakout_volume": {
                    "gates": {"rvol": {"max": 1.199}},
                    "confidence_penalty": 15,
                    "reason": "Cup & Handle breakout requires strong volume confirmation"
                },
                "weak_momentum": {
                    "gates": {"rsi": {"max": 54.999}},
                    "confidence_penalty": 10,
                    "reason": "Lacking momentum for sustained breakout"
                }
            },
            "bonuses": {
                "explosive_volume": {
                    "gates": {"rvol": {"min": 2.5}},
                    "confidence_boost": 15,
                    "reason": "Exceptional volume on the handle breakout"
                }
            }
        },
        "min_pattern_quality": 8.0,
        "min_setup_score": 80,
        "setup_type": "pattern_driven",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": 4.5}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": 4.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.0},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Cup and handle breakout pattern"
    },

    "PATTERN_FLAG_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["flagPennant"],
            "CONFIRMING": ["bollingerSqueeze", "ichimokuSignals"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "threeLineStrike", "cupHandle"]
        },
        "classification_rules": {
            "pattern_detection": {
                "flagPennant": True
            },
            "technical_gates": {
                "rvol": {"min": 1.5},
                "trendStrength": {"min": 5.5}
            },
            "require_fundamentals": False
        },
        "default_priority": 95,
        "context_requirements": {
            "technical": {
                "rvol": {"min": 1.5},
                "trendStrength": {"min": 5.5},
                "adx": {"min": 18},
                "volatilityQuality": {"min": 5.0}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "exhausted_trend": {
                    "gates": {"rsi": {"min": 85}},
                    "confidence_penalty": 15,
                    "reason": "Trend is too overextended, risk of failed breakout"
                },
                "low_volume_push": {
                    "gates": {"rvol": {"max": 0.999}},
                    "confidence_penalty": 20,
                    "reason": "Flag breakout without volume is a bull trap"
                }
            },
            "bonuses": {
                "strong_momentum_push": {
                    "gates": {"macdhistogram": {"min": 0.001}},
                    "confidence_boost": 10,
                    "reason": "MACD momentum fully supports continuation"
                }
            }
        },
        "min_pattern_quality": 7.5,
        "min_setup_score": 80,
        "setup_type": "pattern_driven",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 6.5},
                        "volatilityQuality": {"min": 5.0},
                        "rvol": {"min": 2.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.5},
                        "volatilityQuality": {"min": 4.0},
                        "rvol": {"min": 1.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": 1.2}
                    }
                },

            },
        },
        "description": "Flag/pennant continuation breakout"
    },

    "PATTERN_STRIKE_REVERSAL": {
        "patterns": {
            "PRIMARY": ["threeLineStrike"],
            "CONFIRMING": ["bollingerSqueeze", "ichimokuSignals", "bullishNecklinePattern", "bearishNecklinePattern"],
            "CONFLICTING": ["deathCross", "flagPennant", "minerviniStage2"]
        },
        "classification_rules": {
            "pattern_detection": {
                "threeLineStrike": True
            },
            "technical_gates": {
                "rvol": {"min": 1.3},
                "rsi": {"min": 45}
            },
            "require_fundamentals": False
        },
        "default_priority": 94,
        "context_requirements": {
            "technical": {
                "rvol": {"min": 1.3},
                "rsi": {"min": 45},
                "adx": {"min": 12},
                "volatilityQuality": {"min": 3.0}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "weak_reversal_volume": {
                    "gates": {"rvol": {"max": 0.999}},
                    "confidence_penalty": 15,
                    "reason": "Reversals without volume usually fail"
                },
                "fighting_strong_trend": {
                    "gates": {"trendStrength": {"min": 5.0}, "adx": {"min": 25}},
                    "confidence_penalty": 20,
                    "reason": "Attempting to reverse a very strong trend"
                }
            },
            "bonuses": {
                "extreme_oversold_bounce": {
                    "gates": {"rsi": {"max": 35}},
                    "confidence_boost": 15,
                    "reason": "Bouncing from deeply oversold levels"
                }
            }
        },
        "min_pattern_quality": 7.0,
        "min_setup_score": 80,
        "setup_type": "pattern_driven",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "volatilityQuality": {"min": 3.0},
                        "rvol": {"min": 1.5}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 12},
                        "volatilityQuality": {"min": 2.5},
                        "rvol": {"min": 1.3}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": 1.2}
                    }
                },

            },
        },
        "description": "Three-line strike reversal pattern"
    },

    "PATTERN_GOLDEN_CROSS": {
        "patterns": {
            "PRIMARY": ["goldenCross"],
            "CONFIRMING": ["cupHandle", "minerviniStage2", "ichimokuSignals"],
            "CONFLICTING": ["deathCross", "bullishNecklinePattern", "bearishNecklinePattern"]
        },
        "classification_rules": {
            "pattern_detection": {
                "goldenCross": True
            },
            "technical_gates": {
                "trendStrength": {"min": 3.0},
                "momentumStrength": {"min": 4.0}
            },
            "require_fundamentals": False
        },
        "default_priority": 92,
        "context_requirements": {
            "technical": {
                "trendStrength": {"min": 3.0},
                "momentumStrength": {"min": 4.0},
                "adx": {"min": 15},
                "volatilityQuality": {"min": 3.0}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "chop_zone_cross": {
                    "gates": {"adx": {"max": 14.999}},
                    "confidence_penalty": 20,
                    "reason": "Golden cross in a flat/choppy market is unreliable"
                }
            },
            "bonuses": {
                "strong_trend_confirmed": {
                    "gates": {"adx": {"min": 25}},
                    "confidence_boost": 15,
                    "reason": "Cross confirmed by strong directional trend"
                },
                "volume_backed": {
                    "gates": {"rvol": {"min": 1.5}},
                    "confidence_boost": 10,
                    "reason": "Institutions buying the cross"
                }
            }
        },
        "min_pattern_quality": 7.5,
        "min_setup_score": 75,
        "setup_type": "pattern_driven",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 17},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 3.5}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 3.0},
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 6.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 2.5},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Long-term moving average golden cross"
    },

    # ========================================================
    # MOMENTUM SETUPS
    # ========================================================

    "MOMENTUM_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["darvasBox", "bollingerSqueeze", "cupHandle"],
            "CONFIRMING": ["ichimokuSignals", "goldenCross", "flagPennant", "threeLineStrike"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross", "minerviniStage2"]
        },
        "classification_rules": {
            "pattern_detection": {},  # No specific pattern required
            "technical_gates": {
                "bbpercentb": {"min": 0.98},
                "rsi": {"min": 60},
                "rvol": {"min": 1.5}
            },
            "require_fundamentals": False
        },
        "default_priority": 90,
        "context_requirements": {
            "technical": {
                "bbpercentb": {"min": 0.98},
                "rsi": {"min": 60, "max": 80},
                "rvol": {"min": 1.5},
                "adx": {"min": 18},
                "trendStrength": {"min": 5.0},
                "volatilityQuality": {"min": 4.0},
                "wick_ratio_max": 2.5
            },
            "fundamental": {
                "required": False
            }
        },
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},              # Keep default
                        "trendStrength": {"min": 5.0},   # Keep default
                        "volatilityQuality": {"min": 5.0}  # ⬆️ Tighter for intraday
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": 4.0}  # Standard
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},            # ⬇️ Relax for long_term
                        "trendStrength": {"min": 4.5},   # ⬇️ Lower requirement
                        "volatilityQuality": {"min": None}  # ⬇️ No requirement
                    }
                },

            },
        },
        "validation_modifiers": {
            "penalties": {
                "low_breakout_volume": {
                    "gates": {"rvol": {"max": 1.999}},
                    "confidence_penalty": 10,
                    "reason": "Breakout needs volume confirmation"
                },
                "weak_trend": {
                    "gates": {"trendStrength": {"max": 4.999}},
                    "confidence_penalty": 15,
                    "reason": "Momentum breakout needs strong trend"
                }
            },
            "bonuses": {
                "volume_surge": {
                    "gates": {"rvol": {"min": 3.0}},
                    "confidence_boost": 10,
                    "reason": "Exceptional volume confirmation"
                }
            }
        },
        "min_pattern_quality": 7.0,
        "min_setup_score": 60,
        "setup_type": "momentum",
        "description": "Explosive breakout from consolidation with volume"
    },

    "MOMENTUM_BREAKDOWN": {
        "patterns": {
            "PRIMARY": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross"],
            "CONFIRMING": ["bollingerSqueeze"],
            "CONFLICTING": ["goldenCross", "cupHandle", "minerviniStage2", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "bbpercentb": {"max": 0.02},
                "rsi": {"max": 40},
                "rvol": {"min": 1.5}
            },
            "require_fundamentals": False
        },
        "default_priority": 88,
        "context_requirements": {
            "technical": {
                "bbpercentb": {"max": 0.02},
                "rsi": {"max": 40},
                "rvol": {"min": 1.5},
                "adx": {"min": 18},
                "trendStrength": {"min": 5.0},
                "maTrendSignal": -1
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "low_breakout_volume": {
                    "gates": {"rvol": {"max": 1.999}},
                    "confidence_penalty": 12,
                    "reason": "Breakdown volume weak relative to confirmation threshold"
                }
            },
            "bonuses": {
                "panic_selling": {
                    "gates": {"rvol": {"min": 2.0}},
                    "confidence_boost": 15,
                    "reason": "High volume institutional distribution"
                },
                "heavy_momentum": {
                    "gates": {"rsi": {"max": 30}},
                    "confidence_boost": 10,
                    "reason": "Deep bear momentum confirmed"
                }
            }
        },
        "min_pattern_quality": 7.0,
        "min_setup_score": 60,
        "setup_type": "momentum",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.5},  # Bearish trend
                        "volatilityQuality": {"min": 4.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Bearish breakdown for short/avoid"
    },

    # ========================================================
    # TREND SETUPS
    # ========================================================

    "TREND_PULLBACK": {
        "patterns": {
            "PRIMARY": ["goldenCross", "flagPennant", "ichimokuSignals"],
            "CONFIRMING": ["bollingerSqueeze", "threeLineStrike"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross", "cupHandle"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "trendStrength": {"min": 4.0},
                "price": {"min_metric": "maFast"},
                # TODO: abs(price - maFast) / maFast <= 0.05,
                "rsi": {"min": 50},
                "momentumStrength": {"min": 4.0}
            },
            "require_fundamentals": False
        },
        "default_priority": 80,
        "context_requirements": {
            "technical": {
                "trendStrength": {"min": 4.0},
                "priceVsPrimaryTrendPct": {"max": 5},
                "rsi": {"min": 50},
                "momentumStrength": {"min": 4.0},
                "adx": {"min": 16},
                "volatilityQuality": {"min": 3.5}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "lost_momentum": {
                    "gates": {"rsi": {"max": 51.999}},
                    "confidence_penalty": 10,
                    "reason": "Pullback cut too deep, momentum is broken"
                },
                "high_volume_selling": {
                    "gates": {"rvol": {"min": 2.0}, "price": {"max_metric": "prev_close"}},
                    "confidence_penalty": 15,
                    "reason": "Pullback has too much selling volume"
                }
            },
            "bonuses": {
                "light_volume_pullback": {
                    "gates": {"rvol": {"max": 0.8}},
                    "confidence_boost": 10,
                    "reason": "Healthy, low-volume consolidation"
                }
            }
        },
        "min_pattern_quality": 5.0,
        "min_setup_score": 60,
        "setup_type": "trend_following",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},              # Need strong trend for intraday
                        "trendStrength": {"min": 5.0},   # Higher than default 4.0
                        "volatilityQuality": {"min": 4.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 3.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 5.0}
                    },
                    "technical": {
                        "adx": {"min": None},            # Relaxed for long_term
                        "trendStrength": {"min": 3.5},   # Lower requirement
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Dip-buying in established uptrend"
    },

    "DEEP_PULLBACK": {
        "patterns": {
            "PRIMARY": ["goldenCross", "ichimokuSignals"],
            "CONFIRMING": ["bollingerSqueeze", "threeLineStrike", "cupHandle"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "trendStrength": {"min": 3.5},
                "priceVsPrimaryTrendPct": {"min": -10, "max": -5}
            },
            "require_fundamentals": False
        },
        "default_priority": 75,
        "context_requirements": {
            "technical": {
                "trendStrength": {"min": 3.5},
                "priceVsPrimaryTrendPct": {"min": -10, "max": -5},
                "adx": {"min": 14},
                "volatilityQuality": {"min": 3.0}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "lost_momentum": {
                    "gates": {"rsi": {"max": 34.999}},
                    "confidence_penalty": 20,
                    "reason": "Pullback cut too deep, momentum structurally broken"
                },
                "high_volume_selling": {
                    "gates": {"rvol": {"min": 2.5}, "price": {"max_metric": "prev_close"}},
                    "confidence_penalty": 15,
                    "reason": "Heavy distribution — not a healthy retracement"
                }
            },
            "bonuses": {
                "light_volume_pullback": {
                    "gates": {"rvol": {"max": 0.8}},
                    "confidence_boost": 10,
                    "reason": "Healthy, low-volume retracement into support zone"
                },
                "strong_trend_context": {
                    "gates": {"trendStrength": {"min": 6.0}, "adx": {"min": 25}},
                    "confidence_boost": 8,
                    "reason": "Deep pull in a powerful trend — high probability recovery"
                }
            }
        },
        "min_pattern_quality": 6.0,
        "min_setup_score": 60,
        "setup_type": "trend_following",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": 3.5}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.0},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Deeper retracement in strong trend"
    },

    "TREND_FOLLOWING": {
        "patterns": {
            "PRIMARY": ["goldenCross", "ichimokuSignals"],
            "CONFIRMING": ["flagPennant", "bollingerSqueeze"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross", "threeLineStrike"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "rsi": {"min": 55},
                "macdhistogram": {"min": 0}
            },
            "require_fundamentals": False
        },
        "default_priority": 70,
        "context_requirements": {
            "technical": {
                "rsi": {"min": 55},
                "macdhistogram": {"min": 0},
                "adx": {"min": 20},
                "trendStrength": {"min": 5.0}
            },
            "fundamental": {
                "required": False
            }
        },
        "min_pattern_quality": 5.0,
        "min_setup_score": 60,
        "setup_type": "trend_following",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 22},              # Stronger trend for intraday
                        "trendStrength": {"min": 5.5}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 4.5}
                    }
                },

            },
        },
        "validation_modifiers": {
            "penalties": {
                "decelerating_momentum": {
                    "gates": {"rsislope": {"max": -0.05}, "macdhistogram": {"max": 0}},
                    "confidence_penalty": 12,
                    "reason": "Both RSI and MACD momentum decelerating in trend setup"
                },
                "overextended": {
                    "gates": {"rsi": {"min": 80.001}},
                    "confidence_penalty": 10,
                    "reason": "Overbought — elevated mean-reversion risk"
                }
            },
            "bonuses": {
                "strong_trend_confluence": {
                    "gates": {"adx": {"min": 30}, "trendStrength": {"min": 7.0}},
                    "confidence_boost": 10,
                    "reason": "Strong directional trend with high quality score"
                }
            }
        },
        "description": "Classic trend-following entry"
    },

    "BEAR_TREND_FOLLOWING": {
        "patterns": {
            "PRIMARY": ["deathCross", "ichimokuSignals"],
            "CONFIRMING": ["bollingerSqueeze"],
            "CONFLICTING": ["goldenCross", "cupHandle", "minerviniStage2", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {}, # No mandatory pattern, but primary patterns boost score
            "technical_gates": {
                "rsi": {"max": 45},
                "macdhistogram": {"max": 0},
                "trendStrength": {"min": 5.0}
            },
            "require_fundamentals": False
        },
        "default_priority": 55, # Matches master_config priority override
        "context_requirements": {
            "technical": {
                "rsi": {"max": 45},
                "macdhistogram": {"max": 0},
                "adx": {"min": 20},
                "trendStrength": {"min": 5.0},
                "maTrendSignal": -1 # Indicates bearish alignment
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "low_volume_drop": {
                    "gates": {"rvol": {"max": 0.999}},
                    "confidence_penalty": 15,
                    "reason": "Price drifting lower without selling pressure"
                }
            },
            "bonuses": {
                "panic_selling": {
                    "gates": {"rvol": {"min": 2.0}},
                    "confidence_boost": 15,
                    "reason": "High volume institutional distribution"
                },
                "heavy_momentum": {
                    "gates": {"rsi": {"max": 30}},
                    "confidence_boost": 10,
                    "reason": "Deep bear momentum confirmed"
                }
            }
        },
        "min_pattern_quality": 5.0,
        "min_setup_score": 60,
        "setup_type": "trend_following",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 22},
                        "trendStrength": {"min": 5.5}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 4.5}
                    }
                },

            },
        },
        "description": "Short/sell setup following an established downtrend"
    },
    # ========================================================
    # ACCUMULATION SETUPS
    # ========================================================

    "QUALITY_ACCUMULATION": {
        "patterns": {
            "PRIMARY": ["minerviniStage2", "darvasBox"],
            "CONFIRMING": ["bollingerSqueeze", "cupHandle", "ichimokuSignals"],
            "CONFLICTING": ["bullishNecklinePattern", "bearishNecklinePattern", "deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "bbWidth": {"max": 4.999},
                "rvol": {"max": 0.999},
                "rsi": {"min": 40, "max": 60}
            },
            "fundamental_gates": {
                # TODO: roe >= 20 or roe3yAvg >= 18,
                "roce": {"min": 25},
                "deRatio": {"max": 0.5}
            }
        },
        "default_priority": 78,
        "context_requirements": {
            "technical": {
                "bbWidth": {"max": 5.0},
                "rvol": {"max": 1.0},
                "rsi": {"min": 40, "max": 60},
                "adx": {"max": 25},
                "volatilityQuality": {"min": 5.0}
            },
            "fundamental": {
                "required": True,
                "roe": {"min": 15},
                "roce": {"min": 25},
                "deRatio": {"max": 0.5},
                "piotroskiF": {"min": 7}
            }
        },
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 4.0}  # Relaxed for intraday
                    },
                    "fundamental": {
                        "required": False  # ⚠️ Fundamentals not relevant for intraday
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 6.0}
                    },
                    "technical": {
                        "volatilityQuality": {"min": 2.5}  # Very relaxed
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},           # No trend requirement
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None}
                    },
                    "fundamental": {
                        "dividendyield": {"min": 1.0},
                        "fundamentalScore": {"min": 7.0}
                    }
                },

            },
        },       
        "validation_modifiers": {
            "penalties": {
                "poor_fundamentals": {
                    "gates": {"fundamentalScore": {"max": 4.999}},
                    "confidence_penalty": 20,
                    "reason": "Value play requires strong underlying fundamentals"
                },
                "tight_consolidation_breaking": {
                    "gates": {"bbWidth": {"min": 4.001}},
                    "confidence_penalty": 12,
                    "reason": "Accumulation pattern invalidated by expanding volatility"
                }
            },
            "bonuses": {
                "stellar_fundamentals": {
                    "gates": {"fundamentalScore": {"min": 8.0}},
                    "confidence_boost": 15,
                    "reason": "Exceptional core business metrics"
                },
                "institutional_buying": {
                    "gates": {"rvol": {"min": 1.5}, "price": {"min_metric": "prev_close"}},
                    "confidence_boost": 10,
                    "reason": "Smart money accumulating"
                }
            }
        },
        "min_pattern_quality": 8.0,
        "min_setup_score": 70,
        "setup_type": "accumulation",
        "description": "Institutional accumulation during tight consolidation"
    },
    # ========================================================
    # VALUE SETUPS
    # ========================================================

    "DEEP_VALUE_PLAY": {
        "patterns": {
            "PRIMARY": ["bullishNecklinePattern", "bearishNecklinePattern", "threeLineStrike"],
            "CONFIRMING": ["ichimokuSignals", "cupHandle", "bollingerSqueeze"],
            "CONFLICTING": ["deathCross", "minerviniStage2"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {},
            "require_fundamentals": True,
            "fundamental_gates": {
                "peRatio": {"max": 9.999},
                "fcfYield": {"min": 5.001},
                "roe": {"min": 15}
            }
        },
        "default_priority": 82,
        "context_requirements": {
            "technical": {
                "rsi": {"max": 35},
                "priceVsPrimaryTrendPct": {"max": -10},
                "position52w": {"max": 20},
                "volatilityQuality": {"min": 2.0}
            },
            "fundamental": {
                "required": True,
                "peRatio": {"max": 10.0},
                "fcfYield": {"min": 5.0},
                "roe": {"min": 15}
            }
        },
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 2.5}
                    },
                    "fundamental": {
                        "required": False  # Not relevant for intraday
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 6.0}
                    },
                    "technical": {
                        "volatilityQuality": {"min": 2.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    },
                    "fundamental": {
                        "dividendyield": {"min": 1.5},
                        "fundamentalScore": {"min": 7.0}
                    }
                },

            },
        },
        "validation_modifiers": {
            "penalties": {
                "poor_fundamentals": {
                    "gates": {"fundamentalScore": {"max": 4.999}},
                    "confidence_penalty": 20,
                    "reason": "Value play requires strong underlying fundamentals"
                },
                "high_debt": {
                    "gates": {"deRatio": {"min": 2.0}},
                    "confidence_penalty": 10,
                    "reason": "Company is overleveraged for a safe accumulation"
                }
            },
            "bonuses": {
                "stellar_fundamentals": {
                    "gates": {"fundamentalScore": {"min": 8.0}},
                    "confidence_boost": 15,
                    "reason": "Exceptional core business metrics"
                },
                "high_dividend_yield": {
                    "gates": {"dividendyield": {"min": 4.001}},
                    "confidence_boost": 15,
                    "reason": "Strong dividend payout acts as risk cushion"
                },
                "early_reversal_signal": {
                    "gates": {"rsislope": {"min": 0.05}, "price": {"min_metric": "prev_close"}},
                    "confidence_boost": 10,
                    "reason": "Momentum starting to turn — value inflection point"
                }
            }
        },
        "min_pattern_quality": 6.0,
        "min_setup_score": 70,
        "setup_type": "value",
        "description": "Buying quality stocks at distressed valuations"
    },
    "VALUE_TURNAROUND": {
        "patterns": {
            "PRIMARY": ["bullishNecklinePattern", "bearishNecklinePattern", "goldenCross", "cupHandle"],
            "CONFIRMING": ["ichimokuSignals", "threeLineStrike", "bollingerSqueeze"],
            "CONFLICTING": ["deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "trendStrength": {"min": 3.0, "max": 5.499},
                "rsi": {"min": 45},
                "momentumStrength": {"min": 4.0}
            },
            "require_fundamentals": True,
            "fundamental_gates": {
                "roe": {"min": 18},
                "roce": {"min": 20},
                "peRatio": {"max": 11.999}
            }
        },
        "default_priority": 85,
        "context_requirements": {
            "technical": {
                "trendStrength": {"min": 3.0, "max": 5.5},
                "rsi": {"min": 45},
                "adx": {"min": 15},
                "momentumStrength": {"min": 4.0},
                "volatilityQuality": {"min": 3.0}
            },
            "fundamental": {
                "required": True,
                "roe": {"min": 18},
                "roce": {"min": 20},
                "peRatio": {"max": 12}
            }
        },
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 3.0}
                    },
                    "fundamental": {
                        "required": False
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 6.0}
                    },
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": 8},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "validation_modifiers": {
            "penalties": {
                "weak_fundamentals": {
                    "gates": {"_logic": "OR", "roe": {"max": 19.999}, "roce": {"max": 21.999}},
                    "confidence_penalty": 15,
                    "reason": "Turnaround story needs minimum earnings quality floor"
                },
                "momentum_stalling": {
                    "gates": {"rsislope": {"max": 0}, "macdhistogram": {"max": 0}},
                    "confidence_penalty": 10,
                    "reason": "Both momentum indicators still falling — too early for entry"
                }
            },
            "bonuses": {
                "strong_quality_recovery": {
                    "gates": {"roe": {"min": 22}, "momentumStrength": {"min": 6.0}},
                    "confidence_boost": 12,
                    "reason": "High quality fundamentals with confirmed momentum recovery"
                }
            }
        },
        "min_pattern_quality": 6.0,
        "min_setup_score": 75,
        "setup_type": "value",
        "description": "Fundamentally sound companies showing technical reversal"
    },
    # ========================================================
    # VOLATILITY SETUPS
    # ========================================================

    "VOLATILITY_SQUEEZE": {
        "patterns": {
            "PRIMARY": ["bollingerSqueeze", "darvasBox"],
            "CONFIRMING": ["ichimokuSignals", "goldenCross", "minerviniStage2"],
            "CONFLICTING": ["flagPennant", "threeLineStrike"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "bbWidth": {"max": 0.499},
                "volatilityQuality": {"min": 7.0},
                "adx": {"min": 15},
                # TODO: ttmSqueeze == 'Squeeze On'
            },
            "require_fundamentals": False
        },
        "default_priority": 85,
        "context_requirements": {
            "technical": {
                "bbWidth": {"max": 0.5},
                "volatilityQuality": {"min": 7.0},
                "adx": {"min": 15}
            },
            "fundamental": {
                "required": False
            }
        },
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 7.0}  # Keep tight
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 6.0}  # Slightly relaxed
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 7.0}  # Back to default
                    }
                },

            },
        },
        "validation_modifiers": {
            "penalties": {
                "lacking_direction": {
                    "gates": {"macdhistogram": {"max": -0.501}},
                    "confidence_penalty": 15,
                    "reason": "Squeeze present but no bullish MACD momentum to fire it"
                }
            },
            "bonuses": {
                "explosive_firing": {
                    "gates": {"rvol": {"min": 2.0}},
                    "confidence_boost": 15,
                    "reason": "Squeeze is firing with high volume"
                },
                "strong_directional_bias": {
                    "gates": {"adx": {"min": 20}},
                    "confidence_boost": 10,
                    "reason": "Pre-existing trend supports the squeeze"
                }
            }
        },
        "min_pattern_quality": 7.0,
        "min_setup_score": 70,
        "setup_type": "volatility",
        "description": "Coiled spring ready for explosive move"
    },
        # ========================================================
        # REVERSAL SETUPS
        # ========================================================

    "REVERSAL_MACD_CROSS_UP": {
        "patterns": {
            "PRIMARY": ["threeLineStrike", "bullishNecklinePattern", "bearishNecklinePattern"],
            "CONFIRMING": ["ichimokuSignals", "bollingerSqueeze", "goldenCross"],
            "CONFLICTING": ["deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "macdhistogram": {"min": 0.001},
                "prevmacdhistogram": {"max": 0},
                "trendStrength": {"min": 2.0}
            },
            "require_fundamentals": False
        },
        "default_priority": 75,
        "context_requirements": {
            "technical": {
                "macdhistogram": {"min": 0},
                "prevmacdhistogram": {"max": 0},
                "trendStrength": {"min": 2.0},
                "adx": {"min": 10},
                "volatilityQuality": {"min": 2.5},
                "rsislope": {"min": 0.05}
            },
            "fundamental": {
                "required": False
            }
        },
        "validation_modifiers": {
            "penalties": {
                "weak_volume_reversal": {
                    "gates": {"rvol": {"max": 0.999}},
                    "confidence_penalty": 15,
                    "reason": "Indicator cross without volume commitment"
                },
                "crushing_downtrend": {
                    "gates": {"trendStrength": {"max": 1.5}, "adx": {"min": 30}},
                    "confidence_penalty": 20,
                    "reason": "Attempting to catch a falling knife in a violent downtrend"
                }
            },
            "bonuses": {
                "deep_oversold": {
                    "gates": {"rsi": {"max": 30}},
                    "confidence_boost": 15,
                    "reason": "Elastic snap-back from extreme lows"
                }
            }
        },
        "min_pattern_quality": 6.0,
        "min_setup_score": 65,
        "setup_type": "reversal",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 12},
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 10},
                        "volatilityQuality": {"min": 2.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 6.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Early reversal via MACD momentum shift"
    },

    "REVERSAL_RSI_SWING_UP": {
        "patterns": {
            "PRIMARY": ["threeLineStrike", "bullishNecklinePattern", "bearishNecklinePattern"],
            "CONFIRMING": ["ichimokuSignals", "bollingerSqueeze"],
            "CONFLICTING": ["goldenCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},  # No specific pattern required
            "technical_gates": {
                "rsi": {"max": 34.999},
                "rsislope": {"min": 0.051000000000000004},
                "trendStrength": {"min": 2.0}
            },
            "require_fundamentals": False
        },
        "default_priority": 72,
        "context_requirements": {
            "technical": {
                "rsi": {"max": 35},
                "rsislope": {"min": 0.05},
                "trendStrength": {"min": 2.0},
                "adx": {"min": 12}
            },
            "fundamental": {"required": False}
        },
        "validation_modifiers": {
            "penalties": {
                "weak_volume_reversal": {
                    "gates": {"rvol": {"max": 0.999}},
                    "confidence_penalty": 15,
                    "reason": "Indicator cross without volume commitment"
                },
                "crushing_downtrend": {
                    "gates": {"trendStrength": {"max": 1.5}, "adx": {"min": 30}},
                    "confidence_penalty": 20,
                    "reason": "Attempting to catch a falling knife in a violent downtrend"
                },
                "rsi_not_oversold": {
                    "gates": {"rsi": {"min": 40.001}},
                    "confidence_penalty": 12,
                    "reason": "RSI swing-up requires genuine oversold conditions to be meaningful"
                }
            },
            "bonuses": {
                "deep_oversold": {
                    "gates": {"rsi": {"max": 30}},
                    "confidence_boost": 15,
                    "reason": "Elastic snap-back from extreme lows"
                }
            }
        },
        "min_pattern_quality": 5.0,
        "min_setup_score": 60,
        "setup_type": "reversal",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 12},
                        "volatilityQuality": {"min": 2.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 5.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "RSI oversold bounce reversal"
    },

    "REVERSAL_ST_FLIP_UP": {
        "patterns": {
            "PRIMARY": ["threeLineStrike"],
            "CONFIRMING": ["bollingerSqueeze", "ichimokuSignals"],
            "CONFLICTING": ["goldenCross"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                # TODO: supertrendSignal == 'Bullish',
                "prev_supertrend": {"max": 0},
                "rvol": {"min": 1.2}
            },
            "require_fundamentals": False
        },
        "default_priority": 70,
        "context_requirements": {
            "technical": {
                "rvol": {"min": 1.2},
                "adx": {"min": 14},
                "trendStrength": {"min": 2.5}
            },
            "fundamental": {"required": False}
        },
        "validation_modifiers": {
            "penalties": {},
            "bonuses": {}
        },
        "min_pattern_quality": 6.0,
        "min_setup_score": 65,
        "setup_type": "reversal",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 3.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 2.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 2.0}
                    }
                },

            },
        },
        "description": "Supertrend reversal confirmation"
    },

    "QUALITY_ACCUMULATION_DOWNTREND": {
        "patterns": {
            "PRIMARY": ["minerviniStage2", "darvasBox"],
            "CONFIRMING": ["bollingerSqueeze", "cupHandle"],
            "CONFLICTING": ["goldenCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "trendStrength": {"max": 2.999},
                "bbpercentb": {"min": 0.2, "max": 0.5}
            },
            "fundamental_gates": {
                "roe": {"min": 20},
                "roce": {"min": 25},
                "deRatio": {"max": 0.5}
            },
            "require_fundamentals": True
        },
        "default_priority": 80,
        "context_requirements": {
            "technical": {
                "trendStrength": {"max": 3.0},
                "bbpercentb": {"min": 0.2, "max": 0.5},
                "adx": {"max": 25}
            },
            "fundamental": {
                "required": True,
                "roe": {"min": 20},
                "roce": {"min": 25},
                "deRatio": {"max": 0.5}
            }
        },
        "validation_modifiers": {
            "penalties": {
                "weak_fundamentals": {
                    "gates": {"_logic": "OR", "roe": {"max": 17.999}, "roce": {"max": 21.999}},
                    "confidence_penalty": 20,
                    "reason": "Quality accumulation needs strong fundamentals"
                }
            },
            "bonuses": {}
        },
        "min_pattern_quality": 7.0,
        "min_setup_score": 70,
        "setup_type": "accumulation",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 3.0}
                    },
                    "fundamental": {
                        "required": False  # Not relevant for intraday
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 2.5}
                    },
                    "fundamental": {
                        "required": True,
                        "roe": {"min": 18},      # Slightly relaxed
                        "roce": {"min": 22},
                        "fundamentalScore": {"min": 6.0}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None}
                    },
                    "fundamental": {
                        "required": True,
                        "roe": {"min": 20},
                        "roce": {"min": 25},
                        "deRatio": {"max": 0.5},
                        "fundamentalScore": {"min": 7.0}
                    }
                },

            },
        },
        "description": "Accumulation during downtrend/sideways in quality stocks"
    },

    "SELL_AT_RANGE_TOP": {
        "patterns": {
            "PRIMARY": [],
            "CONFIRMING": ["bullishNecklinePattern", "bearishNecklinePattern"],
            "CONFLICTING": ["goldenCross", "cupHandle"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "bbWidth": {"max": 4.999},
                # TODO: price >= bbHigh * 0.98,
                "rsi": {"min": 60}
            },
            "require_fundamentals": False
        },
        "default_priority": 70,
        "context_requirements": {
            "technical": {
                "bbWidth": {"max": 5.0},
                "rsi": {"min": 60},
                "adx": {"max": 20}
            },
            "fundamental": {"required": False}
        },
        "validation_modifiers": {
            "penalties": {},
            "bonuses": {}
        },
        "min_pattern_quality": 5.0,
        "min_setup_score": 60,
        "setup_type": "exit",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 2.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Range-bound exit at resistance"
    },

    "TAKE_PROFIT_AT_MID": {
        "patterns": {
            "PRIMARY": [],
            "CONFIRMING": [],
            "CONFLICTING": []
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_gates": {
                "bbWidth": {"max": 4.999},
                # TODO: price >= bbMid * 0.98,
                # TODO: price < bbHigh * 0.98,
                "rsi": {"min": 55}
            },
            "require_fundamentals": False
        },
        "default_priority": 60,
        "context_requirements": {
            "technical": {
                "bbWidth": {"max": 5.0},
                "rsi": {"min": 55}
            },
            "fundamental": {"required": False}
        },
        "validation_modifiers": {
            "penalties": {},
            "bonuses": {}
        },
        "min_pattern_quality": 4.0,
        "min_setup_score": 50,
        "setup_type": "exit",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 3.0}
                    }
                },

            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 2.5}
                    }
                },

            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },

            },
        },
        "description": "Partial profit in range-bound consolidation"
    },

    # ========================================================
    # GENERIC FALLBACK
    # ========================================================

    "GENERIC": {
        "patterns": {
            "PRIMARY": [],
            "CONFIRMING": [],
            "CONFLICTING": []
        },
        "classification_rules": {
            # Fallback when no other setup matches
            "pattern_detection": {},
            "technical_gates": {},
            "require_fundamentals": False
        },
        "default_priority": 10,
        "context_requirements": {
            # GENERIC keeps requirements very loose – last-resort setup
            "technical": {
                # Do NOT duplicate global volume / RSI logic here
                "rvol": {"min": None},
                "rsi": {"min": None, "max": None},
                "trendStrength": {"min": None},
                "adx": {"min": None},
                "volatilityQuality": {"min": None}
            },
            "fundamental": {
                "required": False
            }
        },

        # GENERIC-specific validation penalties/bonuses
        "validation_modifiers": {
            "penalties": {
                # Technical extremes (beyond global modifiers)
                "critically_weak_trend": {
                    "gates": {"adx": {"max": 10}, "trendStrength": {"max": 2.0}},
                    "confidence_penalty": 12,
                    "reason": "Critically weak directional movement - GENERIC setup needs minimum trend"
                },
                "extreme_technical_weakness": {
                    "gates": {"technicalScore": {"max": 2.499}},
                    "confidence_penalty": 15,
                    "reason": "Extreme technical weakness - even GENERIC needs basic technical merit"
                },
                "extreme_overbought_no_setup": {
                    "gates": {"rsi": {"min": 85.001}},
                    "confidence_penalty": 12,
                    "reason": "Extreme overbought without specific setup pattern - high reversal risk"
                },

                # Fundamental extremes (filter complete garbage)
                "fundamentally_broken": {
                    "gates": {"fundamentalScore": {"max": 2.999}},
                    "confidence_penalty": 15,
                    "reason": "Fundamentally broken - GENERIC fallback doesn't excuse garbage quality"
                }
            },
            "bonuses": {
                # Reward surprising strength in GENERIC (should have matched specific setup)
                "unexpected_strong_momentum": {
                    "gates": {"momentumStrength": {"min": 7.5}, "trendStrength": {"min": 6.5}},
                    "confidence_boost": 8,
                    "reason": "Strong momentum despite no specific setup - edge case quality"
                },
                "unexpected_quality": {
                    "gates": {"fundamentalScore": {"min": 7.5}, "technicalScore": {"min": 6.5}},
                    "confidence_boost": 10,
                    "reason": "High quality profile despite no specific setup - investigate why no pattern match"
                }
            }
        },

        "min_pattern_quality": 0.0,
        "min_setup_score": 0.0,
        "setup_type": "fallback",
        "horizon_overrides": {},
        "description": "GENERIC fallback when no specific setup pattern matches"
    }
}


# ============================================================
# PATTERN METADATA (Complete Pattern Properties)
# ============================================================

PATTERN_METADATA: Dict[str, Dict[str, Any]] = {
    "bollingerSqueeze": {
        "type": "continuation",
        "timeframe_agnostic": True,
        "direction": "neutral",
        "typical_duration": {"min": 5, "max": 20},
        "failure_rate": 0.25,
        "best_horizons": ["intraday", "short_term"],
        "physics": {
            "target_ratio": 1.0,
            "duration_multiplier": 0.5,
            "max_stop_pct": 4.0,
            "horizons_supported": ["intraday", "short_term"]
        },
        "entry_rules": {
            "intraday": {
                "order_type": "stop_market",
                "trigger": "upper_band",
                "gates": {
                    "rsi": {"min": 50.0},
                    "macdhistogram": {"min": 0.0},
                    "rvol": {"min": 1.5},
                    "squeeze_duration": {"min": 5.0}  # ✅ RESTORED - prevents premature entry
                }





            },
            "short_term": {
                "order_type": "limit",
                "trigger": "close_above_band",
                "gates": {
                    "rsi": {"min": 50.0},
                    "macdhistogram": {"min": 0.0},
                    "rvol": {"min": 1.2},
                    "squeeze_duration": {"min": 3.0}  # ✅ RESTORED - lower for short_term
                }





            },
            "long_term": {
                "order_type": "limit",
                "trigger": "close_above_band",
                "gates": {
                    "rsi": {"min": 45.0},
                    "macdhistogram": {"min": -0.2},
                    "rvol": {"min": 1.0},
                    "squeeze_duration": {"min": 4.0}  # ✅ RESTORED
                }





            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "metadata_keys": {
                    "analytics": ["squeeze_duration", "squeeze_strength"]
                },
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "bbLow", "duration": 2},
                        "bbWidth": {"min": 10.0}  # ⬆️ Relaxed from 8.0
                    },




                    "_logic": "OR",

                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "bbLow", "multiplier": 0.99},
                        "bbWidth": {"min": 12.0}  # ⬆️ Relaxed from 10.0
                    },




                    "_logic": "OR",

                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "bbLow", "duration": 2},
                        "bbWidth": {"min": 15.0}  # ⬆️ Relaxed from 12.0
                    },




                    "_logic": "OR",

                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            }
        }
    },

    "cupHandle": {
        "type": "continuation",
        "timeframe_agnostic": False,
        "direction": "bullish",
        "typical_duration": {"min": 20, "max": 60},
        "failure_rate": 0.20,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            "target_ratio": 0.618,
            "duration_multiplier": 1.2,
            "max_stop_pct": 8.0,
            "min_cup_len": 20,
            "max_cup_depth": 0.50,
            "handle_len": 5,
            "require_volume": False,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "stop_market",
                "trigger": "rim_level",
                "gates": {
                    "price": {"min_metric": "rim_level", "multiplier": 0.995},
                    "rvol": {"min": 1.2}
                }



            },
            "long_term": {
                "order_type": "limit",
                "trigger": "rim_level",
                "gates": {
                    "price": {"min_metric": "rim_level", "multiplier": 0.99},
                    "rvol": {"min": 1.1}
                }



            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["cup_depth_pct", "handle_depth_pct"]
            },
            "breakdown_threshold": {
                # ❌ REMOVED intraday (pattern not suitable for this horizon)
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "handle_low", "multiplier": 0.97, "duration": 2},  # ✅ Define handleLow in indicators
                        "rvol": {"max": 0.8}  # ✅ Volume drying up
                    },




                    "_logic": "OR",

                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "handle_low", "multiplier": 0.95, "duration": 3},
                        "rvol": {"max": 0.7}
                    },




                    "_logic": "OR",

                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            },
            "handle_failure": {
                "max_handle_depth": 0.15,
                "action": "INVALIDATE_PATTERN"
            }
        }
    },

    "darvasBox": {
        "type": "breakout",
        "timeframe_agnostic": True,
        "direction": "bullish",
        "typical_duration": {"min": 8, "max": 30},
        "failure_rate": 0.30,
        "best_horizons": ["intraday", "short_term"],
        "notes": "Darvas box breakdown is binary - no monitor mode",
        "physics": {
            "target_ratio": 2.0,        #2x box depth for T1, 4x for T2
            "duration_multiplier": 1.3,
            "max_stop_pct": 5.0,
            "lookback": 50,
            "box_length": 5,
            "horizons_supported": ["intraday", "short_term"]
        },
        "entry_rules": {
            "intraday": {
                "order_type": "stop_market",
                "trigger": "box_high",
                "gates": {
                    "price": {"min_metric": "box_high", "multiplier": 1.002},  # ✅ Clearance above box
                    "rvol": {"min": 1.5},  # ✅ Volume surge (simplified from volume_surge_required)
                    "box_age_candles": {"max": 50.0}  # ✅ RESTORED - prevents stale boxes
                }




            },
            "short_term": {
                "order_type": "stop_market",
                "trigger": "box_high",
                "gates": {
                    "price": {"min_metric": "box_high", "multiplier": 1.005},
                    "rvol": {"min": 1.3},
                    "box_age_candles": {"max": 30.0}  # ✅ RESTORED - tighter for short_term
                }




            }
        },
        "invalidation": {
            "metadata_keys": {
                # Used for Velocity Analytics only
                "analytics": ["box_height_pct", "box_age_candles"] 
            },
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "box_low", "multiplier": 0.998}
                    },

                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "box_low", "multiplier": 0.995}
                    },

                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "box_low", "multiplier": 0.99, "duration": 2}
                    },

                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_IMMEDIATELY",
                "long_term": "EXIT_ON_CLOSE"
            }
        }
    }
}


# ============================================================
# PATTERN METADATA (Complete Pattern Properties)
# ============================================================

PATTERN_METADATA: Dict[str, Dict[str, Any]] = {
    "bollingerSqueeze": {
        "type": "continuation",
        "timeframe_agnostic": True,
        "direction": "neutral",
        "typical_duration": {"min": 5, "max": 20},
        "failure_rate": 0.25,
        "best_horizons": ["intraday", "short_term"],
        "physics": {
            "target_ratio": 1.0,
            "duration_multiplier": 0.5,
            "max_stop_pct": 4.0,
            "horizons_supported": ["intraday", "short_term"]
        },
        "entry_rules": {
            "intraday": {
                "order_type": "stop_market",
                "trigger": "upper_band",
                "gates": {
                    "rsi": {"min": 50.0},
                    "macdhistogram": {"min": 0.0},
                    "rvol": {"min": 1.5},
                    "squeeze_duration": {"min": 5.0}
                }
            },
            "short_term": {
                "order_type": "limit",
                "trigger": "close_above_band",
                "gates": {
                    "rsi": {"min": 50.0},
                    "macdhistogram": {"min": 0.0},
                    "rvol": {"min": 1.2},
                    "squeeze_duration": {"min": 3.0}
                }
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "close_above_band",
                "gates": {
                    "rsi": {"min": 45.0},
                    "macdhistogram": {"min": -0.2},
                    "rvol": {"min": 1.0},
                    "squeeze_duration": {"min": 4.0}
                }
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "metadata_keys": {
                    "analytics": ["squeeze_duration", "squeeze_strength"]
                },
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "bbLow", "duration": 2},
                        "bbWidth": {"min": 10.0}
                    },
                    "_logic": "OR",
                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "bbLow", "multiplier": 0.99},
                        "bbWidth": {"min": 12.0}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "bbLow", "duration": 2},
                        "bbWidth": {"min": 15.0}
                    },
                    "_logic": "OR",
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            }
        }
    },

    "cupHandle": {
        "type": "continuation",
        "timeframe_agnostic": False,
        "direction": "bullish",
        "typical_duration": {"min": 20, "max": 60},
        "failure_rate": 0.20,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            "target_ratio": 0.618,
            "duration_multiplier": 1.2,
            "max_stop_pct": 8.0,
            "min_cup_len": 20,
            "max_cup_depth": 0.50,
            "handle_len": 5,
            "require_volume": False,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "stop_market",
                "trigger": "rim_level",
                "gates": {
                    "price": {"min_metric": "rim_level", "multiplier": 0.995},
                    "rvol": {"min": 1.2}
                }
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "rim_level",
                "gates": {
                    "price": {"min_metric": "rim_level", "multiplier": 0.99},
                    "rvol": {"min": 1.1}
                }
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["cup_depth_pct", "handle_depth_pct"]
            },
            "breakdown_threshold": {
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "handle_low", "multiplier": 0.97, "duration": 2},
                        "rvol": {"max": 0.8}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "handle_low", "multiplier": 0.95, "duration": 3},
                        "rvol": {"max": 0.7}
                    },
                    "_logic": "OR",
                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            },
            "handle_failure": {
                "max_handle_depth": 0.15,
                "action": "INVALIDATE_PATTERN"
            }
        }
    },

    "darvasBox": {
        "type": "breakout",
        "timeframe_agnostic": True,
        "direction": "bullish",
        "typical_duration": {"min": 8, "max": 30},
        "failure_rate": 0.30,
        "best_horizons": ["intraday", "short_term"],
        "physics": {
            "target_ratio": 2.0,
            "duration_multiplier": 1.3,
            "max_stop_pct": 5.0,
            "lookback": 50,
            "box_length": 5,
            "horizons_supported": ["intraday", "short_term"]
        },
        "entry_rules": {
            "intraday": {
                "order_type": "stop_market",
                "trigger": "box_high",
                "gates": {
                    "price": {"min_metric": "box_high", "multiplier": 1.002},
                    "rvol": {"min": 1.5},
                    "box_age_candles": {"max": 50.0}
                }
            },
            "short_term": {
                "order_type": "stop_market",
                "trigger": "box_high",
                "gates": {
                    "price": {"min_metric": "box_high", "multiplier": 1.005},
                    "rvol": {"min": 1.3},
                    "box_age_candles": {"max": 30.0}
                }
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["box_height_pct", "box_age_candles"] 
            },
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "box_low", "multiplier": 0.998}
                    }
                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "box_low", "multiplier": 0.995}
                    }
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "box_low", "multiplier": 0.99, "duration": 2}
                    }
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_IMMEDIATELY",
                "long_term": "EXIT_ON_CLOSE"
            }
        }
    },

    "minerviniStage2": {
        "type": "accumulation",
        "timeframe_agnostic": False,
        "direction": "bullish",
        "typical_duration": {"min": 10, "max": 40},
        "failure_rate": 0.15,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            "target_ratio": 1.0,
            "duration_multiplier": 1.8,
            "max_stop_pct": 7.0,
            "min_contraction_pct": 1.5,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit",
                "trigger": "pivot_point",
                "gates": {
                    "volatilityQuality": {"max": 1.5},
                    "price": {"min_metric": "pivot_point", "multiplier": 1.01},
                    "position52w": {"min": 80.0}
                }
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "pivot_point",
                "gates": {
                    "volatilityQuality": {"max": 2.0},
                    "price": {"min_metric": "pivot_point", "multiplier": 1.005},
                    "position52w": {"min": 70.0}
                }
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["contraction_pct", "volatility_quality"]
            },
            "breakdown_threshold": {
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "maFast", "multiplier": 0.95, "duration": 2},
                        "volatilityQuality": {"max": 4.0}
                    },
                    "_logic": "OR"
                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            },
            "stage_reversion": {
                "gates": {
                    "price": {"max_metric": "maFast", "duration": 5},
                    "rvol": {"max": 0.8, "duration": 5}
                },
                "_logic": "AND",
                "action": "EXIT_ON_CLOSE"
            }
        }
    },

    "flagPennant": {
        "type": "continuation",
        "timeframe_agnostic": True,
        "direction": "trend_aligned",
        "typical_duration": {"min": 5, "max": 15},
        "failure_rate": 0.25,
        "best_horizons": ["intraday", "short_term", "long_term"],
        "physics": {
            "target_ratio": 0.5,
            "duration_multiplier": 0.8,
            "max_stop_pct": 6.0,
            "horizons_supported": ["intraday", "short_term"]
        },
        "entry_rules": {
            "intraday": {
                "order_type": "stop_market",
                "trigger": "flag_high",
                "gates": {
                    "pole_length": {"min": 8.0},
                    "flag_tightness": {"max": 0.03},
                    "price": {"min_metric": "flag_high", "multiplier": 1.005}
                }
            },
            "short_term": {
                "order_type": "stop_market",
                "trigger": "flag_high",
                "gates": {
                    "pole_length": {"min": 5.0},
                    "flag_tightness": {"max": 0.05},
                    "price": {"min_metric": "flag_high", "multiplier": 1.01}
                }
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["pole_strength", "flag_tightness"]
            },
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "flag_low", "multiplier": 0.998},
                        "trendStrength": {"max": 4.0}
                    },
                    "_logic": "OR",
                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "flag_low", "multiplier": 0.995},
                        "adx": {"max": 15.0}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "flag_low", "multiplier": 0.99},
                        "adx": {"max": 12.0}
                    },
                    "_logic": "OR",
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_IMMEDIATELY",
                "long_term": "EXIT_ON_CLOSE"
            },
            "expiration": {
                "enabled": True,
                "max_duration_candles": {
                    "intraday": 20,
                    "short_term": 15,
                    "long_term": 12
                },
                "action_on_expire": "DOWNGRADE_TO_CONSOLIDATION"
            }
        }
    },

    "threeLineStrike": {
        "type": "reversal",
        "timeframe_agnostic": True,
        "direction": "counter_trend",
        "typical_duration": {"min": 1, "max": 5},
        "failure_rate": 0.35,
        "best_horizons": ["intraday", "short_term"],
        "physics": {
            **DEFAULT_PHYSICS,
            "horizons_supported": ["intraday", "short_term", "long_term"]
        },
        "entry_rules": {
            "intraday": {
                "order_type": "market",
                "trigger": "candle_close",
                "gates": {
                    "strike_candle_body": {"min": 0.6},
                    "rvol": {"min": 1.3}
                }
            },
            "short_term": {
                "order_type": "limit",
                "trigger": "candle_close",
                "gates": {
                    "strike_candle_body": {"min": 0.7},
                    "rvol": {"min": 1.2}
                }
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "strike_low", "multiplier": 0.995, "duration": 2},
                        "rsi": {"max": 45.0}
                    },
                    "_logic": "OR",
                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "strike_low", "multiplier": 0.99, "duration": 3},
                        "rsi": {"max": 40.0}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "strike_low", "multiplier": 0.98, "duration": 3},
                        "macdhistogram": {"max": 0.0}
                    },
                    "_logic": "OR",
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            },
            "expiration": {
                "enabled": True,
                "max_duration_candles": {
                    "intraday": 15,
                    "short_term": 12,
                    "long_term": 10
                },
                "action": "EXIT_ON_CLOSE"
            }
        }
    },

    "ichimokuSignals": {
        "type": "trend_confirmation",
        "timeframe_agnostic": True,
        "direction": "variable",
        "typical_duration": {"min": 5, "max": 100},
        "failure_rate": 0.30,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            **DEFAULT_PHYSICS,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit",
                "trigger": "tenkan_kijun_cross",
                "gates": {
                    "cloud_thickness": {"min": 0.01},
                    "tenkan_kijun_spread": {"min": 0.005},
                    "price": {"min_metric": "ichiSpanA"}
                }
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "kumo_breakout",
                "gates": {
                    "cloud_thickness": {"min": 0.02},
                    "tenkan_kijun_spread": {"min": 0.01},
                    "price": {"min_metric": "ichiSpanA"}
                }
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "ichiSpanA", "multiplier": 0.998, "duration": 2},
                        "ichiTenkan": {"max_metric": "ichiKijun"}
                    },
                    "_logic": "OR",
                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "ichiSpanA", "multiplier": 0.995, "duration": 2},
                        "chikouSpan": {"max_metric": "price"}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "ichiSpanA", "duration": 3},
                        "ichiTenkan": {"max_metric": "ichiKijun", "duration": 3}
                    },
                    "_logic": "AND"
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
        }
    },

    "goldenCross": {
        "type": "trend_change",
        "timeframe_agnostic": False,
        "direction": "bullish",
        "typical_duration": {"min": 30, "max": 200},
        "failure_rate": 0.20,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            "target_ratio": 0,
            "duration_multiplier": 2.0,
            "max_stop_pct": None,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "gates": {
                    "maMid": {"min_metric": "maSlow", "multiplier": 1.002},
                    "rvol": {"min": 1.1}
                }
            },
            "long_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "gates": {
                    "maMid": {"min_metric": "maSlow", "multiplier": 1.005},
                    "rvol": {"min": 1.0}
                }
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "short_term": {
                    "gates": {
                        "maMid": {"max_metric": "maSlow", "duration": 5},
                        "price": {"max_metric": "maMid", "multiplier": 0.95}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "maMid": {"max_metric": "maSlow", "duration": 10},
                        "adx": {"max": 15.0}
                    },
                    "_logic": "OR",
                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "EXIT_ON_CLOSE"
            }
        }
    },

    "deathCross": {
        "type": "trend_change",
        "timeframe_agnostic": False,
        "direction": "bearish",
        "typical_duration": {"min": 30, "max": 200},
        "failure_rate": 0.20,
        "best_horizons": ["long_term"],
        "physics": {
            "target_ratio": 0,
            "duration_multiplier": 2.0,
            "max_stop_pct": None,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "gates": {
                    "maMid": {"max_metric": "maSlow", "multiplier": 0.998},
                    "rvol": {"min": 1.1}
                }
            },
            "long_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "gates": {
                    "maMid": {"max_metric": "maSlow", "multiplier": 0.995},
                    "rvol": {"min": 1.0}
                }
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "short_term": {
                    "gates": {
                        "maMid": {"min_metric": "maSlow", "duration": 5},
                        "rsi": {"min": 60.0}
                    },
                    "_logic": "OR",
                },
                "long_term": {
                    "gates": {
                        "maMid": {"min_metric": "maSlow", "duration": 10},
                        "price": {"min_metric": "maMid", "multiplier": 1.05}
                    },
                    "_logic": "OR",
                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "EXIT_ON_CLOSE"
            }
        }
    },

    "bearishNecklinePattern": {
        "type": "reversal",
        "timeframe_agnostic": False,
        "direction": "bearish",
        "typical_duration": {"min": 15, "max": 50},
        "failure_rate": 0.25,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            **DEFAULT_PHYSICS,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "stop_market",
                "trigger": "neckline",
                "gates": {
                    "price": {"max_metric": "neckline", "multiplier": 0.99},
                    "rvol": {"min": 1.2},
                    "peak_similarity": {"max": 0.02}
                }
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "neckline_confirmation",
                "gates": {
                    "price": {"max_metric": "neckline", "multiplier": 0.995},
                    "peak_similarity": {"max": 0.03}
                }
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["pattern_height_pct", "peak_similarity"]
            },
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"min_metric": "neckline", "multiplier": 1.002, "duration": 1},
                        "rvol": {"max": 0.8}
                    },
                    "_logic": "OR"
                },
                "short_term": {
                    "gates": {
                        "price": {"min_metric": "neckline", "multiplier": 1.005, "duration": 2},
                        "rsi": {"min": 60}
                    },
                    "_logic": "OR"
                },
                "long_term": {
                    "gates": {
                        "price": {"min_metric": "neckline", "multiplier": 1.01, "duration": 3},
                        "macdhistogram": {"min": 0}
                    },
                    "_logic": "OR"
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            }
        }
    },

    "bullishNecklinePattern": {
        "type": "reversal",
        "timeframe_agnostic": False,
        "direction": "bullish",
        "typical_duration": {"min": 15, "max": 50},
        "failure_rate": 0.25,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            **DEFAULT_PHYSICS,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "stop_market",
                "trigger": "neckline",
                "gates": {
                    "price": {"min_metric": "neckline", "multiplier": 1.01},
                    "rvol": {"min": 1.2},
                    "peak_similarity": {"max": 0.02}
                }
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "neckline_confirmation",
                "gates": {
                    "price": {"min_metric": "neckline", "multiplier": 1.005},
                    "peak_similarity": {"max": 0.03}
                }
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["pattern_height_pct", "peak_similarity"]
            },
            "breakdown_threshold": {
                "intraday": {
                    "gates": {
                        "price": {"max_metric": "neckline", "multiplier": 0.998, "duration": 1},
                        "rvol": {"max": 0.8}
                    },
                    "_logic": "OR"
                },
                "short_term": {
                    "gates": {
                        "price": {"max_metric": "neckline", "multiplier": 0.995, "duration": 2},
                        "rsi": {"max": 40}
                    },
                    "_logic": "OR"
                },
                "long_term": {
                    "gates": {
                        "price": {"max_metric": "neckline", "multiplier": 0.99, "duration": 3},
                        "macdhistogram": {"max": 0}
                    },
                    "_logic": "OR"
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            }
        }
    }
}

# ============================================================
# PATTERN INDICATOR KEY MAPPINGS (By Horizon)
# ============================================================

#obsolete patterns todo delete
PATTERN_INDICATOR_MAPPINGS = {
    "bollingerSqueeze": {
        "intraday": "bollingerSqueezeIntraday",
        "short_term": "bollingerSqueezeShortTerm",
        "long_term": "bollingerSqueezeLongTerm"
    },
    "minerviniStage2": {
        "intraday": "minerviniStage2Intraday",
        "short_term": "minerviniStage2ShortTerm",
        "long_term": "minerviniStage2LongTerm"
    },
    "ichimokuSignals": {
        "intraday": "ichimokuSignalsIntraday",
        "short_term": "ichimokuSignalsShortTerm",
        "long_term": "ichimokuSignalsLongTerm"
    },
    "goldenCross": {
        "intraday": "goldenCrossIntraday",
        "short_term": "goldenCrossShortTerm",
        "long_term": "goldenCrossLongTerm"
    },
    "bullishNecklinePattern": {
        "intraday": "bullishNecklineIntraday",
        "short_term": "bullishNecklineShortTerm",
        "long_term": "bullishNecklineLongTerm"
    },
    "bearishNecklinePattern": {
        "intraday": "bearishNecklineIntraday",
        "short_term": "bearishNecklineShortTerm",
        "long_term": "bearishNecklineLongTerm"
    },
    "cupHandle": {
        "intraday": "cupHandleIntraday",
        "short_term": "cupHandleShortTerm",
        "long_term": "cupHandleLongTerm"
    },
    "flagPennant": {
        "intraday": "flagPennantIntraday",
        "short_term": "flagPennantShortTerm",
        "long_term": "flagPennantLongTerm"
    },
    "darvasBox": {
        "intraday": "darvasBoxIntraday",
        "short_term": "darvasBoxShortTerm",
        "long_term": "darvasBoxLongTerm"
    },
    "threeLineStrike": {
        "intraday": "threeLineStrikeIntraday",
        "short_term": "threeLineStrikeShortTerm",
        "long_term": "threeLineStrikeLongTerm"
    }
}


__all__ = [
    SETUP_PATTERN_MATRIX,
    PATTERN_METADATA,
    DEFAULT_PHYSICS,
    PATTERN_INDICATOR_MAPPINGS,
    PATTERN_SCORING_THRESHOLDS
]
