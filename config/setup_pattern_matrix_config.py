# config/setup_pattern_matrix.py
"""
Setup-Pattern Mapping Matrix - PRODUCTION VERSION v3.0

Complete pattern genome with:
✓ Setup classification rules (moved from master_config)
✓ Default priority scores (horizons override)
✓ Default confidence base floors (horizons adjust)
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
            "CONFLICTING": ["doubleTopBottom", "deathCross"]
        },
        "classification_rules": {
            "pattern_detection": {
                "darvasBox": True  # Boolean flag check
            },
            "technical_conditions": [
                "rvol >= 1.5",
                "trendStrength >= 4.0"
            ],
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
                    "condition": "rvol < 1.3",
                    "amount": 15,
                    "reason": "Darvas breakout needs strong volume"
                }
            },
            "bonuses": {
                "explosive_volume": {
                    "condition": "rvol >= 3.0",
                    "amount": 10,
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
                "opportunity": {
                    "confidence": {"min": 70},
                    "rrRatio": {"min": 1.5}
                }
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
                "opportunity": {
                    "confidence": {"min": 65},
                    "rrRatio": {"min": 1.5}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": 1.2}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}  # Not ideal for multibagger
                }
            }
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
            "technical_conditions": [
                "volatilityQuality >= 6.0",
                "rsi >= 50"
            ],
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
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    }
                }
            }
        },
        "validation_modifiers": {
            "penalties": {
                "poor_stage2": {
                    "condition": "position52w < 80",
                    "amount": 20,
                    "reason": "VCP requires stock near 52W highs"
                }
            },
            "bonuses": {}
        },
        "min_pattern_quality": 8.5,
        "min_setup_score": 85,
        "setup_type": "pattern_driven",
        "horizon_overrides": {
            "intraday": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 17},
                        "trendStrength": {"min": 5.5},
                        "volatilityQuality": {"min": 7.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 70}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": 6.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 65}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 5.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 8.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 65}
                }
            }
        },
        "description": "Minervini-style volatility contraction pattern"
    },

    "PATTERN_CUP_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["cupHandle"],
            "CONFIRMING": ["goldenCross", "bollingerSqueeze", "ichimokuSignals"],
            "CONFLICTING": ["doubleTopBottom", "deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {
                "cupHandle": True
            },
            "technical_conditions": [
                "rvol >= 1.2",
                "trendStrength >= 3.5"
            ],
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
                    "condition": "rvol < 1.2",
                    "amount": 15,
                    "reason": "Cup & Handle breakout requires strong volume confirmation"
                },
                "weak_momentum": {
                    "condition": "rsi < 55",
                    "amount": 10,
                    "reason": "Lacking momentum for sustained breakout"
                }
            },
            "bonuses": {
                "explosive_volume": {
                    "condition": "rvol >= 2.5",
                    "amount": 15,
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
                "opportunity": {
                    "confidence": {"min": 65}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": 4.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.0},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 2.5},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            }
        },
        "description": "Cup and handle breakout pattern"
    },

    "PATTERN_FLAG_BREAKOUT": {
        "patterns": {
            "PRIMARY": ["flagPennant"],
            "CONFIRMING": ["bollingerSqueeze", "ichimokuSignals"],
            "CONFLICTING": ["doubleTopBottom", "threeLineStrike", "cupHandle"]
        },
        "classification_rules": {
            "pattern_detection": {
                "flagPennant": True
            },
            "technical_conditions": [
                "rvol >= 1.5",
                "trendStrength >= 5.5"
            ],
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
                    "condition": "rsi >= 85",
                    "amount": 15,
                    "reason": "Trend is too overextended, risk of failed breakout"
                },
                "low_volume_push": {
                    "condition": "rvol < 1.0",
                    "amount": 20,
                    "reason": "Flag breakout without volume is a bull trap"
                }
            },
            "bonuses": {
                "strong_momentum_push": {
                    "condition": "macdhistogram > 0.0",
                    "amount": 10,
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
                "opportunity": {
                    "confidence": {"min": 70}
                }
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
                "opportunity": {
                    "confidence": {"min": 65}
                }
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
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}  # Not ideal for multibagger
                }
            }
        },
        "description": "Flag/pennant continuation breakout"
    },

    "PATTERN_STRIKE_REVERSAL": {
        "patterns": {
            "PRIMARY": ["threeLineStrike"],
            "CONFIRMING": ["bollingerSqueeze", "ichimokuSignals", "doubleTopBottom"],
            "CONFLICTING": ["deathCross", "flagPennant", "minerviniStage2"]
        },
        "classification_rules": {
            "pattern_detection": {
                "threeLineStrike": True
            },
            "technical_conditions": [
                "rvol >= 1.3",
                "rsi >= 45"
            ],
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
                    "condition": "rvol < 1.0",
                    "amount": 15,
                    "reason": "Reversals without volume usually fail"
                },
                "fighting_strong_trend": {
                    "condition": "trendStrength >= 5.0 and adx >= 25",
                    "amount": 20,
                    "reason": "Attempting to reverse a very strong trend"
                }
            },
            "bonuses": {
                "extreme_oversold_bounce": {
                    "condition": "rsi <= 35",
                    "amount": 15,
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
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 12},
                        "volatilityQuality": {"min": 2.5},
                        "rvol": {"min": 1.3}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": 1.2}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None},
                        "rvol": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
        },
        "description": "Three-line strike reversal pattern"
    },

    "PATTERN_GOLDEN_CROSS": {
        "patterns": {
            "PRIMARY": ["goldenCross"],
            "CONFIRMING": ["cupHandle", "minerviniStage2", "ichimokuSignals"],
            "CONFLICTING": ["deathCross", "doubleTopBottom"]
        },
        "classification_rules": {
            "pattern_detection": {
                "goldenCross": True
            },
            "technical_conditions": [
                "trendStrength >= 3.0",
                "momentumStrength >= 4.0"
            ],
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
                    "condition": "adx < 15",
                    "amount": 20,
                    "reason": "Golden cross in a flat/choppy market is unreliable"
                }
            },
            "bonuses": {
                "strong_trend_confirmed": {
                    "condition": "adx >= 25",
                    "amount": 15,
                    "reason": "Cross confirmed by strong directional trend"
                },
                "volume_backed": {
                    "condition": "rvol >= 1.5",
                    "amount": 10,
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
                "opportunity": {
                    "confidence": {"min": 65}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 3.0},
                        "volatilityQuality": {"min": 3.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
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
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 8.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 2.0},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            }
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
            "CONFLICTING": ["doubleTopBottom", "deathCross", "minerviniStage2"]
        },
        "classification_rules": {
            "pattern_detection": {},  # No specific pattern required
            "technical_conditions": [
                "bbpercentb >= 0.98",
                "rsi >= 60",
                "rvol >= 1.5"
            ],
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
                "opportunity": {
                    "confidence": {"min": 65},  # Higher bar for intraday
                    "rrRatio": {"min": 1.5}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": 4.0}  # Standard
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60},
                    "rrRatio": {"min": 1.5}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},            # ⬇️ Relax for long_term
                        "trendStrength": {"min": 4.5},   # ⬇️ Lower requirement
                        "volatilityQuality": {"min": None}  # ⬇️ No requirement
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}  # Lower bar for long_term
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},  # ⬇️ Don't care about momentum
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
        },
        "validation_modifiers": {
            "penalties": {
                "low_breakout_volume": {
                    "condition": "rvol < 2.0",
                    "amount": 10,
                    "reason": "Breakout needs volume confirmation"
                },
                "weak_trend": {
                    "condition": "trendStrength < 5.0",
                    "amount": 15,
                    "reason": "Momentum breakout needs strong trend"
                }
            },
            "bonuses": {
                "volume_surge": {
                    "condition": "rvol >= 3.0",
                    "amount": 10,
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
            "PRIMARY": ["doubleTopBottom", "deathCross"],
            "CONFIRMING": ["bollingerSqueeze"],
            "CONFLICTING": ["goldenCross", "cupHandle", "minerviniStage2", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "bbpercentb <= 0.02",
                "rsi <= 40",
                "rvol >= 1.5"
            ],
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
                    "condition": "rvol < 2.0",
                    "amount": 12,
                    "reason": "Breakdown volume weak relative to confirmation threshold"
                }
            },
            "bonuses": {
                "panic_selling": {
                    "condition": "rvol >= 2.0",
                    "amount": 15,
                    "reason": "High volume institutional distribution"
                },
                "heavy_momentum": {
                    "condition": "rsi <= 30",
                    "amount": 10,
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
                "opportunity": {
                    "confidence": {"min": 65},
                    "rrRatio": {"min": 1.5}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 5.0},
                        "volatilityQuality": {"min": 3.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 65},
                    "rrRatio": {"min": 1.5}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 4.5},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}  # Block for multibagger
                }
            }
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
            "CONFLICTING": ["doubleTopBottom", "deathCross", "cupHandle"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "trendStrength >= 4.0",
                "price > maFast",
                "abs(price - maFast) / maFast <= 0.05",
                "rsi >= 50",
                "momentumStrength >= 4.0"
            ],
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
                    "condition": "rsi < 52",
                    "amount": 10,
                    "reason": "Pullback cut too deep, momentum is broken"
                },
                "high_volume_selling": {
                    "condition": "rvol >= 2.0 and price < prev_close",
                    "amount": 15,
                    "reason": "Pullback has too much selling volume"
                }
            },
            "bonuses": {
                "light_volume_pullback": {
                    "condition": "rvol <= 0.8",
                    "amount": 10,
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
                "opportunity": {
                    "confidence": {"min": 60},
                    "rrRatio": {"min": 1.5}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 16},
                        "trendStrength": {"min": 4.0},
                        "volatilityQuality": {"min": 3.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55},
                    "rrRatio": {"min": 1.4}
                }
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
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.0},   # Very relaxed
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
        },
        "description": "Dip-buying in established uptrend"
    },

    "DEEP_PULLBACK": {
        "patterns": {
            "PRIMARY": ["goldenCross", "ichimokuSignals"],
            "CONFIRMING": ["bollingerSqueeze", "threeLineStrike", "cupHandle"],
            "CONFLICTING": ["doubleTopBottom", "deathCross"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "trendStrength >= 3.5",
                "priceVsPrimaryTrendPct >= -10",
                "priceVsPrimaryTrendPct <= -5"
            ],
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
                    "condition": "rsi < 35",
                    "amount": 20,
                    "reason": "Pullback cut too deep, momentum structurally broken"
                },
                "high_volume_selling": {
                    "condition": "rvol >= 2.5 and price < prev_close",
                    "amount": 15,
                    "reason": "Heavy distribution — not a healthy retracement"
                }
            },
            "bonuses": {
                "light_volume_pullback": {
                    "condition": "rvol <= 0.8",
                    "amount": 10,
                    "reason": "Healthy, low-volume retracement into support zone"
                },
                "strong_trend_context": {
                    "condition": "trendStrength >= 6.0 and adx >= 25",
                    "amount": 8,
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
                "opportunity": {
                    "confidence": {"min": 55},
                    "rrRatio": {"min": 2.0}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 3.5},
                        "volatilityQuality": {"min": 3.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50},
                    "rrRatio": {"min": 2.0}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 3.0},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 48},
                    "rrRatio": {"min": 2.5}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 2.5},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50},
                    "rrRatio": {"min": 3.0}
                }
            }
        },
        "description": "Deeper retracement in strong trend"
    },

    "TREND_FOLLOWING": {
        "patterns": {
            "PRIMARY": ["goldenCross", "ichimokuSignals"],
            "CONFIRMING": ["flagPennant", "bollingerSqueeze"],
            "CONFLICTING": ["doubleTopBottom", "deathCross", "threeLineStrike"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "rsi >= 55",
                "macdhistogram >= 0"
            ],
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
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 4.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": 15},
                        "trendStrength": {"min": 4.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
        },
        "validation_modifiers": {
            "penalties": {
                "decelerating_momentum": {
                    "condition": "rsislope < -0.05 and macdhistogram < 0",
                    "amount": 12,
                    "reason": "Both RSI and MACD momentum decelerating in trend setup"
                },
                "overextended": {
                    "condition": "rsi > 80",
                    "amount": 10,
                    "reason": "Overbought — elevated mean-reversion risk"
                }
            },
            "bonuses": {
                "strong_trend_confluence": {
                    "condition": "adx >= 30 and trendStrength >= 7.0",
                    "amount": 10,
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
            "technical_conditions": [
                "rsi <= 45",
                "macdhistogram <= 0",
                "trendStrength >= 5.0"
            ],
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
                    "condition": "rvol < 1.0",
                    "amount": 15,
                    "reason": "Price drifting lower without selling pressure"
                }
            },
            "bonuses": {
                "panic_selling": {
                    "condition": "rvol >= 2.0",
                    "amount": 15,
                    "reason": "High volume institutional distribution"
                },
                "heavy_momentum": {
                    "condition": "rsi <= 30",
                    "amount": 10,
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
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 20},
                        "trendStrength": {"min": 5.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 18},
                        "trendStrength": {"min": 4.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},            # Not applicable
                        "trendStrength": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}          # Block for multibagger
                }
            }
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
            "CONFLICTING": ["doubleTopBottom", "deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "bbWidth < 5.0",
                "rvol < 1.0",
                "rsi >= 40",
                "rsi <= 60"
            ],
            "fundamental_conditions": [
                "roe >= 20 or roe3yAvg >= 18",
                "roce >= 25",
                "deRatio <= 0.5"
            ]
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
                "opportunity": {
                    "confidence": {"min": 55}
                }
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
                "opportunity": {
                    "confidence": {"min": 45}
                }
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
                "opportunity": {
                    "confidence": {"min": None},  # Will use setup default
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "dividendyield": {"min": 1.0},
                        "fundamentalScore": {"min": 8.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60},  # Higher bar for multibagger
                }
            }
        },       
        "validation_modifiers": {
            "penalties": {
                "poor_fundamentals": {
                    "condition": "fundamentalScore < 5.0",
                    "amount": 20,
                    "reason": "Value play requires strong underlying fundamentals"
                },
                "tight_consolidation_breaking": {
                    "condition": "bbWidth > 4.0",
                    "amount": 12,
                    "reason": "Accumulation pattern invalidated by expanding volatility"
                }
            },
            "bonuses": {
                "stellar_fundamentals": {
                    "condition": "fundamentalScore >= 8.0",
                    "amount": 15,
                    "reason": "Exceptional core business metrics"
                },
                "institutional_buying": {
                    "condition": "rvol >= 1.5 and price > prev_close",
                    "amount": 10,
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
            "PRIMARY": ["doubleTopBottom", "threeLineStrike"],
            "CONFIRMING": ["ichimokuSignals", "cupHandle", "bollingerSqueeze"],
            "CONFLICTING": ["deathCross", "minerviniStage2"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [],
            "require_fundamentals": True,
            "fundamental_conditions": [
                "peRatio < 10.0",
                "fcfYield > 5.0",
                "roe >= 15"
            ]
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
                "opportunity": {
                    "confidence": {"min": 45}
                }
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
                "opportunity": {
                    "confidence": {"min": 40}
                }
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
                "opportunity": {
                    "confidence": {"min": 35}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "dividendyield": {"min": 1.5},
                        "fundamentalScore": {"min": 8.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
        },
        "validation_modifiers": {
            "penalties": {
                "poor_fundamentals": {
                    "condition": "fundamentalScore < 5.0",
                    "amount": 20,
                    "reason": "Value play requires strong underlying fundamentals"
                },
                "high_debt": {
                    "condition": "deRatio >= 2.0",
                    "amount": 10,
                    "reason": "Company is overleveraged for a safe accumulation"
                }
            },
            "bonuses": {
                "stellar_fundamentals": {
                    "condition": "fundamentalScore >= 8.0",
                    "amount": 15,
                    "reason": "Exceptional core business metrics"
                },
                "high_dividend_yield": {
                    "condition": "dividendyield > 4.0",
                    "amount": 15,
                    "reason": "Strong dividend payout acts as risk cushion"
                },
                "early_reversal_signal": {
                    "condition": "rsislope > 0.05 and price > prev_close",
                    "amount": 10,
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
            "PRIMARY": ["doubleTopBottom", "goldenCross", "cupHandle"],
            "CONFIRMING": ["ichimokuSignals", "threeLineStrike", "bollingerSqueeze"],
            "CONFLICTING": ["deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "trendStrength >= 3.0",
                "trendStrength < 5.5",
                "rsi >= 45",
                "momentumStrength >= 4.0"
            ],
            "require_fundamentals": True,
            "fundamental_conditions": [
                "roe >= 18",
                "roce >= 20",
                "peRatio < 12"
            ]
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
                "opportunity": {
                    "confidence": {"min": 50}
                }
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
                "opportunity": {
                    "confidence": {"min": 50}
                }
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
                "opportunity": {
                    "confidence": {"min": 45}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 8.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 65}
                }
            }
        },
        "validation_modifiers": {
            "penalties": {
                "weak_fundamentals": {
                    "condition": "roe < 20 or roce < 22",
                    "amount": 15,
                    "reason": "Turnaround story needs minimum earnings quality floor"
                },
                "momentum_stalling": {
                    "condition": "rsislope < 0 and macdhistogram < 0",
                    "amount": 10,
                    "reason": "Both momentum indicators still falling — too early for entry"
                }
            },
            "bonuses": {
                "strong_quality_recovery": {
                    "condition": "roe >= 22 and momentumStrength >= 6.0",
                    "amount": 12,
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
            "technical_conditions": [
                "bbWidth < 0.5",
                "volatilityQuality >= 7.0",
                "adx >= 15",
                "ttmSqueeze == 'Squeeze On'"
            ],
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
                "opportunity": {
                    "confidence": {"min": 60},
                    "rrRatio": {"min": 1.8}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 6.0}  # Slightly relaxed
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 7.0}  # Back to default
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}  # Block for multibagger
                }
            }
        },
        "validation_modifiers": {
            "penalties": {
                "lacking_direction": {
                    "condition": "macdhistogram < -0.5",
                    "amount": 15,
                    "reason": "Squeeze present but no bullish MACD momentum to fire it"
                }
            },
            "bonuses": {
                "explosive_firing": {
                    "condition": "rvol >= 2.0",
                    "amount": 15,
                    "reason": "Squeeze is firing with high volume"
                },
                "strong_directional_bias": {
                    "condition": "adx >= 20",
                    "amount": 10,
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
            "PRIMARY": ["threeLineStrike", "doubleTopBottom"],
            "CONFIRMING": ["ichimokuSignals", "bollingerSqueeze", "goldenCross"],
            "CONFLICTING": ["deathCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "macdhistogram > 0",
                "prevmacdhistogram <= 0",
                "trendStrength >= 2.0"
            ],
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
                    "condition": "rvol < 1.0",
                    "amount": 15,
                    "reason": "Indicator cross without volume commitment"
                },
                "crushing_downtrend": {
                    "condition": "trendStrength <= 1.5 and adx >= 30",
                    "amount": 20,
                    "reason": "Attempting to catch a falling knife in a violent downtrend"
                }
            },
            "bonuses": {
                "deep_oversold": {
                    "condition": "rsi <= 30",
                    "amount": 15,
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
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 10},
                        "volatilityQuality": {"min": 2.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
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
                "opportunity": {
                    "confidence": {"min": 48}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            }
        },
        "description": "Early reversal via MACD momentum shift"
    },

    "REVERSAL_RSI_SWING_UP": {
        "patterns": {
            "PRIMARY": ["threeLineStrike", "doubleTopBottom"],
            "CONFIRMING": ["ichimokuSignals", "bollingerSqueeze"],
            "CONFLICTING": ["goldenCross", "flagPennant"]
        },
        "classification_rules": {
            "pattern_detection": {},  # No specific pattern required
            "technical_conditions": [
                "rsi < 35",
                "rsislope > 0.05",
                "trendStrength >= 2.0"
            ],
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
                    "condition": "rvol < 1.0",
                    "amount": 15,
                    "reason": "Indicator cross without volume commitment"
                },
                "crushing_downtrend": {
                    "condition": "trendStrength <= 1.5 and adx >= 30",
                    "amount": 20,
                    "reason": "Attempting to catch a falling knife in a violent downtrend"
                },
                "rsi_not_oversold": {
                    "condition": "rsi > 40",
                    "amount": 12,
                    "reason": "RSI swing-up requires genuine oversold conditions to be meaningful"
                }
            },
            "bonuses": {
                "deep_oversold": {
                    "condition": "rsi <= 30",
                    "amount": 15,
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
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 12},
                        "volatilityQuality": {"min": 2.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
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
                "opportunity": {
                    "confidence": {"min": 45}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            }
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
            "technical_conditions": [
                "supertrendSignal == 'Bullish'",
                "prev_supertrend <= 0",
                "rvol >= 1.2"
            ],
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
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": 14},
                        "trendStrength": {"min": 2.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": 2.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 52}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "fundamentalScore": {"min": 7.0}
                    },
                    "technical": {
                        "adx": {"min": None},
                        "trendStrength": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
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
            "technical_conditions": [
                "trendStrength < 3.0",
                "bbpercentb >= 0.2",
                "bbpercentb <= 0.5"
            ],
            "fundamental_conditions": [
                "roe >= 20",
                "roce >= 25",
                "deRatio <= 0.5"
            ],
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
                    "condition": "roe < 18 or roce < 22",
                    "amount": 20,
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
                "opportunity": {
                    "confidence": {"min": 45}
                }
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
                "opportunity": {
                    "confidence": {"min": 40}
                }
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
                "opportunity": {
                    "confidence": {"min": None},  # Use default
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "fundamental": {
                        "required": True,
                        "roe": {"min": 22},      # Higher bar for multibagger
                        "roce": {"min": 28},
                        "deRatio": {"max": 0.4},
                        "fundamentalScore": {"min": 8.0}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            }
        },
        "description": "Accumulation during downtrend/sideways in quality stocks"
    },

    "SELL_AT_RANGE_TOP": {
        "patterns": {
            "PRIMARY": [],
            "CONFIRMING": ["doubleTopBottom"],
            "CONFLICTING": ["goldenCross", "cupHandle"]
        },
        "classification_rules": {
            "pattern_detection": {},
            "technical_conditions": [
                "bbWidth < 5.0",
                "price >= bbHigh * 0.98",
                "rsi >= 60"
            ],
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
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 2.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 60}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}  # Not applicable for multibagger
                }
            }
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
            "technical_conditions": [
                "bbWidth < 5.0",
                "price >= bbMid * 0.98",
                "price < bbHigh * 0.98",
                "rsi >= 55"
            ],
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
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "short_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": 2.5}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 55}
                }
            },
            
            "long_term": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": 50}
                }
            },
            
            "multibagger": {
                "context_requirements": {
                    "technical": {
                        "volatilityQuality": {"min": None}
                    }
                },
                "opportunity": {
                    "confidence": {"min": None}  # Not applicable
                }
            }
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
            "technical_conditions": [],
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
                    "condition": "adx < 10 and trendStrength < 2.0",
                    "amount": 12,
                    "reason": "Critically weak directional movement - GENERIC setup needs minimum trend"
                },
                "extreme_technical_weakness": {
                    "condition": "technicalScore < 2.5",
                    "amount": 15,
                    "reason": "Extreme technical weakness - even GENERIC needs basic technical merit"
                },
                "extreme_overbought_no_setup": {
                    "condition": "rsi > 85",
                    "amount": 12,
                    "reason": "Extreme overbought without specific setup pattern - high reversal risk"
                },

                # Fundamental extremes (filter complete garbage)
                "fundamentally_broken": {
                    "condition": "fundamentalScore < 3.0",
                    "amount": 15,
                    "reason": "Fundamentally broken - GENERIC fallback doesn't excuse garbage quality"
                }
            },
            "bonuses": {
                # Reward surprising strength in GENERIC (should have matched specific setup)
                "unexpected_strong_momentum": {
                    "condition": "momentumStrength >= 7.5 and trendStrength >= 6.5",
                    "amount": 8,
                    "reason": "Strong momentum despite no specific setup - edge case quality"
                },
                "unexpected_quality": {
                    "condition": "fundamentalScore >= 7.5 and technicalScore >= 6.5",
                    "amount": 10,
                    "reason": "High quality profile despite no specific setup - investigate why no pattern match"
                }
            }
        },

        "min_pattern_quality": 0.0,
        "min_setup_score": 0.0,
        "setup_type": "fallback",
        "horizon_overrides": {
            "intraday": {"context_requirements": {"technical": {"adx": {"min": 0},"trendStrength": {"min": 0}}},"opportunity": {"confidence": {"min": 50}}},
            "short_term": {"context_requirements": {"technical": {"adx": {"min": 0},    "trendStrength": {"min": 0}}},"opportunity": {"confidence": {"min": 50}}},
            "long_term": {"context_requirements": {"technical": {"adx": {"min": 0},    "trendStrength": {"min": 0}}},"opportunity": {"confidence": {"min": 50}}},
            "multibagger": {"context_requirements": {"technical": {"adx": {"min": 0},    "trendStrength": {"min": 0}}},"opportunity": {"confidence": {"min": 50}} }
        },
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
                "conditions": [
                    "rsi >= 50",
                    "macdhistogram >= 0",
                    "rvol >= 1.5",
                    "squeeze_duration >= 5"  # ✅ RESTORED - prevents premature entry
                ]
            },
            "short_term": {
                "order_type": "limit",
                "trigger": "close_above_band",
                "conditions": [
                    "rsi >= 50",
                    "macdhistogram >= 0",
                    "rvol >= 1.2",
                    "squeeze_duration >= 3"  # ✅ RESTORED - lower for short_term
                ]
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "close_above_band",
                "conditions": [
                    "rsi >= 45",
                    "macdhistogram >= -0.2",
                    "rvol >= 1.0",
                    "squeeze_duration >= 4"  # ✅ RESTORED
                ]
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "metadata_keys": {
                    "analytics": ["squeeze_duration", "squeeze_strength"]
                },
                "intraday": {
                    "conditions": [
                        "price < bbLow",
                        "bbWidth > 10.0"  # ⬆️ Relaxed from 8.0
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "short_term": {
                    "conditions": [
                        "price < bbLow * 0.99",
                        "bbWidth > 12.0"  # ⬆️ Relaxed from 10.0
                    ],
                    "duration_candles": 1,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "price < bbLow",
                        "bbWidth > 15.0"  # ⬆️ Relaxed from 12.0
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
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
                "conditions": [
                    "price >= rim_level * 0.995",
                    "rvol >= 1.2"
                ]
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "rim_level",
                "conditions": [
                    "price >= rim_level * 0.99",
                    "rvol >= 1.1"
                ]
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["cup_depth_pct", "handle_depth_pct"]
            },
            "breakdown_threshold": {
                # ❌ REMOVED intraday (pattern not suitable for this horizon)
                "short_term": {
                    "conditions": [
                        "price < handle_low * 0.97",  # ✅ Define handleLow in indicators
                        "rvol < 0.8"  # ✅ Volume drying up
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "price < handle_low * 0.95",
                        "rvol < 0.7"
                    ],
                    "duration_candles": 3,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
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
                "conditions": [
                    "price >= box_high * 1.002",  # ✅ Clearance above box
                    "rvol >= 1.5",                # ✅ Volume surge (simplified from volume_surge_required)
                    "box_age_candles <= 50"       # ✅ RESTORED - prevents stale boxes
                ]
            },
            "short_term": {
                "order_type": "stop_market",
                "trigger": "box_high",
                "conditions": [
                    "price >= box_high * 1.005",
                    "rvol >= 1.3",
                    "box_age_candles <= 30"       # ✅ RESTORED - tighter for short_term
                ]
            }
        },
        "invalidation": {
            "metadata_keys": {
                # Used for Velocity Analytics only
                "analytics": ["box_height_pct", "box_age_candles"] 
            },
            "breakdown_threshold": {
                "intraday": {
                    "conditions": ["price < box_low * 0.998"],
                    "duration_candles": 1
                },
                "short_term": {
                    "conditions": ["price < box_low * 0.995"],
                    "duration_candles": 1
                },
                "long_term": {
                    "conditions": ["price < box_low * 0.99"],
                    "duration_candles": 2
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
        "best_horizons": ["short_term", "long_term", "multibagger"],
        "physics": {
            "target_ratio": 1.0,
            "duration_multiplier": 1.8,
            "max_stop_pct": 7.0,
            "min_contraction_pct": 1.5,
            "horizons_supported": ["short_term", "long_term", "multibagger"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit",       # Minervini buys on specific pivots
                "trigger": "pivot_point",
                "conditions": [
                    "volatilityQuality <= 1.5",
                    "price >= pivot_point * 1.01",
                    "position52w >= 80"
                ]
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "pivot_point",
                "conditions": [
                    "volatilityQuality <= 2.0",
                    "price >= pivot_point * 1.005",
                    "position52w >= 70"
                ]
            },
            "multibagger": {
                "order_type": "limit",
                "trigger": "pivot_point",
                "conditions": [
                    "volatilityQuality <= 2.5",
                    "price >= pivot_point * 1.005",
                    "position52w >= 65"
                ]
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["contraction_pct", "volatility_quality"]
            },
            "breakdown_threshold": {
                # ❌ REMOVED intraday (accumulation pattern, not for day trading)
                "short_term": {
                    "conditions": [
                        "price < maFast * 0.95",  # ✅ Use 10-day MA instead of pivot
                        "volatilityQuality < 4.0"  # ✅ VCP breaking down
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "price < maFast * 0.92",
                        "position52w < 70"  # ✅ Fell from highs
                    ],
                    "duration_candles": 3,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "multibagger": {
                    "conditions": [
                        "price < maFast * 0.90",
                        "position52w < 65"
                    ],
                    "duration_candles": 3,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP",
                "multibagger": "TIGHTEN_STOP"
            },
            "stage_reversion": {
                "conditions": [
                    "price < maFast",
                    "rvol < 0.8"  # ✅ Volume declining
                ],
                "duration_candles": 5,
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
                "conditions": [
                    "pole_length >= 8",           # ✅ RESTORED - validates pole strength
                    "flag_tightness <= 0.03",     # ✅ RESTORED - ensures tight consolidation
                    "price >= flag_high * 1.005", # ✅ Breakout clearance
                    "trendStrength >= 5.5"
                ]
            },
            "short_term": {
                "order_type": "stop_market",
                "trigger": "flag_high",
                "conditions": [
                    "pole_length >= 5",           # ✅ RESTORED - lower for short_term
                    "flag_tightness <= 0.05",     # ✅ RESTORED - slightly looser
                    "price >= flag_high * 1.01",
                    "trendStrength >= 5.0"
                ]
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["pole_strength", "flag_tightness"]
            },
            "breakdown_threshold": {
                "intraday": {
                    "conditions": [
                        "price < flag_low * 0.998",
                        "trendStrength < 4.0"  # ✅ Trend weakening
                    ],
                    "duration_candles": 1,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "short_term": {
                    "conditions": [
                        "price < flag_low * 0.995",
                        "adx < 15"
                    ],
                    "duration_candles": 1,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "price < flag_low * 0.99",
                        "adx < 12"
                    ],
                    "duration_candles": 1,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_IMMEDIATELY",
                "long_term": "EXIT_ON_CLOSE"
            },
            "expiration": {
                "enabled": True,  # ✅ FIX: check_pattern_expiration() gates on this key
                "max_duration_candles": {
                    "intraday": 20,
                    "short_term": 15,  # ⬆️ Relaxed from 10
                    "long_term": 12  # ⬆️ Relaxed from 8
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
                "order_type": "market",      # Reversals often require instant entry
                "trigger": "candle_close",
                "conditions": [
                    "strike_candle_body >= 0.6",
                    "rvol >= 1.3"
                ]
            },
            "short_term": {
                "order_type": "limit",
                "trigger": "candle_close",
                "conditions": [
                    "strike_candle_body >= 0.7",
                    "rvol >= 1.2"
                ]
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "intraday": {
                    "conditions": [
                        "price < strike_low * 0.995",  # ✅ Use strike candle low
                        "rsi < 45"  # ✅ Reversal failing
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "short_term": {
                    "conditions": [
                        "price < strike_low * 0.99",
                        "rsi < 40"
                    ],
                    "duration_candles": 3,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "price < strike_low * 0.98",
                        "macdhistogram < 0"
                    ],
                    "duration_candles": 3,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                }
            },
            "action": {
                "intraday": "EXIT_IMMEDIATELY",
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "TIGHTEN_STOP"
            },
            "expiration": {
                "enabled": True,  # ✅ FIX: required by check_pattern_expiration()
                "max_duration_candles": {  # ✅ FIX: was "max_hold_candles" — key mismatch with enhancer
                    "intraday": 15,  # ⬆️ Relaxed from 10
                    "short_term": 12,  # ⬆️ Relaxed from 8
                    "long_term": 10  # ⬆️ Relaxed from 6
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
        "best_horizons": ["short_term", "long_term", "multibagger"],
        "physics": {
            **DEFAULT_PHYSICS,
            "horizons_supported": ["short_term", "long_term", "multibagger"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit",
                "trigger": "tenkan_kijun_cross",
                "conditions": [
                    "cloud_thickness >= 0.01",
                    "tenkan_kijun_spread >= 0.005",
                    "price > ichiSpanA"
                ]
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "kumo_breakout",
                "conditions": [
                    "cloud_thickness >= 0.02",
                    "tenkan_kijun_spread >= 0.01",
                    "price > ichiSpanA"
                ]
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                "intraday": {
                    "conditions": [
                        "price < ichiSpanA * 0.998",  # ✅ Cloud bottom (Senkou A)
                        "ichiTenkan < ichiKijun"  # ✅ Fast line crosses below
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "short_term": {
                    "conditions": [
                        "price < ichiSpanA * 0.995",
                        "chikouSpan < price"  # ✅ Lagging span below price
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "price < ichiSpanA",
                        "ichiTenkan < ichiKijun"
                    ],
                    "duration_candles": 3,
                    "_logic": "AND"  # ✅ Both must occur for long-term
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
        "best_horizons": ["short_term", "long_term", "multibagger"],
        "physics": {
            # FIX: was None - dict.get(key, default) returns None when key exists with None value.
            # Use 0 (falsy sentinel) so callers guard with: target_ratio or 1.0
            "target_ratio": 0,
            "duration_multiplier": 2.0,
            "max_stop_pct": None,
            "horizons_supported": ["short_term", "long_term", "multibagger"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "conditions": [
                    "maMid > maSlow * 1.002",
                    "rvol >= 1.1"
                ]
            },
            "long_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "conditions": [
                    "maMid > maSlow * 1.005",
                    "rvol >= 1.0"
                ]
            },
            "multibagger": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "conditions": [
                    "maMid > maSlow * 1.01",
                    "rvol >= 0.8"
                ]
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                # "intraday (long-term signal, not for day trading)
                "short_term": {
                    "conditions": [
                        "maMid < maSlow",
                        "price < maMid * 0.95"  # ✅ Significant breakdown
                    ],
                    "duration_candles": 5,  # ⬆️ Relaxed from 3
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "maMid < maSlow",
                        "adx < 15"  # ✅ Trend dying
                    ],
                    "duration_candles": 10,  # ⬆️ Relaxed from 4
                    "_logic": "OR",
                    "_duration_applies_to": [0]
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
        "best_horizons": ["long_term", "multibagger"],
        "physics": {
            # FIX: was None - same issue as goldenCross. Use 0 as falsy sentinel.
            "target_ratio": 0,
            "duration_multiplier": 2.0,
            "max_stop_pct": None,
            "horizons_supported": ["short_term", "long_term", "multibagger"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "conditions": [
                    "maMid < maSlow * 0.998",
                    "rvol >= 1.1"
                ]
            },
            "long_term": {
                "order_type": "limit", 
                "trigger": "ma_cross",
                "conditions": [
                    "maMid < maSlow * 0.995",
                    "rvol >= 1.0"
                ]
            }
        },
        "invalidation": {
            "breakdown_threshold": {
                # ❌ REMOVED intraday
                "short_term": {
                    "conditions": [
                        "maMid > maSlow",
                        "rsi > 60",
                    ],
                    "duration_candles": 5,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                },
                "long_term": {
                    "conditions": [
                        "maMid > maSlow",
                        "price > maMid * 1.05"
                    ],
                    "duration_candles": 10,
                    "_logic": "OR",
                    "_duration_applies_to": [0]
                }
            },
            "action": {
                "short_term": "EXIT_ON_CLOSE",
                "long_term": "EXIT_ON_CLOSE"
            }
        }
    },

    "doubleTopBottom": {
        "type": "reversal",
        "timeframe_agnostic": False,
        "direction": "variable",
        "typical_duration": {"min": 15, "max": 50},
        "failure_rate": 0.25,
        "best_horizons": ["short_term", "long_term"],
        "physics": {
            **DEFAULT_PHYSICS,
            "horizons_supported": ["short_term", "long_term"]
        },
        "entry_rules": {
            "short_term": {
                "order_type": "stop_market", # Buy on neckline break
                "trigger": "neckline",
                "conditions": [
                    "price >= neckline * 1.01",
                    "rvol >= 1.2",
                    "peak_similarity <= 0.02"
                ]
            },
            "long_term": {
                "order_type": "limit",
                "trigger": "neckline_confirmation",
                "conditions": [
                    "price >= neckline * 1.005",
                    "peak_similarity <= 0.03"
                ]
            }
        },
        "invalidation": {
            "metadata_keys": {
                "analytics": ["pattern_height_pct", "peak_similarity"],
                "type": "type"  # ✅ Add this to expose bullish/bearish type
            },
            "breakdown_threshold": {
                "intraday": {
                    "conditions": [
                        "type == 'bullish' and price < neckline * 0.998", 
                        "type == 'bearish' and price > neckline * 1.002",
                        "rvol < 0.8" # Common condition: volume dying
                    ],
                    "duration_candles": 1,
                    "_logic": "OR",
                    "_duration_applies_to": [0, 1]  # Both price conditions need duration
                },
                "short_term": {
                    "conditions": [
                        "type == 'bullish' and price < neckline * 0.995",
                        "type == 'bearish' and price > neckline * 1.005",
                        "rsi < 40"  # For bearish, you might want rsi > 60 instead
                    ],
                    "duration_candles": 2,
                    "_logic": "OR",
                    "_duration_applies_to": [0, 1]
                },
                "long_term": {
                    "conditions": [
                        "type == 'bullish' and price < neckline * 0.99",
                        "type == 'bearish' and price > neckline * 1.01",
                        "macdhistogram < 0"  # For bearish, use macdhistogram > 0
                    ],
                    "duration_candles": 3,
                    "_logic": "OR",
                    "_duration_applies_to": [0, 1]
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
                    "intraday": 20,  # ⬆️ Relaxed from 15
                    "short_term": 15,  # ⬆️ Relaxed from 10
                    "long_term": 20  # ⬆️ Relaxed from 12
                },
                "action": "MONITOR"  # ⬇️ Relaxed from EXIT_ON_CLOSE
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
        "long_term": "bollingerSqueezeLongTerm",
        "multibagger": "bollingerSqueezeMultibagger"
    },
    "minerviniStage2": {
        "intraday": "minerviniStage2Intraday",
        "short_term": "minerviniStage2ShortTerm",
        "long_term": "minerviniStage2LongTerm",
        "multibagger": "minerviniStage2Multibagger"
    },
    "ichimokuSignals": {
        "intraday": "ichimokuSignalsIntraday",
        "short_term": "ichimokuSignalsShortTerm",
        "long_term": "ichimokuSignalsLongTerm",
        "multibagger": "ichimokuSignalsMultibagger"
    },
    "goldenCross": {
        "intraday": "goldenCrossIntraday",
        "short_term": "goldenCrossShortTerm",
        "long_term": "goldenCrossLongTerm",
        "multibagger": "goldenCrossMultibagger"
    },
    "doubleTopBottom": {
        "intraday": "doubleTopBottomIntraday",
        "short_term": "doubleTopBottomShortTerm",
        "long_term": "doubleTopBottomLongTerm",
        "multibagger": "doubleTopBottomMultibagger"
    },
    "cupHandle": {
        "intraday": "cupHandleIntraday",
        "short_term": "cupHandleShortTerm",
        "long_term": "cupHandleLongTerm",
        "multibagger": "cupHandleMultibagger"
    },
    "flagPennant": {
        "intraday": "flagPennantIntraday",
        "short_term": "flagPennantShortTerm",
        "long_term": "flagPennantLongTerm",
        "multibagger": "flagPennantMultibagger"
    },
    "darvasBox": {
        "intraday": "darvasBoxIntraday",
        "short_term": "darvasBoxShortTerm",
        "long_term": "darvasBoxLongTerm",
        "multibagger": "darvasBoxMultibagger"
    },
    "threeLineStrike": {
        "intraday": "threeLineStrikeIntraday",
        "short_term": "threeLineStrikeShortTerm",
        "long_term": "threeLineStrikeLongTerm",
        "multibagger": "threeLineStrikeMultibagger"
    }
}


__all__ = [
    SETUP_PATTERN_MATRIX,
    PATTERN_METADATA,
    DEFAULT_PHYSICS,
    PATTERN_INDICATOR_MAPPINGS,
    PATTERN_SCORING_THRESHOLDS
]
