# config/multibagger/multibagger_fundamental_score_config.py
"""
Multibagger Fundamental Score Configuration
============================================
MB-specific metric inclusion, weights, penalties, and bonuses.

These constants are consumed ONLY by mb_compute_fundamental_score()
in multibagger_evaluator.py. The main pipeline's compute_fundamental_score()
is NEVER called for MB — MBQueryOptimizedExtractor.get_fundamental_score()
routes to the MB-specific function instead.

CATEGORY WEIGHTS (must sum to 1.0):
    growth:           0.35  — Primary driver of multibagger returns
    profitability:    0.25  — Business quality proxy
    quality:          0.15  — Financial integrity
    financial_health: 0.10  — Balance sheet durability
    ownership:        0.08  — Management alignment
    valuation:        0.05  — Entry price sanity check
    market:           0.02  — Size context
"""

# ============================================================================
# METRIC INCLUSION — which metrics are active per category for "multibagger"
# ============================================================================

MB_HORIZON_METRIC_INCLUSION = {
    "multibagger": {
        "growth": [
            "profitGrowth3y", "epsGrowth5y", "revenueGrowth5y",
            "fcfGrowth3y", "marketCapCagr",
        ],
        "profitability": [
            "roe", "roce", "roic", "netProfitMargin",
        ],
        "quality": [
            "piotroskiF", "earningsStability", "RDIntensity",
        ],
        "financial_health": [
            "deRatio", "interestCoverage", "fcfYield", "ocfVsProfit",
        ],
        "ownership": [
            "promoterHolding", "institutionalOwnership", "promoterpledge",
        ],
        "valuation": [
            "peRatio", "pegRatio", "peVsSector",
        ],
        "market": [
            "marketCap",
        ],
        # Metrics to exclude from scoring — these have no long-term signal value
        "exclude": [
            "pbRatio", "psRatio", "quarterlyGrowth", "currentRatio",
            "fcfMargin", "ebitdaMargin", "assetTurnover",
            "dividendyield", "dividendPayout", "yieldVsAvg",
            "beta", "position52w", "shortInterest",
            "analystRating", "days_to_earnings",
        ],
    }
}

# ============================================================================
# CATEGORY WEIGHTS (pillar level)
# ============================================================================

MB_HORIZON_FUNDAMENTAL_WEIGHTS = {
    "multibagger": {
        "growth":           0.35,
        "profitability":    0.25,
        "quality":          0.15,
        "financial_health": 0.10,
        "ownership":        0.08,
        "valuation":        0.05,
        "market":           0.02,
    }
}

# ============================================================================
# METRIC WEIGHTS (within each category — flat dict, no category nesting)
# ============================================================================

MB_METRIC_WEIGHTS = {
    "multibagger": {
        # growth (sum = 1.0)
        "profitGrowth3y":   0.20,
        "epsGrowth5y":      0.25,
        "revenueGrowth5y":  0.20,
        "fcfGrowth3y":      0.15,
        "marketCapCagr":    0.20,
        # profitability (sum = 1.0)
        "roe":              0.35,
        "roce":             0.30,
        "roic":             0.25,
        "netProfitMargin":  0.10,
        # quality (sum = 1.0)
        "piotroskiF":       0.50,
        "earningsStability": 0.30,
        "RDIntensity":      0.20,
        # financial_health (sum = 1.0)
        "deRatio":          0.40,
        "interestCoverage": 0.30,
        "fcfYield":         0.20,
        "ocfVsProfit":      0.10,
        # ownership (sum = 1.0)
        "promoterHolding":      0.50,
        "institutionalOwnership": 0.30,
        "promoterpledge":       0.20,
        # valuation (sum = 1.0)
        "peRatio":          0.40,
        "pegRatio":         0.40,
        "peVsSector":       0.20,
        # market (sum = 1.0)
        "marketCap":        1.00,
    }
}

# ============================================================================
# PENALTY RULES
# operator: "<" | ">" | "<=" | ">="
# penalty: fraction deducted from final_score (0.0–1.0 scale maps to 0–10)
# ============================================================================

MB_FUNDAMENTAL_PENALTIES = {
    "multibagger": [
        {
            "metric":    "promoterHolding",
            "operator":  "<",
            "threshold": 25.0,
            "penalty":   3.0,
            "reason":    "Low promoter conviction",
        },
        {
            "metric":    "epsGrowth5y",
            "operator":  "<",
            "threshold": 10.0,
            "penalty":   3.5,
            "reason":    "Insufficient 5-year EPS growth",
        },
        {
            "metric":    "deRatio",
            "operator":  ">",
            "threshold": 1.0,
            "penalty":   2.0,
            "reason":    "High debt constrains compounding",
        },
        {
            "metric":    "piotroskiF",
            "operator":  "<",
            "threshold": 5,
            "penalty":   2.5,
            "reason":    "Poor fundamental quality (Piotroski < 5)",
        },
        {
            "metric":    "roe",
            "operator":  "<",
            "threshold": 10.0,
            "penalty":   2.5,
            "reason":    "ROE too low for compounder thesis",
        },
    ]
}

# ============================================================================
# BONUS RULES (gate-based, uses evaluate_gates)
# bonus: fraction added to final_score (same scale as penalty)
# ============================================================================

MB_FUNDAMENTAL_BONUSES = [
    {
        "gates": {"roe": {"min": 25}, "roce": {"min": 20}},
        "bonus":  2.0,
        "reason": "Exceptional capital efficiency",
    },
    {
        "gates": {"epsGrowth5y": {"min": 20}, "profitGrowth3y": {"min": 20}},
        "bonus":  2.5,
        "reason": "Strong sustained growth",
    },
    {
        "gates": {"piotroskiF": {"min": 8}},
        "bonus":  1.5,
        "reason": "High Piotroski F-Score",
    },
    {
        "gates": {"deRatio": {"max": 0.3}, "interestCoverage": {"min": 10}},
        "bonus":  1.5,
        "reason": "Fortress balance sheet",
    },
    {
        "gates": {"promoterHolding": {"min": 60}},
        "bonus":  2.0,
        "reason": "Strong promoter conviction",
    },
    {
        "gates": {"fcfYield": {"min": 8}, "ocfVsProfit": {"min": 1.2}},
        "bonus":  1.5,
        "reason": "Excellent cash generation",
    },
    {
        "gates": {"marketCapCagr": {"min": 30}},
        "bonus":  2.0,
        "reason": "Multi-year wealth creator",
    },
    {
        "gates": {"roic": {"min": 20}, "roe": {"min": 25}},
        "bonus":  1.8,
        "reason": "Economic moat — high ROIC + ROE",
    },
]
