# config/fundamental_score_config.py
"""
Fundamental Score Aggregation Config
Version: 1.0 - Score-First Design

PHILOSOPHY:
- Fundamentals ALREADY have scores (0-10) from fundamentals.py
- Direct score extraction with optional overrides
- Config defines WEIGHTS + CATEGORIES only
"""

from typing import Any, Dict, Tuple
import logging

logger = logging.getLogger(__name__)
SCORE_FIELD_UNRELIABLE = {"ocfVsProfit", "promoterpledge", "beta"}

# ==============================================================================
# METRIC REGISTRY (Metadata for fundamental metrics)
# ==============================================================================

METRIC_REGISTRY = {
    # ===========================
    # VALUATION METRICS
    # ===========================
    "peRatio": {"type": "numeric","category": "valuation","description": "Price-to-Earnings ratio"},
    "pbRatio": {"type": "numeric","category": "valuation","description": "Price-to-Book ratio"},
    "pegRatio": {"type": "numeric","category": "valuation","description": "PEG ratio (PE / Growth)"},
    "psRatio": {"type": "numeric","category": "valuation","description": "Price-to-Sales ratio"},
    "peVsSector": {"type": "numeric","category": "valuation","description": "PE vs sector average"},
    
    # ===========================
    # PROFITABILITY METRICS
    # ===========================
    "roe": {"type": "numeric","category": "profitability","description": "Return on Equity"},
    "roce": {"type": "numeric","category": "profitability","description": "Return on Capital Employed"},
    "roic": {"type": "numeric","category": "profitability","description": "Return on Invested Capital"},
    "netProfitMargin": {"type": "numeric","category": "profitability","description": "Net profit margin"},
    "operatingMargin": {"type": "numeric","category": "profitability","description": "Operating margin"},
    "ebitdaMargin": {"type": "numeric","category": "profitability","description": "EBITDA margin"},
    
    # ===========================
    # GROWTH METRICS
    # ===========================
    "profitGrowth3y": {"type": "numeric","category": "growth","description": "3-year profit CAGR"},
    "epsGrowth5y": {"type": "numeric","category": "growth","description": "5-year EPS growth"},
    "revenueGrowth5y": {"type": "numeric","category": "growth","description": "5-year revenue CAGR"},
    "fcfGrowth3y": {"type": "numeric","category": "growth","description": "3-year FCF growth"},
    "quarterlyGrowth": {"type": "numeric","category": "growth","description": "Quarterly growth (QoQ)"},
    "marketCapCagr": {"type": "numeric","category": "growth","description": "Market cap CAGR"},
    
    # ===========================
    # FINANCIAL HEALTH METRICS
    # ===========================
    "deRatio": {"type": "numeric","category": "financial_health","description": "Debt-to-Equity ratio"},
    "interestCoverage": {"type": "numeric","category": "financial_health","description": "Interest coverage ratio"},
    "currentRatio": {"type": "numeric","category": "financial_health","description": "Current ratio (liquidity)"},
    "fcfYield": {"type": "numeric","category": "financial_health","description": "Free cash flow yield"},
    "fcfMargin": {"type": "numeric","category": "financial_health","description": "Free cash flow margin"},
    "ocfVsProfit": {"type": "numeric","category": "financial_health","description": "OCF vs Net Income ratio"},
    "beta": {"type": "numeric","category": "financial_health","description": "Beta (volatility vs market)"},
    
    # ===========================
    # QUALITY METRICS
    # ===========================
    "piotroskiF": {"type": "numeric","category": "quality","description": "Piotroski F-Score (0-9)"},
    "earningsStability": {"type": "numeric","category": "quality","description": "Earnings coefficient of variation"},
    "assetTurnover": {"type": "numeric","category": "quality","description": "Asset turnover efficiency"},
    "RDIntensity": {"type": "numeric","category": "quality","description": "R&D spend as % of revenue"},
    "roe3yAvg": {"type": "numeric","category": "quality","description": "3-year average Return on Equity"},
    
    # ===========================
    # OWNERSHIP METRICS
    # ===========================
    "promoterHolding": {"type": "numeric","category": "ownership","description": "Promoter shareholding %"},
    "institutionalOwnership": {"type": "numeric","category": "ownership","description": "Institutional ownership %"},
    "promoterpledge": {"type": "numeric","category": "ownership","description": "Promoter pledge %"},
    
    # ===========================
    # DIVIDEND METRICS
    # ===========================
    "dividendyield": {"type": "numeric","category": "dividend","description": "Dividend yield %"},
    "dividendPayout": {"type": "numeric","category": "dividend","description": "Dividend payout ratio"},
    "yieldVsAvg": {"type": "numeric","category": "dividend","description": "Yield vs 5-year average"},
    
    # ===========================
    # MARKET METRICS
    # ===========================
    "marketCap": {"type": "numeric","category": "market","description": "Market capitalization"},
    "position52w": {"type": "numeric","category": "market","description": "Distance from 52-week high"},
    "shortInterest": {"type": "numeric","category": "market","description": "Short interest metrics"},
    "analystRating": {"type": "text","category": "market","description": "Analyst recommendation"},
    "days_to_earnings": {"type": "numeric","category": "market","description": "Days until next earnings"
    }
}

# ==============================================================================
# HORIZON METRIC PARTICIPATION (What metrics apply per horizon)
# ==============================================================================

HORIZON_METRIC_INCLUSION = {
    "intraday": {
        # Intraday doesn't use fundamentals heavily
        "market": ["marketCap", "position52w"],
        "exclude": ["profitGrowth3y", "epsGrowth5y", "revenueGrowth5y", "fcfGrowth3y","marketCapCagr", "piotroskiF", "earningsStability", "RDIntensity","dividendyield", "dividendPayout", "yieldVsAvg"]
    },
    
    "short_term": {
        "valuation": ["peRatio", "pbRatio", "pegRatio", "psRatio"],
        "profitability": ["roe", "roce", "netProfitMargin", "operatingMargin"],
        "growth": ["quarterlyGrowth", "epsGrowth5y"],
        "financial_health": ["deRatio", "currentRatio", "fcfYield", "beta"],
        "quality": ["piotroskiF"],
        "ownership": ["promoterHolding", "institutionalOwnership"],
        "market": ["marketCap", "position52w"],
        
        "exclude": [
"marketCapCagr", 
"RDIntensity",   
"dividendyield", "dividendPayout", "yieldVsAvg" 
]
    },
    
    "long_term": {
        "valuation": ["peRatio", "pbRatio", "pegRatio", "peVsSector"],
        "profitability": ["roe", "roce", "roic", "netProfitMargin", "operatingMargin", "ebitdaMargin"],
        "growth": ["profitGrowth3y", "epsGrowth5y", "revenueGrowth5y", "fcfGrowth3y"],
        "financial_health": ["deRatio", "interestCoverage", "currentRatio", "fcfYield", "fcfMargin", "ocfVsProfit", "beta"],
        "quality": ["piotroskiF", "earningsStability", "assetTurnover"],
        "ownership": ["promoterHolding", "institutionalOwnership", "promoterpledge"],
        "dividend": ["dividendyield", "dividendPayout"],
        "market": ["marketCap"],
        
        "exclude": [
            "quarterlyGrowth",  # Too short-term
            "position52w",      # Technical concern
            "shortInterest",    # Market timing
            "days_to_earnings"  # Event timing
        ]
    },
    
    "multibagger": {
        "valuation": ["peRatio", "pegRatio", "peVsSector"],  # Growth at reasonable price
        "profitability": ["roe", "roce", "roic", "netProfitMargin"],  # Sustained profitability
        "growth": ["profitGrowth3y", "epsGrowth5y", "revenueGrowth5y", "fcfGrowth3y", "marketCapCagr"],  # ALL growth metrics
        "financial_health": ["deRatio", "interestCoverage", "fcfYield", "ocfVsProfit"],  # Quality balance sheet
        "quality": ["piotroskiF", "earningsStability", "RDIntensity"],  # High-quality business
        "ownership": ["promoterHolding", "institutionalOwnership", "promoterpledge"],  # Skin in the game
        "market": ["marketCap"],  # Size matters for liquidity
        
        "exclude": [
            "pbRatio",          # Less relevant for growth stocks
            "psRatio",          # Can be high for growth
            "quarterlyGrowth",  # Noise over 5 years
            "currentRatio",     # Short-term metric
            "fcfMargin",        # Can be negative during growth phase
            "ebitdaMargin",     # Focus on net profitability
            "assetTurnover",    # Not critical for multibaggers
            "dividendyield", "dividendPayout", "yieldVsAvg",  # Growth stocks rarely pay dividends
            "beta",             # Volatility expected
            "position52w",      # Long-term focus
            "shortInterest",    # Market timing
            "analystRating",    # Often wrong for multibaggers
            "days_to_earnings"  # Irrelevant
        ]
    }
}

# ==============================================================================
# HORIZON CATEGORY WEIGHTS (How to aggregate categories)
# ==============================================================================

HORIZON_FUNDAMENTAL_WEIGHTS = {
    "intraday": {
        "market": 1.0  # Only market size matters for liquidity
    },
    
    "short_term": {"valuation": 0.25,"profitability": 0.20,"growth": 0.20,"financial_health": 0.15,"quality": 0.10,"ownership": 0.05,"market": 0.05
    },
    
    "long_term": {"valuation": 0.15,"profitability": 0.25,"growth": 0.20,"financial_health": 0.20,"quality": 0.10,"ownership": 0.05,"dividend": 0.03,"market": 0.02
    },
    
    "multibagger": {
        "growth": 0.35,           # Growth is KING
        "profitability": 0.25,    # Must be profitable
        "quality": 0.15,          # High-quality business
        "financial_health": 0.10, # Sustainable growth
        "ownership": 0.08,        # Promoter conviction
        "valuation": 0.05,        # Can pay premium for growth
        "market": 0.02            # Size for exit liquidity
    }
}

# ==============================================================================
# METRIC-LEVEL WEIGHTS (Within each category, per horizon)
# ==============================================================================

METRIC_WEIGHTS = {
    # ==========================================================================
    # INTRADAY: Only market metrics matter
    # ==========================================================================
    "intraday": {
        "marketCap": 0.70,      # Liquidity proxy
        "position52w": 0.30     # Price level awareness
    },
    
    # ==========================================================================
    # SHORT-TERM: Balance across all aspects
    # ==========================================================================
    "short_term": {
        # VALUATION (25%)
        "peRatio": 0.35,
        "pbRatio": 0.25,
        "pegRatio": 0.30,
        "psRatio": 0.10,
        
        # PROFITABILITY (20%)
        "roe": 0.35,
        "roce": 0.30,
        "netProfitMargin": 0.20,
        "operatingMargin": 0.15,
        
        # GROWTH (20%)
        "quarterlyGrowth": 0.60,  # Recent momentum
        "epsGrowth5y": 0.40,
        
        # FINANCIAL HEALTH (15%)
        "deRatio": 0.35,
        "currentRatio": 0.25,
        "fcfYield": 0.25,
        "beta": 0.15,
        
        # QUALITY (10%)
        "piotroskiF": 1.0,
        
        # OWNERSHIP (5%)
        "promoterHolding": 0.60,
        "institutionalOwnership": 0.40,
        
        # MARKET (5%)
        "marketCap": 0.70,
        "position52w": 0.30
    },
    
    # ==========================================================================
    # LONG-TERM: Quality and sustainability focus
    # ==========================================================================
    "long_term": {
        # VALUATION (15%)
        "peRatio": 0.30,
        "pbRatio": 0.25,
        "pegRatio": 0.30,
        "peVsSector": 0.15,
        
        # PROFITABILITY (25%)
        "roe": 0.30,
        "roce": 0.25,
        "roic": 0.25,
        "netProfitMargin": 0.10,
        "operatingMargin": 0.05,
        "ebitdaMargin": 0.05,
        
        # GROWTH (20%)
        "profitGrowth3y": 0.25,
        "epsGrowth5y": 0.30,
        "revenueGrowth5y": 0.25,
        "fcfGrowth3y": 0.20,
        
        # FINANCIAL HEALTH (20%)
        "deRatio": 0.20,
        "interestCoverage": 0.15,
        "currentRatio": 0.10,
        "fcfYield": 0.15,
        "fcfMargin": 0.10,
        "ocfVsProfit": 0.10,
        "beta": 0.20,
        
        # QUALITY (10%)
        "piotroskiF": 0.50,
        "earningsStability": 0.30,
        "assetTurnover": 0.20,
        
        # OWNERSHIP (5%)
        "promoterHolding": 0.40,
        "institutionalOwnership": 0.35,
        "promoterpledge": 0.25,
        
        # DIVIDEND (3%)
        "dividendyield": 0.60,
        "dividendPayout": 0.40,
        
        # MARKET (2%)
        "marketCap": 1.0
    },
    
    # ==========================================================================
    # MULTIBAGGER: Growth above all, with quality gates
    # ==========================================================================
    "multibagger": {
        # GROWTH (35%) - SUPREME PRIORITY
        "profitGrowth3y": 0.20,
        "epsGrowth5y": 0.25,
        "revenueGrowth5y": 0.20,
        "fcfGrowth3y": 0.15,
        "marketCapCagr": 0.20,
        
        # PROFITABILITY (25%) - Must be profitable
        "roe": 0.35,
        "roce": 0.30,
        "roic": 0.25,
        "netProfitMargin": 0.10,
        
        # QUALITY (15%) - High-quality business
        "piotroskiF": 0.50,
        "earningsStability": 0.30,
        "RDIntensity": 0.20,
        
        # FINANCIAL HEALTH (10%) - Sustainable growth
        "deRatio": 0.40,
        "interestCoverage": 0.30,
        "fcfYield": 0.20,
        "ocfVsProfit": 0.10,
        
        # OWNERSHIP (8%) - Promoter conviction
        "promoterHolding": 0.50,
        "institutionalOwnership": 0.30,
        "promoterpledge": 0.20,
        
        # VALUATION (5%) - Can pay premium
        "peRatio": 0.40,
        "pegRatio": 0.40,
        "peVsSector": 0.20,
        
        # MARKET (2%)
        "marketCap": 1.0
    }
}

# ==============================================================================
# PENALTIES (Fundamental red flags)
# ==============================================================================

FUNDAMENTAL_PENALTIES = {
    "short_term": [
        {
            "metric": "deRatio",
            "operator": ">",
            "threshold": 2.0,
            "penalty": 0.30,
            "reason": "High debt burden"
        },
        {
            "metric": "promoterpledge",
            "operator": ">",
            "threshold": 20,
            "penalty": 0.25,
            "reason": "High promoter pledge"
        },
        {
            "metric": "currentRatio",
            "operator": "<",
            "threshold": 0.8,
            "penalty": 0.20,
            "reason": "Liquidity crisis risk"
        }
    ],
    
    "long_term": [
        {
            "metric": "deRatio",
            "operator": ">",
            "threshold": 1.5,
            "penalty": 0.25,
            "reason": "Elevated debt levels"
        },
        {
            "metric": "interestCoverage",
            "operator": "<",
            "threshold": 2.0,
            "penalty": 0.30,
            "reason": "Poor interest coverage"
        },
        {
            "metric": "promoterpledge",
            "operator": ">",
            "threshold": 15,
            "penalty": 0.25,
            "reason": "Promoter pledge risk"
        },
        {
            "metric": "roe",
            "operator": "<",
            "threshold": 10,
            "penalty": 0.20,
            "reason": "Below-average ROE"
        }
    ],
    
    "multibagger": [
        {
            "metric": "promoterHolding",
            "operator": "<",
            "threshold": 25,
            "penalty": 0.30,
            "reason": "Low promoter conviction"
        },
        {
            "metric": "epsGrowth5y",
            "operator": "<",
            "threshold": 10,
            "penalty": 0.35,
            "reason": "Insufficient growth track record"
        },
        {
            "metric": "deRatio",
            "operator": ">",
            "threshold": 1.0,
            "penalty": 0.20,
            "reason": "Debt constrains growth"
        },
        {
            "metric": "piotroskiF",
            "operator": "<",
            "threshold": 5,
            "penalty": 0.25,
            "reason": "Poor fundamental quality"
        }
    ]
}

# ==============================================================================
# BONUSES (Fundamental strengths)
# ==============================================================================

FUNDAMENTAL_BONUSES = [
    {
        "condition": "roe >= 25 AND roce >= 20",
        "bonus": 0.20,
        "reason": "Exceptional capital efficiency"
    },
    {
        "condition": "epsGrowth5y >= 20 AND profitGrowth3y >= 20",
        "bonus": 0.25,
        "reason": "Strong sustained growth"
    },
    {
        "condition": "piotroskiF >= 8",
        "bonus": 0.15,
        "reason": "High Piotroski F-Score"
    },
    {
        "condition": "deRatio <= 0.3 AND interestCoverage >= 10",
        "bonus": 0.15,
        "reason": "Fortress balance sheet"
    },
    {
        "condition": "promoterHolding >= 60 AND promoterpledge == 0",
        "bonus": 0.20,
        "reason": "Strong promoter conviction (no pledge)"
    },
    {
        "condition": "fcfYield >= 8 AND ocfVsProfit >= 1.2",
        "bonus": 0.15,
        "reason": "Excellent cash generation"
    },
    {
        "condition": "marketCapCagr >= 30",
        "bonus": 0.20,
        "reason": "Multi-year wealth creator"
    }
]

# ==============================================================================
# SECTOR-SPECIFIC EXCLUSIONS (Optional - for better scoring accuracy)
# ==============================================================================

SECTOR_SPECIFIC_EXCLUSIONS = {
    "Industrials": ["RDIntensity"],          # Ports, shipping, logistics
    "Utilities": ["RDIntensity", "assetTurnover"],
    "Real Estate": ["RDIntensity", "assetTurnover"],
    "Financials": ["RDIntensity", "assetTurnover"],  # Banks, NBFCs
    "Energy": ["RDIntensity"],               # Oil & Gas
}
# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _evaluate_condition(condition: str, fundamentals: Dict) -> bool:
    """Evaluate a simple condition on fundamental metrics."""
    if " AND " not in condition:
        return _evaluate_single_condition(condition, fundamentals)
    
    parts = condition.split(" AND ")
    return all(_evaluate_single_condition(part.strip(), fundamentals) for part in parts)


def _evaluate_single_condition(condition: str, fundamentals: Dict) -> bool:
    """Evaluate a single condition."""
    for op in [">=", "<=", "==", "!=", ">", "<"]:
        if op in condition:
            metric, value = condition.split(op)
            metric = metric.strip()
            value = value.strip()
            
            metric_data = fundamentals.get(metric, {})
            
            # Try to get raw value first, then score
            if isinstance(metric_data, dict):
                actual = metric_data.get("raw")
                if actual is None:
                    actual = metric_data.get("score")
            else:
                actual = metric_data
            
            if actual is None:
                return False
            
            try:
                threshold = float(value)
                actual = float(actual)
                
                if op == ">=": return actual >= threshold
                elif op == "<=": return actual <= threshold
                elif op == ">": return actual > threshold
                elif op == "<": return actual < threshold
                elif op == "==": return actual == threshold
                elif op == "!=": return actual != threshold
            except (ValueError, TypeError):
                return False
    
    return False


def get_active_metrics_for_horizon(horizon: str, sector: str = None) -> dict:
    """Get active metrics for horizon with sector exclusions."""
    cfg = HORIZON_METRIC_INCLUSION.get(horizon, {})

    active = {}
    for category, metrics in cfg.items():
        if category == "exclude":
            continue
        active[category] = list(metrics)  # Make a copy

    # ✅ FIX: Initialize excluded set from config
    excluded = set(cfg.get("exclude", []))

    # ✅ FIX: Safely add sector-specific exclusions
    if sector:
        sector_exclusions = SECTOR_SPECIFIC_EXCLUSIONS.get(sector, [])
        if sector_exclusions:  # Only if not None/empty
            excluded.update(sector_exclusions)
            logger.debug(f"Sector {sector} exclusions: {sector_exclusions}")

    # Remove excluded metrics
    for category in list(active.keys()):  # Use list() to avoid dict size change
        active[category] = [m for m in active[category] if m not in excluded]

    return active


def compute_category_score(
    fundamentals: dict,
    horizon: str,
    category: str,
    active_metrics: list
) -> tuple:
    """
    Compute score for a fundamental category.
    
    Returns:
        (category_score_0_to_10, breakdown_dict)
    """
    weights = METRIC_WEIGHTS.get(horizon, {})
    total_weight = 0.0
    weighted_score = 0.0
    breakdown = {}
    
    for metric in active_metrics:
        metric_weight = weights.get(metric)
        if metric_weight is None:
            continue
        
        metric_data = fundamentals.get(metric)
        score = extract_normalized_score(metric_data, metric, horizon)
        
        score = max(0.0, min(10.0, score))  # Defensive clamp
        
        weighted_score += score * metric_weight
        total_weight += metric_weight
        
        breakdown[metric] = {
            "score": score,
            "weight": metric_weight,
            "contribution": score * metric_weight
        }
    
    if total_weight == 0:
        return 0.0, breakdown
    
    normalized = weighted_score / total_weight
    return round(normalized, 2), breakdown


def apply_fundamental_penalties(fundamentals: dict, horizon: str) -> tuple:
    """
    Apply penalty rules to fundamental score.
    
    Returns:
        (total_penalty, list_of_reasons)
    """
    penalties = FUNDAMENTAL_PENALTIES.get(horizon, [])
    total = 0.0
    reasons = []
    
    for rule in penalties:
        metric = rule["metric"]
        metric_data = fundamentals.get(metric)
        
        if isinstance(metric_data, dict):
            actual = metric_data.get("raw")
            if actual is None:
                actual = metric_data.get("score")
        else:
            actual = metric_data
        
        if actual is None:
            continue
        
        try:
            actual = float(actual)
            threshold = rule["threshold"]
            op = rule["operator"]
            
            triggered = (
                (op == "<" and actual < threshold) or
                (op == ">" and actual > threshold) or
                (op == "<=" and actual <= threshold) or
                (op == ">=" and actual >= threshold)
            )
            
            if triggered:
                total += rule["penalty"]
                reasons.append(rule["reason"])
        except (ValueError, TypeError):
            continue
    
    return round(total, 2), reasons


def _extract_metrics_from_condition(condition: str) -> list:
    """Extract metric names from a condition."""
    import re
    return re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:>=|<=|==|!=|>|<)', condition)

def apply_fundamental_bonuses(fundamentals: dict, horizon: str = "short_term") -> tuple:
    """
    Apply bonus rules to fundamental score.
    
    Returns:
        (total_bonus, list_of_reasons)
    """
    total = 0.0
    reasons = []
    excluded = set(HORIZON_METRIC_INCLUSION.get(horizon, {}).get("exclude", []))
    
    for rule in FUNDAMENTAL_BONUSES:
        rule_metrics = _extract_metrics_from_condition(rule["condition"])
        if any(m in excluded for m in rule_metrics):
            continue
            
        if _evaluate_condition(rule["condition"], fundamentals):
            total += rule["bonus"]
            reasons.append(rule["reason"])
    
    return round(total, 2), reasons


def compute_fundamental_score(fundamentals: dict, horizon: str) -> dict:
    """
    Final fundamental score computation.
    
    Args:
        fundamentals: Dict from fundamentals.py compute_fundamentals()
        horizon: Trading horizon (intraday/short_term/long_term/multibagger)
    
    Returns:
        Rich, explainable score structure
    """
    sector = fundamentals.get("sector", None)
    active = get_active_metrics_for_horizon(horizon, sector)
    category_weights = HORIZON_FUNDAMENTAL_WEIGHTS.get(horizon, {})

    # if active in SECTOR_SPECIFIC_EXCLUSIONS.get(fundamentals.get('sector')):
    #     active.__delattr__
    
    category_scores = {}
    category_breakdown = {}
    total_score = 0.0
    
    # Compute each category
    for category, metrics in active.items():
        if category not in category_weights:
            continue
        
        cat_score, breakdown = compute_category_score(
            fundamentals, horizon, category, metrics
        )
        
        weight = category_weights[category]
        total_score += cat_score * weight
        
        category_scores[category] = {
            "score": cat_score,
            "weight": weight,
            "weighted": round(cat_score * weight, 2)
        }
        category_breakdown[category] = breakdown
    
    # Apply penalties
    penalty, penalty_reasons = apply_fundamental_penalties(fundamentals, horizon)
    
    # Apply bonuses
    bonus, bonus_reasons = apply_fundamental_bonuses(fundamentals, horizon)
    
    # Calculate final score
    final_score = total_score - penalty + bonus
    final_score = round(max(0.0, min(10.0, final_score)), 2)

    return {
        "score": final_score,
        "horizon": horizon,
        "base_score": round(total_score, 2),
        "category_scores": category_scores,
        "penalties": {
            "total": round(penalty, 2),
            "reasons": penalty_reasons
        },
        "bonuses": {
            "total": round(bonus, 2),
            "reasons": bonus_reasons
        },
        "breakdown": category_breakdown
    }

#metaScore computing fundamental score based on config
def extract_normalized_score(
    metric_data: Any,
    metric_name: str,
    horizon: str = "short_term"
) -> float:
    if isinstance(metric_data, dict):
        # For metrics with known inverted directionality, always re-normalize from raw
        if metric_name in SCORE_FIELD_UNRELIABLE:
            raw = metric_data.get("raw") if metric_data.get("raw") is not None else metric_data.get("value")
            if raw is not None:
                try:
                    raw_float = float(str(raw).replace('%', '').replace('x', '').replace(',', '').strip())
                    return _normalize_fundamental_metric(metric_name, raw_float, horizon)
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse '{raw}' for {metric_name}")
            return 0.0

        # For all other metrics, trust the pre-computed score field
        score = metric_data.get("score")
        if score is not None:
            try:
                return max(0.0, min(10.0, float(str(score).replace('%', '').replace('x', '').replace(',', '').strip())))
            except (ValueError, TypeError):
                pass # Fall through to raw if score is invalid

        # Fallback: re-normalize from raw value
        raw = metric_data.get("raw") if metric_data.get("raw") is not None else metric_data.get("value")
        if raw is None:
            logger.warning(f"No score/raw for {metric_name}")
            return 0.0

    elif isinstance(metric_data, (int, float, str)):
        raw = metric_data
    else:
        logger.warning(f"Invalid type for {metric_name}: {type(metric_data)}")
        return 0.0

    # Robust float conversion for raw
    try:
        raw_float = float(str(raw).replace('%', '').replace('x', '').replace(',', '').strip())
        return _normalize_fundamental_metric(metric_name, raw_float, horizon)
    except (ValueError, TypeError):
        logger.warning(f"Could not convert raw value '{raw}' to float for {metric_name}")
        return 0.0

def _normalize_fundamental_metric(
    metric_name: str,
    raw_value: float,
    horizon: str
) -> float:
    """
    Normalize raw fundamental value to 0-10 score.
    Uses same thresholds as compute_fundamentals() but extracted here for reuse.
    """
    # Get metric metadata
    registry_entry = METRIC_REGISTRY.get(metric_name)
    if not registry_entry:
        logger.warning(f"Unknown metric: {metric_name}")
        return 5.0  # Neutral score
    
    category = registry_entry["category"]
    
    # Apply category-specific normalization
    if category == "valuation":
        return _normalize_valuation(metric_name, raw_value)
    elif category == "profitability":
        return _normalize_profitability(metric_name, raw_value)
    elif category == "growth":
        return _normalize_growth(metric_name, raw_value)
    elif category in ["financial_health", "quality"]:
        return _normalize_health_quality(metric_name, raw_value)
    elif category == "ownership":
        return _normalize_ownership(metric_name, raw_value)
    else:
        # Generic linear scaling for unknown categories
        return min(10.0, max(0.0, raw_value / 10))


# Helper normalization functions (extract from fundamentals.py)
def _normalize_valuation(metric: str, val: float) -> float:
    """Lower is better for valuation metrics"""
    if metric == "peRatio":
        if val <= 10: return 10
        elif val <= 15: return 8
        elif val <= 25: return 6
        elif val <= 40: return 4
        else: return 2
    
    elif metric == "pbRatio":
        if val <= 1.0: return 10
        elif val <= 2.0: return 8
        elif val <= 3.0: return 6
        else: return 4
    
    elif metric == "pegRatio":
        if val <= 1.0: return 10
        elif val <= 1.5: return 8
        elif val <= 2.0: return 6
        else: return 4
    elif metric == "peVsSector":   # Lower multiple vs sector = cheaper = better
        if val <= 0.5:  return 10
        elif val <= 0.8: return 8
        elif val <= 1.0: return 6
        elif val <= 1.2: return 4
        else: return 2
    
    return 5.0


def _normalize_profitability(metric: str, val: float) -> float:
    """Higher is better"""
    if metric in ["roe", "roce", "roic"]:
        if val >= 25: return 10
        elif val >= 20: return 9
        elif val >= 15: return 7
        elif val >= 12: return 5
        else: return 3
    
    elif metric in ["netProfitMargin", "operatingMargin"]:
        if val >= 20: return 10
        elif val >= 15: return 8
        elif val >= 10: return 6
        else: return 4
    
    return 5.0


def _normalize_growth(metric: str, val: float) -> float:
    """Higher is better"""
    if val >= 30: return 10
    elif val >= 25: return 9
    elif val >= 20: return 8
    elif val >= 15: return 7
    elif val >= 10: return 6
    else: return max(0, val / 2)


def _normalize_health_quality(metric: str, val: float) -> float:
    """Mixed directionality"""
    if metric == "deRatio":  # Lower is better
        if val <= 0.3: return 10
        elif val <= 0.5: return 8
        elif val <= 1.0: return 5
        elif val <= 2.0: return 3
        else: return 1

    elif metric == "ocfVsProfit":     # ← ADD: Higher is better (OCF > profit = quality earnings)
        if val >= 1.2: return 10      # OCF significantly exceeds profit (high quality)
        elif val >= 1.0: return 8     # OCF roughly matches profit
        elif val >= 0.8: return 5     # Slight shortfall
        elif val >= 0.5: return 3     # Notable gap
        else: return 1                # OCF far below profit (earnings quality concern)
        
    
    elif metric == "currentRatio":  # Higher is better
        if val >= 2.0: return 10
        elif val >= 1.5: return 8
        elif val >= 1.0: return 6
        else: return 4

    elif metric == "beta":
        # Lower beta = more stable. Score inverts at extremes.
        if 0.5 <= val <= 0.9: return 10   # Low vol, less market risk
        elif 0.9 < val <= 1.2: return 7   # Market neutral
        elif 1.2 < val <= 1.5: return 4   # Elevated volatility
        else: return 2                    # High beta or near-zero (illiquid)
    
    elif metric == "piotroskiF":  # 0-9 scale
        return val * (10/9)  # Convert to 0-10
    
    return 5.0  # Neutral for marketCap, position52w, etc.


def _normalize_ownership(metric: str, val: float) -> float:
    """Context-dependent scoring"""
    if metric == "promoterHolding":
        if 50 <= val <= 70: return 10  # Sweet spot
        elif 40 <= val < 50 or 70 < val <= 80: return 8
        elif 30 <= val < 40 or 80 < val <= 90: return 6
        else: return 4
        
    elif metric == "institutionalOwnership":
        if val >= 30: return 10
        elif val >= 20: return 8
        elif val >= 10: return 6
        elif val > 0: return 4
        else: return 2
    
    elif metric == "promoterpledge":  # Lower is better
        if val == 0: return 10
        elif val <= 5: return 8
        elif val <= 10: return 6
        else: return 2
    
    return 5.0
# ==============================================================================
# EXPORT
# ==============================================================================

__all__ = [
    "compute_fundamental_score",
    "METRIC_REGISTRY",
    "HORIZON_METRIC_INCLUSION",
    "HORIZON_FUNDAMENTAL_WEIGHTS",
    "METRIC_WEIGHTS"
]