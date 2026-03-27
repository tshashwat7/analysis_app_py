# config/multibagger/multibagger_evaluator.py
"""
Multibagger Evaluator — Phase 2
=================================
Thread-safe, isolated Phase 2 scorer for multibagger candidates.

ARCHITECTURE — Why a custom extractor stack?
    ConfigExtractor bakes in HORIZON_PILLAR_WEIGHTS and HYBRID_PILLAR_COMPOSITION
    via a module-level import on line 20 of config_extractor.py:
        from config.master_config import HORIZON_PILLAR_WEIGHTS, ...
    These are set at Python import time and cannot be changed by passing a
    different master_config dict.  MBConfigExtractor fixes this by overwriting
    the sections dict entries AFTER super().__init__() completes.

    QueryOptimizedExtractor.get_hybrid_pillar_composition() has its own lazy
    import of HYBRID_PILLAR_COMPOSITION.  MBQueryOptimizedExtractor overrides
    this method to return MB_HYBRID_PILLAR_COMPOSITION instead.

    get_technical_score() and get_fundamental_score() in QOE call the main
    pipeline's compute_technical_score() / compute_fundamental_score() which
    reads from the main config files.  These are overridden here to call
    mb_compute_technical_score() / mb_compute_fundamental_score() that use
    MB-specific weights, penalties, and bonuses.

THREAD SAFETY:
    run_mb_resolver() creates a fresh MBConfigResolver per call.
    It calls resolver.build_evaluation_context_only() directly, bypassing
    build_evaluation_context() which internally calls get_resolver(horizon)
    and would use the cached main-pipeline resolver.
    No global state is patched. Safe to run in a daemon thread.

CALL FLOW:
    run_mb_resolver(symbol, fundamentals, indicators, patterns)
        → MBConfigResolver.__init__()
              → MBQueryOptimizedExtractor.__init__()
                    → MBConfigExtractor.__init__()  [overrides pillar sections]
        → resolver.build_evaluation_context_only()  [scores all 3 pillars]
        → calculate_structural_eligibility(..., extractor=mb_extractor)
        → compute_opportunity_score(..., extractor=mb_extractor)
        → returns structured result dict
"""

import logging
from typing import Any, Dict, Optional, Tuple

from config.config_extractor import ConfigExtractor, ConfigSection
from config.query_optimized_extractor import QueryOptimizedExtractor
from config.config_resolver import ConfigResolver
from config.config_helpers import (
    flatten_market_data_mixed,
    _extract_price_data,
    _extract_patterns,
)
from services.scoring_utils import (
    calculate_structural_eligibility,
    compute_opportunity_score_logic as calc_opp_logic_util,
    calculate_final_decision_score as calc_final_score_util,
    compute_opportunity_score_full,
)
from collections import OrderedDict

from config.multibagger.multibagger_master_config import (
    MB_MASTER_CONFIG,
    MB_HORIZON_PILLAR_WEIGHTS,
    MB_HYBRID_PILLAR_COMPOSITION,
)
from config.multibagger.multibagger_confidence_config import MB_CONFIDENCE_CONFIG
from config.multibagger.multibagger_config import MULTIBAGGER_CONFIG

logger = logging.getLogger(__name__)

_HORIZON = "multibagger"


# =============================================================================
# ISOLATED SCORING FUNCTIONS
# Called by MBQueryOptimizedExtractor to route around the main pipeline's
# hard-imported scoring functions.
# =============================================================================

def mb_compute_fundamental_score(fundamentals: dict, horizon: str) -> dict:
    """
    Thread-safe fundamental scorer using MB-specific weights.

    Replicates the structure of compute_fundamental_score() from
    fundamental_score_config.py but reads from MB constants.
    Returns the same dict shape so ConfigResolver's scoring pipeline
    consumes it identically.
    """
    from config.multibagger.multibagger_fundamental_score_config import (
        MB_HORIZON_METRIC_INCLUSION,
        MB_HORIZON_FUNDAMENTAL_WEIGHTS,
        MB_METRIC_WEIGHTS,
        MB_FUNDAMENTAL_PENALTIES,
        MB_FUNDAMENTAL_BONUSES,
    )
    from config.fundamental_score_config import extract_normalized_score
    from config.gate_evaluator import evaluate_gates

    active_raw   = MB_HORIZON_METRIC_INCLUSION.get(horizon, {}).copy()
    excluded     = set(active_raw.pop("exclude", []))
    cat_weights  = MB_HORIZON_FUNDAMENTAL_WEIGHTS.get(horizon, {})
    metric_wts   = MB_METRIC_WEIGHTS.get(horizon, {})

    category_scores    = {}
    category_breakdown = {}
    total_score        = 0.0

    for category, metrics in active_raw.items():
        if category not in cat_weights:
            continue

        active_metrics = [m for m in metrics if m not in excluded]
        cat_weight_total  = 0.0
        cat_weighted_score = 0.0
        breakdown = {}

        for metric in active_metrics:
            w = metric_wts.get(metric)
            if w is None:
                continue

            metric_data = fundamentals.get(metric)
            score = extract_normalized_score(metric_data, metric, horizon)
            score = max(0.0, min(10.0, score))

            cat_weighted_score += score * w
            cat_weight_total   += w
            breakdown[metric]   = {
                "score":        score,
                "weight":       w,
                "contribution": score * w,
            }

        cat_score  = round(cat_weighted_score / cat_weight_total, 2) if cat_weight_total > 0 else 0.0
        cat_w      = cat_weights[category]
        total_score += cat_score * cat_w

        category_scores[category]    = {
            "score":    cat_score,
            "weight":   cat_w,
            "weighted": round(cat_score * cat_w, 2),
        }
        category_breakdown[category] = breakdown

    # --- Penalties (operator-based, not gate-based) ----------------------
    penalty         = 0.0
    penalty_reasons = []
    for rule in MB_FUNDAMENTAL_PENALTIES.get(horizon, []):
        metric_data = fundamentals.get(rule["metric"])
        if isinstance(metric_data, dict):
            actual = metric_data.get("raw")
        else:
            actual = metric_data
            
        # ✅ P3 FIX: Skip if raw is None — never fall back to 'score' for penalties
        if actual is None:
            continue
        try:
            actual = float(actual)
            op, thresh = rule["operator"], rule["threshold"]
            triggered = (
                (op == "<"  and actual < thresh) or
                (op == ">"  and actual > thresh) or
                (op == "<=" and actual <= thresh) or
                (op == ">=" and actual >= thresh)
            )
            if triggered:
                penalty += rule["penalty"]
                penalty_reasons.append(rule["reason"])
        except (ValueError, TypeError):
            continue

    # --- Bonuses (gate-based) -------------------------------------------
    bonus         = 0.0
    bonus_reasons = []
    for rule in MB_FUNDAMENTAL_BONUSES:
        rule_metrics = [k for k in rule["gates"].keys() if not k.startswith("_")]
        if any(m in excluded for m in rule_metrics):
            continue
        passes, _ = evaluate_gates(rule["gates"], fundamentals)
        if passes:
            bonus += rule["bonus"]
            bonus_reasons.append(rule["reason"])

    final_score = round(max(0.0, min(10.0, total_score - penalty + bonus)), 2)

    return {
        "score":          final_score,
        "horizon":        horizon,
        "base_score":     round(total_score, 2),
        "category_scores": category_scores,
        "breakdown":      category_breakdown,
        "penalties":      {"total": round(penalty, 2), "reasons": penalty_reasons},
        "bonuses":        {"total": round(bonus, 2),   "reasons": bonus_reasons},
    }


def mb_compute_technical_score(indicators: dict, horizon: str) -> dict:
    """
    Thread-safe technical scorer using MB-specific weights.

    NOTE on relStrengthNifty dual-category:
        The metric appears in both "trend" and "momentum" category lists.
        mb_compute_technical_score() routes momentum lookups to the
        "_momentum" suffix key in MB_TECH_METRIC_WEIGHTS so weights
        are applied independently per category without any dict collision.

        # Dual-category routing: metrics in both trend+momentum use "_momentum" suffix
        # for their momentum weight key — see multibagger_technical_score_config.py header.
    """
    from config.multibagger.multibagger_technical_score_config import (
        MB_TECH_HORIZON_METRIC_INCLUSION,
        MB_HORIZON_TECHNICAL_WEIGHTS,
        MB_TECH_METRIC_WEIGHTS,
        MB_TECHNICAL_PENALTIES,
        MB_TECH_BONUSES,
        MB_LIQUIDITY_PENALTY_RULE,
    )
    # ✅ Fix 7.1-4: CAUTION: Private API dependency. 
    # extract_metric_score and _extract_raw_value are needed for MB tech scoring.
    from config.technical_score_config import extract_metric_score, _extract_raw_value
    from config.gate_evaluator import evaluate_gates

    active_raw   = MB_TECH_HORIZON_METRIC_INCLUSION.get(horizon, {}).copy()
    excluded     = set(active_raw.pop("exclude", []))
    cat_weights  = MB_HORIZON_TECHNICAL_WEIGHTS.get(horizon, {})
    metric_wts   = MB_TECH_METRIC_WEIGHTS.get(horizon, {})

    category_scores    = {}
    category_breakdown = {}
    total_score        = 0.0

    for category, metrics in active_raw.items():
        if category not in cat_weights:
            continue

        active_metrics = [m for m in metrics if m not in excluded]
        cat_weight_total   = 0.0
        cat_weighted_score = 0.0
        breakdown = {}

        for metric in active_metrics:
            # relStrengthNifty in momentum category uses the aliased weight key
            weight_key = f"{metric}_momentum" if (category == "momentum" and metric == "relStrengthNifty") else metric
            w = metric_wts.get(weight_key)
            if w is None:
                continue

            score = extract_metric_score(indicators.get(metric), metric, indicators)
            if score is None:
                continue
            score = max(0.0, min(10.0, score))

            cat_weighted_score += score * w
            cat_weight_total   += w
            breakdown[metric]   = {
                "score":        score,
                "weight":       w,
                "contribution": score * w,
            }

        cat_score  = round(cat_weighted_score / cat_weight_total, 2) if cat_weight_total > 0 else 0.0
        cat_w      = cat_weights.get(category, 0)
        total_score += cat_score * cat_w

        category_scores[category]    = {
            "score":    cat_score,
            "weight":   cat_w,
            "weighted": round(cat_score * cat_w, 2),
        }
        category_breakdown[category] = breakdown

    # --- Penalties -------------------------------------------------------
    penalty         = 0.0
    penalty_reasons = []
    for rule in MB_TECHNICAL_PENALTIES.get(horizon, []):
        metric_data = indicators.get(rule["metric"])
        if metric_data is None:
            continue
        is_pt = rule.get("is_passthrough", False)
        actual = extract_metric_score(metric_data, rule["metric"], indicators) if is_pt else _extract_raw_value(metric_data, fallback_to_score=False)
        if actual is None:
            continue
        op, thresh = rule["operator"], rule["threshold"]
        triggered = (
            (op == "<"  and actual < thresh) or
            (op == ">"  and actual > thresh) or
            (op == "<=" and actual <= thresh) or
            (op == ">=" and actual >= thresh)
        )
        if triggered:
            penalty += rule["penalty"]
            penalty_reasons.append(rule["reason"])

    # --- Liquidity penalty -----------------------------------------------
    liq_penalty = 0.0
    liq_reason  = None
    liq_rule    = MB_LIQUIDITY_PENALTY_RULE.get(horizon)
    if liq_rule:
        vol_data  = indicators.get("avg_volume_30Days", {})
        vol_value = vol_data.get("value") or vol_data.get("raw") if isinstance(vol_data, dict) else vol_data
        if vol_value is not None:
            try:
                if float(vol_value) < liq_rule["min_avg_volume"]:
                    liq_penalty = liq_rule["penalty_multiplier"]
                    liq_reason  = liq_rule["reason"]
            except (ValueError, TypeError):
                pass

    # --- Bonuses ---------------------------------------------------------
    bonus         = 0.0
    bonus_reasons = []
    for rule in MB_TECH_BONUSES:
        rule_metrics = [k for k in rule["gates"].keys() if not k.startswith("_")]
        if any(m in excluded for m in rule_metrics):
            continue
        passes, _ = evaluate_gates(rule["gates"], indicators)
        if passes:
            bonus += rule["bonus"]
            bonus_reasons.append(rule["reason"])

    final_score = round(max(0.0, min(10.0, total_score - penalty - liq_penalty + bonus)), 2)

    return {
        "score":          final_score,
        "horizon":        horizon,
        "base_score":     round(total_score, 2),
        "category_scores": category_scores,
        "breakdown":      category_breakdown,
        "penalties":      {
            "total":     round(penalty + liq_penalty, 2),
            "technical": round(penalty, 2),
            "liquidity": round(liq_penalty, 2),
            "reasons":   penalty_reasons + ([liq_reason] if liq_reason else []),
        },
        "bonuses": {"total": round(bonus, 2), "reasons": bonus_reasons},
    }


# =============================================================================
# CUSTOM EXTRACTOR STACK
# =============================================================================

class MBConfigExtractor(ConfigExtractor):
    """
    ConfigExtractor subclass that injects MB-specific config sections
    after super().__init__() has run.

    Overrides:
        1. sections["horizon_pillar_weights"]  — baked in from module-level import
        2. sections["hybrid_pillar_composition"] — baked in from module-level import
        3. self.confidence_config + re-runs extract_confidence_sections()
        4. extract_matrix_sections() — reads from self.master_config (MB_MASTER_CONFIG)
           instead of hard-importing from setup_pattern_matrix_config.py
    """

    def __init__(self, master_config: Dict, horizon: str, log=None):
        # ✅ P2-2 FULL FIX: Pass MB_CONFIDENCE_CONFIG directly to super().__init__
        # This prevents double-extraction and double-import log spam.
        super().__init__(
            master_config, 
            horizon, 
            logger=log, 
            confidence_config_override=MB_CONFIDENCE_CONFIG
        )

        # Overwrite baked-in module-level sections
        for key, val in [
            ("horizon_pillar_weights",    MB_HORIZON_PILLAR_WEIGHTS.get(horizon, {})),
            ("hybrid_pillar_composition", MB_HYBRID_PILLAR_COMPOSITION.get(horizon, {})),
        ]:
            if key in self.sections:
                self.sections[key].data = val
            else:
                self.sections[key] = ConfigSection(
                    data=val, source=f"MB_override.{key}"
                )

    def extract_matrix_sections(self):
        """
        Read setup_pattern_matrix and strategy_matrix from self.master_config
        (= MB_MASTER_CONFIG) instead of the hard file imports in the base class.

        Section keys produced are identical to the base class so the resolver
        finds them under the same names it always queries.
        """
        setup_matrix    = self.master_config.get("setup_pattern_matrix", {})
        strategy_matrix = self.master_config.get("strategy_matrix", {})

        self.sections["setup_pattern_matrix"] = ConfigSection(
            data=setup_matrix, source="MB_MASTER_CONFIG.setup_pattern_matrix"
        )
        self.sections["strategy_matrix"] = ConfigSection(
            data=strategy_matrix, source="MB_MASTER_CONFIG.strategy_matrix"
        )

        for setup_name, setup_config in setup_matrix.items():
            self.sections[f"setup_{setup_name}"] = ConfigSection(
                data=setup_config, source=f"MB.setup.{setup_name}"
            )
            ctx_reqs = setup_config.get("context_requirements", {})
            if ctx_reqs:
                self.sections[f"setup_context_{setup_name}"] = ConfigSection(
                    data=ctx_reqs, source=f"MB.setup.{setup_name}.context_requirements"
                )
            val_mods = setup_config.get("validation_modifiers", {})
            if val_mods:
                self.sections[f"setup_validation_{setup_name}"] = ConfigSection(
                    data=val_mods, source=f"MB.setup.{setup_name}.validation_modifiers"
                )

        for strat_name, strat_config in strategy_matrix.items():
            self.sections[f"strategy_{strat_name}"] = ConfigSection(
                data=strat_config, source=f"MB.strategy.{strat_name}"
            )
            fit_inds = strat_config.get("fit_indicators", {})
            if fit_inds:
                self.sections[f"strategy_fit_{strat_name}"] = ConfigSection(
                    data=fit_inds, source=f"MB.strategy.{strat_name}.fit_indicators"
                )
            scoring_rules = strat_config.get("scoring_rules", {})
            if scoring_rules:
                self.sections[f"strategy_scoring_{strat_name}"] = ConfigSection(
                    data=scoring_rules, source=f"MB.strategy.{strat_name}.scoring_rules"
                )

        self.logger.info("✅ MB matrix sections extracted from MB_MASTER_CONFIG")


class MBQueryOptimizedExtractor(QueryOptimizedExtractor):
    """
    QOE subclass that routes all scoring through the MB-specific functions
    and overrides get_hybrid_pillar_composition() to return MB weights.
    """

    def __init__(self, master_config: Dict, horizon: str, log=None):
        # ✅ P2-1 FULL FIX: Inject MBConfigExtractor into super().__init__
        # This eliminates the "super() overhead" of creating two base extractors.
        mb_base_extractor = MBConfigExtractor(master_config, horizon, log)
        super().__init__(master_config, horizon, logger=log, base_extractor=mb_base_extractor)
        
        # ✅ P2-1 REJECTION FIX: Use OrderedDict to maintain LRU compatibility
        self._gate_cache     = OrderedDict()
        self._pattern_cache  = OrderedDict()

    # -- Scoring overrides -------------------------------------------------

    def get_fundamental_score(self, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
        """Route to MB-specific fundamental scorer."""
        return mb_compute_fundamental_score(fundamentals, self.horizon)

    def get_technical_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Route to MB-specific technical scorer."""
        return mb_compute_technical_score(indicators, self.horizon)

    def get_hybrid_pillar_composition(self) -> Dict:
        """Return MB hybrid composition instead of the hard-imported main config."""
        return MB_HYBRID_PILLAR_COMPOSITION.get(self.horizon, {})


class MBConfigResolver(ConfigResolver):
    """
    ConfigResolver subclass that injects the MB extractor stack.

    Does NOT call super().__init__() because that would create a
    QueryOptimizedExtractor backed by the main MASTER_CONFIG.
    Replicates exactly what ConfigResolver.__init__() does, but uses
    MBQueryOptimizedExtractor instead.

    Runs the same validate_extractor_state() guard so misconfiguration
    fails loudly rather than silently producing wrong scores.
    """

    def __init__(self, master_config: Dict, horizon: str, log=None):
        import logging as _logging
        self.horizon  = horizon
        self.logger   = log or _logging.getLogger(__name__)
        self.extractor = MBQueryOptimizedExtractor(master_config, horizon, self.logger)

        # Keep the same validation gate as the base class
        state = self.extractor.validate_extractor_state()
        if not state.get("valid") or not state.get("has_confidence_config"):
            raise RuntimeError(
                f"MB extractor invalid for horizon={horizon}: "
                f"errors={state.get('errors')}, "
                f"has_conf={state.get('has_confidence_config')}"
            )
        self.logger.info(f"✅ MBConfigResolver initialised for {horizon}")


# =============================================================================
# PHASE 2 RUNNER
# =============================================================================

def run_mb_resolver(
    symbol:       str,
    fundamentals: Dict,
    indicators:   Dict,
    patterns:     Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Execute Phase 2 evaluation for a single MB candidate.

    Thread-safe: creates a fresh resolver per call, patches no globals.
    Bypasses build_evaluation_context() because that function calls
    get_resolver(horizon) internally and would use the cached main resolver.
    Instead, calls resolver.build_evaluation_context_only() directly with
    the MB resolver already in hand.

    Args:
        symbol:       NSE symbol (e.g. "RELIANCE.NS")
        fundamentals: Output of compute_fundamentals() — nested {raw,value,score} dicts
        indicators:   Output of compute_indicators_cached("multibagger") — same format
        patterns:     Optional pre-computed patterns dict

    Returns:
        Dict with keys:
            status, final_score, final_decision_score,
            technical_score, fundamental_score, hybrid_score,
            confidence, setup, strategy, eval_ctx, opportunity
        Or None on unrecoverable error.
    """
    try:
        mb_resolver = MBConfigResolver(MB_MASTER_CONFIG, _HORIZON, logger)
        extractor   = mb_resolver.extractor

        # Flatten inputs — same as build_evaluation_context() does internally
        clean_ind   = flatten_market_data_mixed(indicators   or {})
        clean_fund  = flatten_market_data_mixed(fundamentals or {})
        price_data  = _extract_price_data(indicators, fundamentals)
        clean_price = flatten_market_data_mixed(price_data)
        det_patterns = patterns if patterns else _extract_patterns(indicators, _HORIZON)

        eval_ctx = mb_resolver.build_evaluation_context_only(
            symbol           = symbol,
            fundamentals     = clean_fund,
            indicators       = clean_ind,
            price_data       = clean_price,
            detected_patterns = det_patterns,
        )

        if not eval_ctx or "error" in eval_ctx:
            logger.warning(f"[MB Phase2] {symbol}: eval_ctx error — {eval_ctx.get('error')}")
            return None

        scoring      = eval_ctx.get("scoring", {})
        tech_score   = scoring.get("technical",    {}).get("score", 0.0)
        fund_score   = scoring.get("fundamental",  {}).get("score", 0.0)
        hybrid_score = scoring.get("hybrid",       {}).get("score", 0.0)
        conf_info    = eval_ctx.get("confidence",  {})
        confidence   = conf_info.get("clamped", 0)
        setup        = eval_ctx.get("setup",       {}).get("type", "GENERIC")
        strategy     = eval_ctx.get("strategy",    {}).get("primary_strategy")
        
        # Extract estimated_hold_months from strategy config
        _strategy_cfg         = MB_MASTER_CONFIG.get("strategy_matrix", {}).get(strategy or "", {})
        estimated_hold_months = _strategy_cfg.get("estimated_hold_months")

        # ✅ FIX: Extract entry_trigger from risk_candidates dict
        risk_info = eval_ctx.get("risk_candidates", {})
        if risk_info.get("rr_source") == "pattern":
            entry_trigger = risk_info.get("primary_pattern") or "PATTERN_MATCH"
        else:
            entry_trigger = "TECHNICAL_SETUP"

        eligibility = calculate_structural_eligibility(
            tech_score, fund_score, hybrid_score,
            _HORIZON, extractor=extractor,
        )

        # ✅ P0-1 REJECTION FIX: Use decoupled compute_opportunity_score_full
        opportunity = compute_opportunity_score_full(
            ticker                 = symbol,
            horizon                = _HORIZON,
            eligibility_base_score = eligibility,
            indicators             = clean_ind,
            fundamentals           = clean_fund,
            eval_ctx               = eval_ctx,
            extractor              = extractor,
        )

        final_decision_score = (
            opportunity.get("final_decision_score") or
            opportunity.get("final_score") or          # defensive fallback
            eligibility
        )

        logger.info(
            f"[MB Phase2] {symbol} | "
            f"Fund={fund_score:.1f} Tech={tech_score:.1f} Hybrid={hybrid_score:.1f} "
            f"Elig={eligibility:.1f} Final={final_decision_score:.1f} Conf={confidence:.0f}"
        )

        return {
            "status":                 "SUCCESS",
            "symbol":                 symbol,
            "final_score":            eligibility,
            "final_decision_score":   final_decision_score,
            "technical_score":        tech_score,
            "fundamental_score":      fund_score,
            "hybrid_score":           hybrid_score,
            "confidence":             confidence,
            "setup":                  setup,
            "strategy":               strategy,
            "entry_trigger":          entry_trigger,
            "estimated_hold_months":  estimated_hold_months,
            "eval_ctx":               eval_ctx,
            "opportunity":            opportunity,
        }

    except RuntimeError as e:
        # Resolver validation failure — config problem, not a data problem
        logger.error(f"[MB Phase2] {symbol}: resolver init failed — {e}")
        return None
    except Exception as e:
        logger.error(f"[MB Phase2] {symbol}: unexpected error — {e}", exc_info=True)
        return None
