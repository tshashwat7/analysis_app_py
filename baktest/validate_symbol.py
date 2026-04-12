"""
validate_symbol.py
==================
Drop-in merge of validate_single_symbol.py + validate_gates.py.

Builds eval_ctx ONCE, then runs:
  Part 1 — Math audit  (scores, confidence chain, gates status)
  Part 2 — Gate coverage (resolved gates, thresholds, context paths)

Usage:
    python validate_symbol.py --symbol RELIANCE.NS --horizon short_term
    python validate_symbol.py --symbol INFY.NS --horizon long_term --report /tmp/report.txt
    python validate_symbol.py --symbol TCS.NS --part math      # math audit only
    python validate_symbol.py --symbol TCS.NS --part gates     # gate audit only
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "baktest"))

from config.config_helpers import build_evaluation_context, get_resolver
from config.config_utility.logger_config import setup_logger
from services.data_fetch import _get_val
from services.fundamentals import compute_fundamentals
from services.indicator_cache import compute_indicators_cached


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "None"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write(line: str, sink) -> None:
    print(line)
    sink.write(line + "\n")


def _heading(title: str, sink) -> None:
    _write("", sink)
    _write("=" * 96, sink)
    _write(title, sink)
    _write("=" * 96, sink)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 helpers — Math audit (from validate_single_symbol.py)
# ─────────────────────────────────────────────────────────────────────────────

def _sum_weights(category_scores: Dict[str, Dict[str, Any]]) -> float:
    total = 0.0
    for data in category_scores.values():
        weight = data.get("weight")
        if isinstance(weight, (int, float)):
            total += float(weight)
    return total


def _iter_gate_rows(by_gate: Dict[str, Dict[str, Any]]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for gate_name in sorted(by_gate.keys()):
        yield gate_name, by_gate[gate_name]


def _print_category_breakdown(title: str, pillar: Dict[str, Any], sink) -> None:
    _heading(title, sink)
    score = _get_val(pillar, "score")
    _write(f"Final score: {_fmt_num(score)}", sink)

    category_scores = pillar.get("category_scores", {}) or {}
    if not category_scores:
        _write("No category_scores found.", sink)
        return

    total_weight = _sum_weights(category_scores)
    _write(f"Category weight sum: {_fmt_num(total_weight, 4)}", sink)
    _write("Category breakdown:", sink)
    for category, data in category_scores.items():
        _write(
            f"  {category:18} score={_fmt_num(data.get('score')):>8} "
            f"weight={_fmt_num(data.get('weight'), 4):>8} "
            f"weighted={_fmt_num(data.get('weighted')):>8}",
            sink,
        )

    penalties = pillar.get("penalties", {}) or {}
    bonuses = pillar.get("bonuses", {}) or {}
    if penalties:
        _write(f"Penalty total: {_fmt_num(penalties.get('total', 0))}", sink)
        for reason in penalties.get("reasons", []) or []:
            _write(f"  penalty: {reason}", sink)
    if bonuses:
        _write(f"Bonus total: {_fmt_num(bonuses.get('total', 0))}", sink)
        for reason in bonuses.get("reasons", []) or []:
            _write(f"  bonus: {reason}", sink)


def _print_hybrid_breakdown(hybrid: Dict[str, Any], sink) -> None:
    _heading("HYBRID SCORE", sink)
    final_score = _get_val(hybrid, "score")
    _write(f"Final score: {_fmt_num(final_score)}", sink)
    metrics = hybrid.get("metrics", {}) or {}
    if not metrics:
        _write("No hybrid metrics found.", sink)
        return
    for metric, data in metrics.items():
        m_raw = _get_val(data, "raw")
        m_score = _get_val(data, "score")
        _write(
            f"  {metric:25} raw={_fmt_num(m_raw, 3):>10} score={_fmt_num(m_score):>8}",
            sink,
        )


def _print_confidence(conf: Dict[str, Any], sink) -> None:
    _heading("CONFIDENCE CHAIN", sink)

    base = conf.get("base")
    adjustment_total = (conf.get("adjustments") or {}).get("total")
    final_score = conf.get("final")
    clamped = conf.get("clamped")
    clamp_range = conf.get("clamp_range")

    _write(f"Base floor:       {_fmt_num(base)}", sink)
    _write(f"Total adjustment: {_fmt_num(adjustment_total, 1)}", sink)
    _write(f"Pre-clamp final:  {_fmt_num(final_score, 1)}", sink)
    _write(f"Clamped:          {_fmt_num(clamped, 1)}", sink)
    _write(f"Clamp range:      {clamp_range}", sink)

    if isinstance(base, (int, float)) and isinstance(adjustment_total, (int, float)) and isinstance(final_score, (int, float)):
        expected_final = float(base) + float(adjustment_total)
        diff = abs(expected_final - float(final_score))
        status = "OK" if diff < 1e-6 else f"MISMATCH by {diff:.4f}"
        _write(f"Math check:       base + adjustment = {_fmt_num(expected_final, 1)} -> {status}", sink)

    if (
        isinstance(clamp_range, (list, tuple))
        and len(clamp_range) == 2
        and isinstance(final_score, (int, float))
        and isinstance(clamped, (int, float))
    ):
        expected_clamped = max(clamp_range[0], min(clamp_range[1], final_score))
        status = "OK" if abs(expected_clamped - clamped) < 1e-6 else f"MISMATCH by {abs(expected_clamped - clamped):.4f}"
        _write(f"Clamp check:      expected {_fmt_num(expected_clamped, 1)} -> {status}", sink)

    _write("", sink)
    _write("Adjustment breakdown:", sink)
    for item in (conf.get("adjustments") or {}).get("breakdown", []) or []:
        _write(f"  {item}", sink)

    structured = conf.get("structured_adjustments", []) or []
    if structured:
        _write("", sink)
        _write("Structured adjustments:", sink)
        for item in structured:
            _write(
                f"  {item.get('source','unknown'):16} "
                f"{item.get('name','unknown'):24} "
                f"delta={_fmt_num(item.get('delta'), 2)}",
                sink,
            )


def _print_sector_context(indicators: Dict[str, Any], sink) -> None:
    _heading("SECTOR CONTEXT", sink)
    if not indicators:
        _write("No indicator payload found.", sink)
        return

    sector_name = indicators.get("sectorName")
    sector_benchmark = indicators.get("sectorBenchmark")
    sector_available = indicators.get("sectorDataAvailable")
    sector_trend = indicators.get("sectorTrendScore")
    rs_fast = indicators.get("rsVsSectorFast")
    rs_slow = indicators.get("rsVsSectorSlow")

    def _metric_value(metric):
        if isinstance(metric, dict):
            return metric.get("value")
        return metric

    _write(f"Sector:           {_metric_value(sector_name)}", sink)
    _write(f"Benchmark:        {_metric_value(sector_benchmark)}", sink)
    _write(f"Data available:   {_metric_value(sector_available)}", sink)
    _write(f"Sector trend:     {_fmt_num(_metric_value(sector_trend), 2)}", sink)
    _write(f"RS vs sector fast:{_fmt_num(_metric_value(rs_fast), 2)}", sink)
    _write(f"RS vs sector slow:{_fmt_num(_metric_value(rs_slow), 2)}", sink)


def _print_gates_math(eval_ctx: Dict[str, Any], sink) -> None:
    _heading("STRUCTURAL GATES", sink)
    structural = eval_ctx.get("structural_gates", {}) or {}
    by_gate = structural.get("by_gate", {}) or {}
    none_actuals: List[str] = []

    for gate_name, result in _iter_gate_rows(by_gate):
        actual = result.get("actual")
        if actual is None and result.get("status") not in {"skipped"}:
            none_actuals.append(gate_name)
        _write(
            f"  {gate_name:25} status={result.get('status','?'):8} "
            f"actual={str(actual):14} required={result.get('required')} "
            f"source={result.get('source')}",
            sink,
        )

    overall = structural.get("overall", {}) or {}
    _write("", sink)
    _write(f"Overall passed: {overall.get('passed')}", sink)
    _write(f"Failed gates:   {overall.get('failed_gates', [])}", sink)
    _write(f"Unexpected actual=None gates: {none_actuals}", sink)

    _heading("EXECUTION RULES", sink)
    execution = eval_ctx.get("execution_rules", {}) or {}
    rules = execution.get("rules", {}) or {}
    for rule_name in sorted(rules.keys()):
        result = rules[rule_name]
        _write(
            f"  {rule_name:25} status={result.get('status','?'):10} "
            f"severity={result.get('severity', 0):>3} reason={result.get('reason')}",
            sink,
        )
    _write(f"Overall passed: {((execution.get('overall') or {}).get('passed'))}", sink)


def _print_setup(eval_ctx: Dict[str, Any], sink) -> None:
    _heading("SETUP CLASSIFICATION", sink)
    setup = eval_ctx.get("setup", {}) or {}
    best = setup.get("best", {}) or {}
    candidates = setup.get("candidates", []) or []
    rejected = setup.get("rejected", []) or []

    priority = best.get("priority", 0)
    fit_score = best.get("fit_score", 0)
    composite = best.get("composite_score", 0)

    # Verify 70% Priority, 30% Fit
    expected_comp = (priority * 0.7) + (fit_score * 0.3)
    math_status = "OK" if abs(expected_comp - composite) < 0.1 else f"ERR: Expected {expected_comp:.1f}"

    _write(f"Best setup:      {best.get('type')}", sink)
    _write(f"Priority:        {_fmt_num(priority)} (Weight: 70%)", sink)
    _write(f"Fit score:       {_fmt_num(fit_score, 1)} (Weight: 30%)", sink)
    _write(f"Composite score: {_fmt_num(composite, 1)} [{math_status}]", sink)
    _write(f"Candidates:      {len(candidates)}", sink)

    if candidates:
        _write("", sink)
        _write("Top candidates:", sink)
        for candidate in candidates[:5]:
            _write(
                f"  {candidate.get('type'):25} "
                f"priority={candidate.get('priority')} "
                f"fit={_fmt_num(candidate.get('fit_score'), 1)} "
                f"composite={_fmt_num(candidate.get('composite_score'), 1)}",
                sink,
            )

    if rejected:
        _write("", sink)
        _write("Top rejected setups:", sink)
        for item in rejected[:8]:
            _write(f"  {item.get('type'):25} reason={item.get('reason')}", sink)


def _print_profile_trace(eval_ctx: Dict[str, Any], sink) -> None:
    _heading("PROFILE / OPPORTUNITY TRACE", sink)
    scoring = eval_ctx.get("scoring", {}) or {}

    technical = _get_val(scoring.get("technical", {}), "score")
    fundamental = _get_val(scoring.get("fundamental", {}), "score")
    hybrid = _get_val(scoring.get("hybrid", {}), "score")

    confidence = _get_val(eval_ctx.get("confidence", {}), "clamped")
    opportunity = ((eval_ctx.get("opportunity_gates") or {}).get("overall", {}))

    _write(f"Technical score:   {_fmt_num(technical)}", sink)
    _write(f"Fundamental score: {_fmt_num(fundamental)}", sink)
    _write(f"Hybrid score:      {_fmt_num(hybrid)}", sink)
    _write(f"Confidence:        {_fmt_num(confidence)}", sink)
    _write(f"Opportunity pass:  {opportunity.get('passed')}", sink)
    _write(f"Opportunity fails: {opportunity.get('failed_gates', [])}", sink)


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 helpers — Gate coverage audit (from validate_gates.py)
# ─────────────────────────────────────────────────────────────────────────────

def _status_for_threshold(value: Any, threshold: Dict[str, Any], optional: bool = False) -> str:
    if value is None:
        return "SKIPPED (Optional)" if optional else "MISSING"
    if not threshold:
        return "SKIPPED (No Threshold)"
    min_t = threshold.get("min")
    max_t = threshold.get("max")
    if min_t is not None and value < min_t:
        return f"FAIL (need >= {min_t})"
    if max_t is not None and value > max_t:
        return f"FAIL (need <= {max_t})"
    return "PASS"


def _iter_resolved_gates(resolved: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    for gate_name in sorted(resolved.keys()):
        yield gate_name, resolved[gate_name]


def _print_gate_coverage(eval_ctx: Dict[str, Any], resolver, setup_type: str, sink) -> int:
    """
    Prints resolved gate details for structural and opportunity phases.
    Returns count of non-optional gates with missing values.
    """
    extractor = resolver.extractor
    registry = extractor.get_gate_registry()
    _write(f"Registry metrics: {len(registry)}", sink)

    missing_total = 0

    for phase in ("structural", "opportunity"):
        _heading(f"{phase.upper()} GATES", sink)
        resolved_gates = extractor.get_resolved_gates(phase, setup_type)
        if not resolved_gates:
            _write("No resolved gates.", sink)
            continue

        for gate_name, resolved_gate in _iter_resolved_gates(resolved_gates):
            value = resolver._resolve_gate_value_from_context(gate_name, eval_ctx)
            threshold = resolved_gate.threshold

            registry_meta = registry.get(gate_name, {})
            optional = registry_meta.get("optional", False)

            status = _status_for_threshold(value, threshold, optional=optional)

            if value is None and not optional:
                missing_total += 1

            context_paths = registry_meta.get("context_paths", [])
            _write(
                f"{gate_name:28} actual={_fmt(value):12} "
                f"threshold={str(threshold):28} source={resolved_gate.source:12} "
                f"status={status}",
                sink,
            )
            _write(f"  context_paths={context_paths}", sink)

        # Print phase overall
        overall_key = f"{phase}_gates"
        overall = ((eval_ctx.get(overall_key) or {}).get("overall")) or {}
        _write("", sink)
        _write(f"{phase} overall passed: {overall.get('passed')}", sink)
        _write(f"{phase} failed gates:   {overall.get('failed_gates', [])}", sink)

    return missing_total


def _print_block_trace(eval_ctx: Dict[str, Any], missing_total: int, sink) -> None:
    _heading("BLOCK TRACE", sink)
    structural_overall = ((eval_ctx.get("structural_gates") or {}).get("overall")) or {}
    opportunity_overall = ((eval_ctx.get("opportunity_gates") or {}).get("overall")) or {}
    execution_overall = ((eval_ctx.get("execution_rules") or {}).get("overall")) or {}
    confidence = eval_ctx.get("confidence", {}) or {}

    _write(f"structural passed:  {structural_overall.get('passed')}", sink)
    _write(f"opportunity passed: {opportunity_overall.get('passed')}", sink)
    _write(f"execution passed:   {execution_overall.get('passed')}", sink)
    _write(f"confidence clamped: {_fmt(confidence.get('clamped'))}", sink)
    _write(f"missing gate values: {missing_total}", sink)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(symbol: str, horizon: str, report_path: str, part: str) -> int:
    setup_logger()

    print(f"Fetching production data for {symbol} [{horizon}] ...")
    fundamentals = compute_fundamentals(symbol)
    indicators, patterns = compute_indicators_cached(
        symbol,
        horizon,
        sector=fundamentals.get("sector") if isinstance(fundamentals, dict) else None,
    )

    print("Building evaluation context ...")
    eval_ctx = build_evaluation_context(
        ticker=symbol,
        indicators=indicators,
        fundamentals=fundamentals,
        horizon=horizon,
        patterns=patterns,
    )

    # ✅ Re-run key evaluation phases explicitly for easy isolation during audits.
    # This ensures Part 1 (Math Audit) aligns perfectly with Part 2 (Gate Coverage).
    resolver = get_resolver(horizon)
    
    # Update eval_ctx with fresh resolution results in the same dependency order
    # used by the production pipeline. Confidence consumes execution summary data,
    # so execution_rules must be refreshed before confidence is recalculated.
    eval_ctx["structural_gates"] = resolver._validate_structural_gates(eval_ctx)
    eval_ctx["execution_rules"] = resolver._validate_execution_rules(eval_ctx)
    eval_ctx["confidence"] = resolver._calculate_confidence(eval_ctx)
    eval_ctx["opportunity_gates"] = resolver._validate_opportunity_gates(eval_ctx)

    setup_type = ((eval_ctx.get("setup") or {}).get("type")) or "GENERIC"

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as sink:
        _write(f"Validation run: {datetime.now().isoformat()}", sink)
        _write(f"Symbol:         {symbol}", sink)
        _write(f"Horizon:        {horizon}", sink)
        _write(f"Setup type:     {setup_type}", sink)
        _write(f"Pattern count:  {len(patterns or {})}", sink)
        _write(f"Indicator keys: {len(indicators or {})}", sink)
        _write(f"  sectorTrendScore: {indicators.get('sectorTrendScore') if indicators else 'None'}", sink)
        _write(f"  rsVsSectorFast: {indicators.get('rsVsSectorFast') if indicators else 'None'}", sink)
        _write(f"  rsVsSectorSlow: {indicators.get('rsVsSectorSlow') if indicators else 'None'}", sink)
        _write(f"  sectorName: {indicators.get('sectorName') if indicators else 'None'}", sink)
        _write(f"Fundamental keys: {len(fundamentals or {})}", sink)

        # ── Part 1: Math audit ──────────────────────────────────────────────
        if part in ("math", "all"):
            _write("", sink)
            _write("=" * 96, sink)
            _write("PART 1 — MATH AUDIT", sink)
            _write("=" * 96, sink)

            scoring = eval_ctx.get("scoring", {}) or {}
            _print_category_breakdown("TECHNICAL SCORE", scoring.get("technical", {}) or {}, sink)
            _print_category_breakdown("FUNDAMENTAL SCORE", scoring.get("fundamental", {}) or {}, sink)
            _print_hybrid_breakdown(scoring.get("hybrid", {}) or {}, sink)
            _print_sector_context(indicators or {}, sink)
            _print_confidence(eval_ctx.get("confidence", {}) or {}, sink)
            _print_gates_math(eval_ctx, sink)
            _print_setup(eval_ctx, sink)
            _print_profile_trace(eval_ctx, sink)

        # ── Part 2: Gate coverage audit ─────────────────────────────────────
        if part in ("gates", "all"):
            _write("", sink)
            _write("=" * 96, sink)
            _write("PART 2 — GATE COVERAGE AUDIT", sink)
            _write("=" * 96, sink)

            missing_total = _print_gate_coverage(eval_ctx, resolver, setup_type, sink)
            _print_block_trace(eval_ctx, missing_total, sink)

    print(f"\nWrote validation report to {report_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-symbol math audit + gate coverage (merged validator)"
    )
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--horizon", default="short_term")
    parser.add_argument(
        "--report",
        default=os.path.join(os.path.dirname(__file__), "output", "validation_report.txt"),
    )
    parser.add_argument(
        "--part",
        choices=["math", "gates", "all"],
        default="all",
        help="Which audit section to run (default: all)",
    )
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    try:
        return run(symbol, args.horizon, args.report, args.part)
    except Exception as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
