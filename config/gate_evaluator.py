# config/gate_evaluator.py
"""
Gate Evaluator — Pure, stateless gate evaluation engine.

All structured gate evaluation (min/max/equals/min_metric/max_metric, AND/OR logic)
lives here. Nothing in this module reads config files or holds state.

Importable by any layer:
    from config.gate_evaluator import evaluate_gates, evaluate_invalidation_gates

QueryOptimizedExtractor delegates its instance methods here so callers that already
hold an extractor instance don't need to change call sites.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _unwrap(value: Any) -> Any:
    """
    Unwrap a nested metric dict to its scalar value.

    Priority: value → raw → score → the dict itself (if no keys exist).
    If the input is already a scalar, return it unchanged.
    Note: Skip string values (e.g., "0%") and fall through to numeric alternatives.
    """
    if isinstance(value, dict):
        # Priority: value -> raw -> score
        for key in ["value", "raw", "score"]:
            v = value.get(key)
            if v is not None and not isinstance(v, str):
                return v
        return value.get("raw")
    return value


def _check_single_metric(
    metric: str,
    thresholds: Dict[str, Any],
    value: Any,
    data: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Evaluate every threshold clause for one metric.

    Returns (metric_passed, list_of_failure_strings).
    """
    passed = True
    failures: List[str] = []

    # min
    if "min" in thresholds:
        min_val = thresholds["min"]
        if min_val is not None:
            if value is None:
                passed = False
                failures.append(f"{metric}: missing value for min threshold")
            elif not isinstance(value, (int, float)):
                passed = False
                failures.append(f"{metric}: non-numeric value ({type(value).__name__}) for min check")
            elif value < min_val:
                passed = False
                failures.append(f"{metric}: {value} < min({min_val})")

    # max
    if "max" in thresholds:
        max_val = thresholds["max"]
        if max_val is not None:
            if value is None:
                passed = False
                failures.append(f"{metric}: missing value for max threshold")
            elif not isinstance(value, (int, float)):
                passed = False
                failures.append(f"{metric}: non-numeric value ({type(value).__name__}) for max check")
            elif value > max_val:
                passed = False
                failures.append(f"{metric}: {value} > max({max_val})")

    # equals
    if "equals" in thresholds:
        eq_val = thresholds["equals"]
        if value != eq_val:
            passed = False
            failures.append(f"{metric}: {value} != {eq_val}")

    # min_metric — value must be >= ref_metric * multiplier
    if "min_metric" in thresholds:
        ref_name = thresholds["min_metric"]
        ref_val = _unwrap(data.get(ref_name))
        if ref_val is None:
            passed = False
            failures.append(f"{metric}: ref metric '{ref_name}' missing")
        else:
            mult = thresholds.get("multiplier", 1.0)
            threshold = ref_val * mult
            if value is None:
                passed = False
                failures.append(f"{metric}: missing value for min_metric check")
            elif not isinstance(value, (int, float)):
                passed = False
                failures.append(f"{metric}: non-numeric value ({type(value).__name__}) for min_metric check")
            elif value < threshold:
                passed = False
                failures.append(f"{metric}: {value} < {ref_name}({ref_val}) * {mult}")

    # max_metric — value must be <= ref_metric * multiplier
    if "max_metric" in thresholds:
        ref_name = thresholds["max_metric"]
        ref_val = _unwrap(data.get(ref_name))
        if ref_val is None:
            passed = False
            failures.append(f"{metric}: ref metric '{ref_name}' missing")
        else:
            mult = thresholds.get("multiplier", 1.0)
            threshold = ref_val * mult
            if value is None:
                passed = False
                failures.append(f"{metric}: missing value for max_metric check")
            elif not isinstance(value, (int, float)):
                passed = False
                failures.append(f"{metric}: non-numeric value ({type(value).__name__}) for max_metric check")
            elif value > threshold:
                passed = False
                failures.append(f"{metric}: {value} > {ref_name}({ref_val}) * {mult}")

    return passed, failures


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_gates(
    gates: Dict[str, Any],
    data: Dict[str, Any],
    empty_gates_pass: bool = True,
) -> Tuple[bool, List[str]]:
    """
    Evaluate structured gate conditions against market data.

    Supports:
        - min / max / equals thresholds
        - min_metric / max_metric (cross-metric comparisons with optional multiplier)
        - AND / OR logical operators  (``_logic`` key, default AND)
        - Nested metric dicts  (``{"value": x}`` or ``{"raw": x}``)
        - None threshold values are silently skipped (safe for config placeholders)

    Args:
        gates: Gate config dict.
            Example::

                {
                    "adx":  {"min": 20},
                    "rvol": {"min": 1.5},
                    "_logic": "AND"
                }

        data: Flat or nested metric dict.
            Example::

                {"adx": 25, "rvol": {"value": 2.1, "raw": 2.1}}

        empty_gates_pass: If True (default), a gate dict with no metric keys
            returns (True, []). Set to False for strict mode.

    Returns:
        ``(passes, failures)`` where *failures* is a list of human-readable
        strings describing each failed check (empty when *passes* is True).
    """
    logic = gates.get("_logic", "AND").upper()
    results: List[bool] = []
    failures: List[str] = []

    for metric, thresholds in gates.items():
        if metric.startswith("_"):
            continue

        value = _unwrap(data.get(metric))

        if value is None:
            results.append(False)
            failures.append(f"{metric}: missing from data")
            continue

        if not isinstance(thresholds, dict):
            logger.warning("evaluate_gates: invalid threshold format for %s: %r", metric, thresholds)
            results.append(False)
            failures.append(f"{metric}: invalid threshold config")
            continue

        metric_passed, metric_failures = _check_single_metric(metric, thresholds, value, data)
        results.append(metric_passed)
        if not metric_passed:
            failures.extend(metric_failures)

    if not results:
        return (True, []) if empty_gates_pass else (False, ["No gates defined"])

    passes = any(results) if logic == "OR" else all(results)
    return passes, failures if not passes else []


def evaluate_invalidation_gates(
    gates: Dict[str, Any],
    data: Dict[str, Any],
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Evaluate invalidation / breakdown gate conditions.

    Unlike :func:`evaluate_gates`, this function returns a per-metric result list
    so callers can implement duration tracking (e.g. "condition must hold for N
    candles") even when overall logic is OR.

    Args:
        gates: Gate config dict (same format as :func:`evaluate_gates`).
            Thresholds may include a ``"duration"`` key (int, default 1) which is
            passed through to callers for external duration tracking.
        data: Flat or nested metric dict.

    Returns:
        ``(triggered, gate_results)``

        *triggered* — True if the overall invalidation condition fired.

        *gate_results* — list of dicts, one per metric::

            {
                "metric":    str,
                "triggered": bool,   # True = this specific gate fired
                "duration":  int,    # from threshold config, default 1
                "reason":    str,
            }
    """
    logic = gates.get("_logic", "AND").upper()
    results: List[bool] = []
    gate_results: List[Dict[str, Any]] = []

    for metric, thresholds in gates.items():
        if metric.startswith("_"):
            continue

        value = _unwrap(data.get(metric))
        duration = thresholds.get("duration", 1) if isinstance(thresholds, dict) else 1

        if value is None:
            results.append(False)
            gate_results.append({
                "metric": metric,
                "triggered": False,
                "duration": duration,
                "reason": f"{metric}: missing from data",
            })
            continue

        if not isinstance(thresholds, dict):
            logger.warning(
                "evaluate_invalidation_gates: invalid threshold format for %s: %r",
                metric, thresholds,
            )
            results.append(False)
            gate_results.append({
                "metric": metric,
                "triggered": False,
                "duration": duration,
                "reason": f"{metric}: invalid threshold config",
            })
            continue

        metric_passed, metric_failures = _check_single_metric(metric, thresholds, value, data)
        results.append(metric_passed)
        gate_results.append({
            "metric": metric,
            "triggered": metric_passed,
            "duration": duration,
            "reason": "; ".join(metric_failures) if not metric_passed else "Condition triggered",
        })

    if not results:
        return False, []

    triggered = any(results) if logic == "OR" else all(results)
    return triggered, gate_results
