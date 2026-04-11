import argparse
from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd


RESULTS_DIR = Path(__file__).parent / "output" / "backtest_results"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, encoding="utf-8")


def _health_label(value: float, healthy_low: float, healthy_high: float) -> str:
    if value < healthy_low:
        return "Investigate: too low"
    if value > healthy_high:
        return "Investigate: too high"
    return "Healthy"


def _print_section(title: str) -> None:
    print("\n" + "=" * 84)
    print(title)
    print("=" * 84)


def analyze(results_dir: Path) -> int:
    outcomes = _load_csv(results_dir / "forward_outcomes.csv")
    signals = _load_csv(results_dir / "signal_log.csv")
    confidence = _load_csv(results_dir / "confidence_audit.csv")
    transitions = _load_csv(results_dir / "transitions_log.csv")

    _print_section("WIN RATE BY HORIZON")
    for horizon in sorted(outcomes["horizon"].dropna().unique()):
        h = outcomes[outcomes["horizon"] == horizon]
        tradeable = h[h["signal"].isin(["BUY", "SELL"])]
        if tradeable.empty:
            print(f"{horizon:12} no BUY/SELL rows")
            continue

        t1_rate = tradeable["outcome"].isin(["T1_HIT", "T2_HIT"]).mean()
        sl_rate = (tradeable["outcome"] == "SL_HIT").mean()
        expired_rate = tradeable["outcome"].isin(["TIME_EXIT", "NO_OUTCOME", "OPEN", "EXPIRED"]).mean()
        avg_rr = tradeable["rrRatio"].mean()

        print(
            f"{horizon:12} n={len(tradeable):4d} "
            f"T1%={t1_rate:7.1%} [{_health_label(t1_rate, 0.40, 0.60)}] "
            f"SL%={sl_rate:7.1%} [{_health_label(sl_rate, 0.20, 0.35)}] "
            f"Expired%={expired_rate:7.1%} [{_health_label(expired_rate, 0.15, 0.30)}] "
            f"avg_rr={avg_rr:5.2f}"
        )

    _print_section("CONFIDENCE CALIBRATION")
    merged = pd.merge(
        outcomes[outcomes["signal"].isin(["BUY", "SELL"])],
        confidence[["symbol", "horizon", "date", "conf_clamped"]],
        on=["symbol", "horizon", "date"],
        how="left",
    )
    if merged.empty:
        print("No BUY/SELL rows available for confidence calibration.")
    else:
        merged["won"] = merged["outcome"].isin(["T1_HIT", "T2_HIT"])
        merged["conf_bucket"] = pd.cut(
            merged["conf_clamped"],
            bins=[0, 55, 65, 75, 85, 100],
            labels=["<55", "55-65", "65-75", "75-85", ">85"],
            include_lowest=True,
        )
        calibration = merged.groupby("conf_bucket", observed=False)["won"].agg(["mean", "count"])
        calibration.columns = ["win_rate", "count"]
        print(calibration.to_string())

        low_bucket = calibration.loc["<55", "win_rate"] if "<55" in calibration.index else None
        high_bucket = calibration.loc[">85", "win_rate"] if ">85" in calibration.index else None
        if pd.notna(low_bucket) and pd.notna(high_bucket):
            delta = high_bucket - low_bucket
            print(f"\nHigh-minus-low confidence win-rate delta: {delta:.1%}")
            if delta < 0.15:
                print("Investigate: confidence is weakly calibrated or flat.")
            elif delta > 0.25:
                print("Strong separation: confidence is predictive.")
            else:
                print("Healthy: confidence is meaningfully predictive.")

    _print_section("SIGNAL DISTRIBUTION")
    dist = signals["signal"].value_counts(dropna=False)
    print(dist.to_string())
    buy_sell_rate = (dist.get("BUY", 0) + dist.get("SELL", 0)) / max(len(signals), 1)
    print(f"\nBUY+SELL rate: {buy_sell_rate:.1%}")
    print(f"Assessment: {_health_label(buy_sell_rate, 0.05, 0.15)}")

    _print_section("TOP GATE FAILURES (BLOCKED SIGNALS)")
    blocked = signals[signals["signal"] == "BLOCKED"]
    all_failed = []
    for gates in blocked["struct_gates_failed"].dropna():
        if str(gates).strip():
            all_failed.extend([g for g in str(gates).split("|") if g])
    for gate, count in Counter(all_failed).most_common(10):
        print(f"{gate:30} {count}")
    if not all_failed:
        print("No structural gate failures recorded on BLOCKED rows.")

    _print_section("TOP BUY/SELL -> WATCH/BLOCKED TRANSITIONS")
    if transitions.empty:
        print("No transitions logged.")
    else:
        interesting = transitions[
            transitions["from_signal"].isin(["BUY", "SELL"])
            & transitions["to_signal"].isin(["WATCH", "BLOCKED"])
        ]
        if interesting.empty:
            print("No BUY/SELL downgrades found.")
        else:
            top = interesting["trigger_reason"].fillna("unknown").value_counts().head(10)
            print(top.to_string())

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze historical backtest result CSVs")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    try:
        return analyze(results_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        print("No historical backtest outputs found yet. Run backtest_historical.py before Part 3 analysis.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
