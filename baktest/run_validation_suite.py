import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


ROOT = Path(__file__).parent.resolve()
RESULTS_DIR = ROOT / "output" / "backtest_results"


def _cmd(*parts: str) -> List[str]:
    return [sys.executable, *parts]


def _run_step(name: str, command: List[str]) -> Tuple[bool, float]:
    print("\n" + "=" * 84)
    print(name)
    print("=" * 84)
    print(" ".join(command))
    start = time.time()
    completed = subprocess.run(command, cwd=ROOT)
    elapsed = time.time() - start
    ok = completed.returncode == 0
    print(f"\n{name}: {'PASS' if ok else 'FAIL'} ({elapsed:.1f}s)")
    return ok, elapsed


def _has_backtest_results(results_dir: Path) -> bool:
    required = [
        results_dir / "forward_outcomes.csv",
        results_dir / "signal_log.csv",
        results_dir / "confidence_audit.csv",
        results_dir / "transitions_log.csv",
    ]
    return all(path.exists() for path in required)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the repo validation stack in order.")
    parser.add_argument("--symbol", default="RELIANCE.NS")
    parser.add_argument("--horizon", default="short_term")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default="quick",
        help="quick = single-symbol + gates + synthetic + analysis, full adds historical backtest",
    )
    parser.add_argument(
        "--historical-args",
        default="--quick --horizons short_term",
        help="Extra args forwarded to backtest_historical.py in full mode",
    )
    args = parser.parse_args()

    steps: List[Tuple[str, List[str]]] = [
        (
            "Part 1+2: Symbol Validation (Math + Gate Audit)",
            _cmd("validate_symbol.py", "--symbol", args.symbol, "--horizon", args.horizon),
        ),
        (
            "Synthetic Regression Suite",
            _cmd("backtest_synthetic.py", "--suite", "synthetic", "--verbose"),
        ),
    ]

    if args.mode == "full":
        steps.append(
            (
                "Historical Backtest",
                _cmd("backtest_historical.py", *args.historical_args.split()),
            )
        )

    failed: List[str] = []
    total_elapsed = 0.0
    historical_attempted = False

    for name, command in steps:
        ok, elapsed = _run_step(name, command)
        total_elapsed += elapsed
        if name == "Historical Backtest":
            historical_attempted = True
        if not ok:
            failed.append(name)
            # Stop early on diagnostic failures in the first two parts.
            if name.startswith("Part 1") or name.startswith("Part 2"):
                break

    should_run_part3 = False
    if args.mode == "full":
        should_run_part3 = historical_attempted and _has_backtest_results(RESULTS_DIR)
    else:
        should_run_part3 = _has_backtest_results(RESULTS_DIR)

    if should_run_part3:
        ok, elapsed = _run_step(
            "Part 3: Statistical Result Analysis",
            _cmd("analyze_backtest_results.py"),
        )
        total_elapsed += elapsed
        if not ok:
            failed.append("Part 3: Statistical Result Analysis")

    print("\n" + "=" * 84)
    print("Validation Summary")
    print("=" * 84)
    print(f"Mode: {args.mode}")
    print(f"Symbol: {args.symbol}")
    print(f"Horizon: {args.horizon}")
    print(f"Elapsed: {total_elapsed:.1f}s")
    if args.mode == "quick" and not _has_backtest_results(RESULTS_DIR):
        print("Part 3: skipped (no backtest_results CSVs yet; run --mode full or historical first)")
    elif args.mode == "full" and historical_attempted and not _has_backtest_results(RESULTS_DIR):
        print("Part 3: skipped (historical backtest produced no CSV outputs; run --fetch first)")

    if failed:
        print("Failed steps:")
        for step in failed:
            print(f"  - {step}")
        return 1

    print("All validation steps passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
