# -*- coding: utf-8 -*-
"""
Dry Run Robustness Audit v2
- Uses proper resolver factory (config_helpers.get_resolver)
- Handles Unicode encoding
"""
import sys, os, traceback
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
sys.path.insert(0, r"d:\stockviedeo\stock-analyzer-app")
os.chdir(r"d:\stockviedeo\stock-analyzer-app")

# Suppress verbose logging during dry run
import logging
logging.basicConfig(level=logging.WARNING)

def sep(title):
    print(f"\n{'='*70}\n{title}\n{'='*70}")

# Step 1: Config Import
sep("STEP 1: Config Import Validation")
try:
    from config.master_config import MASTER_CONFIG
    from config.setup_pattern_matrix_config import SETUP_PATTERN_MATRIX
    from config.strategy_matrix_config import STRATEGY_MATRIX
    print(f"  Horizons: {list(MASTER_CONFIG.get('horizons', {}).keys())}")
    print(f"  Setups: {len(SETUP_PATTERN_MATRIX)}")
    print(f"  Strategies: {len(STRATEGY_MATRIX)}")
    print("  [PASS] All configs imported")
except Exception as e:
    print(f"  [FAIL] Config import: {e}")
    traceback.print_exc()
    sys.exit(1)

# Step 2: Resolver init via proper factory
sep("STEP 2: Resolver + Extractor Initialization (via config_helpers)")
try:
    from config.config_helpers import get_resolver, clear_resolver_cache
    clear_resolver_cache()  # Fresh start

    for h in ["intraday", "short_term", "long_term", "multibagger"]:
        resolver = get_resolver(h, use_cache=False)
        state = resolver.extractor.validate_extractor_state()
        status = "PASS" if state["valid"] else "FAIL"
        conf_loaded = state.get("has_confidence_config", "?")
        warnings_list = state.get("warnings", [])
        errors_list = state.get("errors", [])
        print(f"  [{status}] {h}: conf_loaded={conf_loaded}, warnings={len(warnings_list)}, errors={len(errors_list)}")
        for e_ in errors_list:
            print(f"       ERROR: {e_}")
        for w_ in warnings_list:
            print(f"       WARN: {w_}")
except Exception as e:
    print(f"  [FAIL] Resolver init: {e}")
    traceback.print_exc()

# Step 3: Full analysis on RELIANCE
sep("STEP 3: Full Analysis - RELIANCE")
try:
    from main import run_analysis
    result = run_analysis("RELIANCE", "nifty50", mode="full")

    error = result.get("error")
    if error:
        print(f"  [FAIL] Analysis error: {error}")
    else:
        full_report = result.get("full_report", {})
        profiles = full_report.get("profiles", {})
        best_fit = full_report.get("best_fit", "?")
        best_score = full_report.get("best_score", 0)

        print(f"  Symbol: {result.get('symbol','?')}")
        print(f"  Best Fit: {best_fit} (Score: {best_score})")

        for h, p in profiles.items():
            fs = p.get("final_score", 0)
            cat = p.get("category", "?")
            setup = p.get("setup", {}).get("type", "NONE") if isinstance(p.get("setup"), dict) else "?"
            strategy_info = p.get("strategy", {})
            strategy_name = strategy_info.get("type", strategy_info.get("primary", "NONE")) if isinstance(strategy_info, dict) else "?"
            conf_info = p.get("confidence", {})
            conf_score = conf_info.get("clamped", conf_info.get("score", 0)) if isinstance(conf_info, dict) else conf_info
            print(f"    {h:12s} | score={fs:6.2f} | cat={str(cat):8s} | setup={str(setup):25s} | strategy={str(strategy_name):18s} | conf={conf_score}")

        # Trade Plan
        sep("STEP 4: Trade Plan Robustness")
        trade = result.get("trade_recommendation", {})
        if not trade or trade.get("status") == "ERROR":
            print(f"  [INFO] Trade plan: {trade.get('reason', 'No trade')}")
        else:
            signal = trade.get("trade_signal", trade.get("signal", "?"))
            entry = trade.get("entry", None)
            sl = trade.get("stop_loss", None)
            targets = trade.get("targets", {})
            t1 = targets.get("t1", None) if isinstance(targets, dict) else None
            t2 = targets.get("t2", None) if isinstance(targets, dict) else None
            rr = trade.get("rr_ratio", None)
            conf = trade.get("final_confidence", None)
            hold = trade.get("estimated_hold", trade.get("hold_estimate", None))
            hold_weeks = trade.get("hold_weeks", None)

            print(f"  Signal:     {signal}")
            print(f"  Entry:      {entry}")
            print(f"  Stop Loss:  {sl}")
            print(f"  T1:         {t1}")
            print(f"  T2:         {t2}")
            print(f"  RR Ratio:   {rr}")
            print(f"  Confidence: {conf}")
            print(f"  Hold:       {hold}")
            print(f"  Hold Weeks: {hold_weeks}")

            # Robustness checks
            issues = []
            if isinstance(entry, (int, float)) and entry <= 0:
                issues.append("Entry <= 0")
            if signal and "BUY" in str(signal).upper():
                if isinstance(entry, (int,float)) and isinstance(sl, (int,float)):
                    if sl >= entry:
                        issues.append(f"SL ({sl}) >= Entry ({entry}) for BUY")
                    sl_pct = abs(entry - sl) / entry * 100
                    if sl_pct > 15:
                        issues.append(f"SL distance too large: {sl_pct:.1f}%")
                    if sl_pct < 0.3:
                        issues.append(f"SL distance tiny: {sl_pct:.2f}%")
                if isinstance(entry, (int,float)) and isinstance(t1, (int,float)) and t1 <= entry:
                    issues.append(f"T1 ({t1}) <= Entry ({entry})")
                if isinstance(t1, (int,float)) and isinstance(t2, (int,float)) and t2 < t1:
                    issues.append(f"T2 ({t2}) < T1 ({t1})")
            if isinstance(rr, (int,float)) and (rr <= 0 or rr > 20):
                issues.append(f"RR out of range: {rr}")
            if isinstance(conf, (int,float)) and (conf < 0 or conf > 100):
                issues.append(f"Confidence out of [0,100]: {conf}")
            if isinstance(hold_weeks, (int,float)) and hold_weeks > 500:
                issues.append(f"Hold weeks unreasonable: {hold_weeks}")
            if t1 is None:
                issues.append("T1 is None")
            if t2 is None:
                issues.append("T2 is None")
            if sl is None:
                issues.append("SL is None")

            if issues:
                for i in issues:
                    print(f"  [ISSUE] {i}")
            else:
                print("  [PASS] All robustness checks passed")

        # Dump full trade recommendation keys for debugging
        sep("STEP 5: Trade Recommendation Structure")
        if trade:
            def print_keys(d, prefix="  "):
                for k, v in d.items():
                    if isinstance(v, dict) and len(v) < 10:
                        print(f"{prefix}{k}: {{")
                        print_keys(v, prefix + "  ")
                        print(f"{prefix}}}")
                    elif isinstance(v, list) and len(v) < 5:
                        print(f"{prefix}{k}: {v}")
                    else:
                        val_str = str(v)[:80]
                        print(f"{prefix}{k}: {val_str}")
            print_keys(trade)

except Exception as e:
    print(f"  [FAIL] Analysis failed: {e}")
    traceback.print_exc()

sep("AUDIT COMPLETE")
