import sys
import os
import logging

# Add project root to sys.path
sys.path.append(os.getcwd())

# whitelisted 'verify_invariants' in master_config.py allows this
from services.trade_enhancer import validate_mathematical_invariants
from services.signal_engine import finalize_trade_decision

# Setup minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RobustnessTest")

def test_invariant_logic():
    logger.info("--- Testing Invariant Logic ---")
    
    # CASE 1: Long with SL > Entry
    exec_ctx = {"risk": {"direction": "LONG", "entry_price": 100, "stop_loss": 110, "targets": [120]}}
    res = validate_mathematical_invariants(exec_ctx)
    logger.info(f"Invalid Long (SL > Entry): {res}")
    assert res["valid"] is False
    assert "Entry" in res["reason"] and "SL" in res["reason"]

    # CASE 2: Short with SL < Entry
    exec_ctx = {"risk": {"direction": "SHORT", "entry_price": 100, "stop_loss": 90, "targets": [80]}}
    res = validate_mathematical_invariants(exec_ctx)
    logger.info(f"Invalid Short (SL < Entry): {res}")
    assert res["valid"] is False

    # CASE 3: Valid Long
    exec_ctx = {"risk": {"direction": "LONG", "entry_price": 100, "stop_loss": 90, "targets": [110, 120]}}
    res = validate_mathematical_invariants(exec_ctx)
    logger.info(f"Valid Long: {res}")
    assert res["valid"] is True

def test_signal_override():
    logger.info("\n--- Testing Signal Override ---")
    
    # Mock context with invariant failure
    exec_ctx = {
        "invariant_failure": True,
        "can_execute": {"failures": ["Mathematical Invariant Failure: Entry <= SL"]}
    }
    eval_ctx = {"setup": {"type": "GENERIC"}}
    plan = {"symbol": "TEST", "final_confidence": 85, "metadata": {"direction": "LONG"}}
    
    finalize_trade_decision(plan, eval_ctx, exec_ctx)
    
    logger.info(f"Signal with Invariant Failure: {plan.get('signal')}")
    logger.info(f"Status: {plan.get('status')}")
    
    assert plan["signal"] == "INVALID_GEOMETRY"
    assert plan["trade_signal"] == "ERROR"
    assert "Signal blocked by robustness invariants" in plan["note"]

if __name__ == "__main__":
    try:
        test_invariant_logic()
        test_signal_override()
        logger.info("\n✅ ALL ROBUSTNESS TESTS PASSED!")
    except Exception as e:
        logger.error(f"Tests failed: {e}")
        sys.exit(1)
