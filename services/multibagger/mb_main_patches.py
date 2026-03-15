# mb_main_patches.py
"""
Required patches to existing files for MB module integration.
Apply these BEFORE registering mb_router or starting the scheduler.

This file is documentation only — copy each patch to the target file manually.
Nothing here is imported at runtime.
"""

# =============================================================================
# 1. services/db.py — Register MultibaggerCandidate table
# =============================================================================
# Add this import near the other model imports, BEFORE init_db() is defined:
#
#   from services.multibagger.mb_db_model import MultibaggerCandidate  # noqa: F401
#
# SQLAlchemy's Base.metadata.create_all() will auto-create the
# multibagger_candidates table the next time init_db() runs on startup.
# No migration script needed — SQLite creates the table if absent.


# =============================================================================
# 2. main.py — Drop multibagger from full-mode horizons_to_compute
# =============================================================================
# Find (around line 714):
#     horizons_to_compute = ["intraday", "short_term", "long_term", "multibagger"]
#
# Replace with:
#     horizons_to_compute = ["intraday", "short_term", "long_term"]
#
# The MB module runs independently via mb_scheduler. The main API never
# needs to score multibagger in response to a user request.


# =============================================================================
# 3. main.py — Fix meta_scores value profile horizon
# =============================================================================
# In the meta_scores block (around line 855), only score_value_profile
# still uses horizon="multibagger". score_growth_profile and
# score_quality_profile already use horizon="long_term" correctly.
#
# Find:
#     "value": score_value_profile(
#         analysis_data["fundamentals"],
#         horizon="multibagger"    ← WRONG
#     ),
#
# Replace with:
#     "value": score_value_profile(
#         analysis_data["fundamentals"],
#         horizon="long_term"      ← CORRECT
#     ),


# =============================================================================
# 4. main.py — Remove multibagger from macro trend fallback chain
# =============================================================================
# Find (around line 787):
#     for fallback_horizon in ["short_term", "intraday", "long_term", "multibagger"]:
#
# Replace with:
#     for fallback_horizon in ["short_term", "intraday", "long_term"]:
#
# multibagger indicators won't exist in raw_indicators_by_horizon once
# the full-mode patch (#2) is applied. Leaving it in is harmless but noisy.


# =============================================================================
# 5. main.py — Preserve multi_score across writes (_save_analysis_to_db)
# =============================================================================
# The MB scheduler writes multi_score via the MultibaggerCandidate table,
# not via signal_cache. The main pipeline writes multi_score=None whenever
# it runs a 3-horizon full-mode analysis. To preserve the MB score across
# re-analyses, read the existing value before overwriting.
#
# In _save_analysis_to_db, add BEFORE the horizon_data dict is built:
#
#   existing_entry = db.query(SignalCache).filter(SignalCache.symbol == symbol).first()
#   existing_multi = (
#       (existing_entry.horizon_scores or {}).get("multi_score")
#       if existing_entry else None
#   )
#
# Then in horizon_data:
#   "multi_score": value.get("multi_score") or existing_multi,
#
# This is a read-before-write within the same DB session, so no race
# condition — CACHE_LOCK already holds at this point.


# =============================================================================
# 6. signal_engine.py — 3-horizon winner-takes-all selection
# =============================================================================
# The best_horizon selection currently picks from ALL successful profiles,
# which would include multibagger if it ever appears. Add a trading-horizons
# filter so the winner-takes-all is always from the 3 active trading horizons.
#
# Find the best horizon selection block (around line 610):
#     successful_profiles = {
#         h: p for h, p in profiles.items() if p.get("status") == "SUCCESS"
#     }
#     if successful_profiles:
#         best_horizon = max(
#             successful_profiles,
#             key=lambda h: successful_profiles[h]["final_decision_score"],
#         )
#         best_score = successful_profiles[best_horizon]["final_decision_score"]
#     else:
#         best_horizon = None
#         best_score = 0
#
# Replace with:
#
#   TRADING_HORIZONS = {"intraday", "short_term", "long_term"}
#
#   successful_profiles = {
#       h: p for h, p in profiles.items() if p.get("status") == "SUCCESS"
#   }
#   trading_profiles = {
#       h: p for h, p in successful_profiles.items() if h in TRADING_HORIZONS
#   }
#
#   if trading_profiles:
#       best_horizon = max(
#           trading_profiles,
#           key=lambda h: trading_profiles[h]["final_decision_score"],
#       )
#       best_score = trading_profiles[best_horizon]["final_decision_score"]
#   elif successful_profiles:
#       # Fallback: single-mode call with horizon="multibagger" via manual UI override
#       best_horizon = max(
#           successful_profiles,
#           key=lambda h: successful_profiles[h]["final_decision_score"],
#       )
#       best_score = successful_profiles[best_horizon]["final_decision_score"]
#   else:
#       best_horizon = None
#       best_score   = 0


# =============================================================================
# 7. main.py — Register mb_router and start scheduler in FastAPI lifespan
# =============================================================================
# Add these imports near the top of main.py:
#
#   from services.multibagger.mb_routes import mb_router
#   from services.multibagger.mb_scheduler import start_mb_scheduler
#
# Register the router (alongside existing routers):
#   app.include_router(mb_router)
#
# In the FastAPI lifespan startup block (where init_db() is called):
#   start_mb_scheduler()


# =============================================================================
# 8. config/constants.py — Add NSE_UNIVERSE_CSV path
# =============================================================================
# Add:
#   NSE_UNIVERSE_CSV = "data/nifty500.csv"
#
# Create data/nifty500.csv with a "symbol" column containing NSE tickers:
#   symbol
#   RELIANCE.NS
#   TCS.NS
#   HDFCBANK.NS
#   ...
