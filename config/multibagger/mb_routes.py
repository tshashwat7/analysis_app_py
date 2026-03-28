# config/multibagger/mb_routes.py
"""
Multibagger API Routes
========================
FastAPI router for the MB module. Reads exclusively from
MultibaggerCandidate table — never touches signal_cache.

INTEGRATION in main.py:
    from config.multibagger.mb_routes import mb_router
    app.include_router(mb_router)

ENDPOINTS:
    GET  /multibagger/candidates          — All candidates with optional tier filter
    GET  /multibagger/candidates/{symbol} — Single candidate detail
    POST /multibagger/run                 — Trigger an immediate manual cycle (admin)
    GET  /multibagger/status              — Last cycle stats
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from config.config_utility.market_utils import get_current_utc
from services.db import get_db
from config.multibagger.mb_db_model import MultibaggerCandidate
# cycle_status is owned by mb_scheduler (the authoritative writer).
# Imported here so /status reads live values without a circular import.
from config.multibagger.mb_scheduler import cycle_status
from services.auth_utils import get_api_key

logger    = logging.getLogger(__name__)
mb_router = APIRouter(prefix="/multibagger", tags=["Multibagger"])


# =============================================================================
# SERIALISER
# =============================================================================

def _row_to_dict(row: MultibaggerCandidate) -> dict:
    return {
        "symbol":               row.symbol,
        "conviction_tier":      row.conviction_tier,
        "fundamental_score":    row.fundamental_score,
        "technical_score":      row.technical_score,
        "hybrid_score":         row.hybrid_score,
        "final_score":          row.final_score,
        "final_decision_score": row.final_decision_score,
        "confidence":           row.confidence,
        "primary_setup":        row.primary_setup,
        "primary_strategy":     row.primary_strategy,
        "estimated_hold_months": row.estimated_hold_months,
        "gatekeeper_passed":    row.gatekeeper_passed,
        "rejection_reason":     row.rejection_reason,
        "last_evaluated":       row.last_evaluated.isoformat() if row.last_evaluated else None,
        "re_evaluate_date":     row.re_evaluate_date.isoformat() if row.re_evaluate_date else None,
        "prev_conviction_tier": row.prev_conviction_tier,
        "entry_trigger":        row.entry_trigger,
        "tier_changed_at":      row.tier_changed_at.isoformat() if row.tier_changed_at else None,
    }


# =============================================================================
# ENDPOINTS
# =============================================================================

@mb_router.get("/candidates", response_model=List[dict])
def get_candidates(
    tier:    Optional[str] = Query(None, description="Filter by conviction tier: HIGH | MEDIUM | LOW"),
    passed:  Optional[bool] = Query(True,  description="Only show gatekeeper-passed stocks"),
    limit:   int            = Query(100,  ge=1, le=500),
    db:      Session        = Depends(get_db),
    api_key: str            = Depends(get_api_key),
):
    """
    Return multibagger candidates, sorted by final_decision_score descending.
    """
    # ✅ P2-4 FIX: Filter by staleness (35 days)
    # This prevents the dashboard from showing stale results if the weekly run fails
    cutoff = datetime.now(timezone.utc) - timedelta(days=35)
    q = db.query(MultibaggerCandidate).filter(MultibaggerCandidate.last_evaluated >= cutoff)

    if passed:
        q = q.filter(MultibaggerCandidate.gatekeeper_passed == True)   # noqa: E712
    if tier:
        q = q.filter(MultibaggerCandidate.conviction_tier == tier.upper())

    rows = (
        q.order_by(MultibaggerCandidate.final_decision_score.desc().nullslast())
         .limit(limit)
         .all()
    )
    return [_row_to_dict(r) for r in rows]


@mb_router.get("/candidates/{symbol}", response_model=dict)
def get_candidate(symbol: str, db: Session = Depends(get_db), api_key: str = Depends(get_api_key)):
    """Return full detail for a single symbol including thesis_json."""
    row = db.query(MultibaggerCandidate).filter_by(symbol=symbol).first()
    if not row:
        raise HTTPException(status_code=404, detail=f"{symbol} not found in multibagger_candidates")

    d = _row_to_dict(row)
    d["thesis_json"] = row.thesis_json
    return d


@mb_router.get("/status", response_model=dict)
def get_status(api_key: str = Depends(get_api_key)):
    """Return last cycle metadata and next scheduled run time."""
    from config.multibagger.mb_scheduler import _next_sunday_ist, _now_ist
    return {
        **cycle_status,
        "next_scheduled_run": _next_sunday_ist().isoformat(),
        "server_time_ist":    _now_ist().isoformat(),
    }


@mb_router.post("/run", response_model=dict)
def trigger_manual_run(api_key: str = Depends(get_api_key)):
    """
    Trigger an immediate MB cycle in the background.
    Intended for admin/testing use only — not to be called during market hours.
    """
    import threading
    from config.multibagger.mb_scheduler import run_mb_cycle

    def _background():
        try:
            run_mb_cycle()  # run_mb_cycle updates cycle_status directly
        except Exception as e:
            cycle_status["last_run_result"] = f"ERROR:{e}"
            logger.error(f"[MB manual run] Failed: {e}", exc_info=True)

    t = threading.Thread(target=_background, name="mb-manual-run", daemon=True)
    t.start()

    return {"message": "MB cycle triggered in background", "started_at": get_current_utc().isoformat()}
