# config/multibagger/mb_db_model.py
"""
MultibaggerCandidate DB Model
==============================
Isolated SQLAlchemy table for MB screener and evaluator results.

INTEGRATION:
    In services/db.py, add before init_db():
        from config.multibagger.mb_db_model import MultibaggerCandidate  # noqa: F401
    SQLAlchemy's metadata.create_all() will auto-create the table on startup.

DESIGN:
    - Never written to by the main trading pipeline.
    - Written exclusively by multibagger_evaluator.run_mb_phase2().
    - Read by mb_routes.py endpoints.
    - symbol is primary key — one row per stock, upserted on each weekly cycle.
"""
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, Boolean, DateTime, JSON, event

from services.db import Base


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MultibaggerCandidate(Base):
    __tablename__ = "multibagger_candidates"

    # --- Identity -------------------------------------------------------
    symbol              = Column(String,  primary_key=True, index=True)
    conviction_tier     = Column(String,  nullable=True)   # "HIGH" | "MEDIUM" | "LOW"

    # --- Phase 2 Resolver Scores ----------------------------------------
    fundamental_score   = Column(Float,   nullable=True)
    technical_score     = Column(Float,   nullable=True)
    hybrid_score        = Column(Float,   nullable=True)
    final_score         = Column(Float,   nullable=True)   # Structural eligibility
    final_decision_score = Column(Float,  nullable=True)   # After opportunity scoring
    confidence          = Column(Float,   nullable=True)   # Clamped [50, 95]

    # --- Setup / Strategy -----------------------------------------------
    primary_setup       = Column(String,  nullable=True)   # e.g. "QUALITY_ACCUMULATION"
    primary_strategy    = Column(String,  nullable=True)   # e.g. "quality_compounder"
    entry_trigger       = Column(String,  nullable=True)   # Triggering pattern name

    # --- Hold Context ---------------------------------------------------
    estimated_hold_months = Column(Float, nullable=True)   # From strategy config
    thesis_json           = Column(JSON,  nullable=True)   # Rich breakdown dict

    # --- Gatekeeper Tracking --------------------------------------------
    gatekeeper_passed    = Column(Boolean,  default=False)
    gatekeeper_passed_at = Column(DateTime, nullable=True)
    rejection_reason     = Column(String,   nullable=True)  # Phase 1 fail reason

    # --- Cadence --------------------------------------------------------
    last_evaluated       = Column(DateTime, default=_utc_now, onupdate=_utc_now, index=True)
    re_evaluate_date     = Column(DateTime, nullable=True)
    prev_conviction_tier = Column(String,   nullable=True)
    tier_changed_at      = Column(DateTime, nullable=True)


_DATETIME_FIELDS = (
    "last_evaluated",
    "gatekeeper_passed_at",
    "re_evaluate_date",
    "tier_changed_at",
)


from sqlalchemy import event

@event.listens_for(MultibaggerCandidate, 'load')
def enforce_tz_mb_candidate(target, context):
    """Ensure all datetime fields carry UTC timezone after loading from SQLite."""
    for field in ('last_evaluated', 'gatekeeper_passed_at',
                  're_evaluate_date', 'tier_changed_at'):
        val = getattr(target, field, None)
        if val is not None and val.tzinfo is None:
            setattr(target, field, val.replace(tzinfo=timezone.utc))
