# 📈 Pro Stock Analyzer v15.0

> **Institutional-Grade Algorithmic Trading Signal Engine for NSE (India)**

![Version](https://img.shields.io/badge/version-15.1-blue)
![Market](https://img.shields.io/badge/market-NSE%20India-orange)
![Stack](https://img.shields.io/badge/stack-Python%20%7C%20FastAPI%20%7C%20SQLite-green)
![Status](https://img.shields.io/badge/status-Production-brightgreen)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Data Architecture — Phase 0](#data-architecture--phase-0)
   - [3-Tier OHLCV Cache](#3-tier-ohlcv-cache)
   - [Fundamental Cache](#fundamental-cache)
   - [Indicator Polymorphism](#indicator-polymorphism)
   - [Corporate Actions](#corporate-actions)
4. [8-Phase Pipeline](#8-phase-pipeline)
   - [Phase 1 — Config Layer](#phase-1--config-layer)
   - [Phase 2 — Extraction Layer](#phase-2--extraction-layer)
   - [Phase 3 — Signal Engine](#phase-3--signal-engine)
   - [Phase 4 — Config Resolver](#phase-4--config-resolver)
   - [Phase 5 — Trade Enhancer](#phase-5--trade-enhancer)
   - [Phase 6 — Pattern Library](#phase-6--pattern-library)
   - [Phase 7 — Multibagger Pipeline](#phase-7--multibagger-pipeline)
   - [Phase 8 — Orchestrator & DB](#phase-8--orchestrator--db)
5. [Multi-Layer Gating System](#multi-layer-gating-system)
6. [Config Architecture](#config-architecture)
7. [Pattern Library](#pattern-library-1)
8. [Database Schema](#database-schema)
9. [Signal Semantics](#signal-semantics)
10. [Pattern State Lifecycle](#pattern-state-lifecycle)
11. [Key Architecture Decisions](#key-architecture-decisions)
12. [API Endpoints](#api-endpoints)
13. [Presentation Layer & UI](#presentation-layer--ui)
14. [Setup & Installation](#setup--installation)
15. [Configuration Guide](#configuration-guide)
16. [Known Limitations](#known-limitations)
17. [Contributing / Development Notes](#contributing--development-notes)

---

## Overview

Pro Stock Analyzer v15.0 is an institutional-grade algorithmic trading signal engine targeting NSE-listed equities across four time horizons: `intraday`, `short_term`, `long_term`, and `multibagger`. The system ingests raw OHLCV data, evaluates multi-layered structural gate conditions, detects chart patterns with full lifecycle tracking, computes confidence-adjusted trade plans, and persists results with retry-safe database writes.

The core philosophy distinguishes stock *quality* (structural eligibility — "is this stock worth watching?") from trade *timing* (execution context — "can I enter right now?"). These are computed in two explicit, decoupled phases: a `build_evaluation_context_only` pass that produces indicators, scores, gates, confidence, and setup classification; followed by a `build_execution_context_from_evaluation` pass that layers position sizing, order model, and real-time RR validation on top of the already-computed evaluation. A weekly multibagger pipeline runs independently with its own isolated extractor stack, scoring weights, and conviction-tier output.

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│  DATA INGESTION & CACHING (Phase 0)                          │
│  data_fetch.py ──► cache.py (RAM → Parquet → Yahoo Finance)  │
│  fundamentals.py (24h SQLite cache)                          │
│  corporate_actions.py + daily_corp_action_warmer.py          │
│  indicators.py (Polymorphic: EMA/ATR shift by horizon)       │
└──────────────────────┬───────────────────────────────────────┘
                       │ adjusted OHLCV + indicators
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  RAW CONFIG FILES (6 Files — config/)                        │
│  master_config.py  setup_pattern_matrix_config.py            │
│  confidence_config.py  strategy_matrix_config.py             │
│  technical_score_config.py  fundamental_score_config.py      │
└──────────────────────┬───────────────────────────────────────┘
                       │ extracts & merges
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  EXTRACTION LAYER (Phase 2)                                  │
│  config_extractor.py ──► query_optimized_extractor.py        │
│                      └── gate_evaluator.py (stateless)       │
└──────────────────────┬───────────────────────────────────────┘
                       │ typed query API
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  CONFIG BRIDGE (Phase 2.5)                                   │
│  config_helpers.py                                           │
│  [Resolver factory · Context builders · Context accessors]   │
└──────────────────────┬───────────────────────────────────────┘
                       │ build_evaluation_context /
                       │ build_execution_context
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  BRAIN LAYER (Phase 3 + 4)                                   │
│  signal_engine.py ──► config_resolver.py (74 methods)        │
│  [Scoring · Setup Classification · Gates · Confidence]       │
└──────────────────────┬───────────────────────────────────────┘
                       │ eval_ctx + exec_ctx
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  EXECUTION LAYER (Phase 5)                                   │
│  trade_enhancer.py                                           │
│  [Pattern expiry · Invalidation · Market-adaptive RR]        │
└──────────────────────┬───────────────────────────────────────┘
                       │ enhanced execution context
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR + PERSISTENCE (Phase 8)                        │
│  main.py ──► db.py                                           │
│  [FastAPI · ProcessPool · ThreadPool · SQLite · Retry]       │
└──────────────────────────────────────────────────────────────┘

       ┌──────────────────────────────────┐
       │  WEEKLY PIPELINE (Phase 7)       │
       │  mb_scheduler ──► screener       │
       │               └──► evaluator     │
       │  [Separate extractor stack]      │
       └──────────────────────────────────┘

       ┌──────────────────────────────────┐
       │  PATTERN LIBRARY (Phase 6)       │
       │  12 detectors — all return       │
       │  {found, score, quality, meta}   │
       │  pattern_analyzer ──► fusion     │
       └──────────────────────────────────┘
```

---

## Data Architecture — Phase 0

Before a single config gate is evaluated, the system must produce clean, adjusted, horizon-aware OHLCV data and a rich indicator set. Phase 0 is the nervous system that makes this happen silently and efficiently for every symbol across every pipeline invocation.

---

### 3-Tier OHLCV Cache

**Files:** `services/data_fetch.py`, `services/cache.py`

`data_fetch.py` is the single entry point for all raw OHLCV ingestion. It never hits the network if the data is already resident. The lookup follows a strict three-tier fallback:

```
Tier 1 — RAM Cache (in-process dict, sub-millisecond)
    │  hit → return immediately
    │  miss ↓
Tier 2 — Parquet / Disk Cache (local filesystem, ~5–20 ms)
    │  hit → load DataFrame, promote to RAM, return
    │  miss ↓
Tier 3 — Yahoo Finance (network, ~300–800 ms)
         → download → write Parquet → promote to RAM → return
```

`cache.py` manages the in-process RAM store and Parquet persistence layer. It handles cache invalidation (staleness by candle count / TTL), Parquet read/write with schema enforcement, and thread-safe promotion from disk to RAM after a Tier-3 fetch.

**Why this matters:** A full market scan across 200+ symbols processes each symbol in a `ProcessPoolExecutor` worker. Without a Parquet tier, every cold start would fan out 200+ simultaneous Yahoo Finance requests, overwhelming rate limits and inflating scan time by an order of magnitude. With warm Parquet cache, a full scan runs entirely from disk.

---

### Fundamental Cache

**File:** `services/fundamentals.py`

Fundamental data (P/E, P/B, debt-to-equity, promoter holding, etc.) is fetched from Yahoo Finance and Moneycontrol and cached in the SQLite database with a **24-hour TTL**.

```
Request fundamental data for symbol
    │
    ├── Cache row exists AND age < 24h?
    │       └── YES → return cached dict (zero network calls)
    │
    └── NO → fetch from Yahoo Finance / Moneycontrol
              → upsert row in SQLite with current timestamp
              → return fresh dict
```

**Why this matters:** Both the main analysis pipeline and the weekly Multibagger pipeline can evaluate the same stock independently (e.g., RELIANCE appears in both a short-term scan and a multibagger screen within the same day). Without the 24h SQLite cache, each pipeline invocation would issue independent network requests. The cache ensures that whichever pipeline runs first pays the network cost; all subsequent pipelines within the same day read from SQLite at negligible latency.

---

### Indicator Polymorphism

**File:** `services/indicators.py`

`indicators.py` transforms raw OHLCV DataFrames into the indicator dict consumed by `signal_engine.py` and all pattern detectors. The key feature is **polymorphic lookback windows**: the same mathematical indicator is computed with horizon-appropriate parameters rather than fixed constants.

| Indicator | `intraday` | `short_term` | `long_term` |
|---|---|---|---|
| EMA (fast) | 9 | 20 | 50 |
| EMA (slow) | 21 | 50 | 200 |
| MMA (trend) | 14 | 30 | 65 |
| WMA (signal) | 10 | 25 | 55 |
| ATR period | 7 | 14 | 21 |

The horizon is passed into `indicators.py` at call time; the function selects the appropriate parameter block and returns a single unified indicator dict. Downstream consumers (signal engine, pattern detectors, trade enhancer) are horizon-agnostic — they reference keys like `ema_fast` and `atr` without needing to know which lookback was used.

**Why this matters:** Using intraday ATR(7) for a long-term position sizing calculation would produce stop-loss distances an order of magnitude too tight. Polymorphic indicators ensure that every metric is geometrically meaningful for the time frame being evaluated.

---

### Corporate Actions

**Files:** `services/corporate_actions.py`, `daily_corp_action_warmer.py`

Unadjusted OHLCV data is dangerous for algorithmic analysis. A 2-for-1 split produces a 50% price discontinuity that will fire every momentum and breakout signal incorrectly unless the data is adjusted.

`corporate_actions.py` fetches split and dividend events from NSE/Yahoo Finance and applies backward price adjustment to the cached OHLCV series before it enters the analysis pipeline. The adjustment is idempotent — running it twice produces the same result.

`daily_corp_action_warmer.py` is a scheduled background process that pre-fetches and applies corporate action adjustments for the full watchlist each morning before market open. This ensures that by the time the first intraday scan fires, the entire Parquet cache is already adjusted and stale-data signals are not generated at open.

**Coverage:**
- **Stock splits** — backward price and volume adjustment
- **Bonus issues** — treated equivalently to splits for price series purposes
- **Dividends** — ex-dividend price gap adjustment to prevent false breakdown signals

---

## 8-Phase Pipeline

### Phase 1 — Config Layer

**Files:** 6 horizon-agnostic config files under `config/`

**Inputs:** None (static Python dictionaries)  
**Outputs:** Importable config dicts consumed exclusively by `config_extractor.py`

| File | Key Export | Role |
|---|---|---|
| `setup_pattern_matrix_config.py` | `SETUP_PATTERN_MATRIX`, `PATTERN_METADATA` | All trade setups with gate requirements, pattern affinities, horizon overrides, entry rules, invalidation gates, and `DEFAULT_PHYSICS` per pattern |
| `strategy_matrix_config.py` | `STRATEGY_MATRIX` | Strategy DNA: fit indicators with weights, scoring rules with point bonuses, market cap requirements per bracket, horizon fit multipliers |
| `confidence_config.py` | `CONFIDENCE_CONFIG` | Full confidence pipeline config: base floors, ADX bands, volume modifiers, divergence physics, universal adjustments, horizon-specific conditional penalties/bonuses, clamp ranges, min tradeable threshold |
| `technical_score_config.py` | `METRIC_REGISTRY`, `COMPOSITE_SCORING_CONFIG` | Metric scoring types (passthrough, stepped, linear_range, mapping, crossover), category weights per horizon, composite scoring rules for `trendStrength`/`momentumStrength`/`volatilityQuality` |
| `fundamental_score_config.py` | `HORIZON_FUNDAMENTAL_WEIGHTS`, `METRIC_WEIGHTS` | Category weights per horizon, metric-level weights, penalty rules (operator-based), bonus rules (gate-based), sector-specific exclusions |
| `master_config.py` | `MASTER_CONFIG`, `GATE_METRIC_REGISTRY`, `HORIZON_PILLAR_WEIGHTS` | Global constants, horizon execution rules, risk management, RR regime adjustments, `GATE_METRIC_REGISTRY` with `context_paths` and `optional` flags for each gate metric, `HYBRID_METRIC_REGISTRY`, pillar weights |

**Design Principle:** No horizon-specific logic inside these files. Horizon overrides are expressed as nested dicts (`horizon_overrides.intraday`, `horizon_overrides.short_term`, etc.) that the extraction layer merges at query time. No module outside of `config_extractor.py` may import directly from these files.

---

### Phase 2 — Extraction Layer

**Files:** `config/config_extractor.py`, `config/query_optimized_extractor.py`, `config/gate_evaluator.py`

**Inputs:** Raw config dicts from Phase 1 + `horizon: str`  
**Outputs:** Typed query API consumed by Phase 4 (Config Resolver) and Phase 2.5 (Config Bridge)

#### `config_extractor.py` — `ConfigExtractor`

Pre-extracts all config sections at initialization into typed `ConfigSection` objects with `{data, source, is_valid, error}` metadata. Sections are keyed by name (e.g., `"structural_gates"`, `"horizon_confidence_clamp"`) and cached for the lifetime of the resolver instance.

**Critical extraction methods:**

| Method | Purpose |
|---|---|
| `extract_confidence_sections()` | Loads from `confidence_config.py`; raises `ConfigurationError` for required sections; safe defaults for optional ones |
| `extract_matrix_sections()` | Loads `SETUP_PATTERN_MATRIX` + `PATTERN_METADATA`; also extracts per-setup `context_requirements` and `horizon_overrides` as individual keyed sections |
| `extract_gate_sections()` | Populates `structural_gates`, `horizon_structural_gates`, `execution_rules`, `opportunity_gates` — **no** setup-specific gate specs (those live in matrix) |
| `validate_extracted_configs()` | Validates critical sections including confidence config structure; raises `ValueError` / `ConfigurationError` on failure |

**Access pattern:**

```python
from config.config_extractor import ConfigExtractor

extractor = ConfigExtractor(MASTER_CONFIG, "short_term")

# Safe access with default
gates = extractor.get("structural_gates", {})

# Strict access — raises ConfigurationError if missing or invalid
clamp = extractor.get_strict("horizon_confidence_clamp")

# Required access — raises ValueError if missing
setup = extractor.get_required("setup_pattern_matrix")
```

#### `query_optimized_extractor.py` — `QueryOptimizedExtractor`

Wraps `ConfigExtractor` with **85 type-safe query methods** organized in 7 categories. Adds versioned LRU caching for `get_resolved_gates()` and `get_pattern_context()`. All confidence pipeline methods delegate gate evaluation to `gate_evaluator.py`.

**Method categories:**

| Category | Representative Methods |
|---|---|
| Confidence | `get_confidence_clamp()`, `get_setup_baseline_floor()`, `calculate_dynamic_confidence_floor()`, `evaluate_all_confidence_modifiers()`, `calculate_total_confidence_adjustment()` |
| Gates | `get_resolved_gates(phase, setup_type)`, `get_gate_registry()`, `is_gate_enabled()` |
| Patterns | `get_pattern_context(name)` → `PatternContext`, `get_setup_patterns(setup)`, `is_pattern_supported_for_horizon()` |
| Strategy | `get_strategy_fit_indicators()`, `get_strategy_scoring_rules()`, `get_strategy_horizon_multiplier()` |
| Scoring | `get_technical_score()`, `get_fundamental_score()`, `get_hybrid_pillar_composition()` |
| Risk | `get_risk_management_config()`, `get_rr_gates()`, `get_combined_position_sizing_multipliers()` |
| Execution | `get_execution_rules()`, `is_execution_rule_enabled()`, `get_volatility_guards_config()`, `get_time_filters_config()` |

**Gate resolution hierarchy** (inside `get_resolved_gates`):

```
Global gates   (master_config.global.entry_gates)
     ↓ overridden by
Horizon gates  (master_config.horizons.X.entry_gates)
     ↓ overridden by
Setup gates    (setup_pattern_matrix.SETUP.context_requirements
                already merged with horizon_overrides upstream)
```

#### `gate_evaluator.py` — Pure Stateless Gate Engine

A zero-dependency, pure-function module that implements the gate evaluation engine. It is the **single source of truth** for all threshold logic. `QueryOptimizedExtractor` delegates all gate calls here via thin wrapper methods.

**Public API:**

```python
from config.gate_evaluator import evaluate_gates, evaluate_invalidation_gates

# Standard gate check — AND/OR logic, min/max/equals/min_metric/max_metric
passes, failures = evaluate_gates(
    gates={"adx": {"min": 20}, "rvol": {"min": 1.5}, "_logic": "AND"},
    data={"adx": 25, "rvol": 2.1},
    empty_gates_pass=True
)
# → (True, [])

# Invalidation gate check — returns per-metric results with duration field
triggered, gate_results = evaluate_invalidation_gates(
    gates={"price": {"max_metric": "box_low", "multiplier": 0.995, "duration": 2}},
    data={"price": 94.5, "box_low": 100.0}
)
# → (True, [{"metric": "price", "triggered": True, "duration": 2, "reason": "..."}])
```

**Supported threshold clauses per metric:**

| Clause | Behavior |
|---|---|
| `min` | Value must be ≥ threshold |
| `max` | Value must be ≤ threshold |
| `equals` | Value must equal threshold exactly |
| `min_metric` | Value ≥ `data[ref_metric] * multiplier` |
| `max_metric` | Value ≤ `data[ref_metric] * multiplier` |
| `_logic` | `"AND"` (default) or `"OR"` across all metrics |

`None` threshold values are silently skipped (safe config placeholders). Non-numeric values that cannot be compared produce `False` with a descriptive failure message.

---

### Phase 2.5 — Config Bridge Layer

**File:** `config/config_helpers.py` — Business Logic Bridge v3.0

**Inputs:** Raw `indicators`, `fundamentals`, `horizon`, and pre-built `eval_ctx`  
**Outputs:** Populated `eval_ctx` and `exec_ctx` dicts; resolver instance from factory

This module is the **only** entry point through which `signal_engine.py` accesses the entire resolver/extractor stack. It enforces the "build once, access many" contract: context is built exactly once (expensive), then accessed zero-cost through lightweight accessor functions. Direct use of `ConfigResolver` or any extractor method outside of this module is an architecture violation.

#### Resolver Factory

```python
from config.config_helpers import get_resolver, clear_resolver_cache

# Cached per-horizon resolver (30× speedup on repeated calls)
resolver = get_resolver("short_term", use_cache=True)
extractor = resolver.extractor
```

Resolvers are cached in a module-level dict `_resolver_cache` keyed by horizon. Call `clear_resolver_cache()` after config hot-reloads.

#### Context Builders

```python
# BUILD ONCE — expensive (indicator computation, scoring, gate evaluation)
eval_ctx = build_evaluation_context(
    ticker="RELIANCE.NS",
    indicators=clean_indicators,
    fundamentals=clean_fundamentals,
    horizon="short_term",
    patterns=detected_patterns   # optional pre-computed
)

# BUILD ONCE — adds position sizing, risk model, order model
exec_ctx = build_execution_context(
    eval_ctx=eval_ctx,
    capital=100000.0
)
```

Both builders use `flatten_market_data_mixed()` internally to normalize nested indicator dicts `{"value": x, "raw": y, "score": z}` → flat float dicts before passing to the resolver. String values (e.g., `"0%"`) are preserved rather than coerced.

#### Context Accessors (zero-cost data extraction)

```python
setup_type, priority, meta = get_setup_from_context(eval_ctx)
confidence, conf_meta    = get_confidence_from_context(eval_ctx)
gate_result              = check_gates_from_context(eval_ctx, confidence)
strategy_info            = get_strategy_from_context(eval_ctx)
```

`check_gates_from_context` aggregates structural gates, execution rules, opportunity gates, and the confidence floor into a single `{passed, failed_gates, summary}` dict without re-evaluating any thresholds.

---

### Phase 3 — Signal Engine

**File:** `services/signal_engine.py`

**Inputs:** `fundamentals`, `indicators_by_horizon`, `patterns_by_horizon`, optional `requested_horizons`  
**Outputs:** `full_report` dict with per-horizon `profiles` and `best_fit` selection; `trade_plan` dict

The signal engine is the top-level orchestrator. It does **no** config access directly — all config queries flow through `config_helpers.py`.

#### `compute_all_profiles()`

Iterates over the filtered horizon set (full mode: all three trading horizons; single mode: one horizon plus `long_term` for meta-scores). For each horizon:

1. Calls `build_evaluation_context()` to get scoring, setup, confidence, and gate results
2. Calls `calculate_structural_eligibility()` to blend the three pillars: `tech × w_tech + fund × w_fund + hybrid × w_hybrid` with weight redistribution for missing data
3. Calls `compute_opportunity_score()` to add setup priority bonus (0–30%), strategy fit bonus, and confidence conviction bonus — blended as `eligibility×0.70 + normalized_bonus×0.30`
4. Applies `profile_signal` classification: `STRONG` (≥8.0), `MODERATE` (≥7.0), `WEAK` (≥5.5), `AVOID` (<5.5)

Best-horizon selection uses only `TRADING_HORIZONS = {"intraday", "short_term", "long_term"}`. Multibagger scores are populated from the DB (weekly cycle) rather than recomputed on demand.

#### `generate_trade_plan()`

Eight internal stages:

| Stage | Action |
|---|---|
| 1 | Reuse or rebuild `eval_ctx` from winner profile |
| 2 | Build `exec_ctx` via `build_execution_context`; enhance via `trade_enhancer.enhance_execution_context()` |
| 3 | Extract entry, SL, targets from `market_adjusted_targets` (or fallback to `risk` model) |
| 4 | Apply macro adjustment (±30% position sizing vs trend) |
| 5 | Build confidence history and audit trail from adjustments breakdown |
| 6 | Apply `validate_execution_rr()` — mutates `exec_ctx["can_execute"]` |
| 7 | Call `finalize_trade_decision()` — 4-layer signal determination |
| 8 | Generate enhanced narratives via `build_enhanced_summaries()` |

#### `finalize_trade_decision()` — 4-Layer Signal Logic

```
Layer 0 — Structure Gate: primary_found? No → WATCH
Layer 1 — Execution Gate: execution_blocked? → BLOCKED; low conf? → WATCH; else BUY/SELL
Layer 2 — Structural Rescue: RR failure + structural targets + high-conf → rescue to BUY
Layer 3 — Finalize: apply direction, write plan fields
```

---

### Phase 4 — Config Resolver

**File:** `config/config_resolver.py` — v6.0 (4200+ lines, 74 methods)

**Inputs:** `eval_ctx` dict with `indicators`, `fundamentals`, `price_data`, `patterns`  
**Outputs:** Complete `eval_ctx` (evaluation phase) and `execution` dict (execution phase)

The resolver is a **pure decision-making class** with no direct imports from raw config files. Every config value is accessed through `self.extractor` (a `QueryOptimizedExtractor` instance). Its two public APIs:

```python
resolver = ConfigResolver(MASTER_CONFIG, "short_term")

# Phase 1 — evaluation only (no capital/time dependency)
eval_ctx = resolver.build_evaluation_context_only(
    symbol, fundamentals, indicators, price_data, detected_patterns
)

# Phase 2 — execution projection
exec_ctx = resolver.build_execution_context_from_evaluation(
    evaluation_ctx=eval_ctx,
    capital=100000.0
)
```

#### Internal Evaluation Pipeline (8 phases)

```
Phase 1: Foundation
  _calculate_all_scores()       → technical, fundamental, hybrid scores
  detect_volume_signature()     → surge / drought / climax
  detect_divergence()           → bearish/bullish with severity

Phase 2: Setup Classification
  _classify_setup()             → candidates ranked by composite_score
                                  (70% priority × 30% fit quality)

Phase 3: Pattern Validation
  _validate_patterns()          → affinity, invalidation, entry rule gates

Phase 4: Strategy & Preferences
  _classify_strategy()          → 65% DNA fit + 35% setup quality blend
  _apply_setup_preferences()    → compatibility annotation (no blocking)

Phase 5: Structural Gates
  _validate_structural_gates()  → threshold eval via GATE_METRIC_REGISTRY
                                  context_paths; optional gates skipped

Phase 6: Execution Rules
  _validate_execution_rules()   → volatility guards, structure validation,
                                  SL distance, target proximity, divergence

Phase 7: Confidence
  _calculate_confidence()       → full pipeline (see Gating Layer 4)

Phase 8: Opportunity Gates
  _validate_opportunity_gates() → post-confidence gate layer
```

#### Key Method Details

**`_build_trend_context()`**  
Reads ADX regime thresholds from `rr_regime_adjustments` config (not hardcoded). Classifies regime as `strong`/`normal`/`weak` based on ADX only (direction-agnostic — bearish setups with high ADX still receive `strong` regime multipliers).

**`_build_momentum_context()`**  
Reads RSI/MACD thresholds from `momentum_thresholds` config per horizon. Derives divergence severity using adaptive multiples (3× decel threshold = severe, 1.5× = moderate).

**`_classify_setup()`**  
Evaluates all non-blocked setups. Rejects on: horizon block, pattern detection failure, fundamental gate failure, technical gate failure, context requirement failure, fit score < `MIN_FIT_SCORE` (10.0). Ranks by composite score. Exposes top-3 candidates for pattern validation.

**`_calculate_confidence()`**  
Full pipeline with 10 steps — see [Gating Layer 4](#layer-4--confidence-calculation). Execution penalties are **never** scaled by the divergence multiplier (independent dimensions). B8 ceiling (cap at 90 when `rvol ≤ 2.0`) applies **only** to bullish breakout/momentum setups.

**`_build_risk_candidates()`**  
Single ATR-based structural baseline. Tags `rr_source` as `"atr_structural"` when primary patterns exist (RR gate deferred to Stage 2) or `"generic_atr"` when no patterns found (RR gate enforced in Stage 1). No capital logic at this stage.

**`_finalize_risk_model()`**  
Dual constraint: `qty_by_risk = risk_per_trade / risk_per_share` vs `qty_by_capital = max_capital / price`. Takes the smaller of the two. Records `limit_reason` (`"max_capital_cap"` or `"risk_target"`) for UI diagnostics.

---

### Phase 5 — Trade Enhancer

**File:** `services/trade_enhancer.py` — v5.0

**Inputs:** `eval_ctx`, `exec_ctx`, `indicators`, `symbol`, `horizon`  
**Outputs:** Enhanced `exec_ctx` with pattern warnings, invalidation flags, RR regime metadata, timeline estimates, and market-adjusted targets

The trade enhancer is a **post-processing** layer that operates in real-time after the static resolver pass. It adds dynamic validation that the resolver intentionally defers (pattern age, breakdown state, market volatility regime).

#### `enhance_execution_context()`

Processing order:

1. **Context hash verification** — detects if market has moved since evaluation
2. **Pattern expiration** (`check_pattern_expiration`) — uses `formation_time` (Unix float) for real-time age; handles ISO string via `fromisoformat` fallback; compares against `typical_duration.max` from `PatternContext`
3. **Pattern invalidation** (`check_pattern_invalidation`) — evaluates breakdown gates via `extractor.evaluate_invalidation_gates()`; SHORT trades treat breakdown as **confirmation** (W39 fix); multi-candle duration tracking via DB
4. **RR regime multipliers** (`get_rr_regime_multipliers`) — reuses `eval_ctx["trend"]["regime"]` from resolver; maps to config `rr_regime_adjustments`
5. **Market-adaptive targets** (`adjust_targets_for_market_conditions`) — direction-agnostic (auto-detected from SL vs entry); volatility SL buffer from config; target stretch anchored to `current_price`; spread cost capped at 50% of T1 reward
6. **Direction conflict check** — runs **unconditionally** outside `if risk:` guard; maps trend vocabulary (BULLISH/BEARISH) to execution vocabulary (LONG/SHORT); blocks on mismatch

#### `adjust_targets_for_market_conditions()`

All constants sourced from `master_config.global.risk_management`:

- `volatility_buffer_factors`: `{low: 0.0, normal: 0.25, high: 0.5, extreme: 1.0}`
- `min_sl_atr_multiples`: `{intraday: 2.0, short_term: 2.0, long_term: 2.5}`
- `target_adjustment_factors`: `{low: 0.85, normal: 1.0, high: 1.15, extreme: 1.3}`
- `base_spread_pct`: `{intraday: 0.0015, short_term: 0.001, long_term: 0.0008}`

#### `validate_execution_rr()`

Reads `rr_gates` from extractor: `min_t1`, `min_t2`, `min_structural`, `execution_floor`. Applies trend-based relaxation via `trendStrength` from `eval_ctx["indicators"]`. Rescue logic checks ATR vs structural target source before allowing RR override.

#### Idempotency Guard

The expiry penalty (`-20` from config `expiry_penalty`) is gated by `_adj["expiry_applied"]` flag in `exec_ctx["confidence_adjustments"]`. Subsequent calls to `enhance_execution_context` on the same context do not stack the penalty.

---

### Phase 6 — Pattern Library

**Files:** `services/analyzers/pattern_analyzer.py`, `services/patterns/` (12 detector classes), `services/fusion/pattern_fusion.py`

Phase 6 has three distinct sub-components that execute in sequence:

#### `pattern_analyzer.py` — Detector Orchestrator

`pattern_analyzer.py` is the single entry point for all pattern detection. It iterates the full set of 10 detector classes, calls each `detect(df, indicators, horizon)` method, and collects every result into a unified `pattern_results` dict keyed by pattern name. It owns the try/except boundary around each detector call so that a failure in one detector (e.g., a malformed DataFrame slice in `cup_handle.py`) does not abort the remaining nine.

```
pattern_analyzer.analyze(df, indicators, horizon)
    ├── bollinger_squeeze.detect(...)       → result
    ├── cup_handle.detect(...)              → result
    ├── darvas.detect(...)                  → result
    ├── flag_pennant.detect(...)            → result
    ├── minervini_vcp.detect(...)           → result
    ├── three_line_strike.detect(...)       → result
    ├── ichimoku_signals.detect(...)        → result
    ├── golden_cross.detect(...)            → result   (alias "goldenCross")
    ├── death_cross.detect(...)             → result   (alias "deathCross")
    ├── double_top_bottom.BullishNeckline   → result   (alias "bullishNeckline")
    ├── double_top_bottom.BearishNeckline   → result   (alias "bearishNeckline")
    ├── momentum_flow.detect(...)           → result
    └── returns: { "goldenCross": {...}, "bollingerSqueeze": {...}, ... }  (×12 total)
```

#### Detector Contract

All detectors extend `BasePattern` and implement `detect(df, indicators, horizon) → Dict`. Every returned dict has the canonical schema:

```python
{
    "found":   bool,
    "score":   float,   # 0–100, normalized
    "quality": float,   # 0–10
    "meta": {
        # Required fields (consumed by trade_enhancer + resolver)
        "age_candles":         int,
        "formation_time":      float,   # Unix timestamp (primary key for real-time age)
        "formation_timestamp": str,     # ISO string for logging/UI
        "bar_index":           int,
        "type":                str,     # "bullish" | "bearish" (lowercase)
        "invalidation_level":  float,
        "velocity_tracking":   dict,
        "pattern_strength":    str,
        "current_price":       float,
        "horizon":             str,
        # Pattern-specific fields (see Pattern Library section)
    }
}
```

#### `pattern_fusion.py` — Indicators Injector

After `pattern_analyzer` collects all results, `pattern_fusion.merge_pattern_into_indicators(indicators, pattern_results, horizon=horizon, df=df)` iterates the results and, for every pattern where `found=True`, formats the result into a standardised UI object and injects it directly into the `indicators` dict.

**Function signature:**
```python
def merge_pattern_into_indicators(
    indicators: Dict[str, Any],
    pattern_results: Dict[str, Any],
    horizon: str = None,        # used to build scoped alias key
    df: pd.DataFrame = None     # optional — provides idempotent ts field
)
```

**Injected structure per found pattern:**
```python
indicators[alias] = {
    "value":  result.get("quality", 0),      # float 0–10 (quality score, NOT pattern name)
    "found":  result.get("found", False),    # bool — always True at injection time
    "ts":     float(df.index[-1].timestamp()) if df is not None else None,
                                             # idempotent timestamp from last OHLCV bar
    "raw":    result,                        # full detector output dict (meta, score, quality)
    "score":  result.get("score", 0),        # normalized 0–100
    "desc":   result.get("desc", f"Pattern {alias} Detected"),
    "alias":  f"{alias}_{horizon}" if horizon else alias,
    "source": "Pattern"
}
```

> **Note:** `value` is the pattern's **quality float** (0–10), not its name string. Templates displaying a pattern label should use `"alias"` or `"desc"`, not `"value"`. This is the correct production schema as of v15.1.

**In-place mutation notice:** `merge_pattern_into_indicators` mutates the `indicators` dict directly and returns `None`. The call site in `pattern_analyzer.py` deliberately does not capture the return value. The `indicators` dict passed in is the same object that signal_engine and config_resolver subsequently read — no copy is made.

This injection is the bridge between Phase 6 and Phase 3. After `merge_pattern_into_indicators` returns, the `indicators` dict is fully enriched — it contains both the computed technical indicators (EMA, ATR, RSI, etc.) and any live pattern structures. The signal engine and config resolver read from this single unified dict without needing to distinguish between "raw" indicators and "pattern" indicators.

See [Pattern Library](#pattern-library-1) for the full table of all 12 detectors.

---

### Phase 7 — Multibagger Pipeline

**Files:** `config/multibagger/` — isolated module running weekly on Sunday midnight IST

The multibagger pipeline is a **fully isolated** two-phase scan that does not share code paths with the main three-horizon trading loop. It uses its own extractor stack (`MBConfigExtractor` → `MBQueryOptimizedExtractor` → `MBConfigResolver`) that overrides pillar weights, scoring functions, and confidence config without patching any globals.

#### Phase 1 — Screener (`multibagger_screener.py`)

Hard-rejection gatekeeper. Filters on:
- **Universe gates:** sector exclusions, min listing age (365 days), min price (₹20), min market cap (₹500 Cr)
- **Fundamental gates:** `epsGrowth5y ≥ 15%`, `ROCE ≥ 15%`, `ROE ≥ 15%`, `D/E ≤ 1.0`, `promoterHolding ≥ 30%`, `piotroskiF ≥ 6`
- **Technical gates (weekly):** Stage 2 alignment (`Close > MMA6 > MMA12 > MMA24`), max drawdown from 52W high ≤ 30%

Parallel execution: `run_bulk_screener()` uses `ThreadPoolExecutor(max_workers=10)`.

#### Phase 2 — Evaluator (`multibagger_evaluator.py`)

Deep MB resolver using the isolated extractor stack:
- **Pillar weights:** `fund=0.60`, `hybrid=0.30`, `tech=0.10`
- **Scoring functions:** `mb_compute_fundamental_score()` (growth 35%, profitability 25%) and `mb_compute_technical_score()` (trend 50%, momentum 30%) — bypass main pipeline functions
- **Confidence config:** `MB_CONFIDENCE_CONFIG` — clamp `[50, 95]`, `min_tradeable = 60`

**Conviction tiers:**

| Tier | Score | Confidence |
|---|---|---|
| `HIGH` | ≥ 8.5 | ≥ 75% |
| `MEDIUM` | ≥ 7.5 | ≥ 65% |
| `LOW` | ≥ 6.5 | ≥ 60% |

`estimated_hold_months` and `entry_trigger` (pattern name or `"TECHNICAL_SETUP"`) are persisted to the `multibagger_candidates` table.

#### Scheduler (`mb_scheduler.py`)

Daemon thread started in FastAPI lifespan. Calculates next Sunday midnight IST, sleeps, then runs the full cycle. Supports manual trigger via `POST /multibagger/run`. Updates `cycle_status` dict imported by `mb_routes.py` for the `/multibagger/status` endpoint.

---

### Phase 8 — Orchestrator & DB

**Files:** `main.py`, `services/db.py`, `config/config_utility/market_utils.py`

#### `config/config_utility/market_utils.py` — Market Session Guard

Before the orchestrator dispatches any analysis work, `market_utils.py` determines the current trading session. It encodes NSE market hours (09:15–15:30 IST), pre-market (09:00–09:15), and after-hours windows, and exposes a simple query API that `main.py` uses to gate horizon selection:

```
market_utils.get_current_session()
    → "intraday"    (09:15–15:30 IST, trading days only)
    → "pre_market"  (09:00–09:15 IST)
    → "after_hours" (15:30+ IST or weekend)
    → "holiday"     (NSE holiday calendar lookup)
```

**Why this matters for the pipeline:** If `get_current_session()` returns `"after_hours"` or `"holiday"`, the orchestrator suppresses intraday horizon analysis entirely — there is no live price action to evaluate. Attempting an intraday scan on stale EOD data would produce gate evaluations against yesterday's closes dressed up as real-time signals. `market_utils.py` is the single authoritative check that prevents this.

#### `main.py` — FastAPI Application

- **Executors:** `ProcessPoolExecutor` (CPU-bound compute) + `ThreadPoolExecutor` (API + background I/O)
- **Cross-process safety:** `CACHE_LOCK = multiprocessing.Lock()` guards all JSON file writes
- **Cache warmer:** Background `asyncio.Task` (`periodic_warmer`) warms symbols in batches with rate-limit detection and cooling
- **`FULL_HORIZON_SCORES` dict:** In-memory store populated after each worker result (single-symbol) and after each warmer batch (bulk path). Feeds the horizon-toggle confluence dots in the UI

**`run_analysis()` modes:**

| Mode | Horizons Computed | Use Case |
|---|---|---|
| `"full"` | `intraday, short_term, long_term` | Default dashboard load |
| `"single"` | `[requested] + long_term` | Horizon toggle (4× faster) |

#### `_write_signal_cache_with_retry()`

Single-query upsert pattern: queries for existing row, calls `writer_fn(db, entry)`, commits. On `OperationalError` with SQLite lock contention (`"database is locked"` / `"busy"`), retries up to `SIGNAL_CACHE_WRITE_RETRIES` (default 3) with exponential backoff (`0.2s × attempt`). Each retry opens a fresh session.

#### `_save_analysis_to_db()`

Persists both `best_horizon` (system-recommended) and `selected_horizon` (user's active view) as separate indexed columns. Preserves existing `multi_score` from prior MB cycle runs when main pipeline does not recompute multibagger.

#### `_mark_analysis_error_in_db()`

Accepts a `Session` parameter. Writes `signal_text = "ERROR"`, `conf_score = 0`, clears all price fields (`entry_price`, `stop_loss`). Prevents stale cached trade values from being served as fresh after a failed analysis.

#### `run_migrations()` — Registry Pattern

```python
registry = {
    "add_selected_horizon":              migrate_add_selected_horizon,
    "add_direction_column":              migrate_add_direction_column,
    "add_pattern_breakdown_lifecycle":   migrate_add_pattern_breakdown_lifecycle,
}
```

Each migration checks `schema_migrations` table for its own name before running. Uses `engine.begin()` context manager to guarantee connection release on success or error. All migrations log at `WARNING` level. Idempotent by design — safe to re-run on every startup.

#### Cleanup Scheduler

`cleanup_old_breakdown_states()` runs in a background thread every 24 hours:
- **Stage 1 (soft-expire):** Active rows not updated in `days_old` (default 7) days → status `"expired"`
- **Stage 2 (hard-purge):** Resolved/expired rows with `resolved_at > 90 days` → `DELETE`

---

## Multi-Layer Gating System

The system evaluates seven independent gate layers in strict order. No lower layer can override a higher layer's block decision.

```
Layer 0 ─ Horizon Gate
Layer 1 ─ Structural Gates
Layer 2 ─ Execution Rules
Layer 3 ─ Pattern Validation
Layer 4 ─ Confidence Calculation
Layer 5 ─ Opportunity Gates
Layer 6 ─ Entry Permission
Layer 7 ─ Execution Validation
```

### Layer 0 — Horizon Gate

`TRADING_HORIZONS` scope filter. `best_fit` selection considers only `{intraday, short_term, long_term}`. Setups can be horizon-blocked via `confidence_config.horizons.X.setup_floor_overrides[setup] = None`.

### Layer 1 — Structural Gates

Gate metrics: `adx`, `trendStrength`, `volatilityQuality`, `rsi`, `bbpercentb`, `atrPct`, `rvol`, `volume`, `roe`, `deRatio`, `piotroskiF`, and others.

All metrics are resolved from `GATE_METRIC_REGISTRY` via `context_paths` — no hardcoded indicator key names in the resolver. Missing optional metrics (`optional: True` in registry, e.g., `piotroskiF`, `marketTrendScore`) are **skipped** rather than blocking. Missing required metrics produce a `"failed"` gate result.

```
Threshold source priority:
  Global gates (master_config)
       ↓
  Horizon gates (horizon override)
       ↓
  Setup gates (from SETUP_PATTERN_MATRIX context_requirements,
               already merged with horizon_overrides)
```

### Layer 2 — Execution Rules

Complex multi-condition validation requiring custom logic beyond simple thresholds:

| Rule | Logic |
|---|---|
| `volatility_guards` | If `atrPct > extreme_vol_buffer`: require `volatilityQuality ≥ min_quality_breakout`; else `≥ min_quality_normal` |
| `structure_validation` | For BREAKOUT setups: `price ≥ resistance × (1 + breakout_clearance)` |
| `sl_distance_validation` | `0.5 × ATR ≤ SL_distance ≤ 5.0 × ATR` |
| `target_proximity_rejection` | `T1_distance ≥ min_target_distance_pct` |
| `divergence_gate` | `eval_ctx["divergence"]["allow_entry"]` from Phase 1 pre-computation |

Rules can be disabled per-horizon via `entry_gates.execution_rules.X.enabled = False`.

### Layer 3 — Pattern Validation

For each of the top-3 candidate setups, `_validate_patterns()` classifies detected patterns as `PRIMARY`, `CONFIRMING`, or `CONFLICTING`. Then for each found pattern:

- **Invalidation gates** — evaluated via `gate_evaluator.evaluate_invalidation_gates()`. For SHORT trades: breakdown = **confirmation**, not invalidation.
- **Entry rule gates** — evaluated against a namespace of `{indicators + fundamentals + meta}`. E.g., `darvasBox` requires `price ≥ box_high × 1.002` and `box_age_candles ≤ 50`.

### Layer 4 — Confidence Calculation

10-step pipeline:

```
Step 1: Base floor from global.setup_baseline_floors[setup_type]
Step 2: Apply horizon.base_confidence_adjustment (-10 intraday, -5 short, 0 long)
Step 3: Override with horizon.setup_floor_overrides if defined
Step 4: Apply ADX confidence bands (boosts: +20 explosive, +10 strong)
        Apply ADX confidence penalties (weak trend)
Step 5: Apply global.volume_modifiers (surge +10, drought -15, climax -20)
Step 6: Apply global.universal_adjustments (divergence MULTIPLIER, trend bands ADDITIVE)
Step 7: Apply horizon.conditional_adjustments (setup-specific penalties/bonuses)
Step 8: Apply setup validation_modifiers (per-setup penalties/bonuses from matrix)
Step 9: Apply execution rule penalties (ADDITIVE, NOT scaled by divergence multiplier)
        Penalties: -5/warning, -15/violation, ±risk_score adjustment
Step 10: Clamp to [floor, ceiling]
         B8 ceiling: cap at 90 when rvol ≤ 2.0 AND setup is bullish breakout/momentum
```

**Divergence multiplier:** Applies only to `volume_modifiers` and `trend_strength_bands` adjustments. Conditional adjustments, setup modifiers, and execution penalties are **never** scaled by it.

### Layer 5 — Opportunity Gates

Post-confidence layer. Gates evaluated: `confidence ≥ min`, `rrRatio ≥ min` (deferred/skipped for pattern trades where `rr_source = "atr_structural"`; enforced for ATR-fallback trades), `technicalScore ≥ min`, `fundamentalScore ≥ min`, `hybridScore ≥ min`.

The `rrRatio` gate is marked `optional: True` in `GATE_METRIC_REGISTRY` with `skip_reason: "deferred_to_stage2_enhancer"`. This prevents pattern setups from failing the RR gate before pattern geometry is applied by the trade enhancer.

### Layer 6 — Entry Permission

`_build_entry_permission()` aggregates all evaluation-phase validations:

- Structural gates must pass
- Execution rules must pass
- Opportunity gates must pass
- Pattern entry validation must pass (all PRIMARY patterns)
- `eval_ctx["divergence"]["allow_entry"]` must be `True`
- Volume signature must not be `"climax"`
- `eval_ctx["confidence"]["block_entry"]` must be `False`
- No invalidated patterns in the active setup (unconditional — cannot be overridden)

### Layer 7 — Execution Validation

`validate_execution_rr()` with trend-based relaxation. `_finalize_risk_model()` applies dual constraint (risk-per-trade vs max-capital). `_build_time_constraints()` checks intraday avoidance windows. `_can_execute()` combines all checks into final `can_execute` dict with `is_hard_blocked` flag.

---

## Config Architecture

### Three-Tier Resolution

```
Global defaults (master_config.py)
        ↓  overridden by
Horizon-specific (master_config["horizons"]["short_term"])
        ↓  overridden by
Setup-specific (SETUP_PATTERN_MATRIX["MOMENTUM_BREAKOUT"]["context_requirements"])
               + horizon_overrides["short_term"]
```

**All config access goes through `query_optimized_extractor.py`.** Neither `config_resolver.py` nor `signal_engine.py` may import from any config file directly.

### Config Hierarchy Examples

**Structural gate for `adx`:**

```python
# Global: master_config.global.entry_gates.structural.gates
{"adx": {"min": 18}}

# Horizon override: master_config.horizons.intraday.entry_gates.structural
{"adx": {"min": 20}}   # overrides global → effective: 20

# Setup override: SETUP_PATTERN_MATRIX["MOMENTUM_BREAKOUT"]
#   .horizon_overrides.intraday.context_requirements.technical
{"adx": {"min": 18}}   # overrides horizon → effective: 18 for this setup
```

**Confidence floor for `QUALITY_ACCUMULATION`:**

```python
# Global: confidence_config.global.setup_baseline_floors
{"QUALITY_ACCUMULATION": 45}

# Horizon override: confidence_config.horizons.long_term.setup_floor_overrides
{"QUALITY_ACCUMULATION": 55}   # → effective: 55 for long_term

# Block entirely: confidence_config.horizons.intraday.setup_floor_overrides
{"VALUE_TURNAROUND": None}     # → setup blocked for intraday
```

**RR regime multipliers:**

```python
# Global: master_config.global.risk_management.rr_regime_adjustments
{"strong_trend": {"adx": {"min": 35}, "t1_mult": 2.0, "t2_mult": 4.0}}

# Horizon override: master_config.horizons.long_term.risk_management.rr_regime_adjustments
{"strong_trend": {"adx": {"min": 35}, "t1_mult": 2.5, "t2_mult": 5.0}}
# → deep-merged per regime key
```

### Extractor State Validation

```python
resolver = ConfigResolver(MASTER_CONFIG, "short_term")
state = resolver.extractor.validate_extractor_state()
# {
#   "valid": True,
#   "errors": [],
#   "warnings": [],
#   "has_confidence_config": True,
#   "cache_stats": {"gate_cache_size": 12, "pattern_cache_size": 8}
# }
```

Initialization fails hard (`RuntimeError`) if confidence config is missing or if the extractor state is invalid. This prevents silent degradation at startup.

---

## Pattern Library

| Pattern | Alias | Direction | Horizons | Key Meta Fields | Unique Mechanics |
|---|---|---|---|---|---|
| **Bollinger Squeeze** | `bollingerSqueeze` | Neutral | `intraday`, `short_term` | `squeeze_duration`, `squeeze_strength`, `state` | `found=True` only on `SQUEEZE_BREAKOUT`; `found=False` during `SQUEEZE_ON` wait phase. `squeeze_duration` computed by walking rolling `bbWidth` history backward |
| **Cup & Handle** | `cupHandle` | Bullish | `short_term`, `long_term` | `rim_level`, `cup_low`, `handle_low`, `depth_pct`, `handle_depth_pct` | Horizon-aware window via `HORIZON_WINDOWS`; formation_time as Unix float from left rim index |
| **Darvas Box** | `darvasBox` | Bullish | `intraday`, `short_term` | `box_high`, `box_low`, `box_age_candles`, `box_height_pct` | `box_high`/`box_low` not overwritten by `meta.update()`; binary breakdown — no monitor mode |
| **Flag / Pennant** | `flagPennant` | Trend-aligned | `intraday`, `short_term` | `pole_gain_pct`, `flag_low`, `flag_high`, `flag_drift_pct`, `pole_strength` | `is_uptrend=False` default when `maFast` unavailable (safe-fail); pole vs flag window configurable |
| **Golden Cross** | `goldenCross` | Bullish | `short_term`, `long_term`, `multibagger` | `maMid`, `maSlow`, `ma_type`, `cross_strength`, `crossover_fresh` | Split class from `DeathCross`; `GoldenDeathCross` kept as legacy alias; `invalidation_level = maMid` |
| **Death Cross** | `deathCross` | Bearish | `short_term`, `long_term`, `multibagger` | `maMid`, `maSlow`, `ma_type`, `cross_strength` | Uses same `_detect_cross()` helper as `GoldenCross`; appears as `CONFLICTING` in bullish setups |
| **Ichimoku Signals** | `ichimokuSignals` | Variable | `short_term`, `long_term` | `cloud_top`, `cloud_bottom`, `tenkan_kijun_spread`, `cloud_color`, `signal_age` | `signal_age` computed by walking TK series backward (not hardcoded); fresh cross always `age=1` (W50) |
| **Minervini VCP** | `minerviniStage2` | Bullish | `short_term`, `long_term`, `multibagger` | `contraction_pct`, `volatility_quality`, `stage_quality`, `contraction_strength` | `contraction_pct` required by resolver `_calculate_pattern_targets` for `depth = entry × (contraction_pct/100)` |
| **Momentum Flow** | `momentumFlow` | Neutral | `intraday`, `short_term`, `long_term` | `bar_index`, `invalidation_level`, `velocity_tracking`, `pattern_strength`, `current_price` | Full meta schema identical to other patterns for uniform resolver treatment |
| **Three-Line Strike** | `threeLineStrike` | Bull/Bear | `intraday`, `short_term`, `long_term` | `strike_low`, `strike_high`, `prior_range`, `strike_candle_body`, `reversal_confidence` | `type` always lowercase `"bullish"` / `"bearish"`; bearish invalidation anchored to `strike_high` (not `strike_low`) |
| **Bullish Neckline** | `bullishNeckline` | Bullish | `intraday`, `short_term`, `long_term` | `neckline_level`, `left_low`, `right_low`, `pattern_depth_pct` | Double-bottom variant; `invalidation_level = neckline_level`; split from legacy `DoubleTopBottom` class into dedicated `BullishNecklinePattern` |
| **Bearish Neckline** | `bearishNeckline` | Bearish | `intraday`, `short_term`, `long_term` | `neckline_level`, `left_high`, `right_high`, `pattern_height_pct` | Double-top variant; appears as `CONFLICTING` in bullish setups; split from legacy `DoubleTopBottom` class into dedicated `BearishNecklinePattern` |

**Universal meta guarantee:** Every detector that sets `found=True` must populate all 10 standard fields: `age_candles`, `formation_time` (Unix float), `formation_timestamp` (ISO string), `bar_index`, `type`, `invalidation_level`, `velocity_tracking`, `pattern_strength`, `current_price`, `horizon`. The trade enhancer relies on `formation_time` for real-time age calculation.

> **Note on `engulfing.py`:** `EngulfingPattern` exists in `services/patterns/engulfing.py` and is V15.0 meta-contract compliant, but is **not registered** in `pattern_analyzer.py`'s detector list. It is a legacy detector kept for reference. Do not add it to the active registry without also adding a `PATTERN_METADATA` entry in `setup_pattern_matrix_config.py` and assigning it to at least one setup's `PRIMARY` / `CONFIRMING` list.

---

## Database Schema

All tables in `data/trade.db` (SQLite, **WAL mode** with **10s busy_timeout** for high-concurrency safety). Primary signal writes utilize a jittered exponential backoff retry helper.

### `signal_cache`

Per-symbol analysis cache. TTL: 1 hour (enforced at read time in `get_cached()`).

| Column | Type | Notes |
|---|---|---|
| `symbol` | String PK | NSE ticker |
| `best_horizon` | String | System-recommended horizon (`intraday` / `short_term` / `long_term`) |
| `selected_horizon` | String | User's currently-viewed horizon — may differ from `best_horizon` |
| `score` | Float | `final_score × 10` (0–100 scale for UI) |
| `recommendation` | String | `"{profile_signal}--{horizon}"` |
| `signal_text` | String | `BUY` / `SELL` / `WATCH` / `BLOCKED` / `HOLD` / `ERROR` |
| `conf_score` | Integer | Clamped confidence % |
| `rr_ratio` | Float | Final execution RR |
| `entry_price` | Float | Entry trigger level |
| `stop_loss` | Float | Execution SL |
| `direction` | String INDEXED | `"bullish"` / `"bearish"` / `"neutral"` — dedicated column, not extracted from JSON |
| `horizon_scores` | JSON | `{intra_score, short_score, long_score, multi_score, sl_dist, direction, macro_index_name}` |
| `updated_at` | DateTime INDEXED | UTC-aware; used for TTL check |

### `stock_meta`

Symbol metadata: `symbol PK`, `sector`, `industry`, `marketCap`, `is_favorite`, `last_scan_time`.

### `paper_trades`

Paper trading positions: `symbol`, `entry_price`, `target_1`, `target_2`, `stop_loss`, `estimated_hold_days`, `horizon`, `position_size`, `status` (`OPEN`/`WIN`/`LOSS`/`PARTIAL`), `created_at`, `updated_at`.

### `pattern_breakdown_state`

Active/resolved breakdown tracking.

| Column | Notes |
|---|---|
| PK | `(symbol, pattern_name, horizon)` composite |
| `status` | `"active"` → `"expired"` (soft, 7d stale) → hard-purged at 90d |
| `candle_count` | Incremented each candle breakdown condition holds |
| `started_at` | UTC datetime of first breakdown trigger |
| `resolved_at` | Set on soft-expire or confirmation |
| `resolution_reason` | `"cleanup_expired"` / `"resolved"` / `"confirmed"` |
| `price_at_breakdown` | Price when breakdown first triggered |
| `threshold_level` | Pattern invalidation level (pivot, box_low, handle_low) |

### `pattern_breakdown_events`

Append-only event log for every status transition: `(id PK, symbol, pattern_name, horizon, event_type, event_time, candle_count, details JSON)`.

### `pattern_performance_history`

Velocity tracking for ML analytics: `(symbol, pattern_name, horizon, setup_type, detected_at, entry_price, target_1, target_2, stop_loss, t1_reached, t2_reached, stopped_out, days_to_t1, bars_to_t1, days_to_t2, trend_regime, adx_at_entry, rr_ratio, pattern_meta JSON, completed)`.

### `schema_migrations`

Migration version table: `migration_name PK` (applied migrations), `applied_at`. Registry pattern, no Alembic. Idempotent — safe to re-run on every startup.

### `multibagger_candidates`

MB pipeline results: `symbol PK`, `conviction_tier`, `fundamental_score`, `technical_score`, `hybrid_score`, `final_score`, `final_decision_score`, `confidence`, `primary_setup`, `primary_strategy`, `entry_trigger`, `estimated_hold_months`, `thesis_json`, `gatekeeper_passed`, `rejection_reason`, `last_evaluated`, `re_evaluate_date`, `prev_conviction_tier`, `tier_changed_at`.

### `fundamental_cache`

24-hour fundamental data cache: `symbol PK`, `data JSON`, `updated_at`. Shared between main pipeline and MB weekly cycle — eliminates duplicate yfinance calls.

---

## Signal Semantics

All five signal states are **first-class outputs** with explicit semantics. No state is a fallback or default.

| Signal | Condition | Meaning |
|---|---|---|
| `STRONG_BUY` / `BUY` | All gates pass, confidence ≥ threshold, RR valid | Entry is structurally sound and timing is right. Execute. |
| `SELL` | Bearish direction, all gates pass | Short/exit entry. Direction-aware mirror of BUY. |
| `WATCH` | Strong profile but no active entry structure | Stock quality is good but pattern/setup alignment is absent. Monitor for formation. Layer 0 block. |
| `HOLD` | Active position management signal | Not a new entry. Only generated for positions already open. |
| `BLOCKED` | Setup + pattern present but execution gate failed | Entry structure exists but a hard gate (spread, volume, RR floor) prevents execution. Includes first-class `block_reason`. |
| `AVOID` | Fundamental or macro red flags | Fundamental score too low or macro conditions adverse. Not generated by technical analysis path. |

**`WATCH` vs `BLOCKED`:**  
`WATCH` = Layer 0 (no primary pattern detected for setup).  
`BLOCKED` = Layer 6 (pattern found, context valid, but execution gate prevents entry).  
These are never conflated.

---

## Pattern State Lifecycle

```
Detected (found=True, quality ≥ threshold)
        │
        ▼
Active Tracking
        │
        ├── Velocity recorded in pattern_performance_history
        │
        ├── T1 hit → days_to_t1, bars_to_t1 logged
        │
        ├── T2 hit → days_to_t2 logged, completed=True
        │
        ├── Stopped out → stopped_out=True, exit_price logged
        │
        └── Breakdown condition triggered
                │
                ▼
        Duration Tracking (pattern_breakdown_state)
                │
                ├── candle_count < required_duration:
                │       status="active", candle_count++
                │
                └── candle_count ≥ required_duration:
                        │
                        ▼
                Confirmed Breakdown
                        │
                        ├── LONG trade: entry BLOCKED (invalidated=True)
                        │
                        └── SHORT trade: entry CONFIRMED (W39)

Stale Pattern (not updated in 7 days)
        │
        ▼
Soft-expired: status="expired", resolved_at set

Resolved Pattern (90 days after resolved_at)
        │
        ▼
Hard-purged: DELETE from pattern_breakdown_state
             (pattern_breakdown_events retained for ML audit)
```

**Expiration vs Invalidation:**  
- *Expiration* (trade enhancer): Pattern is too old relative to `typical_duration.max`. Triggers confidence penalty (`-20`).  
- *Invalidation* (breakdown tracking): Price breached the pattern's structural level. Triggers `BLOCKED` signal unconditionally.

---

## Key Architecture Decisions

### 1. Config-Driven Gate Evaluation via GATE_METRIC_REGISTRY

All structural and opportunity gate metrics are registered in `GATE_METRIC_REGISTRY` with `context_paths` that describe exactly where in the `eval_ctx` dict the metric value lives. `_resolve_gate_value_from_context()` in the resolver traverses these paths rather than using any hardcoded key lookups. This makes adding a new gate metric a pure config change with no resolver code modification.

### 2. Optional Gates via `optional: True` Flag

Metrics like `piotroskiF`, `marketTrendScore`, and `rrRatio` are marked `optional: True` in `GATE_METRIC_REGISTRY`. When the value is missing, the gate is **skipped** (not failed). If more than 50% of a category's metrics are missing, the result is tagged `low_confidence` but entry is not blocked. This is essential for new listings and data-sparse NSE stocks.

### 3. Soft-Delete Pattern Breakdown States (90-Day ML Retention)

`pattern_breakdown_state` uses a two-stage cleanup model: soft-expire at 7 days (status → `"expired"`), hard-purge at 90 days. This ensures the ML training set (`pattern_performance_history`) always has access to the corresponding breakdown event chain. The `pattern_breakdown_events` audit log is never purged.

### 4. No Alembic — Registry-Pattern Migrations

`schema_migrations` table tracks applied migrations by name. `run_migrations()` iterates a static registry dict. Each migration uses `engine.begin()` for guaranteed connection cleanup. Idempotent by design, runs on every `init_db()` call. This avoids Alembic's environment complexity for a single-developer, single-DB deployment while maintaining auditability.

### 5. WATCH / HOLD / BLOCKED as First-Class Distinct States

These three states are semantically distinct and must never be aliased to each other. `WATCH` = quality without timing. `BLOCKED` = timing blocked by execution gate. `HOLD` = active position management. Conflating them would produce incorrect UI behavior (e.g., a `BLOCKED` signal should display the specific gate failure, while `WATCH` should prompt the user to monitor for pattern formation).

### 6. Execution Penalties Never Scaled by Divergence Multiplier

The divergence multiplier is a signal-quality dimension (is the price/RSI relationship healthy?). Execution penalties (warning/violation from rules like spread guards, volume guards) are a separate execution-risk dimension. Scaling execution penalties by the divergence multiplier would incorrectly compound two orthogonal concerns. `_calculate_confidence()` maintains a separate `exec_adjustment` accumulator that is added after the divergence-scaled modifiers.

### 7. B8 Ceiling — Scoped to Bullish Breakout/Momentum Only

The confidence ceiling of 90 when `rvol ≤ 2.0` is an institutional validation rule: a breakout without volume confirmation cannot be high-conviction. This ceiling is deliberately excluded from bearish setups (`BREAKDOWN`, `BEAR_TREND_FOLLOWING`, `MOMENTUM_FLOW_BREAKDOWN`) and from structural/value setups where volume is intentionally low during accumulation phases.

### 8. Direction Conflict Gate Runs Unconditionally

`enhance_execution_context()` runs the direction conflict check **outside** the `if risk:` guard. This ensures that a trend direction mismatch (e.g., resolver classifies bearish but market-adjusted targets were LONG) is caught even when the risk model is absent or malformed. The check maps vocabulary (`BULLISH→LONG`, `BEARISH→SHORT`) before comparison.

### 9. SHORT Trade Breakdown is Confirmation

Pattern invalidation logic in `check_pattern_invalidation()` skips the breakdown-blocking logic for `position_type == "SHORT"`. A price breaking below the Darvas box floor confirms the short thesis rather than invalidating it. The trade enhancer passes `position_type` derived from `eval_ctx["trend"]["classification"]["direction"]`.

### 10. Decoupled Discovery and Execution Scores

`profile_signal` (the stock quality rating: `STRONG`/`MODERATE`/`WEAK`/`AVOID`) is computed from the structural eligibility score and is **independent** of gates, RR, or execution context. A stock can show `STRONG` profile with `BLOCKED` signal — meaning it is a high-quality company but current execution conditions prevent entry. This decoupling preserves full-fidelity scoring even when structural gates block immediate trading.

---

## API Endpoints

### Analysis

| Method | Path | Description |
|---|---|---|
| `GET` | `/analyze?symbol=X&index=Y&horizon=Z` | Full or single-horizon analysis; renders `result.html` |
| `POST` | `/analyze` | Form-submit version of analysis |
| `POST` | `/quick_scores` | Batch analysis for index scan; returns flat JSON for AG Grid |

### Index & Data

| Method | Path | Description |
|---|---|---|
| `GET` | `/load_index/{index_name}` | Load stock list + corp actions for an index |
| `GET` | `/corporate_actions?ticker=X` | Upcoming/past dividends, splits, bonuses |
| `POST` | `/api/fetch_corp_actions` | Trigger on-demand corp actions refresh (1h cooldown) |
| `GET` | `/api/corp_actions_status` | Cache readiness + count |
| `GET` | `/api/macro_metrics` | World Bank macro indicators |

### Paper Trading

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/paper_trade/add` | Add paper trade position |
| `POST` | `/api/paper_trade/remove` | Remove open paper trade |
| `GET` | `/api/paper_trade/status?symbol=X&horizon=Y` | Check if position is open |
| `GET` | `/api/paper_trade/cmp?symbols=X,Y` | Current market prices for portfolio |
| `GET` | `/paper_trades` | Paper portfolio dashboard (HTML) |

### Multibagger

| Method | Path | Description |
|---|---|---|
| `GET` | `/multibagger/candidates` | All candidates; optional `tier` and `passed` filters |
| `GET` | `/multibagger/candidates/{symbol}` | Full thesis including `thesis_json` |
| `GET` | `/multibagger/status` | Last cycle stats + next scheduled run |
| `POST` | `/multibagger/run` | Trigger immediate manual MB cycle |
| `GET` | `/multibagger_dashboard` | Multibagger picks UI (HTML) |

### Pages

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Main index dashboard |

---

## Presentation Layer & UI

**Files:** `templates/result.html`, `templates/mb_result.html`, `services/summaries.py`

### `summaries.py` — Signal Narrative Generator

Before the FastAPI route serialises the final result dict to JSON, `summaries.py` generates the human-readable signal summary string that appears in both the API response and the AG Grid dashboard. It receives the resolved `signal`, `setup`, `confidence`, `entry_price`, `stop_loss`, and `target` values and composes a concise natural-language sentence:

```
"STRONG_BUY — Minervini VCP breakout on short_term horizon.
 Confidence: 82%. Entry ₹1,420 | SL ₹1,374 | T1 ₹1,512 | T2 ₹1,604.
 R:R 2.1 : 1. Setup quality: HIGH."
```

This string is what gets persisted to `signal_cache.signal_text` in the database, surfaced in the `/analyze` JSON response, and rendered verbatim in the AG Grid tooltip cell. `summaries.py` is the only place this string is constructed — neither the signal engine nor the trade enhancer produce narrative output directly.

The FastAPI routes under `/analyze` and `/multibagger_dashboard` render Jinja2 templates that receive the raw JSON signal output and map it into interactive AG Grid dashboards.

### Template Routing

| Template | Route | Grid Contents |
|---|---|---|
| `result.html` | `/` and `/analyze` responses | Per-symbol signal rows with setup, confidence, entry/SL/target, gate pass/fail badges |
| `mb_result.html` | `/multibagger_dashboard` | Conviction-tiered multibagger picks with fundamental scores, pattern alignment, and holding horizon |

### Confluence Dots

The most distinctive UI element is the **Confluence Dots** display — a three-cell traffic light that shows the signal alignment across all three primary horizons for the same symbol simultaneously:

```
Symbol      Intraday   Swing    Long-Term   Confluence
─────────────────────────────────────────────────────
RELIANCE      🟢         🟢        🟢          STRONG
HDFCBANK      🟢         🟡        🔴          MIXED
INFY          🔴         🟡        🟢          WEAK
```

Each dot maps directly to the `signal` field from the corresponding horizon's analysis result:

| Signal | Dot Color |
|---|---|
| `STRONG_BUY` / `BUY` | 🟢 Green |
| `WATCH` / `HOLD` | 🟡 Yellow |
| `SELL` / `AVOID` / `BLOCKED` | 🔴 Red |
| Not analyzed / missing | ⚪ Grey |

**Confluence logic:** The UI queries all three horizons in parallel and assembles the dot row client-side after all three responses resolve. A stock is flagged as **STRONG confluence** only when all three horizons return `BUY` or `STRONG_BUY` — meaning the momentum thesis is valid on the 5-minute chart, the daily chart, and the weekly chart simultaneously. This multi-timeframe alignment filter is the primary tool traders use to surface the highest-conviction setups from the full scan output.

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- ~4 CPU cores recommended for `ProcessPoolExecutor`
- NSE universe CSV at `data/nifty500.csv` (column: `SYMBOL`)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: `fastapi`, `uvicorn`, `sqlalchemy`, `pandas`, `pandas-ta`, `yfinance`, `numpy`, `scipy`, `pytz`

### Initialize Database

```bash
python -c "from services.db import init_db; init_db()"
```

This runs `Base.metadata.create_all()` for all 8 tables and applies all pending schema migrations.

### Start Server

```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

Do not use `--reload` in production — it breaks the `multiprocessing.Lock()` and `ProcessPoolExecutor` setup.

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WARMER_BATCH_SIZE` | `5` | Symbols per cache warmer batch |
| `WARMER_TOP_N_DURING_MARKET` | `50` | Symbols to warm during market hours |
| `WARMER_MARKET_INTERVAL_SEC` | `900` | Warmer cycle interval during market hours |
| `WARMER_OFFPEAK_INTERVAL_SEC` | `3600` | Warmer cycle interval off-peak |
| `WARMER_BATCH_TIMEOUT_SEC` | `300` | Max seconds per batch before timeout |
| `WARMER_RATE_LIMIT_COOLDOWN_SEC` | `300` | Cooldown on yfinance rate limit detection |
| `SIGNAL_CACHE_WRITE_RETRIES` | `3` | SQLite write retry count |
| `SIGNAL_CACHE_RETRY_SLEEP_SEC` | `0.2` | Base sleep between retries |
| `ENABLE_CACHE_WARMER` | `True` | Toggle background cache warmer |
| `ENABLE_JSON_ENRICHMENT` | `True` | Toggle async JSON name enrichment |

---

## Configuration Guide

### Tuning Gate Thresholds

Gate thresholds live in two places depending on scope:

**Universal gates** (all setups): `config/master_config.py` → `global.entry_gates.structural.gates` and `global.entry_gates.opportunity.gates`

```python
# Raise the minimum ADX for all horizons
"adx": {"min": 22}   # was 18

# Disable a gate globally (set to None)
"piotroskiF": {"min": None}
```

**Horizon overrides**: `master_config.horizons.{horizon}.entry_gates`

```python
# Tighten short_term opportunity confidence requirement
"short_term": {
    "entry_gates": {
        "opportunity": {
            "confidence": {"min": 65}  # was 60
        }
    }
}
```

**Setup-specific gates**: `config/setup_pattern_matrix_config.py` → `SETUP_PATTERN_MATRIX[setup].context_requirements` and `horizon_overrides`

### Tuning Confidence

All confidence parameters are in `config/confidence_config.py`:

```python
# Raise base floor for a specific setup
"setup_baseline_floors": {
    "MOMENTUM_BREAKOUT": 60   # was 55
}

# Add a horizon-specific bonus
"horizons": {
    "short_term": {
        "conditional_adjustments": {
            "bonuses": {
                "my_new_bonus": {
                    "gates": {"adx": {"min": 30}},
                    "confidence_boost": 8,
                    "reason": "Strong ADX bonus"
                }
            }
        }
    }
}
```

### Tuning RR Thresholds

Edit `master_config.py` → `global.risk_management.rr_gates` for global minimums and `horizons.{horizon}.risk_management.rr_gates` for horizon-specific overrides:

```python
"long_term": {
    "risk_management": {
        "rr_gates": {
            "min_t1": 2.5,   # raise minimum T1 RR for long-term
            "min_t2": 3.5,
            "execution_floor": 1.6
        }
    }
}
```

### Adding a New Pattern

1. Create `services/patterns/my_pattern.py` extending `BasePattern`
2. Implement `detect()` returning the canonical `{found, score, quality, meta}` schema
3. Add entry in `PATTERN_METADATA` in `setup_pattern_matrix_config.py` with `physics`, `entry_rules`, and `invalidation` blocks
4. Add alias to relevant setups' `patterns.PRIMARY` / `CONFIRMING` / `CONFLICTING` lists in `SETUP_PATTERN_MATRIX`
5. Register in `services/patterns/pattern_analyzer.py` pattern registry

### Adding a New Setup

1. Add entry in `SETUP_PATTERN_MATRIX` with all required fields: `patterns`, `classification_rules`, `default_priority`, `context_requirements`, `validation_modifiers`, `horizon_overrides`
2. Optionally add horizon priority override in `master_config.calculation_engine.horizon_priority_overrides`
3. Optionally add confidence floor in `confidence_config.global.setup_baseline_floors`

---

## Known Limitations (Refactored v15.1)

### ✅ W64 — Decoupled Conflict Penalty (RESOLVED)

The conflict penalty calculated in `_validate_patterns` is correctly subtracted from `total_adjustment` at **Step 8** of the confidence pipeline. A setup with contradictory patterns (e.g., Golden Cross + Death Cross) now receives a specific penalty (up to -30).

### ✅ W46 — Centralized Horizon Constants (RESOLVED)

The dual-semantics issue of `HORIZON_WINDOWS` has been resolved by decoupling into `HORIZON_WINDOWS_SECONDS` (for DB cleanup) and `HORIZON_WINDOWS_BARS` (for pattern detectors). All patterns now reference the centralized `horizon_constants.py` file.

### ✅ W50 — Ichimoku `signal_age` Precision (RESOLVED)

The `signal_age` ambiguity for fresh crosses has been fixed using a sentinel return value in `_compute_signal_age`. The system now distinguishes between a truly fresh cross (`age=1`) and a cross detected at the lookback boundary.

### ✅ W59 — `FULL_HORIZON_SCORES` Hydration (RESOLVED)

The in-memory score cache is now reliably hydrated by the main process from the results of parallel workers in both single-symbol and bulk warmer paths.

### ✅ SQLite Concurrency (RESOLVED)

Write contention on the `signal_cache` has been mitigated via a combination of `journal_mode=WAL`, a `10,000ms` busy timeout, and the `_write_signal_cache_with_retry` helper in `main.py` which implements jittered exponential backoff.

---

## Contributing / Development Notes

### Code Organization

```
config/
  ├── confidence_config.py            # Confidence pipeline config
  ├── config_extractor.py             # Section extraction + ConfigSection dataclass
  ├── config_helpers.py               # Business logic bridge (only public interface)
  ├── config_resolver.py              # Decision-making layer (74 methods)
  ├── config_utility/
  │   ├── market_utils.py             # NSE market hours / trading session detection
  │   └── logger_config.py            # METRICS, SafeDict, log_failures, track_performance
  ├── fundamental_score_config.py     # Fundamental scoring
  ├── gate_evaluator.py               # Pure stateless gate engine
  ├── master_config.py                # Global + horizon configs
  ├── query_optimized_extractor.py    # Typed query API + caching
  ├── setup_pattern_matrix_config.py
  ├── strategy_matrix_config.py
  ├── technical_score_config.py
  └── multibagger/                    # Isolated MB module

services/
  ├── analyzers/
  │   └── pattern_analyzer.py         # Orchestrates all 12 active pattern detector calls
  ├── cache.py                        # 3-Tier OHLCV caching (RAM/Parquet/TTL)
  ├── corporate_actions.py            # Splits/dividend backward adjustment
  ├── data_fetch.py                   # Raw OHLCV ingestion engine
  ├── db.py                           # SQLAlchemy models + migrations
  ├── fundamentals.py                 # 24h SQLite fundamental cache
  ├── fusion/
  │   └── pattern_fusion.py           # Injects found patterns into indicators dict (idempotent ts)
  ├── indicators.py                   # Polymorphic indicator generation
  ├── signal_engine.py                # Top-level orchestrator
  ├── summaries.py                    # Human-readable signal narrative generator
  ├── trade_enhancer.py               # Real-time pattern validation
  └── patterns/
      ├── base.py                     # BasePattern ABC
      ├── bollinger_squeeze.py
      ├── cup_handle.py
      ├── darvas.py
      ├── double_top_bottom.py        # BullishNecklinePattern + BearishNecklinePattern
      ├── engulfing.py                # EngulfingPattern (legacy — not in active registry)
      ├── flag_pennant.py
      ├── golden_cross.py             # GoldenCross + DeathCross (split classes)
      ├── ichimoku_signals.py
      ├── minervini_vcp.py
      ├── momentum_flow.py
      ├── pattern_state_manager.py
      ├── pattern_velocity_tracking.py
      ├── three_line_strike.py
      └── utils.py

templates/
  ├── result.html                     # AG Grid main dashboard
  └── mb_result.html                  # Multibagger UI

daily_corp_action_warmer.py           # Pre-market corporate action scheduler
main.py                               # FastAPI app + lifespan
```

### Core Principles

1. **Config purity:** No code outside `config_extractor.py` may import from `config/*.py` raw config files. All config access through the extractor stack.
2. **Evaluation purity:** `eval_ctx` produced by `build_evaluation_context_only()` must never be mutated by the trade enhancer. The enhancer should write to `exec_ctx` and note stale-context flags rather than modifying discovery-phase data.
3. **Pattern meta contract:** All detectors must populate all 10 standard meta fields when `found=True`. Missing fields produce silent failures in the trade enhancer and resolver.
4. **Gate metric registry:** New gate metrics must be registered in `GATE_METRIC_REGISTRY` with correct `context_paths` before being referenced in any gate config.
5. **Confidence arithmetic:** Penalties are signed (`-15`, not `15`). The `evaluate_confidence_modifier()` method auto-corrects positive `confidence_penalty` values with a CRITICAL log warning.

### Testing Patterns

The system does not ship with a test suite but the following isolated components are pure functions suitable for unit testing:

- `gate_evaluator.evaluate_gates()` — no dependencies
- `config_extractor.ConfigExtractor` — requires only the config dicts
- All pattern `detect()` methods — require a DataFrame and indicators dict
- `confidence_config.CONFIDENCE_CALCULATION_PIPELINE` — readable step-by-step for manual verification

For integration testing, initialize `ConfigResolver` with `MASTER_CONFIG` for the target horizon and call `build_evaluation_context_only()` with synthetic indicator data to trace the full 8-phase pipeline output.

---

*Pro Stock Analyzer v15.0 — NSE India — Quantitative Trading System*