"""
services/analyzers/pattern_analyzer.py

Changes from previous version
──────────────────────────────
  Import fix:   GoldenDeathCross  → GoldenCross + DeathCross (split to match matrix aliases)
  Import fix:   DoubleTopBottom   → BullishNecklinePattern + BearishNecklinePattern
  Alias fix:    detectors list reflects both cross classes so "deathCross" populates correctly
  Return fix:   analyze() returns raw_results (the pattern dict), NOT enhanced_results (None).
                merge_pattern_into_indicators mutates indicators in-place by design.
                Caller (compute_indicators) receives the tuple (indicators, patterns)
                where indicators is already mutated and patterns = raw_results.
"""

import logging
from typing import Dict, Any
import pandas as pd

from services.patterns.bollinger_squeeze    import BollingerSqueeze
from services.patterns.darvas             import DarvasBoxPattern
from services.patterns.flag_pennant       import FlagPennantPattern
from services.patterns.minervini_vcp      import MinerviniVCPPattern
from services.patterns.cup_handle         import CupHandlePattern
from services.patterns.three_line_strike  import ThreeLineStrikePattern
from services.patterns.ichimoku_signals   import IchimokuSignals
# Split classes: GoldenCross (alias "goldenCross") + DeathCross (alias "deathCross")
from services.patterns.golden_cross import GoldenCross, DeathCross
# Split classes: BullishNecklinePattern + BearishNecklinePattern
from services.patterns.double_top_bottom import (
    BullishNecklinePattern,
    BearishNecklinePattern,
)
from services.patterns.momentum_flow  import MomentumFlowPattern
from services.fusion.pattern_fusion import merge_pattern_into_indicators

logger = logging.getLogger(__name__)


class PatternAnalyzer:
    def __init__(self):
        self.detectors = [
            BollingerSqueeze(config={"squeeze_threshold": 0.08}),
            DarvasBoxPattern(),
            FlagPennantPattern(),
            MinerviniVCPPattern(),
            CupHandlePattern(),
            ThreeLineStrikePattern(),
            IchimokuSignals(),
            GoldenCross(),     # alias "goldenCross"
            DeathCross(),      # alias "deathCross" — separate so matrix CONFLICTING lists work
            BullishNecklinePattern(),
            BearishNecklinePattern(),
            MomentumFlowPattern(),
        ]
        # ✅ P3-2: Circuit breaker for failing detectors
        self.failure_counts: Dict[str, int] = {}
        self.max_failures = 3

    @classmethod
    def get_active_aliases(cls) -> list:
        """Return the aliases of all detectors registered in the analyzer.
        
        This serves as the single source of truth for the config extractor's
        completeness validation at startup.
        """
        # We instantiate a temporary instance to avoid duplicating the detectors list
        # since aliases are instance attributes.
        temp = cls()
        return [d.alias for d in temp.detectors]

    def analyze(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        raw_results: Dict[str, Any] = {}

        for detector in self.detectors:
            # ✅ P3-2: Circuit breaker check
            if self.failure_counts.get(detector.alias, 0) >= self.max_failures:
                continue

            try:
                det_result = detector.detect(df, indicators, horizon)
                if det_result.get("found"):
                    raw_results[detector.alias] = det_result
                # Reset failure count on success
                self.failure_counts[detector.alias] = 0
            except Exception as e:
                self.failure_counts[detector.alias] = self.failure_counts.get(detector.alias, 0) + 1
                logger.error(f"Error in pattern {detector.alias} (Failure {self.failure_counts[detector.alias]}): {e}", exc_info=True)
                # ✅ Fix 10: Do NOT inject _error_ sentinel into raw_results.
                # The error is already logged and tracked in failure_counts (circuit breaker).
                # Injecting a non-standard schema entry would pollute raw_results for
                # any downstream consumer that iterates the dict without key-prefix filtering.


        # Mutates indicators in-place (returns None by design).
        # Do NOT assign the return value — it is always None.
        merge_pattern_into_indicators(indicators, raw_results, horizon=horizon, df=df)

        # Return raw_results so callers that capture the return value get something
        # useful, without duplicating the full indicators dict.
        return raw_results


# Singleton — avoids re-initialising detector instances on every call
analyzer = PatternAnalyzer()


def run_pattern_analysis(
    df: pd.DataFrame,
    indicators: Dict[str, Any],
    horizon: str,
) -> Dict[str, Any]:
    return analyzer.analyze(df, indicators, horizon)