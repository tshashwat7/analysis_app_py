import logging
from typing import Dict, Any, List
import pandas as pd

# Import your patterns
from services.patterns.bollinger_squeeze import BollingerSqueeze
from services.patterns.darvas import DarvasBoxPattern        # <--- NEW
from services.patterns.flag_pennant import FlagPennantPattern # <--- NEW
from services.patterns.minervini_vcp import MinerviniVCPPattern
from services.patterns.cup_handle import CupHandlePattern
from services.patterns.three_line_strike import ThreeLineStrikePattern
from services.patterns.ichimoku_signals import IchimokuSignals
from services.patterns.golden_cross import GoldenDeathCross
from services.patterns.double_top_bottom import DoubleTopBottom
from services.patterns.head_shoulders import HeadShouldersPattern
from services.patterns.engulfing import EngulfingPattern

# Import Fusion
from services.fusion.pattern_fusion import merge_pattern_into_indicators

logger = logging.getLogger(__name__)

class PatternAnalyzer:
    def __init__(self):
        # Initialize detectors once
        self.detectors = [
            BollingerSqueeze(config={"squeeze_threshold": 0.08}), # Strict 8% width
            DarvasBoxPattern(),
            FlagPennantPattern(),
            MinerviniVCPPattern(),
            CupHandlePattern(),
            ThreeLineStrikePattern(),
            IchimokuSignals(),
            GoldenDeathCross(),
            DoubleTopBottom(),
            HeadShouldersPattern(),
            EngulfingPattern()
        ]

    def analyze(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        """
        Runs all configured pattern detectors and merges results into indicators.
        """
        results = {}
        
        for detector in self.detectors:
            try:
                # Run Detection
                det_result = detector.detect(df, indicators, horizon)
                
                if det_result["found"]:
                    results[detector.alias] = det_result
                    
            except Exception as e:
                logger.error(f"Error in pattern {detector.alias}: {e}")

        # FUSION: Inject findings directly into indicators
        # This makes the rest of your system (Scoring/UI) see patterns as metrics
        merge_pattern_into_indicators(indicators, results)
        
        return results

# Singleton instance to avoid re-initializing classes constantly
analyzer = PatternAnalyzer()

def run_pattern_analysis(df, indicators, horizon):
    return analyzer.analyze(df, indicators, horizon)