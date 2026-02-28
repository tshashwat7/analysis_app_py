import logging
from typing import Dict, Any, List
import pandas as pd

# Import your patterns
from services.patterns.bollinger_squeeze import BollingerSqueeze
from services.patterns.darvas import DarvasBoxPattern        
from services.patterns.flag_pennant import FlagPennantPattern # <--- NEW
from services.patterns.minervini_vcp import MinerviniVCPPattern
from services.patterns.cup_handle import CupHandlePattern
from services.patterns.three_line_strike import ThreeLineStrikePattern
from services.patterns.ichimoku_signals import IchimokuSignals
from services.patterns.golden_cross import GoldenDeathCross
from services.patterns.double_top_bottom import DoubleTopBottom
# from services.patterns.head_shoulders import HeadShouldersPattern ## removed to avoid pattern noise
# from services.patterns.engulfing import EngulfingPattern


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
        ]

    def analyze(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        raw_results = {}
        
        for detector in self.detectors:
            try:
                det_result = detector.detect(df, indicators, horizon)
                if det_result["found"]:
                    raw_results[detector.alias] = det_result
            except Exception as e:
                logger.error(f"Error in pattern {detector.alias}: {e}")

        # FUSION: Now returns the ENHANCED "P" structure
        enhanced_results = merge_pattern_into_indicators(indicators, raw_results, horizon=horizon)
        
        return enhanced_results # Both indicators and the return value now match

# Singleton instance to avoid re-initializing classes constantly
analyzer = PatternAnalyzer()

def run_pattern_analysis(df, indicators, horizon):
    return analyzer.analyze(df, indicators, horizon)