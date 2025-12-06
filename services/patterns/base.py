from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import pandas as pd
import logging

logger = logging.getLogger("pattern_engine")

class BasePattern(ABC):
    """
    Abstract Base Class for all technical patterns.
    Enforces a strict output format and provides shared utilities.
    """
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.alias = "base_pattern"
        self.horizons_supported = ["intraday", "short_term", "swing", "long_term"]
        self.debug_mode = self.config.get("debug", False)
        self.coerce_numeric = self.config.get("coerce_numeric", True)
        self.numeric_cols = self.config.get("numeric_cols", ["Open", "High", "Low", "Close", "Volume"])

    @abstractmethod
    def detect(self, df: pd.DataFrame, indicators: Dict[str, Any], horizon: str) -> Dict[str, Any]:
        if getattr(self, "coerce_numeric", False) and df is not None:
            df = self.ensure_numeric_df(df)
        """
        Must return: { "found": bool, "score": float, "quality": float, "meta": {...} }
        """
        pass

    def _normalize_score(self, raw_score: float) -> float:
        """Clamps scores between 0 and 100. Handles None/NaN safely."""
        try:
            if raw_score is None: return 0.0
            return max(0.0, min(100.0, float(raw_score)))
        except (ValueError, TypeError):
            return 0.0
    def ensure_numeric_df(self, df: pd.DataFrame, cols: Optional[list] = None) -> pd.DataFrame:
        """
        Defensive coercion helper: ensures requested columns are numeric.
        - strips commas and percent signs, then uses pd.to_numeric(errors='coerce')
        - returns the modified DataFrame (shallow copy semantics)
        Usage: df = self.ensure_numeric_df(df)
        """
        if df is None:
            return df

        if cols is None:
            cols = self.numeric_cols or []

        # operate on a copy to avoid surprising caller-side mutation
        out = df.copy()
        try:
            for c in cols:
                if c not in out.columns:
                    continue

                # convert object-like columns (strings) into numeric safely
                # convert to string first so we can strip commas, percent symbols etc.
                if out[c].dtype == "object" or self.coerce_numeric:
                    cleaned = out[c].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()
                    out[c] = pd.to_numeric(cleaned, errors="coerce")
        except Exception as e:
            # Do not raise — log and return what we have
            self.log_debug(f"ensure_numeric_df error: {e}")
        return out

    def log_debug(self, message: str, extra: Dict = None):
        """Non-blocking debug logger hook."""
        if self.debug_mode:
            logger.debug(f"[{self.alias}] {message}", extra=extra)

    def _get_val(self, data: Dict[str, Any], key: str) -> Optional[float]:
        """Robust extractor for indicator dictionaries."""
        if key not in data: return None
        item = data[key]
        val = item.get("value") if isinstance(item, dict) else item
        try:
            f_val = float(val)
            import numpy as np
            if np.isnan(f_val) or np.isinf(f_val): return None
            return f_val
        except (ValueError, TypeError):
            return None