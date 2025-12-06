from typing import Optional


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
