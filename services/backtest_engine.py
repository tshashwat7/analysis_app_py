# services/backtest_engine.py
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import pandas_ta as ta

from services.data_layer import ParquetStore

# Import signal engine functions. 
from services.signal_engine import (
    classify_setup,
    generate_trade_plan,
    compute_all_profiles,
    compute_trend_strength,
)

# Optional helper
try:
    from services.signal_engine import compute_volatility_quality
except Exception:
    compute_volatility_quality = None 

from config.constants import (
    HORIZON_FETCH_CONFIG,
    HORIZON_PROFILE_MAP,
    ATR_MULTIPLIERS,
)

logger = logging.getLogger("backtester_v6_1")
# logger.setLevel(logging.DEBUG) # Uncomment to see per-candle logic

# INCREASED CAPITAL to handle high-priced stocks (e.g. TCS > 3000)
CAPITAL_START = 1000000.0 
RISK_PER_TRADE_PCT = 1.0
MAX_CAPITAL_PER_TRADE = 0.25

SLIPPAGE_PCT = 0.0002
COMMISSION_PCT = 0.0002
MIN_COMMISSION = 0.0


def _normalize_horizon(h: str) -> str:
    if not h:
        return "short_term"
    h = h.lower().strip()
    if h == "shortterm":
        return "short_term"
    return h


class BacktestEngine:
    def __init__(self, symbol: str, horizon: str = "short_term", initial_capital: float = CAPITAL_START, debug: bool = False):
        self.symbol = symbol.upper()
        self.horizon = _normalize_horizon(horizon)
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.df: Optional[pd.DataFrame] = None
        self.results: List[Dict[str, Any]] = []
        self.active_trade: Optional[Dict[str, Any]] = None
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

        cfg = HORIZON_FETCH_CONFIG.get(self.horizon) or HORIZON_FETCH_CONFIG.get(self.horizon.replace("_", ""), {})
        self.interval = cfg.get("interval", "1d")
        
        # Hold limits
        self.max_hold_bars = cfg.get("max_hold_bars", 60)
        if self.horizon == "intraday":
            self.max_hold_bars = cfg.get("max_hold_bars", 10)
        elif self.horizon == "long_term":
            self.max_hold_bars = cfg.get("max_hold_bars", 120)

        # ATR multipliers fallback
        self.atr_mult = ATR_MULTIPLIERS.get(self.horizon) if isinstance(ATR_MULTIPLIERS, dict) else None
        if not self.atr_mult:
            self.atr_mult = ATR_MULTIPLIERS.get(self.horizon.replace("_", "")) if isinstance(ATR_MULTIPLIERS, dict) else None
        if not self.atr_mult:
            self.atr_mult = {"tp": 1.5, "sl": 1.0}

    # -------------------------
    # Data Loading & Indicators
    # -------------------------
    def load_and_prep_data(self) -> bool:
        logger.info(f"[{self.symbol}] Loading data for horizon '{self.horizon}' interval {self.interval}")
        self.df = ParquetStore.load_ohlcv(self.symbol, self.interval, lookback_days=None)
        if self.df is None or len(self.df) < 50:
            logger.warning(f"[{self.symbol}] Insufficient history ({None if self.df is None else len(self.df)} rows)")
            return False

        # --- 1. Macro Trend ---
        nifty_df = ParquetStore.load_ohlcv("^NSEI", self.interval, lookback_days=None)
        if nifty_df is not None and not nifty_df.empty:
            try:
                nifty_df['nifty_ema200'] = ta.ema(nifty_df['Close'], length=200)
                nifty_df['nifty_status'] = np.where(nifty_df['Close'] > nifty_df['nifty_ema200'], "Strong Uptrend", "Downtrend")
                self.df = self.df.join(nifty_df[['nifty_status']], how='left')
                self.df['nifty_status'] = self.df['nifty_status'].fillna("Neutral")
            except Exception:
                self.df['nifty_status'] = "Neutral"
        else:
            self.df['nifty_status'] = "Neutral"

        # --- 2. Horizon Specifics ---
        if self.horizon == "long_term":
            fast_len, slow_len, trend_len = 10, 40, 50
            ma_func = ta.wma
        else:
            fast_len, slow_len, trend_len = 20, 50, 200
            ma_func = ta.ema

        # --- 3. Moving Averages ---
        self.df['ema20'] = ma_func(self.df['Close'], length=fast_len)
        self.df['ema50'] = ma_func(self.df['Close'], length=slow_len)
        self.df['ema200'] = ma_func(self.df['Close'], length=trend_len)
        
        # [FIX 1] Calculate EMA20 Slope (Fast Slope) for Trend Strength
        try:
            self.df['ema_slope'] = ta.slope(self.df['ema20'], length=5)
        except Exception:
            self.df['ema_slope'] = self.df['ema20'].diff()

        try:
            self.df['dma_200_slope'] = ta.slope(self.df['ema200'], length=10)
        except Exception:
            self.df['dma_200_slope'] = self.df['ema200'].pct_change() * 100.0

        self.df['prev_close'] = self.df['Close'].shift(1)

        # --- 4. Momentum ---
        self.df['rsi'] = ta.rsi(self.df['Close'], length=14)
        try:
            self.df['rsi_slope'] = ta.slope(self.df['rsi'], length=5)
        except Exception:
            self.df['rsi_slope'] = self.df['rsi'].diff()

        macd = ta.macd(self.df['Close'])
        if macd is not None:
            hist_col = next((c for c in macd.columns if c.lower().startswith("macdh")), None)
            self.df['macd_hist'] = macd[hist_col] if hist_col else 0.0
        else:
            self.df['macd_hist'] = 0.0

        # ADX
        adx_df = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        if adx_df is not None:
            adx_col = next((c for c in adx_df.columns if c.upper().startswith("ADX")), None)
            self.df['adx'] = adx_df[adx_col] if adx_col else np.nan

        # BB & KC & Squeeze
        bb = ta.bbands(self.df['Close'], length=20, std=2.0)
        if bb is not None:
            self.df['bb_upper'] = bb[next(c for c in bb.columns if c.startswith("BBU"))]
            self.df['bb_lower'] = bb[next(c for c in bb.columns if c.startswith("BBL"))]
        
        try:
            kc = ta.kc(self.df['High'], self.df['Low'], self.df['Close'], length=20, scalar=1.5)
            if kc is not None:
                self.df['kc_upper'] = kc[next(c for c in kc.columns if c.startswith("KCU"))]
                self.df['kc_lower'] = kc[next(c for c in kc.columns if c.startswith("KCL"))]
        except Exception:
            self.df['kc_upper'] = np.nan
            self.df['kc_lower'] = np.nan

        # ATR
        try:
            self.df['atr_14'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        except Exception:
            self.df['atr_14'] = (self.df['High'] - self.df['Low']).rolling(14).mean()
        self.df['atr_pct'] = (self.df['atr_14'] / self.df['Close']) * 100.0

        # Volume & RVOL
        self.df['vol_avg'] = ta.sma(self.df['Volume'], length=20)
        self.df['rvol'] = self.df['Volume'] / (self.df['vol_avg'] + 1e-9)
        try:
            self.df['vwap'] = ta.vwap(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])
        except Exception:
            self.df['vwap'] = self.df['ema20']

        # Supertrend
        try:
            st = ta.supertrend(self.df['High'], self.df['Low'], self.df['Close'], length=10, multiplier=3.0)
            if st is not None:
                dir_col = next((c for c in st.columns if "d_" in c), None)
                if dir_col:
                    self.df['supertrend_signal'] = np.where(st[dir_col] > 0, "Bullish", "Bearish")
        except Exception:
            self.df['supertrend_signal'] = None

        # PSAR
        try:
            psar_df = ta.psar(self.df['High'], self.df['Low'], self.df['Close'])
            if psar_df is not None:
                long_col = next((c for c in psar_df.columns if c.startswith("PSARl")), None)
                short_col = next((c for c in psar_df.columns if c.startswith("PSARs")), None)
                if long_col:
                    self.df['psar'] = psar_df[long_col]
                elif short_col:
                    self.df['psar'] = psar_df[short_col]
                else:
                    self.df['psar'] = np.nan
        except Exception:
            self.df['psar'] = np.nan

        self.df.dropna(inplace=True)
        return True

    # -------------------------
    # Payload Builder
    # -------------------------
    def _row_to_payload(self, row: pd.Series) -> Dict[str, Any]:
        # Squeeze logic
        ttm_squeeze = "squeeze off"
        try:
            if (row.get('bb_upper', np.nan) < row.get('kc_upper', np.nan)) and (row.get('bb_lower', np.nan) > row.get('kc_lower', np.nan)):
                ttm_squeeze = "squeeze on"
        except Exception:
            pass

        # PSAR
        psar_val = row.get('psar')
        psar_trend = "bear"
        if psar_val is not None and row['Close'] > psar_val:
            psar_trend = "bull"

        # [FIX] Compute Trend Strength using EMA SLOPE (not DMA200 slope)
        try:
            trend_comp = compute_trend_strength({
                "adx": row.get('adx'),
                "ema_slope": row.get('ema_slope'), # Passing the correct slope now
                "di_plus": None,
                "di_minus": None,
                "supertrend_signal": row.get('supertrend_signal')
            })
            trend_raw = trend_comp.get("raw")
            trend_value = trend_comp.get("value")
            trend_score = trend_comp.get("score")
        except Exception:
            trend_raw = row.get('adx')
            trend_value = 0
            trend_score = 0

        # MACD
        macd_hist = row.get('macd_hist', 0)
        macd_cross_raw = 1 if macd_hist > 0 else 0
        macd_cross_score = 9 if macd_cross_raw == 1 else 2

        # DMA Cross
        dma20 = row.get('ema20')
        dma50 = row.get('ema50')
        dma_cross_raw = 1 if (dma20 and dma50 and dma20 > dma50) else 0
        dma_cross_score = 8 if dma_cross_raw == 1 else 3

        # Price vs 200
        ema200 = row.get('ema200')
        price_vs_200_raw = ((row['Close'] / ema200) - 1) * 100 if ema200 else 0.0
        price_vs_200_score = 10 if price_vs_200_raw > 10 else 7 if price_vs_200_raw > 0 else 3

        # Bollinger Metrics
        bb_upper = row.get('bb_upper')
        bb_lower = row.get('bb_lower')
        bb_percent_b_raw = None
        bb_width_raw = None
        
        if bb_upper is not None and bb_lower is not None and (bb_upper - bb_lower) != 0:
            bb_percent_b_raw = (row['Close'] - bb_lower) / (bb_upper - bb_lower)
            bb_width_raw = (bb_upper - bb_lower) / row['Close']
            
        bb_percent_b_score = 0
        if bb_percent_b_raw is not None:
            bb_percent_b_score = 10 if bb_percent_b_raw > 0.8 else 7 if bb_percent_b_raw > 0.5 else 3

        bb_width_score = 5
        if bb_width_raw is not None:
            bb_width_score = 10 if bb_width_raw < 0.01 else 7 if bb_width_raw < 0.02 else 3

        # Volatility Quality
        volq_raw = row.get('atr_pct', 1.0)
        volq_score = 5
        if compute_volatility_quality:
            try:
                vq = compute_volatility_quality({
                    "atr_pct": row.get('atr_pct'),
                    "bb_width": bb_width_raw,
                    "true_range": row.get('atr_14'),
                    "ttm_squeeze": ttm_squeeze
                })
                volq_raw = vq.get('raw')
                volq_score = vq.get('score')
            except Exception:
                pass

        reg_slope_raw = row.get('dma_200_slope', 0)
        reg_slope_score = 8 if reg_slope_raw > 1.0 else 5 if reg_slope_raw > 0 else 2

        payload = {
            "Price": {"raw": float(row['Close']), "value": float(row['Close']), "score": 5},
            "Open": {"value": float(row['Open'])},
            "prev_close": {"value": row.get('prev_close')},

            "ema_20": {"raw": dma20, "value": dma20},
            "ema_50": {"raw": dma50, "value": dma50},
            "ema_200": {"raw": ema200, "value": ema200},
            "dma_200": {"raw": ema200, "value": ema200},
            "dma_200_slope": {"raw": reg_slope_raw, "value": reg_slope_raw, "score": reg_slope_score},
            # [FIX] Passing correct short-term slope for trend calculations
            "ema_slope": {"value": row.get('ema_slope')},

            "vwap": {"value": row.get('vwap')},

            "bb_high": {"value": row.get('bb_upper')},
            "bb_low": {"value": row.get('bb_lower')},
            "bb_percent_b": {"raw": bb_percent_b_raw, "value": bb_percent_b_raw, "score": bb_percent_b_score},
            "bb_width": {"raw": bb_width_raw, "value": bb_width_raw, "score": bb_width_score},
            "ttm_squeeze": {"raw": ttm_squeeze, "value": ttm_squeeze},

            "macd_histogram": {"raw": macd_hist, "value": macd_hist},
            "macd_cross": {"raw": macd_cross_raw, "value": macd_cross_raw, "score": macd_cross_score},

            "rsi": {"raw": row.get('rsi'), "value": row.get('rsi')},
            "rsi_slope": {"raw": row.get('rsi_slope'), "value": row.get('rsi_slope')},

            "rvol": {"raw": row.get('rvol'), "value": row.get('rvol')},

            "atr_14": {"raw": row.get('atr_14'), "value": row.get('atr_14')},
            "atr_pct": {"raw": row.get('atr_pct'), "value": row.get('atr_pct')},
            "volatility_quality": {"raw": volq_raw, "value": volq_raw, "score": volq_score},

            "adx": {"raw": row.get('adx'), "value": row.get('adx')},

            "trend_strength": {"raw": trend_raw, "value": trend_value, "score": trend_score},

            "dma_20_50_cross": {"raw": dma_cross_raw, "value": dma_cross_raw, "score": dma_cross_score},
            "price_vs_200dma_pct": {"raw": price_vs_200_raw, "value": price_vs_200_raw, "score": price_vs_200_score},
            "reg_slope": {"raw": reg_slope_raw, "value": reg_slope_raw, "score": reg_slope_score},

            "psar_trend": {"raw": psar_trend, "value": psar_trend},
            "psar_level": {"raw": psar_val, "value": psar_val},

            "resistance_1": {"value": row['Close'] * 1.05},
            "resistance_2": {"value": row['Close'] * 1.10},
            "resistance_3": {"value": row['Close'] * 1.15},
            "support_1": {"value": row['Close'] * 0.95},
            "support_2": {"value": row['Close'] * 0.90},
            "support_3": {"value": row['Close'] * 0.85},

            "nifty_trend_score": {"value": row.get('nifty_status', "Neutral")}
        }
        return payload

    # -------------------------
    # Position Sizing
    # -------------------------
    def _calculate_position_size(self,
                                 entry_price: float,
                                 stop_loss: Optional[float],
                                 plan_position_pct: Optional[float] = None) -> int:
        if entry_price is None or entry_price <= 0:
            return 0

        shares = 0
        
        # Method 1: Plan % (from signal engine)
        if plan_position_pct is not None and plan_position_pct > 0:
            pct = plan_position_pct if plan_position_pct <= 1.0 else plan_position_pct / 100.0
            pos_value = self.current_capital * pct
            shares = int(pos_value / entry_price)
            
        # Method 2: Risk-based fallback
        elif stop_loss is not None:
            risk_amount = self.current_capital * (RISK_PER_TRADE_PCT / 100.0)
            risk_per_share = abs(entry_price - stop_loss)
            if risk_per_share > 0:
                shares = int(risk_amount / risk_per_share)

        # [FIX] Minimum Shares Logic for Backtest
        # If signal is valid but capital is too low for 1 share of high-priced stock,
        # force 1 share to allow backtest to show results (real-world you would skip).
        if shares == 0:
            # Check if we have enough cash for just 1 share
            if self.current_capital >= entry_price:
                shares = 1

        # Cap max exposure
        max_cost = self.current_capital * MAX_CAPITAL_PER_TRADE
        if shares * entry_price > max_cost:
            shares = int(max_cost / entry_price)
            
        return max(shares, 0)

    # -------------------------
    # Fees & Slippage
    # -------------------------
    def _apply_slippage_and_fees(self, price: float, shares: int, action: str = "BUY") -> Tuple[float, float]:
        if shares <= 0:
            return price, 0.0
        if action.upper() == "BUY":
            exec_price = price * (1.0 + SLIPPAGE_PCT)
        else:
            exec_price = price * (1.0 - SLIPPAGE_PCT)
        gross = exec_price * shares
        commission = max(gross * COMMISSION_PCT, MIN_COMMISSION)
        return exec_price, commission

    # -------------------------
    # Main Loop
    # -------------------------
    def run(self):
        if not self.load_and_prep_data():
            logger.warning(f"[{self.symbol}] Skipping (no data)")
            return

        self.active_trade = None

        for i in range(1, len(self.df) - 1):
            timestamp = self.df.index[i]
            row = self.df.iloc[i]
            next_row = self.df.iloc[i + 1]
            prev_row = self.df.iloc[i - 1] if i - 1 >= 0 else None

            # 1. Manage existing
            if self.active_trade:
                self._manage_trade(self.active_trade, row, timestamp)
                if self.active_trade.get("status") and self.active_trade["status"] != "OPEN":
                    self.results.append(self.active_trade)
                    self.active_trade = None
                continue

            # 2. Indicators & Profiles
            indicators_payload = self._row_to_payload(row)
            fundamentals = {} 
            profiles_summary = compute_all_profiles(self.symbol, fundamentals, indicators_payload, profile_map=None)
            profile_report = profiles_summary["profiles"].get(self.horizon)

            if not profile_report:
                continue

            # 3. Classify Setup
            setup_type = classify_setup(indicators_payload)

            # Safeguard for DEEP_PULLBACK
            if setup_type == "DEEP_PULLBACK":
                is_green = row['Close'] > row['Open']
                rsi = row.get('rsi', 0)
                if not (is_green or (rsi and rsi > 48)):
                    continue

            # 4. Generate Plan
            plan = generate_trade_plan(profile_report, indicators_payload, macro_trend_status=row.get('nifty_status', "Neutral"))

            # 5. Filter Signals
            if not plan or plan.get("signal") in ("WAIT", "HOLD_NO_RISK", "NO_TRADE") or str(plan.get("signal")).startswith("WAIT"):
                continue

            if plan.get("analytics", {}).get("skipped_low_rr", False):
                continue

            # 6. Execute Entry (Next Open)
            entry_price = float(next_row.get("Open") or next_row.get("Close"))
            stop_loss = plan.get("stop_loss")
            
            # Auto-calculate SL if missing (ATR fallback)
            if stop_loss is None:
                atr = indicators_payload.get("atr_14", {}).get("value")
                if atr:
                    stop_loss = entry_price - (self.atr_mult.get("sl", 1.0) * atr)

            plan_pos_pct = plan.get("position_size")
            if plan_pos_pct is not None and plan_pos_pct > 1: 
                plan_pos_pct = plan_pos_pct / 100.0

            shares = self._calculate_position_size(entry_price, stop_loss, plan_position_pct=plan_pos_pct)
            
            if shares <= 0:
                continue

            real_entry_price, entry_fee = self._apply_slippage_and_fees(entry_price, shares, "BUY")

            self.active_trade = {
                "symbol": self.symbol,
                "entry_date": self.df.index[i + 1],
                "entry_price": round(real_entry_price, 4),
                "shares": int(shares),
                "entry_fee": round(entry_fee, 2),
                "stop_loss": stop_loss,
                "target": plan.get("targets", {}).get("t1"),
                "target_2": plan.get("targets", {}).get("t2"),
                "setup": setup_type,
                "signal": plan.get("signal"),
                "status": "OPEN",
                "bars_held": 0,
                "plan": plan,
            }

        # Close at end
        if self.active_trade:
            last_row = self.df.iloc[-1]
            self._manage_trade(self.active_trade, last_row, self.df.index[-1])
            if self.active_trade.get("status") != "OPEN":
                self.results.append(self.active_trade)

    # -------------------------
    # Trade Management
    # -------------------------
    def _manage_trade(self, trade: Dict[str, Any], row: pd.Series, timestamp):
        trade["bars_held"] += 1
        low = row.get("Low")
        high = row.get("High")
        close = row.get("Close")
        
        exit_price = None
        exit_reason = None

        # Stop Loss
        if low is not None and low <= trade.get("stop_loss", -999):
            exit_reason = "LOSS"
            # Gap handling
            exit_price = row.get("Open") if row.get("Open") < trade.get("stop_loss") else trade.get("stop_loss")

        # Targets
        elif trade.get("target_2") and high >= trade.get("target_2"):
            exit_reason = "WIN_T2"
            exit_price = trade.get("target_2")
        elif trade.get("target") and high >= trade.get("target"):
            exit_reason = "WIN"
            exit_price = trade.get("target")

        # Timeout
        elif trade["bars_held"] > self.max_hold_bars:
            exit_reason = "TIMEOUT"
            exit_price = close

        # Trailing SL (Move to BE)
        else:
            dist = (trade.get("target", 0) - trade["entry_price"])
            profit = (close - trade["entry_price"])
            if dist > 0 and profit > (dist * 0.75):
                be_price = trade["entry_price"] * (1.0 + SLIPPAGE_PCT + 0.001)
                if be_price > trade["stop_loss"]:
                    trade["stop_loss"] = round(be_price, 2)

        if exit_price is not None:
            trade["status"] = exit_reason
            real_exit, exit_fee = self._apply_slippage_and_fees(exit_price, trade["shares"], "SELL")
            trade["exit_price"] = round(real_exit, 4)
            trade["exit_date"] = timestamp
            
            pnl = (real_exit - trade["entry_price"]) * trade["shares"]
            trade["net_pnl"] = round(pnl - trade.get("entry_fee", 0) - exit_fee, 2)
            trade["pnl_pct"] = (trade["net_pnl"] / (trade["entry_price"] * trade["shares"])) * 100
            
            self.current_capital += trade["net_pnl"]

    def summarize(self) -> Dict[str, Any]:
        trades = [t for t in self.results if t.get("net_pnl") is not None]
        total = len(trades)
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        
        win_rate = (len(wins) / total * 100) if total else 0.0
        pnl = sum(t["net_pnl"] for t in trades) if trades else 0.0
        
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "total_trades": total,
            "win_rate": round(win_rate, 1),
            "net_pnl": round(pnl, 2),
            "ending_capital": round(self.current_capital, 2),
            "reward_risk": round( (sum(t['net_pnl'] for t in wins)/len(wins)) / abs(sum(t['net_pnl'] for t in losses)/len(losses)), 2) if wins and losses else 0
        }