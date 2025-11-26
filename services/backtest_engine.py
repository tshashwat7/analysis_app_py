import pandas as pd
import pandas_ta as ta
import numpy as np
import logging
from typing import Dict, Any

# Import your infrastructure
from services.data_layer import ParquetStore
from services.signal_engine import classify_setup, generate_trade_plan
from config.constants import HORIZON_FETCH_CONFIG, HORIZON_PROFILE_MAP

logger = logging.getLogger("backtester")

class BacktestEngine:
    def __init__(self, symbol: str, horizon: str = "short_term"):
        self.symbol = symbol.upper()
        self.horizon = horizon
        self.results = []
        self.df = None
        
        config = HORIZON_FETCH_CONFIG.get(horizon, HORIZON_FETCH_CONFIG["short_term"])
        self.interval = config.get("interval", "1d")
        self.profile = HORIZON_PROFILE_MAP.get(horizon, HORIZON_PROFILE_MAP["short_term"])

    def load_and_prep_data(self):
        logger.info(f"[{self.symbol}] Loading {self.horizon} data ({self.interval})...")
        
        # 1. Load Stock Data
        self.df = ParquetStore.load_ohlcv(self.symbol, self.interval, lookback_days=None) 
        if self.df is None or len(self.df) < 200:
            logger.warning(f"[{self.symbol}] Insufficient history")
            return False
        
        # Initial VWAP Calculation
        self.df['vwap'] = ta.vwap(self.df['High'], self.df['Low'], self.df['Close'], self.df['Volume'])

        # 2. Load Nifty Data (For Macro Trend Simulation)
        nifty_df = ParquetStore.load_ohlcv("^NSEI", self.interval, lookback_days=None)
        if nifty_df is not None and not nifty_df.empty:
            # Calc Nifty 200 EMA
            nifty_df['nifty_ema200'] = ta.ema(nifty_df['Close'], length=200)
            # Determine Trend: "Strong Uptrend" if > EMA200, else "Downtrend"
            nifty_df['nifty_status'] = np.where(
                nifty_df['Close'] > nifty_df['nifty_ema200'], 
                "Strong Uptrend", 
                "Downtrend"
            )
            # Join Nifty Status
            self.df = self.df.join(nifty_df[['nifty_status']], how='left')
            self.df['nifty_status'] = self.df['nifty_status'].fillna("Neutral")
        else:
            self.df['nifty_status'] = "Neutral"
        
        self.df['prev_close'] = self.df['Close'].shift(1)
        # --- VECTORIZED INDICATORS ---
        
        # Trend
        self.df['ema20'] = ta.ema(self.df['Close'], length=20)
        self.df['ema50'] = ta.ema(self.df['Close'], length=50)
        self.df['ema200'] = ta.ema(self.df['Close'], length=200)
        
        # --- MISSING LINK: 200 EMA SLOPE ---
        # We calculate slope over 10 periods
        self.df['ema200_slope'] = ta.slope(self.df['ema200'], length=10)
        
        # Momentum
        self.df['rsi'] = ta.rsi(self.df['Close'], length=14)
        self.df['rsi_slope'] = ta.slope(self.df['rsi'], length=5) 
        
        macd = ta.macd(self.df['Close'])
        if macd is not None:
            # MACD returns 3 columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
            hist_col = next((c for c in macd.columns if c.startswith("MACDh")), None)
            self.df['macd_hist'] = macd[hist_col] if hist_col else 0
        
        # ADX
        adx_df = ta.adx(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        if adx_df is not None:
            self.df['adx'] = adx_df[next(c for c in adx_df.columns if c.startswith("ADX"))]

        # Volatility
        bb = ta.bbands(self.df['Close'], length=20, std=2.0)
        if bb is not None:
            self.df['bb_upper'] = bb[next(c for c in bb.columns if c.startswith("BBU"))]
            self.df['bb_lower'] = bb[next(c for c in bb.columns if c.startswith("BBL"))]
            
        kc = ta.kc(self.df['High'], self.df['Low'], self.df['Close'], length=20, scalar=1.5)
        if kc is not None:
            self.df['kc_upper'] = kc[next(c for c in kc.columns if c.startswith("KCU"))]
            self.df['kc_lower'] = kc[next(c for c in kc.columns if c.startswith("KCL"))]
        
        self.df['atr'] = ta.atr(self.df['High'], self.df['Low'], self.df['Close'], length=14)
        
        # Volume
        self.df['vol_avg'] = ta.sma(self.df['Volume'], length=20)
        self.df['rvol'] = self.df['Volume'] / self.df['vol_avg']

        # SuperTrend
        st = ta.supertrend(self.df['High'], self.df['Low'], self.df['Close'], length=10, multiplier=3.0)
        if st is not None:
            dir_col = next(c for c in st.columns if "d_" in c) 
            self.df["supertrend_signal"] = np.where(
                st[dir_col] > 0, "Bullish", "Bearish"
            )

        # PSAR
        psar_df = ta.psar(self.df['High'], self.df['Low'], self.df['Close'])
        if psar_df is not None:
            long_col = next((c for c in psar_df.columns if c.startswith("PSARl")), None)
            short_col = next((c for c in psar_df.columns if c.startswith("PSARs")), None)
            if long_col and short_col:
                self.df['psar'] = psar_df[long_col].fillna(psar_df[short_col])
            elif long_col: self.df['psar'] = psar_df[long_col]
            elif short_col: self.df['psar'] = psar_df[short_col]
        

        self.df.dropna(inplace=True)
        return True

    def _row_to_payload(self, row) -> Dict[str, Any]:
        """
        Maps row data to the EXACT keys expected by signal_engine.py
        Ref: classify_setup() and generate_trade_plan() keys.
        """
        squeeze_status = "Squeeze Off"
        if 'bb_upper' in row and 'kc_upper' in row:
            if (row['bb_upper'] < row['kc_upper']) and (row['bb_lower'] > row['kc_lower']):
                squeeze_status = "Squeeze On"

        psar_val = row.get('psar')
        psar_trend = "Bearish"
        if psar_val and row['Close'] > psar_val:
            psar_trend = "Bullish"
        prev_close_val = row.get('prev_close')
        # --- CRITICAL FIX: MAPPING TO SNAKE_CASE KEYS ---
        return {
            "Price": {"value": row['Close']},
            "Open": {"value": row['Open']},         
            "prev_close": {"value": prev_close_val},
            # Moving Averages (Snake Case)
            "ema_20": {"value": row.get('ema20')},
            "ema_50": {"value": row.get('ema50')},
            "ema_200": {"value": row.get('ema200')},
            "dma_200": {"value": row.get('ema200')}, # Fallback
            "dma_200_slope": {"value": row.get('ema200_slope')}, # NEW for trend filter
            "vwap": {"value": row.get('vwap')},
            # Bollinger
            "bb_high": {"value": row.get('bb_upper')},
            
            # Momentum
            "rsi": {"value": row.get('rsi')},
            "rsi_slope": {"value": row.get('rsi_slope')},
            "macd_histogram": {"value": row.get('macd_hist')},
            
            # Volatility / Squeeze
            "ttm_squeeze": {"value": squeeze_status},
            "rvol": {"value": row.get('rvol')},
            "atr_14": {"value": row.get('atr')},
            "adx": {"value": row.get('adx')},
            
            # Signals & Context
            "supertrend_signal": {"value": row.get('supertrend_signal')},
            "nifty_trend_score": {"desc": row.get('nifty_status', "Neutral")},
            "volatility_quality": {"value": 5.0}, # Mocked for backtest speed
            
            "psar_trend": {"value": psar_trend},
            "psar_level": {"value": psar_val},
            
            # Mock Pivots
            "resistance_1": {"value": row['Close'] * 1.05},
            "resistance_2": {"value": row['Close'] * 1.10},
            "resistance_3": {"value": row['Close'] * 1.15},
            "support_1": {"value": row['Close'] * 0.95},
            "support_2": {"value": row['Close'] * 0.90},
            "support_3": {"value": row['Close'] * 0.85},
            "trend_strength": {"value": (row.get('adx', 0) / 5) if row.get('adx') else 0},
        }

    def run(self):
        if not self.load_and_prep_data():
            return

        active_trade = None
        
        # Start from index 1 to allow previous candle checks
        for i in range(1, len(self.df)):
            timestamp = self.df.index[i]
            row = self.df.iloc[i]
            prev_row = self.df.iloc[i-1] # For confirmation checks
            
            # 1. Manage Active Trade
            if active_trade:
                self._manage_trade(active_trade, row, timestamp)
                if active_trade['status'] != 'OPEN':
                    self.results.append(active_trade)
                    active_trade = None
                continue

            # 2. Generate Payload
            indicators_payload = self._row_to_payload(row)
            
            # 3. Classify
            setup_type = classify_setup(indicators_payload)
            
            if setup_type not in ["GENERIC", "NEUTRAL / CHOPPY", "NO_TRADE"]:
                
                # --- CONFIRMATION LOGIC ---
                # Don't enter strictly on the signal candle; wait for close logic or reversal
                is_green = row['Close'] > row['Open']
                is_reversal = row['Close'] > prev_row['High']
                
                # Deep Pullback MUST have a reversal confirmation
                if setup_type == "DEEP_PULLBACK" and not (is_green or is_reversal):
                    continue

                # Mock profile
                mock_profile = {"final_score": 8.5, "category": "BUY", "profile": self.horizon}
                macro_status = row.get('nifty_status', "Neutral")
                
                # 4. Plan Trade
                plan = generate_trade_plan(mock_profile, indicators_payload, macro_trend_status=macro_status)
                
                # Only take BUYs for this simplified backtest
                if plan['signal'].startswith("BUY"):
                    active_trade = {
                        "symbol": self.symbol,
                        "entry_date": timestamp,
                        "entry_price": row['Close'], # Market Buy on Close
                        "setup": setup_type,
                        "stop_loss": plan['stop_loss'],
                        "original_sl": plan['stop_loss'],
                        "target": plan['targets']['t1'],
                        "status": "OPEN",
                        "bars_held": 0,
                        "rr": plan['rr_ratio']
                    }

    def _manage_trade(self, trade, row, timestamp):
        trade['bars_held'] += 1
        current_low = row['Low']
        current_high = row['High']
        
        # 1. Stop Loss
        if current_low <= trade['stop_loss']:
            trade['status'] = "LOSS"
            exit_p = min(row['Open'], trade['stop_loss']) if row['Open'] < trade['stop_loss'] else trade['stop_loss']
            trade['exit_price'] = exit_p
            trade['exit_date'] = timestamp
            trade['pnl_pct'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
            return

        # 2. Target
        if current_high >= trade['target']:
            trade['status'] = "WIN"
            trade['exit_price'] = trade['target']
            trade['exit_date'] = timestamp
            trade['pnl_pct'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100
            return

        # 3. Relaxed Breakeven (0.85 threshold)
        dist_to_target = trade['target'] - trade['entry_price']
        current_profit = row['Close'] - trade['entry_price']
        
        if current_profit > (dist_to_target * 0.85):
             trade['stop_loss'] = max(trade['stop_loss'], trade['entry_price'] * 1.001)

        # 4. Time Stop
        max_bars = 60 if "m" in self.interval else 40
        if trade['bars_held'] > max_bars:
            trade['status'] = "TIMEOUT"
            trade['exit_price'] = row['Close']
            trade['exit_date'] = timestamp
            trade['pnl_pct'] = (trade['exit_price'] - trade['entry_price']) / trade['entry_price'] * 100