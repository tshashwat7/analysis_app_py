"""
Phase 3 Backtest: Full Engine Simulation
- DIRECTLY calls generate_trade_plan from signal_engine.py.
- Validates Resistance Guards, Symmetric Logic, and Risk Clamps.
- Handles both LONG and SHORT trades.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
from services.data_layer import ParquetStore

# IMPORT YOUR ENGINE LOGIC
# We now import generate_trade_plan to ensure 1:1 parity with Live Trading
from services.signal_engine import (
    generate_trade_plan, 
    compute_trend_strength,
    compute_momentum_strength
)

# Helpers to mock the 'profile_report' expected by generate_trade_plan
def mock_profile_report(indicators):
    # A simplified mock. In a real backtest, you might want to run full profiling,
    # but for signal logic validation, we primarily need the score to not be 0.
    # We derive a synthetic score from trend/momentum to feed the engine.
    ts = indicators.get("trend_strength", 0)
    ms = indicators.get("momentum_strength", 0)
    base = (ts + ms) / 2.0
    
    # Map 0-100 scales down to 0-10
    final_score = base / 10.0
    
    cat = "HOLD"
    if final_score >= 7.0: cat = "BUY"
    elif final_score <= 3.0: cat = "SELL"
    
    return {
        "final_score": round(final_score, 2),
        "category": cat
    }

class FullEngineStrategy(Strategy):
    """
    Simulation Proxy:
    Passes every candle to signal_engine.py and executes strictly based on the plan.
    """
    
    # Parameters
    horizon = "short_term" 
    
    def init(self):
        """Pre-calculate vector indicators to speed up the loop"""
        c = pd.Series(self.data.Close)
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)
        o = pd.Series(self.data.Open)

        # --- 1. Moving Averages ---
        self.ema20 = self.I(ta.ema, c, length=20)
        self.ema50 = self.I(ta.ema, c, length=50) 
        self.ema200 = self.I(ta.ema, c, length=200)

        # --- 2. Momentum ---
        self.rsi = self.I(ta.rsi, c, length=14)
        def get_macd_hist(ser): return ta.macd(ser).iloc[:, 1]
        self.macd_hist = self.I(get_macd_hist, c)

        # --- 3. Volatility ---
        self.atr = self.I(ta.atr, h, l, c, length=14)
        
        # Bollinger Bands
        def get_bb_upper(ser): return ta.bbands(ser, length=20, std=2.0).iloc[:, 2]
        def get_bb_lower(ser): return ta.bbands(ser, length=20, std=2.0).iloc[:, 0]
        def get_bb_width(ser): return ta.bbands(ser, length=20, std=2.0).iloc[:, 3]
        
        self.bb_upper = self.I(get_bb_upper, c)
        self.bb_lower = self.I(get_bb_lower, c)
        self.bb_width = self.I(get_bb_width, c)
        
        # --- 4. Trend Strength (ADX) ---
        def get_adx(h, l, c): return ta.adx(h, l, c).iloc[:, 0]
        self.adx = self.I(get_adx, h, l, c)
        
        # DI+ / DI- for Trend Composite
        def get_di_plus(h, l, c): return ta.adx(h, l, c).iloc[:, 1] # DMP_14
        def get_di_minus(h, l, c): return ta.adx(h, l, c).iloc[:, 2] # DMN_14
        self.di_plus = self.I(get_di_plus, h, l, c)
        self.di_minus = self.I(get_di_minus, h, l, c)

        # --- 5. Supertrend (Critical for Guards) ---
        def get_supertrend(h, l, c): 
            return ta.supertrend(h, l, c, length=10, multiplier=3.0).iloc[:, 0]
        def get_supertrend_dir(h, l, c): 
            return ta.supertrend(h, l, c, length=10, multiplier=3.0).iloc[:, 1] # 1=Up, -1=Down
        self.supertrend = self.I(get_supertrend, h, l, c)
        self.supertrend_dir = self.I(get_supertrend_dir, h, l, c)

        # --- 6. PSAR ---
        def get_psar(h, l, c):
            psar = ta.psar(h, l, c, af0=0.02, af=0.02, max_af=0.2)
            # Combine long and short columns
            combined = psar.iloc[:, 0].fillna(psar.iloc[:, 1])
            return combined
        self.psar = self.I(get_psar, h, l, c)
        
        # --- 7. Volume ---
        # RVOL approximation (SMA 20 of Volume)
        v = pd.Series(self.data.Volume)
        self.vol_sma = self.I(ta.sma, v, length=20)

    def next(self):
        # 1. Skip warmup
        if np.isnan(self.ema200[-1]) or np.isnan(self.adx[-1]):
            return

        # 2. Construct Data Payload (Mimic what signal_engine expects)
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        avg_vol = self.vol_sma[-1]
        rvol = vol / avg_vol if avg_vol > 0 else 1.0
        
        # Determine PSAR trend string
        psar_val = self.psar[-1]
        psar_trend = "bull" if price > psar_val else "bear"
        
        # Determine Supertrend String
        st_val = self.supertrend[-1]
        st_dir_raw = self.supertrend_dir[-1] # 1 or -1
        st_str = "bull" if st_dir_raw > 0 else "bear"

        indicators = {
            "price": price,
            "Open": self.data.Open[-1],
            "prev_close": self.data.Close[-2],
            "ema_20": self.ema20[-1],
            "ema_50": self.ema50[-1],
            "ema_200": self.ema200[-1],
            "rsi": self.rsi[-1],
            "macd_histogram": self.macd_hist[-1],
            "atr_14": self.atr[-1],
            "atr_pct": (self.atr[-1] / price) * 100,
            
            "bb_high": self.bb_upper[-1],
            "bb_low": self.bb_lower[-1],
            "bb_width": self.bb_width[-1],
            
            "adx": self.adx[-1],
            "di_plus": self.di_plus[-1],
            "di_minus": self.di_minus[-1],
            "rvol": rvol,
            
            "ttm_squeeze": "on" if self.bb_width[-1] < 0.05 else "off",
            "volatility_quality": 10 if self.bb_width[-1] < 0.05 else 5,
            
            "psar_trend": psar_trend,
            "psar_level": psar_val,
            "supertrend_value": st_val,
            "supertrend_signal": st_str,
            
            # Need slope? We can approximate or let engine fallback
            "ema_20_slope": (self.ema20[-1] - self.ema20[-2]) # crude 1-bar slope
        }
        
        # 3. Add Composites (Engine needs these pre-calculated usually)
        # We call the engine's compute functions to be exact
        indicators["trend_strength"] = compute_trend_strength(indicators)["value"]
        indicators["momentum_strength"] = compute_momentum_strength(indicators)["value"]

        # 4. Generate Profile Report (Mocked based on Technicals)
        profile_report = mock_profile_report(indicators)

        # 5. EXECUTE ENGINE
        # This is the "Integration Test" - calling the actual logic
        plan = generate_trade_plan(
            profile_report,
            indicators,
            macro_trend_status="Bullish", # Simulate Bull Market or N/A
            horizon=self.horizon
        )
        
        signal = plan.get("signal", "WAIT")
        reason = plan.get("reason", "")
        
        # 6. Order Management
        
        # EXIT LOGIC (Managed by Backtester stops mostly, but we can force exit on reversal)
        if self.position:
            # If we are long and ST turns bear, close.
            if self.position.is_long and price < st_val:
                self.position.close()
            # If we are short and ST turns bull, close.
            elif self.position.is_short and price > st_val:
                self.position.close()
                
        # ENTRY LOGIC
        # We trust the engine. If it says BUY/SHORT, we execute exactly.
        if not self.position:
            
            if "BUY" in signal and "RISKY" not in signal:
                sl_price = plan["stop_loss"]
                # tp1 = plan["targets"]["t1"] 
                
                # Sanity check for backtester
                if sl_price and sl_price < price:
                    print(f"🟢 BUY  @ {self.data.index[-1]} | {reason} | SL: {sl_price:.2f}")
                    self.buy(sl=sl_price)
            
            elif "SHORT" in signal and "RISKY" not in signal:
                sl_price = plan["stop_loss"]
                
                # Sanity check
                if sl_price and sl_price > price:
                    print(f"🔴 SELL @ {self.data.index[-1]} | {reason} | SL: {sl_price:.2f}")
                    self.sell(sl=sl_price)
            
            elif "WAIT" in signal and "RESISTANCE" in signal:
                # Debug logging to prove Guards work
                # print(f"🛡️ GUARD @ {self.data.index[-1]} | {reason}")
                pass

def run_pro_backtest(symbol="RELIANCE.NS"):
    print(f"\n{'='*60}")
    print(f"🚀 FULL ENGINE SIMULATION: {symbol}")
    print(f"{'='*60}\n")
    
    # Load data
    df = ParquetStore.load_ohlcv(symbol, "1d")
    if df is None or df.empty:
        print("❌ No data found")
        return
    
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.dropna(inplace=True)
    
    # Run backtest
    bt = Backtest(
        df,
        FullEngineStrategy,
        cash=100_000,
        commission=0.002,
        exclusive_orders=True 
    )
    
    stats = bt.run()
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"Start: {df.index[0]}")
    print(f"End:   {df.index[-1]}")
    print(f"Return: {stats['Return [%]']:.2f}%")
    print(f"Win Rate: {stats['Win Rate [%]']:.1f}%")
    print(f"# Trades: {stats['# Trades']}")
    print(f"Sharpe: {stats['Sharpe Ratio']:.2f}")
    
    try:
        filename = f"data/engine_validation_{symbol}.html"
        bt.plot(filename=filename, open_browser=False, superimpose=False)
        print(f"\n📈 Chart saved to {filename}")
    except Exception as e:
        print(f"Plot warning: {e}")

if __name__ == "__main__":
    run_pro_backtest("HINDALCO.NS")