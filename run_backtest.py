# run_backtest.py
import pandas as pd
from services.backtest_engine import BacktestEngine
from config.constants import INDEX_TICKERS

# ==========================================
# CONFIGURATION
# ==========================================
TEST_HORIZON = "short_term"  # Options: 'intraday', 'short_term'
TEST_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
    "TATAMOTORS.NS", "SBIN.NS", "ICICIBANK.NS", "BAJFINANCE.NS"
]
# ==========================================

def run_experiment():
    print(f"🚀 Starting Backtest on {len(TEST_SYMBOLS)} symbols")
    print(f"📅 Horizon: {TEST_HORIZON}")
    print("-" * 60)

    all_trades = []

    for symbol in TEST_SYMBOLS:
        try:
            engine = BacktestEngine(symbol, horizon=TEST_HORIZON)
            engine.run()
            
            trades = engine.results
            all_trades.extend(trades)
            
            # Quick stat per stock
            wins = [t for t in trades if t['status'] == 'WIN']
            if trades:
                win_rate = len(wins) / len(trades) * 100
                print(f"{symbol:<15} | Trades: {len(trades):<3} | Win Rate: {win_rate:.1f}%")
            else:
                print(f"{symbol:<15} | No trades generated.")
                
        except Exception as e:
            print(f"❌ Error processing {symbol}: {e}")

    print("-" * 60)
    
    if not all_trades:
        print("No trades found across all stocks. Check data availability.")
        return

    # --- AGGREGATE STATS ---
    df = pd.DataFrame(all_trades)
    
    print(f"\n📊 OVERALL RESULTS ({TEST_HORIZON.upper()})")
    print(f"Total Trades:     {len(df)}")
    
    win_mask = df['status'] == 'WIN'
    loss_mask = df['status'] == 'LOSS'
    
    win_rate = len(df[win_mask]) / len(df) * 100
    avg_win = df[win_mask]['pnl_pct'].mean() if not df[win_mask].empty else 0
    avg_loss = df[loss_mask]['pnl_pct'].mean() if not df[loss_mask].empty else 0
    
    print(f"Win Rate:         {win_rate:.2f}%")
    print(f"Avg Win:          {avg_win:.2f}%")
    print(f"Avg Loss:         {avg_loss:.2f}%")
    if avg_loss != 0:
        print(f"Reward/Risk (Real): {abs(avg_win/avg_loss):.2f}")

    print("\n🏆 PERFORMANCE BY SETUP TYPE")
    print(df.groupby('setup').agg({
        'status': 'count',
        'pnl_pct': 'mean'
    }).rename(columns={'status': 'Trades', 'pnl_pct': 'Avg Return %'}))

if __name__ == "__main__":
    run_experiment()