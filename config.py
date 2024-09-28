# config.py

from datetime import datetime
import os

# Data parameters
# Use os.path.join to create platform-independent paths

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'DATA')
MARKET_CAP_DIR = os.path.join(DATA_DIR, 'Crypto_Market_Cap')

# Backtest parameters
START_DATE = datetime(2020, 1, 24)
END_DATE = datetime(2024, 7, 26)
INITIAL_CASH = 100_000
MIN_CASH = 1000
MAX_OPEN_POSITIONS = 10
BTC_HISTORICAL_VOL = 0.7


# Strategy parameters
PARAMS = {
    'transaction_cost': 0.1/100,
    'slippage': 0.5/100,
    'max_MC_rank': 100, ##We only consider coins ranked until this MC rank in our investment universe at all times
    'min_volume_thres': 2_000_000, #We only consider coins with this minimum 24-hour trading volume in the investment universe
    'sma_window': 20,
    'vol_window': 7, #Window used as a lookback for volatility targeting
    'vol_target': 1, #Annualized volatility targeted by the portfolio
    'target_vol_thres': 0.25, #We only rebalance when our curren volatility target exceeds our optimal volatility target by this threshold (from up or down)
    'regime_filter': True, #Momentum regime filter based on Bitcoin momentum. 
    'rebalancing': True,
    'rsi_window': 4,
    'min_rsi': 95,
    'ranking_method': 'volume', #mean_deviation
    'lookback_window': 90,
    'n_coins': None,
    'perf_thres': None,
    'long_regime_window': 100,
    'short_regime_window': 8,
    'med_regime_window': 50,
    'position_stop_loss': 0.2,
    'strategy': 'momentum',
    'max_allocation_per_coin': 0.1,
    'allocation_type': 'equal_weight'
}