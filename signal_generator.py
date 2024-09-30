# signal_generator.py

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from config_loader import CONFIG
from shared_state import get_shared_state

class SignalGenerator:
    def __init__(self, data_manager, strategy):
        self.data_manager = data_manager
        self.strategy = strategy
        self.params = CONFIG['params']
        self.MAX_POSITIONS = CONFIG['backtest']['max_open_positions']
        self.shared_state = get_shared_state()

    def generate_signals(self, date, eligible_coins, positions):
        entry_signals = self.strategy.generate_entry_signals(date, eligible_coins, positions)
        exit_signals = self.strategy.generate_exit_signals(date, positions)
        
        self.shared_state.update_open_orders(entry_signals)
        self.shared_state.update_open_orders(exit_signals)

    def rank_coins(self, date, investment_universe, ranking_method):
        lookback_date = date - timedelta(days=self.params['lookback_window'])
        prices = self.data_manager.fetch_historical_prices(investment_universe, lookback_date, date)
        
        if ranking_method == 'volume':
            return self._rank_by_volume(prices, date)
        elif ranking_method == 'momentum':
            return self._rank_by_momentum(prices, date)
        elif ranking_method == 'mean_reversion':
            return self._rank_by_mean_reversion(prices, date)
        else:
            raise ValueError(f"Unknown ranking method: {ranking_method}")

    def _rank_by_volume(self, prices, date):
        volume = prices['Volume'].sum().reset_index()
        volume.columns = ['Ticker', 'Volume']
        volume['Volume'] = volume["Volume"].astype(float)
        volume = volume.loc[volume.Volume != 0]
        volume = volume.sort_values(by='Volume', ascending=False).dropna().reset_index(drop=True)
        return prices, list(volume['Ticker'])

    def _rank_by_momentum(self, prices, date):
        close = prices['Close']
        returns = close.pct_change()
        momentum = returns.mean() / returns.std() * np.sqrt(252)  # Annualized Sharpe ratio as momentum
        ranking = momentum.sort_values(ascending=False).reset_index()
        ranking.columns = ['Ticker', 'Momentum']
        return prices, list(ranking['Ticker'])

    def _rank_by_mean_reversion(self, prices, date):
        close = prices['Close']
        sma = close.rolling(window=20).mean()
        std = close.rolling(window=20).std()
        z_score = (close - sma) / std
        ranking = z_score.iloc[-1].sort_values().reset_index()
        ranking.columns = ['Ticker', 'Z-Score']
        return prices, list(ranking['Ticker'])

    def regime_trend(self, date):
        if self.params['regime_filter']:
            lookback_date = date - timedelta(days=self.params['long_regime_window'])
            btc = self.data_manager.fetch_historical_prices(['BTCUSDT'], lookback_date, date, price_col='Close')
            sma_long = btc.mean()
            sma_med = btc.iloc[-self.params['med_regime_window']:].mean()
            sma_short = btc.iloc[-self.params['short_regime_window']:].mean()
            last_close = btc.iloc[-1]
            bull_market = last_close > sma_med
            if bull_market:
                print('BTC is in uptrend: only considering long positions')
                print(f'BTC Close = {last_close} > 50-day BTC MA = {sma_med}')
            else:
                print('BTC is in downtrend: only considering short positions')
                print(f'BTC Close = {last_close} < 50-day BTC MA = {sma_med}')
            return bull_market
        return None

    def generate_and_prioritize_signals(self, date, eligible_coins, positions):
        # Generate all types of signals
        entry_signals = self.strategy.generate_entry_signals(date, eligible_coins, positions)
        exit_signals = self.strategy.generate_exit_signals(date, positions)
        regime_exit_signals = self.generate_regime_exit_signals(date, positions)

        # Prioritize regime signals and resolve conflicts
        final_signals = {}

        if self.params['regime_filter']:
            bull_market = self.regime_trend(date)
            
            for coin, signal in regime_exit_signals.items():
                final_signals[coin] = signal
                
                # Remove any conflicting entry signals
                if coin in entry_signals:
                    del entry_signals[coin]
                
                # Remove any conflicting exit signals
                if coin in exit_signals:
                    del exit_signals[coin]

            # For remaining entry signals, only keep those aligned with the current regime
            for coin, signal in entry_signals.items():
                if (bull_market and signal['signal'] > 0) or (not bull_market and signal['signal'] < 0):
                    final_signals[coin] = signal

        else:
            # If regime filter is not used, combine entry and exit signals
            final_signals.update(exit_signals)
            final_signals.update(entry_signals)

        self.shared_state.set_open_orders(final_signals)

    def generate_regime_exit_signals(self, date, positions):
        if self.params['regime_filter']:
            bull_market = self.regime_trend(date)
            exit_signals = {}
            
            for coin, position in positions.items():
                units = position['units']
                strategy = position['strategy']
                
                if (bull_market and units < 0) or (not bull_market and units > 0):
                    exit_signals[coin] = {'signal': -np.sign(units), 'strategy': strategy}
            
            return exit_signals
        return {}