from datetime import datetime, timedelta
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import MAX_OPEN_POSITIONS
from shared_state import get_shared_state
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from config import DATA_DIR

class SignalGenerator:
    def __init__(self, data_manager, indicators, portfolio_manager, params):
        self.data_manager = data_manager
        self.indicators = indicators
        self.params = params
        self.MAX_POSITIONS = MAX_OPEN_POSITIONS
        self.shared_state = get_shared_state()
        self.position_entry_dates = {}  # New dictionary to track entry dates
        self.portfolio_manager = portfolio_manager

    def regime_trend(self, date):
        if self.params['regime_filter']:
            lookback_date = date - timedelta(days=self.params['long_regime_window'])
            btc = self.data_manager.fetch_historical_prices(['BTC-USD'], lookback_date, date, price_col='Close')
            sma_long = btc.mean()
            sma_med = btc.iloc[-self.params['med_regime_window']:].mean()
            sma_short = btc.iloc[-self.params['short_regime_window']:].mean()
            last_close = btc.iloc[-1]
            if last_close > sma_long and sma_short > sma_med:
                return True
            else:
                print(f'{date}: BTC is in downtrend: bear market environment.')
                return False
        else:
            return None

    def rank_coins(self, date, investment_universe, ranking_method='momentum'):
        lookback_date = date - timedelta(days=self.params['lookback_window'])
        prices = self.data_manager.fetch_historical_prices(investment_universe, lookback_date, date)
        close = prices[['Close']]
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        close_train = close.loc[close.index.strftime('%Y-%m-%d') < date]

        #if 'volume' == self.params['ranking_method']:

        if ranking_method == 'mean_reversion':
            ewma64 = close.ewm(span=64).mean()
            ewma32 = close.ewm(span=32).mean()
            ewma16 = close.ewm(span=16).mean()
            ewma8 = close.ewm(span=8).mean()
            ewma4 = close.ewm(span=4).mean()

            ewmac16 = (ewma16 - ewma64) / ewma64
            ewmac8 = (ewma8 - ewma32) / ewma32
            ewmac4 = (ewma4 - ewma16) / ewma16

            unscaled_mom = 1/3 * (ewmac16 + ewmac8 + ewmac4)

            vol = close.pct_change().rolling(32).std() * np.sqrt(365)
            scaled_mom = unscaled_mom * vol
            ranking = scaled_mom.iloc[-1].sort_values(ascending=False).reset_index()
            ranking.columns = ['Price', 'Ticker', 'Momentum']
            ranking = ranking.loc[ranking.Momentum>0].iloc[:5] #for now, only long
        

        if ranking_method == 'mean_deviation':
            ewm5 = close.ewm(span=5).mean().iloc[-1]
            ewm16 = close.ewm(span=16).mean().iloc[-1]
            ewm64 = close.ewm(span=64).mean().iloc[-1]
            vol = close.std()
            last_close = close.iloc[-1]

            long_term_trend = (ewm16 - ewm64) / ewm64
            long_term_trend.dropna(inplace=True)
            #print('Long term trend:')
            #print(long_term_trend)
            #short_term_trend = (last_close - ewm5) / vol
            short_term_trend = (last_close - ewm5) / ewm5
            short_term_trend.dropna(inplace=True)
            #print('Short term trend:')
            #print(short_term_trend)

            common_index = long_term_trend.index.intersection(short_term_trend.index)
            long_term_trend = long_term_trend.loc[common_index]
            short_term_trend = short_term_trend.loc[common_index]

            mean_reversion_strength = np.where(
                    (np.sign(short_term_trend) != np.sign(long_term_trend)),
                    0.5 * abs(short_term_trend) + 0.5 * abs(long_term_trend),
                    0
                )

            ranking = pd.DataFrame()
            ranking['Ticker'] = long_term_trend.index.to_list()
            ranking['mean_reversion_strength'] = mean_reversion_strength
            ranking = ranking.sort_values('mean_reversion_strength', ascending=False)
            ranking = ranking.loc[ranking['mean_reversion_strength'] != 0]
            ranking['Ticker'] = ranking['Ticker'].apply(lambda x: x[1])

            # Reset the index to make sure the Ticker column is the first column
            ranking = ranking.reset_index(drop=True)
            ranking.dropna(inplace=True)


        if 'RSI' in self.params['ranking_method']:
            ranking = self.indicators.calculate_rsi(close_train, self.params['rsi_window'])
            if self.params['ranking_method'] == 'RSI_long':
                ranking = ranking.sort_values(by='RSI', ascending=True)
                ranking = ranking.loc[ranking['RSI'] <= self.params['min_rsi']]
            else:
                ranking = ranking.sort_values(by='RSI', ascending=False)
                ranking = ranking.loc[ranking['RSI'] >= self.params['min_rsi']]
        elif self.params['ranking_method'] == 'perf':
            ranking = close_train.iloc[-1] / close.iloc[0] - 1
            ranking = ranking.reset_index()
            ranking.rename(columns={0: 'Performance'}, inplace=True)
            ranking = ranking[['Ticker', 'Performance']]
            ranking = ranking.sort_values(by='Performance', ascending=False)
            ranking = ranking.dropna().reset_index().drop(columns='index')

            if self.params['perf_thres'] is not None:
                ranking = ranking.loc[ranking['Performance'] > self.params['perf_thres']]

            if self.params['n_coins'] is not None:
                ranking = ranking.iloc[:self.params['n_coins']]

        elif ranking_method == 'quality_momentum':
            returns = close.pct_change()
            total_return = (close.iloc[-1] / close.iloc[0]) - 1
            negative_returns_ratio = len(returns[returns < 0]) / len(returns)
            smoothness = 1 - negative_returns_ratio  # Higher value means smoother path
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized

                # Combine factors into quality momentum score
            momentum_score = 1/2 * (total_return + smoothness)
            
            # Adjust score by Sharpe ratio to favor consistent performance
            quality_momentum = momentum_score * (1 + sharpe_ratio)

            ranking = quality_momentum.reset_index()
            ranking.columns = ['Price', 'Ticker', 'Momentum']
            ranking['Momentum'] = ranking["Momentum"].astype(float)
            ranking = ranking[['Ticker', 'Momentum']]
            ranking = ranking.sort_values(by='Momentum', ascending=False).dropna().reset_index().drop(columns='index')

            #print('Quality Momentum Ranking')

            #print(ranking)

        elif ranking_method == 'momentum':
            
            ewma64 = close.ewm(span=64).mean()
            ewma32 = close.ewm(span=32).mean()
            ewma16 = close.ewm(span=16).mean()
            ewma8 = close.ewm(span=8).mean()
            ewma4 = close.ewm(span=4).mean()
            ewma5 = close.ewm(span=5).mean()

            vol = close.pct_change().std()

            mom = (close - ewma5) / ewma5

            ewmac16 = (ewma16 - ewma64) / ewma64
            ewmac8 = (ewma8 - ewma32) / ewma32
            ewmac4 = (ewma4 - ewma16) / ewma16

            unscaled_mom = 1/3 * (ewmac16 + ewmac8 + ewmac4)

            vol = close.pct_change().std() * np.sqrt(365)
            scaled_mom = unscaled_mom / vol
            ranking = scaled_mom.iloc[-1].reset_index()
            ranking.columns = ['Price', 'Ticker', 'Momentum']
            ranking['Momentum'] = ranking["Momentum"].astype(float)
            ranking = ranking[['Ticker', 'Momentum']]
            ranking = ranking.sort_values(by='Momentum', ascending=False)

            #print('Momentum Ranking')

            #print(ranking)
            

        eligible_coins = list(ranking.Ticker.unique())
        return prices, eligible_coins

    def momentum_entry_signals(self, date, eligible_coins, positions):
        SIGNALS = {}

        for coin in eligible_coins:

            if coin not in positions and len(positions) < self.MAX_POSITIONS:
                SIGNALS[coin] = {'signal': 1, 'strategy': 'momentum'} 
                #print(f'Momentum entry signal detected on {date} for {coin}')
                self.position_entry_dates[coin] = pd.to_datetime(date) + timedelta(days=1)  # New dictionary to track entry dates
                self.shared_state.update_open_orders(SIGNALS)

    def momentum_exit_signals(self, date, ranked_coins, positions):
        coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'momentum']
        SIGNALS = {}

        # Get the top 5 coins from the latest ranking
        top_coins = ranked_coins[:self.MAX_POSITIONS]

        for coin in coins_held:
            if coin not in top_coins:
                #print(top_5_coins)
                SIGNALS[coin] = {'signal': -1, 'strategy': 'momentum'}  # Exit signal
                print(f'Momentum exit signal detected on {date} for {coin} (no longer in top 5)')

        self.shared_state.update_open_orders(SIGNALS)

    def mean_reversion_entry_signals_adf(self, date, prices, eligible_coins, positions, accepted_signals='long_only', entry_zscore=1):
        SIGNALS = {}

        for coin in eligible_coins:
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')
            
            data = prices[[('Close', coin)]]
            data.columns = data.columns.droplevel(1)
            
            # Perform ADF test
            result = adfuller(data['Close'].dropna())
            
            # Check if the series is stationary (p-value < 0.05)
            is_stationary = result[1] < 0.1
            
            if is_stationary:
                print(f'Stationary coin detected! {coin}')
                # Calculate z-score
                rolling_mean = data['Close'].rolling(window=60).mean()
                rolling_std = data['Close'].rolling(window=60).std()
                z_score = (data['Close'] - rolling_mean) / rolling_std

                print(f"z_score: {z_score.iloc[-1]}")
                
                # Define entry conditions
                long_entry = z_score < -entry_zscore
                short_entry = z_score > entry_zscore
                
                # Plot the time series and highlight entry areas
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
                
                # Plot price and moving average
                ax1.plot(data.index, data['Close'], label='Price')
                ax1.plot(data.index, rolling_mean, label='30-day MA')
                ax1.set_title(f'{coin} Price and Moving Average')
                ax1.legend()
                
                # Plot z-score and entry areas
                ax2.plot(data.index, z_score, label='Z-score')
                ax2.axhline(y=2, color='r', linestyle='--')
                ax2.axhline(y=-2, color='g', linestyle='--')
                ax2.fill_between(data.index, -entry_zscore, z_score, where=(z_score < -entry_zscore), color='g', alpha=0.3, label='Long Entry')
                ax2.fill_between(data.index, entry_zscore, z_score, where=(z_score > entry_zscore), color='r', alpha=0.3, label='Short Entry')
                ax2.set_title(f'{coin} Z-score and Entry Areas')
                ax2.legend()
                
                # Format x-axis
                ax2.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
                plt.xticks(rotation=45)
                
                plt.tight_layout()
                plt.savefig(f'{DATA_DIR}/{coin}_mean_reversion_analysis_{date}.png')
                plt.close()
                
                # Generate signals
                if accepted_signals == 'all':
                    if long_entry.iloc[-1]:
                        signal = 1
                    elif short_entry.iloc[-1]:
                        signal = -1
                    else:
                        signal = 0
                elif accepted_signals == 'long_only':
                    if long_entry.iloc[-1]:
                        signal = 1
                    else:
                        signal = 0
                elif accepted_signals == 'short_only':
                    if short_entry.iloc[-1]:
                        signal = -1
                    else:
                        signal = 0
                
                
                print(f"signal: {signal} ")
                if coin not in positions and len(positions) < self.MAX_POSITIONS and signal != 0:
                    SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                    print(f'Mean-reversion entry signal detected on {date} for {coin}')
                    self.position_entry_dates[coin] = pd.to_datetime(date) + timedelta(days=1)
                    self.shared_state.update_open_orders(SIGNALS)
                    print(f'Reason: Stationary series, Z-score = {z_score.iloc[-1]:.2f}')
            

        return SIGNALS


    def mean_reversion_entry_signals(self, date, prices, eligible_coins, positions, accepted_signals='long_only'):
        SIGNALS = {}

        for coin in eligible_coins:
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')
            data = prices[[('Open', coin), ('Close', coin), ('High', coin), ('Low', coin)]]
            data.columns = data.columns.droplevel(1)
            data['EMA64'] = data['Close'].ewm(span=64).mean()
            data['EMA16'] = data['Close'].ewm(span=16).mean()
            data['EMA5'] = data['Close'].ewm(span=5).mean()
            vol = data['Close'].std()
            data = data.loc[data.index.strftime('%Y-%m-%d')==date]
            close = data['Close'].iloc[-1]
            trend_ema_long = data['EMA64'].iloc[-1]
            trend_ema_short = data['EMA16'].iloc[-1]
            mr_ema = data['EMA5'].iloc[-1]

            bollinger = (close - mr_ema) / vol

            long_term_trend = (trend_ema_short - trend_ema_long) / trend_ema_long
            short_term_trend = (close - mr_ema) / mr_ema

            if short_term_trend > 0.05 and long_term_trend < -0.05:
            #if bollinger > 0.5 and long_term_trend < 0:
                signal = -1
            
            elif short_term_trend < -0.05 and long_term_trend > 0.05:
            #elif bollinger < -0.5 and long_term_trend > 0:
                signal = 1

            else:
                signal = 0
            
            if accepted_signals == 'long_only':
                if coin not in positions and len(positions) < self.MAX_POSITIONS and signal == 1: #only long
                    SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                    print(f'Mean-reversion entry signal detected on {date} for {coin}')
                    self.position_entry_dates[coin] = pd.to_datetime(date) + timedelta(days=1)  # New dictionary to track entry dates
                    self.shared_state.update_open_orders(SIGNALS)
                    print(f'Reason: Close = {close}, EMA5 = {mr_ema}, EMA16 = {trend_ema_short}, EMA64 = {trend_ema_long}')
            
            elif accepted_signals == 'short_only':
                if coin not in positions and len(positions) < self.MAX_POSITIONS and signal == -1: #only short
                    SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                    print(f'Mean-reversion entry signal detected on {date} for {coin}')
                    self.position_entry_dates[coin] = pd.to_datetime(date) + timedelta(days=1)  # New dictionary to track entry dates
                    self.shared_state.update_open_orders(SIGNALS)
                    print(f'Reason: Close = {close}, EMA5 = {mr_ema}, EMA16 = {trend_ema_short}, EMA64 = {trend_ema_long}')

    
    def momentum_exit_signals2(self, date, positions):
        coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'momentum']
        lookback_date = date - timedelta(days=self.params['lookback_window'])
        prices = self.data_manager.fetch_historical_prices(coins_held, lookback_date, date)
        SIGNALS = {}
        for coin in coins_held:
            current_position = np.sign(positions[coin]['units'])

            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')

            if len(coins_held) > 1:
                data = prices[[('Open', coin), ('Close', coin), ('High', coin), ('Low', coin)]]
                data.columns = data.columns.droplevel(1)
                data = data[['Close']]
            else:
                data = pd.DataFrame(prices)
                data.columns = ['Close']

            
            data['EMA64'] = data['Close'].ewm(span=64).mean()
            data['EMA32'] = data['Close'].ewm(span=32).mean()
            data['EMA16'] = data['Close'].ewm(span=16).mean()
            data['EMA8'] = data['Close'].ewm(span=8).mean()
            data['EMA4'] = data['Close'].ewm(span=4).mean()
            ema5 = data['Close'].ewm(span=5).mean().iloc[-1]
            data = data.loc[data.index.strftime('%Y-%m-%d')==date]

            ewmac16 = (data['EMA16'] - data['EMA64']).iloc[-1]
            ewmac8 = (data['EMA8'] - data['EMA32']).iloc[-1]
            ewmac4 = (data['EMA4'] - data['EMA16']).iloc[-1]

            close = data['Close'].iloc[-1]

            signal = np.sign(1/3 * (ewmac16 + ewmac8 + ewmac4))


            if current_position == 1 and signal != 1:
                signal = -1
            elif current_position == -1 and signal != -1:
                signal = 1
            else:
                signal = 0

            if signal != 0:
                SIGNALS[coin] = {'signal': signal, 'strategy': 'momentum'}
                print(f'Momentum exit signal detected on {date} for {coin}')
                self.shared_state.update_open_orders(SIGNALS)
                #print(f'Reason: Close = {close}, EMA5 = {mr_ema}, EMA16 = {trend_ema_short}, EMA64 = {trend_ema_long}')

    def mean_reversion_exit_signals_adf(self, date, positions):
        lookback_date = date - timedelta(days=self.params['lookback_window'])
        coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'mean_reversion']
        prices = self.data_manager.fetch_historical_prices(coins_held, lookback_date, date)
        print(prices)
        SIGNALS = {}

        for coin in coins_held:
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')

            # Handle both multi-column and single-column price data
            if isinstance(prices, pd.DataFrame):
                if ('Close', coin) in prices.columns:
                    data = prices[('Close', coin)]
                elif coin in prices.columns:
                    data = prices[coin]
            elif isinstance(prices, pd.Series):
                if prices.name == ('Close', coin) or prices.name == coin:
                    data = prices
            
            
            # Perform ADF test
            result = adfuller(data.dropna())
            
            # Check if the series is stationary (p-value < 0.05)
            p_value = result[1] 

            # Calculate z-score
            rolling_mean = data.rolling(window=60).mean()
            rolling_std = data.rolling(window=60).std()
            z_score = (data.iloc[-1] - rolling_mean.iloc[-1]) / rolling_std.iloc[-1]

            print(f'Z-SCORE: {z_score}')

            current_position = positions[coin]['units']
            
            # Check stationarity and z-score
            if p_value > 0.3:
                print(f"{coin} is no longer stationary (p-value: {p_value:.4f}). Exiting position.")
                signal = -np.sign(current_position)
                SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
            elif abs(z_score) < 0.5:
                print(f"{coin} z-score ({z_score:.4f}) is close to mean. Exiting position.")
                signal = -np.sign(current_position)
                SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
            else:
                print(f"{coin} remains in position. p-value: {p_value:.4f}, z-score: {z_score:.4f}")

            if coin in SIGNALS:
                print(f'Exit signal detected on {date} for {coin}')

        self.shared_state.update_open_orders(SIGNALS)


    def mean_reversion_exit_signals2(self, date, positions):
            coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'mean_reversion']
            SIGNALS = {}
            date += timedelta(days=1)

            for coin in coins_held:
                entry_date = self.position_entry_dates.get(coin)
                if entry_date is None:
                    print(f"Warning: No entry date found for {coin}. Skipping exit check.")
                    continue

                days_held = (date - entry_date).days
                current_position = positions[coin]['units']
                entry_price = self.portfolio_manager.get_entry_price(coin)
                current_price = self.data_manager.fetch_current_price(coin, date, 'Close')

                if days_held == 1:
                    if self.is_profitable(current_position, entry_price, current_price):
                        signal = -np.sign(current_position)
                        SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}  # Exit signal
                elif days_held == 2:
                    if self.is_profitable(current_position, entry_price, current_price):
                        signal = -np.sign(current_position)
                        SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}  # Exit signal
                elif days_held >= 3:
                    signal = -np.sign(current_position)
                    SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}  # Exit signal

                if coin in SIGNALS:
                    print(f'Exit signal detected on {date} for {coin} after {days_held} days')

            self.shared_state.update_open_orders(SIGNALS)

    def is_profitable(self, position, entry_price, current_price):
        return (position > 0 and current_price > entry_price) or (position < 0 and current_price < entry_price)
 
    def mean_reversion_exit_signals(self, date, positions, signals_type='all'):
        coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'mean_reversion']
        print(coins_held)
        lookback_date = date - timedelta(days=self.params['lookback_window'])
        prices = self.data_manager.fetch_historical_prices(coins_held, lookback_date, date)
        SIGNALS = {}
        for coin in coins_held:
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')

            if len(coins_held) > 1:
                data = prices[[('Open', coin), ('Close', coin), ('High', coin), ('Low', coin)]]
                data.columns = data.columns.droplevel(1)
                data = data[['Close']]
            else:
                data = pd.DataFrame(prices)
                data.columns = ['Close']

            
            data['EMA64'] = data['Close'].ewm(span=64).mean()
            data['EMA16'] = data['Close'].ewm(span=16).mean()
            data['EMA5'] = data['Close'].ewm(span=5).mean()
            data = data.loc[data.index.strftime('%Y-%m-%d')==date]
            close = data['Close'].iloc[-1]
            trend_ema_long = data['EMA64'].iloc[-1]
            trend_ema_short = data['EMA16'].iloc[-1]
            mr_ema = data['EMA5'].iloc[-1]
            current_position = np.sign(positions[coin]['units'])

            long_term_trend = np.sign(trend_ema_short - trend_ema_long)
            short_term_trend = np.sign(close - mr_ema)

            if current_position == 1 and long_term_trend == short_term_trend:
                signal = -1
            elif current_position == -1 and long_term_trend == short_term_trend:
                signal = 1
            else:
                signal = 0

            if signal != 0 and signals_type == 'all':
                SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                print(f'Mean-reversion exit signal detected on {date} for {coin}')
                self.shared_state.update_open_orders(SIGNALS)
                #print(f'Reason: Close = {close}, EMA5 = {mr_ema}, EMA16 = {trend_ema_short}, EMA64 = {trend_ema_long}')
            
            elif signals_type == 'long' and signal == -1:
                SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                print(f'Mean-reversion exit signal detected on {date} for {coin}')
                self.shared_state.update_open_orders(SIGNALS)

            elif signals_type == 'short' and signal == 1:
                SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                print(f'Mean-reversion exit signal detected on {date} for {coin}')
                self.shared_state.update_open_orders(SIGNALS)

