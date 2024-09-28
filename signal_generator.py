from datetime import datetime, timedelta
import numpy as np
from indicators import Indicators
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from config import MAX_OPEN_POSITIONS
from shared_state import get_shared_state
from portfolio_manager import PortfolioManager

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
            btc = self.data_manager.fetch_historical_prices(['BTCUSDT'], lookback_date, date, price_col='Close')
            sma_long = btc.mean()
            sma_med = btc.iloc[-self.params['med_regime_window']:].mean()
            sma_short = btc.iloc[-self.params['short_regime_window']:].mean()
            last_close = btc.iloc[-1]
            if last_close > sma_med:
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
        volume = prices['Volume']
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        close_train = close.loc[close.index.strftime('%Y-%m-%d') < date]

        #if 'volume' == self.params['ranking_method']:

        if ranking_method == 'volume':

            ranking = volume.reset_index()
            print(ranking)
            ranking.columns = ['Price', 'Ticker', 'Volume']
            ranking['Volume'] = ranking["Volume"].astype(float)
            ranking = ranking[['Ticker', 'Volume']]
            ranking = ranking.sort_values(by='Volume', ascending=False).dropna().reset_index().drop(columns='index')
            print('Volume Ranking')
            print(ranking)


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
            ranking = Indicators.calculate_rsi(close_train, self.params['rsi_window'])
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
        
        elif ranking_method == 'performance':
            print(close)
            total_return = (close.iloc[-1] / close.iloc[0]) - 1
            ranking = total_return.reset_index()
            ranking.columns = ['Price', 'Ticker', 'Performance']
            ranking['Performance'] = ranking["Performance"].astype(float)
            ranking = ranking.sort_values(by='Performance', ascending=False).dropna().reset_index().drop(columns='index')
            #print('Performance Ranking')

            #print(ranking)


        elif ranking_method == 'quality_momentum':
            returns = close.pct_change()
            total_return = (close.iloc[-1] / close.iloc[0]) - 1
            negative_returns_ratio = len(returns[returns < 0]) / len(returns)
            smoothness = 1 - negative_returns_ratio  # Higher value means smoother path
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)  # Annualized

                # Combine factors into quality momentum score
            momentum_score = 1/2 * (total_return + smoothness)

            #we want smoothness > 0.7 and total_return > 0.25 -> momentum_score > 0.425
            
            # Adjust score by Sharpe ratio to favor consistent performance
            #quality_momentum = momentum_score * (1 + sharpe_ratio)
            quality_momentum = sharpe_ratio

            #we want sharpe > 1.2 -> quality momentum > 0.95

            ranking = quality_momentum.reset_index()
            ranking.columns = ['Price', 'Ticker', 'Momentum']
            ranking['Momentum'] = ranking["Momentum"].astype(float)
            ranking = ranking[['Ticker', 'Momentum']]
            ranking = ranking.loc[(ranking['Momentum'] > 0.7) & (ranking['Momentum'] < 5)]
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

            print('Momentum Ranking')

            print(ranking)
            

        eligible_coins = list(ranking.Ticker.unique())
        return prices, eligible_coins

    def momentum_entry_signals(self, date, eligible_coins, positions):
        SIGNALS = {}

        lookback_date = date - timedelta(days=self.params['lookback_window'])
        prices = self.data_manager.fetch_historical_prices(eligible_coins, lookback_date, date)
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')

        for coin in eligible_coins:
            if ('Open', coin) in prices.columns and ('High', coin) in prices.columns and ('Low', coin) in prices.columns and ('Close', coin) in prices.columns:
                data = prices[[('Open', coin), ('High', coin), ('Low', coin), ('Close', coin)]]
                data.columns = ['Open', 'High', 'Low', 'Close']

            ema60 = data['Close'].ewm(span=60).mean()
            ema20 = data['Close'].ewm(span=20).mean()
            sma20 = data['Close'].rolling(20).mean()
            ema5 = data['Close'].ewm(span=5).mean()
            avg_range = Indicators.average_range(data, n=10)
            today_data = data.loc[data.index.strftime('%Y-%m-%d') == date]
            high = data.iloc[-10:-1]['High'].max()

            today_open = today_data['Open'].iloc[0]
            today_close = today_data['Close'].iloc[0]
            today_avg_range = avg_range.loc[date]

            #if today_close > (today_open + today_avg_range) and coin not in positions and len(positions) < self.MAX_POSITIONS:
            if today_close >= high and coin not in positions and len(positions) < self.MAX_POSITIONS:
            #if coin not in positions and len(positions) < self.MAX_POSITIONS:
                SIGNALS[coin] = {'signal': 1, 'strategy': 'momentum'} 
                print(f'Momentum entry signal detected on {date} for {coin}')
                print(f'Reason: Close = {today_close} > 10-day High = {high}')
                self.position_entry_dates[coin] = pd.to_datetime(date) + timedelta(days=1)  # New dictionary to track entry dates
                self.shared_state.update_open_orders(SIGNALS)

    def momentum_exit_signals(self, date, ranked_coins, positions):
        coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'momentum']
        lookback_date = date - timedelta(days=self.params['lookback_window'])
        prices = self.data_manager.fetch_historical_prices(coins_held, lookback_date, date, price_col='Close')
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        SIGNALS = {}

        # Get the top coins from the latest ranking
        top_coins = ranked_coins[:5]

        for coin in coins_held:

            # Handle both multi-column and single-column price data
            if isinstance(prices, pd.DataFrame):
                if ('Close', coin) in prices.columns:
                    data = prices[('Close', coin)]
                elif coin in prices.columns:
                    data = prices[coin]
            elif isinstance(prices, pd.Series):
                if prices.name == ('Close', coin) or prices.name == coin:
                    data = prices

            ema60 = data.ewm(span=60).mean()
            low = data.iloc[-7:].min()
            ema20 = data.ewm(span=20).mean()
            ema5 = data.ewm(span=5).mean()
            sma20 = data.rolling(20).mean()
            high = data.iloc[-10:-1].max()
            data = data.loc[data.index.strftime('%Y-%m-%d')==date]
            close = data.iloc[-1]
            trend_ema_long = ema60.iloc[-1]
            trend_ema_short = ema20.iloc[-1]
    

            if close < trend_ema_short:
                #or trend_ema_short < trend_ema_long or coin not in top_coins:
                #print(top_coins)
                SIGNALS[coin] = {'signal': -1, 'strategy': 'momentum'}  # Exit signal
                print(f'Momentum exit signal detected on {date} for {coin}')
                print(f'Reason: Close = {close} < 20-day EMA = {trend_ema_short}')
            else:
                print(f'Close = {close} > 20-day EMA = {trend_ema_short}: staying Long.')

        self.shared_state.update_open_orders(SIGNALS)


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


    def mean_reversion_exit_signals2(self, date, positions):
            coins_held = [coin for coin in positions if positions[coin]['strategy'] == 'momentum']
            SIGNALS = {}
            date += timedelta(days=1)

            for coin in coins_held:
                entry_date = self.position_entry_dates.get(coin)
                if entry_date is None:
                    print(f"Warning: No entry date found for {coin}. Skipping exit check.")
                    continue

                days_held = (date - entry_date).days
                current_position = positions[coin]
                entry_price = self.portfolio_manager.get_entry_price(coin)
                current_price = self.data_manager.fetch_current_price(coin, date, 'Close')

                if days_held == 1:
                    if self.is_profitable(current_position, entry_price, current_price):
                        SIGNALS[coin] = -np.sign(current_position)  # Exit signal
                elif days_held >= 2:
                    SIGNALS[coin] = -np.sign(current_position)  # Exit signal
               # elif days_held >= 3:
                #    SIGNALS[coin] = -np.sign(current_position)  # Force exit

                if coin in SIGNALS:
                    print(f'Exit signal detected on {date} for {coin} after {days_held} days')

            self.shared_state.update_open_orders(SIGNALS)

    def is_profitable(self, position, entry_price, current_price):
        return (position > 0 and current_price > entry_price) or (position < 0 and current_price < entry_price)
 
    def mean_reversion_exit_signals(self, date, positions):
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

            if signal != 0:
                SIGNALS[coin] = {'signal': signal, 'strategy': 'mean_reversion'}
                print(f'Mean-reversion exit signal detected on {date} for {coin}')
                self.shared_state.update_open_orders(SIGNALS)
                #print(f'Reason: Close = {close}, EMA5 = {mr_ema}, EMA16 = {trend_ema_short}, EMA64 = {trend_ema_long}')
