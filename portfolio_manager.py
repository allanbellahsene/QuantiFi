from datetime import datetime, timedelta
import numpy as np
from config import PARAMS as params
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from shared_state import get_shared_state

class PortfolioManager:
    def __init__(self, initial_cash, min_cash, max_open_positions, transaction_cost, slippage, data_manager):
        self.shared_state = get_shared_state()
        self.Cash = initial_cash
        self.MIN_CASH_LEFT = min_cash
        self.MAX_POSITIONS = max_open_positions
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.data_manager = data_manager
        self.POSITIONS = {}
        self.PROCESSED_ORDERS = {}
        self.HOLDINGS = {}
        self.REBALANCING_ORDERS = {}
        self.PROCESSED_REBALANCING = {}
        self.SIZES = {}
        self.equity = 0
        self.AUM = self.Cash + self.equity
        self.vol_target = params['vol_target']
        self.vol_window = params['vol_window']
        self.vol_threshold = params['target_vol_thres']
        self.POSITION_STOP_LOSS = params['position_stop_loss']
        self.allow_short = {'momentum': False, 'mean_reversion': True, 'combined': True}
        self.params = params

    def size_position(self, date, coin, min_vol=1e-5, max_size=3):

        if self.params['allocation_type'] == 'volatility_target':
            vol = self.get_volatility(date, coin)
            if vol <= min_vol:
                #print(f"Warning: Extremely low volatility ({vol}) detected for {coin} on {date}. Setting to minimum threshold.")
                vol = min_vol
            position_size = min(max_size, self.vol_target / vol)
        else:
            position_size = 1/self.MAX_POSITIONS

        return position_size

    def get_volatility(self, date, coin):
        lookback_date = date - timedelta(days=self.params['lookback_window']) 
        price_df = self.data_manager.fetch_historical_prices([coin], lookback_date, date, price_col='Close')
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        price_df = price_df.loc[price_df.index<date]

        lt_vol = price_df.pct_change().std() * np.sqrt(365)
        st_vol = price_df.iloc[-self.vol_window:].pct_change().std() * np.sqrt(365)

        vol = 0.7 * st_vol + 0.3 * lt_vol

        return vol

    def calculate_risk_parity_weights(self, volatilities, target_vol):
        inv_vols = [1/vol for vol in volatilities]
        total_inv_vol = sum(inv_vols)
        weights = [inv_vol/total_inv_vol for inv_vol in inv_vols]
        
        # Scale weights to meet target volatility
        portfolio_vol = sum(w*v for w, v in zip(weights, volatilities))
        scaling_factor = target_vol / portfolio_vol
        scaled_weights = [w * scaling_factor for w in weights]
        
        return scaled_weights

    def process_orders(self, date):
        ##### NEED TO CHANGE: POSITION SIZE AND NB OF UNITS SHOULD BE CALCULATED IN A SEPARATE METHOD
        #### THIS FUNCTION SHOULD TAKE THE QUANTITY AND TYPE OF ORDER (SELL OR BUY) AS INPUTS
        #### SHOULD ALSO BE ABLE TO PROCESS LIQUIDATION ORDERS HERE
        open_orders = self.shared_state.get_open_orders()
        if len(open_orders) > 0:
                    # Calculate risk parity weights for all coins with open orders
            coins_to_consider = list(open_orders.keys())

            for coin in coins_to_consider:
                price = self.data_manager.fetch_current_price(coin, date, 'Open')
                signal = open_orders[coin]['signal']
                strategy = open_orders[coin]['strategy']
                size = self.size_position(date, coin)
                price = price * (1 + (signal * self.slippage))

                # Check if shorting is allowed for this strategy
                if signal < 0 and coin not in self.POSITIONS and not self.allow_short[strategy]:
                    #print(f"Ignoring short order for {coin} as shorting is not allowed for {strategy} strategy")
                    del open_orders[coin]
                    continue

                if coin not in self.POSITIONS and len(self.POSITIONS) < self.MAX_POSITIONS:
                    print(f'Position size: {size}')
                    desired_nb_units = (self.AUM * size) / price #1/self.MAX_POSITIONS *
                    print(f'Desired number of units: {desired_nb_units}')
                    max_nb_units = (self.Cash-self.MIN_CASH_LEFT) / (price * (1 + self.transaction_cost))
                    print(f'Max nb of units: {max_nb_units}')
                    max_portfolio_units = (self.params['max_allocation_per_coin'] * self.AUM) / price #Allow max 25% allocation in one coin
                    nb_units = min(desired_nb_units, max_nb_units, max_portfolio_units)
                    exposure = nb_units * price
                    actual_size = exposure / self.AUM
                    self.SIZES[coin] = actual_size 
                    proceeds = nb_units * price * (1 + (signal * self.transaction_cost))
                    if nb_units > 0:
                        ORDER = 'SHORT' if signal == -1 else 'LONG'
                        print(f'Processing {ORDER}: traded {nb_units} {coin} at ${price} per unit on {date}')
                        if date not in self.PROCESSED_ORDERS:
                            self.PROCESSED_ORDERS[date] = {}
                        self.PROCESSED_ORDERS[date][coin] = {'type': ORDER, 'qty': nb_units, 'price': price, 'strategy': strategy}
                        self.Cash += -signal * proceeds
                        self.POSITIONS[coin] = {'units': signal * nb_units, 'strategy': strategy}

                elif coin in self.POSITIONS:
                    position = self.POSITIONS[coin]
                    nb_units = position['units']
                    if abs(nb_units) > 0:
                        ORDER = 'SHORT' if signal == -1 else 'LONG'
                        print(f'Closing position on {coin} with a {ORDER} order: traded {nb_units} at ${price} per unit on {date}')
                        if date not in self.PROCESSED_ORDERS:
                            self.PROCESSED_ORDERS[date] = {}
                        self.PROCESSED_ORDERS[date][coin] = {'type': ORDER, 'qty': nb_units, 'price': price, 'strategy': strategy}
                        #print(f"Updated PROCESSED_ORDERS: {self.PROCESSED_ORDERS[date]}")
                        proceeds = abs(nb_units) * price * (1 + signal * self.transaction_cost)
                        #print(f'Total proceeds: {proceeds}')
                        self.Cash += -signal * proceeds
                        del self.POSITIONS[coin]

                del open_orders[coin]
        self.shared_state.set_open_orders(open_orders)

    def get_entry_price(self, coin):
        """
        Get the entry price for a given coin position by looking at the most recent PROCESSED_ORDERS.
        """
        for date in sorted(self.PROCESSED_ORDERS.keys(), reverse=True):
            if coin in self.PROCESSED_ORDERS[date]:
                return self.PROCESSED_ORDERS[date][coin]['price']
        print(f"No entry price found for {coin}")
        return None  # Return None if no entry found

    def process_mean_reversion_orders(self, date): #ORDERS

        if len(self.OPEN_ORDERS) > 0:
          for coin in list(self.OPEN_ORDERS.keys()): #We should actually iterate by order of priority - first through sell signals,
          #then for buy signals, by order of signal strength - see how we rank coins further below
            price = self.fetch_current_price(coin, date, 'Open')
            signal = self.OPEN_ORDERS[coin]
            size = self.size_position(date, coin)
            self.SIZES[coin] = size * signal
            #When buying, price should be more expensive. When selling, price should be cheaper.
            price = price * (1 + (signal * self.slippage))

            if coin not in self.POSITIONS and len(self.POSITIONS) < self.MAX_POSITIONS:
              desired_nb_units = (1/self.MAX_POSITIONS * self.AUM * size) / price
              max_nb_units = (self.Cash-self.MIN_CASH_LEFT) / (price * (1 + self.transaction_cost))
              nb_units = min(desired_nb_units, max_nb_units)
              proceeds = nb_units * price * (1 + (signal * self.transaction_cost))
              if nb_units > 0:
                ORDER = 'SHORT' if signal == -1 else 'LONG'
                print(f'Processing {ORDER}: traded {nb_units} {coin} at ${price} per unit on {date}')
                self.PROCESSED_ORDERS[date] = {coin: ORDER, 'qty': nb_units, 'price': price}
                self.Cash += -signal * proceeds
                self.POSITIONS[coin] = signal * nb_units

            elif coin in self.POSITIONS:
              nb_units = self.POSITIONS.get(coin, 0)
              if abs(nb_units) > 0:
                ORDER = 'SHORT' if signal == -1 else 'LONG'
                print(f'Closing position on {coin} with a {ORDER} order: traded {nb_units} at ${price} per unit on {date}')
                self.PROCESSED_ORDERS[date] = {coin: ORDER, 'qty': nb_units, 'price': price}
                proceeds = abs(nb_units) * price * (1 + signal * self.transaction_cost)
                print(f'Total proceeds: {proceeds}')
                self.Cash += -signal * proceeds
                del self.POSITIONS[coin]

            del self.OPEN_ORDERS[coin]

    def update_portfolio(self, date):
        self.equity = sum(self.POSITIONS[ticker]['units'] * self.data_manager.fetch_current_price(ticker, date, 'Close') for ticker in self.POSITIONS)
        self.HOLDINGS[date] = {ticker: self.POSITIONS[ticker]['units'] * self.data_manager.fetch_current_price(ticker, date, 'Close') for ticker in self.POSITIONS}
        self.AUM = self.equity + self.Cash
        self.POSITIONS = {key: {'units': float(value['units']), 'strategy': value['strategy']} for key, value in self.POSITIONS.items()}


        print(f'Portfolio position holdings on {date}: {self.POSITIONS}')
        date_holdings = self.HOLDINGS[date]
        date_holdings = {key: float(value) for key, value in date_holdings.items()}
        print(f'Portfolio position values on {date}: {date_holdings}')
        current_price = {ticker: self.data_manager.fetch_current_price(ticker, date, 'Close') for ticker in self.POSITIONS}
        print(f'Current prices of held assets: {current_price}')
        print(f'Total Asset Value: ${self.equity}')
        if self.allow_short[self.params['strategy']] == False and self.equity < 0:
            raise ValueError('Equity is negative!')
        print(f'Total Cash Value: ${self.Cash}')
        print(f'Total Portfolio Value = ${self.AUM}')
        print('='*50)
        print('\n')

    def check_rebalancing(self, date):
        if len(self.POSITIONS) > 0:
            for coin, position_info in self.POSITIONS.items():
                vol = self.get_volatility(date, coin)
                current_position = self.POSITIONS[coin]['units']
                current_price = self.data_manager.fetch_current_price(coin, date, 'Open')
                current_exposure = current_position * current_price
                current_allocation = current_exposure / self.AUM
                current_target_vol = vol * current_allocation
                target_allocation = self.vol_target / vol
                target_nb_units = (target_allocation * self.AUM) / current_price
                print(f'Current target vol: {current_target_vol}')
                if current_target_vol > self.vol_target * (1 + self.vol_threshold):
                    print(f'Rebalancing need detected on {date} for {coin}: Need to decrease exposure to meet volatility target')
                    print(f'Current price: {current_price}')
                    print(f'Current held units: {current_position}')
                    print(f'Current exposure = {current_exposure}, current AUM = {self.AUM}')
                    print(f'Current allocation = {current_allocation}. Target allocation = {target_allocation}')
                    order_size = (target_nb_units - current_position) / current_position
                    print(f'Hence, need to sell {order_size * current_position} units')
                    self.REBALANCING_ORDERS[coin] = {
                    'order_size': order_size,
                    'strategy': position_info['strategy']
                    }
                #elif current_target_vol < self.vol_target * (1 - self.vol_threshold):
                #    print('Current vol is too low')
                #    print(f'Rebalancing need detected on {date} for {coin}: Need to increase exposure to meet volatility target')
                #    print(f'Current size = {current_size}. Optimal size = {new_optimal_size}')
                #    order_size = (new_optimal_size - current_size) / current_size
                #    print(f'Hence, need to process an order of {order_size*100: .2f}% of held units on next open')
                #    self.REBALANCING_ORDERS[coin] = {
                #    'order_size': order_size,
                #    'strategy': position_info['strategy']
                #    }

    def process_rebalancing(self, date):
        if len(self.REBALANCING_ORDERS) > 0:
            for coin, rebalance_info in list(self.REBALANCING_ORDERS.items()):
                if coin not in self.POSITIONS or self.POSITIONS[coin]['units'] == 0:
                    del self.REBALANCING_ORDERS[coin]
                    continue

                current_position = self.POSITIONS[coin]['units']
                current_strategy = self.POSITIONS[coin]['strategy']
                price = self.data_manager.fetch_current_price(coin, date, 'Open')
                order_size = rebalance_info['order_size']

                #print(f'ORDER size: {order_size}')

                # Implement a minimum threshold for rebalancing
                if abs(order_size) < 0.01:  # 1% minimum rebalancing threshold
                    continue

                nb_units = order_size * current_position

                #print(f'Nb units to sell: {nb_units}')
                
                # Check if we're trying to short when it's not allowed
                if nb_units < 0 and not self.allow_short[current_strategy]:
                    nb_units = max(nb_units, -current_position)  # Limit to closing the position
                
                #print(f'new nb units: {nb_units}')

                if abs(nb_units) > 0:
                    ORDER = 'LONG' if nb_units > 0 else 'SHORT'
                    if ORDER == 'LONG':
                        del self.REBALANCING_ORDERS[coin]
                        continue
                    else:
                        proceeds = abs(nb_units) * price * (1 + np.sign(nb_units) * self.transaction_cost)

                        # Check if we have enough cash for a long order
                        if ORDER == 'LONG' and proceeds > self.Cash:
                            nb_units = (self.Cash / price) / (1 + self.transaction_cost)
                            proceeds = nb_units * price * (1 + self.transaction_cost)

                        print(f'Processing rebalancing {ORDER} order of {abs(nb_units)} units for {coin} at ${price} per unit on {date}')
                        
                        if date not in self.PROCESSED_REBALANCING:
                            self.PROCESSED_REBALANCING[date] = {}
                        self.PROCESSED_REBALANCING[date][coin] = {
                            'type': ORDER, 
                            'qty': abs(nb_units), 
                            'price': price,
                            'strategy': current_strategy
                        }
                        
                        self.Cash -= np.sign(nb_units) * proceeds
                        self.POSITIONS[coin]['units'] += nb_units

                        # If position is closed, remove it
                        if self.POSITIONS[coin]['units'] == 0:
                            del self.POSITIONS[coin]

                    del self.REBALANCING_ORDERS[coin]

            print(f"Cash balance after rebalancing: ${self.Cash}")

    def exit_all_positions(self, position_type='long'):
        open_orders = self.shared_state.get_open_orders()
        if len(self.POSITIONS) > 0:
            for coin, position in self.POSITIONS.items():
                units = position['units']
                if units > 0 and position_type == 'long':
                    open_orders[coin] = {'signal': -1, 'strategy': position['strategy']}
                    self.shared_state.set_open_orders(open_orders)
                if units < 0 and position_type == 'short':
                    open_orders[coin] = {'signal': 1, 'strategy': position['strategy']}
                    self.shared_state.set_open_orders(open_orders)

    def check_stop_loss(self, date):
        """
        Check unrealized return on each position and liquidate if it exceeds the stop loss threshold.
        
        :param date: Current date in the backtesting loop
        :param stop_loss_threshold: Maximum allowed unrealized loss as a decimal (e.g., 0.1 for 10%)
        """

        positions_to_liquidate = []

        for coin, position_info in self.POSITIONS.items():
            entry_price = self.get_entry_price(coin)
            if entry_price is None:
                print(f"Warning: No entry price found for {coin}. Skipping stop loss check.")
                continue
            
            current_price = self.data_manager.fetch_current_price(coin, date, 'Close')
            
            position = position_info['units']  # Get the position size from the new structure
            strategy = position_info['strategy']  # Get the strategy information
            
            if position > 0:  # Long position
                unrealized_return = (current_price - entry_price) / entry_price
            else:  # Short position
                unrealized_return = (entry_price - current_price) / entry_price
            
            if unrealized_return < -self.POSITION_STOP_LOSS:
                positions_to_liquidate.append(coin)
                print(f"Stop loss triggered for {coin} ({strategy}) on {date}. Unrealized return: {unrealized_return:.2%}")
        
        return positions_to_liquidate


    def liquidate_position(self, date, positions_to_liquidate):
        """
        Liquidate specific positions.
        
        :param date: Current date
        :param positions_to_liquidate: List of coins to liquidate
        """
        for coin in positions_to_liquidate:
            position_info = self.POSITIONS[coin]
            position = position_info['units']
            strategy = position_info['strategy']
            price = self.data_manager.fetch_current_price(coin, date, 'Close')
            proceeds = abs(position) * price
            
            if position > 0:  # Long position
                order_type = 'SELL'
                self.Cash += proceeds * (1 - self.transaction_cost)
            else:  # Short position
                order_type = 'BUY TO COVER'
                self.Cash -= proceeds * (1 + self.transaction_cost)
            
            print(f"Liquidating {coin} position ({strategy}): {order_type} {abs(position)} units at ${price:.2f} per unit on {date}")
            
            if date not in self.PROCESSED_ORDERS:
                self.PROCESSED_ORDERS[date] = {}
            self.PROCESSED_ORDERS[date][coin] = {
                'type': order_type, 
                'qty': abs(position), 
                'price': price, 
                'strategy': strategy
            }
            
            del self.POSITIONS[coin]