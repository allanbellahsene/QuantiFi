import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import*
import plotly.graph_objs as go

from DataHandler import DataHandler


class Portfolio:
    def __init__(self, initial_amount, data_handler, strategy, ftc=0.0, ptc=0.0):
        self.initial_amount = initial_amount
        self.cash_portfolio = initial_amount  # cash available for trades
        self.asset_portfolio = 0  # total value of assets
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.trades = 0
        self.position = 0
        self.buy_signals = []
        self.sell_signals = []
        self.units_history = [0]
        self.fees_paid = []
        self.data_handler = data_handler
        
        self.strategy = strategy
        
        if self.strategy.window is not None and self.strategy.window > 0:
            self.window = self.strategy.window
            self.portfolio_value = [initial_amount]*self.window
            self.cash_history = [initial_amount]*self.window
       
        
        else:
            self.portfolio_value = [initial_amount]
            self.cash_history = [initial_amount]
            
        

    def place_order(self, bar, order_type, amount=None, units=None, verbose=True):
        date, price = self.data_handler.get_date_price(bar)
        cost_per_unit = price * (1 + self.ptc)  # initial cost per unit, without ftc
        

        if units is None:
            max_units_affordable = int((amount - self.ftc) / cost_per_unit)  # considering ftc
            units = max(0, max_units_affordable)  # ensure units is non-negative
            cost_per_unit += self.ftc / units if units != 0 else 0  # update cost_per_unit to include ftc
            

        if order_type == 'buy':
            self.buy_signals.append(bar)
            total_cost = units * cost_per_unit  # use updated cost_per_unit
            if total_cost > self.cash_portfolio:  # Ensure the portfolio has enough cash
                print("Not enough cash to buy")
                return
            self.cash_portfolio -= total_cost
            self.units += units
            self.cash_history.append(self.cash_portfolio)
            self.units_history.append(self.units)
            self.fees_paid.append(self.ptc*price*units+self.ftc)
            

        elif order_type == 'sell':
            
            self.sell_signals.append(bar)
            self.cash_portfolio += (units * price) * (1 - self.ptc)# Not subtracting ftc here
            self.units -= units
            self.cash_history.append(self.cash_portfolio)
            self.units_history.append(self.units)
            self.fees_paid.append(self.ptc*price*units+self.ftc)
                                     
                                    

        self.trades += 1
        self.asset_portfolio = self.units * price  # Update asset_portfolio after the transaction
                                     

        if verbose:
            print(f'{date} | {order_type}ing {units} units at {price:.2f}$ + {self.ptc*price*units+self.ftc}$ fees')
            self.print_balance(bar)
            self.print_net_wealth(bar)



    def close_out(self, bar, verbose=True):
        date, price = self.data_handler.get_date_price(bar)
        self.cash_portfolio += self.units * price
        self.units = 0
        self.trades += 1
        if verbose:
            print(f'{date} | inventory {self.units} units at {price:.2f}')
            print('=' * 55)
        print('Final balance   [$] {:.2f}'.format(self.cash_portfolio)) 
        perf = ((self.cash_portfolio - self.initial_amount) / self.initial_amount * 100)
        print('Net Performance [%] {:.2f}'.format(perf)) 
        print('Trades Executed [#] {:.2f}'.format(self.trades))
        print('Total fees paid  [$] {:.2f}'.format(sum(self.fees_paid)))  # print total fees paid
        print('=' * 55)      

    def update_portfolio_value(self, bar):
        date, price = self.data_handler.get_date_price(bar)
        self.asset_portfolio = self.units * price  # update asset_portfolio
        net_wealth = self.asset_portfolio + self.cash_portfolio
        self.portfolio_value.append(net_wealth)
        
    def update_cash_history(self):
        self.cash_history.append(self.cash_portfolio)

    def update_units_history(self):
        self.units_history.append(self.units)

    def print_balance(self, bar):
        date, price = self.data_handler.get_date_price(bar)
        print('{} | current cash balance {}$'.format(date, np.round(self.cash_portfolio, 2)))
        print('{} | current asset balance {}$'.format(date, np.round(self.asset_portfolio, 2)))

    def print_net_wealth(self, bar):
        date, price =  self.data_handler.get_date_price(bar)
        net_wealth = self.asset_portfolio + self.cash_portfolio
        print('{} | current net wealth {}$'.format(date, net_wealth))
        
    def calculate_returns(self):
    # Calculate daily returns
        self.returns = [0]
        for i in range(1, len(self.portfolio_value)):
            today = self.portfolio_value[i]
            yesterday = self.portfolio_value[i - 1]
            daily_return = (today - yesterday) / yesterday
            self.returns.append(daily_return)
        return self.returns
    

    def calculate_volatility(self, frequency='annual'):
    # Calculate volatility
        if frequency == 'annual':
            H = np.sqrt(252)
        elif frequency == 'monthly':
            H = np.sqrt(30)
        elif frequency == 'daily':
            H = 1
        else:
            raise ValueError("frequency can only be 'annual', 'monthly' or 'daily'")
        self.volatility = np.std(self.returns) * H

    def calculate_max_drawdown(self):
        # Calculate max drawdown
        peak = self.portfolio_value[0]
        max_drawdown = 0
        for i in range(1, len(self.portfolio_value)):
            if self.portfolio_value[i] > peak:
                peak = self.portfolio_value[i]
            else:
                drawdown = (peak - self.portfolio_value[i]) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        self.max_drawdown = max_drawdown

    def calculate_drawdown_duration(self):
        # Calculate drawdown duration
        peak = self.portfolio_value[0]
        drawdown_end = 0
        drawdown_duration = 0
        for i in range(1, len(self.portfolio_value)):
            if self.portfolio_value[i] >= peak:
                peak = self.portfolio_value[i]
                drawdown_end = i
            duration = i - drawdown_end
            if duration > drawdown_duration:
                drawdown_duration = duration
        self.drawdown_duration = drawdown_duration

    def calculate_var(self, confidence_level=0.05):
        # Calculate Value at Risk (VaR)
        sorted_returns = sorted(self.returns)
        index = int(len(sorted_returns) * confidence_level)
        self.var = sorted_returns[index]

    def calculate_expected_shortfall(self, confidence_level=0.05):
        # Calculate Expected Shortfall (CVaR)
        sorted_returns = sorted(self.returns)
        index = int(len(sorted_returns) * confidence_level)
        self.expected_shortfall = -np.mean(sorted_returns[0:index])
    
    def calculate_metrics(self):
        self.calculate_returns()
        self.calculate_volatility()
        self.calculate_max_drawdown()
        self.calculate_drawdown_duration()
        self.calculate_var()
        self.calculate_expected_shortfall()
        self.metrics = pd.DataFrame({
            'Annual Volatility (%)': [round(self.volatility * 100, 2)],
            'Max Drawdown (%)': [round(self.max_drawdown * 100, 2)],
            'Drawdown Duration': [round(self.drawdown_duration, 2)],
            'Value at Risk (%)': [round(self.var * 100, 2)],
            'Expected Shortfall (%)': [round(self.expected_shortfall * 100, 2)]
        })

    def print_metrics(self):
        print("\nRisk Metrics:")
        display(self.metrics)