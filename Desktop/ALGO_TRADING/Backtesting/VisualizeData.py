import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import*
import plotly.graph_objs as go

from DataHandler import DataHandler
from Portfolio import Portfolio
from Strategy import Strategy, MeanReversionStrategy

class VisualizeData:
    def __init__(self, data_handler):
        self.data_handler = data_handler

    def plot_price(self, portfolio):
        data = go.Scatter(x=self.data_handler.data.index, y=self.data_handler.data['price'],
                          mode='lines', name='price', line=dict(color='blue'))
        buy_signals = go.Scatter(x=[self.data_handler.data.index[i] for i in portfolio.buy_signals],
                                 y=[self.data_handler.data['price'].iloc[i] for i in portfolio.buy_signals],
                                 mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up'))
        sell_signals = go.Scatter(x=[self.data_handler.data.index[i] for i in portfolio.sell_signals],
                                  y=[self.data_handler.data['price'].iloc[i] for i in portfolio.sell_signals],
                                  mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down'))
        layout = go.Layout(title='Price Over Time with Buy/Sell Signals',
                           xaxis=dict(title='Time'),
                           yaxis=dict(title='Price'),
                           showlegend=True)
        fig = go.Figure(data=[data, buy_signals, sell_signals], layout=layout)
        fig.show()

    def plot_portfolio_value(self, portfolio):
        data = go.Scatter(x=self.data_handler.data.index[:len(portfolio.portfolio_value)], y=portfolio.portfolio_value,
                          mode='lines', name='Portfolio Value', line=dict(color='blue'))
        layout = go.Layout(title='Portfolio Value Over Time',
                          xaxis=dict(title='Time'),
                          yaxis=dict(title='Value'),
                          showlegend=True)
        fig = go.Figure(data=[data], layout=layout)
        fig.show()
        
    def plot_cumulative_returns(self, portfolio):
        returns = np.array(portfolio.returns)
        cum_returns = np.cumprod(1 + returns) - 1
        cum_returns *= 100
        data = go.Scatter(x=self.data_handler.data.index[:len(portfolio.returns)], y=cum_returns,
                          mode='lines', name='Cumulative Return %', line=dict(color='blue'))
        layout = go.Layout(title='Cumulative Portfolio Return Over Time',
                           xaxis=dict(title='Time'),
                           yaxis=dict(title='Cumulative Return %'),
                           showlegend=True)
        fig = go.Figure(data=[data], layout=layout)
        fig.show()



    def plot_cash_portfolio(self, portfolio):
        data = go.Scatter(x=self.data_handler.data.index[:len(portfolio.cash_history)], y=portfolio.cash_history,
                          mode='lines', name='Cash', line=dict(color='blue'))
        layout = go.Layout(title='Cash Portfolio Value Over Time',
                           xaxis=dict(title='Time'),
                           yaxis=dict(title='Value'),
                           showlegend=True)
        fig = go.Figure(data=[data], layout=layout)
        fig.show()

    def plot_units(self, portfolio):
        data = go.Scatter(x=self.data_handler.data.index[:len(portfolio.units_history)], y=portfolio.units_history,
                          mode='lines', name='Units', line=dict(color='blue'))
        layout = go.Layout(title='Units Over Time',
                           xaxis=dict(title='Time'),
                           yaxis=dict(title='Units'),
                           showlegend=True)
        fig = go.Figure(data=[data], layout=layout)
        fig.show()
        