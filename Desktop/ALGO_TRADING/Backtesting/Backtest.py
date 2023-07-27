from Portfolio import Portfolio
from DataHandler import DataHandler
from Strategy import Strategy, MeanReversionStrategy
from VisualizeData import VisualizeData


class Backtest:
    def __init__(self, portfolio, data_handler, strategy):
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.strategy = strategy
        self.visualizer = VisualizeData(data_handler)

    def run(self):
        self.strategy.run(self.data_handler, self.portfolio)
        #self.strategy.run(self.data_handler)
        self.portfolio.calculate_metrics()
        self.portfolio.print_metrics()
        self.visualizer.plot_price(self.portfolio)
        self.visualizer.plot_portfolio_value(self.portfolio)
        self.visualizer.plot_cumulative_returns(self.portfolio)