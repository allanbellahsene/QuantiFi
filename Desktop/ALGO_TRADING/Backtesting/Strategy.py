from abc import ABC, abstractmethod

class Strategy(ABC):
    def __init__(self):
        self.position = 0

    @abstractmethod
    def should_buy(self, data_handler, portfolio, bar):
        pass

    @abstractmethod
    def should_sell(self, data_handler, portfolio, bar):
        pass

    def run(self, data_handler, portfolio):
        for bar in range(len(data_handler.data)):
            if self.position == 0 and self.should_buy(data_handler, portfolio, bar):
                portfolio.place_order(bar=bar, order_type='buy', amount=portfolio.cash_portfolio)
                self.position = 1

            elif self.position == 1 and self.should_sell(data_handler, portfolio, bar):
                portfolio.place_order(bar=bar, order_type='sell', units=portfolio.units)
                self.position = 0

            portfolio.update_portfolio_value(bar)
            portfolio.update_cash_history()
            portfolio.update_units_history()

        portfolio.close_out(bar)

class MeanReversionStrategy(Strategy):
    def __init__(self, threshold, portfolio):
        super().__init__()
        self.window = portfolio.window
        self.threshold = threshold

    def sma(self, data_handler):
        return data_handler.data['price'].rolling(self.window).mean()

    def should_buy(self, data_handler, portfolio, bar):
        return data_handler.data['price'].iloc[bar] < self.sma(data_handler).iloc[bar] - self.threshold

    def should_sell(self, data_handler, portfolio, bar):
        return data_handler.data['price'].iloc[bar] > self.sma(data_handler).iloc[bar] + self.threshold

class BuyAndHoldStrategy(Strategy):
    def should_buy(self, data_handler, portfolio, bar):
        # Buy on the first bar
        return bar == 0

    def should_sell(self, data_handler, portfolio, bar):
        # Never sell
        return False