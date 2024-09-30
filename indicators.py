import pandas as pd
import numpy as np

class Indicators:
    @staticmethod
    def wwma(values, n):
        return values.ewm(alpha=1/n, adjust=False).mean()

    @staticmethod
    def atr(df, n=10):
        data = df.copy()
        high = data['High']
        low = data['Low']
        close = data['Close']
        data['tr0'] = abs(high - low)
        data['tr1'] = abs(high - close.shift())
        data['tr2'] = abs(low - close.shift())
        tr = data[['tr0', 'tr1', 'tr2']].max(axis=1)
        atr = Indicators.wwma(tr, n)
        return atr

    @staticmethod
    def calculate_rsi(price, window):
        delta = price.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def average_range(df, n=10):
        """Calculate the average range (high to low) over the last n periods."""
        return (df['High'] - df['Low']).rolling(window=n).mean()