import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import*
import plotly.graph_objs as go

class DataHandler:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get_data()

    def get_data(self):
        """
        Imports data from the MySQL table, which contains data from AlphaVantage.
        Uses the import_stock_data() function created in the Functions.py file.
        Returns the log-returns of the symbol.
        """
        raw = import_stock_data(self.symbol, start_date=self.start_date, end_date=self.end_date) 
        raw['return'] = np.log(raw["price"] / raw["price"].shift(1))
        num_na_rows = raw.isna().any(axis=1).sum()
        print("Number of rows with missing values:", num_na_rows)
        raw.dropna(inplace=True)
        return raw
    
    def get_price_ts(self):
        return import_stock_data(self.symbol, start_date=self.start_date, end_date=self.end_date)

    def get_date_price(self, bar):
        date = str(self.data.index[bar])[:10]
        price = self.data.price.iloc[bar]
        #price = np.round(price, 2) (not sure if in reality you buy an asset with a lot of decimals)
        return date, price