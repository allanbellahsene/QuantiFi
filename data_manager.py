
#data_manager.py

import pandas as pd
from datetime import datetime, timedelta
from utils import Utils
from config_loader import CONFIG
import yfinance as yf

class DataManager:
    def __init__(self):
        self.utils = Utils()
        self.DATA_DIR = CONFIG['DATA_DIR']
        self.MARKET_CAP_DIR = CONFIG['MARKET_CAP_DIR']
        self.START_DATE = CONFIG['backtest']['start_date']
        self.END_DATE = CONFIG['backtest']['end_date']
        self.params = CONFIG['params']
        if not self.utils.check_file_exists_os(self.DATA_DIR, "asset_universe.pkl"):
            print(f'Download asset universe from {self.START_DATE} to {self.END_DATE}...')
            self.asset_universe = self.generate_whole_asset_universe()
            self.utils.save_list(self.asset_universe, f"{self.DATA_DIR}/asset_universe.pkl")
            print(f'Asset universe saved in {self.DATA_DIR}/asset_universe.pkl')
        else:
            self.asset_universe = self.utils.load_list(f'{self.DATA_DIR}/asset_universe.pkl')

        first_date = self.START_DATE - timedelta(days=self.params['lookback_window'])
        price_path = f"binance_crypto_futures_data_20190908-20240917.pkl"

        if not self.utils.check_file_exists_os(self.DATA_DIR, price_path):
            print(f'Downloading historical prices for asset universe from {first_date} to {self.END_DATE}...')
            self.PRICE_DF = yf.download(tickers=self.asset_universe, start=first_date, end=self.END_DATE)[['Open', 'High', 'Low', 'Close', 'Volume']]
            self.utils.save_dataframe_pickle(self.PRICE_DF, f'{self.DATA_DIR}/{price_path}')
        else:
            self.PRICE_DF = self.utils.load_dataframe_pickle(f'{self.DATA_DIR}/{price_path}')

    def fetch_current_price(self, coin, date, price_col='Open'):
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        prices_data = self.PRICE_DF.loc[(self.PRICE_DF.index.strftime('%Y-%m-%d') == date)]
        data = prices_data[[(price_col, coin)]]
        data.columns = data.columns.droplevel(1)
        price = data[price_col].iloc[-1]
        return price

    def fetch_historical_prices(self, assets, lookback_date, last_date, price_col=None):
        if isinstance(lookback_date, datetime):
            lookback_date = lookback_date.strftime('%Y-%m-%d')
        if isinstance(last_date, datetime):
            last_date = last_date.strftime('%Y-%m-%d')
        prices = self.PRICE_DF.loc[(self.PRICE_DF.index >= lookback_date) & (self.PRICE_DF.index <= last_date)]
        
        if isinstance(assets, list) and len(assets) > 0:
            mask = [ticker in assets for ticker in prices.columns.get_level_values(1)]
            prices = prices.loc[:, mask]
        elif isinstance(assets, str):
            prices = prices.loc[:, (slice(None), assets)]
        else:
            raise ValueError("assets must be a non-empty list or a string")

        # If price_col is specified, return only that column
        if price_col:
            if isinstance(assets, list) and len(assets) > 1:
                prices = prices.loc[:, (price_col, slice(None))]
            else:
                prices = prices.loc[:, (price_col, assets if isinstance(assets, str) else assets[0])]
            
            # If we're left with a single column, convert to a Series
            if isinstance(prices, pd.DataFrame):
                prices = prices.squeeze(axis=1)

        return prices

    def update_investment_universe(self, date, min_rank, min_vol_thres):
        date_market_cap = date.strftime('%Y-%m-%d').replace('-', '')
        try:
            market_cap_data = pd.read_csv(f'{self.MARKET_CAP_DIR}/market_cap_snapshot_{date_market_cap}.csv')
            market_cap_data['date'] = date
            market_cap_data['volume'] = market_cap_data['volume (24h)'].apply(Utils.convert_dollar_to_float)
            market_cap_data['7d_perf'] = market_cap_data['% 7d'].apply(Utils.convert_pct_to_float)
            symbols = market_cap_data.loc[(market_cap_data.Rank<=min_rank) & (market_cap_data.volume >= min_vol_thres)].Symbol.unique()
            symbols = [s for s in symbols if 'US' not in s and 'DAI' not in s]
            ASSET_UNIVERSE = [s + '-USD' for s in symbols]
            ASSET_UNIVERSE = [asset.replace('-USD', 'USDT') for asset in ASSET_UNIVERSE]
            return ASSET_UNIVERSE
        except:
            return None


    def generate_whole_asset_universe(self):
        files_to_keep = self.utils.import_market_cap_files()
        N = len(files_to_keep)
        min_volume = self.params['min_volume_thres']
        max_MC = self.params['max_MC_rank']

        SYMBOLS = []

        for file in files_to_keep:

            mc = pd.read_csv(file)
            date = self.utils.extract_date(file)

            mc['volume'] = mc['volume (24h)'].apply(self.utils.convert_dollar_to_float)
            mc['7d_perf'] = mc['% 7d'].apply(self.utils.convert_pct_to_float)
            symbols = mc.loc[(mc.Rank<=max_MC) & (mc.volume >= min_volume)].Symbol.unique()

            symbols = [s for s in symbols if 'US' not in s] #remove stable coins
            symbols = [s for s in symbols if 'DAI' not in s]
            ASSET_UNIVERSE = [s + '-USD' for s in symbols]
            ASSET_UNIVERSE = [asset.replace('-USD', 'USDT') for asset in ASSET_UNIVERSE]

            SYMBOLS.append(ASSET_UNIVERSE)

        SYMBOLS = self.utils.flatten_and_unique(SYMBOLS)

        return SYMBOLS

