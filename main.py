import pandas as pd
from datetime import datetime
import numpy as np
from backtest_engine import BacktestEngine
from config import START_DATE, END_DATE, INITIAL_CASH, MIN_CASH, MAX_OPEN_POSITIONS, PARAMS
pd.options.mode.chained_assignment = None  # default='warn'


#### TO DO:
#### SHORT TERM MEAN REVERSION WILL WORK MUCH BETTER ON HOURLY DATA: SAME SIGNALS RULE BUT EXECUTION CAN BE EXECUTED EVERY HOUR
#### ADD CONSIDERATIONS FOR PERP FUTURES TRADING: USE FUTURES DATA INSTEAD OF SPOT, CONSIDER MARGIN REQUIREMENTS AND FUNDING RATE PAYMENTS

if __name__ == "__main__":

    backtest = BacktestEngine(INITIAL_CASH, MIN_CASH, MAX_OPEN_POSITIONS, START_DATE, END_DATE, PARAMS)
    results = backtest.run_backtest()
