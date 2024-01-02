from datetime import timedelta
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import calendar
import random
import numpy as np
import pandas as pd
import pandas_ta as ta
import csv
import sys
from pandas_market_calendars import get_calendar

from YFinanceCache import *

class StockData:

    def __init__(self,
                 filename="Stock Data/sp500_stocks.csv",
                 use_sp500=False,
                 group_by="date",
                 ticker_col="ticker",
                 period_months=6,
                 num_assets=1,
                 include_ti=False,
                 lookback_window=14,
                 indicator_list=None,
                 indicator_args={}
                ):

        self.filename = filename

        # Tracking Variables
        self.start_index = None
        self.current_step = None
        self.i = None
        self.max_steps = None

        # Data Variables
        self.ticker = None
        self.p_months = period_months
        self.num_assets = num_assets
        self.lead_date = None
        self.start_date = None
        self.end_date = None
        self.stock_data = None
        self.leading_data = None

        # Set to true when computing indicators
        self.include_ti = include_ti
        self.lookback = lookback_window
        self.indicator_list = indicator_list
        self.indicator_args = indicator_args

        # Instantiate yfinance wrapper for cache handling
        self.yf = YFinanceCache("yfinance_cache")

        # Set to True if only using SP500 index
        self.use_sp500 = use_sp500

        # Reads stocks from file
        self.stocks = self.get_tickers_from_file(self.filename, group_by, ticker_col)
        self.quarters = [key for key in list(self.stocks.keys())]


    def reset(self, seed, new_ticker=False, new_dates=False):

        random.seed(a=seed)
        self.current_step = 0
        df = None

        if new_dates:
            # Pick a random start date
            date_idx = random.randint(0, len(self.quarters) - 1)
            quarter = self.quarters[date_idx]

            # Determine start and end date for episode
            self.start_date, self.end_date = self.get_trading_period(quarter)
            self.lead_date = self.get_lead_up_period(self.start_date)

        if self.use_sp500:

            df = self.get_stock_data(self.use_sp500)
            self.ticker = "SPY"

        else:

            if new_ticker:

                # Keep picking random stocks until 1 trades for the entire year
                while True:
                    stock_idx = random.randint(0, len(self.stocks[self.quarters[date_idx]]) - 1)
                    self.ticker = self.stocks[self.quarters[date_idx]][stock_idx]
                    df = self.get_stock_data()
                    if self.is_valid_data(df):
                        break

        # Split df into leading data and observed data
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(df['Date'])
        df = df.sort_index()

        # Set instance vars relating to batch of data
        self.stock_data = df
        self.leading_data = self.stock_data.loc[self.stock_data['Date'] <= self.start_date]
        self.start_index = len(self.leading_data)
        self.max_steps = self.stock_data.index.size - 1

        # Reset i
        self.i = self.start_index

        # Add computed values to retrieved data
        if self.include_ti:
            self.validate_technical_indicators()

        # Add computed delta columns (% chng between rows for each col)
        self.validate_delta_columns()

        # Cache dataframe now that any applicable indicators have been pre-computed
        self.yf.update_cache(df=df, start=self.start_date, end=self.end_date, ticker=self.ticker)


    # Defines the technical indicators to be used with data and computes / adds them
    def validate_technical_indicators(self):

        # Gather key data for calculations
        df = self.stock_data
        df['Open'] = df['Open'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # Dynamically check for and add missing indicators to dataset
        for ind in self.indicator_list:
            if ind not in df.columns:
                ind_lower = ind.lower()
                if hasattr(df.ta, ind_lower):
                    ind_method = getattr(df.ta, ind_lower)
                    try:
                        args = self.indicator_args[ind]
                        res = ind_method(**args)
                    except KeyError:
                        res = ind_method()
                    if type(res) == pd.DataFrame:
                        res = res[res.columns[0]]
                else:
                    print(f"Warning: An indicator that was specified doesn't exist in the TA-Lib Python library (\"{ind}\").\nContinuing without it.")
                    self.indicator_list.remove(ind)
                    self.indicator_args.pop(ind)

                df[ind] = res


    # Dynamically check for and add missing pct change columns for each regular column
    def validate_delta_columns(self):
        df = self.stock_data

        # For each column that is not a delta column, and not date or datetime
        for col in filter(lambda s: "_delta" not in s, df.columns):
            if not is_datetime(df[col]):
                delta_col = col + "_delta"
                if delta_col not in df.columns:
                    df[delta_col] = df[col].pct_change()


    def __len__(self):
        return self.max_steps


    # Allows subscription of the data directly from outside of the class
    def __getitem__(self, key):
        t_df = None
        if isinstance(key, slice):
            start_i = key.start or 0
            end_i = key.stop or 0
            t_df = self.stock_data.iloc[self.i + start_i:self.i + end_i]
        elif key is None:
            t_df = self.stock_data.iloc[self.i + 0]
        else:
            t_df = self.stock_data.iloc[self.i + key]

        return t_df


    def next(self):
        self.current_step += 1
        self.i = self.current_step + self.start_index


    def get_leading_data(self):
        return self.leading_data


    def get_lead_up_period(self, start_date):
        # Get period leading up that's half the trading period
        m = start_date.month - (self.p_months // 2)
        y = start_date.year
        if m < 1:
            m += 12
            y -= 1
        _, d = calendar.monthrange(y, m)
        lead_dt = datetime(y, m, d)
        lead_dt = self.find_next_trading_day(lead_dt)

        return lead_dt


        # Calculates a valid start and end for trading period of episode


    def get_trading_period(self, quarter):
        # Get valid start date
        st_dt = self.find_next_trading_day(quarter)

        # Get valid end date
        m = quarter.month + self.p_months
        y = quarter.year
        if m > 12:
            m -= 12
            y += 1
        _, d = calendar.monthrange(y, m)
        end_dt = datetime(y, m, d)

        end_dt = self.find_next_trading_day(end_dt)

        return st_dt, end_dt


        # Finds next trading day provided a date


    def find_next_trading_day(self, trading_date):
        trading_calendar = get_calendar("XNYS")
        while not trading_calendar.valid_days(
                start_date=trading_date,
                end_date=trading_date
        ).size > 0:
            trading_date += timedelta(days=1)
        return trading_date


        # Downloads stock data and checks if it traded for defined period


    def get_stock_data(self, use_sp500=False):

        # Use custom yfinance wrapper to reduce network load
        if use_sp500:
            df = self.yf.download("SPY", start=self.lead_date, end=self.end_date, progress=None)
        else:
            df = self.yf.download(self.ticker, start=self.lead_date, end=self.end_date, progress=None)

        # Ensure the stock was trading for the entire date range
        if df.empty:
            print(f"No trading data available for {self.ticker} between {self.lead_date} and {self.end_date}")
            return None

        return df


    def is_valid_data(self, df):
        if df is None:
            return False
        return not df['Adj Close'].isnull().values.any()


    # Reads tickers from a file and groups them by target column(s)
    def get_tickers_from_file(self, file_path, group_by="date", ticker_col="ticker"):
        data_dict = {}
        idx = None
        is_group_list = type(group_by) == list

        with open(file_path, 'r') as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:

                # Handle grouping by 1 or more attributes in file
                if is_group_list:
                    idx = ""
                    for group in group_by:
                        idx += row[group]
                else:
                    idx = row[group_by]
                    if group_by.upper() == "DATE":
                        idx = datetime.strptime(idx, '%Y-%m-%d')

                ticker = row[ticker_col]

                if idx not in data_dict:
                    data_dict[idx] = []

                data_dict[idx].append(ticker)

        return data_dict
