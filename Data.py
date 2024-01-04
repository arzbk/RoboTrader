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
                 num_assets=1,
                 use_sp500=False,
                 group_by="date",
                 fixed_start_date=False,
                 fixed_portfolio=None,
                 ticker_col="ticker",
                 period_months=6,
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
        self.fixed_start_date = fixed_start_date
        self.fixed_portfolio = fixed_portfolio
        self.p_months = period_months
        self.lead_period = 3 # 3 Months of data prior to observed trading period
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
        self.quarter = None


    def reset(self, seed, new_tickers=False, new_dates=False):

        random.seed(a=seed)
        self.current_step = 0
        df = None

        # Initialize or reset dates for trading
        if self.fixed_start_date and not self.start_date:

            # Determine start and end date for episode
            self.start_date, self.end_date = self.get_trading_period(self.fixed_start_date)
            self.quarter = self.get_last_quarter(self.fixed_start_date)
            self.lead_date = self.get_lead_up_period(self.start_date, day=1)

        elif new_dates:

            # Pick a random start date
            date_idx = random.randint(0, len(self.quarters) - 1)
            self.quarter = self.quarters[date_idx]

            # Determine start and end date for episode
            self.start_date, self.end_date = self.get_trading_period(self.quarter)
            self.lead_date = self.get_lead_up_period(self.start_date)

        # Initialize or reset stock selections and data
        if self.fixed_portfolio and not self.stock_data:

            self.stock_data = {}
            self.leading_data = {}

            for ticker in self.fixed_portfolio:

                df = self.get_stock_data(ticker)
                if self.is_valid_data(df):
                    self.stock_data[ticker] = df

                    # Prepare dataframe for training and testing
                    self.preprocess_dataframe(df, ticker)

                else:
                    print(f"Warning: Stock \"{ticker}\" is invalid from {self.start_date} to {self.end_date}; skipping...")

        elif self.use_sp500 and not self.stock_data:

            df = self.get_stock_data(self.use_sp500)
            self.ticker = "SPY"

            # Prepare dataframe for training and testing
            self.preprocess_dataframe()

        elif new_tickers or not self.stock_data:

            self.stock_data = {}
            self.leading_data = {}

            for i in range(0, self.num_assets):

                ticker = None

                # Keep picking random stocks until 1 is found that satisfies criteria
                while True:
                    stock_idx = random.randint(0, len(self.stocks[self.quarter]) - 1)
                    ticker = self.stocks[self.quarter][stock_idx]
                    df = self.get_stock_data(ticker)
                    if self.is_valid_data(df):
                        self.stock_data[ticker] = df
                        break

                # Prepare dataframe for training and testing
                self.preprocess_dataframe(self.stock_data[ticker], ticker)

        # Reset i for new episode
        self.i = self.start_index

        return list(self.stock_data.keys())

    def preprocess_dataframe(self, df, ticker):

        # Split df into leading data and observed data
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(df['Date'])
        df = df.sort_index()

        # Set instance vars relating to batch of data
        self.leading_data[ticker] = df.loc[df['Date'] <= self.start_date]

        # Keep standard start and stop indexes for all assets in portfolio
        if self.max_steps:
            self.start_index = max(self.start_index, len(self.leading_data[ticker]))
            self.max_steps = min(self.max_steps, df.index.size - 1)

        else:
            self.start_index = len(self.leading_data[ticker])
            self.max_steps = df.index.size - 1

        # Add computed values to retrieved data if included
        if self.include_ti:
            self.validate_technical_indicators(df)

        # Add delta columns (% chng between rows for each col)
        self.validate_delta_columns(df)

        # Cache dataframe now that setup is complete
        self.yf.update_cache(df=df, start=self.start_date, end=self.end_date, ticker=ticker)
        self.stock_data[ticker] = df


    # Defines the technical indicators to be used with data and computes / adds them
    def validate_technical_indicators(self, df):

        # Gather key data for calculations
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
    def validate_delta_columns(self, df):

        # Replace zeros with NaN before below operation
        df[['ROC', 'WILLR']] = df[['ROC', 'WILLR']].replace(0, pd.NA)

        # For each column that is not a delta column, and not date or datetime
        for col in filter(lambda s: "_delta" not in s, df.columns):
            if not is_datetime(df[col]):
                delta_col = col + "_delta"
                if delta_col not in df.columns:
                    df[delta_col] = df[col].pct_change()

        #TODO: Come up with a better more feature-wise solution to handling inf values
        df[['ROC', 'WILLR']] = df[['ROC', 'WILLR']].replace([np.inf, -np.inf], pd.NA)
        df[['ROC', 'WILLR']] = df[['ROC', 'WILLR']].fillna(method='ffill')


    def list_assets(self):
        return list(self.stock_data.keys())


    def __len__(self):
        return self.max_steps


    # Allows subscription of the data directly from outside of the class
    def __getitem__(self, key):
        retval = {}
        if isinstance(key, slice):
            start_i = key.start or 0
            end_i = key.stop or 0
            for ticker in self.stock_data:
                df = self.stock_data[ticker].iloc[self.i + start_i:self.i + end_i]
                retval[ticker] = df
        elif key is None:
            for ticker in self.stock_data:
                retval[ticker] = self.stock_data[ticker].iloc[self.i + 0]
        else:
            for ticker in self.stock_data:
                retval[ticker] = self.stock_data[ticker].iloc[self.i + key]

        return retval


    def next(self):
        self.current_step += 1
        self.i = self.current_step + self.start_index


    def get_leading_data(self):
        return self.leading_data


    def get_last_quarter(self, target_date):

        # Filter dates that are less than the target date
        previous_dates = [date for date in self.quarters if date < target_date]
        if not previous_dates:
            return None

        # Find the closest date by selecting the maximum of the previous dates
        return max(previous_dates)


    # If day isn't specified, use last day of month (based on fiscal quarters) - otherwise use day
    def get_lead_up_period(self, start_date, day=None):
        m = start_date.month - self.lead_period
        y = start_date.year
        while m < 1:
            m += 12
            y -= 1
        d = day if day else calendar.monthrange(y, m)[1]
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
        while m > 12:
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
    def get_stock_data(self, ticker, use_sp500=False):

        # Use custom yfinance wrapper to reduce network load
        if use_sp500:
            df = self.yf.download("SPY", start=self.lead_date, end=self.end_date, progress=None)
        else:
            df = self.yf.download(ticker, start=self.lead_date, end=self.end_date, progress=None)

        # Ensure the stock was trading for the entire date range
        if df.empty:
            print(f"No trading data available for {ticker} between {self.lead_date} and {self.end_date}")
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
