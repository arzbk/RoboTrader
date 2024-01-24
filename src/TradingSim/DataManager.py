import sys
from datetime import timedelta
from datetime import datetime
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import pandas_ta
import time
import multiprocessing as mp
import calendar
import random
import csv
from pandas_market_calendars import get_calendar
from .YFinanceCache import *


class DataAgent:
    def __init__(self,
                 seed,
                 queues,
                 queue_ubound,
                 queue_lbound,
                 queue_states,
                 filename="Stock Data/sp500_stocks.csv",
                 cache_dir="yfinance_cache",
                 num_envs=1,
                 num_assets=1,
                 episodes=None,
                 rolling_window_size=None,
                 group_by="date",
                 fixed_start_date=False,
                 range_start_date=None,
                 range_end_date=None,
                 fixed_portfolio=None,
                 ticker_col="ticker",
                 period_months=6,
                 include_ti=False,
                 lookback_window=14,
                 indicator_list=None,
                 indicator_args={}
                ):

        self.seed = seed

        # Reference to the queue for each worker/env pair
        self.queues = queues
        self.queue_ubound = queue_ubound
        self.queue_lbound = queue_lbound
        self.queue_states = queue_states

        self.filename = filename

        # Tracking Variables
        self.start_index = 0
        self.current_step = 0
        self.i = 0
        self.max_steps = 0

        self.episodes = episodes

        # Variables for calculated columns
        self.lead_period = 6
        self.rolling_window_size = rolling_window_size
        self.include_ti = include_ti
        self.lookback = lookback_window
        self.indicator_list = indicator_list
        self.indicator_args = indicator_args

        # Portfolio variables
        self.fixed_start_date = fixed_start_date
        self.range_start_date = range_start_date
        self.range_end_date = range_end_date
        self.fixed_portfolio = fixed_portfolio
        self.p_months = period_months
        self.lead_period = 6
        self.num_envs = num_envs
        self.num_assets = num_assets
        self.lead_date = None
        self.start_date = None
        self.end_date = None
        self.stock_df = None
        self.leading_df = None

        # env training variables
        self.stock_data = {}
        self.leading_data = {}

        # Instantiate yfinance wrapper for cache handling
        self.yf = YFinanceCache(cache_dir)

        # Reads stocks from file
        self.stocks = self.get_tickers_from_file(self.filename, group_by, ticker_col)
        self.quarters = [key for key in list(self.stocks.keys())]
        self.quarter = None

        # Apply optional start and end range to random dates
        f_start = self.range_start_date or datetime(2000, 1, 1)
        if self.range_end_date:
            f_end = self.get_lead_up_period(self.range_end_date, months=period_months)
        else:
            f_end = datetime(2200, 1, 1)
        self.quarters = list(filter(lambda x: f_start <= x <= f_end, self.quarters))


    def run(self, agent_num):

        cntr = 0

        # Run forever - as long as application is running
        while True:

            while True:

                # Set new random seed every time for new data
                random.seed(time.time())

                df = None
                self.stock_df = {}
                self.leading_df = {}

                # Initialize or reset dates for trading
                if self.fixed_start_date and not self.start_date:

                    # Determine start and end date for episode
                    self.start_date, self.end_date = self.get_trading_period(self.fixed_start_date)
                    self.quarter = self.get_last_quarter(self.fixed_start_date)
                    self.lead_date = self.get_lead_up_period(self.start_date, day=1)

                elif not self.fixed_start_date:

                    # Pick a random start date
                    date_idx = random.randint(0, len(self.quarters) - 1)
                    self.quarter = self.quarters[date_idx]

                    # Determine start and end date for episode
                    self.start_date, self.end_date = self.get_trading_period(self.quarter)
                    self.lead_date = self.get_lead_up_period(self.start_date)

                # Initialize or reset stock selections and data
                if self.fixed_portfolio:

                    for ticker in self.fixed_portfolio:

                        df = self.get_stock_data(ticker)
                        if self.is_valid_data(df):
                            self.stock_df[ticker] = df

                            # Prepare dataframe for training and testing
                            self.preprocess_dataframe(df, ticker, agent_num)

                        else:
                            pass
                            raise Exception(f"""
                            Warning: Stock \"{ticker}\" is invalid from {self.start_date} 
                            to {self.end_date}; skipping...""")

                elif not self.fixed_portfolio:

                    for i in range(0, self.num_assets):

                        ticker = None

                        # Keep picking random stocks until 1 is found that satisfies criteria
                        while True:
                            stock_idx = random.randint(0, len(self.stocks[self.quarter]) - 1)
                            ticker = self.stocks[self.quarter][stock_idx]
                            df = self.get_stock_data(ticker)
                            if self.is_valid_data(df):
                                self.stock_df[ticker] = df
                                break

                        # Prepare dataframe for training and testing
                        self.preprocess_dataframe(self.stock_df[ticker], ticker, agent_num)

                # Put environment training data in queue - ready to go...
                cntr += 1
                print(f"A{agent_num}: {cntr}")
                self.queues[agent_num].put(self.stock_data)


    def preprocess_dataframe(self, df, ticker, agent_num):

        # Split df into leading data and observed data
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(df['Date'])
        df = df.sort_index()

        # Set instance vars relating to batch of data
        self.leading_df[ticker] = df.loc[df['Date'] <= self.start_date]

        # Keep standard start and stop indexes for all assets in portfolio
        if self.max_steps:
            self.start_index = max(self.start_index, len(self.leading_df[ticker]))
            self.max_steps = min(self.max_steps, df.index.size - 1)

        else:
            self.start_index = len(self.leading_df[ticker])
            self.max_steps = df.index.size - 1

        # Add computed values to retrieved data if included
        if self.include_ti:
            self.add_technical_indicators(df)

        # Add delta columns (% chng between rows for each col)
        self.add_computed_columns(df)

        # Cache dataframe now that setup is complete
        df = df.loc[df['Date'] >= self.start_date]
        self.yf.update_cache(df=df, start=self.start_date, end=self.end_date, ticker=ticker, env_num=agent_num)
        self.stock_df[ticker] = df

        # Convert dataframes to numpy arrays and add to cython dicts for fast retrieval
        cols = {}
        for col in self.stock_df[ticker].columns:
            cols[col] = self.stock_df[ticker][col].to_numpy()[self.start_index:]
        self.stock_data[ticker] = cols


    # Defines the technical indicators to be used with data and computes / adds them
    def add_technical_indicators(self, df):

        # Gather key data for calculations
        df['Open'] = df['Open'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        # Apply unique, incrementing epsilon to each column so that no zero division ever occurs

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
                    raise Exception(f"Error: An indicator that was specified doesn't exist in the TA-Lib Python library.")

                df[ind] = res


    # Dynamically check for and add missing pct change columns for each regular column
    def add_computed_columns(self, df):

        # Replace zeros with NaN before below operation
        #df[['ROC', 'WILLR']] = df[['ROC', 'WILLR']].replace(0, pd.NA)

        # For each column that is not a delta column, and not date or datetime
        for col in filter(lambda s: "_delta" not in s, df.columns):
            if not is_datetime(df[col]):
                delta_col = col + "_delta"
                if delta_col not in df.columns:
                    df[delta_col] = df[col].pct_change()
                    if self.rolling_window_size:
                        norm_col = col + "_norm"
                        scaled_col = col + "_scaled"
                        if norm_col not in df.columns:

                            # Apply rolling normalization
                            rolling_mean = df[delta_col].rolling(window=self.rolling_window_size).mean()
                            rolling_std = df[delta_col].rolling(window=self.rolling_window_size).std()
                            df[norm_col] = (df[delta_col] - rolling_mean) / rolling_std

                            # Apply min-max scaling to entire column
                            min_value = df[norm_col].min()
                            max_value = df[norm_col].max()
                            df[scaled_col] = (df[norm_col] - min_value) / (max_value - min_value)


    def list_assets(self):
        return list(self.stock_data.keys())


    def __len__(self):
        return self.max_steps


    def next(self):
        self.current_step += 1
        self.i = self.current_step + self.start_index
        retval = {
            ticker: {
                col: self.stock_data[ticker][col][self.i]
                for col in self.stock_data[ticker].keys()
            }
            for ticker in self.stock_data.keys()
        }

        return retval

    def get_leading_data(self):
        return self.leading_data


    def get_last_quarter(self, target_date):

        # Filter dates that are less than the target date
        previous_dates = [date for date in self.quarters if date < target_date]
        if not previous_dates:
            return None

        # Find the closest date by selecting the maximum of the previous dates
        return max(previous_dates)


    # Downloads stock data and checks if it traded for defined period
    def get_stock_data(self, ticker):

        # Use custom yfinance wrapper to reduce network load
        df = self.yf.download(ticker, start=self.lead_date, end=self.end_date, progress=None)

        # Ensure the stock was trading for the entire date range
        if df.empty:
            #print(f"No trading data available for {ticker} between {self.lead_date} and {self.end_date}")
            return None

        return df


    # If day isn't specified, use last day of month (based on fiscal quarters) - otherwise use day
    def get_lead_up_period(self, start_date, day=None, months=None):
        sub_months = self.lead_period if not months else months
        m = start_date.month - sub_months
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

class DataManager:

    def __init__(
            self,
            seed,
            **kwargs
    ):

        # Set static upper and lower queue bound
        queue_ubound = 6
        queue_lbound = 3

        # Setup variables specifically for multiprocessing
        self.conf = kwargs
        self.num_envs = kwargs['num_envs']
        self.queues = []
        self.queue_states = []
        self.constant_mode = False

        if self.conf['fixed_start_date'] and self.conf['fixed_portfolio']:
            print("Running in single dataset mode")
            self.constant_mode = True
            self.num_envs = 1
            queue_ubound = 1
            queue_lbound = -1

        # Used in the case where we aren't varying data across envs
        self.last_queue_item = None

        # Start the agents with shared resources
        for i in range(self.num_envs):
            self.queues.append(multiprocessing.Queue())
            self.queue_states.append(mp.Value('b', False))

        # Create a worker class instance with reference to the queues
        agent = DataAgent(self.queues, queue_ubound, queue_lbound, self.queue_states, **kwargs)

        # Create and run worker agent for each env
        agents = []
        for agent_num in range(self.num_envs):
            ap = multiprocessing.Process(
                target=agent.run,
                args=(agent_num,)
            )
            agents.append(ap)
            ap.start()


    def is_ready(self):
        if all(self.queue_states):
            return True
        else:
            return False


    def new_dataset(self, i):
        if not self.constant_mode:
            return self.queues[i].get()
        elif self.constant_mode and not self.last_queue_item:
            ds = self.queues[0].get()
            self.last_queue_item = ds
            return self.last_queue_item
        elif self.constant_mode and self.last_queue_item:
            return self.last_queue_item


if __name__ == "__main__":

    ######## Used for test data ###########
    num_envs = 4
    num_assets = 2
    episodes = 250
    lookback_steps = 20
    use_indicators = True
    indicators = ["SMA", "RSI", "EMA"]
    indicator_args = {"RSI": {"timeperiod": lookback_steps}, "SMA": {"timeperiod": lookback_steps}, "EMA": {"timeperiod": lookback_steps}}

    tp_trn = {
        "rolling_window_size": 30,
        "period_months": 60,
        "filename": "../../Stock Data/sp500_stocks.csv",
        "cache_dir": "../../yfinance_cache",
        "fixed_start_date": datetime(2011, 1, 1),
        "range_start_date": None,
        "range_end_date": None,
        "fixed_portfolio": None,
        "num_envs": num_envs,
        "num_assets": num_assets,
        "include_ti": use_indicators,
        "indicator_list": indicators,
        "indicator_args": indicator_args
    }

    # Tell the data manager to create a worker for each env
    dm = DataManager(seed=time.time(), **tp_trn)
