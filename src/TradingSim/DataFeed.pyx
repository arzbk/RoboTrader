import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from datetime import timedelta
import datetime
from dateutil.relativedelta import relativedelta
import multiprocessing
from pandas.api.types import is_datetime64_any_dtype as is_datetime
import pandas_ta
import pandas as pd
pd.options.mode.chained_assignment = None
import random
import time
import sys

from Indicators_c import Indicator

class DataProcessor:

    def __init__(self, indicator_list, indicator_args, rolling_window_size=30):
        self.rolling_window_size = rolling_window_size
        self.indicator_list = indicator_list
        self.indicator_args = indicator_args


    def validate_data(self, data):

        # Gather key data for calculations
        df = pd.DataFrame()
        df['open'] = data['open']
        df['close'] = data['close']
        df['high'] = data['high']
        df['low'] = data['low']
        df['volume'] = data['volume']
        df['date'] = data['date']
        df.set_index('date', inplace=True, drop=False)

        # Count NaN's and Zeros, and check for both - skipping ticker if it contains them
        nan_count = df.isnull().sum().sum()
        zero_count = (df == 0).sum().sum()

        if nan_count > 0:
            print(f'Failed NaN count ({nan_count}) - trying another ticker...')
            return df, False
        elif zero_count > 0:
            print(f'Failed zero count ({zero_count}) - trying another ticker...')
            return df, False
        else:
            return df, True


    def calculate_indicators(self, df):

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
                    raise Exception(
                        f"Error: An indicator that was specified doesn't exist in the TA-Lib Python library.")

                df[ind] = res

        return df


    def prepare_dataset(self, df, start_date):

        # For each column that is not a delta column, and not date or datetime
        for col in filter(lambda s: "_" not in s, df.columns):
            if col not in ['metadata', 'date']:
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

        # Filter out leading data
        df = df[df['date'] >= start_date]

        # Forward fill NaN and inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(method='ffill', inplace=True)

        # Convert data back to numpy array
        data = {}
        for col in df.columns:
            data[col.lower()] = df[col].to_numpy()

        return data


class StockData:
    def __init__(self):
        self.conn = None
        try:
            self.conn = psycopg2.connect(
                dbname='sharadar',
                user='postgres',
                password='unkQRXen_9',
                host='10.243.199.178',
                port='5432'
            )
            dict_cur = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            self.cur = dict_cur
        except psycopg2.DatabaseError as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"An unspecified error occurred: {e}")


    def close(self, commit=True):
        if commit:
            self.conn.commit()
        self.cur.close()
        self.conn.close()


    def get_tickers_for_date_range(self, target_date, p_months):
        tickers = []
        try:
            target_date = datetime.datetime.strftime(target_date, '%Y-%m-%d')

            # Prepare the query
            query = sql.SQL("""
            SELECT 
                s3.date AS algo_lead_date,
                s.date AS algo_start_date,
                s2.date AS algo_end_date,
                t.*
            FROM sharadar.sep s
            INNER JOIN sharadar.tickers t
            ON t.ticker = s.ticker
            INNER JOIN (
                SELECT 
                    s2.ticker, 
                    s2.date
                FROM sharadar.sep s2
                WHERE 
                s2.date = (
                    SELECT 
                      in_sql.date
                  FROM
                      (
                          SELECT 
                            s3.date,
                            ABS(s3.date - DATE(DATE({target_date}) + INTERVAL '{p_months} MONTHS')) as date_diff
                          FROM 
                            sharadar.sep s3
                          WHERE
                              s3.date BETWEEN (DATE(DATE({target_date}) + INTERVAL '{p_months} MONTHS') - 3) AND (DATE(DATE({target_date}) + INTERVAL '{p_months} MONTHS') + 3)
                      ) in_sql
                  ORDER BY in_sql.date_diff
                  LIMIT 1
               )
            ) s2
            ON s2.ticker = s.ticker
            INNER JOIN (
                SELECT 
                    s3.ticker, 
                    s3.date
                FROM sharadar.sep s3
                WHERE 
                s3.date = (
                    SELECT 
                      in_sql.date
                  FROM
                      (
                          SELECT 
                            s4.date,
                            ABS(s4.date - DATE(DATE({target_date}) - INTERVAL '6 MONTHS')) as date_diff
                          FROM 
                            sharadar.sep s4
                          WHERE
                              s4.date BETWEEN (DATE(DATE({target_date}) - INTERVAL '6 MONTHS') - 3) AND (DATE(DATE({target_date}) - INTERVAL '6 MONTHS') + 3)
                      ) in_sql
                  ORDER BY in_sql.date_diff
                  LIMIT 1
               )
            ) s3
            ON s3.ticker = s.ticker
            WHERE s.date = (
                SELECT 
                    in_sql.date
                FROM
                    (
                        SELECT 
                          s4.date,
                          ABS(s4.date - DATE({target_date})) as date_diff
                        FROM 
                          sharadar.sep s4
                        WHERE
                            s4.date BETWEEN (DATE({target_date}) - 3) AND (DATE({target_date}) + 3)
                    ) in_sql
                ORDER BY in_sql.date_diff
                LIMIT 1
            )
            AND t.category IN ('Domestic Common Stock', 'Canadian Common Stock')
            AND CAST(LEFT(t.scalemarketcap, 1) AS INTEGER) >= 4
            AND t.sector = 'Healthcare';
            """).format(
                target_date=sql.Literal(target_date),
                p_months=sql.Literal(p_months)
            )

            # Execute the query
            self.cur.execute(query)
            tickers = self.cur.fetchall()

        except psycopg2.DatabaseError as e:
            print(f"An error occurred while fetching tickers: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return tickers


    def get_dates_between(self, date_1, date_2):
        dates = []
        try:
            # Prepare the query
            query = sql.SQL("""
            SELECT
                DISTINCT s.date
            FROM
                sharadar.sep s
            WHERE
                s.date BETWEEN {date_1} AND {date_2}
            ORDER BY s.date;
            """).format(
                date_1=sql.Literal(datetime.datetime.strftime(date_1, '%Y-%m-%d')),
                date_2=sql.Literal(datetime.datetime.strftime(date_2, '%Y-%m-%d'))
            )

            # Execute the query
            self.cur.execute(query)
            dates = [row[0] for row in self.cur.fetchall()]

        except psycopg2.DatabaseError as e:
            print(f"An error occurred while fetching dates: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return dates


    def get_closest_date(self, target_date):
        closest_date = None
        try:
            # Prepare the query to find the closest date
            query = sql.SQL("""
            SELECT 
                in_sql.date
            FROM
                (
                    SELECT 
                      s.date,
                      ABS(s.date - DATE({target_date})) as date_diff
                    FROM 
                      sharadar.sep s
                    WHERE
                        s.date BETWEEN (DATE({target_date}) - 3) AND (DATE({target_date}) + 3)
                ) in_sql
            ORDER BY in_sql.date_diff
            LIMIT 1;

            """).format(
                target_date=sql.Literal(datetime.datetime.strftime(target_date, '%Y-%m-%d'))
            )

            # Execute the query
            self.cur.execute(query)
            result = self.cur.fetchone()
            if result:
                closest_date = result[0]

        except psycopg2.DatabaseError as e:
            print(f"An error occurred while fetching the closest date: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return closest_date


    def get_ohlcv_data(self, ticker, start_date, end_date):

        ohlcv_data = []
        try:
            # Prepare the query
            query = sql.SQL("""
            SELECT
                s.ticker,
                s.date,
                s.open,
                s.high,
                s.low,
                s.close,
                s.volume
            FROM
                sharadar.sep s
            WHERE
                s.ticker = {ticker}
                AND s.date >= DATE({start_date})
                AND s.date <= DATE({end_date})
            ORDER BY s.date;
            """).format(
                ticker=sql.Literal(ticker),
                start_date=sql.Literal(datetime.datetime.strftime(start_date, '%Y-%m-%d')),
                end_date=sql.Literal(datetime.datetime.strftime(end_date, '%Y-%m-%d'))
            )

            # Execute the query
            self.cur.execute(query)
            ohlcv_data = self.cur.fetchall()

        except psycopg2.DatabaseError as e:
            print(f"An error occurred while fetching OHLCV data: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

        return ohlcv_data


    def _get_sql_value(self, value):
        """
        Convert Python value to SQL-compatible representation based on its type.
        """
        if isinstance(value, str):
            return value
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, (list, tuple)):
            return str(value)
        elif isinstance(value, datetime.date):
            return datetime.datetime.strftime(value, '%Y-%m-%d')


    def insert_preprocessed_data(self, rows, columns):
        column_names = None

        try:

            # Prepare mogrify string
            mogrify_str = str(f"({','.join(['%s' for _ in range(len(columns))])})")

            # Prepare row data for DB
            row_data = tuple(rows)
            """
            for col in columns:
                print(col)

            print(mogrify_str)
            print(len(columns))
            print(len(row_data[0]))
            for i in range(len(row_data[0])):
                print(f"{row_data[0][i]} == {columns[i]}...")
            """

            args_str = ','.join(self.cur.mogrify(mogrify_str, row).decode("utf-8") for row in row_data)

            # Build query
            query_str = "INSERT INTO sharadar.preprocessed ({cols}) VALUES {vals};".format(
                cols=','.join(columns),
                vals=args_str
            )

            # Execute Query
            self.cur.execute(query_str)

            self.conn.commit()

        except psycopg2.DatabaseError as e:
            print(f"An error occurred while adding cached data: {e}")
            self.conn.rollback()
        except Exception as e:
            print(f"An unspecified error occurred: {e}")
            self.conn.rollback()


    def prep_preprocessed_table(self, cols):

        try:

            # Get current columns from the 'preprocessed' table (excluding some columns)
            self.cur.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'preprocessed';")
            current_columns = [
                row[0].lower()
                for row in self.cur.fetchall()
            ]

            # Step A: Ensure dictionary keys match the column names (case-insensitive)
            data_columns = [col.lower() for col in cols]
            missing_columns = [key for key in data_columns if key not in current_columns]

            if len(missing_columns) > 0:
                print("Missing Columns!")

            # Create the missing columns if needed
            for col in missing_columns:
                print(f" - Creating column {col}")
                self.cur.execute(sql.SQL("ALTER TABLE sharadar.preprocessed ADD COLUMN {} NUMERIC NOT NULL").format(
                    sql.Identifier(col)))
                self.conn.commit()
                print(f" - Done creating column {col}")

        except psycopg2.DatabaseError as e:
            print(f"An error occurred while upserting preprocessed data: {e}")
            self.conn.rollback()
        except Exception as e:
            print(f"An unspecified error occurred: {e}")
            self.conn.rollback()


class DataHelper:

    def __init__(self,
                 mode,
                 range_st,
                 range_nd,
                 p_months,
                 num_envs,
                 num_assets,
                 indicator_list,
                 indicator_args,
                 episodes=2,
                 workers_per_episode=1
    ):

        # Get handle for DB
        db = StockData()

        self.mode = mode
        self.seed = 42
        self.num_envs = num_envs
        self.num_workers = workers_per_episode * num_envs
        self.num_assets = num_assets
        self.episodes = episodes

        # Setup queues for workers
        self.output_queues = [multiprocessing.Queue() for _ in range(num_envs)]

        if self.mode == 'load':
            self.input_queues = [multiprocessing.Queue() for _ in range(num_envs)]
            self.dp = DataProcessor(indicator_list, indicator_args)
            self.p_months = p_months
            self.indicator_list = indicator_list
            self.indicator_args = indicator_args

            self.start_dates = self._get_start_dates(range_st, range_nd, episodes)

            # Prepare the preprocessed table before processing starts
            self.col_list = ['open', 'high', 'low', 'close', 'volume'] + self.indicator_list
            proc_cols = []
            for col in self.col_list:
                col = col.lower()
                proc_cols.append(col)
                proc_cols.append(col + '_delta')
                proc_cols.append(col + '_norm')
                proc_cols.append(col + '_scaled')
            self.col_list = proc_cols
            self.col_list.append('env_num')
            db.prep_preprocessed_table(self.col_list)

        db.close()


    def _get_start_dates(self, start_date, end_date, num_dates):

        # Calculate the range of days between start and end dates
        delta = end_date - start_date

        # Generate a random number of days within the range
        st_dates = []
        for _ in range(num_dates):
            random_days = random.randint(0, delta.days)
            random_date = start_date + timedelta(days=random_days)
            st_dates.append(random_date)

        # Calculate the random date by adding random_days to the start_date
        return st_dates


    def _get_tickers(self, db, start_date, p_months):

        # Get ticker data
        tickers = db.get_tickers_for_date_range(
            start_date,
            p_months
        )

        chosen_tickers = []
        for _ in range(self.num_envs):
            env_tickers = []
            for _ in range(self.num_assets * 3):
                env_tickers.append(tickers[random.randint(0, len(tickers) - 1)])
            chosen_tickers.append(env_tickers)

        return chosen_tickers


    def _assign_to_workers(self):

        """
        from pyinstrument import Profiler

        profiler = Profiler()
        profiler.start()
        """

        db = StockData()
        self.tickers = {}

        for prog, dt in enumerate(self.start_dates):
            print(f"- Fetching tickers for date \"{dt}\" ({prog + 1}/{self.episodes})")
            self.tickers[dt] = self._get_tickers(db, dt, self.p_months)
            for i in range(self.num_envs):
                self.input_queues[i].put(self.tickers[dt][i])

        # Signal to workers their work is complete
        for i in range(self.num_envs):
            self.input_queues[i].put("DONE")

        db.close()
        """
        profiler.stop()

        profiler.print()
        """


    def cache_worker(self, worker_num):

        db = StockData()
        row_data = []
        ticker_cntr = 0
        cols = None
        meta_cols = None

        while True:

            ticker_cntr += 1

            eps_data = self.output_queues[worker_num].get()
            if eps_data == "DONE":
                break

            else:
                for ticker in list(eps_data.keys()):

                    # Prepare rows for mass insert to DB
                    data = eps_data[ticker]

                    # Get columns if we don't have them already
                    if not cols:
                        cols = [col.lower() for col in list(data.keys()) if col != 'metadata']
                        meta_cols = [col.lower() for col in list(data['metadata'].keys())]

                    rows = [
                        tuple([ticker])
                        +
                        tuple(
                             db._get_sql_value(data['metadata'][meta_col])
                             for meta_col in meta_cols
                        )
                        +
                        tuple(
                            db._get_sql_value(data[col][i])
                            for col in cols
                        )
                        for i in range(len(data['close']))
                    ]

                    # Add ticker data to chunked master data list
                    row_data = row_data + rows

                    if ticker_cntr >= 10:
                        db.insert_preprocessed_data(rows=row_data, columns=['ticker']+meta_cols+cols)
                        ticker_cntr = 0
                        row_data = []

        db.insert_preprocessed_data(rows=row_data, columns=['ticker']+meta_cols+cols)
        db.close()


    def preprocessing_worker(self, worker_num):

        # Immediately spawn off secondary process to handle writing to cache table
        p = multiprocessing.Process(target=self.cache_worker, args=(worker_num,))
        p.start()

        print(f"Worker {worker_num} is processing.")

        db = StockData()

        # Set unique random seed
        random.seed(a=self.seed + worker_num)

        data = {}
        col_data = []
        ep = 0
        ticker_cnt = 0
        ticker_i = 0

        while True:
            tickers = self.input_queues[worker_num].get()
            if tickers == "DONE":
                break

            print(f"Worker {worker_num}; Episode {ep}")
            train_data = {}

            while ticker_cnt < self.num_assets:

                ticker_data = tickers[ticker_i]

                # Get key dates from query
                lead_date = ticker_data['algo_lead_date']
                start_date = ticker_data['algo_start_date']
                end_date = ticker_data['algo_end_date']

                # Get OHLCV data - looping until a stock without missing vals is found
                rows = db.get_ohlcv_data(ticker_data['ticker'], lead_date, end_date)
                columns = list(rows[0].keys())
                data = {
                    col: [row[col] for row in rows]
                    for col in columns if col != 'ticker'
                }
                df, is_error_free = self.dp.validate_data(data)

                if is_error_free:

                    # Add calculated columns / indicators
                    df = self.dp.calculate_indicators(df)

                    # Difference, normalize, scale, and trim
                    data = self.dp.prepare_dataset(df, start_date)

                    # Add metadata for training data to dict, and queue for commit to DB
                    meta = {
                        'cik_code': ticker_data['cik_code'],
                        'start_date': start_date,
                        'end_date': end_date,
                        'env_num': worker_num,
                        'p_months': self.p_months
                    }
                    data['metadata'] = meta

                    train_data[ticker_data['ticker']] = data

                    ticker_cnt += 1

                ticker_i += 1

            # Add prepared data to the queue for RL Learning
            self.output_queues[worker_num].put(train_data)

            ep += 1
            ticker_i = 0
            ticker_cnt = 0

        print(f"Worker {worker_num} has finished processing.")

        self.output_queues[worker_num].put("DONE")
        p.join()
        db.close()


    def data_worker(self, worker_num):

        print(f"Worker {worker_num} is processing.")

        db = StockData()

        while True:


                    # Add calculated columns / indicators
                    df = self.dp.calculate_indicators(df)

                    # Difference, normalize, scale, and trim
                    data = self.dp.prepare_dataset(df, start_date)

                    # Add metadata for training data to dict, and queue for commit to DB
                    meta = {
                        'cik_code': ticker_data['cik_code'],
                        'start_date': start_date,
                        'end_date': end_date,
                        'env_num': worker_num,
                        'p_months': self.p_months
                    }
                    data['metadata'] = meta

                    train_data[ticker_data['ticker']] = data

                    ticker_cnt += 1

                ticker_i += 1

            # Add prepared data to the queue for RL Learning
            self.output_queues[worker_num].put(train_data)

            ep += 1
            ticker_i = 0
            ticker_cnt = 0

        print(f"Worker {worker_num} has finished processing.")

        self.output_queues[worker_num].put("DONE")
        p.join()
        db.close()


    def fetch_data(self):

        processes = []
        for i in range(self.num_envs):
            # Create a Process that runs the worker function with the worker number as an argument
            p = multiprocessing.Process(target=self.preprocessing_worker, args=(i,))
            processes.append(p)
            p.start()  # Start the process

        self._assign_to_workers()

        # Wait for all processes to complete

        for p in processes:
            p.join()

        # At this point, all workers have completed
        print("All worker processes have completed.")


# Test it out
if __name__ == "__main__":

    num_assets = 2
    p_months = 48
    num_envs = 8
    lookback_steps = 20
    episodes = 250

    # process, or load
    mode = 'load'

    indicators = ["SMA", "RSI", "EMA"]
    indicator_args = {"RSI": {"timeperiod": lookback_steps}, "SMA": {"timeperiod": lookback_steps},
                      "EMA": {"timeperiod": lookback_steps}}

    dh = DataHelper(
        mode='process',
        range_st=datetime.datetime(2007, 1, 1),
        range_nd=datetime.datetime(2012, 1, 1),
        p_months=p_months,
        num_envs=num_envs,
        num_assets=num_assets,
        indicator_list=indicators,
        indicator_args=indicator_args,
        episodes=episodes
    )
    dh.fetch_data()

    print("Data Preprocessed and Ready!")










