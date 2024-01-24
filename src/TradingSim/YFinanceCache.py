import os
import yfinance as yf
import json
import pandas as pd
import multiprocessing

class YFinanceCache:

    # Read in pre-fetched data as well as index
    def __init__(self, cache_folder, index_file="index.txt"):

        self.cache_folder = cache_folder
        self.index = []

        # Create the cache folder if it doesn't exist
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)

        # Read index file so that we know what data we already have
        self.index_filepath = os.path.join(self.cache_folder, index_file)
        if os.path.exists(self.index_filepath):
            with open(self.index_filepath, 'r') as fin:
                self.index = fin.readlines()


    def _compute_uid(self, start=None, end=None, ticker=None):

        # Check if we have data already, and if so, load it
        s_dt = start.strftime("%Y-%m-%d")
        e_dt = end.strftime("%Y-%m-%d")
        uid = f"{ticker}_{s_dt}_{e_dt}"

        return uid


    def update_cache(self, df=None, path=None, uid=None, start=None, end=None, ticker=None, env_num=None):

        if not path:
            uid = self._compute_uid(start, end, ticker)
            path = os.path.join(self.cache_folder + f"/{env_num}/", f"{uid}.csv")

        if uid not in self.index:
            self.index.append(uid + "\n")
            with open(self.index_filepath, 'w') as fout:
                fout.writelines(self.index)

        if os.path.exists(path):
            os.remove(path)
        df.to_csv(path, index=False)


    def download(self, ticker, start=None, end=None, period=None, **kwargs):

        # Check if we have data already, and if so, load it
        uid = self._compute_uid(start, end, ticker)
        cache_filepath = os.path.join(self.cache_folder, f"{uid}.csv")

        # Load data from file if it exists
        if uid in self.index:

            # Read cached data to pandas df
            df = pd.read_csv(cache_filepath)

        # ...If not, download and cache it
        else:

            df = yf.download(ticker, start=start, end=end, progress=False)

        # Return the data regardless of how we got it
        return df

