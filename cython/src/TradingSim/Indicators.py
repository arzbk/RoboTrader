import pandas as pd
import numpy as np


class Indicator:

    def __init__(self,
                 df,
                 col_open='Open',
                 col_close='Close',
                 col_high='High',
                 col_low='Low',
                 col_volume='Volume'
                 ):
        self.df = df
        self.col_open = col_open
        self.col_close = col_close
        self.col_high = col_high
        self.col_low = col_low
        self.col_volume = col_volume

    def SMA(self, window=14):
        return self.df[self.col_close].rolling(window=window).mean()

    def RSI(self, window=14):
        delta = self.df[self.col_close].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def EMA(self, span=12):
        return self.df[self.col_close].ewm(span=span, adjust=False).mean()

    def STOCH(self, window=14):
        lowest_low = self.df[self.col_low].rolling(window=window).min()
        highest_high = self.df[self.col_high].rolling(window=window).max()

        k_percent = ((self.df[self.col_close] - lowest_low) /
                     (highest_high - lowest_low)) * 100

        d_percent = k_percent.rolling(window=3).mean()  # Adjust as needed

        return k_percent #, d_percent

    def MACD(self, short_window=12, long_window=26, signal_window=9):
        short_ema = self.df[self.col_close].ewm(span=short_window, adjust=False).mean()
        long_ema = self.df[self.col_close].ewm(span=long_window, adjust=False).mean()

        macd_line = short_ema - long_ema
        signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()

        return macd_line, signal_line, macd_line - signal_line

    def ADOSC(self, fast_period=3, slow_period=10):
        high_low_diff = self.df[self.col_high] - self.df[self.col_low]
        adosc_numerator = (self.df[self.col_close] - self.df[self.col_low] -
                           (self.df[self.col_high] - self.df[self.col_close])) * self.df[self.col_volume]

        adosc = adosc_numerator.rolling(window=fast_period).sum() / adosc_numerator.rolling(window=slow_period).sum()

        return adosc

    def OBV(self):
        price_diff = self.df[self.col_close].diff()
        obv_direction = np.where(price_diff > 0, 1, -1)
        obv = (obv_direction * self.df[self.col_volume]).cumsum()

        return obv

    def ROC(self, window=12):
        return ((self.df[self.col_close] - self.df[self.col_close].shift(window)) / self.df[self.col_close].shift(
            window)) * 100

    def WILLR(self, window=14):
        highest_high = self.df[self.col_high].rolling(window=window).max()
        lowest_low = self.df[self.col_low].rolling(window=window).min()

        willr = -100 * ((highest_high - self.df[self.col_close]) / (highest_high - lowest_low))

        return willr

    def DISP(self, window=10):
        sma = self.df[self.col_close].rolling(window=window).mean()
        disparity_index = ((self.df[self.col_close] - sma) / sma) * 100

        return disparity_index

