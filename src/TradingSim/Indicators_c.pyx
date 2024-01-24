# indicator.pyx
import numpy as np
cimport numpy as np

cdef class Indicator:
    cdef np.ndarray[float, ndim=1] open_arr
    cdef np.ndarray[float, ndim=1] close_arr
    cdef np.ndarray[float, ndim=1] high_arr
    cdef np.ndarray[float, ndim=1] low_arr
    cdef np.ndarray[int, ndim=1] volume_arr

    def __init__(self,
                 open_arr,
                 close_arr,
                 high_arr,
                 low_arr,
                 volume_arr):
        self.open_arr = open_arr
        self.close_arr = close_arr
        self.high_arr = high_arr
        self.low_arr = low_arr
        self.volume_arr = volume_arr

    def sma(self, int window=14):
        cdef np.ndarray[float, ndim=1] result = np.empty_like(self.close_arr)
        result[:window-1] = np.nan
        for i in range(window-1, len(self.close_arr)):
            result[i] = np.mean(self.close_arr[i-window+1:i+1])
        return result

    def rsi(self, int window=14):
        cdef np.ndarray[float, ndim=1] delta = np.diff(self.close_arr)
        cdef np.ndarray[float, ndim=1] gain = np.maximum(delta, 0)
        cdef np.ndarray[float, ndim=1] loss = -np.minimum(delta, 0)

        cdef np.ndarray[float, ndim=1] avg_gain = np.empty_like(gain)
        cdef np.ndarray[float, ndim=1] avg_loss = np.empty_like(loss)

        avg_gain[:window] = np.cumsum(gain[:window]) / np.arange(1, window + 1)
        avg_loss[:window] = np.cumsum(loss[:window]) / np.arange(1, window + 1)

        for i in range(window, len(self.close_arr)):
            avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i]) / window
            avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i]) / window

        cdef np.ndarray[float, ndim=1] rs = avg_gain / avg_loss
        cdef np.ndarray[float, ndim=1] rsi = 100 - (100 / (1 + rs))

        return rsi

    def ema(self, int span=12):
        cdef np.ndarray[float, ndim=1] result = np.empty_like(self.close_arr)
        result[:span-1] = np.nan
        alpha = 2 / (span + 1)

        result[span-1] = np.mean(self.close_arr[:span])
        for i in range(span, len(self.close_arr)):
            result[i] = alpha * self.close_arr[i] + (1 - alpha) * result[i-1]

        return result

    def stoch(self, int window=14):
        cdef np.ndarray[float, ndim=1] lowest_low = np.minimum.reduceat(self.low_arr, range(0, len(self.low_arr), window))
        cdef np.ndarray[float, ndim=1] highest_high = np.maximum.reduceat(self.high_arr, range(0, len(self.high_arr), window))

        cdef np.ndarray[float, ndim=1] k_percent = ((self.close_arr - lowest_low) / (highest_high - lowest_low)) * 100
        cdef np.ndarray[float, ndim=1] d_percent = np.empty_like(k_percent)
        d_percent[:2] = np.nan

        for i in range(2, len(d_percent)):
            d_percent[i] = np.mean(k_percent[i-2:i+1])

        return k_percent  #, d_percent