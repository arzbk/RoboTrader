# indicator.pyx
import numpy as np
cimport numpy as np

class Transform:
    cdef np.ndarray[float, ndim=1] open_arr
    cdef np.ndarray[float, ndim=1] close_arr
    cdef np.ndarray[float, ndim=1] high_arr
    cdef np.ndarray[float, ndim=1] low_arr
    cdef np.ndarray[int, ndim=1] volume_arr

    cpdef apply_all(self, )