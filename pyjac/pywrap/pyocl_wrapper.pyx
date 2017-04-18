import cython
import numpy as np
cimport numpy as np

cdef extern from "species_rates_kernel.h":
    void species_rates_kernel(np.uint_t problem_size, np.uint_t num_devices, double* phi, double* P, double* dphi)
    void finalize()
    void compiler()

cdef int compiled = 0
@cython.boundscheck(False)
@cython.wraparound(False)
def species_rates(np.uint_t problem_size,
            np.uint_t num_devices,
            np.ndarray[np.float64_t] phi,
            np.ndarray[np.float64_t] P,
            np.ndarray[np.float64_t] dphi):
    global compiled
    if not compiled:
        compiler()
        compiled = True
    species_rates_kernel(problem_size, num_devices, &phi[0], &P[0], &dphi[0])
    return None

def __dealloc__(self):
    finalize()