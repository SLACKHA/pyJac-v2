import cython
import numpy as np
cimport numpy as np

cdef extern from "species_rates_kernel_main.h":
    void species_rates_kernel_call(np.int_t problem_size, np.int_t num_threads, double* phi, double* P, double* dphi)
    void finalize()

@cython.boundscheck(False)
@cython.wraparound(False)
def species_rates(np.int_t problem_size,
            np.int_t num_threads,
            np.ndarray[np.float64_t] phi,
            np.ndarray[np.float64_t] P,
            np.ndarray[np.float64_t] dphi):
    species_rates_kernel_call(problem_size, num_threads, &phi[0], &P[0], &dphi[0])
    return None

def __dealloc__(self):
    finalize()