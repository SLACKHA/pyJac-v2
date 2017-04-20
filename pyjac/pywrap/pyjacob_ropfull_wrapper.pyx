import cython
import numpy as np
cimport numpy as np

cdef extern from "species_rates_kernel_main.h":
    void species_rates_kernel_call(np.int problem_size, np.int num_threads,
                        double* phi,
                        double* P,
                        double* dphi,
                        double* rop_fwd,
                        double* rop_rev,
                        double* pres_mod,
                        double* rop_net,)
    void finalize()

@cython.boundscheck(False)
@cython.wraparound(False)
def species_rates(np.uint_t problem_size,
            np.uint_t num_threads,
            np.ndarray[np.float64_t] phi,
            np.ndarray[np.float64_t] P,
            np.ndarray[np.float64_t] dphi,
            np.ndarray[np.float64_t] rop_fwd,
            np.ndarray[np.float64_t] rop_rev,
            np.ndarray[np.float64_t] pres_mod,
            np.ndarray[np.float64_t] rop_net):
    species_rates_kernel_call(problem_size, num_threads, &phi[0], &P[0], &dphi[0],
        &rop_fwd[0], &rop_rev[0], &pres_mod[0], &rop_net[0])
    return None

def __dealloc__(self):
    finalize()