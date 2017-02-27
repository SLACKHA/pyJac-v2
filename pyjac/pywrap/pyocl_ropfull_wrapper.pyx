import cython
import numpy as np
cimport numpy as np

cdef extern from "species_rates_kernel.h":
    void species_rates_kernel(np.uint_t problem_size, np.uint_t num_devices,
                        double* T, double* P,
                        double* conc, double* wdot,
                        double* rop_fwd, double* rop_rev,
                        double* rop_net, double* pres_mod)
    void finalize()
    void compiler()

cdef int compiled = 0
@cython.boundscheck(False)
@cython.wraparound(False)
def species_rates(np.uint_t problem_size,
            np.uint_t num_devices,
            np.ndarray[np.float64_t] T,
            np.ndarray[np.float64_t] P,
            np.ndarray[np.float64_t] conc,
            np.ndarray[np.float64_t] wdot,
            np.ndarray[np.float64_t] rop_fwd,
            np.ndarray[np.float64_t] rop_rev,
            np.ndarray[np.float64_t] pres_mod,
            np.ndarray[np.float64_t] rop_net,
            np.uint_t force_no_compile = 0):
    global compiled
    if not compiled and not force_no_compile:
        compiler()
        compiled = True
    species_rates_kernel(problem_size, num_devices, &T[0], &P[0], &conc[0], &wdot[0],
        &rop_fwd[0], &rop_rev[0], &pres_mod[0], &rop_net[0])
    return None

def __dealloc__(self):
    finalize()