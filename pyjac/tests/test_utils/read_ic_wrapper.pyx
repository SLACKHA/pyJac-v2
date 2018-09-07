import cython
import numpy as np
cimport numpy as np
from cpython cimport bool

cdef extern from "read_initial_conditions${header_ext}":
    void read_initial_conditions (const char *filename, unsigned int NUM,
                         double *phi_host, double *param_host,
                         const char order);

cdef char C_ord = 'C'
cdef char F_ord = 'F'

@cython.boundscheck(False)
@cython.wraparound(False)
def read_ics(const char* filename,
            np.uint_t NUM,
            np.ndarray[np.float64_t] phi,
            np.ndarray[np.float64_t] param,
            bool C_order):
    read_initial_conditions(filename, NUM, &phi[0], &param[0],
                            C_ord if C_order else F_ord)
    return None
