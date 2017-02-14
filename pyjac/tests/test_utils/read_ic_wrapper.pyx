import cython
import numpy as np
cimport numpy as np
from python_version cimport PY_MAJOR_VERSION

cdef extern from "read_initial_conditions.h":
    void read_initial_conditions (const_char *filename, unsigned int NUM,
                         double *T_host, double *P_host, double *conc_host,
                         const_char order);

cdef const_char_pointer filename = 'data.bin'
cdef const_char C_ord = 'C'
cdef const_char F_ord = 'F'

@cython.boundscheck(False)
@cython.wraparound(False)
def read_ics(np.uint_t NUM,
            np.ndarray[np.float64_t] T,
            np.ndarray[np.float64_t] P,
            np.ndarray[np.float64_t] conc,
            np.bool C_order):
    read_initial_conditions(filename, &T[0], &P[0], &conc[0], C_ord if C_order else F_ord)
    return None

def __dealloc__(self):
    finalize()