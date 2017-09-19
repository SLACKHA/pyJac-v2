import cython
import numpy as np
cimport numpy as np

cdef int compiled = 0

IF TYPE=='build_type.species_rates':
    cdef extern from "species_rates_kernel_main.oclh":
        void species_rates_kernel_call(np.uint_t problem_size, np.uint_t num_devices, double* phi, double* P, double* dphi)
        void finalize()
        void compiler()
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
        species_rates_kernel_call(problem_size, num_devices, &phi[0], &P[0], &dphi[0])
        return None
ELIF TYPE=='build_type.jacobian':
    cdef extern from "jacobian_kernel_main.oclh":
        void jacobian_kernel_call(np.uint_t problem_size, np.uint_t num_devices, double* phi, double* P, double* jac)
        void finalize()
        void compiler()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def jacobian(np.uint_t problem_size,
                 np.uint_t num_devices,
                 np.ndarray[np.float64_t] phi,
                 np.ndarray[np.float64_t] P,
                 np.ndarray[np.float64_t] jac):
        global compiled
        if not compiled:
            compiler()
            compiled = True
        jacobian_kernel_call(problem_size, num_devices, &phi[0], &P[0], &jac[0])
        return None



def __dealloc__(self):
    finalize()