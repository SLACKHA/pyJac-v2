Driver Functions
################

There are two major types of driver functions for pyJac, queue-based methods and
lock-step methods. The correct choice of driver function depends on the target language
and device, and further affects things such as memory usage, and data ordering.

This section will discuss the various driver function types and their rammifications.

.. _driver-function:

=======================
Why implement a driver?
=======================

When using pyJac on it's own, we potentially want to evaluate the chemical kinetic
source rates or Jacobian for many different initial conditions concurrently. To
achieve this, we need to loop over the initial conditions e.g., as in the following
psuedo-code:

```
    for i in initial_conditions:
        jacobian[i] = eval_jacobian(phi[i])
    end
```

where `jacobian` and `phi` represent the chemical kinetic Jacobian and
thermochemical state vector (a.k.a., $\Phi$).

However, when coupling pyJac to an external code (e.g., to an ODE integration library)
these arrays may already be created, for instance in CFD codes, one often sees something
like:

```
    for icell in cells:
        double phi_local[n_spec + 1] = {0};
        // set temperature
        phi_local[0] = phi[icell, 0]
        for ispecies in species:
            // set species
            phi_local[ispecies + 1] = phi[icell, ispecies + 1]
        end

        solve_ode(phi_local, pressure[icell])
    end
```

Here the calling code has implicitly assumed that the ODE integrator operates on local
copies of the global state arrays `phi` and `pressure`.  Hence, pyJac must support
this sort of memory format.

In addition, reacting-flow codes may use different state-variables, e.g., mass-fractions,
mole-fractions, concentrations, etc.!  The driver function provides a natural place to
enable conversion to/from the calling code's state variables to pyJac's state-vector
(see :ref:`state-vector`).

.. _work-size:

========================
Specifying the Work-Size
========================

In pyJac, the work-size is defined as the total number of separate (potentially
vectorized) evaluations of the chemical kinetic properties / source rates / Jacobian
happening concurrently.  This is determined automatically per-language via:

|Language |OpenMP           |OpenCL          |
|:-------:|-----------------|----------------|
|Work-Size|omp_num_threads()|get_num_groups()|

Alternatively, a more intuitive meaning for various devices is a follows:

|Device   |CPU                 |GPU        |
|:-------:|--------------------|-----------|
|Work-Size|# of cores / threads|# of blocks|

Where a 'thread block' for a GPU is defined in the CUDA sense.

.. note::
    While the work-size may be specified at run-time, if it is specified during the
    generation process via the :ref:`work_size_flag`, more optimized code will be
    generated.

For vectorized codes, the work-size is not exactly equal to the number of
thermochemical states that are being evaluated concurrently.  For example, if
a single CPU core is being utilized, but a :ref:`vector-width` of 4 is specified, the
work-size will still be equal to one.

.. _working-buffer:

===================
Memory Requirements
===================

The memory allocated by pyJac is based on a few factors:

1.  The :ref:`work-size` specified during generation or at run-time.

For CPU and Accelerator devices, this tends to be in the 10s of threads.
On a GPU however, typically 100s to 1000s of threads are required to saturate the
throughput of the device.

In pyJac, a non-input/output array of size (per initial-condition) of `N_s`
(e.g., the concentrations) is typically shaped:
```
    concentrations.shape = (work-size, N_s)
```
such that all threads have their own working copy of the `concentrations` array to
work with.

On a GPU, the `work-size` is calculated (in CUDA terminology) as the number of blocks
launched multipled by the size of each block (i.e., `gridDim * blockDim`).  Or in OpenCL
terminology, the output of `get_global_size()`.
On the CPU and MIC however, the `global_size` can typically be set the number of CPU
cores (or threads) the user wishes to use, and the allocated memory size can be
significantly reduced.

For vectorized execution, the shape of the arrays changes slightly to
(note: assuming a wide-vectorized "C"-ordering :see:`vector_split`):
```
    concentrations.shape = (work-size, N_s, vector_width)
```
where the `vector_width` is typically 2--8 for CPUs and MICs, and 64--1024 for GPUs
(note: this corresponds to the block-size in CUDA).


=====================
Lockstep-based driver
=====================

This type of driver is very similar to static-based scheduling in OpenMP (
see `_mp_scheduling`_). Essentially all threads recieve their assigned initial
conditons at startup, and evaluate the Jacobian or source terms for them.

This doesn't have any scheduling overhead, but if different threads take different
amounts of time to complete (e.g., as in ODE integration of different initial
conditions), the work may become unbalanced, and some threads may wait for the others
to complete.


==================
Queue-based driver
==================

This type of driver is based on dynamic scheduling in OpenMP (see `_mp_scheduling`_).
Unlike in the lockstep-driver, threads in the queue-based driver recieve their
assigned initial conditions at runtime.
Specifically, each thread will perform an atomic integer addition on a global counter
to determine the next initial condition to evaluate.
This reduces the effects of varying runtimes between different initial conditions, but
incurs some-overhead due to the atomic counter update.

Queue-based drivers are not-available for target languages / platforms that do not
implement atomic operations for integer types.

.. _mp_scheduling: http://cs.umw.edu/~finlayson/class/fall14/cpsc425/notes/12-scheduling.html
