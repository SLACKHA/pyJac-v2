Data Ordering
#############

pyJac can utilize C (row-major) or F (column-major) data layouts, and additionally implements a vectorized data ordering to improve caching & performance on various platforms.

.. _array_ordering:

==============
Array Ordering
==============

Consider the array of species concentrations at in index :math:`\left(j, k\right)`, :math:`[C]_{j, k}`.
The array has :math:`N_{\text{state}}` (:math:`j = 1 \ldots N_{\text{state}}`) rows corresponding to the number of thermo-chemical states being evaluated and :math:`N_{\text{sp}}` columns (:math:`k = 1 \ldots N_{\text{sp}}`) corresponding to a chemical model with :math:`N_{\text{sp}}` species.

This row / column layout is the one applied for the row-major / column-major layout.


=======================================================
Data Ordering for use with Python or Executable Library
=======================================================

When calling pyJac from either the generated Python wrapper or executable library (See :ref:`interface_vs_exe`), the state vectors / Jacobians should be interpreted using standard "C" \ "F"-ordering.
For example, to apply an F-ordering for a CONV state vector in `numpy`_:

```
# create phi array
phi = np.zeros(n_state, n_spec + 1)
# populate the phi array
# index 0 is the temperature
phi[:, 0] = temperatures[:]
# index 1 is the pressure
phi[:, 1] = pressures[:]
# and indicies 2...n_spec are the moles of the species in the model (excluding the last species)
phi[:, 2:] = moles[:, :-1]
# and finally, convert to F-order
phi = np.copy(phi, order='F')
```

.. _numpy: http://numpy.org
.. _interface_vs_exe: `Difference between Interface and Executable Libraries`

.. _vector_split:

=================================================================
Data Ordering for Calling pyJac from Other Codes (Interface Mode)
=================================================================

When calling pyJac's generated source-term \ Jacobian codes directly from another code (see :ref:`interface_vs_exe`), the supplied data must be in pyJac's own internal data-format.

As described in the pyJac-v2 paper (:ref:`paper`), pyJac uses a vectorized data-ordering for some cases.  Here we will define some terms to improve clarity:

*  The **split_axis** is the axis in the array (**Note: before the split is applied**) that will be split into two new axes, a vector axis and another axis (which may or may not be important).
*  The **vector_axis** results from the splitting, and is of length :ref:`vector-width`.
*  The **grow_axis** is the array axis that will grow with increasing numbers of initial conditions.

For example purposes, we will consider in this section an array of species concentrations (see :cref:`array_ordering`) for 1000 thermo-chemical states for a model with 20 chemical species and a vector width of 8.
The array's shape before splitting is: :math:`\left(1000, 20\right)`.

There are currently three situations where pyJac will automatically utilize a vectorized data ordering:

1)  A shallow-vectorized, C-ordered code is generated.  In this case:

* The **split_axis** corresponds to the initial conditions axis (i.e., zero, in zero-based indexing).
* After the split, the array will be converted to shape :math:`\left(125, 20, 8\right)`,
* the **vector_axis** will be the last axis in the array (after the species axis in :ref:`array_ordering`, and of length :ref:`vector-width`,
* and the **grow_axis** will be axis zero, and will be size `np.ceil(n_state / vector_width)`.

This corresponds to ordering:

.. math::
	[C]_{0, 0}, \ldots [C]_{vw, 0}, [C]_{0, 1}, \ldots [C]_{vw, 1} \ldots, \text{etc.}

for a vector width ":math:`vw`".  This data-layout orders the concentrations for a given species :math:`k` for :math:`vw` thermo-chemical states sequentially in memory, followed by the concentrations of species :math:`k + 1` for the same states.  This is important so that SIMD-instructions do not need to perform expensive gather / scatter operations.

2)  A deep-vectorized, F-ordered code is generated.  This is similar to case #1.

* The **split_axis** corresponds to the last axis in the array, i.e., the species axis (axis one).
* After the split, the array will be converted to shape :math:`\left(8, 1000, 3\right)`,
* the **vector_axis** is the first axis in the array (axis 0), and of length :math:`vw`,
* and the **grow_axis** is the initial-conditions axis (the size of which is unchanged in this case), i.e., axis one.

This corresponds to ordering:

.. math::
	[C]_{0, 0}, \ldots [C]_{0, vw}, [C]_{1, 0}, \ldots [C]_{1, vw} \ldots, \text{etc.}

for a vector width ":math:`vw`".  This data-layout orders the concentrations for a given thermo-chemical state :math:`j` for :math:`vw` species sequentially in memory, followed by the thermo-chemical state :math:`j + 1` for the same species.  This is important to ensure coalesced memory accesses on the GPU.


3)  Explicit-SIMD (:ref:`simd`) is used, and neither of the previous two cases apply.  In this case, the axes of the array may be padded (but not re-ordered) to ensure that the array can properly be vectorized.  For example, again using the species concentration array from :ref:`array_ordering`,  let us consider an array of 20 species, for 1000 thermo-chemical states, and a vector-width of 8.  If a "C"-ordering is used:

* The **split_axis** will be the species axis in unsplit array (axis one).
* After the split, the array will be resized to shape :math:`\left(1000, 3, 8\right)` such that the species axis can be properly vectorized, and:
* The **vector_axis** is the last axis of the array of the split array (axis two).
* The **grow_axis** is axis zero.

Conversely, if a "F"-ordering is used:

* The **split_axis** will be the initial condition axis in unsplit array (axis zero).
* After the split, the array will be resized to shape :math:`\left(8, 125, 20\right)` such that the initial condition axis is properly vectorized, and:
* The **vector_axis** is the first axis of the array of the split array (axis zero).
* The **grow_axis** is axis one.

.. _paper: `dummy`
.. _simd: `dummy2`
.. _vecwidth: `vector-width`
