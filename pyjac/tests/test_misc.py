"""
Tests various functions in the utils function or parts of the test apparatus
themselves
"""

# standard library
from collections import OrderedDict

# package includes
import numpy as np
from parameterized import parameterized, param
from unittest.case import SkipTest
from optionloop import OptionLoop
try:
    from scipy.sparse import csr_matrix, csc_matrix, coo_matrix
except ImportError:
    csr_matrix = None
    csc_matrix = None

# local includes
from pyjac.loopy_utils.loopy_utils import kernel_call
from pyjac.core.array_creator import array_splitter
from pyjac.utils import enum_to_string, listify
from pyjac.core.enum_types import (kernel_type, JacobianFormat, JacobianType)
from pyjac.tests.test_utils import get_comparable, skipif, dense_to_sparse_indicies,\
    select_elements, get_split_elements, sparsify
from pyjac.tests import set_seed

set_seed()


@parameterized([(JacobianType.exact, 'exact'),
                (JacobianType.approximate, 'approximate'),
                (JacobianType.finite_difference, 'finite_difference'),
                (JacobianFormat.sparse, 'sparse'),
                (JacobianFormat.full, 'full'),
                (kernel_type.chem_utils, 'chem_utils'),
                (kernel_type.species_rates, 'species_rates'),
                (kernel_type.jacobian, 'jacobian')])
def test_enum_to_string(enum, string):
    assert enum_to_string(enum) == string


@parameterized([('a', ['a']),
                ([1, 2, 3], [1, 2, 3]),
                ((1, 2, 'a'), [1, 2, 'a']),
                (3, [3])])
def test_listify(value, expected):
    assert listify(value) == expected


@parameterized([param(
    (1024, 4, 4), lambda y, z: y + z <= 4, [np.arange(4), np.arange(4)], (1, 2)),
                param(
    (1024, 6, 6), lambda x, y: (x + y) % 3 != 0, [np.arange(3), np.arange(6)],
                    (1, 2)), param(
    (1024, 10, 10), lambda x, y: x == 0, [np.array([0], np.int32), np.arange(6)],
                    (1, 2)), param(
    (1024, 10, 10), lambda x, y: (x & y) != 0, [np.arange(4, 10), np.arange(6)],
                    (1, 2), tiling=False)
    ])
@skipif(csr_matrix is None, 'scipy missing')
def test_dense_to_sparse_indicies(shape, sparse, mask, axes, tiling=True):
    for order in ['C', 'F']:
        # create matrix
        arr = np.arange(1, np.prod(shape) + 1).reshape(shape, order=order)

        def __slicer(x, y):
            slicer = [slice(None)] * arr.ndim
            slicer[1:] = x, y
            return slicer

        def apply_sparse(x, y):
            arr[__slicer(*np.where(~sparse(x, y)))] = 0

        # sparsify
        np.fromfunction(apply_sparse, arr.shape[1:], dtype=np.int32)
        matrix = csr_matrix if order == 'C' else csc_matrix
        matrix = matrix(arr[0])

        # next, create a sparse copy of the matrix
        sparse_arr = np.zeros((arr.shape[0], matrix.nnz), dtype=arr.dtype)
        it = np.nditer(np.empty(shape[1:]), flags=['multi_index'], order=order)
        i = 0
        while not it.finished:
            if not sparse(*it.multi_index):
                it.iternext()
                continue

            sparse_arr[:, i] = arr[__slicer(*it.multi_index)]
            it.iternext()
            i += 1

        # get the sparse indicies
        row, col = (matrix.indptr, matrix.indices) if order == 'C' \
            else (matrix.indices, matrix.indptr)
        sparse_axes, sparse_inds = dense_to_sparse_indicies(
            mask, axes, col, row, order, tiling=tiling)
        sparse_inds = sparse_inds[-1]

        # and check
        it = np.nditer(np.empty(shape[1:]), flags=['multi_index'], order=order)
        i = 0
        while not it.finished:
            if not sparse(*it.multi_index):
                it.iternext()
                continue

            if not tiling:
                if not (it.multi_index[0] in mask[-2] and
                        it.multi_index[1] == mask[-1][np.where(
                            it.multi_index[0] == mask[-2])]):
                    it.iternext()
                    continue

            if not (it.multi_index[0] in mask[-2] and it.multi_index[1] in mask[-1]):
                it.iternext()
                continue

            # check that the sparse indicies match what we expect
            assert np.all(sparse_arr[:, sparse_inds[i]] == arr[__slicer(
                          *it.multi_index)])
            it.iternext()
            i += 1


@parameterized([param(
    (1024, 4, 4), [np.arange(4), np.arange(4)], (1, 2)),
                param(
    (1024, 6, 6), [np.arange(3), np.arange(6)], (1, 2)), param(
    (1024, 10, 10), [np.array([0], np.int32), np.arange(6)], (1, 2)), param(
    (1024, 10, 10), [np.arange(4, 10), np.arange(6)], (1, 2), tiling=False)
    ])
def test_select_elements(shape, mask, axes, tiling=True):
    # create array
    arr = np.arange(1, np.prod(shape) + 1).reshape(shape)

    dummy_opts = type('', (object,), {
        'depth': None,
        'width': None,
        'order': 'C',
        'is_simd': False})
    asplit = array_splitter(dummy_opts)

    assert np.array_equal(
        select_elements(arr, mask, axes, tiling=tiling).flatten(order='C'),
        # despite the name, this can actually be used for both split & non-split
        # elements and forms a nice test-case answer here
        get_split_elements(arr, asplit, arr.shape, mask, axes, tiling=tiling))


# dummy option loop
def opts_loop(langs=['opencl'],
              width=[4, None],
              depth=[4, None],
              order=['C', 'F'],
              simd=True,
              sparse=False):

    oploop = OptionLoop(OrderedDict(
        [('lang', langs),
         ('width', width),
         ('depth', depth),
         ('order', order),
         ('device_type', 'CPU'),
         ('is_simd', [True, False] if simd else [False]),
         ('jac_format', [JacobianFormat.sparse, JacobianFormat.full] if sparse else
                        [JacobianFormat.full])]))
    for state in oploop:
        if state['depth'] and state['width']:
            continue
        if state['is_simd'] and not (state['depth'] or state['width']):
            state['is_simd'] = False
        yield type('', (object,), state)


def compare_patterns(shape):
    """
    A generator that yields the different comparison patterns that pyJac utilizes
    to test Jacobian / chemical-rate correctness

    Parameters
    ----------
    shape: tuple of int
        The shape of the array to compare

    Yields
    ------
    compare_mask: list of list of int
        The indicies to compare
    compare_axis: list or tuple of int
        The axes to compare, each entry in this variable gives the axis that the
        corresponding entry in `compare_mask` refers to
    tiled: bool [True]
        If True, the mask should be considered in "tiled" form, i.e., each
        combination of mask indicies should be considered, e.g., for a 3-D array

            mask = [(0, 1, 2), (1, 2)], ax = (1, 2)
                -> (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)
                for axis = (1, 2), respectively

        If False, the mask is a list of indicies, e.g.:
            mask = (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)
            axis = (1, 2)

        This form is used occaisionally for Jacobian testing to select individual
        array elements in non-simple patterns
    """

    ndim = len(shape)
    size = shape[-1]
    if ndim < 3:
        # rate subs
        last_axis = ndim - 1
        single_choice = np.expand_dims(
            np.random.choice(shape[-1], 1, replace=False), -1)
        # pattern #1 - a single array entry over all ICs
        yield ([single_choice], (last_axis,), True)

        # pattern #2 - selected elements of the array over all ICs
        choice = np.sort(np.random.choice(shape[-1], size - 3, replace=False))
        yield ([choice], (last_axis,), True)

        # pattern #3 - selected IC's, one array entry
        yield ([choice, single_choice], (0, last_axis), True)

        # pattern #4 - selected IC's multiple array entries
        yield ([choice, choice], (0, last_axis), True)
    else:
        row_ax = ndim - 2
        col_ax = ndim - 1
        # Jacobian
        single_choice = np.random.choice(shape[-1], 1, replace=False)
        choice = np.sort(np.random.choice(shape[-1], size - 3, replace=False))
        choice2 = np.sort(np.random.choice(shape[-1], size - 3, replace=False))
        choice3 = np.sort(np.random.choice(shape[-1], size - 3, replace=False))
        # pattern #1 - a single row. with selected column entries over all ICs
        yield ([single_choice, choice], (row_ax, col_ax), True)

        # pattern #2 - selected elements of row and column entires
        yield ([choice, choice2], (row_ax, col_ax), True)

        # pattern #3 - selected IC's, one row, with selected columns
        yield ([choice, single_choice, choice2], (0, row_ax, col_ax), True)

        # pattern #4 - selected IC's, selected rows, with one columns
        yield ([choice, choice2, single_choice], (0, row_ax, col_ax), True)

        # pattern #4 - all IC's, multiple array entries (tiling mode does not
        # allow for selection of ICs)
        yield ([choice2, choice3], (row_ax, col_ax), False)


class dummy_init(object):
    def __init__(self, arr):
        self.initializer = arr


@parameterized([(2, False), (3, False), (3, True)])
def test_get_comparable_nosplit(ndim, sparse):
    axis_size = 10
    # create array
    arr = np.arange(axis_size**ndim)
    arr = arr.reshape((axis_size,) * ndim)

    if sparse:
        # set some array elements to zero to sparsify it
        choice = np.sort(np.random.choice(axis_size, 3, replace=False))
        choice1 = np.sort(np.random.choice(axis_size, 3, replace=False))
        for x1 in choice:
            for x2 in choice1:
                arr[:, x1, x2] = 0

    # create comparable object
    for i1, (masks, axes, tiling) in enumerate(compare_patterns(arr.shape)):
        comparable = get_comparable([masks], [arr], compare_axis=axes,
                                    tiling=tiling)

        namestore = None
        for i2, opts in enumerate(opts_loop(sparse=sparse)):
            kc = kernel_call('', arr, axes, masks)
            outv = arr.copy()
            if sparse and opts.jac_format == JacobianFormat.sparse:
                if csc_matrix is None:
                    raise SkipTest('Scipy required for sparse Jacobian testing')
                # get the appropriate matrix type
                matrix = csr_matrix if opts.order == 'C' else csc_matrix
                # get the sparse indicies
                matrix = matrix(arr[0, :, :])
                row, col = (matrix.indptr, matrix.indices) if opts.order == 'C' \
                    else (matrix.indices, matrix.indptr)
                # and get the sparse indicies in flat form
                matrix = coo_matrix(arr[0, :, :])
                flat_row, flat_col = matrix.row, matrix.col

                kc.input_args = {}
                kc.input_args['jac'] = arr.copy()
                namestore = type('', (object,), {
                    'jac_row_inds': dummy_init(row),
                    'jac_col_inds': dummy_init(col),
                    'flat_jac_row_inds': dummy_init(flat_row),
                    'flat_jac_col_inds': dummy_init(flat_col)})

                # and finally, sparsify array
                outv = sparsify(outv, col, row, opts.order)

            asplit = array_splitter(opts)
            kc.set_state(asplit, order=opts.order, namestore=namestore,
                         jac_format=opts.jac_format)

            outv = asplit.split_numpy_arrays(outv.copy())[0]
            outv = comparable(kc, outv, 0, False)
            ansv = comparable(kc, kc.transformed_ref_ans[0].copy(), 0, True)

            assert np.array_equal(outv, ansv)
