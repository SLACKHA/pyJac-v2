from __future__ import division

import os
from contextlib import contextmanager
from string import Template
from collections import OrderedDict
import shutil
import logging
import subprocess
import sys
from functools import wraps
import collections
import tempfile

import psutil
from nose import SkipTest

from pyjac.loopy_utils.loopy_utils import (
    get_device_list, kernel_call, populate, auto_run, loopy_options)
from pyjac.core.enum_types import RateSpecialization, JacobianType, JacobianFormat
from pyjac.core.exceptions import MissingPlatformError, BrokenPlatformError
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.core import array_creator as arc
from pyjac.core.mech_auxiliary import write_aux
from pyjac.pywrap import pywrap
from pyjac import utils
from pyjac.utils import platform_is_gpu, clean_dir
from pyjac.core.enum_types import KernelType
from pyjac.tests import _get_test_input, get_test_langs
from pyjac.tests.test_utils.get_test_matrix import load_platforms
try:
    from scipy.sparse import csr_matrix, csc_matrix
except ImportError:
    csr_matrix = None
    csc_matrix = None


from optionloop import OptionLoop
import numpy as np
try:
    # compatability for older numpy
    np_divmod = np.divmod
except AttributeError:
    def np_divmod(a, b, **kwargs):
        div, mod = divmod(a, b)
        return np.asarray(div, **kwargs), np.asarray(mod, **kwargs)

import six


def __get_template(fname):
    with open(fname, 'r') as file:
        return Template(file.read())


script_dir = os.path.dirname(os.path.abspath(__file__))


def get_run_source():
    return __get_template(os.path.join(script_dir, 'test_run.py.in'))


def get_import_source():
    return __get_template(os.path.join(script_dir, 'test_import.py.in'))


def get_read_ics_source():
    return __get_template(os.path.join(script_dir, 'read_ic_setup.py.in'))


@contextmanager
def temporary_build_dirs(cleanup=True):
    """
    Returns a self-cleaning set of directories that may be used as build / object /
    library dirs.

    Returns
    -------
    dir_tuple: (str, str, str)
        The build_dir, obj_dir, and lib_dir respectively
    """
    dirpath = tempfile.mkdtemp()
    # make build, obj, lib
    build = os.path.join(dirpath, 'build')
    obj = os.path.join(dirpath, 'obj')
    lib = os.path.join(dirpath, 'lib')

    utils.create_dir(build)
    utils.create_dir(obj)
    utils.create_dir(lib)
    owd = os.getcwd()
    try:
        os.chdir(dirpath)
        yield build, obj, lib
    finally:
        os.chdir(owd)
        if cleanup:
            clean_dir(dirpath, remove_dir=True)


class kernel_runner(object):
    """
    Simple wrapper that runs one of our kernels to find values (e.g. kf_fall,
    or X_sri)

    Parameters
    ----------
    func : Callable
        The function to use that generates the :class:`knl_info` to run
    args : dict of :class:`numpy.ndarray`
        The arguements to pass to the kernel
    kwargs : dict
        Any other arguments to pass to the func

    Returns
    -------
    vals : list of :class:`numpy.ndarray`
        The values computed by the kernel
    """

    def __init__(self, func, test_size, args={}, kwargs={}):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.test_size = test_size
        self.__name__ = self.func.__name__ + '_runner'

    def __call__(self, loopy_opts, namestore, test_size):
        device = get_device_list()[0]

        infos = self.func(loopy_opts, namestore, test_size=test_size,
                          **self.kwargs)

        try:
            iter(infos)
        except TypeError:
            infos = [infos]

        # create a dummy generator
        gen = k_gen.make_kernel_generator(
            kernel_type=KernelType.dummy,
            name='dummy',
            loopy_opts=loopy_opts,
            kernels=infos,
            namestore=namestore,
            test_size=self.test_size
        )
        gen._make_kernels()
        # setup kernel call and output names
        kc = []
        out_arg_names = []
        for k in gen.kernels:
            written = [x for x in k.get_written_variables()
                       if x not in k.temporary_variables]
            kc.append(
                kernel_call('dummy', [None],
                            out_mask=list(range(len(written))),
                            **self.args))
            out_arg_names.append([
                arg.name for arg in k.args if arg.name in k.get_written_variables()])
            kc[-1].set_state(gen.array_split, loopy_opts.order)

        output = populate(gen.kernels, kc, device=device)
        # turn into dicts
        output = [{oa_name[i]: output[ind][i] for i in range(len(oa_name))}
                  for ind, oa_name in enumerate(out_arg_names)]
        # and collapse into single dict if single kernel
        if len(output) == 1:
            output = output[0]
        return output


class indexer(object):
    """
    Utility class that helps in transformation of old indicies to split array
    indicies
    """

    array_types = (np.ndarray, tuple, list)

    @property
    def vec_width(self):
        """
        Returns the vector-width of the split
        """
        return self.splitter.vector_width

    def offset(self, axis):
        """
        Returns the offset to add to axes as a result of the split
        """

        return axis + (1 if axis >= self.vec_axis else 0)

    def _get_index(self, inds, axes):
        """
        Converts the indicies (:param:`inds`) for the given :param:`axes` to
        their split indicies.

        Parameters
        ----------
        inds: list of :class:`np.ndarray` or list of lists
            The integer indicies to convert, each entry in the list should correspond
            to an entry at the same index in the :param:`axes`
        axes: list of ints or :class:`np.ndarray`
            The axes for each index entry in :param:`inds`

        Returns
        -------
        split: tuple of :class:`np.ndarray`
            The  split indicies that correspond to the passed indicies for the
            split array
        """

        rv = [slice(None)] * self.out_ndim
        axi = next((i for i in six.moves.range(len(axes))
                    if axes[i] == self.split_axis), None)
        if axi is not None:
            # and first index is the floor division of the new dim size
            # the last index is the remainder of the ind by the new dimension size

            # check that this is ind is not a slice
            # if it is we don't need to to anything
            if isinstance(inds[axi], indexer.array_types):
                # it's a numpy array, so we can divmod
                rv[self.offset(self.split_axis)], rv[self.vec_axis] = np_divmod(
                        inds[axi], self.vec_width, dtype=arc.kint_type)

        for i, ax in enumerate(axes):
            if i != axi:
                # there is no change in the actual indexing here

                # check that this is ind is not a slice
                # if it is we don't need to to anything
                if isinstance(inds[i], indexer.array_types):
                    rv[self.offset(ax)] = np.array(inds[i][:]).astype(arc.kint_type)

        return tuple(rv)

    def __init__(self, splitter, ref_shape):
        """
        Creates the :class:`indexer`

        Parameters
        ----------
        splitter: :class:`array_splitter`
            The array splitter that was used on the array for which the split
            indicies are desired
        ref_shape: tuple of int
            The shape of the unsplit array
        """
        self.splitter = splitter
        self.ref_shape = ref_shape

        # calculate split
        split_shape, _, vec_axis, split_axis = self.splitter.split_shape(
            type('', (object,), {'shape': ref_shape}))

        self.ref_ndim = len(ref_shape)
        self.out_ndim = len(split_shape)
        self.split_axis = split_axis
        self.vec_axis = vec_axis

        if vec_axis is None:
            # no split at all
            self._indexer = lambda x, y: x
        else:
            # split
            self._indexer = self._get_index

    def __call__(self, inds, axes):
        """
        Returns the split array indicies for the given :attr:`ref_shape` and
        :attr:`splitter`

        Parameters
        ----------
        inds: list of list of int or list of :class:`numpy.ndarray`
            The indicies in the unsplit array that we want to inspect in the split
            array
        axes: list of list of int or list of :class:`numpy.ndarray`
            The axes in the unsplit array to which each entry in :param:`inds`
            corresponds to

        Notes
        -----
        This class is the "dumb" version of :func:`get_split_elements`, and shouldn't
        be used directly if you don't understand the differences between the two.
        See :func:`get_split_elements` for a complete description of said
        differnences

        Returns
        -------
        mask: tuple of int / slice
            A proper indexing for the split array
        """
        return self._indexer(inds, axes)


class multi_index_iter(object):
    """
    This is a convenience class to provide a buffered access to a multi-dimensional
    iteration (with specifiable iteration-order), while tracking a multi_index.

    This is built to get around the limitations / slow speed of :class:`numpy.nditer`
    which _can_ track the multi_index and control the iteration order, but _cannot_
    do so in a buffered manner (which is crucial for efficient iteration)

    Examples


    Attributes
    ----------
    mask: list of :class:`numpy.ndarray`
        The indicies (per-axis) to iterate over
    order: ['C', 'F']
        The iteration order
    size_limit: int [1e8]
        The number of bytes than can be stored by this :class:`multi_index_iter`
        to avoid huge memory usage while maintaining efficiency.  Default is
        1Gb.  This can be overriden by the test input :ref:`WORKING_MEM`.
    """

    def __init__(self, mask, order, size_limit=_get_test_input(
                    'working_mem', 2.5e8)):
        self.mask = mask[:]
        self.shape = [x.size for x in mask]
        self.total_size = np.prod([x.size for x in mask])
        utils.check_order(order)
        self.order = order
        self._use_memmap = False

        # and determine iteration order
        self._iterorder = list(range(len(self.shape)))
        if self.order == 'C':
            self._iterorder = list(reversed(self._iterorder))

        # and initialize counts / strides
        self._counts = np.zeros(len(self.shape), dtype=arc.kint_type)
        self._cumstrides = np.zeros(len(self.shape), dtype=arc.kint_type)
        self._strides = np.zeros(len(self.shape), dtype=arc.kint_type)

        # first pass, figure out minimum necessary memory
        accum = 1
        for ax in self._iterorder:
            _per_axis = size_limit // len(self.shape)
            stride = _per_axis // (accum * self.shape[ax])
            if not stride and ax != 0:
                # can't fit the non-IC axes in memory... bad sign, but we
                # just have to bump the size limit such that we can, for
                # sanity
                logger = logging.getLogger(__name__)
                size_limit = accum * self.shape[ax] * len(self.shape)
                logger.warn('Multi-indicies for array of shape ({}) cannot '
                            'fit in given size_limit for (non initial-condition)'
                            ' axis {}, with size {}. The size-limit must be '
                            'increased to {}.'.format(
                                utils.stringify_args(self.shape), ax,
                                self.shape[ax], size_limit))
            accum *= self.shape[ax]

        # second pass, set strides & cumulative strides
        accum = 1
        self.size_limit = int(size_limit)
        _per_axis = self.size_limit // len(self.shape)
        for ax in self._iterorder:
            if ax != 0:
                stride = _per_axis // (accum * self.shape[ax])
            else:
                # figure out the number of times we can fit the array
                stride = _per_axis // accum

            assert stride

            self._strides[ax] = np.minimum(stride, self.shape[ax])
            self._cumstrides[ax] = accum
            accum *= self.shape[ax]

        # create index array holder
        self._indicies = np.empty((len(self.shape), self.per_iter),
                                  dtype=arc.kint_type)
        self.finished = False

        assert self.per_iter <= self.size_limit

        logger = logging.getLogger(__name__)
        logger.debug('multi_index_iter created with {} iterations of {} elements '
                     'each'.format(self.num_iters, self.per_iter))

    @property
    def per_iter(self):
        """
        The number of elements processed per-iteration
        """
        last = self._iterorder[-1]
        return self._cumstrides[last] * self._strides[last]

    @property
    def num_iters(self):
        last = self._iterorder[-1]
        return int(np.ceil(self.shape[last] / self._strides[last]))

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next set of multi_indicies to be processed

        Returns
        -------
        indicies: tuple of (:class:`numpy.ndarray`,)
            The multi indicies to process for this iteration
        """

        if self.finished:
            raise StopIteration

        # for the current set of counts, go through the iterorder and populate
        # the _indicies arrays
        for i, ax in enumerate(self._iterorder):
            # create arange bounds
            start = self._counts[ax]
            end = start + self._strides[ax]

            # figure out tiling
            repeats = int(np.ceil(self.total_size / (
                (end - start) * self._cumstrides[ax])))
            inds = np.tile(self.mask[ax][start:end],
                           (self._cumstrides[ax], repeats)).flatten('F')

            # update indicies
            self._indicies[ax, :self.per_iter] = inds[:self.per_iter]

            # update counts
            self._counts[ax] = end

        # check for end of iteration
        if np.all(self._counts >= self.shape):
            self.finished = True

        # and update counts
        for ax in self._iterorder:
            self._counts[ax] %= self.shape[ax]

        return tuple(self._indicies[i, :self.per_iter] for i in range(len(
            self.shape)))

    def next(self):
        return self.__next__()


def get_split_elements(arr, splitter, ref_shape, mask, axes=(1,),
                       tiling=True):
    """
    A helper method to get the elements in a split array for a given
    set of indicies and axis specified for the reference (unsplit) array.

    .. _note-above:

    Note
    ----

    This method is somewhat similar to :class:`indexer` but differs in a few key
    respects:

    1) First, this method returns the desired elements of the split array (instead
       of indicies).

    2) Second, this method properly tiles / repeats the indicies for splitting, e.g.,
       let's say we want to look at the slice 1:3 for axes 0 & 1 for the unsplit
       array "array":

            array[1:3, 1:3]

       Passing a mask of [[1,2,3], [1,2,3]], and axes of (0, 1) to the
       :class:`indexer` would result in just six split indicies returned i.e., for
       indicies (1,1), (2,2), and (3,3).

       :func:`get_split_elements` will tile these indicies such that the elements
       for each combination of the inputs will be returned.  In our example
       this would correspond to nine elements, one each for each of (1,1)
       (1,2), (1,3) ... (3,3).


    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        The split array to compute the split indicies for
    mask: :class:`numpy.ndarray` or list thereof
        The indicies to determine
    ref_ndim: int [2]
        The dimension of the unsplit array
    axes: list of int
        The axes the mask's correspond to. Must be of the same shape / size as mask.
    tiling: bool [True]
        If False, turns off the tiling discussed in the
        :ref:`note-above <note above>`_.  In this mode, each entry of the mask must
        be the same length as all other entries

    Returns
    -------
    sliced: :class:`numpy.ndarray`
        A flattened array of length NxM containing the desired elements of the
        split array, where N is the number of initial conditions of
        :param:`ref_shape` (i.e., ref_shape[0]), and M the number of elements
        obtained by the combined mask arrays (see the above note)
    """

    _get_index = indexer(splitter, ref_shape)

    if not tiling:
        if not len(mask) == len(ref_shape):
            # copy to slice form
            index = [slice(None) for x in six.moves.range(arr.ndim)]
            for i, ax in enumerate(axes):
                index[ax] = mask[i]
            mask = tuple(index)
        # check mask
        # get a size
        size = next(m.size for m in mask if isinstance(m, np.ndarray))
        assert all(m.size == size for m in mask if isinstance(m, np.ndarray)), (
            "Supplied mask elements cannot have differing lengths in non-tiling "
            "mode.")
        return arr[_get_index(mask, axes)].flatten(splitter.data_order)

    # create outputs
    output = None

    # ensure the mask / axes are in the form we expect
    mask = utils.listify(mask)
    axes = utils.listify(axes)
    assert len(axes) == len(mask), "Supplied mask doesn't match given axis/axes"

    # fill in any missing mask / axes
    full_mask = []
    full_axes = np.arange(len(ref_shape))
    for i in range(len(ref_shape)):
        if i in axes:
            full_mask.append(mask[axes.index(i)])
        else:
            full_mask.append(np.arange(ref_shape[i]))

    # create multi index iterator & output
    mii = multi_index_iter(full_mask, splitter.data_order)
    output = np.empty((mii.num_iters, mii.per_iter), dtype=arr.dtype, order='C')
    for i, multi_index in enumerate(mii):
        # convert multi indicies to split indices
        inds = _get_index(multi_index, full_axes)
        # and store
        output[i, :] = arr[inds]

    return output.flatten('C')[:mii.total_size]


# https://stackoverflow.com/a/41234399
def inNd(a, b):
    """
    Helper method that works like in1d, but for N-Dimensional arrays

    Paramaters
    ----------
    a: :class:`numpy.ndarray`
        A M x N array
    b: :class:`numpy.ndarray`
        A K x N array
    Returns
    -------
    a_in_b: :class:`numpy.ndarray`
        An array of Lx1 (L<=K) indicies indicating which indicies in B
        correspond to indicies in A
    """
    return np.where((a[:, None] == b).all(-1).any(0))[0]


# https://stackoverflow.com/a/25655090
def combination(*arrays, **kwargs):
    """
    Helper method that combines the given 1-D arrays in the "ordering" given
    Used to get all combinations of sparse indicies in the array

    Parameters
    ----------
    arrays: list or tuple of :class:`numpy.ndarray`
        The arrays to combine.  Note that all arrays must have the same size
    order: str ['C', 'F']
        The combination order, corresponding to row/column major combination

    Returns
    -------
    combined: :class:`numpy.ndarray`
        The combined array of shape N x :param:`arrays`[0].size
    """

    # assert np.all([np.array_equal(x.shape, arrays[0].shape) for x in arrays[1:]])

    order = kwargs.pop('order')
    if order == 'F':
        # need to have the columns incrementing slower, easier to
        # put the col mask first and then...
        arrays = list(reversed(arrays))
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    if order == 'F':
        # ...flop rows and columns
        ix = ix[:, [1, 0]]
    return ix


# helper methods
def sparse_to_dense_indices(col_inds, row_inds, order, as_inds=True):
    """
    Converts the supplied :param:`col_inds` and :param:`row_inds` to a sparse matrix
    of format :class:`scipy.coo_matrix`and returns an array (or tuple, for
    :param:`as_inds` == False) of the dense indicies that correspond to the supplied
    sparse indicies.

    Notes
    -----
    This method is primarily useful to give a "filtered" list of dense-indicies,
    that is, with identically-zero dense-indicies removed.  Additionally, this
    handles ordering concerns for flattening of dense matricies (i.e., returns the
    proper dense indicies for conversion to a :class:`scipy.csr_matrix` or
    class:`scipy.csc_matrix` depending on :param:`order`)

    Parameters
    ----------
    col_inds: :class:`numpy.ndarray`
        Either the column pointer or column index list (depending on :param:`order`)
    row_inds: :class:`numpy.ndarray`
        Either the row pointer or row index list (depending on :param:`order`)
    order: ['C', 'F']
        The data ordering
    as_inds: bool [True]
        - If True, return dense indicies as a numpy array of shape [Nx2], where N is
        the number of non-zero dense indices
        - If False, these indicies are returned as a tuple

    Returns
    -------
    dense_inds: :class:`numpy.ndarray` or tuple
        The "filtered", ordered dense indicies corresponding to the supplied sparse
        indicies.  Numpy array of shape [Nx2] if :param:`as_inds`, else tuple

    """
    # setup dummy sparse matrix
    if order == 'C':
        matrix = csr_matrix
        inds = col_inds
        indptr = row_inds
    else:
        matrix = csc_matrix
        inds = row_inds
        indptr = col_inds
    # next create and get indicies
    matrix = matrix((np.ones(inds.size), inds, indptr)).tocoo()
    row, col = matrix.row, matrix.col
    if as_inds:
        return np.asarray((row, col)).T
    return row, col


def sparsify(array, col_inds, row_inds, order):
    """
    Returns a sparse version of the dense :param:`array`.
    Useful to convert reference answers to sparse format before splitting / selecting
    elements for comparison.

    Parameters
    ----------
    array: :class:`np.ndarray`
        The dense array to convert to sparse representation
    col_inds: :class:`numpy.ndarray`
        Either the column pointer or column index list (depending on :param:`order`)
    row_inds: :class:`numpy.ndarray`
        Either the row pointer or row index list (depending on :param:`order`)
    order: ['C', 'F']
        The data ordering

    Returns
    -------
    sparse: :class:`np.ndarray`
        The converted sparse array

    See Also
    --------
    :func:`pyjac.functional_tester.test.jacobian_eval.__sparsify`
    """

    inds = sparse_to_dense_indices(col_inds, row_inds, order, as_inds=False)
    index_tuple = [slice(None) for x in six.moves.range(array.ndim)]
    index_tuple[-len(inds):] = inds
    return array[tuple(index_tuple)]


def dense_to_sparse_indicies(mask, axes, col_inds, row_inds, order, tiling=True):
    """
    Helper function to convert dense Jacobian indicies to their corresponding
    sparse indicies.

    Note
    ----
    The :param:`col_inds` and :param:`row_inds` are either a row/column pointer
    or row/column indicies depending on the :param:`order`.  A "C"-ordered matrix
    implies use of a compressed-row sparse matrix, while an "F"-ordered matrix uses
    a compressed-column sparse matrix.

    Typically, you can just pass the values stored in the :class:`kernel_call`'s
    :attr:`row_inds` and :attr:`col_inds`

    Parameters
    ----------
    mask: list of :class:`numpy.ndarray`
        The dense Jacobian indicies to convert to sparse indicies.  Each entry in
        :param:`mask` should correspond to an axis in :param:`axes`.
    axes: list or tuple of int, or -1
        The axes index of the Jacobian array that each entry of :param:`mask`
        corresponds to.
        If :param:`axes` is -1, this indicates that the :param:`mask` consists of
        row & column indicies, but don't need to be combined via :func:`combination`
    col_inds: :class:`numpy.ndarray`
        Either the column pointer or column index list (depending on :param:`order`)
    row_inds: :class:`numpy.ndarray`
        Either the row pointer or row index list (depending on :param:`order`)
    order: ['C', 'F']
        The data ordering
    tiling: bool [True]
        If False, turns off the tiling discussed in
        :ref:`note-above <get_split_elements>`_.
        In this mode, each entry of the mask must be the same length as all other
        entries.

    Returns
    -------
    sparse_axes: tuple of int
        The axes corresponding to the sparse inds
    sparse_inds: :class:`numpy.ndarray`
        The sparse indicies
    """

    # extract row & column masks
    def __row_and_col_mask():
        if not tiling:
            assert len(mask) >= 2, ('The mask must at least include row & column '
                                    'elements when not using tiling mode.')
            # check size of non slices
            submask = [x for x in mask if isinstance(x, np.ndarray)]
            assert all(submask[0].size == m.size for m in submask[1:]), (
                'All elements of the mask must have the same size in tiling mode.')
            return len(mask) - 2, mask[-2], len(mask) - 1, mask[-1]
        row_ind = next(i for i, ind in enumerate(axes)
                       if ind == 1)
        row_mask = mask[row_ind]
        col_ind = next(i for i, ind in enumerate(axes)
                       if ind == 2)
        col_mask = mask[col_ind]
        return row_ind, row_mask, col_ind, col_mask

    # need to collapse the mask
    inds = sparse_to_dense_indices(col_inds, row_inds, order)

    # next we need to find the 1D index of all the row, col pairs in
    # the mask
    row_ind, row_mask, col_ind, col_mask = __row_and_col_mask()

    # remove old inds
    mask = [mask[i] for i in range(len(mask)) if i not in [
        row_ind, col_ind]]
    axes = tuple(x for i, x in enumerate(axes) if i not in [
        row_ind, col_ind])

    # add the sparse indicies
    if tiling:
        new_mask = combination(row_mask, col_mask, order=order)
    else:
        new_mask = np.vstack((row_mask.T, col_mask.T)).T
    mask.append(inNd(new_mask, inds))
    # and the new axis
    axes = axes + (1,)

    return axes, mask


def select_elements(arr, mask, axes, tiling=True):
    """
    Selects elements in the rows/columns of the :param:`arr` that match the given
    :param:`mask.

    Notes
    -----
    This method is built for not-split arrays _only_, and is significantly simpler
    than :func:`get_split_elements`.

    As with :func:`get_split_elements`, each entry in the :param:`mask` corresponds
    to an axis given in :param:`axes`.  However, the mask here simply tells us which
    entries in an axis to select.  For example, for a 3x3 :param:`arr`, with
    :param:`mask` == [[1, 2], [1]] and :param:`axes` == (0, 1), the result will be
    arr[[1, 2], [1]]:

    .. doctest::

    >>> import numpy as np
    >>> arr = np.arange(9).reshape((3, 3))
    >>> select_elements(arr, [[1, 2], [1]], (0, 1))

    .. testoutput:
        array([4, 7])

    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        The array to select from
    mask: list of :class:`numpy.ndarray`
        The selection mask
    axes: list of int
        The integer index of the axes to select from, each entry in :param:`mask`
        should correspond to an axis in this parameter.
    tiling: bool [True]
        Whether tiling mode is turned on, see
        :ref:`note-above <get_split_elements>`_.  By default this is True.

    Returns
    -------
    selected: `class:`numpy.ndarray`
        The selected array
    """

    assert len(axes) == len(mask), 'Given mask does not match compare axes.'

    try:
        # test if list of indicies
        if not tiling:
            if len(mask) != arr.ndim:
                # copy to slice form
                index = [slice(None) for x in six.moves.range(arr.ndim)]
                for i, ax in enumerate(axes):
                    index[ax] = mask[i]
                mask = index
            return arr[tuple(mask)].squeeze()
        # next try iterable

        # multiple axes
        outv = arr
        # account for change in variable size
        ax_fac = 0
        for i, ax in enumerate(axes):
            shape = len(outv.shape)
            inds = mask[i]

            # some versions of numpy complain about implicit casts of
            # the indicies inside np.take
            try:
                inds = inds.astype('int64')
            except (AttributeError, TypeError):
                pass
            outv = np.take(outv, inds, axis=ax-ax_fac)
            if len(outv.shape) != shape:
                ax_fac += shape - len(outv.shape)
        return outv.squeeze()
    except TypeError:
        # finally, take a simple mask
        return np.take(arr, mask, axes).squeeze()


class get_comparable(object):
    """
    A wrapper for the kernel_call's _get_comparable function that fixes
    comparison for split arrays

    Attributes
    ----------
    compare_mask: list of :class:`numpy.ndarray` or list of tuples of
            :class:`numpy.ndarray`
        The default comparison mask.  If multi-dimensional, should be a list
        of tuples of :class:`numpy.ndarray`'s corresponding to the compare axis
    ref_answer: :class:`numpy.ndarray`
        The answer to compare to, used to determine the proper shape
    compare_axis: iterable of int
        The axis (or axes) to compare along
    """

    def __init__(self, compare_mask, ref_answer, compare_axis=(1,), tiling=True):
        self.compare_mask = compare_mask
        if not isinstance(compare_mask, list):
            self.compare_mask = [compare_mask]
        self.ref_answer = ref_answer
        if not isinstance(ref_answer, list):
            self.ref_answer = [ref_answer]
        self.compare_axis = utils.listify(compare_axis)
        self.tiling = tiling

        # check that all the compare masks are of the same
        # length as the compare axis
        assert all(len(x) == len(self.compare_axis) for x in self.compare_mask),\
            "Can't use dissimilar compare masks / axes"

    def __call__(self, kc, outv, index, is_answer=False):
        """
        Get a comparable version of the split/unsplit array

        Parameters
        ----------
        kc: :class:`kernel_call`
            The kernel call object that requires the comparable array for comparison.
            This contains information needed to properly compute the split indicies
        outv: :class:`numpy.ndarray`
            The output corresponding to the kernel calls
        index: int
            The index of the output in the list of outputs corresponding to the
            :class:`kernel_call`, used for determining which mask / answer should be
            used
        is_answer: bool [False]
            If True, :param:`outv` is a reference answer.  This affects how the split
            indicies are calculated for Jacobian comparison
        """
        mask = list(self.compare_mask[index][:])
        ans = self.ref_answer[index]
        try:
            axis = self.compare_axis[:]
        except TypeError:
            axis = self.compare_axis
        ndim = ans.ndim
        ref_shape = ans.shape

        # check for sparse
        if kc.jac_format == JacobianFormat.sparse:
            if csc_matrix is None and csr_matrix is None:
                raise SkipTest('Cannot test sparse matricies without scipy'
                               ' installed')
            axis, mask = dense_to_sparse_indicies(
                mask, axis, kc.col_inds, kc.row_inds, kc.current_order,
                tiling=self.tiling)
            # update the reference shape
            ref_shape = sparsify(ans, kc.col_inds,
                                 kc.row_inds, kc.current_order).shape
            # indicate the drop in dimension
            ndim -= 1

        # check for vectorized data order
        if outv.ndim == ndim:
            # return the default select
            return select_elements(outv, mask, axis, self.tiling)
        else:
            return get_split_elements(outv, kc.current_split, ref_shape, mask,
                                      axis, tiling=self.tiling)


def reduce_oploop(base, add=None):
    """
    Convenience method to turn :param:`base` into an :class:`oploopconcat`

    Parameters
    ----------
    base: list of list of tuples
        Each list inside of base should be convertible to an OrderedDict
    add: list of tuples [None]
        If specified, add this list of tuples to each internal :class:`OptionLoop`

    Returns
    -------
    concat: :class:`oploopconcat`
        The concatentated option loops
    """

    out = None
    for b in base:
        if add is not None:
            b += add
        val = OptionLoop(OrderedDict(b), lambda: False)
        if out is None:
            out = val
        else:
            out = out + val

    return out


def _get_oploop(owner, do_ratespec=False, do_ropsplit=False, do_conp=False,
                langs=get_test_langs(), do_vector=True, do_sparse=False,
                do_approximate=False, do_finite_difference=False,
                sparse_only=False, do_simd=True, **kwargs):

    platforms = load_platforms(owner.store.test_platforms, langs=langs)
    oploop = [('order', ['C', 'F']),
              ('auto_diff', [False])
              ]
    if do_ratespec:
        oploop += [
            ('rate_spec', [x for x in RateSpecialization]),
            ('rate_spec_kernels', [True, False])]
    if do_ropsplit:
        oploop += [
            ('rop_net_kernels', [True])]
    if do_conp:
        oploop += [('conp', [True, False])]
    else:
        oploop += [('conp', [True])]
    if sparse_only:
        oploop += [('jac_format', [JacobianFormat.sparse])]
    elif do_sparse:
        oploop += [('jac_format', [JacobianFormat.sparse, JacobianFormat.full])]
    else:
        oploop += [('jac_format', [JacobianFormat.full])]
    if do_approximate:
        oploop += [('jac_type', [JacobianType.exact, JacobianType.approximate])]
    elif do_finite_difference:
        oploop += [('jac_type', [JacobianType.finite_difference])]
    else:
        oploop += [('jac_type', [JacobianType.exact])]
    if do_simd:
        oploop += [('is_simd', [True, False])]

    for key in sorted(kwargs.keys()):
        # enable user to pass in additional args
        oploop += [(key, utils.listify(kwargs[key]))]

    return reduce_oploop(platforms, oploop)


def _should_skip_oploop(state, skip_test=None, skip_deep_simd=True):
    """
    A unified method to determine whether the :param:`state` is a viable
    configuration for initializing a :class:`loopy_options`

    Parameters
    ----------
    state: dict
        The state of the :class:`optionloop`
    skip_test: six.callable
        Callable functions that take as the arguement the :param:`state` and
        return True IFF the state should be skipped
    skip_deep_simd: bool [True]
        If true, skip explicit-SIMD tests w/ deep vectorizations (not currently
        implemented)

    Returns
    -------
    should_skip: bool
        True IFF the state should be skipped
    """

    if utils.can_vectorize_lang[state['lang']] and (
            state.get('width', False) and state.get('depth', False)):
        # can't vectorize deep and wide concurrently
        return True

    if (state.get('width', False) or state.get('depth', False)) \
            and not utils.can_vectorize_lang[state['lang']]:
        return True

    if not (state.get('width', False) or state.get('depth', False)) \
            and state.get('is_simd', False):
        return True

    if skip_deep_simd and state.get('depth', False) and state.get('is_simd', False):
        return True

    if state['lang'] != 'opencl' and state.get('device_type', ''):
        return True

    if state.get('rate_spec', None) == 'fixed' and state.get('split_kernels', None):
        # not a valid configuration
        return True

    if skip_test and skip_test(state):
        return True

    return False


# inspired by https://stackoverflow.com/a/42983747
class OptionLoopWrapper(object):
    """
    A simple generator for option loop iteration with built-in skip test detection

    Parameters
    ----------
    oploop_base: class:`OptionLoop`
        The base option loop to iterate over.
    skip_test: six.callable [None]
        If not none, a callable function that takes as the arguement the
        current :param:`state` of the option loop and returns True IFF the state
        should be skipped
    yield_index: bool [False]
        If true, yield a tuple of the (index, state) of the oploop enumeration
    skip_deep_simd: bool [None]
        If true, skip deep-SIMD vectorization tests, defaults to value of
        :param:`from_get_oploop`

    Notes
    -----
    A copy will be made of the :param:`oploop_base` such that the original
    option loop remains intact

    Yields
    ------
    loopy_options: :class:`loopy_options`
        The current state of the oploop, converted to loopy options
    """

    def __init__(self, oploop_base, skip_test=None, yield_index=False,
                 ignored_state_vals=['conp'], from_get_oploop=False,
                 skip_deep_simd=None):
        self.oploop = oploop_base.copy()
        self.state = next(oploop_base.copy()).copy()
        self.yield_index = yield_index
        self.skip_test = skip_test
        self.bad_platforms = set()
        self.ignored_state_vals = ignored_state_vals[:]
        self.skip_deep_simd = skip_deep_simd

    @staticmethod
    def from_dict(oploop_base, skip_test=None, yield_index=False,
                  ignored_state_vals=['conp'], skip_deep_simd=False):
        """
        A convenience method that returns a :class:`OptionLoopWrapper` from the
        list of tuples provided (i.e., that may be turned into an option loop)

        Parameters
        ----------
        oploop_base: dict
            The options that may be converted into an :class:`OptionLoop`
        Returns
        -------
        wrapper: :class:`OptionLoopWrapper`
            The constructed wrapper
        """

        return OptionLoopWrapper(OptionLoop(oploop_base),
                                 skip_test=skip_test,
                                 yield_index=yield_index,
                                 ignored_state_vals=ignored_state_vals,
                                 skip_deep_simd=skip_deep_simd)

    @staticmethod
    def from_get_oploop(owner, skip_test=None, yield_index=False,
                        ignored_state_vals=['conp'], **oploop_kwds):
        """
        A convenience method that returns a :class:`OptionLoopWrapper` from the
        corresponding call to :func:`_get_oploop`

        Parameters
        ----------
        owner: :class:`TestClass`
            The test class that owns the :class:`storage` object
        oploop_kwds: dict
            The keywords for the :func:`_get_oploop` call

        Returns
        -------
        wrapper: :class:`OptionLoopWrapper`
            The constructed wrapper
        """

        oploop = _get_oploop(owner, **oploop_kwds)
        return OptionLoopWrapper(oploop, skip_test=skip_test,
                                 yield_index=yield_index,
                                 ignored_state_vals=ignored_state_vals,
                                 skip_deep_simd=True)

    @staticmethod
    def from_oploop(oploop_base, skip_test=None, yield_index=False,
                    ignored_state_vals=['conp'], skip_deep_simd=True):
        """
        A convenience method that returns a :class:`OptionLoopWrapper` from base
        :class:`OptionLoop` provided

        Parameters
        ----------
        oploop_base: :class:`OptionLoop`
            The base :class:`OptionLoop`

        Returns
        -------
        wrapper: :class:`OptionLoopWrapper`
            The constructed wrapper
        """

        return OptionLoopWrapper(oploop_base,
                                 skip_test=skip_test,
                                 yield_index=yield_index,
                                 ignored_state_vals=ignored_state_vals,
                                 skip_deep_simd=skip_deep_simd)

    def send(self, ignored_arg):
        for i, state in enumerate(self.oploop):
            if _should_skip_oploop(state, skip_test=self.skip_test,
                                   skip_deep_simd=self.skip_deep_simd):
                continue
            # build loopy options
            try:
                # and set any ignored state values for reference
                self.state = state.copy()
                # and build options
                opts = loopy_options(**{x: state[x] for x in state
                                     if x not in self.ignored_state_vals})
            except MissingPlatformError:
                # warn and skip future tests
                logger = logging.getLogger(__name__)
                logger.warn('Platform {} not found'.format(state['platform']))
                self.bad_platforms.update([state['platform']])
                continue
            except BrokenPlatformError as e:
                # expected
                logger = logging.getLogger(__name__)
                logger.debug('Skipping bad platform: {}'.format(e.message))
                continue
            if self.yield_index:
                return (i, opts)
            else:
                return opts
        raise StopIteration

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return utils.stringify_args(self.state, kwd=True)

    def throw(self, type=None, value=None, traceback=None):
        raise StopIteration

    def __iter__(self):
        return self

    def next(self):
        return self.send(None)

    def __next__(self):
        return self.next()

    def close(self):
        """
        Raise GeneratorExit inside generator.
        """
        try:
            self.throw(GeneratorExit)
        except (GeneratorExit, StopIteration):
            pass
        else:
            raise RuntimeError("generator ignored GeneratorExit")


def _generic_tester(owner, func, kernel_calls, rate_func, do_ratespec=False,
                    do_ropsplit=False, do_conp=False, do_vector=True,
                    do_sparse=False, langs=None,
                    sparse_only=False, **kwargs):
    """
    A generic testing method that can be used for to test the correctness of
    any _pyJac_ kernel via the supplied :class:`kernel_call`'s

    Parameters
    ----------
    owner: :class:`TestClass`
        The owning TestClass with access to the shared :class:`storage`
    func : :class:`Callable`
        The _pyJac_ kernel generator function, which returns a
        :class:`knl_info`
    kernel_calls : :class:`kernel_call` or list thereof
        Contains the masks and reference answers for kernel testing
    rate_func: :class:`Callable`
        The _pyJac_ function that generates the reaction rate / jacobian
        specification dictionary.  Should be one of :func:`assign_rates` or
        :func:`determine_jac_inds`
    do_ratespec : bool [False]
        If true, test rate specializations and kernel splitting for simple rates
    do_ropsplit : bool [False]
        If true, test kernel splitting for rop_net
    do_conp:  bool [False]
        If true, test for both constant pressure _and_ constant volume
    do_vector: bool [True]
        If true, use vectorization in testing
    langs: ['opencl']
        The testing languages, @see utils.langs for allowed languages
    do_sparse: bool [False]
        If true, test sparse jacobian alongside full
    sparse_only: bool [False]
            Test only the sparse jacobian (e.g. for testing indexing)
    kwargs: dict
        Any additional arguements to pass to the :param:`func`
    """

    if langs is None:
        langs = get_test_langs()

    if 'conp' in kwargs:
        do_conp = False

    reacs = owner.store.reacs
    specs = owner.store.specs

    exceptions = ['device', 'conp']
    bad_platforms = set()

    # listify
    kernel_calls = utils.listify(kernel_calls)

    def __skip_test(state):
        return 'platform' in state and state['platform'] in bad_platforms

    oploops = OptionLoopWrapper.from_get_oploop(
            owner, do_ratespec=do_ratespec, do_ropsplit=do_ropsplit,
            langs=langs, do_conp=do_conp, do_sparse=do_sparse,
            sparse_only=sparse_only, skip_test=__skip_test, yield_index=True,
            ignored_state_vals=exceptions)
    for i, opt in oploops:
        # find rate info
        rate_info = rate_func(reacs, specs, opt.rate_spec)
        try:
            conp = kwargs['conp']
        except KeyError:
            try:
                conp = oploops.state['conp']
            except KeyError:
                conp = True
        # create namestore
        namestore = arc.NameStore(opt, rate_info, conp,
                                  owner.store.test_size)
        # create the kernel info
        infos = utils.listify(func(opt, namestore, test_size=owner.store.test_size,
                                   **kwargs))

        if not infos:
            logger = logging.getLogger(__name__)
            logger.warn('Function {} returned no kernels for testing. '
                        'This typically is caused by a reaction type '
                        'being missing from the mechanism, e.g.: '
                        'taking the deriviative of the net ROP w.r.t Pressure '
                        'for a mechanism without PLOG or CHEB reactions.'.format(
                            func.__name__))
            continue

        # create a dummy kernel generator
        knl = k_gen.make_kernel_generator(
            kernel_type=KernelType.dummy,
            name='spec_rates',
            loopy_opts=opt,
            kernels=infos,
            namestore=namestore,
            test_size=owner.store.test_size
        )

        knl._make_kernels()

        # create a list of answers to check
        for kc in kernel_calls:
            kc.set_state(knl.array_split, opt.order, namestore, opt.jac_format)

        assert auto_run(knl.kernels, kernel_calls, device=opt.device),\
            'Evaluate {} rates failed'.format(func.__name__)


def _full_kernel_test(self, lang, kernel_gen, test_arr_name, test_arr,
                      ktype, call_name, call_kwds={}, looser_tol_finder=None,
                      atol=1e-8, rtol=1e-5, loose_rtol=1e-4, loose_atol=1,
                      **oploop_kwds):

    home_dir = self.store.script_dir

    def __cleanup():
        # remove library
        clean_dir(lib_dir)
        # remove build
        clean_dir(obj_dir)
        # clean dummy builder
        dist_build = os.path.join(home_dir, 'build')
        if os.path.exists(dist_build):
            shutil.rmtree(dist_build)
        # clean sources
        clean_dir(build_dir)

    P = self.store.P
    V = self.store.V
    exceptions = ['conp']

    # load the module tester template
    mod_test = get_run_source()

    bad_platforms = set()

    def __skip_test(state):
        return 'platform' in state and state['platform'] in bad_platforms
    oploops = OptionLoopWrapper.from_get_oploop(
            self, do_conp=True, do_vector=utils.can_vectorize_lang[lang],
            langs=[lang], skip_test=__skip_test,
            yield_index=True, ignored_state_vals=exceptions,
            do_sparse=ktype == KernelType.jacobian,
            **oploop_kwds)

    from pyjac.core.create_jacobian import determine_jac_inds
    sparse_answers = {}
    for i, opts in oploops:
        with temporary_build_dirs() as (build_dir, obj_dir, lib_dir):
            conp = oploops.state['conp']

            key = (conp, opts.jac_type, opts.order)
            if ktype == KernelType.jacobian and \
                    opts.jac_format == JacobianFormat.sparse \
                    and key not in sparse_answers:
                full_jac = test_arr(conp)
                # get indicies
                inds = determine_jac_inds(
                    self.store.reacs, self.store.specs, RateSpecialization.fixed,
                    jacobian_type=opts.jac_type)
                if opts.order == 'C':
                    # use CRS
                    jac_row_inds = inds['jac_inds']['crs']['row_ptr']
                    jac_col_inds = inds['jac_inds']['crs']['col_ind']
                elif opts.order == 'F':
                    # use CCS
                    jac_row_inds = inds['jac_inds']['ccs']['row_ind']
                    jac_col_inds = inds['jac_inds']['ccs']['col_ptr']
                # need a sparse version of the test array
                sparse = sparsify(full_jac, jac_col_inds, jac_row_inds, opts.order)
                sparse_answers[key] = sparse

            # generate kernel
            kgen = kernel_gen(self.store.reacs, self.store.specs, opts, conp=conp,
                              **call_kwds)

            # generate
            data_path = os.path.join(lib_dir, 'data.bin')
            kgen.generate(build_dir, data_filename=data_path)

            # write header
            write_aux(build_dir, opts, self.store.specs, self.store.reacs)

            # generate wrapper
            pywrap(opts.lang, build_dir, build_dir=obj_dir,
                   obj_dir=obj_dir, out_dir=lib_dir,
                   platform=str(opts.platform), ktype=ktype)

            # get arrays
            phi = np.array(
                self.store.phi_cp if conp else self.store.phi_cv,
                order=opts.order, copy=True)

            # set to some minumum to avoid issues w/ Adept comparison of fmin/fmax
            phi[np.where(phi == 0)] = 1e-300
            param = np.array(P if conp else V, copy=True)

            def namify(name):
                return os.path.join(lib_dir, name + '.npy')

            # save args to dir
            def __saver(arr, name, namelist):
                myname = namify(name)
                # need to split inputs / answer
                np.save(myname, arr.flatten('K'))
                namelist.append(name)

            args = []
            __saver(phi, arc.state_vector, args)
            __saver(param, arc.pressure_array if conp else arc.volume_array, args)

            # sort args
            args = [namify(arg) for arg in utils.kernel_argument_ordering(
                args, ktype)]

            # and now the test values
            tests = []
            if six.callable(test_arr):
                if opts.jac_format == JacobianFormat.sparse:
                    test = np.array(sparse_answers[key], copy=True, order=opts.order)
                else:
                    test = np.array(test_arr(conp), copy=True, order=opts.order)
            else:
                test = np.array(test_arr, copy=True, order=opts.order)
            __saver(test, test_arr_name, tests)

            # sort tests
            tests = utils.kernel_argument_ordering(tests, ktype)
            tests = [namify(arg) for arg in utils.kernel_argument_ordering(
                tests, ktype)]

            # find where the reduced pressure term for non-Lindemann falloff /
            # chemically activated reactions is zero

            def __get_looser_tols(ravel_ind, copy_inds,
                                  looser_tols=np.empty((0,))):
                # fill other ravel locations with tiled test size
                stride = 1
                size = np.prod([test.shape[i] for i in range(test.ndim)
                               if i not in copy_inds])
                for i in [x for x in range(test.ndim) if x not in copy_inds]:
                    repeats = int(np.ceil(size / (test.shape[i] * stride)))
                    ravel_ind[i] = np.tile(np.arange(test.shape[i],
                                           dtype=arc.kint_type),
                                           (repeats, stride)).flatten(
                                                order='F')[:size]
                    stride *= test.shape[i]

                # and use multi_ravel to convert to linear for dphi
                # for whatever reason, if we have two ravel indicies with multiple
                # values we need to need to iterate and stitch them together
                if ravel_ind.size:
                    copy = ravel_ind.copy()
                    new_tols = []
                    if copy_inds.size > 1:
                        # check all copy inds are same shape
                        assert np.all(ravel_ind[copy_inds[0]].shape == y.shape
                                      for y in ravel_ind[copy_inds[1:]])
                    for index in np.ndindex(ravel_ind[copy_inds[0]].shape):
                        # create copy w/ replaced index
                        copy[copy_inds] = [np.array(
                            ravel_ind[copy_inds][i][index], dtype=arc.kint_type)
                            for i in range(copy_inds.size)]
                        # and store the raveled indicies
                        new_tols.append(np.ravel_multi_index(
                            copy, test.shape, order=opts.order))

                    # concat
                    new_tols = np.concatenate(new_tols)

                    # get unique
                    new_tols = np.unique(new_tols)

                    # and force to int for indexing
                    looser_tols = np.asarray(
                        np.union1d(looser_tols, new_tols), dtype=arc.kint_type)
                return looser_tols

            looser_tols = np.empty((0,))
            if looser_tol_finder is not None:
                # pull user specified first
                looser_tols = __get_looser_tols(*looser_tol_finder(
                    test, opts.order, kgen.array_split._have_split(),
                    conp))

            # add more loose tolerances where Pr is zero
            last_zeros = np.where(self.store.ref_Pr == 0)[0]
            if last_zeros.size:
                ravel_ind = np.array(
                    [last_zeros] +
                    [np.arange(test.shape[i], dtype=arc.kint_type)
                     for i in range(1, test.ndim)])
                copy_inds = np.array([0])
                looser_tols = __get_looser_tols(ravel_ind, copy_inds,
                                                looser_tols=looser_tols)
            else:
                looser_tols = np.empty((0,))
                copy_inds = np.empty((0,))

            # number of devices is:
            #   number of threads for CPU
            #   1 for GPU
            num_devices = _get_test_input(
                'num_threads', psutil.cpu_count(logical=False))
            if platform_is_gpu(opts.platform):
                num_devices = 1

            # and save the data.bin file in case of testing
            db = np.concatenate((
                np.expand_dims(phi[:, 0], axis=1),
                np.expand_dims(param, axis=1),
                phi[:, 1:]), axis=1)
            with open(data_path, 'wb') as file:
                db.flatten(order='C').tofile(file,)

            looser_tols_str = '[]'
            if looser_tols.size:
                looser_tols_str = ', '.join(np.char.mod('%i', looser_tols))

            # write the module tester
            with open(os.path.join(lib_dir, 'test.py'), 'w') as file:
                file.write(mod_test.safe_substitute(
                    package='pyjac_{lang}'.format(
                        lang=utils.package_lang[opts.lang]),
                    input_args=utils.stringify_args(args, use_quotes=True),
                    test_arrays=utils.stringify_args(tests, use_quotes=True),
                    looser_tols='[{}]'.format(looser_tols_str),
                    loose_rtol=loose_rtol,
                    loose_atol=loose_atol,
                    atol=atol,
                    rtol=rtol,
                    non_array_args='{}, {}'.format(
                        self.store.test_size, num_devices),
                    output_files='',
                    kernel_name=kgen.name.title()))

            try:
                utils.run_with_our_python([os.path.join(lib_dir, 'test.py')])
            except subprocess.CalledProcessError:
                logger = logging.getLogger(__name__)
                logger.debug(oploops.state)
                assert False, '{} error'.format(kgen.name)


def with_check_inds(check_inds={}, custom_checks={}):
    # This wrapper is to be used to ensure that we're comparing the same indicies
    # throughout a testing method (e.g. to those we set to zero on the input side)

    def check_inds_decorator(func):
        @wraps(func)
        def wrapped(self, *args, **kwargs):
            self.__fixed = False

            def __fix_callables():
                if not self.__fixed:
                    for k, v in six.iteritems(check_inds):
                        if six.callable(v):
                            check_inds[k] = v(self)
                    for k, v in six.iteritems(custom_checks):
                        assert six.callable(v)
                        if six.callable(v):
                            check_inds[k] = v(self, *args)
                self.__fixed = True

            def _get_compare(answer):
                """
                Return an appropriate comparable for the specified check_inds
                """
                __fix_callables()
                axes = []
                inds = []
                for ax, ind in sorted(six.iteritems(check_inds), key=lambda x: x[0]):
                    axes.append(ax)
                    inds.append(ind)
                return get_comparable([inds], [answer], tuple(axes),
                                      tiling=kwargs.pop('tiling', True))

            def _set_at(array, value, order='C'):
                """
                Set the value at check_inds in array to value
                """
                __fix_callables()
                mask = np.array([slice(None)] * array.ndim)
                for ax, ind in check_inds.items():
                    if ax == 0:
                        # don't skip resetting any initial conditions,
                        # even if we're only testing some
                        continue
                    mask[ax] = ind
                array[tuple(mask)] = value

            self._get_compare = _get_compare
            self._set_at = _set_at
            return func(self, *args, **kwargs)
        return wrapped
    return check_inds_decorator


# xfail based on https://stackoverflow.com/a/9615578/1667311
def xfail(condition=None, msg=''):
    """
        An implementation of an expected fail for Nose w/ an optional condition
    """
    def xfail_decorator(test):
        # check that condition is valid
        if condition is not None and not condition:
            return test

        @wraps(test)
        def inner(*args, **kwargs):
            try:
                test(*args, **kwargs)
            except Exception:
                raise SkipTest(msg)
            else:
                raise AssertionError('Failure expected')
        return inner

    return xfail_decorator


# based on
# https://github.com/numpy/numpy/blob/v1.14.0/numpy/testing/nose_tools/decorators.py#L91-L165  # noqa
def skipif(condition, msg=''):
    """
    Make function raise SkipTest exception if a given condition is true.
    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.
    Parameters
    ----------
    skip_condition : bool or callable
        Flag to determine whether to skip the decorated test.
    msg : str, optional
        Message to give on raising a SkipTest exception. Default is None.
    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, causes SkipTest
        to be raised when `skip_condition` is True, and the function
        to be called normally otherwise.
    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.
    """

    def skip_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose

        # Allow for both boolean or callable skip conditions.
        if isinstance(condition, collections.Callable):
            skip_val = lambda: condition()  # noqa
        else:
            skip_val = lambda: condition  # noqa

        def get_msg(func, msg=None):
            """Skip message with information about function being skipped."""
            if msg is None:
                out = 'Test skipped due to test condition'
            else:
                out = msg

            return "Skipping test: {}: {}".format(func.__name__, out)

        # We need to define *two* skippers because Python doesn't allow both
        # return with value and yield inside the same function.
        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            if skip_val():
                raise SkipTest(get_msg(f, msg))
            else:
                return f(*args, **kwargs)

        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            if skip_val():
                raise SkipTest(get_msg(f, msg))
            else:
                for x in f(*args, **kwargs):
                    yield x

        # Choose the right skipper to use when building the actual decorator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func

        return nose.tools.make_decorator(f)(skipper)

    return skip_decorator


class runner(object):
    """
    A base class for running the :func:`_run_mechanism_tests`
    """

    def __init__(self, filetype, rtype=KernelType.jacobian):
        self.rtype = rtype
        self.descriptor = 'jac' if rtype == KernelType.jacobian else 'spec'
        self.filetype = filetype

    def pre(self, gas, data, num_conditions, max_vec_width):
        raise NotImplementedError

    def run(self, state, asplit, dirs, data_output, limits):
        raise NotImplementedError

    def check_file(self, file, state, limits={}):
        raise NotImplementedError

    @property
    def max_per_run(self):
        return None

    def get_phi(self, T, param, extra, moles):
        return np.concatenate((np.reshape(T, (-1, 1)),
                               np.reshape(param, (-1, 1)),
                               np.reshape(extra, (-1, 1)),
                               moles[:, :-1]), axis=1)

    def have_limit(self, state, limits):
        """
        Returns the appropriate limit on the number of initial conditions
        based on the runtype

        Parameters
        ----------
        state: dict
            The current run's parameters
        limits: dict
            If supplied, a limit on the number of conditions that may be tested
            at once. Important for larger mechanisms that may cause memory overflows

        Returns
        -------
        num_conditions: int or None
            The limit on the number of initial conditions for this runtype
            If None is returned, the limit is not supplied
        """

        # check rtype
        rtype_str = str(self.rtype)
        rtype_str = rtype_str[rtype_str.index('.') + 1:]
        if limits and rtype_str in limits:
            if self.rtype == KernelType.jacobian:
                # check sparsity
                if state['sparse'] in limits[rtype_str]:
                    return limits[rtype_str][state['sparse']]
            else:
                return limits[rtype_str]

        return None

    def get_filename(self, loopy_opts, platform, conp, num_cores):
        # store vector size
        self.current_vecsize = loopy_opts.vector_width
        desc = self.descriptor
        if self.rtype == KernelType.jacobian:
            desc += '_sparse' if loopy_opts.jac_format == JacobianFormat.sparse \
                else '_full'
        if loopy_opts.jac_type == JacobianType.finite_difference:
            desc = 'fd' + desc

        vecsize = loopy_opts.vector_width if (
            utils.can_vectorize_lang[loopy_opts.lang]
            and loopy_opts.vector_width) else '1'
        vectype = 'w' if loopy_opts.width else 'd' if loopy_opts.depth else 'par'
        if loopy_opts.is_simd:
            vectype += 'simd'
        assert loopy_opts.rate_spec_kernels == loopy_opts.rop_net_kernels
        split = 'split' if loopy_opts.rate_spec_kernels else 'single'
        conp = 'conp' if conp else 'conv'

        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                desc, loopy_opts.lang, vecsize, loopy_opts.order,
                vectype, platform, utils.enum_to_string(loopy_opts.rate_spec),
                split, num_cores, conp) + self.filetype

    def post(self):
        pass


def _run_mechanism_tests(work_dir, test_matrix, prefix, run,
                         raise_on_missing=True):
    """
    This method is used to consolidate looping for the :mod:`peformance_tester`
    and :mod:`functional tester, as they have very similar execution patterns

    Parameters
    ----------
    work_dir: str
        The directory to run / check in
    run: :class:`runner`
        The code / function to be run for each state of the :class:`OptionLoop`
    test_matrix: str
        The testing matrix file, specifing the configurations to test
    prefix: str
        a prefix within the work directory to store the output of this run
    raise_on_missing: bool
        Raise an exception of the specified :param:`test_matrix` file is not found

    Returns
    -------
    None
    """

    # pull the run type
    rtype = run.rtype

    obj_dir = 'obj'
    build_dir = 'out'
    test_dir = 'test'

    # check if validation
    from pyjac.functional_tester.test import validation_runner
    for_validation = isinstance(run, validation_runner)

    # imports needed only for this tester
    from pyjac.tests.test_utils import get_test_matrix as tm
    from pyjac.tests.test_utils import data_bin_writer as dbw
    from pyjac.core.mech_interpret import read_mech_ct
    from pyjac.core.create_jacobian import find_last_species, create_jacobian
    import cantera as ct

    work_dir = os.path.abspath(work_dir)
    no_regen = set(['num_cores'])

    def __needs_regen(old_state, state):
        # find different entries
        keys = set(list(old_state.keys()) + list(state.keys()))
        diffs = [k for k in keys if k not in old_state or k not in state or
                 state[k] != old_state[k]]
        # ensure they're all in the list that doesn't require regeneration
        if all(x in no_regen for x in diffs):
            return False
        return True

    mechanism_list, oploop, max_vec_width = tm.get_test_matrix(
        work_dir, run.rtype, test_matrix, for_validation,
        raise_on_missing)

    if len(mechanism_list) == 0:
        logger = logging.getLogger(__name__)
        logger.error('No mechanisms found for testing in directory:{}, '
                     'exiting...'.format(work_dir))
        sys.exit(-1)

    for mech_name, mech_info in sorted(mechanism_list.items(),
                                       key=lambda x: x[1]['ns']):
        # ensure directory structure is valid
        this_dir = os.path.join(work_dir, mech_name)
        # take into account the prefix
        if prefix:
            this_dir = os.path.join(this_dir, prefix)
            utils.create_dir(this_dir)

        this_dir = os.path.abspath(this_dir)
        my_obj = os.path.join(this_dir, obj_dir)
        my_build = os.path.join(this_dir, build_dir)
        my_test = os.path.join(this_dir, test_dir)
        # store phi paths
        phi_cp_path = os.path.join(this_dir, 'data_cp.bin')
        phi_cv_path = os.path.join(this_dir, 'data_cv.bin')
        utils.create_dir(my_obj)
        utils.create_dir(my_build)
        utils.create_dir(my_test)

        dirs = {'run': this_dir,
                'test': my_test,
                'build': my_build,
                'obj': my_obj}

        def __cleanup():
            # remove library
            clean_dir(my_obj, False)
            # remove build
            clean_dir(my_build, False)
            # clean sources
            clean_dir(my_test, False)
            # clean dummy builder
            dist_build = os.path.join(os.getcwd(), 'build')
            if os.path.exists(dist_build):
                shutil.rmtree(dist_build)

        # get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))
        gas.basis = 'molar'

        # read our species for MW's
        _, specs, _ = read_mech_ct(gas=gas)

        # find the last species
        gas_map = find_last_species(specs, return_map=True)
        del specs
        # update the gas
        specs = gas.species()[:]
        gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                          species=[specs[x] for x in gas_map],
                          reactions=gas.reactions())
        del specs

        # first load data to get species rates, jacobian etc.
        num_conditions, data = dbw.load(
            [], directory=os.path.join(work_dir, mech_name))

        # rewrite data to file in 'C' order
        # dbw.write(this_dir, num_conditions=num_conditions, data=data)

        # apply species mapping to data
        data[:, 2:] = data[:, 2 + gas_map]

        # check limits
        if 'limits' in mech_info:
            def __try_convert(enumtype, value):
                try:
                    value = utils.EnumType(enumtype)(value)
                except KeyError:
                    logger = logging.getLogger(__name__)
                    logger.warn('Unknown limit type {} found in mechanism info file '
                                'for mech {}'.format(value, mech_name))
                    return False
                return value

            for ktype in mech_info['limits']:
                ktype = __try_convert(KernelType, ktype)

        # set T / P arrays from data
        T = data[:num_conditions, 0].flatten()
        P = data[:num_conditions, 1].flatten()
        # set V = 1 such that concentrations == moles
        V = np.ones_like(P)

        # resize data
        moles = data[:num_conditions, 2:].copy()

        run.pre(gas, {'T': T, 'P': P, 'V': V, 'moles': moles},
                num_conditions, max_vec_width)

        # get the phi's
        phi_cp = run.get_phi(T, P, V, moles)
        phi_cv = run.get_phi(T, V, P, moles)
        np.array(phi_cp, order='C', copy=True).flatten('C').tofile(phi_cp_path)
        np.array(phi_cv, order='C', copy=True).flatten('C').tofile(phi_cv_path)

        # clear old data
        del data
        del T
        del P
        del V
        del moles
        del phi_cp
        del phi_cv

        # begin iterations
        from collections import defaultdict
        done_parallel = defaultdict(lambda: False)
        bad_platforms = set()
        old_state = None

        class parallel_skipper(object):
            # this is simple parallelization, don't need to repeat for
            # different vector sizes, simply choose one and go

            def __init__(self):
                self.done_parallel = {}

            def __call__(self, state):
                par_check = tuple(state[x] for x in state if x not in [
                    'width', 'depth'])
                if done_parallel[par_check]:
                    return True
                # else mark done if we're running a parallel check
                if not (state['width'] or state['depth']):
                    done_parallel[par_check] = True
                return False

        def model_skipper(state):
            # we've decided to skip this model for this configuration
            return 'models' in state and mech_name not in state['models']

        def platform_skipper(state):
            return state['platform'] in bad_platforms

        parallel_skip = parallel_skipper()

        wrapper = OptionLoopWrapper.from_oploop(
            oploop.copy(), ignored_state_vals=['conp', 'num_cores', 'models'],
            skip_test=lambda state: any(call(state) for call in [
                parallel_skip, model_skipper, platform_skipper]),
            skip_deep_simd=True)
        for i, opts in enumerate(wrapper):
            # check for regen
            regen = old_state is None or __needs_regen(
                old_state, wrapper.state.copy())
            # remove any old builds
            if regen:
                __cleanup()
            conp = wrapper.state['conp']

            # get the filename
            data_output = run.get_filename(opts, wrapper.state['platform'], conp,
                                           wrapper.state['num_cores'])

            # if already run, continue
            data_output = os.path.join(this_dir, data_output)
            if run.check_file(data_output, wrapper.state.copy(),
                              mech_info['limits']):
                continue

            phi_path = phi_cp_path if conp else phi_cv_path

            try:
                if regen:
                    # don't regenerate code if we don't need to
                    create_jacobian(opts.lang,
                                    gas=gas,
                                    width=opts.width,
                                    depth=opts.depth,
                                    data_order=opts.order,
                                    build_path=my_build,
                                    kernel_type=rtype,
                                    platform=opts.platform_name,
                                    data_filename=phi_path,
                                    split_rate_kernels=opts.rate_spec_kernels,
                                    rate_specialization=opts.rate_spec,
                                    split_rop_net_kernels=opts.rop_net_kernels,
                                    output_full_rop=(
                                        rtype == KernelType.species_rates
                                        and for_validation),
                                    conp=conp,
                                    use_atomic_doubles=opts.use_atomic_doubles,
                                    use_atomic_ints=opts.use_atomic_ints,
                                    jac_format=opts.jac_format,
                                    jac_type=opts.jac_type,
                                    for_validation=for_validation,
                                    mem_limits=test_matrix)
            except MissingPlatformError:
                # can't run on this platform
                bad_platforms.update([opts.platform_name])
                continue
            except BrokenPlatformError as e:
                # expected
                logger = logging.getLogger(__name__)
                logger.debug('Skipping bad platform: {}'.format(e.message))
                continue

            run.run(wrapper.state.copy(), dirs, phi_path, data_output,
                    mech_info['limits'])

            # store the old state
            old_state = wrapper.state.copy()

        # cleanup any answers / arrays created by the runner for this
        # mechanism
        run.post()
    del run


class TestingLogger(object):
    def start_capture(self, logname=None, loglevel=logging.DEBUG):
        """ Start capturing log output to a string buffer.
            @param newLogLevel: Optionally change the global logging level, e.g.
            logging.DEBUG
        """
        self.buffer = six.StringIO()
        self.buffer.write("Log output")

        if logname:
            logger = logging.getLogger(logname)
        else:
            logger = logging.getLogger()
        if loglevel:
            self.oldloglevel = logger.getEffectiveLevel()
            logger.setLevel(loglevel)
        else:
            self.oldloglevel = None

        self.loghandler = logging.StreamHandler(self.buffer)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s "
                                      "- %(message)s")
        self.loghandler.setFormatter(formatter)
        logger.addHandler(self.loghandler)

    def stop_capture(self):
        """ Stop capturing log output.

        @return: Collected log output as string
        """

        # Remove our handler
        logger = logging.getLogger()

        # Restore logging level (if any)
        if self.oldloglevel is not None:
            logger.setLevel(self.oldloglevel)
        logger.removeHandler(self.loghandler)

        self.loghandler.flush()
        self.buffer.flush()

        return self.buffer.getvalue()


__all__ = ["indexer", "get_split_elements", "kernel_runner", "inNd",
           "get_comparable", "combination", "reduce_oploop", "_generic_tester",
           "_full_kernel_test", "_run_mechanism_tests", "runner", "TestingLogger"]
