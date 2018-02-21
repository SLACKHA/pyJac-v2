from __future__ import division

import os
from string import Template
from collections import OrderedDict
import shutil
import logging
from multiprocessing import cpu_count
import subprocess
import sys
from functools import wraps

from ...loopy_utils.loopy_utils import (get_device_list, kernel_call, populate,
                                        auto_run, RateSpecialization, loopy_options,
                                        JacobianType, JacobianFormat)
from ...core.exceptions import MissingPlatformError, BrokenPlatformError
from ...kernel_utils import kernel_gen as k_gen
from ...core import array_creator as arc
from ...core.mech_auxiliary import write_aux
from .. import get_test_platforms
from ...pywrap import generate_wrapper
from ... import utils
from ...libgen import build_type

try:
    from scipy.sparse import csr_matrix, csc_matrix
except:
    csr_matrix = None
    csc_matrix = None


from unittest.case import SkipTest
from optionloop import OptionLoop
import numpy as np
try:
    # compatability for older numpy
    np_divmod = np.divmod
except:
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


def clean_dir(dirname, remove_dir=True):
    if not os.path.exists(dirname):
        return
    for file in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, file)):
            os.remove(os.path.join(dirname, file))
    if remove_dir:
        shutil.rmtree(dirname, ignore_errors=True)


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
            name='dummy',
            loopy_opts=loopy_opts,
            kernels=infos,
            namestore=namestore,
            test_size=self.test_size,
            for_testing=True
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

    @classmethod
    def get_split_axis(self, order='C'):
        """
        Returns the array axis that would be split (if there exists a split)
        based on :param:`order`
        """
        return -1 if order == 'F' else 0

    @classmethod
    def get_split_dim(self, shape, order='C'):
        """
        Returns the split dimension of the supplied :param:`shape` based on the
        supplied data order :param:`order`
        """

        return shape[0 if order == 'F' else -1]

    def _get_F_index(self, inds, axes):
        """
        for the 'F' order split, the last axis is split and moved to the beginning.
        """
        rv = [slice(None)] * self.out_ndim
        axi = next((i for i in six.moves.range(len(axes))
                    if axes[i] == self.ref_ndim - 1), None)
        if axi is not None:
            # the first index is the remainder of the ind by the new dimension size
            # and the last index is the floor division of the new dim size

            # check that this is ind is not a slice
            # if it is we don't need to to anything
            if isinstance(inds[axi], np.ndarray):
                rv[-1], rv[0] = np_divmod(
                    inds[axi], self.split_dim, dtype=np.int32)

        for i, ax in enumerate(axes):
            if i != axi:
                # there is no change in the actual indexing here
                # however, the destination index will be increased by one
                # to account for the new inserted index at the front of the array

                # check that this is ind is not a slice
                # if it is we don't need to to anything
                if isinstance(inds[i], np.ndarray):
                    rv[ax + 1] = inds[i][:].astype(np.int32)

        return rv

    def _get_C_index(self, inds, axes):
        """
        for the 'C' order split, the first axis is split and moved to the end.
        """
        rv = [slice(None)] * self.out_ndim
        axi = next((i for i in six.moves.range(len(axes)) if axes[i] == 0),
                   None)
        if axi is not None:
            # and first index is the floor division of the new dim size
            # the last index is the remainder of the ind by the new dimension size

            # check that this is ind is not a slice
            # if it is we don't need to to anything
            if isinstance(inds[axi], np.ndarray):
                # it's a numpy array, so we can divmod
                rv[0], rv[-1] = np_divmod(
                    inds[axi], self.split_dim, dtype=np.int32)

        for i, ax in enumerate(axes):
            if i != axi:
                # there is no change in the actual indexing here

                # check that this is ind is not a slice
                # if it is we don't need to to anything
                if isinstance(inds[i], np.ndarray):
                    rv[ax] = inds[i][:].astype(np.int32)

        return rv

    def __init__(self, ref_ndim, out_ndim, out_shape, order='C'):
        self.ref_ndim = ref_ndim
        self.out_ndim = out_ndim
        self._indexer = self._get_F_index if order == 'F' else \
            self._get_C_index
        self.split_dim = self.get_split_dim(out_shape, order)

    def __call__(self, inds, axes):
        return self._indexer(inds, axes)


def parse_split_index(arr, mask, order, ref_ndim=2, axis=(1,), stride_arr=None,
                      size_arr=None):
    """
    A helper method to get the index of an element in a split array for all initial
    conditions

    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        The split array to use
    mask: :class:`numpy.ndarray` or list thereof
        The indicies to determine
    order: ['C', 'F']
        The numpy data order
    ref_ndim: int [2]
        The dimension of the unsplit array
    axis: int or list of int
        The axes the mask's correspond to.
        Must be of the same shape / size as mask
    stride_arr: :class:`np.ndarray` or list
        This should _never_ be supplied by the user.  Essentially what happens is
        that for an F-split sparse Jacobian, the size of the split dimension differs
        between the reference answer, and the sparse matrix.  In order to get the
        comparison right, the sparse split must use the strides of the reference
        answer in order to get proper tiling of the mask.
    size_arr: :class:`np.ndarray` or int
        Similar to the stride array, but for converting from a split reference answer
        _to_ a sparse array

    Returns
    -------
    mask: tuple of int / slice
        A proper indexing for the split array
    """

    _get_index = indexer(ref_ndim, arr.ndim, arr.shape, order)

    # handle multi-dim combination of mask
    if not isinstance(mask, np.ndarray):
        size = np.prod([x.size for x in mask])
    else:
        # single dim, use simple mask
        size = mask.size
        mask = [mask]
        assert len(axis) == 1, "Supplied mask doesn't match given axis"

    if size_arr is not None:
        size = np.prod(size_arr)

    if arr.ndim == ref_ndim:
        # no split
        masking = np.array([slice(None)] * arr.ndim)
        for i, x in enumerate(axis):
            masking[x] = mask[i]
        return tuple(masking)

    # get the index arrays
    inds = np.array(_get_index(mask, axis))
    stride_arr = np.array([np.unique(x).size for x in inds], dtype=np.int32) \
        if stride_arr is None else stride_arr
    # get non-slice inds
    non_slice = np.array([i for i, x in enumerate(inds)
                          if isinstance(x, np.ndarray)], dtype=np.int32)
    # create the output masker
    masking = np.array([slice(None)] * arr.ndim)
    masking[non_slice] = [np.empty(size, dtype=np.int32)
                          for x in range(non_slice.size)]

    # need to fill the masking array
    # the first and last indicies are split
    stride = 1
    for i in reversed(range(len(masking[non_slice]))):
        # handled below
        if non_slice[i] == 0:
            continue
        # find the number of times to tile this array
        repeats = int(np.ceil(size / (inds[non_slice][i].size * stride)))
        shape = (stride, repeats)
        # tile and place in mask
        masking[non_slice][i][:] = np.tile(
            inds[non_slice][i], shape).flatten(order='F')[:size]
        # the first and last index are tied together by the split
        if i == len(masking[non_slice]) - 1 and 0 in non_slice:
            masking[0][:] = np.tile(inds[0], shape).flatten(
                order='F')[:size]
        # and update stride
        stride *= stride_arr[i]

    return tuple(masking)


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


class get_comparable(object):
    """
    A wrapper for the kernel_call's _get_comparable function that fixes
    comparison for split arrays

    Properties
    ----------
    compare_mask: list of :class:`numpy.ndarray` or list of tuples of
            :class:`numpy.ndarray`
        The default comparison mask.  If multi-dimensional, should be a list
        of tuples of :class:`numpy.ndarray`'s corresponding to the compare axis
    ref_answer: :class:`numpy.ndarray`
        The answer to compare to, used to determine the proper shape
    compare_axis: -1 or iterable of int
        The axis (or axes) to compare along
    """

    def __init__(self, compare_mask, ref_answer, compare_axis=(1,)):
        self.compare_mask = compare_mask
        if not isinstance(self.compare_mask, list):
            self.compare_mask = [self.compare_mask]

        self.ref_answer = ref_answer
        if not isinstance(self.ref_answer, list):
            self.ref_answer = [ref_answer]

        self.compare_axis = compare_axis

        from collections import Iterable
        if isinstance(self.compare_axis, Iterable):
            # if iterable, check that all the compare masks are of the same
            # length as the compare axis
            assert all(len(x) == len(self.compare_axis) for x in self.compare_mask),\
                "Can't use dissimilar compare masks / axes"

    def __call__(self, kc, outv, index, is_answer=False):
        mask = list(self.compare_mask[index][:])
        ans = self.ref_answer[index]
        try:
            axis = self.compare_axis[:]
        except:
            axis = self.compare_axis
        ndim = ans.ndim

        # helper methods
        def __get_sparse_mat(as_inds=True):
            # setup dummy sparse matrix
            if kc.current_order == 'C':
                matrix = csr_matrix
                inds = kc.col_inds
                indptr = kc.row_inds
            else:
                matrix = csc_matrix
                inds = kc.row_inds
                indptr = kc.col_inds
            # next create and get indicies
            matrix = matrix((np.ones(inds.size), inds, indptr)).tocoo()
            row, col = matrix.row, matrix.col
            if as_inds:
                return np.asarray((row, col)).T
            return row, col

        # extract row & column masks
        def __row_and_col_mask():
            if self.compare_axis == -1:
                return 1, mask[1], 2, mask[2]
            row_ind = next(i for i, ind in enumerate(self.compare_axis)
                           if ind == 1)
            row_mask = mask[row_ind]
            col_ind = next(i for i, ind in enumerate(self.compare_axis)
                           if ind == 2)
            col_mask = mask[col_ind]
            return row_ind, row_mask, col_ind, col_mask

        # check for sparse (ignore answers, which do not get transformed into
        # sparse and should be dealt with as usual)
        stride_arr = None
        size_arr = None
        if kc.jac_format == JacobianFormat.sparse:
            if not is_answer:
                if csc_matrix is None and csr_matrix is None:
                    raise SkipTest('Cannot test sparse matricies without scipy'
                                   ' installed')
                # need to collapse the mask
                inds = __get_sparse_mat()

                # next we need to find the 1D index of all the row, col pairs in
                # the mask
                row_ind, row_mask, col_ind, col_mask = __row_and_col_mask()

                # remove old inds
                mask = [mask[i] for i in range(len(mask)) if i not in [
                    row_ind, col_ind]]
                if self.compare_axis != -1:
                    axis = tuple(x for i, x in enumerate(axis) if i not in [
                        row_ind, col_ind])

                # store ic mask in case we need strides array
                ic_size = mask[0].size if mask and isinstance(mask[0], np.ndarray) \
                    else ans.shape[0]

                # add the sparse indicies
                if self.compare_axis != -1:
                    new_mask = combination(
                        row_mask, col_mask, order=kc.current_order)
                else:
                    new_mask = np.vstack((row_mask.T, col_mask.T)).T
                mask.append(inNd(new_mask, inds))
                # and the new axis
                if self.compare_axis != -1:
                    axis = axis + (1,)
                # and indicate that we've lost a dimension
                ndim -= 1

                if kc.current_order == 'F' and outv.ndim != ndim:
                    # as the split array dimension differs, we need to supply the
                    # same strides as the reference answer
                    size_arr = [mask[row_ind].size]
                    if len(self.compare_mask[index]) == 3:
                        size_arr = [ic_size] + size_arr
                    # and fix the stride such that the rows and columns
                    # move together
                    if kc.current_order == 'F':
                        stride_arr = [1] + [ic_size, 1, 1]

            else:
                # we need to filter the reference answer based on what is actually in
                # the sparse jacoban

                # get the (row, col) indicies of the sparse matrix
                inds = __get_sparse_mat()
                # find the row & column mask
                row_ind, row_mask, col_ind, col_mask = __row_and_col_mask()
                # combine col & row masks
                if self.compare_axis != -1:
                    new_mask = combination(row_mask, col_mask,
                                           order=kc.current_order)
                else:
                    new_mask = np.vstack((row_mask, col_mask)).T
                # find where the sparse indicies correspond to our row & column masks
                new_mask = new_mask[inNd(inds, new_mask)]
                # split back into rows and columns
                mask[row_ind] = new_mask[:, 0]
                mask[col_ind] = new_mask[:, 1]

                if outv.ndim == ndim:
                    axis = -1
                    if len(mask) == 3:
                        # pre-filter IC array
                        outv = outv[mask[0]]
                        mask[0] = slice(None)
                    else:
                        mask.insert(0, slice(None))
                else:
                    # need to take the size to be the number of initial conditions
                    # multiplied by the number of indicies in our mask
                    ic_vals = mask[0] if len(mask) == 3 and isinstance(
                        mask[0], np.ndarray) else np.arange(ans.shape[0])
                    ic_size = ic_vals.size
                    size_arr = [mask[row_ind].size]
                    if len(mask) == 3:
                        size_arr = [ic_size] + size_arr
                    # and fix the stride such that the rows and columns
                    # move together
                    stride_arr = [1] * outv.ndim
                    if kc.current_order == 'F':
                        stride_arr = [1] + [np.unique(ic_vals).size, 1, 1]
                    else:
                        d, m = np_divmod(ic_vals, outv.shape[-1])
                        stride_arr = [np.unique(d).size] \
                            + [1, 1] + [np.unique(m).size]

        # check for vectorized data order
        if outv.ndim == ndim:
            # return the default, as it can handle it
            return kernel_call('', [], compare_mask=[mask],
                               compare_axis=axis)._get_comparable(outv, 0)
        elif axis != -1:
            # get the split indicies
            masking = parse_split_index(
                outv, mask, kc.current_order, ndim, axis, stride_arr, size_arr)

        else:
            # we supplied a list of indicies, all we really have to do is convert
            # them and return

            _get_index = indexer(ndim, outv.ndim, outv.shape, kc.current_order)
            # first check we have a reasonable mask
            assert ndim == len(mask), "Can't use dissimilar compare masks / axes"
            # dummy comparison axis
            comp_axis = np.arange(ndim)
            # convert inds
            masking = tuple(_get_index(mask, comp_axis))

        # and return
        return outv[masking]


def _get_oploop(owner, do_ratespec=False, do_ropsplit=False, do_conp=True,
                langs=['c', 'opencl'], do_vector=True, do_sparse=False,
                do_approximate=False, do_finite_difference=False,
                sparse_only=False):

    platforms = get_test_platforms(owner.store.test_platforms,
                                   do_vector=do_vector, langs=langs)
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
    oploop += [('knl_type', ['map'])]
    out = None
    for p in platforms:
        val = OptionLoop(OrderedDict(p + oploop))
        if out is None:
            out = val
        else:
            out = out + val

    return out


def _generic_tester(owner, func, kernel_calls, rate_func, do_ratespec=False,
                    do_ropsplit=False, do_conp=False, do_vector=True,
                    do_sparse=False, langs=None,
                    sparse_only=False, **kw_args):
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
        from .. import get_test_langs
        langs = get_test_langs()

    oploop = _get_oploop(owner, do_ratespec=do_ratespec, do_ropsplit=do_ropsplit,
                         langs=langs, do_conp=do_conp, do_sparse=do_sparse,
                         sparse_only=sparse_only)

    reacs = owner.store.reacs
    specs = owner.store.specs

    exceptions = ['device', 'conp']
    bad_platforms = set()

    for i, state in enumerate(oploop):
        if utils.can_vectorize_lang[state['lang']] and (
                state['width'] is not None and state['depth'] is not None):
            # can't vectorize deep and wide concurrently
            continue

        # skip bad platforms
        if 'platform' in state and state['platform'] in bad_platforms:
            continue

        try:
            opt = loopy_options(**{x: state[x] for x in state
                                if x not in exceptions})
        except MissingPlatformError:
            # warn and skip future tests
            logger = logging.getLogger(__name__)
            logger.warn('Platform {} not found'.format(state['platform']))
            bad_platforms.update([state['platform']])
            continue
        except BrokenPlatformError as e:
            # expected
            logger = logging.getLogger(__name__)
            logger.info('Skipping bad platform: {}'.format(e.message))
            continue

        # find rate info
        rate_info = rate_func(reacs, specs, opt.rate_spec)
        try:
            conp = state['conp']
        except:
            try:
                conp = kw_args['conp']
            except:
                conp = True
        # create namestore
        namestore = arc.NameStore(opt, rate_info, conp,
                                  owner.store.test_size)
        # create the kernel info
        infos = func(opt, namestore,
                     test_size=owner.store.test_size, **kw_args)

        if not isinstance(infos, list):
            try:
                infos = list(infos)
            except:
                infos = [infos]

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
            name='spec_rates',
            loopy_opts=opt,
            kernels=infos,
            namestore=namestore,
            test_size=owner.store.test_size,
            for_testing=True
        )

        knl._make_kernels()

        # create a list of answers to check
        try:
            for kc in kernel_calls:
                kc.set_state(knl.array_split, state['order'], namestore,
                             state['jac_format'])
        except TypeError as e:
            if str(e) != "'kernel_call' object is not iterable":
                raise e
            kernel_calls.set_state(knl.array_split, state['order'], namestore,
                                   state['jac_format'])

        assert auto_run(knl.kernels, kernel_calls, device=opt.device),\
            'Evaluate {} rates failed'.format(func.__name__)


def _full_kernel_test(self, lang, kernel_gen, test_arr_name, test_arr,
                      btype, call_name, call_kwds={}, looser_tol_finder=None,
                      atol=1e-8, rtol=1e-5, loose_rtol=1e-4, loose_atol=1,
                      **oploop_kwds):
    oploop = _get_oploop(self, do_conp=True, do_vector=lang != 'c', langs=[lang],
                         **oploop_kwds)

    package_lang = {'opencl': 'ocl',
                    'c': 'c'}
    build_dir = self.store.build_dir
    obj_dir = self.store.obj_dir
    lib_dir = self.store.lib_dir
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

    # now start test
    for i, state in enumerate(oploop):
        if utils.can_vectorize_lang[state['lang']] and (
                state['width'] is not None and state['depth'] is not None):
            # can't vectorize both directions at the same time
            continue

        # clean old files
        __cleanup()

        # skip bad platforms
        if 'platform' in state and state['platform'] in bad_platforms:
            continue

        try:
            # create loopy options
            opts = loopy_options(**{x: state[x] for x in state
                                 if x not in exceptions})
        except MissingPlatformError:
            # warn and skip future tests
            logger = logging.getLogger(__name__)
            logger.warn('Platform {} not found'.format(state['platform']))
            bad_platforms.update([state['platform']])
            continue
        except BrokenPlatformError as e:
            # expected
            logger = logging.getLogger(__name__)
            logger.info('Skipping bad platform: {}'.format(e.message))
            continue

        # check to see if device is CPU
        # if (opts.lang == 'opencl' and opts.device_type == cl.device_type.CPU) \
        #        and (opts.depth is None or not opts.use_atomics):
        #    opts.use_private_memory = True

        conp = state['conp']

        # generate kernel
        kgen = kernel_gen(self.store.reacs, self.store.specs, opts, conp=conp,
                          **call_kwds)

        # generate
        kgen.generate(
            build_dir, data_filename=os.path.join(os.getcwd(), 'data.bin'))

        # write header
        write_aux(build_dir, opts, self.store.specs, self.store.reacs)

        # generate wrapper
        generate_wrapper(opts.lang, build_dir, build_dir=obj_dir,
                         out_dir=lib_dir, platform=str(opts.platform),
                         btype=btype)

        # get arrays
        phi = np.array(
            self.store.phi_cp if conp else self.store.phi_cv,
            order=opts.order, copy=True)
        param = np.array(P if conp else V, copy=True)

        # save args to dir
        def __saver(arr, name, namelist):
            myname = os.path.join(lib_dir, name + '.npy')
            # need to split inputs / answer
            np.save(myname, kgen.array_split.split_numpy_arrays(
                arr)[0].flatten('K'))
            namelist.append(myname)

        args = []
        __saver(phi, 'phi', args)
        __saver(param, 'param', args)

        # and now the test values
        tests = []
        if six.callable(test_arr):
            test = np.array(test_arr(conp), copy=True, order=opts.order)
        else:
            test = np.array(test_arr, copy=True, order=opts.order)
        ref_ndim = test.ndim
        __saver(test, test_arr_name, tests)

        # find where the reduced pressure term for non-Lindemann falloff / chemically
        # activated reactions is zero

        # get split arrays
        test, = kgen.array_split.split_numpy_arrays(test)

        def __get_looser_tols(ravel_ind, copy_inds,
                              looser_tols=np.empty((0,))):
            # fill other ravel locations with tiled test size
            stride = 1
            size = np.prod([test.shape[i] for i in range(test.ndim)
                           if i not in copy_inds])
            for i in [x for x in range(test.ndim) if x not in copy_inds]:
                repeats = int(np.ceil(size / (test.shape[i] * stride)))
                ravel_ind[i] = np.tile(np.arange(test.shape[i], dtype=np.int32),
                                       (repeats, stride)).flatten(
                                            order='F')[:size]
                stride *= test.shape[i]

            # and use multi_ravel to convert to linear for dphi
            # for whatever reason, if we have two ravel indicies with multiple values
            # we need to need to iterate and stitch them together
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
                        ravel_ind[copy_inds][i][index], dtype=np.int32)
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
                    np.union1d(looser_tols, new_tols), dtype=np.int32)
            return looser_tols

        looser_tols = np.empty((0,))
        if looser_tol_finder is not None:
            # pull user specified first
            looser_tols = __get_looser_tols(*looser_tol_finder(
                test, opts.order, kgen.array_split._have_split(),
                state['conp']))

        # add more loose tolerances where Pr is zero
        last_zeros = np.where(self.store.ref_Pr == 0)[0]
        if last_zeros.size:
            if kgen.array_split._have_split():
                ravel_ind = parse_split_index(test, (last_zeros,), opts.order,
                                              ref_ndim=ref_ndim, axis=(0,))
                # and list
                ravel_ind = np.array(ravel_ind)

                # just choose the initial condition indicies
                if opts.order == 'C':
                    # wide split, take first and last index
                    copy_inds = np.array([0, test.ndim - 1], dtype=np.int32)
                elif opts.order == 'F':
                    # deep split, take just the IC index at 1
                    copy_inds = np.array([1], dtype=np.int32)
            else:
                ravel_ind = np.array(
                    [last_zeros] + [np.arange(test.shape[i], dtype=np.int32)
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
        num_devices = int(cpu_count() / 2)
        if platform_is_gpu(opts.platform):
            num_devices = 1

        # and save the data.bin file in case of testing
        db = np.concatenate((
            np.expand_dims(phi[:, 0], axis=1),
            np.expand_dims(param, axis=1),
            phi[:, 1:]), axis=1)
        with open(os.path.join(lib_dir, 'data.bin'), 'wb') as file:
            db.flatten(order=opts.order).tofile(file,)

        looser_tols_str = '[]'
        if looser_tols.size:
            looser_tols_str = ', '.join(np.char.mod('%i', looser_tols))
        # write the module tester
        with open(os.path.join(lib_dir, 'test.py'), 'w') as file:
            file.write(mod_test.safe_substitute(
                package='pyjac_{lang}'.format(
                    lang=package_lang[opts.lang]),
                input_args=', '.join('"{}"'.format(x) for x in args),
                test_arrays=', '.join('"{}"'.format(x) for x in tests),
                looser_tols='[{}]'.format(looser_tols_str),
                loose_rtol=loose_rtol,
                loose_atol=loose_atol,
                atol=atol,
                rtol=rtol,
                non_array_args='{}, {}'.format(
                    self.store.test_size, num_devices),
                call_name=call_name,
                output_files=''))

        try:
            subprocess.check_call([
                'python{}.{}'.format(
                    sys.version_info[0], sys.version_info[1]),
                os.path.join(lib_dir, 'test.py')])
            # cleanup
            for x in args + tests:
                os.remove(x)
            os.remove(os.path.join(lib_dir, 'test.py'))
        except subprocess.CalledProcessError:
            logger = logging.getLogger(__name__)
            logger.debug(state)
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
                return get_comparable([inds], [answer], tuple(axes))

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


class runner(object):
    """
    A base class for running the :func:`_run_mechanism_tests`
    """

    def __init__(self, rtype=build_type.jacobian):
        self.rtype = rtype
        self.descriptor = 'jac' if rtype == build_type.jacobian else 'spec'

    def pre(self, gas, data, num_conditions, max_vec_width):
        raise NotImplementedError

    def run(self, state, asplit, dirs, data_output, limits):
        raise NotImplementedError

    def get_filename(self, state):
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
        if limits is not None and rtype_str in limits:
            if self.rtype == build_type.jacobian:
                # check sparsity
                if state['sparse'] in limits[rtype_str]:
                    return limits[rtype_str][state['sparse']]
            else:
                return limits[rtype_str]

        return None

    def post(self):
        pass


def _run_mechanism_tests(work_dir, test_platforms, prefix, run, mem_limits='',
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
    test_platforms: str
        The testing platforms file, specifing the configurations to test
    prefix: str
        a prefix within the work directory to store the output of this run
    raise_on_missing: bool
        Raise an exception of the specified :param:`test_platforms` file is not found

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
    from ...functional_tester.test import validation_runner
    for_validation = isinstance(run, validation_runner)

    # imports needed only for this tester
    from . import get_test_matrix as tm
    from . import data_bin_writer as dbw
    from ...core.mech_interpret import read_mech_ct
    from ...core.array_creator import array_splitter
    from ...core.create_jacobian import find_last_species, create_jacobian
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
        work_dir, run.rtype, test_platforms, for_validation,
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
        dbw.write(this_dir, num_conditions=num_conditions, data=data)

        # apply species mapping to data
        data[:, 2:] = data[:, 2 + gas_map]

        # figure out the number of conditions to test
        num_conditions = int(
            np.floor(num_conditions / max_vec_width) * max_vec_width)

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

            def __change_limit(keylist):
                subdict = mech_info['limits']
                keylist = [str(key)[str(key).index('.') + 1:].lower()
                           for key in keylist]
                for i, key in enumerate(keylist):
                    if key not in subdict:
                        return
                    if i < len(keylist) - 1:
                        # recurse
                        subdict = subdict[key]
                    else:
                        lim = int(np.floor(subdict[key] / max_vec_width)
                                  * max_vec_width)
                        if lim != subdict[key]:
                            subdict[key] = lim
                            logger = logging.getLogger(__name__)
                            logger.info(
                                'Changing limit for mech {name} ({keys}) '
                                'from {old} to {new} to ensure even '
                                'divisbility by vector width'.format(
                                    name=mech_name,
                                    keys='.'.join(keylist),
                                    old=subdict[key],
                                    new=lim))

            for btype in mech_info['limits']:
                btype = __try_convert(build_type, btype)
                if btype == build_type.jacobian:
                    __change_limit([btype, JacobianFormat.sparse])
                    __change_limit([btype, JacobianFormat.full])
                else:
                    __change_limit([btype])

        # set T / P arrays from data
        T = data[:num_conditions, 0].flatten()
        P = data[:num_conditions, 1].flatten()
        # set V = 1 such that concentrations == moles
        V = np.ones_like(P)

        # resize data
        moles = data[:num_conditions, 2:].copy()

        run.pre(gas, {'T': T, 'P': P, 'V': V, 'moles': moles},
                num_conditions, max_vec_width)

        # clear old data
        del data
        del T
        del P
        del V
        del moles

        # begin iterations
        from collections import defaultdict
        done_parallel = defaultdict(lambda: False)
        op = oploop.copy()
        bad_platforms = set()
        old_state = None
        for i, state in enumerate(op):
            # check for regen
            regen = old_state is None or __needs_regen(old_state, state.copy())
            # remove any old builds
            if regen:
                __cleanup()
            lang = state['lang']
            vecsize = state['vecsize']
            order = state['order']
            wide = state['wide']
            deep = state['deep']
            platform = state['platform']
            rate_spec = state['rate_spec']
            split_kernels = state['split_kernels']
            conp = state['conp']
            par_check = tuple(state[x] for x in state if x != 'vecsize')
            sparse = state['sparse']
            jac_type = state['jac_type']

            if platform in bad_platforms:
                continue
            if not (deep or wide) and done_parallel[par_check]:
                # this is simple parallelization, don't need to repeat for
                # different vector sizes, simply choose one and go
                continue
            elif not (deep or wide):
                # mark done
                done_parallel[par_check] = True

            if rate_spec == 'fixed' and split_kernels:
                continue  # not a thing!

            if deep and wide:
                # can't do both simultaneously
                continue

            # get the filename
            data_output = run.get_filename(state.copy())

            # if already run, continue
            data_output = os.path.join(this_dir, data_output)
            if run.check_file(data_output, state.copy(), mech_info['limits']):
                continue

            # store phi path
            phi_path = os.path.join(this_dir, 'data.bin')

            try:
                if regen:
                    # don't regenerate code if we don't need to
                    create_jacobian(lang,
                                    gas=gas,
                                    vector_size=vecsize,
                                    wide=wide,
                                    deep=deep,
                                    data_order=order,
                                    build_path=my_build,
                                    skip_jac=rtype == build_type.species_rates,
                                    platform=platform,
                                    data_filename=phi_path,
                                    split_rate_kernels=split_kernels,
                                    rate_specialization=rate_spec,
                                    split_rop_net_kernels=split_kernels,
                                    output_full_rop=(
                                        rtype == build_type.species_rates),
                                    conp=conp,
                                    use_atomics=state['use_atomics'],
                                    jac_format=sparse,
                                    jac_type=jac_type,
                                    for_validation=for_validation,
                                    seperate_kernels=state['seperate_kernels'],
                                    mem_limits=mem_limits)
            except MissingPlatformError:
                # can't run on this platform
                bad_platforms.update([platform])
                continue
            except BrokenPlatformError as e:
                # expected
                logger = logging.getLogger(__name__)
                logger.info('Skipping bad platform: {}'.format(e.message))
                continue

            # get an array splitter
            width = state['vecsize'] if state['wide'] else None
            depth = state['vecsize'] if state['deep'] else None
            order = state['order']
            asplit = array_splitter(type('', (object,), {
                'width': width, 'depth': depth, 'order': order}))

            run.run(state.copy(), asplit, dirs, phi_path, data_output,
                    mech_info['limits'])

            # store the old state
            old_state = state.copy()

        # cleanup any answers / arrays created by the runner for this
        # mechanism
        run.post()
    del run


def platform_is_gpu(platform):
    """
    Attempts to determine if the given platform name corresponds to a GPU

    Parameters
    ----------
    platform_name: str or :class:`pyopencl.platform`
        The name of the platform to check

    Returns
    -------
    is_gpu: bool or None
        True if platform found and the device type is GPU
        False if platform found and the device type is not GPU
        None otherwise
    """
    # filter out C or other non pyopencl platforms
    if not platform:
        return False
    try:
        import pyopencl as cl
        if isinstance(platform, cl.Platform):
            return platform.get_devices()[0].type == cl.device_type.GPU

        for p in cl.get_platforms():
            if platform.lower() in p.name.lower():
                # match, get device type
                dtype = set(d.type for d in p.get_devices())
                assert len(dtype) == 1, (
                    "Mixed device types on platform {}".format(p.name))
                # fix cores for GPU
                if cl.device_type.GPU in dtype:
                    return True
                return False
    except ImportError:
        pass
    return None
