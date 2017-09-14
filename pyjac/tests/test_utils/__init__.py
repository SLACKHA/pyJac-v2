from __future__ import division

import os
from string import Template
from ...loopy_utils.loopy_utils import (get_device_list, kernel_call, populate,
                                        auto_run, RateSpecialization, loopy_options)
from ...core.exceptions import MissingPlatformError
from ...kernel_utils import kernel_gen as k_gen
from ...core import array_creator as arc
from .. import get_test_platforms
from optionloop import OptionLoop
import numpy as np
import six

from collections import OrderedDict
import logging


def __get_template(fname):
    with open(fname, 'r') as file:
        return Template(file.read())


def get_import_source():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return __get_template(os.path.join(script_dir, 'test_import.py.in'))


def get_read_ics_source():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return __get_template(os.path.join(script_dir, 'read_ic_setup.py.in'))


def clean_dir(dirname, remove_dir=True):
    if not os.path.exists(dirname):
        return
    for file in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, file)):
            os.remove(os.path.join(dirname, file))
    if remove_dir:
        os.rmdir(dirname)


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

    def __call__(self, eqs, loopy_opts, namestore, test_size):
        device = get_device_list()[0]

        infos = self.func(eqs, loopy_opts, namestore, test_size=test_size,
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
            test_size=self.test_size
        )
        gen._make_kernels()
        kc = kernel_call('dummy',
                         [None],
                         **self.args)
        kc.set_state(gen.array_split, loopy_opts.order)
        self.out_arg_names = [[
            x for x in k.get_written_variables()
            if x not in k.temporary_variables]
            for k in gen.kernels]
        return populate(gen.kernels, kc, device=device)[0]


class indexer(object):
    """
    Utility class that helps in transformation of old indicies to split array
    indicies
    """

    def _get_F_index(self, inds, axes):
        """
        for the 'F' order split, the last axis is split and moved to the beginning.
        """
        rv = [slice(None)] * self.out_ndim
        axi = next((i for i in six.moves.range(len(axes))
                    if axes[i] == self.ref_ndim - 1),
                   None)
        if axi is not None:
            # the first index is the remainder of the ind by the new dimension size
            # and the last index is the floor division of the new dim size
            rv[-1], rv[0] = np.divmod(inds[axi], self.out_shape[0], dtype=np.int32)

        for i, ax in enumerate(axes):
            if ax != axi:
                # there is no change in the actual indexing here
                # however, the destination index will be increased by one
                # to account for the new inserted index at the front of the array
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
            rv[0], rv[-1] = np.divmod(inds[axi], self.out_shape[-1], dtype=np.int32)

        for i, ax in enumerate(axes):
            if i != axi:
                # there is no change in the actual indexing here
                rv[ax] = inds[i][:].astype(np.int32)

        return rv

    def __init__(self, ref_ndim, out_ndim, out_shape, order='C'):
        self.ref_ndim = ref_ndim
        self.out_shape = out_shape
        self.out_ndim = out_ndim
        self._indexer = self._get_F_index if order == 'F' else \
            self._get_C_index

    def __call__(self, inds, axes):
        return self._indexer(inds, axes)


def parse_split_index(arr, ind, order, ref_ndim=2, axis=1):
    """
    A helper method to get the index of an element in a split array for all initial
    conditions

    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        The split array to use
    ind: int or list of int
        The element index(ices)
    order: ['C', 'F']
        The numpy data order
    ref_ndim: int [2]
        The dimension of the unsplit array
    axis: int or list of int
        The axes the ind's correspond to

    Returns
    -------
    index: tuple of int / slice
        A proper indexing for the split array
    """

    dim = arr.ndim
    index = [slice(None)] * dim

    _get_index = indexer(ref_ndim, arr.ndim, arr.shape, order)

    try:
        for i, ax in _get_index(ind, axis):
            index[ax] = i
    except TypeError:
        inds, axs = _get_index(arr, (ind,), (axis,))
        for i, ax in zip(inds, axs):
            index[ax] = i
    return tuple(index)


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
    compare_axis: int or list of int
        The axis (or axes) to compare along
    """

    def __init__(self, compare_mask, ref_answer, compare_axis=1):
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

    def __call__(self, kc, outv, index):
        mask = self.compare_mask[index]
        ans = self.ref_answer[index]

        # check for vectorized data order
        if outv.ndim == ans.ndim:
            # return the default, as it can handle it
            return kernel_call('', [], compare_mask=[mask],
                               compare_axis=self.compare_axis)._get_comparable(
                               outv, 0)

        if self.compare_axis != -1:
            _get_index = indexer(ans.ndim, outv.ndim, outv.shape, kc.current_order)
            # this is a list of indicies in dimensions to take
            # handle multi-dim combination of mask
            if not isinstance(mask, np.ndarray):
                size = np.prod([x.size for x in mask])
            else:
                # single dim, use simple mask
                size = mask.size

            # get the index arrays
            inds = np.array(_get_index(mask, self.compare_axis))
            stride_arr = np.array([np.unique(x).size for x in inds], dtype=np.int32)
            # get non-slice inds
            non_slice = np.array([i for i, x in enumerate(inds)
                                  if isinstance(x, np.ndarray)], dtype=np.int32)
            # create the output masker
            masking = np.array([slice(None)] * outv.ndim)
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

            return outv[tuple(masking)]

        raise NotImplementedError


def _get_eqs_and_oploop(owner, do_ratespec=False, do_ropsplit=False,
                        do_conp=True, langs=['opencl'], do_vector=True):

    platforms = get_test_platforms(do_vector=do_vector, langs=langs)
    eqs = {'conp': owner.store.conp_eqs,
           'conv': owner.store.conv_eqs}
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
    oploop += [('knl_type', ['map'])]
    out = None
    for p in platforms:
        val = OptionLoop(OrderedDict(p + oploop))
        if out is None:
            out = val
        else:
            out = out + val

    return eqs, out


def _generic_tester(owner, func, kernel_calls, rate_func, do_ratespec=False,
                    do_ropsplit=False, do_conp=False, do_vector=True,
                    langs=['opencl'], **kw_args):
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
    kwargs: dict
        Any additional arguements to pass to the :param:`func`
    """

    eqs, oploop = _get_eqs_and_oploop(owner, do_ratespec=do_ratespec,
                                      do_ropsplit=do_ropsplit, do_conp=do_conp)

    reacs = owner.store.reacs
    specs = owner.store.specs

    exceptions = ['device', 'conp']
    bad_platforms = set()

    for i, state in enumerate(oploop):
        if state['width'] is not None and state['depth'] is not None:
            continue

        # skip bad platforms
        if 'platform' in state and state['platform'] in bad_platforms:
            continue

        try:
            opt = loopy_options(**{x: state[x] for x in state
                                if x not in exceptions})
        except MissingPlatformError:
            # warn and skip future tests
            logging.warn('Platform {} not found'.format(
                state['platform']))
            bad_platforms.update([state['platform']])
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
        infos = func(eqs, opt, namestore,
                     test_size=owner.store.test_size, **kw_args)

        if not isinstance(infos, list):
            try:
                infos = list(infos)
            except:
                infos = [infos]

        # create a dummy kernel generator
        knl = k_gen.make_kernel_generator(
            name='spec_rates',
            loopy_opts=opt,
            kernels=infos,
            test_size=owner.store.test_size
        )

        knl._make_kernels()

        # create a list of answers to check
        try:
            for kc in kernel_calls:
                kc.set_state(knl.array_split, state['order'])
        except TypeError as e:
            if str(e) != "'kernel_call' object is not iterable":
                raise e
            kernel_calls.set_state(knl.array_split, state['order'])

        assert auto_run(knl.kernels, kernel_calls, device=opt.device),\
            'Evaluate {} rates failed'.format(func.__name__)
