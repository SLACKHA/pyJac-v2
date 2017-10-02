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
                                        auto_run, RateSpecialization, loopy_options)
from ...core.exceptions import MissingPlatformError
from ...kernel_utils import kernel_gen as k_gen
from ...core import array_creator as arc
from ...core.mech_auxiliary import write_aux
from .. import get_test_platforms
from ...pywrap import generate_wrapper
from ... import utils


from optionloop import OptionLoop
import numpy as np
import pyopencl as cl
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
                    if axes[i] == self.ref_ndim - 1), None)
        if axi is not None:
            # the first index is the remainder of the ind by the new dimension size
            # and the last index is the floor division of the new dim size

            # check that this is ind is not a slice
            # if it is we don't need to to anything
            if isinstance(inds[axi], np.ndarray):
                rv[-1], rv[0] = np.divmod(
                    inds[axi], self.out_shape[0], dtype=np.int32)

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
                rv[0], rv[-1] = np.divmod(
                    inds[axi], self.out_shape[-1], dtype=np.int32)

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
        self.out_shape = out_shape
        self.out_ndim = out_ndim
        self._indexer = self._get_F_index if order == 'F' else \
            self._get_C_index

    def __call__(self, inds, axes):
        return self._indexer(inds, axes)


def parse_split_index(arr, mask, order, ref_ndim=2, axis=(1,)):
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

    Returns
    -------
    index: tuple of int / slice
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

    if arr.ndim == ref_ndim:
        # no split
        masking = np.array([slice(None)] * arr.ndim)
        for i, x in enumerate(axis):
            masking[x] = mask[i]
        return tuple(masking)

    # get the index arrays
    inds = np.array(_get_index(mask, axis))
    stride_arr = np.array([np.unique(x).size for x in inds], dtype=np.int32)
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

    def __call__(self, kc, outv, index):
        mask = self.compare_mask[index]
        ans = self.ref_answer[index]

        # check for vectorized data order
        if outv.ndim == ans.ndim:
            # return the default, as it can handle it
            return kernel_call('', [], compare_mask=[mask],
                               compare_axis=self.compare_axis)._get_comparable(
                               outv, 0)

        _get_index = indexer(ans.ndim, outv.ndim, outv.shape, kc.current_order)
        if self.compare_axis != -1:
            # get the split indicies
            masking = parse_split_index(
                outv, mask, kc.current_order, ans.ndim, self.compare_axis)

        else:
            # we supplied a list of indicies, all we really have to do is convert
            # them and return

            # first check we have a reasonable mask
            assert ans.ndim == len(mask), "Can't use dissimilar compare masks / axes"
            # dummy comparison axis
            comp_axis = np.arange(ans.ndim)
            # convert inds
            masking = tuple(_get_index(mask, comp_axis))

        # and return
        return outv[masking]


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
            logging.warn('Platform {} not found'.format(state['platform']))
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


def _full_kernel_test(self, lang, kernel_gen, test_arr_name, test_arr,
                      btype, call_name, **oploop_kwds):
    eqs, oploop = _get_eqs_and_oploop(
            self, do_conp=True, do_vector=lang != 'c', langs=[lang],
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
            logging.warn('Platform {} not found'.format(state['platform']))
            bad_platforms.update([state['platform']])
            continue

        # check to see if device is CPU
        # if (opts.lang == 'opencl' and opts.device_type == cl.device_type.CPU) \
        #        and (opts.depth is None or not opts.use_atomics):
        #    opts.use_private_memory = True

        conp = state['conp']

        # generate kernel
        kgen = kernel_gen(eqs, self.store.reacs, self.store.specs, opts, conp=conp)

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
        __saver(test, test_arr_name, tests)

        # find where the reduced pressure term for non-Lindemann falloff / chemically
        # activated reactions is zero

        # get split arrays
        test, = kgen.array_split.split_numpy_arrays(test)

        # find where Pr is zero
        last_zeros = np.where(self.store.ref_Pr == 0)[0]

        # turn into updated form
        if kgen.array_split._have_split():
            ravel_ind = parse_split_index(test, (last_zeros,), opts.order,
                                          ref_ndim=3, axis=(0,))
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

        # fill other ravel locations with tiled test size
        stride = 1
        size = np.prod([test.shape[i] for i in range(test.ndim)
                       if i not in copy_inds])
        for i in [x for x in range(test.ndim) if x not in copy_inds]:
            repeats = int(np.ceil(size / (test.shape[i] * stride)))
            ravel_ind[i] = np.tile(np.arange(test.shape[i], dtype=np.int32),
                                   (repeats, stride)).flatten(order='F')[:size]
            stride *= test.shape[i]
        # and use multi_ravel to convert to linear for dphi
        # for whatever reason, if we have two ravel indicies with multiple values
        # we need to need to iterate and stitch them together
        looser_tols = np.empty((0,))
        for index in np.ndindex(tuple(x.shape[0] for x in ravel_ind[copy_inds])):
            # create copy w/ replaced index
            copy = ravel_ind.copy()
            copy[copy_inds] = [np.array(
                ravel_ind[copy_inds][i][x], dtype=np.int32)
                for i, x in enumerate(index)]
            # amd take union of the iterated ravels
            looser_tols = np.union1d(looser_tols, np.ravel_multi_index(
                copy, test.shape, order=opts.order))

        # and force to int for indexing
        looser_tols = np.array(looser_tols, dtype=np.int32)

        # number of devices is:
        #   number of threads for CPU
        #   1 for GPU
        num_devices = int(cpu_count() / 2)
        if lang == 'opencl' and opts.device_type == cl.device_type.GPU:
            num_devices = 1

        # and save the data.bin file in case of testing
        db = np.concatenate((
            np.expand_dims(phi[:, 0], axis=1),
            np.expand_dims(param, axis=1),
            phi[:, 1:]), axis=1)
        with open(os.path.join(lib_dir, 'data.bin'), 'wb') as file:
            db.flatten(order=opts.order).tofile(file,)

        # write the module tester
        from ...libgen import build_type
        with open(os.path.join(lib_dir, 'test.py'), 'w') as file:
            file.write(mod_test.safe_substitute(
                package='pyjac_{lang}'.format(
                    lang=package_lang[opts.lang]),
                input_args=', '.join('"{}"'.format(x) for x in args),
                test_arrays=', '.join('"{}"'.format(x) for x in tests),
                looser_tols='[{}]'.format(
                    ', '.join(str(x) for x in looser_tols)),
                rtol=1e-1 if btype != build_type.jacobian else 1e0,
                atol=1e-8,
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
        except subprocess.CalledProcessError as e:
            logging.debug(state)
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
                for ax, ind in six.iteritems(check_inds):
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
                    mask[ax] = ind
                array[tuple(mask)] = value

            self._get_compare = _get_compare
            self._set_at = _set_at
            return func(self, *args, **kwargs)
        return wrapped
    return check_inds_decorator
