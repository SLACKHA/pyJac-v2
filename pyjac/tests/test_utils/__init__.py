import os
from string import Template
from ...loopy_utils.loopy_utils import get_device_list, kernel_call, populate
from ...kernel_utils import kernel_gen as k_gen
import numpy as np
import six
from collections import defaultdict


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

    def _get_F_index(arr, i, ax):
        """
        for the 'F' order split, the last axis is split and moved to the beginning.
        """
        dim = arr.ndim
        if ax != dim - 1:
            # there is no change in the actual indexing here
            # however, the destination index will be increased by one
            # to account for the new inserted index at the front of the array
            return (i,), (ax + 1,)
        # here, we must change the indicies
        # the first index is the remainder of the ind by the new dimension size
        # and the last index is the floor division of the new dim size
        (i % arr.shape[0], i // arr.shape[0]), (0, dim - 1)

    def _get_C_index(arr, i, ax):
        """
        for the 'C' order split, the first axis is split and moved to the end.
        """
        dim = arr.ndim
        if ax != 0:
            # there is no change in the actual indexing here
            return (i,), (ax,)
        # here, we must change the indicies
        # and first index is the floor division of the new dim size
        # the last index is the remainder of the ind by the new dimension size
        (i // arr.shape[-1], i % arr.shape[0]), (0, dim - 1)

    def __init__(self, order='C'):
        self._indexer = indexer._get_F_index if order == 'F' else \
            indexer._get_C_index

    def __call__(self, arr, i, ax):
        return self._indexer(arr, i, ax)


def parse_split_index(arr, ind, order, axis=1):
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
    axis: int or list of int
        The axes the ind's correspond to

    Returns
    -------
    index: tuple of int / slice
        A proper indexing for the split array
    """

    dim = arr.ndim
    index = [slice(None)] * dim

    _get_index = indexer(order)

    try:
        for i, ax in enumerate(zip(ind, axis)):
            set_inds, set_axs = _get_index(arr, i, ax)
            for i, ax in zip(set_inds, set_axs):
                index[ax] = i
    except TypeError:
        inds, axs = _get_index(arr, ind, axis)
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

        assert all(len(x) == len(self.compare_axis) for x in self.compare_mask), (
            "Can't use dissimilar compare masks / axes")

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
            _get_index = indexer(kc.current_order)
            # this is a list of indicies in dimensions to take
            try:
                # try multi-dim
                enumerate(self.compare_axis)
                # get max size
                for i, ax in enumerate(self.compare_axis):
                    ind_list = defaultdict(lambda: list())
                    # get comparable index
                    for ind in mask[i]:
                        for indi, axi in zip(*_get_index(outv, ind, ax)):
                            # and update take inds
                            ind_list[axi].append(indi)
                    # finally turn into iter
                    ind_list = six.iteritems(ind_list)
                    for axi, inds in ind_list:
                        outv = np.take(outv, inds, axis=axi)
            except TypeError:
                ind_list = defaultdict(lambda: list())
                # otherwise, this is a single dim
                for ind in mask:
                    for indi, axi in zip(*_get_index(outv, ind, self.compare_axis)):
                        # and update take inds
                        ind_list[axi].append(indi)

                # and turn into "take" params
                ind_list = six.iteritems(ind_list)
                for axi, inds in ind_list:
                    outv = np.take(outv, inds, axis=axi)

            return outv

        return NotImplementedError
