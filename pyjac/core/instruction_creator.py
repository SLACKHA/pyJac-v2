# -*- coding: utf-8 -*-
"""Contains various utility classes for creating update instructions
(e.g., handling updating rates of progress w/ reverse ROP, pressure mod term,
etc.)
"""

from __future__ import division

import logging
import inspect
from string import Template


import loopy as lp
import numpy as np
from loopy.types import AtomicType
from .array_creator import var_name


class array_splitter(object):
    """
        A convenience object that handles splitting arrays to improve vectorized
        data-access patterns, etc.

        Can handle reshaping of both loopy and numpy arrays to the desired shape

        Properties
        ----------
        depth: int [None]
            If is not None, the vector-width to use for deep-vectorization
        wide: bool [False]
            If is not None, the vector-width to use for wide-vectorization
        data_order: ['C', 'F']
            The data ordering of the kernel
    """

    def __init__(self, loopy_opts):
        self.depth = loopy_opts.depth
        self.width = loopy_opts.width
        self.vector_width = self.depth if self.depth is not None else self.width
        self.data_order = loopy_opts.order

    def _have_split(self):
        """
        Returns True if there is anything for this :class:`array_splitter` to do
        """

        return self.vector_width is not None and ((
            self.data_order == 'C' and self.width) or (
            self.data_order == 'F' and self.depth))

    def _split_array_axis_inner(self, kernel, array_name, split_axis, dest_axis,
                                count, order='C'):
        if count == 1:
            return kernel

        # {{{ adjust arrays

        from loopy.kernel.tools import ArrayChanger
        from loopy.symbolic import SubstitutionRuleMappingContext
        from loopy.transform.padding import ArrayAxisSplitHelper

        achng = ArrayChanger(kernel, array_name)
        ary = achng.get()

        from pytools import div_ceil

        # {{{ adjust shape

        new_shape = ary.shape
        assert new_shape is not None, 'Cannot split auto-sized arrays'
        new_shape = list(new_shape)
        axis_len = new_shape[split_axis]
        outer_len = div_ceil(axis_len, count)
        new_shape[split_axis] = outer_len
        new_shape.insert(dest_axis, count)
        new_shape = tuple(new_shape)

        # }}}

        # {{{ adjust dim tags

        if ary.dim_tags is None:
            raise RuntimeError("dim_tags of '%s' are not known" % array_name)
        new_dim_tags = list(ary.dim_tags)

        old_dim_tag = ary.dim_tags[split_axis]

        from loopy.kernel.array import FixedStrideArrayDimTag
        if not isinstance(old_dim_tag, FixedStrideArrayDimTag):
            raise RuntimeError("axis %d of '%s' is not tagged fixed-stride".format(
                split_axis, array_name))

        new_dim_tags.insert(dest_axis, FixedStrideArrayDimTag(1))
        # fix strides
        toiter = reversed(list(enumerate(new_shape))) if order == 'C' \
            else enumerate(new_shape)

        stride = 1
        for i, shape in toiter:
            new_dim_tags[i] = new_dim_tags[i].copy(stride=stride)
            stride *= shape

        new_dim_tags = tuple(new_dim_tags)

        # }}}

        # {{{ adjust dim_names

        new_dim_names = ary.dim_names
        if new_dim_names is not None:
            new_dim_names = list(new_dim_names)
            existing_name = new_dim_names[split_axis]
            outer_name = existing_name + "_outer"
            new_dim_names[split_axis] = outer_name
            new_dim_names.insert(dest_axis, existing_name + "_inner")
            new_dim_names = tuple(new_dim_names)

        # }}}

        kernel = achng.with_changed_array(ary.copy(
            shape=new_shape, dim_tags=new_dim_tags, dim_names=new_dim_names))

        # }}}

        var_name_gen = kernel.get_var_name_generator()

        def split_access_axis(expr):
            idx = expr.index
            if not isinstance(idx, tuple):
                idx = (idx,)
            idx = list(idx)

            axis_idx = idx[split_axis]

            from loopy.symbolic import simplify_using_aff
            inner_index = simplify_using_aff(kernel, axis_idx % count)
            outer_index = simplify_using_aff(kernel, axis_idx // count)
            idx[split_axis] = outer_index
            idx.insert(dest_axis, inner_index)
            return expr.aggregate.index(tuple(idx))

        rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, var_name_gen)
        aash = ArrayAxisSplitHelper(rule_mapping_context,
                                    set([array_name]), split_access_axis)
        kernel = rule_mapping_context.finish_kernel(aash.map_kernel(kernel))

        return kernel

    def _split_loopy_arrays(self, kernel):
        """
        Splits the :class:`loopy.GlobalArg`'s that form the given kernel's arguements
        to conform to this split pattern

        Parameters
        ----------
        kernel : `loopy.LoopKernel`
            The kernel to apply the splits to

        Returns
        -------
        split_kernel : `loopy.LoopKernel`
            The kernel with the array splittings applied
        """

        if not self._have_split():
            return kernel

        for array_name in [x.name for x in kernel.args
                           if isinstance(x, lp.GlobalArg)]:
            if self.data_order == 'C' and self.width:
                split_axis = 0
                dest_axis = len(x.shape)
            else:
                split_axis = len(x.shape) - 1
                dest_axis = 0

            kernel = self._split_array_axis_inner(
                kernel, array_name, split_axis, dest_axis,
                self.vector_width, self.data_order)

        return kernel

    def _split_numpy_array(self, input_array):
        """
        Spits the supplied numpy array according to desired pattern

        Parameters
        ----------
        input_array : :class:`numpy.ndarray`
            The input array to split

        Returns
        -------
        output : :class:`numpy.ndarray`
            The properly split / resized numpy array
        """

        if not self._have_split():
            return input_array

        def _split_and_pad(arr, axis, width, ax_trans):
            # get the last split as the ceiling
            end = np.ceil(arr.shape[axis] / width) * width
            # create split indicies
            indicies = np.arange(width, end + 1, width, dtype=np.int32)
            # split array
            arr = np.split(arr, indicies, axis=axis)
            # filter out empties
            arr = [a for a in arr if a.size]
            # check for pad
            if arr[-1].shape[axis] != width:
                pads = [(0, 0) for x in arr[-1].shape]
                pads[axis] = (0, width - arr[-1].shape[axis])
                arr[-1] = np.pad(arr[-1], pads, 'constant')
            # get joined
            arr = np.stack(arr, axis=axis)
            # and move array dims
            return np.moveaxis(arr, *ax_trans).copy(order=self.data_order)

        # figure out split
        dim = len(input_array.shape) - 1
        if self.data_order == 'C' and self.width:
            return _split_and_pad(input_array, 0, self.width, (dim, -1))
        elif self.data_order == 'F' and self.depth:
            return _split_and_pad(input_array, dim, self.depth, (-1, 0))

    def split_numpy_arrays(self, arrays):
        """
        Splits the provided numpy arrays

        See :func:`_split_numpy_array`

        Parameters
        ----------
        arrays: list of :class:`numpy.ndarray`
            The arrays to split

        Returns
        -------
        out_arrays: list of :class:`numpy.ndarray`
            The split arrays
        """

        if isinstance(arrays, np.ndarray):
            arrays = [arrays]

        return [self._split_numpy_array(a) for a in arrays]


def use_atomics(loopy_opts):
    """ Convenience method to detect whether atomics will be used or not.
        Useful in that we need to apply atomic modifiers to some instructions,
        but _not_ the sequential specializer

    Parameters
    ----------
    loopy_opts: :class:`loopy_utils.loopy_opts`
        The loopy options used to create this kernel.

    Returns
    -------
    use_atomics : [bool]
        Whether an atomic specializer would be returned by
        :meth:`get_deep_specializer`
    """

    return loopy_opts.depth and loopy_opts.use_atomics


def get_deep_specializer(loopy_opts, atomic_ids=[], split_ids=[], init_ids=[],
                         force_sequential=False):
    """
    Returns a deep specializer to enable deep vectorization using either
    atomic updates or a sequential (single-lane/thread) "dummy" deep
    vectorizor (for implementations w/o 64-bit atomic instructions, e.g.
    intel's opencl)

    Parameters
    ----------
    loopy_opts: :class:`loopy_utils.loopy_opts`
        The loopy options used to create this kernel.  Determines the type
        of deep specializer to return
    atomic_ids: list of str
        A list of instruction id-names that require atomic updates for
        deep vectorization
    split_ids: list of str
        For instructions where (e.g.) a sum starts with a constant term,
        the easiest way to handle it is to split it over all the threads /
        lanes and have them contribute equally to the final sum.
        These instructions ids should be passed as split_ids
    init_ids: list of str
        List of instructions that initialize atomic variables

    Returns
    -------
    can_vectorize: bool [True]
        Whether the resulting kernel can be properly vectorized or not
    vectorization_specializer: :class:`Callable`(:class:`loopy.LoopKernel`)
        A function that takes as an argument the constructed kernel, and
        modifies it so as to enable the required vectorization
    """

    if not loopy_opts.depth:
        # no need to do anything
        return True, None

    if loopy_opts.use_atomics and not force_sequential:
        return True, atomic_deep_specialization(
            loopy_opts.depth, atomic_ids=atomic_ids,
            split_ids=split_ids, init_ids=init_ids)
    else:
        return False, dummy_deep_specialization()


class within_inames_specializer(object):
    """
    A simple class designed to ensure all kernels are vectorizable
    by putting instructions that do not use the local hardware axis inside the
    correct loop.

    This should _not_ be used for anything but deep-vectorizations
    """
    def __init__(self, var_name=var_name):
        self.var_name = var_name

    def __call__(self, knl):
        # get resulting tags
        in_tag = '{}_inner'.format(self.var_name)
        for insn in knl.instructions:
            if not insn.within_inames & frozenset([in_tag]):
                # add a fake dependency on the vectorized iname
                insn.within_inames |= frozenset([in_tag])

        return knl.copy(instructions=knl.instructions[:])


class atomic_deep_specialization(within_inames_specializer):
    """
    A class that turns write race instructions to atomics to enable deep
    vectorization

    vec_width: int
        The vector width to split over
    atomic_ids : list of str
        A list of instruction id-names that require atomic updates for
        deep vectorization
    split_ids : list of str
        For instructions where (e.g.) a sum starts with a constant term,
        the easiest way to handle it is to split it over all the threads /
        lanes and have them contribute equally to the final sum.
        These instructions ids should be passed as split_ids
    init_ids: list of str
        Lit of instruction id-names that require atomic inits for deep vectorization
    """

    def __init__(self, vec_width, atomic_ids=[], split_ids=[], init_ids=[]):
        def _listify(x):
            if not isinstance(x, list):
                return [x]
            return x
        # set parameters
        self.vec_width = vec_width
        self.atomic_ids = _listify(atomic_ids)[:]
        self.split_ids = _listify(split_ids)[:]
        self.init_ids = _listify(init_ids)[:]

        # and parent constructor
        super(atomic_deep_specialization, self).__init__()

    def __call__(self, knl):
        insns = knl.instructions[:]
        data = knl.args[:]

        # we generally need to infer the dtypes for temporary variables at this stage
        # in case any of them are atomic
        from loopy.type_inference import infer_unknown_types
        from loopy.types import to_loopy_type
        from pymbolic.primitives import Sum
        knl = infer_unknown_types(knl, expect_completion=True)
        temps = knl.temporary_variables.copy()

        def _check_atomic_data(insn):
            # get the kernel arg written by this insn
            written = insn.assignee_var_names()[0]
            ind = next(
                (i for i, d in enumerate(data) if d.name == written), None)
            # make sure the dtype is atomic, if not update it
            if ind is not None and not isinstance(data[ind].dtype, AtomicType):
                data[ind] = data[ind].copy(for_atomic=True)
            elif ind is None:
                assert written in temps, (
                    'Cannot find written atomic variable: {}'.format(written))
                if not isinstance(temps[written].dtype, AtomicType):
                    temps[written] = temps[written].copy(dtype=to_loopy_type(
                        temps[written].dtype, for_atomic=True))
            return written

        for insn_ind, insn in enumerate(insns):
            if insn.id in self.atomic_ids:
                written = _check_atomic_data(insn)
                # and force the insn to an atomic update
                insns[insn_ind] = insn.copy(
                    atomicity=(lp.AtomicUpdate(written),))
            elif insn.id in self.init_ids:
                written = _check_atomic_data(insn)
                # setup an atomic init
                insns[insn_ind] = insn.copy(
                    atomicity=(lp.AtomicInit(written),))
            elif insn.id in self.split_ids:
                if isinstance(insn.expression, Sum) and \
                        insn.assignee in insn.expression.children:
                    written = _check_atomic_data(insn)
                    # get children that are not the assignee and re-sum
                    others = Sum(tuple(
                        x for x in insn.expression.children if x != insn.assignee))
                    # finally implement the split as a += sum(others) / vec_width
                    insns[insn_ind] = insn.copy(
                        expression=insn.assignee + others / self.vec_width,
                        atomicity=(lp.AtomicUpdate(written),))
                else:
                    # otherwise can simply divide
                    insns[insn_ind] = insn.copy(
                        expression=insn.expression / self.vec_width)

        # now force all instructions into inner loop
        return super(atomic_deep_specialization, self).__call__(
            knl.copy(instructions=insns, args=data, temporary_variables=temps))


class dummy_deep_specialization(within_inames_specializer):
    """
    A reusable-class to enable serialized deep vectorizations (i.e. reductions
    on a single OpenCL lane)
    """

    def __call__(self, knl):
        # do a dummy split
        knl = lp.split_iname(knl, self.var_name, 1, inner_tag='l.0')
        # now call the base to force all instructions inside the inner loop
        return super(dummy_deep_specialization, self).__call__(knl)


def default_pre_instructs(result_name, var_str, INSN_KEY):
    """
    Simple helper method to return a number of precomputes based off the passed
    instruction key

    Parameters
    ----------
    result_name : str
        The loopy temporary variable name to store in
    var_str : str
        The stringified representation of the variable to construct
    key : ['INV', 'LOG', 'VAL']
        The transform / value to precompute

    Returns
    -------
    precompute : str
        A loopy instruction in the form:
            '<>result_name = fn(var_str)'
    """
    default_preinstructs = {'INV': '1 / {}'.format(var_str),
                            'LOG': 'log({})'.format(var_str),
                            'VAL': '{}'.format(var_str),
                            'LOG10': 'log10({})'.format(var_str)}
    return Template("<>${result} = ${value}").safe_substitute(
        result=result_name,
        value=default_preinstructs[INSN_KEY])


def get_update_instruction(mapstore, mask_arr, base_update_insn):
    """
    Handles updating a value by a possibly specified (masked value),
    for example this can be used to generate an instruction (or set thereof)
    to update the net rate of progress by the reverse or pressure modification
    term

    Parameters
    ----------
    mapstore: :class:`array_creator.MapStore`
        The base mapstore used in creation of this kernel
    mask_arr: :class:`array_creator.creator`
        The array to use as a mask to determine whether the :param:`base_value`
        should be updated
    base_update_insn: str
        The update instruction to use as a base.  This may be surrounded by
        and if statement or possibly discarded altogether depending on the
        :param:`mask_arr` and :param:`mapstore`

    Returns
    -------
    update_insn: str
        The update instruction string(s), possibly empty if no update is
        required
    """

    if not mapstore.is_finalized:
        _, _, line_number, function_name, _, _ = inspect.stack()[1]
        logging.warn('Call to get_update_instruction() from {0}:{1}'
                     ' used non-finalized mapstore, finalizing now...'.format(
                         function_name, line_number))

        mapstore.finalize()

    # empty mask
    if not mask_arr:
        return ''

    # ensure mask array in domains
    assert mask_arr in mapstore.domain_to_nodes, (
        'Cannot create update instruction - mask array '
        ' {} not in mapstore domains'.format(
            mask_arr.name))

    # check to see if there are any empty mask entries
    mask = mapstore.domain_to_nodes[mask_arr]
    mask = mask if mask.parent in mapstore.transformed_domains \
        else None
    if mask:
        mask_iname = mask.iname
        # if so, return a guarded update insn
        return Template(
            """
    if ${mask_iname} >= 0
        ${base_update_insn}
    end
    """).safe_substitute(**locals())

    # else return the base update insn
    return base_update_insn
