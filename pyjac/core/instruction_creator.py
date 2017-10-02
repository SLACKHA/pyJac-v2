# -*- coding: utf-8 -*-
"""Contains various utility classes for creating update instructions
(e.g., handling updating rates of progress w/ reverse ROP, pressure mod term,
etc.)
"""

from __future__ import division

import logging
import inspect
from string import Template

import six
import loopy as lp
import numpy as np
from loopy.types import AtomicType
from .array_creator import var_name


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
        return False, dummy_deep_specialization(
            write_races=atomic_ids + split_ids + init_ids)


class write_race_silencer(object):
    """
    Turns off warnings for loopy's detection of writing across a vectorized-iname
    as we handle his with either the :class:`dummy_deep_specialization` or
    the :class:`atomic_deep_specialization`
    """

    def __init__(self, write_races=[]):
        self.write_races = ['write_race({name})'.format(name=x) for x in write_races]

    def __call__(self, knl):
        return knl.copy(silenced_warnings=knl.silenced_warnings + self.write_races)


class within_inames_specializer(write_race_silencer):
    """
    A simple class designed to ensure all kernels are vectorizable
    by putting instructions that do not use the local hardware axis inside the
    correct loop.

    This should _not_ be used for anything but deep-vectorizations
    """
    def __init__(self, var_name=var_name, **kwargs):
        self.var_name = var_name
        super(within_inames_specializer, self).__init__(**kwargs)

    def __call__(self, knl):
        # get resulting tags
        in_tag = '{}_inner'.format(self.var_name)
        for insn in knl.instructions:
            if not insn.within_inames & frozenset([in_tag]):
                # add a fake dependency on the vectorized iname
                insn.within_inames |= frozenset([in_tag])

        return super(within_inames_specializer, self).__call__(
            knl.copy(instructions=knl.instructions[:]))


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
        super(atomic_deep_specialization, self).__init__(
            write_races=atomic_ids + split_ids + init_ids)

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
