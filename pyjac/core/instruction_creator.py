# -*- coding: utf-8 -*-
"""Contains various utility classes for creating update instructions
(e.g., handling updating rates of progress w/ reverse ROP, pressure mod term,
etc.)
"""

import logging
import inspect
from string import Template
import loopy as lp
from loopy.types import AtomicType
from .array_creator import var_name


def get_deep_specializer(loopy_opts, atomic_ids=[], split_ids=[]):
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
    atomic_ids : list of str
        A list of instruction id-names that require atomic updates for
        deep vectorization
    split_ids : list of str
        For instructions where (e.g.) a sum starts with a constant term,
        the easiest way to handle it is to split it over all the threads /
        lanes and have them contribute equally to the final sum.
        These instructions ids should be passed as split_ids

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

    if loopy_opts.use_atomics:
        return True, atomic_deep_specialization(
            atomic_ids=atomic_ids, split_ids=split_ids)
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

    atomic_ids : list of str
        A list of instruction id-names that require atomic updates for
        deep vectorization
    split_ids : list of str
        For instructions where (e.g.) a sum starts with a constant term,
        the easiest way to handle it is to split it over all the threads /
        lanes and have them contribute equally to the final sum.
        These instructions ids should be passed as split_ids
    """

    def __init__(self, atomic_ids=[], split_ids=[]):
        def _listify(x):
            if not isinstance(x, list):
                return [x]
            return x
        # set parameters
        self.atomic_ids = _listify(atomic_ids)[:]
        self.split_ids = _listify(split_ids)[:]

        # and parent constructor
        super(atomic_deep_specialization, self).__init__()

    def __call__(self, knl):
        insns = knl.instructions[:]
        data = knl.args[:]
        for insn_ind, insn in enumerate(insns):
            if insn.id in self.atomic_ids:
                # get the kernel arg written by this insn
                written = insn.assignee_var_names()[0]
                ind = next(
                    i for i, d in enumerate(data) if d.name == written)
                # make sure the dtype is atomic, if not update it
                if not isinstance(data[ind].dtype, AtomicType):
                    data[ind] = data[ind].copy(for_atomic=True)
                # and force the insn to an atomic update
                insns[insn_ind] = insn.copy(
                    atomicity=(lp.AtomicUpdate(written),))

        # now force all instructions into inner loop
        return super(atomic_deep_specialization, self).__call__(
            knl.copy(instructions=insns, args=data))


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
