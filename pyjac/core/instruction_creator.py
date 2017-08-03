# -*- coding: utf-8 -*-
"""Contains various utility classes for creating update instructions
(e.g., handling updating rates of progress w/ reverse ROP, pressure mod term,
etc.)
"""

import logging
import inspect
from string import Template
import loopy as lp


class atomic_deep_specialzation(object):

    """
    A reusable-class to enable serialized deep vectorizations (i.e. reductions
    on a single OpenCL lane)
    """

    def __init__(self, atomic_ids=[], ):
        pass

    def __call__(self, knl):
        # do a dummy split
        knl = lp.split_iname(knl, 'i', 1, inner_tag='l.0')
        for insn in knl.instructions:
            if not insn.within_inames & frozenset(['i_inner', 'i_outer']):
                # add a fake dependency on the split iname
                insn.within_inames |= frozenset(['i_inner'])

        return knl.copy(instructions=knl.instructions[:])


class dummy_deep_specialzation(object):

    """
    A reusable-class to enable serialized deep vectorizations (i.e. reductions
    on a single OpenCL lane)
    """

    def __init__(self):
        pass

    def __call__(self, knl):
        # do a dummy split
        knl = lp.split_iname(knl, 'i', 1, inner_tag='l.0')
        for insn in knl.instructions:
            if not insn.within_inames & frozenset(['i_inner', 'i_outer']):
                # add a fake dependency on the split iname
                insn.within_inames |= frozenset(['i_inner'])

        return knl.copy(instructions=knl.instructions[:])


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
