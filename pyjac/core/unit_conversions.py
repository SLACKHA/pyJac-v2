# -*- coding: utf-8 -*-
"""Module for converting various input formats into pyJac's internal format"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
from string import Template

# Non-standard librarys
import numpy as np

# Local imports
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.core import array_creator as arc
from pyjac.core import instruction_creator as ic
from pyjac.core.array_creator import (global_ind, var_name, default_inds)


def mass_to_mole_factions(loopy_opts, namestore, conp=True, test_size=None):
    """Converts input state vector from mass fractions to mole fractions and state
       variables depending on constant pressure vs constant volue assumption

    Parameters
    ----------
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    conp : bool
        If true, generate equations using constant pressure assumption
        If false, use constant volume equations
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Notes
    -----
    Assumes that this is being called at input only!
    This allows us to make the (generally) unsafe assumption that the mole factions
    are _equivalent_ to the moles, as the total number of moles will adjust to
    satisfy the ideal gas relation.


    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    # first kernel, determine molecular weight
    mapstore = arc.MapStore(loopy_opts, namestore.num_specs_no_ns, test_size)

    # first, create all arrays
    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(loopy_opts, test_size))

    # need "Yi" and molecular weight / factor arrays

    # add / apply maps
    mapstore.check_and_add_transform(namestore.n_arr,
                                     namestore.phi_spec_inds,
                                     force_inline=True)
    mapstore.check_and_add_transform(namestore.mw_post_arr,
                                     namestore.num_specs_no_ns,
                                     force_inline=True)

    Yi_arr, Yi_str = mapstore.apply_maps(namestore.n_arr, *default_inds)
    mw_inv_arr, mw_inv_str = mapstore.apply_maps(namestore.mw_inv, var_name)
    mw_work_arr, mw_work_str = mapstore.apply_maps(namestore.mw_work, global_ind)

    # add arrays
    kernel_data.extend([Yi_arr, mw_inv_arr, mw_work_arr])

    # initialize molecular weight
    pre_instructions = Template(
        """
            ${mw_work_str} = W_ns_inv {id=init}
            <> work = 0 {id=init_work}
        """
    ).safe_substitute(**locals())

    instructions = Template(
        """
            work = work + (${mw_inv_str} - W_ns_inv) * ${Yi_str} \
                {id=update, dep=init*}
        """).safe_substitute(**locals())

    barrier = ic.get_barrier(loopy_opts, local_memory=False,
                             id='break', dep='update')
    post_instructions = Template(
        """
        ${barrier}
        ${mw_work_str} = (${mw_work_str} + work) {id=final, dep=break, nosync=init}
        """).substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['final'], init_ids=['init'])

    mw_kernel = k_gen.knl_info(name='molecular_weight_inverse',
                               pre_instructions=[pre_instructions],
                               instructions=instructions,
                               post_instructions=[post_instructions],
                               mapstore=mapstore,
                               var_name=var_name,
                               kernel_data=kernel_data,
                               can_vectorize=can_vectorize,
                               vectorization_specializer=vec_spec,
                               parameters={'W_ns_inv': 1. / np.float64(
                                                namestore.mw_arr[-1])},
                               silenced_warnings=['write_race(final)',
                                                  'write_race(init)'])

    # now convert to moles
    mapstore = arc.MapStore(loopy_opts, namestore.num_specs_no_ns, test_size)

    # first, create all arrays
    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(loopy_opts, test_size))

    # need input "Yi", molecular weight, and moles array

    # add / apply maps
    mapstore.check_and_add_transform(namestore.n_arr,
                                     namestore.phi_spec_inds,
                                     force_inline=True)

    n_arr, n_str = mapstore.apply_maps(namestore.n_arr, *default_inds)
    mw_work_arr, mw_work_str = mapstore.apply_maps(namestore.mw_work, global_ind)
    mw_inv_arr, mw_inv_str = mapstore.apply_maps(namestore.mw_inv, var_name)

    # add arrays
    kernel_data.extend([n_arr, mw_inv_arr, mw_work_arr])

    pre_instructions = Template(
        '<> mw = 1 / ${mw_work_str} {id=init}').safe_substitute(**locals())
    instructions = Template(
        """
            ${n_str} = ${n_str} * ${mw_inv_str} * mw {dep=init}
        """).safe_substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(loopy_opts)
    mf_kernel = k_gen.knl_info(name='mole_fraction',
                               pre_instructions=[pre_instructions],
                               instructions=instructions,
                               mapstore=mapstore,
                               var_name=var_name,
                               kernel_data=kernel_data,
                               can_vectorize=can_vectorize,
                               vectorization_specializer=vec_spec)

    return [mw_kernel, mf_kernel]
