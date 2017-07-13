#! /usr/bin/env python
"""Creates source code for calculating analytical Jacobian matrix.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
import os
from math import log
from string import Template

# Local imports
from .. import utils
from . import mech_interpret as mech
from . import rate_subs as rate
from . import mech_auxiliary as aux
from ..sympy_utils import sympy_interpreter as sp_interp
from ..loopy_utils import loopy_utils as lp_utils
from ..loopy_utils import preambles_and_manglers as lp_pregen
from . import array_creator as arc
from ..kernel_utils import kernel_gen as k_gen
from . import reaction_types as rtypes
from . import chem_model as chem

# external packages

global_ind = 'j'
"""str: The global initial condition index

This is the string index for the global condition loop in generated kernels
of :module:`rate_subs`
"""


var_name = 'i'
"""str: The inner loop index

This is the string index for the inner loops in generated kernels of
:module:`rate_subs`
"""


default_inds = (global_ind, var_name)
"""str: The default indicies used in main loops of :module:`rate_subs`

This is the string indicies for the main loops for generated kernels in
:module:`rate_subs`
"""


def thermo_temperature_derivative(nicename, eqs, loopy_opts, namestore,
                                  test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the concentration weighted specific energy sum.

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dci_dnj`


    Parameters
    ----------
    nicename : ['dcp', 'dcv', 'db']
        The polynomial derivative to calculate
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    conp : bool [True]
        If supplied, True for constant pressure jacobian. False for constant
        volume [Default: True]

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    eq = eqs['conp'] if nicename in ['dcp'] else eqs['conv']
    return rate.polyfit_kernel_gen(
        nicename, eq, loopy_opts, namestore, test_size)


def dEdot_dnj(eqs, loopy_opts, namestore, test_size=None,
              conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the extra variable (i.e. V or P depending on conp/conv)
    w.r.t. the molar variables


    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    conp : bool [True]
        If supplied, True for constant pressure jacobian. False for constant
        volume [Default: True]

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # create arrays
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    num_specs = namestore.num_specs.initializer[-1] + 1
    # k loop
    spec_k = 'spec_k'
    extra_inames = [(spec_k, '0 <= spec_k < {}'.format(
        num_specs - 1))]

    mw_lp, mw_str = mapstore.apply_maps(
        namestore.mw_post_arr, spec_k)
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)
    jac_lp, dnk_dnj_str = mapstore.apply_maps(
        namestore.jac, global_ind, spec_k, var_name, affine={
            var_name: 2,
            spec_k: 2
        })
    _, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, 1, var_name, affine={
            var_name: 2,
        })
    _, dTdot_dnj_str = mapstore.apply_maps(
        namestore.jac, global_ind, 0, var_name, affine={
            var_name: 2,
        })

    P_lp, P_str = mapstore.apply_maps(
            namestore.P_arr, global_ind)
    T_lp, T_str = mapstore.apply_maps(
            namestore.T_arr, global_ind)

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    kernel_data.extend([mw_lp, V_lp, P_lp, T_lp, jac_lp])

    extra_var_str = V_str if conp else P_str
    fixed_var_str = P_str if conp else V_str
    instructions = Template("""
    <> sum = 0 {id=init}
    for ${spec_k}
        sum = sum + (1 - ${mw_str}) * ${dnk_dnj_str} {id=sum, dep=*}
    end

    ${jac_str} = ${T_str} * Ru * sum / ${fixed_var_str} + ${extra_var_str} * ${dTdot_dnj_str} / ${T_str} {dep=sum, nosync=sum}
    """).safe_substitute(**locals())

    return k_gen.knl_info(name='d{}dot_dnj'.format('P' if conp else 'V'),
                          extra_inames=extra_inames,
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          parameters={'Ru': chem.RU}
                          )


def dTdot_dnj(eqs, loopy_opts, namestore, test_size=None,
              conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the partial derivatives of dT/dt with respect to the molar species
    quanities

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    conp : bool [True]
        If supplied, True for constant pressure jacobian. False for constant
        volume [Default: True]

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # create arrays
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    num_specs = namestore.num_specs.initializer[-1] + 1
    # k loop
    spec_k = 'spec_k'
    extra_inames = [(spec_k, '0 <= spec_k < {}'.format(
        num_specs - 1))]

    spec_heat_lp, spec_heat_k_str = mapstore.apply_maps(
        namestore.spec_heat, *default_inds)
    _, spec_heat_ns_str = mapstore.apply_maps(
        namestore.spec_heat_ns, global_ind)
    energy_lp, energy_k_str = mapstore.apply_maps(
        namestore.spec_energy, global_ind, spec_k)
    _, energy_ns_str = mapstore.apply_maps(
        namestore.spec_energy_ns, global_ind)
    spec_heat_tot_lp, spec_heat_total_str = mapstore.apply_maps(
        namestore.spec_heat_total, global_ind)
    mw_lp, mw_str = mapstore.apply_maps(
        namestore.mw_post_arr, spec_k)
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)
    T_dot_lp, T_dot_str = mapstore.apply_maps(
        namestore.T_dot, global_ind)
    jac_lp, jac_spec_str = mapstore.apply_maps(
        namestore.jac, global_ind, spec_k, var_name, affine={
            var_name: 2,
            spec_k: 2
        })
    _, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, '0', var_name, affine={
            var_name: 2,
        })

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    kernel_data.extend([spec_heat_lp, energy_lp, spec_heat_tot_lp, mw_lp,
                        V_lp, T_dot_lp, jac_lp])

    instructions = Template("""
    <> sum = 0 {id=init}
    for ${spec_k}
        sum = sum + (${energy_k_str} - ${energy_ns_str} * ${mw_str}) * ${jac_spec_str} {id=sum, dep=*}
    end

    ${jac_str} = -(sum + ${T_dot_str} * (${spec_heat_k_str} - ${spec_heat_ns_str})) / (${V_str} * ${spec_heat_total_str}) {dep=sum, nosync=sum}
    """).safe_substitute(**locals())

    return k_gen.knl_info(name='dTdot_dnj',
                          extra_inames=extra_inames,
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore
                          )


def total_specific_energy(eqs, loopy_opts, namestore, test_size=None,
                          conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the concentration weighted specific energy sum.

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    conp : bool [True]
        If supplied, True for constant pressure jacobian. False for constant
        volume [Default: True]

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # create arrays
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs,
                            namestore.num_specs)

    spec_heat_lp, spec_heat_str = mapstore.apply_maps(
        namestore.spec_heat, *default_inds)
    conc_lp, conc_str = mapstore.apply_maps(
        namestore.conc_arr, *default_inds)
    spec_heat_tot_lp, spec_heat_total_str = mapstore.apply_maps(
        namestore.spec_heat_total, global_ind)

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    kernel_data.extend([spec_heat_lp, conc_lp, spec_heat_tot_lp])

    pre_instructions = Template('${spec_heat_total_str} = 0').safe_substitute(
        spec_heat_total_str=spec_heat_total_str)
    instructions = Template("""
        ${spec_heat_total_str} = ${spec_heat_total_str} + ${spec_heat_str} * ${conc_str}
    """).safe_substitute(
        spec_heat_total_str=spec_heat_total_str,
        spec_heat_str=spec_heat_str,
        conc_str=conc_str)

    return k_gen.knl_info(name='{}_total'.format(namestore.spec_heat.name),
                          pre_instructions=[pre_instructions],
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore
                          )


def __dci_dnj(loopy_opts, namestore,
              do_ns=False, fall_type=rtypes.falloff_form.none):
    """Generates instructions, kernel arguements, and data for calculating
    derivatives of the third body concentrations / falloff blending factors
    with respect to the molar quantity of a species

    Notes
    -----
    This is method is split into two kernels, the first handles derivatives of
    reactions with respect to the third body species in the reaction,
    that is if reaction `i` contains species `k, k+1...k+n`, and third body
    species `j, j+1... j+m`, it will consider the derivative of
    species `k` with respect to `j, j+1...j+m`, and so on with `k+1`,etc.

    The second kernel handles reactions where the last species in the mechanism
    has a non-default (non-unity) third body efficiency.
    In this case, using the strict formulation results in non-zero derivatives
    for _all_ species in the mechanism due to the conservation of mass
    formulation of the last species. That is, it will compute the derivative of
    `k` w.r.t species `1, 2, ... Ns - 1` where Ns is the last species in the
    mechanism.

    This second kernel may be turned off to increase sparsity (at the expense)
    of an incorrect jacobian.  This is often desired for implicit integrators
    which often only need an approximation to the Jacobian anyways.

    Additionally, we use a trick here to simplify the these methods for the
    strict formulation.  Namely, we _assume_ that the third body efficiency of
    the last species is unity in the first kernel.  In the second kernel, we
    then update the derivative of _all_ species assuming the third body species
    efficiency of a species `j` is unity.  By doing so, we obtain the correct
    multiplier ($\alpha_{i,j} - \alpha_{i, N_s}$) in all cases, and we save
    ourselves a fair bit of complicated indexing / looping.


    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        Contains keys 'fwd', 'rev' and 'net', with booleans corresponding to
        whether all nu values for that direction are integers.
        If True, powers of concentrations will be evaluated using
        multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    if fall_type == rtypes.falloff_form.none:
        our_inds = namestore.num_thd_only if not do_ns \
            else namestore.thd_only_ns_inds
        rxn_range = namestore.thd_only_map if not do_ns \
            else namestore.thd_only_ns_map
        thd_range = namestore.thd_only_map if not do_ns \
            else namestore.thd_only_ns_map
        map_onto = namestore.thd_map
    elif fall_type == rtypes.falloff_form.sri:
        our_inds = namestore.num_sri if not do_ns \
            else namestore.sri_ns_inds
        rxn_range = namestore.sri_map if not do_ns \
            else namestore.sri_has_ns
        thd_range = namestore.fall_to_thd_map
        map_onto = namestore.fall_map
    elif fall_type == rtypes.falloff_form.troe:
        our_inds = namestore.num_troe if not do_ns \
            else namestore.troe_ns_inds
        rxn_range = namestore.troe_map if not do_ns \
            else namestore.troe_has_ns
        thd_range = namestore.fall_to_thd_map
        map_onto = namestore.fall_map
    elif fall_type == rtypes.falloff_form.lind:
        our_inds = namestore.num_lind if not do_ns \
            else namestore.lind_ns_inds
        rxn_range = namestore.lind_map if not do_ns \
            else namestore.lind_has_ns
        thd_range = namestore.fall_to_thd_map
        map_onto = namestore.fall_map

    knl_name = 'dci_{}_dnj{}'.format(
        {rtypes.falloff_form.none: 'thd',
         rtypes.falloff_form.lind: 'lind',
         rtypes.falloff_form.sri: 'sri',
         rtypes.falloff_form.troe: 'troe'
         }[fall_type], '' if not do_ns else '_ns')

    if not rxn_range.initializer.size:
        # can't create kernel from empty reaction range
        return None

    # main loop is over third body rxns (including falloff / chemically
    # activated)
    mapstore = arc.MapStore(loopy_opts, rxn_range, rxn_range)

    # indicies
    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    if fall_type != rtypes.falloff_form.none:
        # the fall to third map depending on the reaction range
        mapstore.check_and_add_transform(thd_range, rxn_range)

    # and from the fall / third index to the actual reaction index
    mapstore.check_and_add_transform(map_onto, rxn_range)

    # the third body parameters map into the rxn_range
    mapstore.check_and_add_transform(namestore.thd_offset, thd_range)
    mapstore.check_and_add_transform(namestore.thd_type, thd_range)

    # while the net offsets map onto the map_onto range
    mapstore.check_and_add_transform(namestore.rop_fwd, map_onto)
    mapstore.check_and_add_transform(namestore.rxn_to_spec_offsets, map_onto)

    # and the reverse rop needs a map onto the rev mask
    mapstore.check_and_add_transform(namestore.rev_mask, map_onto)
    mapstore.check_and_add_transform(namestore.rop_rev, namestore.rev_mask)

    # falloff transforms
    if fall_type != rtypes.falloff_form.none:
        # pr is on the falloff index
        mapstore.check_and_add_transform(namestore.Pr, rxn_range)
        # while the pressure modification term is indexed by the third body
        # index
        mapstore.check_and_add_transform(
            namestore.pres_mod, namestore.fall_to_thd_map)
        # kf is on the real reaction index
        mapstore.check_and_add_transform(namestore.kf, map_onto)
        # Fi is indexed on the falloff index
        mapstore.check_and_add_transform(namestore.Fi, rxn_range)
        # kf_fall is on the falloff reaction index
        mapstore.check_and_add_transform(namestore.kf_fall, rxn_range)
        # and the falloff type is on the falloff index
        mapstore.check_and_add_transform(namestore.fall_type, rxn_range)

        if fall_type == rtypes.falloff_form.sri:
            # get the sri arrays, keyed on the SRI index
            mapstore.check_and_add_transform(namestore.sri_a, our_inds)
            mapstore.check_and_add_transform(namestore.sri_b, our_inds)
            mapstore.check_and_add_transform(namestore.sri_c, our_inds)
            mapstore.check_and_add_transform(namestore.X_sri, our_inds)
        elif fall_type == rtypes.falloff_form.troe:
            # get the troe arrays, keyed on the troe index
            mapstore.check_and_add_transform(namestore.Atroe, our_inds)
            mapstore.check_and_add_transform(namestore.Btroe, our_inds)
            mapstore.check_and_add_transform(namestore.Fcent, our_inds)

    # third body efficiencies, types and offsets
    thd_offset_lp, thd_offset_str = mapstore.apply_maps(
        namestore.thd_offset, var_name)
    _, thd_offset_next_str = mapstore.apply_maps(
        namestore.thd_offset, var_name, affine=1)

    # Ns efficiency is the last efficiency of this reaction
    # hence, we can simply take the next reaction and subtract 1
    thd_eff_ns_lp, thd_eff_ns_str = mapstore.apply_maps(
        namestore.thd_eff, thd_offset_next_str, affine=-1)

    # third body type is on main loop
    thd_type_lp, thd_type_str = mapstore.apply_maps(
        namestore.thd_type, var_name)

    # efficiencies / species are on the inner loop
    spec_k = 'spec_k'
    spec_j = 'spec_j'
    spec_j_ind = 'spec_j_ind'
    spec_k_ind = 'spec_k_ind'

    # third body eff of species j
    thd_eff_lp, thd_eff_j_str = mapstore.apply_maps(
        namestore.thd_eff, spec_j_ind)

    # species j
    thd_spec_lp, spec_j_str = mapstore.apply_maps(
        namestore.thd_spec, spec_j_ind)

    # get net species
    rxn_to_spec_offsets_lp, rxn_to_spec_offsets_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name)
    _, rxn_to_spec_offsets_next_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name, affine=1)
    # species 'k' is on k loop
    specs_lp, spec_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec, spec_k_ind)
    # get product nu
    nu_lp, prod_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, spec_k_ind, affine=spec_k_ind)
    # and reac nu
    _, reac_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, spec_k_ind, affine=spec_k_ind)

    jac_map = (spec_k, spec_j)

    # rop
    rop_fwd_lp, rop_fwd_str = mapstore.apply_maps(
        namestore.rop_fwd, *default_inds)
    rop_rev_lp, rop_rev_str = mapstore.apply_maps(
        namestore.rop_rev, *default_inds)
    rev_mask_lp, rev_mask_str = mapstore.apply_maps(
        namestore.rev_mask, var_name)

    # and jacobian
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, *jac_map, affine={x: 2 for x in jac_map}
    )

    # update data and extra inames
    kernel_data.extend([thd_offset_lp, thd_eff_ns_lp, thd_type_lp, thd_eff_lp,
                        thd_spec_lp, rxn_to_spec_offsets_lp, specs_lp, nu_lp,
                        rop_fwd_lp, rop_rev_lp, rev_mask_lp, jac_lp])

    parameters = {}
    manglers = []

    fall_update = ''
    # if we have a falloff term, need to calcule the dFi
    if fall_type != rtypes.falloff_form.none:
        # add entries needed by all falloff reactions
        Pr_lp, Pr_str = mapstore.apply_maps(namestore.Pr, *default_inds)
        Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)
        pres_mod_lp, pres_mod_str = mapstore.apply_maps(
            namestore.pres_mod, *default_inds)
        kf_lp, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)
        kf_fall_lp, kf_fall_str = mapstore.apply_maps(
            namestore.kf_fall, *default_inds)
        fall_type_lp, fall_type_str = mapstore.apply_maps(
            namestore.fall_type, var_name)
        # update data
        kernel_data.extend(
            [Pr_lp, pres_mod_lp, kf_lp, kf_fall_lp, fall_type_lp, Fi_lp])

        # handle type specific parameters
        if fall_type == rtypes.falloff_form.sri:
            sri_a_lp, sri_a_str = mapstore.apply_maps(
                namestore.sri_a, var_name)
            sri_b_lp, sri_b_str = mapstore.apply_maps(
                namestore.sri_b, var_name)
            sri_c_lp, sri_c_str = mapstore.apply_maps(
                namestore.sri_c, var_name)
            sri_X_lp, sri_X_str = mapstore.apply_maps(
                namestore.X_sri, *default_inds)

            # and the temperature
            T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)

            # add data, and put together falloff string
            kernel_data.extend([sri_a_lp, sri_b_lp, sri_c_lp, sri_X_lp, T_lp])

            dFi = Template(
                """
        <> dFi = -2 * ${sri_X_str} * ${sri_X_str} * log(${sri_a_str} * exp(-${sri_b_str} / ${T_str}) + exp(-${T_str} / ${sri_c_str})) * log(fmax(${Pr_str}, 1e-300d)) / (fmax(${Pr_str}, 1e-300d) * logtensquared) {id=dFi}
        """).substitute(
                sri_X_str=sri_X_str,
                sri_a_str=sri_a_str,
                sri_b_str=sri_b_str,
                sri_c_str=sri_c_str,
                T_str=T_str,
                Pr_str=Pr_str
            )

            parameters['logtensquared'] = log(10) * log(10)
            manglers.append(lp_pregen.fmax())

        elif fall_type == rtypes.falloff_form.troe:
            Atroe_lp, Atroe_str = mapstore.apply_maps(
                namestore.Atroe, *default_inds)
            Btroe_lp, Btroe_str = mapstore.apply_maps(
                namestore.Btroe, *default_inds)
            Fcent_lp, Fcent_str = mapstore.apply_maps(
                namestore.Fcent, *default_inds)

            kernel_data.extend([Atroe_lp, Btroe_lp, Fcent_lp])

            dFi = Template(
                """
        <> dFi = ${Atroe_str} * ${Atroe_str} + ${Btroe_str} * ${Btroe_str} {id=dFi_init}
        dFi = -2 * ${Atroe_str} * ${Btroe_str} * (0.14 * ${Atroe_str} + ${Btroe_str}) * log(fmax(${Fcent_str}, 1e-300d)) / (fmax(${Pr_str}, 1e-300d) * dFi * dFi * logten) {id=dFi, dep=dFi_init}
        """).substitute(
                Atroe_str=Atroe_str,
                Btroe_str=Btroe_str,
                Fcent_str=Fcent_str,
                Pr_str=Pr_str
            )
            parameters['logten'] = log(10)
            manglers.append(lp_pregen.fmax())
        else:
            # lindeman
            dFi = '<> dFi = 0d {id=dFi}'

        # finally handle the falloff vs chemically activated updater
        fall_update = Template("""
        ${dFi}

        # set parameters for both
        <> Fi_fac = dFi {id=dFi_fac_init, dep=dFi}
        if ${fall_type_str}
            # chemically activated
            <>k0 = ${kf_str} {id=kf_chem}
            <>kinf = ${kf_fall_str} {id=kinf_chem}
        else
            # fall-off
            kinf = ${kf_str} {id=kinf_fall}
            k0 = ${kf_fall_str} {id=kf_fall}
            Fi_fac = ${Pr_str} * Fi_fac + 1 {id=dFi_fac_up, dep=dFi_fac_init}
        end

        # and update dFi
        dFi =  k0 * (${Fi_str} * Fi_fac - ${pres_mod_str}) / (kinf * (${Pr_str} + 1)) {id=fall, dep=kf_*:kinf_*:dFi_fac_*}
        """).substitute(
            dFi=dFi,
            fall_type_str=fall_type_str,
            kf_str=kf_str,
            kf_fall_str=kf_fall_str,
            Pr_str=Pr_str,
            Fi_str=Fi_str,
            pres_mod_str=pres_mod_str
        )

    extra_inames = [
        (spec_k_ind, 'rxn_off <= {} < rxn_off_next'.format(spec_k_ind))]
    if not do_ns:
        extra_inames.append(
            (spec_j_ind, 'thd_off <= {} < thd_off_next'.format(spec_j_ind)))

        instructions = Template("""
        <> thd_off = ${thd_offset_str}
        <> thd_off_next = ${thd_offset_next_str}
        <> rxn_off = ${rxn_to_spec_offsets_str}
        <> rxn_off_next = ${rxn_to_spec_offsets_next_str}
        <> ropi = ${rop_fwd_str} {id=ropi}
        if ${rev_mask_str} >= 0
            ropi = ropi - ${rop_rev_str} {id=ropi_up, dep=ropi}
        end
        ${fall_update} # insert falloff form if necessary
        for ${spec_j_ind}
            <> spec_j = ${spec_j_str}
            if ${spec_j} != ${ns}
                <> dci = 0 {id=ci_init}
                if ${thd_type_str} == ${mix}
                    # here we assume the last species has an efficiency of 1
                    # if this is not true, it will be fixed in the Ns kernel
                    dci = ${thd_eff_j_str} - 1.0 {id=ci_up, dep=ci_init}
                end
                if ${thd_type_str} == ${species}
                    # if we get here, delta(j, m) is true by default
                    # hence derivative is one
                    dci = 1d {id=ci_up2, dep=ci_init}
                end
                for ${spec_k_ind}
                    <> ${spec_k} = ${spec_k_str}
                    <> nu_k= ${prod_nu_k_str} - ${reac_nu_k_str}
                    if ${spec_k} != ${ns}
                        ${jac_str} = ${jac_str} + nu_k * dci * ropi${fall_mul_str}
                    end
                end
            end
        end
        """).substitute(
            thd_offset_str=thd_offset_str,
            thd_offset_next_str=thd_offset_next_str,
            rxn_to_spec_offsets_str=rxn_to_spec_offsets_str,
            rxn_to_spec_offsets_next_str=rxn_to_spec_offsets_next_str,
            thd_eff_ns_str=thd_eff_ns_str,
            rop_fwd_str=rop_fwd_str,
            rop_rev_str=rop_rev_str,
            rev_mask_str=rev_mask_str,
            thd_eff_j_str=thd_eff_j_str,
            spec_j_ind=spec_j_ind,
            spec_j=spec_j,
            spec_j_str=spec_j_str,
            spec_k=spec_k,
            spec_k_str=spec_k_str,
            spec_k_ind=spec_k_ind,
            prod_nu_k_str=prod_nu_k_str,
            reac_nu_k_str=reac_nu_k_str,
            jac_str=jac_str,
            ns=namestore.num_specs.initializer[-1],
            thd_type_str=thd_type_str,
            mix=int(rtypes.thd_body_type.mix),
            species=int(rtypes.thd_body_type.species),
            fall_update=fall_update,
            fall_mul_str=(' * dFi {dep=fall}'
                          if fall_type != rtypes.falloff_form.none else '')
        )
    else:
        extra_inames.append((spec_j, '0 <= {} < {}'.format(
            spec_j, namestore.num_specs.initializer[-1])))

        instructions = Template("""
        <> rxn_off = ${rxn_to_spec_offsets_str}
        <> rxn_off_next = ${rxn_to_spec_offsets_next_str}
        <> ns_thd_eff = ${thd_eff_ns_str}
        <> ropi = ${rop_fwd_str} {id=ropi}
        if ${rev_mask_str} >= 0
            ropi = ropi - ${rop_rev_str} {id=ropi_up, dep=ropi}
        end
        <> dci = 0 {id=ci_init}
        ${fall_update} # insert falloff form if necessary
        if ${thd_type_str} == ${mix}
            # non-specified species have a efficiency of 1.0
            # and we have already deducted an efficiency of 1.0 from the
            # specified species
            # Hence, this formulation accounts for both parts of \alpha_j - \alpha_ns
            dci = 1.0 - ns_thd_eff {id=ci_up, dep=ci_init}
        end
        if ${thd_type_str} == ${species}
            dci = -1.0d {id=ci_up2, dep=ci_init}
        end
        for ${spec_k_ind}
            <> ${spec_k} = ${spec_k_str}
            <> nu_k= ${prod_nu_k_str} - ${reac_nu_k_str}
            if ${spec_k} != ${ns}
                for ${spec_j}
                    ${jac_str} = ${jac_str} + nu_k * dci * ropi${fall_mul_str}
                end
            end
        end
        """).substitute(
            thd_offset_str=thd_offset_str,
            thd_offset_next_str=thd_offset_next_str,
            rxn_to_spec_offsets_str=rxn_to_spec_offsets_str,
            rxn_to_spec_offsets_next_str=rxn_to_spec_offsets_next_str,
            thd_eff_ns_str=thd_eff_ns_str,
            rop_fwd_str=rop_fwd_str,
            rop_rev_str=rop_rev_str,
            rev_mask_str=rev_mask_str,
            thd_eff_j_str=thd_eff_j_str,
            spec_j_ind=spec_j_ind,
            spec_j=spec_j,
            spec_j_str=spec_j_str,
            spec_k=spec_k,
            spec_k_str=spec_k_str,
            spec_k_ind=spec_k_ind,
            prod_nu_k_str=prod_nu_k_str,
            reac_nu_k_str=reac_nu_k_str,
            jac_str=jac_str,
            ns=namestore.num_specs.initializer[-1],
            thd_type_str=thd_type_str,
            mix=int(rtypes.thd_body_type.mix),
            species=int(rtypes.thd_body_type.species),
            fall_update=fall_update,
            fall_mul_str=(' * dFi {dep=fall}'
                          if fall_type != rtypes.falloff_form.none else '')
        )

    inames, ranges = zip(*extra_inames)
    # join inames
    extra_inames = [
        (','.join(inames), ' and '.join(ranges))]

    return k_gen.knl_info(name=knl_name,
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          extra_inames=extra_inames,
                          mapstore=mapstore,
                          parameters=parameters,
                          manglers=manglers
                          )


def dci_thd_dnj(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    derivatives of the pressure modification term of third body reactions
    with respect to the molar quantity of a species

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dci_dnj`


    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        Contains keys 'fwd', 'rev' and 'net', with booleans corresponding to
        whether all nu values for that direction are integers.
        If True, powers of concentrations will be evaluated using
        multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False)]
    ns_info = __dci_dnj(loopy_opts, namestore, True)
    if ns_info:
        infos.append(ns_info)
    return infos


def dci_lind_dnj(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    derivatives of the pressure modification term of Lindemann falloff
    reactions with respect to the molar quantity of a species

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dci_dnj`


    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        Contains keys 'fwd', 'rev' and 'net', with booleans corresponding to
        whether all nu values for that direction are integers.
        If True, powers of concentrations will be evaluated using
        multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False, rtypes.falloff_form.lind)]
    ns_info = __dci_dnj(loopy_opts, namestore, True, rtypes.falloff_form.lind)
    if ns_info:
        infos.append(ns_info)
    return infos


def dci_sri_dnj(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    derivatives of the pressure modification term of SRI falloff
    reactions with respect to the molar quantity of a species

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dci_dnj`


    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        Contains keys 'fwd', 'rev' and 'net', with booleans corresponding to
        whether all nu values for that direction are integers.
        If True, powers of concentrations will be evaluated using
        multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False, rtypes.falloff_form.sri)]
    ns_info = __dci_dnj(loopy_opts, namestore, True, rtypes.falloff_form.sri)
    if ns_info:
        infos.append(ns_info)
    return infos


def dci_troe_dnj(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    derivatives of the pressure modification term of Troe falloff
    reactions with respect to the molar quantity of a species

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dci_dnj`


    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        Contains keys 'fwd', 'rev' and 'net', with booleans corresponding to
        whether all nu values for that direction are integers.
        If True, powers of concentrations will be evaluated using
        multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False, rtypes.falloff_form.troe)]
    ns_info = __dci_dnj(loopy_opts, namestore, True, rtypes.falloff_form.troe)
    if ns_info:
        infos.append(ns_info)
    return infos


def dRopi_dnj(eqs, loopy_opts, namestore, allint, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    derivatives of the Rate of Progress with respect to the molar quantity of
    a species

    Notes
    -----
    This is method is split into two kernels, the first handles derivatives of
    reactions with respect to the species in the reaction, that is if reaction
    `i` contains species `k, k+1...k+n`, it will consider the derivative of
    species `k` with respect to `k, k+1, k+2... k+n`, and so on with `k+1`,etc.

    The second kernel handles reactions where the last species in the mechanism
    is present in the reaction.  In this case, using the strict formulation
    results in non-zero derivatives for _all_ species in the mechanism due to
    the conservation of mass formulation of the last species. That is, it will
    compute the derivative of `k` w.r.t species `1, 2, ... Ns - 1` where Ns is
    the last species in the mechanism.

    This second kernel may be turned off to increase sparsity (at the expense)
    of an incorrect jacobian.  This is often desired for implicit integrators
    which often only need an approximation to the Jacobian anyways.

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        Contains keys 'fwd', 'rev' and 'net', with booleans corresponding to
        whether all nu values for that direction are integers.
        If True, powers of concentrations will be evaluated using
        multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # start developing the kernel

    def __dropidnj(do_ns=False):

        spec_j = 'spec_j'
        spec_k = 'spec_k'
        net_ind_k = 'net_ind_k'
        net_ind_j = 'net_ind_j'

        net_ind_inner = 'net_ind_inner'
        spec_inner = 'spec_inner'
        inner_inds = (net_ind_j, net_ind_inner) if not do_ns else (
            net_ind_inner,)

        jac_map = (spec_k, spec_j)

        # indicies
        kernel_data = []
        if namestore.test_size == 'problem_size':
            kernel_data.append(namestore.problem_size)

        rxn_range = namestore.num_reacs if not do_ns else namestore.rxn_has_ns

        mapstore = arc.MapStore(loopy_opts, rxn_range, rxn_range)
        # get net offsets

        # may need offset on all arrays on the main loop if do_ns,
        # hence check for transforms
        mapstore.check_and_add_transform(
            namestore.rxn_to_spec_offsets, rxn_range)

        if do_ns:
            # check for transform on forward
            mapstore.check_and_add_transform(namestore.kf, rxn_range)

            # and reverse
            mapstore.check_and_add_transform(
                namestore.rev_mask, rxn_range)

            # check and add transforms for pressure mod
            mapstore.check_and_add_transform(
                namestore.thd_mask, rxn_range)

        else:
            # add default transforms
            mapstore.check_and_add_transform(
                namestore.kr, namestore.rev_mask)
            mapstore.check_and_add_transform(
                namestore.pres_mod, namestore.thd_mask)

        rxn_to_spec_offsets_lp, rxn_to_spec_offsets_str = mapstore.apply_maps(
            namestore.rxn_to_spec_offsets, var_name)
        _, rxn_to_spec_offsets_next_str = mapstore.apply_maps(
            namestore.rxn_to_spec_offsets, var_name, affine=1)

        # get net species
        net_specs_lp, net_spec_k_str = mapstore.apply_maps(
            namestore.rxn_to_spec, net_ind_k)

        # get product nu
        net_nu_lp, prod_nu_k_str = mapstore.apply_maps(
            namestore.rxn_to_spec_prod_nu, net_ind_k, affine=net_ind_k)
        # and reac nu
        _, reac_nu_k_str = mapstore.apply_maps(
            namestore.rxn_to_spec_reac_nu, net_ind_k, affine=net_ind_k)

        # Check for forward / rev / third body maps with/without NS
        rev_mask_lp, rev_mask_str = mapstore.apply_maps(
            namestore.rev_mask, var_name)
        kr_lp = None
        pres_mod_lp = None
        if do_ns:
            # create mask string and use as kr index
            rev_mask_lp, rev_mask_str = mapstore.apply_maps(
                namestore.rev_mask, var_name)
            kr_lp, kr_str = mapstore.apply_maps(
                namestore.kr, global_ind, rev_mask_str)

            # create mask string and use as pmod index
            pmod_mask_lp, pmod_mask_str = mapstore.apply_maps(
                namestore.thd_mask, var_name)
            pres_mod_lp, pres_mod_str = mapstore.apply_maps(
                namestore.pres_mod, global_ind, pmod_mask_str)
        else:

            # default creators:
            rev_mask_lp, rev_mask_str = mapstore.apply_maps(
                namestore.rev_mask, var_name)
            kr_lp, kr_str = mapstore.apply_maps(
                namestore.kr, *default_inds)

            pmod_mask_lp, pmod_mask_str = mapstore.apply_maps(
                namestore.thd_mask, var_name)
            pres_mod_lp, pres_mod_str = mapstore.apply_maps(
                namestore.pres_mod, *default_inds)

        # get fwd / rev rates

        kf_lp, kf_str = mapstore.apply_maps(
            namestore.kf, *default_inds)

        # next we need the forward / reverse nu's and species
        specs_lp, spec_j_str = mapstore.apply_maps(
            namestore.rxn_to_spec, net_ind_j)
        _, spec_inner_str = mapstore.apply_maps(
            namestore.rxn_to_spec, net_ind_inner)
        _, spec_j_prod_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_prod_nu, net_ind_j, affine=net_ind_j)
        _, spec_j_reac_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_reac_nu, net_ind_j, affine=net_ind_j)
        _, inner_prod_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_prod_nu, net_ind_inner, affine=net_ind_inner)
        _, inner_reac_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_reac_nu, net_ind_inner, affine=net_ind_inner)

        # finally, we need the concentrations for the fwd / rev loops
        conc_lp, conc_inner_str = mapstore.apply_maps(
            namestore.conc_arr, global_ind, spec_inner)

        # and finally the jacobian
        jac_lp, jac_str = mapstore.apply_maps(
            namestore.jac, global_ind, *jac_map, affine={x: 2 for x in jac_map}
        )

        kernel_data.extend([rxn_to_spec_offsets_lp, net_specs_lp, net_nu_lp,
                            pres_mod_lp, kf_lp, kr_lp, conc_lp, jac_lp,
                            pmod_mask_lp, rev_mask_lp])

        # now start creating the instructions

        extra_inames = [
            (net_ind_k,
             'net_offset <= {} < net_offset_next'.format(net_ind_k))]
        for ind in inner_inds:
            extra_inames.append(
                (ind, 'inner_offset <= {} < inner_offset_next'.format(ind)))

        if not do_ns:
            inner = ("""
                for ${net_ind_j}
                    <> ${spec_j} = ${spec_j_str} {id=spec_j}
                    if ${spec_j} != ${ns}
                        <> Sj_fwd = ${spec_j_reac_nu_str} {id=Sj_fwd_init}
                        <> Sj_rev = ${spec_j_prod_nu_str} {id=Sj_rev_init}
                        for ${net_ind_inner}
                            <> nu_fwd = ${inner_reac_nu_str} {id=nuf_inner}
                            <> nu_rev = ${inner_prod_nu_str} {id=nur_inner}
                            <> ${spec_inner} = ${spec_inner_str}
                            # handle nu
                            if ${spec_inner} == ${spec_j}
                                nu_fwd = nu_fwd - 1 {id=nuf_inner_up, dep=nuf_inner}
                            end
                            if ${spec_inner} == ${spec_j}
                                nu_rev = nu_rev - 1 {id=nur_inner_up, dep=nur_inner}
                            end
                            Sj_fwd = Sj_fwd * fast_powi(${conc_inner_str}, nu_fwd) {id=Sj_fwd_up, dep=Sj_fwd_init:nuf_inner_up}
                            Sj_rev = Sj_rev * fast_powi(${conc_inner_str}, nu_rev) {id=Sj_rev_up, dep=Sj_rev_init:nur_inner_up}
                        end
                        # and update Jacobian
                        ${jac_str} = ${jac_str} + (kf_i * Sj_fwd - kr_i * Sj_rev) * ci * nu_k {id=jac_up, dep=Sj_fwd_up:Sj_rev_up:ci_up:nu_k:spec_k}
                    end
                end
            """)
        else:
            extra_inames.append((spec_j, '0 <= {} < {}'.format(
                spec_j, namestore.num_specs.initializer[-1])))
            inner = ("""
                <> Sns_fwd = 1.0d {id=Sns_fwd_init}
                <> Sns_rev = 1.0d {id=Sns_rev_init}
                for ${net_ind_inner}
                    <> nu_fwd = ${inner_reac_nu_str} {id=nuf_inner}
                    <> nu_rev = ${inner_prod_nu_str} {id=nur_inner}
                    <> ${spec_inner} = ${spec_inner_str}
                    # handle nu
                    if ${spec_inner} == ${ns}
                        Sns_fwd = Sns_fwd * nu_fwd {id=Sns_fwd_up, dep=Sns_fwd_init}
                        nu_fwd = nu_fwd - 1 {id=nuf_inner_up, dep=nuf_inner:Sns_fwd_up}
                    end
                    Sns_fwd = Sns_fwd * fast_powi(${conc_inner_str}, nu_fwd) {id=Sns_fwd_up2, dep=Sns_fwd_up:nuf_inner_up}
                    if ${spec_inner} == ${ns}
                        Sns_rev = Sns_rev * nu_rev {id=Sns_rev_up, dep=Sns_rev_init}
                        nu_rev = nu_rev - 1 {id=nur_inner_up, dep=nur_inner:Sns_rev_up}
                    end
                    Sns_rev = Sns_rev * fast_powi(${conc_inner_str}, nu_rev) {id=Sns_rev_up2, dep=Sns_rev_up:nur_inner_up}
                end
                # and update Jacobian for all species in this row
                <> jac_updater =  (kr_i * Sns_rev - kf_i * Sns_fwd) * ci * nu_k {id=jac_up, dep=Sns_fwd_up*:Sns_rev_up*:ci_up:nu_k:spec_k:kf:kr*}
                for ${spec_j}
                    ${jac_str} = ${jac_str} + jac_updater
                end
            """)

        instructions = Template("""
            # loop over all species in reaction
            <> ci = 1.0d {id=ci_set}
            if ${pmod_mask_str} >= 0
                ci = ${pres_mod_str} {id=ci_up, dep=ci_set}
            end
            <> kf_i = ${kf_str} {id=kf}
            <> kr_i = 0.0d {id=kr}
            if ${rev_mask_str} >= 0
                kr_i = ${kr_str} {id=kr2, dep=kr}
            end

            <> net_offset = ${rxn_to_spec_offsets_str}
            <> net_offset_next = ${rxn_to_spec_offsets_next_str}
            <> inner_offset = ${rxn_to_spec_offsets_str}
            <> inner_offset_next = ${rxn_to_spec_offsets_next_str}
            # loop over net species
            for ${net_ind_k}
                # get species and nu k
                <> ${spec_k} = ${net_spec_k_str} {id=spec_k}
                if ${spec_k} != ${ns}
                    <> nu_k = ${prod_nu_k_str} - ${reac_nu_k_str} {id=nu_k}
                    # put in inner
                    ${inner}
                end
            end
        """).safe_substitute(inner=inner)
        instructions = Template(instructions).substitute(
            rxn_to_spec_offsets_str=rxn_to_spec_offsets_str,
            rxn_to_spec_offsets_next_str=rxn_to_spec_offsets_next_str,
            net_ind_k=net_ind_k,
            spec_k=spec_k,
            net_spec_k_str=net_spec_k_str,
            prod_nu_k_str=prod_nu_k_str,
            reac_nu_k_str=reac_nu_k_str,
            net_ind_inner=net_ind_inner,
            net_ind_j=net_ind_j,
            spec_j=spec_j,
            spec_j_str=spec_j_str,
            spec_j_reac_nu_str=spec_j_reac_nu_str,
            spec_j_prod_nu_str=spec_j_prod_nu_str,
            inner_reac_nu_str=inner_reac_nu_str,
            inner_prod_nu_str=inner_prod_nu_str,
            spec_inner=spec_inner,
            spec_inner_str=spec_inner_str,
            conc_inner_str=conc_inner_str,
            kf_str=kf_str,
            kr_str=kr_str,
            jac_str=jac_str,
            pmod_mask_str=pmod_mask_str,
            rev_mask_str=rev_mask_str,
            pres_mod_str=pres_mod_str,
            ns=namestore.num_specs.initializer[-1]
        )

        inames, ranges = zip(*extra_inames)
        # join inames
        extra_inames = [
            (','.join(inames), ' and '.join(ranges))]
        return k_gen.knl_info(name='dRopidnj{}'.format('_ns' if do_ns else ''),
                              instructions=instructions,
                              var_name=var_name,
                              kernel_data=kernel_data,
                              extra_inames=extra_inames,
                              mapstore=mapstore,
                              preambles=[
                                   lp_pregen.fastpowi_PreambleGen(),
                                   lp_pregen.fastpowf_PreambleGen()]
                              )

    return [__dropidnj(False), __dropidnj(True)]


def create_jacobian(lang,
                    mech_name=None,
                    therm_name=None,
                    gas=None,
                    vector_size=None,
                    wide=False,
                    deep=False,
                    ilp=False,
                    unr=None,
                    build_path='./out/',
                    last_spec=None,
                    skip_jac=False,
                    auto_diff=False,
                    platform='',
                    data_order='C',
                    rate_specialization='full',
                    split_rate_kernels=True,
                    split_rop_net_kernels=False,
                    spec_rates_sum_over_reac=True,
                    conp=True,
                    data_filename='data.bin',
                    output_full_rop=False
                    ):
    """Create Jacobian subroutine from mechanism.

    Parameters
    ----------
    lang : {'c', 'opencl'}
        Language type.
    mech_name : str, optional
        Reaction mechanism filename (e.g. 'mech.dat').
        This or gas must be specified
    therm_name : str, optional
        Thermodynamic database filename (e.g. 'therm.dat')
        or nothing if info in mechanism file.
    gas : cantera.Solution, optional
        The mechanism to generate the Jacobian for.  This or ``mech_name`` must be specified
    vector_size : int
        The SIMD vector width to use.  If the targeted platform is a GPU, this is the GPU block size
    wide : bool
        If true, use a 'wide' vectorization strategy. Cannot be specified along with 'deep'.
    deep : bool
        If true, use a 'deep' vectorization strategy.  Cannot be specified along with 'wide'.  Currently broken
    unr : int
        If supplied, unroll inner loops (i.e. those that would be affected by a deep vectorization).
        Can be used in conjunction with deep or wide parallelism
    build_path : str, optional
        The output directory for the jacobian files
    last_spec : str, optional
        If specified, the species to assign to the last index.
        Typically should be N2, Ar, He or another inert bath gas
    skip_jac : bool, optional
        If ``True``, only the reaction rate subroutines will be generated
    auto_diff : bool, optional
        If ``True``, generate files for use with the Adept autodifferention library.
    platform : {'CPU', 'GPU', or other vendor specific name}
        The OpenCL platform to run on.
        *   If 'CPU' or 'GPU', the first available matching platform will be used
        *   If a vendor specific string, it will be passed to pyopencl to get the platform
    data_order : {'C', 'F'}
        The data ordering, 'C' (row-major) recommended for deep vectorizations, while 'F' (column-major)
        recommended for wide vectorizations
    rate_specialization : {'fixed', 'hybrid', 'full'}
        The level of specialization in evaluating reaction rates.
        'Full' is the full form suggested by Lu et al. (citation)
        'Hybrid' turns off specializations in the exponential term (Ta = 0, b = 0)
        'Fixed' is a fixed expression exp(logA + b logT + Ta / T)
    split_rate_kernels : bool
        If True, and the :param"`rate_specialization` is not 'Fixed', split different evaluation types
        into different kernels
    split_rop_net_kernels : bool
        If True, break different ROP values (fwd / back / pdep) into different kernels
    spec_rates_sum_over_reac : bool
        Controls the manner in which the species rates are calculated
        *  If True, the summation occurs as:
            for reac:
                rate = reac_rates[reac]
                for spec in reac:
                    spec_rate[spec] += nu(spec, reac) * reac_rate
        *  If False, the summation occurs as:
            for spec:
                for reac in spec_to_reacs[spec]:
                    rate = reac_rates[reac]
                    spec_rate[spec] += nu(spec, reac) * reac_rate
        *  Of these, the first choice appears to be slightly more efficient, likely due to less
        thread divergence / SIMD wastage, HOWEVER it causes issues with deep vectorization
        (an atomic update of the spec_rate is needed, and doesn't appear to work in current loopy)
        Hence, we supply both.
        *  Note that if True, and deep vectorization is passed this switch will be ignored
        and a warning will be issued
    conp : bool
        If True, use the constant pressure assumption.  If False, use the constant volume assumption.
    data_filename : str
        If specified, the path to the data.bin file that will be used for kernel testing
    output_full_rop : bool
        If ``True``, output forward and reversse rates of progress
        Useful in testing, as there are serious floating point errors for
        net production rates near equilibrium, invalidating direct comparison to Cantera
    Returns
    -------
    None

    """
    if auto_diff or not skip_jac:
        raise NotImplementedError()

    if lang != 'c' and auto_diff:
        print('Error: autodifferention only supported for C')
        sys.exit(2)

    if auto_diff:
        skip_jac = True

    lang = lang.lower()
    if lang not in utils.langs:
        print('Error: language needs to be one of: ')
        for l in utils.langs:
            print(l)
        sys.exit(2)

    # configure options
    width = None
    depth = None
    if wide:
        width = vector_size
    elif deep:
        depth = vector_size

    rspec = ['fixed', 'hybrid', 'full']
    rate_spec_val = next(
        (i for i, x in enumerate(rspec) if rate_specialization.lower() == x), None)
    assert rate_spec_val is not None, 'Error: rate specialization value {} not recognized.\nNeeds to be one of: {}'.format(
        rate_specialization, ', '.join(rspec))
    rate_spec_val = lp_utils.RateSpecialization(rate_spec_val)

    # create the loopy options
    loopy_opts = lp_utils.loopy_options(width=width,
                                        depth=depth,
                                        ilp=False,
                                        unr=unr,
                                        lang=lang,
                                        order=data_order,
                                        rate_spec=rate_spec_val,
                                        rate_spec_kernels=split_rate_kernels,
                                        rop_net_kernels=split_rop_net_kernels,
                                        spec_rates_sum_over_reac=spec_rates_sum_over_reac,
                                        platform=platform)

    # create output directory if none exists
    utils.create_dir(build_path)

    if auto_diff:
        with open(os.path.join(build_path, 'ad_jacob.h'), 'w') as file:
            file.write('#ifndef AD_JAC_H\n'
                       '#define AD_JAC_H\n'
                       'void eval_jacob (const double t, const double pres, '
                       'const double* y, double* jac);\n'
                       '#endif\n'
                       )

    assert mech_name is not None or gas is not None, 'No mechanism specified!'

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if gas is not None or mech_name.endswith(tuple(['.cti', '.xml'])):
        elems, specs, reacs = mech.read_mech_ct(mech_name, gas)
    else:
        elems, specs, reacs = mech.read_mech(mech_name, therm_name)

    if not specs:
        print('No species found in file: {}'.format(mech_name))
        sys.exit(3)

    if not reacs:
        print('No reactions found in file: {}'.format(mech_name))
        sys.exit(3)

    # #check to see if the last_spec is specified
    # if last_spec is not None:
    #     #find the index if possible
    #     isp = next((i for i, sp in enumerate(specs)
    #                if sp.name.lower() == last_spec.lower().strip()),
    #                None
    #                )
    #     if isp is None:
    #         print('Warning: User specified last species {} '
    #               'not found in mechanism.'
    #               '  Attempting to find a default species.'.format(last_spec)
    #               )
    #         last_spec = None
    #     else:
    #         last_spec = isp
    # else:
    #     print('User specified last species not found or not specified.  '
    #           'Attempting to find a default species')
    # if last_spec is None:
    #     wt = chem.get_elem_wt()
    #     #check for N2, Ar, He, etc.
    #     candidates = [('N2', wt['n'] * 2.), ('Ar', wt['ar']),
    #                     ('He', wt['he'])]
    #     for sp in candidates:
    #         match = next((isp for isp, spec in enumerate(specs)
    #                       if sp[0].lower() == spec.name.lower() and
    #                       sp[1] == spec.mw),
    #                         None)
    #         if match is not None:
    #             last_spec = match
    #             break
    #     if last_spec is not None:
    #         print('Default last species '
    #               '{} found.'.format(specs[last_spec].name)
    #               )
    # if last_spec is None:
    #     print('Warning: Neither a user specified or default last species '
    #           'could be found. Proceeding using the last species in the '
    #           'base mechanism: {}'.format(specs[-1].name))
    #     last_spec = len(specs) - 1

    # #pick up the last_spec and drop it at the end
    # temp = specs[:]
    # specs[-1] = temp[last_spec]
    # specs[last_spec] = temp[-1]

    # write headers
    aux.write_aux(build_path, loopy_opts, specs, reacs)

    eqs = {}
    eqs['conp'] = sp_interp.load_equations(conp)[1]
    eqs['conv'] = sp_interp.load_equations(not conp)[1]

    # now begin writing subroutines
    kgen = rate.write_specrates_kernel(eqs, reacs, specs, loopy_opts,
                                       conp=conp, output_full_rop=output_full_rop)

    # generate
    kgen.generate(build_path, data_filename=data_filename)

    if skip_jac == False:
        # write Jacobian subroutine
        touched = write_jacobian(build_path, lang, specs,
                                 reacs, seen_sp, smm)

        write_sparse_multiplier(build_path, lang, touched, len(specs))

    return 0


if __name__ == "__main__":
    args = utils.get_parser()

    create_jacobian(lang=args.lang,
                    mech_name=args.input,
                    therm_name=args.thermo,
                    optimize_cache=args.cache_optimizer,
                    initial_state=args.initial_conditions,
                    num_blocks=args.num_blocks,
                    num_threads=args.num_threads,
                    no_shared=args.no_shared,
                    L1_preferred=args.L1_preferred,
                    multi_thread=args.multi_thread,
                    force_optimize=args.force_optimize,
                    build_path=args.build_path,
                    last_spec=args.last_species,
                    auto_diff=args.auto_diff
                    )
