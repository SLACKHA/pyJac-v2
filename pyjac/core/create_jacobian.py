#! /usr/bin/env python
"""Creates source code for calculating analytical Jacobian matrix.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import sys
from math import log
from string import Template
import logging

# Local imports
from .. import utils
from . import mech_interpret as mech
from . import rate_subs as rate
from . import mech_auxiliary as aux
from ..sympy_utils import sympy_interpreter as sp_interp
from ..loopy_utils import loopy_utils as lp_utils
from ..loopy_utils import preambles_and_manglers as lp_pregen
from ..loopy_utils.loopy_utils import JacobianType, JacobianFormat
from . import array_creator as arc
from ..kernel_utils import kernel_gen as k_gen
from .reaction_types import reaction_type, falloff_form, thd_body_type
from . import chem_model as chem
from . import instruction_creator as ic
from .array_creator import (global_ind, var_name, default_inds)
from .rate_subs import assign_rates

# external
import numpy as np


def determine_jac_inds(reacs, specs, rate_spec, jacobian_type=JacobianType.exact):
    """
    From a given set of reactions, determine the populated jacobian indicies.
    Additionally, populate the rate information from :meth:`pyjac.core.assign_rates`

    Parameters
    ----------
    reacs : list of `ReacInfo`
        The reactions in the mechanism
    specs : list of `SpecInfo`
        The species in the mechanism
    rate_spec : `RateSpecialization` enum
        The specialization option specified
    JacobianType : :class:`JacobianType`
        The Jacobian type to be constructed, a full Jacobian has no approximations
        for reactions including the last species while an approximate Jacobian
        ignores the contributions from species not involved

    Notes
    -----

    See also :meth:`pyjac.core.assign_rates`, :class:`JacobianType`

    Returns
    -------
    jac_info : dict of parameters
        Keys are 'jac_inds', which contains:
            'flat': a flattened list of non-zero jacobian indicies.
            'ccs': a dictionary of 'col_ind' and 'row_ptr' representing the indicies
                in a compressed column storage format
            'crs': a dictionary of 'col_ind' and 'row_ptr' representing the indicies
                in a compressed row storage format

        Additionally, `jac_info` will contain the results from
        :meth:`pyjac.core.assign_rates`
    """

    val = assign_rates(reacs, specs, rate_spec)

    inds = []
    row_size = len(specs) + 1  # Ns - 1 species + temperature + extra variable
    species_offset = 2  # temperature + extra variable

    def __add_row(row):
        inds.extend([(row, x) for x in range(row_size)])

    # The first row is all derivatives of the dT/dt term, and no entries are
    # zero
    __add_row(0)

    # the second row is derivatives of the extra variable, and again is
    # non-zero
    __add_row(1)

    # From here on out:
    #
    # The first entry is the derivative of dnj/dt w.r.t Temperature
    #       -> this is non-zero if this species has a non-zero net stoich. coeff in
    #          any reaction
    #
    # The second entry is the derivative of dnj/dt w.r.t. the extra variable (P/V)
    #       -> this is non-zero if this species has a non-zero net stoich. coeff in
    #          any reaction

    def __offset(arr):
        return np.array(np.concatenate(
            (np.cumsum(arr) - arr, np.array([np.sum(arr)]))),
            dtype=np.int32)

    # get list of species that have a non-zero nu in some reaction
    non_zero_specs = val['net_per_spec']['map']
    rxn_count = __offset(val['net_per_spec']['reac_count'])
    rxn_maps = val['net_per_spec']['reacs']
    rxn_to_specs_map = val['net']['reac_to_spec']
    num_specs_in_rxn = val['net']['num_reac_to_spec']
    num_specs_in_rxn = __offset(num_specs_in_rxn)
    has_ns = val['reac_has_ns']
    thd_has_ns = val['thd']['has_ns']
    thd_spec = val['thd']['spec']
    thd_map = val['thd']['map']
    num_specs_in_thd = __offset(val['thd']['spec_num'])

    assert np.unique(non_zero_specs).size == non_zero_specs.size
    for spec in non_zero_specs:
        row = spec + species_offset
        nonzero_derivs = set([0, 1])

        def __add_specs(slist):
            # add species to derivative list
            nonzero_derivs.update([x + species_offset for x in slist
                                   if x + species_offset < row_size])

        # add the temperature and extra var derivative
        nonzero_derivs.update([0, 1])

        # now we go through the reactions for which this species is non-zero
        inner_ind = np.where(non_zero_specs == spec)[0][0]
        for rxn in rxn_maps[rxn_count[inner_ind]:rxn_count[inner_ind + 1]]:
            # get third body index
            thd_ind = None
            if rxn in thd_map:
                thd_ind = np.where(thd_map == rxn)[0][0]

            # if the last species directly participates in the reaction, and we're
            # looking for a full Jacobian, this entire row has non-zero
            # derivatives
            if rxn in has_ns or thd_ind in thd_has_ns and \
                    jacobian_type == JacobianType.exact:
                __add_specs(range(row_size))
                break

            # update species in the reaction
            __add_specs(rxn_to_specs_map[
                num_specs_in_rxn[rxn]:num_specs_in_rxn[rxn + 1]])

            if thd_ind is not None:
                # update third body species in the reaction
                __add_specs(thd_spec[
                    num_specs_in_thd[thd_ind]:num_specs_in_thd[thd_ind + 1]])

        # finally add the non-zero derivatives
        if len(nonzero_derivs):
            inds.extend([(row, x) for x in sorted(nonzero_derivs)])

    # get the compressed storage
    rows, cols = zip(*inds)

    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)

    # get a column-major version for flat inds
    inds_F = np.array(inds, copy=True)
    offset = 0
    for i, col in enumerate(np.unique(cols)):
        # find the rows that contains this col
        row_F = rows[np.where(cols == col)[0]]
        # place in inds
        inds_F[offset:offset + row_F.size] = np.asarray(
            (row_F, [col] * row_F.size), dtype=np.int32).T
        offset += row_F.size

    # turn into row and colum counts
    row_ptr = []
    col_ind = []

    col_ptr = []
    row_ind = []
    for i in range(row_size):
        try:
            in_row = np.where(rows == i)[0]
            row_ptr.append(in_row.size)
            col_ind.extend(cols[in_row])
        except:
            pass

        try:
            in_col = np.where(cols == i)[0]
            col_ptr.append(in_col.size)
            row_ind.extend(rows[in_col])
        except:
            pass

    # update indicies in return value
    val['jac_inds'] = {
        'flat_C': np.asarray(inds, dtype=np.int32),
        'flat_F': np.asarray(inds_F, dtype=np.int32),
        'crs': {'col_ind': np.array(col_ind, dtype=np.int32),
                'row_ptr': __offset(row_ptr)},
        'ccs': {'row_ind': np.array(row_ind, dtype=np.int32),
                'col_ptr': __offset(col_ptr)},
    }
    return val


def reset_arrays(eqs, loopy_opts, namestore, test_size=None, conp=True):
    """Resets the Jacobian array for use in the evaluations

    kernel

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_nonzero_jac_inds,
                            namestore.num_nonzero_jac_inds)

    # first, create all arrays
    kernel_data = []

    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

    k_ind = 'row'
    j_ind = 'col'

    if loopy_opts.jac_format == JacobianFormat.sparse:
        jac_lp, jac_str = mapstore.apply_maps(namestore.jac, global_ind, var_name,
                                              ignore_lookups=True)
        instructions = Template(
            """
                ${jac_str} = 0d {id=reset}
            """).substitute(**locals())

        kernel_data.extend([jac_lp])
    else:
        # need jac_array
        jac_lp, jac_str = mapstore.apply_maps(
            namestore.jac, global_ind, k_ind, j_ind)
        # and row / col inds
        row_lp, row_str = mapstore.apply_maps(namestore.flat_jac_row_inds, var_name)
        col_lp, col_str = mapstore.apply_maps(namestore.flat_jac_col_inds, var_name)

        instructions = Template(
            """
                <> ${k_ind} = ${row_str}
                <> ${j_ind} = ${col_str}
                ${jac_str} = 0d {id=reset}
            """).substitute(**locals())

        kernel_data.extend([jac_lp, row_lp, col_lp])

    # add arrays
    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, init_ids=['reset'])

    return k_gen.knl_info(name='reset_arrays',
                          instructions=instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec)


@ic.with_conditional_jacobian
def __dcidE(eqs, loopy_opts, namestore, test_size=None,
            rxn_type=reaction_type.thd, conp=True, jac_create=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term for all third body /
    falloff / chemically activated reactions with respect to the extra variable
    (volume / pressure) for constant pressure/volume respectively


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
    rxn_type: [reaction_type.thd, falloff_form.lind, falloff_form.sri,
               falloff_form.troe]
        The reaction type to generate the pressure modification derivative for
    conp : bool [True]
        If supplied, True for constant pressure jacobian. False for constant
        volume [Default: True]
    jac_create: Callable
        The conditional Jacobian instruction creator from :mod:`instruction_creator`
    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    num_range_dict = {reaction_type.thd: namestore.num_thd_only,
                      falloff_form.lind: namestore.num_lind,
                      falloff_form.sri: namestore.num_sri,
                      falloff_form.troe: namestore.num_troe}
    thd_range_dict = {reaction_type.thd: namestore.thd_only_map,
                      falloff_form.lind: namestore.fall_to_thd_map,
                      falloff_form.sri: namestore.fall_to_thd_map,
                      falloff_form.troe: namestore.fall_to_thd_map}
    fall_range_dict = {falloff_form.lind: namestore.lind_map,
                       falloff_form.sri: namestore.sri_map,
                       falloff_form.troe: namestore.troe_map}
    name_description = {reaction_type.thd: 'thd',
                        falloff_form.lind: 'lind',
                        falloff_form.sri: 'sri',
                        falloff_form.troe: 'troe'}
    # get num
    num_range = num_range_dict[rxn_type]
    thd_range = thd_range_dict[rxn_type]
    rxn_range = namestore.thd_map

    # number of species
    ns = namestore.num_specs[-1]

    # create mapstore
    mapstore = arc.MapStore(loopy_opts, num_range, num_range)

    # setup static mappings

    if rxn_type in fall_range_dict:
        fall_range = fall_range_dict[rxn_type]

        # falloff index depends on num_range
        mapstore.check_and_add_transform(fall_range, num_range)

        # and the third body index depends on the falloff index
        mapstore.check_and_add_transform(thd_range, fall_range)

    # and finally the reaction index depends on the third body index
    mapstore.check_and_add_transform(rxn_range, thd_range)

    # setup third body stuff
    mapstore.check_and_add_transform(namestore.thd_type, thd_range)
    mapstore.check_and_add_transform(namestore.thd_offset, thd_range)

    # and place rop's / species maps, etc.
    mapstore.check_and_add_transform(namestore.rop_fwd, rxn_range)
    mapstore.check_and_add_transform(namestore.rev_mask, rxn_range)
    mapstore.check_and_add_transform(namestore.rop_rev, namestore.rev_mask)
    mapstore.check_and_add_transform(namestore.rxn_to_spec_offsets, rxn_range)

    # pressure mod term
    mapstore.check_and_add_transform(namestore.pres_mod, thd_range)
    if rxn_type != reaction_type.thd:
        # falloff type
        mapstore.check_and_add_transform(namestore.fall_type, fall_range)
        # need falloff blending / reduced pressure
        mapstore.check_and_add_transform(namestore.Fi, fall_range)
        mapstore.check_and_add_transform(namestore.Pr, fall_range)

        # in addition we (most likely) need the falloff / regular kf rates
        mapstore.check_and_add_transform(namestore.kf, rxn_range)
        mapstore.check_and_add_transform(namestore.kf_fall, fall_range)

    # get the third body types
    thd_type_lp, thd_type_str = mapstore.apply_maps(
        namestore.thd_type, var_name)
    thd_offset_lp, thd_offset_next_str = mapstore.apply_maps(
        namestore.thd_offset, var_name, affine=1)
    # get third body efficiencies & species
    thd_eff_lp, thd_eff_last_str = mapstore.apply_maps(
        namestore.thd_eff, thd_offset_next_str, affine=-1)
    thd_spec_lp, thd_spec_last_str = mapstore.apply_maps(
        namestore.thd_spec, thd_offset_next_str, affine=-1)

    k_ind = 'k_ind'
    # nu offsets
    nu_offset_lp, offset_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name)
    _, offset_next_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name, affine=1)
    # setup species k loop
    extra_inames = [(k_ind, 'offset <= {} < offset_next'.format(k_ind))]

    # reac and prod nu for net
    nu_lp, reac_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, k_ind, affine=k_ind)
    _, prod_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, k_ind, affine=k_ind)
    # get species
    spec_lp, spec_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec, k_ind)
    # and jac
    jac_update_insn = (
        "${jac_str} = ${jac_str} + (${prod_nu_k_str} - ${reac_nu_k_str}) * "
        "${factor} {id=jac, dep=${deps}}")
    jac_lp, jac_update_insn = jac_create(
        mapstore, namestore.jac, global_ind, spec_k_str, 1,
        affine={spec_k_str: 2}, insn=jac_update_insn, deps='dci_*')

    # ropnet
    rop_fwd_lp, rop_fwd_str = mapstore.apply_maps(
        namestore.rop_fwd, *default_inds)
    rop_rev_lp, rop_rev_str = mapstore.apply_maps(
        namestore.rop_rev, *default_inds)

    # T, P, V
    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    V_lp, V_str = mapstore.apply_maps(namestore.V_arr, global_ind)
    P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)
    pres_mod_lp, pres_mod_str = mapstore.apply_maps(
        namestore.pres_mod, *default_inds)

    # update kernel data
    kernel_data.extend([thd_type_lp, thd_offset_lp, thd_eff_lp, thd_spec_lp,
                        nu_offset_lp, nu_lp, spec_lp, rop_fwd_lp, rop_rev_lp,
                        jac_lp, pres_mod_lp, T_lp, V_lp, P_lp])

    mix = int(thd_body_type.mix)
    spec = int(thd_body_type.species)
    unity = int(thd_body_type.unity)

    parameters = {'Ru': chem.RU}
    pre_instructions = [Template(
        '<> rt_inv = 1 / (Ru * ${T_str})').safe_substitute(**locals())]
    # the factor to multiply the 'factor' term by for updat
    fac_term = (
        '* rt_inv * {} * rop_net '.format(V_str) if not conp else ' * rop_net')
    # the k0 / kinf factor
    manglers = []
    # by default we are using the third body factors (these may be changed
    # in the falloff types below)
    factor = 'dci_thd_dE'
    # the pressure modification term to use (pres_mod for thd, Pr for falloff)
    fall_instructions = ''
    if rxn_type != reaction_type.thd:
        # update factors
        factor = 'dci_fall_dE'
        # create arrays
        fall_type_lp, fall_type_str = mapstore.apply_maps(
            namestore.fall_type, var_name)
        Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)
        Pr_lp, Pr_str = mapstore.apply_maps(namestore.Pr, *default_inds)
        # set the pmod factor
        kf_lp, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)
        kf_fall_lp, kf_fall_str = mapstore.apply_maps(
            namestore.kf_fall, *default_inds)

        kernel_data.extend([pres_mod_lp, fall_type_lp, Fi_lp, Pr_lp, kf_lp,
                            kf_fall_lp])

        # factor used for the Theta_Pr part of conp Pr derivative
        fall_instructions = Template(
            '<int32> not_unity = ${thd_type_str} != ${unity}').safe_substitute(
            **locals())
        conp_theta_pr_fac = Template('- ${Pr_str} * not_unity').safe_substitute(
            **locals()) if conp else ''

        # check for Troe / SRI
        if rxn_type == falloff_form.troe:
            Atroe_lp, Atroe_str = mapstore.apply_maps(
                namestore.Atroe, *default_inds)
            Btroe_lp, Btroe_str = mapstore.apply_maps(
                namestore.Btroe, *default_inds)
            Fcent_lp, Fcent_str = mapstore.apply_maps(
                namestore.Fcent, *default_inds)
            kernel_data.extend([Atroe_lp, Btroe_lp, Fcent_lp])
            dFi_instructions = Template("""
                <> absqsq = ${Atroe_str} * ${Atroe_str} + \
                    ${Btroe_str} * ${Btroe_str} {id=ab_init}
                absqsq = absqsq * absqsq {id=ab_fin, dep=ab_init}
                <> dFi = -2 * ${Atroe_str} * ${Btroe_str} * log(${Fcent_str}) \
                * (0.14 * ${Atroe_str} + ${Btroe_str}) * \
                (mod ${conp_theta_pr_fac}) / \
                (${Pr_str} * absqsq * logten) {id=dFi_final, dep=ab_fin}
            """).safe_substitute(**locals())
            parameters['logten'] = log(10)
        elif rxn_type == falloff_form.sri:
            X_lp, X_str = mapstore.apply_maps(namestore.X_sri, *default_inds)
            a_lp, a_str = mapstore.apply_maps(namestore.sri_a, var_name)
            b_lp, b_str = mapstore.apply_maps(namestore.sri_b, var_name)
            c_lp, c_str = mapstore.apply_maps(namestore.sri_c, var_name)
            kernel_data.extend([X_lp, a_lp, b_lp, c_lp])
            pre_instructions.append(
                ic.default_pre_instructs('Tval', T_str, 'VAL'))
            pre_instructions.append(
                ic.default_pre_instructs('Tinv', T_str, 'INV'))
            manglers.append(lp_pregen.fmax())

            sri_fac = (Template("""\
                log((${a_str} * exp(Tval * cinv) + exp(${b_str} * Tinv)) * \
                     exp(-${b_str} * Tinv - Tval * cinv))\
                """) if conp else Template("""\
                log(${a_str} * exp(-${b_str} * Tinv) + exp(-Tval * cinv))\
                """)).safe_substitute(
                **locals())
            dFi_instructions = Template("""
                <> cinv = 1 / ${c_str}
                <> dFi = -2 * ${X_str} * ${X_str} * (\
                    mod ${conp_theta_pr_fac}) * ${sri_fac} * \
                    log(fmax(${Pr_str}, 1e-300d)) / \
                    (fmax(${Pr_str}, 1e-300d) * logtensquared) {id=dFi_final}
            """).safe_substitute(**locals())
            parameters['logtensquared'] = log(10) * log(10)
        else:
            dFi_instructions = '<> dFi = 0 {id=dFi_final}'

        # the theta_Pr term that appears inside the pressure modification
        # term.  We simplify it to this to put in the falloff (non-chemically
        # activated) if statement
        conp_theta_pr_outer_fac = Template(
            '- not_unity * ${pres_mod_str}').safe_substitute(
            **locals()) if conp else ''
        # change the sign on this factor
        conp_theta_pr_fac = Template(' + ${Pr_str} * not_unity').safe_substitute(
            **locals()) if conp else ''
        fall_finish = '{} * rop_net '.format(V_str) if not conp else \
            ' rop_net'
        Pfac = ' * ' + P_str if conp else ''
        fall_instructions = Template("""
        ${fall_instructions}
        if ${fall_type_str}
            # chemically activated
            <>kf_0 = ${kf_str} {id=kf_chem}
            <>kf_inf = ${kf_fall_str} {id=kf_inf_chem}
        else
            # fall-off
            kf_0 = ${kf_fall_str} {id=kf_fall}
            kf_inf = ${kf_str} {id=kf_inf_fall}
        end
        mod = mod${Pfac} * rt_inv * kf_0 / \
            kf_inf {id=mod_final, dep=kf*:mod_mix:mod_spec}
        ${dFi_instructions}
        <> dci_fall_dE = ${pres_mod_str} * \
            ((-mod${conp_theta_pr_fac})/ (${Pr_str} + 1) + \
                dFi) {id=dci_fall_init}
        if not ${fall_type_str}
            # falloff
            dci_fall_dE = dci_fall_dE + ${Fi_str} * mod / (${Pr_str} + 1) \
                ${conp_theta_pr_outer_fac} {id=dci_fall_up1, dep=dci_fall_init}
        end
        dci_fall_dE = dci_fall_dE * ${fall_finish} \
            {id=dci_fall_final, dep=dci_fall_up1}
        """).safe_substitute(**locals())

    rop_net_rev_update = ic.get_update_instruction(
        mapstore, namestore.rop_rev,
        Template(
            'rop_net = rop_net - ${rop_rev_str} \
                        {id=rop_net_up, dep=rop_net_init}').safe_substitute(
            **locals()))

    # need to update mod term by subtracting off either pres_mod or Pr
    # for conp-third body or falloff respectively
    fall_deps = ':kf*' if rxn_type != reaction_type.thd else ''
    mod_update = Template("""
    if ${thd_type_str} != ${unity}
        mod = mod * ${P_str} * rt_inv$ - ${pres_mod_str} \
            {id=mod_up, dep=mod_mix:mod_spec${fall_deps}}
    end
    """).safe_substitute(**locals()) if conp and rxn_type == reaction_type.thd\
        else ''

    # mod term initialization depends on conp / conv
    mod_init = Template("""
    <> mod = ${thd_type_str} == ${mix} {id=mod_init}
    """).safe_substitute(**locals()) if conp else (
        '<> mod = 1 {id=mod_init}')

    # set the third body factor for the reactions (thd only)
    thd_factor_set = Template(
        '<> dci_thd_dE = mod${fac_term} \
        {id=dci_thd_init, dep=mod*:rop_net*}').safe_substitute(
        **locals()) if rxn_type == reaction_type.thd else ''

    # set up third body only instruction
    # (which get the Pr deriv for falloff)
    thd_mod_insns = Template("""
    ${mod_init}
    if ${thd_type_str} == ${mix} and ${thd_spec_last_str} == ${ns}
        mod = ${thd_eff_last_str} {id=mod_mix, dep=mod_init}
    end
    if ${thd_type_str} == ${spec}
        mod = ${thd_spec_last_str} == ${ns} {id=mod_spec, dep=mod_init}
    end
    ${mod_update}
    ${thd_factor_set}
    """).safe_substitute(**locals())

    # and instructions
    instructions = Template(Template("""
    <> rop_net = ${rop_fwd_str} {id=rop_net_init}
    ${rop_net_rev_update}
    ${thd_mod_insns}
    ${fall_instructions}
    <> offset = ${offset_str}
    <> offset_next = ${offset_next_str}
    for ${k_ind}
        if ${spec_k_str} != ${ns}
            ${jac_update_insn}
        end
    end
    """).safe_substitute(**locals())).safe_substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'])

    return k_gen.knl_info(name='dci_{}_dE'.format(
        name_description[rxn_type]),
        extra_inames=extra_inames,
        instructions=instructions,
        pre_instructions=pre_instructions,
        var_name=var_name,
        kernel_data=kernel_data,
        mapstore=mapstore,
        parameters=parameters,
        manglers=manglers,
        can_vectorize=can_vectorize,
        vectorization_specializer=vec_spec
    )


def dci_thd_dE(eqs, loopy_opts, namestore, test_size=None,
               conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term of third body reactions
    w.r.t. the extra variable (volume / pressure) for constant pressure/volume
    respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dcidE`

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidE(eqs, loopy_opts, namestore, test_size,
                                reaction_type.thd, conp=conp)]
            if x is not None]


def dci_lind_dE(eqs, loopy_opts, namestore, test_size=None,
                conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term of Lindemann falloff
    reactions w.r.t. the extra variable (volume / pressure)
    for constant pressure/volume respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dcidE`

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidE(eqs, loopy_opts, namestore, test_size,
                                falloff_form.lind, conp=conp)]
            if x is not None]


def dci_troe_dE(eqs, loopy_opts, namestore, test_size=None,
                conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term of Troe falloff
    reactions w.r.t. the extra variable (volume / pressure)
    for constant pressure/volume respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dcidE`

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidE(eqs, loopy_opts, namestore, test_size,
                                falloff_form.troe, conp=conp)]
            if x is not None]


def dci_sri_dE(eqs, loopy_opts, namestore, test_size=None,
               conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term of SRI falloff
    reactions w.r.t. the extra variable (volume / pressure)
    for constant pressure/volume respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dcidE`

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidE(eqs, loopy_opts, namestore, test_size,
                                falloff_form.sri, conp=conp)]
            if x is not None]


def __dRopidE(eqs, loopy_opts, namestore, test_size=None,
              do_ns=False, rxn_type=reaction_type.elementary, maxP=None,
              maxT=None, conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress (for all reaction types)
    with respect to the extra variable -- Volume / Pressure, depending on
    constant pressure / volume accordingly


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
    do_ns : bool [False]
        If True, generate kernel to handle derivatives of the last species'
        concentration w.r.t. temperature
    rxn_type : [reaction_type.thd, reaction_type.plog, reaction_type.cheb]
        The type of reaction to generate fore
    maxP: int [None]
        The maximum number of pressure interpolations of any reaction in
        the mechanism.
        - For PLOG - the maximum number of pressure interpolations of any
        reaction in the mechanism.
        - For CHEB - The maximum degree of temperature polynomials for
        chebyshev reactions in this mechanism
    maxT : int [None]
        The maximum degree of temperature polynomials for chebyshev reactions
        in this mechanism
    conp : bool [True]
        If supplied, True for constant pressure jacobian. False for constant
        volume [Default: True]

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # indicies
    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # check rxn type
    if rxn_type in [reaction_type.plog, reaction_type.cheb] and do_ns:
        return None  # this is handled by do_ns w / elementary
    elif rxn_type == reaction_type.plog:
        assert maxP, ('The maximum # of pressure interpolations must be'
                      ' supplied')
    elif rxn_type == reaction_type.cheb:
        assert maxP, ('The maximum chebyshev pressure polynomial degree must '
                      ' be supplied')
        assert maxT, ('The maximum chebyshev temperature polynomial degree '
                      ' must be supplied')

    num_range_dict = {reaction_type.elementary: namestore.num_simple,
                      reaction_type.plog: namestore.num_plog,
                      reaction_type.cheb: namestore.num_cheb}
    # get num
    num_range = num_range_dict[rxn_type] if not do_ns else\
        namestore.num_rxn_has_ns

    # return an empty kernel if it doesn't apply
    if num_range is None or not num_range.initializer.size:
        return None

    # number of species
    ns = namestore.num_specs[-1]

    mapstore = arc.MapStore(loopy_opts, num_range, num_range)

    rxn_range_dict = {reaction_type.elementary: namestore.simple_map,
                      reaction_type.plog: namestore.plog_map,
                      reaction_type.cheb: namestore.cheb_map}

    # add map
    rxn_range = rxn_range_dict[rxn_type] if not do_ns else\
        namestore.rxn_has_ns
    mapstore.check_and_add_transform(rxn_range, num_range)

    # rev mask depends on actual reaction index
    mapstore.check_and_add_transform(namestore.rev_mask, rxn_range)
    # thd_mask depends on actual reaction index
    mapstore.check_and_add_transform(namestore.thd_mask, rxn_range)
    # pres mod is on thd_mask
    mapstore.check_and_add_transform(
        namestore.pres_mod, namestore.thd_mask)
    # nu's are on the actual rxn index
    mapstore.check_and_add_transform(
        namestore.rxn_to_spec_offsets, rxn_range)

    # specific transforms
    if not do_ns:
        # fwd ROP is on actual rxn index
        mapstore.check_and_add_transform(namestore.rop_fwd, rxn_range)
        # rev ROP is on rev mask
        mapstore.check_and_add_transform(namestore.rop_rev, namestore.rev_mask)
    else:
        # kf is on real index
        mapstore.check_and_add_transform(namestore.kf, rxn_range)
        # kr is on rev rxn index
        mapstore.check_and_add_transform(namestore.kr, namestore.rev_mask)

    # extra inames
    net_ind = 'net_ind'
    k_ind = 'k_ind'

    # common variables
    # temperature
    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    # Volume
    V_lp, V_str = mapstore.apply_maps(namestore.V_arr, global_ind)
    # pressure mod term
    pres_mod_lp, pres_mod_str = mapstore.apply_maps(
        namestore.pres_mod, *default_inds)
    # nu offsets
    nu_offset_lp, nu_offset_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name)
    _, nu_offset_next_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name, affine=1)
    # reac and prod nu for net
    nu_lp, net_reac_nu_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, net_ind, affine=net_ind)
    _, net_prod_nu_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, net_ind, affine=net_ind)
    # get species
    spec_lp, spec_str = mapstore.apply_maps(
        namestore.rxn_to_spec, net_ind)
    _, spec_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec, k_ind)
    # and jac
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, spec_k_str, 1, affine={spec_k_str: 2})

    # add to data
    kernel_data.extend([T_lp, V_lp, pres_mod_lp, nu_offset_lp, nu_lp, spec_lp,
                        jac_lp])

    extra_inames = [
        (net_ind, 'offset <= {} < offset_next'.format(net_ind)),
        (k_ind, 'offset <= {} < offset_next'.format(k_ind))]
    parameters = {}
    pre_instructions = []
    if not do_ns:
        if not conp and \
                rxn_type not in [reaction_type.plog, reaction_type.cheb]:
            # the non-NS portion of these reactions is zero by definition
            return None
        # rop's & pres mod
        rop_fwd_lp, rop_fwd_str = mapstore.apply_maps(
            namestore.rop_fwd, *default_inds)
        rop_rev_lp, rop_rev_str = mapstore.apply_maps(
            namestore.rop_rev, *default_inds)

        kernel_data.extend([rop_fwd_lp, rop_rev_lp])

        if conp:
            # handle updates for dRopi_dE's
            rev_update = ic.get_update_instruction(
                mapstore, namestore.rop_rev,
                Template(
                    'dRopi_dE = dRopi_dE + nu_rev * ${rop_rev_str} \
                {id=dE_update, dep=dE_init}').safe_substitute(**locals()))

            deps = ':'.join(['dE_update'] if rev_update else [])
            pres_mod_update = ''
            if rxn_type not in [reaction_type.plog, reaction_type.cheb]:
                pres_mod_update = ic.get_update_instruction(
                    mapstore, namestore.pres_mod,
                    Template(
                        'dRopi_dE = dRopi_dE * ${pres_mod_str} \
                    {id=dE_final, dep=${deps}}').safe_substitute(**locals()))

            # if conp, need to include the fwd / reverse ROP an extra time
            # to account for the qi term resulting from d/dV (qi * V)
            # the negative is to account for the difference in signs between
            # this and the Rop * sum(nu) terms
            start = -1 if conp else 0
            # all constant pressure cases are the same (Rop * sum of nu)
            instructions = Template("""
                <> nu_fwd = ${start}
                <> nu_rev = ${start}
                for ${net_ind}
                    nu_fwd = nu_fwd + ${net_reac_nu_str} {id=nuf_up}
                    nu_rev = nu_rev + ${net_prod_nu_str} {id=nur_up}
                end
                <> dRopi_dE = -nu_fwd * ${rop_fwd_str} {id=dE_init, dep=nu*}
                ${rev_update}
                ${pres_mod_update}
                """).safe_substitute(**locals())
        else:
            # conv
            if rxn_type == reaction_type.plog:
                # conp & plog
                lo_ind = 'lo'
                hi_ind = 'hi'
                param_ind = 'p'
                # create extra arrays
                P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)
                # number of plog rates per rxn
                plog_num_param_lp, plog_num_param_str = mapstore.apply_maps(
                    namestore.plog_num_param, var_name)
                # pressure ranges
                plog_params_lp, pressure_mid_lo = mapstore.apply_maps(
                    namestore.plog_params, 0, var_name, param_ind)
                _, pressure_mid_hi = mapstore.apply_maps(
                    namestore.plog_params, 0, var_name, param_ind,
                    affine={param_ind: 1})
                _, pres_lo_str = mapstore.apply_maps(
                    namestore.plog_params, 0, var_name, lo_ind)
                _, pres_hi_str = mapstore.apply_maps(
                    namestore.plog_params, 0, var_name, hi_ind)
                # arrhenius params
                _, A_lo_str = mapstore.apply_maps(
                    namestore.plog_params, 1, var_name, lo_ind)
                _, A_hi_str = mapstore.apply_maps(
                    namestore.plog_params, 1, var_name, hi_ind)
                _, beta_lo_str = mapstore.apply_maps(
                    namestore.plog_params, 2, var_name, lo_ind)
                _, beta_hi_str = mapstore.apply_maps(
                    namestore.plog_params, 2, var_name, hi_ind)
                _, Ta_lo_str = mapstore.apply_maps(
                    namestore.plog_params, 3, var_name, lo_ind)
                _, Ta_hi_str = mapstore.apply_maps(
                    namestore.plog_params, 3, var_name, hi_ind)
                _, pressure_lo = mapstore.apply_maps(
                    namestore.plog_params, 0, var_name, 0)
                _, pressure_hi = mapstore.apply_maps(
                    namestore.plog_params, 0, var_name, 'numP')
                kernel_data.extend([P_lp, plog_num_param_lp, plog_params_lp])

                # add plog instruction
                pre_instructions.extend([ic.default_pre_instructs(
                    'logP', P_str, 'LOG'), ic.default_pre_instructs(
                    'logT', T_str, 'LOG'), ic.default_pre_instructs(
                    'Tinv', T_str, 'INV')])

                # and dkf instructions
                dkf_instructions = Template("""
                    <> lo = 0 {id=lo_init}
                    <> hi = numP {id=hi_init}
                    <> numP = ${plog_num_param_str} - 1
                    for ${param_ind}
                        if ${param_ind} <= numP and \
                                (logP > ${pressure_mid_lo}) and \
                                (logP <= ${pressure_mid_hi})
                            lo = ${param_ind} {id=set_lo, dep=lo_init}
                            hi = ${param_ind} + 1 {id=set_hi, dep=hi_init}
                        end
                    end
                    <> dkf = 0 {id=dkf_init}
                    # not out of range
                    if logP > ${pressure_lo} and logP <= ${pressure_hi}
                        dkf = (${A_hi_str} - ${A_lo_str} + \
                            logT * (${beta_hi_str} - ${beta_lo_str}) - \
                            (${Ta_hi_str} - ${Ta_lo_str}) * Tinv) / \
                            (${P_str} * (${pres_hi_str} - ${pres_lo_str})) \
                            {id=dkf_final, dep=dkf_init:set_*}
                    end
                """).safe_substitute(**locals())
                extra_inames.append((
                    param_ind, '0 <= {} < {}'.format(param_ind, maxP - 1)))
            elif rxn_type == reaction_type.cheb:
                # conp & cheb
                # max degrees in mechanism
                # derivative by P decreases pressure poly degree by 1
                poly_max = int(max(maxP - 1, maxT))

                # extra inames
                pres_poly_ind = 'k'
                temp_poly_ind = 'm'
                poly_compute_ind = 'p'
                lim_ind = 'dummy'
                # derivative by P decreases pressure poly degree by 1
                extra_inames.extend([
                    (pres_poly_ind, '0 <= {} < {}'.format(
                        pres_poly_ind, maxP - 1)),
                    (temp_poly_ind, '0 <= {} < {}'.format(
                        temp_poly_ind, maxT)),
                    (poly_compute_ind, '2 <= {} < {}'.format(
                        poly_compute_ind, poly_max))])

                # create arrays

                num_P_lp, num_P_str = mapstore.apply_maps(
                    namestore.cheb_numP, var_name)
                num_T_lp, num_T_str = mapstore.apply_maps(
                    namestore.cheb_numT, var_name)
                # derivative by P forces us to look 1 over in the pressure
                # polynomial index
                params_lp, params_str = mapstore.apply_maps(
                    namestore.cheb_params, var_name, temp_poly_ind,
                    pres_poly_ind,
                    affine={pres_poly_ind: 1})
                plim_lp, _ = mapstore.apply_maps(
                    namestore.cheb_Plim, var_name, lim_ind)
                tlim_lp, _ = mapstore.apply_maps(
                    namestore.cheb_Tlim, var_name, lim_ind)

                # workspace vars are based only on their polynomial indicies
                pres_poly_lp, ppoly_str = mapstore.apply_maps(
                    namestore.cheb_pres_poly, pres_poly_ind)
                temp_poly_lp, tpoly_str = mapstore.apply_maps(
                    namestore.cheb_temp_poly, temp_poly_ind)

                # create temperature and pressure arrays
                P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)

                kernel_data.extend([params_lp, num_P_lp, num_T_lp, plim_lp,
                                    P_lp, tlim_lp, pres_poly_lp, temp_poly_lp])

                # preinstructions
                pre_instructions.extend(
                    [ic.default_pre_instructs('logP', P_str, 'LOG'),
                     ic.default_pre_instructs('Tinv', T_str, 'INV')])

                # various strings for preindexed limits, params, etc
                _, Pmin_str = mapstore.apply_maps(
                    namestore.cheb_Plim, var_name, '0')
                _, Pmax_str = mapstore.apply_maps(
                    namestore.cheb_Plim, var_name, '1')
                _, Tmin_str = mapstore.apply_maps(
                    namestore.cheb_Tlim, var_name, '0')
                _, Tmax_str = mapstore.apply_maps(
                    namestore.cheb_Tlim, var_name, '1')

                # the various indexing for the pressure / temperature
                # polynomials
                _, ppoly0_str = mapstore.apply_maps(
                    namestore.cheb_pres_poly, '0')
                _, ppoly1_str = mapstore.apply_maps(
                    namestore.cheb_pres_poly, '1')
                _, ppolyp_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                    poly_compute_ind)
                _, ppolypm1_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                      poly_compute_ind,
                                                      affine=-1)
                _, ppolypm2_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                      poly_compute_ind,
                                                      affine=-2)
                _, tpoly0_str = mapstore.apply_maps(
                    namestore.cheb_temp_poly, '0')
                _, tpoly1_str = mapstore.apply_maps(
                    namestore.cheb_temp_poly, '1')
                _, tpolyp_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                    poly_compute_ind)
                _, tpolypm1_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                      poly_compute_ind,
                                                      affine=-1)
                _, tpolypm2_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                      poly_compute_ind,
                                                      affine=-2)

                dkf_instructions = Template("""
                    <>numP = ${num_P_str} - 1 {id=plim} # derivative by P
                    <>numT = ${num_T_str} {id=tlim}
                    <> Tred = (2 * Tinv - ${Tmax_str}- ${Tmin_str}) / \
                        (${Tmax_str} - ${Tmin_str})
                    <> Pred = (2 * logP - ${Pmax_str} - ${Pmin_str}) / \
                        (${Pmax_str} - ${Pmin_str})
                    ${ppoly0_str} = 1
                    ${ppoly1_str} = 2 * Pred # derivative by P
                    ${tpoly0_str} = 1
                    ${tpoly1_str} = Tred

                    # compute polynomial terms
                    for p
                        if p < numP
                            ${ppolyp_str} = 2 * Pred * ${ppolypm1_str} - \
                                ${ppolypm2_str} {id=ppoly, dep=plim}
                        end
                        if p < numT
                            ${tpolyp_str} = 2 * Tred * ${tpolypm1_str} - \
                                ${tpolypm2_str} {id=tpoly, dep=tlim}
                        end
                    end

                    <> dkf = 0 {id=dkf_init}
                    for m
                        <>temp = 0
                        for k
                            # derivative by P
                            temp = temp + (k + 1) * ${ppoly_str} * \
                                ${params_str} {id=temp, dep=ppoly:tpoly}
                        end
                        dkf = dkf + ${tpoly_str} * temp \
                            {id=dkf_update, dep=temp:dkf_init}
                    end
                    dkf = dkf * 2 * logten / (${P_str} * \
                        (${Pmax_str} - ${Pmin_str})) {id=dkf, dep=dkf_update}
                """).safe_substitute(**locals())
                parameters['logten'] = log(10)

            rev_update = ic.get_update_instruction(
                mapstore, namestore.rop_rev,
                Template(
                    'dRopi_dE = dRopi_dE - ${rop_rev_str} \
                {id=dE_update, dep=dE_init}').safe_substitute(**locals()))

            rev_dep = 'dE_update:' if rev_update else ''

            # and put together instructions
            instructions = Template("""
            ${dkf_instructions}
            <> dRopi_dE = ${rop_fwd_str} {id=dE_init}
            ${rev_update}
            dRopi_dE = dRopi_dE * dkf * ${V_str} \
                {id=dE_final, dep=${rev_dep}dkf_*}
            """).safe_substitute(**locals())
    else:
        # create kf / kr
        kf_lp, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)
        kr_lp, kr_str = mapstore.apply_maps(namestore.kr, *default_inds)
        P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)
        conc_lp, conc_str = mapstore.apply_maps(
            namestore.conc_arr, global_ind, 'net_spec')

        if conp:
            pre_instructions.append(Template(
                '<>fac = ${P_str} / (Ru * ${T_str})'
            ).safe_substitute(**locals()))
        else:
            pre_instructions.append(Template(
                '<>fac = ${V_str} / (Ru * ${T_str})'
            ).safe_substitute(**locals()))

        # create Ns nu's
        _, ns_reac_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_reac_nu, 'offset_next',
            affine='offset_next - 2')
        _, ns_prod_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_prod_nu, 'offset_next',
            affine='offset_next - 2')

        kernel_data.extend([kf_lp, kr_lp, P_lp, conc_lp])
        parameters['Ru'] = chem.RU

        rev_update = ic.get_update_instruction(
            mapstore, namestore.kr,
            Template(
                'kr_i = ${kr_str} \
                    {id=kr_up, dep=kr_in}').safe_substitute(**locals()))

        pres_mod_update = ''
        if rxn_type not in [reaction_type.plog, reaction_type.cheb]:
            pres_mod_update = ic.get_update_instruction(
                mapstore, namestore.pres_mod,
                Template('ci = ${pres_mod_str} \
                        {id=ci, dep=ci_init}').safe_substitute(
                    **locals()),)

        instructions = Template("""
        <> kr_i = 0 {id=kr_in}
        ${rev_update}
        <> ci = 1 {id=ci_init}
        ${pres_mod_update}
        <> Sns_fwd = ${ns_reac_nu_str} {id=Sns_fwd_init}
        <> Sns_rev = ${ns_prod_nu_str} {id=Sns_rev_init}
        for ${net_ind}
            <> nu_fwd = ${net_reac_nu_str} {id=nuf_inner}
            <> nu_rev = ${net_prod_nu_str} {id=nur_inner}
            <> net_spec = ${spec_str}
            # handle nu
            if net_spec == ${ns}
                nu_fwd = nu_fwd - 1 {id=nuf_inner_up, dep=nuf_inner}
            end
            Sns_fwd = Sns_fwd * fast_powi(${conc_str}, nu_fwd) \
                {id=Sns_fwd_up, dep=nuf_inner_up}
            if net_spec == ${ns}
                nu_rev = nu_rev - 1 {id=nur_inner_up, dep=nur_inner}
            end
            Sns_rev = Sns_rev * fast_powi(${conc_str}, nu_rev) \
                {id=Sns_rev_up, dep=nur_inner_up}
        end
        <> dRopi_dE = (Sns_fwd * ${kf_str} - Sns_rev * kr_i) * ci \
            * fac {id=dE_final, dep=Sns*}
        """).substitute(**locals())

    # get nuk's
    _, reac_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, k_ind, affine=k_ind)
    _, prod_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, k_ind, affine=k_ind)
    instructions = Template("""
        <> offset = ${nu_offset_str}
        <> offset_next = ${nu_offset_next_str}
        ${instructions}
        for ${k_ind}
            if ${spec_k_str} != ${ns}
                ${jac_str} = ${jac_str} + (${prod_nu_k_str} - ${reac_nu_k_str}) \
                    * dRopi_dE {id=jac, dep=dE*}
            end
        end
    """).substitute(**locals())

    name_description = {reaction_type.elementary: '',
                        reaction_type.plog: '_plog',
                        reaction_type.cheb: '_cheb'}

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'])
    return k_gen.knl_info(name='dRopi{}_d{}{}'.format(
        name_description[rxn_type],
        'V' if conp else 'P',
        '_ns' if do_ns else ''),
        extra_inames=extra_inames,
        instructions=instructions,
        pre_instructions=pre_instructions,
        var_name=var_name,
        kernel_data=kernel_data,
        mapstore=mapstore,
        preambles=[lp_pregen.fastpowi_PreambleGen(),
                   lp_pregen.fastpowf_PreambleGen()],
        parameters=parameters,
        can_vectorize=can_vectorize,
        vectorization_specializer=vec_spec
    )


def dRopidE(eqs, loopy_opts, namestore, test_size=None, conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress (for non-pressure dependent reaction
    types) with respect to the extra variable -- volume/pressure for constant
    volume / pressure respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dRopidE`

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dRopidE(eqs, loopy_opts, namestore,
                                  test_size=test_size, do_ns=False,
                                  conp=conp),
                        __dRopidE(eqs, loopy_opts, namestore,
                                  test_size=test_size, do_ns=True,
                                  conp=conp)]
            if x is not None]


def dRopi_plog_dE(eqs, loopy_opts, namestore, test_size=None, conp=True,
                  maxP=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress (for PLOG reactions)
    with respect to the extra variable -- volume/pressure for constant
    volume / pressure respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dRopidE`

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
    maxP: int [None]
        The maximum number of pressure interpolations of any reaction in
        the mechanism.

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    ret = [__dRopidE(eqs, loopy_opts, namestore,
                     test_size=test_size, do_ns=False,
                     rxn_type=reaction_type.plog, conp=conp,
                     maxP=maxP)]
    if test_size == 'problem_size':
        # include the ns version for convenience in testing
        ret.append(__dRopidE(eqs, loopy_opts, namestore,
                             test_size=test_size, do_ns=True,
                             rxn_type=reaction_type.plog, conp=conp,
                             maxP=maxP))
    return [x for x in ret if x is not None]


def dRopi_cheb_dE(eqs, loopy_opts, namestore, test_size=None, conp=True,
                  maxP=None, maxT=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress (for CHEB reactions)
    with respect to the extra variable -- volume/pressure for constant
    volume / pressure respectively

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dRopidE`

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
    maxP : int [None]
        The maximum degree of pressure polynomials for chebyshev reactions in
        this mechanism
    maxT : int [None]
        The maximum degree of temperature polynomials for chebyshev reactions
        in this mechanism

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    ret = [__dRopidE(eqs, loopy_opts, namestore,
                     test_size=test_size, do_ns=False,
                     rxn_type=reaction_type.cheb, conp=conp,
                     maxP=maxP, maxT=maxT)]
    if test_size == 'problem_size':
        # include the ns version for convenience in testing
        ret.append(__dRopidE(eqs, loopy_opts, namestore,
                             test_size=test_size, do_ns=True,
                             rxn_type=reaction_type.cheb, conp=conp,
                             maxP=maxP, maxT=maxT))
    return [x for x in ret if x is not None]


def dTdotdE(eqs, loopy_opts, namestore, test_size, conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of change of temperature with respect to
    the extra variable (volume/pressure for const. pressure / volume
    respectively)


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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # indicies
    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    mapstore = arc.MapStore(
        loopy_opts, namestore.num_specs_no_ns, namestore.num_specs_no_ns)

    # create arrays
    T_lp, T_str = mapstore.apply_maps(
        namestore.T_arr, global_ind)
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)

    # specific heats and energies
    spec_heat_tot_lp, spec_heat_total_str = mapstore.apply_maps(
        namestore.spec_heat_total, global_ind)
    spec_heat_lp, spec_heat_str = mapstore.apply_maps(
        namestore.spec_heat, *default_inds)
    _, spec_heat_ns_str = mapstore.apply_maps(
        namestore.spec_heat_ns, global_ind)
    spec_energy_lp, spec_energy_str = mapstore.apply_maps(
        namestore.spec_energy, *default_inds)
    _, spec_energy_ns_str = mapstore.apply_maps(
        namestore.spec_energy_ns, global_ind)

    # concs
    conc_lp = None
    if conp:
        conc_lp, conc_str = mapstore.apply_maps(
            namestore.conc_arr, *default_inds)

    # rates
    wdot_lp, wdot_str = mapstore.apply_maps(
        namestore.spec_rates, *default_inds)
    Tdot_lp, Tdot_str = mapstore.apply_maps(
        namestore.T_dot, global_ind)

    # molecular weights
    mw_lp, mw_str = mapstore.apply_maps(
        namestore.mw_post_arr, var_name)

    # jacobian entries
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, 0, 1)
    _, dnkdot_de_str = mapstore.apply_maps(
        namestore.jac, global_ind, var_name, 1, affine={var_name: 2})

    parameters = {}
    kernel_data.extend([x for x in [
        spec_heat_tot_lp, spec_heat_lp, spec_energy_lp, wdot_lp, Tdot_lp,
        mw_lp, jac_lp, conc_lp, T_lp, V_lp] if x is not None])
    if conp:
        pre_instructions = ['<> dTsum = 0',
                            '<> specsum = 0']
        instructions = Template("""
            specsum = specsum + (${spec_energy_str} - ${spec_energy_ns_str} * \
                ${mw_str}) * (${dnkdot_de_str} - ${wdot_str}) {id=up, dep=*}
            dTsum = dTsum + (${spec_heat_str} - ${spec_heat_ns_str}) * \
                ${conc_str} {id=up2, dep=*}
        """).safe_substitute(**locals())
        post_instructions = [Template("""
            <> spec_inv = 1 / (${spec_heat_total_str} * ${V_str})
            ${jac_str} = ${jac_str} + (${Tdot_str} * dTsum - specsum) * spec_inv \
                {id=jac, dep=up*, nosync=up*}
            """).safe_substitute(**locals())]
    else:
        parameters['Ru'] = chem.RU
        pre_instructions = ['<> sum = 0',
                            ic.default_pre_instructs('Vinv', V_str, 'INV')]
        instructions = Template("""
            sum = sum + (${spec_energy_str} - ${spec_energy_ns_str} * \
                ${mw_str}) * ${dnkdot_de_str} {id=up, dep=*}
        """).safe_substitute(**locals())
        post_instructions = [Template("""
            <>spec_inv = 1 / (${spec_heat_total_str})
            ${jac_str} = ${jac_str} - (${Tdot_str} * ${spec_heat_ns_str} \
                / (Ru * ${T_str})) * spec_inv {id=jac_split}
            ${jac_str} = ${jac_str} - (sum * Vinv) * spec_inv \
                {id=jac, dep=up, nosync=up}
            """).safe_substitute(**locals())]

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'], split_ids=['jac_split'])
    return k_gen.knl_info(name='dTdotd{}'.format('V' if conp else 'P'),
                          instructions=instructions,
                          pre_instructions=pre_instructions,
                          post_instructions=post_instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          parameters=parameters,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec
                          )


def dEdotdE(eqs, loopy_opts, namestore, test_size, conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of change of volume / pressure
    with respect to the extra variable (volume/pressure for const. pressure /
    volume respectively)

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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # indicies
    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    mapstore = arc.MapStore(
        loopy_opts, namestore.net_nonzero_spec, namestore.net_nonzero_spec)

    # create arrays
    T_lp, T_str = mapstore.apply_maps(
        namestore.T_arr, global_ind)
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)
    P_lp, P_str = mapstore.apply_maps(
        namestore.P_arr, global_ind)

    # jacobian entries
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, 1, 1)
    _, dnkdot_de_str = mapstore.apply_maps(
        namestore.jac, global_ind, var_name, 1, affine={var_name: 2})
    _, dTdot_de_str = mapstore.apply_maps(
        namestore.jac, global_ind, 0, 1)

    # rates
    Tdot_lp, Tdot_str = mapstore.apply_maps(
        namestore.T_dot, global_ind)

    # molecular weights
    mw_lp, mw_str = mapstore.apply_maps(
        namestore.mw_post_arr, var_name)

    kernel_data.extend([
        T_lp, V_lp, P_lp, jac_lp, Tdot_lp, mw_lp])

    var_str = V_str if conp else P_str
    param_str = P_str if conp else V_str

    pre_instructions = [Template("""
        <> sum = 0 {id=init}
        ... nop {id=index_dummy} # included to avoid non-existant dep check
        """).safe_substitute(**locals())]

    instructions = Template("""
        # hook the sum depenency onto any resulting index changes for the
        # net_non_zero sum
        sum = sum + (1 - ${mw_str}) * ${dnkdot_de_str} {id=up, dep=index*}
    """).safe_substitute(**locals())

    post_instructions = [Template("""
        ${jac_str} = ${jac_str} + Ru * ${T_str} * sum / ${param_str} \
            {id=jac, dep=up, nosync=up}
        ${jac_str} = ${jac_str} + (${var_str} * ${dTdot_de_str} + ${Tdot_str}) \
            / ${T_str} {id=jac_split}
    """).safe_substitute(**locals())]

    parameters = {'Ru': chem.RU}

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'], split_ids=['jac_split'])

    return k_gen.knl_info(name='d{0}dotd{0}'.format('V' if conp else 'P'),
                          instructions=instructions,
                          pre_instructions=pre_instructions,
                          post_instructions=post_instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec,
                          parameters=parameters
                          )


def dTdotdT(eqs, loopy_opts, namestore, test_size=None, conp=True):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of change of temprature with respect to
    temperature


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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # indicies
    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    ns = namestore.num_specs[-1]

    mapstore = arc.MapStore(
        loopy_opts, namestore.num_specs_no_ns, namestore.num_specs_no_ns)

    # Temperature
    T_lp, T_str = mapstore.apply_maps(
        namestore.T_arr, global_ind)

    # spec sum
    spec_heat_tot_lp, spec_heat_total_str = mapstore.apply_maps(
        namestore.spec_heat_total, global_ind)

    # dT/dt
    dTdt_lp, Tdot_str = mapstore.apply_maps(
        namestore.T_dot, global_ind)

    # spec heat
    spec_heat_lp, spec_heat_str = mapstore.apply_maps(
        namestore.spec_heat, *default_inds)

    # and derivative w.r.t T
    dspec_heat_lp, dspec_heat_str = mapstore.apply_maps(
        namestore.dspec_heat, *default_inds)

    # Ns derivative
    _, dspec_heat_ns_str = mapstore.apply_maps(
        namestore.dspec_heat, global_ind, ns)

    # last species spec heat
    _, spec_heat_ns_str = mapstore.apply_maps(
        namestore.spec_heat_ns, global_ind)

    # energy
    spec_energy_lp, spec_energy_str = mapstore.apply_maps(
        namestore.spec_energy, *default_inds)

    # last species energy
    _, spec_energy_ns_str = mapstore.apply_maps(
        namestore.spec_energy_ns, global_ind)

    # molecular weights
    mw_lp, mw_str = mapstore.apply_maps(
        namestore.mw_post_arr, var_name)

    # concentrations
    conc_lp, conc_str = mapstore.apply_maps(
        namestore.conc_arr, *default_inds)

    # last species concentration
    _, conc_ns_str = mapstore.apply_maps(
        namestore.conc_arr, global_ind, ns)

    # volume
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)

    # spec rates
    wdot_lp, wdot_str = mapstore.apply_maps(
        namestore.spec_rates, *default_inds)

    # and finally molar rates
    _, ndot_str = mapstore.apply_maps(
        namestore.jac, global_ind, var_name, 0, affine={var_name: 2})

    # jacobian entry
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, 0, 0)

    kernel_data.extend([spec_heat_tot_lp, dTdt_lp, spec_heat_lp, dspec_heat_lp,
                        spec_energy_lp, mw_lp, conc_lp, V_lp, wdot_lp,
                        jac_lp, T_lp])

    pre_instructions = Template("""
    <> dTsum = ((${spec_heat_ns_str} * Tinv - ${dspec_heat_ns_str}) \
        * ${conc_ns_str}) {id=split}
    <> rate_sum = 0
    """).safe_substitute(**locals()).split('\n')
    pre_instructions.extend([
        ic.default_pre_instructs('Vinv', V_str, 'INV'),
        ic.default_pre_instructs('Tinv', T_str, 'INV')])

    instructions = Template("""
        dTsum = dTsum + (${spec_heat_ns_str} * Tinv - ${dspec_heat_str}) \
            * ${conc_str}
        rate_sum = rate_sum + ${wdot_str} * \
            (-${spec_heat_str} + ${mw_str} * ${spec_heat_ns_str}) + \
            Vinv * ${ndot_str} * \
            (-${spec_energy_str} + ${spec_energy_ns_str} * ${mw_str}) \
                {id=rate_update, dep=*}
    """).safe_substitute(**locals())

    post_instructions = Template("""
        ${jac_str} = ${jac_str} + (${Tdot_str} * dTsum + rate_sum) \
            / ${spec_heat_total_str} {id=jac, dep=rate_update, nosync=rate_update}
    """).safe_substitute(**locals()).split('\n')

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'], split_ids=['split'])

    return k_gen.knl_info(name='dTdot_dT',
                          instructions=instructions,
                          pre_instructions=pre_instructions,
                          post_instructions=post_instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec,
                          )


def dEdotdT(eqs, loopy_opts, namestore, test_size=None, conp=False):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of change of the extra variable (volume/pressure
    for constant pressure/volume respectively) with respect to temperature


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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    ns = namestore.num_specs[-1]

    mapstore = arc.MapStore(
        loopy_opts, namestore.num_specs_no_ns, namestore.num_specs_no_ns)

    # create arrays
    mw_lp, mw_str = mapstore.apply_maps(namestore.mw_post_arr, var_name)
    V_lp, V_str = mapstore.apply_maps(namestore.V_arr, global_ind)
    P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)
    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    jac_lp, jac_str = mapstore.apply_maps(namestore.jac, global_ind, 1, 0)
    _, dnkdot_dT_str = mapstore.apply_maps(
        namestore.jac, global_ind, var_name, 0, affine={var_name: 2})
    _, dTdot_dT_str = mapstore.apply_maps(namestore.jac, global_ind, 0, 0)
    wdot_lp, wdot_str = mapstore.apply_maps(
        namestore.spec_rates, *default_inds)
    dphi_lp, Tdot_str = mapstore.apply_maps(namestore.T_dot, global_ind)

    kernel_data.extend([mw_lp, V_lp, P_lp, T_lp, jac_lp, wdot_lp, dphi_lp])

    # instructions
    pre_instructions = ['<> sum = 0',
                        ic.default_pre_instructs('Tinv', T_str, 'INV')]
    if conp:
        pre_instructions.append(
            ic.default_pre_instructs('Vinv', V_str, 'INV'))
        # sums
        instructions = Template("""
            sum = sum + (1 - ${mw_str}) * (Vinv * ${dnkdot_dT_str} + Tinv * \
                ${wdot_str}) {id=sum, dep=*}
        """).safe_substitute(**locals())
        # sum finish
        post_instructions = [Template("""
            ${jac_str} = ${jac_str} + Ru * ${T_str} * ${V_str} * sum / ${P_str} \
                {id=jac, dep=sum, nosync=sum}
            ${jac_str} = ${jac_str} + ${V_str} * Tinv * \
                (${dTdot_dT_str} - Tinv * ${Tdot_str}) {id=jac_split}
        """).safe_substitute(**locals())]
    else:
        pre_instructions.append(Template(
            '<> fac = ${T_str} / ${V_str}').safe_substitute(**locals()))
        instructions = Template("""
            sum = sum + (1 - ${mw_str}) * (${dnkdot_dT_str} * fac + \
                ${wdot_str}) {id=sum, dep=*}
        """).safe_substitute(**locals())
        post_instructions = [Template("""
            ${jac_str} = ${jac_str} + Ru * sum {id=jac, nosync=sum, dep=sum}
            ${jac_str} = ${jac_str} + ${P_str} * \
                (${dTdot_dT_str} - ${Tdot_str} * Tinv) * Tinv {id=jac_split}
        """).safe_substitute(**locals())]

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'], split_ids=['jac_split'])

    parameters = {'Ru': chem.RU}
    return k_gen.knl_info(name='d{}dotdT'.format('V' if conp else 'P'),
                          instructions=instructions,
                          pre_instructions=pre_instructions,
                          post_instructions=post_instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          parameters=parameters,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec
                          )


def __dcidT(eqs, loopy_opts, namestore, test_size=None,
            rxn_type=reaction_type.thd):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term for all third body /
    falloff / chemically activated reactions with respect to temperature


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
    rxn_type: [reaction_type.thd, falloff_form.lind, falloff_form.sri,
               falloff_form.troe]
        The reaction type to generate the pressure modification derivative for

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    num_range_dict = {reaction_type.thd: namestore.num_thd_only,
                      falloff_form.lind: namestore.num_lind,
                      falloff_form.sri: namestore.num_sri,
                      falloff_form.troe: namestore.num_troe}
    thd_range_dict = {reaction_type.thd: namestore.thd_only_map,
                      falloff_form.lind: namestore.fall_to_thd_map,
                      falloff_form.sri: namestore.fall_to_thd_map,
                      falloff_form.troe: namestore.fall_to_thd_map}
    fall_range_dict = {falloff_form.lind: namestore.lind_map,
                       falloff_form.sri: namestore.sri_map,
                       falloff_form.troe: namestore.troe_map}
    name_description = {reaction_type.thd: 'thd',
                        falloff_form.lind: 'lind',
                        falloff_form.sri: 'sri',
                        falloff_form.troe: 'troe'}
    # get num
    num_range = num_range_dict[rxn_type]
    thd_range = thd_range_dict[rxn_type]
    rxn_range = namestore.thd_map

    # number of species
    ns = namestore.num_specs[-1]

    # create mapstore
    mapstore = arc.MapStore(loopy_opts, num_range, num_range)

    # setup static mappings

    if rxn_type in fall_range_dict:
        fall_range = fall_range_dict[rxn_type]

        # falloff index depends on num_range
        mapstore.check_and_add_transform(fall_range, num_range)

        # and the third body index depends on the falloff index
        mapstore.check_and_add_transform(thd_range, fall_range)

    # and finally the reaction index depends on the third body index
    mapstore.check_and_add_transform(rxn_range, thd_range)

    # setup third body stuff
    mapstore.check_and_add_transform(namestore.thd_type, thd_range)
    mapstore.check_and_add_transform(namestore.thd_offset, thd_range)

    # and place rop's / species maps, etc.
    mapstore.check_and_add_transform(namestore.rop_fwd, rxn_range)
    mapstore.check_and_add_transform(namestore.rev_mask, rxn_range)
    mapstore.check_and_add_transform(namestore.rop_rev, namestore.rev_mask)
    mapstore.check_and_add_transform(namestore.rxn_to_spec_offsets, rxn_range)

    if rxn_type != reaction_type.thd:
        # pressure mod term
        mapstore.check_and_add_transform(namestore.pres_mod, thd_range)
        # falloff type
        mapstore.check_and_add_transform(namestore.fall_type, fall_range)
        # need falloff blending / reduced pressure
        mapstore.check_and_add_transform(namestore.Fi, fall_range)
        mapstore.check_and_add_transform(namestore.Pr, fall_range)

        # in addition we (most likely) need the falloff / regular kf rates
        mapstore.check_and_add_transform(namestore.kf, rxn_range)
        mapstore.check_and_add_transform(namestore.kf_fall, fall_range)

        # and the beta / Ta parameters for falloff / regular kf
        mapstore.check_and_add_transform(namestore.fall_beta, fall_range)
        mapstore.check_and_add_transform(namestore.fall_Ta, fall_range)

        # the regular kf params require the simple_mask
        mapstore.check_and_add_transform(namestore.simple_mask, rxn_range)
        mapstore.check_and_add_transform(
            namestore.simple_beta, namestore.simple_mask)
        mapstore.check_and_add_transform(
            namestore.simple_Ta, namestore.simple_mask)

    # get the third body types
    thd_type_lp, thd_type_str = mapstore.apply_maps(
        namestore.thd_type, var_name)
    thd_offset_lp, thd_offset_next_str = mapstore.apply_maps(
        namestore.thd_offset, var_name, affine=1)
    # get third body efficiencies & species
    thd_eff_lp, thd_eff_last_str = mapstore.apply_maps(
        namestore.thd_eff, thd_offset_next_str, affine=-1)
    thd_spec_lp, thd_spec_last_str = mapstore.apply_maps(
        namestore.thd_spec, thd_offset_next_str, affine=-1)

    k_ind = 'k_ind'
    # nu offsets
    nu_offset_lp, offset_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name)
    _, offset_next_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name, affine=1)
    # setup species k loop
    extra_inames = [(k_ind, 'offset <= {} < offset_next'.format(k_ind))]

    # reac and prod nu for net
    nu_lp, reac_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, k_ind, affine=k_ind)
    _, prod_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, k_ind, affine=k_ind)
    # get species
    spec_lp, spec_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec, k_ind)
    # and jac
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, spec_k_str, 0, affine={spec_k_str: 2})

    # ropnet
    rop_fwd_lp, rop_fwd_str = mapstore.apply_maps(
        namestore.rop_fwd, *default_inds)
    rop_rev_lp, rop_rev_str = mapstore.apply_maps(
        namestore.rop_rev, *default_inds)

    # T, P, V
    T_lp, T_str = mapstore.apply_maps(
        namestore.T_arr, global_ind)
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)
    P_lp, P_str = mapstore.apply_maps(
        namestore.P_arr, global_ind)

    # update kernel data
    kernel_data.extend([thd_type_lp, thd_offset_lp, thd_eff_lp, thd_spec_lp,
                        nu_offset_lp, nu_lp, spec_lp, rop_fwd_lp, rop_rev_lp,
                        jac_lp, T_lp, V_lp, P_lp])

    pre_instructions = [ic.default_pre_instructs('Tinv', T_str, 'INV')]
    parameters = {}
    manglers = []
    # by default we are using the third body factors (these may be changed
    # in the falloff types below)
    factor = 'dci_thd_dT'
    thd_fac = ' * {} * rop_net '.format(V_str)
    fall_instructions = ''
    if rxn_type != reaction_type.thd:
        # update factors
        factor = 'dci_fall_dT'
        thd_fac = ''
        # create arrays
        pres_mod_lp, pres_mod_str = mapstore.apply_maps(
            namestore.pres_mod, *default_inds)
        fall_type_lp, fall_type_str = mapstore.apply_maps(
            namestore.fall_type, var_name)
        Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)
        Pr_lp, Pr_str = mapstore.apply_maps(namestore.Pr, *default_inds)
        kf_lp, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)
        s_beta_lp, s_beta_str = mapstore.apply_maps(
            namestore.simple_beta, var_name)
        s_Ta_lp, s_Ta_str = mapstore.apply_maps(
            namestore.simple_Ta, var_name)
        kf_fall_lp, kf_fall_str = mapstore.apply_maps(
            namestore.kf_fall, *default_inds)
        f_beta_lp, f_beta_str = mapstore.apply_maps(
            namestore.fall_beta, var_name)
        f_Ta_lp, f_Ta_str = mapstore.apply_maps(
            namestore.fall_Ta, var_name)

        kernel_data.extend([pres_mod_lp, fall_type_lp, Fi_lp, Pr_lp, kf_lp,
                            s_beta_lp, s_Ta_lp, kf_fall_lp, f_beta_lp,
                            f_Ta_lp])

        # check for Troe / SRI
        if rxn_type == falloff_form.troe:
            Atroe_lp, Atroe_str = mapstore.apply_maps(
                namestore.Atroe, *default_inds)
            Btroe_lp, Btroe_str = mapstore.apply_maps(
                namestore.Btroe, *default_inds)
            Fcent_lp, Fcent_str = mapstore.apply_maps(
                namestore.Fcent, *default_inds)
            troe_a_lp, troe_a_str = mapstore.apply_maps(
                namestore.troe_a, var_name)
            troe_T1_lp, troe_T1_str = mapstore.apply_maps(
                namestore.troe_T1, var_name)
            troe_T2_lp, troe_T2_str = mapstore.apply_maps(
                namestore.troe_T2, var_name)
            troe_T3_lp, troe_T3_str = mapstore.apply_maps(
                namestore.troe_T3, var_name)
            kernel_data.extend([Atroe_lp, Btroe_lp, Fcent_lp, troe_a_lp,
                                troe_T1_lp, troe_T2_lp, troe_T3_lp])
            pre_instructions.append(
                ic.default_pre_instructs('Tval', T_str, 'VAL'))
            dFi_instructions = Template("""
                <> dFcent = -${troe_a_str} * ${troe_T1_str} * \
                exp(-Tval * ${troe_T1_str}) + (${troe_a_str} - 1) * ${troe_T3_str} *\
                exp(-Tval * ${troe_T3_str}) + ${troe_T2_str} * Tinv * Tinv * \
                    exp(-${troe_T2_str} * Tinv)
                <> logFcent = log(${Fcent_str})
                <> absq = ${Atroe_str} * ${Atroe_str} + ${Btroe_str} * ${Btroe_str} \
                    {id=ab_init}
                <> absqsq = absq * absq {id=ab_fin}
                <> dFi = -${Btroe_str} * (2 * ${Atroe_str} * ${Fcent_str} * \
                (0.14 * ${Atroe_str} + ${Btroe_str}) * \
                (${Pr_str} * theta_Pr + theta_no_Pr) * logFcent + \
                ${Pr_str} * dFcent * (2 * ${Atroe_str} * \
                (1.1762 * ${Atroe_str} - 0.67 * ${Btroe_str}) * logFcent \
                - ${Btroe_str} * absq * logten)) / \
                (${Fcent_str} * ${Pr_str} * absqsq * logten) {id=dFi_final}
            """).safe_substitute(**locals())
            parameters['logten'] = log(10)
        elif rxn_type == falloff_form.sri:
            X_lp, X_str = mapstore.apply_maps(namestore.X_sri, *default_inds)
            a_lp, a_str = mapstore.apply_maps(namestore.sri_a, var_name)
            b_lp, b_str = mapstore.apply_maps(namestore.sri_b, var_name)
            c_lp, c_str = mapstore.apply_maps(namestore.sri_c, var_name)
            d_lp, d_str = mapstore.apply_maps(namestore.sri_d, var_name)
            e_lp, e_str = mapstore.apply_maps(namestore.sri_e, var_name)
            kernel_data.extend([X_lp, a_lp, b_lp, c_lp, d_lp, e_lp])
            pre_instructions.append(
                ic.default_pre_instructs('Tval', T_str, 'VAL'))
            manglers.append(lp_pregen.fmax())

            dFi_instructions = Template("""
                <> cinv = 1 / ${c_str}
                <> dFi = -${X_str} * (\
                exp(-Tval * cinv) * cinv - ${a_str} * ${b_str} * Tinv * \
                Tinv * exp(-${b_str} * Tinv)) / \
                (${a_str} * exp(-${b_str} * Tinv) + exp(-Tval * cinv)) \
                + ${e_str} * Tinv - \
                2 * ${X_str} * ${X_str} * \
                log(${a_str} * exp(-${b_str} * Tinv) + exp(-Tval * cinv)) * \
                (${Pr_str} * theta_Pr + theta_no_Pr) * \
                log(fmax(${Pr_str}, 1e-300d)) / \
                (fmax(${Pr_str}, 1e-300d) * logtensquared) {id=dFi_final}
            """).safe_substitute(**locals())
            parameters['logtensquared'] = log(10) * log(10)
        else:
            dFi_instructions = '<> dFi = 0 {id=dFi_final}'

        fall_instructions = Template("""
        if ${fall_type_str}
            # chemically activated
            <>kf_0 = ${kf_str} {id=kf_chem}
            <>beta_0 = ${s_beta_str} {id=beta0_chem}
            <>Ta_0 = ${s_Ta_str} {id=Ta0_chem}
            <>kf_inf = ${kf_fall_str} {id=kf_inf_chem}
            <>beta_inf = ${f_beta_str} {id=betaf_chem}
            <>Ta_inf = ${f_Ta_str} {id=Taf_chem}
        else
            # fall-off
            kf_0 = ${kf_fall_str} {id=kf_fall}
            beta_0 = ${f_beta_str} {id=beta0_fall}
            Ta_0 = ${f_Ta_str} {id=Ta0_fall}
            kf_inf = ${kf_str} {id=kf_inf_fall}
            beta_inf = ${s_beta_str} {id=betaf_fall}
            Ta_inf = ${s_Ta_str} {id=Taf_fall}
        end
        <> pmod = ${pres_mod_str}
        <> theta_Pr = Tinv * (beta_0 - beta_inf + (Ta_0 - Ta_inf) * Tinv) \
            {id=theta_Pr, dep=beta*:kf*:Ta*}
        <> theta_no_Pr = dci_thd_dT * kf_0 / kf_inf {id=theta_No_Pr, dep=kf*}
        ${dFi_instructions}
        <> dci_fall_dT = pmod * (-(${Pr_str} * theta_Pr + theta_no_Pr) / \
            (${Pr_str} + 1) + dFi) {id=dfall_init}
        if not ${fall_type_str}
            # falloff
            dci_fall_dT = dci_fall_dT + theta_Pr * pmod + ${Fi_str} * theta_no_Pr / \
                (${Pr_str} + 1) {id=dfall_up1, dep=dfall_init}
        end
        dci_fall_dT = dci_fall_dT * ${V_str} * rop_net \
            {id=dfall_final, dep=dfall_up1}
        """).safe_substitute(**locals())

    rop_net_rev_update = ic.get_update_instruction(
        mapstore, namestore.rop_rev,
        Template(
            'rop_net = rop_net - ${rop_rev_str} \
                        {id=rop_net_up, dep=rop_net_init}').safe_substitute(
            **locals()))
    mix = int(thd_body_type.mix)
    spec = int(thd_body_type.species)
    # and instructions
    instructions = Template("""
    <> rop_net = ${rop_fwd_str} {id=rop_net_init}
    ${rop_net_rev_update}
    <> mod = 1 {id=mod_init}
    if ${thd_type_str} == ${mix} and ${thd_spec_last_str} == ${ns}
        mod = ${thd_eff_last_str} {id=mod_mix, dep=mod_init}
    end
    if ${thd_type_str} == ${spec}
        mod = ${thd_spec_last_str} == ${ns} {id=mod_spec, dep=mod_init}
    end
    <> dci_thd_dT = -${P_str} * mod * Ru_inv * Tinv * Tinv${thd_fac} \
        {dep=mod*:rop_net*}
    ${fall_instructions}
    <> offset = ${offset_str}
    <> offset_next = ${offset_next_str}
    for ${k_ind}
        if ${spec_k_str} != ${ns}
            ${jac_str} = ${jac_str} + (${prod_nu_k_str} - ${reac_nu_k_str}) * \
                ${factor} {id=jac}
        end
    end
    """).safe_substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'])

    parameters.update({'Ru_inv': 1.0 / chem.RU})
    return k_gen.knl_info(name='dci_{}_dT'.format(
        name_description[rxn_type]),
        extra_inames=extra_inames,
        instructions=instructions,
        pre_instructions=pre_instructions,
        var_name=var_name,
        kernel_data=kernel_data,
        mapstore=mapstore,
        parameters=parameters,
        manglers=manglers,
        can_vectorize=can_vectorize,
        vectorization_specializer=vec_spec
    )


def dci_thd_dT(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term w.r.t. Temperature
    for third body reactions

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidT(eqs, loopy_opts, namestore, test_size,
                                reaction_type.thd)] if x is not None]


def dci_lind_dT(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term w.r.t. Temperature
    for Lindemann falloff reactions

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidT(eqs, loopy_opts, namestore, test_size,
                                falloff_form.lind)] if x is not None]


def dci_troe_dT(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term w.r.t. Temperature
    for Troe falloff reactions

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidT(eqs, loopy_opts, namestore, test_size,
                                falloff_form.troe)] if x is not None]


def dci_sri_dT(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the pressure modification term w.r.t. Temperature
    for SRI falloff reactions

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dcidT(eqs, loopy_opts, namestore, test_size,
                                falloff_form.sri)] if x is not None]


def __dRopidT(eqs, loopy_opts, namestore, test_size=None,
              do_ns=False, rxn_type=reaction_type.elementary, maxP=None,
              maxT=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress (for all reaction types)
    with respect to temperature


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
    do_ns : bool [False]
        If True, generate kernel to handle derivatives of the last species'
        concentration w.r.t. temperature
    rxn_type : [reaction_type.thd, reaction_type.plog, reaction_type.cheb]
        The type of reaction to generate fore
    maxP: int [None]
        The maximum number of pressure interpolations of any reaction in
        the mechanism.
        - For PLOG - the maximum number of pressure interpolations of any
        reaction in the mechanism.
        - For CHEB - The maximum degree of temperature polynomials for
        chebyshev reactions in this mechanism
    maxT : int [None]
        The maximum degree of temperature polynomials for chebyshev reactions
        in this mechanism

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # indicies
    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # check rxn type
    if rxn_type in [reaction_type.plog, reaction_type.cheb] and do_ns:
        return None  # this is handled by do_ns w / elementary
    elif rxn_type == reaction_type.plog:
        assert maxP, ('The maximum # of pressure interpolations must be'
                      ' supplied')
    elif rxn_type == reaction_type.cheb:
        assert maxP, ('The maximum chebyshev pressure polynomial degree must '
                      ' be supplied')
        assert maxT, ('The maximum chebyshev temperature polynomial degree '
                      ' must be supplied')

    num_range_dict = {reaction_type.elementary: namestore.num_simple,
                      reaction_type.plog: namestore.num_plog,
                      reaction_type.cheb: namestore.num_cheb}
    # get num
    num_range = num_range_dict[rxn_type] if not do_ns else\
        namestore.num_rxn_has_ns

    # return an empty kernel if it doesn't apply
    if num_range is None or not num_range.initializer.size:
        return None

    # number of species
    ns = namestore.num_specs[-1]

    mapstore = arc.MapStore(loopy_opts, num_range, num_range)

    rxn_range_dict = {reaction_type.elementary: namestore.simple_map,
                      reaction_type.plog: namestore.plog_map,
                      reaction_type.cheb: namestore.cheb_map}

    # add map
    rxn_range = rxn_range_dict[rxn_type] if not do_ns else\
        namestore.rxn_has_ns
    mapstore.check_and_add_transform(rxn_range, num_range)

    # rev mask depends on actual reaction index
    mapstore.check_and_add_transform(namestore.rev_mask, rxn_range)
    # thd_mask depends on actual reaction index
    mapstore.check_and_add_transform(namestore.thd_mask, rxn_range)
    # pres mod is on thd_mask
    mapstore.check_and_add_transform(
        namestore.pres_mod, namestore.thd_mask)
    # nu's are on the actual rxn index
    mapstore.check_and_add_transform(
        namestore.rxn_to_spec_offsets, rxn_range)

    # specific transforms
    if not do_ns:
        # fwd ROP is on actual rxn index
        mapstore.check_and_add_transform(namestore.rop_fwd, rxn_range)
        # rev ROP is on rev mask
        mapstore.check_and_add_transform(
            namestore.rop_rev, namestore.rev_mask)
    else:
        # kf is on real index
        mapstore.check_and_add_transform(namestore.kf, rxn_range)
        # kr is on rev rxn index
        mapstore.check_and_add_transform(namestore.kr, namestore.rev_mask)

    # extra inames
    net_ind = 'net_ind'
    k_ind = 'k_ind'

    # common variables
    # temperature
    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    # Volume
    V_lp, V_str = mapstore.apply_maps(namestore.V_arr, global_ind)
    # get rev / thd mask
    rev_mask_lp, rev_mask_str = mapstore.apply_maps(
        namestore.rev_mask, var_name)
    thd_mask_lp, thd_mask_str = mapstore.apply_maps(
        namestore.thd_mask, var_name)
    pres_mod_lp, pres_mod_str = mapstore.apply_maps(
        namestore.pres_mod, *default_inds)
    # nu offsets
    nu_offset_lp, nu_offset_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name)
    _, nu_offset_next_str = mapstore.apply_maps(
        namestore.rxn_to_spec_offsets, var_name, affine=1)
    # reac and prod nu for net
    nu_lp, net_reac_nu_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, net_ind, affine=net_ind)
    _, net_prod_nu_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, net_ind, affine=net_ind)
    # get species
    spec_lp, spec_str = mapstore.apply_maps(
        namestore.rxn_to_spec, net_ind)
    _, spec_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec, k_ind)
    # and jac
    jac_lp, jac_str = mapstore.apply_maps(
        namestore.jac, global_ind, spec_k_str, 0, affine={spec_k_str: 2})

    # add to data
    kernel_data.extend([T_lp, V_lp, rev_mask_lp, thd_mask_lp, pres_mod_lp,
                        nu_offset_lp, nu_lp, spec_lp, jac_lp])

    extra_inames = [
        (net_ind, 'offset <= {} < offset_next'.format(net_ind)),
        (k_ind, 'offset <= {} < offset_next'.format(k_ind))]
    parameters = {}
    pre_instructions = []
    if not do_ns:
        # get rxn parameters, these are based on the simple index
        beta_lp, beta_str = mapstore.apply_maps(
            namestore.simple_beta, var_name)
        Ta_lp, Ta_str = mapstore.apply_maps(
            namestore.simple_Ta, var_name)
        # rop's & pres mod
        rop_fwd_lp, rop_fwd_str = mapstore.apply_maps(
            namestore.rop_fwd, *default_inds)
        rop_rev_lp, rop_rev_str = mapstore.apply_maps(
            namestore.rop_rev, *default_inds)
        # and finally dBkdT
        dB_lp, dBk_str = mapstore.apply_maps(
            namestore.db, global_ind, spec_str)

        kernel_data.extend([
            beta_lp, Ta_lp, rop_fwd_lp, rop_rev_lp, dB_lp])

        pre_instructions = [ic.default_pre_instructs(
            'Tinv', T_str, 'INV')]
        if rxn_type == reaction_type.plog:
            lo_ind = 'lo'
            hi_ind = 'hi'
            param_ind = 'p'
            # create extra arrays
            P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)
            # number of plog rates per rxn
            plog_num_param_lp, plog_num_param_str = mapstore.apply_maps(
                namestore.plog_num_param, var_name)
            # pressure ranges
            plog_params_lp, pressure_mid_lo = mapstore.apply_maps(
                namestore.plog_params, 0, var_name, param_ind)
            _, pressure_mid_hi = mapstore.apply_maps(
                namestore.plog_params, 0, var_name, param_ind,
                affine={param_ind: 1})
            _, pres_lo_str = mapstore.apply_maps(
                namestore.plog_params, 0, var_name, lo_ind)
            _, pres_hi_str = mapstore.apply_maps(
                namestore.plog_params, 0, var_name, hi_ind)
            # arrhenius params
            _, beta_lo_str = mapstore.apply_maps(
                namestore.plog_params, 2, var_name, lo_ind)
            _, beta_hi_str = mapstore.apply_maps(
                namestore.plog_params, 2, var_name, hi_ind)
            _, Ta_lo_str = mapstore.apply_maps(
                namestore.plog_params, 3, var_name, lo_ind)
            _, Ta_hi_str = mapstore.apply_maps(
                namestore.plog_params, 3, var_name, hi_ind)
            _, pressure_lo = mapstore.apply_maps(
                namestore.plog_params, 0, var_name, 0)
            _, pressure_hi = mapstore.apply_maps(
                namestore.plog_params, 0, var_name, 'numP')
            kernel_data.extend([P_lp, plog_num_param_lp, plog_params_lp])

            # add plog instruction
            pre_instructions.append(ic.default_pre_instructs(
                'logP', P_str, 'LOG'))

            # and dkf instructions
            dkf_instructions = Template("""
                <> lo = 0 {id=lo_init}
                <> hi = numP {id=hi_init}
                <> numP = ${plog_num_param_str} - 1
                for ${param_ind}
                    if ${param_ind} <= numP and (logP > ${pressure_mid_lo}) and \
                            (logP <= ${pressure_mid_hi})
                        lo = ${param_ind} {id=set_lo, dep=lo_init}
                        hi = ${param_ind} + 1 {id=set_hi, dep=hi_init}
                    end
                end
                if logP > ${pressure_hi} # out of range above
                    <> dkf = (${beta_hi_str} + ${Ta_hi_str} * Tinv) * Tinv \
                        {id=dkf_init_hi, dep=set_*}
                else
                    dkf = (${beta_lo_str} + ${Ta_lo_str} * Tinv) * Tinv \
                        {id=dkf_init_lo, dep=set_*}
                end
                if logP > ${pressure_lo} and logP <= ${pressure_hi}
                    # not out of range
                    dkf = dkf + Tinv * (logP - ${pres_lo_str}) * \
                    (${beta_hi_str} - ${beta_lo_str} + \
                        (${Ta_hi_str} - ${Ta_lo_str}) * Tinv) / \
                    (${pres_hi_str} - ${pres_lo_str}) {id=dkf_final, dep=dkf_init*}
                end
            """).safe_substitute(**locals())
            extra_inames.append((
                param_ind, '0 <= {} < {}'.format(param_ind, maxP - 1)))
        elif rxn_type == reaction_type.cheb:
            # max degrees in mechanism
            poly_max = int(max(maxP, maxT - 1))

            # extra inames
            pres_poly_ind = 'k'
            temp_poly_ind = 'm'
            poly_compute_ind = 'p'
            lim_ind = 'dummy'
            extra_inames.extend([
                (pres_poly_ind, '0 <= {} < {}'.format(pres_poly_ind, maxP)),
                (temp_poly_ind, '0 <= {} < {}'.format(
                    temp_poly_ind, maxT - 1)),
                (poly_compute_ind, '2 <= {} < {}'.format(
                    poly_compute_ind, poly_max))])

            # create arrays

            num_P_lp, num_P_str = mapstore.apply_maps(
                namestore.cheb_numP, var_name)
            num_T_lp, num_T_str = mapstore.apply_maps(
                namestore.cheb_numT, var_name)
            params_lp, params_str = mapstore.apply_maps(
                namestore.cheb_params, var_name, temp_poly_ind, pres_poly_ind,
                affine={temp_poly_ind: 1})
            plim_lp, _ = mapstore.apply_maps(
                namestore.cheb_Plim, var_name, lim_ind)
            tlim_lp, _ = mapstore.apply_maps(
                namestore.cheb_Tlim, var_name, lim_ind)

            # workspace vars are based only on their polynomial indicies
            pres_poly_lp, ppoly_str = mapstore.apply_maps(
                namestore.cheb_pres_poly, pres_poly_ind)
            temp_poly_lp, tpoly_str = mapstore.apply_maps(
                namestore.cheb_temp_poly, temp_poly_ind)

            # create temperature and pressure arrays
            P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)

            kernel_data.extend([params_lp, num_P_lp, num_T_lp, plim_lp, P_lp,
                                tlim_lp, pres_poly_lp, temp_poly_lp])

            # preinstructions
            pre_instructions.extend(
                [ic.default_pre_instructs('logP', P_str, 'LOG')])

            # various strings for preindexed limits, params, etc
            _, Pmin_str = mapstore.apply_maps(
                namestore.cheb_Plim, var_name, '0')
            _, Pmax_str = mapstore.apply_maps(
                namestore.cheb_Plim, var_name, '1')
            _, Tmin_str = mapstore.apply_maps(
                namestore.cheb_Tlim, var_name, '0')
            _, Tmax_str = mapstore.apply_maps(
                namestore.cheb_Tlim, var_name, '1')

            # the various indexing for the pressure / temperature polynomials
            _, ppoly0_str = mapstore.apply_maps(namestore.cheb_pres_poly, '0')
            _, ppoly1_str = mapstore.apply_maps(namestore.cheb_pres_poly, '1')
            _, ppolyp_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                poly_compute_ind)
            _, ppolypm1_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                  poly_compute_ind,
                                                  affine=-1)
            _, ppolypm2_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                  poly_compute_ind,
                                                  affine=-2)
            _, tpoly0_str = mapstore.apply_maps(namestore.cheb_temp_poly, '0')
            _, tpoly1_str = mapstore.apply_maps(namestore.cheb_temp_poly, '1')
            _, tpolyp_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                poly_compute_ind)
            _, tpolypm1_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                  poly_compute_ind,
                                                  affine=-1)
            _, tpolypm2_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                  poly_compute_ind,
                                                  affine=-2)

            dkf_instructions = Template("""
                <>numP = ${num_P_str} {id=plim}
                <>numT = ${num_T_str} - 1 {id=tlim}
                <> Tred = (2 * Tinv - ${Tmax_str}- ${Tmin_str}) / \
                    (${Tmax_str} - ${Tmin_str})
                <> Pred = (2 * logP - ${Pmax_str} - ${Pmin_str}) / \
                    (${Pmax_str} - ${Pmin_str})
                ${ppoly0_str} = 1
                ${ppoly1_str} = Pred
                ${tpoly0_str} = 1
                ${tpoly1_str} = 2 * Tred

                # compute polynomial terms
                for p
                    if p < numP
                        ${ppolyp_str} = 2 * Pred * ${ppolypm1_str} - \
                            ${ppolypm2_str} {id=ppoly, dep=plim}
                    end
                    if p < numT
                        ${tpolyp_str} = 2 * Tred * ${tpolypm1_str} - \
                            ${tpolypm2_str} {id=tpoly, dep=tlim}
                    end
                end

                <> dkf = 0 {id=dkf_init}
                for m
                    <>temp = 0
                    for k
                        temp = temp + ${ppoly_str} * ${params_str} \
                            {id=temp, dep=ppoly:tpoly}
                    end
                    dkf = dkf + (m + 1) * ${tpoly_str} * temp \
                        {id=dkf_update, dep=temp:dkf_init}
                end
                dkf = -dkf * 2 * logten * Tinv * Tinv / (${Tmax_str} - ${Tmin_str}) \
                    {id=dkf, dep=dkf_update}
            """).safe_substitute(**locals())
            parameters['logten'] = log(10)
        else:
            dkf_instructions = Template(
                '<> dkf = (${beta_str} + ${Ta_str} * Tinv) * Tinv {id=dkf}'
            ).safe_substitute(**locals())

        # and put together instructions
        instructions = Template("""
        ${dkf_instructions}
        <> dRopidT = ${rop_fwd_str} * dkf {id=init, dep=dkf*}
        <> ci = 1 {id=ci_init}
        if ${rev_mask_str} >= 0
            <> dBk_sum = 0
            for ${net_ind}
                dBk_sum = dBk_sum + \
                    (${net_prod_nu_str} - ${net_reac_nu_str}) * ${dBk_str} {id=up}
            end
            dRopidT = dRopidT - ${rop_rev_str} * \
                (dkf - dBk_sum) {id=rev, dep=init:up}
        end
        if ${thd_mask_str} >= 0
            ci = ${pres_mod_str} {id=ci}
        end
        dRopidT = dRopidT * ci * ${V_str} {id=Ropi_final, dep=rev:ci*}
        """).safe_substitute(**locals())
    else:
        # create kf / kr
        kf_lp, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)
        kr_lp, kr_str = mapstore.apply_maps(namestore.kr, *default_inds)
        P_lp, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)
        conc_lp, conc_str = mapstore.apply_maps(
            namestore.conc_arr, global_ind, 'net_spec')

        # create Ns nu's
        _, ns_reac_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_reac_nu, 'offset_next',
            affine='offset_next - 2')
        _, ns_prod_nu_str = mapstore.apply_maps(
            namestore.rxn_to_spec_prod_nu, 'offset_next',
            affine='offset_next - 2')

        kernel_data.extend([kf_lp, kr_lp, P_lp, conc_lp])
        parameters['Ru'] = chem.RU

        instructions = Template("""
        <> kr_i = 0 {id=kr_in}
        if ${rev_mask_str} >= 0
            kr_i = ${kr_str} {id=kr_up}
        end
        <> ci = 1 {id=ci_init}
        if ${thd_mask_str} >= 0
            ci = ${pres_mod_str} {id=ci}
        end
        <> Sns_fwd = ${ns_reac_nu_str} {id=Sns_fwd_init}
        <> Sns_rev = ${ns_prod_nu_str} {id=Sns_rev_init}
        for ${net_ind}
            <> nu_fwd = ${net_reac_nu_str} {id=nuf_inner}
            <> nu_rev = ${net_prod_nu_str} {id=nur_inner}
            <> net_spec = ${spec_str}
            # handle nu
            if net_spec == ${ns}
                nu_fwd = nu_fwd - 1 {id=nuf_inner_up, dep=nuf_inner}
            end
            Sns_fwd = Sns_fwd * fast_powi(${conc_str}, nu_fwd) \
                {id=Sns_fwd_up, dep=nuf_inner_up}
            if net_spec == ${ns}
                nu_rev = nu_rev - 1 {id=nur_inner_up, dep=nur_inner}
            end
            Sns_rev = Sns_rev * fast_powi(${conc_str}, nu_rev) \
                {id=Sns_rev_up, dep=nur_inner_up}
        end
        <> dRopidT = (Sns_rev * kr_i - Sns_fwd * ${kf_str}) * \
            ${V_str} * ci * ${P_str} / (Ru * ${T_str} * ${T_str}) \
            {id=Ropi_final, dep=Sns*}
        """).substitute(**locals())

    # get nuk's
    _, reac_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, k_ind, affine=k_ind)
    _, prod_nu_k_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, k_ind, affine=k_ind)
    instructions = Template("""
        <> offset = ${nu_offset_str}
        <> offset_next = ${nu_offset_next_str}
        ${instructions}
        for ${k_ind}
            if ${spec_k_str} != ${ns}
                ${jac_str} = ${jac_str} + \
                    (${prod_nu_k_str} - ${reac_nu_k_str}) * dRopidT \
                    {id=jac, dep=Ropi_final}
            end
        end
    """).substitute(**locals())

    name_description = {reaction_type.elementary: '',
                        reaction_type.plog: '_plog',
                        reaction_type.cheb: '_cheb'}

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'])

    return k_gen.knl_info(name='dRopi{}_dT{}'.format(
        name_description[rxn_type],
        '_ns' if do_ns else ''),
        extra_inames=extra_inames,
        instructions=instructions,
        pre_instructions=pre_instructions,
        var_name=var_name,
        kernel_data=kernel_data,
        mapstore=mapstore,
        preambles=[lp_pregen.fastpowi_PreambleGen(),
                   lp_pregen.fastpowf_PreambleGen()],
        parameters=parameters,
        can_vectorize=can_vectorize,
        vectorization_specializer=vec_spec
    )


def dRopidT(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress (for non-pressure dependent reaction
    types) with respect to temperature

    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dRopidT`

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
    maxP: int [None]
        The maximum number of pressure interpolations of any reaction in
        the mechanism.

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dRopidT(eqs, loopy_opts, namestore,
                                  test_size=test_size, do_ns=False),
                        __dRopidT(eqs, loopy_opts, namestore,
                                  test_size=test_size, do_ns=True)]
            if x is not None]


def dRopi_plog_dT(eqs, loopy_opts, namestore, test_size=None, maxP=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress for PLOG reactions
    with respect to temperature


    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dRopidT`


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
    maxP: int [None]
        The maximum number of pressure interpolations of any reaction in
        the mechanism.

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dRopidT(eqs, loopy_opts, namestore,
                                  rxn_type=reaction_type.plog,
                                  test_size=test_size, do_ns=False,
                                  maxP=maxP)]
            if x is not None]


def dRopi_cheb_dT(eqs, loopy_opts, namestore, test_size=None, maxP=None,
                  maxT=None):
    """Generates instructions, kernel arguements, and data for calculating
    the derivative of the rate of progress for Chebyshev reactions
    with respect to temperature


    Notes
    -----
    See :meth:`pyjac.core.create_jacobian.__dRopidT`


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
    maxP : int [None]
        The maximum degree of pressure polynomials for chebyshev reactions in
        this mechanism
    maxT : int [None]
        The maximum degree of temperature polynomials for chebyshev reactions
        in this mechanism

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    return [x for x in [__dRopidT(eqs, loopy_opts, namestore,
                                  rxn_type=reaction_type.cheb,
                                  test_size=test_size, do_ns=False,
                                  maxP=maxP, maxT=maxT)]
            if x is not None]


def thermo_temperature_derivative(nicename, eqs, loopy_opts, namestore,
                                  test_size=None):
    """Generates instructions, kernel arguements, and data for calculating
    the concentration weighted specific energy sum.


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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    eq = eqs['conp'] if nicename in ['dcp'] else eqs['conv']
    return rate.polyfit_kernel_gen(
        nicename, eq, loopy_opts, namestore, test_size)


@ic.with_conditional_jacobian
def dEdot_dnj(eqs, loopy_opts, namestore, test_size=None,
              conp=True, jac_create=None):
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
    jac_create: Callable
        The conditional Jacobian instruction creator from :mod:`instruction_creator`

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # create arrays
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    # k loop is _only_ over non-zero dnk/dnj deriviatives
    ns = namestore.num_specs[-1]
    num_nonzero_specs = namestore.num_net_nonzero_spec.size
    if ns in namestore.num_net_nonzero_spec.initializer:
        num_nonzero_specs -= 1
    # k loop
    i_spec_k = 'i_spec_k'
    extra_inames = [(i_spec_k, '0 <= i_spec_k < {}'.format(
        num_nonzero_specs))]
    nonzero_lp, spec_k = mapstore.apply_maps(
        namestore.net_nonzero_spec, i_spec_k)

    mw_lp, mw_str = mapstore.apply_maps(
        namestore.mw_post_arr, spec_k)
    V_lp, V_str = mapstore.apply_maps(
        namestore.V_arr, global_ind)
    # dnk/dnj jacobian set
    dnkdnj_insn = ("sum = sum + (1 - ${mw_str}) * ${jac_str} {id=sum, dep=${deps}}")
    jac_lp, dnkdnj_insn = jac_create(
        mapstore, namestore.jac, global_ind, spec_k, var_name, affine={
            var_name: 2,
            spec_k: 2
        }, insn=dnkdnj_insn, deps='*')
    # and the dedot / dnj instruction
    dedotdnj_insn = (
        "${jac_str} = ${jac_str} + ${T_str} * Ru * sum / ${fixed_var_str} + "
        "${extra_var_str} * ${dTdot_dnj_str} / ${T_str} "
        "{id=jac, dep=${deps}, nosync=sum}")
    _, dedotdnj_insn = jac_create(
        mapstore, namestore.jac, global_ind, 1, var_name, affine={
            var_name: 2,
        }, insn=dedotdnj_insn, deps='sum', entry_exists=True)
    # and finally do a simple string creation for dTdot / dnj
    # NOTE: do not precompute index here as 1. it's only called once
    # 2. it has to exist by defn and 3. we haven't trained the creator to hanle
    # multiple indicies
    _, dTdot_dnj_str = jac_create(
        mapstore, namestore.jac, global_ind, 0, var_name, affine={
            var_name: 2,
        }, entry_exists=True, index_insn=False)

    P_lp, P_str = mapstore.apply_maps(
        namestore.P_arr, global_ind)
    T_lp, T_str = mapstore.apply_maps(
        namestore.T_arr, global_ind)

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    kernel_data.extend([mw_lp, V_lp, P_lp, T_lp, jac_lp, nonzero_lp])

    extra_var_str = V_str if conp else P_str
    fixed_var_str = P_str if conp else V_str
    instructions = Template(Template("""
    <> sum = 0 {id=init}
    for ${i_spec_k}
        ${dnkdnj_insn}
    end
    ${dedotdnj_insn}
    """).safe_substitute(**locals())).safe_substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(loopy_opts, atomic_ids=['jac'])

    return k_gen.knl_info(name='d{}dot_dnj'.format('V' if conp else 'P'),
                          extra_inames=extra_inames,
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          parameters={'Ru': chem.RU},
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec
                          )


@ic.with_conditional_jacobian
def dTdot_dnj(eqs, loopy_opts, namestore, test_size=None,
              conp=True, jac_create=None):
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
    jac_create: Callable
        The conditional Jacobian instruction creator from :mod:`instruction_creator`

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # create arrays
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    # k loop is _only_ over non-zero dnk/dnj deriviatives
    ns = namestore.num_specs[-1]
    num_nonzero_specs = namestore.num_net_nonzero_spec.size
    if ns in namestore.num_net_nonzero_spec.initializer:
        num_nonzero_specs -= 1
    # k loop
    i_spec_k = 'i_spec_k'
    extra_inames = [(i_spec_k, '0 <= i_spec_k < {}'.format(
        num_nonzero_specs))]
    nonzero_lp, spec_k = mapstore.apply_maps(
        namestore.net_nonzero_spec, i_spec_k)
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

    # start creating the Jacobian

    # species jacobian sum
    species_jac_insn = ("sum = sum + (${energy_k_str} - ${energy_ns_str} * "
                        "${mw_str}) * ${jac_str} {id=sum, dep=${deps}}")
    jac_lp, species_jac_insn = jac_create(
        mapstore, namestore.jac, global_ind, spec_k, var_name, affine={
            var_name: 2,
            spec_k: 2
        }, insn=species_jac_insn, deps='*')

    # dTdot/dnj jacobian set
    tdot_jac_insn = (
        "${jac_str} = -(sum + ${T_dot_str} * "
        "(${spec_heat_k_str} - ${spec_heat_ns_str})) / "
        "(${V_str} * ${spec_heat_total_str}) {id=jac, dep=${deps}, nosync=sum}")
    _, tdot_jac_insn = jac_create(
        mapstore, namestore.jac, global_ind, '0', var_name, affine={
            var_name: 2,
        }, entry_exists=True, insn=tdot_jac_insn, deps='sum')

    kernel_data = []
    if namestore.test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    kernel_data.extend([spec_heat_lp, energy_lp, spec_heat_tot_lp, mw_lp,
                        V_lp, T_dot_lp, jac_lp, nonzero_lp])

    instructions = Template(Template("""
    <> sum = 0 {id=init}
    for ${i_spec_k}
        ${species_jac_insn}
    end
    ${tdot_jac_insn}
    """).safe_substitute(**locals())).safe_substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(loopy_opts, init_ids=['jac'])

    return k_gen.knl_info(name='dTdot_dnj',
                          extra_inames=extra_inames,
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec
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
    knl_list : list of :class:`knl_info`
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

    barrier = (
        '... lbarrier {id=break, dep=init}'
        if loopy_opts.use_atomics and loopy_opts.depth else
        '... nop {id=break, dep=init}')
    pre_instructions = Template("""
        <>spec_tot = 0
        ${spec_heat_total_str} = 0 {id=init}
        """).safe_substitute(**locals())
    instructions = Template("""
        spec_tot = spec_tot + ${spec_heat_str} * \
            ${conc_str} {id=update}
    """).safe_substitute(**locals())
    post_instructions = Template("""
        ${barrier}
        ${spec_heat_total_str} = ${spec_heat_total_str} + spec_tot \
            {id=sum, dep=update:init:break, nosync=init}
    """).safe_substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['sum'], init_ids=['init'])

    return k_gen.knl_info(name='{}_total'.format(namestore.spec_heat.name),
                          pre_instructions=[pre_instructions],
                          instructions=instructions,
                          post_instructions=[post_instructions],
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec
                          )


def __dci_dnj(loopy_opts, namestore,
              do_ns=False, fall_type=falloff_form.none):
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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    if fall_type == falloff_form.none:
        our_inds = namestore.num_thd_only if not do_ns \
            else namestore.thd_only_ns_inds
        rxn_range = namestore.thd_only_map if not do_ns \
            else namestore.thd_only_ns_map
        thd_range = namestore.thd_only_map if not do_ns \
            else namestore.thd_only_ns_map
        map_onto = namestore.thd_map
    elif fall_type == falloff_form.sri:
        our_inds = namestore.num_sri if not do_ns \
            else namestore.sri_ns_inds
        rxn_range = namestore.sri_map if not do_ns \
            else namestore.sri_has_ns
        thd_range = namestore.fall_to_thd_map
        map_onto = namestore.fall_map
    elif fall_type == falloff_form.troe:
        our_inds = namestore.num_troe if not do_ns \
            else namestore.troe_ns_inds
        rxn_range = namestore.troe_map if not do_ns \
            else namestore.troe_has_ns
        thd_range = namestore.fall_to_thd_map
        map_onto = namestore.fall_map
    elif fall_type == falloff_form.lind:
        our_inds = namestore.num_lind if not do_ns \
            else namestore.lind_ns_inds
        rxn_range = namestore.lind_map if not do_ns \
            else namestore.lind_has_ns
        thd_range = namestore.fall_to_thd_map
        map_onto = namestore.fall_map

    knl_name = 'dci_{}_dnj{}'.format(
        {falloff_form.none: 'thd',
         falloff_form.lind: 'lind',
         falloff_form.sri: 'sri',
         falloff_form.troe: 'troe'
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

    if fall_type != falloff_form.none:
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
    if fall_type != falloff_form.none:
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

        if fall_type == falloff_form.sri:
            # get the sri arrays, keyed on the SRI index
            mapstore.check_and_add_transform(namestore.sri_a, our_inds)
            mapstore.check_and_add_transform(namestore.sri_b, our_inds)
            mapstore.check_and_add_transform(namestore.sri_c, our_inds)
            mapstore.check_and_add_transform(namestore.X_sri, our_inds)
        elif fall_type == falloff_form.troe:
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
    if fall_type != falloff_form.none:
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
        if fall_type == falloff_form.sri:
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
        <> dFi = -2 * ${sri_X_str} * ${sri_X_str} * log(${sri_a_str} * \
           exp(-${sri_b_str} / ${T_str}) + exp(-${T_str} / ${sri_c_str})) * \
           log(fmax(${Pr_str}, 1e-300d)) / \
           (fmax(${Pr_str}, 1e-300d) * logtensquared) {id=dFi}
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

        elif fall_type == falloff_form.troe:
            Atroe_lp, Atroe_str = mapstore.apply_maps(
                namestore.Atroe, *default_inds)
            Btroe_lp, Btroe_str = mapstore.apply_maps(
                namestore.Btroe, *default_inds)
            Fcent_lp, Fcent_str = mapstore.apply_maps(
                namestore.Fcent, *default_inds)

            kernel_data.extend([Atroe_lp, Btroe_lp, Fcent_lp])

            dFi = Template(
                """
        <> dFi = ${Atroe_str} * ${Atroe_str} + ${Btroe_str} * ${Btroe_str} \
            {id=dFi_init}
        dFi = -2 * ${Atroe_str} * ${Btroe_str} * \
        (0.14 * ${Atroe_str} + ${Btroe_str}) * log(fmax(${Fcent_str}, 1e-300d)) / \
        (fmax(${Pr_str}, 1e-300d) * dFi * dFi * logten) {id=dFi, dep=dFi_init}
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
        dFi =  k0 * (${Fi_str} * Fi_fac - ${pres_mod_str}) / \
            (kinf * (${Pr_str} + 1)) {id=fall, dep=kf_*:kinf_*:dFi_fac_*}
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
                        ${jac_str} = ${jac_str} + nu_k * dci * ropi${fall_mul_str} \
                            {id=jac${fall_dep_str}}
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
            ns=namestore.num_specs[-1],
            thd_type_str=thd_type_str,
            mix=int(thd_body_type.mix),
            species=int(thd_body_type.species),
            fall_update=fall_update,
            fall_mul_str=(' * dFi ' if fall_type != falloff_form.none else ''),
            fall_dep_str=(
                ', dep=fall' if fall_type != falloff_form.none else '')
        )
    else:
        extra_inames.append((spec_j, '0 <= {} < {}'.format(
            spec_j, namestore.num_specs[-1])))

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
                    ${jac_str} = ${jac_str} + nu_k * dci * ropi${fall_mul_str} \
                        {id=jac${fall_dep_str}}
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
            ns=namestore.num_specs[-1],
            thd_type_str=thd_type_str,
            mix=int(thd_body_type.mix),
            species=int(thd_body_type.species),
            fall_update=fall_update,
            fall_mul_str=(' * dFi ' if fall_type != falloff_form.none else ''),
            fall_dep_str=(
                ', dep=fall' if fall_type != falloff_form.none else '')
        )

    inames, ranges = zip(*extra_inames)
    # join inames
    extra_inames = [
        (','.join(inames), ' and '.join(ranges))]

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['jac'])

    return k_gen.knl_info(name=knl_name,
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          extra_inames=extra_inames,
                          mapstore=mapstore,
                          parameters=parameters,
                          manglers=manglers,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec
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
    knl_list : list of :class:`knl_info`
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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False, falloff_form.lind)]
    ns_info = __dci_dnj(loopy_opts, namestore, True, falloff_form.lind)
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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False, falloff_form.sri)]
    ns_info = __dci_dnj(loopy_opts, namestore, True, falloff_form.sri)
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
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    infos = [__dci_dnj(loopy_opts, namestore, False, falloff_form.troe)]
    ns_info = __dci_dnj(loopy_opts, namestore, True, falloff_form.troe)
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
    knl_list : list of :class:`knl_info`
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
        if do_ns and rxn_range.initializer is None or not rxn_range.initializer.size:
            return None

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
                            Sj_fwd = Sj_fwd * fast_powi(${conc_inner_str}, nu_fwd) \
                                {id=Sj_fwd_up, dep=Sj_fwd_init:nuf_inner_up}
                            Sj_rev = Sj_rev * fast_powi(${conc_inner_str}, nu_rev) \
                                {id=Sj_rev_up, dep=Sj_rev_init:nur_inner_up}
                        end
                        # and update Jacobian
                        ${jac_str} = ${jac_str} + (kf_i * Sj_fwd - kr_i * Sj_rev) * \
                            ci * nu_k \
                            {id=jac, dep=Sj_fwd_up:Sj_rev_up:ci_up:nu_k:spec_k}
                    end
                end
            """)
        else:
            extra_inames.append((spec_j, '0 <= {} < {}'.format(
                spec_j, namestore.num_specs[-1])))
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
                        nu_fwd = nu_fwd - 1 \
                            {id=nuf_inner_up, dep=nuf_inner:Sns_fwd_up}
                    end
                    Sns_fwd = Sns_fwd * fast_powi(${conc_inner_str}, nu_fwd) \
                        {id=Sns_fwd_up2, dep=Sns_fwd_up:nuf_inner_up}
                    if ${spec_inner} == ${ns}
                        Sns_rev = Sns_rev * nu_rev {id=Sns_rev_up, dep=Sns_rev_init}
                        nu_rev = nu_rev - 1 \
                            {id=nur_inner_up, dep=nur_inner:Sns_rev_up}
                    end
                    Sns_rev = Sns_rev * fast_powi(${conc_inner_str}, nu_rev) \
                        {id=Sns_rev_up2, dep=Sns_rev_up:nur_inner_up}
                end
                # and update Jacobian for all species in this row
                <> jac_updater =  (kr_i * Sns_rev - kf_i * Sns_fwd) * ci * nu_k \
                    {id=jac_up, dep=Sns_fwd_up*:Sns_rev_up*:ci_up:nu_k:spec_k:kf:kr*}
                for ${spec_j}
                    ${jac_str} = ${jac_str} + jac_updater {id=jac}
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
            ns=namestore.num_specs[-1]
        )

        inames, ranges = zip(*extra_inames)
        # join inames
        extra_inames = [
            (','.join(inames), ' and '.join(ranges))]

        can_vectorize, vec_spec = ic.get_deep_specializer(
            loopy_opts, atomic_ids=['jac'])
        return k_gen.knl_info(name='dRopidnj{}'.format('_ns' if do_ns else ''),
                              instructions=instructions,
                              var_name=var_name,
                              kernel_data=kernel_data,
                              extra_inames=extra_inames,
                              mapstore=mapstore,
                              preambles=[
                                   lp_pregen.fastpowi_PreambleGen(),
                                   lp_pregen.fastpowf_PreambleGen()],
                              can_vectorize=can_vectorize,
                              vectorization_specializer=vec_spec
                              )

    return [x for x in [__dropidnj(False), __dropidnj(True)] if x is not None]


def get_jacobian_kernel(eqs, reacs, specs, loopy_opts, conp=True,
                        test_size=None, auto_diff=False):
    """Helper function that generates kernels for
       evaluation of reaction rates / rate constants / and species rates

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    reacs : list of :class:`ReacInfo`
        List of species in the mechanism.
    specs : list of :class:`SpecInfo`
        List of species in the mechanism.
    loopy_opts : :class:`loopy_options` object
        A object containing all the loopy options to execute
    conp : bool
        If true, generate equations using constant pressure assumption
        If false, use constant volume equations
    test_size : int
        If not None, this kernel is being used for testing.
    auto_diff : bool
        If ``True``, generate files for Adept autodifferention library.

    Returns
    -------
    kernel_gen : :class:`kernel_generator`
        The generator responsible for creating the resulting Jacobian code

    """

    # figure out rates and info
    rate_info = determine_jac_inds(reacs, specs, loopy_opts.rate_spec,
                                   loopy_opts.jac_type)

    # set test size
    if test_size is None:
        test_size = 'problem_size'

    # create the namestore
    nstore = arc.NameStore(loopy_opts, rate_info, conp, test_size)

    kernels = []
    barriers = []

    def __add_knl(knls, klist=None):
        if klist is None:
            klist = kernels
        try:
            klist.extend(knls)
        except:
            klist.append(knls)

    # barrier management
    def __insert_at(name, before=True):
        if loopy_opts.depth:
            ind = next((i for i, knl in enumerate(kernels)
                        if knl.name == name), None)
            if ind is not None:
                if before:
                    barriers.append((ind - 1, ind, 'global'))
                else:
                    barriers.append((ind, ind + 1, 'global'))

    # Note:
    # the order in which these kernels get added is important
    # the kernel generator uses the input order to generate the wrapping
    # kernel calls
    # hence, any data dependencies should be expressed in the order added here

    # reset kernels
    __add_knl(reset_arrays(eqs, loopy_opts, nstore, test_size=test_size))

    # first, add the species derivatives

    allint = {'net': rate_info['net']['allint']}

    # insert a barrier to a) wait for the end of the species rates
    # and b) wait for the end of the jacobian reset
    __insert_at(kernels[-1].name, before=False)
    # rate of progress derivatives
    __add_knl(dRopi_dnj(eqs, loopy_opts, nstore, allint, test_size=test_size))

    # and the third body / falloff derivatives
    if rate_info['thd']['num']:
        __add_knl(dci_thd_dnj(eqs, loopy_opts, nstore, test_size=test_size))

        if rate_info['fall']['lind']['num']:
            __add_knl(
                dci_lind_dnj(eqs, loopy_opts, nstore, test_size=test_size))

        if rate_info['fall']['sri']['num']:
            __add_knl(
                dci_sri_dnj(eqs, loopy_opts, nstore, test_size=test_size))

        if rate_info['fall']['troe']['num']:
            __add_knl(
                dci_troe_dnj(eqs, loopy_opts, nstore, test_size=test_size))

    # total spec heats
    __add_knl(total_specific_energy(
        eqs, loopy_opts, nstore, conp=conp, test_size=test_size))

    # and thermo derivatives
    __add_knl(thermo_temperature_derivative(
        nstore.dspec_heat.name, eqs, loopy_opts, nstore, test_size=test_size))

    if rate_info['rev']['num']:
        __add_knl(thermo_temperature_derivative(
            nstore.db.name, eqs, loopy_opts, nstore, test_size=test_size))

    # next, the temperature derivative w.r.t. species
    __add_knl(dTdot_dnj(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))
    # (depends on total_specific_energy)
    __insert_at(kernels[-1].name)

    # and the extra var deriv
    __add_knl(
        dEdot_dnj(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))
    # (depends on dTdot_dnj)
    __insert_at(kernels[-1].name)

    # temperature derivatives
    __add_knl(dRopidT(eqs, loopy_opts, nstore, test_size=test_size))

    # check for plog
    if rate_info['plog']['num']:
        __add_knl(dRopi_plog_dT(eqs, loopy_opts, nstore,
                                maxP=rate_info['plog']['max_P'],
                                test_size=test_size))

    # check for chebyshev
    if rate_info['cheb']['num']:
        __add_knl(dRopi_cheb_dT(eqs, loopy_opts, nstore,
                                maxP=np.max(rate_info['cheb']['num_P']),
                                maxT=np.max(rate_info['cheb']['num_T']),
                                test_size=test_size))

    # check for third body terms
    if rate_info['thd']['num']:
        __add_knl(dci_thd_dT(eqs, loopy_opts, nstore, test_size=test_size))

        if rate_info['fall']['lind']['num']:
            __add_knl(
                dci_lind_dT(eqs, loopy_opts, nstore, test_size=test_size))

        if rate_info['fall']['sri']['num']:
            __add_knl(dci_sri_dT(eqs, loopy_opts, nstore, test_size=test_size))

        if rate_info['fall']['troe']['num']:
            __add_knl(
                dci_troe_dT(eqs, loopy_opts, nstore, test_size=test_size))

    # total tempertature derivative
    __add_knl(dTdotdT(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))
    # barrier for dnj / dT
    __insert_at(kernels[-1].name)

    # total extra var derivative w.r.t T
    __add_knl(dEdotdT(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))
    # barrier for dependency on dTdotdT
    __insert_at(kernels[-1].name)

    # finally, do extra var derivatives
    __add_knl(dRopidE(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))

    # check for plog
    if rate_info['plog']['num']:
        __add_knl(dRopi_plog_dE(
            eqs, loopy_opts, nstore, maxP=rate_info['plog']['max_P'],
            conp=conp, test_size=test_size))

    # check for cheb
    if rate_info['cheb']['num']:
        __add_knl(dRopi_cheb_dE(
            eqs, loopy_opts, nstore,
            maxP=np.max(rate_info['cheb']['num_P']),
            maxT=np.max(rate_info['cheb']['num_T']),
            conp=conp, test_size=test_size))

    # and the third body / falloff derivativatives
    if rate_info['thd']['num']:
        __add_knl(dci_thd_dE(
            eqs, loopy_opts, nstore, conp=conp, test_size=test_size))

        if rate_info['fall']['lind']['num']:
            __add_knl(
                dci_lind_dE(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))

        if rate_info['fall']['sri']['num']:
            __add_knl(dci_sri_dE(eqs, loopy_opts, nstore, conp=conp,
                                 test_size=test_size))

        if rate_info['fall']['troe']['num']:
            __add_knl(
                dci_troe_dE(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))

    # and the temperature derivative w.r.t. the extra var
    __add_knl(dTdotdE(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))
    # inser barrier for dnj / dE from the previous kernels
    __insert_at(kernels[-1].name)

    # total extra var derivative w.r.t the extra var
    __add_knl(dEdotdE(eqs, loopy_opts, nstore, conp=conp, test_size=test_size))
    # barrier for dependency on dEdotdE
    __insert_at(kernels[-1].name)

    input_arrays = ['phi', 'P_arr' if conp else 'V_arr']
    output_arrays = ['jac']

    # create the specrates subkernel
    sgen = rate.get_specrates_kernel(eqs, reacs, specs, loopy_opts, conp=conp)
    sub_kernels = sgen.kernels[:]
    # and finally fix the barriers to account for the sub kernels
    offset = len(sub_kernels)
    barriers = [(i1 + offset, i2 + offset, bartype)
                for i1, i2, bartype in barriers]
    # and return the full generator
    return k_gen.make_kernel_generator(
        loopy_opts=loopy_opts,
        name='jacobian_kernel',
        kernels=sub_kernels + kernels,
        namestore=nstore,
        depends_on=[sgen],
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        auto_diff=auto_diff,
        test_size=test_size,
        barriers=barriers)


def find_last_species(specs, last_spec=None, return_map=False):
    """
    Find a suitable species to move to the end of the mechanism, taking into account
    a user specified species, :param:`last_spec` if supplied.

    Notes
    -----
    If the user does not specify a species, or it cannot be found the first species
    of Nitrogen (N_2), Argon (Ar) and Helium (He) found in the mechanism will be
    used. If none of these species can be found, the mechanism will be unchanged.

    Parameters
    ----------
    specs: list of :class:`SpecInfo`
        The species in the mechanism
    last_spec: str [None]
        The name of the last species specified by the user
    return_map: bool [False]
        If True, return a mapping that can be used to map species / data
        for the mechanism, e.g.:
            ```
            map = find_last_species(specs, return_map=True)
            concs = concs[map]
            ```
        If false, an updated species list will be returned

    Returns
    -------
    map: list of :class:`SpecInfo` or :class:`numpy.ndarray`
        Depending on value of :param:`return_map`, returns an updated species list
        or mapping to achieve the same
    """
    # check to see if the last_spec is specified
    if last_spec is not None:
        # find the index if possible
        isp = next((i for i, sp in enumerate(specs)
                    if sp.name.lower() == last_spec.lower().strip()),
                   None
                   )
        if isp is None:
            logging.warn('User specified last species {} not found in mechanism.'
                         '  Attempting to find a default species.'.format(last_spec))
            last_spec = None
        else:
            last_spec = isp
    else:
        logging.warn('User specified last species not found or not specified.  '
                     'Attempting to find a default species')
    if last_spec is None:
        wt = chem.get_elem_wt()
        # check for N2, Ar, He, etc.
        candidates = [('N2', wt['n'] * 2.), ('Ar', wt['ar']),
                      ('He', wt['he'])]
        for sp in candidates:
            match = next((isp for isp, spec in enumerate(specs)
                          if sp[0].lower() == spec.name.lower() and
                          sp[1] == spec.mw), None)
            if match is not None:
                last_spec = match
                break
        if last_spec is not None:
            logging.info('Default last species {} found.'.format(
                specs[last_spec].name))
    if last_spec is None:
        logging.warn('Neither a user specified or default last species '
                     'could be found. Proceeding using the last species in the '
                     'base mechanism: {}'.format(specs[-1].name))
        last_spec = len(specs) - 1

    if return_map:
        gas_map = np.arange(len(specs), dtype=np.int32)
        gas_map[last_spec:-1] = gas_map[last_spec + 1:]
        gas_map[-1] = last_spec
        return gas_map

    # else, pick up the last_spec and drop it at the end
    specs.append(specs.pop(last_spec))
    return specs


def create_jacobian(lang, mech_name=None, therm_name=None, gas=None,
                    vector_size=None, wide=False, deep=False, ilp=None, unr=None,
                    build_path='./out/', last_spec=None, skip_jac=False, platform='',
                    data_order='C', rate_specialization='full',
                    split_rate_kernels=True, split_rop_net_kernels=False,
                    conp=True, data_filename='data.bin', output_full_rop=False,
                    use_atomics=True, jac_type='exact', jac_format='full'):
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
        The mechanism to generate the Jacobian for.  This or ``mech_name`` must be
        specified
    vector_size : int
        The SIMD vector width to use.  If the targeted platform is a GPU,
        this is the GPU block size
    wide : bool
        If true, use a 'wide' vectorization strategy. Cannot be specified along with
        'deep'.
    deep : bool
        If true, use a 'deep' vectorization strategy. Cannot be specified along with
        'wide'.
    unr : int
        If supplied, unroll inner loops (i.e. those that would be affected by a
        deep vectorization). Can be used in conjunction with deep or wide parallelism
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
        *   If a vendor specific string, it will be passed to pyopencl to get the
            platform
    data_order : {'C', 'F'}
        The data ordering, 'C' (row-major) recommended for deep vectorizations,
        while 'F' (column-major) recommended for wide vectorizations
    rate_specialization : {'fixed', 'hybrid', 'full'}
        The level of specialization in evaluating reaction rates.
        'Full' is the full form suggested by Lu et al. (citation)
        'Hybrid' turns off specializations in the exponential term (Ta = 0, b = 0)
        'Fixed' is a fixed expression exp(logA + b logT + Ta / T)
    split_rate_kernels : bool
        If True, and the :param"`rate_specialization` is not 'Fixed', split different
        valuation types into different kernels
    split_rop_net_kernels : bool
        If True, break different ROP values (fwd / back / pdep) into different
        kernels
    conp : bool
        If True, use the constant pressure assumption.  If False, use the constant
        volume assumption.
    data_filename : str
        If specified, the path to the data.bin file that will be used for kernel
        testing
    output_full_rop : bool
        If ``True``, output forward and reversse rates of progress
        Useful in testing, as there are serious floating point errors for
        net production rates near equilibrium, invalidating direct comparison to
        Cantera
    jac_type: ['exact', 'approximate']
        The type of Jacobian kernel to generate.  An approximate Jacobian ignores
        derivatives of the last species with respect to other species in the
        mechanism -- i.e. :math:`\frac{\partial n_{N_s}}{\partial n_{j}}` -- in the
        reaction rate derivatives.

        This can significantly increase sparsity for mechanisms containing reactions
        that include the last species directly, or as a third-body species with a
        non-unity efficiency, but gives results in an approxmiate Jacobian, and thus
        is more suitable to use with implicit integration techniques.
    jac_format: ['full', 'sparse']
        If 'sparse', the Jacobian will be encoded using a compressed row or column
        storage format (for a data order of 'C' and 'F' respectively).
    Returns
    -------
    None

    """

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

    # convert enums
    rate_spec_val = utils.EnumType(lp_utils.RateSpecialization)(
        rate_specialization.lower())
    jac_format = utils.EnumType(JacobianFormat)(
        jac_format.lower())
    jac_type = utils.EnumType(JacobianType)(
        jac_type.lower())

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
                                        platform=platform,
                                        use_atomics=use_atomics,
                                        jac_format=jac_format,
                                        jac_type=jac_type)

    # create output directory if none exists
    utils.create_dir(build_path)

    assert mech_name is not None or gas is not None, 'No mechanism specified!'

    # Interpret reaction mechanism file, depending on Cantera or
    # Chemkin format.
    if gas is not None or mech_name.endswith(tuple(['.cti', '.xml'])):
        elems, specs, reacs = mech.read_mech_ct(mech_name, gas)
    else:
        elems, specs, reacs = mech.read_mech(mech_name, therm_name)

    if not specs:
        logging.error('No species found in file: {}'.format(mech_name))
        sys.exit(3)

    if not reacs:
        logging.error('No reactions found in file: {}'.format(mech_name))
        sys.exit(3)

    # find and move last species to end
    specs = find_last_species(specs, last_spec=last_spec)

    # write headers
    aux.write_aux(build_path, loopy_opts, specs, reacs)

    eqs = {}
    eqs['conp'] = sp_interp.load_equations(conp)[1]
    eqs['conv'] = sp_interp.load_equations(not conp)[1]

    # now begin writing subroutines
    if not skip_jac:
        # get Jacobian subroutines
        gen = get_jacobian_kernel(eqs, reacs, specs, loopy_opts, conp=conp)
        #  write_sparse_multiplier(build_path, lang, touched, len(specs))
    else:
        # just specrates
        gen = rate.get_specrates_kernel(eqs, reacs, specs, loopy_opts,
                                        conp=conp, output_full_rop=output_full_rop)

    # write the kernel
    gen.generate(build_path, data_filename=data_filename)
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
                    auto_diff=args.auto_diff,
                    )
