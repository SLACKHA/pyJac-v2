# -*- coding: utf-8 -*-
"""Module for writing species/reaction rate subroutines.

This is kept separate from Jacobian creation module in order
to create only the rate subroutines if desired.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import logging
from string import Template
from collections import OrderedDict

# Non-standard librarys
import loopy as lp
import numpy as np
from loopy.kernel.data import AddressSpace as scopes

# Local imports
from pyjac import utils
from pyjac.core import chem_model as chem
from pyjac.core.enum_types import reaction_type, falloff_form, thd_body_type,\
    RateSpecialization, KernelType
from pyjac.core import instruction_creator as ic
from pyjac.core import array_creator as arc
from pyjac.core.array_creator import (global_ind, var_name, default_inds)
from pyjac.kernel_utils import kernel_gen as k_gen


def inputs_and_outputs(conp, ktype=KernelType.species_rates, output_full_rop=False):
    """
    A convenience method such that kernel inputs / output argument names are
    available for inspection

    Parameters
    ----------
    conp: bool
        If true, use constant-pressure formulation, else constant-volume
    ktype: :class:`KernelType`
        The kernel type to return the arguments for -- if unspecified, defaults to
        species rates

    Returns
    -------
    input_args: list of str
        The input arguments to kernels generated in this file
    output_args: list of str
        The output arguments to kernels generated in this file
    """
    if ktype == KernelType.species_rates:
        input_args = utils.kernel_argument_ordering(
            [arc.state_vector, arc.pressure_array if conp else arc.volume_array],
            ktype)
        output_args = [arc.state_vector_rate_of_change]
    elif ktype == KernelType.chem_utils:
        input_args = [arc.state_vector]
        output_args = utils.kernel_argument_ordering(
            [arc.enthalpy_array, arc.constant_pressure_specific_heat,
             arc.rate_const_thermo_coeff_array] if conp else [
             arc.internal_energy_array, arc.constant_volume_specific_heat,
             arc.rate_const_thermo_coeff_array], ktype)
    else:
        raise NotImplementedError()

    if output_full_rop:
        output_args += ['rop_fwd', 'rop_rev', 'pres_mod', 'rop_net']

    return input_args, output_args


def assign_rates(reacs, specs, rate_spec):
    """
    From a given set of reactions, determine the rate types for evaluation

    Parameters
    ----------
    reacs : list of `ReacInfo`
        The reactions in the mechanism
    specs : list of `SpecInfo`
        The species in the mechanism
    rate_spec : `RateSpecialization` enum
        The specialization option specified

    Notes
    -----

    For simple Arrhenius evaluations, the rate type keys are:

    if rate_spec == RateSpecialization.full
        0 -> kf = A
        1 -> kf = A * T * T * T ...
        2 -> kf = exp(logA + b * logT)
        3 -> kf = exp(logA - Ta / T)
        4 -> kf = exp(logA + b * logT - Ta / T)

    if rate_spec = RateSpecialization.hybrid
        0 -> kf = A
        1 -> kf = A * T * T * T ...
        2 -> kf = exp(logA + b * logT - Ta / T)

    if rate_spec == RateSpecialization.fixed
        0 -> kf = exp(logA + b * logT - Ta / T)

    Note that the reactions in 'fall', 'chem' and 'thd' are also in
            'simple'
        Further, there are duplicates between 'thd' and 'fall' / 'chem'

    Returns
    -------
    rate_info : dict of parameters
        Keys are 'simple', 'plog', 'cheb', 'fall', 'chem', 'thd'
        Values are further dictionaries including addtional rate info, number,
        offset, maps, etc.
    """

    assert rate_spec in RateSpecialization
    # determine specialization
    full = rate_spec == RateSpecialization.full
    # hybrid = rate_spec == RateSpecialization.hybrid
    fixed = rate_spec == RateSpecialization.fixed

    # find fwd / reverse rate parameters
    # first, the number of each
    rev_map = np.array([i for i, x in enumerate(reacs) if x.rev],
                       dtype=arc.kint_type)
    num_rev = len(rev_map)
    # next, find the species / nu values
    nu_sum = []
    net_num_spec = []
    net_spec = []
    net_nu = []
    reac_has_ns = []
    ns_nu = []
    for i_rxn, rxn in enumerate(reacs):
        # get list of species in reaction
        spec_list = sorted(set(rxn.reac[:] + rxn.prod[:]))
        # add species / num
        net_spec.extend(spec_list)
        net_num_spec.append(len(spec_list))
        # get fwd / reverse nu for species
        for spec in spec_list:
            # get reactant index
            ind = next((i for i, x in enumerate(rxn.reac) if x == spec), None)
            reac_nu = rxn.reac_nu[ind] if ind is not None else 0

            # get product index
            ind = next((i for i, x in enumerate(rxn.prod) if x == spec), None)
            prod_nu = rxn.prod_nu[ind] if ind is not None else 0

            # and add nu values
            net_nu.extend([prod_nu, reac_nu])

        # and nu sum for equilibrium constants
        nu_sum.append(sum([utils.get_nu(isp, rxn) for isp in spec_list]))

        # handle fwd / rev nu for last species indicator
        ns_reac_ind = next((i for i, x in enumerate(rxn.reac[:])
                            if x == len(specs) - 1), None)
        ns_reac_nu = rxn.reac_nu[ns_reac_ind] if ns_reac_ind is not None else 0
        ns_prod_ind = next((i for i, x in enumerate(rxn.prod[:])
                            if x == len(specs) - 1), None)
        ns_prod_nu = rxn.prod_nu[ns_prod_ind] if ns_prod_ind is not None else 0
        if ns_reac_nu or ns_prod_nu:
            reac_has_ns.append(i_rxn)
            ns_nu.extend([ns_prod_nu, ns_reac_nu])

    # create numpy versions
    reac_has_ns = np.array(reac_has_ns, dtype=arc.kint_type)
    net_nu_integer = all(utils.is_integer(nu) for nu in net_nu)
    if net_nu_integer:
        nu_sum = np.array(nu_sum, dtype=arc.kint_type)
        net_nu = np.array(net_nu, dtype=arc.kint_type)
        ns_nu = np.array(ns_nu, dtype=arc.kint_type)
    else:
        nu_sum = np.array(nu_sum)
        net_nu = np.array(net_nu)
        ns_nu = np.array(ns_nu)
    net_num_spec = np.array(net_num_spec, dtype=arc.kint_type)
    net_spec = np.array(net_spec, dtype=arc.kint_type)

    # sometimes we need the net properties forumlated per species rather than
    # per reaction as above
    spec_to_reac = []
    spec_nu = []
    spec_reac_count = []
    spec_list = []
    for ispec, spec in enumerate(specs):
        # first, find all non-zero nu reactions
        reac_list = [x for x in [(irxn, utils.get_nu(ispec, rxn))
                                 for irxn, rxn in enumerate(reacs)] if x[1]]
        if reac_list:
            reac_list, nu_list = zip(*reac_list)
            spec_to_reac.extend(reac_list)
            spec_nu.extend(nu_list)
            spec_reac_count.append(len(reac_list))
            spec_list.append(ispec)

    spec_to_reac = np.array(spec_to_reac, dtype=arc.kint_type)
    if net_nu_integer:
        spec_nu = np.array(spec_nu, dtype=arc.kint_type)
    else:
        spec_nu = np.array(spec_nu)
    spec_reac_count = np.array(spec_reac_count, dtype=arc.kint_type)
    spec_list = np.array(spec_list, dtype=arc.kint_type)

    def __seperate(reacs, matchers):
        # find all reactions / indicies that match this offset
        rate = [(i, x) for i, x in enumerate(reacs) if any(x.match(y) for y in
                                                           matchers)]
        mapping = np.empty(0, dtype=arc.kint_type)
        num = 0
        if rate:
            mapping, rate = zip(*rate)
            mapping = np.array(mapping, dtype=arc.kint_type)
            rate = list(rate)
            num = len(rate)

        return rate, mapping, num

    # count / seperate reactions with simple arrhenius rates
    simple_rate, simple_map, num_simple = __seperate(
        reacs, [reaction_type.elementary, reaction_type.thd,
                reaction_type.fall, reaction_type.chem])

    def __specialize(rates, fall=False):
        fall_types = None
        num = len(rates)
        rate_type = np.zeros((num,), dtype=arc.kint_type)
        if fall:
            fall_types = np.zeros((num,), dtype=arc.kint_type)
        # reaction parameters
        A = np.zeros((num,), dtype=np.float64)
        b = np.zeros((num,), dtype=np.float64)
        Ta = np.zeros((num,), dtype=np.float64)

        for i, reac in enumerate(rates):
            if (reac.high or reac.low) and fall:
                if reac.high:
                    Ai, bi, Tai = reac.high
                    # mark as chemically activated
                    fall_types[i] = int(reaction_type.chem)
                else:
                    # we want k0, hence default factor is fine
                    Ai, bi, Tai = reac.low
                    fall_types[i] = int(reaction_type.fall)  # mark as falloff
            else:
                # assign rate params
                Ai, bi, Tai = reac.A, reac.b, reac.E
            # generic assign
            A[i] = np.log(Ai)
            b[i] = bi
            Ta[i] = Tai

            if fixed:
                rate_type[i] = 0
                continue
            # assign rate types
            if bi == 0 and Tai == 0:
                A[i] = Ai
                rate_type[i] = 0
            elif bi == int(bi) and bi and Tai == 0:
                A[i] = Ai
                rate_type[i] = 1
            elif Tai == 0 and bi != 0:
                rate_type[i] = 2
            elif bi == 0 and Tai != 0:
                rate_type[i] = 3
            else:
                rate_type[i] = 4
            if not full:
                rate_type[i] = rate_type[i] if rate_type[i] <= 1 else 2

        # subtract off the falloff type to make this a simple 0/1 comparison
        if fall:
            fall_types -= int(reaction_type.fall)
        return rate_type, A, b, Ta, fall_types

    simple_rate_type, A, b, Ta, _ = __specialize(simple_rate)

    # finally determine the advanced rate evaulation types
    plog_reacs, plog_map, num_plog = __seperate(
        reacs, [reaction_type.plog])

    # create the plog arrays
    num_pressures = []
    plog_params = []
    for p in plog_reacs:
        num_pressures.append(len(p.plog_par))
        plog_params.append(p.plog_par)
    num_pressures = np.array(num_pressures, dtype=arc.kint_type)

    cheb_reacs, cheb_map, num_cheb = __seperate(
        reacs, [reaction_type.cheb])

    # create the chebyshev arrays
    cheb_n_pres = []
    cheb_n_temp = []
    cheb_plim = []
    cheb_tlim = []
    cheb_coeff = []
    for cheb in cheb_reacs:
        cheb_n_pres.append(cheb.cheb_n_pres)
        cheb_n_temp.append(cheb.cheb_n_temp)
        cheb_coeff.append(cheb.cheb_par)
        cheb_plim.append(cheb.cheb_plim)
        cheb_tlim.append(cheb.cheb_tlim)
    cheb_n_pres = np.array(cheb_n_pres, dtype=arc.kint_type)
    cheb_n_temp = np.array(cheb_n_temp, dtype=arc.kint_type)
    cheb_plim = np.array(cheb_plim)
    cheb_tlim = np.array(cheb_tlim)
    cheb_coeff = np.array(cheb_coeff)

    # find falloff types
    fall_reacs, fall_map, num_fall = __seperate(
        reacs, [reaction_type.fall, reaction_type.chem])
    fall_rate_type, fall_A, fall_b, fall_Ta, fall_types = __specialize(
        fall_reacs, True)
    # find blending type
    blend_type = np.array([next(int(y) for y in x.type if isinstance(
        y, falloff_form)) for x in fall_reacs], dtype=arc.kint_type)
    # seperate parameters based on blending type
    # lindeman
    lind_map = np.where(blend_type == int(falloff_form.lind))[
        0].astype(dtype=arc.kint_type)
    # sri
    sri_map = np.where(blend_type == int(falloff_form.sri))[
        0].astype(dtype=arc.kint_type)
    sri_reacs = [reacs[fall_map[i]] for i in sri_map]
    sri_par = [reac.sri_par for reac in sri_reacs]
    # now fill in defaults as needed
    for par_set in sri_par:
        if len(par_set) != 5:
            par_set.extend([1, 0])
    if len(sri_par):
        sri_a, sri_b, sri_c, sri_d, sri_e = [
            np.array(x, dtype=np.float64) for x in zip(*sri_par)]
    else:
        sri_a, sri_b, sri_c, sri_d, sri_e = [
            np.empty(shape=(0,)) for i in range(5)]
    # and troe
    troe_map = np.where(blend_type == int(falloff_form.troe))[
        0].astype(dtype=arc.kint_type)
    troe_reacs = [reacs[fall_map[i]] for i in troe_map]
    troe_par = [reac.troe_par for reac in troe_reacs]
    # now fill in defaults as needed
    for par_set in troe_par:
        if len(par_set) != 4:
            par_set.append(0)
    try:
        troe_a, troe_T3, troe_T1, troe_T2 = [
            np.array(x, dtype=np.float64) for x in zip(*troe_par)]
        # and invert
        troe_T1 = 1. / troe_T1
        troe_T3 = 1. / troe_T3
    except ValueError:
        troe_a = np.empty(0)
        troe_T3 = np.empty(0)
        troe_T1 = np.empty(0)
        troe_T2 = np.empty(0)

    # find third-body types
    thd_reacs, thd_map, num_thd = __seperate(
        reacs, [reaction_type.fall, reaction_type.chem, reaction_type.thd])
    # find third body type
    thd_type = np.array([next(int(y) for y in x.type if isinstance(
        y, thd_body_type)) for x in thd_reacs], dtype=arc.kint_type)

    # first, we must do some surgery to get _our_ form of the thd-body
    # efficiencies
    last_spec = len(specs) - 1
    thd_spec_num = []
    thd_spec = []
    thd_eff = []
    thd_has_ns = []
    thd_ns_eff = []
    for i, x in enumerate(thd_reacs):
        if x.match(thd_body_type.species):
            thd_spec_num.append(1)
            thd_spec.append(x.pdep_sp)
            thd_eff.append(1)
            if x.pdep_sp == last_spec:
                thd_has_ns.append(i)
                thd_ns_eff.append(1)

        elif x.match(thd_body_type.unity):
            thd_spec_num.append(0)
        else:
            thd_spec_num.append(len(x.thd_body_eff))
            spec, eff = zip(*x.thd_body_eff)
            thd_spec.extend(spec)
            thd_eff.extend(eff)
            ind = next((ind for ind, s in enumerate(spec) if s == last_spec),
                       None)
            if ind is not None:
                thd_has_ns.append(i)
                thd_ns_eff.append(eff[ind])

    thd_spec_num = np.array(thd_spec_num, dtype=arc.kint_type)
    thd_spec = np.array(thd_spec, dtype=arc.kint_type)
    thd_eff = np.array(thd_eff, dtype=np.float64)
    thd_has_ns = np.array(thd_has_ns, dtype=arc.kint_type)
    thd_ns_eff = np.array(thd_ns_eff, dtype=np.float64)

    # thermo properties
    poly_dim = specs[0].hi.shape[0]
    Ns = len(specs)

    # pick out a values and T_mid
    a_lo = np.zeros((Ns, poly_dim), dtype=np.float64)
    a_hi = np.zeros((Ns, poly_dim), dtype=np.float64)
    T_mid = np.zeros((Ns,), dtype=np.float64)
    for ind, spec in enumerate(specs):
        a_lo[ind, :] = spec.lo[:]
        a_hi[ind, :] = spec.hi[:]
        T_mid[ind] = spec.Trange[1]

    # post processing

    # chebyshev parameter reordering
    pp_cheb_coeff = np.empty(0)
    pp_cheb_plim = np.empty(0)
    pp_cheb_tlim = np.empty(0)
    if num_cheb:
        pp_cheb_coeff = np.zeros((num_cheb, int(np.max(cheb_n_temp)),
                                  int(np.max(cheb_n_pres))))
        for i, p in enumerate(cheb_coeff):
            pp_cheb_coeff[i, :cheb_n_temp[i], :cheb_n_pres[i]] = p[:, :]

        # limits for cheby polys
        pp_cheb_plim = np.log(np.array(cheb_plim, dtype=np.float64))
        pp_cheb_tlim = 1. / np.array(cheb_tlim, dtype=np.float64)

    # plog parameter reorder
    pp_plog_params = np.empty(0)
    maxP = None
    if num_plog:
        # max # of parameters for sizing
        maxP = np.max(num_pressures)
        # for simplicity, we're going to use a padded form
        pp_plog_params = np.zeros((4, num_plog, maxP))
        for m in range(4):
            for i, numP in enumerate(num_pressures):
                for j in range(numP):
                    pp_plog_params[m, i, j] = plog_params[i][j][m]

        # take the log of P and A
        hold = np.seterr(divide='ignore')
        pp_plog_params[0, :, :] = np.log(pp_plog_params[0, :, :])
        pp_plog_params[1, :, :] = np.log(pp_plog_params[1, :, :])
        pp_plog_params[np.where(np.isinf(pp_plog_params))] = 0
        np.seterr(**hold)

    # molecular weights
    mws = np.array([spec.mw for spec in specs])
    mw_post = mws[:-1] / mws[-1]

    # min / max T
    minT = None
    maxT = None
    for spec in specs:
        if minT is None:
            minT = spec.Trange[0]
        else:
            minT = np.maximum(minT, spec.Trange[0])
        if maxT is None:
            maxT = spec.Trange[-1]
        else:
            maxT = np.minimum(maxT, spec.Trange[-1])

    return {'simple': {'A': A, 'b': b, 'Ta': Ta, 'type': simple_rate_type,
                       'num': num_simple, 'map': simple_map},
            'plog': {'map': plog_map, 'num': num_plog,
                     'num_P': num_pressures, 'params': plog_params,
                     'max_P': maxP,
                     'post_process': {'params': pp_plog_params},
                     },
            'cheb': {'map': cheb_map, 'num': num_cheb,
                     'num_P': cheb_n_pres, 'num_T': cheb_n_temp,
                     'params': cheb_coeff, 'Tlim': cheb_tlim,
                     'Plim': cheb_plim,
                     'post_process': {
                         'params': pp_cheb_coeff,
                         'Plim': pp_cheb_plim,
                         'Tlim': pp_cheb_tlim
                     }},
            'fall': {'map': fall_map, 'num': num_fall,
                     'ftype': fall_types, 'blend': blend_type,
                     'A': fall_A, 'b': fall_b, 'Ta': fall_Ta,
                     'type': fall_rate_type,
                     'sri':
                     {'map': sri_map,
                         'num': sri_map.size,
                         'a': sri_a,
                         'b': sri_b,
                         'c': sri_c,
                         'd': sri_d,
                         'e': sri_e
                      },
                     'troe':
                     {'map': troe_map,
                         'num': troe_map.size,
                         'a': troe_a,
                         'T3': troe_T3,
                         'T1': troe_T1,
                         'T2': troe_T2
                      },
                     'lind': {'map': lind_map,
                              'num': lind_map.size}
                     },
            'thd': {'map': thd_map, 'num': num_thd,
                    'type': thd_type, 'spec_num': thd_spec_num,
                    'spec': thd_spec, 'eff': thd_eff,
                    'has_ns': thd_has_ns, 'eff_ns': thd_ns_eff},
            'fwd': {'map': np.arange(len(reacs)), 'num': len(reacs)},
            'rev': {'map': rev_map, 'num': num_rev},
            'net': {'num_reac_to_spec': net_num_spec, 'nu_sum': nu_sum,
                    'nu': net_nu, 'reac_to_spec': net_spec,
                    'allint': net_nu_integer},
            'net_per_spec': {'reac_count': spec_reac_count, 'nu': spec_nu,
                             'reacs': spec_to_reac, 'map': spec_list,
                             'allint': net_nu_integer},
            'Nr': len(reacs),
            'Ns': len(specs),
            'thermo': {
                'a_lo': a_lo,
                'a_hi': a_hi,
                'T_mid': T_mid},
            'mws': mws,
            'mw_post': mw_post,
            'reac_has_ns': reac_has_ns,
            'ns_nu': ns_nu,
            'minT': minT,
            'maxT': maxT
            }


def reset_arrays(loopy_opts, namestore, test_size=None):
    """Resets the dphi and wdot arrays for use in the rate evaluations

    kernel

    Parameters
    ----------
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
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    def __create(arr, nrange, name):
        # create reset kernel
        mapstore = arc.MapStore(loopy_opts, nrange, test_size)

        # first, create all arrays
        kernel_data = []

        # add problem size
        kernel_data.extend(arc.initial_condition_dimension_vars(
            loopy_opts, test_size))

        # need arrays
        arr_lp, arr_str = mapstore.apply_maps(arr, *default_inds)
        kernel_data.append(arr_lp)
        instructions = Template(
            """
                ${arr_str} = 0d {id=set}
            """).substitute(**locals())

        # currently both reset arrays are atomic (if used)
        can_vectorize, vec_spec = ic.get_deep_specializer(
            loopy_opts, init_ids=['set'])

        return k_gen.knl_info(name=name,
                              instructions=instructions,
                              mapstore=mapstore,
                              var_name=var_name,
                              kernel_data=kernel_data,
                              can_vectorize=can_vectorize,
                              vectorization_specializer=vec_spec)

    return [__create(namestore.n_dot, namestore.phi_inds, 'ndot_reset'),
            __create(namestore.spec_rates, namestore.num_specs, 'wdot_reset')]


def get_concentrations(loopy_opts, namestore, conp=True,
                       test_size=None):
    """Determines concentrations from moles and state variables depending
    on constant pressure vs constant volue assumption

    kernel

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    mapstore = arc.MapStore(loopy_opts, namestore.num_specs_no_ns, test_size)

    fixed_inds = (global_ind,)

    # first, create all arrays
    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # need P, V, T and n arrays

    # add / apply maps
    mapstore.check_and_add_transform(namestore.n_arr,
                                     namestore.phi_spec_inds,
                                     force_inline=True)
    mapstore.check_and_add_transform(namestore.conc_arr,
                                     namestore.num_specs_no_ns,
                                     force_inline=True)
    mapstore.check_and_add_transform(namestore.conc_ns_arr,
                                     namestore.num_specs_no_ns,
                                     force_inline=True)

    n_arr, n_str = mapstore.apply_maps(namestore.n_arr, *default_inds)
    P_arr, P_str = mapstore.apply_maps(namestore.P_arr, *fixed_inds)
    V_arr, V_str = mapstore.apply_maps(namestore.V_arr, *fixed_inds)
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, *fixed_inds)
    conc_arr, conc_str = mapstore.apply_maps(namestore.conc_arr, *default_inds)

    _, conc_ns_str = mapstore.apply_maps(namestore.conc_ns_arr, *fixed_inds)

    # add arrays
    kernel_data.extend([n_arr, P_arr, V_arr, T_arr, conc_arr])

    precompute = ic.PrecomputedInstructions(loopy_opts)
    V_inv = 'V_inv'
    V_inv_insn = precompute(V_inv, V_str, 'INV', guard=ic.VolumeGuard(loopy_opts))
    T_val = 'T_val'
    T_val_insn = precompute(T_val, T_str, 'VAL', guard=ic.TemperatureGuard(
        loopy_opts))

    pre_instructions = Template(
        """${V_inv_insn}
           ${T_val_insn}
           <>n_sum = 0 {id=n_init}
           ${conc_ns_str} = ${P_str} / (R_u * ${T_val}) {id=cns_init}
        """).substitute(**locals())

    mole_guard = ic.Guard(loopy_opts, minv=utils.small)
    n_guarded = mole_guard(n_str)
    nsp_guarded = mole_guard('n_sum')

    instructions = Template(
        """
            <> n = ${n_guarded}
            ${conc_str} = n * V_inv {id=cn_init}
            n_sum = n_sum + n {id=n_update, dep=n_init}
        """).substitute(**locals())

    barrier = ic.get_barrier(loopy_opts, id='break', dep='cns_init')
    post_instructions = Template(
        """
        ${barrier}
        ${conc_ns_str} = ${conc_ns_str} - ${nsp_guarded} * V_inv \
            {id=cns_set, dep=n_update:break, nosync=cns_init}
        """).substitute(**locals())

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['cns_set'],
        init_ids=['cns_init', 'cn_init'])

    return k_gen.knl_info(name='get_concentrations',
                          pre_instructions=[pre_instructions],
                          instructions=instructions,
                          post_instructions=[post_instructions],
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec,
                          manglers=mole_guard.manglers,
                          parameters={'R_u': np.float64(chem.RU)})


def get_molar_rates(loopy_opts, namestore, conp=True,
                    test_size=None):
    """Generates instructions, kernel arguements, and data for the
       molar derivatives
    kernel

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    mapstore = arc.MapStore(loopy_opts, namestore.num_specs_no_ns, test_size)

    # first, create all arrays
    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    fixed_inds = (global_ind,)

    # wdot, dphi, V

    # add / apply maps
    mapstore.check_and_add_transform(namestore.n_dot,
                                     namestore.phi_spec_inds)

    mapstore.check_and_add_transform(namestore.spec_rates,
                                     namestore.num_specs_no_ns)

    wdot_lp, wdot_str = mapstore.apply_maps(namestore.spec_rates,
                                            *default_inds)

    ndot_lp, ndot_str = mapstore.apply_maps(namestore.n_dot,
                                            *default_inds)

    V_lp, V_str = mapstore.apply_maps(namestore.V_arr, *fixed_inds)

    V_val = 'V_val'
    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)

    pre_instructions = precompute(V_val, V_str, 'VAL', guard=ic.VolumeGuard(
        loopy_opts))

    kernel_data.extend([V_lp, ndot_lp, wdot_lp])

    instructions = Template(
        """
        ${ndot_str} = ${V_val} * ${wdot_str} {id=set}
        """
    ).substitute(
        ndot_str=ndot_str,
        V_val=V_val,
        wdot_str=wdot_str
    )

    vec_spec = None
    can_vectorize = True
    if ic.use_atomics(loopy_opts):
        can_vectorize, vec_spec = ic.get_deep_specializer(
            loopy_opts, init_ids=['set'])

    return k_gen.knl_info(name='get_molar_rates',
                          pre_instructions=[pre_instructions],
                          instructions=instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec)


def get_extra_var_rates(loopy_opts, namestore, conp=True,
                        test_size=None):
    """Generates instructions, kernel arguements, and data for the
       derivative of the "extra" variable -- P or V depending on conV/conP
       assumption respectively
    kernel

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    mapstore = arc.MapStore(loopy_opts, namestore.num_specs_no_ns, test_size)

    # first, create all arrays
    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    fixed_inds = (global_ind,)

    # get arrays

    mapstore.check_and_add_transform(namestore.spec_rates,
                                     namestore.num_specs_no_ns,
                                     force_inline=True)

    mapstore.check_and_add_transform(namestore.mw_post_arr,
                                     namestore.num_specs_no_ns,
                                     force_inline=True)

    wdot_lp, wdot_str = mapstore.apply_maps(namestore.spec_rates,
                                            *default_inds)

    Edot_lp, Edot_str = mapstore.apply_maps(namestore.E_dot,
                                            *fixed_inds)

    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, *fixed_inds)
    P_lp, P_str = mapstore.apply_maps(namestore.P_arr, *fixed_inds)
    Tdot_lp, Tdot_str = mapstore.apply_maps(namestore.T_dot, *fixed_inds)
    mw_lp, mw_str = mapstore.apply_maps(namestore.mw_post_arr, *(var_name,))
    V_lp, V_str = mapstore.apply_maps(namestore.V_arr, *fixed_inds)

    kernel_data.extend([wdot_lp, Edot_lp, T_lp, P_lp, Tdot_lp, mw_lp])

    pre = ['<>dE = 0.0d {id=dE_init}']
    precompute = ic.PrecomputedInstructions(loopy_opts)
    T_val = 'T'
    pre.append(precompute(T_val, T_str, 'VAL', guard=ic.TemperatureGuard(
            loopy_opts)))

    if conp:
        V_val = 'V_val'
        pre.append(precompute(V_val, V_str, 'VAL', guard=ic.VolumeGuard(loopy_opts)))
        pre_instructions = [
            Template('${Edot_str} = ${V_val} * ${Tdot_str} / ${T_val} \
                     {id=init, dep=precompute*}').safe_substitute(
                **locals()),
        ] + pre
    else:
        pre_instructions = [
            Template('${Edot_str} = ${P_str} * ${Tdot_str} / ${T_val} \
                     {id=init, dep=precompute*}').safe_substitute(
                **locals()),
        ] + pre

    instructions = Template(
        """
            dE = dE + (1.0 - ${mw_str}) * ${wdot_str} {id=sum, dep=dE_init}
        """
    ).safe_substitute(**locals())

    if conp:
        kernel_data.append(V_lp)

        if ic.use_atomics(loopy_opts):
            # need to fix the post instructions to work atomically
            pre_instructions = pre[:]
            post_instructions = [Template(
                """
                temp_sum[0] = ${V_val} * ${Tdot_str} / ${T_val} \
                    {id=temp_init, dep=*:precompute*, atomic}
                ... lbarrier {id=lb1, dep=temp_init}
                temp_sum[0] = temp_sum[0] + \
                    ${V_val} * dE * ${T_val} * R_u / ${P_str} \
                    {id=temp_sum, dep=lb1*:sum, nosync=temp_init, atomic}
                ... lbarrier {id=lb2, dep=temp_sum}
                ${Edot_str} = temp_sum[0] \
                    {id=final, dep=lb2, atomic, nosync=temp_init:temp_sum}
                """
            ).safe_substitute(**locals())]
            kernel_data.append(lp.TemporaryVariable('temp_sum', dtype=np.float64,
                                                    scope=scopes.LOCAL, shape=(1,)))
        else:
            post_instructions = [Template(
                """
                ${Edot_str} = ${Edot_str} + ${V_val} * dE * ${T_val} * R_u / \
                    ${P_str} {id=end, dep=sum:init, nosync=init}
                """).safe_substitute(**locals())
            ]

    else:
        if ic.use_atomics(loopy_opts):
            # need to fix the post instructions to work atomically
            pre_instructions = pre[:]
            post_instructions = [Template(
                """
                temp_sum[0] = ${P_str} * ${Tdot_str} / ${T_val} \
                    {id=temp_init, dep=*:precompute*, atomic}
                ... lbarrier {id=lb1, dep=temp_init}
                temp_sum[0] = temp_sum[0] + ${T_val} * R_u * dE \
                    {id=temp_sum, dep=lb1*:sum, nosync=temp_init, atomic}
                ... lbarrier {id=lb2, dep=temp_sum}
                ${Edot_str} = temp_sum[0] \
                    {id=final, dep=lb2, atomic, nosync=temp_init:temp_sum}
                """
            ).safe_substitute(**locals())]
            kernel_data.append(lp.TemporaryVariable('temp_sum', dtype=np.float64,
                                                    scope=scopes.LOCAL, shape=(1,)))
        else:
            post_instructions = [Template(
                """
                ${Edot_str} = ${Edot_str} + ${T_val} * R_u * dE {id=end, \
                    dep=sum:init, nosync=init}
                """
            ).safe_substitute(**locals())]

    if loopy_opts.depth:
        can_vectorize, vec_spec = ic.get_deep_specializer(
            loopy_opts, atomic_ids=['final', 'temp_sum'],
            init_ids=['init', 'temp_init'])
    else:
        can_vectorize = True
        vec_spec = ic.write_race_silencer(['end'])

    return k_gen.knl_info(name='get_extra_var_rates',
                          pre_instructions=pre_instructions,
                          instructions=instructions,
                          post_instructions=post_instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          parameters={'R_u': np.float64(chem.RU)},
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec,
                          silenced_warnings=['write_race(end)'])


def get_temperature_rate(loopy_opts, namestore, conp=True,
                         test_size=None):
    """Generates instructions, kernel arguements, and data for the
       temperature derivative
    kernel

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    mapstore = arc.MapStore(loopy_opts, namestore.num_specs, test_size)
    fixed_inds = (global_ind,)

    # first, create all arrays
    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # add / apply maps
    mapstore.check_and_add_transform(namestore.spec_rates,
                                     namestore.num_specs,
                                     force_inline=True)
    mapstore.check_and_add_transform(namestore.conc_arr,
                                     namestore.num_specs,
                                     force_inline=True)

    # add all non-mapped arrays
    if conp:
        h_lp, h_str = mapstore.apply_maps(namestore.h, *default_inds)
        cp_lp, cp_str = mapstore.apply_maps(namestore.cp, *default_inds)
        energy_str = h_str
        spec_heat_str = cp_str
        kernel_data.extend([h_lp, cp_lp])
    else:
        u_lp, u_str = mapstore.apply_maps(namestore.u, *default_inds)
        cv_lp, cv_str = mapstore.apply_maps(namestore.cv, *default_inds)
        energy_str = u_str
        spec_heat_str = cv_str
        kernel_data.extend([u_lp, cv_lp])

    conc_lp, conc_str = mapstore.apply_maps(namestore.conc_arr, *default_inds)
    Tdot_lp, Tdot_str = mapstore.apply_maps(namestore.T_dot, *fixed_inds)
    wdot_lp, wdot_str = mapstore.apply_maps(namestore.spec_rates,
                                            *default_inds)

    kernel_data.extend([conc_lp, Tdot_lp, wdot_lp])

    pre_instructions = [Template(
        """
        <> upper = 0 {id=uinit}
        <> lower = 0 {id=linit}
        # handled by reset_arrays
        # ${Tdot_str} = 0 {id=init}
        """).safe_substitute(**locals())]
    instructions = Template(
        """
            upper = upper + ${energy_str} * ${wdot_str} {id=sum1, dep=uinit}
            lower = lower + ${conc_str} * ${spec_heat_str} {id=sum2, dep=linit}
        """
    ).safe_substitute(**locals())

    post_instructions = [Template(
        """
        ${Tdot_str} = ${Tdot_str} - upper / lower {id=final, dep=sum*}
        """
    ).safe_substitute(**locals())]

    if ic.use_atomics(loopy_opts):
        # need to fix the post instructions to work atomically
        post_instructions = [Template(
            """
            temp_sum[0] = 0 {id=temp_init, atomic}
            ... lbarrier {id=lb1, dep=temp_init}
            temp_sum[0] = temp_sum[0] + lower {id=temp_sum, dep=lb1:sum*, \
                nosync=temp_init, atomic}
            ... lbarrier {id=lb2, dep=temp_sum}
            ${Tdot_str} = ${Tdot_str} - upper / temp_sum[0] \
                {id=final, dep=lb2, nosync=temp_sum:temp_init, atomic}
            """
        ).safe_substitute(**locals())]
        kernel_data.append(lp.TemporaryVariable('temp_sum', dtype=np.float64,
                                                scope=scopes.LOCAL, shape=(1,)))

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, init_ids=['init', 'temp_init'],
        atomic_ids=['temp_sum', 'final'])

    return k_gen.knl_info(name='temperature_rate',
                          pre_instructions=pre_instructions,
                          instructions=instructions,
                          post_instructions=post_instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec)


def get_spec_rates(loopy_opts, namestore, conp=True,
                   test_size=None):
    """Generates instructions, kernel arguements, and data for the
       temperature derivative
    kernel

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # various indicies
    spec_ind = 'spec_ind'
    ispec = 'ispec'

    # create map store
    mapstore = arc.MapStore(loopy_opts, namestore.num_reacs, test_size)

    # create arrays
    spec_lp, spec_str = mapstore.apply_maps(namestore.rxn_to_spec,
                                            ispec)
    num_spec_offsets_lp, \
        num_spec_offsets_str = \
        mapstore.apply_maps(namestore.rxn_to_spec_offsets, var_name)
    num_spec_offsets_next_lp, \
        num_spec_offsets_next_str = \
        mapstore.apply_maps(namestore.rxn_to_spec_offsets,
                            var_name, affine=1)
    nu_lp, prod_nu_str = mapstore.apply_maps(
        namestore.rxn_to_spec_prod_nu, ispec, affine=ispec)
    _, reac_nu_str = mapstore.apply_maps(
        namestore.rxn_to_spec_reac_nu, ispec, affine=ispec)
    rop_net_lp, rop_net_str = mapstore.apply_maps(namestore.rop_net,
                                                  *default_inds)
    wdot_lp, wdot_str = mapstore.apply_maps(namestore.spec_rates,
                                            global_ind, spec_ind)

    # update kernel args
    kernel_data.extend(
        [spec_lp, num_spec_offsets_lp, nu_lp, rop_net_lp, wdot_lp])

    # now the instructions
    instructions = Template(
        """
    <>net_rate = ${rop_net_str} {id=rate_init}
    <>offset = ${num_spec_offsets_str}
    <>offset_next = ${num_spec_offsets_next_str}
    for ispec
        <> ${spec_ind} = ${spec_str} # (offset handled in wdot str)
        <> nu = ${prod_nu_str} - ${reac_nu_str}
        ${wdot_str} = ${wdot_str} + nu * net_rate {id=sum}
    end
    """).safe_substitute(**locals())

    # extra inames
    extra_inames = [('ispec', 'offset <= ispec < offset_next')]

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['sum'])

    return k_gen.knl_info(name='spec_rates',
                          instructions=instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          extra_inames=extra_inames,
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec)


def get_rop_net(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the net
    Rate of Progress kernels

    Parameters
    ----------
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

    # create net rop kernel

    kernel_data = OrderedDict([('fwd', [])])
    maps = OrderedDict([('fwd',
                         arc.MapStore(loopy_opts, namestore.num_reacs, test_size))])
    transforms = {'fwd': namestore.num_reacs}

    separated_kernels = loopy_opts.rop_net_kernels
    if separated_kernels:
        kernel_data['rev'] = []
        maps['rev'] = arc.MapStore(loopy_opts, namestore.num_rev_reacs, test_size)
        transforms['rev'] = namestore.rev_map
        kernel_data['pres_mod'] = []
        maps['pres_mod'] = arc.MapStore(loopy_opts, namestore.num_thd, test_size)
        transforms['pres_mod'] = namestore.thd_map
    else:
        transforms['rev'] = namestore.rev_mask
        transforms['pres_mod'] = namestore.thd_mask

    def __add_data(knl, data):
        if separated_kernels:
            kernel_data[knl].append(data)
        else:
            kernel_data['fwd'].append(data)

    def __add_to_all(data):
        for kernel in kernel_data:
            try:
                for d in data:
                    __add_data(kernel, d)
            except TypeError:
                __add_data(kernel, data)

    def __get_map(knl):
        if separated_kernels:
            return maps[knl]
        else:
            return maps['fwd']

    # add transforms
    if not separated_kernels:
        # if separate kernels, we're looping over the rev / pres mod indicies
        # directly, hence no transforms
        if namestore.rop_rev:
            __get_map('rev').check_and_add_transform(
                namestore.rop_rev, transforms['rev'])

        if namestore.pres_mod:
            __get_map('pres_mod').check_and_add_transform(
                namestore.pres_mod, transforms['pres_mod'])

    for name in kernel_data:
        # check for map / mask
        __get_map(name).\
            check_and_add_transform(namestore.rop_net,
                                    transforms[name])

    __add_to_all(
        arc.initial_condition_dimension_vars(loopy_opts, test_size))

    # create the fwd rop array / str
    # this never has a map / mask
    rop_fwd_lp, rop_fwd_str = __get_map('fwd').\
        apply_maps(namestore.rop_fwd, *default_inds)

    __add_data('fwd', rop_fwd_lp)

    if namestore.rop_rev is not None:
        # we have reversible reactions

        # apply the maps
        rop_rev_lp, rop_rev_str = __get_map('rev').\
            apply_maps(namestore.rop_rev, *default_inds)

        # add data
        __add_data('rev', rop_rev_lp)

    if namestore.pres_mod is not None:
        # we have pres mod reactions

        # apply the maps
        pres_mod_lp, pres_mod_str = __get_map('pres_mod').\
            apply_maps(namestore.pres_mod, *default_inds)

        # add data
        __add_data('pres_mod', pres_mod_lp)

    # add rop net to all kernels:
    rop_strs = {}
    for name in kernel_data:

        # apply map
        rop_net_lp, rop_net_str = __get_map(name).\
            apply_maps(namestore.rop_net, *default_inds)

        # and add to data
        __add_data(name, rop_net_lp)

        # store rop_net str for later
        rop_strs[name] = rop_net_str

    if not separated_kernels:
        # now the instructions
        instructions = Template(
            """
        <>net_rate = ${rop_fwd_str} {id=rate_update}
        ${rev_update}
        ${pmod_update}
        ${rop_net_str} = net_rate {dep=rate_update*}
        """).safe_substitute(rop_fwd_str=rop_fwd_str,
                             rop_net_str=rop_strs['fwd'])

        # reverse update
        rev_update_instructions = ic.get_update_instruction(
            __get_map('rev'), namestore.rop_rev,
            Template(
                """
            net_rate = net_rate - ${rop_rev_str} \
                {id=rate_update_rev, dep=rate_update}
            """).safe_substitute(
                rop_rev_str=rop_rev_str))

        # pmod update
        pmod_update_instructions = ic.get_update_instruction(
            __get_map('pres_mod'), namestore.pres_mod,
            Template(
                """
            net_rate = net_rate * ${pres_mod_str} \
                {id=rate_update_pmod, dep=rate_update${rev_dep}}
            """).safe_substitute(
                rev_dep=':rate_update_rev' if namestore.rop_rev is not None
                    else '',
                pres_mod_str=pres_mod_str))

        instructions = Template(instructions).safe_substitute(
            rev_update=rev_update_instructions,
            pmod_update=pmod_update_instructions)

        instructions = '\n'.join(
            [x for x in instructions.split('\n') if x.strip()])

        return k_gen.knl_info(name='rop_net_fixed',
                              instructions=instructions,
                              var_name=var_name,
                              kernel_data=kernel_data['fwd'],
                              mapstore=maps['fwd'])

    else:
        infos = []
        for kernel in kernel_data:
            if kernel == 'fwd':
                instructions = Template(
                    """
            ${rop_net_str} = ${rop_fwd_str} {id=rop_net_fwd}
                    """).safe_substitute(rop_fwd_str=rop_fwd_str,
                                         rop_net_str=rop_strs['fwd'])
            elif kernel == 'rev':
                instructions = Template(
                    """
            ${rop_net_str} = ${rop_net_str} - ${rop_rev_str} {id=rop_net_rev}
                    """).safe_substitute(rop_rev_str=rop_rev_str,
                                         rop_net_str=rop_strs['rev'])
            else:
                instructions = Template(
                    """
            ${rop_net_str} = ${rop_net_str} * ${pres_mod_str} {id=rop_net_pres_mod}
                    """).safe_substitute(pres_mod_str=pres_mod_str,
                                         rop_net_str=rop_strs['pres_mod'])

            instructions = '\n'.join(
                [x for x in instructions.split('\n') if x.strip()])
            infos.append(k_gen.knl_info(name='rop_net_{}'.format(kernel),
                                        instructions=instructions,
                                        var_name=var_name,
                                        kernel_data=kernel_data[kernel],
                                        mapstore=maps[kernel],
                                        silenced_warnings=['write_race(rop_net_{})'
                                        .format(kernel)]))
        return infos


def get_rop(loopy_opts, namestore, allint={'net': False}, test_size=None):
    """Generates instructions, kernel arguements, and data for the Rate of Progress
    kernels

    Parameters
    ----------
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    allint : dict
        If allint['net'] is True, powers of concentrations
        will be evaluated using multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    maps = {}

    # create ROP kernels

    def __rop_create(direction):
        spec_loop = 'ispec'
        spec_ind = 'spec_ind'

        # indicies
        kernel_data = []
        # add problem size
        kernel_data.extend(arc.initial_condition_dimension_vars(
            loopy_opts, test_size))
        if direction == 'fwd':
            inds = namestore.num_reacs
            mapinds = namestore.num_reacs
        else:
            inds = namestore.num_rev_reacs
            mapinds = namestore.rev_map

        maps[direction] = arc.MapStore(loopy_opts, inds, test_size)
        themap = maps[direction]

        # add transforms for the offsets
        themap.check_and_add_transform(
            namestore.rxn_to_spec_offsets, mapinds)

        # we need species lists, nu lists, etc.

        # offsets are on main loop, no map
        num_spec_offsets_lp, num_spec_offsets_str = themap.apply_maps(
            namestore.rxn_to_spec_offsets, var_name)

        # next offset to calculate num species
        _, num_spec_offsets_next_str = themap.apply_maps(
            namestore.rxn_to_spec_offsets, var_name, affine=1)

        # nu lists are on main loop, no map
        nus = 'rxn_to_spec_reac_nu' if direction == 'fwd'\
            else 'rxn_to_spec_prod_nu'
        nu = getattr(namestore, nus)
        nu_lp, nu_str = themap.apply_maps(nu, spec_loop, affine=spec_loop)

        # species lists are in ispec loop, use that iname
        spec_lp, spec_str = themap.apply_maps(
            namestore.rxn_to_spec, spec_loop)

        # rate constants on main loop, no map
        rateconst = namestore.kf if direction == 'fwd' else namestore.kr
        rateconst_arr, rateconst_str = themap.apply_maps(
            rateconst, *default_inds)

        # concentrations in ispec loop, also use offset for phi
        concs_lp, concs_str = themap.apply_maps(
            namestore.conc_arr, global_ind, spec_ind)

        # and finally the ROP values in the mainloop, no map
        rop = getattr(namestore, 'rop_' + direction)
        rop_arr, rop_str = themap.apply_maps(rop, *default_inds)

        # update kernel data
        kernel_data.extend(
            [num_spec_offsets_lp, concs_lp, nu_lp, spec_lp,
             rateconst_arr, rop_arr])

        # instructions
        rop_instructions = Template(
            """
    <>rop_temp = ${rateconst_str} {id=rop_init}
    <>spec_offset = ${num_spec_offsets_str}
    <>spec_offset_next = ${num_spec_offsets_next_str}
    for ${spec_loop}
        <>${spec_ind} = ${spec_str} {id=spec_ind}
        ${rop_temp_eval}
    end
    ${rop_str} = rop_temp {dep=rop_fin*}
    """).safe_substitute(rateconst_str=rateconst_str,
                         rop_str=rop_str,
                         spec_loop=spec_loop,
                         num_spec_offsets_str=num_spec_offsets_str,
                         num_spec_offsets_next_str=num_spec_offsets_next_str,
                         spec_str=spec_str,
                         spec_ind=spec_ind)

        power_func = ic.power_function(loopy_opts,
                                       is_integer_power=allint['net'],
                                       is_vector=loopy_opts.is_simd)
        conc_power = power_func(concs_str, nu_str)
        # if all integers, it's much faster to use multiplication
        roptemp_eval = Template(
            """
    rop_temp = rop_temp * ${conc_power} {id=rop_fin, dep=rop_init}
    """).safe_substitute(conc_power=conc_power)

        rop_instructions = utils.subs_at_indent(rop_instructions,
                                                rop_temp_eval=roptemp_eval)

        # and finally extra inames
        extra_inames = [
            (spec_loop, 'spec_offset <= {} < spec_offset_next'.format(spec_loop))]

        # and return the rateconst
        return k_gen.knl_info(name='rop_eval_{}'.format(direction),
                              instructions=rop_instructions,
                              var_name=var_name,
                              kernel_data=kernel_data,
                              extra_inames=extra_inames,
                              mapstore=maps[direction],
                              manglers=[power_func])

    infos = [__rop_create('fwd')]
    if namestore.rop_rev is not None:
        infos.append(__rop_create('rev'))
    return infos


def get_rxn_pres_mod(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for pressure
    modification term of the forward reaction rates.

    Parameters
    ----------
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

    # check for empty
    if namestore.thd_only_map is None:
        info_list = [None]
    else:
        # start developing the ci kernel
        # rate info and reac ind

        kernel_data = []
        # add problem size
        kernel_data.extend(arc.initial_condition_dimension_vars(
            loopy_opts, test_size))

        # create the third body conc pres-mod kernel

        thd_map = arc.MapStore(loopy_opts, namestore.thd_only_map, test_size)

        # get the third body concs
        thd_lp, thd_str = thd_map.apply_maps(namestore.thd_conc,
                                             *default_inds)

        # and the pressure mod term
        pres_mod_lp, pres_mod_str = thd_map.apply_maps(namestore.pres_mod,
                                                       *default_inds)

        thd_instructions = Template("""
        ${pres_mod} = ${thd_conc} {id=set_pres_mod}
        """).safe_substitute(pres_mod=pres_mod_str,
                             thd_conc=thd_str)

        # and the args
        kernel_data.extend([thd_lp, pres_mod_lp])

        # add to the info list
        info_list = [
            k_gen.knl_info(name='ci_thd',
                           instructions=thd_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=thd_map,
                           silenced_warnings=['write_race(set_pres_mod)'])]

    # check for empty
    if namestore.num_fall is None:
        info_list.append(None)
    else:
        # and now the falloff kernel
        kernel_data = []
        # add problem size
        kernel_data.extend(arc.initial_condition_dimension_vars(
            loopy_opts, test_size))

        fall_map = arc.MapStore(loopy_opts, namestore.num_fall, test_size)

        # the pressure mod term uses fall_to_thd_map/mask
        fall_map.check_and_add_transform(namestore.pres_mod,
                                         namestore.fall_to_thd_map)

        # the falloff vs chemically activated indicator
        fall_type_lp, fall_type_str = \
            fall_map.apply_maps(namestore.fall_type, var_name)

        # the blending term
        Fi_lp, Fi_str = \
            fall_map.apply_maps(namestore.Fi, *default_inds)

        # the Pr array
        Pr_lp, Pr_str = \
            fall_map.apply_maps(namestore.Pr, *default_inds)

        pres_mod_lp, pres_mod_str = \
            fall_map.apply_maps(namestore.pres_mod, *default_inds)

        # update the args
        kernel_data.extend([Fi_lp, Pr_lp, fall_type_lp, pres_mod_lp])

        fall_instructions = Template("""
        <>ci_temp = ${Fi_str} / (1 + ${Pr_str}) {id=ci_decl}
        if not ${fall_type}
            ci_temp = ci_temp * ${Pr_str} {id=ci_update, dep=ci_decl}
        end
        ${pres_mod} = ci_temp {dep=ci_update}
    """).safe_substitute(Fi_str=Fi_str,
                         Pr_str=Pr_str,
                         pres_mod=pres_mod_str,
                         fall_type=fall_type_str
                         )

        # add to the info list
        info_list.append(
            k_gen.knl_info(name='ci_fall',
                           instructions=fall_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=fall_map))
    return info_list


def get_rev_rates(loopy_opts, namestore, allint, test_size=None):
    """Generates instructions, kernel arguements, and data for reverse reaction
    rates

    Parameters
    ----------
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

    # check for empty
    if namestore.num_rev_reacs is None:
        return None

    # start developing the Kc kernel
    kernel_data = []
    spec_ind = 'spec_ind'
    spec_loop = 'ispec'

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # add the reverse map
    rev_map = arc.MapStore(loopy_opts, namestore.num_rev_reacs, test_size)

    # map from reverse reaction index to forward reaction index
    rev_map.check_and_add_transform(
        namestore.nu_sum, namestore.rev_map)
    rev_map.check_and_add_transform(
        namestore.rxn_to_spec, namestore.rev_map)
    rev_map.check_and_add_transform(
        namestore.rxn_to_spec_offsets, namestore.rev_map)
    rev_map.check_and_add_transform(
        namestore.kf, namestore.rev_map)

    # create nu_sum on main loop
    nu_sum_lp, nu_sum_str = rev_map.apply_maps(namestore.nu_sum,
                                               var_name)

    # all species in reaction on spec loop
    spec_lp, spec_str = rev_map.apply_maps(namestore.rxn_to_spec,
                                           spec_loop)

    # species offsets on main loop
    num_spec_offsets_lp, num_spec_offsets_str = rev_map.apply_maps(
        namestore.rxn_to_spec_offsets, var_name)

    # species offset on main loop with offset of 1
    _, num_spec_offsets_next_str = rev_map.apply_maps(
        namestore.rxn_to_spec_offsets, var_name, affine=1)

    # B array on spec_ind
    B_lp, B_str = rev_map.apply_maps(namestore.b, global_ind, spec_ind)

    # net nu on species loop
    nu_lp, prod_nu_str = rev_map.apply_maps(namestore.rxn_to_spec_prod_nu,
                                            spec_loop, affine=spec_loop)
    _, reac_nu_str = rev_map.apply_maps(namestore.rxn_to_spec_reac_nu,
                                        spec_loop, affine=spec_loop)

    # the Kc array on the main loop, no map as this is only reversible
    Kc_lp, Kc_str = rev_map.apply_maps(namestore.Kc, *default_inds)

    # create the kf array / str
    kf_arr, kf_str = rev_map.apply_maps(
        namestore.kf, *default_inds)

    # create the kr array / str (no map as we're looping over rev inds)
    kr_arr, kr_str = rev_map.apply_maps(
        namestore.kr, *default_inds)

    # update kernel data
    kernel_data.extend([nu_sum_lp, spec_lp, num_spec_offsets_lp,
                        B_lp, Kc_lp, nu_lp, kf_arr, kr_arr])

    # get the right power function
    power_func = ic.power_function(loopy_opts,
                                   is_integer_power=allint['net'],
                                   # both values here are scalar
                                   is_vector=False)
    pressure_power = power_func('P_val', 'P_sum_end')

    # create the pressure product loop
    pressure_prod = Template("""
    <> P_sum_end = abs(${nu_sum}) {id=P_bound}
    if ${nu_sum} > 0
        <> P_val = P_a / R_u {id=P_val_decl}
    else
        P_val = R_u / P_a {id=P_val_decl1}
    end
    <> P_sum = ${pressure_power} {id=P_accum, dep=P_val_decl*}
    """).substitute(nu_sum=nu_sum_str,
                    pressure_power=pressure_power)

    expg = ic.GuardedExp(loopy_opts)
    b_exp = expg('B_sum')
    # and the b sum loop
    Bsum_inst = Template("""
    <>offset = ${spec_offset} {id=offset}
    <>spec_end = ${spec_offset_next} {id=B_bound}
    <>B_sum = 0 {id=B_init}
    for ${spec_loop}
        <>${spec_ind} = ${spec_mapper} {dep=offset:B_bound}
        <>net_nu = ${prod_nu_str} - ${reac_nu_str}
        if net_nu != 0
            B_sum = B_sum + net_nu * ${B_val} {id=B_accum, dep=B_init}
        end
    end
    B_sum = ${b_exp} {id=B_final, dep=B_accum}
    """).substitute(spec_offset=num_spec_offsets_str,
                    spec_offset_next=num_spec_offsets_next_str,
                    spec_loop=spec_loop,
                    spec_ind=spec_ind,
                    spec_mapper=spec_str,
                    nu_val=nu_sum_str,
                    prod_nu_str=prod_nu_str,
                    reac_nu_str=reac_nu_str,
                    B_val=B_str,
                    b_exp=b_exp
                    )

    Rate_assign = Template("""
    <>Kc_temp = P_sum * B_sum {dep=P_accum*:B_final}
    ${Kc_str} = Kc_temp
    ${kr_str} = ${kf_str} / Kc_temp
    """).safe_substitute(**locals())

    instructions = '\n'.join([Bsum_inst, pressure_prod, Rate_assign])

    # create the extra inames
    extra_inames = [(spec_loop, 'offset <= {} < spec_end'.format(spec_loop))]

    # and return the rateinfo
    return k_gen.knl_info(name='rateconst_Kc',
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=rev_map,
                          extra_inames=extra_inames,
                          parameters={
                              'P_a': np.float64(chem.PA),
                              'R_u': np.float64(chem.RU)},
                          manglers=[power_func, expg])


def get_thd_body_concs(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for third body
    concentrations

    Parameters
    ----------
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

    # check for empty
    if namestore.thd_inds is None:
        return None

    spec_ind = 'spec_ind'
    spec_loop = 'ispec'
    spec_offset = 'offset'

    # create mapstore over number of third reactions
    mapstore = arc.MapStore(loopy_opts, namestore.thd_inds, test_size)

    # create args

    # get concentrations
    # in species loop
    concs_lp, concs_str = mapstore.apply_maps(
        namestore.conc_arr, global_ind, spec_ind)

    # get third body concentrations (by defn same as third reactions)
    thd_lp, thd_str = mapstore.apply_maps(namestore.thd_conc, *default_inds)

    # get T and P arrays
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    P_arr, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)

    # and the third body descriptions

    # efficiency list
    thd_eff_lp, thd_eff_str = mapstore.apply_maps(
        namestore.thd_eff, spec_loop)
    # non-unity species in thd-body conc
    thd_spec_lp, thd_spec_str = mapstore.apply_maps(
        namestore.thd_spec, spec_loop)
    # offset to spec / efficiency arrays
    thd_offset_lp, thd_offset_str = mapstore.apply_maps(
        namestore.thd_offset, var_name)
    # third body type
    thd_type_lp, thd_type_str = mapstore.apply_maps(
        namestore.thd_type, var_name)
    # get next offset to determine num of thd body eff's in rxnq
    _, thd_offset_next_str = mapstore.apply_maps(
        namestore.thd_offset, var_name, affine=1)

    # kernel data
    kernel_data = []
    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # add arrays
    kernel_data.extend([P_arr, T_arr, concs_lp, thd_lp, thd_type_lp,
                        thd_eff_lp, thd_spec_lp, thd_offset_lp])

    # maps
    # extra loops
    extra_inames = [(spec_loop, '{} <= {} < spec_end'.format(
        spec_offset, spec_loop))]

    # generate instructions and sub in instructions
    instructions = Template("""
<int32> not_spec = ${thd_type} != ${species}
<> ${offset_name} = ${offset} {id=offset}
<> spec_end = ${offset_next} {id=num0}
<> thd_temp = ${P_str} * not_spec / (R * ${T_str}) {id=thd1, dep=num0}
for ${spec_loop}
    <> ${spec_ind} = ${thd_spec} {id=ind1}
    thd_temp = thd_temp + (${thd_eff} - not_spec) * ${conc_thd_spec} {id=thdcalc,\
        dep=ind1:thd1}
end
${thd_str} = thd_temp {dep=thd*}
""").safe_substitute(
        offset_name=spec_offset,
        offset=thd_offset_str,
        offset_next=thd_offset_next_str,
        thd_eff=thd_eff_str,
        conc_thd_spec=concs_str,
        thd_str=thd_str,
        spec_loop=spec_loop,
        spec_ind=spec_ind,
        thd_spec=thd_spec_str,
        P_str=P_str,
        T_str=T_str,
        thd_type=thd_type_str,
        species=int(thd_body_type.species)
    )

    # create info
    return k_gen.knl_info('eval_thd_body_concs',
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          extra_inames=extra_inames,
                          parameters={'R': chem.RU},
                          mapstore=mapstore)


def get_cheb_arrhenius_rates(loopy_opts, namestore, maxP, maxT,
                             test_size=None):
    """Generates instructions, kernel arguements, and data for cheb rate constants

    Parameters
    ----------
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    maxP : int
        The maximum degree of pressure polynomials for chebyshev reactions in
        this mechanism
    maxT : int
        The maximum degree of temperature polynomials for chebyshev reactions
        in this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    # check for empty map
    if namestore.cheb_map is None:
        return None

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.num_cheb, test_size)

    # max degrees in mechanism
    poly_max = int(np.maximum(maxP, maxT))

    # extra inames
    pres_poly_ind = 'k'
    temp_poly_ind = 'm'
    poly_compute_ind = 'p'
    lim_ind = 'dummy'
    extra_inames = [(pres_poly_ind, '0 <= {} < {}'.format(
        pres_poly_ind, maxP)),
        (temp_poly_ind, '0 <= {} < {}'.format(
                        temp_poly_ind, maxT)),
        (poly_compute_ind, '2 <= {} < {}'.format(poly_compute_ind, poly_max))]

    # create arrays

    # kf is based on the map
    mapstore.check_and_add_transform(namestore.kf, namestore.cheb_map)

    num_P_lp, num_P_str = mapstore.apply_maps(namestore.cheb_numP, var_name)
    num_T_lp, num_T_str = mapstore.apply_maps(namestore.cheb_numT, var_name)
    params_lp, params_str = mapstore.apply_maps(namestore.cheb_params,
                                                var_name,
                                                temp_poly_ind,
                                                pres_poly_ind)
    plim_lp, _ = mapstore.apply_maps(namestore.cheb_Plim, var_name, lim_ind)
    tlim_lp, _ = mapstore.apply_maps(namestore.cheb_Tlim, var_name, lim_ind)

    # workspace vars are based only on their polynomial indicies
    pres_poly_lp, ppoly_k_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                    pres_poly_ind)
    temp_poly_lp, tpoly_m_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                    temp_poly_ind)

    # create temperature and pressure arrays
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    P_arr, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)

    # get the forward rate constants
    kf_arr, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)

    # update kernel data
    kernel_data = []
    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    kernel_data.extend([params_lp, num_P_lp, num_T_lp, plim_lp, tlim_lp,
                        pres_poly_lp, temp_poly_lp, T_arr, P_arr, kf_arr])

    # preinstructions
    logP = 'logP'
    Tinv = 'Tinv'
    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)

    preinstructs = [precompute(logP, P_str, 'LOG'),
                    precompute(Tinv, T_str, 'INV', guard=ic.TemperatureGuard(
                        loopy_opts))]

    # various strings for preindexed limits, params, etc
    _, Pmin_str = mapstore.apply_maps(namestore.cheb_Plim, var_name, '0')
    _, Pmax_str = mapstore.apply_maps(namestore.cheb_Plim, var_name, '1')
    _, Tmin_str = mapstore.apply_maps(namestore.cheb_Tlim, var_name, '0')
    _, Tmax_str = mapstore.apply_maps(namestore.cheb_Tlim, var_name, '1')

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

    eguard = ic.GuardedExp(loopy_opts, exptype=utils.exp_10_fun[loopy_opts.lang])
    exp10fun = eguard('kf_temp')

    instructions = Template("""
<>Pmin = ${Pmin_str}
<>Tmin = ${Tmin_str}
<>Pmax = ${Pmax_str}
<>Tmax = ${Tmax_str}
<>Tred = (-Tmax - Tmin + 2 * ${Tinv}) / (Tmax - Tmin)
<>Pred = (-Pmax - Pmin + 2 * ${logP}) / (Pmax - Pmin)
<>numP = ${num_P_str} {id=plim}
<>numT = ${num_T_str} {id=tlim}
${ppoly0_str} = 1 {id=ppoly_init}
${ppoly1_str} = Pred {id=ppoly_init1}
${tpoly0_str} = 1 {id=tpoly_init}
${tpoly1_str} = Tred {id=tpoly_init2}
#<> poly_end = max(numP, numT)
# compute polynomial terms
for p
    if p < numP
        ${ppolyp_str} = 2 * Pred * ${ppolypm1_str} - ${ppolypm2_str} \
            {id=ppoly, dep=plim:ppoly_init*}
    end
    if p < numT
        ${tpolyp_str} = 2 * Tred * ${tpolypm1_str} - ${tpolypm2_str} \
            {id=tpoly, dep=tlim:tpoly_init*}
    end
end
<> kf_temp = 0 {id=kf_init}
for m
    <>temp = 0 {id=temp_init}
    for k
        if k < numP
            temp = temp + ${ppoly_k_str} * ${params_str} {id=temp,
                dep=ppoly:tpoly:kf_init:temp_init}
        end
    end
    if m < numT
        kf_temp = kf_temp + ${tpoly_m_str} * temp {id=kf, dep=temp:kf_init}
    end
end

${kf_str} = ${exp10fun} {id=set, dep=kf}
""")

    instructions = instructions.safe_substitute(**locals())
    vec_spec = ic.write_race_silencer(['set'])

    return k_gen.knl_info('rateconst_cheb',
                          instructions=instructions,
                          pre_instructions=preinstructs,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          extra_inames=extra_inames,
                          vectorization_specializer=vec_spec,
                          manglers=[eguard])


def get_plog_arrhenius_rates(loopy_opts, namestore, maxP, test_size=None):
    """Generates instructions, kernel arguements, and data for p-log rate constants

    Parameters
    ----------
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    maxP : int
        The maximum number of pressure interpolations of any reaction in
        the mechanism.
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    # check for empty plog (if so, return empty)
    if namestore.plog_map is None:
        return None

    # parameter indicies
    arrhen_ind = 'm'
    param_ind = 'k'
    lo_ind = 'lo_ind'
    hi_ind = 'hi_ind'

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.plog_map, test_size)

    # number of parameter sets per reaction
    mapstore.check_and_add_transform(namestore.plog_num_param,
                                     namestore.num_plog)

    # plog parameters
    mapstore.check_and_add_transform(namestore.plog_params,
                                     namestore.num_plog)
    # fwd rate constants
    mapstore.check_and_add_transform(namestore.kf, namestore.plog_map)

    plog_num_param_lp, plog_num_param_str = mapstore.apply_maps(
        namestore.plog_num_param, var_name)
    plog_params_lp, plog_params_str = mapstore.apply_maps(
        namestore.plog_params, arrhen_ind, var_name, param_ind)

    # temperature / pressure arrays
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    P_arr, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)

    # create temporary storage variables
    low_lp = lp.TemporaryVariable(
        'low', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    hi_lp = lp.TemporaryVariable(
        'hi', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)

    # forward rxn rate constants
    kf_arr, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)

    # data
    kernel_data = []
    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # update kernel data
    kernel_data.extend([plog_params_lp, plog_num_param_lp, T_arr,
                        P_arr, low_lp, hi_lp, kf_arr])
    # extra loops
    extra_inames = [(param_ind, '0 <= {} < {}'.format(param_ind, maxP - 1)),
                    (arrhen_ind, '0 <= {} < 4'.format(arrhen_ind))]

    # specific indexing strings
    _, pressure_lo = mapstore.apply_maps(
        namestore.plog_params, 0, var_name, 0)
    _, pressure_hi = mapstore.apply_maps(
        namestore.plog_params, 0, var_name, 'numP')
    _, pressure_mid_lo = mapstore.apply_maps(
        namestore.plog_params, 0, var_name, param_ind)
    _, pressure_mid_hi = mapstore.apply_maps(
        namestore.plog_params, 0, var_name, param_ind, affine={param_ind: 1})
    _, pressure_general_lo = mapstore.apply_maps(
        namestore.plog_params, arrhen_ind, var_name, lo_ind)
    _, pressure_general_hi = mapstore.apply_maps(
        namestore.plog_params, arrhen_ind, var_name, hi_ind)

    # precompute names
    logP = 'logP'
    logT = 'logT'
    Tinv = 'Tinv'

    # exponentials
    expg = ic.GuardedExp(loopy_opts)
    exp_kf = expg('kf_temp')

    # instructions
    instructions = Template(
        """
        <>lower = ${logP} <= ${pressure_lo} # check below range
        if lower
            <>lo_ind = 0 {id=ind00, nosync=ind10}
            <>hi_ind = 0 {id=ind01, nosync=ind11}
        end
        <>numP = ${plog_num_param_str} - 1
        <>upper = ${logP} > ${pressure_hi} # check above range
        if upper
            lo_ind = numP {id=ind10, nosync=ind00}
            hi_ind = numP {id=ind11, nosync=ind01}
        end
        <>oor = lower or upper
        for k
            # check that
            # 1. inside this reactions parameter's still
            # 2. inside pressure range
            <> midcheck = (k < numP) and (${logP} > ${pressure_mid_lo}) \
                and (${logP} <= ${pressure_mid_hi})
            if midcheck
                lo_ind = k {id=ind20, dep=ind10:ind00}
                hi_ind = k + 1 {id=ind21, dep=ind11:ind01}
            end
        end
        # load pressure and reaction parameters into temp arrays
        for m
            low[m] = ${pressure_general_lo} {id=lo, dep=ind*}
            hi[m] = ${pressure_general_hi} {id=hi, dep=ind*}
        end
        # eval logkf's
        <>logk1 = low[1] + ${logT} * low[2] - low[3] * ${Tinv}  {id=a1, dep=lo}
        <>logk2 = hi[1] + ${logT} * hi[2] - hi[3] * ${Tinv} {id=a2, dep=hi}
        <>kf_temp = logk1 {id=a_oor}
        if not oor
            # if not out of bounds, compute interpolant
            kf_temp = (-logk1 + logk2) * (${logP} - low[0]) / (hi[0] - low[0]) + \
                kf_temp {id=a_found, dep=a1:a2:a_oor}
        end
        ${kf_str} = ${exp_kf} {id=kf, dep=a_found}
""").safe_substitute(**locals())

    vec_spec = ic.write_race_silencer(['kf'])

    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)
    guardT = ic.TemperatureGuard(loopy_opts)
    preinsns = [precompute(Tinv, T_str, 'INV', guard=guardT),
                precompute(logT, T_str, 'LOG', guard=guardT),
                precompute(logP, P_str, 'LOG')]

    # and return
    return [k_gen.knl_info(name='rateconst_plog',
                           instructions=instructions,
                           pre_instructions=preinsns,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           extra_inames=extra_inames,
                           vectorization_specializer=vec_spec,
                           # silence warning about scatter load of plog parameters
                           # due to varying pressure
                           silenced_warnings=['vectorize_failed'],
                           manglers=[expg, precompute])]


def get_reduced_pressure_kernel(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the reduced
    pressure evaluation kernel

    Parameters
    ----------
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

    # check for empty
    if namestore.fall_map is None:
        return None

    # create the mapper
    mapstore = arc.MapStore(loopy_opts, namestore.fall_map, test_size)

    # create the various necessary arrays

    kernel_data = []
    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))
    # add maps / masks

    # kf is over all reactions
    mapstore.check_and_add_transform(namestore.kf, namestore.fall_map)
    # kf_fall / Pr / fall type are over falloff reactions
    mapstore.check_and_add_transform(namestore.kf_fall, namestore.num_fall)
    mapstore.check_and_add_transform(namestore.Pr, namestore.num_fall)
    mapstore.check_and_add_transform(namestore.fall_type, namestore.num_fall)
    # third body concentrations are over thd_map
    mapstore.check_and_add_transform(namestore.fall_to_thd_map,
                                     namestore.num_fall)
    mapstore.check_and_add_transform(namestore.thd_conc,
                                     namestore.fall_to_thd_map)

    # simple arrhenius rates
    kf_arr, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)
    # simple arrhenius rates using falloff (alternate) parameters
    kf_fall_arr, kf_fall_str = mapstore.apply_maps(namestore.kf_fall,
                                                   *default_inds)

    # temperatures
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)

    # create a Pr array
    Pr_arr, Pr_str = mapstore.apply_maps(namestore.Pr, *default_inds)

    # third-body concentration
    thd_conc_lp, thd_conc_str = mapstore.apply_maps(namestore.thd_conc,
                                                    *default_inds)
    # and finally the falloff types
    fall_type_lp, fall_type_str = mapstore.apply_maps(namestore.fall_type,
                                                      var_name)

    # append all arrays to the kernel data
    kernel_data.extend([T_arr, thd_conc_lp, kf_arr, kf_fall_arr, Pr_arr,
                        fall_type_lp])

    # create instruction set
    pr_instructions = Template("""
if ${fall_type_str}
    # chemically activated
    <>k0 = ${kf_str} {id=k0_c, nosync=k0_f}
    <>kinf = ${kf_fall_str} {id=kinf_c, nosync=kinf_f}
else
    # fall-off
    kinf = ${kf_str} {id=kinf_f, nosync=kinf_c}
    k0 = ${kf_fall_str} {id=k0_f, nosync=k0_c}
end
# prevent reduced pressure from ever being truly zero
${Pr_str} = ${thd_conc_str} * k0 / kinf {id=set, dep=k*}
""")

    # sub in strings
    pr_instructions = pr_instructions.safe_substitute(**locals())
    vec_spec = ic.write_race_silencer(['set'])

    # and finally return the resulting info
    return [k_gen.knl_info('red_pres',
                           instructions=pr_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           vectorization_specializer=vec_spec)]


def get_troe_kernel(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the Troe
    falloff evaluation kernel

    Parameters
    ----------
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

    # check for empty
    if namestore.troe_map is None:
        return None

    # rate info and reac ind
    kernel_data = []

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.num_troe, test_size)

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # add maps / masks
    mapstore.check_and_add_transform(namestore.Fi, namestore.troe_map)
    mapstore.check_and_add_transform(namestore.Pr, namestore.troe_map)

    # create the Pr loopy array / string
    Pr_lp, Pr_str = mapstore.apply_maps(namestore.Pr, *default_inds)

    # create Fi loopy array / string
    Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)

    # create the Fcent loopy array / str
    Fcent_lp, Fcent_str = mapstore.apply_maps(namestore.Fcent, *default_inds)

    # create the Atroe loopy array / str
    Atroe_lp, Atroe_str = mapstore.apply_maps(namestore.Atroe, *default_inds)

    # create the Btroe loopy array / str
    Btroe_lp, Btroe_str = mapstore.apply_maps(namestore.Btroe, *default_inds)

    # create the temperature array
    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)

    # update the kernel_data
    kernel_data.extend([Pr_lp, T_lp, Fi_lp, Fcent_lp, Atroe_lp, Btroe_lp])

    # get troe params and create arrays
    troe_a_lp, troe_a_str = mapstore.apply_maps(namestore.troe_a, var_name)
    troe_T3_lp, troe_T3_str = mapstore.apply_maps(namestore.troe_T3, var_name)
    troe_T1_lp, troe_T1_str = mapstore.apply_maps(namestore.troe_T1, var_name)
    troe_T2_lp, troe_T2_str = mapstore.apply_maps(namestore.troe_T2, var_name)
    # update the kernel_data
    kernel_data.extend([troe_a_lp, troe_T3_lp, troe_T1_lp, troe_T2_lp])

    # get generic power function
    power_func = ic.power_function(loopy_opts,
                                   is_integer_power=False,
                                   is_vector=loopy_opts.is_simd,
                                   guard_nonzero=True)
    pow_fcent = power_func(
        'Fcent_temp', '(1 / (((Atroe_temp * Atroe_temp) / '
                      '(Btroe_temp * Btroe_temp) + 1)))')

    # exponentials
    expg = ic.GuardedExp(loopy_opts)
    T1_exp = expg('-T * {troe_T1_str}'.format(troe_T1_str=troe_T1_str))
    T3_exp = expg('-T * {troe_T3_str}'.format(troe_T3_str=troe_T3_str))
    T2_exp = expg('-{troe_T2_str} / T'.format(troe_T2_str=troe_T2_str))

    # logs
    logg = ic.GuardedLog(loopy_opts, logtype=utils.log_10_fun[loopy_opts.lang])
    log_fcent = logg('Fcent_temp')
    log_pr = logg(Pr_str)

    # make the instructions
    troe_instructions = Template("""
    <>Fcent_temp = ${troe_a_str} * ${T1_exp} \
        + (1 - ${troe_a_str}) * ${T3_exp} {id=Fcent_decl}
    if ${troe_T2_str} != 0
        Fcent_temp = Fcent_temp + ${T2_exp} \
            {id=Fcent_decl2, dep=Fcent_decl}
    end
    ${Fcent_str} = Fcent_temp {id=Fcent_decl3, dep=Fcent_decl2}
    <>logFcent = ${log_fcent} {dep=Fcent_decl3}
    <>logPr = ${log_pr}
    <>Atroe_temp = -0.67 * logFcent + logPr - 0.4 {dep=Fcent_decl*}
    <>Btroe_temp = -1.1762 * logFcent - 0.14 * logPr + 0.806 {dep=Fcent_decl*}
    ${Atroe_str} = Atroe_temp
    ${Btroe_str} = Btroe_temp
    ${Fi_str} = ${pow_fcent} {id=Fi, dep=Fcent_decl*}
    """).safe_substitute(**locals())

    vec_spec = ic.write_race_silencer(['Fi'])
    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)

    return [k_gen.knl_info('fall_troe',
                           pre_instructions=[
                            precompute('T', T_str, 'VAL',
                                       guard=ic.TemperatureGuard(loopy_opts))],
                           instructions=troe_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           vectorization_specializer=vec_spec,
                           manglers=[power_func, expg, logg])]


def get_sri_kernel(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the SRI
    falloff evaluation kernel

    Parameters
    ----------
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

    # check for empty
    if namestore.sri_map is None:
        return None

    kernel_data = []

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.num_sri, test_size)

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # maps and transforms
    mapstore.check_and_add_transform(namestore.Fi, namestore.sri_map)
    mapstore.check_and_add_transform(namestore.Pr, namestore.sri_map)

    # Create the temperature array
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)

    # create Fi array / mapping
    Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)
    # and Pri array / mapping
    Pr_lp, Pr_str = mapstore.apply_maps(namestore.Pr, *default_inds)

    kernel_data.extend([T_arr, Fi_lp, Pr_lp])

    # create SRI parameters
    X_sri_lp, X_sri_str = mapstore.apply_maps(namestore.X_sri, *default_inds)
    sri_a_lp, sri_a_str = mapstore.apply_maps(namestore.sri_a, var_name)
    sri_b_lp, sri_b_str = mapstore.apply_maps(namestore.sri_b, var_name)
    sri_c_lp, sri_c_str = mapstore.apply_maps(namestore.sri_c, var_name)
    sri_d_lp, sri_d_str = mapstore.apply_maps(namestore.sri_d, var_name)
    sri_e_lp, sri_e_str = mapstore.apply_maps(namestore.sri_e, var_name)

    kernel_data.extend(
        [X_sri_lp, sri_a_lp, sri_b_lp, sri_c_lp, sri_d_lp, sri_e_lp])

    # precomputes
    Tinv = 'Tinv'
    Tval = 'Tval'

    # get generic power function
    noint_power = ic.power_function(loopy_opts, is_integer_power=False,
                                    is_vector=loopy_opts.is_simd)
    possible_int_power = ic.power_function(loopy_opts,
                                           is_integer_power=isinstance(
                                            sri_e_lp.dtype, np.integer),
                                           is_vector=loopy_opts.is_simd)
    # guarded exponential
    expg = ic.GuardedExp(loopy_opts)
    sri_b_exp = expg(Template('-${sri_b_str} * ${Tinv}').safe_substitute(**locals()))
    sri_c_exp = expg(Template('-${Tval} / ${sri_c_str}').safe_substitute(**locals()))

    # perform power evaluations
    sri_power_base = Template('(${sri_a_str} * ${sri_b_exp} + '
                              '${sri_c_exp})').safe_substitute(**locals())
    sri_power = noint_power(sri_power_base, 'X_temp')

    sri_power_optional = possible_int_power(Tval, sri_e_str)

    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)
    Tguard = ic.TemperatureGuard(loopy_opts)
    pre_instructions = [precompute(Tval, T_str, 'VAL', guard=Tguard),
                        precompute(Tinv, T_str, 'INV', guard=Tguard)]

    # get logarithm
    logg = ic.GuardedLog(loopy_opts, logtype=utils.log_10_fun[loopy_opts.lang])
    log_pr = logg(Pr_str)

    # create instruction set
    sri_instructions = Template("""
    <>logPr = ${log_pr}
    <>X_temp = 1 / (logPr * logPr + 1) {id=X_decl}
    <>Fi_temp = ${sri_power} {id=Fi_decl, dep=X_decl}
    if ${sri_d_str} != 1.0
        Fi_temp = Fi_temp * ${sri_d_str} {id=Fi_decl1, dep=Fi_decl}
    end
    if ${sri_e_str} != 0.0
        Fi_temp = Fi_temp * ${sri_power_optional} {id=Fi_decl2, dep=Fi_decl1}
    end
    ${Fi_str} = Fi_temp {dep=Fi_decl*}
    ${X_sri_str} = X_temp
    """).safe_substitute(**locals())

    return [k_gen.knl_info('fall_sri',
                           instructions=sri_instructions,
                           pre_instructions=pre_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           manglers=[noint_power, possible_int_power, precompute,
                                     logg, expg])]


def get_lind_kernel(loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the Lindeman
    falloff evaluation kernel

    Parameters
    ----------
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

    # check for empty
    if namestore.lind_map is None:
        return None

    # set of equations is irrelevant for non-derivatives

    kernel_data = []

    # add problem size
    kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # create Fi array / str

    mapstore = arc.MapStore(loopy_opts, namestore.lind_map, test_size)

    Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)
    kernel_data.append(Fi_lp)

    # create instruction set
    lind_instructions = Template("""
    ${Fi_str} = 1 {id=Fi_init}
    """).safe_substitute(Fi_str=Fi_str)

    return [k_gen.knl_info('fall_lind',
                           instructions=lind_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           silenced_warnings=['write_race(Fi_init)'])]


def get_simple_arrhenius_rates(loopy_opts, namestore, test_size=None,
                               falloff=False):
    """Generates instructions, kernel arguements, and data for specialized forms
    of simple (non-pressure dependent) rate constants

    Parameters
    ----------
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    falloff : bool
        If true, generate rate kernel for the falloff rates, i.e. either
        k0 or kinf depending on whether the reaction is falloff or chemically
        activated

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    # find options, sizes, etc.
    if falloff:
        tag = 'fall'

        # check for empty falloff (if so, return empty)
        if namestore.fall_map is None:
            return None

        mapstore = arc.MapStore(loopy_opts, namestore.fall_map, test_size)
        # define the rtype iteration domain

        def get_rdomain(rtype):
            if rtype < 0:
                return namestore.num_fall, namestore.num_fall,\
                    namestore.num_fall
            else:
                return getattr(namestore, 'fall_rtype_{}_inds'.format(rtype)),\
                    getattr(namestore, 'fall_rtype_{}_inds'.format(rtype)),\
                    getattr(namestore, 'fall_rtype_{}_num'.format(rtype))
        rdomain = get_rdomain
    else:
        tag = 'simple'
        mapstore = arc.MapStore(loopy_opts, namestore.simple_map, test_size)
        # define the rtype iteration domain

        def get_rdomain(rtype):
            if rtype < 0:
                return namestore.num_simple, namestore.simple_map,\
                    namestore.num_simple
            else:
                return getattr(namestore,
                               'simple_rtype_{}_inds'.format(rtype)), \
                    getattr(namestore,
                            'simple_rtype_{}_map'.format(rtype)), \
                    getattr(namestore,
                            'simple_rtype_{}_num'.format(rtype))
        rdomain = get_rdomain

    # first assign the reac types, parameters
    full = loopy_opts.rate_spec == RateSpecialization.full
    hybrid = loopy_opts.rate_spec == RateSpecialization.hybrid
    fixed = loopy_opts.rate_spec == RateSpecialization.fixed
    separated_kernels = loopy_opts.rate_spec_kernels
    logger = logging.getLogger(__name__)
    if fixed and separated_kernels:
        separated_kernels = False
        logger.warn('Cannot use separated kernels with a fixed '
                    'RateSpecialization, disabling...')

    base_kernel_data = []
    # add problem size
    base_kernel_data.extend(arc.initial_condition_dimension_vars(
        loopy_opts, test_size))

    # if we need the rtype array, add it
    if not separated_kernels and not fixed:
        rtype_attr = getattr(namestore, '{}_rtype'.format(tag))
        # get domain and corresponing kf inds
        domain, inds, num = rdomain(-1)
        # add map
        mapstore.check_and_add_transform(rtype_attr, domain)
        # create
        rtype_lp, rtype_str = mapstore.apply_maps(rtype_attr, var_name)
        # add
        base_kernel_data.append(rtype_lp)

    # create temperature array / str
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    base_kernel_data.insert(0, T_arr)

    Tinv = 'Tinv'
    logT = 'logT'
    Tval = 'Tval'
    guardT = ic.TemperatureGuard(loopy_opts)
    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)
    default_preinstructs = {Tinv:
                            precompute(Tinv, T_str, 'INV', guard=guardT),
                            logT:
                            precompute(logT, T_str, 'LOG', guard=guardT),
                            Tval:
                            precompute(Tval, T_str, 'VAL', guard=guardT)}

    # guarded exponential evaluator
    expg = ic.GuardedExp(loopy_opts)
    # generic kf assigment str
    kf_assign = Template("${kf_str} = ${rate} {id=rate_eval0, \
                         nosync=${deps}}")
    expkf_assign = Template(Template(
        "${kf_str} = ${exp_guarded} {id=rate_eval${id}, nosync=${deps}}")
        .safe_substitute(exp_guarded=expg('${rate}')))

    # put rateconst info args in dict for unpacking convenience
    extra_args = {'kernel_data': base_kernel_data,
                  'var_name': var_name,
                  'manglers': [expg, precompute]}

    def __get_instructions(rtype, mapper, kernel_data, beta_iter=1,
                           single_kernel_rtype=None, Tval=Tval, logT=logT,
                           Tinv=Tinv, specializations=set(),
                           preambles=[],
                           manglers=[]):
        # get domain
        domain, inds, num = rdomain(rtype)

        # use the single_kernel_rtype to find instructions
        if rtype < 0:
            rtype = single_kernel_rtype

        # get attrs
        A_attr = getattr(namestore, '{}_A'.format(tag))
        b_attr = getattr(namestore, '{}_beta'.format(tag))
        Ta_attr = getattr(namestore, '{}_Ta'.format(tag))
        kf_attr = getattr(namestore, 'kf' if tag == 'simple' else 'kf_fall')

        # ensure the map inds are keyed off the num
        if (separated_kernels or fixed) and not falloff:
            mapper.check_and_add_transform(inds, num)

        # add maps
        mapper.check_and_add_transform(A_attr, domain)
        mapper.check_and_add_transform(b_attr, domain)
        mapper.check_and_add_transform(Ta_attr, domain)
        mapper.check_and_add_transform(kf_attr, inds)

        # create A / b / ta
        A_lp, A_str = mapper.apply_maps(A_attr, var_name)
        b_lp, b_str = mapper.apply_maps(b_attr, var_name)
        Ta_lp, Ta_str = mapper.apply_maps(Ta_attr, var_name)
        kf_lp, kf_str = mapper.apply_maps(kf_attr, *default_inds)

        # add arrays
        kernel_data.extend([A_lp, b_lp, Ta_lp, kf_lp])

        # get rate equations
        rate_eqn_pre = Template(
            "${A_str} + ${logT} * ${b_str} - ${Ta_str} * ${Tinv}"
            ).safe_substitute(**locals())
        rate_eqn_pre_noTa = Template(
            "${A_str} + ${logT} * ${b_str}").safe_substitute(**locals())
        rate_eqn_pre_nobeta = Template(
            "${A_str} - ${Ta_str} * ${Tinv}").safe_substitute(
            **locals())

        def __deps(assign):
            if single_kernel_rtype is None:
                # not a combined kernel
                return ' '
            elif assign == expkf_assign:
                rate_evals = [0, 1, 2, 3, 4] if full else [0, 1, 2] if hybrid else \
                    [0]
                rate_evals = sorted(set(rate_evals) & specializations)
                deps = ['rate_eval{}'.format(x) for x in rate_evals]
                if 1 in rate_evals:
                    # add beta iter deps
                    deps += ['a*']
                deps = ':'.join(deps)
                return '{}'.format(':' + deps if deps else '')
            elif assign == kf_assign:
                return 'rate_eval*:a*'
            else:
                return 'rate_eval*'

        manglers = [x for x in manglers]
        preambles = [x for x in preambles]
        # the simple formulation
        if fixed or (hybrid and rtype == 2) or (full and rtype == 4):
            retv = expkf_assign.safe_substitute(rate=str(rate_eqn_pre),
                                                deps=__deps(expkf_assign),
                                                id=rtype)
        # otherwise check type and return appropriate instructions with
        # array strings substituted in
        elif rtype == 0:
            retv = kf_assign.safe_substitute(rate=A_str, deps=__deps(kf_assign))
        elif rtype == 1:
            power_func = ic.power_function(
                loopy_opts, is_integer_power=True,
                is_positive_power=True,
                is_vector=loopy_opts.is_simd)
            T_power = power_func(Tval, 'b_end')
            beta_iter_str = Template("""
            <int32> b_end = ${b_str}
            kf_temp = kf_temp * ${T_power} {id=a2, dep=a1}
            ${kf_str} = kf_temp {id=rate_eval1, dep=a2, nosync=${deps}}
            """).safe_substitute(b_str=b_str, deps=__deps(''), T_power=T_power)
            # this is about the one place where we must do this directly
            manglers.extend(power_func.manglers + [
                x.func_mangler for x in power_func.preambles])
            preambles.extend(power_func.preambles)

            retv = Template(
                """
                <>kf_temp = ${A_str} {id=a1}
                ${beta_iter_str}
                """).safe_substitute(**locals())
        elif rtype == 2:
            retv = expkf_assign.safe_substitute(
                rate=str(rate_eqn_pre_noTa), deps=__deps(expkf_assign),
                id=rtype)
        elif rtype == 3:
            retv = expkf_assign.safe_substitute(
                rate=str(rate_eqn_pre_nobeta), deps=__deps(expkf_assign),
                id=rtype)

        return Template(retv).safe_substitute(kf_str=kf_str), preambles, manglers

    # various specializations of the rate form
    specializations = {}
    i_a_only = k_gen.knl_info(name='a_only_{}'.format(tag),
                              instructions='',
                              mapstore=mapstore,
                              **extra_args)

    # the default is a single multiplication
    # if needed, this will be expanded to a for-loop multiplier
    i_beta_int = k_gen.knl_info(name='beta_int_{}'.format(tag),
                                instructions='',
                                mapstore=mapstore,
                                pre_instructions=[
                                    default_preinstructs[Tval],
                                    default_preinstructs[Tinv]],
                                **extra_args)
    i_beta_exp = k_gen.knl_info('beta_exp_{}'.format(tag),
                                instructions='',
                                mapstore=mapstore,
                                pre_instructions=[
                                    default_preinstructs[Tinv],
                                    default_preinstructs[logT]],
                                **extra_args)
    i_ta_exp = k_gen.knl_info('ta_exp_{}'.format(tag),
                              instructions='',
                              mapstore=mapstore,
                              pre_instructions=[
        default_preinstructs[Tinv],
        default_preinstructs[logT]],
        **extra_args)
    i_full = k_gen.knl_info('rateconst_full{}'.format(tag),
                            instructions='',
                            mapstore=mapstore,
                            pre_instructions=[
        default_preinstructs[Tinv],
        default_preinstructs[logT]],
        **extra_args)

    # set up the simple arrhenius rate specializations
    if fixed:
        specializations[0] = i_full
    else:
        specializations[0] = i_a_only
        specializations[1] = i_beta_int
        if full:
            specializations[2] = i_beta_exp
            specializations[3] = i_ta_exp
            specializations[4] = i_full
        else:
            specializations[2] = i_full

    # filter out unused specializations
    for rtype in specializations.copy():
        domain, _, _ = rdomain(rtype)
        if domain is None or not domain.initializer.size:
            # kernel doesn't act on anything, don't add it to output
            del specializations[rtype]

    # find out if beta iteration needed
    beta_iter = False
    b_attr = getattr(namestore, '{}_beta'.format(tag))
    rtype_attr = getattr(namestore, '{}_rtype'.format(tag))
    # find locations
    locs = np.where(rtype_attr.initializer == 1)
    b_vals = b_attr.initializer[locs]
    if b_vals.size:
        # if max b exponent > 1, need to iterate
        beta_iter = int(np.max(np.abs(b_vals)))

    # if single kernel, and not a fixed exponential
    if not separated_kernels and not fixed:
        # need to enclose each branch in it's own if statement
        do_conditional = len(specializations) > 1
        instruction_list = []
        man = []
        pre = []
        for i in specializations:
            if do_conditional:
                instruction_list.append(
                    'if {1} == {0}'.format(i, rtype_str))
            insns, preambles, manglers = __get_instructions(
                -1,
                arc.MapStore(loopy_opts, mapstore.map_domain, test_size),
                specializations[i].kernel_data,
                beta_iter,
                single_kernel_rtype=i,
                specializations=set(specializations.keys()),
                manglers=specializations[i].manglers,
                preambles=specializations[i].preambles)
            instruction_list.extend([
                '\t' + x for x in insns.split('\n') if x.strip()])
            if do_conditional:
                instruction_list.append('end')
            if manglers:
                man.extend(manglers)
            if preambles:
                pre.extend(preambles)
        # and combine them
        kernel_data = list(specializations.values())[0].kernel_data
        specializations = {-1: k_gen.knl_info(
                           'rateconst_singlekernel_{}'.format(tag),
                           instructions='\n'.join(instruction_list),
                           pre_instructions=list(
                               default_preinstructs.values()),
                           mapstore=mapstore,
                           kernel_data=kernel_data,
                           var_name=var_name,
                           manglers=man,
                           preambles=pre,
                           silenced_warnings=['write_race(rate_eval0)',
                                              'write_race(rate_eval1)',
                                              'write_race(rate_eval2)',
                                              'write_race(rate_eval3)',
                                              'write_race(rate_eval4)',
                                              'write_race(a2)'])}

    out_specs = {}
    # and do some finalizations for the specializations
    for rtype, info in specializations.items():
        # this is handled above
        if rtype < 0:
            out_specs[rtype] = info
            continue

        # turn off warning
        info.kwargs['silenced_warnings'] = ['write_race(rate_eval{})'.format(rtype),
                                            'write_race(a2)']

        domain, _, num = rdomain(rtype)
        if domain is None or not domain.initializer.size:
            # kernel doesn't act on anything, don't add it to output
            continue

        # next create a mapper for this rtype
        mapper = arc.MapStore(loopy_opts, domain, test_size)

        # set as mapper
        info.mapstore = mapper

        # if a specific rtype, get the instructions here
        if rtype >= 0:
            info.instructions, info.preambles, info.manglers = __get_instructions(
                rtype, mapper, info.kernel_data, beta_iter,
                specializations=sorted(set(specializations.keys())),
                manglers=specializations[rtype].manglers,
                preambles=specializations[rtype].preambles)

        out_specs[rtype] = info

    return list(out_specs.values())


def get_specrates_kernel(reacs, specs, loopy_opts, conp=True, test_size=None,
                         auto_diff=False, output_full_rop=False,
                         mem_limits='', **kwargs):
    """Helper function that generates kernels for
       evaluation of reaction rates / rate constants / and species rates

    Parameters
    ----------
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
    output_full_rop : bool
        If ``True``, output forward and reversse rates of progress
        Useful in testing, as there are serious floating point errors for
        net production rates near equilibrium, invalidating direct comparison to
        Cantera
    mem_limits: str ['']
        Path to a .yaml file indicating desired memory limits that control the
        desired maximum amount of global / local / or constant memory that
        the generated pyjac code may allocate.  Useful for testing,f or otherwise
        limiting memory usage during runtime. The keys of this file are the
        members of :class:`pyjac.kernel_utils.memory_limits.mem_type`
    kwargs: dict
        Arguements for the construction of the :class:`kernel_generator`

    Returns
    -------
    kernel_gen : :class:`kernel_generator`
        The generator responsible for creating the resulting code

    """

    # figure out rates and info
    rate_info = assign_rates(reacs, specs, loopy_opts.rate_spec)

    # create the namestore
    nstore = arc.NameStore(loopy_opts, rate_info, conp, test_size)

    kernels = []

    def __add_knl(knls, klist=None):
        if klist is None:
            klist = kernels
        try:
            klist.extend([x for x in knls if x is not None])
        except TypeError:
            if knls is not None:
                klist.append(knls)

    # Note:
    # the order in which these kernels get added is important
    # the kernel generator uses the input order to generate the wrapping
    # kernel calls
    # hence, any data dependencies should be expressed in the order added here

    # reset kernels
    __add_knl(reset_arrays(loopy_opts, nstore, test_size=test_size))

    # first, add the concentration kernel
    __add_knl(get_concentrations(loopy_opts, nstore, conp=conp,
                                 test_size=test_size))

    # get the simple arrhenius k_gen.knl_info's
    __add_knl(get_simple_arrhenius_rates(loopy_opts,
                                         nstore, test_size=test_size))

    # check for plog
    if rate_info['plog']['num']:
        # generate the plog kernel
        __add_knl(get_plog_arrhenius_rates(loopy_opts,
                                           nstore, rate_info['plog']['max_P'],
                                           test_size=test_size))

    # check for chebyshev
    if rate_info['cheb']['num']:
        __add_knl(get_cheb_arrhenius_rates(loopy_opts,
                                           nstore,
                                           np.max(rate_info['cheb']['num_P']),
                                           np.max(rate_info['cheb']['num_T']),
                                           test_size=test_size))

    # check for third body terms
    if rate_info['thd']['num']:
        # add the initial third body conc eval kernel
        __add_knl(get_thd_body_concs(loopy_opts,
                                     nstore, test_size))

    # check for falloff
    if rate_info['fall']['num']:
        # get the falloff rates
        __add_knl(get_simple_arrhenius_rates(loopy_opts,
                                             nstore, test_size=test_size,
                                             falloff=True))
        # and the reduced pressure
        __add_knl(get_reduced_pressure_kernel(loopy_opts,
                                              nstore, test_size=test_size))
        # and finally any blending functions (depend on reduced pressure)
        if rate_info['fall']['lind']['num']:
            __add_knl(get_lind_kernel(loopy_opts,
                                      nstore, test_size=test_size))
        if rate_info['fall']['troe']['num']:
            __add_knl(get_troe_kernel(loopy_opts,
                                      nstore, test_size=test_size))
        if rate_info['fall']['sri']['num']:
            __add_knl(get_sri_kernel(loopy_opts,
                                     nstore, test_size=test_size))

    # thermo polynomial dimension
    depends_on = []
    # check for reverse rates
    if rate_info['rev']['num']:
        # add the 'b' eval
        __add_knl(polyfit_kernel_gen('b', loopy_opts,
                                     nstore, test_size))
        # addd the 'b' eval to depnediencies
        depends_on.append(kernels[-1])
        # add Kc / rev rates
        __add_knl(get_rev_rates(loopy_opts,
                                nstore,
                                allint={'net': rate_info['net']['allint']},
                                test_size=test_size))

    # check for falloff
    if rate_info['fall']['num']:
        # and the Pr evals
        __add_knl(get_rxn_pres_mod(loopy_opts,
                                   nstore, test_size))

    # add ROP
    __add_knl(get_rop(loopy_opts,
                      nstore, allint={'net': rate_info['net']['allint']},
                      test_size=test_size))
    # add ROP net
    __add_knl(get_rop_net(loopy_opts,
                          nstore, test_size))
    # add spec rates
    __add_knl(get_spec_rates(loopy_opts,
                             nstore, test_size))

    # add molar rates
    __add_knl(get_molar_rates(loopy_opts, nstore, conp=conp,
                              test_size=test_size))

    if conp:
        # get h / cp evals
        __add_knl(polyfit_kernel_gen('h', loopy_opts, nstore,
                                     test_size), depends_on)
        __add_knl(polyfit_kernel_gen('cp', loopy_opts, nstore,
                                     test_size), depends_on)
    else:
        # and u / cv
        __add_knl(polyfit_kernel_gen('u', loopy_opts, nstore,
                                     test_size), depends_on)
        __add_knl(polyfit_kernel_gen('cv', loopy_opts, nstore,
                                     test_size), depends_on)
    # and add to source rates
    __add_knl(depends_on[-2:])

    # and temperature rates
    __add_knl(get_temperature_rate(loopy_opts,
                                   nstore, test_size=test_size, conp=conp))
    # and finally the extra variable rates
    __add_knl(get_extra_var_rates(loopy_opts, nstore, conp=conp,
                                  test_size=test_size))

    # get a wrapper for the dependecies
    thermo_in, thermo_out = inputs_and_outputs(conp, KernelType.chem_utils)
    thermo_wrap = k_gen.make_kernel_generator(kernel_type=KernelType.chem_utils,
                                              loopy_opts=loopy_opts,
                                              kernels=depends_on,
                                              namestore=nstore,
                                              input_arrays=thermo_in,
                                              output_arrays=thermo_out,
                                              auto_diff=auto_diff,
                                              test_size=test_size,
                                              mem_limits=mem_limits
                                              )

    barriers = []
    if loopy_opts.depth:
        def __insert_at(name, before=True):
            ind = next((i for i, knl in enumerate(kernels)
                        if knl.name == name), None)
            if ind is not None:
                if before:
                    barriers.append((ind - 1, ind, 'global'))
                else:
                    barriers.append((ind, ind + 1, 'global'))
        # need to add barriers
        # barrier at third bodies for get_concentrations
        __insert_at('eval_thd_body_concs', True)
        # barrier for reduced pressure based on thd body concs and kf_fall
        __insert_at('red_pres', True)
        # barrier before fall_troe for Pr
        if rate_info['fall']['num']:
            if not rate_info['fall']['troe']['num']:
                # try the fall_sri
                __insert_at('fall_sri', True)
            else:
                # put before troe
                __insert_at('fall_troe', True)
            # barrier before the falloff ci's for the Fi's
            __insert_at('ci_fall', True)
        if rate_info['rev']['num']:
            # barrier before Kc for b evals
            __insert_at('rateconst_Kc', True)
        if loopy_opts.rop_net_kernels:
            # need sync after each rop_net
            for x in ['rop_net_rev', 'rop_net_pres_mod']:
                __insert_at(x, True)
        elif rate_info['rev']['num'] or rate_info['thd']['num']:
            # if it's a fixed rop net, and there are reverse
            # or third body reactions
            __insert_at('rop_net_fixed', True)
        # barrier at species rates for the net ROP
        __insert_at('spec_rates', True)
        # barrier at molar rates for wdot
        __insert_at('get_molar_rates', True)
        # barrier at temperature rates for thermo parameters
        __insert_at('temperature_rate', True)
        # and at the extra variable rates for Tdot
        __insert_at('get_extra_var_rates', True)

    input_arrays, output_arrays = inputs_and_outputs(
        conp, output_full_rop=output_full_rop)
    if output_full_rop:
        if not rate_info['rev']['num']:
            output_arrays = [x for x in output_arrays if x != 'rop_rev']
        if not rate_info['thd']['num']:
            output_arrays = [x for x in output_arrays if x != 'pres_mod']
    return k_gen.make_kernel_generator(
        loopy_opts=loopy_opts,
        kernel_type=KernelType.species_rates,
        kernels=kernels,
        namestore=nstore,
        depends_on=[thermo_wrap],
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        auto_diff=auto_diff,
        test_size=test_size,
        barriers=barriers,
        mem_limits=mem_limits,
        **kwargs)


def polyfit_kernel_gen(nicename, loopy_opts, namestore, test_size=None):
    """Helper function that generates kernels for
       evaluation of various thermodynamic species properties

    Parameters
    ----------
    nicename : str
        The variable name to use in generated code
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not None, this kernel is being used for testing.

    Returns
    -------
    knl : :class:`loopy.LoopKernel`
        The generated loopy kernel for code generation / testing

    """

    # check for empty
    if nicename in ['b', 'db'] and namestore.rev_map is None:
        return None

    param_ind = 'dummy'
    loop_index = 'k'
    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.num_specs, test_size, loop_index)

    knl_data = []
    # add problem size
    knl_data.extend(arc.initial_condition_dimension_vars(loopy_opts, test_size))

    # get correctly ordered arrays / strings
    a_lo_lp, _ = mapstore.apply_maps(namestore.a_lo, loop_index, param_ind)
    a_hi_lp, _ = mapstore.apply_maps(namestore.a_hi, loop_index, param_ind)
    poly_dim = namestore.a_lo.shape[1]
    T_mid_lp, T_mid_str = mapstore.apply_maps(namestore.T_mid, loop_index)

    # create the input/temperature arrays
    T_lp, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    out_lp, out_str = mapstore.apply_maps(getattr(namestore, nicename),
                                          global_ind, loop_index)

    knl_data.extend([a_lo_lp, a_hi_lp, T_mid_lp, T_lp, out_lp])

    # create string indexes for a_lo/a_hi
    a_lo_strs = [mapstore.apply_maps(namestore.a_lo, loop_index, str(i))[1]
                 for i in range(poly_dim)]
    a_hi_strs = [mapstore.apply_maps(namestore.a_hi, loop_index, str(i))[1]
                 for i in range(poly_dim)]
    # mapping of nicename -> eqn
    eqn_maps = {'cp': Template(
        "Ru * (T * (T * (T * (T * ${a4} + ${a3}) + ${a2}) + ${a1}) + ${a0})"),
        'dcp': Template(
        "Ru * (T * (T *(4 * T * ${a4} + 3 * ${a3} ) + 2 * ${a2}) + ${a1})"),
        'h': Template(
        "Ru * (T * (T * (T * (T * (T * ${a4} / 5 + ${a3} / 4) + ${a2} / 3) + "
        "${a1} / 2) + ${a0}) + ${a5})"),
        'cv': Template(
        "Ru * (T * (T * (T * (T * ${a4} + ${a3}) + ${a2}) + ${a1}) + ${a0} - 1)"),
        'dcv': Template(
        "Ru * (T * (T * (4 * T * ${a4} + 3 * ${a3} ) + 2 * ${a2}) + ${a1})"),
        'u': Template(
        "Ru * (T * (T * (T * (T * (T * ${a4} / 5 + ${a3} / 4) + ${a2} / 3) + "
        "${a1} / 2) + ${a0}) - T + ${a5})"),
        'b': Template(
        "T * (T * (T * (T * ${a4} / 20 + ${a3} / 12) + ${a2} / 6) + ${a1} / 2) + "
        "(${a0} - 1) * logT - ${a0} + ${a6} - ${a5} * Tinv"),
        'db': Template(
        "T * (T * (T * ${a4} / 5 + ${a3} / 4) + ${a2} / 3) + ${a1} / 2 + "
        "Tinv * (${a0} - 1 + ${a5} * Tinv)")}
    # create lo / hi equation
    lo_eq = eqn_maps[nicename].safe_substitute(
        {'a' + str(i): a_lo for i, a_lo in enumerate(a_lo_strs)})
    hi_eq = eqn_maps[nicename].safe_substitute(
        {'a' + str(i): a_hi for i, a_hi in enumerate(a_hi_strs)})

    # guard the temperature to avoid SigFPE's in equil. constant eval
    guard = ic.TemperatureGuard(loopy_opts)

    Tval = 'T'
    # create a precomputed instruction generator
    precompute = ic.PrecomputedInstructions(loopy_opts)
    preinstructs = [precompute(Tval, T_str, 'VAL', guard=guard)]
    if nicename in ['db', 'b']:
        preinstructs.append(precompute('Tinv', T_str, 'INV', guard=guard))
        if nicename == 'b':
            preinstructs.append(precompute('logT', T_str, 'LOG', guard=guard))

    return k_gen.knl_info(instructions=Template("""
        if ${Tval} < ${T_mid_str}
            ${out_str} = ${lo_eq} {id=low, nosync=hi}
        else
            ${out_str} = ${hi_eq} {id=hi, nosync=low}
        end
        """).safe_substitute(**locals()),
                          kernel_data=knl_data,
                          pre_instructions=preinstructs,
                          name='eval_{}'.format(nicename),
                          parameters={'Ru': chem.RU},
                          var_name=loop_index,
                          mapstore=mapstore,
                          manglers=[precompute],
                          silenced_warnings=['write_race(low)',
                                             'write_race(hi)'])


def write_chem_utils(reacs, specs, loopy_opts, conp=True,
                     test_size=None, auto_diff=False,
                     mem_limits='', **kwargs):
    """Helper function that generates kernels for
       evaluation of species thermodynamic quantities

    Parameters
    ----------
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
    mem_limits: str ['']
        Path to a .yaml file indicating desired memory limits that control the
        desired maximum amount of global / local / or constant memory that
        the generated pyjac code may allocate.  Useful for testing, or otherwise
        limiting memory usage during runtime. The keys of this file are the
        members of :class:`pyjac.kernel_utils.memory_limits.mem_type`
    kwargs: dict
        Arguements for the construction of the :class:`kernel_generator`

    Returns
    -------
    kernel_gen : :class:`kernel_generator`
        The generator responsible for creating the resulting code

    """

    # figure out rates and info
    rate_info = assign_rates(reacs, specs, loopy_opts.rate_spec)

    # create the namestore
    nstore = arc.NameStore(loopy_opts, rate_info, conp, test_size)

    # generate the kernels
    output = ['cp', 'h', 'b'] if conp else ['cv', 'u', 'b']
    kernels = []
    for nicename in output:
        kernels.append(polyfit_kernel_gen(nicename, loopy_opts,
                                          nstore, test_size))

    return k_gen.make_kernel_generator(
        loopy_opts=loopy_opts,
        kernel_type=KernelType.chem_utils,
        kernels=kernels,
        namestore=nstore,
        input_arrays=['phi'],
        output_arrays=output,
        auto_diff=auto_diff,
        test_size=test_size,
        mem_limits=mem_limits,
        **kwargs
    )


if __name__ == "__main__":
    utils.create(kernel_type=KernelType.species_rates)
