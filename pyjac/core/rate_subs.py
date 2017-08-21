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
import sympy as sp
import loopy as lp
import numpy as np
from loopy.kernel.data import temp_var_scope as scopes
from ..loopy_utils import loopy_utils as lp_utils

# Local imports
from .. import utils
from . import chem_model as chem
from ..kernel_utils import kernel_gen as k_gen
from ..sympy_utils import sympy_utils as sp_utils
from . reaction_types import reaction_type, falloff_form, thd_body_type, \
    reversible_type
from . import array_creator as arc
from ..loopy_utils import preambles_and_manglers as lp_pregen
from . import instruction_creator as ic
from .array_creator import (global_ind, var_name, default_inds)


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

    if rate_spec == lp_utils.rate_specialization.fixed
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

    # determine specialization
    full = rate_spec == lp_utils.RateSpecialization.full
    # hybrid = rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = rate_spec == lp_utils.RateSpecialization.fixed

    # find fwd / reverse rate parameters
    # first, the number of each
    rev_map = np.array([i for i, x in enumerate(reacs) if x.rev],
                       dtype=np.int32)
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
    reac_has_ns = np.array(reac_has_ns, dtype=np.int32)
    net_nu_integer = all(utils.is_integer(nu) for nu in net_nu)
    if net_nu_integer:
        nu_sum = np.array(nu_sum, dtype=np.int32)
        net_nu = np.array(net_nu, dtype=np.int32)
        ns_nu = np.array(ns_nu, dtype=np.int32)
    else:
        nu_sum = np.array(nu_sum)
        net_nu = np.array(net_nu)
        ns_nu = np.array(ns_nu)
    net_num_spec = np.array(net_num_spec, dtype=np.int32)
    net_spec = np.array(net_spec, dtype=np.int32)

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

    spec_to_reac = np.array(spec_to_reac, dtype=np.int32)
    if net_nu_integer:
        spec_nu = np.array(spec_nu, dtype=np.int32)
    else:
        spec_nu = np.array(spec_nu)
    spec_reac_count = np.array(spec_reac_count, dtype=np.int32)
    spec_list = np.array(spec_list, dtype=np.int32)

    def __seperate(reacs, matchers):
        # find all reactions / indicies that match this offset
        rate = [(i, x) for i, x in enumerate(reacs) if any(x.match(y) for y in
                                                           matchers)]
        mapping = np.empty(0, dtype=np.int32)
        num = 0
        if rate:
            mapping, rate = zip(*rate)
            mapping = np.array(mapping, dtype=np.int32)
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
        rate_type = np.zeros((num,), dtype=np.int32)
        if fall:
            fall_types = np.zeros((num,), dtype=np.int32)
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
    num_pressures = np.array(num_pressures, dtype=np.int32)

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
    cheb_n_pres = np.array(cheb_n_pres, dtype=np.int32)
    cheb_n_temp = np.array(cheb_n_temp, dtype=np.int32)
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
        y, falloff_form)) for x in fall_reacs], dtype=np.int32)
    # seperate parameters based on blending type
    # lindeman
    lind_map = np.where(blend_type == int(falloff_form.lind))[
        0].astype(dtype=np.int32)
    # sri
    sri_map = np.where(blend_type == int(falloff_form.sri))[
        0].astype(dtype=np.int32)
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
        0].astype(dtype=np.int32)
    troe_reacs = [reacs[fall_map[i]] for i in troe_map]
    troe_par = [reac.troe_par for reac in troe_reacs]
    # now fill in defaults as needed
    for par_set in troe_par:
        if len(par_set) != 4:
            par_set.append(0)
    try:
        troe_a, troe_T3, troe_T1, troe_T2 = [
            np.array(x, dtype=np.float64) for x in zip(*troe_par)]
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
        y, thd_body_type)) for x in thd_reacs], dtype=np.int32)

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

    thd_spec_num = np.array(thd_spec_num, dtype=np.int32)
    thd_spec = np.array(thd_spec, dtype=np.int32)
    thd_eff = np.array(thd_eff, dtype=np.float64)
    thd_has_ns = np.array(thd_has_ns, dtype=np.int32)
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
                'T_mid': T_mid
            },
            'mws': mws,
            'mw_post': mw_post,
            'reac_has_ns': reac_has_ns,
            'ns_nu': ns_nu
            }


def reset_arrays(eqs, loopy_opts, namestore, test_size=None):
    """Resets the dphi and wdot arrays for use in the rate evaluations

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

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    mapstore = arc.MapStore(loopy_opts,
                            namestore.phi_inds,
                            namestore.phi_inds)

    # first, create all arrays
    kernel_data = []

    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

    # need dphi and wdot arrays

    ndot_lp, ndot_str = mapstore.apply_maps(namestore.n_dot, *default_inds)
    wdot_lp, wdot_str = mapstore.apply_maps(namestore.spec_rates, *default_inds)

    # number of species
    ns = namestore.num_specs.initializer[-1]
    # and index
    ind = var_name

    # add arrays
    kernel_data.extend([ndot_lp, wdot_lp])

    instructions = Template(
        """
            ${ndot_str} = 0d
            if ${ind} < ${ns}
                ${wdot_str} = 0d
            end
        """).substitute(**locals())

    return k_gen.knl_info(name='reset_arrays',
                          instructions=instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data)


def get_concentrations(eqs, loopy_opts, namestore, conp=True,
                       test_size=None):
    """Determines concentrations from moles and state variables depending
    on constant pressure vs constant volue assumption

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

    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    fixed_inds = (global_ind,)

    # first, create all arrays
    kernel_data = []

    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

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

    pre_instructions = Template(
        """<>V_inv = 1.0d / ${V_str}
           <>n_sum = 0
           ${cns_str} = ${P_str} / (R_u * ${T_str}) {id=cns_init}
        """).substitute(
            P_str=P_str,
            V_str=V_str,
            T_str=T_str,
            cns_str=conc_ns_str)

    instructions = Template(
        """
            ${conc_str} = ${n_str} * V_inv {id=cn_init}
            n_sum = n_sum + ${n_str} {id=n_update}
        """).substitute(
            conc_str=conc_str,
            n_str=n_str
    )

    post_instructions = Template(
        """
        ${cns_str} = ${cns_str} - n_sum * V_inv {id=cns_set, dep=n_update}
        """).substitute(cns_str=conc_ns_str)

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
                          parameters={'R_u': np.float64(chem.RU)})


def get_molar_rates(eqs, loopy_opts, namestore, conp=True,
                    test_size=None):
    """Generates instructions, kernel arguements, and data for the
       molar derivatives
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

    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    # first, create all arrays
    kernel_data = []

    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

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

    V_lp, V_str = mapstore.apply_maps(namestore.V_arr,
                                      *fixed_inds)

    V_val = 'V_val'
    pre_instructions = ic.default_pre_instructs(V_val, V_str, 'VAL')

    kernel_data.extend([V_lp, ndot_lp, wdot_lp])

    instructions = Template(
        """
        ${ndot_str} = ${V_val} * ${wdot_str}
        """
    ).substitute(
        ndot_str=ndot_str,
        V_val=V_val,
        wdot_str=wdot_str
    )

    return k_gen.knl_info(name='get_molar_rates',
                          pre_instructions=[pre_instructions],
                          instructions=instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data)


def get_extra_var_rates(eqs, loopy_opts, namestore, conp=True,
                        test_size=None):
    """Generates instructions, kernel arguements, and data for the
       derivative of the "extra" variable -- P or V depending on conV/conP
       assumption respectively
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

    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs_no_ns,
                            namestore.num_specs_no_ns)

    # first, create all arrays
    kernel_data = []

    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

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

    if conp:
        pre_instructions = [
            Template('${Edot_str} = ${V_str} * ${Tdot_str} / ${T_str} \
                     {id=init}').safe_substitute(
                **locals()),
            '<>dE = 0.0d'
        ]
    else:
        pre_instructions = [
            Template('${Edot_str} = ${P_str} * ${Tdot_str} / ${T_str}\
                     {id=init}').safe_substitute(
                **locals()),
            '<>dE = 0.0d'
        ]

    instructions = Template(
        """
            dE = dE + (1.0 - ${mw_str}) * ${wdot_str} {id=sum}
            """
    ).safe_substitute(**locals())

    if conp:
        kernel_data.append(V_lp)

        if loopy_opts.use_atomics and loopy_opts.depth:
            # need to fix the post instructions to work atomically
            pre_instructions = ['<>dE = 0.0d']
            post_instructions = [Template(
                """
                temp_sum = ${V_str} * ${Tdot_str} / ${T_str} {id=temp_init, dep=*,\
                                                              atomic}
                ... lbarrier {id=lb1, dep=temp_init}
                temp_sum = temp_sum + ${V_str} * dE * ${T_str} * R_u /  ${P_str} \
                    {id=temp_sum, dep=lb1*:sum, nosync=temp_init, atomic}
                ... lbarrier {id=lb2, dep=temp_sum}
                ${Edot_str} = temp_sum {id=final, dep=lb2, atomic, nosync=temp_init}
                """
                ).safe_substitute(**locals())]
            kernel_data.append(lp.TemporaryVariable('temp_sum', dtype=np.float64,
                                                    scope=scopes.LOCAL))
        else:
            post_instructions = [Template(
                """
                ${Edot_str} = ${Edot_str} + ${V_str} * dE * ${T_str} * R_u / \
                    ${P_str} {id=end, dep=sum:init, nosync=init}
                """).safe_substitute(**locals())
            ]

    else:
        if loopy_opts.use_atomics and loopy_opts.depth:
            # need to fix the post instructions to work atomically
            pre_instructions = ['<>dE = 0.0d']
            post_instructions = [Template(
                """
                temp_sum = ${P_str} * ${Tdot_str} / ${T_str} {id=temp_init, dep=*,\
                                                              atomic}
                ... lbarrier {id=lb1, dep=temp_init}
                temp_sum = temp_sum + ${T_str} * R_u * dE \
                    {id=temp_sum, dep=lb1*:sum, nosync=temp_init, atomic}
                ... lbarrier {id=lb2, dep=temp_sum}
                ${Edot_str} = temp_sum {id=final, dep=lb2, atomic, nosync=temp_init}
                """
                ).safe_substitute(**locals())]
            kernel_data.append(lp.TemporaryVariable('temp_sum', dtype=np.float64,
                                                    scope=scopes.LOCAL))
        else:
            post_instructions = [Template(
                """
                ${Edot_str} = ${Edot_str} + ${T_str} * R_u * dE {id=end, dep=sum:init, \
                                                                 nosync=init}
                """
            ).safe_substitute(**locals())]

    can_vectorize, vec_spec = ic.get_deep_specializer(
        loopy_opts, atomic_ids=['final', 'temp_sum'],
        init_ids=['init', 'temp_init'])
    return k_gen.knl_info(name='get_extra_var_rates',
                          pre_instructions=pre_instructions,
                          instructions=instructions,
                          post_instructions=post_instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          parameters={'R_u': np.float64(chem.RU)},
                          can_vectorize=can_vectorize,
                          vectorization_specializer=vec_spec)


def get_temperature_rate(eqs, loopy_opts, namestore, conp=True,
                         test_size=None):
    """Generates instructions, kernel arguements, and data for the
       temperature derivative
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

    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs,
                            namestore.num_specs)
    fixed_inds = (global_ind,)

    # here, the equation form _does_ matter
    if conp:
        term = next(x for x in eqs['conp'] if
                    str(x) == 'frac{text{d} T }{text{d} t }')
        term = eqs['conp'][term]
    else:
        term = next(x for x in eqs['conv'] if
                    str(x) == 'frac{text{d} T }{text{d} t }')
        term = eqs['conv'][term]

    # first, create all arrays
    kernel_data = []

    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

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
        <> upper = 0
        <> lower = 0
        ${Tdot_str} = 0 {id=init}
        """).safe_substitute(**locals())]
    instructions = Template(
        """
            upper = upper + ${energy_str} * ${wdot_str} {id=sum1}
            lower = lower + ${conc_str} * ${spec_heat_str} {id=sum2}
        """
    ).safe_substitute(**locals())

    post_instructions = [Template(
        """
        ${Tdot_str} = ${Tdot_str} - upper / lower {id=final, dep=sum*}
        """
    ).safe_substitute(**locals())]

    if loopy_opts.use_atomics and loopy_opts.depth:
        # need to fix the post instructions to work atomically
        post_instructions = [Template(
            """
            temp_sum = 0 {id=temp_init, atomic}
            temp_sum = temp_sum + lower {id=temp_sum, dep=temp_init:sum*,\
                                         nosync=temp_init, atomic}
            ... lbarrier {id=lb2, dep=temp_sum}
            ${Tdot_str} = ${Tdot_str} - upper / temp_sum {id=final, dep=lb2:init, \
                                                          nosync=init, atomic}
            """
            ).safe_substitute(**locals())]
        kernel_data.append(lp.TemporaryVariable('temp_sum', dtype=np.float64,
                                                scope=scopes.LOCAL))

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


def get_spec_rates(eqs, loopy_opts, namestore, conp=True,
                   test_size=None):
    """Generates instructions, kernel arguements, and data for the
       temperature derivative
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
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

    # various indicies
    spec_ind = 'spec_ind'
    ispec = 'ispec'

    # create map store
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_reacs,
                            namestore.num_reacs)

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


def get_rop_net(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the net
    Rate of Progress kernels

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

    # create net rop kernel

    kernel_data = OrderedDict([('fwd', [])])
    maps = OrderedDict([('fwd',
                         arc.MapStore(loopy_opts, namestore.num_reacs,
                                      namestore.num_reacs))])
    transforms = {'fwd': namestore.num_reacs}

    separated_kernels = loopy_opts.rop_net_kernels
    if separated_kernels:
        kernel_data['rev'] = []
        maps['rev'] = arc.MapStore(loopy_opts, namestore.num_rev_reacs,
                                   namestore.rev_mask)
        transforms['rev'] = namestore.rev_map
        kernel_data['pres_mod'] = []
        maps['pres_mod'] = arc.MapStore(loopy_opts, namestore.num_thd,
                                        namestore.thd_mask)
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

    if test_size == 'problem_size':
        __add_to_all(namestore.problem_size)

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
            ${rop_net_str} = ${rop_fwd_str}
                    """).safe_substitute(rop_fwd_str=rop_fwd_str,
                                         rop_net_str=rop_strs['fwd'])
            elif kernel == 'rev':
                instructions = Template(
                    """
            ${rop_net_str} = ${rop_net_str} - ${rop_rev_str}
                    """).safe_substitute(rop_rev_str=rop_rev_str,
                                         rop_net_str=rop_strs['rev'])
            else:
                instructions = Template(
                    """
            ${rop_net_str} = ${rop_net_str} * ${pres_mod_str}
                    """).safe_substitute(pres_mod_str=pres_mod_str,
                                         rop_net_str=rop_strs['pres_mod'])

            instructions = '\n'.join(
                [x for x in instructions.split('\n') if x.strip()])
            infos.append(k_gen.knl_info(name='rop_net_{}'.format(kernel),
                                        instructions=instructions,
                                        var_name=var_name,
                                        kernel_data=kernel_data[kernel],
                                        mapstore=maps[kernel]))
        return infos


def get_rop(eqs, loopy_opts, namestore, allint={'net': True}, test_size=None):
    """Generates instructions, kernel arguements, and data for the Rate of Progress
    kernels

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems.
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
        if test_size == 'problem_size':
            kernel_data.append(namestore.problem_size)
        if direction == 'fwd':
            inds = namestore.num_reacs
            mapinds = namestore.num_reacs
        else:
            inds = namestore.num_rev_reacs
            mapinds = namestore.rev_map

        maps[direction] = arc.MapStore(loopy_opts, inds, inds)
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

        # if all integers, it's much faster to use multiplication
        allint_eval = Template(
            """
    rop_temp = rop_temp * fast_powi(${concs_str}, ${nu_str}) {id=rop_fin}
    """).safe_substitute(
            nu_str=nu_str,
            concs_str=concs_str)

        # if we need to use powers, do so
        fractional_eval = Template(
            """
    if int(${nu_str}) == ${nu_str}
        ${allint}
    else
        rop_temp = rop_temp * fast_powf(${concs_str}, ${nu_str}) {id=rop_fin2}
    end
    """).safe_substitute(nu_str=nu_str,
                         concs_str=concs_str)
        fractional_eval = k_gen.subs_at_indent(fractional_eval, 'allint',
                                               allint_eval)

        if not allint['net']:
            rop_instructions = k_gen.subs_at_indent(rop_instructions,
                                                    'rop_temp_eval',
                                                    fractional_eval)
        else:
            rop_instructions = k_gen.subs_at_indent(rop_instructions,
                                                    'rop_temp_eval',
                                                    allint_eval)

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
                              preambles=[
            lp_pregen.fastpowi_PreambleGen(),
            lp_pregen.fastpowf_PreambleGen()])

    infos = [__rop_create('fwd')]
    if namestore.rop_rev is not None:
        infos.append(__rop_create('rev'))
    return infos


def get_rxn_pres_mod(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for pressure
    modification term of the forward reaction rates.

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

    # start developing the ci kernel
    # rate info and reac ind

    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # create the third body conc pres-mod kernel

    thd_map = arc.MapStore(loopy_opts, namestore.thd_only_map,
                           namestore.thd_only_mask)

    # get the third body concs
    thd_lp, thd_str = thd_map.apply_maps(namestore.thd_conc,
                                         *default_inds)

    # and the pressure mod term
    pres_mod_lp, pres_mod_str = thd_map.apply_maps(namestore.pres_mod,
                                                   *default_inds)

    thd_instructions = Template("""
    ${pres_mod} = ${thd_conc}
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
                       mapstore=thd_map)]

    # and now the falloff kernel
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    fall_map = arc.MapStore(loopy_opts, namestore.num_fall,
                            namestore.num_fall)

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


def get_rev_rates(eqs, loopy_opts, namestore, allint, test_size=None):
    """Generates instructions, kernel arguements, and data for reverse reaction
    rates

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
    # start developing the Kc kernel
    kernel_data = []
    spec_ind = 'spec_ind'
    spec_loop = 'ispec'

    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # set of eqn's doesn't matter
    conp_eqs = eqs['conp']

    # add the reverse map
    rev_map = arc.MapStore(loopy_opts, namestore.num_rev_reacs,
                           namestore.rev_mask)

    # find Kc equation
    Kc_sym = next(x for x in conp_eqs if str(x) == '{K_c}[i]')
    Kc_eqn = conp_eqs[Kc_sym]
    nu_sym = next(x for x in Kc_eqn.free_symbols if str(x) == 'nu[k, i]')
    B_sym = next(x for x in Kc_eqn.free_symbols if str(x) == 'B[k]')
    kr_sym = next(x for x in conp_eqs if str(x) == '{k_r}[i]')
    # kf_sym = next(x for x in conp_eqs if str(x) == '{k_f}[i]')

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

    # modify Kc equation
    Kc_eqn = sp_utils.sanitize(conp_eqs[Kc_sym],
                               symlist={'nu': nu_sym,
                                        'B': B_sym},
                               subs={
        sp.Sum(nu_sym, (sp.Idx('k'), 1, sp.Symbol('N_s'))): nu_sum_str})

    # insert the B sum into the Kc equation
    Kc_eqn_Pres = next(
        x for x in sp.Mul.make_args(Kc_eqn) if x.has(sp.Symbol('R_u')))
    Kc_eqn_exp = Kc_eqn / Kc_eqn_Pres
    Kc_eqn_exp = sp_utils.sanitize(Kc_eqn_exp,
                                   symlist={'nu': nu_sym,
                                            'B': B_sym},
                                   subs={
                                       sp.Sum(B_sym * nu_sym,
                                              (sp.Idx('k'), 1, sp.Symbol('N_s')
                                               )): 'B_sum'})

    # create the kf array / str
    kf_arr, kf_str = rev_map.apply_maps(
        namestore.kf, *default_inds)

    # create the kr array / str (no map as we're looping over rev inds)
    kr_arr, kr_str = rev_map.apply_maps(
        namestore.kr, *default_inds)

    # get the kr eqn
    Kc_temp_str = 'Kc_temp'
    # for some reason this substitution is poorly behaved
    # hence we just do this rather than deriving from sympy for the moment
    # kr_eqn = sp.Symbol(kf_str) / sp.Symbol(Kc_temp_str)
    kr_eqn = sp_utils.sanitize(conp_eqs[kr_sym][
        (reversible_type.non_explicit,)],
        symlist={'{k_f}[i]': sp.Symbol('kf[i]'),
                 '{K_c}[i]': sp.Symbol('Kc[i]')},
        subs={'kf[i]': kf_str,
              'Kc[i]': Kc_temp_str})

    # update kernel data
    kernel_data.extend([nu_sum_lp, spec_lp, num_spec_offsets_lp,
                        B_lp, Kc_lp, nu_lp, kf_arr, kr_arr])

    # create the pressure product loop
    pressure_prod = Template("""
    <> P_sum_end = abs(${nu_sum}) {id=P_bound}
    if ${nu_sum} > 0
        <> P_val = P_a / R_u {id=P_val_decl}
    else
        P_val = R_u / P_a {id=P_val_decl1}
    end
    <> P_sum = fast_powi(P_val, P_sum_end) {id=P_accum, dep=P_val_decl*}
    """).substitute(nu_sum=nu_sum_str)

    if not allint['net']:
        # if not all integers, need to add outer if statment to check integer
        # status
        pressure_prod_temp = Template("""
    <> P_sum_end = abs(${nu_sum}) {id=P_bound}
    if ${nu_sum} > 0
        <> P_val = P_a / R_u {id=P_val_decl}
    else
        P_val = R_u / P_a {id=P_val_decl1}
    end
    if (int)${nu_sum} == ${nu_sum}
        P_sum = fast_powi(P_val, P_sum_end) {id=P_accum, dep=P_val_decl*}
    else
        P_sum = fast_powf(P_val, ${nu_sum}) {id=P_accum2, dep=P_val_decl*}
    end""").substitute(nu_sum=nu_sum_str)

        pressure_prod = k_gen.subs_at_indent(pressure_prod_temp, 'pprod',
                                             pressure_prod)

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
    B_sum = exp(B_sum) {id=B_final, dep=B_accum}
    """).substitute(spec_offset=num_spec_offsets_str,
                    spec_offset_next=num_spec_offsets_next_str,
                    spec_loop=spec_loop,
                    spec_ind=spec_ind,
                    spec_mapper=spec_str,
                    nu_val=nu_sum_str,
                    prod_nu_str=prod_nu_str,
                    reac_nu_str=reac_nu_str,
                    B_val=B_str
                    )

    Rate_assign = Template("""
    <>${Kc_temp_str} = P_sum * B_sum {dep=P_accum*:B_final}
    ${Kc_val} = ${Kc_temp_str}
    ${kr_val} = ${rev_eqn}
    """).safe_substitute(Kc_val=Kc_str,
                         Kc_temp_str=Kc_temp_str,
                         kr_val=kr_str,
                         rev_eqn=kr_eqn)

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
                          preambles=[lp_pregen.fastpowi_PreambleGen(),
                                     lp_pregen.fastpowf_PreambleGen()])


def get_thd_body_concs(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for third body
    concentrations

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

    spec_ind = 'spec_ind'
    spec_loop = 'ispec'
    spec_offset = 'offset'

    # create mapstore over number of third reactions
    mapstore = arc.MapStore(loopy_opts, namestore.thd_inds, namestore.thd_inds)

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
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

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
    thd_temp = thd_temp + (${thd_eff} - not_spec) * ${conc_thd_spec} {id=thdcalc, dep=ind1}
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


def get_cheb_arrhenius_rates(eqs, loopy_opts, namestore, maxP, maxT,
                             test_size=None):
    """Generates instructions, kernel arguements, and data for cheb rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
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

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.cheb_map,
                            namestore.cheb_mask)

    # the equation set doesn't matter for this application
    # just use conp
    conp_eqs = eqs['conp']

    # find the cheb equation
    cheb_eqn = next(x for x in conp_eqs if str(x) == 'log({k_f}[i])/log(10)')
    cheb_form, cheb_eqn = cheb_eqn, conp_eqs[cheb_eqn][(reaction_type.cheb,)]
    cheb_form = sp.Pow(10, sp.Symbol('kf_temp'))

    # make nice symbols
    Tinv = sp.Symbol('Tinv')
    logP = sp.Symbol('logP')
    Pmax, Pmin, Tmax, Tmin = sp.symbols('Pmax Pmin Tmax Tmin')
    Pred, Tred = sp.symbols('Pred Tred')

    # get tilde{T}, tilde{P}
    T_red = next(x for x in conp_eqs if str(x) == 'tilde{T}')
    P_red = next(x for x in conp_eqs if str(x) == 'tilde{P}')

    Pred_eqn = sp_utils.sanitize(conp_eqs[P_red],
                                 subs={sp.log(sp.Symbol('P_{min}')): Pmin,
                                       sp.log(sp.Symbol('P_{max}')): Pmax,
                                       sp.log(sp.Symbol('P')): logP})

    Tred_eqn = sp_utils.sanitize(conp_eqs[T_red],
                                 subs={sp.S.One / sp.Symbol('T_{min}'): Tmin,
                                       sp.S.One / sp.Symbol('T_{max}'): Tmax,
                                       sp.S.One / sp.Symbol('T'): Tinv})

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

    # parameters, counts and limit arrays are based on number of
    # chebyshev reacs
    mapstore.check_and_add_transform(namestore.cheb_numP, namestore.num_cheb)
    mapstore.check_and_add_transform(namestore.cheb_numT, namestore.num_cheb)
    mapstore.check_and_add_transform(namestore.cheb_params, namestore.num_cheb)
    mapstore.check_and_add_transform(namestore.cheb_Plim, namestore.num_cheb)
    mapstore.check_and_add_transform(namestore.cheb_Tlim, namestore.num_cheb)

    # and the kf based on the map
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
    pres_poly_lp, pres_poly_str = mapstore.apply_maps(namestore.cheb_pres_poly,
                                                      pres_poly_ind)
    temp_poly_lp, temp_poly_str = mapstore.apply_maps(namestore.cheb_temp_poly,
                                                      temp_poly_ind)

    # create temperature and pressure arrays
    T_arr, T_str = mapstore.apply_maps(namestore.T_arr, global_ind)
    P_arr, P_str = mapstore.apply_maps(namestore.P_arr, global_ind)

    # get the forward rate constants
    kf_arr, kf_str = mapstore.apply_maps(namestore.kf, *default_inds)

    # update kernel data
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    kernel_data.extend([params_lp, num_P_lp, num_T_lp, plim_lp, tlim_lp,
                        pres_poly_lp, temp_poly_lp, T_arr, P_arr, kf_arr])

    # preinstructions
    logP = 'logP'
    Tinv = 'Tinv'
    preinstructs = [ic.default_pre_instructs(logP, P_str, 'LOG'),
                    ic.default_pre_instructs(Tinv, T_str, 'INV')]

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

    instructions = Template("""
<>Pmin = ${Pmin_str}
<>Tmin = ${Tmin_str}
<>Pmax = ${Pmax_str}
<>Tmax = ${Tmax_str}
<>Tred = ${Tred_str}
<>Pred = ${Pred_str}
<>numP = ${numP_str} {id=plim}
<>numT = ${numT_str} {id=tlim}
${ppoly_0} = 1
${ppoly_1} = Pred
${tpoly_0} = 1
${tpoly_1} = Tred
#<> poly_end = max(numP, numT)
# compute polynomial terms
for p
    if p < numP
        ${ppoly_p} = 2 * Pred * ${ppoly_pm1} - ${ppoly_pm2} {id=ppoly, dep=plim}
    end
    if p < numT
        ${tpoly_p} = 2 * Tred * ${tpoly_pm1} - ${tpoly_pm2} {id=tpoly, dep=tlim}
    end
end
<> kf_temp = 0
for m
    <>temp = 0
    for k
        temp = temp + ${ppoly_k} * ${chebpar_km} {id=temp, dep=ppoly:tpoly}
    end
    kf_temp = kf_temp + ${tpoly_m} * temp {id=kf, dep=temp}
end

${kf_str} = ${kf_eval} {dep=kf}
""")

    instructions = instructions.safe_substitute(
        kf_str=kf_str,
        Tred_str=str(Tred_eqn),
        Pred_str=str(Pred_eqn),
        Pmin_str=Pmin_str,
        Pmax_str=Pmax_str,
        Tmin_str=Tmin_str,
        Tmax_str=Tmax_str,
        ppoly_0=ppoly0_str,
        ppoly_1=ppoly1_str,
        ppoly_k=pres_poly_str,
        ppoly_p=ppolyp_str,
        ppoly_pm1=ppolypm1_str,
        ppoly_pm2=ppolypm2_str,
        tpoly_0=tpoly0_str,
        tpoly_1=tpoly1_str,
        tpoly_m=temp_poly_str,
        tpoly_p=tpolyp_str,
        tpoly_pm1=tpolypm1_str,
        tpoly_pm2=tpolypm2_str,
        chebpar_km=params_str,
        numP_str=num_P_str,
        numT_str=num_T_str,
        kf_eval=str(cheb_form))

    return k_gen.knl_info('rateconst_cheb',
                          instructions=instructions,
                          pre_instructions=preinstructs,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=mapstore,
                          extra_inames=extra_inames)


def get_plog_arrhenius_rates(eqs, loopy_opts, namestore, maxP, test_size=None):
    """Generates instructions, kernel arguements, and data for p-log rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume
        systems
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

    rate_eqn = get_rate_eqn(eqs)

    # find the plog equation
    plog_eqn = next(x for x in eqs['conp'] if str(x) == 'log({k_f}[i])')
    _, plog_eqn = plog_eqn, eqs[
        'conp'][plog_eqn][(reaction_type.plog,)]

    # now we do some surgery to obtain a form w/o 'logs' as we'll take them
    # explicitly in python
    logP = sp.Symbol('logP')
    logP1 = sp.Symbol('low[0]')
    logP2 = sp.Symbol('hi[0]')
    logk1 = sp.Symbol('logk1')
    logk2 = sp.Symbol('logk2')
    plog_eqn = sp_utils.sanitize(plog_eqn, subs={sp.log(sp.Symbol('k_1')): logk1,
                                                 sp.log(sp.Symbol('k_2')): logk2,
                                                 sp.log(sp.Symbol('P')): logP,
                                                 sp.log(sp.Symbol('P_1')): logP1,
                                                 sp.log(sp.Symbol('P_2')): logP2})

    # and specialize the k1 / k2 equations
    A1 = sp.Symbol('low[1]')
    b1 = sp.Symbol('low[2]')
    Ta1 = sp.Symbol('low[3]')
    k1_eq = sp_utils.sanitize(rate_eqn, subs={sp.Symbol('A[i]'): A1,
                                              sp.Symbol('beta[i]'): b1,
                                              sp.Symbol('Ta[i]'): Ta1})
    A2 = sp.Symbol('hi[1]')
    b2 = sp.Symbol('hi[2]')
    Ta2 = sp.Symbol('hi[3]')
    k2_eq = sp_utils.sanitize(rate_eqn, subs={sp.Symbol('A[i]'): A2,
                                              sp.Symbol('beta[i]'): b2,
                                              sp.Symbol('Ta[i]'): Ta2})

    # parameter indicies
    arrhen_ind = 'm'
    param_ind = 'k'
    lo_ind = 'lo_ind'
    hi_ind = 'hi_ind'

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.plog_map,
                            namestore.plog_mask)

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
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

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

    # instructions
    instructions = Template(
        """
        <>lower = ${logP} <= ${pressure_lo} # check below range
        if lower
            <>lo_ind = 0 {id=ind00}
            <>hi_ind = 0 {id=ind01}
        end
        <>numP = ${plog_num_param_str} - 1
        <>upper = ${logP} > ${pressure_hi} # check above range
        if upper
            lo_ind = numP {id=ind10}
            hi_ind = numP {id=ind11}
        end
        <>oor = lower or upper
        for k
            # check that
            # 1. inside this reactions parameter's still
            # 2. inside pressure range
            <> midcheck = (k <= numP) and (${logP} > ${pressure_mid_lo}) and (${logP} <= ${pressure_mid_hi})
            if midcheck
                lo_ind = k {id=ind20}
                hi_ind = k + 1 {id=ind21}
            end
        end
        for m
            low[m] = ${pressure_general_lo} {id=lo, dep=ind*}
            hi[m] = ${pressure_general_hi} {id=hi, dep=ind*}
        end
        <>logk1 = ${loweq} {id=a1, dep=lo}
        <>logk2 = ${hieq} {id=a2, dep=hi}
        <>kf_temp = logk1 {id=a_oor}
        if not oor
            kf_temp = ${plog_eqn} {id=a_found, dep=a1:a2}
        end
        ${kf_str} = exp(kf_temp) {id=kf, dep=a_oor:a_found}
""").safe_substitute(loweq=k1_eq, hieq=k2_eq, plog_eqn=plog_eqn,
                     kf_str=kf_str,
                     logP=logP,
                     plog_num_param_str=plog_num_param_str,
                     pressure_lo=pressure_lo,
                     pressure_hi=pressure_hi,
                     pressure_mid_lo=pressure_mid_lo,
                     pressure_mid_hi=pressure_mid_hi,
                     pressure_general_lo=pressure_general_lo,
                     pressure_general_hi=pressure_general_hi
                     )

    # and return
    return [k_gen.knl_info(name='rateconst_plog',
                           instructions=instructions,
                           pre_instructions=[
                               ic.default_pre_instructs(Tinv, T_str, 'INV'),
                               ic.default_pre_instructs(logT, T_str, 'LOG'),
                               ic.default_pre_instructs(logP, P_str, 'LOG')],
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           extra_inames=extra_inames)]


def get_reduced_pressure_kernel(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the reduced
    pressure evaluation kernel

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

    conp_eqs = eqs['conp']  # conp / conv irrelevant for rates

    # create the mapper
    mapstore = arc.MapStore(loopy_opts, namestore.fall_map,
                            namestore.fall_mask)

    # create the various necessary arrays

    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

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

    # create Pri eqn
    Pri_sym = next(x for x in conp_eqs if str(x) == 'P_{r, i}')
    # make substituions to get a usable form
    pres_mod_sym = sp.Symbol(thd_conc_str)
    Pri_eqn = sp_utils.sanitize(conp_eqs[Pri_sym][(thd_body_type.mix,)],
                                subs={'[X]_i': pres_mod_sym,
                                      'k_{0, i}': 'k0',
                                      'k_{infty, i}': 'kinf'}
                                )

    # create instruction set
    pr_instructions = Template("""
if ${fall_type_str}
    # chemically activated
    <>k0 = ${kf_str} {id=k0_c}
    <>kinf = ${kf_fall_str} {id=kinf_c}
else
    # fall-off
    kinf = ${kf_str} {id=kinf_f}
    k0 = ${kf_fall_str} {id=k0_f}
end
${Pr_str} = ${Pr_eq} {dep=k*}
""")

    # sub in strings
    pr_instructions = pr_instructions.safe_substitute(
        fall_type_str=fall_type_str,
        kf_str=kf_str,
        kf_fall_str=kf_fall_str,
        Pr_str=Pr_str,
        Pr_eq=Pri_eqn)

    # and finally return the resulting info
    return [k_gen.knl_info('red_pres',
                           instructions=pr_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore)]


def get_troe_kernel(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the Troe
    falloff evaluation kernel

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

    # set of equations is irrelevant for non-derivatives
    conp_eqs = eqs['conp']

    # rate info and reac ind
    kernel_data = []

    # create mapper
    mapstore = arc.MapStore(loopy_opts,
                            namestore.troe_map, namestore.troe_mask)

    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # add maps / masks
    mapstore.check_and_add_transform(namestore.Fcent, namestore.num_troe)
    mapstore.check_and_add_transform(namestore.Atroe, namestore.num_troe)
    mapstore.check_and_add_transform(namestore.Btroe, namestore.num_troe)

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

    # find the falloff form equations
    Fi_sym = next(x for x in conp_eqs if str(x) == 'F_{i}')
    keys = conp_eqs[Fi_sym]
    Fi = {}
    for key in keys:
        fall_form = next(x for x in key if isinstance(x, falloff_form))
        Fi[fall_form] = conp_eqs[Fi_sym][key]

    # get troe syms / eqs
    Fcent = next(x for x in conp_eqs if str(x) == 'F_{cent}')
    Atroe = next(x for x in conp_eqs if str(x) == 'A_{Troe}')
    Btroe = next(x for x in conp_eqs if str(x) == 'B_{Troe}')
    Fcent_eq, Atroe_eq, Btroe_eq = conp_eqs[
        Fcent], conp_eqs[Atroe], conp_eqs[Btroe]

    # get troe params and create arrays
    troe_a_lp, troe_a_str = mapstore.apply_maps(namestore.troe_a, var_name)
    troe_T3_lp, troe_T3_str = mapstore.apply_maps(namestore.troe_T3, var_name)
    troe_T1_lp, troe_T1_str = mapstore.apply_maps(namestore.troe_T1, var_name)
    troe_T2_lp, troe_T2_str = mapstore.apply_maps(namestore.troe_T2, var_name)
    # update the kernel_data
    kernel_data.extend([troe_a_lp, troe_T3_lp, troe_T1_lp, troe_T2_lp])
    # sub into eqs
    Fcent_eq = sp_utils.sanitize(Fcent_eq, subs={
        'a': troe_a_str,
        'T^{*}': troe_T1_str,
        'T^{***}': troe_T3_str,
        'T^{**}': troe_T2_str,
    })

    # now separate into optional / base parts
    Fcent_base_eq = sp.Add(
        *[x for x in sp.Add.make_args(Fcent_eq)
          if not sp.Symbol(troe_T2_str) in x.free_symbols])
    Fcent_opt_eq = Fcent_eq - Fcent_base_eq

    # develop the Atroe / Btroe eqs
    Atroe_eq = sp_utils.sanitize(Atroe_eq, subs=OrderedDict([
        ('F_{cent}', Fcent_str),
        ('P_{r, i}', Pr_str),
        (sp.log(sp.Symbol(Pr_str), 10), 'logPr'),
        (sp.log(sp.Symbol(Fcent_str), 10), sp.Symbol('logFcent'))
    ]))

    Btroe_eq = sp_utils.sanitize(Btroe_eq, subs=OrderedDict([
        ('F_{cent}', Fcent_str),
        ('P_{r, i}', Pr_str),
        (sp.log(sp.Symbol(Pr_str), 10), 'logPr'),
        (sp.log(sp.Symbol(Fcent_str), 10), sp.Symbol('logFcent'))
    ]))

    Fcent_temp_str = 'Fcent_temp'
    # finally, work on the Fi form
    Fi_eq = sp_utils.sanitize(Fi[falloff_form.troe], subs=OrderedDict([
        ('F_{cent}', Fcent_temp_str),
        ('A_{Troe}', Atroe_str),
        ('B_{Troe}', Btroe_str)
    ]))

    # separate into Fcent and power
    Fi_base_eq = next(x for x in Fi_eq.args if str(x) == Fcent_temp_str)
    Fi_pow_eq = next(x for x in Fi_eq.args if str(x) != Fcent_temp_str)
    Fi_pow_eq = sp_utils.sanitize(Fi_pow_eq, subs=OrderedDict([
        (sp.Pow(sp.Symbol(Atroe_str), 2), sp.Symbol('Atroe_squared')),
        (sp.Pow(sp.Symbol(Btroe_str), 2), sp.Symbol('Btroe_squared'))
    ]))

    # make the instructions
    troe_instructions = Template("""
    <>T = ${T_str}
    <>${Fcent_temp} = ${Fcent_base_eq} {id=Fcent_decl} # this must be a temporary to avoid a race on future assignments
    if ${troe_T2_str} != 0
        ${Fcent_temp} = ${Fcent_temp} + ${Fcent_opt_eq} {id=Fcent_decl2, dep=Fcent_decl}
    end
    ${Fcent_str} = ${Fcent_temp} {id=Fcent_decl3, dep=Fcent_decl2}
    <> Fcent_val = fmax(1e-300d, ${Fcent_temp}) {id=Fcv, dep=Fcent_decl3}
    <>Pr_val = fmax(1e-300d, ${Pr_str}) {id=Prv}
    <>logFcent = log10(Fcent_val) {dep=Fcv}
    <>logPr = log10(Pr_val) {dep=Prv}
    <>Atroe_temp = ${Atroe_eq} {dep=Fcent_decl*}
    <>Btroe_temp = ${Btroe_eq} {dep=Fcent_decl*}
    ${Atroe_str} = Atroe_temp # this must be a temporary to avoid a race on future assignments
    ${Btroe_str} = Btroe_temp # this must be a temporary to avoid a race on future assignments
    <>Atroe_squared = Atroe_temp * Atroe_temp
    <>Btroe_squared = Btroe_temp * Btroe_temp
    ${Fi_str} = ${Fi_base_eq}**(${Fi_pow_eq}) {dep=Fcent_decl*}
    """).safe_substitute(T_str=T_str,
                         Fcent_temp=Fcent_temp_str,
                         Fcent_str=Fcent_str,
                         Fcent_base_eq=Fcent_base_eq,
                         Fcent_opt_eq=Fcent_opt_eq,
                         troe_T2_str=troe_T2_str,
                         Pr_str=Pr_str,
                         Atroe_eq=Atroe_eq,
                         Btroe_eq=Btroe_eq,
                         Atroe_str=Atroe_str,
                         Btroe_str=Btroe_str,
                         Fi_str=Fi_str,
                         Fi_base_eq=Fi_base_eq,
                         Fi_pow_eq=Fi_pow_eq)

    return [k_gen.knl_info('fall_troe',
                           instructions=troe_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           manglers=[lp_pregen.fmax()])]


def get_sri_kernel(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the SRI
    falloff evaluation kernel

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

    # set of equations is irrelevant for non-derivatives
    conp_eqs = eqs['conp']
    kernel_data = []

    # create mapper
    mapstore = arc.MapStore(loopy_opts, namestore.sri_map, namestore.sri_mask)

    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # maps and transforms
    for arr in [namestore.X_sri, namestore.sri_a, namestore.sri_b,
                namestore.sri_c, namestore.sri_d, namestore.sri_e]:
        mapstore.check_and_add_transform(arr, namestore.num_sri)

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

    # start creating SRI kernel
    Fi_sym = next(x for x in conp_eqs if str(x) == 'F_{i}')
    Fi = next(val for key, val in conp_eqs[Fi_sym].items()
              if falloff_form.sri in key)

    # find Pr symbol
    Pri_sym = next(x for x in conp_eqs if str(x) == 'P_{r, i}')

    # get SRI symbols
    X_sri_sym = next(x for x in Fi.free_symbols if str(x) == 'X')

    # create SRI eqs
    X_sri_eq = conp_eqs[X_sri_sym].subs(
        sp.Pow(sp.log(Pri_sym, 10), 2), 'logPr * logPr')
    Fi_sri_eq = Fi.copy()
    Fi_sri_eq = sp_utils.sanitize(Fi_sri_eq,
                                  subs={
                                      'a': sri_a_str,
                                      'b': sri_b_str,
                                      'c': sri_c_str,
                                      'd': sri_d_str,
                                      'e': sri_e_str,
                                      'X': 'X_temp'
                                  })
    # do some surgery on the Fi_sri_eq to get the optional parts
    Fi_sri_base = next(x for x in sp.Mul.make_args(Fi_sri_eq)
                       if any(str(y) == sri_a_str for y in x.free_symbols))
    Fi_sri_opt = Fi_sri_eq / Fi_sri_base
    Fi_sri_d_opt = next(x for x in sp.Mul.make_args(Fi_sri_opt)
                        if any(str(y) == sri_d_str for y in x.free_symbols))
    Fi_sri_e_opt = next(x for x in sp.Mul.make_args(Fi_sri_opt)
                        if any(str(y) == sri_e_str for y in x.free_symbols))

    # create instruction set
    sri_instructions = Template("""
    <>Pr_val = fmax(1e-300d, ${pr_str}) {id=Pri}
    <>logPr = log10(Pr_val) {dep=Pri}
    <>X_temp = ${Xeq} {id=X_decl} # this must be a temporary to avoid a race on Fi_temp assignment
    <>Fi_temp = ${Fi_sri} {id=Fi_decl, dep=X_decl}
    if ${d_str} != 1.0
        Fi_temp = Fi_temp * ${d_eval} {id=Fi_decl1, dep=Fi_decl}
    end
    if ${e_str} != 0.0
        Fi_temp = Fi_temp * ${e_eval} {id=Fi_decl2, dep=Fi_decl}
    end
    ${Fi_str} = Fi_temp {dep=Fi_decl*}
    ${X_str} = X_temp
    """).safe_substitute(T_str=T_str,
                         pr_str=Pr_str,
                         X_str=X_sri_str,
                         Xeq=X_sri_eq,
                         Fi_sri=Fi_sri_base,
                         d_str=sri_d_str,
                         d_eval=Fi_sri_d_opt,
                         e_str=sri_e_str,
                         e_eval=Fi_sri_e_opt,
                         Fi_str=Fi_str)

    return [k_gen.knl_info('fall_sri',
                           instructions=sri_instructions,
                           pre_instructions=[
                               ic.default_pre_instructs('T', T_str, 'VAL')],
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore,
                           manglers=[lp_pregen.fmax()])]


def get_lind_kernel(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for the Lindeman
    falloff evaluation kernel

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

    # set of equations is irrelevant for non-derivatives

    kernel_data = []

    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # create Fi array / str

    mapstore = arc.MapStore(loopy_opts,
                            namestore.lind_map, namestore.lind_mask)

    Fi_lp, Fi_str = mapstore.apply_maps(namestore.Fi, *default_inds)
    kernel_data.append(Fi_lp)

    # create instruction set
    lind_instructions = Template("""
    ${Fi_str} = 1
    """).safe_substitute(Fi_str=Fi_str)

    return [k_gen.knl_info('fall_lind',
                           instructions=lind_instructions,
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore)]


def get_simple_arrhenius_rates(eqs, loopy_opts, namestore, test_size=None,
                               falloff=False):
    """Generates instructions, kernel arguements, and data for specialized forms
    of simple (non-pressure dependent) rate constants

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
        mapstore = arc.MapStore(loopy_opts, namestore.fall_map,
                                namestore.fall_mask)
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
        mapstore = arc.MapStore(loopy_opts, namestore.simple_map,
                                namestore.simple_mask)
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
    full = loopy_opts.rate_spec == lp_utils.RateSpecialization.full
    hybrid = loopy_opts.rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = loopy_opts.rate_spec == lp_utils.RateSpecialization.fixed
    separated_kernels = loopy_opts.rate_spec_kernels
    if fixed and separated_kernels:
        separated_kernels = False
        logging.warn('Cannot use separated kernels with a fixed '
                     'RateSpecialization, disabling...')

    base_kernel_data = []
    if test_size == 'problem_size':
        base_kernel_data.append(namestore.problem_size)

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

    # put rateconst info args in dict for unpacking convenience
    extra_args = {'kernel_data': base_kernel_data,
                  'var_name': var_name}

    default_preinstructs = {'Tinv':
                            ic.default_pre_instructs('Tinv', T_str, 'INV'),
                            'logT':
                            ic.default_pre_instructs('logT', T_str, 'LOG'),
                            'Tval':
                            ic.default_pre_instructs('Tval', T_str, 'VAL')}

    # generic kf assigment str
    kf_assign = Template("${kf_str} = ${rate}")
    expkf_assign = Template("${kf_str} = exp(${rate})")

    def get_instructions(rtype, mapper, kernel_data, beta_iter=1,
                         single_kernel_rtype=None):
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
        rate_eqn_pre = get_rate_eqn(eqs)
        rate_eqn_pre = sp_utils.sanitize(rate_eqn_pre,
                                         symlist={
                                             'A[i]': A_str,
                                             'Ta[i]': Ta_str,
                                             'beta[i]': b_str,
                                         })

        manglers = []
        # the simple formulation
        if fixed or (hybrid and rtype == 2) or (full and rtype == 4):
            retv = expkf_assign.safe_substitute(rate=str(rate_eqn_pre))
        # otherwise check type and return appropriate instructions with
        # array strings substituted in
        elif rtype == 0:
            retv = kf_assign.safe_substitute(rate=A_str)
        elif rtype == 1:
            if beta_iter > 1:
                beta_iter_str = Template("""
                <> b_end = abs(${b_str})
                <> ${kf_str} = kf_temp * pown(T_iter, b_end) {id=a4, dep=a3:a2:a1}
                ${kf_str} = kf_temp {dep=a4}
                """).safe_substitute(b_str=b_str)
                manglers.append(k_gen.MangleGen('pown',
                                                (np.float64, np.int32),
                                                np.float64))
            else:
                beta_iter_str = ("${kf_str} = kf_temp * T_iter"
                                 " {id=a4, dep=a3:a2:a1}")
            retv = Template(
                """
                <> T_iter = Tval {id=a1}
                if ${b_str} < 0
                    T_iter = Tinv {id=a2, dep=a1}
                end
                <>kf_temp = ${A_str} {id=a3}
                ${beta_iter}
                """).safe_substitute(A_str=A_str,
                                     b_str=b_str,
                                     beta_iter=beta_iter_str)
        elif rtype == 2:
            retv = expkf_assign.safe_substitute(
                rate=str(rate_eqn_pre.subs(Ta_str, 0)))
        elif rtype == 3:
            retv = expkf_assign.safe_substitute(
                rate=str(rate_eqn_pre.subs(b_str, 0)))

        return Template(retv).safe_substitute(kf_str=kf_str), manglers

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
                                    default_preinstructs['Tval'],
                                    default_preinstructs['Tinv']],
                                **extra_args)
    i_beta_exp = k_gen.knl_info('beta_exp_{}'.format(tag),
                                instructions='',
                                mapstore=mapstore,
                                pre_instructions=[
                                    default_preinstructs['Tinv'],
                                    default_preinstructs['logT']],
                                **extra_args)
    i_ta_exp = k_gen.knl_info('ta_exp_{}'.format(tag),
                              instructions='',
                              mapstore=mapstore,
                              pre_instructions=[
        default_preinstructs['Tinv'],
        default_preinstructs['logT']],
        **extra_args)
    i_full = k_gen.knl_info('rateconst_full{}'.format(tag),
                            instructions='',
                            mapstore=mapstore,
                            pre_instructions=[
        default_preinstructs['Tinv'],
        default_preinstructs['logT']],
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
        if len(specializations) > 1:
            instruction_list = []
            func_manglers = []
            for i in specializations:
                instruction_list.append(
                    'if {1} == {0}'.format(i, rtype_str))
                insns, manglers = get_instructions(
                    -1,
                    arc.MapStore(loopy_opts, mapstore.map_domain,
                                 mapstore.mask_domain),
                    specializations[i].kernel_data,
                    beta_iter,
                    single_kernel_rtype=i)
                instruction_list.extend([
                    '\t' + x for x in insns.split('\n') if x.strip()])
                instruction_list.append('end')
                if manglers:
                    func_manglers.extend(manglers)
        # and combine them
        specializations = {-1: k_gen.knl_info(
                           'rateconst_singlekernel_{}'.format(tag),
                           instructions='\n'.join(instruction_list),
                           pre_instructions=list(
                               default_preinstructs.values()),
                           mapstore=mapstore,
                           kernel_data=specializations[0].kernel_data,
                           var_name=var_name,
                           manglers=func_manglers)}

    out_specs = {}
    # and do some finalizations for the specializations
    for rtype, info in specializations.items():
        # this is handled above
        if rtype < 0:
            out_specs[rtype] = info
            continue

        domain, _, num = rdomain(rtype)
        if domain is None or not domain.initializer.size:
            # kernel doesn't act on anything, don't add it to output
            continue

        # next create a mapper for this rtype
        mapper = arc.MapStore(loopy_opts, domain, domain)

        # set as mapper
        info.mapstore = mapper

        # if a specific rtype, get the instructions here
        if rtype >= 0:
            info.instructions, info.manglers = get_instructions(
                rtype, mapper, info.kernel_data, beta_iter)

        out_specs[rtype] = info

    return list(out_specs.values())


def write_specrates_kernel(eqs, reacs, specs,
                           loopy_opts, conp=True,
                           test_size=None, auto_diff=False,
                           output_full_rop=False):
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
    output_full_rop : bool
        If ``True``, output forward and reversse rates of progress
        Useful in testing, as there are serious floating point errors for
        net production rates near equilibrium, invalidating direct comparison to Cantera

    Returns
    -------
    kernel_gen : :class:`kernel_generator`
        The generator responsible for creating the resulting code

    """

    # figure out rates and info
    rate_info = assign_rates(reacs, specs, loopy_opts.rate_spec)

    # set test size
    if test_size is None:
        test_size = 'problem_size'

    # create the namestore
    nstore = arc.NameStore(loopy_opts, rate_info, conp, test_size)

    kernels = []

    def __add_knl(knls, klist=None):
        if klist is None:
            klist = kernels
        try:
            klist.extend(knls)
        except:
            klist.append(knls)

    # Note:
    # the order in which these kernels get added is important
    # the kernel generator uses the input order to generate the wrapping
    # kernel calls
    # hence, any data dependencies should be expressed in the order added here

    # get the simple arrhenius k_gen.knl_info's
    __add_knl(get_simple_arrhenius_rates(eqs, loopy_opts,
                                         nstore, test_size=test_size))

    # check for plog
    if rate_info['plog']['num']:
        # generate the plog kernel
        __add_knl(get_plog_arrhenius_rates(eqs, loopy_opts,
                                           nstore, rate_info['plog']['max_P'],
                                           test_size=test_size))

    # check for chebyshev
    if rate_info['cheb']['num']:
        __add_knl(get_cheb_arrhenius_rates(eqs,
                                           loopy_opts,
                                           nstore,
                                           np.max(rate_info['cheb']['num_P']),
                                           np.max(rate_info['cheb']['num_T']),
                                           test_size=test_size))

    # check for third body terms
    if rate_info['thd']['num']:
        # add the initial third body conc eval kernel
        __add_knl(get_thd_body_concs(eqs, loopy_opts,
                                     nstore, test_size))

    # check for falloff
    if rate_info['fall']['num']:
        # get the falloff rates
        __add_knl(get_simple_arrhenius_rates(eqs, loopy_opts,
                                             nstore, test_size=test_size,
                                             falloff=True))
        # and the reduced pressure
        __add_knl(get_reduced_pressure_kernel(eqs, loopy_opts,
                                              nstore, test_size=test_size))
        # and finally any blending functions (depend on reduced pressure)
        if rate_info['fall']['lind']['num']:
            __add_knl(get_lind_kernel(eqs, loopy_opts,
                                      nstore, test_size=test_size))
        if rate_info['fall']['troe']['num']:
            __add_knl(get_troe_kernel(eqs, loopy_opts,
                                      nstore, test_size=test_size))
        if rate_info['fall']['sri']['num']:
            __add_knl(get_sri_kernel(eqs, loopy_opts,
                                     nstore, test_size=test_size))

    # thermo polynomial dimension
    depends_on = []
    # check for reverse rates
    if rate_info['rev']['num']:
        # add the 'b' eval
        __add_knl(polyfit_kernel_gen('b', eqs['conp'], loopy_opts,
                                     nstore, test_size))
        # addd the 'b' eval to depnediencies
        depends_on.append(kernels[-1])
        # add Kc / rev rates
        __add_knl(get_rev_rates(eqs, loopy_opts,
                                nstore,
                                allint={'fwd': rate_info['fwd']['allint'],
                                        'rev': rate_info['rev']['allint'],
                                        'net': rate_info['net']['allint']},
                                test_size=test_size))

    # check for falloff
    if rate_info['fall']['num']:
        # and the Pr evals
        __add_knl(get_rxn_pres_mod(eqs, loopy_opts,
                                   nstore, test_size))

    # add ROP
    __add_knl(get_rop(eqs, loopy_opts,
                      nstore, allint={'fwd': rate_info['fwd']['allint'],
                                      'rev': rate_info['rev']['allint'],
                                      'net': rate_info['net']['allint']},
                      test_size=test_size))
    # add ROP net
    __add_knl(get_rop_net(eqs, loopy_opts,
                          nstore, test_size))
    # add spec rates
    __add_knl(get_spec_rates(eqs, loopy_opts,
                             nstore, test_size))

    if conp:
        # get h / cp evals
        __add_knl(polyfit_kernel_gen('h', eqs['conp'], loopy_opts, nstore,
                                     test_size))
        __add_knl(polyfit_kernel_gen('cp', eqs['conp'], loopy_opts, nstore,
                                     test_size))
    else:
        # and u / cv
        __add_knl(polyfit_kernel_gen('u', eqs['conv'], loopy_opts, nstore,
                                     test_size))
        __add_knl(polyfit_kernel_gen('cv', eqs['conv'], loopy_opts, nstore,
                                     test_size))
    # add the thermo kernels to our dependencies
    depends_on.extend(kernels[-2:])
    # and temperature rates
    __add_knl(get_temperature_rate(eqs, loopy_opts,
                                   nstore, test_size=test_size, conp=conp))

    # get a wrapper for the dependecies
    thermo_wrap = k_gen.make_kernel_generator(name='chem_utils_kernel',
                                              loopy_opts=loopy_opts,
                                              kernels=depends_on,
                                              input_arrays=['T_arr'],
                                              output_arrays=[
                                                  'h', 'cp'] if conp else
                                              ['u', 'cv'],
                                              auto_diff=auto_diff,
                                              test_size=test_size
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
        # first, find reduced pressure
        __insert_at('red_pres', True)
        # barrier before fall_troe for Pr
        __insert_at('fall_troe', True)
        # barrier before the falloff ci's for the Fi's
        __insert_at('ci_fall', True)
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
        __insert_at('spec_rates', True)
        __insert_at('temperature_rate', True)

    input_arrays = ['phi', 'P_arr', 'dphi']
    output_arrays = ['dphi']
    if output_full_rop:
        output_arrays += ['rop_fwd']
        if rate_info['rev']['num']:
            output_arrays += ['rop_rev']
        if rate_info['thd']['num']:
            output_arrays += ['pres_mod']
        output_arrays += ['rop_net']
    return k_gen.make_kernel_generator(
        loopy_opts=loopy_opts,
        name='species_rates_kernel',
        kernels=kernels,
        external_kernels=depends_on,
        depends_on=[thermo_wrap],
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        init_arrays={'dphi': 0},
        auto_diff=auto_diff,
        test_size=test_size,
        barriers=barriers)


def get_rate_eqn(eqs, index='i'):
    """Helper routine that returns the Arrenhius rate constant in exponential
    form.

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    index : str
        The index to generate the equations for, 'i' by default
    Returns
    -------
    rate_eqn_pre : `sympy.Expr`
        The rate constant before taking the exponential (sympy does odd things upon doing so)
        This is used for various simplifications

    """

    conp_eqs = eqs['conp']

    # define some dummy symbols for loopy writing
    E_sym = sp.Symbol('Ta[{ind}]'.format(ind=index))
    A_sym = sp.Symbol('A[{ind}]'.format(ind=index))
    T_sym = sp.Symbol('T')
    b_sym = sp.Symbol('beta[{ind}]'.format(ind=index))
    symlist = {'Ta[i]': E_sym,
               'A[i]': A_sym,
               'T': T_sym,
               'beta[i]': b_sym}
    Tinv_sym = sp.Symbol('Tinv')
    logA_sym = sp.Symbol('A[{ind}]'.format(ind=index))
    logT_sym = sp.Symbol('logT')

    # the rate constant is indep. of conp/conv, so just use conp for simplicity
    kf_eqs = [x for x in conp_eqs if str(x) == '{k_f}[i]']

    # do some surgery on the equations
    kf_eqs = {key: (x, conp_eqs[x][key])
              for x in kf_eqs for key in conp_eqs[x]}

    # first load the arrenhius rate equation
    rate_eqn = next(kf_eqs[x]
                    for x in kf_eqs if reaction_type.elementary in x)[1]
    rate_eqn = sp_utils.sanitize(rate_eqn,
                                 symlist=symlist,
                                 subs={sp.Symbol('{E_{a}}[i]') / (sp.Symbol('R_u') * T_sym):
                                       E_sym * Tinv_sym})

    # finally, alter to exponential form:
    rate_eqn_pre = sp.log(A_sym) + sp.log(T_sym) * b_sym - E_sym * Tinv_sym
    rate_eqn_pre = rate_eqn_pre.subs([(sp.log(A_sym), logA_sym),
                                      (sp.log(T_sym), logT_sym)])

    return rate_eqn_pre


def polyfit_kernel_gen(nicename, eqs, loopy_opts, namestore, test_size=None):
    """Helper function that generates kernels for
       evaluation of various thermodynamic species properties

    Parameters
    ----------
    nicename : str
        The variable name to use in generated code
    eqs : dict of `sympy.Symbol`
        Dictionary defining conditional equations for the variables (keys)
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

    param_ind = 'dummy'
    loop_index = 'k'
    # create mapper
    mapstore = arc.MapStore(loopy_opts,
                            namestore.num_specs,
                            namestore.num_specs,
                            loop_index)

    knl_data = []
    if test_size == 'problem_size':
        knl_data.append(namestore.problem_size)

    if loopy_opts.width is not None and loopy_opts.depth is not None:
        raise Exception('Cannot specify both SIMD/SIMT width and depth')

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

    # mapping of nicename -> varname
    var_maps = {'cp': '{C_p}[k]',
                'dcp': 'frac{text{d} {C_p} }{text{d} T }[k]',
                'h': 'H[k]',
                'cv': '{C_v}[k]',
                'dcv': 'frac{text{d} {C_v} }{text{d} T }[k]',
                'u': 'U[k]',
                'b': 'B[k]',
                'db': 'frac{text{d} B }{text{d} T }[k]'}
    varname = var_maps[nicename]

    # get variable and equation
    var = next(v for v in eqs.keys() if str(v) == varname)
    eq = eqs[var]

    # create string indexes for a_lo/a_hi
    a_lo_strs = [mapstore.apply_maps(namestore.a_lo, loop_index, str(i))[1]
                 for i in range(poly_dim)]
    a_hi_strs = [mapstore.apply_maps(namestore.a_hi, loop_index, str(i))[1]
                 for i in range(poly_dim)]
    # use to create lo / hi equation
    from ..sympy_utils import sympy_addons as sp_add
    from collections import defaultdict
    a_list = defaultdict(
        str,
        [(x.args[-1], x) for x in eq.free_symbols
         if isinstance(x, sp_add.MyIndexed)])

    lo_eq_str = str(eq.subs([(a_list[i], a_lo_strs[i])
                             for i in range(poly_dim)]))
    hi_eq_str = str(eq.subs([(a_list[i], a_hi_strs[i])
                             for i in range(poly_dim)]))

    T_val = 'T'
    preinstructs = [ic.default_pre_instructs(T_val, T_str, 'VAL')]

    return k_gen.knl_info(instructions=Template("""
        for k
            if ${T_val} < ${T_mid_str}
                ${out_str} = ${lo_eq}
            else
                ${out_str} = ${hi_eq}
            end
        end
        """).safe_substitute(
        out_str=out_str,
        lo_eq=lo_eq_str,
        hi_eq=hi_eq_str,
        T_mid_str=T_mid_str,
        T_val=T_val),
        kernel_data=knl_data,
        pre_instructions=preinstructs,
        name='eval_{}'.format(nicename),
        parameters={'R_u': chem.RU},
        var_name=loop_index,
        mapstore=mapstore)


def write_chem_utils(specs, eqs, loopy_opts,
                     test_size=None, auto_diff=False):
    """Write subroutine to evaluate species thermodynamic properties.

    Notes
    -----
    Thermodynamic properties include:  enthalpy, energy, specific heat
    (constant pressure and volume).

    Parameters
    ----------
    specs : list of `SpecInfo`
        List of species in the mechanism.
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    test_size : int
        If not None, this kernel is being used for testing.
    auto_diff : bool
        If ``True``, generate files for Adept autodifferention library.

    Returns
    -------
    global_defines : list of :class:`loopy.TemporaryVariable`
        The global variables for this kernel that need definition in the memory manager

    """

    if test_size is None:
        test_size = 'problem_size'

    # generate the kernels
    conp_eqs = eqs['conp']
    conv_eqs = eqs['conv']

    nicenames = ['cp', 'h', 'cv', 'u', 'b']
    kernels = []
    for nicename in nicenames:
        eq = conp_eqs if nicename in ['h', 'cp'] else conv_eqs
        kernels.append(polyfit_kernel_gen(nicename,
                                          eq, specs, loopy_opts, test_size))

    return k_gen.make_kernel_generator(
        loopy_opts=loopy_opts,
        name='chem_utils',
        kernels=kernels,
        input_arrays=['phi'],
        output_arrays=nicenames,
        auto_diff=auto_diff,
        test_size=test_size
    )


if __name__ == "__main__":
    from . import create_jacobian
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
                    skip_jac=True,
                    last_spec=args.last_species,
                    auto_diff=args.auto_diff
                    )
