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

    Returns
    -------
    rate_info : dict of parameters
        Keys are 'simple', 'plog', 'cheb', 'fall', 'chem', 'thd'
        Values are further dictionaries including addtional rate info, number,
        offset, maps, etc.

    Notes
    -----
        Note that the reactions in 'fall', 'chem' and 'thd' are also in
            'simple'
        Further, there are duplicates between 'thd' and 'fall' / 'chem'
    """

    # determine specialization
    full = rate_spec == lp_utils.RateSpecialization.full
    hybrid = rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = rate_spec == lp_utils.RateSpecialization.fixed

    # find fwd / reverse rate parameters
    # first, the number of each
    rev_map = np.array([i for i, x in enumerate(reacs) if x.rev],
                       dtype=np.int32)
    num_rev = len(rev_map)
    # next, find the species / nu values
    fwd_spec = []
    fwd_num_spec = []
    fwd_nu = []
    rev_spec = []
    rev_num_spec = []
    rev_nu = []
    nu_sum = []
    net_num_spec = []
    net_nu = []
    net_spec = []
    fwd_allnu_integer = True
    rev_allnu_integer = True
    for rxn in reacs:
        # fwd
        fwd_spec.extend(rxn.reac[:])
        fwd_num_spec.append(len(rxn.reac))
        fwd_nu.extend(rxn.reac_nu[:])
        if rxn.rev:
            # and rev
            rev_spec.extend(rxn.prod[:])
            rev_num_spec.append(len(rxn.prod))
            rev_nu.extend(rxn.prod_nu[:])
        # finally, net values
        spec = list(sorted(set(rxn.reac + rxn.prod)))
        net_spec.extend(spec)
        net_num_spec.append(len(spec))
        nu = [utils.get_nu(isp, rxn) for isp in spec]
        net_nu.extend(nu)
        # and nu sum for equilibrium constants
        nu_sum.append(sum(nu))

    # create numpy versions
    fwd_spec = np.array(fwd_spec, dtype=np.int32)
    fwd_num_spec = np.array(fwd_num_spec, dtype=np.int32)
    if any(not utils.is_integer(nu) for nu in fwd_nu):
        fwd_nu = np.array(fwd_nu)
        fwd_allnu_integer = False
    else:
        fwd_nu = np.array(fwd_nu, dtype=np.int32)
    rev_spec = np.array(rev_spec, dtype=np.int32)
    rev_num_spec = np.array(rev_num_spec, dtype=np.int32)
    if any(not utils.is_integer(nu) for nu in rev_nu):
        rev_nu = np.array(rev_nu)
        fwd_allnu_integer = False
    else:
        rev_nu = np.array(rev_nu, dtype=np.int32)

    net_nu_integer = all(utils.is_integer(nu) for nu in net_nu)
    if net_nu_integer:
        nu_sum = np.array(nu_sum, dtype=np.int32)
        net_nu = np.array(net_nu, dtype=np.int32)
    else:
        nu_sum = np.array(nu_sum)
        net_nu = np.array(net_nu)
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
        mapping = []
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
                    fall_types[i] = 1  # mark as chemically activated
                else:
                    # we want k0, hence default factor is fine
                    Ai, bi, Tai = reac.low
                    fall_types[i] = 0  # mark as falloff
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
    troe_a, troe_T3, troe_T1, troe_T2 = [
        np.array(x, dtype=np.float64) for x in zip(*troe_par)]

    # find third-body types
    thd_reacs, thd_map, num_thd = __seperate(
        reacs, [reaction_type.fall, reaction_type.chem, reaction_type.thd])
    # find third body type
    thd_type = np.array([next(int(y) for y in x.type if isinstance(
        y, thd_body_type)) for x in thd_reacs], dtype=np.int32)
    # find the species indicies
    thd_spec_num = []
    thd_spec = []
    thd_eff = []
    for x in thd_reacs:
        if x.match(thd_body_type.species):
            thd_spec_num.append(1)
            thd_spec.append(x.pdep_sp)
            thd_eff.append(1)
        elif x.match(thd_body_type.unity):
            thd_spec_num.append(0)
        else:
            thd_spec_num.append(len(x.thd_body_eff))
            spec, eff = zip(*x.thd_body_eff)
            thd_spec.extend(spec)
            thd_eff.extend(eff)
    thd_spec_num = np.array(thd_spec_num, dtype=np.int32)
    thd_spec = np.array(thd_spec, dtype=np.int32)
    thd_eff = np.array(thd_eff, dtype=np.float64)

    # post processing

    # first, we must do some surgery to get _our_ form of the thd-body
    # efficiencies
    last_spec = len(specs) - 1
    pp_thd_eff_ns = np.ones(num_thd)
    pp_thd_num_specs = thd_spec_num.copy()
    pp_thd_spec = thd_spec.copy()
    pp_thd_eff = thd_eff.copy()
    # go through the efficiency maps
    for i in range(pp_thd_num_specs.size):
        # get number of thd body species and offset
        num = pp_thd_num_specs[i]
        offset = np.sum(pp_thd_num_specs[:i])
        # check if Ns has a non-default efficiency
        if last_spec in pp_thd_spec[offset:offset+num]:
            ind = np.where(pp_thd_spec[offset:offset+num] == last_spec)[0][0]
            # set the efficiency
            pp_thd_eff_ns[i] = pp_thd_eff[offset + ind]
            # delete from the species list
            pp_thd_spec = np.delete(pp_thd_spec, offset + ind)
            # delete from effiency list
            pp_thd_eff = np.delete(pp_thd_eff, offset + ind)
            # subtract from species num
            pp_thd_num_specs[i] -= 1
        # and subtract from efficiencies
        pp_thd_eff[offset:offset+pp_thd_num_specs[i]] -= pp_thd_eff_ns[i]
        if pp_thd_eff_ns[i] != 1:
            # we need to add all the other species :(
            # get updated species list
            to_add = np.array(range(len(specs) - 1), dtype=np.int32)
            # set default efficiency
            eff = np.full((len(specs) - 1,), 1 - pp_thd_eff_ns[i])
            # fill in existing efficiencies
            eff[pp_thd_spec[offset:offset+pp_thd_num_specs[i]]] = \
                pp_thd_eff[offset:offset+pp_thd_num_specs[i]]
            # delete from the species list / efficiencies
            pp_thd_spec = np.delete(pp_thd_spec,
                                    range(offset, offset+pp_thd_num_specs[i]))
            pp_thd_eff = np.delete(pp_thd_eff,
                                   range(offset, offset+pp_thd_num_specs[i]))
            # insert new lists
            pp_thd_spec = np.insert(pp_thd_spec, offset, to_add)
            pp_thd_eff = np.insert(pp_thd_eff, offset, eff)
            # and update number of species
            pp_thd_num_specs[i] = to_add.size

    # chebyshev parameter reordering
    pp_cheb_coeff = np.zeros((num_cheb, int(np.max(cheb_n_temp)),
                              int(np.max(cheb_n_pres))))
    for i, p in enumerate(cheb_coeff):
        pp_cheb_coeff[i, :cheb_n_temp[i], :cheb_n_pres[i]] = p[:, :]

    # limits for cheby polys
    pp_cheb_plim = np.log(np.array(cheb_plim, dtype=np.float64))
    pp_cheb_tlim = 1. / np.array(cheb_tlim, dtype=np.float64)

    # plog parameter reorder
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

    return {'simple': {'A': A, 'b': b, 'Ta': Ta, 'type': simple_rate_type,
                       'num': num_simple, 'map': simple_map},
            'plog': {'map': plog_map, 'num': num_plog,
                     'num_P': num_pressures, 'params': plog_params,
                     'post_process': {'params': pp_plog_params}
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
                    'post_process': {
                        'eff_ns': pp_thd_eff_ns,
                        'spec_num': pp_thd_num_specs,
                        'spec': pp_thd_spec,
                        'eff': pp_thd_eff
                    }},
            'fwd': {'map': np.arange(len(reacs)), 'num': len(reacs),
                    'num_reac_to_spec':  fwd_num_spec,
                    'reac_to_spec': fwd_spec,
                    'nu': fwd_nu, 'allint': fwd_allnu_integer},
            'rev': {'map': rev_map, 'num': num_rev,
                    'num_reac_to_spec':  rev_num_spec,
                    'reac_to_spec': rev_spec,
                    'nu': rev_nu, 'allint': rev_allnu_integer},
            'net': {'num_reac_to_spec': net_num_spec, 'nu_sum': nu_sum,
                    'nu': net_nu, 'reac_to_spec': net_spec,
                    'allint': net_nu_integer},
            'net_per_spec': {'reac_count': spec_reac_count, 'nu': spec_nu,
                             'reacs': spec_to_reac, 'map': spec_list,
                             'allint': net_nu_integer},
            'Nr': len(reacs),
            'Ns': len(specs)}


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
    rate_list : list of :class:`knl_info`
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
    mapstore.check_and_add_transform(namestore.conc_dot,
                                     namestore.phi_spec_inds,
                                     force_inline=True)
    mapstore.check_and_add_transform(namestore.conc_arr,
                                     namestore.phi_spec_inds,
                                     force_inline=True)

    # add all non-mapped arrays
    if conp:
        h_lp, h_str = mapstore.apply_maps(namestore.h_arr, *default_inds)
        cp_lp, cp_str = mapstore.apply_maps(namestore.cp_arr, *default_inds)
        kernel_data.extend([h_lp, cp_lp])
    else:
        u_lp, u_str = mapstore.apply_maps(namestore.u_arr, *default_inds)
        cv_lp, cv_str = mapstore.apply_maps(namestore.cv_arr, *default_inds)
        kernel_data.extend([u_lp, cv_lp])

    conc_lp, conc_str = mapstore.apply_maps(namestore.conc_arr, *default_inds)
    Tdot_lp, Tdot_str = mapstore.apply_maps(namestore.T_dot, *fixed_inds)

    kernel_data.extend([conc_lp, Tdot_lp])

    conc_dot_lp, conc_dot_str = mapstore.apply_maps(namestore.conc_dot,
                                                    *default_inds)
    # kernel_data.add(conc_dot_lp)

    # put together conv/conp terms
    if conp:
        term = sp_utils.sanitize(term, subs={
            'H[k]': h_str,
            'dot{omega}[k]': conc_dot_str,
            '[C][k]': conc_str,
            '{C_p}[k]': cp_str
        })
    else:
        term = sp_utils.sanitize(term, subs={
            'U[k]': u_str,
            'dot{omega}[k]': conc_dot_str,
            '[C][k]': conc_str,
            '{C_v}[k]': cv_str
        })
    # now split into upper / lower halves
    factor = -1

    def separate(term):
        upper = sp.Mul(
            *[x for x in sp.Mul.make_args(term) if not x.has(sp.Pow) and x.has(sp.Sum)])
        lower = factor / (term / upper)  # take inverse
        upper = sp.Mul(
            *[x if not x.has(sp.Sum) else x.function for x in sp.Mul.make_args(upper)])
        lower = lower.function
        return upper, lower

    upper, lower = separate(term)

    pre_instructions = Template(
        '${Tdot_str} = ${factor} * simul_reduce(sum, ${var_name}, ${upper_term})'
        ' / simul_reduce(sum, ${var_name}, ${lower_term}) {id=sum}'
    ).safe_substitute(
        Tdot_str=Tdot_str,
        factor=factor,
        var_name=var_name)

    instructions = Template(pre_instructions).safe_substitute(
        upper_term=str(upper),
        lower_term=str(lower))

    can_vectorize = loopy_opts.depth is None
    # finally do vectorization ability and specializer

    def __vec_spec_wide(knl):
        # split the reduction
        knl = lp.split_reduction_outward(knl, 'j_outer')
        # and remove the sum_0 barrier
        knl = lp.realize_reduction(knl)
        # remove depends of update on end accumulator
        instruction_list = [insn if 'update' not in insn.id
                            else insn.copy(depends_on_is_final=True)
                            for insn in knl.instructions]
        # remove dummy sync of end accumulator on updates
        instruction_list = [insn if insn.id != 'sum_0'
                            else insn.copy(no_sync_with=insn.no_sync_with |
                                           frozenset([('sum_i_update', 'any')]))
                            for insn in instruction_list]
        return knl.copy(instructions=instruction_list)

    def __vec_spec_deep(knl):
        # do a dummy split
        knl = lp.split_iname(knl, 'i', 1, inner_tag='l.0')
        # split outwards and expand reductions
        knl = lp.split_reduction_outward(knl, 'i_inner')
        knl = lp.realize_reduction(knl)
        # abuse rename to fix 'two l.0 for sum_0'
        knl = lp.rename_iname(
            knl, 'red_i_inner_0', 'red_i_inner', existing_ok=True)
        # resolve dependencies and syncs
        sum_0_insn = next(
            insn for insn in knl.instructions if insn.id == 'sum_0')
        for insn in knl.instructions:
            # fix sum_0 sync problems
            if insn != sum_0_insn and 'sum_i' in insn.id:
                sum_0_insn.no_sync_with |= frozenset([(insn.id, 'any')])
            # check depends
            if insn != sum_0_insn and \
                    namestore.T_dot.name in insn.read_dependency_names():
                insn.depends_on_is_final = True
        return knl.copy(instructions=knl.instructions[:])
    vec_spec = __vec_spec_wide
    if loopy_opts.depth:
        vec_spec = __vec_spec_deep

    return k_gen.knl_info(name='temperature_rate',
                          pre_instructions=[instructions],
                          instructions='',
                          mapstore=mapstore,
                          var_name='i',
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
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator for both
        equation types
    """

    # find summation direction and consistency check
    over_reac = loopy_opts.spec_rates_sum_over_reac
    deep = loopy_opts.depth is not None
    if deep and over_reac:
        logging.warn('Cannot use summation over reaction with a deep '
                     'vectorization [not currently supported].'
                     '  Disabling...')
        over_reac = False

    kernel_data = []
    # add problem size
    if namestore.problem_size is not None:
        kernel_data.append(namestore.problem_size)

    if over_reac:
        # various indicies
        spec_ind = 'spec_ind'
        spec_map = 'spec_map'

        # create map store
        mapstore = arc.MapStore(loopy_opts,
                                namestore.num_reacs,
                                namestore.num_reacs)

        # create arrays
        spec_lp, spec_str = mapstore.apply_maps(namestore.net_reac_to_spec_map,
                                                spec_map)
        num_spec_offsets_lp, \
            num_spec_offsets_str = \
            mapstore.apply_maps(namestore.net_reac_to_spec_offsets, var_name)
        num_spec_offsets_next_lp, \
            num_spec_offsets_next_str = \
            mapstore.apply_maps(namestore.net_reac_to_spec_offsets,
                                var_name + ' + 1')
        net_nu_lp, net_nu_str = \
            mapstore.apply_maps(namestore.net_reac_to_spec_nu,
                                spec_map)
        rop_net_lp, rop_net_str = mapstore.apply_maps(namestore.rop_net,
                                                      *default_inds)
        dphi_lp, dphi_str = mapstore.apply_maps(namestore.conc_dot,
                                                global_ind, spec_ind + ' + 1')

        # update kernel args
        kernel_data.extend(
            [spec_lp, num_spec_offsets_lp, net_nu_lp, rop_net_lp, dphi_lp])

        # now the instructions
        instructions = Template(
            """
        <>net_rate = ${rop_net_str} {id=rate_init}
        <>offset = ${num_spec_offsets_str}
        <>num_spec = ${num_spec_offsets_next_str} - offset
        for ispec
            <> ${spec_map} = offset + ispec
            <> ${spec_ind} = ${spec_str} # (offset handled in wdot str)
            <> nu = ${nu_str}
            ${dphi_str} = ${dphi_str} + nu * net_rate
        end
        """).safe_substitute(rop_net_str=rop_net_str,
                             spec_str=spec_str,
                             spec_map=spec_map,
                             nu_str=net_nu_str,
                             spec_ind=spec_ind,
                             dphi_str=dphi_str,
                             num_spec_offsets_str=num_spec_offsets_str,
                             num_spec_offsets_next_str=num_spec_offsets_next_str)

        # extra inames
        extra_inames = [('ispec', '0 <= ispec < num_spec')]

    else:
        # various indicies
        reac_ind = 'reac_ind'
        reac_map = 'reac_map'

        # create map store
        mapstore = arc.MapStore(loopy_opts,
                                namestore.net_nonzero_spec,
                                namestore.net_nonzero_spec)

        # add mappings
        mapstore.check_and_add_transform(namestore.conc_dot,
                                         namestore.net_nonzero_phi)

        # create arrays

        # inner loop vars depend on reac_map
        reac_lp, reac_str = mapstore.apply_maps(namestore.net_spec_to_reac,
                                                reac_map)
        net_nu_lp, net_nu_str = \
            mapstore.apply_maps(namestore.net_spec_to_reac_nu,
                                reac_map)
        # offsets depend on 'i'
        num_reac_offsets_lp, \
            num_reac_offsets_str = \
            mapstore.apply_maps(namestore.net_spec_to_reac_offsets, var_name)
        num_reac_offsets_next_lp, \
            num_reac_offsets_next_str = \
            mapstore.apply_maps(namestore.net_spec_to_reac_offsets,
                                var_name + ' + 1')
        # rop net depends on reac_ind
        rop_net_lp, rop_net_str = mapstore.apply_maps(namestore.rop_net,
                                                      global_ind, reac_ind)
        # dphi dep
        dphi_lp, dphi_str = mapstore.apply_maps(namestore.conc_dot,
                                                *default_inds)
        kernel_data.extend(
            [reac_lp, net_nu_lp, num_reac_offsets_lp, dphi_lp, rop_net_lp])

        # now the instructions
        instructions = Template(
            """
        <>rxn_offset = ${num_reac_offsets_str}
        <>num_rxn = ${num_reac_offsets_next_str} - rxn_offset
        for irxn
            <>${reac_map} = rxn_offset + irxn
            <>${reac_ind} = ${reac_str}
            ${dphi_str} = ${dphi_str} + ${net_nu_str} * ${rop_net_str}
        end
        """).safe_substitute(reac_map=reac_map,
                             reac_ind=reac_ind,
                             num_reac_offsets_str=num_reac_offsets_str,
                             num_reac_offsets_next_str=num_reac_offsets_next_str,
                             reac_str=reac_str,
                             rop_net_str=rop_net_str,
                             net_nu_str=net_nu_str,
                             dphi_str=dphi_str)

        extra_inames = [('irxn', '0 <= irxn < num_rxn')]

    return k_gen.knl_info(name='spec_rates',
                          instructions=instructions,
                          mapstore=mapstore,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          extra_inames=extra_inames)


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
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # create net rop kernel

    kernel_data = OrderedDict([('fwd', [])])
    maps = OrderedDict([('fwd',
                         arc.MapStore(loopy_opts, namestore.num_reacs,
                                      namestore.num_reacs))])

    separated_kernels = loopy_opts.rop_net_kernels
    if separated_kernels:
        kernel_data['rev'] = []
        maps['rev'] = arc.MapStore(loopy_opts, namestore.rev_map,
                                   namestore.rev_mask)
        kernel_data['pres_mod'] = []
        maps['pres_mod'] = arc.MapStore(loopy_opts, namestore.thd_map,
                                        namestore.thd_mask)

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

    if test_size == 'problem_size':
        __add_to_all(namestore.problem_size)

    # create the fwd rop array / str
    # this never has a map / mask
    rop_fwd_lp, rop_fwd_str = __get_map('fwd').\
        apply_maps(namestore.rop_fwd, *default_inds)

    __add_data('fwd', rop_fwd_lp)

    if namestore.rop_rev is not None:
        # we have reversible reactions

        # first check for map / mask
        if not separated_kernels:
            # if a single kernel, add a mask for reversible reactions
            __get_map('rev').check_and_add_transform(namestore.rop_rev,
                                                     namestore.rev_mask)
        # otherwise, no map needed as the seperate rev kernel loops over
        # the rev indicies

        # apply the maps
        rop_rev_lp, rop_rev_str = __get_map('rev').\
            apply_maps(namestore.rop_rev, *default_inds)

        # add data
        __add_data('rev', rop_rev_lp)

    if namestore.pres_mod is not None:
        # we have pres mod reactions

        # first check for map / mask
        if not separated_kernels:
            # if a single kernel, add a mask for pmod reactions
            __get_map('pres_mod').\
                check_and_add_transform(namestore.pres_mod,
                                        namestore.thd_mask)
        # otherwise, no map needed as the seperate pmod kernel loops over
        # the pmod indicies

        # apply the maps
        pres_mod_lp, pres_mod_str = __get_map('pres_mod').\
            apply_maps(namestore.pres_mod, *default_inds)

        # add data
        __add_data('pres_mod', pres_mod_lp)

    # add rop net to all kernels:
    rop_strs = {}
    for name in kernel_data:
        # check for map / mask
        if separated_kernels and name != 'fwd':
            # fwd kernel has no map or mask (guarenteed same indicies)
            # other kernels may have a map, so check
            mapval = namestore.rev_map if name == 'rev'\
                else namestore.thd_map
            __get_map(name).\
                check_and_add_transform(namestore.rop_net,
                                        mapval)

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
        if namestore.rop_rev is not None:
            rev_update_instructions = Template(
                """
            net_rate = net_rate - ${rop_rev_str} {id=rate_update_rev, dep=rate_update}
            """).safe_substitute(
                rop_rev_str=rop_rev_str)

            # If there is a mask, we need to encase this in an if statement
            if namestore.rop_rev in __get_map('rev').transformed_variables:
                rev_update_instructions = Template(
                    """
            if ${rev_mask} >= 0
                ${cur_inst}
            end
            """).safe_substitute(cur_inst=rev_update_instructions,
                                 rev_mask=__get_map('rev').
                                 transformed_variables[namestore.rop_rev]
                                 .new_iname)
        else:
            rev_update_instructions = ''

        # pmod update
        if namestore.pres_mod is not None:
            pmod_update_instructions = Template(
                """
        net_rate = net_rate * ${pres_mod_str} {id=rate_update_pmod, dep=rate_update${rev_dep}}
        """).safe_substitute(
                rev_dep=':rate_update_rev' if namestore.rop_rev is not None
                    else '',
                pres_mod_str=pres_mod_str)
            if namestore.pres_mod in \
                    __get_map('pres_mod').transformed_variables:
                # num pmod != num rxns
                pmod_update_instructions = Template(
                    """
            if ${pres_mask} >= 0
                ${cur_inst}
            end
            """).safe_substitute(cur_inst=pmod_update_instructions,
                                 pres_mask=__get_map('pres_mod').
                                 transformed_variables[namestore.pres_mod]
                                 .new_iname)
        else:
            pmod_update_instructions = ''

        instructions = Template(instructions).safe_substitute(
            rev_update=rev_update_instructions,
            pmod_update=pmod_update_instructions)

        instructions = '\n'.join(
            [x for x in instructions.split('\n') if x.strip()])

        return k_gen.knl_info(name='rop_net_fixed',
                              instructions=instructions,
                              var_name='i',
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
                                        var_name='i',
                                        kernel_data=kernel_data[kernel],
                                        mapstore=maps[kernel]))
        return infos


def get_rop(eqs, loopy_opts, namestore, allint, test_size=None):
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
        Contains keys 'fwd' and 'rev', with booleans corresponding to whether
        all nu values for that direction are integers.  If True, powers of
        concentrations will be evaluated using multiplications
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
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
            inds = namestore.rev_map
            mapinds = namestore.rev_mask

        maps[direction] = arc.MapStore(loopy_opts, inds, mapinds)
        themap = maps[direction]

        # we need species lists, nu lists, etc.

        # offsets are on main loop, no map
        offsets = getattr(namestore, direction + '_reac_to_spec_offsets')
        num_spec_offsets_lp, num_spec_offsets_str = themap.apply_maps(
            offsets, var_name)

        # next offset to calculate num species
        _, num_spec_offsets_next_str = themap.apply_maps(
            offsets, var_name + ' + 1')

        # nu lists are on main loop, no map
        nu = getattr(namestore, 'nu_' + direction)
        nu_lp, nu_str = themap.apply_maps(nu, spec_loop)

        # species lists are in ispec loop, use that iname
        speclist = getattr(namestore, direction + '_reac_to_spec')
        spec_lp, spec_str = themap.apply_maps(speclist, spec_loop)

        # rate constants on main loop, no map
        rateconst = namestore.kf if direction == 'fwd' else namestore.kr
        rateconst_arr, rateconst_str = themap.apply_maps(
            rateconst, *default_inds)

        # concentrations in ispec loop, also use offset for phi
        concs_lp, concs_str = themap.apply_maps(
            namestore.conc_arr, global_ind, spec_ind + ' + 1')

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
    <>num_spec = ${num_spec_offsets_next_str} - spec_offset
    for ${spec_loop}
        <>spec_map = spec_offset + ispec
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
    <>conc_temp = 1.0d {id=conc_init}
    <>nu = ${nu_str}
    for inu
        conc_temp = conc_temp * ${concs_str} {id=conc_update, dep=spec_ind:conc_init}
    end
    rop_temp = rop_temp * conc_temp {id=rop_fin, dep=conc_update}""").safe_substitute(
            nu_str=nu_str,
            concs_str=concs_str)

        # if we need to use powers, do so
        fractional_eval = Template(
            """
    if int(${nu_str}) == ${nu_str}
        ${allint}
    else
        rop_temp = rop_temp * (${concs_str})**(${nu_str}) {id=rop_fin2}
    end
    """).safe_substitute(nu_str=nu_str,
                         concs_str=concs_str)
        fractional_eval = k_gen.subs_at_indent(fractional_eval, 'allint',
                                               allint_eval)

        if not allint[direction]:
            rop_instructions = k_gen.subs_at_indent(rop_instructions,
                                                    'rop_temp_eval',
                                                    fractional_eval)
        else:
            rop_instructions = k_gen.subs_at_indent(rop_instructions,
                                                    'rop_temp_eval',
                                                    allint_eval)

        # and finally extra inames
        extra_inames = [(spec_loop, 'spec_offset <= {} < spec_offset + num_spec'.format(spec_loop)),
                        ('inu', '0 <= inu < nu')]

        # and return the rateconst
        return k_gen.knl_info(name='rop_eval_{}'.format(direction),
                              instructions=rop_instructions,
                              var_name=var_name,
                              kernel_data=kernel_data,
                              extra_inames=extra_inames,
                              mapstore=maps[direction])

    infos = [__rop_create('fwd')]
    if namestore.rop_rev is not None:
        infos.append(__rop_create('rev'))
    return infos


def get_rxn_pres_mod(eqs, loopy_opts, namestore, test_size=None):
    """Generates instructions, kernel arguements, and data for pressure modification
    term of the forward reaction rates.

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    namestore : :class:`array_creator.NameStore`
        The namestore / creator for this method
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    # start developing the ci kernel
    # rate info and reac ind

    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # create the third body conc pres-mod kernel

    thd_map = arc.MapStore(loopy_opts, namestore.thd_map,
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

    fall_map = arc.MapStore(loopy_opts, namestore.fall_map,
                            namestore.fall_mask)

    # the falloff vs chemically activated indicator
    fall_type_lp, fall_type_str = \
        fall_map.apply_maps(namestore.fall_type, var_name)

    # the blending term
    Fi_lp, Fi_str = \
        fall_map.apply_maps(namestore.Fi, *default_inds)

    # the Pr array
    Pr_lp, Pr_str = \
        fall_map.apply_maps(namestore.Pr, *default_inds)

    # and the pressure mod term (use fall_to_thd_map/mask)
    fall_map.check_and_add_transform(namestore.pres_mod,
                                     namestore.fall_to_thd_map)

    pres_mod_lp, pres_mod_str = \
        fall_map.apply_maps(namestore.pres_mod, *default_inds)

    # update the args
    kernel_data.extend([Fi_lp, Pr_lp, fall_type_lp, pres_mod_lp])

    fall_instructions = Template("""
    <>ci_temp = ${Fi_str} / (1 + ${Pr_str}) {id=ci_decl}
    if ${fall_type} == 0
        ci_temp = ci_temp * ${Pr_str} {id=ci_update, dep=ci_decl}
    end
    ${pres_mod} = ci_temp {dep=ci_update}
""").safe_substitute(Fi_str=Fi_str,
                     Pr_str=Pr_str,
                     pres_mod=pres_mod_str,
                     fall_type=fall_type_str)

    # add to the info list
    info_list.append(
        k_gen.knl_info(name='ci_fall',
                       instructions=fall_instructions,
                       var_name=var_name,
                       kernel_data=kernel_data,
                       mapstore=fall_map))

    return info_list


def get_rev_rates(eqs, loopy_opts, namestore, allint, test_size=None):
    """Generates instructions, kernel arguements, and data for reverse reaction rates

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
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
    # start developing the Kc kernel
    kernel_data = []
    spec_ind = 'spec_ind'
    spec_loop = 'ispec'

    if test_size == 'problem_size':
        kernel_data.append(namestore.problem_size)

    # set of eqn's doesn't matter
    conp_eqs = eqs['conp']

    # add the reverse map
    rev_map = arc.MapStore(loopy_opts, namestore.rev_map, namestore.rev_mask)

    # find Kc equation
    Kc_sym = next(x for x in conp_eqs if str(x) == '{K_c}[i]')
    Kc_eqn = conp_eqs[Kc_sym]
    nu_sym = next(x for x in Kc_eqn.free_symbols if str(x) == 'nu[k, i]')
    B_sym = next(x for x in Kc_eqn.free_symbols if str(x) == 'B[k]')
    kr_sym = next(x for x in conp_eqs if str(x) == '{k_r}[i]')
    # kf_sym = next(x for x in conp_eqs if str(x) == '{k_f}[i]')

    # create nu_sum on main loop
    # this may require a map
    rev_map.check_and_add_transform(
        namestore.reac_to_spec_nu_sum, namestore.rev_map)
    nu_sum_lp, nu_sum_str = rev_map.apply_maps(namestore.reac_to_spec_nu_sum,
                                               var_name)

    # all species in reaction on spec loop
    spec_lp, spec_str = rev_map.apply_maps(namestore.net_reac_to_spec_map,
                                           spec_loop)

    # species offsets on main loop
    # this may require a map
    rev_map.check_and_add_transform(
        namestore.net_reac_to_spec_offsets, namestore.rev_map)
    num_spec_offsets_lp, num_spec_offsets_str = rev_map.apply_maps(
        namestore.net_reac_to_spec_offsets, var_name)

    # species offset on main loop with offset of 1
    _, num_spec_offsets_next_str = rev_map.apply_maps(
        namestore.net_reac_to_spec_offsets, var_name, affine=1)

    # B array on spec_ind
    B_lp, B_str = rev_map.apply_maps(namestore.b_arr, global_ind, spec_ind)

    # net nu on species loop
    net_nu_lp, net_nu_str = rev_map.apply_maps(namestore.net_reac_to_spec_nu,
                                               spec_loop)

    # the Kc array on the main loop, no map as this is only reversible
    Kc_lp, Kc_str = rev_map.apply_maps(namestore.Kc, *default_inds)

    # modify Kc equation
    Kc_eqn = sp_utils.sanitize(conp_eqs[Kc_sym],
                               symlist={'nu[k, i]': nu_sym,
                                        'B[k]': B_sym},
                               subs={
        sp.Sum(nu_sym, (sp.Idx('k'), 1, sp.Symbol('N_s'))): nu_sum_str})

    # insert the B sum into the Kc equation
    Kc_eqn_Pres = next(
        x for x in sp.Mul.make_args(Kc_eqn) if x.has(sp.Symbol('R_u')))
    Kc_eqn_exp = Kc_eqn / Kc_eqn_Pres
    Kc_eqn_exp = sp_utils.sanitize(Kc_eqn_exp,
                                   symlist={'nu[k, i]': nu_sym,
                                            'B[k]': B_sym},
                                   subs={
                                       sp.Sum(B_sym * nu_sym,
                                              (sp.Idx('k'), 1, sp.Symbol('N_s')
                                               )): 'B_sum'})

    # create the kf array / str
    # this may require a map
    rev_map.check_and_add_transform(namestore.kf, namestore.rev_map)
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
                        B_lp, Kc_lp, net_nu_lp, kf_arr, kr_arr])

    # create the pressure product loop
    pressure_prod = Template("""
    <> P_sum_end = abs(${nu_sum}) {id=P_bound}
    <> P_sum = 1.0d {id=P_init}
    if ${nu_sum} > 0
        <> P_val = P_a / R_u {id=P_val_decl}
    else
        P_val = R_u / P_a {id=P_val_decl1}
    end
    for P_sum_ind
        P_sum = P_sum * P_val {id=P_accum, dep=P_val_decl:P_val_decl1:P_bound:P_init}
    end
    """).safe_substitute(nu_sum=nu_sum_str)

    if not allint['net']:
        # if not all integers, need to add outer if statment to check integer
        # status
        pressure_prod_temp = Template("""
    if int(${nu_sum}) == ${nu_sum}
        ${pprod}
    else
        P_sum = (P_a / R_u)**(${nu_sum}) {id=P_accum}
    end""").safe_substitute(nu_sum=nu_sum_str)

        pressure_prod = k_gen.subs_at_indent(pressure_prod_temp, 'pprod',
                                             pressure_prod)

    # and the b sum loop
    Bsum_inst = Template("""
    <>offset = ${spec_offset} {id=offset}
    <>spec_end = ${spec_offset_next} {id=B_bound}
    <>B_sum = 0 {id=B_init}
    for ${spec_loop}
        <>${spec_ind} = ${spec_mapper} {dep=offset:B_bound}
        if ${net_nu} != 0
            B_sum = B_sum + ${net_nu} * ${B_val} {id=B_accum, dep=B_init}
        end
    end
    B_sum = exp(B_sum) {id=B_final, dep=B_accum}
    """).safe_substitute(spec_offset=num_spec_offsets_str,
                         spec_offset_next=num_spec_offsets_next_str,
                         spec_loop=spec_loop,
                         spec_ind=spec_ind,
                         spec_mapper=spec_str,
                         nu_val=nu_sum_str,
                         net_nu=net_nu_str,
                         B_val=B_str
                         )

    Rate_assign = Template("""
    <>${Kc_temp_str} = P_sum * B_sum {dep=P_accum:B_final}
    ${Kc_val} = ${Kc_temp_str}
    ${kr_val} = ${rev_eqn}
    """).safe_substitute(Kc_val=Kc_str,
                         Kc_temp_str=Kc_temp_str,
                         kr_val=kr_str,
                         rev_eqn=kr_eqn)

    instructions = '\n'.join([Bsum_inst, pressure_prod, Rate_assign])

    # create the extra inames
    extra_inames = [('P_sum_ind', '0 <= P_sum_ind < P_sum_end'),
                    (spec_loop, 'offset <= {} < spec_end'.format(spec_loop))]

    # and return the rateinfo
    return k_gen.knl_info(name='rateconst_Kc',
                          instructions=instructions,
                          var_name=var_name,
                          kernel_data=kernel_data,
                          mapstore=rev_map,
                          extra_inames=extra_inames,
                          parameters={
                              'P_a': np.float64(chem.PA),
                              'R_u': np.float64(chem.RU)})


def get_thd_body_concs(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for third body concentrations

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator
    """

    Ns = rate_info['Ns']
    num_thd = rate_info['thd']['num']
    reac_ind = 'i'

    # create args
    indicies = ['${species_ind}', 'j']
    concs_lp, concs_str, _ = lp_utils.get_loopy_arg('conc',
                                                    indicies=indicies,
                                                    dimensions=(Ns, test_size),
                                                    order=loopy_opts.order)
    concs_str = Template(concs_str)

    indicies = ['${reac_ind}', 'j']
    thd_lp, thd_str, _ = lp_utils.get_loopy_arg('thd_conc',
                                                indicies=indicies,
                                                dimensions=(
                                                    num_thd, test_size),
                                                order=loopy_opts.order)

    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    P_arr = lp.GlobalArg('P_arr', shape=(test_size,), dtype=np.float64)

    thd_eff_ns = np.ones(num_thd)
    num_specs = rate_info['thd']['spec_num'].copy()
    spec_list = rate_info['thd']['spec'].copy()
    thd_effs = rate_info['thd']['eff'].copy()

    last_spec = Ns - 1
    # first, we must do some surgery to get _our_ form of the thd-body
    # efficiencies
    for i in range(num_specs.size):
        num = num_specs[i]
        offset = np.sum(num_specs[:i])
        # check if Ns has a non-default efficiency
        if last_spec in spec_list[offset:offset+num]:
            ind = np.where(spec_list[offset:offset+num] == last_spec)[0][0]
            # set the efficiency
            thd_eff_ns[i] = thd_effs[offset + ind]
            # delete from the species list
            spec_list = np.delete(spec_list, offset + ind)
            # delete from effiency list
            thd_effs = np.delete(thd_effs, offset + ind)
            # subtract from species num
            num_specs[i] -= 1
        # and subtract from efficiencies
        thd_effs[offset:offset+num_specs[i]] -= thd_eff_ns[i]
        if thd_eff_ns[i] != 1:
            # we need to add all the other species :(
            # get updated species list
            to_add = np.array(range(Ns - 1), dtype=np.int32)
            # and efficiencies
            eff = np.array([1 - thd_eff_ns[i] for spec in range(Ns - 1)])
            eff[spec_list[offset:offset+num_specs[i]]
                ] = thd_effs[offset:offset+num_specs[i]]
            # delete from the species list / efficiencies
            spec_list = np.delete(
                spec_list, range(offset, offset+num_specs[i]))
            thd_effs = np.delete(thd_effs,  range(offset, offset+num_specs[i]))
            # insert
            spec_list = np.insert(spec_list, offset, to_add)
            thd_effs = np.insert(thd_effs, offset, eff)
            # and update number
            num_specs[i] = to_add.size

    # and temporary variables:
    thd_type_lp, thd_type_str = __1Dcreator(
        'thd_type', rate_info['thd']['type'], scope=scopes.GLOBAL)
    thd_eff_lp, thd_eff_str = __1Dcreator('thd_eff', thd_effs)
    thd_spec_lp, thd_spec_str = __1Dcreator('thd_spec', spec_list)
    thd_num_spec_lp, thd_num_spec_str = __1Dcreator('thd_spec_num', num_specs)
    thd_eff_ns_lp, thd_eff_ns_str = __1Dcreator('thd_eff_ns', thd_eff_ns)

    # calculate offsets
    offsets = np.cumsum(num_specs, dtype=np.int32) - num_specs
    thd_offset_lp, thd_offset_str = __1Dcreator('thd_offset', offsets)

    # kernel data
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    kernel_data.extend([T_arr, P_arr, concs_lp, thd_lp, thd_type_lp,
                        thd_eff_lp, thd_spec_lp, thd_num_spec_lp, thd_offset_lp,
                        thd_eff_ns_lp])

    # maps
    out_map = {}
    outmap_name = 'out_map'
    indicies = rate_info['thd']['map'].astype(dtype=np.int32)
    indicies = k_gen.handle_indicies(indicies, reac_ind, out_map, kernel_data)
    # extra loops
    extra_inames = [('k', '0 <= k < num')]

    # generate instructions
    instructions = Template("""
<> offset = ${offset} {id=offset}
<> num_temp = ${num_spec_str} {id=num0}
if ${type_str} == 1 # single species
    <> thd_temp = ${conc_spec} {id=thd0, dep=num0}
    num_temp = 0 {id=num1, dep=num0}
else
    thd_temp = P_arr[j] * ${thd_eff_ns_str} / (R * T_arr[j]) {id=thd1, dep=num0}
end
<> num = num_temp {dep=num*}
for k
    thd_temp = thd_temp + ${thd_eff} * ${conc_thd_spec} {id=thdcalc}
end
${thd_str} = thd_temp {dep=thd*}
""")

    # sub in instructions
    instructions = Template(
        instructions.safe_substitute(
            offset=thd_offset_str,
            type_str=thd_type_str,
            conc_spec=concs_str.safe_substitute(
                species_ind=Template(thd_spec_str).safe_substitute(
                    reac_ind='offset')),
            num_spec_str=thd_num_spec_str,
            thd_eff=Template(thd_eff_str).safe_substitute(
                reac_ind='offset + k'),
            conc_thd_spec=concs_str.safe_substitute(
                species_ind=Template(thd_spec_str).safe_substitute(
                    reac_ind='offset + k')),
            thd_str=thd_str,
            thd_eff_ns_str=thd_eff_ns_str
        )
    ).safe_substitute(reac_ind=reac_ind)

    # create info
    info = k_gen.knl_info('eval_thd_body_concs',
                          instructions=instructions,
                          var_name=reac_ind,
                          kernel_data=kernel_data,
                          extra_inames=extra_inames,
                          indicies=indicies,
                          parameters={'R': chem.RU},
                          extra_subs={'reac_ind': reac_ind})
    return info


def get_cheb_arrhenius_rates(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for cheb rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    # the equation set doesn't matter for this application
    # just use conp
    conp_eqs = eqs['conp']

    rate_eqn_pre = get_rate_eqn(eqs)
    # find the cheb equation
    cheb_eqn = next(x for x in conp_eqs if str(x) == 'log({k_f}[i])/log(10)')
    cheb_form, cheb_eqn = cheb_eqn, conp_eqs[cheb_eqn][(reaction_type.cheb,)]
    cheb_form = sp.Pow(10, sp.Symbol('kf_temp'))

    # make nice symbols
    Tinv = sp.Symbol('T_inv')
    logP = sp.Symbol('logP')
    Pmax, Pmin, Tmax, Tmin = sp.symbols('Pmax Pmin Tmax Tmin')
    Pred, Tred = sp.symbols('Pred Tred')

    # get tilde{T}, tilde{P}
    T_red = next(x for x in conp_eqs if str(x) == 'tilde{T}')
    P_red = next(x for x in conp_eqs if str(x) == 'tilde{P}')

    Pred_eqn = sp_utils.sanitize(conp_eqs[P_red], subs={sp.log(sp.Symbol('P_{min}')): Pmin,
                                                        sp.log(sp.Symbol('P_{max}')): Pmax,
                                                        sp.log(sp.Symbol('P')): logP})

    Tred_eqn = sp_utils.sanitize(conp_eqs[T_red], subs={sp.S.One / sp.Symbol('T_{min}'): Tmin,
                                                        sp.S.One / sp.Symbol('T_{max}'): Tmax,
                                                        sp.S.One / sp.Symbol('T'): Tinv})

    # number of cheb reactions
    num_cheb = rate_info['cheb']['num']
    # degree of pressure polynomial per reaction
    num_P = np.array(rate_info['cheb']['num_P'], dtype=np.int32)
    # degree of temperature polynomial per reaction
    num_T = np.array(rate_info['cheb']['num_T'], dtype=np.int32)

    # max degrees in mechanism
    maxP = int(np.max(num_P))
    maxT = int(np.max(num_T))
    minP = int(np.min(num_P))
    minT = int(np.min(num_T))
    poly_max = int(np.maximum(maxP, maxT))

    # now we start defining parameters / temporary variable

    # workspace vars
    pres_poly_lp = lp.TemporaryVariable('pres_poly', shape=(poly_max,),
                                        dtype=np.float64, scope=scopes.PRIVATE, read_only=False)
    temp_poly_lp = lp.TemporaryVariable('temp_poly', shape=(poly_max,),
                                        dtype=np.float64, scope=scopes.PRIVATE, read_only=False)
    pres_poly_str = Template('pres_poly[${pres_poly_ind}]')
    temp_poly_str = Template('temp_poly[${temp_poly_ind}]')

    # chebyshev parameters
    params = np.zeros((num_cheb, maxT, maxP))
    for i, p in enumerate(rate_info['cheb']['params']):
        params[i, :num_T[i], :num_P[i]] = p[:, :]

    indicies = ['${reac_ind}', '${temp_poly_ind}', '${pres_poly_ind}']
    params_lp = lp.TemporaryVariable('cheb_params', shape=params.shape,
                                     initializer=params, scope=scopes.GLOBAL, read_only=True)
    params_str = Template('cheb_params[' + ','.join(indicies) + ']')

    # finally the min/maxs & param #'s
    numP_lp = lp.TemporaryVariable('cheb_numP', shape=num_P.shape,
                                   initializer=num_P, dtype=np.int32, read_only=True, scope=scopes.GLOBAL)
    numP_str = 'cheb_numP[${reac_ind}]'
    numT_lp = lp.TemporaryVariable('cheb_numT', shape=num_T.shape,
                                   initializer=num_T, dtype=np.int32, read_only=True, scope=scopes.GLOBAL)
    numT_str = 'cheb_numT[${reac_ind}]'

    # limits for cheby polys
    Plim = np.log(np.array(rate_info['cheb']['Plim'], dtype=np.float64))
    Tlim = 1. / np.array(rate_info['cheb']['Tlim'], dtype=np.float64)

    indicies = ['${reac_ind}', '${lim_ind}']
    plim_lp = lp.TemporaryVariable('cheb_plim', shape=Plim.shape,
                                   initializer=Plim, scope=scopes.GLOBAL, read_only=True)
    tlim_lp = lp.TemporaryVariable('cheb_tlim', shape=Tlim.shape,
                                   initializer=Tlim, scope=scopes.GLOBAL, read_only=True)
    plim_str = Template('cheb_plim[' + ','.join(indicies) + ']')
    tlim_str = Template('cheb_tlim[' + ','.join(indicies) + ']')

    T_arr, T_arr_str, _ = lp_utils.get_loopy_arg('T_arr',
                                                 indicies=['j'],
                                                 dimensions=[test_size],
                                                 order=loopy_opts.order,
                                                 dtype=np.float64)
    P_arr, P_arr_str, _ = lp_utils.get_loopy_arg('P_arr',
                                                 indicies=['j'],
                                                 dimensions=[test_size],
                                                 order=loopy_opts.order,
                                                 dtype=np.float64)
    kernel_data = [params_lp, numP_lp, numT_lp, plim_lp, tlim_lp,
                   pres_poly_lp, temp_poly_lp, T_arr, P_arr]

    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    reac_ind = 'i'
    out_map = {}
    outmap_name = 'out_map'
    indicies = rate_info['cheb']['map'].astype(dtype=np.int32)
    Nr = rate_info['Nr']

    indicies = k_gen.handle_indicies(indicies, reac_ind,
                                     out_map, kernel_data, inmap_name=outmap_name,
                                     force_zero=True)

    # get the proper kf indexing / array
    kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf',
                                                        indicies=[
                                                            reac_ind, 'j'],
                                                        dimensions=[
                                                            Nr, test_size],
                                                        order=loopy_opts.order,
                                                        map_name=out_map)
    maps = []
    if reac_ind in map_result:
        maps.append(map_result[reac_ind])

    # add to kernel data
    kernel_data.append(kf_arr)

    # preinstructions
    preinstructs = [k_gen.PLOG_PREINST_KEY, k_gen.TINV_PREINST_KEY]

    # extra loops
    pres_poly_ind = 'k'
    temp_poly_ind = 'm'
    extra_inames = [(pres_poly_ind, '0 <= {} < {}'.format(pres_poly_ind, maxP)),
                    (temp_poly_ind, '0 <= {} < {}'.format(
                        temp_poly_ind, maxT)),
                    ('p', '2 <= p < {}'.format(poly_max))]

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

${kf_str} = exp10(kf_temp) {dep=kf}
""")

    instructions = Template(instructions.safe_substitute(
        kf_str=kf_str,
        Tred_str=str(Tred_eqn),
        Pred_str=str(Pred_eqn),
        Pmin_str=plim_str.safe_substitute(lim_ind=0),
        Pmax_str=plim_str.safe_substitute(lim_ind=1),
        Tmin_str=tlim_str.safe_substitute(lim_ind=0),
        Tmax_str=tlim_str.safe_substitute(lim_ind=1),
        ppoly_0=pres_poly_str.safe_substitute(pres_poly_ind=0),
        ppoly_1=pres_poly_str.safe_substitute(pres_poly_ind=1),
        ppoly_k=pres_poly_str.safe_substitute(pres_poly_ind=pres_poly_ind),
        ppoly_p=pres_poly_str.safe_substitute(pres_poly_ind='p'),
        ppoly_pm1=pres_poly_str.safe_substitute(pres_poly_ind='p - 1'),
        ppoly_pm2=pres_poly_str.safe_substitute(pres_poly_ind='p - 2'),
        tpoly_0=temp_poly_str.safe_substitute(temp_poly_ind=0),
        tpoly_1=temp_poly_str.safe_substitute(temp_poly_ind=1),
        tpoly_m=temp_poly_str.safe_substitute(temp_poly_ind=temp_poly_ind),
        tpoly_p=temp_poly_str.safe_substitute(temp_poly_ind='p'),
        tpoly_pm1=temp_poly_str.safe_substitute(temp_poly_ind='p - 1'),
        tpoly_pm2=temp_poly_str.safe_substitute(temp_poly_ind='p - 2'),
        chebpar_km=params_str.safe_substitute(temp_poly_ind=temp_poly_ind,
                                              pres_poly_ind=pres_poly_ind),
        numP_str=numP_str,
        numT_str=numT_str,
        kf_eval=str(cheb_form),
        num_cheb=num_cheb)).safe_substitute(reac_ind=reac_ind)

    return k_gen.knl_info('rateconst_cheb', instructions=instructions, pre_instructions=preinstructs,
                          var_name=reac_ind, kernel_data=kernel_data, maps=maps,
                          extra_inames=extra_inames, indicies=indicies,
                          extra_subs={'reac_ind': reac_ind})


def get_plog_arrhenius_rates(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for p-log rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    rate_eqn = get_rate_eqn(eqs)

    # find the plog equation
    plog_eqn = next(x for x in eqs['conp'] if str(x) == 'log({k_f}[i])')
    plog_form, plog_eqn = plog_eqn, eqs[
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

    # total # of plog reactions
    num_plog = rate_info['plog']['num']
    # number of parameter sets per reaction
    num_params = rate_info['plog']['num_P']

    # create the loopy equivalents
    params = rate_info['plog']['params']
    # create a copy
    params_temp = [p[:] for p in params]
    # max # of parameters for sizing
    maxP = np.max(num_params)

    # for simplicity, we're going to use a padded form
    params = np.zeros((4, num_plog, maxP))
    for m in range(4):
        for i, numP in enumerate(num_params):
            for j in range(numP):
                params[m, i, j] = params_temp[i][j][m]

    # take the log of P and A
    hold = np.seterr(divide='ignore')
    params[0, :, :] = np.log(params[0, :, :])
    params[1, :, :] = np.log(params[1, :, :])
    params[np.where(np.isinf(params))] = 0
    np.seterr(**hold)

    # default params indexing order
    inds = ['${m}', '${reac_ind}', '${param_ind}']
    # make loopy version
    plog_params_lp = lp.TemporaryVariable('plog_params', shape=params.shape,
                                          initializer=params, scope=scopes.GLOBAL, read_only=True)
    param_str = Template('plog_params[' + ','.join(inds) + ']')

    # and finally the loopy version of num_params
    num_params_lp = lp.TemporaryVariable('plog_num_params', shape=lp.auto,
                                         initializer=num_params, read_only=True, scope=scopes.GLOBAL)

    # create temporary variables
    low_lp = lp.TemporaryVariable(
        'low', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    hi_lp = lp.TemporaryVariable(
        'hi', shape=(4,), scope=scopes.PRIVATE, dtype=np.float64)
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    P_arr = lp.GlobalArg('P_arr', shape=(test_size,), dtype=np.float64)

    # start creating the k_gen.knl_info's
    # data
    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))
    kernel_data.extend([plog_params_lp, num_params_lp, T_arr,
                        P_arr, low_lp, hi_lp])

    # reac ind
    reac_ind = 'i'

    # extra loops
    extra_inames = [('k', '0 <= k < {}'.format(maxP - 1)), ('m', '0 <= m < 4')]

    # see if we need an output mask
    out_map = {}
    indicies = rate_info['plog']['map'].astype(dtype=np.int32)
    Nr = rate_info['Nr']

    indicies = k_gen.handle_indicies(indicies, reac_ind, out_map,
                                     kernel_data, inmap_name='plog_inds')

    # get the proper kf indexing / array
    kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf',
                                                        indicies=[
                                                            reac_ind, 'j'],
                                                        dimensions=[
                                                            Nr, test_size],
                                                        map_name=out_map,
                                                        order=loopy_opts.order)
    kernel_data.append(kf_arr)

    # handle map info
    maps = []
    if reac_ind in map_result:
        maps.append(map_result[reac_ind])

    # instructions
    instructions = Template(Template(
        """
        <>lower = logP <= ${pressure_lo} # check below range
        if lower
            <>lo_ind = 0 {id=ind00}
            <>hi_ind = 0 {id=ind01}
        end
        <>numP = plog_num_params[${reac_ind}] - 1
        <>upper = logP > ${pressure_hi} # check above range
        if upper
            lo_ind = numP {id=ind10}
            hi_ind = numP {id=ind11}
        end
        <>oor = lower or upper
        for k
            # check that
            # 1. inside this reactions parameter's still
            # 2. inside pressure range
            <> midcheck = (k <= numP) and (logP > ${pressure_mid_lo}) and (logP <= ${pressure_mid_hi})
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
                     pressure_lo=param_str.safe_substitute(m=0, param_ind=0),
                     pressure_hi=param_str.safe_substitute(
                         m=0, param_ind='numP'),
                     pressure_mid_lo=param_str.safe_substitute(
                         m=0, param_ind='k'),
                     pressure_mid_hi=param_str.safe_substitute(
                         m=0, param_ind='k + 1'),
                     pressure_general_lo=param_str.safe_substitute(
                         m='m', param_ind='lo_ind'),
                     pressure_general_hi=param_str.safe_substitute(
                         m='m', param_ind='hi_ind')
                     )
    ).safe_substitute(reac_ind=reac_ind)

    # and return
    return [k_gen.knl_info(name='rateconst_plog', instructions=instructions,
                           pre_instructions=[
                               k_gen.TINV_PREINST_KEY, k_gen.TLOG_PREINST_KEY, k_gen.PLOG_PREINST_KEY],
                           var_name=reac_ind, kernel_data=kernel_data,
                           maps=maps, extra_inames=extra_inames, indicies=indicies,
                           extra_subs={'reac_ind': reac_ind})]


def get_reduced_pressure_kernel(eqs, loopy_opts, rate_info, test_size=None):
    """Generates instructions, kernel arguements, and data for the reduced
    pressure evaluation kernel

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    reac_ind = 'i'
    conp_eqs = eqs['conp']  # conp / conv irrelevant for rates

    # create the various necessary arrays

    kernel_data = []
    if test_size == 'problem_size':
        kernel_data.append(lp.ValueArg(test_size, dtype=np.int32))

    map_instructs = []
    inmaps = {}

    # falloff index mapping
    indicies = ['${reac_ind}', 'j']
    k_gen.handle_indicies(rate_info['fall']['map'], '${reac_ind}', inmaps, kernel_data,
                          inmap_name='fall_map',
                          force_zero=True)

    # simple arrhenius rates
    kf_arr, kf_str, map_inst = lp_utils.get_loopy_arg('kf',
                                                      indicies=indicies,
                                                      dimensions=[
                                                          rate_info['Nr'], test_size],
                                                      order=loopy_opts.order,
                                                      map_name=inmaps)
    if '${reac_ind}' in map_inst:
        map_instructs.append(map_inst['${reac_ind}'])

    # simple arrhenius rates using falloff (alternate) parameters
    kf_fall_arr, kf_fall_str, _ = lp_utils.get_loopy_arg('kf_fall',
                                                         indicies=indicies,
                                                         dimensions=[
                                                             rate_info['fall']['num'], test_size],
                                                         order=loopy_opts.order)

    # temperatures
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)

    # create a Pr array
    Pr_arr, Pr_str, _ = lp_utils.get_loopy_arg('Pr',
                                               indicies=indicies,
                                               dimensions=[
                                                   rate_info['fall']['num'], test_size],
                                               order=loopy_opts.order)

    # third-body concentrations
    thd_indexing_str = '${reac_ind}'
    thd_index_map = {}

    # check if a mapping is needed
    thd_inds = rate_info['thd']['map']
    fall_inds = rate_info['fall']['map']
    if not np.array_equal(thd_inds, fall_inds):
        # add a mapping for falloff index -> third body conc index
        thd_map_name = 'thd_map'
        thd_map = np.where(np.in1d(thd_inds, fall_inds))[0].astype(np.int32)

        k_gen.handle_indicies(thd_map, thd_indexing_str, thd_index_map,
                              kernel_data, inmap_name=thd_map_name, force_map=True)

    # create the array
    indicies = ['${reac_ind}', 'j']
    thd_conc_lp, thd_conc_str, map_inst = lp_utils.get_loopy_arg('thd_conc',
                                                                 indicies=indicies,
                                                                 dimensions=(
                                                                     rate_info['thd']['num'], test_size),
                                                                 order=loopy_opts.order,
                                                                 map_name=thd_index_map,
                                                                 map_result='thd_i')
    if '${reac_ind}' in map_inst:
        map_instructs.append(map_inst['${reac_ind}'])

    # and finally the falloff types
    fall_type_lp, fall_type_str = __1Dcreator('fall_type', rate_info['fall']['ftype'],
                                              scope=scopes.GLOBAL)

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
if fall_type[${reac_ind}]
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
    pr_instructions = Template(pr_instructions.safe_substitute(
        kf_str=kf_str,
        kf_fall_str=kf_fall_str,
        Pr_str=Pr_str,
        Pr_eq=Pri_eqn)).safe_substitute(
        reac_ind=reac_ind)

    # and finally return the resulting info
    return [k_gen.knl_info('red_pres',
                           instructions=pr_instructions,
                           var_name=reac_ind,
                           kernel_data=kernel_data,
                           indicies=np.array(
                               range(rate_info['fall']['num']), dtype=np.int32),
                           maps=map_instructs,
                           extra_subs={'reac_ind': reac_ind})]


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
    rate_list : list of :class:`knl_info`
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
    <> Fcent_val = fmax(${Fcent_temp}, 1e-300d) {id=Fcv, dep=Fcent_decl3}
    <>Pr_val = fmax(${Pr_str}, 1e-300d) {id=Prv}
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
                           manglers=[
                               k_gen.MangleGen('fmax',
                                               (np.float64, np.float64),
                                               np.float64)])]


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
    rate_list : list of :class:`knl_info`
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
        mapstore.check_and_add_transform(namestore.arr, namestore.num_sri)

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
    <>T = ${T_str}
    <>Pr_val = 1e-300 {id=Pri}
    if ${pr_str} > 1e-300
        Pr_val = ${pr_str} {id=Prv, dep=Pri}
    end
    <>logPr = log10(Pr_val) {dep=Prv}
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
                           var_name=var_name,
                           kernel_data=kernel_data,
                           mapstore=mapstore)]


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
    rate_list : list of :class:`knl_info`
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


def get_simple_arrhenius_rates(eqs, loopy_opts, rate_info, test_size=None,
                               falloff=False):
    """Generates instructions, kernel arguements, and data for specialized forms
    of simple (non-pressure dependent) rate constants

    Parameters
    ----------
    eqs : dict
        Sympy equations / variables for constant pressure / constant volume systems
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    rate_info : dict
        The output of :method:`assign_rates` for this mechanism
    test_size : int
        If not none, this kernel is being used for testing.
        Hence we need to size the arrays accordingly
    falloff : bool
        If true, generate rate kernel for the falloff rates, i.e. either
        k0 or kinf depending on whether the reaction is falloff or chemically activated

    Returns
    -------
    rate_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    # find options, sizes, etc.
    if falloff:
        tag = 'fall'
        name_mod = '_fall'
        Nr = rate_info['fall']['num']
    else:
        tag = 'simple'
        name_mod = ''
        Nr = rate_info['Nr']

    # first assign the reac types, parameters
    full = loopy_opts.rate_spec == lp_utils.RateSpecialization.full
    hybrid = loopy_opts.rate_spec == lp_utils.RateSpecialization.hybrid
    fixed = loopy_opts.rate_spec == lp_utils.RateSpecialization.fixed
    separated_kernels = loopy_opts.rate_spec_kernels
    if fixed and separated_kernels:
        separated_kernels = False
        logging.warn('Cannot use separated kernels with a fixed RateSpecialization, '
                     'disabling...')

    # define loopy arrays
    A_name = 'A{}'.format(name_mod)
    A_lp = lp.TemporaryVariable(A_name, shape=lp.auto,
                                initializer=rate_info[tag]['A'],
                                read_only=True, scope=scopes.GLOBAL)
    b_name = 'beta{}'.format(name_mod)
    b_lp = lp.TemporaryVariable(b_name, shape=lp.auto,
                                initializer=rate_info[tag]['b'],
                                read_only=True, scope=scopes.GLOBAL)
    Ta_name = 'Ta{}'.format(name_mod)
    Ta_lp = lp.TemporaryVariable(Ta_name, shape=lp.auto,
                                 initializer=rate_info[tag]['Ta'],
                                 read_only=True, scope=scopes.GLOBAL)
    T_arr = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)

    simple_arrhenius_data = []
    if test_size == 'problem_size':
        simple_arrhenius_data += [lp.ValueArg(test_size, dtype=np.int32)]
    simple_arrhenius_data.extend([A_lp, b_lp, Ta_lp, T_arr])

    # if we need the rtype array, add it
    if not separated_kernels and not fixed:
        rtype_lp = lp.TemporaryVariable('rtype{}'.format(name_mod), shape=lp.auto,
                                        initializer=rate_info[tag]['type'],
                                        read_only=True, scope=scopes.PRIVATE)
        simple_arrhenius_data.append(rtype_lp)

    # the reaction index variable
    reac_ind = 'i'

    # put rateconst info args in dict for unpacking convenience
    extra_args = {'kernel_data': simple_arrhenius_data,
                  'var_name': reac_ind,
                  'extra_subs': {'reac_ind': reac_ind}}

    default_preinstructs = [k_gen.TINV_PREINST_KEY, k_gen.TLOG_PREINST_KEY]

    # generic kf assigment str
    kf_assign = Template("${kf_str} = ${rate}")
    expkf_assign = Template("${kf_str} = exp(${rate})")

    def get_instructions(rtype, index='i'):
        # get rate equations
        rate_eqn_pre = get_rate_eqn(eqs, index=index)
        index_str = '[{}]'.format(index)
        rate_eqn_pre = sp_utils.sanitize(rate_eqn_pre,
                                         symlist={
                                             'A' + index_str: A_name + index_str,
                                             'Ta' + index_str: Ta_name + index_str,
                                             'beta' + index_str: b_name + index_str,
                                         })

        # the simple formulation
        if fixed or (hybrid and rtype == 2) or (full and rtype == 4):
            return expkf_assign.safe_substitute(rate=str(rate_eqn_pre))

        # otherwise check type and return appropriate
        if rtype == 0:
            return kf_assign.safe_substitute(rate=A_name + index_str)
        elif rtype == 1:
            return Template(
                """
                <> T_val = T_arr[j] {id=a1}
                <> negval = ${b_name}[${index}] < 0
                if negval
                    T_val = T_inv {id=a2, dep=a1}
                end
                <>kf_temp = ${A_name}[${index}] {id=a3, dep=a2}
                ${beta_iter}
                """).safe_substitute(index=index, A_name=A_name,
                                     b_name=b_name)
        elif rtype == 2:
            return expkf_assign.safe_substitute(
                rate=str(rate_eqn_pre.subs(Ta_name + index_str, 0)))
        elif rtype == 3:
            return expkf_assign.safe_substitute(
                rate=str(rate_eqn_pre.subs(b_name + index_str, 0)))

    # various specializations of the rate form
    specializations = {}
    i_a_only = k_gen.knl_info(name='a_only{}'.format(name_mod),
                              instructions='',
                              **extra_args)

    # the default is a single multiplication
    # if needed, this will be expanded to a for-loop multiplier
    beta_iter = "${kf_str} = kf_temp * T_val {id=a4, dep=a3}"
    i_beta_int = k_gen.knl_info(name='beta_int{}'.format(name_mod),
                                instructions='',
                                pre_instructions=[k_gen.TINV_PREINST_KEY],
                                **extra_args)
    i_beta_exp = k_gen.knl_info('rateconst_beta_exp{}'.format(name_mod),
                                instructions='',
                                pre_instructions=default_preinstructs,
                                **extra_args)
    i_ta_exp = k_gen.knl_info('rateconst_ta_exp{}'.format(name_mod),
                              instructions='',
                              pre_instructions=default_preinstructs,
                              **extra_args)
    i_full = k_gen.knl_info('rateconst_full{}'.format(name_mod),
                            instructions='',
                            pre_instructions=default_preinstructs,
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

    if not separated_kernels and not fixed:
        # need to enclose each branch in it's own if statement
        if len(specializations) > 1:
            instruction_list = []
            inds = set()
            for i in specializations:
                instruction_list.append(
                    'if rtype{1}[i] == {0}'.format(i, name_mod))
                instruction_list.extend(['\t' + x for x in get_instructions(
                    i).split('\n') if x.strip()])
                instruction_list.append('end')
        # and combine them
        specializations = {-1: k_gen.knl_info('rateconst_singlekernel{}'.format(name_mod),
                                              instructions='\n'.join(
                                                  instruction_list),
                                              pre_instructions=[
                                                  k_gen.TINV_PREINST_KEY, k_gen.TLOG_PREINST_KEY],
                                              **extra_args)}

    out_specs = {}
    # and do some finalizations for the specializations
    for rtype, info in specializations.items():
        # first, get indicies
        if rtype < 0:
            # select all for joined kernel
            info.indicies = np.arange(
                0, rate_info[tag]['type'].size, dtype=np.int32)
        else:
            # otherwise choose just our rtype
            info.indicies = np.where(
                rate_info[tag]['type'] == rtype)[0].astype(dtype=np.int32)

        if not info.indicies.size:
            # kernel doesn't act on anything, don't add it to output
            continue

        # check maxb / iteration
        if (separated_kernels and (info.name == i_beta_int.name)) or \
                (not separated_kernels and not fixed):
            # find b's for this type
            bvals = rate_info[tag]['b'][np.where(rate_info[tag]['type'] == 1)]
            if bvals.size:
                # find max b exponent
                maxb = np.max(np.abs(bvals))
                # if we need to iterate
                if maxb > 1:
                    # add an extra iname, and the resulting iteraton loop
                    info.extra_inames.append(('k', '0 <= k < b_end'))
                    beta_iter = """
                <> b_end = abs(${b_name}[${index}])
                for k
                    kf_temp = kf_temp * T_val {id=a4, dep=a3}
                end
                ${kf_str} = kf_temp {dep=a4}
                    """

        # check if we need an input map
        maps = {}
        dummy_in_index = 'i_in'  # used as input map if needed
        dummy_out_index = 'i_out'
        inmap_name = 'input_map'
        outmap_name = 'output_map'
        # need to store these here for the moment, until
        # _after_ we check for the output map
        out_inds = None
        if not falloff:
            out_inds = rate_info[tag]['map'][info.indicies]
        info.indicies = k_gen.handle_indicies(info.indicies, info.var_name,
                                              maps, info.kernel_data, inmap_name=inmap_name,
                                              output_indicies=out_inds,
                                              outmap_name=outmap_name)

        index = info.var_name
        # handle input mapping info if needed
        if info.var_name in maps:
            info.maps.append(lp_utils.generate_map_instruction(
                info.var_name, dummy_in_index, inmap_name))
            # change the var_name
            index = dummy_in_index
        kf_ind = index
        # handle output mapping info if needed
        if info.var_name + '_out' in maps:
            # need to map from the actual loop index
            info.maps.append(lp_utils.generate_map_instruction(
                info.var_name, dummy_out_index, outmap_name))
            # change the kf_ind
            kf_ind = dummy_out_index

        # get the proper kf indexing / array
        kf_arr, kf_str, map_result = lp_utils.get_loopy_arg('kf' + name_mod,
                                                            indicies=[
                                                                kf_ind, 'j'],
                                                            dimensions=[
                                                                Nr, test_size],
                                                            order=loopy_opts.order)
        info.kernel_data.append(kf_arr)

        if rtype >= 0:
            info.instructions = get_instructions(rtype, index=index)

        # substitute in whatever beta_iter / kf_str we found
        info.instructions = Template(info.instructions).safe_substitute(
            beta_iter=beta_iter)
        info.instructions = Template(info.instructions).safe_substitute(
            kf_str=kf_str,
            A_name=A_name,
            b_name=b_name,
            Ta_name=Ta_name,
            index=index)
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
    kernel_gen : :class:`wrapping_kernel_generator`
        The generator responsible for creating the resulting code

    """

    # figure out rates and info
    rate_info = assign_rates(reacs, specs, loopy_opts.rate_spec)

    # set test size
    if test_size is None:
        test_size = 'problem_size'

    # create the namestore
    nstore = arc.NameStore(loopy_opts, rate_info, test_size)

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
                                           nstore, test_size=test_size))

    # check for chebyshev
    if rate_info['cheb']['num']:
        __add_knl(get_cheb_arrhenius_rates(eqs, loopy_opts,
                                           nstore, test_size=test_size))

    # check for third body terms
    if rate_info['thd']['num']:
        # add the initial third body conc eval kernel
        __add_knl(get_thd_body_concs(eqs, loopy_opts,
                                     nstore, test_size))

    # check for falloff
    if rate_info['fall']['num']:
        # get the falloff rates
        __add_knl(get_simple_arrhenius_rates(eqs, loopy_opts,
                                             nstore, test_size=test_size, falloff=True))
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

    depends_on = []
    # check for reverse rates
    if rate_info['rev']['num']:
        # add the 'b' eval
        __add_knl(polyfit_kernel_gen('b', eqs['conp'], specs, loopy_opts,
                                     test_size))
        # addd the 'b' eval to depnediencies
        depends_on.append(kernels[-1])
        # add Kc / rev rates
        __add_knl(get_rev_rates(eqs, loopy_opts,
                                nstore, test_size=test_size))

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
        __add_knl(polyfit_kernel_gen('h', eqs['conp'], specs, loopy_opts,
                                     test_size))
        __add_knl(polyfit_kernel_gen('cp', eqs['conp'], specs, loopy_opts,
                                     test_size))
    else:
        # and u / cv
        __add_knl(polyfit_kernel_gen('u', eqs['conv'], specs, loopy_opts,
                                     test_size))
        __add_knl(polyfit_kernel_gen('cv', eqs['conv'], specs, loopy_opts,
                                     test_size))
    # add the thermo kernels to our dependencies
    depends_on.extend(kernels[-2:])
    # and temperature rates
    __add_knl(get_temperature_rate(eqs, loopy_opts,
                                   nstore, test_size=test_size, conp=conp))

    # get a wrapper for the dependecies
    thermo_wrap = k_gen.wrapping_kernel_generator(name='chem_utils_kernel',
                                                  loopy_opts=loopy_opts,
                                                  kernels=depends_on,
                                                  input_arrays=['T_arr'],
                                                  output_arrays=[
                                                      'h', 'cp'] if conp else ['u', 'cv'],
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

    input_arrays = ['T_arr', 'P_arr', 'conc', 'wdot']
    output_arrays = ['wdot']
    if output_full_rop:
        output_arrays += ['rop_fwd']
        if rate_info['rev']['num']:
            output_arrays += ['rop_rev']
        if rate_info['thd']['num']:
            output_arrays += ['pres_mod']
        output_arrays += ['rop_net']
    return k_gen.wrapping_kernel_generator(
        loopy_opts=loopy_opts,
        name='species_rates_kernel',
        kernels=kernels,
        external_kernels=depends_on,
        depends_on=[thermo_wrap],
        input_arrays=input_arrays,
        output_arrays=output_arrays,
        init_arrays={'wdot': 0},
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
    Tinv_sym = sp.Symbol('T_inv')
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


def polyfit_kernel_gen(nicename, eqs, specs,
                       loopy_opts, test_size=None):
    """Helper function that generates kernels for
       evaluation of various thermodynamic species properties

    Parameters
    ----------
    nicename : str
        The variable name to use in generated code
    eqs : dict of `sympy.Symbol`
        Dictionary defining conditional equations for the variables (keys)
    specs : list of `SpecInfo`
        List of species in the mechanism.
    lang : {'c', 'cuda', 'opencl'}
        Programming language.
    loopy_opts : `loopy_options` object
        A object containing all the loopy options to execute
    test_size : int
        If not None, this kernel is being used for testing.

    Returns
    -------
    knl : :class:`loopy.LoopKernel`
        The generated loopy kernel for code generation / testing

    """

    if test_size is None:
        test_size = 'problem_size'

    if loopy_opts.width is not None and loopy_opts.depth is not None:
        raise Exception('Cannot specify both SIMD/SIMT width and depth')

    # mapping of nicename -> varname
    var_maps = {'cp': '{C_p}[k]',
                'h': 'H[k]', 'cv': '{C_v}[k]',
                'u': 'U[k]', 'b': 'B[k]'}
    varname = var_maps[nicename]

    var = next(v for v in eqs.keys() if str(v) == varname)
    eq = eqs[var]
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

    # get correctly ordered arrays / strings
    indicies = ['k', '${param_val}']
    a_lo_lp = lp.TemporaryVariable('a_lo', shape=a_lo.shape, initializer=a_lo,
                                   scope=scopes.GLOBAL, read_only=True)
    a_hi_lp = lp.TemporaryVariable('a_hi', shape=a_hi.shape, initializer=a_hi,
                                   scope=scopes.GLOBAL, read_only=True)
    a_lo_str = Template('a_lo[' + ','.join(indicies) + ']')
    a_hi_str = Template('a_hi[' + ','.join(indicies) + ']')
    T_mid_lp = lp.TemporaryVariable('T_mid', shape=T_mid.shape, initializer=T_mid, read_only=True,
                                    scope=scopes.GLOBAL)

    k = sp.Idx('k')
    lo_eq_str = str(eq.subs([(sp.IndexedBase('a')[k, i],
                              a_lo_str.safe_substitute(param_val=i)) for i in range(poly_dim)]))
    hi_eq_str = str(eq.subs([(sp.IndexedBase('a')[k, i],
                              a_hi_str.safe_substitute(param_val=i)) for i in range(poly_dim)]))

    target = lp_utils.get_target(loopy_opts.lang)

    # create the input arrays arrays
    T_lp = lp.GlobalArg('T_arr', shape=(test_size,), dtype=np.float64)
    out_lp, out_str, _ = lp_utils.get_loopy_arg(nicename,
                                                indicies=['k', 'j'],
                                                dimensions=(Ns, test_size),
                                                order=loopy_opts.order)

    knl_data = [a_lo_lp, a_hi_lp, T_mid_lp, T_lp, out_lp]

    if test_size == 'problem_size':
        knl_data = [lp.ValueArg('problem_size', dtype=np.int32)] + knl_data

    return k_gen.knl_info(instructions=Template("""
        for j
            <> T = T_arr[j]
            for k
                if T < T_mid[k]
                    ${out_str} = ${lo_eq}
                else
                    ${out_str} = ${hi_eq}
                end
            end
        end
        """).safe_substitute(out_str=out_str, lo_eq=lo_eq_str, hi_eq=hi_eq_str),
                          kernel_data=knl_data,
                          name='eval_{}'.format(nicename),
                          parameters={'R_u': chem.RU},
                          var_name='k',
                          indicies=k_gen.handle_indicies(np.arange(Ns, dtype=np.int32), 'k', None, []))


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

    file_prefix = ''
    if auto_diff:
        file_prefix = 'ad_'

    target = lp_utils.get_target(loopy_opts.lang)

    # generate the kernels
    conp_eqs = eqs['conp']
    conv_eqs = eqs['conv']

    nicenames = ['cp', 'h', 'cv', 'u', 'b']
    kernels = []
    headers = []
    code = []
    for nicename in nicenames:
        eq = conp_eqs if nicename in ['h', 'cp'] else conv_eqs
        kernels.append(polyfit_kernel_gen(nicename,
                                          eq, specs, loopy_opts, test_size))

    return k_gen.wrapping_kernel_generator(
        loopy_opts=loopy_opts,
        name='chem_utils',
        kernels=kernels,
        input_arrays=['T_arr'],
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
