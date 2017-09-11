# system
import os
from collections import OrderedDict, defaultdict
import subprocess
import sys
import shutil
from multiprocessing import cpu_count

# local imports
from ..core.rate_subs import (write_specrates_kernel, get_rate_eqn,
                              assign_rates, get_simple_arrhenius_rates,
                              get_plog_arrhenius_rates, get_lind_kernel,
                              get_cheb_arrhenius_rates, get_thd_body_concs,
                              get_reduced_pressure_kernel, get_sri_kernel,
                              get_troe_kernel, get_rev_rates, get_rxn_pres_mod,
                              get_rop, get_rop_net, get_spec_rates,
                              get_temperature_rate, get_concentrations,
                              get_molar_rates, get_extra_var_rates, reset_arrays)
from ..loopy_utils.loopy_utils import (auto_run, loopy_options,
                                       RateSpecialization,
                                       get_device_list, populate,
                                       kernel_call)
from . import TestClass, get_test_platforms
from ..core.reaction_types import reaction_type, falloff_form, thd_body_type
from ..kernel_utils import kernel_gen as k_gen
from ..core.mech_auxiliary import write_aux
from ..pywrap.pywrap_gen import generate_wrapper
from ..tests import test_utils as test_utils
from ..core import array_creator as arc

# modules
from optionloop import OptionLoop
import cantera as ct
import numpy as np
from nose.plugins.attrib import attr
from parameterized import parameterized
import pyopencl as cl


class kf_wrapper(object):
    """
    Simple wrapper that calculates Kf / Kf_fall based on order for use in a
    given function

    Parameters
    ----------
    owner : :class:`TestClass`
        The owning test class (for access to the :class:`storage`)
    function : FunctionType
        The function to call
    """

    def __init__(self, owner, function, **kwargs):
        self.store = owner.store
        self.func = function
        self.kf_val = []
        self.kf_fall_val = []
        self.kwargs = kwargs
        self.kwargs['kf'] = lambda x: np.array(
            self.kf_val[0], order=x, copy=True)
        self.kwargs['kf_fall'] = lambda x: np.array(
            np.array(self.kf_fall_val[0], order=x, copy=True))

    def __call__(self, eqs, loopy_opts, namestore, test_size):
        # check if we've found the kf / kf_fall values yet
        if not self.kf_val:
            # first we have to get the simple arrhenius rates
            # in order to evaluate the reduced pressure

            runner = test_utils.kernel_runner(
                get_simple_arrhenius_rates,
                self.store.test_size,
                {'phi': self.kwargs['phi']},
                {'falloff': True})

            self.kf_fall_val.append(
                runner(eqs, loopy_opts, namestore, test_size)[0])

            # next with regular parameters
            runner = test_utils.kernel_runner(
                get_simple_arrhenius_rates,
                self.store.test_size,
                {'phi': self.kwargs['phi']})

            self.kf_val.append(
                runner(eqs, loopy_opts, namestore, test_size)[0])

        # finally we can call the function
        return self.func(
            eqs, loopy_opts, namestore, test_size)


def parse_split_index(arr, ind, order):
    """
    A helper method to get the index of an element in a split array for all initial
    conditions

    Parameters
    ----------
    arr: :class:`numpy.ndarray`
        The split array to use
    ind: int
        The element index
    order: ['C', 'F']
        The numpy data order
    """

    # the index is a linear combination of the first and last indicies
    # in the split array
    if order == 'F':
        # For 'F' order, where vw is the vector width:
        # (0, 1, ... vw - 1) in the first index corresponds to the
        # last index = 0
        # (vw, vw+1, vw + 2, ... 2vw - 1) corresponds to the last index = 1,
        # etc.
        return (ind % arr.shape[0], slice(None), ind // arr.shape[0])
    else:
        # For 'C' order, where (s, l) corresponds to the second and last
        # index in the array:
        #
        # ((0, 0), (1, 0), (2, 0)), etc. corresponds to index (0, 1, 2)
        # for IC 0
        # ((0, 1), (1, 1), (2, 1)), etc. corresponds to index (0, 1, 2)
        # for IC 1, etc.

        return (slice(None), ind, slice(None))


class get_comparable(object):
    def __init__(self, compare_mask, ref_answer):
        self.compare_mask = compare_mask
        if not isinstance(self.compare_mask, list):
            self.compare_mask = [self.compare_mask]
        self.ref_answer = ref_answer

    def __call__(self, kc, outv, index):
        mask = self.compare_mask[index]

        # check for vectorized data order
        if outv.ndim == self.ref_answer.ndim:
            from ..loopy_utils.loopy_utils import kernel_call
            # return the default, as it can handle it
            return kernel_call('', [], compare_mask=[mask])._get_comparable(outv, 0)

        ind_list = []
        # get comparable index
        for ind in mask:
            ind_list.append(parse_split_index(outv, ind, kc.current_order))

        ind_list = zip(*ind_list)
        # filter slice arrays from parser
        for i in range(len(ind_list)):
            if all(x == slice(None) for x in ind_list[i]):
                ind_list[i] = slice(None)

        return outv[ind_list]


class SubTest(TestClass):

    def test_get_rate_eqs(self):
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        pre = get_rate_eqn(eqs)

        # check the form
        assert 'exp(' + str(pre) + \
            ')' == 'exp(A[i] - T_inv*Ta[i] + beta[i]*logT)'

        pre = get_rate_eqn(eqs, index='j')

        # check the form
        assert 'exp(' + str(pre) + \
            ')' == 'exp(A[j] - T_inv*Ta[j] + beta[j]*logT)'

    def test_assign_rates(self):
        reacs = self.store.reacs
        specs = self.store.specs
        result = assign_rates(reacs, specs, RateSpecialization.fixed)

        # test rate type
        assert np.all(result['simple']['type'] == 0)

        # import gas in cantera for testing
        gas = self.store.gas

        # test fwd / rev maps, nu, species etc.
        assert result['fwd']['num'] == gas.n_reactions
        assert np.array_equal(result['fwd']['map'], np.arange(gas.n_reactions))
        rev_inds = np.array(
            [i for i in range(gas.n_reactions) if gas.is_reversible(i)])
        assert np.array_equal(result['rev']['map'], rev_inds)
        assert result['rev']['num'] == rev_inds.size
        fwd_specs = []
        fwd_nu = []
        rev_specs = []
        rev_nu = []
        nu_sum = []
        net_nu = []
        net_specs = []
        net_num_specs = []
        reac_count = defaultdict(lambda: 0)
        spec_nu = defaultdict(lambda: [])
        spec_to_reac = defaultdict(lambda: [])
        for ireac, reac in enumerate(gas.reactions()):
            temp_nu_sum_dict = defaultdict(lambda: 0)
            for spec, nu in sorted(reac.reactants.items(), key=lambda x: gas.species_index(x[0])):
                fwd_specs.append(gas.species_index(spec))
                fwd_nu.append(nu)
                temp_nu_sum_dict[spec] -= nu
            assert result['fwd']['num_reac_to_spec'][
                ireac] == len(reac.reactants)
            for spec, nu in sorted(reac.products.items(), key=lambda x: gas.species_index(x[0])):
                if ireac in rev_inds:
                    rev_specs.append(gas.species_index(spec))
                    rev_nu.append(nu)
                temp_nu_sum_dict[spec] += nu
            if ireac in rev_inds:
                rev_ind = np.where(rev_inds == ireac)[0]
                assert result['rev']['num_reac_to_spec'][
                    rev_ind] == len(reac.products)
            temp_specs, temp_nu_sum = zip(*[(gas.species_index(x[0]), x[1]) for x in
                                            sorted(temp_nu_sum_dict.items(), key=lambda x: gas.species_index(x[0]))])
            net_specs.extend(temp_specs)
            net_num_specs.append(len(temp_specs))
            net_nu.extend(temp_nu_sum)
            nu_sum.append(sum(temp_nu_sum))
            for spec, nu in temp_nu_sum_dict.items():
                spec_ind = gas.species_index(spec)
                if nu:
                    reac_count[spec_ind] += 1
                    spec_nu[spec_ind].append(nu)
                    spec_to_reac[spec_ind].append(ireac)

        assert np.array_equal(fwd_specs, result['fwd']['reac_to_spec'])
        assert np.array_equal(fwd_nu, result['fwd']['nu'])
        assert np.array_equal(rev_specs, result['rev']['reac_to_spec'])
        assert np.array_equal(rev_nu, result['rev']['nu'])
        assert np.array_equal(nu_sum, result['net']['nu_sum'])
        assert np.array_equal(net_nu, result['net']['nu'])
        assert np.array_equal(net_num_specs, result['net']['num_reac_to_spec'])
        assert np.array_equal(net_specs, result['net']['reac_to_spec'])
        spec_inds = sorted(reac_count.keys())
        assert np.array_equal([reac_count[x] for x in spec_inds],
                              result['net_per_spec']['reac_count'])
        assert np.array_equal([y for x in spec_inds for y in spec_nu[x]],
                              result['net_per_spec']['nu'])
        assert np.array_equal([y for x in spec_inds for y in spec_to_reac[x]],
                              result['net_per_spec']['reacs'])
        assert np.array_equal(spec_inds,
                              result['net_per_spec']['map'])

        def __get_rate(reac, fall=False):
            try:
                Ea = reac.rate.activation_energy
                b = reac.rate.temperature_exponent
                if fall:
                    return None
                return reac.rate
            except:
                if not fall:
                    # want the normal rates
                    if isinstance(reac, ct.FalloffReaction) and not isinstance(reac, ct.ChemicallyActivatedReaction):
                        rate = reac.high_rate
                    else:
                        rate = reac.low_rate
                else:
                    # want the other rates
                    if isinstance(reac, ct.FalloffReaction) and not isinstance(reac, ct.ChemicallyActivatedReaction):
                        rate = reac.low_rate
                    else:
                        rate = reac.high_rate
                return rate
            return Ea, b

        def __tester(result, spec_type):
            # test return value
            assert 'simple' in result and 'cheb' in result and 'plog' in result

            # test num, map
            plog_inds = []
            cheb_inds = []
            if result['plog']['num']:
                plog_inds, plog_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                                              if isinstance(x, ct.PlogReaction)])
            if result['cheb']['num']:
                cheb_inds, cheb_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                                              if isinstance(x, ct.ChebyshevReaction)])

            def rate_checker(our_params, ct_params, rate_forms, force_act_nonlog=False):
                act_energy_ratios = []
                for ourvals, ctvals, form in zip(*(our_params, ct_params, rate_forms)):
                    # activation energy, check rate form
                    # if it's fixed specialization, or the form >= 2
                    if (spec_type == RateSpecialization.fixed or form >= 2) and not force_act_nonlog:
                        # it's in log form
                        assert np.isclose(
                            ourvals[0], np.log(ctvals.pre_exponential_factor))
                    else:
                        assert np.isclose(
                            ourvals[0], ctvals.pre_exponential_factor)
                    # temperature exponent doesn't change w/ form
                    assert np.isclose(ourvals[1], ctvals.temperature_exponent)
                    # activation energy, either the ratios should be constant or
                    # it should be zero
                    if ourvals[2] == 0 or ctvals.activation_energy == 0:
                        assert ourvals[2] == ctvals.activation_energy
                    else:
                        act_energy_ratios.append(
                            ourvals[2] / ctvals.activation_energy)
                # check that all activation energy ratios are the same
                assert np.all(
                    np.isclose(act_energy_ratios, act_energy_ratios[0]))

            # check rate values
            if plog_inds:
                assert np.array_equal(
                    result['plog']['num_P'], [len(p.rates) for p in plog_reacs])
                for i, reac_params in enumerate(result['plog']['params']):
                    for j, rates in enumerate(plog_reacs[i].rates):
                        assert np.isclose(reac_params[j][0], rates[0])
                    # plog uses a weird form, so use force_act_nonlog
                    rate_checker([rp[1:] for rp in reac_params], [rate[1] for rate in plog_reacs[i].rates],
                                 [2 for rate in plog_reacs[i].rates], force_act_nonlog=True)

            simple_inds = sorted(list(set(range(gas.n_reactions)).difference(
                set(plog_inds).union(set(cheb_inds)))))
            assert result['simple']['num'] == len(simple_inds)
            assert np.array_equal(
                result['simple']['map'], np.array(simple_inds))
            # test the simple reaction rates
            simple_reacs = [gas.reaction(i) for i in simple_inds]
            rate_checker([(result['simple']['A'][i], result['simple']['b'][i],
                           result['simple']['Ta'][i]) for i in range(result['simple']['num'])],
                         [__get_rate(reac, False) for reac in simple_reacs],
                         result['simple']['type'])

            # test the falloff (alternate) rates
            fall_reacs = [gas.reaction(i) for i in result['fall']['map']]
            rate_checker([(result['fall']['A'][i], result['fall']['b'][i],
                           result['fall']['Ta'][i]) for i in range(result['fall']['num'])],
                         [__get_rate(reac, True) for reac in fall_reacs],
                         result['fall']['type'])

        __tester(result, RateSpecialization.fixed)

        result = assign_rates(reacs, specs, RateSpecialization.hybrid)

        def test_assign(type_max, fall):
            # test rate type
            rtypes = []
            for reac in gas.reactions():
                if not (isinstance(reac, ct.PlogReaction) or isinstance(reac, ct.ChebyshevReaction)):
                    rate = __get_rate(reac, fall)
                    if rate is None:
                        continue
                    Ea = rate.activation_energy
                    b = rate.temperature_exponent
                    if Ea == 0 and b == 0:
                        rtypes.append(0)
                    elif Ea == 0 and int(b) == b:
                        rtypes.append(1)
                    elif Ea == 0:
                        rtypes.append(2)
                    elif b == 0:
                        rtypes.append(3)
                    else:
                        rtypes.append(4)
                    rtypes[-1] = min(rtypes[-1], type_max)
            return rtypes

        # test rate type
        assert np.array_equal(result['simple']['type'],
                              test_assign(2, False))
        assert np.array_equal(result['fall']['type'],
                              test_assign(2, True))
        __tester(result, RateSpecialization.hybrid)

        result = assign_rates(reacs, specs, RateSpecialization.full)

        # test rate type
        assert np.array_equal(result['simple']['type'],
                              test_assign(5, False))
        assert np.array_equal(result['fall']['type'],
                              test_assign(5, True))
        __tester(result, RateSpecialization.full)

        # ALL BELOW HERE ARE INDEPENDENT OF SPECIALIZATIONS
        if result['cheb']['num']:
            cheb_inds, cheb_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                                          if isinstance(x, ct.ChebyshevReaction)])
            assert result['cheb']['num'] == len(cheb_inds)
            assert np.array_equal(result['cheb']['map'], np.array(cheb_inds))

        if result['plog']['num']:
            plog_inds, plog_reacs = zip(*[(i, x) for i, x in enumerate(gas.reactions())
                                          if isinstance(x, ct.PlogReaction)])
            assert result['plog']['num'] == len(plog_inds)
            assert np.array_equal(result['plog']['map'], np.array(plog_inds))

        # test the thd / falloff / chem assignments
        assert np.array_equal(result['fall']['map'],
                              [i for i, x in enumerate(gas.reactions()) if (isinstance(x,
                                                                                       ct.FalloffReaction) or isinstance(x, ct.ChemicallyActivatedReaction))])
        fall_reacs = [gas.reaction(y) for y in result['fall']['map']]
        # test fall vs chemically activated
        assert np.array_equal(result['fall']['ftype'],
                              np.array([reaction_type.fall if (isinstance(x, ct.FalloffReaction) and not
                                                               isinstance(x, ct.ChemicallyActivatedReaction)) else reaction_type.chem for x in
                                        fall_reacs], dtype=np.int32) - int(reaction_type.fall))
        # test blending func
        blend_types = []
        for x in fall_reacs:
            if isinstance(x.falloff, ct.TroeFalloff):
                blend_types.append(falloff_form.troe)
            elif isinstance(x.falloff, ct.SriFalloff):
                blend_types.append(falloff_form.sri)
            else:
                blend_types.append(falloff_form.lind)
        assert np.array_equal(
            result['fall']['blend'], np.array(blend_types, dtype=np.int32))
        # test parameters
        # troe
        if result['fall']['troe']['num']:
            troe_reacs = [
                x for x in fall_reacs if isinstance(x.falloff, ct.TroeFalloff)]
            troe_par = [x.falloff.parameters for x in troe_reacs]
            troe_a, troe_T3, troe_T1, troe_T2 = [
                np.array(x) for x in zip(*troe_par)]
            assert np.array_equal(result['fall']['troe']['a'], troe_a)
            assert np.array_equal(result['fall']['troe']['T3'], troe_T3)
            assert np.array_equal(result['fall']['troe']['T1'], troe_T1)
            assert np.array_equal(result['fall']['troe']['T2'], troe_T2)
            # and map
            assert np.array_equal([fall_reacs.index(x) for x in troe_reacs],
                                  result['fall']['troe']['map'])
        # sri
        if result['fall']['sri']['num']:
            sri_reacs = [
                x for x in fall_reacs if isinstance(x.falloff, ct.SriFalloff)]
            sri_par = [x.falloff.parameters for x in sri_reacs]
            sri_a, sri_b, sri_c, sri_d, sri_e = [
                np.array(x) for x in zip(*sri_par)]
            assert np.array_equal(result['fall']['sri']['a'], sri_a)
            assert np.array_equal(result['fall']['sri']['b'], sri_b)
            assert np.array_equal(result['fall']['sri']['c'], sri_c)
            assert np.array_equal(result['fall']['sri']['d'], sri_d)
            assert np.array_equal(result['fall']['sri']['e'], sri_e)
            # and map
            assert np.array_equal([fall_reacs.index(x) for x in sri_reacs],
                                  result['fall']['sri']['map'])
        # lindemann
        if result['fall']['lind']['num']:
            assert np.array_equal(result['fall']['lind']['map'],
                                  [i for i, x in enumerate(fall_reacs) if not isinstance(x.falloff, ct.TroeFalloff)
                                   and not isinstance(x.falloff, ct.SriFalloff)])

        # and finally test the third body stuff
        # test map
        third_reac_inds = [i for i, x in enumerate(gas.reactions()) if (isinstance(x,
                                                                                   ct.FalloffReaction) or isinstance(x, ct.ChemicallyActivatedReaction)
                                                                        or isinstance(x, ct.ThreeBodyReaction))]
        assert np.array_equal(result['thd']['map'], third_reac_inds)
        # construct types, efficiencies, species, and species numbers
        thd_type = []
        thd_eff = []
        thd_sp = []
        thd_sp_num = []
        for ind in third_reac_inds:
            eff_dict = gas.reaction(ind).efficiencies
            eff = sorted(eff_dict, key=lambda x: gas.species_index(x))
            if not len(eff):
                thd_type.append(thd_body_type.unity)
            elif (len(eff) == 1 and eff_dict[eff[0]] == 1 and
                    gas.reaction(ind).default_efficiency == 0):
                thd_type.append(thd_body_type.species)
            else:
                thd_type.append(thd_body_type.mix)
            thd_sp_num.append(len(eff))
            for spec in eff:
                thd_sp.append(gas.species_index(spec))
                thd_eff.append(eff_dict[spec])
        # and test
        assert np.array_equal(
            result['thd']['type'], np.array(thd_type, dtype=np.int32))
        assert np.array_equal(result['thd']['eff'], thd_eff)
        assert np.array_equal(result['thd']['spec_num'], thd_sp_num)
        assert np.array_equal(result['thd']['spec'], thd_sp)

    def __get_eqs_and_oploop(self, do_ratespec=False, do_ropsplit=None,
                             do_conp=True, langs=['opencl'], do_vector=True):

        platforms = get_test_platforms(do_vector=do_vector, langs=langs)
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        oploop = [('order', ['C', 'F']),
                  ('auto_diff', [False])
                  ]
        if do_ratespec:
            oploop += [
                ('rate_spec', [x for x in RateSpecialization]),
                ('rate_spec_kernels', [True, False])]
        if do_ropsplit:
            oploop += [
                ('rop_net_kernels', [True])]
        if do_conp:
            oploop += [('conp', [True, False])]
        oploop += [('knl_type', ['map'])]
        out = None
        for p in platforms:
            val = OptionLoop(OrderedDict(p + oploop))
            if out is None:
                out = val
            else:
                out = out + val

        return eqs, out

    def __generic_rate_tester(self, func, kernel_calls, do_ratespec=False,
                              do_ropsplit=None, do_conp=False, **kw_args):
        """
        A generic testing method that can be used for rate constants, third bodies,
        ...

        Parameters
        ----------
        func : function
            The function to test
        kernel_calls : :class:`kernel_call` or list thereof
            Contains the masks and reference answers for kernel testing
        do_ratespec : bool
            If true, test rate specializations and kernel splitting for simple rates
        do_ropsplit : bool
            If true, test kernel splitting for rop_net
        """

        eqs, oploop = self.__get_eqs_and_oploop(
            do_ratespec, do_ropsplit, do_conp=do_conp)

        reacs = self.store.reacs
        specs = self.store.specs

        exceptions = ['device', 'conp']

        for i, state in enumerate(oploop):
            if state['width'] is not None and state['depth'] is not None:
                continue
            opt = loopy_options(**{x: state[x] for x in
                                   state if x not in exceptions})

            # find rate info
            rate_info = assign_rates(reacs, specs, opt.rate_spec)

            conp = True
            if 'conp' in kw_args:
                conp = kw_args['conp']

            # create namestore
            namestore = arc.NameStore(opt, rate_info, conp,
                                      self.store.test_size)
            # create the kernel info
            infos = func(eqs, opt, namestore,
                         test_size=self.store.test_size, **kw_args)

            if not isinstance(infos, list):
                try:
                    infos = list(infos)
                except:
                    infos = [infos]

            # create a dummy kernel generator
            knl = k_gen.make_kernel_generator(
                name='spec_rates',
                loopy_opts=opt,
                kernels=infos,
                test_size=self.store.test_size
            )
            knl._make_kernels()

            # create a list of answers to check
            try:
                for kc in kernel_calls:
                    kc.set_state(knl.array_split, state['order'])
            except:
                kernel_calls.set_state(knl.array_split, state['order'])

            assert auto_run(knl.kernels, kernel_calls, device=opt.device), \
                'Evaluate {} rates failed'.format(func.__name__)

    def __test_rateconst_type(self, rtype):
        """
        Performs tests for a single reaction rate type

        Parameters
        ----------
        rtype : {'simple', 'plog', 'cheb'}
            The reaction type to test
        """

        phi = self.store.phi_cp
        P = self.store.P
        ref_const = self.store.fwd_rate_constants

        reacs = self.store.reacs

        masks = {
            'simple': (
                [np.array([i for i, x in enumerate(reacs)
                           if x.match((
                               reaction_type.elementary,
                               reaction_type.fall,
                               reaction_type.chem))])],
                get_simple_arrhenius_rates),
            'plog': (
                [np.array([i for i, x in enumerate(reacs)
                           if x.match((reaction_type.plog,))])],
                get_plog_arrhenius_rates),
            'cheb': (
                [np.array([i for i, x in enumerate(reacs)
                           if x.match((reaction_type.cheb,))])],
                get_cheb_arrhenius_rates)}

        args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                'kf': lambda x: np.zeros_like(ref_const, order=x)}
        if rtype != 'simple':
            args['P_arr'] = P

        kw_args = {}
        if rtype == 'plog':
            kw_args['maxP'] = np.max([
                len(rxn.rates) for rxn in self.store.gas.reactions()
                if isinstance(rxn, ct.PlogReaction)])
        elif rtype == 'cheb':
            kw_args['maxP'] = np.max([
                rxn.nPressure for rxn in self.store.gas.reactions()
                if isinstance(rxn, ct.ChebyshevReaction)])
            kw_args['maxT'] = np.max([
                rxn.nTemperature for rxn in self.store.gas.reactions()
                if isinstance(rxn, ct.ChebyshevReaction)])

        def __simple_post(kc, out):
            if len(out[0].shape) == 3:
                # vectorized data order
                # the index is the combination of the first axis and the last
                for i, thd_i in enumerate(self.store.thd_inds):
                    f, s, l = parse_split_index(
                        out[0], thd_i, kc.current_order)
                    if kc.current_order == 'F':
                        out[0][f, s, l] *= self.store.ref_pres_mod[:, i]
                    else:
                        out[0][f, s, l] *= np.reshape(
                            self.store.ref_pres_mod[:, i],
                            out[0][f, s, l].shape,
                            order='C')
            else:
                out[0][:, self.store.thd_inds] *= self.store.ref_pres_mod

        compare_mask, rate_func = masks[rtype]
        post = None if rtype != 'simple' else __simple_post

        # see if mechanism has this type
        if not compare_mask[0].size:
            return

        compare_mask = [get_comparable(compare_mask, ref_const)]

        # create the kernel call
        kc = kernel_call(rtype,
                         ref_const,
                         compare_mask=compare_mask,
                         post_process=post, **args)

        self.__generic_rate_tester(
            rate_func, kc, do_ratespec=rtype == 'simple', **kw_args)

    @attr('long')
    def test_simple_rate_constants(self):
        self.__test_rateconst_type('simple')

    @attr('long')
    def test_plog_rate_constants(self):
        self.__test_rateconst_type('plog')

    @attr('long')
    def test_cheb_rate_constants(self):
        self.__test_rateconst_type('cheb')

    @attr('long')
    def test_set_concentrations(self):
        phi = self.store.phi_cp
        P = self.store.P
        V = self.store.V
        ref_ans = self.store.concs.copy()

        # do conp
        args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                'P_arr': lambda x: np.array(P, order=x, copy=True)}

        # create the kernel call
        kc = kernel_call('eval_', ref_ans, **args)
        self.__generic_rate_tester(get_concentrations, kc, conp=True)

        # do conv
        phi = self.store.phi_cv
        args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                'V_arr': lambda x: np.array(V, order=x, copy=True)}

        # create the kernel call
        kc = kernel_call('eval_', ref_ans, **args)
        self.__generic_rate_tester(get_concentrations, kc, conp=False)

    @attr('long')
    def test_thd_body_concs(self):
        phi = self.store.phi_cp
        concs = self.store.concs
        P = self.store.P
        ref_ans = self.store.ref_thd.copy()
        args = {'conc': lambda x: np.array(concs, order=x, copy=True),
                'phi': lambda x: np.array(phi, order=x, copy=True),
                'P_arr': lambda x: np.array(P, order=x, copy=True)}

        # create the kernel call
        kc = kernel_call('eval_thd_body_concs', ref_ans, **args)
        self.__generic_rate_tester(get_thd_body_concs, kc)

    @attr('long')
    def test_reduced_pressure(self):
        phi = self.store.phi_cp.copy()
        ref_thd = self.store.ref_thd.copy()
        ref_ans = self.store.ref_Pr.copy()
        args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                'thd_conc': lambda x: np.array(ref_thd, order=x, copy=True),
                }

        wrapper = kf_wrapper(self, get_reduced_pressure_kernel, **args)

        # create the kernel call
        kc = kernel_call('pred', ref_ans, **wrapper.kwargs)
        self.__generic_rate_tester(wrapper, kc, do_ratespec=True)

    @attr('long')
    def test_sri_falloff(self):
        ref_phi = self.store.phi_cp
        ref_Pr = self.store.ref_Pr
        ref_ans = self.store.ref_Sri.copy().squeeze()
        args = {'Pr': lambda x: np.array(ref_Pr, order=x, copy=True),
                'phi': lambda x: np.array(ref_phi, order=x, copy=True),
                }

        # get SRI reaction mask
        sri_mask = np.where(
            np.in1d(self.store.fall_inds, self.store.sri_inds))[0]
        if not sri_mask.size:
            return
        # create the kernel call
        kc = kernel_call('fall_sri', ref_ans, out_mask=[0],
                         compare_mask=[get_comparable(sri_mask, ref_ans)], **args)
        self.__generic_rate_tester(get_sri_kernel, kc)

    @attr('long')
    def test_troe_falloff(self):
        phi = self.store.phi_cp
        ref_Pr = self.store.ref_Pr
        ref_ans = self.store.ref_Troe.copy().squeeze()
        args = {'Pr': lambda x: np.array(ref_Pr, order=x, copy=True),
                'phi': lambda x: np.array(phi, order=x, copy=True),
                }

        # get Troe reaction mask
        troe_mask = np.where(
            np.in1d(self.store.fall_inds, self.store.troe_inds))[0]
        if not troe_mask.size:
            return
        # create the kernel call
        kc = kernel_call('fall_troe', ref_ans, out_mask=[0],
                         compare_mask=[get_comparable(troe_mask, ref_ans)], **args)
        self.__generic_rate_tester(get_troe_kernel, kc)

    @attr('long')
    def test_lind_falloff(self):
        ref_ans = self.store.ref_Lind.copy()
        # get lindeman reaction mask
        lind_mask = np.where(
            np.in1d(self.store.fall_inds, self.store.lind_inds))[0]
        if not lind_mask.size:
            return
        # need a seperate answer mask to deal with the shape difference
        # in split arrays
        ans_mask = [np.arange(self.store.lind_inds.size, dtype=np.int32)]
        # create the kernel call
        kc = kernel_call('fall_lind', ref_ans,
                         compare_mask=[get_comparable(lind_mask, ref_ans)],
                         ref_ans_compare_mask=[get_comparable(ans_mask, ref_ans)])
        self.__generic_rate_tester(get_lind_kernel, kc)

    @attr('long')
    def test_rev_rates(self):
        ref_fwd_rates = self.store.fwd_rate_constants.copy()
        ref_kc = self.store.equilibrium_constants.copy()
        ref_B = self.store.ref_B_rev.copy()
        ref_rev = self.store.rev_rate_constants.copy()
        args = {'b': lambda x: np.array(ref_B, order=x, copy=True),
                'kf': lambda x: np.array(ref_fwd_rates, order=x, copy=True)}

        # create the dictionary for nu values stating if all integer
        allint = {'net':
                  np.allclose(np.mod(self.store.gas.reactant_stoich_coeffs() -
                                     self.store.gas.product_stoich_coeffs(),
                                     1), 0)}

        # create the kernel call
        kc = kernel_call('Kc', [ref_kc, ref_rev],
                         out_mask=[0, 1], **args)

        self.__generic_rate_tester(get_rev_rates, kc, allint=allint)

    @attr('long')
    def test_pressure_mod(self):
        ref_pres_mod = self.store.ref_pres_mod.copy()
        ref_Pr = self.store.ref_Pr.copy()
        ref_Fi = self.store.ref_Fall.copy()
        ref_thd = self.store.ref_thd.copy()

        args = {'Fi': lambda x: np.array(ref_Fi, order=x, copy=True),
                'thd_conc': lambda x: np.array(ref_thd, order=x, copy=True),
                'Pr': lambda x: np.array(ref_Pr, order=x, copy=True),
                'pres_mod': lambda x: np.zeros_like(ref_pres_mod, order=x)}

        thd_only_inds = np.where(
            np.logical_not(np.in1d(self.store.thd_inds,
                                   self.store.fall_inds)))[0]
        fall_only_inds = np.where(np.in1d(self.store.thd_inds,
                                          self.store.fall_inds))[0]

        # create the kernel call
        kc = [kernel_call('ci_thd', [ref_pres_mod],
                          out_mask=[0],
                          compare_mask=[get_comparable(thd_only_inds, ref_pres_mod)],
                          input_mask=['Fi', 'Pr'],
                          strict_name_match=True, **args),
              kernel_call('ci_fall', [ref_pres_mod],
                          out_mask=[0],
                          compare_mask=[get_comparable(
                            fall_only_inds, ref_pres_mod)],
                          input_mask=['thd_conc'],
                          strict_name_match=True, **args)]
        self.__generic_rate_tester(get_rxn_pres_mod, kc)

    @attr('long')
    def test_rop(self):
        fwd_rate_constants = self.store.fwd_rate_constants.copy()
        rev_rate_constants = self.store.rev_rate_constants.copy()
        fwd_rxn_rate = self.store.fwd_rxn_rate.copy()
        rev_rxn_rate = self.store.rev_rxn_rate.copy()
        conc = self.store.concs.copy()

        # create the dictionary for nu values stating if all integer
        allint = {'net':
                  np.allclose(np.mod(self.store.gas.product_stoich_coeffs(),
                                     1), 0) and
                  np.allclose(np.mod(self.store.gas.reactant_stoich_coeffs(),
                                     1), 0)}

        args = {'kf': lambda x:
                np.array(fwd_rate_constants, order=x, copy=True),
                'kr': lambda x:
                np.array(rev_rate_constants, order=x, copy=True),
                'conc': lambda x:
                np.array(conc, order=x, copy=True)}

        kc = [kernel_call('rop_eval_fwd', [fwd_rxn_rate],
                          input_mask=['kr'],
                          strict_name_match=True, **args),
              kernel_call('rop_eval_rev', [rev_rxn_rate],
                          input_mask=['kf'],
                          strict_name_match=True, **args)]
        self.__generic_rate_tester(get_rop, kc, allint=allint)

    @attr('long')
    def test_rop_net(self):
        fwd_removed = self.store.fwd_rxn_rate.copy()
        fwd_removed[:, self.store.thd_inds] = fwd_removed[
            :, self.store.thd_inds] / self.store.ref_pres_mod
        thd_in_rev = np.where(
            np.in1d(self.store.thd_inds, self.store.rev_inds))[0]
        rev_update_map = np.where(
            np.in1d(self.store.rev_inds, self.store.thd_inds[thd_in_rev]))[0]
        rev_removed = self.store.rev_rxn_rate.copy()
        rev_removed[:, rev_update_map] = rev_removed[
            :, rev_update_map] / self.store.ref_pres_mod[:, thd_in_rev]

        # remove ref pres mod = 0 (this is a 0 rate)
        fwd_removed[np.where(np.isnan(fwd_removed))] = 0
        rev_removed[np.where(np.isnan(rev_removed))] = 0
        args = {'rop_fwd': lambda x: np.array(fwd_removed, order=x, copy=True),
                'rop_rev': lambda x: np.array(rev_removed, order=x, copy=True),
                'pres_mod': lambda x: np.array(self.store.ref_pres_mod,
                                               order=x, copy=True)
                }

        # first test w/o the splitting
        kc = kernel_call('rop_net', [self.store.rxn_rates], **args)
        self.__generic_rate_tester(get_rop_net, kc)

        def __input_mask(self, arg_name):
            names = ['fwd', 'rev', 'pres_mod']
            return next(x for x in names if x in self.name) in arg_name

        def __chainer(self, out_vals):
            self.kernel_args['rop_net'] = out_vals[-1][0]

        # next test with splitting
        kc = [kernel_call('rop_net_fwd', [self.store.rxn_rates],
                          input_mask=__input_mask, strict_name_match=True,
                          check=False, **args),
              kernel_call('rop_net_rev', [self.store.rxn_rates],
                          input_mask=__input_mask, strict_name_match=True,
                          check=False, chain=__chainer, **args),
              kernel_call('rop_net_pres_mod', [self.store.rxn_rates],
                          input_mask=__input_mask, strict_name_match=True,
                          chain=__chainer, **args)]
        self.__generic_rate_tester(get_rop_net, kc, do_ropsplit=True)

    @attr('long')
    def test_spec_rates(self):
        args = {'rop_net': lambda x: np.array(self.store.rxn_rates, order=x,
                                              copy=True),
                'wdot': lambda x: np.zeros_like(self.store.species_rates,
                                                order=x)}
        wdot = self.store.species_rates
        kc = kernel_call('spec_rates', [wdot],
                         compare_mask=[
                            get_comparable(np.arange(self.store.gas.n_species),
                                           wdot)],
                         **args)

        # test regularly
        self.__generic_rate_tester(get_spec_rates, kc)

    @attr('long')
    def test_temperature_rates(self):
        args = {'wdot': lambda x: np.array(self.store.species_rates.copy(), order=x,
                                           copy=True),
                'conc': lambda x: np.array(self.store.concs, order=x, copy=True),
                'cp': lambda x: np.array(self.store.spec_cp, order=x, copy=True),
                'h': lambda x: np.array(self.store.spec_h, order=x, copy=True),
                'cv': lambda x: np.array(self.store.spec_cv, order=x, copy=True),
                'u': lambda x: np.array(self.store.spec_u, order=x, copy=True),
                'dphi': lambda x: np.zeros_like(self.store.dphi_cp, order=x)}

        kc = [kernel_call('temperature_rate', [self.store.dphi_cp],
                          input_mask=['cv', 'u'],
                          compare_mask=[get_comparable(np.array(0, dtype=np.int32),
                                                       self.store.dphi_cp)],
                          **args)]

        # test conp
        self.__generic_rate_tester(get_temperature_rate, kc,
                                   conp=True)

        # test conv
        kc = [kernel_call('temperature_rate', [self.store.dphi_cv],
                          input_mask=['cp', 'h'],
                          compare_mask=[get_comparable(np.array(0, dtype=np.int32),
                                                       self.store.dphi_cv)],
                          **args)]
        # test conv
        self.__generic_rate_tester(get_temperature_rate, kc,
                                   conp=False)

    @attr('long')
    def test_get_molar_rates(self):
        args = {
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
            'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True)}

        kc = [kernel_call('get_molar_rates', [self.store.dphi_cp],
                          input_mask=['cv', 'u'],
                          compare_mask=[
                            get_comparable(
                                2 + np.arange(self.store.gas.n_species - 1),
                                self.store.dphi_cp)],
                          **args)]

        # test conp
        self.__generic_rate_tester(get_molar_rates, kc,
                                   conp=True)

        args = {
            'V_arr': lambda x: np.array(
                self.store.V, order=x, copy=True),
            'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True)}
        # test conv
        kc = [kernel_call('get_molar_rates', [self.store.dphi_cv],
                          input_mask=['cp', 'h'],
                          compare_mask=[get_comparable(
                            2 + np.arange(self.store.gas.n_species - 1),
                            self.store.dphi_cv)],
                          **args)]
        # test conv
        self.__generic_rate_tester(get_molar_rates, kc,
                                   conp=False)

    @attr('long')
    def test_get_extra_var_rates(self):
        dphi = np.zeros_like(self.store.dphi_cp)
        dphi[:, 0] = self.store.conp_temperature_rates[:]
        args = {
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
            'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True),
            'P_arr': lambda x: np.array(
                self.store.P, order=x, copy=True),
            'dphi': lambda x: np.array(
                dphi, order=x, copy=True)}

        kc = [kernel_call('get_extra_var_rates', [self.store.dphi_cp],
                          input_mask=['cv', 'u'],
                          compare_mask=[get_comparable(np.array([1], dtype=np.int32),
                                                       self.store.dphi_cp)],
                          **args)]

        # test conp
        self.__generic_rate_tester(get_extra_var_rates, kc,
                                   conp=True)

        dphi = np.zeros_like(self.store.dphi_cv)
        dphi[:, 0] = self.store.conv_temperature_rates[:]
        args = {
            'phi': lambda x: np.array(
                self.store.phi_cv, order=x, copy=True),
            'V_arr': lambda x: np.array(
                self.store.V, order=x, copy=True),
            'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True),
            'dphi': lambda x: np.array(
                dphi, order=x, copy=True)}

        # test conv
        kc = [kernel_call('get_extra_var_rates', [self.store.dphi_cv],
                          input_mask=['cp', 'h'],
                          compare_mask=[get_comparable(np.array([1], dtype=np.int32),
                                                       self.store.dphi_cv)],
                          **args)]
        # test conv
        self.__generic_rate_tester(get_extra_var_rates, kc,
                                   conp=False)

    @attr('long')
    def test_reset_arrays(self):
        args = {
            'dphi': lambda x: np.array(
                self.store.dphi_cp, order=x, copy=True),
            'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True)}

        kc = [kernel_call('ndot_reset', [np.zeros_like(self.store.dphi_cp)],
                          strict_name_match=True, input_mask=['wdot'], **args),
              kernel_call('wdot_reset', [np.zeros_like(self.store.species_rates)],
                          strict_name_match=True, input_mask=['dphi'], **args)]

        # test conp
        self.__generic_rate_tester(reset_arrays, kc)

    @parameterized.expand([('opencl',), ('c',)])
    @attr('long')
    def test_specrates(self, lang):
        eqs, oploop = self.__get_eqs_and_oploop(
                do_ratespec=True, do_ropsplit=True,
                do_conp=True,
                do_vector=lang != 'c',
                langs=[lang])

        package_lang = {'opencl': 'ocl',
                        'c': 'c'}
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir

        def __cleanup():
            # remove library
            test_utils.clean_dir(lib_dir)
            # remove build
            test_utils.clean_dir(obj_dir)
            # clean dummy builder
            dist_build = os.path.join(build_dir, 'build')
            if os.path.exists(dist_build):
                shutil.rmtree(dist_build)
            # clean sources
            test_utils.clean_dir(build_dir)

        P = self.store.P
        V = self.store.V
        exceptions = ['conp']

        # load the module tester template
        mod_test = test_utils.get_import_source()

        # now start test
        for i, state in enumerate(oploop):
            __cleanup()
            if state['width'] is not None and state['depth'] is not None:
                continue
            opts = loopy_options(**{x: state[x] for x in
                                    state if x not in exceptions})

            # check to see if device is CPU
            # if (opts.lang == 'opencl' and opts.device_type == cl.device_type.CPU) \
            #        and (opts.depth is None or not opts.use_atomics):
            #    opts.use_private_memory = True

            conp = state['conp']

            # generate kernel
            kgen = write_specrates_kernel(eqs, self.store.reacs,
                                          self.store.specs, opts,
                                          conp=conp)

            # generate
            kgen.generate(
                build_dir, data_filename=os.path.join(os.getcwd(), 'data.bin'))

            # write header
            write_aux(build_dir, opts, self.store.specs, self.store.reacs)

            # generate wrapper
            generate_wrapper(opts.lang, build_dir, build_dir=obj_dir,
                             out_dir=lib_dir, platform=str(opts.platform))

            # get arrays
            phi = np.array(
                self.store.phi_cp if conp else self.store.phi_cv,
                order=opts.order, copy=True)
            dphi = np.array(self.store.dphi_cp if conp else self.store.dphi_cv,
                            order=opts.order, copy=True)
            param = np.array(P if conp else V, copy=True)

            # save args to dir
            def __saver(arr, name, namelist):
                myname = os.path.join(lib_dir, name + '.npy')
                # need to split inputs / answer
                np.save(myname, kgen.array_split.split_numpy_arrays(
                    arr)[0].flatten('K'))
                namelist.append(myname)

            args = []
            __saver(phi, 'phi', args)
            __saver(param, 'param', args)

            # and now the test values
            tests = []
            __saver(dphi, 'dphi', tests)

            # find where the last species concentration is zero to mess with the
            # tolerances there

            # get split arrays
            concs, dphi = kgen.array_split.split_numpy_arrays([
                self.store.concs, self.store.dphi_cp if conp
                else self.store.dphi_cv])

            # index into concs to ge the last species
            if concs.ndim == 3:
                slice_ind = parse_split_index(
                    concs, len(self.store.specs) - 1, opts.order)
            else:
                slice_ind = [slice(None), -1]

            # find where it's zero
            last_zeros = np.where(concs[slice_ind] == 0)
            count = 0

            # create a ravel index
            ravel_ind = list(slice_ind[:])
            for i in range(len(slice_ind)):
                if slice_ind[i] != slice(None):
                    ravel_ind[i] = np.arange(dphi.shape[i], dtype=np.int32)
                else:
                    ravel_ind[i] = last_zeros[count]
                    count += 1

            # and use multi_ravel to convert to linear for dphi
            looser_tols = np.ravel_multi_index(ravel_ind, dphi.shape,
                                               order=opts.order)

            num_devices = cpu_count() / 2
            if lang == 'opencl' and opts.device_type == cl.device_type.GPU:
                num_devices = 1

            # write the module tester
            with open(os.path.join(lib_dir, 'test.py'), 'w') as file:
                file.write(mod_test.safe_substitute(
                    package='pyjac_{lang}'.format(
                        lang=package_lang[opts.lang]),
                    input_args=', '.join('"{}"'.format(x) for x in args),
                    test_arrays=', '.join('"{}"'.format(x) for x in tests),
                    looser_tols='[{}]'.format(
                        ', '.join(str(x) for x in looser_tols)),
                    rtol=5e-3,
                    atol=1e-8,
                    non_array_args='{}, {}'.format(
                        self.store.test_size, num_devices),
                    call_name='species_rates',
                    output_files=''))

            try:
                subprocess.check_call([
                    'python{}.{}'.format(
                        sys.version_info[0], sys.version_info[1]),
                    os.path.join(lib_dir, 'test.py')])
                # cleanup
                for x in args + tests:
                    os.remove(x)
                os.remove(os.path.join(lib_dir, 'test.py'))
            except:
                assert False, 'Species rates error'
