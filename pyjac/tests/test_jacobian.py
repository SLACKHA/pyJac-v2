from . import TestClass
from ..core.rate_subs import (
    get_concentrations,
    get_rop, get_rop_net, get_spec_rates, get_molar_rates, get_thd_body_concs,
    get_rxn_pres_mod, get_reduced_pressure_kernel, get_lind_kernel,
    get_sri_kernel, get_troe_kernel, get_simple_arrhenius_rates,
    polyfit_kernel_gen, get_plog_arrhenius_rates, get_cheb_arrhenius_rates,
    get_rev_rates, get_temperature_rate, get_extra_var_rates)
from ..loopy_utils.loopy_utils import (loopy_options, RateSpecialization,
                                       kernel_call, set_adept_editor, populate)
from ..core.create_jacobian import (
    dRopi_dnj, dci_thd_dnj, dci_lind_dnj, dci_sri_dnj, dci_troe_dnj,
    total_specific_energy, dTdot_dnj, dEdot_dnj, thermo_temperature_derivative,
    dRopidT, dRopi_plog_dT, dRopi_cheb_dT, dTdotdT, dci_thd_dT, dci_lind_dT,
    dci_troe_dT, dci_sri_dT, dEdotdT, dTdotdE, dEdotdE, dRopidE, dRopi_plog_dE,
    dRopi_cheb_dE, dci_thd_dE, dci_lind_dE, dci_troe_dE, dci_sri_dE,
    determine_jac_inds, reset_arrays)
from ..core import array_creator as arc
from ..core.reaction_types import reaction_type, falloff_form
from ..kernel_utils import kernel_gen as k_gen
from .test_utils import kernel_runner, get_comparable, _generic_tester

import numpy as np
import six
import loopy as lp
import cantera as ct

from nose.plugins.attrib import attr
from unittest.case import SkipTest


class editor(object):

    def __init__(self, independent, dependent,
                 problem_size, order, do_not_set=[]):

        self.independent = independent
        indep_size = next(x for x in independent.shape if x != problem_size)
        self.dependent = dependent
        dep_size = next(x for x in dependent.shape if x != problem_size)
        self.problem_size = problem_size

        # create the jacobian
        self.output = arc.creator('jac', np.float64,
                                  (problem_size, dep_size, indep_size),
                                  order=order)
        self.output = self.output(*['i', 'j', 'k'])[0]
        try:
            self.do_not_set = do_not_set[:]
        except:
            self.do_not_set = [do_not_set]

    def set_single_kernel(self, single_kernel):
        """
        It's far easier to use two generated kernels, one that uses the full
        problem size (for calling via loopy), and another that uses a problem
        size of 1, to work with Adept indexing in the AD kernel
        """
        self.single_kernel = single_kernel

    def __call__(self, knl):
        return set_adept_editor(knl, self.single_kernel, self.problem_size,
                                self.independent, self.dependent, self.output,
                                self.do_not_set)


class SubTest(TestClass):
    def _generic_jac_tester(self, func, kernel_calls, do_ratespec=False,
                            do_ropsplit=None,
                            do_conp=False,
                            **kw_args):
        """
        A generic testing method that can be used for testing jacobian kernels

        This is primarily a thin wrapper for :func:`_generic_tester`

        Parameters
        ----------
        func : function
            The function to test
        kernel_calls : :class:`kernel_call` or list thereof
            Contains the masks and reference answers for kernel testing
        do_ratespec : bool [False]
            If true, test rate specializations and kernel splitting for simple rates
        do_ropsplit : bool [False]
            If true, test kernel splitting for rop_net
        do_conp:  bool [False]
            If true, test for both constant pressure _and_ constant volume
        """

        _generic_tester(self, func, kernel_calls, determine_jac_inds,
                        do_ratespec=do_ratespec, do_ropsplit=do_ropsplit,
                        do_conp=do_conp, **kw_args)

    def _make_array(self, array):
        """
        Creates an array for comparison to an autorun kernel from the result
        of __get_jacobian

        Parameters
        ----------
        array : :class:`numpy.ndarray`
            The input Jacobian array

        Returns
        -------
        reshaped : :class:`numpy.ndarray`
            The reshaped  / reordered array for comparison to the autorun
            kernel
        """

        for i in range(array.shape[0]):
            # reshape inner array
            array[i, :, :] = np.reshape(array[i, :, :].flatten(order='K'),
                                        array.shape[1:],
                                        order='F')

        return array

    def _get_jacobian(self, func, kernel_call, editor, ad_opts, conp,
                      extra_funcs=[],
                      **kw_args):
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        # find rate info
        rate_info = determine_jac_inds(
            self.store.reacs,
            self.store.specs,
            ad_opts.rate_spec)
        # create namestore
        namestore = arc.NameStore(ad_opts, rate_info, conp,
                                  self.store.test_size)

        # get kw args this function expects
        def __get_arg_dict(check, **in_args):
            try:
                arg_count = check.func_code.co_argcount
                args = check.func_code.co_varnames[:arg_count]
            except:
                arg_count = check.__code__.co_argcount
                args = check.__code__.co_varnames[:arg_count]

            args_dict = {}
            for k, v in six.iteritems(in_args):
                if k in args:
                    args_dict[k] = v
            return args_dict

        # create the kernel info
        infos = []
        info = func(eqs, ad_opts, namestore,
                    test_size=self.store.test_size,
                    **__get_arg_dict(func, **kw_args))
        try:
            infos.extend(info)
        except:
            infos.append(info)

        # create a dummy kernel generator
        knl = k_gen.make_kernel_generator(
            name='jacobian',
            loopy_opts=ad_opts,
            kernels=infos,
            test_size=self.store.test_size,
            extra_kernel_data=[editor.output]
        )
        knl._make_kernels()

        # get list of current args
        have_match = kernel_call.strict_name_match
        new_args = []
        new_kernels = []
        for k in knl.kernels:
            if have_match and kernel_call.name != k.name:
                continue

            new_kernels.append(k)
            for arg in k.args:
                if arg not in new_args and not isinstance(
                        arg, lp.TemporaryVariable):
                    new_args.append(arg)

        knl = new_kernels[:]

        # generate dependencies with full test size to get extra args
        infos = []
        for f in extra_funcs:
            info = f(eqs, ad_opts, namestore,
                     test_size=self.store.test_size,
                     **__get_arg_dict(f, **kw_args))
            try:
                infos.extend(info)
            except:
                infos.append(info)

        for i in infos:
            for arg in i.kernel_data:
                if arg not in new_args and not isinstance(
                        arg, lp.TemporaryVariable):
                    new_args.append(arg)

        for i in range(len(knl)):
            knl[i] = knl[i].copy(args=new_args[:])

        # and a generator for the single kernel
        single_name = arc.NameStore(ad_opts, rate_info, conp, 1)
        single_info = []
        for f in extra_funcs + [func]:
            info = f(eqs, ad_opts, single_name,
                     test_size=1,
                     **__get_arg_dict(f, **kw_args))
            try:
                for i in info:
                    if f == func and have_match and kernel_call.name != i.name:
                        continue
                    single_info.append(i)
            except:
                if f == func and have_match and kernel_call.name != info.name:
                    continue
                single_info.append(info)

        single_knl = k_gen.make_kernel_generator(
            name='spec_rates',
            loopy_opts=ad_opts,
            kernels=single_info,
            test_size=1,
            extra_kernel_data=[editor.output]
        )

        single_knl._make_kernels()

        # set in editor
        editor.set_single_kernel(single_knl.kernels)

        kernel_call.set_state(single_knl.array_split, ad_opts.order)

        # add dummy 'j' arguement
        kernel_call.kernel_args['j'] = -1
        # and place output
        kernel_call.kernel_args[editor.output.name] = np.zeros(
            editor.output.shape,
            order=editor.output.order)

        # run kernel
        populate(
            [knl[0]], kernel_call,
            editor=editor)

        return self._make_array(kernel_call.kernel_args[editor.output.name])

    def _make_namestore(self, conp):
        # get number of sri reactions
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        ad_opts = loopy_options(order='C', knl_type='map', lang='c',
                                auto_diff=True)

        # create namestore
        namestore = arc.NameStore(ad_opts, rate_info, conp, self.store.test_size)

        return namestore, rate_info

    @attr('long')
    def test_dropi_dnj(self):

        # test conp
        namestore, rate_info = self._make_namestore(True)
        ad_opts = namestore.loopy_opts

        # set up arguements
        allint = {'net': rate_info['net']['allint']}
        # create the editor
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        args = {'rop_fwd': lambda x: np.zeros_like(
            self.store.fwd_rxn_rate, order=x),
            'rop_rev': lambda x: np.zeros_like(
            self.store.rev_rxn_rate, order=x),
            'pres_mod': lambda x: np.array(
            self.store.ref_pres_mod, order=x, copy=True),
            'rop_net': lambda x: np.zeros_like(
            self.store.rxn_rates, order=x),
            'phi': lambda x: np.array(
            self.store.phi_cp, order=x, copy=True),
            'P_arr': lambda x: np.array(
            self.store.P, order=x, copy=True),
            'kf': lambda x: np.array(
            self.store.fwd_rate_constants, order=x, copy=True),
            'kr': lambda x: np.array(
            self.store.rev_rate_constants, order=x, copy=True),
            'conc': lambda x: np.zeros_like(
            self.store.concs, order=x),
            'wdot': lambda x: np.zeros_like(
            self.store.species_rates, order=x),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x)
        }

        # obtain the finite difference jacobian
        kc = kernel_call('dRopidnj', [self.store.rxn_rates], **args)

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, True,
            extra_funcs=[get_concentrations, get_rop, get_rop_net,
                         get_spec_rates],
            do_not_set=[namestore.rop_fwd, namestore.rop_rev,
                        namestore.conc_arr, namestore.spec_rates,
                        namestore.presmod],
            allint=allint)

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        jac_size = rate_info['Ns'] + 1
        args = {
            'kf': lambda x: np.array(
                self.store.fwd_rate_constants, order=x, copy=True),
            'kr': lambda x: np.array(
                self.store.rev_rate_constants, order=x, copy=True),
            'pres_mod': lambda x: np.array(
                self.store.ref_pres_mod, order=x, copy=True),
            'conc': lambda x: np.array(
                self.store.concs, order=x, copy=True),
            'jac': lambda x: np.zeros(
                (self.store.test_size, jac_size, jac_size), order=x)
        }
        # and test
        kc = [kernel_call('dRopidnj', [fd_jac], check=False,
                          strict_name_match=True, **args),
              kernel_call('dRopidnj_ns', [fd_jac], compare_mask=[
                get_comparable([(
                  2 + np.arange(self.store.gas.n_species - 1),
                  2 + np.arange(self.store.gas.n_species - 1))], [fd_jac],
                  compare_axis=(1, 2))],
            compare_axis=(1, 2), chain=_chainer, strict_name_match=True,
            allow_skip=True,
            **args)]

        return self._generic_jac_tester(dRopi_dnj, kc, allint=allint)

    def __get_check(self, include_test, rxn_test=None):
        include = set()
        exclude = set()
        # get list of species not in falloff / chemically activated
        for i_rxn, rxn in enumerate(self.store.gas.reactions()):
            if rxn_test is None or rxn_test(rxn):
                specs = set(
                    list(rxn.products.keys()) + list(rxn.reactants.keys()))
                nonzero_specs = set()
                for spec in specs:
                    nu = 0
                    if spec in rxn.products:
                        nu += rxn.products[spec]
                    if spec in rxn.reactants:
                        nu -= rxn.reactants[spec]
                    if nu != 0:
                        nonzero_specs.update([spec])
                if include_test(rxn):
                    include.update(nonzero_specs)
                else:
                    exclude.update(nonzero_specs)

        test = set(self.store.gas.species_index(x)
                   for x in include - exclude)
        return np.array(sorted(test)) + 2

    def __get_dci_check(self, include_test):
        return self.__get_check(include_test, lambda rxn:
                                isinstance(rxn, ct.FalloffReaction) or
                                isinstance(rxn, ct.ThreeBodyReaction))

    def __get_comp_extractor(self, kc, mask):
        cm = mask.compare_mask[0] if isinstance(mask, get_comparable) else mask
        if len(cm) != 3:
            return tuple([None]) * 4  # only compare masks w/ conditions

        # first, invert the conditions mask
        cond, x, y = cm
        cond = np.where(np.logical_not(
            np.in1d(np.arange(self.store.test_size), cond)))[0]

        if not cond.size:
            return tuple([None]) * 4   # nothing to test

        def __get_val(vals, mask):
            outv = vals.copy()
            for ax, m in enumerate(mask):
                outv = np.take(outv, m, axis=ax)
            return outv
        extractor = __get_val

        # create a new compare mask if necessary
        if isinstance(mask, get_comparable):
            mask = get_comparable(
                compare_mask=[(cond, x, y)],
                compare_axis=mask.compare_axis,
                ref_answer=mask.ref_answer)

            # and redefine the value extractor
            def __get_val(vals, *args):
                return mask(kc, vals, 0)
            extractor = __get_val

        # and return the extractor
        return extractor, cond, x, y

    @attr('long')
    def test_dci_thd_dnj(self):
        # test conp
        namestore, rate_info = self._make_namestore(True)
        ad_opts = namestore.loopy_opts

        # setup arguemetns
        # create the editor
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        args = {'rop_fwd': lambda x: np.array(
            self.store.fwd_rxn_rate, order=x, copy=True),
            'rop_rev': lambda x: np.array(
            self.store.rev_rxn_rate, order=x, copy=True),
            'pres_mod': lambda x: np.array(
            self.store.ref_pres_mod, order=x, copy=True),
            'rop_net': lambda x: np.zeros_like(
            self.store.rxn_rates, order=x),
            'phi': lambda x: np.array(
            self.store.phi_cp, order=x, copy=True),
            'P_arr': lambda x: np.array(
            self.store.P, order=x, copy=True),
            'conc': lambda x: np.zeros_like(
            self.store.concs, order=x),
            'wdot': lambda x: np.zeros_like(
            self.store.species_rates, order=x),
            'thd_conc': lambda x: np.zeros_like(
            self.store.ref_thd, order=x),
            'Fi': lambda x: np.array(
            self.store.ref_Fall, order=x, copy=True),
            'Pr': lambda x: np.array(
            self.store.ref_Pr, order=x, copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x)
        }

        # obtain the finite difference jacobian
        kc = kernel_call('dci_thd_nj', [None], **args)

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, True,
            extra_funcs=[get_concentrations, get_thd_body_concs,
                         get_rxn_pres_mod, get_rop_net,
                         get_spec_rates],
            do_not_set=[
                namestore.conc_arr, namestore.spec_rates, namestore.rop_net])

        # setup args
        jac_size = rate_info['Ns'] + 1
        args = {
            'rop_fwd': lambda x: np.array(
                self.store.fwd_rxn_rate, order=x, copy=True),
            'rop_rev': lambda x: np.array(
                self.store.rev_rxn_rate, order=x, copy=True),
            'jac': lambda x: np.zeros(
                (self.store.test_size, jac_size, jac_size), order=x)
        }

        # get list of species not in falloff / chemically activated
        # to get the check mask
        test = self.__get_dci_check(lambda x: isinstance(
            x, ct.ThreeBodyReaction))

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        # and get mask
        comp = get_comparable(
                  [(test, 2 + np.arange(self.store.gas.n_species - 1))],
                  [fd_jac], compare_axis=(1, 2))
        kc = [kernel_call('dci_thd_dnj', [fd_jac], check=False,
                          strict_name_match=True, **args),
              kernel_call('dci_thd_dnj_ns', comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True, **args)]

        return self._generic_jac_tester(dci_thd_dnj, kc)

    def nan_compare(self, kc, our_val, ref_val, mask, allow_our_nans=False):
        # get the condition extractor
        extractor, cond, x, y = self.__get_comp_extractor(kc, mask)
        if extractor is None:
            # no need to test
            return True

        def __compare(our_vals, ref_vals):
            # sometimes if only one value is selected, we end up with a
            # non-dimensional array
            if not ref_vals.shape and ref_vals:
                ref_vals = np.expand_dims(ref_vals, axis=0)
            if not our_vals.shape and our_vals:
                our_vals = np.expand_dims(our_vals, axis=0)
            # find where close
            bad = np.where(np.logical_not(np.isclose(ref_vals, our_vals)))
            good = np.where(np.isclose(ref_vals, our_vals))

            # make sure all the bad conditions here in the ref val are nan's
            is_correct = np.all(np.isnan(ref_vals[bad]))

            # or failing that, just that they're much "larger" than the other
            # entries (sometimes the Pr will not be exactly zero if it's
            # based on the concentration of the last species)
            fac = 1 if not good[0].size else np.max(np.abs(ref_vals[good]))
            is_correct = is_correct or (
                (np.min(np.abs(ref_vals[bad])) / fac) > 1e12)

            # and ensure all our values are 'large' but finite numbers
            # (defined here by > 1e295)
            # _or_ allow_our_nans is True _and_ they're all nan's
            is_correct = is_correct and (
                np.all(np.abs(our_vals[bad]) >= 1e285)
                or (allow_our_nans and np.all(np.isnan(our_vals[bad]))))

            return is_correct

        return __compare(extractor(our_val, (cond, x, y)),
                         extractor(ref_val, (cond, x, y)))

    def our_nan_compare(self, kc, our_val, ref_val, mask):
        return self.nan_compare(kc, our_val, ref_val, mask, allow_our_nans=True)

    def __get_removed(self):
        # get our form of rop_fwd / rop_rev
        fwd_removed = self.store.fwd_rxn_rate.copy()
        rev_removed = self.store.rev_rxn_rate.copy()
        if self.store.thd_inds.size:
            fwd_removed[:, self.store.thd_inds] = fwd_removed[
                :, self.store.thd_inds] / self.store.ref_pres_mod
            thd_in_rev = np.where(
                np.in1d(self.store.thd_inds, self.store.rev_inds))[0]
            rev_update_map = np.where(
                np.in1d(
                    self.store.rev_inds, self.store.thd_inds[thd_in_rev]))[0]
            rev_removed[:, rev_update_map] = rev_removed[
                :, rev_update_map] / self.store.ref_pres_mod[:, thd_in_rev]
            # remove ref pres mod = 0 (this is a 0 rate)
            fwd_removed[np.where(np.isnan(fwd_removed))] = 0
            rev_removed[np.where(np.isnan(rev_removed))] = 0

        return fwd_removed, rev_removed

    def __get_kf_and_fall(self, conp=True):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        # create args and parameters
        phi = self.store.phi_cp if conp else self.store.phi_cv
        args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                'kf': lambda x: np.zeros_like(self.store.fwd_rate_constants,
                                              order=x)}
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        opts = loopy_options(order='C', knl_type='map', lang='opencl')
        namestore = arc.NameStore(opts, rate_info, True, self.store.test_size)

        # get kf
        runner = kernel_runner(get_simple_arrhenius_rates,
                               self.store.test_size, args)
        kf = runner(eqs, opts, namestore, self.store.test_size)[0]

        if self.store.ref_Pr.size:
            args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                    'kf_fall': lambda x: np.zeros_like(self.store.ref_Fall, order=x)}
            # get kf_fall
            runner = kernel_runner(get_simple_arrhenius_rates,
                                   self.store.test_size, args,
                                   {'falloff': True})
            kf_fall = runner(eqs, opts, namestore, self.store.test_size)[0]
        else:
            kf_fall = None

        if namestore.num_plog is not None:
            args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                    'kf': lambda x: np.array(kf, order=x, copy=True)}
            if conp:
                args['P_arr'] = lambda x: np.array(
                    self.store.P, order=x, copy=True)
            # get plog
            runner = kernel_runner(self.__get_plog_call_wrapper(rate_info),
                                   self.store.test_size, args)
            kf = runner(eqs, opts, namestore, self.store.test_size)[0]

        if namestore.num_cheb is not None:
            args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                    'kf': lambda x: np.array(kf, order=x, copy=True)}
            if conp:
                args['P_arr'] = lambda x: np.array(
                    self.store.P, order=x, copy=True)
            # get plog
            runner = kernel_runner(self.__get_cheb_call_wrapper(rate_info),
                                   self.store.test_size, args)
            kf = runner(eqs, opts, namestore, self.store.test_size)[0]

        return kf, kf_fall

    def __get_kr(self, kf):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        args = {
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'b': lambda x: np.array(
                self.store.ref_B_rev, order=x, copy=True),
            'out_mask': [0, 1]}
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        opts = loopy_options(order='C', knl_type='map', lang='opencl')
        namestore = arc.NameStore(opts, rate_info, True, self.store.test_size)
        allint = {'net': rate_info['net']['allint']}

        # get kf
        runner = kernel_runner(get_rev_rates,
                               self.store.test_size, args, {'allint': allint})
        out = runner(eqs, opts, namestore, self.store.test_size)
        return out[next(i for i, x in enumerate(runner.out_arg_names[0])
                        if x == 'kr')]

    def __get_db(self):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        opts = loopy_options(order='C', knl_type='map', lang='opencl')
        namestore = arc.NameStore(opts, rate_info, True, self.store.test_size)
        # need dBk/dT
        args = {
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
        }

        def __call_wrapper(eqs, loopy_opts, namestore, test_size):
            return thermo_temperature_derivative(
                'db', eqs,
                loopy_opts, namestore,
                test_size)
        # get kf
        runner = kernel_runner(__call_wrapper, self.store.test_size, args)
        return runner(eqs, opts, namestore, self.store.test_size)[0]

    @attr('long')
    def test_dci_lind_dnj(self):
        # test conp
        namestore, rate_info = self._make_namestore(True)
        ad_opts = namestore.loopy_opts

        # set up arguements
        allint = {'net': rate_info['net']['allint']}

        fwd_removed, rev_removed = self.__get_removed()

        # setup arguements
        # create the editor
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        kf, kf_fall = self.__get_kf_and_fall()

        args = {'rop_fwd': lambda x: np.array(
            fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(
            rev_removed, order=x, copy=True),
            'pres_mod': lambda x: np.zeros_like(
            self.store.ref_pres_mod, order=x),
            'rop_net': lambda x: np.zeros_like(
            self.store.rxn_rates, order=x),
            'phi': lambda x: np.array(
            self.store.phi_cp, order=x, copy=True),
            'P_arr': lambda x: np.array(
            self.store.P, order=x, copy=True),
            'conc': lambda x: np.zeros_like(
            self.store.concs, order=x),
            'wdot': lambda x: np.zeros_like(
            self.store.species_rates, order=x),
            'thd_conc': lambda x: np.zeros_like(
            self.store.ref_thd, order=x),
            'Fi': lambda x: np.zeros_like(self.store.ref_Fall, order=x),
            'Pr': lambda x: np.zeros_like(self.store.ref_Pr, order=x),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
        }

        # obtain the finite difference jacobian
        kc = kernel_call('dci_lind_nj', [None], **args)

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, True,
            extra_funcs=[get_concentrations, get_thd_body_concs,
                         get_reduced_pressure_kernel, get_lind_kernel,
                         get_rxn_pres_mod, get_rop_net,
                         get_spec_rates],
            do_not_set=[namestore.conc_arr, namestore.spec_rates,
                        namestore.rop_net, namestore.Fi],
            allint=allint)

        # setup args
        args = {
            'rop_fwd': lambda x: np.array(
                fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(
                rev_removed, order=x, copy=True),
            'Pr': lambda x: np.array(
                self.store.ref_Pr, order=x, copy=True),
            'Fi': lambda x: np.array(
                self.store.ref_Fall, order=x, copy=True),
            'pres_mod': lambda x: np.array(
                self.store.ref_pres_mod, order=x, copy=True),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
        }

        lind_test = self.__get_dci_check(
            lambda rxn: isinstance(rxn, ct.FalloffReaction) and
            rxn.falloff.type == 'Simple')

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        # and get mask
        comp = get_comparable(compare_mask=[(
            lind_test, 2 + np.arange(self.store.gas.n_species - 1))],
            ref_answer=[fd_jac],
            compare_axis=(1, 2))
        kc = [kernel_call('dci_lind_dnj', comp.ref_answer, check=False,
                          strict_name_match=True, **args),
              kernel_call('dci_lind_dnj_ns', comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True, **args)]

        return self._generic_jac_tester(dci_lind_dnj, kc)

    def __get_sri_params(self, namestore):
        sri_args = {'Pr': lambda x: np.array(
            self.store.ref_Pr, order=x, copy=True),
            'phi': lambda x: np.array(
            self.store.phi_cp, order=x, copy=True),
            'out_mask': [1]}
        runner = kernel_runner(get_sri_kernel, self.store.test_size, sri_args)
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        opts = loopy_options(order='C', knl_type='map', lang='opencl')
        X = runner(eqs, opts, namestore, self.store.test_size)[0]
        return X

    @attr('long')
    def test_dci_sri_dnj(self):
        # test conp
        namestore, rate_info = self._make_namestore(True)
        ad_opts = namestore.loopy_opts

        # set up arguements
        allint = {'net': rate_info['net']['allint']}

        # get our form of rop_fwd / rop_rev
        fwd_removed, rev_removed = self.__get_removed()

        # setup arguements
        # create the editor
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        # get kf / kf_fall
        kf, kf_fall = self.__get_kf_and_fall()
        # create X
        X = self.__get_sri_params(namestore)

        args = {
            'pres_mod': lambda x: np.zeros_like(
                self.store.ref_pres_mod, order=x),
            'thd_conc': lambda x: np.array(
                self.store.ref_thd, order=x, copy=True),
            'Fi': lambda x: np.zeros_like(self.store.ref_Fall, order=x),
            'Pr': lambda x: np.array(self.store.ref_Pr, order=x, copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'X': lambda x: np.zeros_like(X, order=x),
            'phi': lambda x: np.array(self.store.phi_cp, order=x, copy=True),
            'P_arr': lambda x: np.array(self.store.P, order=x, copy=True),
            'conc': lambda x: np.zeros_like(self.store.concs, order=x),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
            'wdot': lambda x: np.zeros_like(self.store.species_rates, order=x),
            'rop_fwd': lambda x: np.array(
                fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(
                rev_removed, order=x, copy=True),
            'rop_net': lambda x: np.zeros_like(self.store.rxn_rates, order=x)
        }

        # obtain the finite difference jacobian
        kc = kernel_call('dci_sri_nj', [None], **args)

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, True,
            extra_funcs=[get_concentrations, get_thd_body_concs,
                         get_reduced_pressure_kernel, get_sri_kernel,
                         get_rxn_pres_mod, get_rop_net, get_spec_rates],
            do_not_set=[namestore.conc_arr, namestore.Fi, namestore.X_sri,
                        namestore.thd_conc],
            allint=allint)

        # setup args
        args = {
            'rop_fwd': lambda x: np.array(
                fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(
                rev_removed, order=x, copy=True),
            'Pr': lambda x: np.array(
                self.store.ref_Pr, order=x, copy=True),
            'Fi': lambda x: np.array(
                self.store.ref_Fall, order=x, copy=True),
            'pres_mod': lambda x: np.array(
                self.store.ref_pres_mod, order=x, copy=True),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
            'X': lambda x: np.array(X, order=x, copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True)
        }

        sri_test = self.__get_dci_check(
            lambda rxn: isinstance(rxn, ct.FalloffReaction) and
            rxn.falloff.type == 'SRI')

        # find non-NaN SRI entries for testing
        # NaN entries will be handled by :func:`nan_compare`
        to_test = np.where(np.all(
            self.store.ref_Pr[:, self.store.sri_to_pr_map] != 0.0, axis=1))[0]

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        # and get mask
        comp = get_comparable(compare_mask=[(
            to_test, sri_test, 2 + np.arange(self.store.gas.n_species - 1))],
            compare_axis=(0, 1, 2),
            ref_answer=[fd_jac])
        kc = [kernel_call('dci_sri_dnj', comp.ref_answer, check=False,
                          strict_name_match=True, **args),
              kernel_call('dci_sri_dnj_ns', comp.ref_answer,
                          compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True,
                          other_compare=self.nan_compare, rtol=5e-4, **args)]

        return self._generic_jac_tester(dci_sri_dnj, kc)

    def __get_troe_params(self, namestore):
        troe_args = {'Pr': lambda x: np.array(
            self.store.ref_Pr, order=x, copy=True),
            'phi': lambda x: np.array(
            self.store.phi_cp, order=x, copy=True),
            'out_mask': [1, 2, 3]}
        runner = kernel_runner(
            get_troe_kernel, self.store.test_size, troe_args)
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        opts = loopy_options(order='C', knl_type='map', lang='opencl')
        Fcent, Atroe, Btroe = runner(
            eqs, opts, namestore, self.store.test_size)
        return Fcent, Atroe, Btroe

    @attr('long')
    def test_dci_troe_dnj(self):
        # test conp
        namestore, rate_info = self._make_namestore(True)
        ad_opts = namestore.loopy_opts

        # set up arguements
        allint = {'net': rate_info['net']['allint']}

        # get our form of rop_fwd / rop_rev
        fwd_removed, rev_removed = self.__get_removed()

        # setup arguements
        # create the editor
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        # get kf / kf_fall
        kf, kf_fall = self.__get_kf_and_fall()
        Fcent, Atroe, Btroe = self.__get_troe_params(namestore)

        args = {
            'pres_mod': lambda x: np.zeros_like(
                self.store.ref_pres_mod, order=x),
            'thd_conc': lambda x: np.array(
                self.store.ref_thd, order=x, copy=True),
            'Fi': lambda x: np.zeros_like(self.store.ref_Fall, order=x),
            'Pr': lambda x: np.array(self.store.ref_Pr, order=x, copy=True),
            'phi': lambda x: np.array(self.store.phi_cp, order=x, copy=True),
            'P_arr': lambda x: np.array(self.store.P, order=x, copy=True),
            'conc': lambda x: np.zeros_like(self.store.concs, order=x),
            'wdot': lambda x: np.zeros_like(self.store.species_rates, order=x),
            'rop_fwd': lambda x: np.array(
                fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(
                rev_removed, order=x, copy=True),
            'rop_net': lambda x: np.zeros_like(self.store.rxn_rates, order=x),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
            'Atroe': lambda x: np.zeros_like(Atroe, order=x),
            'Btroe': lambda x: np.zeros_like(Btroe, order=x),
            'Fcent': lambda x: np.zeros_like(Fcent, order=x)
        }

        # obtain the finite difference jacobian
        kc = kernel_call('dci_sri_nj', [None], **args)

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, True,
            extra_funcs=[get_concentrations, get_thd_body_concs,
                         get_reduced_pressure_kernel, get_troe_kernel,
                         get_rxn_pres_mod, get_rop_net, get_spec_rates],
            do_not_set=[namestore.conc_arr, namestore.Fi, namestore.Atroe,
                        namestore.Btroe, namestore.Fcent, namestore.thd_conc],
            allint=allint)

        # setup args
        args = {
            'rop_fwd': lambda x: np.array(
                fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(
                rev_removed, order=x, copy=True),
            'Pr': lambda x: np.array(
                self.store.ref_Pr, order=x, copy=True),
            'Fi': lambda x: np.array(
                self.store.ref_Fall, order=x, copy=True),
            'pres_mod': lambda x: np.array(
                self.store.ref_pres_mod, order=x, copy=True),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
            'Atroe': lambda x: np.array(Atroe, order=x, copy=True),
            'Btroe': lambda x: np.array(Btroe, order=x, copy=True),
            'Fcent': lambda x: np.array(Fcent, order=x, copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
        }

        troe_test = self.__get_dci_check(
            lambda rxn: isinstance(rxn, ct.FalloffReaction) and
            rxn.falloff.type == 'Troe')

        # find non-NaN Troe entries for testing
        # NaN entries will be handled by :func:`nan_compare`
        to_test = np.where(np.all(
            self.store.ref_Pr[:, self.store.troe_to_pr_map] != 0.0, axis=1))[0]

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        comp = get_comparable(compare_mask=[(
                  to_test,
                  troe_test,
                  2 + np.arange(self.store.gas.n_species - 1))],
                ref_answer=[fd_jac], compare_axis=(0, 1, 2))
        # and get mask
        kc = [kernel_call('dci_troe_dnj', comp.ref_answer, check=False,
                          strict_name_match=True, **args),
              kernel_call('dci_troe_dnj_ns', comp.ref_answer,
                          compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True,
                          other_compare=self.nan_compare, **args)]

        return self._generic_jac_tester(dci_troe_dnj, kc)

    def test_total_specific_energy(self):
        # conp
        ref_cp = np.sum(self.store.concs * self.store.spec_cp, axis=1)

        # cp args
        cp_args = {'cp': lambda x: np.array(
            self.store.spec_cp, order=x, copy=True),
            'conc': lambda x: np.array(
            self.store.concs, order=x, copy=True),
            'cp_tot': lambda x: np.zeros_like(ref_cp, order=x)}

        # call
        kc = [kernel_call('cp_total', [ref_cp], strict_name_match=True,
                          **cp_args)]

        self._generic_jac_tester(total_specific_energy, kc, conp=True)

        # conv
        ref_cv = np.sum(self.store.concs * self.store.spec_cv, axis=1)

        # cv args
        cv_args = {'cv': lambda x: np.array(
            self.store.spec_cv, order=x, copy=True),
            'conc': lambda x: np.array(
            self.store.concs, order=x, copy=True),
            'cv_tot': lambda x: np.zeros_like(ref_cp, order=x)}

        # call
        kc = [kernel_call('cv_total', [ref_cv], strict_name_match=True,
                          **cv_args)]

        self._generic_jac_tester(total_specific_energy, kc, conp=False)

    def __get_fall_call_wrapper(self):
        def fall_wrapper(eqs, loopy_opts, namestore, test_size):
            return get_simple_arrhenius_rates(eqs, loopy_opts, namestore,
                                              test_size, falloff=True)
        return fall_wrapper

    def __get_plog_call_wrapper(self, rate_info):
        def plog_wrapper(eqs, loopy_opts, namestore, test_size):
            return get_plog_arrhenius_rates(eqs, loopy_opts, namestore,
                                            rate_info['plog']['max_P'],
                                            test_size)
        return plog_wrapper

    def __get_cheb_call_wrapper(self, rate_info):
        def cheb_wrapper(eqs, loopy_opts, namestore, test_size):
            return get_cheb_arrhenius_rates(eqs, loopy_opts, namestore,
                                            np.max(rate_info['cheb']['num_P']),
                                            np.max(rate_info['cheb']['num_T']),
                                            test_size)
        return cheb_wrapper

    def __get_poly_wrapper(self, name, conp):
        def poly_wrapper(eqs, loopy_opts, namestore, test_size):
            desc = 'conp' if conp else 'conv'
            return polyfit_kernel_gen(name, eqs[desc],
                                      loopy_opts, namestore, test_size)
        return poly_wrapper

    def __get_full_jac(self, conp=True):
        # see if we've already computed this, no need to redo if we have it
        attr = 'fd_jac' + ('_cp' if conp else '_cv')
        if hasattr(self.store, attr):
            return getattr(self.store, attr).copy()

        # creates a FD version of the full species Jacobian
        namestore, rate_info = self._make_namestore(conp)
        ad_opts = namestore.loopy_opts
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        def __create_arr(order, inds):
            return np.zeros((self.store.test_size, inds.size), order=order)

        # setup args
        allint = {'net': rate_info['net']['allint']}
        args = {
            'phi': lambda x: np.array(self.store.phi_cp if conp
                                      else self.store.phi_cv, order=x,
                                      copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'wdot': lambda x: np.zeros_like(self.store.species_rates,
                                            order=x),
            'Atroe': lambda x: __create_arr(x, self.store.troe_inds),
            'Btroe': lambda x: __create_arr(x, self.store.troe_inds),
            'Fcent': lambda x: __create_arr(x, self.store.troe_inds),
            'Fi': lambda x: __create_arr(x, self.store.fall_inds),
            'Pr': lambda x: __create_arr(x, self.store.fall_inds),
            'X': lambda x: __create_arr(x, self.store.sri_inds),
            'conc': lambda x: np.zeros_like(self.store.concs,
                                            order=x),
            'dphi': lambda x: np.zeros_like(self.store.dphi_cp,
                                            order=x),
            'kf': lambda x: np.zeros_like(self.store.fwd_rate_constants,
                                          order=x),
            'kf_fall': lambda x: __create_arr(x, self.store.fall_inds),
            'kr': lambda x: np.zeros_like(self.store.rev_rate_constants,
                                          order=x),
            'pres_mod': lambda x: np.zeros_like(self.store.ref_pres_mod,
                                                order=x),
            'rop_fwd': lambda x: np.zeros_like(self.store.fwd_rxn_rate,
                                               order=x),
            'rop_rev': lambda x: np.zeros_like(self.store.rev_rxn_rate,
                                               order=x),
            'rop_net': lambda x: np.zeros_like(self.store.rxn_rates,
                                               order=x),
            'thd_conc': lambda x: np.zeros_like(self.store.ref_thd,
                                                order=x),
            'b': lambda x: np.zeros_like(self.store.ref_B_rev,
                                         order=x),
            'Kc': lambda x: np.zeros_like(self.store.equilibrium_constants,
                                          order=x)
        }
        if conp:
            args['P_arr'] = lambda x: np.array(
                self.store.P, order=x, copy=True)
            args['h'] = lambda x: np.zeros_like(
                self.store.spec_h, order=x)
            args['cp'] = lambda x: np.zeros_like(
                self.store.spec_cp, order=x)
        else:
            args['V_arr'] = lambda x: np.array(
                self.store.V, order=x, copy=True)
            args['u'] = lambda x: np.zeros_like(
                self.store.spec_u, order=x)
            args['cv'] = lambda x: np.zeros_like(
                self.store.spec_cv, order=x)

        # obtain the finite difference jacobian
        kc = kernel_call('dnkdnj', [None], **args)

        __b_call_wrapper = self.__get_poly_wrapper('b', conp)

        __cp_call_wrapper = self.__get_poly_wrapper('cp', conp)

        __cv_call_wrapper = self.__get_poly_wrapper('cv', conp)

        __h_call_wrapper = self.__get_poly_wrapper('h', conp)

        __u_call_wrapper = self.__get_poly_wrapper('u', conp)

        def __extra_call_wrapper(eqs, loopy_opts, namestore, test_size):
            return get_extra_var_rates(eqs, loopy_opts, namestore,
                                       conp=conp, test_size=test_size)

        def __temperature_wrapper(eqs, loopy_opts, namestore, test_size):
            return get_temperature_rate(eqs, loopy_opts, namestore,
                                        conp=conp, test_size=test_size)

        jac = self._get_jacobian(
            __extra_call_wrapper, kc, edit, ad_opts, conp,
            extra_funcs=[get_concentrations, get_simple_arrhenius_rates,
                         self.__get_plog_call_wrapper(rate_info),
                         self.__get_cheb_call_wrapper(rate_info),
                         get_thd_body_concs, self.__get_fall_call_wrapper(),
                         get_reduced_pressure_kernel, get_lind_kernel,
                         get_sri_kernel, get_troe_kernel,
                         __b_call_wrapper, get_rev_rates,
                         get_rxn_pres_mod, get_rop, get_rop_net,
                         get_spec_rates] + (
                [__h_call_wrapper, __cp_call_wrapper] if conp else
                [__u_call_wrapper, __cv_call_wrapper]) + [
                get_molar_rates, __temperature_wrapper],
            allint=allint)

        # store the jacobian for later
        setattr(self.store, attr, jac)

        return jac

    def test_dTdot_dnj(self):
        # conp

        # get total cp
        cp_sum = np.sum(self.store.concs * self.store.spec_cp, axis=1)

        # get species jacobian
        jac = self.__get_full_jac(True)

        # instead of whittling this down to the actual answer [:, 0, 2:], it's
        # way easier to keep this full sized such that we can use the same
        # :class:`get_comparable` object as the output from the kernel
        ref_answer = jac.copy()

        # reset other values
        jac[:, :, :2] = 0
        jac[:, :2, :] = 0

        # cp args
        cp_args = {'cp': lambda x: np.array(
            self.store.spec_cp, order=x, copy=True),
            'h': lambda x: np.array(
                self.store.spec_h, order=x, copy=True),
            'cp_tot': lambda x: np.array(
                cp_sum, order=x, copy=True),
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
            'dphi': lambda x: np.array(
                self.store.dphi_cp, order=x, copy=True),
            'jac': lambda x: np.array(
                jac, order=x, copy=True)}

        comp = get_comparable(compare_mask=[(np.array([0]),
                                             np.arange(2, jac.shape[1]))],
                              compare_axis=(1, 2),
                              ref_answer=[ref_answer])

        # call
        kc = [kernel_call('dTdot_dnj', comp.ref_answer,
                          compare_axis=comp.compare_axis, compare_mask=[comp],
                          equal_nan=True, **cp_args)]

        self._generic_jac_tester(dTdot_dnj, kc, conp=True)

        # conv
        cv_sum = np.sum(self.store.concs * self.store.spec_cv, axis=1)

        # get species jacobian
        jac = self.__get_full_jac(False)

        # instead of whittling this down to the actual answer [:, 0, 2:], it's
        # way easier to keep this full sized such that we can use the same
        # :class:`get_comparable` object as the output from the kernel
        ref_answer = jac.copy()

        # reset other values
        jac[:, :, :2] = 0
        jac[:, :, :2] = 0
        jac[:, :2, :] = 0

        # cv args
        cv_args = {'cv': lambda x: np.array(
            self.store.spec_cv, order=x, copy=True),
            'u': lambda x: np.array(
                self.store.spec_u, order=x, copy=True),
            'cv_tot': lambda x: np.array(
                cv_sum, order=x, copy=True),
            'dphi': lambda x: np.array(
                self.store.dphi_cv, order=x, copy=True),
            'V_arr': lambda x: np.array(
                self.store.V, order=x, copy=True),
            'jac': lambda x: np.array(
                jac, order=x, copy=True)}

        comp = get_comparable(compare_mask=[(np.array([0]),
                                             np.arange(2, jac.shape[1]))],
                              compare_axis=(1, 2),
                              ref_answer=[ref_answer])
        # call
        kc = [kernel_call('dTdot_dnj', comp.ref_answer,
                          compare_axis=comp.compare_axis, compare_mask=[comp],
                          equal_nan=True, **cv_args)]

        self._generic_jac_tester(dTdot_dnj, kc, conp=False)

    def test_dEdot_dnj(self):
        # conp

        # get species jacobian
        jac = self.__get_full_jac(True)

        # instead of whittling this down to the actual answer [:, 1, 2:], it's
        # way easier to keep this full sized such that we can use the same
        # :class:`get_comparable` object as the output from the kernel
        ref_answer = jac.copy()

        # reset set value for kernel
        jac[:, 1, :] = 0

        # cp args
        cp_args = {
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
            'jac': lambda x: np.array(
                jac, order=x, copy=True),
            'P_arr': lambda x: np.array(
                self.store.P, order=x, copy=True)}

        comp = get_comparable(ref_answer=[ref_answer], compare_axis=(1, 2),
                              compare_mask=[(np.array([1]),
                                             np.arange(2, jac.shape[1]))])

        # call
        kc = [kernel_call('dVdot_dnj', comp.ref_answer,
                          compare_axis=comp.compare_axis, compare_mask=[comp],
                          equal_nan=True, strict_name_match=True, **cp_args)]

        self._generic_jac_tester(dEdot_dnj, kc, conp=True)

        # get species jacobian
        jac = self.__get_full_jac(False)

        # instead of whittling this down to the actual answer [:, 1, 2:], it's
        # way easier to keep this full sized such that we can use the same
        # :class:`get_comparable` object as the output from the kernel
        ref_answer = jac.copy()

        # reset set value for kernel
        jac[:, 1, :] = 0

        # cv args
        cv_args = {
            'phi': lambda x: np.array(
                self.store.phi_cv, order=x, copy=True),
            'jac': lambda x: np.array(
                jac, order=x, copy=True),
            'V_arr': lambda x: np.array(
                self.store.V, order=x, copy=True)}

        comp = get_comparable(ref_answer=[ref_answer], compare_axis=(1, 2),
                              compare_mask=[(np.array([1]),
                                             np.arange(2, jac.shape[1]))])

        # call
        kc = [kernel_call('dPdot_dnj', comp.ref_answer,
                          compare_axis=comp.compare_axis, compare_mask=[comp],
                          equal_nan=True, strict_name_match=True, **cv_args)]

        self._generic_jac_tester(dEdot_dnj, kc, conp=False)

    def test_thermo_derivatives(self):
        def __test_name(myname):
            conp = myname in ['cp']
            namestore, rate_info = self._make_namestore(conp)
            ad_opts = namestore.loopy_opts

            phi = self.store.phi_cp if conp else self.store.phi_cv

            # dname/dT
            edit = editor(
                namestore.T_arr, getattr(namestore, myname),
                self.store.test_size,
                order=ad_opts.order)

            args = {
                'phi': lambda x: np.array(
                    phi, order=x, copy=True),
            }

            # obtain the finite difference jacobian
            kc = kernel_call(myname, [None], **args)

            def __call_wrapper(eqs, loopy_opts, namestore, test_size):
                return thermo_temperature_derivative(
                    name, eqs,
                    loopy_opts, namestore,
                    test_size)
            name = myname
            ref_ans = self._get_jacobian(
                __call_wrapper, kc, edit, ad_opts, namestore.conp)
            ref_ans = ref_ans[:, :, 0]

            # force all entries to zero for split comparison
            name = 'd' + myname
            args.update({name: lambda x: np.zeros_like(ref_ans, order=x)})
            # call
            kc = [kernel_call(myname, [ref_ans], **args)]

            self._generic_jac_tester(__call_wrapper, kc)

        __test_name('cp')
        __test_name('cv')
        __test_name('b')

    def test_dRopidT(self, rxn_type=reaction_type.elementary,
                     test_variable=False, conp=True):
        # test conp (form doesn't matter)
        namestore, rate_info = self._make_namestore(conp)
        ad_opts = namestore.loopy_opts

        # setup arguements
        # create the editor
        edit = editor(
            namestore.T_arr if not test_variable else namestore.E_arr,
            namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        # get kf / kf_fall
        kf, _ = self.__get_kf_and_fall()
        # and kr
        kr = self.__get_kr(kf)

        args = {
            'pres_mod': lambda x: np.array(
                self.store.ref_pres_mod, order=x, copy=True),
            'conc': lambda x: np.zeros_like(self.store.concs, order=x),
            'wdot': lambda x: np.zeros_like(self.store.species_rates, order=x),
            'rop_fwd': lambda x: np.zeros_like(
                self.store.fwd_rxn_rate, order=x),
            'rop_rev': lambda x: np.zeros_like(
                self.store.rev_rxn_rate, order=x),
            'rop_net': lambda x: np.zeros_like(self.store.rxn_rates, order=x),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'b': lambda x: np.zeros_like(
                    self.store.ref_B_rev, order=x),
            'Kc': lambda x: np.zeros_like(
                self.store.equilibrium_constants, order=x),
        }

        if test_variable:
            args.update({
                'kf': lambda x: np.array(kf, order=x, copy=True),
                'kr': lambda x: np.array(kr, order=x, copy=True)
            })

        else:
            args.update({
                'kf': lambda x: np.zeros_like(kf, order=x),
                'kr': lambda x: np.zeros_like(kr, order=x),
                #  'kf_fall': lambda x: np.zeros_like(
                #    self.store.ref_Pr, order=x)
            })

        if conp:
            args.update({
                'P_arr': lambda x: np.array(self.store.P, order=x, copy=True),
                'phi': lambda x: np.array(
                    self.store.phi_cp, order=x, copy=True),
            })
        else:
            args.update({
                'V_arr': lambda x: np.array(self.store.V, order=x, copy=True),
                'phi': lambda x: np.array(
                    self.store.phi_cv, order=x, copy=True),
            })

        # obtain the finite difference jacobian
        kc = kernel_call('dRopidT', [None], **args)

        allint = {'net': rate_info['net']['allint']}
        rate_sub = get_simple_arrhenius_rates
        if rxn_type == reaction_type.plog:
            rate_sub = self.__get_plog_call_wrapper(rate_info)
        elif rxn_type == reaction_type.cheb:
            rate_sub = self.__get_cheb_call_wrapper(rate_info)

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, conp,
            extra_funcs=[get_concentrations, rate_sub,
                         self.__get_poly_wrapper('b', conp),
                         get_rev_rates, get_rop, get_rop_net, get_spec_rates],
            allint=allint)

        # get our form of rop_fwd / rop_rev
        fwd_removed, rev_removed = self.__get_removed()

        # setup args
        args = {
            'rop_fwd': lambda x: np.array(fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(rev_removed, order=x, copy=True),
            'pres_mod': lambda x: np.array(
                self.store.ref_pres_mod, order=x, copy=True),
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'kr': lambda x: np.array(kr, order=x, copy=True),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
            'conc': lambda x: np.array(self.store.concs, order=x, copy=True)
        }

        if conp:
            args.update({
                'phi': lambda x: np.array(
                    self.store.phi_cp, order=x, copy=True),
                'P_arr': lambda x: np.array(self.store.P, order=x, copy=True)
            })
        else:
            args.update({
                'phi': lambda x: np.array(
                    self.store.phi_cv, order=x, copy=True),
                'V_arr': lambda x: np.array(self.store.V, order=x, copy=True),
            })

        # input_mask = []
        if not test_variable:
            # and finally dBk/dT
            dBkdT = self.__get_db()
            args['db'] = lambda x: np.array(dBkdT, order=x, copy=True)

        # input masking
        input_mask = ['V_arr']
        if rxn_type == reaction_type.elementary:
            input_mask.append('P_arr')
        elif test_variable:
            # needed for the test variable for the extras
            input_mask = []
            if conp and rxn_type != reaction_type.elementary:
                input_mask = ['P_arr']

        def _chainer(self, out_vals):
            if out_vals[-1][0] is not None:
                self.kernel_args['jac'] = out_vals[-1][0].copy(
                    order=self.current_order)

        def include(rxn):
            if rxn_type == reaction_type.plog:
                return isinstance(rxn, ct.PlogReaction)
            elif rxn_type == reaction_type.cheb:
                return isinstance(rxn, ct.ChebyshevReaction)
            else:
                return not (isinstance(rxn, ct.PlogReaction)
                            or isinstance(rxn, ct.ChebyshevReaction))

        # set variable name and check index
        var_name = 'T'
        diff_index = 0
        if test_variable:
            var_name = 'V' if conp else 'P'
            diff_index = 1

        # get descriptor
        name_desc = ''
        other_args = {'conp': conp} if test_variable else {}
        tester = dRopidT if not test_variable else dRopidE
        if rxn_type == reaction_type.plog:
            name_desc = '_plog'
            tester = dRopi_plog_dT if not test_variable else dRopi_plog_dE
            other_args['maxP'] = rate_info['plog']['max_P']
        elif rxn_type == reaction_type.cheb:
            name_desc = '_cheb'
            tester = dRopi_cheb_dT if not test_variable else dRopi_cheb_dE
            other_args['maxP'] = np.max(rate_info['cheb']['num_P'])
            other_args['maxT'] = np.max(rate_info['cheb']['num_T'])

        test_conditions = np.arange(self.store.test_size)
        if test_variable:
            # find states where the last species conc should be zero, as this
            # can cause some problems in the FD Jac
            test_conditions = np.where(self.store.concs[:, -1] != 0)[0]

        rtol = 1e-3
        atol = 1e-7

        def _small_compare(kc, our_vals, ref_vals, mask):
            # get the condition extractor
            extractor, cond, x, y = self.__get_comp_extractor(kc, mask)
            if extractor is None:
                # no need to test
                return True

            # find where there isn't a match
            outv = extractor(our_vals, (cond, x, y))
            refv = extractor(ref_vals, (cond, x, y))
            check = np.where(
                np.logical_not(np.isclose(outv, refv, rtol=rtol)))[0]

            correct = True
            if check.size:
                # check that our values are zero (which is correct)
                correct = np.all(outv[check] == 0)

                # and that the reference values are "small"
                correct &= np.all(np.abs(refv[check]) <= atol)

            return correct

        test = self.__get_check(include)

        comp = get_comparable(compare_mask=[(test_conditions, test,
                                            np.array([diff_index]))],
                              ref_answer=[fd_jac],
                              compare_axis=(0, 1, 2))

        # and get mask
        kc = [kernel_call('dRopi{}_d{}'.format(name_desc, var_name),
                          comp.ref_answer, check=False,
                          strict_name_match=True,
                          allow_skip=test_variable,
                          input_mask=['kf', 'kr', 'conc'] + input_mask,
                          **args),
              kernel_call('dRopi{}_d{}_ns'.format(name_desc, var_name),
                          comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_mask, chain=_chainer,
                          strict_name_match=True, allow_skip=True,
                          rtol=rtol, atol=atol, other_compare=_small_compare,
                          input_mask=['db', 'rop_rev', 'rop_fwd'],
                          **args)]

        return self._generic_jac_tester(tester, kc, **other_args)

    def test_dRopi_plog_dT(self):
        self.test_dRopidT(reaction_type.plog)

    def test_dRopi_cheb_dT(self):
        self.test_dRopidT(reaction_type.cheb)

    def test_dRopi_dE(self):
        self.test_dRopidT(reaction_type.elementary, True, conp=True)
        self.test_dRopidT(reaction_type.elementary, True, conp=False)

    def test_dRopi_plog_dE(self):
        self.test_dRopidT(reaction_type.plog, True, conp=True)
        self.test_dRopidT(reaction_type.plog, True, conp=False)

    def test_dRopi_cheb_dE(self):
        self.test_dRopidT(reaction_type.cheb, True, conp=True)
        self.test_dRopidT(reaction_type.cheb, True, conp=False)

    def __get_non_ad_params(self, conp):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        opts = loopy_options(order='C', knl_type='map', lang='opencl')
        namestore = arc.NameStore(opts, rate_info, conp, self.store.test_size)

        return namestore, rate_info, opts, eqs

    def test_dTdot_dT(self):
        def __subtest(conp):
            # conp
            fd_jac = self.__get_full_jac(conp)

            spec_heat = self.store.spec_cp if conp else self.store.spec_cv
            namestore, rate_info, opts, eqs = self.__get_non_ad_params(conp)
            phi = self.store.phi_cp if conp else self.store.phi_cv
            dphi = self.store.dphi_cp if conp else self.store.dphi_cv

            spec_heat = np.sum(self.store.concs * spec_heat, axis=1)

            jac = fd_jac.copy()
            jac[:, 0, 0] = 0

            # get dcp
            args = {'phi': lambda x: np.array(
                phi, order=x, copy=True)}
            dc = kernel_runner(self.__get_poly_wrapper(
                'dcp' if conp else 'dcv', conp),
                self.store.test_size, args)(
                eqs, opts, namestore, self.store.test_size)[0]

            args = {'conc': lambda x: np.array(
                self.store.concs, order=x, copy=True),
                'dphi': lambda x: np.array(
                dphi, order=x, copy=True),
                'phi': lambda x: np.array(
                phi, order=x, copy=True),
                'jac': lambda x: np.array(
                jac, order=x, copy=True),
                'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True)
            }

            if conp:
                args.update({
                    'h': lambda x: np.array(
                        self.store.spec_h, order=x, copy=True),
                    'cp': lambda x: np.array(
                        self.store.spec_cp, order=x, copy=True),
                    'dcp': lambda x: np.array(
                        dc, order=x, copy=True),
                    'cp_tot': lambda x: np.array(
                        spec_heat, order=x, copy=True)})
            else:
                args.update({
                    'u': lambda x: np.array(
                        self.store.spec_u, order=x, copy=True),
                    'cv': lambda x: np.array(
                        self.store.spec_cv, order=x, copy=True),
                    'dcv': lambda x: np.array(
                        dc, order=x, copy=True),
                    'cv_tot': lambda x: np.array(
                        spec_heat, order=x, copy=True),
                    'V_arr': lambda x: np.array(
                        self.store.V, order=x, copy=True)})

            # find NaN's
            to_test = np.setdiff1d(np.arange(self.store.test_size),
                                   np.unique(np.where(np.isnan(jac))[0]),
                                   assume_unique=True)
            comp = get_comparable(compare_mask=[(
                to_test, np.array([0]), np.array([0]))],
                                  compare_axis=(0, 1, 2),
                                  ref_answer=[fd_jac])
            kc = kernel_call('dTdot_dT', comp.ref_answer, check=True,
                             compare_mask=[comp], compare_axis=comp.compare_axis,
                             equal_nan=True, other_compare=self.our_nan_compare,
                             **args)

            return self._generic_jac_tester(dTdotdT, kc, conp=conp)

        # test conp
        __subtest(True)
        # test conv
        __subtest(False)

    def test_dci_thd_dT(self, rxn_type=reaction_type.thd, test_variable=False,
                        conp=True):
        # test conp (form doesn't matter)
        namestore, rate_info = self._make_namestore(conp)
        ad_opts = namestore.loopy_opts

        # setup arguements

        # get our form of rop_fwd / rop_rev
        fwd_removed, rev_removed = self.__get_removed()
        kf, kf_fall = self.__get_kf_and_fall()
        args = {
            'pres_mod': lambda x: np.zeros_like(
                self.store.ref_pres_mod, order=x),
            'conc': lambda x: np.zeros_like(self.store.concs, order=x),
            'wdot': lambda x: np.zeros_like(self.store.species_rates, order=x),
            'rop_fwd': lambda x: np.array(fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(rev_removed, order=x, copy=True),
            'rop_net': lambda x: np.zeros_like(self.store.rxn_rates, order=x),
            'thd_conc': lambda x: np.zeros_like(self.store.ref_thd, order=x),
            'kf': lambda x: np.zeros_like(kf, order=x),
            'kf_fall': lambda x: np.zeros_like(kf_fall, order=x),
            'Pr': lambda x: np.zeros_like(self.store.ref_Pr, order=x),
            'Fi': lambda x: np.zeros_like(self.store.ref_Fall, order=x),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x)
        }

        if rxn_type == falloff_form.troe:
            args.update({
                'Fcent': lambda x: np.zeros((
                    self.store.test_size, self.store.troe_inds.size), order=x),
                'Atroe': lambda x: np.zeros((
                    self.store.test_size, self.store.troe_inds.size), order=x),
                'Btroe': lambda x: np.zeros((
                    self.store.test_size, self.store.troe_inds.size), order=x),
                })
        elif rxn_type == falloff_form.sri:
            args.update({
                'X': lambda x: np.zeros((
                    self.store.test_size, self.store.sri_inds.size), order=x)
                })

        if conp:
            args.update({
                'P_arr': lambda x: np.array(self.store.P, order=x, copy=True),
                'phi': lambda x: np.array(
                    self.store.phi_cp, order=x, copy=True)
            })
        else:
            args.update({
                'V_arr': lambda x: np.array(self.store.V, order=x, copy=True),
                'phi': lambda x: np.array(
                    self.store.phi_cv, order=x, copy=True)
            })

        # obtain the finite difference jacobian
        kc = kernel_call('dci_dT', [None], **args)
        # create the editor
        edit = editor(
            namestore.T_arr if not test_variable else namestore.E_arr,
            namestore.n_dot, self.store.test_size,
            order=ad_opts.order)

        rate_sub = None
        if rxn_type == falloff_form.lind:
            rate_sub = get_lind_kernel
        elif rxn_type == falloff_form.sri:
            rate_sub = get_sri_kernel
        elif rxn_type == falloff_form.troe:
            rate_sub = get_troe_kernel
        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, conp,
            extra_funcs=[x for x in [get_concentrations, get_thd_body_concs,
                                     get_simple_arrhenius_rates,
                                     self.__get_fall_call_wrapper(),
                                     get_reduced_pressure_kernel, rate_sub,
                                     get_rxn_pres_mod, get_rop_net,
                                     get_spec_rates] if x is not None])

        # setup args

        # create rop net w/o pres mod
        args = {
            'rop_fwd': lambda x: np.array(fwd_removed, order=x, copy=True),
            'rop_rev': lambda x: np.array(rev_removed, order=x, copy=True),
            # 'conc': lambda x: np.zeros_like(self.store.concs, order=x),
            'jac': lambda x: np.zeros(namestore.jac.shape, order=x),
        }

        if conp:
            args.update({
                'P_arr': lambda x: np.array(self.store.P, order=x, copy=True),
                'phi': lambda x: np.array(
                    self.store.phi_cp, order=x, copy=True)
            })
        else:
            args.update({
                'V_arr': lambda x: np.array(self.store.V, order=x, copy=True),
                'phi': lambda x: np.array(
                    self.store.phi_cv, order=x, copy=True)
            })

        if test_variable:
            args.update({
                'pres_mod': lambda x: np.array(
                    self.store.ref_pres_mod, order=x, copy=True)
                })

        def over_rxn(rxn):
            if rxn_type == reaction_type.thd:
                # for third body, only need to worry about these
                return isinstance(rxn, ct.ThreeBodyReaction)
            # otherwise, need to look at all 3body/fall, and exclude wrong type
            return isinstance(rxn, ct.ThreeBodyReaction) or isinstance(
                rxn, ct.FalloffReaction)

        def include(rxn):
            if rxn_type == reaction_type.thd:
                return True
            elif rxn_type == falloff_form.lind:
                desc = 'Simple'
            elif rxn_type == falloff_form.sri:
                desc = 'SRI'
            elif rxn_type == falloff_form.troe:
                desc = 'Troe'
            return (isinstance(rxn, ct.FalloffReaction)
                    and rxn.falloff.type == desc)

        tester = dci_thd_dT if not test_variable else dci_thd_dE
        to_test = np.arange(self.store.test_size)
        if rxn_type != reaction_type.thd:
            args.update({
                'Pr': lambda x: np.array(
                    self.store.ref_Pr, order=x, copy=True),
                'Fi': lambda x: np.array(
                    self.store.ref_Fall, order=x, copy=True),
                'kf': lambda x: np.array(kf, order=x, copy=True),
                'kf_fall': lambda x: np.array(kf_fall, order=x, copy=True),
                'pres_mod': lambda x: np.array(
                    self.store.ref_pres_mod, order=x, copy=True)
            })
            if rxn_type == falloff_form.lind:
                to_test = np.where(np.all(
                    self.store.ref_Pr[:, self.store.lind_to_pr_map] != 0.0,
                    axis=1))[0]
                tester = dci_lind_dT if not test_variable else dci_lind_dE
            elif rxn_type == falloff_form.sri:
                to_test = np.where(np.all(
                    self.store.ref_Pr[:, self.store.sri_to_pr_map] != 0.0,
                    axis=1))[0]
                tester = dci_sri_dT if not test_variable else dci_sri_dE
                X = self.__get_sri_params(namestore)
                args.update({'X': lambda x: np.array(X, order=x, copy=True)})
            elif rxn_type == falloff_form.troe:
                to_test = np.where(np.all(
                    self.store.ref_Pr[:, self.store.troe_to_pr_map] != 0.0,
                    axis=1))[0]
                tester = dci_troe_dT if not test_variable else dci_troe_dE
                Fcent, Atroe, Btroe = self.__get_troe_params(namestore)
                args.update({
                    'Fcent': lambda x: np.array(Fcent, order=x, copy=True),
                    'Atroe': lambda x: np.array(Atroe, order=x, copy=True),
                    'Btroe': lambda x: np.array(Btroe, order=x, copy=True)
                })

        test = self.__get_check(include, over_rxn)

        if test_variable and conp:
            # need to adjust the reference answer to account for the addition
            # of the net ROP resulting from the volume derivative in this
            # Jacobian entry.

            starting_jac = np.zeros(namestore.jac.shape)

            from ..utils import get_nu
            for i, rxn in enumerate(self.store.reacs):

                # this is a bit tricky: in order to get the proper derivatives
                # in the auto-differentiation version, we set the falloff
                # blending term to zero for reaction types
                # not under consideration.  This has the side effect of forcing
                # the net ROP for these reactions to be zero in the AD-code
                #
                # Therefore, we must exclude the ROP for falloff/chemically
                # activated reactions when looking at the third body
                # derivatives
                ct_rxn = self.store.gas.reaction(i)
                if isinstance(ct_rxn, ct.FalloffReaction) and \
                        rxn_type == reaction_type.thd:
                    continue
                elif rxn_type == reaction_type.thd and not include(ct_rxn):
                    continue

                # get the net rate of progress
                ropnet = self.store.rxn_rates[:, i]
                for spec in set(rxn.reac + rxn.prod):
                    # ignore last species
                    if spec < len(self.store.specs) - 1:
                        # and the nu value for each species
                        nu = get_nu(spec, rxn)
                        # now subtract of this reactions contribution to
                        # this species' Jacobian entry
                        starting_jac[:, spec + 2, 1] += nu * ropnet

            # and place in the args
            args.update({'jac': lambda x: np.array(
                starting_jac, order=x, copy=True)})

        # and get mask
        check_ind = 1 if test_variable else 0
        comp = get_comparable(compare_mask=[(to_test, test, np.array([check_ind]))],
                              compare_axis=(0, 1, 2),
                              ref_answer=[fd_jac])
        kc = [kernel_call('dci_dT',
                          comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis,
                          other_compare=self.nan_compare,
                          **args)]

        extra_args = {}
        if test_variable:
            extra_args['conp'] = conp

        return self._generic_jac_tester(tester, kc, **extra_args)

    def test_dci_lind_dT(self):
        self.test_dci_thd_dT(falloff_form.lind)

    def test_dci_troe_dT(self):
        self.test_dci_thd_dT(falloff_form.troe)

    def test_dci_sri_dT(self):
        self.test_dci_thd_dT(falloff_form.sri)

    def test_dci_thd_dE(self):
        self.test_dci_thd_dT(reaction_type.thd, test_variable=True, conp=True)
        self.test_dci_thd_dT(reaction_type.thd, test_variable=True, conp=False)

    def test_dci_lind_dE(self):
        self.test_dci_thd_dT(falloff_form.lind, test_variable=True, conp=True)
        self.test_dci_thd_dT(falloff_form.lind, test_variable=True, conp=False)

    def test_dci_troe_dE(self):
        self.test_dci_thd_dT(falloff_form.troe, test_variable=True, conp=True)
        self.test_dci_thd_dT(falloff_form.troe, test_variable=True, conp=False)

    def test_dci_sri_dE(self):
        self.test_dci_thd_dT(falloff_form.sri, test_variable=True, conp=True)
        self.test_dci_thd_dT(falloff_form.sri, test_variable=True, conp=False)

    def test_dEdot_dT(self):
        def __subtest(conp):
            # conp
            fd_jac = self.__get_full_jac(conp)

            namestore, rate_info, opts, eqs = self.__get_non_ad_params(conp)
            phi = self.store.phi_cp if conp else self.store.phi_cv
            dphi = self.store.dphi_cp if conp else self.store.dphi_cv

            jac = fd_jac.copy()
            jac[:, 1, 0] = 0

            args = {'dphi': lambda x: np.array(
                dphi, order=x, copy=True),
                'phi': lambda x: np.array(
                phi, order=x, copy=True),
                'jac': lambda x: np.array(
                jac, order=x, copy=True),
                'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True)
            }

            if conp:
                args['P_arr'] = lambda x: np.array(
                    self.store.P, order=x, copy=True)
            else:
                args['V_arr'] = lambda x: np.array(
                    self.store.V, order=x, copy=True)

            # exclude purposefully included nan's
            to_test = np.setdiff1d(np.arange(self.store.test_size),
                                   np.unique(np.where(np.isnan(jac))[0]),
                                   assume_unique=True)
            comp = get_comparable(compare_mask=[
                (to_test, np.array([1]), np.array([0]))],
                                  compare_axis=(0, 1, 2),
                                  ref_answer=[fd_jac]
                                  )

            # and get mask
            kc = [kernel_call('dEdotdT', comp.ref_answer, compare_mask=[comp],
                              compare_axis=comp.compare_axis,
                              other_compare=self.our_nan_compare,
                              **args)]

            return self._generic_jac_tester(dEdotdT, kc, conp=conp)

        __subtest(True)
        __subtest(False)

    def test_dTdot_dE(self):
        def __subtest(conp):
            # conp
            fd_jac = self.__get_full_jac(conp)

            namestore, rate_info, opts, eqs = self.__get_non_ad_params(conp)
            phi = self.store.phi_cp if conp else self.store.phi_cv
            dphi = self.store.dphi_cp if conp else self.store.dphi_cv
            spec_heat = self.store.spec_cp if conp else self.store.spec_cv
            spec_heat_tot = np.sum(self.store.concs * spec_heat, axis=1)
            spec_energy = self.store.spec_h if conp else self.store.spec_u

            jac = fd_jac.copy()
            jac[:, 0, 1] = 0

            args = {'dphi': lambda x: np.array(
                dphi, order=x, copy=True),
                'phi': lambda x: np.array(
                phi, order=x, copy=True),
                'jac': lambda x: np.array(
                jac, order=x, copy=True),
                'wdot': lambda x: np.array(
                self.store.species_rates, order=x, copy=True)
            }

            if conp:
                args.update({
                    'cp': lambda x: np.array(
                        spec_heat, order=x, copy=True),
                    'cp_tot': lambda x: np.array(
                        spec_heat_tot, order=x, copy=True),
                    'h': lambda x: np.array(
                        spec_energy, order=x, copy=True),
                    'conc': lambda x: np.array(
                        self.store.concs, order=x, copy=True)},
                    )
            else:
                args.update({'V_arr': lambda x: np.array(
                    self.store.V, order=x, copy=True),
                    'cv': lambda x: np.array(
                        spec_heat, order=x, copy=True),
                    'cv_tot': lambda x: np.array(
                        spec_heat_tot, order=x, copy=True),
                    'u': lambda x: np.array(
                        spec_energy, order=x, copy=True)})

            # exclude purposefully included nan's
            to_test = np.setdiff1d(np.arange(self.store.test_size),
                                   np.unique(np.where(np.isnan(jac))[0]),
                                   assume_unique=True)

            comp = get_comparable(ref_answer=[fd_jac],
                                  compare_mask=[
                                  (to_test, np.array([1]), np.array([0]))],
                                  compare_axis=(0, 1, 2))
            # and get mask
            kc = [kernel_call('dTdotdE', comp.ref_answer,
                              compare_mask=[comp], compare_axis=comp.compare_axis,
                              other_compare=self.our_nan_compare,
                              **args)]

            return self._generic_jac_tester(dTdotdE, kc, conp=conp)

        __subtest(True)
        __subtest(False)

    def test_dEdot_dE(self):
        def __subtest(conp):
            # conp
            fd_jac = self.__get_full_jac(conp)

            namestore, rate_info, opts, eqs = self.__get_non_ad_params(conp)
            phi = self.store.phi_cp if conp else self.store.phi_cv
            dphi = self.store.dphi_cp if conp else self.store.dphi_cv
            jac = fd_jac.copy()
            jac[:, 1, 1] = 0

            args = {'dphi': lambda x: np.array(
                dphi, order=x, copy=True),
                'phi': lambda x: np.array(
                phi, order=x, copy=True),
                'jac': lambda x: np.array(
                jac, order=x, copy=True),
            }

            if conp:
                args.update({'P_arr': lambda x: np.array(
                    self.store.P, order=x, copy=True)})
            else:
                args.update({'V_arr': lambda x: np.array(
                    self.store.V, order=x, copy=True)})

            # exclude purposefully included nan's
            to_test = np.setdiff1d(np.arange(self.store.test_size),
                                   np.unique(np.where(np.isnan(jac))[0]),
                                   assume_unique=True)

            comp = get_comparable(ref_answer=[fd_jac],
                                  compare_mask=[
                                    (to_test, np.array([1]), np.array([1]))],
                                  compare_axis=(0, 1, 2))

            # and get mask
            kc = [kernel_call('dEdotdE', comp.ref_answer,
                              compare_mask=[comp], compare_axis=comp.compare_axis,
                              other_compare=self.our_nan_compare,
                              **args)]

            return self._generic_jac_tester(dEdotdE, kc, conp=conp)

        __subtest(True)
        __subtest(False)

    def test_index_determination(self):
        try:
            from scipy.sparse import csr_matrix, csc_matrix
        except:
            raise SkipTest('Cannot test sparse Jacobian without scipy')
        # find FD jacobian
        jac = self.__get_full_jac(True)
        # find our non-zero indicies
        ret = determine_jac_inds(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)['jac_inds']
        non_zero_inds = ret['flat']
        non_zero_inds = non_zero_inds.T

        jac_inds = np.where(jac != 0)[1:3]
        jac_inds = np.column_stack((jac_inds[0], jac_inds[1]))
        jac_inds = np.unique(jac_inds, axis=0).T

        assert np.allclose(jac_inds, non_zero_inds)

        # create a jacobian of the max of all the FD's to avoid zeros due to
        # zero rates
        jac = np.max(np.abs(jac), axis=0)

        # create a CRS
        crs = csr_matrix(jac)
        assert np.allclose(ret['crs']['row_ptr'], crs.indptr) and \
            np.allclose(ret['crs']['col_ind'], crs.indices)

        # and repeat with CCS
        ccs = csc_matrix(jac)
        assert np.allclose(ret['ccs']['col_ptr'], ccs.indptr) and \
            np.allclose(ret['ccs']['row_ind'], ccs.indices)

    def test_reset_arrays(self):
        namestore, _, _, _ = self.__get_non_ad_params(True)
        # find our non-zero indicies
        ret = determine_jac_inds(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)['jac_inds']
        non_zero_inds = ret['flat']

        jac_shape = namestore.jac.shape

        def __set(order):
            x = np.zeros(jac_shape, order=order)
            x[:, non_zero_inds[:, 0], non_zero_inds[:, 1]] = 1
            return x
        args = {'jac': __set}

        comp = get_comparable(ref_answer=[np.zeros(jac_shape)],
                              compare_mask=[
                                (slice(None), non_zero_inds[:, 0],
                                    non_zero_inds[:, 1])],
                              compare_axis=-1)

        # and get mask
        kc = kernel_call('reset_arrays', comp.ref_answer, compare_mask=[comp],
                         compare_axis=comp.compare_axis, **args)

        return self._generic_jac_tester(reset_arrays, kc)
