import numpy as np
import logging
import six
import loopy as lp
import cantera as ct

from nose.plugins.attrib import attr
from unittest.case import SkipTest
from parameterized import parameterized
try:
    from scipy.sparse import csr_matrix, csc_matrix
except ImportError:
    csr_matrix = None
    csc_matrix = None

from pyjac.core.rate_subs import (
    get_concentrations,
    get_rop, get_rop_net, get_spec_rates, get_molar_rates, get_thd_body_concs,
    get_rxn_pres_mod, get_reduced_pressure_kernel, get_lind_kernel,
    get_sri_kernel, get_troe_kernel, get_simple_arrhenius_rates,
    polyfit_kernel_gen, get_plog_arrhenius_rates, get_cheb_arrhenius_rates,
    get_rev_rates, get_temperature_rate, get_extra_var_rates)
from pyjac.loopy_utils.loopy_utils import (
    loopy_options, kernel_call, set_adept_editor, populate, get_target)
from pyjac.core.enum_types import RateSpecialization, FiniteDifferenceMode
from pyjac.core.create_jacobian import (
    dRopi_dnj, dci_thd_dnj, dci_lind_dnj, dci_sri_dnj, dci_troe_dnj,
    total_specific_energy, dTdot_dnj, dEdot_dnj, thermo_temperature_derivative,
    dRopidT, dRopi_plog_dT, dRopi_cheb_dT, dTdotdT, dci_thd_dT, dci_lind_dT,
    dci_troe_dT, dci_sri_dT, dEdotdT, dTdotdE, dEdotdE, dRopidE, dRopi_plog_dE,
    dRopi_cheb_dE, dci_thd_dE, dci_lind_dE, dci_troe_dE, dci_sri_dE,
    determine_jac_inds, reset_arrays, get_jacobian_kernel,
    finite_difference_jacobian)
from pyjac.core import array_creator as arc
from pyjac.core.enum_types import reaction_type, falloff_form
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.tests import get_test_langs, TestClass
from pyjac.tests.test_utils import (
    kernel_runner, get_comparable, _generic_tester,
    _full_kernel_test, with_check_inds, inNd, skipif, xfail)
from pyjac.core.enum_types import KernelType
from pyjac import utils


class editor(object):
    def __init__(self, independent, dependent,
                 problem_size, order, do_not_set=[],
                 skip_on_missing=None):

        def __replace_problem_size(shape):
            new_shape = []
            for x in shape:
                if x != arc.problem_size.name:
                    new_shape.append(x)
                else:
                    new_shape.append(problem_size)
            return tuple(new_shape)

        assert len(independent.shape) == 2
        self.independent = independent.copy(shape=__replace_problem_size(
            independent.shape))
        indep_size = independent.shape[1]

        assert len(dependent.shape) == 2
        self.dependent = dependent.copy(shape=__replace_problem_size(
            dependent.shape))
        dep_size = dependent.shape[1]
        self.problem_size = problem_size

        # create the jacobian
        self.output = arc.creator('jac', np.float64,
                                  (problem_size, dep_size, indep_size),
                                  order=order)
        self.output = self.output(*['i', 'j', 'k'])[0]
        self.do_not_set = utils.listify(do_not_set)
        self.skip_on_missing = skip_on_missing

    def set_single_kernel(self, single_kernel):
        """
        It's far easier to use two generated kernels, one that uses the full
        problem size (for calling via loopy), and another that uses a problem
        size of 1, to work with Adept indexing in the AD kernel
        """
        self.single_kernel = single_kernel

    def set_skip_on_missing(self, func):
        """
            If set, skip if the :class:`kernel_info` returned by this function
            is None
        """
        self.skip_on_missing = func

    def __call__(self, knl):
        return set_adept_editor(knl, self.single_kernel, self.problem_size,
                                self.independent, self.dependent, self.output,
                                self.do_not_set)


# various convenience wrappers
def _get_fall_call_wrapper():
    def fall_wrapper(loopy_opts, namestore, test_size):
        return get_simple_arrhenius_rates(loopy_opts, namestore,
                                          test_size, falloff=True)
    return fall_wrapper


def _get_plog_call_wrapper(rate_info):
    def plog_wrapper(loopy_opts, namestore, test_size):
        if rate_info['plog']['num']:
            return get_plog_arrhenius_rates(loopy_opts, namestore,
                                            rate_info['plog']['max_P'],
                                            test_size)
    return plog_wrapper


def _get_cheb_call_wrapper(rate_info):
    def cheb_wrapper(loopy_opts, namestore, test_size):
        if rate_info['cheb']['num']:
            return get_cheb_arrhenius_rates(loopy_opts, namestore,
                                            np.max(rate_info['cheb']['num_P']),
                                            np.max(rate_info['cheb']['num_T']),
                                            test_size)
    return cheb_wrapper


def _get_poly_wrapper(name, conp):
    def poly_wrapper(loopy_opts, namestore, test_size):
        return polyfit_kernel_gen(name, loopy_opts, namestore, test_size)
    return poly_wrapper


def _get_ad_jacobian(self, test_size, conp=True, pregen=None, return_kernel=False):
    """
    Convenience method to evaluate the finite difference Jacobian from a given
    Phi / parameter set

    Parameters
    ----------
    test_size: int
        The number of conditions to test
    conp: bool
        If True, CONP else CONV
    pregen: Callable [None]
        If not None, this corresponds to a previously generated AD-Jacobian kernel
        Used in the validation tester to speed up chunked Jacobian evaluation
    return_kernel: bool [False]
        If True, we want __get_jacobian to return the kernel and kernel call
        rather than the evaluated array (to be used with :param:`pregen`)
    """

    class create_arr(object):
        def __init__(self, dim):
            self.dim = dim

        @classmethod
        def new(cls, inds):
            if isinstance(inds, np.ndarray):
                dim = inds.size
            elif isinstance(inds, list):
                dim = len(inds)
            elif isinstance(inds, arc.creator):
                dim = inds.initializer.size
            elif isinstance(inds, int):
                dim = inds
            else:
                return None
            return cls(dim)

        def __call__(self, order):
            return np.zeros((test_size, self.dim), order=order)

    # get rate info
    rate_info = determine_jac_inds(
        self.store.reacs, self.store.specs, RateSpecialization.fixed)

    # create loopy options
    # --> have to turn off the temperature guard to avoid fmin / max issues with
    #     Adept
    ad_opts = loopy_options(order='C', lang='c', auto_diff=True)

    # create namestore
    store = arc.NameStore(ad_opts, rate_info, conp, test_size)

    # and the editor
    edit = editor(store.n_arr, store.n_dot, test_size,
                  order=ad_opts.order)
    # setup args
    phi = self.store.phi_cp if conp else self.store.phi_cv
    allint = {'net': rate_info['net']['allint']}
    args = {
        'phi': lambda x: np.array(phi, order=x, copy=True),
        'jac': lambda x: np.zeros((test_size,) + store.jac.shape[1:], order=x),
        'wdot': create_arr.new(store.num_specs),
        'Atroe': create_arr.new(store.num_troe),
        'Btroe': create_arr.new(store.num_troe),
        'Fcent': create_arr.new(store.num_troe),
        'Fi': create_arr.new(store.num_fall),
        'Pr': create_arr.new(store.num_fall),
        'X': create_arr.new(store.num_sri),
        'conc': create_arr.new(store.num_specs),
        'dphi': lambda x: np.zeros_like(phi, order=x),
        'kf': create_arr.new(store.num_reacs),
        'kf_fall': create_arr.new(store.num_fall),
        'kr': create_arr.new(store.num_rev_reacs),
        'pres_mod': create_arr.new(store.num_thd),
        'rop_fwd': create_arr.new(store.num_reacs),
        'rop_rev': create_arr.new(store.num_rev_reacs),
        'rop_net': create_arr.new(store.num_reacs),
        'thd_conc': create_arr.new(store.num_thd),
        'b': create_arr.new(store.num_specs),
        'Kc': create_arr.new(store.num_rev_reacs)
    }
    if conp:
        args['P_arr'] = lambda x: np.array(self.store.P, order=x, copy=True)
        args['h'] = create_arr.new(store.num_specs)
        args['cp'] = create_arr.new(store.num_specs)
    else:
        args['V_arr'] = lambda x: np.array(self.store.V, order=x, copy=True)
        args['u'] = create_arr.new(store.num_specs)
        args['cv'] = create_arr.new(store.num_specs)

    # trim unused args
    args = {k: v for k, v in six.iteritems(args) if v is not None}

    # obtain the finite difference jacobian
    kc = kernel_call('dnkdnj', [None], **args)

    # check for pregenerated kernel
    if pregen is not None:
        return pregen(kc)

    __b_call_wrapper = _get_poly_wrapper('b', conp)

    __cp_call_wrapper = _get_poly_wrapper('cp', conp)

    __cv_call_wrapper = _get_poly_wrapper('cv', conp)

    __h_call_wrapper = _get_poly_wrapper('h', conp)

    __u_call_wrapper = _get_poly_wrapper('u', conp)

    def __extra_call_wrapper(loopy_opts, namestore, test_size):
        return get_extra_var_rates(loopy_opts, namestore,
                                   conp=conp, test_size=test_size)

    def __temperature_wrapper(loopy_opts, namestore, test_size):
        return get_temperature_rate(loopy_opts, namestore,
                                    conp=conp, test_size=test_size)

    return _get_jacobian(
        self, __extra_call_wrapper, kc, edit, ad_opts, conp,
        extra_funcs=[get_concentrations, get_simple_arrhenius_rates,
                     _get_plog_call_wrapper(rate_info),
                     _get_cheb_call_wrapper(rate_info),
                     get_thd_body_concs, _get_fall_call_wrapper(),
                     get_reduced_pressure_kernel, get_lind_kernel,
                     get_sri_kernel, get_troe_kernel,
                     __b_call_wrapper, get_rev_rates,
                     get_rxn_pres_mod, get_rop, get_rop_net,
                     get_spec_rates] + (
            [__h_call_wrapper, __cp_call_wrapper] if conp else
            [__u_call_wrapper, __cv_call_wrapper]) + [
            get_molar_rates, __temperature_wrapper],
        allint=allint, return_kernel=return_kernel)


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


def _get_jacobian(self, func, kernel_call, editor, ad_opts, conp, extra_funcs=[],
                  return_kernel=False, **kwargs):
    """
    Computes an autodifferentiated kernel, exposed to external classes in order
    to share with the :mod:`functional_tester`

    Parameters
    ----------
    func: Callable
        The function to autodifferentiate
    kernel_call: :class:`kernel_call`
        The kernel call with arguements, etc. to use
    editor: :class:`editor`
        The jacobian editor responsible for creating the AD kernel
    ad_opts: :class:`loopy_options`
        The AD enabled loopy options object
    extra_funcs: list of Callable
        Additional functions that must be called before :param:`func`.
        These can be used to chain together functions to find derivatives of
        complicated values (e.g. ROP)
    return_kernel: bool [False]
        If True, return a callable function that takes as as an arguement the
        new kernel_call w/ updated args and returns the result
        Note: The user is responsible for checking that the arguements are of
            valid shape
    kwargs: dict
        Additional args for :param:`func

    Returns
    -------
    ad_jac : :class:`numpy.ndarray`
        The resulting autodifferentiated jacobian.  The shape of which depends on
        the values specified in the editor
    """
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
            # py2-3 compat
            arg_count = check.func_code.co_argcount
            args = check.func_code.co_varnames[:arg_count]
        except AttributeError:
            arg_count = check.__code__.co_argcount
            args = check.__code__.co_varnames[:arg_count]

        args_dict = {}
        for k, v in six.iteritems(in_args):
            if k in args:
                args_dict[k] = v
        return args_dict

    # create the kernel info
    infos = []
    info = func(ad_opts, namestore,
                test_size=self.store.test_size,
                **__get_arg_dict(func, **kwargs))

    infos.extend(utils.listify(info))

    # create a dummy kernel generator
    knl = k_gen.make_kernel_generator(
        kernel_type=KernelType.jacobian,
        loopy_opts=ad_opts,
        kernels=infos,
        namestore=namestore,
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
    def __raise(f):
        raise SkipTest('Mechanism {} does not contain derivatives corresponding to '
                       '{}'.format(self.store.gas.name, f.__name__))
    infos = []
    for f in extra_funcs:
        info = f(ad_opts, namestore,
                 test_size=self.store.test_size,
                 **__get_arg_dict(f, **kwargs))
        is_skip = editor.skip_on_missing is not None and \
            f == editor.skip_on_missing

        if is_skip and any(x is None for x in utils.listify(info)):
            # empty map (e.g. no PLOG)
            __raise(f)
        infos.extend([x for x in utils.listify(info) if x is not None])

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
        info = f(ad_opts, single_name,
                 test_size=1,
                 **__get_arg_dict(f, **kwargs))

        for i in utils.listify(info):
            if f == func and have_match and kernel_call.name != i.name:
                continue
            if i is None:
                # empty map (e.g. no PLOG)
                continue
            single_info.append(i)

    single_knl = k_gen.make_kernel_generator(
        kernel_type=KernelType.species_rates,
        loopy_opts=ad_opts,
        kernels=single_info,
        namestore=single_name,
        test_size=1,
        extra_kernel_data=[editor.output]
    )

    single_knl._make_kernels()

    # set in editor
    editor.set_single_kernel(single_knl.kernels)

    kernel_call.set_state(single_knl.array_split, ad_opts.order)

    # and place output
    kernel_call.kernel_args[editor.output.name] = np.zeros(
        editor.output.shape,
        order=editor.output.order)
    # and finally tell us not to copy
    kernel_call.do_not_copy.add(editor.output.name)

    if return_kernel:
        def __pregen(kernel_call):
            # setup the kernel call
            # reset the state
            kernel_call.set_state(single_knl.array_split, ad_opts.order)
            # and place output
            kernel_call.kernel_args[editor.output.name] = np.zeros(
                editor.output.shape,
                order=editor.output.order)
            # and finally tell us not to copy
            kernel_call.do_not_copy.add(editor.output.name)
            # run
            populate([knl[0]], kernel_call, editor=editor)
            # get result
            return _make_array(self, kernel_call.kernel_args[editor.output.name])
        return __pregen

    # run kernel
    populate([knl[0]], kernel_call, editor=editor)

    return _make_array(self, kernel_call.kernel_args[editor.output.name])


class SubTest(TestClass):
    """
    The base Jacobian tester class
    """

    def setUp(self):
        # steal the global function decls

        self._get_jacobian = lambda *args, **kwargs: _get_jacobian(
            self, *args, **kwargs)
        self._make_array = lambda *args, **kwargs: _make_array(
            self, *args, **kwargs)
        self._get_ad_jacobian = lambda *args, **kwargs: _get_ad_jacobian(
            self, *args, **kwargs)

        super(SubTest, self).setUp()

    def _generic_jac_tester(self, func, kernel_calls, do_ratespec=False,
                            do_ropsplit=None, do_conp=False, do_sparse=True,
                            sparse_only=False, **kwargs):
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
        do_sparse: bool [True]
            Test the sparse Jacobian as well
        sparse_only: bool [False]
            Test only the sparse jacobian (e.g. for testing indexing)
        """

        _generic_tester(self, func, kernel_calls, determine_jac_inds,
                        do_ratespec=do_ratespec, do_ropsplit=do_ropsplit,
                        do_conp=do_conp, do_sparse=do_sparse,
                        sparse_only=sparse_only, **kwargs)

    def _make_namestore(self, conp):
        # get number of sri reactions
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        ad_opts = loopy_options(order='C', lang='c', auto_diff=True)

        # create namestore
        namestore = arc.NameStore(ad_opts, rate_info, conp, self.store.test_size)

        return namestore, rate_info

    @attr('long')
    @with_check_inds(check_inds={
        1: lambda self: 2 + np.arange(self.store.gas.n_species - 1),
        2: lambda self: 2 + np.arange(self.store.gas.n_species - 1)})
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

        comp = self._get_compare(fd_jac)
        # and test
        kc = [kernel_call('dRopidnj', [fd_jac], check=False,
                          strict_name_match=True, **args),
              kernel_call('dRopidnj_ns', comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True, **args)]

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
                    if spec == self.store.gas.species_names[-1]:
                        # ns derivative -> no jacobian entry
                        continue
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

        def __get_val(vals, mask, **kwargs):
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
            def __get_val(vals, *args, **kwargs):
                return mask(kc, vals, 0, **kwargs)
            extractor = __get_val

        # and return the extractor
        return extractor, cond, x, y

    @attr('long')
    @with_check_inds(check_inds={
        # get list of species not in falloff / chemically activated
        # to get the check mask
        1: lambda self: self.__get_dci_check(
            lambda x: isinstance(x, ct.ThreeBodyReaction)),
        2: lambda self: 2 + np.arange(self.store.gas.n_species - 1)})
    def test_dci_thd_dnj(self):
        # test conp
        namestore, rate_info = self._make_namestore(True)
        ad_opts = namestore.loopy_opts

        # setup arguemetns
        # create the editor
        edit = editor(
            namestore.n_arr, namestore.n_dot, self.store.test_size,
            order=ad_opts.order, skip_on_missing=get_thd_body_concs)

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

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        # and get mask
        comp = self._get_compare(fd_jac)
        kc = [kernel_call('dci_thd_dnj', comp.ref_answer, check=False,
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
                (np.min(np.abs(ref_vals[bad])) / fac) > 1e10)

            # and ensure all our values are 'large' but finite numbers
            # _or_ allow_our_nans is True _and_ they're all nan's
            is_correct = is_correct and (
                (allow_our_nans and np.all(np.isnan(our_vals[bad]))) or
                np.all(np.abs(our_vals[bad]) >= utils.inf_cutoff))

            return is_correct

        return __compare(extractor(our_val, (cond, x, y)),
                         extractor(ref_val, (cond, x, y), is_answer=True))

    def our_nan_compare(self, kc, our_val, ref_val, mask):
        return self.nan_compare(kc, our_val, ref_val, mask, allow_our_nans=True)

    def __get_removed(self):
        # get our form of rop_fwd / rop_rev
        fwd_removed = self.store.fwd_rxn_rate.copy()
        rev_removed = self.store.rev_rxn_rate.copy()
        if self.store.thd_inds.size:
            with np.errstate(divide='ignore', invalid='ignore'):
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
        opts = loopy_options(order='C', lang='c')
        namestore = arc.NameStore(opts, rate_info, True, self.store.test_size)

        # get kf
        runner = kernel_runner(get_simple_arrhenius_rates,
                               self.store.test_size, args)
        kf = runner(opts, namestore, self.store.test_size)['kf']

        if self.store.ref_Pr.size:
            args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                    'kf_fall': lambda x: np.zeros_like(self.store.ref_Fall, order=x)}
            # get kf_fall
            runner = kernel_runner(get_simple_arrhenius_rates,
                                   self.store.test_size, args,
                                   {'falloff': True})
            kf_fall = runner(opts, namestore, self.store.test_size)['kf_fall']
        else:
            kf_fall = None

        if namestore.num_plog is not None:
            args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                    'kf': lambda x: np.array(kf, order=x, copy=True)}
            if conp:
                args['P_arr'] = lambda x: np.array(
                    self.store.P, order=x, copy=True)
            # get plog
            runner = kernel_runner(_get_plog_call_wrapper(rate_info),
                                   self.store.test_size, args)
            kf = runner(opts, namestore, self.store.test_size)['kf']

        if namestore.num_cheb is not None:
            args = {'phi': lambda x: np.array(phi, order=x, copy=True),
                    'kf': lambda x: np.array(kf, order=x, copy=True)}
            if conp:
                args['P_arr'] = lambda x: np.array(
                    self.store.P, order=x, copy=True)
            # get plog
            runner = kernel_runner(_get_cheb_call_wrapper(rate_info),
                                   self.store.test_size, args)
            kf = runner(opts, namestore, self.store.test_size)['kf']

        return kf, kf_fall

    def __get_kr(self, kf):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        args = {
            'kf': lambda x: np.array(kf, order=x, copy=True),
            'b': lambda x: np.array(
                self.store.ref_B_rev, order=x, copy=True)}
        opts = loopy_options(order='C', lang='c')
        namestore = arc.NameStore(opts, rate_info, True, self.store.test_size)
        allint = {'net': rate_info['net']['allint']}

        # get kf
        runner = kernel_runner(get_rev_rates,
                               self.store.test_size, args, {'allint': allint})
        kr = runner(opts, namestore, self.store.test_size)['kr']
        return kr

    def __get_db(self):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)
        opts = loopy_options(order='C', lang='c')
        namestore = arc.NameStore(opts, rate_info, True, self.store.test_size)
        # need dBk/dT
        args = {
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
        }

        def __call_wrapper(loopy_opts, namestore, test_size):
            return thermo_temperature_derivative(
                'db',
                loopy_opts, namestore,
                test_size)
        # get db
        runner = kernel_runner(__call_wrapper, self.store.test_size, args)
        return runner(opts, namestore, self.store.test_size)['db']

    @attr('long')
    @with_check_inds(check_inds={
        1: lambda self: self.__get_dci_check(
            lambda rxn: isinstance(rxn, ct.FalloffReaction) and
            rxn.falloff.type == 'Simple'),
        2: lambda self: 2 + np.arange(self.store.gas.n_species - 1)
        })
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
            order=ad_opts.order, skip_on_missing=get_lind_kernel)

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

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        # and get mask
        comp = self._get_compare(fd_jac)
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
            self.store.phi_cp, order=x, copy=True)}
        runner = kernel_runner(get_sri_kernel, self.store.test_size, sri_args)
        opts = loopy_options(order='C', lang='c')
        X = runner(opts, namestore, self.store.test_size)['X']
        return X

    @attr('long')
    @with_check_inds(check_inds={
        # find non-NaN SRI entries for testing
        # NaN entries will be handled by :func:`nan_compare`
        0: lambda self: np.where(np.all(
            self.store.ref_Pr[:, self.store.sri_to_pr_map] != 0.0, axis=1))[0],
        1: lambda self: self.__get_dci_check(
            lambda rxn: isinstance(rxn, ct.FalloffReaction) and
            rxn.falloff.type == 'SRI'),
        2: lambda self: 2 + np.arange(self.store.gas.n_species - 1)
        })
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
            order=ad_opts.order, skip_on_missing=get_sri_kernel)

        if not rate_info['fall']['sri']['num']:
            raise SkipTest('No SRI reactions in mechanism {}'.format(
                self.store.gas.name))

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

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        # and get mask
        comp = self._get_compare(fd_jac)
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
            self.store.phi_cp, order=x, copy=True)}
        runner = kernel_runner(
            get_troe_kernel, self.store.test_size, troe_args)
        opts = loopy_options(order='C', lang='c')
        Fcent, Atroe, Btroe = [runner(
            opts, namestore, self.store.test_size)[x] for x in
            ['Fcent', 'Atroe', 'Btroe']]
        return Fcent, Atroe, Btroe

    @attr('long')
    @with_check_inds(check_inds={
        # find non-NaN Troe entries for testing
        # NaN entries will be handled by :func:`nan_compare`
        0: lambda self: np.where(np.all(
            self.store.ref_Pr[:, self.store.troe_to_pr_map] != 0.0, axis=1))[0],
        1: lambda self: self.__get_dci_check(
            lambda rxn: isinstance(rxn, ct.FalloffReaction) and
            rxn.falloff.type == 'Troe'),
        2: lambda self: 2 + np.arange(self.store.gas.n_species - 1)
        })
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
            order=ad_opts.order, skip_on_missing=get_troe_kernel)

        if not rate_info['fall']['troe']['num']:
            raise SkipTest('No Troe reactions in mechanism {}'.format(
                self.store.gas.name))

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

        def _chainer(self, out_vals):
            self.kernel_args['jac'] = out_vals[-1][0].copy(
                order=self.current_order)

        comp = self._get_compare(fd_jac)
        # and get mask
        kc = [kernel_call('dci_troe_dnj', comp.ref_answer, check=False,
                          strict_name_match=True, **args),
              kernel_call('dci_troe_dnj_ns', comp.ref_answer,
                          compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True,
                          other_compare=self.nan_compare, **args)]

        return self._generic_jac_tester(dci_troe_dnj, kc)

    @attr('long')
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

        self._generic_jac_tester(total_specific_energy, kc, conp=True,
                                 do_sparse=False)

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

        self._generic_jac_tester(total_specific_energy, kc, conp=False,
                                 do_sparse=False)

    def __get_full_jac(self, conp=True):
        # see if we've already computed this, no need to redo if we have it
        attr = 'fd_jac' + ('_cp' if conp else '_cv')
        if hasattr(self.store, attr):
            return getattr(self.store, attr).copy()

        # get the jacobian
        jac = self._get_ad_jacobian(self.store.test_size, conp=conp)

        # store the jacobian for later
        setattr(self.store, attr, jac.copy())

        return jac

    @attr('long')
    @with_check_inds(check_inds={
        1: np.array([0]),
        2: lambda self: np.arange(2, self.store.jac_dim)
        })
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

        # reset the values to be populated
        self._set_at(jac, 0)

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

        comp = self._get_compare(ref_answer)

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

        # reset the values to be populated
        self._set_at(jac, 0)

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

        comp = self._get_compare(ref_answer)
        # call
        kc = [kernel_call('dTdot_dnj', comp.ref_answer,
                          compare_axis=comp.compare_axis, compare_mask=[comp],
                          equal_nan=True, **cv_args)]

        self._generic_jac_tester(dTdot_dnj, kc, conp=False)

    @attr('long')
    @with_check_inds(check_inds={
        1: np.array([1]),
        2: lambda self: np.arange(2, self.store.jac_dim)
        })
    def test_dEdot_dnj(self):
        # conp

        # get species jacobian
        jac = self.__get_full_jac(True)

        # instead of whittling this down to the actual answer [:, 1, 2:], it's
        # way easier to keep this full sized such that we can use the same
        # :class:`get_comparable` object as the output from the kernel
        ref_answer = jac.copy()

        # reset values to be populated by kernel
        self._set_at(jac, 0)

        # cp args
        cp_args = {
            'phi': lambda x: np.array(
                self.store.phi_cp, order=x, copy=True),
            'jac': lambda x: np.array(
                jac, order=x, copy=True),
            'P_arr': lambda x: np.array(
                self.store.P, order=x, copy=True)}

        # get the compare mask
        comp = self._get_compare(ref_answer)
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

        # reset values to be populated by kernel
        self._set_at(jac, 0)

        # cv args
        cv_args = {
            'phi': lambda x: np.array(
                self.store.phi_cv, order=x, copy=True),
            'jac': lambda x: np.array(
                jac, order=x, copy=True),
            'V_arr': lambda x: np.array(
                self.store.V, order=x, copy=True)}

        # get the compare mask
        comp = self._get_compare(ref_answer)
        # call
        kc = [kernel_call('dPdot_dnj', comp.ref_answer,
                          compare_axis=comp.compare_axis, compare_mask=[comp],
                          equal_nan=True, strict_name_match=True, **cv_args)]

        self._generic_jac_tester(dEdot_dnj, kc, conp=False)

    @attr('long')
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

            def __call_wrapper(loopy_opts, namestore, test_size):
                return thermo_temperature_derivative(
                    name,
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

            self._generic_jac_tester(__call_wrapper, kc, do_sparse=False)

        __test_name('cp')
        __test_name('cv')
        __test_name('b')

    def __run_ropi_test(self, rxn_type=reaction_type.elementary,
                        test_variable=False, conp=True):
        #  setup for FD jac
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
        }

        if test_variable and (rxn_type == reaction_type.elementary or conp):
            args.update({
                'kf': lambda x: np.array(kf, order=x, copy=True),
                'kr': lambda x: np.array(kr, order=x, copy=True)
            })

        else:
            args.update({
                'kf': lambda x: np.zeros_like(kf, order=x),
                'kr': lambda x: np.zeros_like(kr, order=x),
                'b': lambda x: np.zeros_like(
                        self.store.ref_B_rev, order=x),
                'Kc': lambda x: np.zeros_like(
                    self.store.equilibrium_constants, order=x),
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
            if not rate_info['plog']['num']:
                raise SkipTest('No PLOG reactions in mechanism {}'.format(
                    self.store.gas.name))
            rate_sub = _get_plog_call_wrapper(rate_info)
        elif rxn_type == reaction_type.cheb:
            if not rate_info['cheb']['num']:
                raise SkipTest('No Chebyshev reactions in mechanism {}'.format(
                    self.store.gas.name))
            rate_sub = _get_cheb_call_wrapper(rate_info)

        edit.set_skip_on_missing(rate_sub)
        rate_sub = [rate_sub] + [_get_poly_wrapper('b', conp), get_rev_rates]

        if test_variable and (rxn_type == reaction_type.elementary or conp):
            rate_sub = []

        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, conp,
            extra_funcs=[get_concentrations] + rate_sub +
                        [get_rop, get_rop_net, get_spec_rates],
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

        # set variable name and check index
        var_name = 'T'
        if test_variable:
            var_name = 'V' if conp else 'P'

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
            refv = extractor(ref_vals, (cond, x, y), is_answer=True)
            check = np.where(
                np.logical_not(np.isclose(outv, refv, rtol=rtol)))[0]

            correct = True
            if check.size:
                # check that our values are zero (which is correct)
                correct = np.all(outv[check] == 0)

                # and that the reference values are "small"
                correct &= np.all(np.abs(refv[check]) <= atol)

            return correct

        # get compare mask
        comp = self._get_compare(fd_jac)
        kc = [kernel_call('dRopi{}_d{}'.format(name_desc, var_name),
                          comp.ref_answer, check=False,
                          strict_name_match=True,
                          allow_skip=test_variable,
                          input_mask=['kf', 'kr', 'conc'] + input_mask,
                          **args),
              kernel_call('dRopi{}_d{}_ns'.format(name_desc, var_name),
                          comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis, chain=_chainer,
                          strict_name_match=True, allow_skip=True,
                          rtol=rtol, atol=atol, other_compare=_small_compare,
                          input_mask=['db', 'rop_rev', 'rop_fwd'],
                          **args)]

        return self._generic_jac_tester(tester, kc, **other_args)

    @attr('long')
    @with_check_inds(check_inds={
        1: lambda self: self.__get_check(
            lambda rxn: not (isinstance(rxn, ct.PlogReaction)
                             or isinstance(rxn, ct.ChebyshevReaction))),
        2: np.array([0])
        })
    def test_dRopidT(self):
        self.__run_ropi_test()

    @attr('long')
    @with_check_inds(check_inds={
        1: lambda self: self.__get_check(
            lambda rxn: isinstance(rxn, ct.PlogReaction)),
        2: np.array([0])
        })
    def test_dRopi_plog_dT(self):
        self.__run_ropi_test(reaction_type.plog)

    @attr('long')
    @with_check_inds(check_inds={
        1: lambda self: self.__get_check(
            lambda rxn: isinstance(rxn, ct.ChebyshevReaction)),
        2: np.array([0])
        })
    def test_dRopi_cheb_dT(self):
        self.__run_ropi_test(reaction_type.cheb)

    @attr('long')
    @with_check_inds(check_inds={
        # find states where the last species conc should be zero, as this
        # can cause some problems in the FD Jac
        0: lambda self: np.where(self.store.concs[:, -1] != 0)[0],
        1: lambda self: self.__get_check(
            lambda rxn: not (isinstance(rxn, ct.PlogReaction)
                             or isinstance(rxn, ct.ChebyshevReaction))),
        2: np.array([1])
        })
    def test_dRopi_dE(self):
        self.__run_ropi_test(test_variable=True, conp=True)
        self.__run_ropi_test(test_variable=True, conp=False)

    @attr('long')
    @with_check_inds(check_inds={
        # find states where the last species conc should be zero, as this
        # can cause some problems in the FD Jac
        0: lambda self: np.where(self.store.concs[:, -1] != 0)[0],
        1: lambda self: self.__get_check(
            lambda rxn: isinstance(rxn, ct.PlogReaction)),
        2: np.array([1])
        })
    def test_dRopi_plog_dE(self):
        self.__run_ropi_test(reaction_type.plog, True, conp=True)
        self.__run_ropi_test(reaction_type.plog, True, conp=False)

    @attr('long')
    @with_check_inds(check_inds={
        # find states where the last species conc should be zero, as this
        # can cause some problems in the FD Jac
        0: lambda self: np.where(self.store.concs[:, -1] != 0)[0],
        1: lambda self: self.__get_check(
            lambda rxn: isinstance(rxn, ct.ChebyshevReaction)),
        2: np.array([1])
        })
    def test_dRopi_cheb_dE(self):
        self.__run_ropi_test(reaction_type.cheb, True, conp=True)
        self.__run_ropi_test(reaction_type.cheb, True, conp=False)

    def __get_non_ad_params(self, conp):
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = determine_jac_inds(reacs, specs, RateSpecialization.fixed)

        opts = loopy_options(order='C', lang='c')
        namestore = arc.NameStore(opts, rate_info, conp, self.store.test_size)

        return namestore, rate_info, opts

    @with_check_inds(check_inds={
        1: np.array([0]),
        2: np.array([0]),
        }, custom_checks={
        #  find NaN's
        0: lambda self, conp: np.setdiff1d(
            np.arange(self.store.test_size), np.unique(np.where(np.isnan(
                self.__get_full_jac(conp)))[0]), assume_unique=True)
        }
    )
    def __run_dtdot_dt(self, conp):
        # get the full jacobian
        fd_jac = self.__get_full_jac(conp)

        spec_heat = self.store.spec_cp if conp else self.store.spec_cv
        namestore, rate_info, opts = self.__get_non_ad_params(conp)
        phi = self.store.phi_cp if conp else self.store.phi_cv
        dphi = self.store.dphi_cp if conp else self.store.dphi_cv

        spec_heat = np.sum(self.store.concs * spec_heat, axis=1)

        jac = fd_jac.copy()

        # reset values to be populated
        self._set_at(jac, 0)

        # get dcp
        args = {'phi': lambda x: np.array(
            phi, order=x, copy=True)}
        dc_name = 'dcp' if conp else 'dcv'
        dc = kernel_runner(_get_poly_wrapper(dc_name, conp),
                           self.store.test_size, args)(
                           opts, namestore, self.store.test_size)[dc_name]

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
        comp = self._get_compare(fd_jac)
        kc = kernel_call('dTdot_dT', comp.ref_answer, check=True,
                         compare_mask=[comp], compare_axis=comp.compare_axis,
                         equal_nan=True, other_compare=self.our_nan_compare,
                         **args)

        return self._generic_jac_tester(dTdotdT, kc, conp=conp)

    @attr('long')
    def test_dTdot_dT(self):
        # test conp
        self.__run_dtdot_dt(True)
        # test conv
        self.__run_dtdot_dt(False)

    def __run_dci_thd_dvar(self, rxn_type=reaction_type.thd, test_variable=False,
                           conp=True):
        # setup the namestore and options
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
        # tell the editor to raise a skip test if we don't have this type of falloff
        # / rxn
        edit.set_skip_on_missing(rate_sub)
        fd_jac = self._get_jacobian(
            get_molar_rates, kc, edit, ad_opts, conp,
            extra_funcs=[x for x in [get_concentrations, get_thd_body_concs,
                                     get_simple_arrhenius_rates,
                                     _get_fall_call_wrapper(),
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

        tester = dci_thd_dT if not test_variable else dci_thd_dE
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
                tester = dci_lind_dT if not test_variable else dci_lind_dE
            elif rxn_type == falloff_form.sri:
                tester = dci_sri_dT if not test_variable else dci_sri_dE
                X = self.__get_sri_params(namestore)
                args.update({'X': lambda x: np.array(X, order=x, copy=True)})
            elif rxn_type == falloff_form.troe:
                tester = dci_troe_dT if not test_variable else dci_troe_dE
                Fcent, Atroe, Btroe = self.__get_troe_params(namestore)
                args.update({
                    'Fcent': lambda x: np.array(Fcent, order=x, copy=True),
                    'Atroe': lambda x: np.array(Atroe, order=x, copy=True),
                    'Btroe': lambda x: np.array(Btroe, order=x, copy=True)
                })

        # get the compare mask
        comp = self._get_compare(fd_jac)
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

                def __is_fall(rxn):
                    return (reaction_type.fall in rxn.type or
                            reaction_type.chem in rxn.type)

                # if we're looking at third bodies, and it's a falloff
                if rxn_type == reaction_type.thd and __is_fall(rxn):
                    continue
                # or if we're looking at falloffs, and it doesn't match
                elif rxn_type != reaction_type.thd and rxn_type not in rxn.type\
                        and __is_fall(rxn):
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

        rtol = 1e-5
        atol = 1e-8
        if test_variable and rxn_type == falloff_form.sri:
            # this tends to be a bit more finicky
            rtol = 5e-4
            atol = 1e-5

        kc = [kernel_call('dci_dT',
                          comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis,
                          other_compare=self.nan_compare,
                          rtol=rtol, atol=atol,
                          **args)]

        extra_args = {}
        if test_variable:
            extra_args['conp'] = conp

        return self._generic_jac_tester(tester, kc, **extra_args)

    @attr('long')
    @with_check_inds(check_inds={
        # for third body, only need to worry about these
        1: lambda self: self.__get_check(
            lambda rxn: True,
            lambda rxn: isinstance(rxn, ct.ThreeBodyReaction)),
        2: np.array([0])})
    def test_dci_thd_dT(self):
        self.__run_dci_thd_dvar()

    @attr('long')
    @with_check_inds(check_inds={
        0: lambda self: np.where(np.all(
                    self.store.ref_Pr[:, self.store.lind_to_pr_map] != 0.0,
                    axis=1))[0],
        # need to look at all 3body/fall, and exclude wrong type
        1: lambda self: self.__get_check(
            lambda rxn: (isinstance(rxn, ct.FalloffReaction) and
                         rxn.falloff.type == 'Simple'),
            lambda rxn: (isinstance(rxn, ct.ThreeBodyReaction) or
                         isinstance(rxn, ct.FalloffReaction))),
        2: np.array([0])})
    def test_dci_lind_dT(self):
        self.__run_dci_thd_dvar(falloff_form.lind)

    @attr('long')
    @with_check_inds(check_inds={
        0: lambda self: np.where(np.all(
                    self.store.ref_Pr[:, self.store.troe_to_pr_map] != 0.0,
                    axis=1))[0],
        # need to look at all 3body/fall, and exclude wrong type
        1: lambda self: self.__get_check(
            lambda rxn: (isinstance(rxn, ct.FalloffReaction) and
                         rxn.falloff.type == 'Troe'),
            lambda rxn: (isinstance(rxn, ct.ThreeBodyReaction) or
                         isinstance(rxn, ct.FalloffReaction))),
        2: np.array([0])})
    def test_dci_troe_dT(self):
        self.__run_dci_thd_dvar(falloff_form.troe)

    @attr('long')
    @with_check_inds(check_inds={
        0: lambda self: np.where(np.all(
                    self.store.ref_Pr[:, self.store.sri_to_pr_map] != 0.0,
                    axis=1))[0],
        # need to look at all 3body/fall, and exclude wrong type
        1: lambda self: self.__get_check(
            lambda rxn: (isinstance(rxn, ct.FalloffReaction) and
                         rxn.falloff.type == 'SRI'),
            lambda rxn: (isinstance(rxn, ct.ThreeBodyReaction) or
                         isinstance(rxn, ct.FalloffReaction))),
        2: np.array([0])})
    def test_dci_sri_dT(self):
        self.__run_dci_thd_dvar(falloff_form.sri)

    @attr('long')
    @with_check_inds(check_inds={
        # for third body, only need to worry about these
        1: lambda self: self.__get_check(
            lambda rxn: True,
            lambda rxn: isinstance(rxn, ct.ThreeBodyReaction)),
        2: np.array([1])})
    def test_dci_thd_dE(self):
        self.__run_dci_thd_dvar(reaction_type.thd, test_variable=True, conp=True)
        self.__run_dci_thd_dvar(reaction_type.thd, test_variable=True, conp=False)

    @attr('long')
    @with_check_inds(check_inds={
        0: lambda self: np.where(np.all(
                    self.store.ref_Pr[:, self.store.lind_to_pr_map] != 0.0,
                    axis=1))[0],
        # need to look at all 3body/fall, and exclude wrong type
        1: lambda self: self.__get_check(
            lambda rxn: (isinstance(rxn, ct.FalloffReaction) and
                         rxn.falloff.type == 'Simple'),
            lambda rxn: (isinstance(rxn, ct.ThreeBodyReaction) or
                         isinstance(rxn, ct.FalloffReaction))),
        2: np.array([1])})
    def test_dci_lind_dE(self):
        self.__run_dci_thd_dvar(falloff_form.lind, test_variable=True, conp=True)
        self.__run_dci_thd_dvar(falloff_form.lind, test_variable=True, conp=False)

    @attr('long')
    @with_check_inds(check_inds={
        0: lambda self: np.where(np.all(
                    self.store.ref_Pr[:, self.store.troe_to_pr_map] != 0.0,
                    axis=1))[0],
        # need to look at all 3body/fall, and exclude wrong type
        1: lambda self: self.__get_check(
            lambda rxn: (isinstance(rxn, ct.FalloffReaction) and
                         rxn.falloff.type == 'Troe'),
            lambda rxn: (isinstance(rxn, ct.ThreeBodyReaction) or
                         isinstance(rxn, ct.FalloffReaction))),
        2: np.array([1])})
    def test_dci_troe_dE(self):
        self.__run_dci_thd_dvar(falloff_form.troe, test_variable=True, conp=True)
        self.__run_dci_thd_dvar(falloff_form.troe, test_variable=True, conp=False)

    @attr('long')
    @with_check_inds(check_inds={
        0: lambda self: np.where(np.all(
                    self.store.ref_Pr[:, self.store.sri_to_pr_map] != 0.0,
                    axis=1))[0],
        # need to look at all 3body/fall, and exclude wrong type
        1: lambda self: self.__get_check(
            lambda rxn: (isinstance(rxn, ct.FalloffReaction) and
                         rxn.falloff.type == 'SRI'),
            lambda rxn: (isinstance(rxn, ct.ThreeBodyReaction) or
                         isinstance(rxn, ct.FalloffReaction))),
        2: np.array([1])})
    def test_dci_sri_dE(self):
        self.__run_dci_thd_dvar(falloff_form.sri, test_variable=True, conp=True)
        self.__run_dci_thd_dvar(falloff_form.sri, test_variable=True, conp=False)

    @with_check_inds(check_inds={
            1: np.array([1]),
            2: np.array([0])},
        custom_checks={
            # exclude purposefully included nan's
            0: lambda self, conp: np.setdiff1d(
                np.arange(self.store.test_size),
                np.unique(np.where(np.isnan(self.__get_full_jac(conp)))[0]),
                assume_unique=True)
        })
    def __run_test_dedot_dt(self, conp):
        # conp
        fd_jac = self.__get_full_jac(conp)

        namestore, rate_info, opts = self.__get_non_ad_params(conp)
        phi = self.store.phi_cp if conp else self.store.phi_cv
        dphi = self.store.dphi_cp if conp else self.store.dphi_cv

        jac = fd_jac.copy()
        self._set_at(jac, 0)

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

        # and get mask
        comp = self._get_compare(fd_jac)
        kc = [kernel_call('dEdotdT', comp.ref_answer, compare_mask=[comp],
                          compare_axis=comp.compare_axis,
                          other_compare=self.our_nan_compare,
                          **args)]

        return self._generic_jac_tester(dEdotdT, kc, conp=conp)

    @attr('long')
    def test_dEdot_dT(self):
        self.__run_test_dedot_dt(True)
        self.__run_test_dedot_dt(False)

    @with_check_inds(check_inds={
            1: np.array([0]),
            2: np.array([1])
        },
        custom_checks={
            # exclude purposefully included nan's
            0: lambda self, conp: np.setdiff1d(
                np.arange(self.store.test_size), np.unique(
                    np.where(np.isnan(self.__get_full_jac(conp)))[0]),
                assume_unique=True)
        })
    def __run_test_dtdot_de(self, conp):
        # get the full jacobian
        fd_jac = self.__get_full_jac(conp)

        namestore, rate_info, opts = self.__get_non_ad_params(conp)
        phi = self.store.phi_cp if conp else self.store.phi_cv
        dphi = self.store.dphi_cp if conp else self.store.dphi_cv
        spec_heat = self.store.spec_cp if conp else self.store.spec_cv
        spec_heat_tot = np.sum(self.store.concs * spec_heat, axis=1)
        spec_energy = self.store.spec_h if conp else self.store.spec_u

        jac = fd_jac.copy()
        self._set_at(jac, 0)

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

        # and get mask
        comp = self._get_compare(fd_jac)
        kc = [kernel_call('dTdotdE', comp.ref_answer,
                          compare_mask=[comp], compare_axis=comp.compare_axis,
                          other_compare=self.our_nan_compare,
                          **args)]

        return self._generic_jac_tester(dTdotdE, kc, conp=conp)

    @attr('long')
    def test_dTdot_dE(self):
        self.__run_test_dtdot_de(True)
        self.__run_test_dtdot_de(False)

    @with_check_inds(check_inds={
            1: np.array([1]),
            2: np.array([1])
        },
        custom_checks={
            # exclude purposefully included nan's
            0: lambda self, conp: np.setdiff1d(
                np.arange(self.store.test_size), np.unique(np.where(np.isnan(
                    self.__get_full_jac(conp)))[0]), assume_unique=True)
        })
    def __run_test_dedot_de(self, conp):
        # get the full jacobian
        fd_jac = self.__get_full_jac(conp)

        namestore, rate_info, opts = self.__get_non_ad_params(conp)
        phi = self.store.phi_cp if conp else self.store.phi_cv
        dphi = self.store.dphi_cp if conp else self.store.dphi_cv
        jac = fd_jac.copy()
        # reset populated values
        self._set_at(jac, 0)

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

        # and get mask
        comp = self._get_compare(fd_jac)
        kc = [kernel_call('dEdotdE', comp.ref_answer,
                          compare_mask=[comp], compare_axis=comp.compare_axis,
                          other_compare=self.our_nan_compare,
                          **args)]

        return self._generic_jac_tester(dEdotdE, kc, conp=conp)

    @attr('long')
    def test_dEdot_dE(self):
        self.__run_test_dedot_de(True)
        self.__run_test_dedot_de(False)

    @attr('long')
    @skipif(csr_matrix is None, 'Cannot test sparse Jacobian without scipy')
    def test_index_determination(self):
        # find FD jacobian
        jac = self.__get_full_jac(True)
        # find our non-zero indicies
        inds = determine_jac_inds(self.store.reacs, self.store.specs,
                                  RateSpecialization.fixed)
        non_zero_specs = inds['net_per_spec']['map']
        if self.store.gas.n_species - 1 in non_zero_specs:
            # remove last species
            non_zero_specs = non_zero_specs[:-1]
        ret = inds['jac_inds']
        non_zero_inds = ret['flat_C']
        non_zero_inds = non_zero_inds.T

        # set all T and V derivatives to nonzero by assumption
        jac[:, non_zero_specs + 2, 0:2] = 1
        # convert nan's or inf's to some non-zero number
        jac[np.where(~np.isfinite(jac))] = 1
        # create a jacobian of the max of all the FD's to avoid zeros due to
        # zero rates
        jac = np.max(np.abs(jac), axis=0)

        # and get non-zero indicies
        jac_inds = np.column_stack(np.where(jac)).T
        if jac_inds.shape != non_zero_specs.shape:
            logger = logging.getLogger(__name__)
            logger.warn(
                "Autodifferentiated Jacobian sparsity pattern "
                "does not match pyJac's.  There are legitimate reasons"
                "why this might be the case -- e.g., matching "
                "arrhenius parameters for two reactions containing "
                "the same species, with one reaction involving the "
                "(selected) last species in the mechanism -- if you "
                "are not sure why this error is appearing, feel free to "
                "contact the developers to ensure this is not a bug.")
            # get missing values
            missing = np.where(~np.in1d(
                # as those that are in our detected non-zero indicies
                np.arange(non_zero_inds.shape[1]),
                # and not in the auto-differentiated non-zero indicies
                inNd(jac_inds.T, non_zero_inds.T)))[0]
            # check that all missing values are zero in the jacobian
            assert np.all(jac[non_zero_inds[0, missing],
                              non_zero_inds[1, missing]] == 0)
            # set them to non-zero for testing below
            jac[non_zero_inds[0, missing], non_zero_inds[1, missing]] = 1
        else:
            assert np.allclose(jac_inds, non_zero_inds)

        # create a CRS
        crs = csr_matrix(jac)
        assert np.array_equal(ret['crs']['row_ptr'], crs.indptr) and \
            np.array_equal(ret['crs']['col_ind'], crs.indices)

        # and repeat with CCS
        ccs = csc_matrix(jac)
        assert np.array_equal(ret['ccs']['col_ptr'], ccs.indptr) and \
            np.array_equal(ret['ccs']['row_ind'], ccs.indices)

    @attr('long')
    def test_reset_arrays(self):
        # find our non-zero indicies
        ret = determine_jac_inds(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)['jac_inds']

        non_zero_inds = ret['flat_C']
        jac_shape = (self.store.test_size,) + (len(self.store.specs) + 1,) * 2

        def __set(order):
            x = np.zeros(jac_shape, order=order)
            x[:, non_zero_inds[:, 0], non_zero_inds[:, 1]] = 1
            return x

        args = {'jac': __set}

        comp = get_comparable(ref_answer=[np.zeros(jac_shape)],
                              compare_mask=[
                                (slice(None), non_zero_inds[:, 0],
                                    non_zero_inds[:, 1])],
                              compare_axis=np.arange(len(jac_shape)),
                              tiling=False)

        # and get mask
        kc = kernel_call('reset_arrays', comp.ref_answer, compare_mask=[comp],
                         compare_axis=comp.compare_axis, **args)

        return self._generic_jac_tester(reset_arrays, kc)

    def test_sparse_indexing(self):
        from pyjac.core import instruction_creator as ic
        # a simple test to ensure our sparse indexing is working correctly

        @ic.with_conditional_jacobian
        def __kernel_creator(loopy_opts, namestore, test_size=None, jac_create=None):
            from pyjac.core.enum_types import JacobianFormat
            if loopy_opts.jac_format != JacobianFormat.sparse:
                raise SkipTest('Not relevant')
            from pyjac.core import array_creator as arc
            from pyjac.kernel_utils.kernel_gen import knl_info
            from string import Template
            from pyjac.loopy_utils.preambles_and_manglers import jac_indirect_lookup
            # get ptrs and inds
            if loopy_opts.order == 'C':
                inds = namestore.jac_col_inds
                indptr = namestore.jac_row_inds
            else:
                inds = namestore.jac_row_inds
                indptr = namestore.jac_col_inds
            # create maps
            mapstore = arc.MapStore(loopy_opts, namestore.net_nonzero_phi,
                                    self.store.test_size)
            mapstore.finalize()
            var_name = mapstore.tree.iname
            global_ind = arc.global_ind

            kernel_data = []

            # convert to loopy & str
            inds_arr, inds_str = mapstore.apply_maps(inds, var_name)
            indptr_arr, indptr_str = mapstore.apply_maps(indptr, var_name)
            _, indptr_next_str = mapstore.apply_maps(indptr, var_name,
                                                     affine=1)
            kernel_data.extend([inds_arr, indptr_arr])
            # and create set instructions
            jac_lp, set1_insn = jac_create(
                mapstore, namestore.jac, global_ind, var_name, 0,
                insn='${jac_str} = jac_index {id=set1, nosync=set*, dep=${deps}}')
            _, set2_insn = jac_create(
                mapstore, namestore.jac, global_ind, var_name, 1,
                insn='${jac_str} = jac_index {id=set2, nosync=set*, dep=${deps}}')
            _, set3_insn = jac_create(
                mapstore, namestore.jac, global_ind, var_name, 'spec + 2',
                insn='${jac_str} = jac_index {id=set3, nosync=set*, dep=${deps}}')
            _, set4_insn = jac_create(
                mapstore, namestore.jac, global_ind, var_name, 'spec + 2',
                insn='${jac_str} = jac_index {id=set4, nosync=set*, dep=${deps}}')
            _, set5_insn = jac_create(
                mapstore, namestore.jac, global_ind, var_name, 'spec + 2',
                insn='${jac_str} = jac_index {id=set5, nosync=set*, dep=${deps}}')
            _, set6_insn = jac_create(
                mapstore, namestore.jac, global_ind, var_name, 'spec + 2',
                insn='${jac_str} = jac_index {id=set6, nosync=set*, dep=${deps}}')

            target = get_target(loopy_opts.lang, device=loopy_opts.device)
            lookup = jac_indirect_lookup(
                namestore.jac_col_inds if loopy_opts.order == 'C'
                else namestore.jac_row_inds, target)

            kernel_data.append(jac_lp)
            # create get species -> rxn offsets & rxns
            offsets, s_t_r_offset_str = mapstore.apply_maps(
                namestore.spec_to_rxn_offsets, 'ind')
            _, s_t_r_next_offset_str = mapstore.apply_maps(
                namestore.spec_to_rxn_offsets, 'ind', affine=1)
            rxn, rxn_str = mapstore.apply_maps(
                namestore.spec_to_rxn, 'i_rxn')
            # now get rxn -> species and offsets
            rxn_offsets, r_t_s_offset_str = mapstore.apply_maps(
                namestore.rxn_to_spec_offsets, rxn_str)
            _, r_t_s_next_offset_str = mapstore.apply_maps(
                namestore.rxn_to_spec_offsets, rxn_str, affine=1)
            spec, spec_str = mapstore.apply_maps(
                namestore.rxn_to_spec, 'is1')
            kernel_data.extend([offsets, rxn_offsets, spec, rxn])

            # set ns
            ns = namestore.num_specs[-1]

            # finally we need the third body indicies
            thd_mask, thd_mask_str = mapstore.apply_maps(
                namestore.thd_mask, rxn_str)
            thd_offset, toffset_str = mapstore.apply_maps(
                namestore.thd_offset, thd_mask_str)
            _, toffset_next_str = mapstore.apply_maps(
                namestore.thd_offset, thd_mask_str, affine=1)
            thd_spec, thd_spec_str = mapstore.apply_maps(
                namestore.thd_spec, 'i_thd')
            kernel_data.extend([thd_mask, thd_offset, thd_spec])

            # add all species for temperature / variable loop

            numspec, specloop2_str = mapstore.apply_maps(
                namestore.num_specs_no_ns, 'is2')
            numspec, specloop3_str = mapstore.apply_maps(
                namestore.num_specs_no_ns, 'is3')
            kernel_data.append(numspec)

            # create a custom has_ns mask over all reaction types
            has_ns_mask = np.full(namestore.num_reacs.size, 0, dtype=arc.kint_type)
            if namestore.rxn_has_ns is not None:
                has_ns_mask[namestore.rxn_has_ns.initializer] = 1
            if namestore.thd_only_ns_inds is not None:
                has_ns_mask[namestore.thd_map[
                    namestore.thd_only_ns_inds.initializer]] = 1
            if namestore.sri_has_ns is not None:
                has_ns_mask[namestore.thd_map[namestore.fall_to_thd_map[
                    namestore.sri_has_ns.initializer]]] = 1
            if namestore.troe_has_ns is not None:
                has_ns_mask[namestore.thd_map[namestore.fall_to_thd_map[
                    namestore.troe_has_ns.initializer]]] = 1
            if namestore.lind_has_ns is not None:
                has_ns_mask[namestore.thd_map[namestore.fall_to_thd_map[
                    namestore.lind_has_ns.initializer]]] = 1
            has_ns_mask = arc.creator('has_ns', initializer=has_ns_mask,
                                      dtype=arc.kint_type, shape=has_ns_mask.shape,
                                      order=loopy_opts.order)
            has_ns, has_ns_str = mapstore.apply_maps(
                has_ns_mask, rxn_str)
            kernel_data.append(has_ns)

            extra_inames = [('i_rxn', 'roffset <= i_rxn < roffset_next'),
                            ('is1', 'soffset <= is1 < soffset_next'),
                            ('is2,is3', '0 <= is2,is3 < {}'.format(
                                namestore.num_specs_no_ns.size)),
                            ('i_thd', 'toffset <= i_thd <= toffset_next')]
            # subs
            subs = locals().copy()
            subs.update({
                        'lookup': lookup.name,
                        'indptr': indptr_str})

            # make kernel
            instructions = Template(Template("""
                # for each phi entry in the jacobian
                # temperature
                ${set1_insn}
                # extra variable
                ${set2_insn}
                if ${var_name} >= 2
                    # and then go through the reactions for this species
                    <> ind = i - 2
                    <> roffset = ${s_t_r_offset_str}
                    <> roffset_next = ${s_t_r_next_offset_str}
                    for i_rxn
                        # and now loop over species in reaction
                        <> soffset = ${r_t_s_offset_str}
                        <> soffset_next = ${r_t_s_next_offset_str}
                        for is1
                            <> spec = ${spec_str}
                            if spec != ${ns}
                                # do lookup
                                ${set3_insn}
                            end
                        end
                        # and species in third body (if present)
                        if ${thd_mask_str} != -1
                            <>toffset = ${toffset_str}
                            <>toffset_next = ${toffset_next_str}
                            for i_thd
                                spec = ${thd_spec_str}
                                if spec != ${ns}
                                    ${set4_insn}
                                end
                            end
                        end
                        if ${has_ns_str}
                            # turn on every species
                            for is2
                                spec = ${specloop2_str}
                                # do lookup
                                ${set5_insn}
                            end
                        end
                    end
                else
                    for is3
                        spec = ${specloop3_str}
                        # do lookup
                        ${set6_insn}
                    end
                end
            """).safe_substitute(**subs)).safe_substitute(**subs)

            return knl_info(instructions=instructions,
                            mapstore=mapstore,
                            name='index_test',
                            kernel_data=kernel_data,
                            extra_inames=extra_inames,
                            manglers=[lookup],
                            seq_dependencies=True,
                            silenced_warnings=['write_race(set1)',
                                               'write_race(set2)',
                                               'write_race(set3)',
                                               'write_race(set4)',
                                               'write_race(set5)',
                                               'write_race(set6)'])

        # create reference answer
        # get number of non-zero jacobian inds
        jac_inds = determine_jac_inds(self.store.reacs, self.store.specs,
                                      RateSpecialization.fixed)['jac_inds']
        jac_size = len(self.store.specs) + 1
        num_nonzero_jac = jac_inds['flat_C'].shape[0]
        # get reference answer and comparable
        ref_ans = np.tile(np.arange(num_nonzero_jac, dtype=arc.kint_type),
                          (self.store.test_size, 1))
        compare_mask = (slice(None), np.arange(num_nonzero_jac))

        def __get_compare(kc, outv, index, is_answer=False):
            from pyjac.tests.test_utils import indexer
            _get_index = indexer(kc.current_split, ref_ans.shape)
            # first check we have a reasonable mask
            assert ref_ans.ndim == len(compare_mask), (
                "Can't use dissimilar compare masks / axes")
            # dummy comparison axis
            comp_axis = np.arange(ref_ans.ndim)
            # convert inds
            masking = tuple(_get_index(compare_mask, comp_axis))
            return outv[masking]

        # and finally create args
        args = {
            'jac': lambda x: np.zeros((self.store.test_size, jac_size, jac_size),
                                      order=x)
        }

        kc = kernel_call('index_test', ref_ans, compare_mask=[__get_compare],
                         compare_axis=np.arange(ref_ans.ndim), tiling=False, **args)

        return self._generic_jac_tester(__kernel_creator, kc, sparse_only=True)

    @parameterized.expand([(x,) for x in get_test_langs()])
    @attr('fullkernel')
    def test_jacobian(self, lang):
        _full_kernel_test(self, lang, get_jacobian_kernel, 'jac',
                          lambda conp: self.__get_full_jac(conp),
                          ktype=KernelType.jacobian, call_name='jacobian')

    @parameterized.expand([(x,) for x in get_test_langs()])
    @attr('fullkernel')
    @xfail(msg='Finite Difference Jacobian currently broken')
    def test_fd_jacobian(self, lang):
        def __looser_tol_finder(arr, order, have_split, conp):
            last_spec_name = self.store.gas.species_names[-1]
            # look for derivatives resulting from the last species' prescense in the
            # reaction

            ns_derivs = set()

            def __add(rxn, wrt=None):
                # find net nu
                from collections import defaultdict
                nu = defaultdict(lambda: 0)
                for spec in rxn.products:
                    nu[spec] += rxn.products[spec]
                for spec in rxn.reactants:
                    nu[spec] -= rxn.reactants[spec]

                nonzero = [self.store.gas.species_index(spec) for spec in nu
                           if nu[spec] != 0 and spec != last_spec_name]
                # get default derivative (all species)
                if wrt is None:
                    wrt = range(self.store.gas.n_species - 1)
                # add any with nonzero and not last species
                ns_derivs.update([(x, y) for x in nonzero for y in wrt])

            for rxn in self.store.gas.reactions():
                # first, check for last species in reaction
                in_rxn = set(k for k in list(rxn.products.keys())
                             + list(rxn.reactants.keys()))
                if last_spec_name in in_rxn:
                    __add(rxn)
                    continue

                # next, check for third body / falloff
                try:
                    # if last species is third body
                    if last_spec_name in rxn.efficiencies:
                        # add full row
                        __add(rxn)
                    else:
                        # third body efficiencies in general can give poor
                        # FD approximations
                        __add(rxn, tuple(self.store.gas.species_index(x) for x in
                                         rxn.efficiencies))
                except AttributeError:
                    # no efficiencies
                    pass

            # finally, we need to turn these into 1d indicies
            row, col = zip(*sorted(ns_derivs))
            row_size = self.store.gas.n_species + 1
            # turn into numpy arrays
            row = np.array(row) + 2
            col = np.array(col) + 2
            # temperature and extra variable deriviatives w.r.t species are
            # inherently noisier

            first_rows = [0] * (row_size - 2) + [1] * (row_size - 2)
            if conp:
                # as are species / temperature rates w.r.t. V
                first_rows += list(range(row_size))
            else:
                # temperature rates can be a bit wonky here
                first_rows += list(range(row_size))
            row = np.insert(row, 0, first_rows)
            first_cols = list(range(2, row_size)) + list(range(2, row_size))
            if conp:
                # species / temperature rates w.r.t. V
                first_cols += [1] * row_size
            else:
                # temperature rates can be a bit wonky here
                first_cols += [0] * row_size
            col = np.insert(col, 0, first_cols)
            # and the extra variable derivative

            from .test_utils import parse_split_index
            # also have to override the size & stride arrays as these indicies
            # can be indexed together already -- this keeps us from blowing chunks
            # performance wise in the __get_looser_tols's multi ravel loop
            ravel_ind = parse_split_index(arr, (row, col),
                                          order, ref_ndim=3, axis=(1, 2),
                                          size_arr=[row.size],
                                          stride_arr=[1] * arr.ndim)
            copy_inds = np.array([1, 2])
            if have_split and order == 'F':
                # the last dimension has been split
                copy_inds = np.array([0, 2, 3])

            # return ravel inds, copy axes
            return np.array(ravel_ind), copy_inds

        # we're really not testing here for correctness, rather that
        # we meet some _reasonable_ (but large) tolerances
        _full_kernel_test(self, lang, finite_difference_jacobian, 'jac',
                          lambda conp: self.__get_full_jac(conp),
                          ktype=KernelType.jacobian, call_name='jacobian',
                          do_finite_difference=True,
                          atol=100, rtol=100, loose_rtol=1e7, loose_atol=100,
                          looser_tol_finder=__looser_tol_finder,
                          call_kwds={'mode': FiniteDifferenceMode.central,
                                     'order': 8})
