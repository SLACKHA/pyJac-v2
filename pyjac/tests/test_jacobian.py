from . import TestClass
from ..core.rate_subs import (get_sri_kernel, assign_rates)
from ..loopy_utils.loopy_utils import (auto_run, loopy_options,
                                       RateSpecialization,
                                       get_device_list,
                                       kernel_call,
                                       set_adept_editor,
                                       populate
                                       )
from ..core import array_creator as arc
from ..kernel_utils import kernel_gen as k_gen

import numpy as np
from nose.plugins.attrib import attr
from optionloop import OptionLoop

from collections import OrderedDict


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
    def __get_eqs_and_oploop(self, do_ratespec=False, do_ropsplit=None,
                             do_spec_per_reac=False,
                             use_platform_instead=False,
                             do_conp=False,
                             do_vector=True,
                             langs=['opencl']):
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        vectypes = [4, None] if do_vector else [None]
        oploop = [('lang', langs),
                  ('width', vectypes[:]),
                  ('depth', vectypes[:]),
                  ('order', ['C', 'F']),
                  ('ilp', [False]),
                  ('unr', [None, 4]),
                  ('auto_diff', [True])
                  ]
        if do_ratespec:
            oploop += [
                ('rate_spec', [x for x in RateSpecialization]),
                ('rate_spec_kernels', [True, False])]
        if do_ropsplit:
            oploop += [
                ('rop_net_kernels', [True])]
        if do_spec_per_reac:
            oploop += [
                ('spec_rates_sum_over_reac', [True, False])]
        if use_platform_instead:
            oploop += [('platform', ['CPU', 'GPU'])]
        else:
            oploop += [('device', get_device_list())]
        if do_conp:
            oploop += [('conp', [True, False])]
        oploop += [('knl_type', ['map'])]
        oploop = OptionLoop(OrderedDict(oploop))

        return eqs, oploop

    def __generic_jac_tester(self, func, kernel_calls, do_ratespec=False,
                              do_ropsplit=None, do_spec_per_reac=False,
                              editor=None,
                              **kw_args):
        """
        A generic testing method that can be used for rate constants,
        third bodies, ...

        Parameters
        ----------
        func : function
            The function to test
        kernel_calls : :class:`kernel_call` or list thereof
            Contains the masks and reference answers for kernel testing
        do_ratespec : bool
            If true, test rate specializations and kernel splitting for simple
            rates
        do_ropsplit : bool
            If true, test kernel splitting for rop_net
        do_spec_per_reac : bool
            If true, test species rates summing over reactions as well
        """

        eqs, oploop = self.__get_eqs_and_oploop(
            do_ratespec, do_ropsplit, do_spec_per_reac,
            do_vector=False)

        reacs = self.store.reacs
        specs = self.store.specs

        for i, state in enumerate(oploop):
            if state['width'] is not None and state['depth'] is not None:
                continue
            opt = loopy_options(**{x: state[x] for x in
                                   state if x != 'device'})
            # find rate info
            rate_info = assign_rates(reacs, specs, opt.rate_spec)
            # create namestore
            namestore = arc.NameStore(opt, rate_info, self.store.test_size)
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
                test_size=self.store.test_size,
                extra_kernel_data=[editor.output]
            )

            knl._make_kernels()

            # create a list of answers to check
            try:
                for kc in kernel_calls:
                    kc.set_state(state['order'])
            except:
                kernel_calls.set_state(state['order'])

            assert auto_run(knl.kernels, kernel_calls, device=state['device'],
                            editor=editor), \
                'Evaluate {} rates failed'.format(func.__name__)

    def __make_array(self, array):
        """
        Creates an array for comparison to an autorun kernel from the result
        of __get_jacobian
        """

        for i in range(array.shape[0]):
            # reshape inner array
            array[i, :, :] = np.reshape(array[i, :, :].flatten(order='K'),
                                        array.shape[1:],
                                        order='F')

    def __get_jacobian(self, func, kernel_call, editor, ad_opts,
                       **kw_args):
        eqs = {'conp': self.store.conp_eqs,
               'conv': self.store.conv_eqs}
        # find rate info
        rate_info = assign_rates(
            self.store.reacs,
            self.store.specs,
            ad_opts.rate_spec)
        # create namestore
        namestore = arc.NameStore(ad_opts, rate_info, self.store.test_size)
        # create the kernel info
        infos = func(eqs, ad_opts, namestore,
                     test_size=self.store.test_size, **kw_args)

        # create a dummy kernel generator
        knl = k_gen.make_kernel_generator(
                name='jacobian',
                loopy_opts=ad_opts,
                kernels=infos,
                test_size=self.store.test_size,
                extra_kernel_data=[editor.output]
            )
        knl._make_kernels()

        # and a generator for the single kernel
        single_name = arc.NameStore(ad_opts, rate_info, 1)
        single_info = func(eqs, ad_opts, single_name,
                           test_size=1, **kw_args)

        single_knl = k_gen.make_kernel_generator(
            name='spec_rates',
            loopy_opts=ad_opts,
            kernels=single_info,
            test_size=1,
            extra_kernel_data=[editor.output]
        )

        single_knl._make_kernels()

        # set in editor
        editor.set_single_kernel(single_knl.kernels[0])

        kernel_call.set_state(ad_opts.order)

        # add dummy 'j' arguement
        kernel_call.kernel_args['j'] = -1
        # and place output
        kernel_call.kernel_args[editor.output.name] = np.zeros(
            editor.output.shape,
            order=editor.output.order)

        import loopy as lp
        knl.kernels[0] = lp.set_options(knl.kernels[0], write_wrapper=True)

        # run kernel
        populate(
            knl.kernels, kernel_call, device=get_device_list()[0],
            editor=editor)

        return self.__make_array(kernel_call.kernel_args[editor.output.name])

    @attr('long')
    def test_sri_derivatives(self):
        ref_phi = self.store.phi
        ref_Pr = self.store.ref_Pr.copy()
        ref_ans = self.store.ref_Sri.copy().squeeze()
        args = {'Pr': lambda x: np.array(ref_Pr, order=x, copy=True),
                'phi': lambda x: np.array(ref_phi, order=x, copy=True),
                }

        # get number of sri reactions
        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = assign_rates(reacs, specs, RateSpecialization.fixed)

        num_sri = np.arange(rate_info['fall']['sri']['num'], dtype=np.int32)
        if not num_sri.size:
            return
        # create the kernel calls
        kc = kernel_call('fall_sri', ref_ans, out_mask=[1],
                         compare_mask=[num_sri], **args)

        ad_opts = loopy_options(order='C', knl_type='map', lang='c',
                                auto_diff=True)

        # create namestore
        namestore = arc.NameStore(ad_opts,
                                  rate_info, self.store.test_size)
        myedit = editor(namestore.T_arr, namestore.Fi,
                        self.store.test_size, order=ad_opts.order,
                        do_not_set=namestore.X_sri)

        answer = self.__get_jacobian(get_sri_kernel, kc, myedit, ad_opts)

        kc.ref_answer = answer

        self.__generic_jac_tester(get_sri_kernel, kc, editor=myedit)
