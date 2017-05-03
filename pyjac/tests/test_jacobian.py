from . import TestClass
from ..core.rate_subs import (get_sri_kernel, assign_rates)
from ..loopy_utils.loopy_utils import (auto_run, loopy_options,
                                       RateSpecialization,
                                       get_device_list,
                                       kernel_call,
                                       set_adept_editor
                                       )
from ..core import array_creator as arc
from ..kernel_utils import kernel_generator as k_gen

import numpy as np
from nose.plugins.attrib import attr
from optionloop import OptionLoop

from collections import OrderedDict


class editor(object):

    def __init__(self, independent, dependent,
                 problem_size):

        self.independent = independent
        indep_size = next(x for x in independent.shape if x != problem_size)
        self.dependent = dependent
        dep_size = next(x for x in dependent.shape if x != problem_size)
        self.problem_size = problem_size

        # create the jacobian
        self.output = arc.creator('jac', np.float64,
                                  (dep_size, indep_size, problem_size),
                                  'F')

    def __call__(self, knl):
        set_adept_editor(knl, self.problem_size,
                         self.independent, self.dependent, self.output)


class SubTest(TestClass):

    @attr('long')
    def test_sri_derivatives(self):
        ref_phi = self.store.phi
        ref_Pr = self.store.ref_Pr.copy()
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
                         compare_mask=[sri_mask], **args)

        reacs = self.store.reacs
        specs = self.store.specs
        rate_info = assign_rates(reacs, specs, RateSpecialization.fixed)
        # create namestore
        namestore = arc.NameStore(object(order='F', kernel_type='map'),
                                  rate_info, self.store.test_size)
        myedit = editor(namestore.phi, namestore.Fi,
                        self.store.test_size)
        self.__generic_rate_tester(get_sri_kernel, kc, editor=myedit)

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

    def __generic_rate_tester(self, func, kernel_calls, do_ratespec=False,
                              do_ropsplit=None, do_spec_per_reac=False,
                              editor=None,
                              **kw_args):
        """
        A generic testing method that can be used for rate constants, third bodies, ...

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
        do_spec_per_reac : bool
            If true, test species rates summing over reactions as well
        """

        eqs, oploop = self.__get_eqs_and_oploop(
            do_ratespec, do_ropsplit, do_spec_per_reac)

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
                test_size=self.store.test_size
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
