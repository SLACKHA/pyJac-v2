# system
from collections import OrderedDict

# local imports
from ..core.rate_subs import polyfit_kernel_gen, assign_rates
from ..loopy_utils.loopy_utils import (auto_run, loopy_options,
                                       get_device_list, kernel_call,
                                       RateSpecialization)
from ..kernel_utils import kernel_gen as k_gen
from . import TestClass, get_test_platforms
from ..core.array_creator import NameStore

# modules
from optionloop import OptionLoop
from nose.plugins.attrib import attr
import numpy as np


class SubTest(TestClass):
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

    def __subtest(self, ref_ans, nicename, eqs):
        platforms = get_test_platforms(do_vector=True, langs=['opencl'])
        start = [('order', ['C', 'F']),
                 ('auto_diff', False),
                 ('knl_type', 'map')]
        oploop = None
        for p in platforms:
            val = OptionLoop(OrderedDict(p + start))
            if oploop is None:
                oploop = val
            else:
                oploop = oploop + val

        test_size = self.store.test_size
        for i, state in enumerate(oploop):
            if state['width'] and state['depth']:
                continue

            opt = loopy_options(
                **{x: state[x] for x in state if x != 'device'})
            rate_info = assign_rates(self.store.reacs, self.store.specs,
                                     RateSpecialization.fixed)
            namestore = NameStore(opt, rate_info, True, test_size)
            knl = polyfit_kernel_gen(nicename, eqs, opt, namestore,
                                     test_size=test_size)

            args = {'phi': np.array(self.store.phi_cp, order=opt.order, copy=True),
                    nicename: np.zeros_like(ref_ans, order=opt.order)}
            # create the kernel call
            kc = kernel_call('eval_' + nicename,
                             [ref_ans],
                             **args)

            # create a dummy kernel generator
            knl = k_gen.make_kernel_generator(
                name='chem_utils',
                loopy_opts=opt,
                kernels=[knl],
                test_size=test_size
            )
            knl._make_kernels()

            # now run
            kc.set_state(knl.array_split, state['order'])
            assert auto_run(knl.kernels, kc, device=opt.device)

    @attr('long')
    def test_cp(self):
        self.__subtest(self.store.spec_cp,
                       'cp', self.store.conp_eqs)

    @attr('long')
    def test_cv(self):
        self.__subtest(self.store.spec_cv,
                       'cv', self.store.conp_eqs)

    @attr('long')
    def test_h(self):
        self.__subtest(self.store.spec_h,
                       'h', self.store.conp_eqs)

    @attr('long')
    def test_u(self):
        self.__subtest(self.store.spec_u,
                       'u', self.store.conv_eqs)

    @attr('long')
    def test_b(self):
        self.__subtest(self.store.spec_b,
                       'b', self.store.conp_eqs)
