# system
from collections import OrderedDict

# local imports
from ..core.rate_subs import polyfit_kernel_gen, assign_rates
from ..loopy_utils.loopy_utils import (auto_run, loopy_options,
                                       get_device_list, kernel_call,
                                       RateSpecialization)
from ..kernel_utils import kernel_gen as k_gen
from . import TestClass, get_test_platforms
from .test_utils import _generic_tester
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

    def __subtest(self, ref_ans, nicename):
        def __wrapper(eqs, opt, namestore, test_size=None, **kw_args):
            eq = eqs['conp'] if nicename in ['cp', 'h', 'b'] else eqs['conv']
            return polyfit_kernel_gen(nicename, eq, opt, namestore,
                                      test_size=test_size)

        # create args
        args = {'phi': lambda x: np.array(self.store.phi_cp, order=x, copy=True),
                nicename: lambda x: np.zeros_like(ref_ans, order=x)}
        # create the kernel call
        kc = kernel_call('eval_' + nicename,
                         [ref_ans],
                         **args)

        return _generic_tester(self, __wrapper, [kc], assign_rates)

    @attr('long')
    def test_cp(self):
        self.__subtest(self.store.spec_cp, 'cp')

    @attr('long')
    def test_cv(self):
        self.__subtest(self.store.spec_cv, 'cv')

    @attr('long')
    def test_h(self):
        self.__subtest(self.store.spec_h, 'h')

    @attr('long')
    def test_u(self):
        self.__subtest(self.store.spec_u, 'u')

    @attr('long')
    def test_b(self):
        self.__subtest(self.store.spec_b, 'b')
