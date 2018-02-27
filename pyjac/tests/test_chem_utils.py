# local imports
from ..core.rate_subs import polyfit_kernel_gen, assign_rates
from ..loopy_utils.loopy_utils import kernel_call
from . import TestClass
from .test_utils import _generic_tester

# modules
from nose.plugins.attrib import attr
import numpy as np


class SubTest(TestClass):
    def __subtest(self, ref_ans, nicename):
        def __wrapper(opt, namestore, test_size=None, **kwargs):
            return polyfit_kernel_gen(nicename, opt, namestore,
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
