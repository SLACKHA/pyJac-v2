# system
from collections import OrderedDict

# local imports
from ..core.rate_subs import polyfit_kernel_gen, assign_rates
from ..loopy_utils.loopy_utils import (auto_run, loopy_options,
                                       get_device_list, kernel_call,
                                       RateSpecialization)
from ..kernel_utils import kernel_gen as k_gen
from . import TestClass
from ..core.array_creator import NameStore

# modules
from optionloop import OptionLoop
from nose.plugins.attrib import attr
import numpy as np


class SubTest(TestClass):
    def __subtest(self, ref_ans, nicename, eqs):
        oploop = OptionLoop(OrderedDict([('lang', ['opencl']),
                                         ('width', [4, None]),
                                         ('depth', [4, None]),
                                         ('ilp', [True, False]),
                                         ('unr', [None, 4]),
                                         ('order', ['C', 'F']),
                                         ('device', get_device_list())]))

        poly_dim = self.store.specs[0].hi.size
        test_size = self.store.test_size
        for i, state in enumerate(oploop):
            try:
                opt = loopy_options(
                    **{x: state[x] for x in state if x != 'device'})
                rate_info = assign_rates(self.store.reacs, self.store.specs,
                                         RateSpecialization.fixed)
                namestore = NameStore(opt, rate_info, True, test_size)
                knl = polyfit_kernel_gen(nicename, eqs, opt, namestore,
                                         poly_dim,
                                         test_size=test_size)

                # create the kernel call
                kc = kernel_call('eval_' + nicename,
                         [ref_ans],
                         phi=np.array(self.store.phi,
                                      order=state['order'],
                                      copy=True))

                # create a dummy kernel generator
                knl = k_gen.make_kernel_generator(
                    name='chem_utils',
                    loopy_opts=opt,
                    kernels=[knl],
                    test_size=test_size
                )
                knl._make_kernels()

                # now run
                kc.set_state(state['order'])
                assert auto_run(knl.kernels, kc, device=state['device'])
            except Exception as e:
                if not(state['width'] and state['depth']):
                    raise e

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
