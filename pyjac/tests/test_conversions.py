# local imports
from pyjac.core.unit_conversions import (mass_to_mole_factions)
from pyjac.core.rate_subs import assign_rates
from pyjac.tests import TestClass
from pyjac.loopy_utils.loopy_utils import kernel_call
from pyjac.tests.test_utils import _generic_tester, get_comparable

# modules
import numpy as np
from nose.plugins.attrib import attr


class SubTest(TestClass):
    def __generic_conversion_tester(
            self, func, kernel_calls, do_conp=False, **kwargs):
        """
        A generic testing method that can be used for various conversion tests
        This is primarily a thin wrapper for :func:`_generic_tester`

        Parameters
        ----------
        func : function
            The function to test
        kernel_calls : :class:`kernel_call` or list thereof
            Contains the masks and reference answers for kernel testing
        do_conp:  bool [False]
            If true, test for both constant pressure _and_ constant volume
        """

        _generic_tester(self, func, kernel_calls, assign_rates,
                        do_conp=do_conp, **kwargs)

    @attr('long')
    def test_mass_to_mole_fractions(self):
        # create a hybrid input array
        Yphi = np.concatenate((self.store.T.reshape(-1, 1),
                               self.store.V.reshape(-1, 1),
                               self.store.Y[:, :-1]), axis=1)

        args = {'phi': lambda x: np.array(Yphi, order=x, copy=True),
                'mw_work': lambda x: np.zeros(self.store.test_size, order=x)}

        def __chainer(self, out_vals):
            self.kernel_args['mw_work'] = out_vals[-1][0]

        # first test w/o the splitting
        compare_mask = [get_comparable(
            (np.arange(self.store.test_size),), 1. / self.store.mw,
            compare_axis=(0,))]
        kc = kernel_call('molecular_weight_inverse',
                         [1. / self.store.mw],
                         strict_name_match=True,
                         compare_axis=(0,),
                         compare_mask=compare_mask,
                         **args)
        mole_fractions = (self.store.mw * (
            self.store.Y[:, :-1] / self.store.gas.molecular_weights[:-1]).T).T
        # create a reference answer of same shape just to simply comparison
        ref_answer = np.concatenate((self.store.T.reshape(-1, 1),
                                    self.store.V.reshape(-1, 1),
                                    mole_fractions), axis=1)
        compare_mask = [get_comparable(
            (np.arange(2, self.store.jac_dim),), ref_answer)]
        kc2 = kernel_call('mole_fraction',
                          [ref_answer],
                          strict_name_match=True,
                          compare_mask=compare_mask,
                          compare_axis=(1,),
                          chain=__chainer,
                          **args)
        self.__generic_conversion_tester(mass_to_mole_factions, [kc, kc2])
