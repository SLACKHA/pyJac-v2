from pyjac.core.enum_types import falloff_form, reaction_type, reversible_type, \
    thd_body_type
from pyjac.tests import TestClass


class SubTest(TestClass):

    def test_finalize(self):
        reacs = self.store.reacs
        for i, reac in enumerate(reacs):
            # for each reaction, test that we have the correct enums

            # test the reaction type
            if reac.pdep:
                # for falloff/chemically activated
                # also test the falloff form
                if reac.low:
                    assert reaction_type.fall in reac.type
                    assert all(x not in reac.type for x in reaction_type
                               if x != reaction_type.fall)
                else:
                    assert reaction_type.chem in reac.type
                    assert all(x not in reac.type for x in reaction_type
                               if x != reaction_type.chem)
                if reac.sri:
                    assert falloff_form.sri in reac.type
                elif reac.troe:
                    assert falloff_form.troe in reac.type
                else:
                    assert falloff_form.lind in reac.type
            elif reac.thd_body:
                assert reaction_type.thd in reac.type
                assert all(x not in reac.type for x in reaction_type
                           if x != reaction_type.thd)
            elif reac.plog:
                assert reaction_type.plog in reac.type
                assert all(x not in reac.type for x in reaction_type
                           if x != reaction_type.plog)
            elif reac.cheb:
                assert reaction_type.cheb in reac.type
                assert all(x not in reac.type for x in reaction_type
                           if x != reaction_type.cheb)
            else:
                assert reaction_type.elementary in reac.type
                assert all(x not in reac.type for x in reaction_type
                           if x != reaction_type.elementary)

            # test the reversible type
            if reac.rev:
                if reac.rev_par:
                    assert reversible_type.explicit in reac.type
                    assert all(x not in reac.type for x in reversible_type
                               if x != reversible_type.explicit)
                else:
                    assert reversible_type.non_explicit in reac.type
                    assert all(x not in reac.type for x in reversible_type
                               if x != reversible_type.non_explicit)
            else:
                assert reversible_type.non_reversible in reac.type
                assert all(x not in reac.type for x in reversible_type
                           if x != reversible_type.non_reversible)

            # finally test the third body types
            if reac.pdep or reac.thd_body:
                if reac.pdep_sp:
                    assert thd_body_type.species in reac.type
                    assert all(x not in reac.type for x in thd_body_type
                               if x != thd_body_type.species)
                elif not reac.thd_body_eff:
                    assert thd_body_type.unity in reac.type
                    assert all(x not in reac.type for x in thd_body_type
                               if x != thd_body_type.unity)
                else:
                    assert thd_body_type.mix in reac.type
                    assert all(x not in reac.type for x in thd_body_type
                               if x != thd_body_type.mix)

    def test_rxn_strings(self):
        for i, rxn in enumerate(self.store.reacs):
            rstr = str(rxn)
            rstr = rstr[rstr.index(': ') + 2:]
            assert str(rstr) == str(self.store.gas.reaction(i).equation)
