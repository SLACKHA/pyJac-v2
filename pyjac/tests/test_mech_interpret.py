import os
import subprocess
import tempfile
import difflib
import re

import cantera as ct
from cantera import __version__ as ct_version

from pyjac.utils import reassign_species_lists
from pyjac.core.create_jacobian import create_jacobian, find_last_species
from pyjac.core.mech_interpret import read_mech, read_mech_ct
from pyjac.tests.test_utils import xfail
from pyjac.tests import script_dir, TestClass, get_mechanism_file


ck_file = os.path.join(script_dir, 'test.inp')
cti_file = os.path.join(script_dir, 'test.cti')


@xfail(ct_version <= '2.3.0', msg=(
       "fail until "
       "https://github.com/Cantera/cantera/pull/497 is published in 2.4.0"))
def test_ck_is_cti():
    """ tests that the test.inp mechanism corresponds to the test.cti mechanism"""
    with open(cti_file, 'r') as file:
        cti = file.readlines()

    # call ck2cti
    with tempfile.NamedTemporaryFile(mode='r', suffix='.cti') as file:
        # write
        subprocess.check_call(['ck2cti', '--input', ck_file, '--output', file.name])
        # read
        file.seek(0)
        ck = file.readlines()

    # process unicodes
    for i in range(len(cti)):
        cti[i] = re.sub("u'", "'", cti[i])

    # check diff
    assert not len([x for x in difflib.unified_diff(ck, cti)])


def test_mech_interpret_runs():
    """ test mechanism intpreter for both cantera and chemkin, and that results
        match"""
    _, specs_ck, reacs_ck = read_mech(ck_file, None)
    _, specs_cti, reacs_cti = read_mech_ct(cti_file)

    # reassign
    reassign_species_lists(reacs_ck, specs_ck)
    reassign_species_lists(reacs_cti, specs_cti)

    assert len(reacs_ck) == len(reacs_cti)
    for i in range(len(reacs_ck)):
        assert reacs_ck[i] == reacs_cti[i]
    assert len(specs_ck) == len(specs_cti)
    for i in range(len(specs_ck)):
        specs_ck[i] == specs_cti[i]


def test_equality_checking():
    """ test species and reaction equality checking"""
    _, specs_ck, reacs_ck = read_mech(ck_file, None)
    _, specs_cti, reacs_cti = read_mech_ct(cti_file)

    # reassign
    reassign_species_lists(reacs_ck, specs_ck)
    reassign_species_lists(reacs_cti, specs_cti)

    assert reacs_ck[0] == reacs_cti[0]
    for i in range(1, len(reacs_ck)):
        assert reacs_ck[0] != reacs_cti[i]
    assert specs_ck[0] == specs_cti[0]
    for i in range(1, len(specs_ck)):
        assert specs_ck[0] != specs_cti[i]


class Tester(TestClass):
    def test_heikki_issue(self):
        # tests issue raised by heikki via email re: incorrect re-ordering of species
        # post call to reassign_species_lists
        mech = get_mechanism_file()
        gas = ct.Solution(mech)
        # read our species for MW's
        _, specs, _ = read_mech_ct(gas=gas)

        # find the last species
        gas_map = find_last_species(specs, return_map=True)
        del specs
        # update the gas
        specs = gas.species()[:]
        gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                          species=[specs[x] for x in gas_map],
                          reactions=gas.reactions())
        del specs

        _, base_specs, base_reacs = read_mech_ct(gas=gas)
        # and reassign
        reassign_species_lists(base_reacs, base_specs)

        reacs, specs = create_jacobian(
            'c', mech_name=mech, last_spec=base_specs[-1].name,
            test_mech_interpret_vs_backend=True)

        assert all(r1 == r2 for r1, r2 in zip(*(reacs, base_reacs)))
        assert all(s1 == s2 for s1, s2 in zip(*(specs, base_specs)))
