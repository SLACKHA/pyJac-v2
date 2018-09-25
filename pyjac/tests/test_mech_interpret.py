import os
import subprocess
import tempfile
import difflib
import re

import cantera as ct
from cantera import __version__ as ct_version

from pyjac.utils import reassign_species_lists
from pyjac.core.create_jacobian import create_jacobian, find_last_species
from pyjac.core.enum_types import reaction_sorting
from pyjac.core.mech_interpret import read_mech, read_mech_ct
from pyjac.tests.test_utils import xfail, OptionLoopWrapper
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


def test_mechanism_sorting():
    # perform sort
    _, specs_ck, reacs_ck = read_mech(ck_file, None, reaction_sorting.simd)
    # ensure we have a good sort
    from pyjac.core.enum_types import (
        reaction_type, falloff_form, reversible_type, thd_body_type)

    enum_order = (reaction_type, falloff_form, thd_body_type, reversible_type)

    def check(start=0, end=len(reacs_ck), depth=0):
        if depth == len(enum_order):
            return
        for enum in enum_order[depth]:
            this_start = None
            this_end = None
            # pass #1, find start and end of this enum
            for i in range(start, end):
                if reacs_ck[i].match(enum) and this_start is None:
                    this_start = i
                    continue
                if not reacs_ck[i].match(enum) and (
                        this_end is None and this_start is not None):
                    # end of this section
                    this_end = i - 1
                    break
                elif this_start is not None:
                    # should all by of this type
                    assert reacs_ck[i].match(enum)

            if this_start is None:
                # no matches, nothing futher to check for this enum
                continue
            if this_end is None:
                # all matches
                this_end = end

            check(this_start, this_end, depth+1)

    check()


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

        for opts in OptionLoopWrapper.from_get_oploop(self):
            reacs, specs = create_jacobian(
                opts.lang,
                mech_name=mech,
                vector_size=opts.vector_width,
                width=opts.width,
                depth=opts.depth,
                last_spec=base_specs[-1].name,
                platform=opts.platform_name.lower(),
                data_order=opts.order,
                explicit_simd=opts.is_simd,
                test_mech_interpret_vs_backend=True)

            assert all(r1 == r2 for r1, r2 in zip(*(reacs, base_reacs)))
            assert all(s1 == s2 for s1, s2 in zip(*(specs, base_specs)))
