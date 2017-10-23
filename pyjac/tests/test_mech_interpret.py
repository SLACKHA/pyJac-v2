from . import script_dir
from ..core.mech_interpret import read_mech, read_mech_ct

import os
import subprocess
import tempfile
import difflib

ck_file = os.path.join(script_dir, 'test.inp')
cti_file = os.path.join(script_dir, 'test.cti')


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

    # check diff
    assert not len([x for x in difflib.unified_diff(ck, cti)])


def test_mech_interpret_runs():
    """ test mechanism intpreter for both cantera and chemkin, and that results
        match"""
    _, reacs_ck, specs_ck = read_mech(ck_file, None)
    _, reacs_cti, specs_cti = read_mech_ct(cti_file)

    assert len(reacs_ck) == len(reacs_cti)
    for i in range(len(reacs_ck)):
        assert reacs_ck[i] == reacs_cti[i]

    assert len(specs_ck) == len(specs_cti)
    for i in range(len(specs_ck)):
        assert specs_ck[i] == specs_cti[i]
