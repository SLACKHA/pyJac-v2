import numpy as np
from textwrap import dedent

from pyjac.core import array_creator as arc
from pyjac.core.instruction_creator import get_update_instruction
from pyjac.core.array_creator import MapStore, creator
from pyjac.tests.test_utils import TestingLogger
from pyjac.tests import set_seed

set_seed()


def test_get_update_instruction():
    # test the update instruction creator

    # first, create some domains
    map_np = np.arange(12, dtype=arc.kint_type)
    map_domain = creator('map', map_np.dtype, map_np.shape, 'C', initializer=map_np)

    # and a dummy loopy options and mapstore
    loopy_opts = type('', (object,), {
        'use_working_buffer': False, 'pre_split': False})
    mapstore = MapStore(loopy_opts, map_domain, True)

    # add a new domain
    domain_np = np.arange(12, dtype=arc.kint_type) + 2
    domain = creator('domain', domain_np.dtype, domain_np.shape, 'C',
                     initializer=domain_np)

    mapstore.check_and_add_transform(domain, map_domain)

    # test #1, non-finalized mapstore produces warming
    # capture logging
    tl = TestingLogger()
    tl.start_capture(logname='pyjac.core.instruction_creator')

    get_update_instruction(mapstore, map_domain, 'dummy')

    logs = tl.stop_capture()
    assert 'non-finalized mapstore' in logs

    # test #2, empty map produces no-op with correct ID
    test_instrution = '<> test = domain[i] {id=anchor}'
    insn = get_update_instruction(mapstore, None, test_instrution)
    assert insn == '... nop {id=anchor}'

    # test #3, non-transformed domain doesn't result in guarded update insn
    insn = get_update_instruction(mapstore, domain, test_instrution)
    assert insn == test_instrution

    # test #4, transformed domain results in guarded update insn
    mapstore = MapStore(loopy_opts, map_domain, True)
    domain_np = np.full_like(domain_np, -1)
    choice = np.sort(np.random.choice(domain_np.size, domain_np.size - 3,
                                      replace=False))
    domain_np[choice] = np.arange(choice.size)
    domain = creator('domain', domain_np.dtype, domain_np.shape, 'C',
                     initializer=domain_np)
    variable = creator('variable', domain_np.dtype, domain_np.shape, 'C')

    mapstore.check_and_add_transform(domain, map_domain)
    mapstore.check_and_add_transform(variable, domain)
    insn = get_update_instruction(mapstore, variable, test_instrution)
    test = dedent("""
        if i_0 >= 0
            <> test = domain[i] {id=anchor}
        end
    """).strip()
    assert dedent(insn).strip() == test
