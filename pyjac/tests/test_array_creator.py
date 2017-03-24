#compatibility
from builtins import range

#local imports
from ..core import array_creator as arc
from . import TestClass

#nose tools
from nose.tools import assert_raises
from parameterized import parameterized

#modules
import loopy as lp
import numpy as np

def __dummy_opts(knl_type):
    class dummy(object):
        def __init__(self, knl_type):
            self.knl_type = knl_type
    return dummy(knl_type)


def test_creator_asserts():
    #check dtype
    with assert_raises(AssertionError):
        arc.creator('', np.int32, (10,), 'C',
            initializer=np.arange(10, dtype=np.float64))
    #test shape
    with assert_raises(AssertionError):
        arc.creator('', np.int32, (11,), 'C',
            initializer=np.arange(10, dtype=np.float32))


@parameterized(['map'])
def test_non_contiguous_input(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(6, 12)),
            dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0


@parameterized(['map', 'mask'])
def test_contiguous_input(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0


@parameterized(['mask'])
def test_invalid_mask_input(maptype):
    lp_opt = __dummy_opts(maptype)
    mask = np.full((10,), -1, np.int32)
    mask[1] = 3
    mask[3] = 6
    mask[4] = 11
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(6, 12)),
            dtype=np.int32))

    with assert_raises(AssertionError):
        mstore = arc.MapStore(lp_opt, c, c, 'i')
        assert len(mstore.transformed_domains) == 0


# this test only makes sense for a map
@parameterized(['map'])
def test_contiguous_offset_input(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a creator
    c2 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform('x', 'i', c2)

    assert len(mstore.transformed_domains) == 1
    assert 'x' in mstore.transformed_variables


@parameterized(['mask'])
def test_mask_input(maptype):
    lp_opt = __dummy_opts(maptype)
    #create dummy mask
    mask = np.full((10,), -1, np.int32)
    mask[1] = 3
    mask[3] = 6
    mask[4] = 7
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=mask)

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    mask2 = mask[:]
    mask2[4] = 8
    #add a creator
    c2 = arc.creator('', np.int32, (10,), 'C',
        initializer=mask2)
    mstore.check_and_add_transform('x', 'i', c2)

    assert len(mstore.transformed_domains) == 1
    assert 'x' in mstore.transformed_variables


@parameterized(['map'])
def test_multiple_inputs(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform('x', 'i', c2)

    assert len(mstore.transformed_domains) == 0

    #add a mapped variable
    c3 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(5)) + list(range(6, 11)),
            dtype=np.int32))
    mstore.check_and_add_transform('x', 'i', c3)
    assert len(mstore.transformed_domains) == 1
    assert np.array_equal(mstore.map_domain.initializer,
        np.arange(10, dtype=np.int32))
    assert 'x' in mstore.transformed_variables

    #add another mapped variable
    c4 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(5, 11)),
            dtype=np.int32))
    mstore.check_and_add_transform('x2', 'i', c4)
    assert len(mstore.transformed_domains) == 2
    assert 'x2' in mstore.transformed_variables


@parameterized(['mask'])
def test_multiple_mask_inputs(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform('x', 'i', c2)

    assert len(mstore.transformed_domains) == 0

    #add a masked variable
    mask = np.full((10,), -1, np.int32)
    mask[0] = 2
    c3 = arc.creator('', np.int32, (10,), 'C',
        initializer=mask)
    mstore.check_and_add_transform('x', 'i', c3)
    assert len(mstore.transformed_domains) == 1
    assert 'x' in mstore.transformed_variables

    #add another mapped variable
    mask2 = mask[:]
    mask2[2] = 2
    c4 = arc.creator('', np.int32, (10,), 'C',
        initializer=mask2)
    mstore.check_and_add_transform('x2', 'i', c4)
    assert len(mstore.transformed_domains) == 2
    assert 'x2' in mstore.transformed_variables


@parameterized(['map'])
def test_offset_base(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(5, 11)),
        dtype=np.int32))
    mstore.check_and_add_transform('x', 'i', c2)

    assert len(mstore.transformed_domains) == 1
    assert np.array_equal(mstore.map_domain.initializer,
        np.arange(10, dtype=np.int32))
    assert 'x' in mstore.transformed_variables


@parameterized(['map'])
def test_map_variable_creator(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
        initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a variable
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(5, 11)),
        dtype=np.int32))
    mstore.check_and_add_transform(var, 'i', domain)
    var, var_str, map_insn = mstore.apply_maps(var, 'i')

    assert isinstance(var, lp.GlobalArg)
    assert var_str == 'var[i_map]'
    assert map_insn == '<>i_map = domain[i]'

@parameterized(['mask'])
def test_mask_variable_creator(maptype):
    lp_opt = __dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a variable
    mask = np.full((10,), -1, np.int32)
    mask[0] = 2
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
        initializer=mask)
    mstore.check_and_add_transform(var, 'i', domain)
    var, var_str, map_insn = mstore.apply_maps(var, 'i')

    assert isinstance(var, lp.GlobalArg)
    assert var_str == 'var[i_mask]'
    assert map_insn == '<>i_mask = domain[i]'