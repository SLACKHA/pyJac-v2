#compatibility
from builtins import range

#local imports
from ..core import array_creator as arc
from . import TestClass

#nose tools
from nose.tools import assert_raises

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

def test_non_contiguous_input():
    lp_opt = __dummy_opts('map')
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(6, 12)),
            dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 1


def test_contiguous_input():
    lp_opt = __dummy_opts('map')
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0


def test_contiguous_offset_input():
    lp_opt = __dummy_opts('map')
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

def test_multiple_inputs():
    lp_opt = __dummy_opts('map')
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
    assert 'x' in mstore.transformed_variables

    #add another mapped variable
    c4 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(5, 11)),
            dtype=np.int32))
    mstore.check_and_add_transform('x2', 'i', c4)
    assert len(mstore.transformed_domains) == 2
    assert 'x2' in mstore.transformed_variables

def test_offset_base():
    lp_opt = __dummy_opts('map')
    c = arc.creator('', np.int32, (10,), 'C',
        initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    #add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
        initializer=np.array(list(range(4)) + list(range(5, 11)),
        dtype=np.int32))
    mstore.check_and_add_transform('x', 'i', c2)

    assert len(mstore.transformed_domains) == 2
    assert 'x' in mstore.transformed_variables
