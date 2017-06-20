# compatibility
from builtins import range

# local imports
from ..core import array_creator as arc
from . import TestClass

# nose tools
from nose.tools import assert_raises
from nose.plugins.attrib import attr
from parameterized import parameterized

# modules
import loopy as lp
import numpy as np


def _dummy_opts(knl_type):
    class dummy(object):
        def __init__(self, knl_type, order='C'):
            self.knl_type = knl_type
            self.order = order
    return dummy(knl_type)


def test_creator_asserts():
    # check dtype
    with assert_raises(AssertionError):
        arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.float64))
    # test shape
    with assert_raises(AssertionError):
        arc.creator('', np.int32, (11,), 'C',
                    initializer=np.arange(10, dtype=np.float32))


@parameterized(['map'])
def test_non_contiguous_input(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.array(list(range(4)) + list(range(6, 12)),
                                         dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0


@parameterized(['map', 'mask'])
def test_contiguous_input(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0


@parameterized(['mask'])
def test_invalid_mask_input(maptype):
    lp_opt = _dummy_opts(maptype)
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
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a creator
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform('x', c2, 'i')

    assert len(mstore.transformed_domains) == 1
    assert 'x' in mstore.transformed_variables


@parameterized(['mask'])
def test_mask_input(maptype):
    lp_opt = _dummy_opts(maptype)
    # create dummy mask
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
    # add a creator
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=mask2)
    mstore.check_and_add_transform('x', c2, 'i')

    assert len(mstore.transformed_domains) == 1
    assert 'x' in mstore.transformed_variables


@parameterized(['map'])
def test_multiple_inputs(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform('x', c2, 'i')

    assert len(mstore.transformed_domains) == 0

    # add a mapped variable
    c3 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(5)) + list(range(6, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform('x', c3, 'i')
    assert len(mstore.transformed_domains) == 1
    assert np.array_equal(mstore.map_domain.initializer,
                          np.arange(10, dtype=np.int32))
    assert 'x' in mstore.transformed_variables

    # test different vaiable with same map
    c4 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(5)) + list(range(6, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform('x3', c3, 'i')
    assert len(mstore.transformed_domains) == 1
    assert 'x3' in mstore.transformed_variables

    # add another mapped variable
    c4 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(4)) + list(range(5, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform('x2', c4, 'i')
    assert len(mstore.transformed_domains) == 2
    assert 'x2' in mstore.transformed_variables


@parameterized(['mask'])
def test_multiple_mask_inputs(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform('x', c2, 'i')

    assert len(mstore.transformed_domains) == 0

    # add a masked variable
    mask = np.full((10,), -1, np.int32)
    mask[0] = 2
    c3 = arc.creator('', np.int32, (10,), 'C',
                     initializer=mask)
    mstore.check_and_add_transform('x', c3, 'i')
    assert len(mstore.transformed_domains) == 1
    assert 'x' in mstore.transformed_variables

    # add another mapped variable
    mask2 = mask[:]
    mask2[2] = 2
    c4 = arc.creator('', np.int32, (10,), 'C',
                     initializer=mask2)
    mstore.check_and_add_transform('x2', c4, 'i')
    assert len(mstore.transformed_domains) == 2
    assert 'x2' in mstore.transformed_variables


@parameterized(['map'])
def test_offset_base(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(4)) + list(range(5, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform('x', c2, 'i')

    assert len(mstore.transformed_domains) == 1
    assert np.array_equal(mstore.map_domain.initializer,
                          np.arange(10, dtype=np.int32))
    assert 'x' in mstore.transformed_variables


@parameterized(['map'])
def test_map_variable_creator(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a variable
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=np.array(list(range(4)) +
                                              list(range(5, 11)),
                                              dtype=np.int32))
    mstore.check_and_add_transform(var, domain, 'i')
    var, var_str = mstore.apply_maps(var, 'i')

    assert isinstance(var, lp.GlobalArg)
    assert var_str == 'var[i_map]'
    assert '<> i_map = domain[i]' in mstore.transform_insns


@parameterized(['map'])
def test_map_to_larger(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (5,), 'C',
                    initializer=np.arange(5, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a variable
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=np.arange(10, dtype=np.int32))
    # this should work
    mstore.check_and_add_transform(var, domain, 'i')
    var, var_str = mstore.apply_maps(var, 'i')

    assert isinstance(var, lp.GlobalArg)
    assert var_str == 'var[i_map]'
    assert '<> i_map = domain[i]' in mstore.transform_insns


@parameterized(['mask'])
def test_mask_variable_creator(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    # add a variable
    mask = np.full((10,), -1, np.int32)
    mask[0] = 2
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mask)
    mstore.check_and_add_transform(var, domain, 'i')
    var, var_str = mstore.apply_maps(var, 'i')

    assert isinstance(var, lp.GlobalArg)
    assert var_str == 'var[i_mask]'
    assert '<> i_mask = domain[i]' in mstore.transform_insns


@parameterized(['mask'])
def test_mask_iname_domains(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add a variable
    mask = np.full((10,), -1, np.int32)
    mask[0] = 2
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mask)
    mstore.check_and_add_transform(var, domain, 'i')

    assert mstore.get_iname_domain() == ('i', '0 <= i <= 9')


@parameterized(['map'])
def test_map_iname_domains(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert mstore.get_iname_domain() == ('i', '3 <= i <= 12')

    # add an affine map
    mapv = np.arange(10, dtype=np.int32)
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(var, domain, 'i')

    assert mstore.get_iname_domain() == ('i', '3 <= i <= 12')

    # add a non-affine map, domain should bounce to 0-based
    mapv = np.array(list(range(3)) + list(range(4, 11)), dtype=np.int32)
    var = arc.creator('var2', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(var, domain, 'i')
    assert mstore.get_iname_domain() == ('i', '0 <= i <= 9')

    # check non-contigous
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.array(list(range(3)) + list(range(4, 11)),
                                         dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert mstore.get_iname_domain() == ('i', '0 <= i <= 9')


def test_fixed_creator_indices():
    c = arc.creator('base', np.int32, ('isize', 'jsize'), 'C',
                    fixed_indicies=[(0, 1)])
    assert c('j')[1] == 'base[1, j]'


@parameterized(['map', 'mask'])
def test_force_inline(maptype):
    lp_opt = _dummy_opts(maptype)
    if maptype == 'map':
        mapv = np.arange(0, 5, dtype=np.int32)
    else:
        mapv = np.full(11, -1, dtype=np.int32)
        mapv[:10] = np.arange(10, dtype=np.int32)
    c = arc.creator('base', np.int32, mapv.shape, 'C',
                    initializer=mapv)

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add an affine map
    if maptype == 'map':
        mapv += 1
    else:
        mapv[0] = -1
        mapv[1:] = np.arange(10, dtype=np.int32)
    var = arc.creator('var', np.int32, mapv.shape, 'C')
    domain = arc.creator('domain', np.int32, mapv.shape, 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(var, domain, 'i', force_inline=True)
    var, var_str = mstore.apply_maps(var, 'i')
    assert var_str == 'var[i + 1]'.format(maptype)
    assert len(mstore.transform_insns) == 0


@attr('long')
class SubTest(TestClass):
    def test_namestore_init(self):
        lp_opt = _dummy_opts('map')
        from ..core.rate_subs import assign_rates
        from ..loopy_utils.loopy_utils import RateSpecialization
        rate_info = assign_rates(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)
        arc.NameStore(lp_opt, rate_info, True, self.store.test_size)
