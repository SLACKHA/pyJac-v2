# compatibility
from six.moves import range

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


def _dummy_opts(knl_type, order='C', use_private_memory=False):
    class dummy(object):
        def __init__(self, knl_type, order='C', use_private_memory=False):
            self.knl_type = knl_type
            self.order = order
            self.use_private_memory = use_private_memory
    return dummy(knl_type, order=order, use_private_memory=use_private_memory)


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

    # test that creation of mapstore with non-contiguous map forces
    # generation of input map
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.array(list(range(4)) + list(range(6, 12)),
                                         dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    mstore.finalize()
    assert len(mstore.transformed_domains) == 1
    assert mstore.tree.parent is not None
    assert np.allclose(mstore.tree.parent.domain.initializer, np.arange(10))


@parameterized(['map', 'mask'])
def test_contiguous_input(maptype):

    # test that creation of mapstore with contiguous map has no effect
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
        mstore.finalize()
        assert len(mstore.transformed_domains) == 0


def __create_var(name, size=(10,)):
    return arc.creator(name, np.int32, size, 'C')


def test_contiguous_offset_input():
    lp_opt = _dummy_opts('map')
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add a creator that can be mapped affinely
    c2 = arc.creator('c2', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    x = __create_var('x')
    mstore.check_and_add_transform(x, c2, 'i')
    mstore.finalize()

    # test affine mapping in there
    assert len(mstore.transformed_domains) == 1
    assert mstore.domain_to_nodes[c2] in mstore.transformed_domains
    assert mstore.domain_to_nodes[x].parent.domain == c2
    assert mstore.domain_to_nodes[x].iname == 'i + -3'


def test_contiguous_offset_input_map():
    # same as the above, but check that a non-affine mappable transform
    # results in an input map

    lp_opt = _dummy_opts('map')
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add a creator that can be mapped affinely
    c2 = arc.creator('c2', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    x = __create_var('x')
    mstore.check_and_add_transform(x, c2, 'i')

    # and another creator that can't be affinely mapped
    c3 = arc.creator('c3', np.int32, (10,), 'C',
                     initializer=np.array(list(range(4)) + list(range(6, 12)),
                                          dtype=np.int32))
    x2 = __create_var('x2')
    mstore.check_and_add_transform(x2, c3, 'i')
    mstore.finalize()

    # test affine mapping is not transformed (should be moved to input map)
    assert len(mstore.transformed_domains) == 2
    assert mstore.domain_to_nodes[x] not in mstore.transformed_domains
    # check that non-affine and original indicies in there
    assert mstore.domain_to_nodes[c3] in mstore.transformed_domains
    assert mstore.domain_to_nodes[x2].parent.domain == c3
    # and that the tree has been transformed
    assert mstore.tree in mstore.transformed_domains


def test_input_map_domain_transfer():
    # check that a domain on the tree that matches the input map gets
    # transfered to the input map

    lp_opt = _dummy_opts('map')
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add a creator that matches the coming input map
    c2 = arc.creator('c2', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    x = __create_var('x')
    mstore.check_and_add_transform(x, c2, 'i')

    # and another creator that forces the input map
    c3 = arc.creator('c3', np.int32, (10,), 'C',
                     initializer=np.array(list(range(4)) + list(range(6, 12)),
                                          dtype=np.int32))
    x2 = __create_var('x2')
    mstore.check_and_add_transform(x2, c3, 'i')
    mstore.finalize()

    # test that c2 isn't transformed, and resides on new base
    assert len(mstore.transformed_domains) == 2
    assert mstore.domain_to_nodes[c2] not in mstore.transformed_domains
    assert mstore.domain_to_nodes[c2].parent == mstore.tree.parent
    assert mstore.domain_to_nodes[c2].insn is None
    # check that non-affine mapping in there
    assert mstore.domain_to_nodes[c3] in mstore.transformed_domains
    # and the original base
    assert mstore.domain_to_nodes[c] in mstore.transformed_domains


def test_duplicate_iname_detection():
    # ensures the same transform isn't picked up multiple times
    lp_opt = _dummy_opts('map')

    # create dummy map
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # create a mapped domain
    c2 = arc.creator('c', np.int32, (10,), 'C',
                     initializer=np.array(list(range(3)) +
                                          list(range(4, 11)), dtype=np.int32))

    # add two variables to the same domain
    mstore.check_and_add_transform(__create_var('x'), c2)
    mstore.check_and_add_transform(__create_var('x2'), c2)

    mstore.finalize()

    # ensure there's only one transform insn issued
    assert len(mstore.transform_insns) == 1
    assert [x for x in mstore.transform_insns][0] == \
        mstore.domain_to_nodes[c2].insn

    # now repeat with the variables having initializers
    # to test that leaves aren't mapped
    lp_opt = _dummy_opts('map')

    # create dummy map
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # create a mapped domain
    c2 = arc.creator('c', np.int32, (10,), 'C',
                     initializer=np.array(list(range(3)) +
                                          list(range(4, 11)), dtype=np.int32))

    # add two variables to the same domain
    x = __create_var('x')
    x.initializer = np.arange(10)
    x2 = __create_var('x2')
    x2.initializer = np.arange(10)
    mstore.check_and_add_transform(x, c2)
    mstore.check_and_add_transform(x2, c2)

    mstore.finalize()

    # ensure there's only one transform insn issued
    assert len(mstore.transform_insns) == 1
    assert [y for y in mstore.transform_insns][0] == \
        mstore.domain_to_nodes[c2].insn


def test_map_range_update():
    lp_opt = _dummy_opts('map')
    # test a complicated chaining / input map case

    # create dummy map
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # next add a creator that doesn't need a map
    c2 = arc.creator('c2', np.int32, (10,), 'C',
                     initializer=np.arange(10, 0, -1, dtype=np.int32))
    mstore.check_and_add_transform(c2, c, 'i')

    # and a creator that only needs an affine map
    c3 = arc.creator('c3', np.int32, (10,), 'C',
                     initializer=np.arange(4, 14, dtype=np.int32))
    mstore.check_and_add_transform(c3, c2, 'i')

    # and add a creator that will trigger a transform for c3
    c4 = arc.creator('c4', np.int32, (10,), 'C',
                     initializer=np.arange(4, 14, dtype=np.int32))
    mstore.check_and_add_transform(c4, c3, 'i')

    # and another affine
    c5 = arc.creator('c5', np.int32, (10,), 'C',
                     initializer=np.arange(3, 13, dtype=np.int32))
    mstore.check_and_add_transform(c5, c4, 'i')
    # and we need a final variable to test c5
    x = __create_var('x')
    mstore.check_and_add_transform(x, c5, 'i')
    mstore.finalize()

    # there should be an affine input map of + 3
    assert (mstore.domain_to_nodes[c] == mstore.tree and
            mstore.tree.insn is None and mstore.tree.iname == 'i + 3'
            and mstore.tree.parent is not None)
    # c2 should be on the tree
    assert (mstore.domain_to_nodes[c2].parent == mstore.tree and
            mstore.domain_to_nodes[c2].insn == '<> i_1 = c2[i + 3]')
    # c3 should be an regular transform off c2
    assert (mstore.domain_to_nodes[c3].parent == mstore.domain_to_nodes[c2] and
            mstore.domain_to_nodes[c3].insn == '<> i_2 = c3[i_1]')
    # c4 should not have a transform (and thus should take the iname of c3)
    assert (mstore.domain_to_nodes[c4].parent == mstore.domain_to_nodes[c3] and
            mstore.domain_to_nodes[c4].insn is None
            and mstore.domain_to_nodes[c4].iname == 'i_2')
    # and c5 should be an affine of -1 off c4 (using c3's iname)
    assert (mstore.domain_to_nodes[c5].parent == mstore.domain_to_nodes[c4] and
            mstore.domain_to_nodes[c5].insn is None
            and mstore.domain_to_nodes[c5].iname == 'i_2 + -1')


@parameterized(['mask'])
def test_mask_input(maptype):
    lp_opt = _dummy_opts(maptype)
    # create dummy mask
    mask = np.full((10,), -1, np.int32)
    mask[1] = 3
    mask[3] = 6
    mask[4] = 7
    c = arc.creator('c', np.int32, (10,), 'C',
                    initializer=mask)

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    mask2 = np.array(mask, copy=True)
    mask2[4] = 8
    # add a creator
    c2 = arc.creator('c2', np.int32, (10,), 'C',
                     initializer=mask2)
    mstore.check_and_add_transform(__create_var('x'), c2, 'i')
    mstore.finalize()

    assert len(mstore.transformed_domains) == 1
    assert mstore.domain_to_nodes[c2] in mstore.transformed_domains


@parameterized(['map'])
def test_multiple_inputs(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    mstore.check_and_add_transform(__create_var('x2'), c2, 'i')

    # add a mapped variable
    c3 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(5)) + list(range(6, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform(__create_var('x3'), c3, 'i')

    # test different vaiable with same map
    c4 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(5)) + list(range(6, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform(__create_var('x4'), c4, 'i')

    # add another mapped variable
    c5 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.array(list(range(4)) + list(range(5, 11)),
                                          dtype=np.int32))
    mstore.check_and_add_transform(__create_var('x5'), c5, 'i')

    mstore.finalize()

    assert mstore.domain_to_nodes[c2] not in mstore.transformed_domains
    assert mstore.domain_to_nodes[c3] in mstore.transformed_domains
    assert mstore.domain_to_nodes[c4] in mstore.transformed_domains
    assert mstore.domain_to_nodes[c5] in mstore.transformed_domains

    assert len(mstore.transformed_domains) == 3
    assert np.array_equal(mstore.map_domain.initializer,
                          np.arange(10, dtype=np.int32))


@parameterized(['map', 'mask'])
def test_bad_multiple_variable_map(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # add a variable
    c2 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.arange(10, dtype=np.int32))
    x2 = __create_var('x2')
    mstore.check_and_add_transform(x2, c2, 'i')

    c3 = arc.creator('', np.int32, (10,), 'C',
                     initializer=np.arange(3, 13, dtype=np.int32))
    # add the same variable as a different domain, and check error
    with assert_raises(AssertionError):
        mstore.check_and_add_transform(x2, c3, 'i')


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
    x2 = __create_var('x2')
    mstore.check_and_add_transform(x2, c2, 'i')

    assert len(mstore.transformed_domains) == 0

    # add a masked variable
    mask = np.full((10,), -1, np.int32)
    mask[0] = 2
    c3 = arc.creator('', np.int32, (10,), 'C',
                     initializer=mask)
    x3 = __create_var('x3')
    mstore.check_and_add_transform(x3, c3, 'i')

    # add another mapped variable
    mask2 = mask[:]
    mask2[2] = 2
    c4 = arc.creator('', np.int32, (10,), 'C',
                     initializer=mask2)
    x4 = __create_var('x4')
    mstore.check_and_add_transform(x4, c4, 'i')
    mstore.finalize()

    assert len(mstore.transformed_domains) == 2
    # check c2 in transforms
    assert mstore.domain_to_nodes[c2] not in mstore.transformed_domains
    # check x2 parent
    assert mstore.domain_to_nodes[x2].parent == mstore.domain_to_nodes[c2]
    # check c3 in transforms
    assert mstore.domain_to_nodes[c3] in mstore.transformed_domains
    # and x3 parent
    assert mstore.domain_to_nodes[x3].parent == mstore.domain_to_nodes[c3]
    # check c4 in transforms
    assert mstore.domain_to_nodes[c4] in mstore.transformed_domains
    # check x4 parent
    assert mstore.domain_to_nodes[x4].parent == mstore.domain_to_nodes[c4]


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
    x = __create_var('x')
    mstore.check_and_add_transform(x, c2, 'i')
    mstore.finalize()

    assert len(mstore.transformed_domains) == 2
    assert np.array_equal(mstore.map_domain.initializer,
                          np.arange(10, dtype=np.int32))
    assert mstore.domain_to_nodes[c2] in mstore.transformed_domains
    assert mstore.domain_to_nodes[x].parent == mstore.domain_to_nodes[c2]


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
    assert var_str == 'var[i_1]'
    assert '<> i_1 = domain[i + 3]' in mstore.transform_insns


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
    assert var_str == 'var[i_0]'
    assert '<> i_0 = domain[i]' in mstore.transform_insns


@parameterized(['map'])
def test_chained_maps(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (5,), 'C',
                    initializer=np.arange(5, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    assert len(mstore.transformed_domains) == 0

    def __get_iname(domain):
        return mstore.domain_to_nodes[domain].iname

    # add a variable
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=np.arange(10, dtype=np.int32))
    # this should work
    mstore.check_and_add_transform(var, domain, 'i')

    # now add a chained map
    var2 = arc.creator('var2', np.int32, (10,), 'C')
    domain2 = arc.creator('domain2', np.int32, (10,), 'C',
                          initializer=np.arange(10, dtype=np.int32))

    mstore.check_and_add_transform(domain2, domain)
    mstore.check_and_add_transform(var2, domain2)

    # and finally put another chained map that does require a transform
    var3 = arc.creator('var3', np.int32, (10,), 'C')
    domain3 = arc.creator('domain3', np.int32, (10,), 'C',
                          initializer=np.array(list(range(3)) +
                                               list(range(4, 11)),
                                               dtype=np.int32))

    mstore.check_and_add_transform(domain3, domain2)
    mstore.check_and_add_transform(var3, domain3)

    # now create variables and test
    var_lp, var_str = mstore.apply_maps(var, 'i')

    # test that the base map is there
    assert '<> {} = domain[i]'.format(__get_iname(domain)) in \
        mstore.transform_insns

    # var 1 should be based off domain's iname i_0
    assert var_str == 'var[{}]'.format(__get_iname(var))

    # var 2's iname should be based off domain2's iname
    # however since there is no need for map between domain and domain 2
    # this should _still_be i_0
    var2_lp, var2_str = mstore.apply_maps(var2, 'i')

    assert var2_str == 'var2[{}]'.format(__get_iname(var2))

    # and var 3 should be based off domain 3's iname, i_3
    var3_lp, var3_str = mstore.apply_maps(var3, 'i')
    assert var3_str == 'var3[{}]'.format(__get_iname(var3))
    assert (
        '<> {} = domain3[{}]'.format(
            __get_iname(var3), __get_iname(domain2))
        in mstore.transform_insns)


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
    assert var_str == 'var[i_0]'
    assert '<> i_0 = domain[i]' in mstore.transform_insns


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

    mstore.finalize()
    assert mstore.get_iname_domain() == ('i', '0 <= i <= 9')


@parameterized(['map'])
def test_map_iname_domains(maptype):
    lp_opt = _dummy_opts(maptype)
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.arange(3, 13, dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    mstore.finalize()
    assert mstore.get_iname_domain() == ('i', '3 <= i <= 12')

    # add an affine map
    mstore = arc.MapStore(lp_opt, c, c, 'i')
    mapv = np.arange(10, dtype=np.int32)
    var = arc.creator('var', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(var, domain, 'i')
    mstore.finalize()
    assert mstore.get_iname_domain() == ('i', '3 <= i <= 12')

    # add a non-affine map, domain should bounce to 0-based
    mstore = arc.MapStore(lp_opt, c, c, 'i')
    mapv = np.array(list(range(3)) + list(range(4, 11)), dtype=np.int32)
    var = arc.creator('var2', np.int32, (10,), 'C')
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(var, domain, 'i')
    mstore.finalize()
    assert mstore.get_iname_domain() == ('i', '0 <= i <= 9')

    # check non-contigous
    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.array(list(range(3)) + list(range(4, 11)),
                                         dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')
    mstore.finalize()
    assert mstore.get_iname_domain() == ('i', '0 <= i <= 9')


def test_leaf_inames():
    lp_opt = _dummy_opts('map')

    c = arc.creator('base', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))
    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # create one map
    mapv = np.array(list(range(3)) + list(range(4, 11)), dtype=np.int32)
    mapv2 = np.array(list(range(2)) + list(range(3, 11)), dtype=np.int32)
    domain2 = arc.creator('domain2', np.int32, (10,), 'C',
                          initializer=mapv2)
    domain = arc.creator('domain', np.int32, (10,), 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(domain2, domain, 'i')

    # and another
    var = arc.creator('var', np.int32, (10,), 'C')
    mstore.check_and_add_transform(var, domain2, 'i')

    # now create var
    _, d_str = mstore.apply_maps(domain, 'i')
    _, d2_str = mstore.apply_maps(domain2, 'i')
    _, v_str = mstore.apply_maps(var, 'i')

    assert d_str == 'domain[i]'
    assert d2_str == 'domain2[i_0]'
    assert v_str == 'var[i_1]'
    assert '<> i_0 = domain[i]' in mstore.transform_insns
    assert '<> i_1 = domain2[i_0]' in mstore.transform_insns


def test_input_map_pickup():
    lp_opt = _dummy_opts('map')

    # test that creation of mapstore with non-contiguous map forces
    # non-transformed variables to pick up the right iname
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.array(list(range(4)) + list(range(6, 12)),
                                         dtype=np.int32))

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    # create a variable
    x = __create_var('x')
    _, x_str = mstore.apply_maps(x, 'i')

    assert 'i_0' in x_str


def test_fixed_creator_indices():
    c = arc.creator('base', np.int32, ('isize', 'jsize'), 'C',
                    fixed_indicies=[(0, 1)])
    assert c('j')[1] == 'base[1, j]'


@parameterized(['map'])
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
        mapv = np.array(mapv, copy=True) + 1
    else:
        mapv[0] = -1
        mapv[1:] = np.arange(10, dtype=np.int32)
    var = arc.creator('var', np.int32, mapv.shape, 'C')
    domain = arc.creator('domain', np.int32, mapv.shape, 'C',
                         initializer=mapv)
    mstore.check_and_add_transform(var, domain, 'i')
    _, var_str = mstore.apply_maps(var, 'i')
    assert var_str == 'var[i + 1]'
    assert len(mstore.transform_insns) == 0


def test_private_memory_creations():
    lp_opt = _dummy_opts('map', use_private_memory=True)

    # make a creator to form the base of the mapstore
    c = arc.creator('', np.int32, (10,), 'C',
                    initializer=np.arange(10, dtype=np.int32))

    # and the array to test
    arr = arc.creator('a', np.int32, (10, 10), 'C')

    # and a final "input" array
    inp = arc.creator('b', np.int32, (10, 10), 'C',
                      is_input_or_output=True)

    mstore = arc.MapStore(lp_opt, c, c, 'i')

    arr_lp, arr_str = mstore.apply_maps(arr, 'j', 'i')
    assert isinstance(arr_lp, lp.TemporaryVariable) and arr_lp.shape == (10,)
    assert arr_str == 'a[i]'

    inp_lp, inp_str = mstore.apply_maps(inp, 'j', 'i')
    assert isinstance(inp_lp, lp.GlobalArg) and inp_lp.shape == (10, 10)
    assert inp_str == 'b[j, i]'

    # now test input without the global index
    arr_lp, arr_str = mstore.apply_maps(arr, 'k', 'i')
    assert isinstance(arr_lp, lp.GlobalArg) and arr_lp.shape == (10, 10)
    assert arr_str == 'a[k, i]'


class SubTest(TestClass):
    @attr('long')
    def test_namestore_init(self):
        lp_opt = _dummy_opts('map')
        from ..core.rate_subs import assign_rates
        from ..loopy_utils.loopy_utils import RateSpecialization
        rate_info = assign_rates(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)
        arc.NameStore(lp_opt, rate_info, True, self.store.test_size)

    @attr('long')
    def test_input_private_memory_creations(self):
        lp_opt = _dummy_opts('map', use_private_memory=True)
        from ..core.rate_subs import assign_rates
        from ..loopy_utils.loopy_utils import RateSpecialization
        rate_info = assign_rates(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)
        # create name and mapstores
        nstore = arc.NameStore(lp_opt, rate_info, True, self.store.test_size)
        mstore = arc.MapStore(lp_opt, nstore.phi_inds, nstore.phi_inds, 'i')

        # create known input
        jac_lp, jac_str = mstore.apply_maps(nstore.jac, 'j', 'k', 'i')

        assert isinstance(jac_lp, lp.GlobalArg) and jac_lp.shape == nstore.jac.shape
        assert jac_str == 'jac[j, k, i]'
