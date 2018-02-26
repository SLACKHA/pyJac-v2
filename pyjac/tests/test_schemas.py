"""
A unit tester that loads the schemas packaged with :mod:`pyJac` and validates
the given example specifications against them.
"""

# system
from os.path import isfile, join
from collections import OrderedDict

# external
import six
import cantera as ct
from nose.tools import assert_raises

# internal
from ..libgen.libgen import build_type
from ..loopy_utils.loopy_utils import JacobianFormat
from ..utils import func_logger, enum_to_string, listify
from .test_utils import xfail
from . import script_dir as test_mech_dir
from .test_utils.get_test_matrix import load_models, load_from_key, model_key, \
    load_platforms, load_tests
from ..examples import examples_dir
from ..schemas import schema_dir, __prefixify, build_and_validate


@func_logger
def runschema(schema, source, should_fail=False, includes=[]):

    # add common
    includes.append('common_schema.yaml')

    # check source / schema / includes, and prepend prefixes
    def __check(file, fdir=schema_dir):
        assert isinstance(file, six.string_types), 'Schema file should be string'
        file = __prefixify(file, fdir)
        assert isfile(file), 'File {} not found.'.format(file)
        return file

    schema = __check(schema)
    source = __check(source, examples_dir)
    includes = [__check(inc) for inc in includes]

    # define inner tester
    @xfail(should_fail)
    def _internal(source, schema, includes):
        # make schema
        built = build_and_validate(schema, source, includes=includes)
        assert built is not None
        return built

    return _internal(source, schema, includes)


def test_test_platform_schema_specification():
    runschema('test_platform_schema.yaml', 'test_platforms.yaml')


def test_codegen_platform_schema_specification():
    runschema('codegen_platform.yaml', 'codegen_platform.yaml')


def test_load_codegen():
    from ..loopy_utils.loopy_utils import load_platform
    from pyopencl import Platform
    platform = load_platform(__prefixify(
            'codegen_platform.yaml', examples_dir))
    assert isinstance(platform.platform, Platform) or platform.platform == 'intel'
    assert platform.width == 4
    assert not platform.depth
    assert platform.use_atomics is False


def test_matrix_schema_specification():
    runschema('test_matrix_schema.yaml', 'test_matrix.yaml',
              includes=['platform_schema.yaml'])


def __get_test_matrix():
    return build_and_validate('test_matrix_schema.yaml', __prefixify(
        'test_matrix.yaml', examples_dir), includes=['platform_schema.yaml'])


def test_parse_models():
    models = load_models('', __get_test_matrix())

    # test the test mechanism
    assert 'TestMech' in models
    gas = ct.Solution(join(test_mech_dir, 'test.cti'))
    assert gas.n_species == models['TestMech']['ns']
    assert 'limits' in models['TestMech']

    def __test_limit(enumlist, limit):
        stypes = [enum_to_string(enum) for enum in listify(enumlist)]
        root = models['TestMech']['limits']
        for i, stype in enumerate(stypes):
            assert stype in root
            if i == len(stypes) - 1:
                assert root[stype] == limit
            else:
                root = root[stype]

    __test_limit(build_type.species_rates, 10000000)
    __test_limit([build_type.jacobian, JacobianFormat.sparse], 100000)
    __test_limit([build_type.jacobian, JacobianFormat.full], 1000)

    # test gri-mech
    assert 'CH4' in models
    gas = ct.Solution(models['CH4']['mech'])
    assert models['CH4']['ns'] == gas.n_species


def test_load_from_key():
    matrix = __get_test_matrix()
    # test that we have 2 models
    assert len(load_from_key(matrix, model_key)) == 2
    assert len(load_from_key(matrix, '^$')) == 0


def test_load_platform():
    platforms = load_platforms(__get_test_matrix(), raise_on_empty=True)
    platforms = [OrderedDict(p) for p in platforms]

    intel = next(p for p in platforms if p['platform'] == 'intel')
    assert intel['lang'] == 'opencl'
    assert intel['use_atomics'] is False
    assert intel['width'] == [2, 4, 8, None]
    assert intel['depth'] is None

    openmp = next(p for p in platforms if p['platform'] == 'OpenMP')
    assert openmp['lang'] == 'c'
    assert openmp['width'] is None
    assert openmp['depth'] is None

    # test empty platform w/ raise -> assert
    with assert_raises(Exception):
        load_platforms(None, raise_on_empty=True)

    # test empty platform
    platforms = load_platforms(None, langs=['c'], raise_on_empty=False)
    assert len(platforms) == 1
    openmp = OrderedDict(platforms[0])
    assert openmp['lang'] == 'c'
    assert openmp['platform'] == 'OpenMP'
    assert len(platforms[0]) == 2


def test_load_tests():
    tests = load_tests(__get_test_matrix(), 'test_matrix_schema.yaml')
    assert len(tests) == 3
