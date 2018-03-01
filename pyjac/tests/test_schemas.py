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
from tempfile import NamedTemporaryFile

# internal
from pyjac.libgen.libgen import build_type
from pyjac.loopy_utils.loopy_utils import JacobianFormat, JacobianType
from pyjac.utils import func_logger, enum_to_string, listify
from pyjac.tests.test_utils import xfail
from pyjac.tests import script_dir as test_mech_dir
from pyjac.tests.test_utils.get_test_matrix import load_models, load_platforms, \
    load_tests, get_test_matrix, num_cores_default
from pyjac.examples import examples_dir
from pyjac.schemas import schema_dir, __prefixify, build_and_validate
from pyjac.core.exceptions import OverrideCollisionException, DuplicateTestException
from pyjac.loopy_utils.loopy_utils import load_platform
from pyjac.kernel_utils.memory_manager import load_memory_limits

current_test_langs = ['c', 'opencl']
"""
:attr:`current_test_langs`
    we need to define the languages to test here separately from the
    :func:`get_test_langs` as these tests depend on the entire platform / test
    matrix files
"""


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


def test_load_test_platforms():
    platforms = load_platforms(
        runschema('test_platform_schema.yaml', 'test_platforms.yaml'),
        langs=current_test_langs,
        raise_on_empty=True)
    platforms = [OrderedDict(p) for p in platforms]

    # amd
    amd = next(p for p in platforms if 'amd' in p['platform'].lower())
    assert amd['lang'] == 'opencl'
    assert amd['use_atomics'] is True

    def __fuzz_equal(arr):
        return arr == [2, 4, None] or arr == [2, 4]
    assert __fuzz_equal(amd['width'])
    assert __fuzz_equal(amd['depth'])
    assert amd['depth'] != amd['width']

    # openmp
    openmp = next(p for p in platforms if 'openmp' in p['platform'].lower())
    assert openmp['lang'] == 'c'
    assert openmp['width'] is None
    assert openmp['depth'] is None

    # nvidia
    openmp = next(p for p in platforms if 'nvidia' in p['platform'].lower())
    assert openmp['lang'] == 'opencl'
    assert openmp['width'] == [64, 128, 256]
    assert openmp['depth'] is None
    assert openmp['use_atomics'] is False

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


def test_codegen_platform_schema_specification():
    runschema('codegen_platform.yaml', 'codegen_platform.yaml')


def test_load_codegen():
    from pyopencl import Platform
    platform = load_platform(__prefixify(
            'codegen_platform.yaml', examples_dir))
    assert isinstance(platform.platform, Platform)
    assert platform.width == 4
    assert not platform.depth
    assert platform.use_atomics is True


def test_matrix_schema_specification():
    runschema('test_matrix_schema.yaml', 'test_matrix.yaml')


def __get_test_matrix(**kwargs):
    return build_and_validate('test_matrix_schema.yaml', __prefixify(
        'test_matrix.yaml', examples_dir),
        **kwargs)


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


def test_load_platforms_from_matrix():
    platforms = load_platforms(__get_test_matrix(allow_unknown=True),
                               langs=current_test_langs,
                               raise_on_empty=True)
    platforms = [OrderedDict(p) for p in platforms]

    intel = next(p for p in platforms if 'intel' in p['platform'].lower())
    assert intel['lang'] == 'opencl'
    assert intel['use_atomics'] is False
    assert intel['width'] == [2, 4, 8, None]
    assert intel['depth'] is None

    openmp = next(p for p in platforms if 'openmp' in p['platform'].lower())
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
    assert openmp['platform'].lower() == 'openmp'
    assert len(platforms[0]) == 2


def test_duplicate_tests_fails():
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: openmp
            lang: c
            vectype: [par]
        test-list:
          - type: performance
            eval-type: jacobian
          - type: performance
            eval-type: both
        """)
        file.seek(0)

        with assert_raises(DuplicateTestException):
            tests = build_and_validate('test_matrix_schema.yaml', file.name)
            load_tests(tests, file.name)

    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: openmp
            lang: c
            vectype: [par]
        test-list:
          - type: performance
            eval-type: jacobian
            sparse:
                num_cores: [1]
            finite_difference:
                num_cores: [1]
        """)
        file.seek(0)

        with assert_raises(OverrideCollisionException):
            tests = build_and_validate('test_matrix_schema.yaml', file.name)
            load_tests(tests, file.name)


def test_load_tests():
    # load tests doesn't do any processing other than collision / duplicate
    # checking, hence we just check we get the right number of tests
    tests = load_tests(__get_test_matrix(), 'test_matrix_schema.yaml')
    assert len(tests) == 3


def test_get_test_matrix():
    # test that the example test matrix is specified correctly
    def update(want, state, key, seen):
        if state[key] not in seen[key]:
            if six.callable(want[key]):
                want[key](state, want, seen)
            else:
                want[key].remove(state[key])
                seen[key].add(state[key])
        return want, seen

    def run(want, loop, final_checks=None):
        from copy import deepcopy
        seen = defaultdict(lambda: set())
        test = deepcopy(want)
        for state in loop:
            for key in test:
                update(test, state, key, seen)
        # assert we didn't miss anything (that isn't callable -- those handle
        # themselves)
        assert not any(len(v) for v in test.values() if not six.callable(v))
        if final_checks:
            assert final_checks(seen)

    test_matrix = __prefixify('test_matrix.yaml', examples_dir)

    # get the species validation test
    _, loop, max_vec_width = get_test_matrix('.', build_type.species_rates,
                                             test_matrix, True,
                                             langs=current_test_langs,
                                             raise_on_missing=True)
    assert max_vec_width == 8
    from collections import defaultdict

    def vecsize_check(state, want, seen):
        if state['lang'] == 'c':
            assert state['vecsize'] is None
            assert state['wide'] is False
            assert state['deep'] is False
        else:
            seen['vecsize'].add(state['vecsize'])

    def check_final_vecsizes(seen):
        return sorted(seen['vecsize']) == [2, 4, 8]

    # check we have reasonable values
    base = {'platform': ['intel', 'openmp'],
            'wide': [True, False],
            'vecsize': vecsize_check,
            'conp': [True, False],
            'order': ['C', 'F'],
            'num_cores': num_cores_default()[0]}
    run(base, loop, final_checks=check_final_vecsizes)

    # repeat for jacobian
    _, loop, _ = get_test_matrix('.', build_type.jacobian,
                                 test_matrix, True,
                                 langs=current_test_langs,
                                 raise_on_missing=True)
    jacbase = base.copy()
    jacbase.update({
        'sparse': [enum_to_string(JacobianFormat.sparse),
                   enum_to_string(JacobianFormat.full)],
        'jac_type': [enum_to_string(JacobianType.exact)],
        'use_atomics': [True, False]})  # true for OpenMP by default
    run(jacbase, loop, final_checks=check_final_vecsizes)

    # next, do species performance
    _, loop, _ = get_test_matrix('.', build_type.species_rates,
                                 test_matrix, False,
                                 langs=current_test_langs,
                                 raise_on_missing=True)
    want = base.copy()
    want.update({'order': ['F']})
    run(want, loop, final_checks=check_final_vecsizes)

    # and finally, the Jacobian performance
    _, loop, _ = get_test_matrix('.', build_type.jacobian,
                                 test_matrix, False,
                                 langs=current_test_langs,
                                 raise_on_missing=True)
    want = jacbase.copy()
    # no more openmp
    want.update({'use_atomics': [False]})

    def update_jactype(state, want, seen):
        if state['jac_type'] == enum_to_string(JacobianType.finite_difference):
            assert state['num_cores'] == 1
            assert state['vecsize'] is None
            assert state['wide'] is False
            assert state['depth'] is False
            assert state['order'] == 'C'
            assert state['conp'] is True
        else:
            assert state['vecsize'] == 4

    want.update({'platform': ['intel'],
                 'jac_type': update_jactype})

    def check_final_vecsizes(seen):
        return len(seen['vecsize'] - set([4, None])) == 0
    run(want, loop, final_checks=check_final_vecsizes)

    # test gpu vs cpu specs
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: nvidia
            lang: opencl
            vectype: [wide]
            vecsize: [128]
          - name: intel
            lang: opencl
            vectype: [wide]
            vecsize: [4]
        test-list:
          - type: performance
            eval-type: jacobian
            sparse:
                gpuvecsize: [64]
                order: ['F']
            full:
                vecsize: [2]
                gpuorder: ['C']
        """)
        file.flush()

        _, loop, _ = get_test_matrix('.', build_type.jacobian,
                                     file.name, False,
                                     langs=current_test_langs,
                                     raise_on_missing=True)

    from pyjac.tests import platform_is_gpu

    def sparsetest(state, want, seen):
        if state['sparse'] == enum_to_string(JacobianFormat.sparse):
            if platform_is_gpu(state['platform']):
                assert state['vecsize'] == 64
            else:
                assert state['vecsize'] == 4
                assert state['order'] == 'F'
        else:
            if platform_is_gpu(state['platform']):
                assert state['order'] == 'C'
                assert state['vecsize'] == 128
            else:
                assert state['vecsize'] == 2

    want = {'sparse': sparsetest}
    run(want, loop)


def test_load_memory_limits():
    limits = load_memory_limits(__prefixify('codegen_platform.yaml', examples_dir))
    assert limits['global'] == 1e9
    assert limits['constant'] == 1e6
    assert limits['alloc'] == 100e6

    limits = load_memory_limits(__prefixify('test_matrix.yaml', examples_dir))
    assert limits['global'] == 5e9
    assert limits['local'] == 1e6
    assert limits['constant'] == 64e3
    assert limits['alloc'] == 1e9
