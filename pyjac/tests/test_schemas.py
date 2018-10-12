"""
A unit tester that loads the schemas packaged with :mod:`pyJac` and validates
the given example specifications against them.
"""

# system
from os.path import isfile, join
from collections import OrderedDict
from tempfile import NamedTemporaryFile

# external
import six
import cantera as ct
from nose.tools import assert_raises
from pytools.py_codegen import remove_common_indentation

# internal
from pyjac.core.enum_types import KernelType, JacobianFormat, JacobianType
from pyjac.utils import enum_to_string, listify, is_iterable
from pyjac.tests.test_utils import xfail
from pyjac.tests import script_dir as test_mech_dir
from pyjac.tests.test_utils.get_test_matrix import load_models, load_platforms, \
    load_tests, get_test_matrix, num_cores_default
from pyjac.examples import examples_dir
from pyjac.schemas import schema_dir, __prefixify, build_and_validate
from pyjac.core.exceptions import OverrideCollisionException, \
    DuplicateTestException, InvalidOverrideException, \
    InvalidInputSpecificationException, ValidationError
from pyjac.loopy_utils.loopy_utils import load_platform
from pyjac.kernel_utils.memory_limits import memory_limits, memory_type

current_test_langs = ['c', 'opencl']
"""
:attr:`current_test_langs`
    we need to define the languages to test here separately from the
    :func:`get_test_langs` as these tests depend on the entire platform / test
    matrix files
"""


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
    assert amd['use_atomic_doubles'] is True
    assert amd['use_atomic_ints'] is True
    assert amd['is_simd'] == [True]

    def __fuzz_equal(arr):
        return arr == [2, 4, None] or arr == [2, 4]
    assert __fuzz_equal(amd['width'])
    assert __fuzz_equal(amd['depth'])
    assert amd['depth'] == amd['width']

    # openmp
    openmp = next(p for p in platforms if 'openmp' in p['platform'].lower())
    assert openmp['lang'] == 'c'
    assert openmp['width'] is None
    assert openmp['depth'] is None

    # nvidia
    nvidia = next(p for p in platforms if 'nvidia' in p['platform'].lower())
    assert nvidia['lang'] == 'opencl'
    assert nvidia['width'] == [64, 128, 256, None]
    assert nvidia['depth'] is None
    assert nvidia['use_atomic_doubles'] is False
    assert nvidia['use_atomic_ints'] is True

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
    assert platform.use_atomic_doubles is True
    assert platform.is_simd


def test_bad_simd_specification_in_codegen():
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        platform:
            name: portable
            lang: opencl
            # deep vectorization
            depth: 4
            is_simd: True
        """))
        file.seek(0)

        with assert_raises(ValidationError):
            build_and_validate('codegen_platform.yaml', file.name)


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

    __test_limit(KernelType.species_rates, 10000000)
    __test_limit([KernelType.jacobian, JacobianFormat.sparse], 100000)
    __test_limit([KernelType.jacobian, JacobianFormat.full], 1000)

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
    assert intel['use_atomic_doubles'] is False
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
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: openmp
            lang: c
        test-list:
          - test-type: performance
            eval-type: jacobian
          - test-type: performance
            eval-type: both
        """))
        file.seek(0)

        with assert_raises(DuplicateTestException):
            tests = build_and_validate('test_matrix_schema.yaml', file.name)
            load_tests(tests, file.name)

    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: openmp
            lang: c
        test-list:
          - test-type: performance
            eval-type: jacobian
            exact:
                sparse:
                    num_cores: [1]
                full:
                    num_cores: [1]
        """))
        file.seek(0)

        tests = build_and_validate('test_matrix_schema.yaml', file.name)
        load_tests(tests, file.name)

    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: openmp
            lang: c
        test-list:
          - test-type: performance
            eval-type: jacobian
            exact:
                both:
                    num_cores: [1]
                full:
                    num_cores: [1]
        """))
        file.seek(0)

        with assert_raises(OverrideCollisionException):
            tests = build_and_validate('test_matrix_schema.yaml', file.name)
            load_tests(tests, file.name)


def test_load_tests():
    # load tests doesn't do any processing other than collision / duplicate
    # checking, hence we just check we get the right number of tests
    tests = load_tests(__get_test_matrix(), 'test_matrix_schema.yaml')
    assert len(tests) == 3


def test_override():
    # test the base override schema
    with NamedTemporaryFile(mode='w', suffix='.yaml') as file:
        file.write(remove_common_indentation(
            """
            override:
                num_cores: [1]
                order: ['F']
                gpuorder: ['C']
                conp: ['conp']
                width: [2, 4]
                gpuwidth: [128]
                models: ['C2H4']
            """))
        file.flush()
        file.seek(0)
        data = build_and_validate('common_schema.yaml', file.name)['override']
    assert data['num_cores'] == [1]
    assert data['order'] == ['F']
    assert data['gpuorder'] == ['C']
    assert data['conp'] == ['conp']
    assert data['width'] == [2, 4]
    assert data['gpuwidth'] == [128]
    assert data['models'] == ['C2H4']

    # now test embedded overrides
    with NamedTemporaryFile(mode='w', suffix='.yaml') as file:
        file.write(remove_common_indentation(
            """
            model-list:
              - name: CH4
                mech: gri30.cti
                path:
            platform-list:
              - lang: c
                name: openmp
            test-list:
              - test-type: performance
                # limit to intel
                platforms: [intel]
                eval-type: jacobian
                exact:
                    both:
                        num_cores: [1]
                        order: [F]
                        gpuorder: [C]
                        conp: [conp]
                        depth: [2, 4]
                        gpudepth: [128]
                        models: [C2H4]
            """))
        file.flush()
        file.seek(0)
        data = build_and_validate('test_matrix_schema.yaml', file.name,
                                  update=True)

    data = data['test-list'][0]['exact']['both']
    assert data['num_cores'] == [1]
    assert data['order'] == ['F']
    assert data['gpuorder'] == ['C']
    assert data['conp'] == ['conp']
    assert data['depth'] == [2, 4]
    assert data['gpudepth'] == [128]
    assert data['models'] == ['C2H4']


def test_get_test_matrix():
    # test that the example test matrix is specified correctly
    def update(want, state, key, seen):
        if state[key] not in seen[key]:
            if six.callable(want[key]):
                want[key](state, want, seen)
            else:
                if is_iterable(state[key]):
                    for k in state[key]:
                        if k not in seen[key]:
                            want[key].remove(k)
                            seen[key].add(k)
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

    import logging
    logger = logging.getLogger(__name__)
    logger.debug('loading test matrix schema')
    test_matrix = __prefixify('test_matrix.yaml', examples_dir)

    # get the species validation test
    logger.debug('loading test matrix from file')
    _, loop, max_vec_width = get_test_matrix('.', KernelType.species_rates,
                                             test_matrix, True,
                                             langs=current_test_langs,
                                             raise_on_missing=True)
    assert max_vec_width == 8
    from collections import defaultdict

    def width_check(state, want, seen):
        if state['lang'] == 'c':
            assert state['width'] is None
            assert state['depth'] is None
        else:
            seen['width'].add(state['width'])

    def check_final_widths(seen):
        return not (set(seen['width']) - set([None, 2, 4, 8]))

    # check we have reasonable values
    base = {'platform': ['intel', 'openmp'],
            'width': width_check,
            'conp': [True, False],
            'order': ['C', 'F'],
            'num_cores': num_cores_default()[0]}
    logger.debug('check 1')
    run(base, loop, final_checks=check_final_widths)

    # repeat for jacobian
    logger.debug('loading test matrix from file [1]')
    _, loop, _ = get_test_matrix('.', KernelType.jacobian,
                                 test_matrix, True,
                                 langs=current_test_langs,
                                 raise_on_missing=True)
    jacbase = base.copy()
    jacbase.update({
        'jac_format': [enum_to_string(JacobianFormat.sparse),
                       enum_to_string(JacobianFormat.full)],
        'jac_type': [enum_to_string(JacobianType.exact)],
        'use_atomic_doubles': [True, False]})  # true for OpenMP by default
    logger.debug('check 2')
    run(jacbase, loop, final_checks=check_final_widths)

    # next, do species performance
    logger.debug('loading test matrix from file [2]')
    _, loop, _ = get_test_matrix('.', KernelType.species_rates,
                                 test_matrix, False,
                                 langs=current_test_langs,
                                 raise_on_missing=True)
    want = base.copy()
    want.update({'order': ['F']})
    logger.debug('check 3')
    run(want, loop, final_checks=check_final_widths)

    # and finally, the Jacobian performance
    logger.debug('loading test matrix from file [4]')
    _, loop, _ = get_test_matrix('.', KernelType.jacobian,
                                 test_matrix, False,
                                 langs=current_test_langs,
                                 raise_on_missing=True)
    want = jacbase.copy()
    # no more openmp
    want.update({'use_atomic_doubles': [False]})

    def update_jactype(state, want, seen):
        if state['jac_type'] == enum_to_string(JacobianType.finite_difference):
            assert state['num_cores'] == 1
            assert state['width'] is None
            assert state['depth'] is None
            assert state['order'] == 'C'
            assert state['conp'] is True
        else:
            assert state['width'] in [4, None]

    want.update({'platform': ['intel'],
                 'jac_type': update_jactype})

    def check_final_widths(seen):
        return len(seen['width'] - set([4, None])) == 0
    logger.debug('check 5')
    run(want, loop, final_checks=check_final_widths)

    # test gpu vs cpu specs
    logger.debug('writing temp file')
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
          - name: intel
            lang: opencl
            width: [4]
        test-list:
          - test-type: performance
            eval-type: jacobian
            exact:
                sparse:
                    gpuwidth: [64]
                    order: ['F']
                full:
                    width: [2]
                    gpuorder: ['C']
        """))
        file.flush()

        logger.debug('loading test matrix from file [5]')
        _, loop, _ = get_test_matrix('.', KernelType.jacobian,
                                     file.name, False,
                                     langs=current_test_langs,
                                     raise_on_missing=True)

    from pyjac.utils import platform_is_gpu

    def sparsetest(state, want, seen):
        if state['jac_type'] == enum_to_string(JacobianType.exact):
            if state['jac_format'] == enum_to_string(JacobianFormat.sparse):
                if platform_is_gpu(state['platform']):
                    assert state['width'] in [64, None]
                else:
                    assert state['width'] in [4, None]
                    assert state['order'] == 'F'
            else:
                if platform_is_gpu(state['platform']):
                    assert state['order'] == 'C'
                    assert state['width'] in [128, None]
                else:
                    assert state['width'] in [2, None]

    want = {'jac_format': sparsetest}
    logger.debug('check 6')
    run(want, loop)

    # test model override
    logger.debug('writing temp file 2')
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
          - name: H2
            path:
            mech: h2o2.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
        test-list:
          - test-type: performance
            eval-type: jacobian
            finite_difference:
                both:
                    models: ['H2']
        """))
        file.flush()

        logger.debug('loading test matrix from file [6]')
        _, loop, _ = get_test_matrix('.', KernelType.jacobian,
                                     file.name, False,
                                     langs=current_test_langs,
                                     raise_on_missing=True)

    def modeltest(state, want, seen):
        if state['jac_type'] == enum_to_string(JacobianType.finite_difference):
            assert set(state['models']) == set(['H2'])
        else:
            assert set(state['models']) == set(['H2', 'CH4'])

    want = {'models': modeltest}
    logger.debug('check 7')
    run(want, loop)

    # finally test bad model spec
    logger.debug('writing temp file 3')
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
          - name: H2
            path:
            mech: h2o2.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
        test-list:
          - test-type: performance
            eval-type: jacobian
            finite_difference:
                both:
                    models: ['BAD']
        """))
        file.flush()

        logger.debug('loading test matrix from file [7]')
        with assert_raises(InvalidOverrideException):
            get_test_matrix('.', KernelType.jacobian,
                            file.name, False,
                            langs=current_test_langs,
                            raise_on_missing=True)

    # test gpu vectype specification
    logger.debug('writing temp file 4')
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
          - name: H2
            path:
            mech: h2o2.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
          - name: openmp
            lang: c
        test-list:
          - test-type: performance
            eval-type: jacobian
            finite_difference:
                both:
                    width: []
                    depth: []
                    gpuwidth: [128]
        """))
        file.flush()

        logger.debug('loading test matrix from file [8]')
        _, loop, _ = get_test_matrix('.', KernelType.jacobian,
                                     file.name, False,
                                     langs=current_test_langs,
                                     raise_on_missing=True)

    def modeltest(state, want, seen):
        if state['jac_type'] == enum_to_string(JacobianType.finite_difference):
            if state['platform'] == 'openmp':
                assert not bool(state['width'])
            else:
                assert state['width'] == 128

    want = {'models': modeltest}
    logger.debug('check 8')
    run(want, loop)

    # test that source terms evaluations don't inherit exact jacobian overrides
    logger.debug('writing temp file 5')
    with NamedTemporaryFile(mode='w', suffix='.yaml') as file:
        file.write(remove_common_indentation(
            """
            model-list:
              - name: CH4
                mech: gri30.cti
                path:
            platform-list:
              - lang: c
                name: openmp
            test-list:
              - test-type: performance
                # limit to intel
                platforms: [openmp]
                eval-type: both
                exact:
                    both:
                        num_cores: [1]
                        order: [F]
                        gpuorder: [C]
                        conp: [conp]
                        width: [2, 4]
                        gpuwidth: [128]
                        models: []
            """))
        file.flush()
        logger.debug('loading test matrix from file [9]')
        _, loop, _ = get_test_matrix('.', KernelType.species_rates,
                                     file.name, False,
                                     langs=current_test_langs,
                                     raise_on_missing=True)

    want = {'platform': ['openmp'],
            'width': [None],
            'conp': [True, False],
            'order': ['C', 'F'],
            'models': ['CH4'],
            'num_cores': num_cores_default()[0]}
    logger.debug('check 9')
    run(want, loop)


def test_load_memory_limits():
    # create dummy loopy opts
    def __dummy_opts(name):
        return type('', (object,), {'platform_name': name, 'lang': '',
                                    'order': ''})
    # test codegen
    limits = memory_limits.get_limits(
        __dummy_opts('portable'), [],
        __prefixify('codegen_platform.yaml', examples_dir))
    assert limits.limits[memory_type.m_global] == 1e9
    assert limits.limits[memory_type.m_constant] == 1e6
    assert limits.limits[memory_type.m_alloc] == 100e6
    assert len(limits.limits) == 3

    # test test_matrix -- this includes per-platform specification
    limits = memory_limits.get_limits(
        __dummy_opts('intel'), [],
        __prefixify('test_matrix.yaml', examples_dir))
    assert limits.limits[memory_type.m_global] == 5e9
    assert limits.limits[memory_type.m_constant] == 64e3
    assert limits.limits[memory_type.m_local] == 1e6
    assert limits.limits[memory_type.m_alloc] == 1e9
    assert len(limits.limits) == 4
    limits = memory_limits.get_limits(
        __dummy_opts('openmp'), [],
        __prefixify('test_matrix.yaml', examples_dir))
    assert limits.limits[memory_type.m_global] == 5e10
    assert len(limits.limits) == 1

    # and finally, test bad specifications
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
        memory-limits:
          - global: 5 Gb
            platforms: [nvidia]
          - global: 10 Gb
            platforms: [nvidia]
        test-list:
          - test-type: performance
            eval-type: jacobian
        """))
        file.flush()

        with assert_raises(InvalidInputSpecificationException):
            limits = memory_limits.get_limits(
                __dummy_opts('nvidia'), [],
                file.name)

    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
        memory-limits:
          - global: 5 Gb
          - global: 10 Gb
        test-list:
          - test-type: performance
            eval-type: jacobian
        """))
        file.flush()

        with assert_raises(InvalidInputSpecificationException):
            limits = memory_limits.get_limits(
                __dummy_opts('nvidia'), [],
                file.name)

    # try with file w/o limits
    with NamedTemporaryFile('w', suffix='.yaml') as file:
        file.write(remove_common_indentation("""
        model-list:
          - name: CH4
            path:
            mech: gri30.cti
        platform-list:
          - name: nvidia
            lang: opencl
            width: [128]
        test-list:
          - test-type: performance
            eval-type: jacobian
        """))
        file.flush()

        limits = memory_limits.get_limits(__dummy_opts('nvidia'), [], file.name)
        assert limits.limits == {}
