"""
A unit tester that loads the schemas packaged with :mod:`pyJac` and validates
the given example specifications against them.
"""

from os.path import isfile, join
import six
import cantera as ct

# internal
from ..libgen.libgen import build_type
from ..loopy_utils.loopy_utils import JacobianFormat
from ..utils import func_logger, enum_to_string, listify
from .test_utils import xfail
from . import script_dir as test_mech_dir
from .test_utils.get_test_matrix import load_models
from ..examples import examples_dir
from ..schemas import schema_dir, get_validators, build_schema, validate, \
    __prefixify, build_and_validate


@func_logger
def runschema(schema, source, validators=get_validators(),
              should_fail=False, includes=[]):

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
    def _internal(source, schema, validators, includes):
        # make schema
        schema = build_schema(schema, includes=includes)
        return validate(schema, source) is not None

    assert _internal(source, schema, validators, includes)


def test_test_platform_schema_specification():
    runschema('test_platform_schema.yaml', 'test_platforms.yaml')


def test_codegen_platform_schema_specification():
    runschema('codegen_platform.yaml', 'codegen_platform.yaml')


def test_matrix_schema_specification():
    runschema('test_matrix_schema.yaml', 'test_matrix.yaml',
              includes=['platform_schema.yaml'])


def test_parse_models():
    matrix = build_and_validate('test_matrix_schema.yaml', __prefixify(
        'test_matrix.yaml', examples_dir), includes=['platform_schema.yaml'])
    models = load_models('', matrix)

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
