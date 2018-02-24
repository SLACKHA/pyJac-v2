"""

A unit tester that loads the schemas packaged with :mod:`pyJac` and validates
the given example specifications against them.

"""

from os.path import isfile
import six

# internal
from ..utils import func_logger
from .test_utils import xfail
from ..examples import examples_dir
from ..schemas import schema_dir, get_validators, build_schema, validate, __prefixify


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


def test_test_platform_schema():
    runschema('test_platform_schema.yaml', 'test_platforms.yaml')


def test_codegen_platform_schema():
    runschema('codegen_platform.yaml', 'codegen_platform.yaml')


def test_matrix_schema():
    runschema('test_matrix_schema.yaml', 'test_matrix.yaml',
              includes=['platform_schema.yaml'])
