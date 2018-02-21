"""

A unit tester that loads the schemas packaged with :mod:`pyJac` and validates
the given example specifications against them.

"""

from os.path import join, isfile
import six
import yamale

# internal
from ..utils import func_logger
from .test_utils import xfail
from ..examples import examples_dir
from ..schemas import schema_dir, get_validators


@func_logger
def runschema(schema, source, validators=get_validators(),
              should_fail=False):
    def __prefixify(file, dirname):
        if dirname not in file:
            return join(dirname, file)
        return file

    # check source / schema, and prepend prefixes
    assert isinstance(schema, six.string_types), 'Schema file should be string'
    schema = __prefixify(schema, schema_dir)
    assert isinstance(source, six.string_types), 'Source file should be string'
    source = __prefixify(source, examples_dir)

    # define inner tester
    @xfail(should_fail)
    def _internal(source, schema, validators):
        for f in [source, schema]:
            assert isfile(f), 'File {} not found.'.format(f)
        # make schema
        schema = yamale.make_schema(schema, validators=validators)
        # make data
        source = yamale.make_data(source)
        # and validate
        return yamale.validate(schema, source) is not None

    assert _internal(source, schema, validators)


def test_testplatform_schema():
    runschema('platform_schema.yaml', 'test_platforms.yaml')


def test_platform_schema():
    runschema('platform_schema.yaml', 'codegen_platform.yaml')
