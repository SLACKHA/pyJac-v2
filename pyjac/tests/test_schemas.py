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
              should_fail=False, includes=[]):
    def __prefixify(file, dirname):
        if dirname not in file:
            return join(dirname, file)
        return file

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
        schema = yamale.make_schema(schema, validators=validators)
        # next, go through additional includes and add to schema
        for inc in includes:
            # parse other schema
            inc = yamale.readers.parse_file(inc, 'PyYAML')
            for i in inc:
                for k, v in six.iteritems(i):
                    try:
                        schema[k]
                    except KeyError:
                        # new include
                        schema.add_include({k: v})
        # make data
        source = yamale.make_data(source)
        # and validate
        return yamale.validate(schema, source) is not None

    assert _internal(source, schema, validators, includes)


def test_platform_schema_validation():
    runschema('platform_schema.yaml', 'test_platforms.yaml')
    runschema('platform_schema.yaml', 'codegen_platform.yaml')


def test_mechanism_schema():
    runschema('test_matrix_schema.yaml', 'test_matrix.yaml',
              includes=['platform_schema.yaml'])
