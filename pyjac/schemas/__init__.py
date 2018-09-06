# system
from os.path import abspath, dirname, join, isfile
import re
import logging

# external
import six
import yaml
from cerberus import Validator

# internal
from pyjac.utils import langs, stringify_args, listify
from pyjac.core.exceptions import ValidationError, validation_error_to_string

# define path to schemas
schema_dir = abspath(dirname(__file__))


def parse_bytestr(self, value):
    """
    Helper method that let's us "coerce" outside of cerberus, as that doesn't like
    that...

    Returns
    -------
    bytes: int
        The value converted to bytes
    """

    match = re.search(r'^\s*(\d+)\s*([mMkKgG]?[bB])\s*$', value)
    if not match:
        self._error('String {} specified for type "bytes" could '
                    'not be parsed.  Expected format example: 10 GB'.format(
                        value))

    size, unit = match.groups()
    size = int(size)
    if size < 0:
        self._error('Size {} specified for type "bytes" less than zero'.format(
                    value))

    unit = unit.lower()
    if unit == 'b':
        unit = 1
    elif unit == 'kb':
        unit = 1e3
    elif unit == 'mb':
        unit = 1e6
    elif unit == 'gb':
        unit = 1e9
    else:
        self._error('Unknown unit type {}. Allowed types are (case-insensative)'
                    'B, KB, MB, GB.'.format(unit))

    return int(unit * size)


class CustomValidator(Validator):
    def __internal_validator(self, field, valuelist, valid, message, necessary=True):
        valuelist = listify(valuelist)
        if six.callable(valid):
            badvals = [x for x in valuelist if not valid(x)]
        else:
            badvals = [x for x in valuelist if x not in valid]
        if badvals and necessary:
            args = (badvals,)
            if not six.callable(valid):
                args = (badvals, valid)

            self._error(field, message.format(
                *tuple(stringify_args(x) for x in args)))

    def _validate_isvecsize(self, isvecsize, field, value):
        """ Test that the specified value is a proper vector size

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        # TODO: implement per-platform vecsize checks
        # valid values include any power of two (or 3 for OpenCL)
        self.__internal_validator(
            field, value, lambda x: x == 3 or (x & (x - 1)) == 0,
            'Value(s) {} not valid vector size.'
            'Must be a power of two (or the value three, for OpenCL)')

    def _validate_isvectype(self, isvectype, field, value):
        """ Test that the specified value is a proper vector width

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """
        allowed = ['par', 'wide', 'deep']
        self.__internal_validator(
            field, value, allowed,
            'Value(s) {} not valid vector vectorization types.'
            'Allowed values are: {}.')

    def _validate_isvalidlang(self, isvalidlang, field, value):
        """ Test that the specified value is a proper vector width

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """

        self.__internal_validator(
            field, value, langs,
            'Value(s) {} not valid language.'
            'Allowed values are: {}.')
        return True

    def _validate_type_bytestr(self, value):
        """
        Enables validation for `bytestr` schema attribute.
        :param value: field value.
        """

        parse_bytestr(self, value)
        return True

    def _validate_is_platform(self, is_platform, field, value):
        """ Test that the specified value is an ok platform

        The rule's arguments are validated against this schema:
        {'type': 'boolean'}
        """

        # todo implement
        raise NotImplementedError


def __prefixify(file, dirname=schema_dir):
    if dirname not in file:
        return join(dirname, file)
    return file


def build_schema(schema, includes=['common_schema.yaml'],
                 validatorclass=CustomValidator, allow_unknown=False,
                 update=False):
    """
    Creates a schema / parses a schema and adds the additonal given includes

    Parameters
    ----------
    schema: str
        The schema to parse
    validators: list of :class:`Validator` [:func:`get_validators()`]
        The validators to use, by defaut use the output of get_validators()
    includes: list of str
        Additional schema to use for includes
    validatorclass: :class:`Validator` [:class:`CustomValidator`]
        The type of validator to build
    allow_unknown: bool [False]
        Allow unknown keys
    update: bool [False]
        Allow partial specification of data, useful for testing specific parts
        of schemas

    Returns
    -------
        validator: :class:`Validator`
            The constructed validator
    """

    def __recursive_replace(root, schemaname, schema):
        for key, value in six.iteritems(root):
            if key == 'schema' and schemaname == value:
                if 'type' in schema and schema['type'] == root['type']:
                    root[key] = schema['schema'].copy()
                else:
                    root[key] = schema.copy()
            elif isinstance(value, dict):
                root[key] = __recursive_replace(value, schemaname, schema)
        return root

    with open(__prefixify(schema), 'r') as file:
        schema = yaml.load(file)

    for include in includes:
        include = __prefixify(include)
        if not isfile(include):
            raise IOError('Schema file {} does not exist.'.format(include))
        with open(include, 'r') as file:
            common = yaml.load(file)

        # rather than use the schema registry, it's safer to directly replace
        for key, value in six.iteritems(common):
            schema = __recursive_replace(schema, key, value)

    return validatorclass(schema, allow_unknown=allow_unknown, update=update)


def validate(validator, source, filename=''):
    """
    Validates the passed source file from the pre-built schema, and returns the
    result

    Parameters
    ----------
    validator: :class:`CustomValidator`
        The built validator
    source: str
        Path to the source file

    Returns
    -------
    data: dict
        The validated data
    """

    # make data
    with open(source, 'r') as file:
        sourcedict = yaml.load(file)
    # and validate
    if not validator.validate(sourcedict):
        logger = logging.getLogger(__name__)
        logger.error(validation_error_to_string(validator.errors))
        raise ValidationError(source, filename)

    return validator.validated(sourcedict)


def build_and_validate(schema, source, validator=CustomValidator, includes=[],
                       allow_unknown=False, update=False):
    """
    Builds schema from file, validates source from file and returns results.
    Convience method for :func:`build_schema` and :func:`validate`

    Parameters
    ----------
    Parameters
    ----------
    schema: str
        The schema to parse
    source: str
        Path to the source file
    validators: list of :class:`Validator` [:func:`get_validators()`]
        The validators to use, by defaut use the output of get_validators()
    includes: list of str
        Additional schema to use for includes
    allow_unknown: bool [False]
        Allow unknown keys
    update: bool [False]
        Allow partial specification of data, useful for testing specific parts
        of schemas

    Returns
    -------
    data: dict
        The validated data
    """
    includes = listify(includes)
    if 'common_schema.yaml' != source:
        includes.append('common_schema.yaml')
    built = build_schema(schema, validatorclass=validator, includes=includes,
                         allow_unknown=allow_unknown, update=update)
    return validate(built, source, filename=schema)
