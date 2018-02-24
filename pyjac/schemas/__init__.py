# system
from os.path import abspath, dirname, join
import re
import logging

# external
import six
import yamale
from yamale.validators import DefaultValidators, Validator, Integer, String, Map

# internal
from ..utils import func_logger, langs, can_vectorize_lang, stringify_args, listify

# define path to schemas
schema_dir = abspath(dirname(__file__))


def get_list_validator(tagname, validlist):
    class ListValidator(Validator):
        tag = tagname

        @func_logger(name=tagname)
        def _is_valid(self, value):
            value = listify(value)
            badvals = [x for x in value if x not in validlist]
            if badvals:
                logger = logging.getLogger(__name__)
                logger.error('Value(s) {} not in allowed mapping {}. '
                             'Allowed values: {}'.format(
                                badvals, tagname, stringify_args(validlist)))
                return False
            return True

    return ListValidator()


class Pow2Validator(Integer):
    tag = 'pow2'

    @func_logger(name=tag)
    def _is_valid(self, value):
        rv = super(Pow2Validator, self)._is_valid(value) and \
            (value != 0) and ((value & (value - 1)) == 0)
        if not rv:
            logger = logging.getLogger(__name__)
            logger.error('Value {} not a power of two.'.format(
                          value))
        return rv


class BytesValidator(String):
    tag = 'bytes'

    @func_logger(name=tag)
    def _is_valid(self, value):
        logger = logging.getLogger(__name__)
        # first split value
        match = re.search(r'^\s*(\d+)\s*([mMkKgG]?[bB])\s*$')
        if not match:
            logger.error('String {} specified for type "bytes" could '
                         'not be parsed.  Expected format example: 10 GB'.format(
                            value))
            return False

        size, unit = match.groups()[1:]
        size = int(size)
        if size < 0:
            logger.error('Size {} specified for type "bytes" less than zero'.format(
                         value))
            return False
        unit = unit.lower()
        if unit == 'b':
            unit = 1
        elif unit == 'kb':
            unit = 1e3
        elif unit == 'mb':
            unit = 1e6
        elif unit == 'gb':
            unit = 1e9

        return unit * size


class OverrideValidator(Map):
    tag = 'override'

    @func_logger(name=tag)
    def _is_valid(self, value):
        from ..tests.test_utils.get_test_matrix import (
            allowed_overrides, allowed_override_keys)
        logger = logging.getLogger(__name__)

        if not isinstance(value, dict):
            logger.debug('Override improperly specified: {}'.format(value))
            return False

        # next check for valid keys
        if not all(k in allowed_override_keys for k in value.keys()):
            logger.error('Invalid override key specified: {}'.format(
                next(k for k in value.keys() if k not in allowed_override_keys)))
            return False

        # next, check that all subkeys are allowed
        for key in value.keys():
            for k, v in value[key]:
                # check that the override type is valid
                if k not in allowed_overrides:
                    logger.error('Invalid override {} specified for key {}. '
                                 'Allowed values are: {}'.format(
                                    k, key, ', '.join(allowed_overrides.keys())))
                    return False
                override, values = allowed_overrides[k]
                v = listify(v)
                # if the 'values' is a type,
                if isinstance(values, type):
                    bad = next((vi for vi in v if not isinstance(vi, values)), None)
                    if bad is not None:
                        logger.error('Invalid value type specified for key {}. '
                                     'Allowed values type is: {}'.format(
                                        '.'.join([key, k]), str(value)))
                    # and convert to type
                    value[key][k] = [values(vi) for vi in v]
                else:
                    bad = next((vi for vi in v if not values), None)
                    if bad is not None:
                        logger.error('Invalid value type specified for key {}. '
                                     'Allowed values are: {}'.format(
                                        '.'.join([key, k]), ', '.join(
                                            str(vi) for vi in value)))
        return value


def get_validators():
    validators = DefaultValidators.copy()  # This is a dictionary

    lang = get_list_validator('lang', langs)
    validators[lang.tag] = lang

    can_vectorize = get_list_validator('can_vectorize', can_vectorize_lang)
    validators[can_vectorize.tag] = can_vectorize

    vectype = get_list_validator('vectype', ['wide', 'deep', 'par'])
    validators[vectype.tag] = vectype

    pow2 = Pow2Validator()
    validators[pow2.tag] = pow2

    byte = BytesValidator()
    validators[byte.tag] = byte

    override = OverrideValidator()
    validators[override.tag] = override

    return validators


def __prefixify(file, dirname=schema_dir):
    if dirname not in file:
        return join(dirname, file)
    return file


def build_schema(schema, validators=get_validators(), includes=[]):
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

    Returns
    -------
        schema: :class:`yamale.Schema`
            The constructed schema
    """

    # add common
    includes = [__prefixify(x) for x in listify(includes)]
    includes.append(__prefixify('common_schema.yaml'))

    # ensure schema is properly path'd
    schema = __prefixify(schema)

    # build base schema
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

    return schema


def validate(schema, source):
    """
    Validates the passed source file from the pre-built schema, and returns the
    result

    Parameters
    ----------
    schema: :class:`yamale.Schema`
        The built schema
    source: str
        Path to the source file

    Returns
    -------
    data: dict
        The validated data
    """

    # make data
    source = yamale.make_data(source)
    # and validate
    return yamale.validate(schema, source)


def build_and_validate(schema, source, validators=get_validators(), includes=[]):
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

    Returns
    -------
    data: dict
        The validated data
    """
    schema = build_schema(schema, validators=validators, includes=includes)
    return validate(schema, source)[0]
