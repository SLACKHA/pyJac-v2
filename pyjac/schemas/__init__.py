import os
from yamale.validators import DefaultValidators, Validator, Integer, String
from ..utils import func_logger, langs, can_vectorize_lang, stringify_args
import re
import logging

# define path to schemas
schema_dir = os.path.abspath(os.path.dirname(__file__))


def get_list_validator(tagname, validlist):
    class ListValidator(Validator):
        tag = tagname

        @func_logger(name=tagname)
        def _is_valid(self, value):
            if value not in validlist:
                logger = logging.getLogger(__name__)
                logger.error('Value {} not in allowed mapping {}.'
                             'Allowed values: {}'.format(
                                value, tagname, stringify_args(validlist)))
            return value in validlist

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

    return validators
