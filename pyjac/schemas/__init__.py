import os
from yamale.validators import DefaultValidators, Validator
from ..utils import func_logger, langs, can_vectorize_lang

# define path to schemas
schema_dir = os.path.abspath(os.path.dirname(__file__))


def get_map_validator(tag, validmap):
    class MapValidator(Validator):
        @func_logger(name=tag)
        def _is_valid(self, value):
            return tag in validmap and validmap[tag]

    return MapValidator()


def get_list_validator(tag, validlist):
    class ListValidator(Validator):
        @func_logger(name=tag)
        def _is_valid(self, value):
            return tag in validlist

    return ListValidator()


def get_validators():
    validators = DefaultValidators.copy()  # This is a dictionary

    lang = get_list_validator('lang', langs)
    validators[lang.tag] = lang

    can_vectorize = get_list_validator('can_vectorize', can_vectorize_lang)
    validators[can_vectorize.tag] = can_vectorize

    vectype = get_list_validator('vectype', ['wide', 'deep', 'par'])
    validators[vectype.tag] = vectype

    return validators
