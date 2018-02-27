# test various functions in the utils function, or elsewhere

from parameterized import parameterized

from pyjac.utils import enum_to_string, listify
from pyjac.loopy_utils.loopy_utils import JacobianType, JacobianFormat
from pyjac.libgen import build_type


@parameterized([(JacobianType.exact, 'exact'),
                (JacobianType.approximate, 'approximate'),
                (JacobianType.finite_difference, 'finite_difference'),
                (JacobianFormat.sparse, 'sparse'),
                (JacobianFormat.full, 'full'),
                (build_type.chem_utils, 'chem_utils'),
                (build_type.species_rates, 'species_rates'),
                (build_type.jacobian, 'jacobian')])
def test_enum_to_string(enum, string):
    assert enum_to_string(enum) == string


@parameterized([('a', ['a']),
                ([1, 2, 3], [1, 2, 3]),
                ((1, 2, 'a'), [1, 2, 'a']),
                (3, [3])])
def test_listify(value, expected):
    assert listify(value) == expected
