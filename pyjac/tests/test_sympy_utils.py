# local imports
from ..sympy_utils import sympy_utils as sp_util
from ..sympy_utils import sympy_addons as sp_add

# modules
import sympy as sp


def test_symlist():
    # make symbols
    A = sp_add.MyIndexedBase('A')
    B = sp_add.MyIndexedBase('B')
    i = sp.Idx('i')

    expr = A[i] + B[i]

    A_dummy = sp.Symbol('A[i]')
    B_dummy = sp.Symbol('B[i]')

    free_syms = {A_dummy, B_dummy}
    expr_san = sp_util.sanitize(expr)
    assert expr_san.free_symbols.intersection(free_syms) == free_syms

    expr_san = sp_util.sanitize(expr, symlist={
        'A[i]': A_dummy,
        'B[i]': B_dummy
    })
    assert expr_san.free_symbols.intersection(free_syms) == free_syms


def test_subs():
    # make symbols
    A = sp_add.MyIndexedBase('A')
    B = sp_add.MyIndexedBase('B')
    i = sp.Idx('i')

    expr = A[i] + B[i]

    A_dummy = sp.Symbol('A[i]')
    B_dummy = sp.Symbol('B[i]')
    C_dummy = sp.Symbol('C[i]')

    expr_san = sp_util.sanitize(expr,
                                subs={
                                    A_dummy + B_dummy: C_dummy
                                })
    assert expr_san == C_dummy
