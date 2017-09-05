from ..core.instruction_creator import array_splitter
import numpy as np
import loopy as lp


class dummy_loopy_opts(object):
    def __init__(self, depth=None, width=None, order='C'):
        self.depth = depth
        self.width = width
        self.order = order


def test_npy_array_splitter_c_wide():
    # create opts
    opts = dummy_loopy_opts(width=8, order='C')

    # create array split
    asplit = array_splitter(opts)

    def _create(dim):
        # create some test arrays
        arr = np.zeros((dim, dim))
        for i in range(dim):
            arr[i, :] = i
        return arr

    arr, = asplit.split_numpy_arrays(_create(10))

    # check dim
    assert arr.shape == (2, 10, 8)
    # check first
    assert np.array_equal(arr[0, 0, :], np.arange(0, 8))
    # and check after the first 8 run out
    test = np.zeros(8)
    test[0:2] = np.arange(8, 10)
    assert np.array_equal(arr[1, 0, :], test)

    # now test with evenly sized
    arr, = asplit.split_numpy_arrays(_create(16))
    assert arr.shape == (2, 16, 8)
    assert np.array_equal(arr[0, 0, :], np.arange(0, 8))
    assert np.array_equal(arr[1, 0, :], np.arange(8, 16))


def test_npy_array_splitter_f_deep():
    # create opts
    opts = dummy_loopy_opts(depth=8, order='F')

    # create array split
    asplit = array_splitter(opts)

    # create some test arrays
    arr = np.zeros((10, 10), order='F')
    for i in range(10):
        arr[:, i] = i
    arr, = asplit.split_numpy_arrays(arr)

    # check dim
    assert arr.shape == (8, 10, 2)
    # check first
    assert np.array_equal(arr[:, 0, 0], np.arange(0, 8))
    # and check after the first 8 run out
    test = np.zeros(8)
    test[0:2] = np.arange(8, 10)
    assert np.array_equal(arr[:, 0, 1], test)

    # now test with evenly sized
    arr = np.zeros((16, 16), order='F')
    for i in range(16):
        arr[:, i] = i
    arr, = asplit.split_numpy_arrays(arr)
    assert arr.shape == (8, 16, 2)
    assert np.array_equal(arr[:, 0, 0], np.arange(0, 8))
    assert np.array_equal(arr[:, 0, 1], np.arange(8, 16))


def test_lpy_array_splitter_c_wide():
    from pymbolic.primitives import Subscript, Variable

    # create opts
    opts = dummy_loopy_opts(width=8, order='C')

    # create array split
    asplit = array_splitter(opts)

    def _create():
        # create a test kernel
        arg1 = lp.GlobalArg('a1', shape=(10, 10))
        arg2 = lp.GlobalArg('a2', shape=(16, 16))

        return lp.make_kernel(
            '{[i]: 0 <= i < 10}',
            """
                a1[i, 0] = 0 {id=a1}
                a2[i, 0] = 0 {id=a2}
            """,
            [arg1, arg2],
            target=lp.CTarget())

    k = lp.split_iname(_create(), 'i', 8)
    k = asplit._split_loopy_arrays(k)

    # test that it runs
    k()

    # check dim and shape
    a1 = next(x for x in k.args if x.name == 'a1')
    assert a1.shape == (2, 10, 8)
    # and indexing
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a1')
    assert isinstance(assign, Subscript) and assign.index == (
        Variable('i_outer'), 0, Variable('i_inner'))

    # now test with evenly sized
    a2 = next(x for x in k.args if x.name == 'a2')
    assert a2.shape == (2, 16, 8)
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a2')
    assert isinstance(assign, Subscript) and assign.index == (
        Variable('i_outer'), 0, Variable('i_inner'))


def test_lpy_array_splitter_f_deep():
    from pymbolic.primitives import Subscript, Variable

    # create opts
    opts = dummy_loopy_opts(depth=8, order='F')

    # create array split
    asplit = array_splitter(opts)

    def _create():
        # create a test kernel
        arg1 = lp.GlobalArg('a1', shape=(10, 10), order='F')
        arg2 = lp.GlobalArg('a2', shape=(16, 16), order='F')

        return lp.make_kernel(
            '{[i]: 0 <= i < 10}',
            """
                a1[0, i] = 0 {id=a1}
                a2[0, i] = 0 {id=a2}
            """,
            [arg1, arg2],
            target=lp.CTarget())

    k = lp.split_iname(_create(), 'i', 8)
    k = asplit._split_loopy_arrays(k)

    # test that it runs
    k()

    # check dim
    a1 = next(x for x in k.args if x.name == 'a1')
    assert a1.shape == (8, 10, 2)
    # and indexing
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a1')
    assert isinstance(assign, Subscript) and assign.index == (
        Variable('i_inner'), 0, Variable('i_outer'))

    # now test with evenly sized
    a2 = next(x for x in k.args if x.name == 'a2')
    assert a2.shape == (8, 16, 2)
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a2')
    assert isinstance(assign, Subscript) and assign.index == (
        Variable('i_inner'), 0, Variable('i_outer'))
