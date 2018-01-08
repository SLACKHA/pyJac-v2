from __future__ import division

from ..core.array_creator import array_splitter
from ..core.instruction_creator import get_deep_specializer
import numpy as np
import loopy as lp


class dummy_loopy_opts(object):
    def __init__(self, depth=None, width=None, order='C'):
        self.depth = depth
        self.width = width
        self.order = order


def __internal(asplit, shape, order='C', wide=8):
    """
    Assumes shape is square
    """
    side = shape[0]
    # create array
    arr = np.zeros(shape, order=order)
    # set values
    ind = [slice(None)] * arr.ndim
    for i in range(shape[-1]):
        ind[-1 if order == 'F' else 0] = i
        arr[ind] = i
    # split
    arr, = asplit.split_numpy_arrays(arr)
    shape = list(shape)
    if order == 'F':
        # put new dim at front
        shape.insert(0, wide)
        # and adjust end dim
        shape[-1] = np.ceil(side / wide)
    else:
        # put new dim at end
        shape.insert(len(shape), wide)
        # and adjust start dim
        shape[0] = np.ceil(side / wide)
    # check dim
    assert np.array_equal(arr.shape, shape)
    # and values
    ind = [0] * (arr.ndim - 1)
    if order == 'F':
        ind = [slice(None)] + ind
        set_at = -1
    else:
        ind = ind + [slice(None)]
        set_at = 0

    start = 0
    while start < side:
        ind[set_at] = int(start / wide)
        test = np.zeros(wide)
        ar = np.arange(start, np.minimum(start + wide, side))
        test[:ar.size] = ar[:]
        assert np.array_equal(arr[ind], test)
        start += wide


def test_npy_array_splitter_c_wide():
    # create opts
    opts = dummy_loopy_opts(width=8, order='C')

    # create array split
    asplit = array_splitter(opts)

    def _test(shape):
        __internal(asplit, shape, order='C', wide=opts.width)

    # test with small square
    _test((10, 10))

    # now test with evenly sized
    _test((16, 16))

    # finally, try with 3d arrays
    _test((10, 10, 10))
    _test((16, 16, 16))


def test_npy_array_splitter_f_deep():
    # create opts
    opts = dummy_loopy_opts(depth=8, order='F')

    # create array split
    asplit = array_splitter(opts)

    def _test(shape):
        __internal(asplit, shape, order='F', wide=opts.depth)

    # test with small square
    _test((10, 10))

    # now test with evenly sized
    _test((16, 16))

    # finally, try with 3d arrays
    _test((10, 10, 10))
    _test((16, 16, 16))


def _create(order='C', target=lp.ExecutableCTarget()):
    # create a test kernel
    arg1 = lp.GlobalArg('a1', shape=(10, 10), order=order)
    arg2 = lp.GlobalArg('a2', shape=(16, 16), order=order)

    return lp.make_kernel(
        '{[i]: 0 <= i < 10}',
        """
            a1[0, i] = 1 {id=a1}
            a2[0, i] = 1 {id=a2}
        """,
        [arg1, arg2],
        target=target)


def test_lpy_array_splitter_c_wide():
    from pymbolic.primitives import Subscript, Variable

    # create opts
    opts = dummy_loopy_opts(width=8, order='C')

    # create array split
    asplit = array_splitter(opts)

    k = lp.split_iname(_create('C'), 'i', 8)
    k = asplit.split_loopy_arrays(k)

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

    k = lp.split_iname(_create('F'), 'i', 8)
    k = asplit.split_loopy_arrays(k)

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


def test_atomic_deep_vec_with_small_split():
    # test that an :class:`atomic_deep_specialization` with split smaller than
    # the vector width uses the correct splitting size

    def __test(loop_size, vec_width):
        knl = lp.make_kernel(
            '{{[i]: 0 <= i < {}}}'.format(loop_size),
            """
            <> x = 1.0
            a1[0] = a1[0] + x {id=set}
            ... lbarrier {id=wait, dep=set}
            for i
                a1[0] = a1[0] + 1 {id=a1, dep=set:wait, nosync=set}
            end
            """,
            [lp.GlobalArg('a1', shape=(loop_size,), order='C', dtype=np.float32)],
            target=lp.PyOpenCLTarget())
        loopy_opts = type('', (object,), {'depth': vec_width, 'order': 'C',
                                          'use_atomics': True})
        knl = lp.split_iname(knl, 'i', vec_width, inner_tag='l.0')

        # feed through deep specializer
        _, ds = get_deep_specializer(loopy_opts, atomic_ids=['a1'],
                                     split_ids=['set'], use_atomics=True,
                                     is_write_race=True, split_size=loop_size)
        knl = ds(knl)

        val = np.minimum(loop_size, vec_width)
        assert 'x / {:.1f}f'.format(val) in lp.generate_code(knl)[0]

    # test kernel w/ loop size smaller than split
    __test(10, 16)
    # test kernel w/ loop size larger than split
    __test(16, 8)
