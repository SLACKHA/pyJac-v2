from __future__ import division

from collections import OrderedDict

import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_1  # noqa
from parameterized import parameterized, param
from unittest.case import SkipTest
from optionloop import OptionLoop

from pyjac.core.array_creator import array_splitter
from pyjac.core.instruction_creator import get_deep_specializer
from pyjac.tests.test_utils import indexer, parse_split_index

VECTOR_WIDTH = 8


class dummy_loopy_opts(object):
    def __init__(self, depth=None, width=None, order='C', is_simd=False):
        self.depth = depth
        self.width = width
        self.order = order
        self.is_simd = is_simd


def opts_loop(width=[VECTOR_WIDTH, None],
              depth=[VECTOR_WIDTH, None],
              order=['C', 'F'],
              is_simd=True,
              skip_non_vec=True):

    oploop = OptionLoop(OrderedDict(
        [('width', width),
         ('depth', depth),
         ('order', order),
         ('is_simd', is_simd)]))
    for state in oploop:
        if state['depth'] and state['width']:
            continue
        if skip_non_vec and not (state['depth'] or state['width']):
            continue
        if state['is_simd'] and not (state['depth'] or state['width']):
            state['is_simd'] = False
        yield param(state)


def __get_ref_answer(base, asplit):
    vw = asplit.vector_width
    side = base.shape[0]
    split_shape = asplit.split_shape(base)[0]
    order = asplit.data_order

    def slicify(slicer, inds):
        slicer = slicer.copy()
        count = 0
        for i in range(len(slicer)):
            if not slicer[i]:
                slicer[i] = index[count]
                count += 1

        assert count == len(inds), 'Not all indicies used!'
        return slicer

    # create answer
    # setup
    ans = np.zeros(split_shape, dtype=np.int)

    # setup
    count = 0
    side_count = 0
    if order == 'F':
        inds = slice(1, None)
        slicer = [slice(None)] + [None] * len(split_shape[inds])
    else:
        inds = slice(None, -1)
        slicer = [None] * len(split_shape[inds]) + [slice(None)]
    it = np.nditer(np.zeros(split_shape[inds]),
                   flags=['multi_index'], order=order)

    if (order == 'C' and asplit.depth) or (order == 'F' and asplit.width):
        # SIMD - no split
        # array populator
        while not it.finished:
            index = it.multi_index[:]
            # create a column or row
            offset = np.arange(side_count, side_count + vw)
            mask = (offset >= side)
            offset[np.where(mask)] = 0
            offset[np.where(~mask)] += count
            # set
            ans[slicify(slicer, index)] = offset[:]
            # update counters
            side_count = side_count + vw
            if side_count >= side:
                # reset
                side_count = 0
                count += side
            it.iternext()
    else:
        # SIMD - split
        # array populator
        while not it.finished:
            index = it.multi_index[:]
            # create a column or row
            offset = side_count + (np.arange(count, count + vw)) * side ** (
                base.ndim - 1)
            mask = (offset >= side**base.ndim)
            offset[np.where(mask)] = 0
            # set row
            ans[slicify(slicer, index)] = offset[:]
            # update counters
            side_count = side_count + 1
            if side_count >= side ** (base.ndim - 1):
                # reset
                side_count = 0
                count += vw

            it.iternext()

    return ans


def __get_ref_shape(asplit, base_shape):
    side = base_shape[0]
    shape = list(base_shape)

    vw = asplit.vector_width
    if asplit.data_order == 'F' and asplit.depth:
        # put new dim at front
        insert_at = 0
        # and shrink last dim
        change_at = -1
    elif asplit.data_order == 'F':
        if not asplit.is_simd:
            raise SkipTest('No split for non-explicit SIMD F-ordered '
                           'shallow vectorization')
        assert asplit.is_simd
        # insert at front
        insert_at = 0
        # and change old first dim
        change_at = 1
    elif asplit.data_order == 'C' and asplit.width:
        # put new dim at end
        insert_at = len(shape)
        # and adjust start dim
        change_at = 0
    else:
        if not asplit.is_simd:
            raise SkipTest('No split for non-explicit SIMD C-ordered '
                           'deep vectorization')
        # put new dim at end
        insert_at = len(shape)
        # and adjust old end dim
        change_at = len(shape) - 1
    # insert
    shape.insert(insert_at, vw)
    # and adjust end dim
    shape[change_at] = int(np.ceil(side / vw))
    return shape


def __internal(asplit, shape, order='C', width=None, depth=None):
    """
    Assumes shape is square
    """

    # create array
    base = np.arange(np.prod(shape)).reshape(shape, order=order)

    # split
    arr, = asplit.split_numpy_arrays(base.copy())

    # check shape
    assert np.array_equal(arr.shape, __get_ref_shape(asplit, base.shape))

    # check answer
    assert np.array_equal(__get_ref_answer(base, asplit), arr)


def _split_doc(func, num, params):
    test = '_helper_for_'
    name = func.__name__
    if test in name:
        name = name[name.index(test) + len(test):]

    p = params[0][0]
    width = p['width']
    depth = p['depth']
    order = p['order']
    return "{} with: [width={}, depth={}, order={}]".format(
        name, width, depth, order)


@parameterized(opts_loop,
               doc_func=_split_doc)
def test_npy_array_splitter(state):
    # create opts
    opts = dummy_loopy_opts(**state)

    # create array split
    asplit = array_splitter(opts)

    def _test(shape):
        __internal(asplit, shape, order=state['order'], width=opts.width,
                   depth=opts.depth)

    # test with small square
    _test((10, 10))

    # now test with evenly sized
    _test((16, 16))

    # finally, try with 3d arrays
    _test((10, 10, 10))
    _test((16, 16, 16))


def _create(order='C'):
    # create a test kernel
    arg1 = lp.GlobalArg('a1', shape=(10, 10), order=order)
    arg2 = lp.GlobalArg('a2', shape=(16, 16), order=order)

    return lp.make_kernel(
        '{[i]: 0 <= i < 10}',
        """
            a1[0, i] = 1 {id=a1}
            a2[0, i] = 1 {id=a2}
        """,
        [arg1, arg2])


@parameterized(opts_loop,
               doc_func=_split_doc)
def test_lpy_array_splitter(state):
    from pymbolic.primitives import Subscript, Variable, Product, Sum

    # create opts
    opts = dummy_loopy_opts(**state)

    # create array split
    asplit = array_splitter(opts)

    k = lp.split_iname(_create(state['order']), 'i', VECTOR_WIDTH,
                       inner_tag='l.0' if not state['is_simd'] else 'vec')
    a1_hold = k.arg_dict['a1'].copy()
    a2_hold = k.arg_dict['a2'].copy()
    k = asplit.split_loopy_arrays(k)

    # ensure there's no loopy errors
    lp.generate_code_v2(k).device_code()

    def __indexer():
        if state['order'] == 'C':
            if state['width']:
                return (0, Variable('i_inner') +
                        Variable('i_outer') * VECTOR_WIDTH, 0)
            else:
                return (0, Variable('i_outer'), Variable('i_inner'))
        else:
            if state['width']:
                return (0, 0, Sum((
                    Variable('i_inner'), Product(
                        (Variable('i_outer'), VECTOR_WIDTH)))))

            else:
                return (Variable('i_inner'), 0, Variable('i_outer'))

    # check dim
    a1 = k.arg_dict['a1']
    assert a1.shape == asplit.split_shape(a1_hold)[0]
    # and indexing
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a1')
    # construct index
    assert isinstance(assign, Subscript) and assign.index == __indexer()

    # now test with evenly sized
    a2 = k.arg_dict['a2']
    assert a2.shape == asplit.split_shape(a2_hold)[0]
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a2')
    assert isinstance(assign, Subscript) and assign.index == __indexer()


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
    __test(16, VECTOR_WIDTH)


@parameterized(opts_loop,
               doc_func=_split_doc)
def test_get_split_shape(state):
    # create opts
    opts = dummy_loopy_opts(**state)

    # create array split
    asplit = array_splitter(opts)

    def __test(splitter, shape):
        # make a dummy array
        arr = np.zeros(shape)
        # get the split shape
        sh, gr, vec, spl = asplit.split_shape(arr)
        # first -- test against numpy splitter to ensure we get the right shape
        assert sh == asplit.split_numpy_arrays(arr)[0].shape

        # next, the "grow" axis is either the first axis ("C") or the second axis
        # for "F"
        grow = state['order'] == 'F'
        assert gr == grow

        # and the vec_axis is in front if 'F' else in back
        vec_axis = len(shape) if state['order'] == 'C' else 0
        assert vec == vec_axis

        # and finally, the split axis
        split_axis = 0 if state['width'] else len(shape) - 1
        assert spl == split_axis

    # test with small square
    __test(asplit, (10, 10))

    # now test with evenly sized
    __test(asplit, (16, 16))

    # finally, try with 3d arrays
    __test(asplit, (10, 10, 10))
    __test(asplit, (16, 16, 16))

    # and finally test with some randomly sized arrays
    for i in range(50):
        shape = np.random.randint(1, 12, size=np.random.randint(2, 5))
        __test(asplit, shape)


@parameterized(lambda: opts_loop(skip_non_vec=False),
               doc_func=_split_doc)
def test_indexer(state):
    # create opts
    opts = dummy_loopy_opts(**state)
    asplit = array_splitter(opts)

    def __test(splitter, shape):
        # make a dummy array
        arr = np.arange(np.prod(shape)).reshape(shape)
        index = indexer(splitter, shape)

        # split
        split_arr = splitter.split_numpy_arrays(arr)[0]

        # create the indicies to check
        check_inds = tuple(np.arange(x) for x in shape)
        check_axes = tuple(range(len(shape)))

        # get indicies
        new_indicies = index(check_inds, check_axes)

        # and create indexer for unsplit array
        old_inds = [slice(None)] * len(shape)
        for i in range(len(check_axes)):
            old_inds[i] = check_inds[i]

        assert np.allclose(split_arr[new_indicies], arr[tuple(old_inds)])

    # test with small square
    __test(asplit, (10, 10))

    # now test with evenly sized
    __test(asplit, (16, 16))

    # finally, try with 3d arrays
    __test(asplit, (10, 10, 10))
    __test(asplit, (16, 16, 16))


@parameterized(lambda: opts_loop(skip_non_vec=False),
               doc_func=_split_doc)
def test_parse_split_index(state):
    # create opts
    opts = dummy_loopy_opts(**state)
    asplit = array_splitter(opts)

    def __test(splitter, shape):
        # make a dummy array
        arr = np.arange(np.prod(shape)).reshape(shape)
        # split
        split_arr = splitter.split_numpy_arrays(arr)[0]

        # create the indicies to check
        check_inds = tuple(np.arange(x) for x in shape)
        check_axes = tuple(range(len(shape)))

        import pdb; pdb.set_trace()
        mask = parse_split_index(split_arr, splitter, arr.shape, check_inds,
                                 check_axes)

        # and create indexer for unsplit array
        old_inds = [slice(None)] * len(shape)
        for i in range(len(check_axes)):
            old_inds[i] = check_inds[i]

        assert np.allclose(split_arr[mask], arr[tuple(old_inds)])

    # test with small square
    __test(asplit, (10, 10))

    # now test with evenly sized
    __test(asplit, (16, 16))

    # finally, try with 3d arrays
    __test(asplit, (10, 10, 10))
    __test(asplit, (16, 16, 16))
