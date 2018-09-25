from __future__ import division

from collections import OrderedDict

import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from parameterized import parameterized, param
from unittest.case import SkipTest

from pyjac.core.array_creator import array_splitter
from pyjac.core.instruction_creator import get_deep_specializer
from pyjac.loopy_utils.loopy_utils import kernel_call
from pyjac.tests import get_test_langs
from pyjac.tests.test_utils import indexer, get_split_elements, OptionLoopWrapper

VECTOR_WIDTH = 8


def opts_loop(width=[VECTOR_WIDTH, None],
              depth=[VECTOR_WIDTH, None],
              order=['C', 'F'],
              is_simd=True,
              skip_non_vec=True,
              langs=get_test_langs(),
              skip_test=None):

    oploop = OrderedDict(
        [('width', width),
         ('depth', depth),
         ('order', order),
         ('is_simd', is_simd),
         ('lang', langs)])

    def skip(state):
        s = (skip_non_vec and not (state['depth'] or state['width']))
        if skip_test is not None:
            s = s or skip_test(state)
        return s

    for opts in OptionLoopWrapper.from_dict(oploop, skip_test=skip):
        yield param(opts)


def __get_ref_answer(base, asplit):
    vw = asplit.vector_width
    side = base.shape[0]
    split_shape = asplit.split_shape(base)[0]
    order = asplit.data_order

    def slicify(slicer, inds):
        slicer = slicer[:]
        count = 0
        for i in range(len(slicer)):
            if not slicer[i]:
                slicer[i] = index[count]
                count += 1

        assert count == len(inds), 'Not all indicies used!'
        return tuple(slicer)

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
    width = p.width
    depth = p.depth
    order = p.order
    return "{} with: [width={}, depth={}, order={}]".format(
        name, width, depth, order)


@parameterized(opts_loop,
               doc_func=_split_doc)
def test_npy_array_splitter(opts):
    # create array split
    asplit = array_splitter(opts)

    def _test(shape):
        __internal(asplit, shape, order=opts.order, width=opts.width,
                   depth=opts.depth)

    # test with small square
    _test((10, 10))

    # now test with evenly sized
    _test((16, 16))

    # finally, try with 3d arrays
    _test((10, 10, 10))
    _test((16, 16, 16))


@parameterized(lambda: opts_loop(width=[None]),
               doc_func=_split_doc)
def test_lpy_deep_array_splitter(opts):
    from pymbolic.primitives import Subscript, Variable
    # create array split
    asplit = array_splitter(opts)

    # create a test kernel
    size = VECTOR_WIDTH * 3
    loop_bound = VECTOR_WIDTH * 2
    arg1 = lp.GlobalArg('a1', shape=(size, size), order=opts.order)
    arg2 = lp.GlobalArg('a2', shape=(16, 16), order=opts.order)

    k = lp.make_kernel(
        '{{[i]: 0 <= i < {}}}'.format(loop_bound),
        """
            a1[0, i] = 1 {id=a1}
            a2[0, i] = 1 {id=a2}
        """,
        [arg1, arg2],
        silenced_warnings=['no_device_in_pre_codegen_checks'],
        target=lp.OpenCLTarget())

    k = lp.split_iname(k, 'i', VECTOR_WIDTH,
                       inner_tag='l.0' if not opts.is_simd else 'vec')
    a1_hold = k.arg_dict['a1'].copy()
    a2_hold = k.arg_dict['a2'].copy()
    k = asplit.split_loopy_arrays(k)

    # ensure there's no loopy errors
    lp.generate_code_v2(k).device_code()

    def __indexer():
        if opts.order == 'C':
            return (0, Variable('i_outer'), Variable('i_inner'))
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


# currently only have SIMD for wide-vectorizations
@parameterized(lambda: opts_loop(depth=[None]),
               doc_func=_split_doc)
def test_lpy_wide_array_splitter(opts):
    from pymbolic.primitives import Subscript, Variable
    # create array split
    asplit = array_splitter(opts)

    # create a test kernel
    arg1 = lp.GlobalArg('a1', shape=(10, 10), order=opts.order)
    arg2 = lp.GlobalArg('a2', shape=(16, 16), order=opts.order)

    k = lp.make_kernel(
        ['{[i]: 0 <= i < 10}',
         '{{[j_outer]: 0 <= j_outer < {}}}'.format(int(np.ceil(10 / VECTOR_WIDTH))),
         '{{[j_inner]: 0 <= j_inner < {}}}'.format(VECTOR_WIDTH)],
        """
        for i, j_outer, j_inner
            a1[j_outer, i] = 1 {id=a1}
            a2[j_outer, i] = 1 {id=a2}
        end
        """,
        [arg1, arg2],
        silenced_warnings=['no_device_in_pre_codegen_checks'],
        target=lp.OpenCLTarget())

    a1_hold = k.arg_dict['a1'].copy()
    a2_hold = k.arg_dict['a2'].copy()
    k = asplit.split_loopy_arrays(k)
    k = lp.tag_inames(k, {'j_inner': 'l.0' if not opts.is_simd else 'vec'})

    # ensure there's no loopy errors
    lp.generate_code_v2(k).device_code()

    def __indexer():
        if opts.order == 'C':
            return (Variable('j_outer'), Variable('i'), Variable('j_inner'))
        else:
            return (Variable('j_inner'), Variable('j_outer'), Variable('i'))

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


@parameterized(lambda: opts_loop(depth=[None]),
               doc_func=_split_doc)
def test_lpy_iname_presplit(opts):
    """
    Tests that inames access to pre-split inames in non-split loopy arrays are
    correctly handled
    """
    from pymbolic.primitives import Subscript, Variable
    # create array split
    asplit = array_splitter(opts)

    # create a test kernel
    arg1 = lp.GlobalArg('a1', shape=(20, 10), order=opts.order)
    arg2 = lp.GlobalArg('a2', shape=(16, 16), order=opts.order)

    k = lp.make_kernel(
        ['{[i]: 0 <= i < 10}',
         '{{[j_outer]: 0 <= j_outer < {}}}'.format(int(np.ceil(10 / VECTOR_WIDTH))),
         '{{[j_inner]: 0 <= j_inner < {}}}'.format(VECTOR_WIDTH)],
        """
            a1[j_outer, i] = 1 {id=a1}
            a2[j_outer, i] = 1 {id=a2}
        """,
        [arg1, arg2],
        silenced_warnings=['no_device_in_pre_codegen_checks'],
        target=lp.OpenCLTarget())

    k = asplit.split_loopy_arrays(k, dont_split=['a1', 'a2'])

    # ensure there's no loopy errors
    lp.generate_code_v2(k).device_code()

    def __indexer():
        return (Variable('j_outer') * VECTOR_WIDTH + Variable('j_inner'),
                Variable('i'))

    # check indexing
    assign = next(insn.assignee for insn in k.instructions if insn.id == 'a1')
    # construct index
    assert isinstance(assign, Subscript) and assign.index == __indexer()

    # now test with evenly sized
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
            target=lp.OpenCLTarget(),
            silenced_warnings=['no_device_in_pre_codegen_checks'])
        loopy_opts = type('', (object,), {'depth': vec_width, 'order': 'C',
                                          'use_atomic_doubles': True})
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
def test_get_split_shape(opts):
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
        grow = opts.order == 'F'
        assert gr == grow

        # and the vec_axis is in front if 'F' else in back
        vec_axis = len(shape) if opts.order == 'C' else 0
        assert vec == vec_axis

        # and finally, the split axis
        split_axis = 0 if opts.width else len(shape) - 1
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
def test_indexer(opts):
    asplit = array_splitter(opts)

    def __test(splitter, shape):
        # make a dummy array
        arr = np.arange(np.prod(shape)).reshape(shape)
        index = indexer(splitter, shape)

        # split
        split_arr = splitter.split_numpy_arrays(arr)[0]

        # loop over every index in the array
        check_axes = tuple(range(len(shape)))
        it = np.nditer(arr, flags=['multi_index'], order=opts.order)
        while not it.finished:
            # get indicies
            check_inds = tuple((x,) for x in it.multi_index)
            new_indicies = index(check_inds, check_axes)
            # check that it matches the old array value
            assert split_arr[new_indicies] == arr[it.multi_index]
            it.iternext()

    # test with small square
    __test(asplit, (10, 10))

    # now test with evenly sized
    __test(asplit, (16, 16))

    # finally, try with 3d arrays
    __test(asplit, (10, 10, 10))
    __test(asplit, (16, 16, 16))


@parameterized(lambda: opts_loop(skip_non_vec=False),
               doc_func=_split_doc)
def test_get_split_elements(opts):
    # create opts
    asplit = array_splitter(opts)

    def __test(shape, check_inds=None, check_axes=None, tiling=True):
        # make a dummy array
        arr = np.arange(1, np.prod(shape) + 1).reshape(shape)
        # split
        split_arr = asplit.split_numpy_arrays(arr)[0]

        if check_inds is None:
            assert tiling
            # create the indicies to check
            check_inds = tuple(np.arange(x) for x in shape)
            check_axes = tuple(range(len(shape)))
            ans = arr.flatten(opts.order)
        elif tiling:
            assert check_axes is not None
            assert check_inds is not None
            ans = kernel_call('', arr, check_axes, [check_inds])._get_comparable(
                arr, 0, True).flatten(opts.order)
        else:
            slicer = [slice(None)] * arr.ndim
            assert all(check_inds[0].size == ci.size for ci in check_inds[1:])
            for i, ax in enumerate(check_axes):
                slicer[ax] = check_inds[i]
            ans = arr[tuple(slicer)].flatten(opts.order)

        # and compare to the old (unsplit) matrix
        assert np.allclose(
            get_split_elements(split_arr, asplit, arr.shape, check_inds, check_axes,
                               tiling=tiling),
            ans)

    # test with small square
    __test((10, 10))

    # now test with evenly sized
    __test((16, 16))

    # finally, try with 3d arrays
    __test((10, 10, 10))
    # and some non-full check-inds / axes
    __test((10, 10, 10), [np.arange(3, 7), np.arange(2, 4)], (0, 1))
    __test((10, 10, 10), [np.arange(3, 7), np.arange(2, 4)], (1, 2))
    __test((10, 10, 10), [np.arange(3, 7), np.arange(2, 4)], (0, 2))
    __test((10, 10, 10), [np.arange(3, 7), np.arange(2, 4)], (0, 1))
    __test((16, 16, 16))
    __test((16, 16, 16), [np.arange(3, 7), np.arange(2, 4)], (1, 2))
    __test((16, 16, 16), [np.arange(3, 7), np.arange(2, 4)], (0, 2))
    __test((16, 16, 16), [np.arange(3, 7), np.arange(2, 4)], (0, 1))
    # and some non-tiled axes
    __test((10, 10, 10), [np.arange(0, 4), np.arange(3, 7), np.arange(2, 6)],
           (0, 1, 2), tiling=False)
    __test((16, 16, 16), [np.arange(0, 4), np.arange(3, 7), np.arange(2, 6)],
           (0, 1, 2), tiling=False)
    __test((16, 16, 16), [np.arange(0, 4), np.arange(3, 7), np.arange(2, 6)],
           (0, 1, 2), tiling=False)

    # try with a really large array
    __test((100000, 16, 16), [np.arange(3, 50000), np.arange(2, 10), np.array([7])],
           (0, 1, 2))
    __test((100000, 16, 16), [np.arange(0, 4), np.arange(3, 7), np.arange(2, 6)],
           (0, 1, 2), tiling=False)
