# TODO way more tests here
from __future__ import division

from tempfile import NamedTemporaryFile
from collections import OrderedDict
import re
from string import Template

import loopy as lp
import numpy as np
from parameterized import parameterized
from loopy.kernel.data import AddressSpace as scopes
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from pytools.py_codegen import remove_common_indentation

from pyjac.core.array_creator import array_splitter, problem_size, MapStore, \
    creator, kint_type
from pyjac.kernel_utils.memory_limits import memory_limits, memory_type, \
    get_string_strides
from pyjac.kernel_utils.kernel_gen import find_inputs_and_outputs, \
    _unSIMDable_arrays, knl_info, make_kernel_generator
from pyjac.tests.test_utils import xfail, OptionLoopWrapper
from pyjac.core.enum_types import KernelType


def opts_loop(langs=['opencl'],
              width=[4, None],
              depth=[4, None],
              order=['C', 'F'],
              is_simd=None):

    return OptionLoopWrapper.from_dict(
        OrderedDict(
            [('lang', langs),
             ('width', width),
             ('depth', depth),
             ('order', order),
             ('device_type', 'CPU'),
             ('is_simd', is_simd)]))


@parameterized([(np.int32,), (np.int64,)])
def test_stride_limiter(dtype):
    # tests an issue, particularly for the Intel OpenCL runtime where integers in
    # array indexing that overflow the int32 max result in segfaults in kernel

    # The long term fix is probably to allow the user to specify the dtype via
    # command line or platform file, but for now we simply limit the maximum # of
    # conditions per run

    from pymbolic import parse
    arry_name = 'a'
    extractor = re.compile(r'{}\[(.+)\] = i'.format(arry_name))
    dim_size = 1000000
    for opt in opts_loop(is_simd=False):
        split = array_splitter(opt)
        # create a really big loopy array
        ary = lp.GlobalArg(arry_name, shape=(problem_size.name, dim_size),
                           dtype=dtype)
        # make a dummy kernel with this argument to populate dim tags
        knl = lp.make_kernel(['{{[i]: 0 <= i < {}}}'.format(dim_size),
                              '{{[j]: 0 <= j < {}}}'.format(problem_size.name)],
                             '{}[j, i] = i'.format(arry_name),
                             [ary, problem_size])
        # split said array
        knl = split.split_loopy_arrays(knl)
        ary = knl.args[0]
        # get limits object
        limits = None
        with NamedTemporaryFile(suffix='.yaml', mode='w') as temp:
            temp.write("""
                       memory-limits:
                           alloc:
                              {0} B
                           global:
                              {0} B
                       """.format(
                        str(np.iinfo(dtype).max * dtype().itemsize),
                        str(np.iinfo(dtype).max * dtype().itemsize)))
            temp.seek(0)
            limits = memory_limits.get_limits(
                opt, {memory_type.m_global: [ary]}, temp.name,
                get_string_strides()[0],
                dtype=dtype, limit_int_overflow=True)
        # and feed through stride limiter
        limit = limits.integer_limited_problem_size(ary, dtype=dtype)
        # get the intruction from the kernel
        knl = lp.generate_code_v2(knl).device_code()
        # regex the array indexing out
        index = extractor.search(knl).group(1)
        # sub out 'i', 'j' and 'problem_size'
        repl = {'i': str(dim_size - 1),
                'j': str(limit - 1),
                'j_inner': str(opt.vector_width),
                'problem_size': str(limit)}
        pattern = re.compile(r'\b(' + '|'.join(repl.keys()) + r')\b')
        index = pattern.sub(lambda x: repl[x.group()], index)
        index = re.sub('/', '//', index)
        max_index = parse(index)
        assert isinstance(max_index, (int, float))
        assert max_index < np.iinfo(dtype).max

        # finally, test that we get the same limit from can_fit
        assert limit == limits.can_fit(mtype=memory_type.m_global)[0]


def test_get_kernel_input_and_output():
    # make a kernel
    knl = lp.make_kernel('{[i]: 0 <= i < 2}',
                         '<> a = 1')
    assert not len(find_inputs_and_outputs(knl))

    knl = lp.make_kernel('{[i]: 0 <= i < 2}',
                         '<> a = b[i]',
                         [lp.GlobalArg('b', shape=(2,))])
    assert find_inputs_and_outputs(knl) == set(['b'])

    knl = lp.make_kernel('{[i]: 0 <= i < 2}',
                         '<> a = b[i] + c[i]',
                         [lp.GlobalArg('b', shape=(2,)),
                          lp.TemporaryVariable('c', shape=(2,), scope=scopes.GLOBAL)
                          ],
                         silenced_warnings=['read_no_write(c)'])
    assert find_inputs_and_outputs(knl) == set(['b', 'c'])


@xfail(msg="Loopy currently doesn't allow vector inames in conditionals, have "
           "to rethink this test.")
def test_unsimdable():
    from loopy.kernel.array import (VectorArrayDimTag)
    inds = ('j', 'i')
    test_size = 16
    for opt in opts_loop(is_simd=True):
        # make a kernel via the mapstore / usual methods
        base = creator('base', dtype=kint_type, shape=(10,), order=opt.order,
                       initializer=np.arange(10, dtype=kint_type))
        mstore = MapStore(opt, base, store.test_size)

        def __create_var(name, size=(test_size, 10)):
            return creator(name, kint_type, size, opt.order)

        # now create different arrays:

        # one that will cause a map transform
        mapt = creator('map', dtype=kint_type, shape=(10,), order=opt.order,
                       initializer=np.array(list(range(0, 3)) + list(range(4, 11)),
                       kint_type))
        mapv = __create_var('mapv')
        mstore.check_and_add_transform(mapv, mapt)

        # one that is only an affine transform
        affinet = creator('affine', dtype=kint_type, shape=(10,), order=opt.order,
                          initializer=np.arange(2, 12, dtype=kint_type))
        affinev = __create_var('affinev', (test_size, 12))
        mstore.check_and_add_transform(affinev, affinet)

        # and one that is a child of the affine transform
        affinet2 = creator('affine2', dtype=kint_type, shape=(10,), order=opt.order,
                           initializer=np.arange(4, 14, dtype=kint_type))
        mstore.check_and_add_transform(affinet2, affinet)
        # and add a child to it
        affinev2 = __create_var('affinev2', (test_size, 14))
        mstore.check_and_add_transform(affinev2, affinet2)

        # and finally, a child of the map transform
        mapt2 = creator('map2', dtype=kint_type, shape=(10,), order=opt.order,
                        initializer=np.array(list(range(0, 2)) + list(range(3, 11)),
                        kint_type))
        mstore.check_and_add_transform(mapt2, mapt)
        # and a child
        mapv2 = __create_var('mapv2')
        mstore.check_and_add_transform(mapv2, mapt2)

        # now create an kernel info
        affine_lp, affine_str = mstore.apply_maps(affinev, *inds)
        affine2_lp, affine2_str = mstore.apply_maps(affinev2, *inds)
        map_lp, map_str = mstore.apply_maps(mapv, *inds)
        map2_lp, map2_str = mstore.apply_maps(mapv2, *inds)

        instructions = Template(remove_common_indentation("""
        ${affine_str} = 0
        ${affine2_str} = 0
        ${map_str} = 0
        ${map2_str} = 0
        """)).safe_substitute(**locals())

        info = knl_info('test', instructions, mstore, kernel_data=[
            affine_lp, affine2_lp, map_lp, map2_lp])

        # create a dummy kgen
        kgen = make_kernel_generator(opt, KernelType.dummy, [info],
                                     type('namestore', (object,), {'jac': 0}),
                                     test_size=test_size, name='test')

        # make kernels
        kgen._make_kernels()

        # and call simdable
        cant_simd = _unSIMDable_arrays(kgen.kernels[0], opt, mstore,
                                       warn=False)

        if opt.depth:
            assert sorted(cant_simd) == [mapt2.name, map_lp.name, map2_lp.name]
        else:
            assert cant_simd == []

        # make sure we can generate code
        lp.generate_code_v2(kgen.kernels[0]).device_code()

        if not kgen.array_split.vector_width:
            continue

        # check that we've vectorized all arrays
        assert all(len(arr.shape) == 3 for arr in kgen.kernels[0].args)

        # get the split axis
        _, _, vec_axis, _ = kgen.array_split.split_shape(affine_lp)

        assert all(isinstance(arr.dim_tags[vec_axis], VectorArrayDimTag)
                   for arr in kgen.kernels[0].args if arr.name not in cant_simd)
