"""
Tests for the memory manager for code-generation

Copyright Nicholas Curtis 2018
"""
import shutil
import os
import subprocess
from collections import OrderedDict
from textwrap import dedent

from six.moves import cPickle as pickle
import pyopencl as cl
import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from nose.tools import assert_raises

from pyjac import utils
from pyjac.utils import temporary_directory
from pyjac.core import array_creator as arc
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core.enum_types import DeviceMemoryType
from pyjac.kernel_utils.kernel_gen import CallgenResult
from pyjac.kernel_utils.memory_tools import get_memory, DeviceNamer, HostNamer, \
    host_langs
from pyjac.libgen.libgen import get_toolchain, compile, link
from pyjac.loopy_utils.loopy_utils import get_target
from pyjac.tests import script_dir
from pyjac.tests.test_utils import OptionLoopWrapper, get_test_langs, \
    temporary_build_dirs


def __test_cases():
    return OptionLoopWrapper.from_dict(
        OrderedDict(
            [('lang', get_test_langs()),
             ('order', ['C', 'F']),
             ('width', [4, None]),
             ('depth', [4, None]),
             ('device_type', (cl.device_type.CPU, cl.device_type.GPU, None)),
             ('is_simd', [True, False]),
             ('dev_mem_type', (DeviceMemoryType.pinned, DeviceMemoryType.mapped))]),
        ignored_state_vals=['dev_mem_type'],
        skip_test=lambda x: x['lang'] == 'c' and
        x['dev_mem_type'] == DeviceMemoryType.pinned)


def type_map(lang):
    target = get_target(lang)
    type_map = {}
    type_map[lp.to_loopy_type(np.float64, target=target)] = 'double'
    type_map[lp.to_loopy_type(np.int32, target=target)] = 'int'
    type_map[lp.to_loopy_type(np.int64, target=target)] = 'long int'
    return type_map


def test_memory_tools_alloc():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen)

        # create some arrays
        a1 = lp.GlobalArg('a1', shape=(arc.problem_size,), dtype=np.int32)
        a2 = lp.GlobalArg('a2', shape=(arc.problem_size, 10), dtype=np.float64)

        # test alloc
        if opts.lang == 'c':
            # test default
            assert 'a1 = (int*)malloc(problem_size * sizeof(int))' in mem.alloc(
                False, a1)
            # test namer
            mem2 = get_memory(callgen, host_namer=HostNamer())
            assert 'h_a1 = (int*)malloc(problem_size * sizeof(int))' \
                in mem2.alloc(False, a1)
            # test more complex shape / other dtypes
            assert 'a2 = (double*)malloc(10 * problem_size * sizeof(double))'\
                in mem.alloc(False, a2)
            # test device mem
            assert 'a2 = (double*)malloc(10 * per_run * sizeof(double))'\
                in mem.alloc(True, a2)
            # and ic specification
            assert 'a2 = (double*)malloc(10 * run * sizeof(double))'\
                in mem.alloc(True, a2, num_ics='run')
        elif opts.lang == 'opencl':
            # test default host
            assert 'a1 = (int*)malloc(problem_size * sizeof(int))' in mem.alloc(
                False, a1)
            assert (('CL_MEM_ALLOC_HOST_PT' in mem.alloc(True, a1)) ==
                    (wrapper.state['dev_mem_type'] == DeviceMemoryType.pinned))
            # test default device
            assert ('a1 = clCreateBuffer(context, CL_MEM_READ_WRITE') \
                in mem.alloc(True, a1)
            assert 'per_run * sizeof(int)' in mem.alloc(True, a1)
            # test readonly
            assert 'CL_MEM_READ_ONLY' in mem.alloc(True, a1, readonly=True)
        else:
            raise NotImplementedError


def test_memory_tools_sync():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen)

        # not implemented as all calls are currently blocking
        assert not mem.sync()


def test_memory_tools_free():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen)

        # create a test array
        a1 = lp.GlobalArg('a1', shape=(arc.problem_size,), dtype=np.int32)

        # test frees
        if opts.lang == 'c':
            assert mem.free(True, a1) == 'free(a1);'
            assert mem.free(False, a1) == 'free(a1);'
        elif opts.lang == 'opencl':
            assert mem.free(False, a1) == 'free(a1);'
            assert mem.free(True, a1) == 'check_err(clReleaseMemObject(a1));'
        else:
            raise NotImplementedError

        # and test w/ device prefix
        mem = get_memory(callgen, device_namer=DeviceNamer('this'))
        # test frees
        if opts.lang == 'c':
            assert mem.free(False, a1) == 'free(a1);'
        elif opts.lang == 'opencl':
            assert mem.free(True, a1) == 'check_err(clReleaseMemObject(this->d_a1));'
        else:
            raise NotImplementedError


def test_memory_tools_memset():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen)

        # create a test array
        a1 = lp.GlobalArg('a1', shape=(arc.problem_size, 10), dtype=np.int32)
        d1 = lp.GlobalArg('d1', shape=(arc.problem_size, 10, 10), dtype=np.float64)

        # test memset
        if opts.lang == 'c':
            assert mem.memset(True, a1) == \
                'memset(a1, 0, 10 * per_run * sizeof(int));'
            assert mem.memset(False, a1) == \
                'memset(a1, 0, 10 * problem_size * sizeof(int));'
            # check double
            assert 'sizeof(double)' in mem.memset(False, d1)
            assert '100 * problem_size' in mem.memset(False, d1)
            # check ic spec
            assert '100 * dummy' in mem.memset(True, d1, num_ics='dummy')

        elif opts.lang == 'opencl':
            assert mem.memset(False, a1) == \
                'memset(a1, 0, 10 * problem_size * sizeof(int));'
            dev = mem.memset(True, a1)
            if wrapper.state['dev_mem_type'] == DeviceMemoryType.pinned:
                # pinned -> should have a regular memset
                assert 'memset(temp_i, 0, 10 * per_run * sizeof(int));' in dev
                # and map / unmaps
                assert ('clEnqueueMapBuffer(queue, a1, CL_TRUE, CL_MAP_WRITE, '
                        '0, 10 * per_run * sizeof(int), 0, NULL, NULL, &return_code)'
                        ) in dev
                assert ('check_err(clEnqueueUnmapMemObject(queue, a1, temp_i, 0, '
                        'NULL, NULL));') in dev

                # check namer
                mem2 = get_memory(callgen, device_namer=DeviceNamer('data'),
                                  host_namer=HostNamer('data'))
                dev = mem2.memset(True, a1)
                assert ', data->d_a1, ' in dev
                assert 'data->h_temp_i = ' in dev
                mem3 = get_memory(callgen, device_namer=DeviceNamer(
                    'data', postfix='_test'))
                dev = mem3.memset(True, a1)
                assert ', data->d_a1_test, ' in dev
            else:
                # check for opencl 1.2 memset
                assert ('clEnqueueFillBuffer(queue, a1, &zero, sizeof(double), 0, '
                        '10 * per_run * sizeof(int), 0, NULL, NULL)') in dev
                # check for opencl <= 1.1 memset
                assert ('clEnqueueWriteBuffer(queue, a1, CL_TRUE, 0, '
                        '10 * per_run * sizeof(int), zero, 0, NULL, NULL)') in dev

        else:
            raise NotImplementedError


def test_memory_tools_copy():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen, host_namer=HostNamer(), device_namer=DeviceNamer())

        # create a test array
        a1 = lp.GlobalArg('a1', shape=(arc.problem_size), dtype=np.int32)
        a2 = lp.GlobalArg('a2', shape=(arc.problem_size, 10), dtype=np.int32)
        d3 = lp.GlobalArg('d3', shape=(arc.problem_size, 10, 10), dtype=np.float64)

        # test frees
        if opts.lang == 'c':
            # test host constant copy
            assert mem.copy(True, a1, host_constant=True) == (
                'memcpy(d_a1, h_a1, problem_size * sizeof(int));')
            # test copy to device
            assert mem.copy(True, a1) == ('memcpy(d_a1, &h_a1[offset * 1], '
                                          'this_run * sizeof(int));')
            # test copy from device
            if opts.order == 'C':
                assert mem.copy(False, a2) == ('memcpy(&h_a2[offset * 10], d_a2, '
                                               '10 * this_run * sizeof(int));')
            else:
                assert mem.copy(False, a2) == ('memcpy2D_out(h_a2, problem_size, '
                                               'd_a2, per_run, offset, '
                                               'this_run * sizeof(int), '
                                               '10);')
            if opts.order == 'C':
                assert mem.copy(True, d3) == ('memcpy(d_d3, &h_d3[offset * 100], '
                                              '100 * this_run * sizeof(double));')
            else:
                assert mem.copy(True, d3, num_ics='test', num_ics_this_run='test2')\
                    == ('memcpy2D_in(d_d3, test, h_d3, problem_size, offset, '
                        'test2 * sizeof(double), 100);')
        elif opts.lang == 'opencl':
            dev = mem.copy(True, a1, host_constant=True)
            if wrapper.state['dev_mem_type'] == DeviceMemoryType.pinned:
                assert 'clEnqueueUnmapMemObject' in dev
                assert ('h_temp_i = (int*)clEnqueueMapBuffer(queue, d_a1, CL_TRUE, '
                        'CL_MAP_WRITE, 0, problem_size * sizeof(int), 0, NULL, '
                        'NULL, &return_code);') in dev
                assert 'memcpy(h_temp_i, h_a1, problem_size * sizeof(int));' in dev
            else:
                # mapped
                assert ('clEnqueueWriteBuffer(queue, d_a1, CL_TRUE, 0, '
                        'problem_size * sizeof(int), &h_a1, 0, NULL, NULL)') in dev

            dev = mem.copy(False, d3, offset='test', num_ics_this_run='test2')
            if wrapper.state['dev_mem_type'] == DeviceMemoryType.pinned:
                assert ('h_temp_d = (double*)clEnqueueMapBuffer(queue, d_d3, '
                        'CL_TRUE, CL_MAP_READ, 0, 100 * per_run * sizeof(double)'
                        ', 0, NULL, NULL, &return_code);') in dev
                if opts.order == 'C':
                    assert ('memcpy(&h_d3[test * 100], h_temp_d, '
                            '100 * test2 * sizeof(double));') in dev
                else:
                    assert ('memcpy2D_out(h_d3, h_temp_d, '
                            '(size_t[]) {test * sizeof(double), 0, 0}, '
                            '(size_t[]) {test2 * sizeof(double), 100, 1}, '
                            'per_run * sizeof(double), 0, '
                            'problem_size * sizeof(double), 0);')
            else:
                # mapped
                if opts.order == 'C':
                    assert ('clEnqueueReadBuffer(queue, d_d3, CL_TRUE, 0, '
                            '100 * test2 * sizeof(double), &h_d3[test*100], '
                            '0, NULL, NULL)') in dev
                else:
                    assert 'size_t buffer_origin[3] = {0, 0, 0};' in dev
                    assert ('size_t host_origin[3] = {test * sizeof(double)'
                            ', 0, 0};') in dev
                    assert ('size_t region[3] = {test2 * sizeof(double)'
                            ', 100, 1};') in dev
                    assert ('clEnqueueReadBufferRect(queue, d_d3, CL_TRUE, '
                            '&buffer_origin[0], '
                            '&host_origin[0], '
                            '&region[0], '
                            'per_run * sizeof(double), 0, '
                            'problem_size * sizeof(double), 0, h_d3, 0, NULL, NULL)'
                            ) in dev
        else:
            raise NotImplementedError


def test_memory_tools_defn():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen, host_namer=HostNamer(), device_namer=DeviceNamer())

        a1 = lp.GlobalArg('a1', shape=(arc.problem_size), dtype=np.int32)
        a2 = lp.GlobalArg('a2', shape=(arc.problem_size, 10), dtype=np.int64)
        d3 = lp.GlobalArg('d3', shape=(arc.problem_size, 10, 10), dtype=np.float64)
        a4 = lp.ValueArg('a4', dtype=np.int64)
        a5 = lp.ValueArg('a5', dtype=np.int32)
        a6 = lp.TemporaryVariable('a6', initializer=np.array([0, 1, 2]),
                                  read_only=True)

        if opts.lang == 'opencl':
            assert mem.define(True, a1) == 'cl_mem d_a1;'
            assert mem.define(False, a2) == 'long int* h_a2;'
            assert mem.define(True, d3) == 'cl_mem d_d3;'
            assert mem.define(False, a4) == 'long int h_a4;'
            assert mem.define(True, a5) == 'cl_uint d_a5;'
            assert mem.define(True, a5) == 'cl_uint d_a5;'
            with assert_raises(Exception):
                mem.define(True, a6, host_constant=True)
            assert mem.define(False, a6, host_constant=True) == \
                'const long int h_a6[3] = {0, 1, 2};'

        elif opts.lang == 'c':
            assert mem.define(True, a1) == 'int* d_a1;'
            assert mem.define(False, a2) == 'long int* h_a2;'
            assert mem.define(True, d3) == 'double* d_d3;'
            assert mem.define(False, a4) == 'long int h_a4;'
            assert mem.define(True, a5) == 'int d_a5;'
            with assert_raises(Exception):
                mem.define(True, a6, host_constant=True)
            assert mem.define(False, a6, host_constant=True) == \
                'const long int h_a6[3] = {0, 1, 2};'
        else:
            raise NotImplementedError


def test_buffer_sizes():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        # create a memory manager
        mem = get_memory(callgen, host_namer=HostNamer(), device_namer=DeviceNamer())

        # test with value arg
        a1 = lp.GlobalArg('a1', shape=(arc.problem_size), dtype=np.int32)
        assert mem.non_ic_size(a1) == '1'
        assert mem.buffer_size(True, a1, num_ics='per_run') == \
            'per_run * sizeof(int)'
        assert mem.buffer_size(False, a1) == 'problem_size * sizeof(int)'

        # test with Variable
        from pymbolic.primitives import Variable
        a1 = lp.GlobalArg('a1', shape=(Variable(arc.problem_size.name)),
                          dtype=np.int32)
        assert mem.non_ic_size(a1) == '1'
        assert mem.buffer_size(True, a1, num_ics='per_run') == \
            'per_run * sizeof(int)'
        assert mem.buffer_size(False, a1) == 'problem_size * sizeof(int)'


def test_can_load():
    """
    Tests whether the external cog code-gen app can load our serialized objects
    """

    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map(opts.lang))
        with temporary_directory() as tdir:
            with open(os.path.join(tdir, 'test.cpp'), mode='w') as file:
                file.write("""
                    /*[[[cog
                        import cog
                        import os
                        import pickle
                        # next, unserialize the callgen
                        with open(callgen, 'rb') as file:
                            call = pickle.load(file)

                        # and create a memory manager
                        from pyjac.kernel_utils.memory_tools import get_memory
                        mem = get_memory(call)
                        cog.outl('success!')
                       ]]]
                       [[[end]]]*/""")

            # and serialize mem
            with open(os.path.join(tdir, 'callgen.pickle'), 'wb') as file:
                pickle.dump(callgen, file)

            # and call cog
            from cogapp import Cog
            cmd = [
                'cog', '-e', '-d', '-Dcallgen={}'.format(
                    os.path.join(tdir, 'callgen.pickle')),
                '-o', os.path.join(tdir, 'test'), os.path.join(tdir, 'test.cpp')]
            Cog().callableMain(cmd)

            with open(os.path.join(tdir, 'test'), 'r') as file:
                assert file.read().strip() == 'success!'


def test_strided_copy():
    wrapper = __test_cases()
    for opts in wrapper:
        lang = opts.lang
        order = opts.order
        depth = opts.depth
        width = opts.width

        with temporary_build_dirs() as (build_dir, obj_dir, lib_dir):
            vec_size = depth if depth else (width if width else 0)
            # set max per run such that we will have a non-full run (1024 - 1008)
            # this should also be evenly divisible by depth and width
            # (as should the non full run)
            max_per_run = 16
            # number of ics should be divisibly by depth and width
            ics = max_per_run * 8 + vec_size
            if vec_size:
                assert ics % vec_size == 0
                assert max_per_run % vec_size == 0
                assert int(np.floor(ics / max_per_run) * max_per_run) % vec_size == 0

            # build initial callgen
            callgen = CallgenResult(
                order=opts.order, lang=opts.lang,
                dev_mem_type=wrapper.state['dev_mem_type'],
                type_map=type_map(opts.lang))

            # set type
            dtype = np.dtype('float64')

            # create test arrays
            def __create(shape):
                if not isinstance(shape, tuple):
                    shape = (shape,)
                shape = (ics,) + shape
                arr = np.zeros(shape, dtype=dtype, order=order)
                arr.flat[:] = np.arange(np.prod(shape))
                return arr
            arrays = [__create(16), __create(10), __create(20), __create((20, 20)),
                      __create(())]
            const = [np.arange(10, dtype=dtype)]

            # max size for initialization in kernel
            max_size = max([x.size for x in arrays])

            def _get_dtype(dtype):
                return lp.to_loopy_type(
                    dtype, target=get_target(opts.lang))

            lp_arrays = [lp.GlobalArg('a{}'.format(i),
                                      shape=(arc.problem_size.name,) + a.shape[1:],
                                      order=order,
                                      dtype=_get_dtype(arrays[i].dtype))
                         for i, a in enumerate(arrays)] + \
                        [lp.TemporaryVariable(
                            'a{}'.format(i + len(arrays)),
                            dtype=_get_dtype(dtype), order=order,
                            initializer=const[i],
                            read_only=True, shape=const[i].shape)
                         for i in range(len(const))]
            const = lp_arrays[len(arrays):]

            # now update args
            callgen = callgen.copy(name='test',
                                   input_args={'test': [x for x in lp_arrays
                                               if x not in const]},
                                   output_args={'test' : []},
                                   host_constants={'test': const})

            temp_fname = os.path.join(build_dir, 'in' + utils.file_ext[lang])
            fname = os.path.join(build_dir, 'test' + utils.file_ext[lang])
            with open(temp_fname, 'w') as file:
                file.write(dedent("""
       /*[[[cog
            # expected globals:
            #   callgen      - path to serialized callgen object
            #   lang         - the language to use
            #   problem_size - the problem size
            #   max_per_run  - the run-size
            #   max_size     - the maximum array size
            #   order        - The data ordering

            import cog
            import os
            import numpy as np
            from six.moves import cPickle as pickle

            # unserialize the callgen
            with open(callgen, 'rb') as file:
                callgen = pickle.load(file)

            # determine the headers to include
            lang_headers = []
            if lang == 'opencl':
                lang_headers.extend([
                                '#include "memcpy_2d.oclh"',
                                '#include "vectorization.oclh"',
                                '#include <CL/cl.h>',
                                '#include "error_check.oclh"'])
            elif lang == 'c':
                lang_headers.extend([
                    '#include "memcpy_2d.hpp"',
                    '#include "error_check.hpp"'])
            cog.outl('\\n'.join(lang_headers))
            ]]]
            [[[end]]]*/

            // normal headers
            #include <stdlib.h>
            #include <string.h>
            #include <assert.h>


            int main()
            {
                /*[[[cog
                    if lang == 'opencl':
                        cog.outl(
                    'double* h_temp_d;\\n'
                    'int* h_temp_i;\\n'
                    '// create a context / queue\\n'
                    'int lim = 10;\\n'
                    'cl_uint num_platforms;\\n'
                    'cl_uint num_devices;\\n'
                    'cl_platform_id platform [lim];\\n'
                    'cl_device_id device [lim];\\n'
                    'cl_int return_code;\\n'
                    'cl_context context;\\n'
                    'cl_command_queue queue;\\n'
                    'check_err(clGetPlatformIDs(lim, platform, &num_platforms));\\n'
                    'for (int i = 0; i < num_platforms; ++i)\\n'
                    '{\\n'
                    '    check_err(clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, '
                    '    lim, device, &num_devices));\\n'
                    '    if(num_devices > 0)\\n'
                    '        break;\\n'
                    '}\\n'
                    'context = clCreateContext(NULL, 1, &device[0], NULL, NULL, '
                    '&return_code);\\n'
                    'check_err(return_code);\\n'
                    '//create queue\\n'
                    'queue = clCreateCommandQueue(context, device[0], 0, '
                    '&return_code);\\n'
                    'check_err(return_code);\\n')
                ]]]
                [[[end]]]*/

                /*[[[cog

                    # determine maximum array size
                    cog.outl('double zero [{max_size}] = {{0}};'.format(
                        max_size=max_size))

                    # init variables
                    cog.outl('int problem_size = {};'.format(problem_size))
                    cog.outl('int per_run = {};'.format(max_per_run))
                  ]]]
                  [[[end]]]*/

                /*[[[cog
                    # create memory tool
                    from string import Template
                    import loopy as lp
                    from pyjac.kernel_utils.memory_tools import get_memory
                    from pyjac.kernel_utils.memory_tools import HostNamer
                    from pyjac.kernel_utils.memory_tools import DeviceNamer
                    mem = get_memory(callgen, host_namer=HostNamer(),
                                     device_namer=DeviceNamer())

                    # declare host and device arrays
                    for arr in callgen.kernel_args['test'] + callgen.work_arrays:
                        if not isinstance(arr, lp.ValueArg):
                            cog.outl(mem.define(False, arr))
                            cog.outl(mem.define(True, arr))
                    # define host constants
                    for arr in callgen.host_constants['test']:
                        cog.outl(mem.define(False, arr, host_constant=True,
                                            force_no_const=True))
                        cog.outl(mem.define(True, arr))

                    # and declare the temporary array
                    cog.outl(mem.define(True, lp.GlobalArg(
                        'temp_d', dtype=lp.to_loopy_type(np.float64))))

                    # allocate host and device arrays
                    for arr in callgen.kernel_args['test'] + callgen.work_arrays:
                        if not isinstance(arr, lp.ValueArg):
                            cog.outl(mem.alloc(False, arr))
                            cog.outl(mem.alloc(True, arr))
                    for arr in callgen.host_constants['test']:
                        # alloc device version of host constant
                        cog.outl(mem.alloc(True, arr))
                        # copy host constants
                        cog.outl(mem.copy(True, arr, host_constant=True))

                    def _get_size(arr):
                        size = 1
                        for x in arr.shape:
                            if not isinstance(x, int):
                                assert x.name == 'problem_size'
                                size *= int(problem_size)
                            else:
                                size *= x
                        return size

                    # save copies of host arrays
                    host_copies = [Template(
                        '${type} ${save} [${size}] = {${vals}};\\n'
                        'memset(${host}, 0, ${size} * sizeof(${type}));'
                        ).safe_substitute(
                            save='h_' + arr.name + '_save',
                            host='h_' + arr.name,
                            size=_get_size(arr),
                            vals=', '.join([str(x) for x in np.arange(
                                _get_size(arr)).flatten(order)]),
                            type=callgen.type_map[arr.dtype])
                            for arr in callgen.kernel_args['test'] +
                                       callgen.host_constants['test']]
                    for hc in host_copies:
                        cog.outl(hc)
                  ]]]
                  [[[end]]]*/

            // kernel
            for (size_t offset = 0; offset < problem_size; offset += per_run)
            {
                int this_run = problem_size - offset < per_run ? \
                    problem_size - offset : per_run;
                /* Memory Transfers into the kernel, if any */
                /*[[[cog
                  mem2 = get_memory(callgen, host_namer=HostNamer(postfix='_save'),
                                    device_namer=DeviceNamer())
                  for arr in callgen.kernel_args['test']:
                      cog.outl(mem2.copy(True, arr))
                  ]]]
                  [[[end]]]*/

                /* Memory Transfers out */
                /*[[[cog
                  for arr in callgen.kernel_args['test']:
                      cog.outl(mem.copy(False, arr))
                  ]]]
                  [[[end]]]*/
            }

                /*[[[cog
                    # and finally check
                    check_template = Template(
                        'for(int i = 0; i < ${size}; ++i)\\n'
                        '{\\n'
                        '    assert(${host}[i] == ${save}[i]);\\n'
                        '}\\n')
                    checks = [check_template.safe_substitute(
                        host=mem.get_name(False, arr),
                        save=mem2.get_name(False, arr),
                        size=_get_size(arr))
                              for arr in callgen.kernel_args['test']]
                    for check in checks:
                        cog.outl(check)
                  ]]]
                  [[[end]]]*/

                /*[[[cog
                    if lang == 'opencl':
                        cog.outl('check_err(clFlush(queue));')
                        cog.outl('check_err(clReleaseCommandQueue(queue));')
                        cog.outl('check_err(clReleaseContext(context));')
                  ]]]
                  [[[end]]]*/
                return 0;
            }
            """.strip()))

            # serialize callgen
            with open(os.path.join(build_dir, 'callgen.pickle'), 'wb') as file:
                pickle.dump(callgen, file)

            # cogify
            from cogapp import Cog
            cmd = [
                'cog', '-e', '-d', '-Dcallgen={}'.format(
                    os.path.join(build_dir, 'callgen.pickle')),
                '-Dmax_per_run={}'.format(max_per_run),
                '-Dproblem_size={}'.format(ics),
                '-Dmax_size={}'.format(max_size),
                '-Dlang={}'.format(lang),
                '-Dorder={}'.format(order),
                '-o', fname, temp_fname]
            Cog().callableMain(cmd)

            files = [fname]
            # write aux
            write_aux(build_dir, opts, [], [])

            # copy any deps
            def __copy_deps(lang, scan_path, out_path, change_extension=True,
                            ffilt=None, nfilt=None):
                deps = [x for x in os.listdir(scan_path) if os.path.isfile(
                    os.path.join(scan_path, x)) and not x.endswith('.in')]
                if ffilt is not None:
                    deps = [x for x in deps if ffilt in x]
                if nfilt is not None:
                    deps = [x for x in deps if nfilt not in x]
                files = []
                for dep in deps:
                    dep_dest = dep
                    dep_is_header = dep.endswith(utils.header_ext[lang])
                    ext = (utils.file_ext[lang] if not dep_is_header
                           else utils.header_ext[lang])
                    if change_extension and not dep.endswith(ext):
                        dep_dest = dep[:dep.rfind('.')] + ext
                    shutil.copyfile(os.path.join(scan_path, dep),
                                    os.path.join(out_path, dep_dest))
                    if not dep_is_header:
                        files.append(os.path.join(out_path, dep_dest))
                return files

            scan = os.path.join(script_dir, os.pardir, 'kernel_utils', lang)
            files += __copy_deps(lang, scan, build_dir, nfilt='.py')
            scan = os.path.join(script_dir, os.pardir, 'kernel_utils', 'common')
            files += __copy_deps(host_langs[lang], scan, build_dir,
                                 change_extension=False, ffilt='memcpy_2d')

            # build
            toolchain = get_toolchain(lang)
            obj_files = compile(
                lang, toolchain, files, source_dir=build_dir, obj_dir=obj_dir)
            lib = link(toolchain, obj_files, 'memory_test', lib_dir=lib_dir)
            # and run
            subprocess.check_call(lib)
