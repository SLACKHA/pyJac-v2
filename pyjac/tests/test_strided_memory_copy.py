"""
Tests for the memory manager for code-generation

Copyright Nicholas Curtis 2018
"""
import shutil
import os
from string import Template
import subprocess
from collections import OrderedDict

import pyopencl as cl
from parameterized import parameterized
import numpy as np
import loopy as lp
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa

from pyjac import utils
from pyjac.core import array_creator as arc
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core.array_creator import array_splitter
from pyjac.kernel_utils.kernel_gen import CallgenResult
from pyjac.kernel_utils.memory_tools import DeviceMemoryType, get_memory, \
    DeviceNamer, HostNamer
from pyjac.kernel_utils.memory_manager import memory_manager, host_langs
from pyjac.libgen.libgen import compiler, file_struct, libgen
from pyjac.tests import build_dir, obj_dir, lib_dir, script_dir
from pyjac.tests.test_utils import clean_dir
from pyjac.tests.test_utils import OptionLoopWrapper, get_test_langs


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


type_map = {}
type_map[lp.to_loopy_type(np.float64)] = 'double'
type_map[lp.to_loopy_type(np.int32)] = 'int'
type_map[lp.to_loopy_type(np.int64)] = 'long int'


def test_memory_tools_alloc():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map)
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
            assert (('CL_MEM_ALLOC_HOST_PTR' in mem.alloc(True, a1)) ==
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
                                type_map=type_map)
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
                                type_map=type_map)
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


def test_memory_tools_memset():
    wrapper = __test_cases()
    for opts in wrapper:
        # create a dummy callgen
        callgen = CallgenResult(order=opts.order, lang=opts.lang,
                                dev_mem_type=wrapper.state['dev_mem_type'],
                                type_map=type_map)
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
                assert ('(int*)clEnqueueMapBuffer(queue, a1, CL_TRUE, CL_MAP_WRITE, '
                        '0, 10 * per_run * sizeof(int), 0, NULL, NULL, &return_code)'
                        ) in dev
                assert ('check_err(clEnqueueUnmapMemObject(queue, a1, temp_i, 0, '
                        'NULL, NULL));') in dev

                # check namer
                mem2 = get_memory(callgen, device_namer=DeviceNamer('data'))
                dev = mem2.memset(True, a1)
                assert ', data->d_a1, ' in dev
                assert 'data->d_temp_i = ' in dev
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
                                type_map=type_map)
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
                assert ('d_temp_i = (int*)clEnqueueMapBuffer(queue, d_a1, CL_TRUE, '
                        'CL_MAP_WRITE, 0, problem_size * sizeof(int), 0, NULL, '
                        'NULL, &return_code);') in dev
                assert 'memcpy(d_temp_i, h_a1, problem_size * sizeof(int));' in dev
            else:
                # mapped
                assert ('clEnqueueWriteBuffer(queue, d_a1, CL_TRUE, 0, '
                        'problem_size * sizeof(int), &h_a1, 0, NULL, NULL)') in dev

            dev = mem.copy(False, d3, offset='test', num_ics_this_run='test2')
            if wrapper.state['dev_mem_type'] == DeviceMemoryType.pinned:
                assert ('d_temp_d = (double*)clEnqueueMapBuffer(queue, d_d3, '
                        'CL_TRUE, CL_MAP_READ, 0, 100 * per_run * sizeof(double)'
                        ', 0, NULL, NULL, &return_code);') in dev
                if opts.order == 'C':
                    assert ('memcpy(&h_d3[test * 100], d_temp_d, '
                            '100 * test2 * sizeof(double));') in dev
                else:
                    assert ('memcpy2D_out(h_d3, d_temp_d, '
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
                    assert ('clEnqueueReadBufferRect(queue, d_d3, CL_TRUE, '
                            '(size_t[]) {0, 0, 0}, '
                            '(size_t[]) {test * sizeof(double), 0, 0}, '
                            '(size_t[]) {test2 * sizeof(double), 100, 1}, '
                            'per_run * sizeof(double), 0, '
                            'problem_size * sizeof(double), 0, h_d3, 0, NULL, NULL)'
                            ) in dev
        else:
            raise NotImplementedError


@parameterized(__test_cases)
def test_strided_copy(state):
    lang = state['lang']
    order = state['order']
    depth = state['depth']
    width = state['width']

    # cleanup
    clean_dir(build_dir)
    clean_dir(obj_dir)
    clean_dir(lib_dir)

    # create
    utils.create_dir(build_dir)
    utils.create_dir(obj_dir)
    utils.create_dir(lib_dir)

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
    lp_arrays = [lp.GlobalArg('a{}'.format(i), shape=('problem_size',) + a.shape[1:],
                              order=order, dtype=(arrays + const)[i].dtype)
                 for i, a in enumerate(arrays)] + \
                [lp.TemporaryVariable('a{}'.format(i + len(arrays)), dtype=dtype,
                 order=order, initializer=const[i], read_only=True,
                 shape=const[i].shape) for i in range(len(const))]
    const = lp_arrays[len(arrays):]

    dtype = 'double'

    # create array splitter
    opts = type('', (object,), state)
    asplit = array_splitter(opts)

    # split numpy
    arrays = asplit.split_numpy_arrays(arrays)
    # make dummy knl
    knl = lp.make_kernel('{[i]: 0 <= i <= 1}',
                         """
                            if i > 1
                                a0[i, i] = 0
                                a1[i, i] = 0
                                a2[i, i] = 0
                                a3[i, i, i] = 0
                                a4[i] = 0
                                <> k = a5[i]
                            end
                         """, lp_arrays)
    # split loopy
    lp_arrays = asplit.split_loopy_arrays(knl).args

    # now create a simple library
    mem = memory_manager(opts.lang, opts.order, asplit,
                         dev_type=state['device_type'],
                         strided_c_copy=lang == 'c')
    mem.add_arrays([x for x in lp_arrays],
                   in_arrays=[x.name for x in lp_arrays if x not in const],
                   out_arrays=[x.name for x in lp_arrays if x not in const],
                   host_constants=const)

    # create "kernel"
    size_type = 'int'
    lang_headers = []
    if lang == 'opencl':
        lang_headers.extend([
                        '#include "memcpy_2d.oclh"',
                        '#include "vectorization.oclh"',
                        '#include <CL/cl.h>',
                        '#include "ocl_errorcheck.oclh"'])
        size_type = 'cl_uint'
    elif lang == 'c':
        lang_headers.extend([
            '#include "memcpy_2d.h"',
            '#include "error_check.h"'])

    # kernel must copy in and out, using the mem_manager's format
    knl = Template("""
    for (size_t offset = 0; offset < problem_size; offset += per_run)
    {
        ${type} this_run = problem_size - offset < per_run ? \
            problem_size - offset : per_run;
        /* Memory Transfers into the kernel, if any */
        ${mem_transfers_in}

        /* Memory Transfers out */
        ${mem_transfers_out}
    }
    """).safe_substitute(type=size_type,
                         mem_transfers_in=mem._mem_transfers(
                            to_device=True, host_postfix='_save'),
                         mem_transfers_out=mem.get_mem_transfers_out(),
                         problem_size=ics
                         )

    # create the host memory allocations
    host_names = ['h_' + arr.name for arr in lp_arrays]
    host_allocs = mem.get_mem_allocs(True, host_postfix='')

    # device memory allocations
    device_allocs = mem.get_mem_allocs(False)

    # copy to save for test
    host_name_saves = ['h_' + a.name + '_save' for a in lp_arrays]
    host_const_allocs = mem.get_host_constants()
    host_copies = [Template(
        """
        ${type} ${save} [${size}] = {${vals}};
        memset(${host}, 0, ${size} * sizeof(${type}));
        """).safe_substitute(
            save='h_' + lp_arrays[i].name + '_save',
            host='h_' + lp_arrays[i].name,
            size=arrays[i].size,
            vals=', '.join([str(x) for x in arrays[i].flatten()]),
            type=dtype)
        for i in range(len(arrays))]

    # and finally checks
    check_template = Template("""
        for(int i = 0; i < ${size}; ++i)
        {
            assert(${host}[i] == ${save}[i]);
        }
    """)
    checks = [check_template.safe_substitute(host=host_names[i],
                                             save=host_name_saves[i],
                                             size=arrays[i].size)
              for i in range(len(arrays))]

    # and preambles
    ocl_preamble = """
    double* temp_d;
    int* temp_i;
    // create a context / queue
    int lim = 10;
    cl_uint num_platforms;
    cl_uint num_devices;
    cl_platform_id platform [lim];
    cl_device_id device [lim];
    cl_int return_code;
    cl_context context;
    cl_command_queue queue;
    check_err(clGetPlatformIDs(lim, platform, &num_platforms));
    for (int i = 0; i < num_platforms; ++i)
    {
        check_err(clGetDeviceIDs(platform[i], CL_DEVICE_TYPE_ALL, lim, device,
            &num_devices));
        if(num_devices > 0)
            break;
    }
    context = clCreateContext(NULL, 1, &device[0], NULL, NULL, &return_code);
    check_err(return_code);

    //create queue
    queue = clCreateCommandQueue(context, device[0], 0, &return_code);
    check_err(return_code);
    """
    preamble = ''
    if lang == 'opencl':
        preamble = ocl_preamble

    end = ''
    if lang == 'opencl':
        end = """
        check_err(clFlush(queue));
        check_err(clReleaseCommandQueue(queue));
        check_err(clReleaseContext(context));
    """

    file_src = Template("""
${lang_headers}
#include <stdlib.h>
#include <string.h>
#include <assert.h>


void main()
{
    ${preamble}

    double zero [${max_dim}] = {0};

    ${size_type} problem_size = ${problem_size};
    ${size_type} per_run = ${max_per_run};

    ${host_allocs}
    ${host_const_allocs}
    ${mem_declares}
    ${device_allocs}

    ${mem_saves}

    ${host_constant_copy}

    ${knl}

    ${checks}

    ${end}

    exit(0);
}
    """).safe_substitute(lang_headers='\n'.join(lang_headers),
                         mem_declares=mem.get_defns(),
                         host_allocs=host_allocs,
                         host_const_allocs=host_const_allocs,
                         device_allocs=device_allocs,
                         mem_saves='\n'.join(host_copies),
                         host_constant_copy=mem.get_host_constants_in(),
                         checks='\n'.join(checks),
                         knl=knl,
                         preamble=preamble,
                         end=end,
                         size_type=size_type,
                         max_per_run=max_per_run,
                         problem_size=ics,
                         max_dim=max([x.size for x in arrays]))

    # write file
    fname = os.path.join(build_dir, 'test' + utils.file_ext[lang])
    with open(fname, 'w') as file:
        file.write(file_src)
    files = [fname]

    # write aux
    write_aux(build_dir, opts, [], [])

    # copy any deps
    def __copy_deps(lang, scan_path, out_path, change_extension=True,
                    ffilt=None):
        deps = [x for x in os.listdir(scan_path) if os.path.isfile(
            os.path.join(scan_path, x)) and not x.endswith('.in')]
        if ffilt is not None:
            deps = [x for x in deps if ffilt in x]
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
    files += __copy_deps(lang, scan, build_dir)
    scan = os.path.join(script_dir, os.pardir, 'kernel_utils', 'common')
    files += __copy_deps(host_langs[lang], scan, build_dir, change_extension=False,
                         ffilt='memcpy_2d')

    # build
    files = [file_struct(lang, lang, f[:f.rindex('.')], [build_dir], [],
                         build_dir, obj_dir, True, True) for f in files]
    assert not any(compiler(x) for x in files)
    lib = libgen(lang, obj_dir, lib_dir, [x.filename for x in files],
                 True, False, True)
    lib = os.path.join(lib_dir, lib)
    # and run
    subprocess.check_call(lib)
