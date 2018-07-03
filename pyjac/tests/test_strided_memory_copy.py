# tests the memory manager's copy abilities
import shutil
import os
from string import Template
import subprocess
from collections import OrderedDict

import pyopencl as cl
from parameterized import parameterized, param
import numpy as np
import loopy as lp
from optionloop import OptionLoop
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa

from pyjac.libgen.libgen import compiler, file_struct, libgen
from pyjac.kernel_utils.memory_manager import memory_manager, host_langs
from pyjac.tests import build_dir, obj_dir, lib_dir, script_dir
from pyjac.core.array_creator import array_splitter
from pyjac.tests.test_utils import clean_dir
from pyjac import utils
from pyjac.core.mech_auxiliary import write_aux


def __test_cases():
    for state in OptionLoop(OrderedDict(
            [('lang', ['opencl', 'c']), ('order', ['C', 'F']),
             ('width', [4, None]), ('depth', [4, None]),
             ('device_type', (cl.device_type.CPU, cl.device_type.GPU, None)),
             ('is_simd', [True, False])])):
        if state['depth'] and state['width']:
            continue
        elif (state['depth'] is not None or state['width'] is not None) \
                and state['lang'] == 'c':
            continue
        elif (state['lang'] == 'c' and state['device_type'] is not None):
            continue
        elif state['is_simd'] and (
                state['lang'] == 'c' or
                state['device_type'] != cl.device_type.CPU or
                not state['width']):
            continue
        yield param(state)


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
