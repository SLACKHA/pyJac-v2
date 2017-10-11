# tests the memory manager's copy abilities

from parameterized import parameterized, param
from ..libgen import compiler, file_struct
from ..libgen.libgen import libgen
from ..kernel_utils.memory_manager import memory_manager
from . import build_dir, obj_dir, lib_dir, script_dir
import numpy as np
import loopy as lp
from ..core.array_creator import array_splitter
from .test_utils import clean_dir
from .. import utils
import os
from string import Template
import subprocess
from optionloop import OptionLoop
from collections import OrderedDict
from ..core.mech_auxiliary import write_aux
import shutil


def __test_cases():
    for state in OptionLoop(OrderedDict(
            [('lang', ['opencl', 'c']), ('order', ['C', 'F']),
             ('width', [4, None]), ('depth', [4, None])])):
        yield param(state)


@parameterized(__test_cases)
def test_strided_copy(state):
    lang = state['lang']
    order = state['order']
    depth = state['depth']
    width = state['width']
    if depth and width:
        return True

    # cleanup
    clean_dir(build_dir)
    clean_dir(obj_dir)
    clean_dir(lib_dir)

    # create
    utils.create_dir(build_dir)
    utils.create_dir(obj_dir)
    utils.create_dir(lib_dir)

    # number of ics, should be divisibly by depth and width
    max_per_run = 10
    ics = max_per_run * 16
    if depth:
        assert ics % depth == 0
    if width:
        assert ics % width == 0
    dtype = np.dtype('float64')

    # create test arrays
    def __create(shape):
        if not isinstance(shape, tuple):
            shape = (shape,)
        shape = (ics,) + shape
        return np.array(np.random.rand(*shape), dtype=dtype, order=order)
    arrays = [__create(16), __create(10), __create(20), __create((20, 20))]
    lp_arrays = [lp.GlobalArg('a{}'.format(i), shape=('problem_size',) + a.shape[1:],
                              order=order, dtype=arrays[i].dtype)
                 for i, a in enumerate(arrays)]

    dtype = 'double'

    # create array splitter
    opts = type('', (object,), {'width': width, 'depth': depth, 'order': order,
                                'lang': lang})
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
                            end
                         """, lp_arrays)
    # split loopy
    lp_arrays = asplit.split_loopy_arrays(knl).args

    # now create a simple library
    mem = memory_manager(opts.lang, opts.order)
    mem.add_arrays([x for x in lp_arrays], in_arrays=[x.name for x in lp_arrays],
                   out_arrays=[x.name for x in lp_arrays])

    # create "kernel"
    lang_headers = ['#include "memcpy_2d.h"']
    size_type = 'int'
    if lang == 'opencl':
        lang_headers = ['#include "vectorization.oclh"',
                        '#include <CL/cl.h>',
                        '#include "ocl_errorcheck.oclh"']
        size_type = 'cl_uint'

    # kernel must copy in and out, using the mem_manager's format
    knl = Template("""
    ${type} problem_size = ${problem_size};
    ${type} per_run = ${max_per_run} < problem_size ? ${max_per_run} :
        ${problem_size};
    for (size_t offset = 0; offset < problem_size; offset += per_run)
    {
        ${type} this_run = problem_size - offset < per_run ? \
            problem_size - offset : per_run;
        /* Memory Transfers into the kernel, if any */
        ${mem_transfers_in}

        /* Memory Transfers out */
        ${mem_transfers_out}
    }
    """).safe_substitute(max_per_run=max_per_run,
                         type=size_type,
                         mem_transfers_in=mem._mem_transfers(
                            to_device=True, host_postfix='_save'),
                         mem_transfers_out=mem.get_mem_transfers_out(),
                         problem_size=ics
                         )

    # create the host memory allocations
    host_names = ['h_' + a.name for a in lp_arrays]
    host_allocs = [Template("${type} ${name}[${size}] = {${vals}};").safe_substitute(
        type=dtype, name=host_names[i], size=arrays[i].size, vals=', '.join([
            '{}'.format(x) for x in arrays[i].flatten('K')]))
            for i in range(len(arrays))]

    # device memory allocations
    device_names = ['d_' + a.name for a in lp_arrays]
    device_allocs = Template("${type} ${name}[${size}];")
    if lang == 'opencl':
        device_allocs = Template("""
        ${name} = clCreateBuffer(context, CL_MEM_READ_WRITE, ${size}, \
            NULL, &return_code);
        check_err(return_code);
        """)
    device_allocs = [device_allocs.safe_substitute(
        name=device_names[i], size=arrays[i].size,
        type=dtype) for i in range(len(arrays))]

    # copy to save for test
    host_name_saves = ['h_' + a.name + '_save' for a in lp_arrays]
    host_copies = [Template(
        """
        ${type} ${save} [${size}] = {0};
        memcpy(${save}, ${host}, ${size} * sizeof(${type}));
        memset(${host}, 0, ${size} * sizeof(${type}));
        """).safe_substitute(
            save=host_name_saves[i], host=host_names[i], size=arrays[i].size,
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

${mem_declares}

void main()
{
    ${preamble}

    ${host_allocs}

    ${device_allocs}

    ${mem_saves}

    ${knl}

    ${checks}

    ${end}

    exit(0);
}
    """).safe_substitute(lang_headers='\n'.join(lang_headers),
                         mem_declares=mem.get_defns(),
                         host_allocs='\n'.join(host_allocs),
                         device_allocs='\n'.join(device_allocs),
                         mem_saves='\n'.join(host_copies),
                         checks='\n'.join(checks),
                         knl=knl,
                         preamble=preamble,
                         end=end)

    # write file
    fname = os.path.join(build_dir, 'test' + utils.file_ext[lang])
    with open(fname, 'w') as file:
        file.write(file_src)
    files = [fname]

    # write aux
    write_aux(build_dir, opts, [], [])

    # copy any deps
    def __copy_deps(scan_path, out_path, change_extension=True):
        deps = [x for x in os.listdir(scan_path) if os.path.isfile(
            os.path.join(scan_path, x)) and not x.endswith('.in')]
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
    files += __copy_deps(scan, build_dir)

    # build
    files = [file_struct(lang, lang, f[:f.rindex('.')], [build_dir], [],
                         build_dir, obj_dir, True, True) for f in files]
    assert not any(compiler(x) for x in files)
    lib = libgen(lang, obj_dir, lib_dir, [x.filename for x in files],
                 True, False, True)
    lib = os.path.join(lib_dir, lib)
    # and run
    subprocess.check_call(lib)
