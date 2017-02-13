"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
import re
from argparse import ArgumentParser
import multiprocessing
import shutil
from collections import defaultdict, OrderedDict

from string import Template

# Related modules
import numpy as np

try:
    import cantera as ct
    from cantera import ck2cti
except ImportError:
    print('Error: Cantera must be installed.')
    raise

try:
    from optionloop import OptionLoop
except ImportError:
    print('Error: optionloop must be installed.')
    raise

# Local imports
from .. import utils
from ..core.create_jacobian import create_jacobian
from ..libgen import (generate_library, libs, compiler, file_struct,
                      get_cuda_path, flags, get_file_list
                      )
from .. import site_conf as site

from . import data_bin_writer as dbw
from .. get_test_matrix import get_test_matrix

STATIC = False
"""bool: CUDA only works for static libraries"""

def check_step_file(filename, steplist):
    """Checks file for existing data, returns number of runs left

    Parameters
    ----------
    filename : str
        Name of file with data
    steplist : list of int
        List of different numbers of steps

    Returns
    -------
    runs : dict
        Dictionary with number of runs left for each step

    """
    #checks file for existing data
    #and returns number of runs left to do
    #for each # of does in steplist
    runs = {}
    for step in steplist:
        runs[step] = 0
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        for line in lines:
            try:
                vals = line.split(',')
                if len(vals) == 2:
                    vals = [float(v) for v in vals]
                    runs[vals[0]] += 1
            except:
                pass
        return runs
    except:
        return runs


def check_file(filename):
    """Checks file for existing data, returns number of completed runs

    Parameters
    ----------
    filename : str
        Name of file with data

    Returns
    -------
    num_completed : int
        Number of completed runs

    """
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        num_completed = 0
        to_find = 4
        for line in lines:
            try:
                vals = line.split(',')
                if len(vals) == to_find:
                    i = int(vals[0])
                    f = float(vals[1])
                    f2 = float(vals[2])
                    f3 = float(vals[3])
                    num_completed += 1
            except:
                pass
        return num_completed
    except:
        return 0


def getf(x):
    return os.path.basename(x)


def cmd_link(lang, shared):
    """Return linker command.

    Parameters
    ----------
    lang : {'icc', 'c', 'cuda'}
        Programming language
    shared : bool
        ``True`` if shared

    Returns
    -------
    cmd : list of `str`
        List with linker command

    """
    cmd = None
    if lang == 'opencl':
        cmd = ['gcc']
    elif lang == 'c':
        cmd = ['gcc']
    elif lang == 'cuda':
        cmd = ['nvcc'] if not shared else ['g++']
    else:
        print('Lang must be one of {opecl, c}')
        raise
    return cmd


exceptions = ['-xc']
def linker(lang, test_dir, filelist, platform=''):
    args = cmd_link(lang, not STATIC)
    args.extend([x for x in flags[lang] if x not in exceptions])
    args.extend([os.path.join(test_dir, getf(f) + '.o') for f in filelist])
    args.extend(['-o', os.path.join(test_dir, 'speedtest')])
    args.extend(libs[lang])
    rpath = ''
    if lang == 'opencl':
        rpath = next(x for x in site.CL_PATHS if
            x in platform.lower())
        if rpath:
            rpath = site.CL_PATHS[rpath]
            args.extend(['-Wl,-rpath', rpath])
            args.extend(['-L', rpath])

    args.append('-lm')

    try:
        print(' '.join(args))
        subprocess.check_call(args)
    except subprocess.CalledProcessError:
        print('Error: linking of test program failed.')
        sys.exit(1)


def performance_tester(home, work_dir):
    """Runs performance testing for pyJac

    Parameters
    ----------
    home : str
        Directory of source code files
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """
    build_dir = 'out'
    test_dir = 'test'
    work_dir = os.path.abspath(work_dir)

    mechanism_list, ocl_params, max_vec_width = get_test_matrix(work_dir)

    if len(mechanism_list) == 0:
        print('No mechanisms found for performance testing in '
              '{}, exiting...'.format(work_dir)
              )
        sys.exit(-1)

    repeats = 10

    for mech_name, mech_info in sorted(mechanism_list.items(),
                                       key=lambda x:x[1]['ns']
                                       ):
        #get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))

        #ensure directory structure is valid
        this_dir = os.path.join(work_dir, mech_name)
        this_dir = os.path.abspath(this_dir)
        os.chdir(this_dir)
        my_build = os.path.join(this_dir, build_dir)
        my_test = os.path.join(this_dir, test_dir)
        subprocess.check_call(['mkdir', '-p', my_build])
        subprocess.check_call(['mkdir', '-p', my_test])

        current_data_order = None

        the_path = os.getcwd()
        first_run = True
        op = OptionLoop(ocl_params, lambda: False)

        for i, state in enumerate(op):
            lang = state['lang']
            vecsize = state['vecsize']
            order = state['order']
            wide = state['wide']
            deep = state['deep']
            platform = state['platform']
            rate_spec = state['rate_spec']
            split_kernels = state['split_kernels']
            num_cores = state['num_cores']
            if not deep and not wide and vecsize != max_vec_width:
                continue #this is simple parallelization, don't need vector size

            if rate_spec == 'fixed' and split_kernels:
                continue #not a thing!

            if order != current_data_order:
                #rewrite data to file in 'C' order
                num_conditions = dbw.write(os.path.join(work_dir, mech_name))

            #figure out the number of steps
            step_size = max_vec_width
            while step_size < num_conditions:
                if step_size * 2 >= num_conditions:
                    break
                step_size *= 2

            temp_lang = 'c'
            data_output = ('{}_{}_{}_{}_{}_{}_{}_{}'.format(lang, vecsize, order,
                            'w' if wide else 'd' if deep else 'par',
                            platform, rate_spec, 'split' if split_kernels else 'single',
                            num_cores
                            ) +
                           '_output.txt'
                           )

            data_output = os.path.join(the_path, data_output)
            num_completed = check_file(data_output)
            todo = {step_size: repeats - num_completed}
            if not any(todo[x] > 0 for x in todo):
                continue

            try:
                create_jacobian(lang,
                    mech_name=mech_info['mech'],
                    vector_size=vecsize,
                    wide=wide,
                    deep=deep,
                    data_order=order,
                    build_path=my_build,
                    skip_jac=True,
                    auto_diff=False,
                    platform=platform,
                    data_filename=os.path.join(work_dir, mech_name, 'data.bin'),
                    split_rate_kernels=split_kernels,
                    rate_specialization=rate_spec,
                    split_rop_net_kernels=split_kernels
                    )
            except:
                print('generation failed...')
                print(i, state)
                print()
                print()
                continue


            #get file lists
            i_dirs, files = get_file_list(build_dir, lang)

            structs = [file_struct(lang, lang, f, i_dirs,
               [], my_build, my_test, not STATIC) for f in files]

            pool = multiprocessing.Pool()
            results = pool.map(compiler, structs)
            pool.close()
            pool.join()
            if any(r == -1 for r in results):
               sys.exit(-1)

            linker(lang, my_test, files, platform)

            with open(data_output, 'a+') as file:
                for stepsize in todo:
                    for i in range(todo[stepsize]):
                        print(i, "/", todo[stepsize])
                        subprocess.check_call(
                            [os.path.join(the_path,
                            my_test, 'speedtest'),
                            str(stepsize), str(num_cores)], stdout=file
                            )
