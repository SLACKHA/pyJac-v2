"""Module used to create a shared/static library from pyJac files.
"""
from __future__ import print_function

import os
import subprocess
import sys
import multiprocessing
import platform
import logging

from .. import utils
from .. import siteconf as site
from enum import Enum
from ..core.exceptions import CompilationError


class build_type(Enum):
    chem_utils = 1,
    species_rates = 2,
    jacobian = 3

    def __int__(self):
        return self.value


def lib_ext(shared):
    """Returns the appropriate library extension based on the shared flag"""
    return '.a' if not shared else '.so'


cmd_compile = dict(c='gcc',
                   opencl='gcc',
                   # cuda='nvcc'
                   )


def cmd_lib(lang, shared):
    """Returns the appropriate compilation command for creation of the library based
    on the language and shared flag"""
    if lang in ['c', 'opencl']:
        return ['ar', 'rcs'] if not shared else ['gcc', '-shared']
    # elif lang == 'cuda':
    #    return ['nvcc', '-lib'] if not shared else ['nvcc', '-shared']


includes = dict(c=['/usr/local/include/'],
                opencl=site.CL_INC_DIR
                )

shared_flags = dict(c=['-fPIC'],
                    opencl=['-fPIC'],
                    cuda=['-Xcompiler', '"-fPIC"']
                    )
shared_exec_flags = dict(c=['-pie', '-Wl,-E'],
                         opencl=['-pie', '-Wl,-E'])

opt_flags = ['-O3']
debug_flags = ['-O0', '-g']
compile_flags = debug_flags if 'PYJAC_DEBUG' in os.environ else opt_flags

flags = dict(c=site.CC_FLAGS + compile_flags + ['-fopenmp', '-std=c99'],
             opencl=site.CC_FLAGS + compile_flags + ['-xc', '-std=c99'])

libs = dict(c=['-lm', '-fopenmp'],
            opencl=['-l' + x for x in site.CL_LIBNAME]
            )


def which(file):
    """A substitute for the `which` command, searches the PATH for
    a given file"""
    for path in os.environ["PATH"].split(os.pathsep):
        if os.path.exists(os.path.join(path, file)):
            return os.path.join(path, file)

    return None


def get_cuda_path():
    """Returns location of CUDA (nvcc) on the system.

    Parameters
    ----------
    None

    Returns
    -------
    cuda_path : str
        Path where CUDA (nvcc) is found on the system.

    """
    cuda_path = which('nvcc')
    logger = logging.getLogger(__name__)
    if cuda_path is None:
        logger.warn('nvcc not found!')
        return []

    sixtyfourbit = platform.architecture()[0] == '64bit'
    cuda_path = os.path.dirname(os.path.dirname(cuda_path))
    cuda_path = os.path.join(cuda_path,
                             'lib{}'.format('64' if sixtyfourbit else '')
                             )
    return [cuda_path]


lib_dirs = dict(c=[],
                # cuda=get_cuda_path(),
                opencl=site.CL_LIB_DIR)
run_dirs = dict(c=[],
                # cuda=get_cuda_path(),
                opencl=site.CL_LIB_DIR)


def compiler(fstruct):
    """Given a file structure, this method will compile the source file for the
    language and options specified

    Parameters
    ----------
    fstruct : `file_struct`
        An information struct that holds the various compilation options

    Returns
    -------
    success : int
        0 if the compilation process was sucessful, -1 otherwise

    Notes
    -----
    Designed to work with a multiprocess compilation workflow
    """
    args = [cmd_compile[fstruct.build_lang]]
    if fstruct.auto_diff:
        args = ['g++']
    args.extend(flags[fstruct.build_lang])
    if fstruct.auto_diff:
        args = [x for x in args if 'std=' not in x]

    # always use fPIC in case we're building wrapper
    args.extend(shared_flags[fstruct.build_lang])
    if fstruct.as_executable:
        args.extend(shared_exec_flags[fstruct.build_lang])
    # and any other flags
    args.extend(fstruct.args)
    # includes
    include = ['-I{}'.format(d) for d in fstruct.i_dirs +
               includes[fstruct.build_lang]
               ]
    args.extend(include)
    args.extend([
        '-{}c'.format('d' if fstruct.lang == 'cuda' else ''),
        os.path.join(fstruct.source_dir, fstruct.filename +
                     utils.file_ext[fstruct.build_lang]
                     ),
        '-o', os.path.join(fstruct.obj_dir,
                           os.path.basename(fstruct.filename) + '.o')
    ])
    args = [val for val in args if val.strip()]
    try:
        print(' '.join(args))
        subprocess.check_call(args)
    except OSError:
        logger = logging.getLogger(__name__)
        logger.error(
            'Compiler {} not found, generation of pyjac library failed.'.format(
                args[0]))
        return -1
    except subprocess.CalledProcessError as exc:
        logger = logging.getLogger(__name__)
        logger.error('Error: compilation failed for file {} with error:{}'.format(
            fstruct.filename + utils.file_ext[fstruct.build_lang],
            exc.output))
        return exc.returncode
    return 0


def libgen(lang, obj_dir, out_dir, filelist, shared, auto_diff, as_executable):
    """Create a library from a list of compiled files

    Parameters
    ----------
    Parameters
    ----------
    obj_dir : str
        Path with object files
    out_dir : str
        Path to place the library in
    lang : {'c', 'cuda'}
        Programming language
    filelist : List of `str`
        The list of object files to include in the library
    auto_diff : Optional[bool]
        Optional; if ``True``, include autodifferentiation

    """
    command = cmd_lib(lang, shared)

    if lang == 'opencl':
        desc = 'ocl'
    elif lang == 'c':
        if auto_diff:
            desc = 'ad'
        else:
            desc = 'c'

    libname = 'lib{}_pyjac'.format(desc)

    # remove the old library
    if os.path.exists(os.path.join(out_dir, libname + lib_ext(shared))):
        os.remove(os.path.join(out_dir, libname + lib_ext(shared)))
    if os.path.exists(os.path.join(out_dir, libname + lib_ext(not shared))):
        os.remove(os.path.join(out_dir, libname + lib_ext(not shared)))

    # add optimization / debug flags
    command.extend(compile_flags)

    command += lib_ext(shared)

    if not shared and lang != 'cuda':
        command += [os.path.join(out_dir, libname)]

    # add the files
    command.extend(
        [os.path.join(obj_dir, os.path.basename(f) + '.o') for f in filelist])

    if shared and not as_executable:
        command.extend(shared_flags[lang])
    elif as_executable:
        command.extend(shared_exec_flags[lang])

    if shared or lang == 'cuda':
        command += ['-o']
        command += [os.path.join(out_dir, libname)]

        command += ['-L{}'.format(path) for path in lib_dirs[lang]]
        command.extend(libs[lang])

    try:
        print(' '.join(command))
        subprocess.check_call(command)
    except OSError:
        logger = logging.getLogger(__name__)
        logging.error(
            'Compiler {} not found, generation of pyjac library failed.'.format(
                command[0]))
        sys.exit(-1)
    except subprocess.CalledProcessError as exc:
        logger = logging.getLogger(__name__)
        logger.error('Generation of pyjac library failed with error: {}'.format(
            exc.output))
        sys.exit(exc.returncode)

    return libname


class file_struct(object):

    """A simple structure designed to enable multiprocess compilation
    """

    def __init__(self, lang, build_lang, filename, i_dirs, args,
                 source_dir, obj_dir, shared, as_executable):
        """
        Parameters
        ----------
        lang : str
            Compiler to use
        build_lang : {'c', 'cuda'}
            Programming language
        file_name : str
            The file to compile
        i_dirs : List of str
            List of include directorys for compilation
        args : List of str
            List of additional arguements
        source_dir : str
            The directory the file is located in
        obj_dir : str
            The directory to place the compiled object file in
        shared : bool
            If true, this is creating a shared library
        as_executable: bool
            If true, this is a shared library that is also executable
        """

        self.lang = lang
        self.build_lang = build_lang
        self.filename = filename
        self.i_dirs = i_dirs
        self.args = args
        self.source_dir = source_dir
        self.obj_dir = obj_dir
        self.shared = shared
        self.auto_diff = False
        self.as_executable = as_executable


def get_file_list(source_dir, lang, btype):
    """

    Parameters
    ----------
    source_dir : str
        Path with source files
    lang : {'c', 'cuda'}
        Programming language
    btype: :class:`build_type`
        The type of library being built

    Returns
    -------
    i_dirs : list of `str`
        List of include directories
    files : list of `str`
        List of files

    """
    i_dirs = [source_dir]
    files = ['read_initial_conditions', 'timer']

    # look for right code in the directory
    file_base = 'jacobian_kernel'
    if btype == build_type.species_rates:
        file_base = 'species_rates_kernel'
    elif btype == build_type.chem_utils:
        file_base = 'chem_utils_kernel'

    if lang == 'opencl':
        files += [file_base + x for x in ['_compiler', '_main']]
        files += ['ocl_errorcheck']
    elif lang == 'c':
        files += [file_base + x for x in ['', '_main']]
        files += ['error_check']

    flists = []
    for flist in flists:
        try:
            with open(os.path.join(source_dir, flist[0], flist[1].format(lang)),
                      'r') as file:
                vals = file.readline().strip().split(' ')
                vals = [os.path.join(flist[0],
                                     f[:f.index(utils.file_ext[lang])]) for f in vals
                        ]
                files += vals
                i_dirs.append(os.path.join(source_dir, flist[0]))
        except:
            pass
    if lang == 'cuda':
        files += ['gpu_memory']

    return i_dirs, files


def generate_library(lang, source_dir, obj_dir=None, out_dir=None, shared=None,
                     btype=build_type.jacobian, as_executable=False):
    """Generate shared/static library for pyJac files.

    Parameters
    ----------
    lang : {'c', 'cuda'}
        Programming language
    source_dir : str
        Path of folder with pyJac files
    obj_dir : Optional[str]
        Optional; path of folder to store generated object files
    shared : bool
        If ``True``, generate shared library (vs. static)
    finite_difference : Optional[bool]
        If ``True``, include finite differences
    auto_diff : bool
        If ``True``, include autodifferentiation
    btype: :class:`build_type` [build_type.jacobian]
        The type of library being built
    as_executable: bool [False]
        If true, the generated library should use the '-fPIE' flag (or equivalent)
        to be executable

    Returns
    -------
    Location of generated library

    """
    # check lang
    logger = logging.getLogger(__name__)
    if lang not in flags.keys():
        logger.error('Cannot generate library for unknown language {}'.format(lang))
        sys.exit(-1)

    shared = shared and lang != 'cuda'

    if lang == 'cuda' and shared:
        logger.error('CUDA does not support linking of shared device libraries.')
        sys.exit(-1)

    if not shared and as_executable:
        logger.error('Can only make an executable out of a shared library')
        sys.exit(-1)

    build_lang = lang if lang != 'icc' else 'c'

    source_dir = os.path.abspath(os.path.abspath(source_dir))
    if obj_dir is None:
        obj_dir = os.path.join(os.getcwd(), 'obj')
    else:
        obj_dir = os.path.abspath(os.path.abspath(obj_dir))
    if not os.path.exists(obj_dir):
        os.makedirs(obj_dir)
    if out_dir is None:
        out_dir = os.getcwd()
    else:
        out_dir = os.path.abspath(out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    obj_dir = os.path.abspath(obj_dir)
    out_dir = os.path.abspath(out_dir)

    # get file lists
    i_dirs, files = get_file_list(source_dir, build_lang, btype)

    # Compile generated source code
    structs = [file_struct(lang, build_lang, f, i_dirs, [],
                           source_dir, obj_dir, shared, as_executable)
               for f in files]

    pool = multiprocessing.Pool()
    results = pool.map(compiler, structs)
    pool.close()
    pool.join()
    if any(r != 0 for r in results):
        failures = [i for i, r in enumerate(results) if r != -1]
        raise CompilationError([structs[i].filename for i in failures])

    libname = libgen(lang, obj_dir, out_dir, files, shared, False, as_executable)
    return os.path.join(out_dir, libname)
