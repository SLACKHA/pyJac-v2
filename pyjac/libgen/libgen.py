"""Module used to create a shared/static library from pyJac files.
"""
from __future__ import print_function

import os
import logging
import six

from codepy.toolchain import GCCToolchain
from codepy import CompileError

from pyjac import utils
from pyjac import siteconf as site
from pyjac.core.enum_types import KernelType
from pyjac.core.exceptions import CompilationError, LinkingError, \
    LibraryGenerationError


def lib_ext(shared):
    """Returns the appropriate library extension based on the shared flag"""
    return '.a' if not shared else '.so'


cmd_compile = dict(c='g++',
                   opencl='g++',
                   # cuda='nvcc'
                   )

includes = dict(c=[],
                opencl=site.CL_INC_DIR
                )

shared_flags = dict(c=['-fPIC', '-shared'],
                    opencl=['-fPIC', '-shared']
                    )
shared_exec_flags = dict(c=['-pie', '-Wl,-E'],
                         opencl=['-pie', '-Wl,-E'])

opt_flags = ['-O3', '-mtune=native']
debug_flags = ['-O0', '-g']


flags = dict(c=site.CC_FLAGS + ['-fopenmp', '-std=c++11'],
             opencl=site.CC_FLAGS + ['-xc++', '-std=c++11'])
ldflags = dict(c=['-fopenmp'] + site.LDFLAGS,
               opencl=[] + site.LDFLAGS)
libs = dict(c=['m'],
            opencl=site.CL_LIBNAME[:]
            )

lib_dirs = dict(c=[],
                # cuda=get_cuda_path(),
                opencl=site.CL_LIB_DIR)
run_dirs = dict(c=[],
                # cuda=get_cuda_path(),
                opencl=site.CL_LIB_DIR)


def get_toolchain(lang, shared=True, executable=True, **kwargs):
    """
    Return a codepy :class:`Toolchain` to build / link pyJac files.

    Parameters
    ----------
    lang: str
        The language to build
    shared: bool [True]
        If true, build a shared library
    executable: bool [True]
        If true, build a _executable_ shared library; note: requires
        :param:`shared`=True
    **kwargs:
    """

    # compilation flags
    compile_flags = opt_flags
    from pyjac.utils import get_env_val
    # read debug flag from ENV or config
    if get_env_val('debug'):
        compile_flags = debug_flags

    # link flags
    linkflags = ldflags[lang]
    if shared and not executable:
        linkflags += shared_flags[lang]
        compile_flags += shared_flags[lang]
    elif executable:
        if not shared:
            logger = logging.getLogger(__name__)
            logger.error('Cannot create an executable non-shared library!')
            raise LibraryGenerationError()

        compile_flags += shared_flags[lang]
        linkflags += shared_exec_flags[lang]
    if run_dirs[lang]:
        for rdir in utils.listify(run_dirs[lang]):
            linkflags += ['-Wl,-rpath,{}'.format(rdir)]
    so_ext = lib_ext(shared)

    toolchain_args = {'cc': cmd_compile[lang][:],
                      'cflags': (flags[lang] + compile_flags)[:],
                      'ldflags': linkflags[:],
                      'include_dirs': includes[lang][:],
                      'library_dirs': lib_dirs[lang][:],
                      'libraries': libs[lang][:],
                      'so_ext': so_ext,
                      'o_ext': '.o',
                      'defines': [],
                      'undefines': []}

    # merge in user kwargs
    for k, v in six.iteritems(kwargs):
        if k not in toolchain_args or not toolchain_args[k]:
            # empty or user supplied only
            toolchain_args[k] = v
        elif isinstance(toolchain_args[k], list):
            # may simply append to the list
            v = utils.listify(v)
            toolchain_args[k] += v[:]
        else:
            # else, replace
            toolchain_args[k] = v

    return GCCToolchain(**toolchain_args)


def get_file_list(source_dir, lang, ktype, file_base=None):
    """

    Parameters
    ----------
    source_dir : str
        Path with source files
    lang : {'c', 'cuda'}
        Programming language
    ktype: :class:`KernelType`
        The type of library being built
    file_base : str [None]
        If :param:`ktype` == KernelType.dummy, use this as the base filename.

    Returns
    -------
    i_dirs : list of `str`
        List of include directories
    files : list of `str`
        List of files

    """
    i_dirs = [source_dir]
    files = ['read_initial_conditions', 'timer']

    # determine which files to compile
    deps = {KernelType.jacobian: [KernelType.species_rates, KernelType.chem_utils],
            KernelType.species_rates: [KernelType.chem_utils],
            KernelType.chem_utils: []}

    file_bases = {KernelType.jacobian: 'jacobian',
                  KernelType.species_rates: 'species_rates',
                  KernelType.chem_utils: 'chem_utils',
                  KernelType.dummy: file_base}

    if ktype == KernelType.dummy:
        assert file_base is not None
        pass

    modifiers = {'opencl': ['_compiler', '_main'],
                 'c': ['', '_main', '_driver']}

    # handle base kernel type
    files += [file_bases[ktype] + x for x in modifiers[lang]]

    # and add dependencies
    if lang != 'opencl' and ktype != KernelType.dummy:
        for dep in deps[ktype]:
            files += [file_bases[dep]]

    # error checking
    files += ['error_check']
    return i_dirs, files


def compile(lang, toolchain, files, source_dir='', obj_dir=''):
    """
    Compiles the source files with the given toolchain

    Parameters
    ----------
    lang: str
        The language to compile for, used to determine the file extension
    toolchain: :class:`codepy.Toolchain`
        The toolchain to build the files with
    files: list of str
        The list of source files.  If :param:`source_dir` is not specified, these
        should be a absolute path to the file.
    source_dir: str ['']
        If specified, the base directory the source files are located in
    obj_dir: str ['']
        If specified, place the object files in this directory

    Returns
    -------
    objs: list of str
        The compiled object files

    Raises
    ------
    CompilationError
    """

    extension = utils.file_ext[lang]
    obj_files = []
    for file in files:
        try:
            # get source file
            if not source_dir:
                file_base = os.path.basename(file[:file.index(extension)])
            else:
                file_base = file[:file.index(extension)]
                file = os.path.join(source_dir, file)

            # get object file
            obj_file = file_base + toolchain.o_ext
            if obj_dir:
                obj_file = os.path.join(obj_dir, obj_file)
            obj_files.append(obj_file)

            # compile
            toolchain.build_object(obj_files[-1], [file])
        except CompileError as e:
            logger = logging.getLogger(__name__)
            logger.error('Error compiling file: {}'.format(file))
            raise CompilationError(file)

    return obj_files


def link(toolchain, obj_files, libname, lib_dir=''):
    """
    Link the given object files into a library

    Parameters
    ----------
    toolchain: :class:`codepy.Toolchain`
        The toolchain to link the files with
    object_files: list of str
        The list of object files to link
    libname: str
        The output library name
    lib_dir: str ['']
        If specified, place the linked library in this directory

    Returns
    -------
    libname: str
        The full path to the library, this is just :param:`libname` if
        :param:`lib_dir` is unspecified

    Raises
    ------
    LinkingError
    """

    if lib_dir:
        libname = os.path.join(lib_dir, libname)

    # filter out language flags, if necessary, to avoid having the compiler thing
    # we are trying to compile the object files
    filtered = ['-xc++']
    if any(y in toolchain.cflags for y in filtered):
        toolchain = toolchain.copy(cflags=[
            x for x in toolchain.cflags if x not in filtered])

    try:
        toolchain.link_extension(libname, obj_files)
    except CompileError:
        logger = logging.getLogger(__name__)
        logger.error('Generation of pyjac library failed.')
        raise LinkingError(obj_files)

    return libname


def generate_library(lang, source_dir, obj_dir=None, out_dir=None, shared=None,
                     ktype=KernelType.jacobian, as_executable=False, **kwargs):
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
    ktype: :class:`KernelType` [KernelType.jacobian]
        The type of library being built
    as_executable: bool [False]
        If true, the generated library should use the '-fPIE' flag (or equivalent)
        to be executable

    Keyword Arguments
    -----------------
    file_base: str
        Used for creation of libraries for :param:`ktype`==KernelType.dummy -- the
        base filename (generator name) for this library

    Returns
    -------
    Location of generated library

    """
    # check lang
    logger = logging.getLogger(__name__)
    if lang not in flags.keys():
        logger.error('Cannot generate library for unknown language {}'.format(lang))
        raise LibraryGenerationError()

    if lang == 'cuda' and shared:
        logger.error('CUDA does not support linking of shared device libraries.')
        raise LibraryGenerationError()

    if not shared and as_executable:
        logger.error('Can only make an executable out of a shared library')
        raise LibraryGenerationError()

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
    i_dirs, files = get_file_list(source_dir, build_lang, ktype,
                                  file_base=kwargs.pop('file_base', None))

    # get toolchain
    toolchain = get_toolchain(lang, shared, as_executable)
    toolchain = toolchain.copy(include_dirs=toolchain.include_dirs + i_dirs)

    # compile
    ext = utils.file_ext[lang]
    obj_files = compile(lang, toolchain, [x + ext for x in files],
                        source_dir=source_dir, obj_dir=obj_dir)

    # and link
    if lang == 'opencl':
        desc = 'ocl'
    elif lang == 'c':
        desc = 'c'
    libname = 'lib{}_pyjac'.format(desc)
    libname += toolchain.so_ext
    return link(toolchain, obj_files, libname, lib_dir=out_dir)
