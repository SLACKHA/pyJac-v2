"""Module for generating Python wrapper around pyJac code.
"""
import sys
import os
import subprocess
from string import Template
import logging

from ..libgen import generate_library, build_type
from .. import site_conf as site


def generate_setup(setupfile, pyxfile, home_dir, build_dir, out_dir, libname,
                   extra_include_dirs=[], libraries=[], libdirs=[],
                   btype=build_type.jacobian):
    """Helper method to fill in the template .in files

    Parameters
    ----------
    setupfile : str
        Filename of the setup file template
    pyxfile : str
        Filename of the pyx file template
    home_dir : str
        Home directory path
    build_dir : str
        Build directory path
    out_dir : str
        Output directory path
    libname : str
        Library name
    extra_include_dirs : Optional[list of str]
        Optional; if supplied, extra include directions for the python wrapper
    libraries : Optional[list of str]
        Optional; if supplied extra libraries to use
    libdirs : Optional[list of str]
        Optional; if supplied, library directories

    Returns
    -------
    None

    """

    # load and create the setup file
    with open(setupfile, 'r') as file:
        src = Template(file.read())

    def __arr_create(arr):
        return ', '.join(["'{}'".format(x) for x in arr])

    nice_pyx_name = pyxfile[:pyxfile.rindex('.in')]
    file_data = {'homepath': home_dir,
                 'buildpath': build_dir,
                 'libname': libname,
                 'outpath': out_dir,
                 'extra_include_dirs': __arr_create(extra_include_dirs),
                 'libs': __arr_create(libraries),
                 'libdirs': __arr_create(libdirs),
                 'wrapper': nice_pyx_name
                 }
    src = src.safe_substitute(file_data)
    with open(setupfile[:setupfile.rindex('.in')], 'w') as file:
        file.write(src)

    # and the wrapper file
    # load and create the setup file
    with open(pyxfile, 'r') as file:
        src = Template(file.read())

    nice_name = str(btype)
    nice_name = nice_name[nice_name.index('.') + 1:]
    file_data = {'knl': nice_name}

    src = src.safe_substitute(file_data)
    with open(nice_pyx_name, 'w') as file:
        file.write(src)


def distutils_dir_name(dname):
    """Returns the name of a distutils build directory

    Parameters
    ----------
    dname : str
        Base directory name

    Returns
    -------
    Name of a distutils build directory

    """
    import sys
    import sysconfig
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info
                    )


home_dir = os.path.abspath(os.path.dirname(__file__))


def generate_wrapper(lang, source_dir, build_dir=None, out_dir=None,
                     obj_dir=None, platform='', output_full_rop=False,
                     btype=build_type.jacobian):
    """Generates a Python wrapper for the given language and source files

    Parameters
    ----------
    lang : {'cuda', 'c', 'tchem'}
        Programming language of pyJac (cuda, c) or TChem
    source_dir : str
        Directory path of source files.
    build_dir : str
        Directory path of the generated c/cuda/opencl library
    out_dir : Optional [str]
        Directory path for the output python library
    obj_dir: Optional [str]
        Directory path to place the compiled objects
    platform : Optional[str]
        Optional; if specified, the platform for OpenCL execution
    output_full_rop : bool
        If ``True``, output forward and reversse rates of progress
        -- Useful in testing, as there are serious floating point errors for
        net production rates near equilibrium, invalidating direct comparison to
        Cantera
    Returns
    -------
    None

    """

    source_dir = os.path.abspath(source_dir)

    if out_dir is None:
        out_dir = os.getcwd()

    if build_dir is None:
        build_dir = os.path.join('build', distutils_dir_name('temp'))

    shared = False
    ext = '.so' if shared else '.a'
    lib = None
    if lang != 'tchem':
        # first generate the library
        lib = generate_library(lang, source_dir, out_dir=build_dir, obj_dir=obj_dir,
                               shared=shared, btype=btype)
        lib = os.path.abspath(lib)
        if shared:
            lib = lib[lib.index('lib') + len('lib'):lib.index(ext)]

    extra_include_dirs = []
    libraries = []
    libdirs = []
    rpath = ''
    if lang == 'opencl':
        extra_include_dirs.extend(site.CL_INC_DIR)
        libraries.extend(site.CL_LIBNAME)

    if lang == 'c':
        setupfile = 'pyjacob_setup.py.in'
        pyxfile = 'pyjacob_wrapper.pyx.in'
    elif lang == 'opencl':
        setupfile = 'pyocl_setup.py.in'
        pyxfile = 'pyocl_wrapper.pyx.in'
    else:
        logger = logging.getLogger(__name__)
        logger.error('Language {} not recognized'.format(lang))
        sys.exit(-1)

    if output_full_rop:
        # modify the wrapper
        pyxfile = pyxfile[:pyxfile.rindex('_wrapper')] + '_ropfull' + pyxfile[
            pyxfile.rindex('_wrapper'):]

    generate_setup(os.path.join(home_dir, setupfile),
                   os.path.join(home_dir, pyxfile), home_dir, source_dir,
                   build_dir, lib, extra_include_dirs, libraries, libdirs,
                   btype=btype)

    python_str = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])

    # save current
    cwd = os.getcwd()
    try:
        # change to the script dir to avoid long build path
        os.chdir(home_dir)
        # buold
        call = [python_str, os.path.join(home_dir,
                                         setupfile[:setupfile.index('.in')]),
                'build_ext', '--build-lib', out_dir
                ]
        if rpath:
            call += ['--rpath', rpath]

        subprocess.check_call(call)
    finally:
        # and return to base dir
        os.chdir(cwd)
