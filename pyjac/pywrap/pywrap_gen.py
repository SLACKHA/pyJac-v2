"""Module for generating Python wrapper around pyJac code.
"""
import sys
import os
import logging

from six.moves import cPickle as pickle
from pytools import ImmutableRecord
from cogapp import Cog

from pyjac.libgen import generate_library
from pyjac.core.enum_types import KernelType
from pyjac.core.create_jacobian import inputs_and_outputs as jac_args
from pyjac.core.rate_subs import inputs_and_outputs as rate_args
from pyjac.kernel_utils.kernel_gen import DocumentingRecord
from pyjac import siteconf as site
from pyjac import utils


class WrapperGen(ImmutableRecord, DocumentingRecord):
    """
    A serializable class for python wrapper generation

    Attributes
    ----------
    name: str
        The name of the generated kernel
    kernel_args: list of str
        The input / output arguments of the kernel
    lang: str
        The language this wrapper is being generated for
    """

    def __init__(self, name='', kernel_args=[], lang='c'):
        docs = self.init_docs(lang)
        ImmutableRecord.__init__(self, name=name, kernel_args=kernel_args, lang=lang,
                                 docs=docs)


class SetupGen(ImmutableRecord):
    """
    A serializable class for python setup.py generation

    Attributes
    ----------
    lang: str
        The generated language
    package_lang : str
        The short-name of the language/wrapper being generated
    wrapper : str
        Filename / path to the generated pyx file
    build_dir : str
        Build directory path
    name : str
        The base-name the generated python wrapper should take
    libname : str
        The path to the generated library
    include_dirs : Optional[list of str]
        Optional; if supplied, extra include directions for the python wrapper
    libraries : Optional[list of str]
        Optional; if supplied extra libraries to use
    libdirs : Optional[list of str]
        Optional; if supplied, library directories

    """

    def __init__(self, name='', libname='', include_dirs=[], package_lang='c',
                 wrapper='', lang='', build_dir='', libraries=[],
                 libdirs=[]):
        ImmutableRecord.__init__(self, name=name, lang=lang, libname=libname,
                                 include_dirs=include_dirs, libraries=libraries,
                                 package_lang=package_lang, wrapper=wrapper,
                                 build_dir=build_dir, libdirs=libdirs)


def generate_setup(lang, setupfile, pyxfile, home_dir, build_dir, out_dir, libname,
                   extra_include_dirs=[], libraries=[], libdirs=[],
                   ktype=KernelType.jacobian):
    """Helper method to fill in the template .in files

    Parameters
    ----------
    lang : str
        The language of the wrapper being generated
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
    setup: str
        The path to the generated setup.py file
    """

    setup = SetupGen(name='pyjac',
                     libname=libname,
                     include_dirs=extra_include_dirs,
                     package_lang=utils.package_lang[lang],
                     wrapper=pyxfile,
                     lang=lang,
                     build_dir=build_dir,
                     libraries=libraries,
                     libdirs=libdirs)
    # serialize
    # dump wrapper
    with utils.temporary_directory() as tdir:
        setupgen = os.path.join(tdir, 'setupgen.pickle')
        with open(setupgen, 'wb') as file:
            pickle.dump(setup, file)

        infile = setupfile
        outfile = os.path.basename(infile[:infile.rindex('.in')])
        outfile = os.path.join(out_dir, outfile)
        # and cogify
        try:
            Cog().callableMain([
                        'cogapp', '-e', '-d', '-Dsetupgen={}'.format(setupgen),
                        '-o', outfile, infile])
        except Exception:
            logger = logging.getLogger(__name__)
            logger.error('Error generating python setup file: {}'.format(outfile))
            raise

    return outfile


def generate_wrapper(lang, pyxfile, build_dir, ktype=KernelType.jacobian,
                     additional_inputs=[], additional_outputs=[],
                     nice_name=None):
    """
    Generate the Cython wrapper file

    Parameters
    ----------
    lang : str
        The language of the wrapper being generated
    pyxfile : str
        Filename of the pyx file template
    build_dir : str
        The path to place the generated cython wrapper in
    ktype : :class:`KernelType` [KernelType.jacobian]
        The type of wrapper to generate
    additional_inputs : list of str
        If supplied, treat these arguments as additional input variables
    additional_outputs : list of str
        If supplied, treat these arguments as additional output variables
    nice_name: str [None]
        If supplied, use this instead of :param:`ktype` to derive the kernel name

    Returns
    -------
    wrapper: str
        The path to the generated python wrapper
    """

    # create wrappergen
    if nice_name is None:
        nice_name = utils.enum_to_string(ktype)

    if ktype == KernelType.jacobian:
        inputs, outputs = jac_args(True)
        # replace 'P_arr' w/ 'param' for clarity
        replacements = {'P_arr': 'param'}
    elif ktype != KernelType.dummy:
        inputs, outputs = rate_args(True, ktype)
        replacements = {'cp': 'specific_heat',
                        'cv': 'specific_heat',
                        'h': 'specific_energy',
                        'u': 'specific_energy'}
    else:
        assert additional_outputs
        assert additional_inputs
        replacements = {}
        inputs = additional_inputs[:]
        outputs = additional_outputs[:]

    def extend(names, args=[]):
        for name in names:
            if name in replacements:
                name = replacements[name]
            if name not in args:
                args.append(name)
        return args

    args = extend(outputs, extend(inputs))
    wrapper = WrapperGen(name=nice_name, kernel_args=args, lang=lang)

    # dump wrapper
    with utils.temporary_directory() as tdir:
        wrappergen = os.path.join(tdir, 'wrappergen.pickle')
        with open(wrappergen, 'wb') as file:
            pickle.dump(wrapper, file)

        infile = pyxfile
        outfile = 'pyjac_{}.pyx'.format(utils.package_lang[lang])
        outfile = os.path.join(build_dir, outfile)
        # and cogify
        try:
            Cog().callableMain([
                        'cogapp', '-e', '-d', '-Dwrappergen={}'.format(wrappergen),
                        '-o', outfile, infile])
        except Exception:
            logger = logging.getLogger(__name__)
            logger.error('Error generating python wrapper file: {}'.format(outfile))
            raise

    return outfile


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
    import sysconfig
    f = "{dirname}.{platform}-{version[0]}.{version[1]}"
    return f.format(dirname=dname,
                    platform=sysconfig.get_platform(),
                    version=sys.version_info
                    )


home_dir = os.path.abspath(os.path.dirname(__file__))


def pywrap(lang, source_dir, build_dir=None, out_dir=None,
           obj_dir=None, platform='', additional_outputs=[],
           ktype=KernelType.jacobian, **kwargs):
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
    additional_outputs : list of str
        If specified, these additional arguments should be considered outputs of the
        generated kernel call. Useful in testing, to allow output of the forward,
        reverse, pressure depenedent and net rates of progress for a more thorough
        comparison to Cantera (specifically, to quantify floating point errors for
        net production rates near equilibrium)
    ktype : :class:`KernelType` [KernelType.jacobian]
        The type of wrapper to generate

    Keyword Arguments
    -----------------
    file_base: str
        Used for creation of libraries for :param:`ktype`==KernelType.dummy -- the
        base filename (generator name) for this library
    additional_inputs: list of str [[]]
        Use to supply additional input argument names to the generator process;
        currently this is only used for :param:`ktype`==KernelType.dummy


    Returns
    -------
    None

    """

    utils.check_lang(lang)
    source_dir = os.path.abspath(source_dir)

    if out_dir is None:
        out_dir = os.getcwd()

    if obj_dir is None:
        obj_dir = os.path.join(os.getcwd(), 'obj')

    if build_dir is None:
        build_dir = os.path.join(os.getcwd(), 'build', distutils_dir_name('temp'))

    shared = True
    # first generate the library
    lib = generate_library(lang, source_dir, out_dir=build_dir, obj_dir=obj_dir,
                           shared=shared, ktype=ktype,
                           file_base=kwargs.get('file_base', None))
    lib = os.path.abspath(lib)

    extra_include_dirs = []
    libraries = []
    libdirs = []
    rpath = ''
    if lang == 'opencl':
        extra_include_dirs.extend(site.CL_INC_DIR)
        libraries.extend(site.CL_LIBNAME)

    setupfile = 'pyjacob_setup.py.in'
    pyxfile = 'pyjacob_wrapper.pyx.in'

    # generate wrapper
    wrapper = generate_wrapper(lang, os.path.join(home_dir, pyxfile), build_dir,
                               ktype=ktype, additional_outputs=additional_outputs,
                               additional_inputs=kwargs.pop('additional_inputs', []),
                               nice_name=kwargs.get('file_base', None))

    # generate setup
    setup = generate_setup(
        lang, os.path.join(home_dir, setupfile), wrapper,
        home_dir, source_dir, build_dir, lib,
        extra_include_dirs, libraries, libdirs,
        ktype=ktype)

    # and build / run
    call = [setup, 'build_ext', '--build-lib', out_dir,
            '--build-temp', obj_dir]
    if rpath:
        call += ['--rpath', rpath]

    utils.run_with_our_python(call)
