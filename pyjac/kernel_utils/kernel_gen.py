"""
kernel_gen.py - generators used for kernel creation
"""

import shutil
import textwrap
import os
import re
from string import Template
import logging
from collections import defaultdict
import six
from six.moves import cPickle as pickle

import loopy as lp
from loopy.types import NumpyType, AtomicNumpyType, to_loopy_type
from loopy.version import LOOPY_USE_LANGUAGE_VERSION_2018_2  # noqa
from loopy.kernel.data import AddressSpace as scopes
try:
    import pyopencl as cl
except ImportError:
    cl = None
import numpy as np
import cgen
from pytools import ImmutableRecord
from cogapp import Cog

from pyjac.kernel_utils import file_writers as filew
from pyjac.kernel_utils.memory_limits import memory_limits, \
    memory_type, MemoryGenerationResult, get_string_strides
from pyjac.core.enum_types import DeviceMemoryType
from pyjac import siteconf as site
from pyjac import utils
from pyjac.loopy_utils import loopy_utils as lp_utils
from pyjac.loopy_utils import preambles_and_manglers as lp_pregen
from pyjac.core.array_creator import problem_size as p_size
from pyjac.core.array_creator import work_size as w_size
from pyjac.core.array_creator import global_ind
from pyjac.core import array_creator as arc
from pyjac.core.enum_types import DriverType, KernelType
from pyjac.core.instruction_creator import PreambleMangler

script_dir = os.path.abspath(os.path.dirname(__file__))


rhs_work_name = 'rwk'
"""
Name of the generated work-array for generic double-precision work vectors
"""

local_work_name = 'lwk'
"""
Name of the generated work-array for generic double-precision work vectors in
the __local address space
"""

int_work_name = 'iwk'
"""
Name of the generated work-array for generic integer work vectors
"""

time_array = lp.ArrayArg('t', dtype=np.float64, shape=(p_size.name,),
                         address_space=scopes.GLOBAL)
"""
The time array that we prepend to our function signatures for compatibility
"""


class FakeCall(object):
    """
    In some cases, e.g. finite differnce jacobians, we need to place a dummy
    call in the kernel that loopy will accept as valid.  Then it needs to
    be substituted with an appropriate call to the kernel generator's kernel

    Attributes
    ----------
    dummy_call: str
        The dummy call passed to loopy to replaced during code-generation
    replace_in: :class:`loopy.LoopKernel`
        The kernel to replace the dummy call in.
    replace_with: :class:`loopy.LoopKernel`
        The kernel to replace the dummy call with.
    """

    def __init__(self, dummy_call, replace_in, replace_with):
        self.dummy_call = dummy_call
        self.replace_in = replace_in
        self.replace_with = replace_with

    def match(self, kernel, kernel_body):
        """
        Return true IFF :param:`kernel` matches :attr:`replace_in`

        Params
        ------
        kernel: class:`loopy.LoopKernel`
            The kernel to test
        kernel_body: str
            The generated kernel body
        """

        match = kernel.name == self.replace_in.name
        if match:
            assert self.dummy_call in kernel_body
            return True
        return False


class vecwith_fixer(object):

    """
    Simple utility class to force a constant vector width
    even when the loop being vectorized is shorted than the desired width

    clean : :class:`loopy.LoopyKernel`
        The 'clean' version of the kernel, that will be used for
        determination of the gridsize / vecsize
    vecsize : int
        The desired vector width
    """

    def __init__(self, clean, vecsize):
        self.clean = clean
        self.vecsize = vecsize

    def __call__(self, insn_ids, ignore_auto=False):
        # fix for variable too small for vectorization
        grid_size, lsize = self.clean.get_grid_sizes_for_insn_ids(
            insn_ids, ignore_auto=ignore_auto)
        lsize = lsize if not bool(self.vecsize) else \
            self.vecsize
        return grid_size, (lsize,)


def make_kernel_generator(loopy_opts, *args, **kwargs):
    """
    Factory generator method to return the appropriate
    :class:`kernel_generator` type based on the target language in the
    :param:`loopy_opts`

    Parameters
    ----------
    loopy_opts : :class:`LoopyOptions`
        The specified user options
    *args : tuple
        The other positional args to pass to the :class:`kernel_generator`
    **kwargs : dict
        The keyword args to pass to the :class:`kernel_generator`
    """
    if loopy_opts.lang == 'c':
        if not loopy_opts.auto_diff:
            return c_kernel_generator(loopy_opts, *args, **kwargs)
        if loopy_opts.auto_diff:
            return autodiff_kernel_generator(loopy_opts, *args, **kwargs)
    if loopy_opts.lang == 'opencl':
        return opencl_kernel_generator(loopy_opts, *args, **kwargs)
    if loopy_opts.lang == 'ispc':
        return ispc_kernel_generator(loopy_opts, *args, **kwargs)
    raise NotImplementedError()


def find_inputs_and_outputs(knl):
    """
    Convienence method that returns the name of all input/output array's for a given
    :class:`loopy.LoopKernel`

    Parameters
    ----------
    knl: :class:`loopy.LoopKernel`:
        The kernel to check

    Returns
    -------
    inputs_and_outputs: set of str
        The names of the written / read arrays
    """

    return (knl.get_read_variables() | knl.get_written_variables()) & \
        knl.global_var_names()


def _unSIMDable_arrays(knl, loopy_opts, mstore, warn=True):
    """
    Determined which  inputs / outputs are directly indexed with the base iname,
    or whether a map was applied.  In the latter case it is not safe to convert
    the array to a true vectorize access, as we have no guarentee that the
    index can be converted into an integer access

    Parameters
    ----------
    knl: :class:`loopy.LoopKernel`
        The loopy kernel to check
    loopy_opts: :class:`LoopyOptions`
        the loopy options object
    mstore: :class:`pyjac.core.array_creator.mapstore`
        The mapstore created for the kernel
    warn: bool [True]
        If true, fire off a warning of the arrays that could not be vectorized

    Returns
    -------
    unsimdable: list of str
        List of array names that cannot be safely converted to SIMD

    """

    if not loopy_opts.depth:
        # can convert all arrays to SIMD
        return []

    # this is test is made quite easy by checking the mapstore's tree

    # first, get all inputs / outputs
    io = find_inputs_and_outputs(knl)

    # check each array
    owners = arc.search_tree(mstore.absolute_root, io)
    cant_simd = []
    for ary, owner in zip(io, owners):
        # see if we can get from the owner to the absolute root without encountering
        # any non-affine transforms
        while owner and owner != mstore.absolute_root:
            if not owner.domain_transform.affine:
                cant_simd.append(ary)
                break
            owner = owner.parent

    if cant_simd and warn:
        logger = logging.getLogger(__name__)
        logger.warn('Arrays ({}) could not be fully vectorized. '
                    'You might achieve better performance by applying mechanism '
                    'sorting.'.format(utils.stringify_args(cant_simd)))
    return cant_simd


class TargetCheckingRecord(ImmutableRecord):
    """
    A simple base class that overrides :class:`ImmutableRecord`'s pickling behavior
    to assert that all :class:`loopy.KernelArg`'s have a properly set :attr:`dtype`,
    that is, the :attr:`target` of the dtype is set, such that they may be unpickled
    """

    def __check(self, field, value):
        if isinstance(value, lp.KernelArgument):
            assert value.dtype.target is not None, (
                'Argument {} in field {} has unset dtype'.format(value.name, field))
        elif isinstance(value, NumpyType):
            assert value.target is not None, (
                'dtype {} in field {} has unset dtype'.format(value, field))
        elif isinstance(value, list):
            return all(self.__check(field, x) for x in value)
        elif isinstance(value, dict):
            return all(self.__check(field, x) for x in value.values()) and \
                all(self.__check(field, x) for x in value.keys())
        return True

    def __getstate__(self):
        for field in self.__class__.fields:
            if hasattr(self, field):
                self.__check(field, getattr(self, field))

        return super(TargetCheckingRecord, self).__getstate__()


class CodegenResult(TargetCheckingRecord):
    """
    A convenience class that provides storage for intermediate code-generation
    results.

    Attributes
    ----------
    pointer_unpacks: list of str
        A list of pointer de-references that convert the working buffer(s) to
        individual arrays
    instructions: list of str
        The instructions of the generated wrapper kernel
    preambles: list of str
        Macros, non-vector functions, and other miscellania to place at the top of
        the generated file
    extra_kernels: list of str
        The additional kernels that the generated wrapper kernel calls
    kernel: :class:`loopy.LoopKernel`
        The skeleton of the wrapper kernel, used for generating the correct call
        signature
    pointer_offsets: dict of str -> str
        Stored offsets for pointer unpacks (used in driver kernel creation)
    inits: dict of str->str
        Dictionary mapping constant array name -> initialization to avoid duplication
    name: str
        The name of the generated kernel
    dependencies: list of str
        The list of dependencies for this code-generation result
    """

    def __init__(self, pointer_unpacks=[], instructions=[], preambles=[],
                 extra_kernels=[], kernel=None, pointer_offsets={}, inits={},
                 name='', dependencies=[]):
        ImmutableRecord.__init__(self, pointer_unpacks=pointer_unpacks,
                                 instructions=instructions, preambles=preambles,
                                 extra_kernels=extra_kernels, kernel=kernel,
                                 pointer_offsets=pointer_offsets, inits=inits,
                                 name=name, dependencies=dependencies)


def kernel_arg_docs():
    return {'phi': ('double', 'The state vector'),
            'P_arr': ('double', 'The array of pressures.'),
            'V_arr': ('double', 'The array of volumes'),
            'dphi': ('double', 'The time rate of change of the state-vector'),
            'jac': ('double', 'The Jacobian of the time-rate of change of '
                              'the state vector'),
            'problem_size': ('size_t', 'The total number of conditions to execute '
                             'this kernel over')}


# heh
def langue_docs(lang):
    if lang == 'opencl':
        return {
            'work_size': ('size_t', 'The number of OpenCL groups to launch.\n'
                                    'If using GPUs, this is the # of CUDA blocks '
                                    'to use.\n'
                                    'If for CPUs, this is the number of logical '
                                    'cores to use.'),
            'do_not_compile': ('bool', 'If true, the OpenCL kernel has already been '
                                       'compiled (e.g., via previous kernel call) '
                                       'and does not need recompilation. False by '
                                       'default.\n\n Note: If this kernel object '
                                       'has already been executed, the OpenCL '
                                       'kernel has been compiled and will not be '
                                       'recompiled regardless of the status of '
                                       'this flag.')
        }
    elif lang == 'c':
        return {
            'work_size': ('size_t', 'The number of OpenMP threads to use.'),
            'do_not_compile': ('bool', 'Unused -- incuded for consistent '
                               'signatures.')
        }


class DocumentingRecord(object):
    """
    Note, the base class is responsible for passing the 'docs' attribute to the
    immutablerecord
    """

    @staticmethod
    def init_docs(lang, docs=None, language_docs=None):
        docs = {}
        if not docs:
            docs = kernel_arg_docs()
        if not language_docs:
            language_docs = langue_docs(lang)
        docs.update(language_docs)
        return docs

    def get_docs(self, arg):
        """
        Returns the :attr:`docs` matching this :param:`arg`'s :attr:`name`,
        if available, or else a default place-holder string.

        Parameters
        ----------
        arg: :class:`loopy.KernelArgument` or str
            The argument to generate documentation for

        Returns
        -------
        (dtype, docstring): tuple of str
            The type and docstring of the argument
        """

        try:
            name = arg.name
        except AttributeError:
            assert isinstance(arg, str)
            name = arg

        if name in self.docs:
            return self.docs[name]
        else:
            return ('???', 'Unknown kernel argument {}.'.format(name))


class CallgenResult(TargetCheckingRecord, DocumentingRecord):
    """
    A convenience class that provides intermediate storage for generation of the
    calling program

    Attributes
    ----------
    name: str
        The name of the generated kernel
    cl_level: str ['']
        If supplied, OpenCL level for macro definitions
    work_arrays: list of :class:`loopy.ArrayArg`
        The list of work-buffers created for the top-level kernel
    input_args: dict of str -> list of :class:`loopy.ArrayArg`
        A dictionary mapping of kernel name -> global input args for this
        kernel
    output_args: dict of str -> list of :class:`loopy.ArrayArg`
        A dictionary mapping of kernel name -> global output args for this
        kernel
    host_constants: dict of str -> list of :class:`loopy.TemporaryVariables`
        A dictionary mapping of kernel name -> constant variables to be placed in
        the working buffer
    docs: dict of str->str
        A mapping of kernel argument names to their
    local_size: int [1]
        The OpenCL vector width, set to 1 by default for all other languages
    max_ic_per_run: int [None]
        The maximum number of initial conditions allowed per kernel-call
        due to memory constaints
    max_ws_per_run: int
        The maximum number of OpenCL groups / CUDA threads / OpenMP threads allowed
        per kernel-call
    lang: str ['c']
        The language this kernel is being generated for.
    order: str {'C', 'F'}
        The data ordering
    species_names : list of str
        The species names for this model
    rxn_strings : list of str
        The stringified versions of the reactions for this model
    dev_mem_type: :class:`DeviceMemoryType`
        The type of device memory to used, 'pinned', or 'mapped'
    type_map: dict of :class:`LoopyType` -> str
        The mapping of loopy types to ctypes
    source_names: list of str
        The list of filenames of kernels to be compiled. Relevant for OpenCL, such
        that they are available in code.
    platform: str
        The OpenCL platform name, if applicable
    build_options: str
        The OpenCL build options, if applicable
    device_type: int
        The OpenCL device type, if applicable
    input_data_path: str
        The path to the input data binary, if applicable
    for_validation: bool [False]
        If true, save copies of local arrays to file(s) for validation testing.
    binname: str
        The path to the compiled OpenCL binary, if applicable
    """

    def __init__(self, name='', work_arrays=[], input_args={}, output_args={},
                 cl_level='', docs={}, local_size=1, max_ic_per_run=None,
                 max_ws_per_run=None, lang='c', order='C', species_names=[],
                 rxn_strings=[], dev_mem_type=DeviceMemoryType.mapped, type_map={},
                 host_constants={}, source_names={}, platform='', build_options='',
                 device_type=None, input_data_path='', for_validation=False,
                 binname='', language_docs=None):

        docs = self.init_docs(lang, docs=docs, language_docs=language_docs)
        ImmutableRecord.__init__(self, name=name, work_arrays=work_arrays,
                                 input_args=input_args, output_args=output_args,
                                 cl_level=cl_level, docs=docs, local_size=local_size,
                                 max_ic_per_run=max_ic_per_run,
                                 max_ws_per_run=max_ws_per_run, order=order,
                                 species_names=species_names,
                                 rxn_strings=rxn_strings,
                                 lang=lang, dev_mem_type=dev_mem_type,
                                 type_map=type_map, host_constants=host_constants,
                                 source_names=source_names, platform=platform,
                                 build_options=build_options,
                                 device_type=device_type,
                                 input_data_path=input_data_path,
                                 for_validation=for_validation,
                                 binname=binname)

    def _get_data(self, include_work=False):
        data = {}

        def _update(dictv):
            for key, vals in six.iteritems(dictv):
                if key in data:
                    vals = [x for x in vals if x not in data[key]]
                    data[key].extend(vals[:])
                else:
                    data[key] = vals[:]

        _update(self.input_args)
        _update(self.output_args)

        if include_work:
            for key in data:
                data[key].extend(self.work_arrays[:])

        # get a clean copy of input / output args for consistent sorting
        args = [x.name for x in self.input_args[self.name] +
                self.output_args[self.name]]
        for key in data:
            data[key] = utils.kernel_argument_ordering(
                data[key], dummy_args=args, kernel_type=self.kernel_type,
                for_validation=self.for_validation)

        return data

    @property
    def kernel_type(self):
        try:
            kernel_type = utils.EnumType(KernelType)(self.name)
        except Exception:
            kernel_type = KernelType.dummy
        return kernel_type

    @property
    def kernel_data(self):
        """
        Returns a dictionary kernel name-> complete list of kernel arguments
        and work arrays
        """
        return self._get_data(True)

    @property
    def kernel_args(self):
        """
        Returns a dictionary kernel name-> complete list of kernel arguments
        """
        return self._get_data(False)


class CompgenResult(TargetCheckingRecord):
    """
    A convenience class that provides storage for intermediate compilation file
    generation

    Attributes
    ----------
    source_names: list of str
        The file sources
    platform: str
        The OpenCL platform name
    outname: str
        The output binary name for the kernel
    build_options: str
        The OpenCL build options
    """

    def __init__(self, name='', source_names=[], platform=None, outname='',
                 build_options=''):
        ImmutableRecord.__init__(self, name=name, source_names=source_names,
                                 platform=platform, outname=outname,
                                 build_options=build_options)


class ReadgenRecord(TargetCheckingRecord):
    """
    A convenience class that provides storage for initial condition reading from
    binary files

    Attributes
    ----------
    inputs: list of :class:`loopy.ArrayArg`
        The input args to be read from the binary file
    lang: str ['c']
        The language this kernel is being generated for.
    order: str {'C', 'F'}
        The data ordering
    type_map: dict of :class:`LoopyType` -> str
        The mapping of loopy types to ctypes
    """

    def __init__(self, lang='', type_map={}, order='', inputs=[]):
        ImmutableRecord.__init__(self, lang=lang, order=order, inputs=inputs,
                                 type_map=type_map)

    @property
    def dev_mem_type(self):
        # a dummy property to shadow Callgen
        return -1


class kernel_generator(object):

    """
    The base class for the kernel generators
    """

    def __init__(self, loopy_opts, kernel_type, kernels,
                 namestore,
                 name=None,
                 external_kernels=[],
                 input_arrays=[],
                 output_arrays=[],
                 test_size=None,
                 auto_diff=False,
                 depends_on=[],
                 array_props={},
                 barriers=[],
                 extra_kernel_data=[],
                 extra_global_kernel_data=[],
                 extra_preambles=[],
                 is_validation=False,
                 fake_calls=[],
                 mem_limits='',
                 for_testing=False,
                 compiler=None,
                 driver_type=DriverType.lockstep,
                 use_pinned=True):
        """
        Parameters
        ----------
        loopy_opts : :class:`LoopyOptions`
            The specified user options
        kernel_type : :class:`pyjac.enums.KernelType`
            The kernel type; used as a name and for inclusion of other headers
        kernels : list of :class:`loopy.LoopKernel`
            The kernels / calls to wrap
        namestore: :class:`NameStore`
            The namestore object used in creation of this kernel.
            This is used to pull any extra data (e.g. the Jacobian row/col inds)
            as needed
        input_arrays : list of str
            The names of the input arrays of this kernel
        output_arrays : list of str
            The names of the output arrays of this kernel
        test_size : int
            If specified, the # of conditions to test
        auto_diff : bool
            If true, this will be used for automatic differentiation
        depends_on : list of :class:`kernel_generator`
            If supplied, this kernel depends on the supplied depencies
        array_props : dict
            Mapping of various switches to array names:
                doesnt_need_init
                    * Arrays in this list do not need initialization
                      [defined for host arrays only]
        barriers : list of tuples
            List of global memory barriers needed, (knl1, knl2, barrier_type)
        extra_kernel_data : list of :class:`loopy.ArrayBase`
            Extra kernel arguements to add to this kernel
        extra_global_kernel_data : list of :class:`loopy.ArrayBase`
            Extra kernel arguements to add _only_ to this kernel (and not any
            subkernels)
        extra_preambles: list of :class:`PreambleGen`
            Preambles to add to subkernels
        is_validation: bool [False]
            If true, this kernel generator is being used to validate pyJac
            Hence we need to save our output data to a file
        fake_calls: list of :class:`FakeCall`
            Calls to smuggle past loopy
        mem_limits: str ['']
            Path to a .yaml file indicating desired memory limits that control the
            desired maximum amount of global / local / or constant memory that
            the generated pyjac code may allocate.  Useful for testing, or otherwise
            limiting memory usage during runtime. The keys of this file are the
            members of :class:`pyjac.kernel_utils.memory_limits.mem_type`
        for_testing: bool [False]
            If true, this kernel generator will be used for unit testing
        compiler: :class:`loopy.CCompiler` [None]
            An instance of a loopy compiler (or subclass there-of, e.g.
            :class:`pyjac.loopy_utils.AdeptCompiler`), or None
        driver_type: :class:`DriverType`
            The type of kernel driver to generate
        use_pinned: bool [True]
            If true, use pinned memory for device host buffers (e.g., on CUDA or
            OpencL).  If false, use normal device buffers / copies
        """

        self.compiler = compiler
        self.loopy_opts = loopy_opts
        self.array_split = arc.array_splitter(loopy_opts)
        self.lang = loopy_opts.lang
        self.target = lp_utils.get_target(self.lang, self.loopy_opts.device,
                                          self.compiler)
        self.mem_limits = mem_limits

        # Used for pinned memory kernels to enable splitting evaluation over multiple
        # kernel calls
        self.arg_name_maps = {p_size: 'per_run'}

        self.kernel_type = kernel_type
        self._name = name
        if name is not None:
            assert self.kernel_type == KernelType.dummy
        self.kernels = kernels
        self.namestore = namestore
        self.test_size = test_size
        self.auto_diff = auto_diff

        # update kernel inputs / outputs
        self.in_arrays = input_arrays[:]
        self.out_arrays = output_arrays

        self.type_map = {}
        self.type_map[to_loopy_type(np.float64, target=self.target)] = 'double'
        self.type_map[to_loopy_type(np.int32, target=self.target)] = 'int'
        self.type_map[to_loopy_type(np.int64, target=self.target)] = 'long int'

        self.depends_on = depends_on[:]
        self.array_props = array_props.copy()
        self.all_arrays = []
        self.barriers = barriers[:]

        # extra kernel parameters to be added to subkernels
        self.extra_kernel_data = extra_kernel_data[:]
        # extra kernel parameters to be added only to this subkernel
        self.extra_global_kernel_data = extra_global_kernel_data[:]

        self.extra_preambles = extra_preambles[:]
        # check for Jacobian type
        self.jacobian_lookup = None
        if isinstance(namestore.jac, arc.jac_creator):
            # need to add the row / column inds
            self.extra_kernel_data.extend([self.namestore.jac_row_inds([''])[0],
                                           self.namestore.jac_col_inds([''])[0]])

            # and the preamble
            self.extra_preambles.append(lp_pregen.jac_indirect_lookup(
                self.namestore.jac_col_inds if self.loopy_opts.order == 'C'
                else self.namestore.jac_row_inds, self.target))
            self.jacobian_lookup = self.extra_preambles[-1].array.name

        # calls smuggled past loopy
        self.fake_calls = fake_calls[:]
        # set testing
        self.for_testing = isinstance(test_size, int)
        # setup driver type
        self.driver_type = driver_type
        # and pinned
        self.use_pinned = use_pinned
        # mark owners
        self.owner = None
        # validation
        self.for_validation = False

        def __mark(dep):
            for x in dep.depends_on:
                x.owner = dep
                __mark(x)
        __mark(self)

        # the base skeleton for sub kernel creation
        self.skeleton = textwrap.dedent(
            """
            for j
                ${pre}
                for ${var_name}
                    ${main}
                end
                ${post}
            end
            """)
        if self.loopy_opts.pre_split:
            # pre split skeleton
            self.skeleton = textwrap.dedent(
                """
                for j_outer
                    for j_inner
                        ${pre}
                        for ${var_name}
                            ${main}
                        end
                        ${post}
                    end
                end
                """)

    @property
    def name(self):
        """
        Return the name of this kernel generator, based on :attr:`kernel_type
        """

        if self.kernel_type == KernelType.dummy:
            return self._name
        return utils.enum_to_string(self.kernel_type)

    @property
    def unique_pointers(self):
        """
        Return True IFF the user specified the :attr:`loopy_opts.unique_pointers`
        """
        return self.loopy_opts.unique_pointers

    @property
    def work_size(self):
        """
        Returns either the integer :attr:`loopy_opts.work_size` (if specified by
        user) or the name of the `work_size` variable
        """

        if self.unique_pointers:
            return self.vec_width if self.vec_width else 1
        return w_size.name

    @property
    def target_preambles(self):
        """
        Preambles based on the target language

        Returns
        -------
        premables: list of str
            The string preambles for this :class:`kernel_generator`
        """

        return []

    @property
    def vec_width(self):
        """
        Returns the vector width of this :class:`kernel_generator`
        """
        if self.loopy_opts.depth:
            return self.loopy_opts.depth
        if self.loopy_opts.width:
            return self.loopy_opts.width
        return 0

    @property
    def hoist_locals(self):
        """
        If true (e.g., in a subclass), this type of generator requires that local
        memory be hoisted up to / defined in the type-level kernel.

        This is typically the case for languages such as OpenCL and CUDA, but not
        C / OpenMP
        """
        return False

    @property
    def file_prefix(self):
        """
        Prefix for filenames based on autodifferentiaton status
        """
        file_prefix = ''
        if self.auto_diff:
            file_prefix = 'ad_'
        return file_prefix

    def apply_barriers(self, instructions, barriers=None):
        """
        A method stud that can be overriden to apply synchonization barriers
        to vectorized code

        Parameters
        ----------

        instructions: list of str
            The instructions for this kernel
        barriers: list of (int, int)
            The integer indicies between which to insert instructions
            If not supplied, :attr:`barriers` will be used

        Returns
        -------

        instructions : list of str
            The instructions passed in
        """
        return instructions

    def get_assumptions(self, test_size, for_driver=False):
        """
        Returns a list of assumptions on the loop domains
        of generated subkernels

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel
        for_driver: bool [False]
            If this kernel is a driver function

        Returns
        -------

        assumptions : list of str
            List of assumptions to apply to the generated sub kernel
        """

        return []

    def get_inames(self, test_size, for_driver=False):
        """
        Returns the inames and iname_ranges for subkernels created using
        this generator

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel
        for_driver : bool [False]
            If True, utilize the entire test size

        Returns
        -------
        inames : list of str
            The string inames to add to created subkernels by default
        iname_domains : list of str
            The iname domains to add to created subkernels by default
        """

        # need to implement a pre-split, to avoid loopy mangling the inner / outer
        # parallel inames
        pre_split = self.loopy_opts.pre_split

        gind = global_ind
        if not self.for_testing:
            # if we're not testing, or in a driver function the kernel must only be
            # executed once, as the loop over the work-size has been lifted to the
            # driver kernels
            test_size = self.loopy_opts.initial_condition_loopsize

        if pre_split:
            gind += '_outer'

        inames = [gind]
        domains = ['0 <= {} < {}'.format(gind, test_size)]

        if self.loopy_opts.pre_split:
            if self.for_testing or self.unique_pointers:
                # reduced test size
                test_size = int(test_size / self.vec_width)
            # add/fixup dummy j_inner domain
            lind = global_ind + '_inner'
            inames[-1] = (gind, lind)
            domains[-1] = ('0 <= {lind} < {vw} and '
                           '0 <= {gind} < {end}'.format(
                            lind=lind, gind=gind, end=test_size,
                            vw=self.vec_width))

        return inames, domains

    def add_depencencies(self, k_gens):
        """
        Adds the supplied :class:`kernel_generator`s to this
        one's dependency list.  Functionally this means that this kernel
        generator will know how to compile and execute functions
        from the dependencies

        Parameters
        ----------
        k_gens : list of :class:`kernel_generator`
            The dependencies to add to this kernel
        """

        self.depends_on.extend(k_gens)

    def _with_target(self, kernel_arg, for_atomic=False):
        """
        Returns a copy of :param:`kernel_arg` with it's :attr:`dtype.target` set
        for proper pickling

        Parameters
        ----------
        kernel_arg: :class:`loopy.KernelArgument`
            The argument to convert
        for_atomic: bool [False]
            If true, convert to an :class:`AtomicNumpyType`

        Returns
        -------
        updated: :class:`loopy.KernelArgument`
            The argument with correct target set in the dtype
        """

        return kernel_arg.copy(
            dtype=to_loopy_type(kernel_arg.dtype, for_atomic=for_atomic,
                                target=self.target).with_target(self.target))

    def _make_kernels(self, kernels=[], **kwargs):
        """
        Turns the supplied kernel infos into loopy kernels,
        and vectorizes them!

        Parameters
        ----------
        None

        Returns
        -------
        kernels: list of :class:`loopy.LoopKernel`
        """

        use_ours = False
        if not kernels:
            use_ours = True
            kernels = self.kernels

        # now create the kernels!
        for i, info in enumerate(kernels):
            # if external, or already built
            if isinstance(info, lp.LoopKernel):
                continue
            # create kernel from k_gen.knl_info
            kernels[i] = self.make_kernel(info, self.target, self.test_size,
                                          for_driver=kwargs.get('for_driver', False))
            # apply vectorization
            kernels[i] = self.apply_specialization(
                self.loopy_opts,
                info.var_name,
                kernels[i],
                vecspec=info.vectorization_specializer,
                can_vectorize=info.can_vectorize,
                unrolled_vector=info.unrolled_vector)

            dont_split = kwargs.get('dont_split', [])

            # update the kernel args
            kernels[i] = self.array_split.split_loopy_arrays(
                kernels[i], dont_split=dont_split)

            if info.split_specializer:
                kernels[i] = info.split_specializer(kernels[i])

            # and add a mangler
            # func_manglers.append(create_function_mangler(kernels[i]))

            # set the editor
            kernels[i] = lp_utils.set_editor(kernels[i])

        # need to call make_kernels on dependencies
        for x in self.depends_on:
            if use_ours:
                x._make_kernels()

        return kernels

    def __copy_deps(self, scan_path, out_path, change_extension=True):
        """
        Convenience function to copy the dependencies of this
        :class:`kernel_generator` to our own output path

        Parameters
        ----------

        scan_path : str
            The path the dependencies were written to
        out_path : str
            The path this generator is writing to
        change_ext : bool
            If True, any dependencies that do not end with the proper file
            extension, see :any:`utils.file_ext`

        """

        deps = [x for x in os.listdir(scan_path) if os.path.isfile(
            os.path.join(scan_path, x)) and not x.endswith('.in')]
        for dep in deps:
            dep_dest = dep
            dep_is_header = dep.endswith(utils.header_ext['c'])
            ext = (utils.file_ext[self.lang] if not dep_is_header
                   else utils.header_ext[self.lang])
            if change_extension and not dep.endswith(ext):
                dep_dest = dep[:dep.rfind('.')] + ext
            shutil.copyfile(os.path.join(scan_path, dep),
                            os.path.join(out_path, dep_dest))

    def order_kernel_args(self, args):
        """
        Returns the ordered kernel arguments for this :class:`kernel_generator`
        """
        sorting_args = self.in_arrays + self.out_arrays
        return utils.kernel_argument_ordering(args, self.kernel_type,
                                              for_validation=self.for_validation,
                                              dummy_args=sorting_args)

    def generate(self, path, data_order=None, data_filename='data.bin',
                 for_validation=False, species_names=[], rxn_strings=[]):
        """
        Generates wrapping kernel, compiling program (if necessary) and
        calling / executing program for this kernel

        Parameters
        ----------
        path : str
            The output path
        data_order : {'C', 'F'}
            If specified, the ordering of the binary input data
            which may differ from the loopy order
        data_filename : Optional[str]
            If specified, the path to the data file for reading / execution
            via the command line
        for_validation: bool [False]
            If True, this kernel is being generated to validate pyJac, hence we need
            to save output data to a file
        species_names: list of str
            The list of species in the model
        rxn_strings: list of str
            Stringified versions of the reactions in the model

        Returns
        -------
        None
        """

        self.for_validation = for_validation
        utils.create_dir(path)
        self._make_kernels()
        callgen, record, result = self._generate_wrapping_kernel(path)
        callgen = self._generate_driver_kernel(path, record, result, callgen)
        callgen = self._generate_compiling_program(path, callgen)
        _, callgen = self._generate_calling_program(
            path, data_filename, callgen, record, for_validation=for_validation,
            species_names=species_names, rxn_strings=rxn_strings)
        self._generate_calling_header(path, callgen)
        self._generate_common(path, record)

        # finally, copy any dependencies to the path
        lang_dir = os.path.join(script_dir, self.lang)
        self.__copy_deps(lang_dir, path, change_extension=False)

    def _generate_common(self, path, record):
        """
        Creates the common files (used by all target languages) for this
        kernel generator

        Parameters
        ----------
        path : str
            The output path for the common files
        record: :class:`MemoryGenerationResult`
            The memory storage generated for this kernel

        Returns
        -------
        None
        """

        inputs = [x for x in record.args if x.name in self.in_arrays]

        # create readgen
        readgen = ReadgenRecord(
            lang=self.loopy_opts.lang,
            type_map=self.type_map,
            order=self.loopy_opts.order,
            inputs=inputs)

        # serialize
        readout = os.path.join(path, 'readgen.pickle')
        with open(readout, 'wb') as file:
            pickle.dump(readgen, file)

        def run(input, output):
            # cogify
            try:
                Cog().callableMain([
                            'cogapp', '-e', '-d', '-Dreadgen={}'.format(readout),
                            '-o', output, input])
            except Exception:
                logger = logging.getLogger(__name__)
                logger.error('Error generating initial conditions reader:'
                             ' {}'.format(output))
                raise

        common = os.path.join(script_dir, 'common')
        # generate reader
        infile = os.path.join(common, 'read_initial_conditions.cpp.in')
        outfile = os.path.join(path, 'read_initial_conditions' +
                               utils.file_ext[self.lang])
        run(infile, outfile)
        # generate header
        infile = os.path.join(common, 'read_initial_conditions.hpp.in')
        outfile = os.path.join(path, 'read_initial_conditions' +
                               utils.header_ext[self.lang])
        run(infile, outfile)

        # and any other deps
        self.__copy_deps(common, path)

    def _generate_calling_header(self, path, callgen):
        """
        Creates the header file for this kernel

        Parameters
        ----------
        path : str
            The output path for the header file
        callgen: :class:`CallgenResult`
            The current callgen object used to generate the calling program

        Returns
        -------
        file: str
            The path to the generated file
        """

        # serialize
        callout = os.path.join(path, 'callgen.pickle')
        with open(callout, 'wb') as file:
            pickle.dump(callgen, file)

        infile = os.path.join(script_dir, 'common', 'kernel.hpp.in')
        filename = os.path.join(path, self.name + '_main' + utils.header_ext[
                self.lang])

        # cogify
        try:
            Cog().callableMain([
                        'cogapp', '-e', '-d', '-Dcallgen={}'.format(callout),
                        '-o', filename, infile])
        except Exception:
            logger = logging.getLogger(__name__)
            logger.error('Error generating calling header {}'.format(filename))
            raise

        return filename

    def _special_kernel_subs(self, path, callgen):
        """
        Substitutes kernel template parameters that are specific to a
        target languages, to be specialized by subclasses of the
        :class:`kernel_generator`

        Parameters
        ----------
        path : str
            The output path to write files to
        callgen : :class:`CallgenResult`
            The intermediate call-generation store

        Returns
        -------
        updated : :class:`CallgenResult`
            The updated call-generation storage
        """
        return callgen

    def _set_sort(self, arr):
        return sorted(set(arr), key=lambda x: arr.index(x))

    def _generate_calling_program(self, path, data_filename, callgen, record,
                                  for_validation=False, species_names=[],
                                  rxn_strings=[]):
        """
        Needed for all languages, this generates a simple C file that
        reads in data, sets up the kernel call, executes, etc.

        Parameters
        ----------
        path : str
            The output path to write files to
        data_filename : str
            The path to the data file for command line input
        callgen: :class:`CallgenResult`
            The current callgen object used to generate the calling program
        record: :class:`MemoryGenerationResult`
            The memory record for the generated program
        for_validation: bool [False]
            If True, this kernel is being generated to validate pyJac, hence we need
            to save output data to a file
        species_names: list of str
            The list of species in the model
        rxn_strings: list of str
            Stringified versions of the reactions in the model


        Returns
        -------
        file: str
            The output file name
        callgen: :class:`CallgenResult`
            The updated callgen result
        """

        # vec width
        vec_width = self.vec_width
        if not vec_width:
            # set to default
            vec_width = 1
        elif self.loopy_opts.is_simd:
            # SIMD has a vector width, but the launch size is still 1
            vec_width = 1

        # update callgen
        callgen = callgen.copy(
            local_size=vec_width,
            order=self.loopy_opts.order,
            lang=self.lang,
            type_map=self.type_map.copy(),
            input_data_path=data_filename,
            for_validation=for_validation,
            species_names=species_names,
            rxn_strings=rxn_strings)

        # any target specific substitutions
        callgen = self._special_kernel_subs(path, callgen)

        # serialize
        callout = os.path.join(path, 'callgen.pickle')
        with open(callout, 'wb') as file:
            pickle.dump(callgen, file)

        infile = os.path.join(script_dir, 'common', 'kernel.cpp.in')
        filename = os.path.join(path, self.name + '_main' + utils.file_ext[
                self.lang])

        # cogify
        try:
            Cog().callableMain([
                        'cogapp', '-e', '-d', '-Dcallgen={}'.format(callout),
                        '-o', filename, infile])
        except Exception:
            logger = logging.getLogger(__name__)
            logger.error('Error generating calling file {}'.format(filename))
            raise

        return filename, callgen

    def _generate_compiling_program(self, path, callgen):
        """
        Needed for some languages (e.g., OpenCL) this may be overriden in
        subclasses to generate a program that compilers the kernel

        Parameters
        ----------
        path : str
            The output path for the compiling program
        callgen: :class:`CallgenResult`
            The current callgen result, to be updated with driver info

        Returns
        -------
        callgen: :class:`CallgenResult`
            The updated callgen result
        """

        return callgen

    @classmethod
    def _temporary_to_arg(cls, temp):
        """
        Returns the :class:`loopy.ArrayArg` version of the
        :class:`loopy.TemporaryVariable` :param:`temp`
        """

        assert isinstance(temp, lp.TemporaryVariable)
        return lp.ArrayArg(
            address_space=scopes.LOCAL,
            **{k: v for k, v in six.iteritems(vars(temp))
               if k in ['name', 'shape', 'dtype', 'dim_tags']})

    def _migrate_locals(self, kernel, ldecls):
        """
        Migrates local variables in :param:`ldecls` to the arguements of the
        given :param:`kernel`

        Parameters
        ----------
        kernel: :class:`loopy.LoopKernel`
            The kernel to modify
        ldecls: list of :class:`loopy.TemporaryVariable`
            The local variables to migrate

        Returns
        -------
        mod: :class:`loopy.LoopKernel`
            A modified kernel with the given local variables moved from the
            :attr:`loopy.LoopKernel.temporary_variables` to the kernel's
            :attr:`loopy.LoopKernel.args`

        """

        assert all(x.address_space == scopes.LOCAL for x in ldecls)
        ltemps, largs = utils.partition(ldecls, lambda x: isinstance(
            x, lp.TemporaryVariable))
        # only need to process the local temporaries
        names = set([x.name for x in ltemps])
        return kernel.copy(
            args=kernel.args[:] + [self._temporary_to_arg(x) for x in ldecls],
            temporary_variables={
                key: val for key, val in six.iteritems(kernel.temporary_variables)
                if not set([key]) & names})

    def __get_kernel_defn(self, knl, passed_locals=[], remove_work_const=False):
        """
        Returns the kernel definition string for this :class:`kernel_generator`,
        taking into account any migrated local variables

        Note: relies on building steps that occur in
        :func:`_generate_wrapping_kernel` -- will raise an error if called before
        this method

        Parameters
        ----------
        knl: None
            If supplied, this is used instead of the generated kernel
        passed_locals: list of :class:`cgen.CLLocal`
            __local variables declared in the wrapping kernel scope, that must
            be passed into this kernel, as __local defn's in subfunctions
            are not well defined, `function qualifiers in OpenCL <https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/functionQualifiers.html>` # noqa
        remove_work_const: bool [False]
            If true, modify the returned kernel definition to remove const references
            to work arrays

        Returns
        -------
        defn: str
            The kernel definition
        """

        if knl is None:
            raise Exception('Must call _generate_wrapping_kernel first')

        remove_working = True

        if passed_locals:
            knl = self._migrate_locals(knl, passed_locals)
        defn_str = lp_utils.get_header(knl)
        if remove_working:
            defn_str = self._remove_work_size(defn_str)
        if remove_work_const:
            defn_str = self._remove_work_array_consts(defn_str)
        return defn_str[:defn_str.index(';')]

    def _get_kernel_call(self, knl=None, passed_locals=[]):
        """
        Returns a function call for the given kernel :param:`knl` to be used
        as an instruction.

        If :param:`knl` is None, returns the kernel call for
        this :class:`kernel_generator`

        Parameters
        ----------
        knl: :class:`loopy.LoopKernel`
            The loopy kernel to generate a call for
        passed_locals: list of :class:`cgen.CLLocal`
            __local variables declared in the wrapping kernel scope, that must
            be passed into this kernel, as __local defn's in subfunctions
            are not well defined, `function qualifiers in OpenCL <https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/functionQualifiers.html>` # noqa

        Returns
        -------
        call: str
            The resulting function call
        """

        # default is the generated kernel
        if knl is None:
            args = self.kernel_data + [
                x for x in self.extra_global_kernel_data + self.extra_kernel_data
                if isinstance(x, lp.KernelArgument)]
            if passed_locals:
                # put a dummy object that we can reference the name of in the
                # arguements
                args += [type('', (object,), {'name': l.subdecl.name})
                         for l in passed_locals]
            name = self.name
        else:
            # otherwise used passed kernel
            if passed_locals:
                knl = self.__migrate_locals(knl, passed_locals)
            args = knl.args
            name = knl.name

        args = [x.name for x in args]

        return Template("${name}(${args});\n").substitute(
            name=name,
            args=', '.join(args)
            )

    def _compare_args(self, arg1, arg2, allow_shape_mismatch=False):
        """
        Convenience method to test equality of :class:`loopy.KernelArgument`s

        Returns true IFF :param:`arg1` == :param:`arg2`, OR they differ only in
        their atomicity
        """

        def __atomify(arg):
            return self._with_target(arg, for_atomic=True)

        def __shapify(arg1, arg2):
            a1 = self._with_target(arg1, for_atomic=True)
            a2 = self._with_target(arg2, for_atomic=True)
            a1 = a1.copy(shape=a2.shape, dim_tags=a2.dim_tags)
            return a1 == a2

        return arg1 == arg2 or (__atomify(arg1) == __atomify(arg2)) or (
            allow_shape_mismatch and __shapify(arg1, arg2))

    def _process_args(self, kernels=[], allowed_conflicts=[]):
        """
        Processes the arguements for all kernels in this generator (and subkernels
        from dependencies) to:

            1. Check for duplicates
            2. Separate arguments by type (Local / Global / Readonly / Value), etc.

        Notes
        -----
        - The list of local arguments in the returned :class:`MemoryGenerationResult`
        `record` will be non-empty IFF the kernel generator's :attr:`hoist_locals`
        is true
        - If :param:`allowed_conflicts` is supplied, we are assumed to be in a driver
        kernel, and only the 'global' (i.e., `problem_size`'d) variable will be kept,
        as the `work_size`'d variable is assumed to be a local copy

        Parameters
        ----------
        kernels: list of :class:`loopy.LoopKernel`
            The kernels to process
        allowed_conflicts: list of str
            The names of arguments that are allowed to conflict between kernels.

        Returns
        -------
        record: :class:`MemoryGenerationResult`
            A record of the processed arguments, :see:`MemoryGenerationResult`
            The list of global arguments for the top-level wrapping kernel
        kernels: list of :class:`loopy.LoopKernel`
            The (potentially) updated kernels w/ local definitions hoisted as
            necessary
        """

        if not kernels:
            kernels = self.kernels[:]

        # find complete list of kernel data
        args = [arg for dummy in kernels for arg in dummy.args]

        # add our additional kernel data, if any
        args.extend([x for x in self.extra_kernel_data if isinstance(
            x, lp.KernelArgument)])

        kernel_data = []
        # now, scan the arguments for duplicates
        nameset = sorted(set(d.name for d in args))
        for name in nameset:
            same_name = []
            for x in args:
                if x.name == name and not any(x == y for y in same_name):
                    same_name.append(x)

            def __raise():
                raise Exception('Cannot resolve different arguements of '
                                'same name: {}'.format(', '.join(
                                    str(x) for x in same_name)))

            # check allowed_conflicts
            if len(same_name) != 1 and same_name[0].name in allowed_conflicts:
                # find the version of same name that contains the work size,
                # as this is the 'local' version
                work_sized = next((x for x in same_name
                                  if any(w_size.name in str(y) for y in x.shape)),
                                  None)
                if not work_sized:
                    __raise()

                # and remove
                same_name.remove(work_sized)

            if len(same_name) != 1:
                # need to see if differences are resolvable
                atomic = next((x for x in same_name if
                               isinstance(x.dtype, AtomicNumpyType)), None)

                if atomic is not None:
                    other = next(x for x in same_name if x != atomic)
                    # check that all other properties are the same
                    if not self._compare_args(other, atomic,
                                              other.name in allowed_conflicts):
                        __raise()

                    # otherwise, they're the same and the only difference is the
                    # the atomic - so remove the non-atomic
                    same_name.remove(other)

                    # Hence, we try to copy all the other kernels with this arg in it
                    # with the atomic arg
                    for i, knl in enumerate(self.kernels):
                        if other in knl.args:
                            kernels[i] = knl.copy(args=[
                                x if x != other else atomic for x in knl.args])

            if len(same_name) != 1:
                # if we don't have an atomic, or we have multiple different
                # args of the same name...
                __raise()

            same_name = same_name.pop()
            kernel_data.append(same_name)

        # split checked data into arguements and valueargs
        valueargs, args = utils.partition(
            kernel_data, lambda x: isinstance(x, lp.ValueArg))

        # get list of arguments on readonly
        readonly = set(
                arg.name for dummy in kernels for arg in dummy.args
                if not any(arg.name in d.get_written_variables()
                           for d in kernels)
                and not isinstance(arg, lp.ValueArg))

        # check (non-private) temporary variable duplicates
        temps = [arg for dummy in kernels
                 for arg in dummy.temporary_variables.values()
                 if isinstance(arg, lp.TemporaryVariable) and
                 arg.address_space != scopes.PRIVATE and
                 arg.address_space != lp.auto]
        # and add extra kernel data, if any
        temps.extend([x for x in self.extra_kernel_data if isinstance(
            x, lp.TemporaryVariable) and
            x.address_space != scopes.PRIVATE and
            x.address_space != lp.auto])
        copy = temps[:]
        temps = []
        for name in sorted(set(x.name for x in copy)):
            same_names = [x for x in copy if x.name == name]
            if len(same_names) > 1:
                if not all(x == same_names[0] for x in same_names[1:]):
                    raise Exception('Cannot resolve different arguments of '
                                    'same name: {}'.format(', '.join(
                                        str(x) for x in same_names)))
            temps.append(same_names[0])

        # work on temporary variables
        local = []
        if self.hoist_locals:
            local_temps = [x for x in temps if x.address_space == scopes.LOCAL]
            # go through kernels finding local temporaries, and convert to local args
            for i, knl in enumerate(kernels):
                lt = [x for x in local_temps
                      if x in knl.temporary_variables.values()]
                if lt:
                    # convert kernel's local temporarys to local args
                    kernels[i] = self._migrate_locals(knl, lt)
                    # and add to list
                    local.extend([x for x in lt if x not in local])
                    # and remove from temps
                    temps = [x for x in temps if x not in lt]
        # and add any local args
        largs, args = utils.partition(args,
                                      lambda x: x.address_space == scopes.LOCAL)
        local.extend(largs)

        # finally, separate the constants from the temporaries
        # for opencl < 2.0, a constant global can only be a
        # __constant
        constants, temps = utils.partition(temps, lambda x: x.read_only)

        return (MemoryGenerationResult(args=args, local=local, readonly=readonly,
                                       constants=constants, valueargs=valueargs),
                kernels)

    def _process_memory(self, record):
        """
        Determine memory usage / limits, host constant migrations, etc.

        Parameters
        ----------
        record: :class:`MemoryGenerationResult`
            A record of the processed arguments, :see:`MemoryGenerationResult`
            The list of global arguments for the top-level wrapping kernel

        Returns
        -------
        updated_record: :class:`loopy.MemoryGenerationResult`
            The updated memory generation result
        mem_limits: :class:`memory_limits`
            The generated memory limit object
        """

        # now, do our memory calculations to determine if we can fit
        # all our data in memory
        mem_types = defaultdict(lambda: list())

        # store globals
        for arg in [x for x in record.args if not isinstance(x, lp.ValueArg)]:
            mem_types[memory_type.m_global].append(arg)

        # store locals
        mem_types[memory_type.m_local].extend(record.local)

        # and constants
        mem_types[memory_type.m_constant].extend(record.constants)

        # check if we're over our constant memory limit
        mem_limits = memory_limits.get_limits(
            self.loopy_opts, mem_types,
            string_strides=get_string_strides()[0],
            input_file=self.mem_limits,
            limit_int_overflow=self.loopy_opts.limit_int_overflow)

        args = record.args[:]
        constants = record.constants[:]
        readonly = record.readonly.copy()
        host_constants = []
        if not all(x >= 0 for x in mem_limits.can_fit()):
            # we need to convert our __constant temporary variables to
            # __global kernel args until we can fit
            type_changes = defaultdict(lambda: list())
            # we can't remove the sparse indicies as we can't pass pointers
            # to loopy preambles
            gtemps = constants[:]
            if self.jacobian_lookup:
                gtemps = [x for x in constants if self.jacobian_lookup not in x.name]
            # sort by largest size
            gtemps = sorted(gtemps, key=lambda x: np.prod(x.shape), reverse=True)
            type_changes[memory_type.m_global].append(gtemps[0])
            gtemps = gtemps[1:]
            while not all(x >= 0 for x in mem_limits.can_fit(
                    with_type_changes=type_changes)):
                if not gtemps:
                    logger = logging.getLogger(__name__)
                    logger.exception('Cannot fit kernel {} in memory'.format(
                        self.name))
                    # should never get here, but still...
                    raise Exception()

                type_changes[memory_type.m_global].append(gtemps[0])
                gtemps = gtemps[1:]

            # once we've converted enough, we need to physically change the types
            for x in [v for arrs in type_changes.values() for v in arrs]:
                args.append(
                    self._with_target(
                        lp.GlobalArg(x.name, dtype=x.dtype, shape=x.shape)))
                readonly.add(args[-1].name)
                host_constants.append(x)

                # and update the types
                mem_types[memory_type.m_constant].remove(x)
                mem_types[memory_type.m_global].append(x)

            mem_limits = memory_limits.get_limits(
                self.loopy_opts, mem_types, string_strides=get_string_strides()[0],
                input_file=self.mem_limits,
                limit_int_overflow=self.loopy_opts.limit_int_overflow)

        return record.copy(args=args, constants=constants, readonly=readonly,
                           host_constants=host_constants), mem_limits

    def _compress_to_working_buffer(self, record, for_driver=False):
        """
        Compresses the kernel arguments in the :class:`MemoryGenerationResult`
        into working buffers (depending on memory scope & data type), and returns
        the updated record.

        Parameters
        ----------
        record: :class:`MemoryGenerationResult`
            The memory record that holds the kernel arguments
        for_driver: bool [False]
            If True, include kernel arguments in the resulting working buffers

        Returns
        -------
        updated_record: :class:`MemoryGenerationResult`
            The record, with the working buffer(s) stored in :attr:`kernel_data`
        codegen: :class:`CodegenResult`
            The intermediate code generation result w/ stored :attr:`pointer_unpacks`
        """

        # compress our kernel args into a working buffer
        offset_args = record.args[:]
        # partition by memory scope
        largs = record.local[:]
        # partition by data-type
        itype = to_loopy_type(arc.kint_type, target=self.target)
        iargs, dargs = utils.partition(offset_args, lambda x: x.dtype == itype)

        if record.host_constants:
            # include host constants in integer/double workspaces
            inames = set([i.name for i in iargs])
            dnames = set([d.name for d in dargs])
            for hc in record.host_constants:
                if hc.dtype == itype and not set([hc.name]) & inames:
                    iargs += [hc]
                elif not set([hc.name]) & dnames:
                    dargs += [hc]

        # and create buffers for all
        assert dargs, 'No kernel data!'

        def __generate(args, name, scope=scopes.GLOBAL, result=None):
            copy = args[:]
            # first, we sort by kernel argument ordering so any potentially
            # duplicated kernel args are placed at the end and can be safely
            # extracted in the driver
            args = self.order_kernel_args(args)

            if not (len(args) or for_driver):
                if result is None:
                    result = CodegenResult()
                # we've filtered out all the arguments in this kernel, return a dummy
                return self._with_target(
                    lp.ArrayArg(name, shape=(1,),
                                order=self.loopy_opts.order,
                                dtype=copy[0].dtype,
                                address_space=scope), for_atomic=False), result

            # get the pointer unpackings
            size_per_wi, static, offsets = self._get_working_buffer(args)
            unpacks = []
            for k, (dtype, size, offset, s) in six.iteritems(offsets):
                assert s == scope
                unpacks.append(self._get_pointer_unpack(
                    k, size, offset, dtype, scope))
            if not result:
                result = CodegenResult(pointer_unpacks=unpacks,
                                       pointer_offsets=offsets)
            else:
                new_offsets = result.pointer_offsets.copy()
                new_offsets.update(offsets)
                result = result.copy(
                    pointer_unpacks=result.pointer_unpacks + unpacks,
                    pointer_offsets=new_offsets)

            # create working buffer
            from pymbolic.primitives import Variable
            shape = static + Variable(w_size.name) * size_per_wi
            for_atomic = isinstance(args[0].dtype, AtomicNumpyType)
            wb = self._with_target(
                lp.ArrayArg(name, shape=shape,
                            order=self.loopy_opts.order,
                            dtype=args[0].dtype,
                            address_space=scope), for_atomic=for_atomic)
            return wb, result

        # globals
        wb, codegen = __generate(dargs, rhs_work_name)
        record = record.copy(kernel_data=record.kernel_data + [wb])

        if largs:
            # locals
            wb, codegen = __generate(largs, local_work_name, scope=scopes.LOCAL,
                                     result=codegen)
            record = record.copy(kernel_data=record.kernel_data + [wb])

        if iargs:
            # integers
            wb, codegen = __generate(iargs, int_work_name, result=codegen)
            record = record.copy(kernel_data=record.kernel_data + [wb])

        return record, codegen

    def _specialize_pointers(self, record, result):
        """
        Specialize the base pointers in the :param:`result` such that:
            1. The pointer unpacks only contain arrays that correspond to the kernels
               in the calling :class:`kernel_generator`
            2. The pointer unpacks do not contain any of the calling
               :class:`kernel_generator`'s input or output args

        Parameters
        ----------
        record: :class:`MemoryGenerationResult`
            The memory record that holds the kernel arguments
        result: :class:`CodegenResult`
            The current codegen result containing the pointer unpacks

        Returns
        -------
        updated_result: :class:`CodegenResult`
            The updated code generation result with pointers specialized for
            the calling :class:`kernel_generator`
        """

        args = set(self.in_arrays + self.out_arrays)
        data = set([x.name for x in record.args] + [x.name for x in record.local])

        unpacks = []
        offsets = {}

        for (arry, offset), unpack in zip(*(six.iteritems(result.pointer_offsets),
                                            result.pointer_unpacks)):
            if (arry not in args) and (arry in data):
                offsets[arry] = offset
                unpacks.append(unpack)

        return result.copy(pointer_unpacks=unpacks, pointer_offsets=offsets)

    def _dummy_wrapper_kernel(self, kernel_data, readonly, vec_width,
                              as_dummy_call=False, for_driver=False):
        """
        Generates a dummy loopy kernel to function as the global wrapper

        Parameters
        ----------
        kernel_data: list of :class:`loopy.KernelArgument`'s
            The kernel data to use for signature generation
        vec_width: int [0]
            If non-zero, the vector width to use in kernel width fixing
        as_dummy_call: bool [False]
            If True, this is being generated as a dummy call smuggled past loopy
            e.g., for a Finite Difference jacobian call to the species rates kernel
            Hence, we need to add any :attr:`extra_kernel_data` to our kernel defn

        Returns
        -------
        knl: :class:`loopy.LoopKernel`
            The generated dummy kernel

        """

        # assign to non-readonly to prevent removal
        def _name_assign(arr, use_atomics=True):
            if arr.name not in readonly and not isinstance(arr, lp.ValueArg) and \
                    arr.name not in [time_array.name]:
                return arr.name + '[{ind}] = 0 {atomic}'.format(
                    ind=', '.join(['0'] * len(arr.shape)),
                    atomic='{atomic}'
                           if isinstance(arr.dtype, AtomicNumpyType) and use_atomics
                           else '')
            return ''

        # data
        kdata = self.order_kernel_args(kernel_data[:])
        if as_dummy_call:
            # add extra kernel args
            kdata.extend([x for x in self.extra_kernel_data
                          if isinstance(x, lp.KernelArgument)])
        insns = '\n'.join(_name_assign(arr) for arr in kdata)

        # name
        name = self.name + ('_driver' if for_driver else '')

        # domains
        domains = ['{{[{iname}]: 0 <= {iname} < {size}}}'.format(
                iname='i',
                size=self.vec_width)]

        knl = lp.make_kernel(domains, insns, kdata, name=name,
                             target=self.target)

        if self.vec_width and not self.loopy_opts.is_simd:
            ggs = vecwith_fixer(knl.copy(), self.vec_width)
            knl = knl.copy(overridden_get_grid_sizes_for_insn_ids=ggs)

        return knl

    def _migrate_host_constants(self, kernels, host_constants):
        """
        Moves temporary variables to global arguments based on the
        host constants for this :class:`kernel_generator`

        Parameters
        ----------
        kernels: list of :class:`loopy.LoopKernel`
            The kernels to transform
        host_constants: list of :class:`loopy.GlobalArg`
            The list of __constant temporary variables that were converted to
            __global args

        Returns
        -------
        migrated: :class:`loopy.LoopKernel`
            The kernel with any host constants transformed to input arguments
        """

        for i in range(len(kernels)):
            transferred = set([const.name for const in host_constants
                               if const.name in kernels[i].temporary_variables])
            # need to transfer these to arguments
            if transferred:
                # filter temporaries
                new_temps = {t: v for t, v in six.iteritems(
                             kernels[i].temporary_variables) if t not in transferred}
                # create new args
                new_args = [self._with_target(lp.GlobalArg(
                    t, shape=v.shape, dtype=v.dtype, order=v.order,
                    dim_tags=v.dim_tags))
                    for t, v in six.iteritems(kernels[i].temporary_variables)
                    if t in transferred]
                kernels[i] = kernels[i].copy(
                    args=kernels[i].args + new_args, temporary_variables=new_temps)

        return kernels

    def _get_working_buffer(self, args):
        """
        Determine the size of the working buffer required to store the :param:`args`
        in a global working array, and return offsets for determing array indexing

        Parameters
        ----------
        args: list of :class:`loopy.KernelArguments`
            The kernel arguments to collapse into a working buffer

        Returns
        -------
        size_per_work_item: int
            The size (in number of values of dtype of :param:`args`)
            of the working buffer per work-group item
        static_size: int
            The size (in number of values of dtype of :param:`args`) of the working
            buffer (independent of # of work-group items)
        offsets: dict of str -> (dtype, size, offset)
            A mapping of kernel argument names to:
                - the stringified dtype
                - the calculated size of the argument, and
                - offset in the working buffer for this argument's local pointer
        """

        regex = re.compile(r'{}((?:\s*\*\s*)(\d+))?'.format(w_size.name))

        def _get_size(ssize):
            match = regex.search(str(ssize))
            if match:
                multiplier = match.groups()[-1]
                if multiplier:
                    return int(multiplier)
                return 1
            raise NotImplementedError

        size_per_work_item = 0
        static_size = 0
        offsets = {}
        mapping = {}
        if self.unique_pointers:
            mapping = {v.name: v
                       for k, v in six.iteritems(vars(self.namestore))
                       if isinstance(v, arc.creator)}

        def _offset():
            # if we have unique pointers, the work-size is fixed to an integer, and
            # will already be baked into the size of the array
            work_size = self.work_size if not self.unique_pointers else 1
            return '{} * {}'.format(size_per_work_item, work_size)

        for arg in args:
            buffer_size = None
            offset = None
            # split the shape into the work-item and other dimensions
            isizes, ssizes = utils.partition(arg.shape, lambda x: isinstance(x, int))
            if len(ssizes) >= 1:
                # check we have a work size in ssizes
                buffer_size = int(np.prod(isizes) * _get_size(ssizes[0]))
                # offset must be calculated _before_ updating size_per_work_item
                offset = _offset()
                size_per_work_item += buffer_size
            elif not len(ssizes):
                # static size
                buffer_size = int(np.prod(isizes))
                offset = _offset()
                ic_dep = False
                if self.unique_pointers:
                    # need to test if this is per work-item or not
                    if arg.name in mapping and not mapping[arg.name].is_temporary:
                        size_per_work_item += buffer_size
                        ic_dep = True

                if not ic_dep:
                    static_size += buffer_size

            # store offset and increment size
            offsets[arg.name] = (arg.dtype, buffer_size, offset, arg.address_space)

        return size_per_work_item, static_size, offsets

    def _get_pointer_unpack(self, array, size, offset, dtype, scope=scopes.GLOBAL,
                            set_null=False, for_driver=False):
        """
        A method stub to implement the pattern:
        ```
            double* array = &rwk[offset]
        ```
        per target.  Overridden in subclasses

        Parameters
        ----------
        array: str
            The array name
        size: str
            The size of the array
        offset: str
            The stringified offset
        dtype: :class:`loopy.LoopyType`
            The array type
        scope: :class:`loopy.AddressSpace`
            The memory scope
        set_null: bool [False]
            If True, set the unpacked pointer to NULL
        for_driver: bool [False]
            If True, this pointer is being unpacked for the driver, as such
            any value of :attr:`unique_pointers` should be ignored

        Returns
        -------
        unpack: str
            The stringified pointer unpacking statement
        """
        raise NotImplementedError

    @classmethod
    def _remove_const_array(cls, text, arry):
        """
        Similar to :func:`_remove_work_array_consts`, but for removing const defn
        of the given :param:`array`
        """

        replacers = [(re.compile(r'(double(?:\d+)?)\s*const\s*\*__restrict__\s*{}'.
                      format(re.escape(arry))),
                      r'\1 *__restrict__ {}'.format(re.escape(arry)))]
        for r, s in replacers:
            text = r.sub(s, text)
        return text

    @classmethod
    def _remove_work_array_consts(cls, text):
        """
        Hack -- TODO: need a way to specify that an array isn't constant even if
        the kernel in question doesn't write to it in loopy.
        """

        replacers = [(
            re.compile(r'(double const \*__restrict__ {})'.format(rhs_work_name)),
            r'double *__restrict__ {}'.format(rhs_work_name)), (
            re.compile(r'(__local volatile double const \*__restrict__ {})'.format(
                local_work_name)),
            r'__local volatile double *__restrict__ {}'.format(local_work_name)), (
            re.compile(r'(int const \*__restrict__ {})'.format(int_work_name)),
            r'int *__restrict__ {}'.format(int_work_name)), (
            re.compile(r'(long int const \*__restrict__ {})'.format(int_work_name)),
            r'long int *__restrict__ {}'.format(int_work_name))]
        for r, s in replacers:
            text = r.sub(s, text)
        return text

    def deconstify(self, text, readonly=None):
        """
        Convenience method to run :param:`text` through :func:`_remove_const_array`
        for all of :attr`in_arrays`and :attr:`out_arrays`
        """
        if readonly is None:
            readonly = []
        deconst = [arr for arr in self.in_arrays + self.out_arrays
                   if arr not in readonly]
        for arr in deconst:
            text = self._remove_const_array(text, arr)
        return text

    @classmethod
    def _remove_work_size(cls, text):
        """
        Hack -- TODO: whip up define-based array sizing for loopy
        """

        replacers = [  # full replacement
                     (re.compile(r'(, int const work_size, )'), r', '),
                     # rhs )
                     (re.compile(r'(, int const work_size\))'), r')'),
                     # lhs (
                     (re.compile(r'(\(int const work_size, )'), r'('),
                     (re.compile(r'(\(work_size, )'), '('),
                     (re.compile(r'(, work_size, )'), ', '),
                     (re.compile(r'(, work_size\))'), ')')]
        for r, s in replacers:
            text = r.sub(s, text)
        return text

    def _get_kernel_ownership(self):
        """
        Determine which generator in the dependency tree owns which kernel

        Returns
        -------
        owner: dict of str->:class:`kernel_generator`
            A mapping of kernel name to it's owner
        """

        # figure out ownership
        def __rec_dep_owner(gen, owner={}):
            if gen.depends_on:
                for dep in gen.depends_on:
                    owner = __rec_dep_owner(dep, owner)
            for k in gen.kernels:
                if k in gen.kernels and k.name not in owner:
                    owner[k.name] = gen
            return owner

        return __rec_dep_owner(self)

    def _merge_kernels(self, record, result, kernels=[], fake_calls={},
                       for_driver=False, cache={}):
        """
        Generate and merge the supplied kernels, and return the resulting code in
        string form

        Parameters
        ----------
        record: :class:`MemoryGenerationResult`
            The memory generation result containing the processed kernel data
        result: :class:`CodegenResult`
            The current code-gen result
        kernels: list of :class:`loopy.LoopKernel` []
            The kernels to merge, if not supplied, use :attr:`kernels`
        fake_calls: dict of str -> kernel_generator
            In some cases, e.g. finite differnce jacobians, we need to place a dummy
            call in the kernel that loopy will accept as valid.  Then it needs to
            be substituted with an appropriate call to the kernel generator's kernel
        for_driver: bool [False]
            Whether the kernels are being merged for a driver kernel

        Returns
        -------
        result: :class:`CodegenResult`
            The updated codegen result
        """

        if not kernels:
            kernels = self.kernels

        if not fake_calls:
            fake_calls = self.fake_calls

        # generate the kernel code
        preambles = []
        extra_kernels = []
        inits = {}
        instructions = []
        local_decls = []

        # figure out ownership
        owner = self._get_kernel_ownership()
        deps = self._get_deps(include_self=True)

        def _get_func_body(cgr, subs={}):
            """
            Returns the function declaration w/o initializers or preambles
            from a :class:`loopy.GeneratedProgram`
            """
            # get body
            if isinstance(cgr.ast, cgen.FunctionBody):
                body = str(cgr.ast)
            else:
                body = str(cgr.ast.contents[-1])

            # apply any substitutions
            for k, v in six.iteritems(subs):
                body = body.replace(k, v)

            # feed through get_code to get any corrections
            return lp_utils.get_code(body, self.loopy_opts)

        # split into bodies, preambles, etc.
        for i, k, in enumerate(kernels):
            # todo: hack -- until we have a proper OpenMP target in Loopy, we
            # need to set the inner work-size dimension to 1
            # (to avoid an explicit work-size loop)
            if self.lang == 'c':
                k = lp.fix_parameters(k, **{w_size.name: 1})

            # drivers own all their own kernels
            i_own = for_driver or (k.name in owner and owner[k.name] == self)
            dep_own = for_driver or (k.name in owner and owner[k.name] in deps)
            # make kernel
            cgr = None
            dp = None
            if i_own:
                # only generate code for kernels we actually own to avoid duplication
                cgr = lp.generate_code_v2(k)
                # grab preambles
                for _, preamble in cgr.device_preambles:
                    preamble = textwrap.dedent(preamble)
                    if preamble and preamble not in preambles:
                        preambles.append(preamble)

                # now scan device program
                assert len(cgr.device_programs) == 1
                dp = cgr.device_programs[0]

            init_list = {}
            if i_own and isinstance(dp.ast, cgen.Collection):
                # look for preambles
                for item in dp.ast.contents:
                    # initializers go in the preamble
                    if isinstance(item, cgen.Initializer):
                        def _rec_check_name(decl):
                            if 'name' in vars(decl):
                                return decl.name, decl.name in record.readonly
                            elif 'subdecl' in vars(decl):
                                return _rec_check_name(decl.subdecl)
                            return '', False
                        # check for migrated constant
                        name, const = _rec_check_name(item.vdecl)
                        if const:
                            continue
                        if name not in init_list:
                            init_list[name] = str(item)

                    # blanklines and bodies can be ignored (as they will be added
                    # below)
                    elif not (isinstance(item, cgen.Line)
                              or isinstance(item, cgen.FunctionBody)):
                        raise NotImplementedError(type(item))
                # and add to inits
                inits.update(init_list)
            else:
                # no preambles / initializers
                assert (not i_own) or isinstance(dp.ast, cgen.FunctionBody)

            # we need to place the call in the instructions and the extra kernels
            # in their own array

            if i_own:
                # only place the kernel defn in this file, IFF we own it
                extra = self._remove_work_array_consts(
                    self._remove_work_size(_get_func_body(dp, {})))
                extra_kernels.append(extra)
                if fake_calls:
                    # check to see if this kernel has a fake call to replace
                    fk = next((x for x in fake_calls if x.match(k, extra)), None)
                    if fk:
                        # replace call in instructions to call to kernel
                        knl_call = self._remove_work_size(self._get_kernel_call(
                            knl=fk.replace_with, passed_locals=local_decls))
                        extra_kernels[-1] = extra_kernels[-1].replace(
                            fk.dummy_call, knl_call[:-2])
                # and add defn to preamble
                preambles += [self._remove_work_array_consts(
                    self._remove_work_size(
                        lp_utils.get_header(k, codegen_result=cgr)))]

            # get instructions
            if i_own or dep_own:
                insns = self._remove_work_size(self._get_kernel_call(k))
                instructions.append(insns)

        # determine vector width
        vec_width = self.loopy_opts.depth
        if not bool(vec_width):
            vec_width = self.loopy_opts.width
        if not bool(self.vec_width):
            vec_width = 0

        # and save kernel data
        kernel = self._dummy_wrapper_kernel(
            record.kernel_data, record.readonly, vec_width,
            for_driver=for_driver)

        # insert barriers if any
        if not for_driver:
            instructions = self.apply_barriers(instructions)

        # add pointer unpacking
        if not for_driver:
            # driver places unpacks outside of loops
            instructions[0:0] = result.pointer_unpacks[:]

        # add local declaration to beginning of instructions
        instructions[0:0] = [str(x) for x in local_decls]

        # add any target preambles
        preambles = [x for x in self.target_preambles if x not in preambles] \
            + preambles
        preambles = [textwrap.dedent(x) for x in preambles]

        # and place in codegen
        return result.copy(instructions=instructions, preambles=preambles,
                           extra_kernels=extra_kernels, kernel=kernel,
                           inits=inits, name=self.name)

    def _get_deps(self, include_self=True):
        """
        Parameters
        ----------
        include_self: bool [True]
            If True, include the calling kernel generator at the front of the list
        Returns
        -------
        deps: list of :class:`kernel_generator`
            The recursive list of dependencies for this kernel generator
        """

        deps = [self] if include_self else []
        deps += [x for x in self.depends_on[:] if x not in deps]
        for dep in self.depends_on:
            deps += [x for x in dep._get_deps() if x not in deps]
        return deps

    def _deduplicate(self, record, results):
        """
        Handles de-duplication of constant array data and preambles in
        subkernels of the top-level wrapping kernel

        Parameters
        ----------
        record: :class:`MemoryGenerationResult` [None]
            The base wrapping kernel's memory results
        result: :class:`CodegenResult` [None]
            The base wrapping kernel's code-gen results

        Returns
        -------
        results: list of :class:`CodegenResult`
            The code-generation results for sub-kernels, with constants & preambles
            deduplicated. Note that the modified version of :param:`result` is
            stored in results[0]
        """

        # cleanup duplicate inits / premables
        init_list = {}
        preamble_list = []
        out = []
        for result in reversed(results):
            # remove shared inits
            result = result.copy(inits={k: v for k, v in six.iteritems(result.inits)
                                        if k not in init_list})
            result = result.copy(preambles=[v for v in result.preambles
                                            if v not in preamble_list])
            # update
            init_list.update(result.inits)
            preamble_list.extend(result.preambles)
            # and store
            out.append(result)

        return out

    def _set_dependencies(self, codegen_results):
        """
        Sets the dependency field of the codegen results for file generation

        Parameters
        ----------
        codegen_results:  list of :class:`CodegenResult`
            The (almost) finalized codegen results, listified by
            :func:`_deduplicate`

        Returns
        -------
        updated_results:  list of :class:`CodegenResult`
            The results with the :attr:`dependencies` set.
        """

        generators = self._get_deps(include_self=True)
        for i, result in enumerate(codegen_results):
            owner = next(x for x in generators if x.name == result.name)
            deps = owner._get_deps()
            codegen_results[i] = result.copy(
                dependencies=[x.name for x in deps if x != owner.name] +
                result.dependencies)

        return codegen_results

    def _set_kernel_data(self, record, for_driver=False):
        """
        Updates the :param:`record` to contain the correct :param:`kernel_data`

        Parameters
        ----------
        path : str
            The output path to write files to
        record: :class:`MemoryGenerationResult` [None]
            If not None, this wrapping kernel is being generated as a sub-kernel
            (and hence, should reuse the owning kernel's record)
        result: :class:`CodegenResult` [None]
            If not None, this wrapping kernel is being generated as a sub-kernel
            (and hence, should reuse the owning kernel's results)

        Returns
        -------
        record: :class:`MemoryGenerationResult`
            The resulting memory object
        """

        # our working data for the driver consists of:
        # 1. The dummy 'time' array
        # 2. All working data for the underlying kernels
        # 3. The global kernel args
        # 4. the problem size variable (if for_driver)

        # first, find kernel args global kernel args (by name)
        kernel_data = [x for x in record.args if x.name in set(
            self.in_arrays + self.out_arrays)]

        # and add problem size
        if for_driver:
            kernel_data.append(self._with_target(p_size))
        else:
            # add the time array w/ local size
            kernel_data += [time_array.copy(
                shape=(self.loopy_opts.initial_condition_dimsize,),
                order=self.loopy_opts.order)]
        # update
        kernel_data = record.kernel_data + kernel_data
        # and sort
        kernel_data = self.order_kernel_args(kernel_data)
        return record.copy(kernel_data=kernel_data)

    def _generate_wrapping_kernel(self, path, record=None, result=None,
                                  kernels=None, **kwargs):
        """
        Generates a wrapper around the various subkernels in this
        :class:`kernel_generator` (rather than working through loopy's fusion)

        Parameters
        ----------
        path : str
            The output path to write files to
        record: :class:`MemoryGenerationResult` [None]
            If not None, this wrapping kernel is being generated as a sub-kernel
            (and hence, should reuse the owning kernel's record)
        result: :class:`CodegenResult` [None]
            If not None, this wrapping kernel is being generated as a sub-kernel
            (and hence, should reuse the owning kernel's results)
        kernels: list of :class:`LoopKernel` [None]
            The kernels to generate, if not supplied (i.e., for the top-level
            kernel generator) use :attr:`kernels`

        Keyword Arguments
        -----------------
        return_codegen_results: bool [False]
            For testing only -- if True, return the codegen results for each of the
            dependent :class:`kernel_generator`
        return_memgen_records: bool [False]
            For testing only -- if True, return the memory generation results for
            each of the dependent :class:`kernel_generator`

        Returns
        -------
        callgen: :class:`CallgenResult`
            The current callgen result, containing the names of the generated files
        record: :class:`MemoryGenerationResult`
            The resulting memory object
        result: :class:`CodegenResult`
            The resulting code-generation object
        """

        assert all(
            isinstance(x, lp.LoopKernel) for x in self.kernels), (
            'Cannot generate wrapper before calling _make_kernels')

        # whether this is the top-level kernel
        is_owner = record is None
        assert (kernels is not None) != is_owner
        if not kernels:
            kernels = self.kernels

        if is_owner:
            # we must process the kernel args / memory / host constants on the owner
            # such that we have a consistent working buffer
            record, kernels = self._process_args(kernels)
            # process memory
            record, mem_limits = self._process_memory(record)
            # update subkernels for host constants
            if record.host_constants:
                kernels = self._migrate_host_constants(
                    kernels, record.host_constants)
            # generate working buffer
            record, result = self._compress_to_working_buffer(record)

            # specialize the pointers
            result = self._specialize_pointers(record, result)
        else:
            # make local copies of inputs
            result = result.copy()
            owner_record = record.copy()

            record, _ = self._process_args([x.copy() for x in self.kernels])

            # specialize the pointers
            result = self._specialize_pointers(record, result)

            # add any working buffers from the owner
            record = record.copy(kernel_data=record.kernel_data + [
                x for x in owner_record.kernel_data if x.name in [
                    int_work_name, rhs_work_name, local_work_name]])

        # get the kernel arguments for this :class:`kernel_generator`
        record = self._set_kernel_data(record)

        # add work size
        record = record.copy(kernel_data=record.kernel_data + [self._with_target(
            w_size)])

        # get the instructions, preambles and kernel
        result = self._merge_kernels(record, result, kernels=kernels)
        source_names = []
        if is_owner and self.depends_on:
            results = [result]
            # generate wrapper for deps
            deps = self._get_deps(include_self=False)
            memgen_records = [record]
            # generate subkernels
            for kgen in deps:
                _, mr, dr = kgen._generate_wrapping_kernel(path, record, result,
                                                           kernels=kernels)
                results.append(dr)
                memgen_records.append(mr)
            # remove duplicate constant/preamble definitions
            codegen_results = self._deduplicate(record, results)
            # and set sub-kernel dependencies
            codegen_results = self._set_dependencies(codegen_results)
        elif is_owner:
            codegen_results = [result]

        if is_owner:
            # write kernels to file
            for dr in codegen_results:
                source_names.append(self._to_file(path, dr))

        if is_owner and kwargs.get('return_codegen_results', False):
            return codegen_results

        if is_owner and kwargs.get('return_memgen_records', False):
            return memgen_records

        return CallgenResult(source_names=source_names), record, result

    def _to_file(self, path, result, for_driver=False):
        """
        Write the generated kernel data to file

        Parameters
        ----------
        path: str
            The directory to write to
        result: :class:`CodegenResult`
            The code-gen result to write to file
        for_driver: bool [False]
            Whether we're writing a driver kernel or not

        Returns
        -------
        filename: str
            The name of the generated file
        """

        # get filename
        basename = result.name
        name = basename
        if for_driver:
            name += '_driver'

        # first, load the wrapper as a template
        with open(os.path.join(
                script_dir,
                self.lang,
                'wrapping_kernel{}.in'.format(utils.file_ext[self.lang])),
                'r') as file:
            file_str = file.read()
            file_src = Template(file_str)

        # create the file
        filename = os.path.join(path, self.file_prefix + name + utils.file_ext[
            self.lang])
        with filew.get_file(filename, self.lang, include_own_header=True) as file:
            instructions = utils._find_indent(file_str, 'body', '\n'.join(
                result.instructions))
            lines = file_src.safe_substitute(
                defines='',
                preamble='',
                func_define=self.__get_kernel_defn(
                    result.kernel, remove_work_const=True),
                body=instructions,
                extra_kernels='\n'.join(result.extra_kernels))

            if self.auto_diff:
                lines = [x.replace('double', 'adouble') for x in lines]
            file.add_lines(lines)

        # and the header file
        headers = []
        if for_driver:
            # include header to base call
            headers.append(basename + utils.header_ext[self.lang])
            if utils.can_vectorize_lang[self.lang]:
                # add the vectorization header
                headers.append('vectorization' + utils.header_ext[self.lang])
        else:
            # include sub kernels
            for x in result.dependencies:
                headers.append(x + utils.header_ext[self.lang])

        # include the preambles as well, such that they can be
        # included into other files to avoid duplications
        preambles = '\n'.join(result.preambles + sorted(list(result.inits.values())))
        preambles = preambles.split('\n')
        preambles.extend([
            self.__get_kernel_defn(result.kernel, remove_work_const=True) +
            utils.line_end[self.lang]])

        with filew.get_header_file(
            os.path.join(path, self.file_prefix + name + utils.header_ext[
                self.lang]), self.lang) as file:

            file.add_headers(headers)
            if self.auto_diff:
                file.add_headers('adept.h')
                file.add_lines('using adept::adouble;\n')
                preambles = preambles.replace('double', 'adouble')
            file.add_lines(preambles)

        return filename

    def _get_local_unpacks(self, wrapper, args, null_args=[]):
        """
        Converts pointer unpacks from :param:`wrapper` to '_local' versions
        for the driver kernel

        Parameters
        ----------
        wrapper: :class:`CodegenResult`
            The code-generation object for the wrapped kernel
        args: list of :class:`loopy.ArrayArgs`
            The args to convert
        null_args: list of str
            The names of args to "unpack" as NULL

        Returns
        -------
        result: :class:`CodegenResult`
            The code-generation result with the updated pointer unpacks
        """

        def _name(arg):
            return arg.name + arc.local_name_suffix

        unpacks = [(_name(x), wrapper.pointer_offsets[x.name]) for x in args
                   if isinstance(x, lp.ArrayArg)]
        for null in null_args:
            unpacks.append((null.name, (null.dtype, 0, 0, scopes.GLOBAL)))

        local_unpacks = []
        for k, (dtype, size, offset, scope) in unpacks:
            if self.unique_pointers:
                # reset from inner kernel where each pointer had a single
                # workgroup / thread under consideration
                offset = '{} * {}'.format(offset, arc.work_size.name)
            local_unpacks.append(
                self._get_pointer_unpack(k, size, offset, dtype, scope,
                                         set_null=any(k == null.name for null
                                                      in null_args), for_driver=True)
                )

        return CodegenResult(pointer_unpacks=local_unpacks)

    def _generate_driver_kernel(self, path, wrapper_memory, wrapper_result,
                                callgen):
        """
        Generates a driver kernel that is responsible for looping through the entire
        set of initial conditions for testing / execution.  This is useful so that
        an external program can easily link to the wrapper kernel generated by this
        :class:`kernel_generator` and handle their own iteration over conditions
        (e.g., as in an ODE solver). :see:`driver-function` for more

        Parameters
        ----------
        path: str
            The path to place the driver kernel in
        wrapper_memory: :class:`MemoryGenerationResult`
            The memory configuration of the wrapper kernel we are trying to drive
        wrapper_result: :class:`CodegenResult`
            The resulting code-generation object
        callgen: :class:`CallgenResult`
            The current callgen result, to be updated with driver info

        Returns
        -------
        updated_callgen: :class:`CallgenResult`
            The updated callgen result w/ driver source, and IC limits added
        """

        from pyjac.core import driver_kernels as drivers

        # make driver kernels
        knl_info = drivers.get_driver(
                self.loopy_opts, self.namestore, self.in_arrays,
                self.out_arrays, self, test_size=self.test_size)

        if self.driver_type == DriverType.lockstep:
            template = drivers.lockstep_driver_template(self.loopy_opts, self)
        else:
            raise NotImplementedError

        our_arg_names = self.in_arrays + self.out_arrays

        kernels = self._make_kernels(knl_info, for_driver=True,
                                     # mark input / output arrays as un-simdable
                                     # to trigger correct scatter / gather copy
                                     # if necessary
                                     dont_split=our_arg_names)

        def localfy(kernel):
            """
            Return a copy of the kernel w/ our argument names converted to 'local'
            equivalensts
            """
            to_local_names = our_arg_names[:]
            if self.unique_pointers:
                to_local_names.extend([rhs_work_name, local_work_name,
                                       int_work_name])

            return kernel.copy(args=[
                x if x.name not in to_local_names
                else x.copy(name=x.name + arc.local_name_suffix)
                for x in kernel.args])

        # now we must modify the driver kernel, such that it expects the appropriate
        # data
        assert kernels[1].name == 'driver'
        kernels[1] = kernels[1].copy(
            args=kernels[1].args + [x for x in wrapper_memory.kernel_data
                                    if x not in kernels[1].args])
        # and rename to pass local arrays
        kernels[1] = localfy(kernels[1])

        # process arguments
        record, kernels = self._process_args(
            kernels, allowed_conflicts=our_arg_names)

        # process memory
        record, mem_limits = self._process_memory(record)

        # set kernel data
        record = self._set_kernel_data(record, for_driver=True)

        # and compress
        driver_memory, driver_result = self._compress_to_working_buffer(
            wrapper_memory.copy(), for_driver=True)

        # and add work data
        # add the sub-kernel's work arrays
        work_arrays = [x for x in driver_memory.kernel_data if x.name in
                       [rhs_work_name, local_work_name, int_work_name,
                        w_size.name]]
        # and remove the duplicates / smaller work arrays
        dupl = {}
        for arry in work_arrays:
            if isinstance(arry, lp.ValueArg):
                continue
            if arry.name not in dupl:
                dupl[arry.name] = arry
            elif arry.name in [rhs_work_name, local_work_name, int_work_name]:
                def _size(a):
                    from pymbolic import substitute
                    from pymbolic.primitives import Product, Sum
                    assert len(a.shape) == 1
                    if isinstance(a.shape[0], Product) or isinstance(
                            a.shape[0], Sum):
                        return substitute(a.shape[0], **{w_size.name: 1})
                    return a.shape[0]

                if _size(arry) > _size(dupl[arry.name]):
                    dupl[arry.name] = arry
        work_arrays = list(dupl.values())

        # next, we need to determine where in the working buffer the arrays
        # we need in the driver live
        result = self._get_local_unpacks(driver_result, record.kernel_data,
                                         null_args=[time_array])
        if self.unique_pointers:
            # add a local pointer unpack to the working buffers
            for wrk in [x for x in work_arrays if x.name in [
                        rhs_work_name, local_work_name, int_work_name]]:
                # find pointer unpack with smallest matching offset
                smallest = None
                for x, (dtype, size, offset, scope) in six.iteritems(
                        driver_result.pointer_offsets):
                    if not any(x == y.name for y in record.kernel_data):
                        continue
                    if scope != wrk.address_space:
                        continue
                    if smallest is None:
                        smallest = eval(offset)
                    elif eval(offset) < smallest:
                        smallest = eval(offset)
                if smallest is None:
                    assert len(wrk.shape) == 1  # should be lwk or iwk
                    smallest = wrk.shape[0]

                # and add a local unpack for the work buffer
                unpack = self._get_pointer_unpack(
                    wrk.name + '_local', smallest, '0', dtype,
                    wrk.address_space, for_driver=True)
                result = result.copy(pointer_unpacks=result.pointer_unpacks +
                                     [unpack])

        record = record.copy(kernel_data=self.order_kernel_args(
            record.kernel_data + work_arrays + [self._with_target(w_size)]))

        # get the instructions, preambles and kernel
        result = self._merge_kernels(
            record, result, kernels=kernels, fake_calls=[FakeCall(
                self.name + '()', kernels[1], localfy(wrapper_result.kernel))],
            for_driver=True)

        # remove constants from output args in result
        result.extra_kernels[1] = self.deconstify(
            result.extra_kernels[1], wrapper_memory.readonly)
        # and the header
        for i in range(len(result.preambles)):
            if 'driver(' in result.preambles[i]:
                result.preambles[i] = self.deconstify(result.preambles[i],
                                                      wrapper_memory.readonly)
                break

        if self.loopy_opts.depth:
            # insert barriers between:
            # first copy and kernel call
            barriers = [(0, 1, 'global')]
            # and the kernel call / copy-out
            barriers += [(1, 2, 'global')]
            result = result.copy(instructions=self.apply_barriers(
                result.instructions, barriers))

        # slot instructions into template
        result = result.copy(instructions=[
            utils.subs_at_indent(template, insns='\n'.join(result.instructions),
                                 unpacks='\n'.join(result.pointer_unpacks))])

        filename = self._to_file(path, result, for_driver=True)

        max_ic_per_run, max_ws_per_run = mem_limits.can_fit(memory_type.m_global)
        # normalize to divide evenly into vec_width
        if self.vec_width != 0:
            max_ic_per_run = np.floor(
                max_ic_per_run / self.vec_width) * self.vec_width

        work_arrays = self.order_kernel_args(
            [self._with_target(p_size)] + work_arrays)

        # update callgen
        callgen = callgen.copy(name=self.name,
                               source_names=callgen.source_names + [filename],
                               max_ic_per_run=int(max_ic_per_run),
                               max_ws_per_run=int(max_ws_per_run),
                               input_args={self.name: [
                                x for x in record.args if x.name in self.in_arrays]},
                               output_args={self.name: [
                                x for x in record.args
                                if x.name in self.out_arrays]},
                               work_arrays=work_arrays,
                               host_constants={
                                self.name: wrapper_memory.host_constants[:]})
        return callgen

    def remove_unused_temporaries(self, knl):
        """
        Convenience method to remove unused temporary variables from created
        :class:`loopy.LoopKernel`'s

        ...with exception of the arrays used in the preambles
        """
        new_args = []

        exp_knl = lp.expand_subst(knl)

        refd_vars = set(knl.all_params())
        for insn in exp_knl.instructions:
            refd_vars.update(insn.dependency_names())

        from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
        from loopy.symbolic import get_dependencies
        from itertools import chain

        def tolerant_get_deps(expr, parse=False):
            if expr is None or expr is lp.auto:
                return set()
            if parse and isinstance(expr, tuple):
                from loopy.kernel.array import _pymbolic_parse_if_necessary
                expr = tuple(_pymbolic_parse_if_necessary(x) for x in expr)
            return get_dependencies(expr)

        for ary in chain(knl.args, six.itervalues(knl.temporary_variables)):
            if isinstance(ary, ArrayBase):
                refd_vars.update(
                    tolerant_get_deps(ary.shape)
                    | tolerant_get_deps(ary.offset, parse=True))

                for dim_tag in ary.dim_tags:
                    if isinstance(dim_tag, FixedStrideArrayDimTag):
                        refd_vars.update(
                            tolerant_get_deps(dim_tag.stride))

        for arg in knl.temporary_variables:
            if arg in refd_vars:
                new_args.append(arg)

        return knl.copy(temporary_variables={arg: knl.temporary_variables[arg]
                                             for arg in new_args})

    def make_kernel(self, info, target, test_size, for_driver=False):
        """
        Convience method to create loopy kernels from :class:`knl_info`'s

        Parameters
        ----------
        info : :class:`knl_info`
            The rate contstant info to generate the kernel from
        target : :class:`loopy.TargetBase`
            The target to generate code for
        test_size : int/str
            The integer (or symbolic) problem size
        for_driver : bool [False]
            If True, include the work-size loop in full

        Returns
        -------
        knl : :class:`loopy.LoopKernel`
            The generated loopy kernel
        """

        # and the skeleton kernel
        skeleton = self.skeleton[:]

        # convert instructions into a list for convienence
        instructions = info.instructions
        if isinstance(instructions, str):
            instructions = textwrap.dedent(info.instructions)
            instructions = [x for x in instructions.split('\n') if x.strip()]

        # load inames
        if not info.iname_domain_override:
            our_inames, our_iname_domains = self.get_inames(
                test_size, for_driver=for_driver)
        else:
            our_inames, our_iname_domains = zip(*info.iname_domain_override)
            our_inames, our_iname_domains = list(our_inames), \
                list(our_iname_domains)

        inames = [info.var_name] + our_inames
        # add map instructions
        instructions = list(info.mapstore.transform_insns) + instructions

        # look for extra inames, ranges
        iname_range = []

        assumptions = info.assumptions[:]

        # find the start index for 'i'
        iname, iname_domain = info.mapstore.get_iname_domain()

        # add to ranges
        iname_range.append(iname_domain)
        iname_range.extend(our_iname_domains)

        assumptions = []
        assumptions.extend(self.get_assumptions(test_size, for_driver=for_driver))

        for iname, irange in info.extra_inames:
            inames.append(iname)
            iname_range.append(irange)

        # construct the kernel args
        pre_instructions = info.pre_instructions[:]
        post_instructions = info.post_instructions[:]

        def subs_preprocess(key, value):
            # find the instance of ${key} in kernel_str
            result = utils._find_indent(skeleton, key, value)
            return Template(result).safe_substitute(var_name=info.var_name)

        kernel_str = Template(skeleton).safe_substitute(
            var_name=info.var_name,
            pre=subs_preprocess('${pre}', '\n'.join(pre_instructions)),
            post=subs_preprocess('${post}', '\n'.join(post_instructions)),
            main=subs_preprocess('${main}', '\n'.join(instructions)))

        # finally do extra subs
        if info.extra_subs:
            kernel_str = Template(kernel_str).safe_substitute(
                **info.extra_subs)

        iname_arr = []
        # generate iname strings
        for iname, irange in zip(*(inames, iname_range)):
            if isinstance(iname, tuple):
                # multi-domain
                iname = ', '.join(iname)

            iname_arr.append(Template(
                '{[${iname}]:${irange}}').safe_substitute(
                iname=iname,
                irange=irange
            ))

        # get extra mapping data
        extra_kernel_data = [domain(node.iname)[0] for domain, node in
                             six.iteritems(info.mapstore.domain_to_nodes)
                             if not node.is_leaf()]

        extra_kernel_data += self.extra_kernel_data[:]

        # check for duplicate kernel data (e.g. multiple phi arguements)
        kernel_data = []
        for k in info.kernel_data + extra_kernel_data:
            if k not in kernel_data:
                kernel_data.append(k)

        # make the kernel
        knl = lp.make_kernel(iname_arr,
                             kernel_str,
                             kernel_data=kernel_data,
                             name=info.name,
                             target=target,
                             assumptions=' and '.join(assumptions),
                             default_offset=0,
                             **info.kwargs)
        # fix parameters
        if info.parameters:
            knl = lp.fix_parameters(knl, **info.parameters)
        if self.unique_pointers:
            # fix work size
            knl = lp.fix_parameters(knl, **{w_size.name: self.work_size})
        if not knl.loop_priority:
            # prioritize and return
            priority = []
            for iname in inames:
                try:
                    iname = iname.split(',')
                except AttributeError:
                    pass
                priority.extend(iname)
            knl = lp.prioritize_loops(knl, priority)
        # check manglers
        if info.manglers:
            knl = lp.register_function_manglers(knl, info.manglers)

        preambles = info.preambles + self.extra_preambles[:]
        # check preambles
        if preambles:
            # register custom preamble functions
            knl = lp.register_preamble_generators(knl, preambles)
            # also register their function manglers
            knl = lp.register_function_manglers(knl, [
                p.func_mangler for p in preambles])

        return self.remove_unused_temporaries(knl)

    @classmethod
    def apply_specialization(cls, loopy_opts, inner_ind, knl,
                             vecspec=None, can_vectorize=True,
                             get_specialization=False,
                             unrolled_vector=False):
        """
        Applies wide / deep vectorization and/or ILP loop unrolling
        to a loopy kernel

        Parameters
        ----------
        loopy_opts : :class:`loopy_options` object
            A object containing all the loopy options to execute
        inner_ind : str
            The inner loop index variable
        knl : :class:`loopy.LoopKernel`
            The kernel to transform
        vecspec : :function:
            An optional specialization function that is applied after
            vectorization to fix hanging loopy issues
        can_vectorize : bool
            If False, cannot be vectorized in the normal manner, hence
            vecspec must be used to vectorize.
        get_specialization : bool [False]
            If True, the specialization will not be _applied_ to the kernel, instead
            a dictionary mapping inames -> tags will be returned
        unrolled_vector : bool [False]
            If True, apply 'unr' instead of 'vec' for explicit-SIMD inames.
            Useful for driver kernels

        Returns
        -------
        knl : :class:`loopy.LoopKernel`
            The transformed kernel

        OR

        iname_map: dict
            A dictionary mapping inames -> tags, only returned if
            :param:`get_specialization` is True
        """

        # before doing anything, find vec width
        # and split variable
        vec_width = None
        to_split = None
        i_tag = inner_ind
        j_tag = global_ind
        depth = loopy_opts.depth
        width = loopy_opts.width
        if depth:
            to_split = inner_ind
            vec_width = depth
            i_tag += '_outer'
        elif width:
            to_split = global_ind
            vec_width = width
            j_tag += '_outer'
        if not can_vectorize:
            assert vecspec is not None, ('Cannot vectorize a non-vectorizable '
                                         'kernel {} without a specialized '
                                         'vectorization function'.format(
                                             knl.name))
        specialization = {}

        # if we're splitting
        # apply specified optimizations
        if to_split and can_vectorize:
            # and assign the l0 axis to the correct variable
            tag = 'l.0'
            if loopy_opts.is_simd:
                tag = 'vec' if not unrolled_vector else 'unr'
            if get_specialization:
                specialization[to_split + '_inner'] = tag
            elif loopy_opts.pre_split:
                # apply the pre-split
                knl = lp.tag_inames(knl, [(to_split + '_inner', tag)])
            else:
                knl = lp.split_iname(knl, to_split, vec_width, inner_tag=tag)

        if utils.can_vectorize_lang[loopy_opts.lang]:
            # tag 'global_ind' as g0, use simple parallelism
            if get_specialization:
                specialization[j_tag] = 'g.0'
            else:
                knl = lp.tag_inames(knl, [(j_tag, 'g.0')])

        # if we have a specialization
        if vecspec and not get_specialization:
            knl = vecspec(knl)

        if bool(vec_width) and not loopy_opts.is_simd and not get_specialization:
            # finally apply the vector width fix above
            ggs = vecwith_fixer(knl.copy(), vec_width)
            knl = knl.copy(overridden_get_grid_sizes_for_insn_ids=ggs)

        # now do unr / ilp
        if loopy_opts.unr is not None:
            if get_specialization:
                specialization[i_tag + '_inner'] = 'unr'
            else:
                knl = lp.split_iname(knl, i_tag, loopy_opts.unr, inner_tag='unr')
        elif loopy_opts.ilp:
            if get_specialization:
                specialization[i_tag] = 'ilp'
            else:
                knl = lp.tag_inames(knl, [(i_tag, 'ilp')])

        return knl if not get_specialization else specialization


class c_kernel_generator(kernel_generator):

    """
    A C-kernel generator that handles OpenMP parallelization
    """

    def __init__(self, *args, **kwargs):
        super(c_kernel_generator, self).__init__(*args, **kwargs)

    def get_inames(self, test_size, for_driver=False):
        """
        Returns the inames and iname_ranges for subkernels created using
        this C kernel-generator

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel
        for_driver : bool [False]
            If True, utilize the entire test size

        Returns
        -------
        inames : list of str
            The string inames to add to created subkernels by default
        iname_domains : list of str
            The iname domains to add to created subkernels by default
        """

        # Currently C has no vectorization capabilities, and unless we're in a driver
        # function, we should only be executing the kernel once, hence:

        if not (for_driver or self.for_testing):
            return [global_ind], ['0 <= {} < 1'.format(global_ind)]

        if not self.for_testing:
            test_size = p_size.name

        return [global_ind],  ['0 <= {} < {}'.format(global_ind, test_size)]

    def _get_pointer_unpack(self, array, size, offset, dtype, scope=scopes.GLOBAL,
                            set_null=False, for_driver=False):
        """
        A method stub to implement the pattern:
        ```
            double* array = &rwk[offset]
        ```
        per target.  By default this returns the pointer unpack for C, but it may
        be overridden in subclasses

        Parameters
        ----------
        array: str
            The array name
        size: str
            The size of the array
        offset: str
            The stringified offset
        dtype: :class:`loopy.LoopyType`
            The array type
        scope: :class:`loopy.AddressSpace`
            The memory scope
        set_null: bool [False]
            If True, set the unpacked pointer to NULL
        for_driver: bool [False]
            If True, this pointer is being unpacked for the driver, as such
            any value of :attr:`unique_pointers` should be ignored

        Returns
        -------
        unpack: str
            The stringified pointer unpacking statement
        """

        if set_null:
            return '{dtype}* __restrict__ {array} = NULL;'.format(
                dtype=self.type_map[dtype],
                array=array)

        unique = ' + {size} * omp_get_thread_num()'.format(size=size)
        if self.unique_pointers and not for_driver:
            unique = ''

        return ('{dtype}* __restrict__ {array} = {work} + {offset}'
                '{unique};'.format(
                    dtype=self.type_map[dtype],
                    array=array,
                    work=rhs_work_name,
                    offset=offset,
                    unique=unique))

    @property
    def target_preambles(self):
        """
        Preambles for OpenMP

        Notes
        -----
        This defines the work-size variable for OpenCL as the number of groups
        launched by the OpenCL kernel (if the user has not specified a value)

        Returns
        -------
        premables: list of str
            The string preambles for this :class:`kernel_generator`
        """

        work_size = """
        #ifndef work_size
            #define work_size (omp_get_num_threads())
        #endif
        """

        return [work_size]

    def _special_kernel_subs(self, path, callgen):
        """
        An override of the :method:`kernel_generator._special_wrapping_subs`
        that implements C-specific wrapping kernel arguement passing

        Parameters
        ----------
        path : str
            The output path to write files to
        callgen : :class:`CallgenResult`
            The intermediate call-generation store

        Returns
        -------
        updated : :class:`CallgenResult`
            The updated call-generation storage
        """

        return callgen

    def get_assumptions(self, test_size, for_driver=False):
        """
        Returns a list of assumptions on the loop domains
        of generated subkernels

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel
        for_driver: bool [False]
            If this kernel is a driver function

        Returns
        -------

        assumptions : list of str
            List of assumptions to apply to the generated sub kernel
        """

        # we never need these for C
        return []


class autodiff_kernel_generator(c_kernel_generator):

    """
    A C-Kernel generator specifically designed to work with the
    autodifferentiation scheme.  Handles adding jacobian, etc.
    """

    def __init__(self, *args, **kwargs):

        from pyjac.loopy_utils.loopy_utils import AdeptCompiler
        kwargs.setdefault('compiler', AdeptCompiler())
        super(autodiff_kernel_generator, self).__init__(*args, **kwargs)

    def add_jacobian(self, jacobian):
        """
        Adds the jacobian object to the extra kernel data for inclusion in
        generation (to be utilized during the edit / AD process)

        Parameters
        ----------

        jacobian : :class:`loopy.GlobalArg`
            The loopy arguement to add to the method signature

        Returns
        -------
        None
        """

        self.extra_kernel_data.append(jacobian)


class ispc_kernel_generator(kernel_generator):

    def __init__(self, *args, **kwargs):
        super(ispc_kernel_generator, self).__init__(*args, **kwargs)

    # TODO: fill in


class opencl_kernel_generator(kernel_generator):

    """
    An opencl specific kernel generator
    """

    def __init__(self, *args, **kwargs):
        super(opencl_kernel_generator, self).__init__(*args, **kwargs)

        self.barrier_templates = {
            'global': 'barrier(CLK_GLOBAL_MEM_FENCE)',
            'local': 'barrier(CLK_LOCAL_MEM_FENCE)'
        }

        # set atomic types
        self.type_map[to_loopy_type(np.float64, for_atomic=True,
                                    target=self.target)] = 'double'
        self.type_map[to_loopy_type(np.int32, for_atomic=True,
                                    target=self.target)] = 'int'
        self.type_map[to_loopy_type(np.int64, for_atomic=True,
                                    target=self.target)] = 'long int'

    @property
    def target_preambles(self):
        """
        Preambles for OpenCL

        Notes
        -----
        This defines the work-size variable for OpenCL as the number of groups
        launched by the OpenCL kernel (if the user has not specified a value)

        Returns
        -------
        premables: list of str
            The string preambles for this :class:`kernel_generator`
        """

        work_size = """
        #ifndef work_size
            #define work_size (({int_type}) get_num_groups(0))
        #endif
        """.format(int_type=self.type_map[to_loopy_type(arc.kint_type)])

        return [work_size]

    def _get_pointer_unpack(self, array, size, offset, dtype, scope=scopes.GLOBAL,
                            set_null=False, for_driver=False):
        """
        Implement the pattern
        ```
            __scope double __restrict__* array = &rwk[offset]
        ```
        for OpenCL

        Parameters
        ----------
        array: str
            The array name
        size: str
            The size of the array
        offset: str
            The stringified offset
        dtype: :class:`loopy.LoopyType`
            The array type
        scope: :class:`loopy.AddressSpace`
            The memory scope
        set_null: bool [False]
            If True, set the unpacked pointer to NULL
        for_driver: bool [False]
            If True, this pointer is being unpacked for the driver, as such
            any value of :attr:`unique_pointers` should be ignored

        Returns
        -------
        unpack: str
            The stringified pointer unpacking statement
        """

        dtype = self.type_map[dtype]
        if scope == scopes.GLOBAL:
            scope_str = 'global'
            volatile = ''
            work_str = rhs_work_name
        elif scope == scopes.LOCAL:
            scope_str = 'local'
            volatile = ' volatile'
            work_str = local_work_name
        else:
            raise NotImplementedError
        scope_str = '__{}'.format(scope_str)

        cast = ''
        if self.loopy_opts.is_simd:
            # convert to double4 etc
            if not set_null:
                dtype += str(self.vec_width)
            cast = '({} {}*)'.format(scope_str, dtype)

        unique = ''
        if self.unique_pointers and for_driver:
            unique = ' + {} * {}'.format(size, 'get_group_id(0)')

        if set_null:
            return '{scope}{volatile} {dtype}* __restrict__ {array} = 0;'.format(
                scope=scope_str, volatile=volatile,
                dtype=dtype, array=array)

        return ('{scope}{volatile} {dtype}* __restrict__ {array}'
                ' = {cast}({work_str} + {offset}{unique});').format(
            scope=scope_str, volatile=volatile,
            dtype=dtype, array=array, cast=cast,
            work_str=work_str, offset=offset, unique=unique)

    def _special_kernel_subs(self, path, callgen):
        """
        An override of the :method:`kernel_generator._special_kernel_subs`
        that implements OpenCL specific kernel substitutions

        Parameters
        ----------
        path : str
            The output path to write files to
        callgen : :class:`CallgenResult`
            The intermediate call-generation store

        Returns
        -------
        updated : :class:`CallgenResult`
            The updated call-generation storage
        """

        # open cl specific
        callgen = callgen.copy(
            cl_level=int(float(self._get_cl_level()) * 100),
            platform=self.platform_str,
            build_options=self.build_options(path),
            dev_mem_type=(DeviceMemoryType.pinned if self.use_pinned else
                          DeviceMemoryType.mapped),
            device_type=self.loopy_opts.device_type
            )

        return callgen

    def _get_cl_level(self):
        """
        Searches the supplied platform for a OpenCL level.  If not found,
        uses the level from the site config

        Parameters
        ----------
        None

        Returns
        -------
        cl_level: str
            The stringified OpenCL standard level
        """

        # try get the platform's CL level
        try:
            device_level = self.loopy_opts.device.opencl_c_version.split()
            for d in device_level:
                try:
                    float(d)
                    return d
                    break
                except ValueError:
                    pass
        except AttributeError:
            # default to the site level
            return site.CL_VERSION

    def build_options(self, path):
        """
        Returns a string version of the OpenCL build options,
        based on the :file:`siteconf.py`
        """
        # for the build options, we turn to the siteconf
        build_options = ['-I' + x for x in site.CL_INC_DIR + [path]]
        build_options.extend(site.CL_FLAGS)
        build_options.append('-cl-std=CL{}'.format(self._get_cl_level()))
        return ' '.join(build_options)

    @property
    def platform_str(self):
        # get the platform from the options
        if self.loopy_opts.platform_is_pyopencl:
            platform_str = self.loopy_opts.platform.get_info(
                cl.platform_info.VENDOR)
        else:
            logger = logging.getLogger(__name__)
            logger.warn('OpenCL platform name "{}" could not be checked as '
                        'PyOpenCL not found, using user supplied platform '
                        'name.'.format(self.loopy_opts.platform_name))
            platform_str = self.loopy_opts.platform_name
        return platform_str

    def _generate_compiling_program(self, path, callgen):
        """
        Needed for OpenCL, this generates a simple C file that
        compiles and stores the binary OpenCL kernel generated w/ the wrapper

        Parameters
        ----------
        path : str
            The output path to write files to
        callgen: :class:`CallgenResult`
            The current callgen result

        Returns
        -------
        callgen: :class:`CallgenResult`
            The callgen result, updated with the path to the compiler program
        """

        outname = os.path.join(path, self.name + '.bin')
        result = CompgenResult(name=self.name,
                               source_names=callgen.source_names[:],
                               platform=self.platform_str,
                               outname=outname,
                               build_options=self.build_options(path))
        # serialize
        compout = os.path.join(path, 'comp.pickle')
        with open(compout, 'wb') as file:
            pickle.dump(result, file)

        # input
        infile = os.path.join(
            script_dir, self.lang, 'opencl_kernel_compiler.cpp.in')

        # output
        filename = os.path.join(
            path, self.name + '_compiler' + utils.file_ext[self.lang])

        # call cog
        try:
            Cog().callableMain([
                    'cogapp', '-e', '-d', '-Dcompgen={}'.format(compout),
                    '-o', filename, infile])
        except Exception:
            logger = logging.getLogger(__name__)
            logger.error('Error generating compiling file {}'.format(filename))
            raise

        return callgen.copy(source_names=callgen.source_names + [filename],
                            binname=result.outname)

    def apply_barriers(self, instructions, barriers=None):
        """
        An override of :method:`kernel_generator.apply_barriers` that
        applies synchronization barriers to OpenCL kernels

        Parameters
        ----------

        instructions: list of str
            The instructions for this kernel
        barriers: list of (int, int)
            The integer indicies between which to insert instructions
            If not supplied, :attr:`barriers` will be used

        Returns
        -------

        synchronized_instructions : list of str
            The instruction list with the barriers inserted
        """

        # first, recursively apply barriers
        if barriers is None:
            barriers = [b for d in reversed(self._get_deps(include_self=True))
                        for b in d.barriers]

        instructions = list(enumerate(instructions))
        for barrier in barriers:
            # find insert index (the second barrier ind)
            index = next(ind for ind, inst in enumerate(instructions)
                         if inst[0] == barrier[1])
            # check that we're inserting between the required barriers
            assert barrier[0] == instructions[index - 1][0]
            # and insert
            instructions.insert(index, (-1, self.barrier_templates[barrier[2]]
                                        + utils.line_end[self.lang]))
        # and get rid of indicies
        instructions = [inst[1] for inst in instructions]
        return instructions

    @property
    def hoist_locals(self):
        """
        In OpenCL we need to strip out any declaration of a __local variable in
        subkernels, as these must be defined in the called in the kernel scope

        This entails hoisting local declarations up to the wrapping
        kernel for non-separated OpenCL kernels as __local variables in
        sub-functions are not well defined in the standard:
        https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/functionQualifiers.html # noqa
        """

        return True


class knl_info(object):

    """
    A composite class that contains the various parameters, etc.
    needed to create a simple kernel

    name : str
        The kernel name
    instructions : str or list of str
        The kernel instructions
    mapstore : :class:`array_creator.MapStore`
        The MapStore object containing map domains, indicies, etc.
    pre_instructions : list of str
        The instructions to execute before the inner loop
    post_instructions : list of str
        The instructions to execute after end of inner loop but before end
        of outer loop
    var_name : str
        The inner loop variable
    kernel_data : list of :class:`loopy.ArrayBase`
        The arguements / temporary variables for this kernel
    extra_inames : list of tuple
        A list of (iname, domain) tuples the form the extra loops in this kernel
    assumptions : list of str
        Assumptions to pass to the loopy kernel
    parameters : dict
        Dictionary of parameter values to fix in the loopy kernel
    extra subs : dict
        Dictionary of extra string substitutions to make in kernel generation
    can_vectorize : bool
        If False, the vectorization specializer must be used to vectorize this kernel
    vectorization_specializer : function
        If specified, use this specialization function to fix problems that would
        arise in vectorization
    split_specializer : function
        If specified, run this function to fixup an hanging ends after the
        kernel splits are applied
    unrolled_vector : bool [False]
        If true, apply 'unr' instead of 'vec' to any resulting explicit-SIMD iname
    **kwargs: dict
        Any other keyword args to pass to :func:`loopy.make_kernel`
    """

    def __init__(self, name, instructions, mapstore, pre_instructions=[],
                 post_instructions=[],
                 var_name='i', kernel_data=None,
                 extra_inames=[],
                 assumptions=[], parameters={},
                 extra_subs={},
                 vectorization_specializer=None,
                 can_vectorize=True,
                 manglers=[],
                 iname_domain_override=[],
                 split_specializer=None,
                 unrolled_vector=False,
                 preambles=[],
                 **kwargs):

        def __listify(arr):
            if isinstance(arr, str):
                return [arr]
            return arr
        self.name = name
        self.instructions = instructions
        self.mapstore = mapstore
        self.pre_instructions = __listify(pre_instructions)[:]
        self.post_instructions = __listify(post_instructions)[:]
        self.var_name = var_name
        if isinstance(kernel_data, set):
            kernel_data = list(kernel_data)
        self.kernel_data = kernel_data[:]
        self.extra_inames = extra_inames[:]
        self.assumptions = assumptions[:]
        self.parameters = parameters.copy()
        self.extra_subs = extra_subs
        self.can_vectorize = can_vectorize
        self.vectorization_specializer = vectorization_specializer
        self.split_specializer = split_specializer
        self.manglers = []
        # copy if supplied
        self.preambles = [x for x in preambles]
        for mangler in manglers:
            if isinstance(mangler, PreambleMangler):
                self.manglers.extend(mangler.manglers)
                self.preambles.extend(mangler.preambles)
            elif isinstance(mangler, lp_pregen.MangleGen):
                self.manglers.append(mangler)
            else:
                self.preambles.append(mangler)
                self.manglers.append(mangler.func_mangler)

        self.iname_domain_override = iname_domain_override[:]
        self.kwargs = kwargs.copy()
        self.unrolled_vector = unrolled_vector


def create_function_mangler(kernel, return_dtypes=()):
    """
    Returns a function mangler to interface loopy kernels with function calls
    to other kernels (e.g. falloff rates from the rate kernel, etc.)

    Parameters
    ----------
    kernel : :class:`loopy.LoopKernel`
        The kernel to create an interface for
    return_dtypes : list :class:`numpy.dtype` returned from the kernel, optional
        Most likely an empty list
    Returns
    -------
    func : :method:`MangleGen`.__call__
        A function that will return a :class:`loopy.kernel.data.CallMangleInfo` to
        interface with the calling :class:`loopy.LoopKernel`
    """
    from ..loopy_utils.preambles_and_manglers import MangleGen

    dtypes = []
    for arg in kernel.args:
        if not isinstance(arg, lp.TemporaryVariable):
            dtypes.append(arg.dtype)
    mg = MangleGen(kernel.name, tuple(dtypes), return_dtypes)
    return mg.__call__
