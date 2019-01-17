from __future__ import print_function

import logging
import os
import stat
import re
import six
from string import Template

# package imports
import loopy as lp
from loopy.target.c.c_execution import CPlusPlusCompiler
import numpy as np
import warnings

try:
    import pyopencl as cl
    from pyopencl.tools import clear_first_arg_caches
except ImportError:
    cl = None
    pass

# local imports
from pyjac import utils
from pyjac.core.enum_types import (RateSpecialization, JacobianType, JacobianFormat,
                                   KernelType)
from pyjac.core import array_creator as arc
from pyjac.core.exceptions import (MissingPlatformError, MissingDeviceError,
                                   BrokenPlatformError)
from pyjac.loopy_utils.loopy_edit_script import substitute as codefix
from pyjac.schemas import build_and_validate

edit_script = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'loopy_edit_script.py')
adept_edit_script = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 'adept_edit_script.py')


def load_platform(codegen):
    """
    Loads a code-generation platform from a file, and returns the corresponding
    :class:`loopy_options`

    Parameters
    ----------
    codegen: str
        The user-specified code-generation platform yaml file

    Returns
    -------
    :class:`loopy_options`
        The loaded platform

    Raises
    ------
    :class:`cerberus.ValidationError`: A validation error if the supplied codegen
        platform doesn't comply with the :doc:`../schemas/codegen_platform.yaml`
    """

    platform = build_and_validate('codegen_platform.yaml', codegen)['platform']
    width = platform.get('width', None)
    depth = platform.get('depth', None)
    # TODO: implement memory limits loading here
    # optional params get passed as kwargs
    kwargs = {}
    if 'order' in platform and platform['order'] is not None:
        kwargs['order'] = platform['order']
    if 'atomic_doubles' in platform:
        kwargs['use_atomic_doubles'] = platform['atomic_doubles']
    if 'atomic_ints' in platform:
        kwargs['use_atomic_ints'] = platform['atomic_ints']
    return loopy_options(width=width, depth=depth, lang=platform['lang'],
                         platform=platform['name'], **kwargs)


class loopy_options(object):

    """
    Loopy Objects class

    Attributes
    ----------
    width : int
        If not None, the SIMD lane/SIMT block width.
        Cannot be specified along with depth
    depth : int
        If not None, the SIMD lane/SIMT block depth.
        Cannot be specified along with width
    ilp : bool
        If True, use the ILP tag on the species loop.
        Cannot be specified along with unr
    unr : int
        If not None, the unroll length to apply to the species loop.
        Cannot be specified along with ilp
    order : {'C', 'F'}
        The memory layout of the arrays, C (row major)
        or Fortran (column major)
    lang : ['opencl', 'c', 'cuda']
        One of the supported languages
    rate_spec : RateSpecialization
        Controls the level to which Arrenhius rate evaluations are specialized
    rate_spec_kernels : bool
        If True, break different Arrenhius rate specializations into different
        kernels
    rop_net_kernels : bool
        If True, break different ROP values (fwd / back / pdep) into different
        kernels
    platform : {'CPU', 'GPU', or other vendor specific name}
        The OpenCL platform to run on.
        *   If 'CPU' or 'GPU', the first available matching platform will be
            used
        *   If a vendor specific string, it will be passed to pyopencl to get
            the platform
    use_atomic_doubles : bool [True]
        Use atomic updates where necessary for proper deep-vectorization
        If not, a sequential deep-vectorization (with only one thread/lane
        active) will be used
    use_atomic_ints : bool [True]
        Use atomic integer operations for the driver kernel.
    jac_type: :class:`JacobianType` [JacobianType.full]
        The type of Jacobian kernel (full or approximate) to generate
    jac_format: :class:`JacobianFormat` [JacobianFormat.full]
        The format of Jacobian kernel (full or sparse) to generate
    is_simd: bool [None]
        If supplied, override the user-specified flag :param:`explicit_simd`, used
        for testing.
    unique_pointers: bool [False]
        If specified, this indicates that the pointers passed to the generated pyJac
        methods will be unique (i.e., distinct per OpenMP thread /
        OpenCL work-group). This option is most useful for coupling to external
        codes an that have already been parallelized.
    explicit_simd: bool [False]
        Attempt to utilize explict-SIMD instructions in OpenCL
    """
    def __init__(self, width=None, depth=None, ilp=False, unr=None,
                 lang='opencl', order='C', rate_spec=RateSpecialization.fixed,
                 rate_spec_kernels=False, rop_net_kernels=False,
                 platform='', kernel_type=KernelType.jacobian, auto_diff=False,
                 use_atomic_doubles=True, use_atomic_ints=True,
                 jac_type=JacobianType.exact, jac_format=JacobianFormat.full,
                 device=None, device_type=None, is_simd=None,
                 unique_pointers=False, explicit_simd=None):
        self.width = width
        self.depth = depth
        if not utils.can_vectorize_lang[lang]:
            assert not (width or depth), (
                "Can't use a vectorized form with unvectorizable language,"
                " {}".format(lang))
        assert not (width and depth), (
            'Cannot use deep and wide vectorizations simulataneously')
        self.ilp = ilp
        self.unr = unr
        utils.check_lang(lang)
        self.lang = lang
        utils.check_order(order)
        self.order = order
        self.rate_spec = utils.to_enum(rate_spec, RateSpecialization)
        self.rate_spec_kernels = rate_spec_kernels
        self.rop_net_kernels = rop_net_kernels
        self.platform = platform
        self.device_type = device_type
        self.device = device
        self.auto_diff = auto_diff
        self.use_atomic_doubles = use_atomic_doubles
        self.use_atomic_ints = use_atomic_ints
        self.jac_format = utils.to_enum(jac_format, JacobianFormat)
        self.jac_type = utils.to_enum(jac_type, JacobianType)
        self._is_simd = is_simd
        self.explicit_simd = explicit_simd
        self.explicit_simd_warned = False
        if self.lang != 'opencl' and self.explicit_simd:
            logger = logging.getLogger(__name__)
            logger.warn('explicit-SIMD flag has no effect on non-OpenCL targets.')
        self.kernel_type = utils.to_enum(kernel_type, KernelType)
        self.unique_pointers = unique_pointers

        if self._is_simd or self.explicit_simd:
            assert width or depth, (
                'Cannot use explicit SIMD types without vectorization')

        # need to find the first platform that has the device of the correct
        # type
        if self.lang == 'opencl' and not self.platform_is_pyopencl \
                and cl is not None:
            self.device_type = cl.device_type.ALL
            check_name = None
            if self.platform_name.lower() == 'cpu':
                self.device_type = cl.device_type.CPU
            elif self.platform_name.lower() == 'gpu':
                self.device_type = cl.device_type.GPU
            elif self.platform_name.lower() == 'accelerator':
                self.device_type = cl.device_type.ACCELERATOR
            else:
                check_name = self.platform
            self.platform = None
            platforms = cl.get_platforms()
            for p in platforms:
                try:
                    cl.Context(
                        dev_type=self.device_type,
                        properties=[(cl.context_properties.PLATFORM, p)])
                    if not check_name or check_name.lower() in p.get_info(
                            cl.platform_info.NAME).lower():
                        self.platform = p
                        break
                except cl.cffi_cl.RuntimeError:
                    pass
            if not self.platform:
                raise MissingPlatformError(platform)
            if not isinstance(self.device, cl.Device) and (
                    self.device_type is not None):
                # finally a matching device
                self.device = self.platform.get_devices(
                    device_type=self.device_type)
                if not self.device:
                    raise MissingDeviceError(self.device_type, self.platform)
                self.device = self.device[0]
                self.device_type = self.device.get_info(cl.device_info.TYPE)
        elif self.lang == 'opencl':
            self.device_type = 'CL_DEVICE_TYPE_ALL'

        # check for broken vectorizations
        self.raise_on_broken()

    @property
    def limit_int_overflow(self):
        """
        Deals with issue of integer overflow in array indexing
        """
        return self.lang == 'c' or self.lang == 'opencl' and \
            ('intel' in self.platform_name.lower() or
             'portable' in self.platform_name.lower())

    def raise_on_broken(self):
        # Currently, NVIDIA w/ neither deep nor wide-vectorizations (
        #   i.e. a "parallel" implementation) breaks sometimes on OpenCL
        if self.lang == 'opencl' and cl is not None:
            if not (self.width or self.depth) \
                    and self.device_type == cl.device_type.GPU:
                if 'nvidia' in self.platform_name.lower():
                    raise BrokenPlatformError(self)
                # otherwise, simply warn
                logger = logging.getLogger(__name__)
                logger.warn('Some GPU implementation(s)--NVIDIA--give incorrect'
                            'values sporadically without either a deep or wide'
                            'vectorization. Use at your own risk.')
            if self.width and not self.is_simd and \
                    self.device_type == cl.device_type.CPU:
                logger = logging.getLogger(__name__)
                if 'intel' in self.platform_name.lower():
                    logger.error('Intel OpenCL is currently broken for wide, '
                                 'non-explicit-SIMD vectorizations on the CPU.  '
                                 'Use the --explicit_simd flag.')
                    raise BrokenPlatformError(self)
                if not self.explicit_simd and self._is_simd is None:
                    # only warn if user didn't supply
                    logger.warn('You may wish to use the --explicit_simd flag to '
                                'utilize explicit-vector data-types (and avoid '
                                'implicit vectorization, which may yield sub-optimal'
                                ' results).')
            if 'portable' in self.platform_name.lower() and self.unique_pointers:
                logger = logging.getLogger(__name__)
                logger.error('Portable OpenCL is currently broken for '
                             'unique_pointers.')
                raise BrokenPlatformError(self)

    @property
    def is_simd(self):
        """
        Utility to determine whether to tell Loopy to apply explicit-simd
        vectorization or not

        Returns
        -------
        is_simd: bool
            True if we should attempt to explicitly vectorize the data / arrays
        """

        # priority to test-specification
        if self._is_simd is not None:
            return self._is_simd

        if not (self.width or self.depth):
            return False

        # currently SIMD is enabled only wide-CPU vectorizations (
        # deep-vectorizations will require further loopy upgrades)

        if not self.width:
            if self.explicit_simd:
                logger = logging.getLogger(__name__)
                logger.warn('Explicit-SIMD deep-vectorization currently not '
                            'implemented, ignoring user-specified SIMD flag')
            return False

        if self.explicit_simd is not None:
            # user specified
            return self.explicit_simd

        if not cl:
            if self.explicit_simd is None and not self.explicit_simd_warned:
                logger = logging.getLogger(__name__)
                logger.warn('Cannot determine whether to use explicit-SIMD '
                            'instructions as PyOpenCL was not found.  Either '
                            'install PyOpenCL or use the "--explicit_simd" '
                            'command line argument. Assuming not SIMD.')
                self.explicit_simd_warned = True
            return self.explicit_simd

        if self.lang == 'opencl':
            return self.device_type != cl.device_type.GPU
        return True

    @property
    def pre_split(self):
        """
        It is sometimes advantageous to 'pre-split' the outer loop into an
        inner (vector) iname and an outer (parallel) iname, particularly when
        using explicit-SIMD w/ loopy (and avoid having to figure out how to simplify
        floor-div's of the problem size in loopy)

        If this property is True, utilize a pre-split.
        """

        return self.width and arc.array_splitter._have_split_static(self)

    @property
    def initial_condition_dimsize(self):
        """
        Return the necessary IC dimension size based on this :class:`loopy_options`
        """

        ws = arc.work_size.name
        if not self.pre_split and self.width:
            return '{}*{}'.format(ws, self.width)

        return ws

    @property
    def initial_condition_loopsize(self):
        """
        Return the necessary loop bound for the global index of inner kernel loops
        based on this :class:`loopy_options`
        """

        if self.unique_pointers:
            return self.vector_width if self.vector_width else 1
        if not self.pre_split and self.width:
            return '{}*{}'.format(arc.work_size.name, self.width)
        return arc.work_size.name

    @property
    def vector_width(self):
        """
        Returns the vector width for this :class:`loopy_options` or None if
        unvectorized
        """
        if not (self.width or self.depth):
            return None
        return self.width if self.width else self.depth

    @property
    def has_scatter(self):
        """
        Utility to determine whether the target supports scatter writes

        Currently, only Intel's OpenCL implementation does not (CPU-only 16.1.1)
        and if attempted, it breaks the auto-vectorization

        Parameters
        ----------
        None

        Returns
        -------
        has_scatter: bool
            Whether the target supports scatter operations or not
        """
        return not (self.lang == 'opencl' and 'intel' in self.platform_name.lower())

    @property
    def platform_is_pyopencl(self):
        """
        Return true, IFF :attr:`platform` is an instance of a
        :class:`pyopencl.Platform`
        """

        return self.platform and cl is not None and isinstance(
            self.platform, cl.Platform)

    @property
    def platform_name(self):
        """
        Returns the suppled OpenCL platform name, or None if not available
        """

        if self.platform_is_pyopencl:
            return self.platform.name

        return self.platform


def get_device_list():
    """
    Returns the available pyopencl devices

    Parameters
    ----------
    None

    Returns
    -------
    devices : list of :class:`pyopencl.Device`
        The devices recognized by pyopencl
    """
    device_list = []
    for p in cl.get_platforms():
        device_list.append(p.get_devices())
    # don't need multiple gpu's etc.
    return [x[0] for x in device_list if x]


def get_context(device='0'):
    """
    Simple method to generate a pyopencl context

    Parameters
    ----------
    device : str or :class:`pyopencl.Device`
        The pyopencl string (or device class) denoting the device to use,
        defaults to '0'

    Returns
    -------
    ctx : :class:`pyopencl.Context`
        The running context
    """

    # os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
    if isinstance(device, str):
        os.environ['PYOPENCL_CTX'] = device
        ctx = cl.create_some_context(interactive=False)
    else:
        ctx = cl.Context(devices=[device])

    return ctx


def get_header(knl, codegen_result=None):
    """
    Returns header definition code for a :class:`loopy.LoopKernel`

    Parameters
    ----------
    knl : :class:`loopy.LoopKernel`
        The kernel to generate a header definition for
    codegen_result : :class:`loopy.CodeGenerationResult`
        If supplied, the pre-generated code-gen result for this kernel (speeds up
        header generation)

    Returns
    -------
    Generated device header code

    Notes
    -----
    The kernel's Target and name should be set for proper functioning
    """

    return str(lp.generate_header(knl, codegen_result=codegen_result)[0])


def __set_editor(knl, script):
    # set the edit script as the 'editor'
    os.environ['EDITOR'] = script

    # turn on code editing
    edit_knl = lp.set_options(knl, edit_code=True)

    return edit_knl


def set_editor(knl):
    """
    Returns a copy of knl set up for various automated bug-fixes

    Parameters
    ----------
    knl : :class:`loopy.LoopKernel`
        The kernel to generate code for

    Returns
    -------
    edit_knl : :class:`loopy.LoopKernel`
        The kernel set up for editing
    """

    return __set_editor(knl, edit_script)


def set_adept_editor(knl,
                     base_kernels,
                     problem_size=8192,
                     independent_variable=None,
                     dependent_variable=None,
                     output=None,
                     do_not_set=[]):
    """
    Returns a copy of knl set up for various automated bug-fixes

    Parameters
    ----------
    knl : :class:`loopy.LoopKernel`
        The kernel to generate code for
    base_kernels : :class:`loopy.LoopKernel`
        The kernel :param:`knl` and all dependencies required for Jacobian
        evaluation. These kernels, should be generated for a problem_size of 1
        to facilitate indexing in the wrapped kernel
    problem_size : int
        The size of the testing problem
    independent_variable : :class:`array_creator.creator`
        The independent variables to compute the Jacobian with respect to
    dependent_variable : :class:`array_creator.creator`
        The dependent variables to find the Jacobian of
    output : :class:`array_creator.creator`
        The array to store the column-major
        Jacobian in, ordered by thermo-chemical condition
    do_not_set : list of :class:`array_creator.creator`
        Other variables that are computed in this kernel (and hence shouldn't)
        be set

    Returns
    -------
    edit_knl : :class:`loopy.LoopKernel`
        The kernel set up for editing
    """

    # load template
    with open(adept_edit_script + '.in', 'r') as file:
        src = file.read()

    def __get_size_and_stringify(variable):
        sizes = variable.shape
        indicies = ['ad_j', 'i']
        out_str = variable.name + '[{index}]'
        from pyjac.core.array_creator import creator
        if isinstance(variable, creator):
            if variable.order == 'C':
                # last index varies fastest, so stride of 'i' is 1
                sizes = reversed(sizes)
                indicies = reversed(indicies)
        elif isinstance(variable, lp.kernel.data.ArrayBase):
            # find index of test_size
            strides = [x.stride for x in variable.dim_tags]
            sizes = variable.shape
            # if first stride is not problem_size, this is 'C' ordered
            # hence reverse indicies
            if strides[0] != problem_size:
                sizes = reversed(sizes)
                indicies = reversed(indicies)

        if len(variable.shape) == 1:
            return 1, out_str.format(index='ad_j')
        if len(variable.shape) > 2:
            assert variable.name == 'jac'
            size = np.product([x for x in sizes if x != problem_size])
            # can't operate on this
            return None, None

        out_index = ''
        offset = 1
        out_size = None
        for size, index in zip(sizes, indicies):
            if out_index:
                out_index += ' + '
            if str(size) == arc.work_size.name:
                # per work-size = 1 in this context as we're operating per-thread
                pass
            elif size != problem_size:
                assert out_size is None, (
                    'Cannot determine variable size!')
                out_size = size
            out_index += '{} * {}'.format(index, offset)
            offset *= size

        return out_size, out_str.format(index=out_index)

    # find the dimension / string representation of the independent
    # and dependent variables

    indep_size, indep = __get_size_and_stringify(independent_variable)
    dep_size, dep = __get_size_and_stringify(dependent_variable)

    # initializers
    init_template = Template("""
        std::vector<adouble> ad_${name} (${size});
        """)
    set_template = Template("""
        for (int i = 0; i < ${size}; ++i)
        {
            ad_${name}[i].set_value(${indexed});
        }
        """)
    zero_template = Template("""
        for(int i = 0; i < ${size}; ++i)
        {
            ad_${name}[i].set_value(0.0);
        }
        """)

    # get set of written vars
    written_vars = knl.get_written_variables()
    for k in base_kernels:
        written_vars |= k.get_written_variables()

    initializers = []
    for arg in knl.args:
        if arg.name != dependent_variable.name \
                and not isinstance(arg, lp.ValueArg):
            size, indexed = __get_size_and_stringify(arg)
            if size is not None:
                # add initializer
                initializers.append(init_template.substitute(
                    name=arg.name,
                    size=size,
                ))
                if indexed is not None and arg.name not in written_vars:
                    initializers.append(set_template.substitute(
                        name=arg.name,
                        indexed=indexed,
                        size=size
                    ))
                else:
                    initializers.append(zero_template.substitute(
                        name=arg.name,
                        size=size
                    ))

    dep_set_template = Template("""
        for (int i = 0; i < ${size}; ++i)
        {
            ${indexed} = ad_${name}[i].value();
        }
        """)

    setters = []
    for var in [dependent_variable] + do_not_set:
        size, ind = __get_size_and_stringify(var)
        setters.append(dep_set_template.substitute(
            indexed=ind,
            name=var.name,
            size=size))
    setters = '\n'.join(setters)

    jac_size = dep_size * indep_size
    # find the output name
    jac_base_offset = '&' + output.name + \
        '[ad_j * {dep_size} * {indep_size}]'.format(
            dep_size=dep_size, indep_size=indep_size)

    # get header defn
    header = get_header(knl)
    header = header[:header.index(';')]

    # replace the "const" on the jacobian
    header = re.sub(r'double\s*const(?=[^,]+{name})'.format(name=output.name),
                    'double', header)

    # and function call

    kernel_calls = []
    for k in base_kernels:
        arg_list = [arg.name for arg in k.args]
        for i, arg in enumerate(arg_list):
            name = arg[:]
            if arg != output.name:
                name = 'ad_' + name
            if arg != 'j':
                name = '&' + name + '[0]'
            arg_list[i] = name

        kernel_calls.append('ad_{name}({args});'.format(
            name=k.name,
            args=', '.join(arg_list)))

    # fill in template
    with open(adept_edit_script, 'w') as file:
        file.write(utils.subs_at_indent(
            src,
            problem_size=problem_size,
            ad_indep_name='ad_' + independent_variable.name,
            # indep=indep,
            # indep_name=independent_variable.name,
            indep_size=indep_size,
            ad_dep_name='ad_' + dependent_variable.name,
            # dep=dep,
            # dep_name=dependent_variable.name,
            dep_size=dep_size,
            jac_base_offset=jac_base_offset,
            # jac_size=jac_size,
            jac_name=output.name,
            function_defn=header,
            kernel_calls='\n'.join(kernel_calls),
            initializers='\n'.join(initializers),
            base_kernels='\n'.join([get_code(x) for x in base_kernels]),
            setters=setters
        ))

    # and make it executable
    st = os.stat(adept_edit_script)
    os.chmod(adept_edit_script, st.st_mode | stat.S_IEXEC)

    return __set_editor(knl, adept_edit_script)


def get_code(knl, opts=None):
    """
    Returns the device code for a :class:`loopy.LoopKernel` or
    fixes alreay generated code

    Parameters
    ----------
    knl : :class:`loopy.LoopKernel` or str
        The kernel to generate code for.  If knl is a string, it is assumed
        to be pregenerated code, and only the editor script must be called
    opts: :class:`loopy_options`
        The options used in created the kernel -- used to detect platform specific
        fixes.  Ignored if not supplied

    Returns
    -------
    code: str
        Generated device code

    Notes
    -----
    The kernel's Target and name should be set for proper functioning
    """

    if isinstance(knl, str):
        code = knl
    else:
        code, _ = lp.generate_code(knl)

    extra_subs = {}
    if opts is None:
        # ignore
        pass
    elif opts.lang == 'opencl' and (
        'intel' in opts.platform_name.lower()
            and ((opts.order == 'C' and opts.width) or (
                 opts.order == 'F' and opts.depth) or (
                 opts.order == 'F' and opts.width))):
        # If True, this is a finite-difference Jacobian on an Intel OpenCL platform
        # Hence we have to tell the codefixer about the intel bug
        # https://software.intel.com/en-us/forums/opencl/topic/748841
        extra_subs[r'__kernel void __attribute__ \(\(reqd_work_group_size\(\d+, 1, 1'
                   r'\)\)\) species_rates_kernel'] = r'void species_rates_kernel'

    return codefix('stdin', text_in=code, extra_subs=extra_subs)


def not_is_close(arr1, arr2, **kwargs):
    """
    A utility method that returns the result of:
        numpy.where(numpy.logical_not(numpy.isclose(arr1, arr2, **kwargs)))
    Since I use if often in testing

    Parameters
    ----------
    arr1: :class:`np.ndarray`
        Array to compare
    arr2: :class:`np.ndarray`
        Reference answer
    **kwargs: dict
        Keyword args for :func:`numpy.isclose`

    Returns
    -------
    inds: tuple of :class:`numpy.ndarray`
        result of:
        `numpy.where(numpy.logical_not(numpy.isclose(arr1, arr2, **kwargs)))`
    """

    return np.where(np.logical_not(np.isclose(arr1, arr2, **kwargs)))


class kernel_call(object):

    """
    A wrapper for the various parameters (e.g. args, masks, etc.)
    for calling / executing a loopy kernel
    """

    def __init__(self, name, ref_answer, compare_axis=1, compare_mask=None,
                 out_mask=None, input_mask=[], strict_name_match=False,
                 chain=None, check=True, post_process=None,
                 allow_skip=False, other_compare=None, atol=1e-8,
                 rtol=1e-5, equal_nan=False, ref_ans_compare_mask=None,
                 tiling=True, **input_args):
        """
        The initializer for the :class:`kernel_call` object

        Parameters
        ----------
        name : str
            The kernel name, used for matching
        ref_answer : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            The reference answer to compare to
        compare_axis : int, optional
            An axis to apply the compare_mask along, unused if compare_mask
            is None
        compare_mask : :class:`numpy.ndarray` or list of :class:`numpy.ndarray`
            An optional list of indexes to compare, useful when the kernel only
            computes partial results. Should match length of ref_answer
        ref_ans_compare_mask : :class:`numpy.ndarray` or
                list of :class:`numpy.ndarray`
            Same as the compare_mask, but for the reference answer.
            Necessary for some kernel tests, as the reference answer is not the same
            size as the output, which causes issues for split arrays.
            If not supplied, the regular :param:`compare_mask` will be used
        tiling: bool, [True]
            If True (default), the elements in the :param:`compare_mask` should be
            combined, e.g., if two arrays [[1, 2] and [3, 4]] are supplied to
            :param:`compare_mask` with tiling turned on, four resulting indicies will
            be compared -- [1, 3], [1, 4], [2, 3], and [2, 4].  If tiling is turned
            of, the compare mask will be treated as a list of indicies, e.g., (for
            the previous example) -- [1, 3] and [2, 4].
        out_mask : int, optional
            The index(ices) of the returned array to aggregate.
            Should match length of ref_answer
        input_mask : list of str or function, optional
            An optional list of input arguements to filter out
            If a function is passed, the expected signature is along the
            lines of:
                def fcn(self, arg_name):
                    ...
            and returns True iff this arg_name should be used
        strict_name_match : bool, optional
            If true, only kernels exactly matching this name will be excecuted
            Defaut is False
        chain : function, optional
            If not None, a function of signature similar to:
                def fcn(self, out_values):
                    ....
            is expected.
            This function should take the output values from a previous
            kernel call, and place in the input args for this kernel call as
            necessary
        post_process : function, optional
            If not None, a function of signature similar to:
                def fcn(self, out_values):
                    ....
            is expected.  This function should take the output values from this
            kernel call, and process them as expected to compare to results.
            Currently used only in comparison of reaction rates to
            Cantera (to deal w/ falloff etc.)
        check : bool
            If False, do not check result (useful when chaining to check only
            the last result)
            Default is True
        allow_skip : bool
            If True, allow this kernel call to be check results
            without actually executing a kernel (checks the last kernel
            that was executed). This is useful for selectively turning off
            kernels (e.g. if there are no reverse reactions)
        other_compare : Callable, optional
            If supplied, a function that compares output values not checked
            in by this kernel call.  This is useful in the case of NaN's
            resulting from derivatives of (e.g.,) log(0), to ensure our
            arrays are spitting out very large (but finite) numbers
        rtol : float [Default 1e-5]
            The relative tolerance for comparison to reference answers.
            For Jacobian correctness testing this may have to be loosened
        atol : float [Default 1e-8]
            The absolute tolerance for comparison to reference answers.
        equal_nan : bool [False]
            If supplied, whether to consider NaN's equal for reference testing
        input_args : dict of `numpy.array`s
            The arguements to supply to the kernel

        Returns
        -------
        out_ref : list of :class:`numpy.ndarray`
            The value(s) of the evaluated :class:`loopy.LoopKernel`
        """

        self.name = name
        self.ref_answer = ref_answer
        if isinstance(ref_answer, list):
            num_check = len(ref_answer)
        else:
            num_check = 1
            self.ref_answer = [ref_answer]
        self.compare_axis = compare_axis
        if compare_mask is not None:
            self.compare_mask = compare_mask
        else:
            self.compare_mask = [None for i in range(num_check)]
        if ref_ans_compare_mask is not None:
            self.ref_ans_compare_mask = ref_ans_compare_mask
        else:
            self.ref_ans_compare_mask = [None for i in range(num_check)]

        self.out_mask = out_mask
        self.input_mask = input_mask
        self.input_args = input_args
        self.strict_name_match = strict_name_match
        self.kernel_args = None
        self.chain = chain
        self.post_process = post_process
        self.check = check
        self.current_order = None
        self.allow_skip = allow_skip
        self.other_compare = other_compare
        self.tiling = tiling
        # pull any rtol / atol from env / test config as specified by user
        from pyjac.utils import get_env_val
        rtol = float(get_env_val('rtol', rtol))
        atol = float(get_env_val('atol', atol))

        self.rtol = rtol
        self.atol = atol
        self.equal_nan = equal_nan
        self.do_not_copy = set()

    def is_my_kernel(self, knl):
        """
        Tests whether this kernel should be run with this call

        Parameters
        ----------
        knl : :class:`loopy.LoopKernel`
            The kernel to call
        """

        if self.strict_name_match:
            return self.name == knl.name
        return True

    def set_state(self, array_splitter, order='F',
                  namestore=None, jac_format=JacobianFormat.full):
        """
        Updates the kernel arguements, and  and compare axis to the order given
        If the 'arg' is a function, it will be called to get the correct answer

        Parameters
        ----------
        array_splitter: :class:`pyjac.core.array_creator.array_splitter`
            The array splitter of the owning
            :class:`kernek_utils.kernel_gen.kernel.kernel_generator`, used to
            operate on numpy arrays if necessary
        order : {'C', 'F'}
            The memory layout of the arrays, C (row major) or
            Fortran (column major)
        namestore : :class:`NameStore`
            Must be supplied if :param:`jac_format` is of type
            :class:`JacobianFormat.sparse`, in order to pull row / column indicies
            for conversion to / from sparse matricies
        jac_format: :class:`JacobianFormat` [JacobianFormat.full]
            If sparse, we are testing a sparse matrix (and :param:`namestore` must
            be supplied)
        """
        self.current_order = order

        # filter out bad input
        args_copy = self.input_args.copy()
        if self.input_mask is not None:
            if six.callable(self.input_mask):
                args_copy = {x: args_copy[x] for x in args_copy
                             if self.input_mask(self, x)}
            else:
                args_copy = {x: args_copy[x] for x in args_copy
                             if x not in self.input_mask}

        for key in args_copy:
            if six.callable(args_copy[key]):
                # it's a function
                args_copy[key] = args_copy[key](order)

        self.kernel_args = args_copy
        self.transformed_ref_ans = [np.array(ans, order=order, copy=True)
                                    for ans in self.ref_answer]

        self.jac_format = jac_format
        if jac_format == JacobianFormat.sparse:
            from pyjac.tests.test_utils import sparsify
            # need to convert the jacobian arg to a sparse representation
            # the easiest way to deal with this is to convert the kernel argument
            # to the sparse dimensions

            # Then afterwards we can use the row / col inds as an intermediate
            # index in the comparison step
            self.kernel_args['jac'] = np.array(self.kernel_args['jac'][
                :,
                namestore.flat_jac_row_inds.initializer,
                namestore.flat_jac_col_inds.initializer],
                order=order,
                copy=True)
            # save for comparable
            self.row_inds = namestore.jac_row_inds.initializer
            self.col_inds = namestore.jac_col_inds.initializer

            # sparsify transformed answer
            self.transformed_ref_ans = [
                sparsify(array, self.col_inds, self.row_inds, self.current_order)
                if array.ndim >= 3 else array for array in self.transformed_ref_ans]

        # and finally feed through the array splitter
        self.current_split = array_splitter
        self.kernel_args = array_splitter.split_numpy_arrays(self.kernel_args)
        self.transformed_ref_ans = array_splitter.split_numpy_arrays(
            self.transformed_ref_ans)

    def __call__(self, knl, queue):
        """
        Calls the kernel, filtering input / output args as required

        Parameters
        ----------
        knl : :class:`loopy.LoopKernel`
            The kernel to call
        queue : :class:`pyopencl.Queue`
            The command queue

        Returns
        -------
        out : list of :class:`numpy.ndarray`
            The (potentially filtered) output variables
        """

        if isinstance(knl.target, lp.PyOpenCLTarget):
            evt, out = knl(queue, out_host=True, **self.kernel_args)
        elif isinstance(knl.target, lp.CTarget):
            evt, out = knl(**{
                k: v.copy(order=self.current_order) if (
                    isinstance(v, np.ndarray) and k not in self.do_not_copy)
                else v for k, v in self.kernel_args.items()})
        else:
            raise NotImplementedError

        if self.out_mask is not None:
            return [out[ind] for ind in self.out_mask]
        else:
            return [out[0]]

    def _get_comparable(self, variable, index, is_answer=False):
        """
        Selects the data to compare from the supplied variable depending on
        the compare mask / axes supplied
        """

        mask = self.ref_ans_compare_mask[index] if is_answer \
            else self.compare_mask[index]

        if mask is None and is_answer:
            # use the regular compare mask, as the reference answer specific one
            # was not supplied
            mask = self.compare_mask[index]

        # if no mask
        if mask is None:
            return variable

        if six.callable(mask):
            # see if it's a supplied callable
            return mask(self, variable, index, is_answer=is_answer)

        from pyjac.tests.test_utils import select_elements
        return select_elements(variable, mask, self.compare_axis, tiling=self.tiling)

    def compare(self, output_variables):
        """
        Compare the output variables to the given reference answer

        Parameters
        ----------
        output_variables : :class:`numpy.ndarray` or :class:`numpy.ndarray`
            The output variables to test

        Returns
        -------
        match : bool
            True IFF the masked output variables match the supplied reference answer
            for this :class:`kernel_call`
        """

        def _check_mask(mask):
            # check that the mask is one of:
            # 1. a list of length equal to the size of the number of outputs
            # 2. a list of indicies (indicated by the compare axis set to -1)
            # 3. a callable function / object that can figure out extracting the
            #    comparable entries on it's own
            assert (isinstance(mask, list) and
                    len(mask) == len(output_variables)) or \
                not self.tiling or six.callable(mask), (
                    'Compare mask does not match output variables!')

        _check_mask(self.compare_mask)
        _check_mask(self.ref_ans_compare_mask)

        allclear = True
        for i in range(len(output_variables)):
            outv = output_variables[i].copy()
            ref_answer = self.transformed_ref_ans[i].copy()
            if self.compare_mask[i] is not None:
                outv = self._get_comparable(outv, i)
                if outv.shape != ref_answer.shape:
                    # apply the same transformation to the answer
                    ref_answer = self._get_comparable(ref_answer, i, is_answer=True)
            else:
                outv = outv.squeeze()
                ref_answer = ref_answer.squeeze()
            allclear = allclear and np.allclose(outv, ref_answer,
                                                rtol=self.rtol,
                                                atol=self.atol,
                                                equal_nan=self.equal_nan)

            if self.other_compare is not None:
                allclear = allclear and self.other_compare(
                    self, output_variables[i].copy(),
                    self.transformed_ref_ans[i].copy(), self.compare_mask[i])

        return allclear


def populate(knl, kernel_calls, device='0',
             editor=None):
    """
    This method runs the supplied :class:`loopy.LoopKernel` (or list thereof),
    and is often used by :function:`auto_run`

    Parameters
    ----------
    knl : :class:`loopy.LoopKernel` or list of :class:`loopy.LoopKernel`
        The kernel to test, if a list of kernels they will be successively
        applied and the end result compared
    kernel_calls : :class:`kernel_call` or list thereof
        The masks / ref_answers, etc. to use in testing
    device : str
        The pyopencl string denoting the device to use, defaults to '0'
    editor : callable
        If not none, a callable function or object that takes a
        :class:`loopy.LoopKernel` as the sole arguement, and returns the kernel
        with editing turned on (for used with auto-differentiation)

        If not specified, the default (opencl) editor will be invoked

    Returns
    -------
    out_ref : list of :class:`numpy.ndarray`
        The value(s) of the evaluated :class:`loopy.LoopKernel`
    """

    assert len(knl), 'No kernels supplied!'

    # create context
    ctx = None
    if any(isinstance(k.target, lp.PyOpenCLTarget) for k in knl):
        ctx = get_context(device)

    if editor is None:
        editor = set_editor

    def __inner(queue=None):
        output = []
        kc_ind = 0
        oob = False
        while not oob:
            # handle weirdness between list / non-list input
            try:
                kc = kernel_calls[kc_ind]
                kc_ind += 1
            except IndexError:
                oob = True
                break  # reached end of list
            except TypeError:
                # not a list
                oob = True  # break on next run
                kc = kernel_calls

            # create the outputs
            if kc.out_mask is not None:
                out_ref = [None for i in kc.out_mask]
            else:
                out_ref = [None]

            found = False
            # run kernels
            for k in knl:
                # test that we want to run this one
                if kc.is_my_kernel(k):
                    found = True
                    # set the editor to avoid intel bugs
                    test_knl = editor(k)
                    if isinstance(test_knl.target, lp.PyOpenCLTarget):
                        # recreate with device
                        test_knl = test_knl.copy(
                            target=lp.PyOpenCLTarget(device=device))

                    # check for chaining
                    if kc.chain:
                        kc.chain(kc, output)

                    # run!
                    out = kc(test_knl, queue)

                    if kc.post_process:
                        kc.post_process(kc, out)

                    # output mapping
                    if all(x is None for x in out_ref):
                        # if the outputs are none, we init to zeros
                        # and avoid copying zeros over later data!
                        out_ref = [np.zeros_like(x) for x in out]

                    for ind in range(len(out)):
                        # get indicies that are non-zero (already in there)
                        # or non infinity/nan

                        # try w/o finite check (I'm paranoid, don't want to mask)
                        # any bad data
                        copy_inds = np.where(np.logical_not(out[ind] == 0))
                        # copy_inds = np.where(np.logical_not(
                        #    np.logical_or(np.isinf(out[ind]),
                        #                  out[ind] == 0, np.isnan(out[ind]))),
                        # )
                        out_ref[ind][copy_inds] = out[ind][copy_inds]

            output.append(out_ref)
            assert found or kc.allow_skip, (
                'No kernels could be found to match kernel call {}'.format(
                    kc.name))
        return output

    if ctx is not None:
        with cl.CommandQueue(ctx) as queue:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=cl.CompilerWarning)
                output = __inner(queue)
            queue.flush()
        # release context
        clear_first_arg_caches()
        del ctx
    else:
        output = __inner()

    return output


def auto_run(knl, kernel_calls, device='0'):
    """
    This method tests the supplied :class:`loopy.LoopKernel` (or list thereof)
    against a reference answer

    Parameters
    ----------
    knl : :class:`loopy.LoopKernel` or list of :class:`loopy.LoopKernel`
        The kernel to test, if a list of kernels they will be successively
        applied and the end result compared
    kernel_calls : :class:`kernel_call`
        The masks / ref_answers, etc. to use in testing
    device : str
        The pyopencl string denoting the device to use, defaults to '0'
    input_args : dict of `numpy.array`s
        The arguements to supply to the kernel

    Returns
    -------
    result : bool
        True if all tests pass
    """

    # run kernel

    # check lists
    if not isinstance(knl, list):
        knl = [knl]

    out = populate(knl, kernel_calls, device=device)
    try:
        result = True
        for i, kc in enumerate(kernel_calls):
            if kc.check:
                ind = i
                if kc.allow_skip and all(x is None for x in out[i]):
                    # find the last one for which we have data
                    ind = next(ind for ind in reversed(range(i))
                               if not any(x is None for x in out[ind]))
                result = result and kc.compare(out[ind])
        return result
    except TypeError as e:
        if str(e) == "'kernel_call' object is not iterable":
            # if not iterable
            return kernel_calls.compare(out[0])
        raise e


def get_target(lang, device=None, compiler=None):
    """

    Parameters
    ----------
    lang : str
        One of the supported languages, {'c', 'cuda', 'opencl'}
    device : :class:`pyopencl.Device`
        If supplied, and lang is 'opencl', passed to the
        :class:`loopy.PyOpenCLTarget`
    compiler: str
        If supplied, the C-compiler to use

    Returns
    -------
    The correct loopy target type
    """

    utils.check_lang(lang)

    # set target
    if lang == 'opencl':
        if cl is not None:
            return lp.PyOpenCLTarget(device=device)
        return lp.OpenCLTarget()
    elif lang == 'c':
        return lp.ExecutableCTarget(compiler=compiler)
    elif lang == 'cuda':
        return lp.CudaTarget()
    elif lang == 'ispc':
        return lp.ISPCTarget()


class AdeptCompiler(CPlusPlusCompiler):

    def __init__(self, *args, **kwargs):
        from ..siteconf import ADEPT_INC_DIR, ADEPT_LIB_DIR, ADEPT_LIBNAME
        from ..siteconf import CXXFLAGS
        defaults = kwargs.copy()
        defaults['libraries'] = ADEPT_LIBNAME
        if 'cflags' not in defaults:
            defaults['cflags'] = []
        if CXXFLAGS:
            defaults['cflags'] = [x for x in CXXFLAGS
                                  if x not in defaults['cflags']
                                  and x.strip()]
        if ADEPT_LIB_DIR:
            defaults['library_dirs'] = ADEPT_LIB_DIR
        if ADEPT_INC_DIR:
            defaults['cflags'] = ['-I{}'.format(x) for x in ADEPT_INC_DIR]

        # update to use any user specified info
        defaults.update(kwargs)

        # get toolchain
        from pyjac.libgen import get_toolchain
        toolchain = get_toolchain('c', executable=False, **defaults)

        # and create
        super(AdeptCompiler, self).__init__(toolchain=toolchain)

    def build(self, *args, **kwargs):
        """override from CPlusPlusCompiler to load Adept into ctypes and avoid
           missing symbol errors"""
        from ctypes.util import find_library
        from ctypes import CDLL, RTLD_GLOBAL
        CDLL(find_library('adept'), mode=RTLD_GLOBAL)

        return super(AdeptCompiler, self).build(*args, **kwargs)
