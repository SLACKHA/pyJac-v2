"""
memory_manager.py - generators for defining / allocating / transfering memory
for kernel creation
"""

from __future__ import division

from string import Template
import six
import logging
import re

import numpy as np
import loopy as lp
from enum import Enum
from loopy.types import to_loopy_type
from pytools import ImmutableRecord
#  import resource
#  align_size = resource.getpagesize()

from pyjac.core.array_creator import problem_size as p_size
from pyjac.core.array_creator import work_size as w_size
from pyjac.core import array_creator as arc
from pyjac import utils
from pyjac.utils import func_logger
from pyjac.schemas import build_and_validate, parse_bytestr
from pyjac.core.exceptions import ValidationError, \
    InvalidInputSpecificationException

try:
    from pyopencl import device_type
    DTYPE_CPU = device_type.CPU
except ImportError:
    DTYPE_CPU = -1


class memory_type(Enum):
    m_constant = 0,
    m_local = 1,
    m_global = 2,
    m_alloc = 3
    #  m_pagesize = 4

    def __int__(self):
        return self.value


@func_logger
def load_memory_limits(input_file, schema='common_schema.yaml'):
    """
    Conviencence method for loading inputs from memory limits file

    Parameters
    ----------
    input_file:
        The input file to load
    """

    def __limitfy(limits):
        # note: this is only safe because we've already validated.
        # hence, NO ERRORS!
        if limits:
            return {k: parse_bytestr(object, v) if not k == 'platforms' else v
                    for k, v in six.iteritems(limits)}
    if input_file:
        try:
            memory_limits = build_and_validate('common_schema.yaml', input_file,
                                               allow_unknown=True)
            return [__limitfy(memory_limits['memory-limits'])]
        except ValidationError:
            # TODO: fix this -- need a much better way of specifying / passing in
            # limits
            memory_limits = build_and_validate('test_matrix_schema.yaml', input_file,
                                               allow_unknown=True)
            return [__limitfy(x) for x in memory_limits['memory-limits']]
        except KeyError:
            # no limits
            pass

    return {}


class MemoryGenerationResult(ImmutableRecord):
    """
    A placeholder class that holds intermediate memory results

    Attributes
    ----------
    args: list of :class:`loopy.ArrayArg`
        The list of global arguments for the generated kernel
    local: list of :class:`loopy.ArrayArg` with :attr:`address_space` LOCAL
        The list of local argument variables to be defined at the top-level
        wrapping kernel.
    readonly: set of str
        The names of arguements in the top-level kernel that are never written to
    constants: list of :class:`loopy.TemporaryVariable` with :attr:`address_space` GLOBAL and :attr:`readonly` True  # noqa
        The constant data to define in the top-level kernel
    valueargs: list of :class:`loopy.ValueArg`
        The value arguments passed into the top-level wrapping kernel
    host_constants: list of :class:`loopy.GlobalArg`
        The __constant data that was necessary to move to __global data for
        space reasons
    kernel_data: list of :class:`loopy.KernelArguement`
        The list of kernel arguments required to call the generated kernel.
        These typically contain compressed working buffer version(s) of :attr:`args`
        that may be unpacked in the kernel, in addition to :class:`valueargs`
        that should be passed in as is.
    """

    def __init__(self, args=[], local=[], readonly=[], constants=[],
                 valueargs=[], host_constants=[], kernel_data=[]):
        ImmutableRecord.__init__(
            self, args=args, local=local, readonly=readonly, constants=constants,
            valueargs=valueargs, host_constants=host_constants,
            kernel_data=kernel_data)


class memory_limits(object):
    """
    Helps determine whether a kernel is using too much constant / shared memory,
    etc.

    Properties
    ----------
    arrays: dict
            A mapping of :class:`loopy.TemporaryVariable` or :class:`loopy.GlobalArg`
            to :class:`memory_type`, representing the types of all arrays to be
            included
    limits: dict
        A dictionary with keys 'shared', 'constant' and 'local' indicated the
        total amount of each memory type available on the device
    string_strides: list of compiled regular expressions
        A list of regular expression that may be used as array sizes (i.e.,
        for the 'problem_size' variable)
    dtype: np.dtype [np.int32]
            The index type of the kernel to be generated. Default is a 32-bit int
    limit_int_overflow: bool
        If true, limit the maximum number of conditions that can be run to avoid
        int32 overflow
    """

    def __init__(self, lang, order, arrays, limits, string_strides=[],
                 dtype=np.int32, limit_int_overflow=False):
        """
        Initializes a :class:`memory_limits`
        """
        self.lang = lang
        self.order = order
        self.arrays = arrays
        self.limits = limits
        self.string_strides = [re.compile(re.escape(s)) if isinstance(s, str)
                               else s for s in string_strides]
        self.dtype = dtype
        self.limit_int_overflow = limit_int_overflow

    def integer_limited_problem_size(self, arry, dtype=np.int32):
        """
        A convenience method to determine the maximum problem size that will not
        result in an integer overflow in index (mainly, Intel OpenCL).

        This is calculated by determining the maximum index of the array, and then
        dividing the maximum value of :param:`dtype` by this stride

        Parameters
        ----------
        arry: lp.ArrayBase
            The array to test
        dtype: np.dtype [np.int32]
            The integer type to use

        Returns
        -------
        num_ics: int
            The number of initial conditions that can be tested for this array
            without integer overflow in addressing
        """

        # convert problem_size -> 1 in order to determine max per-run size
        # from array shape
        def floatify(val):
            if not (isinstance(val, float) or isinstance(val, int)):
                ss = next((s for s in self.string_strides if s.search(
                    str(val))), None)
                assert ss is not None, 'malformed strides'
                from pymbolic import parse
                # we're interested in the number of conditions we can test
                # hence, we substitute '1' for the problem size, and divide the
                # array stisize by the # of
                val = parse(str(val).replace(p_size.name, '1'))
                val = parse(str(val).replace(w_size.name, '1'))
                assert isinstance(val, (float, int)), (arry.name, val)
            return val
        return int(np.iinfo(dtype).max // np.prod([floatify(x) for x in arry.shape]))

    def arrays_with_type_changes(self, mtype=memory_type.m_constant,
                                 with_type_changes={}):
        """
        Returns the list of :attr:`arrays` that are of :param:`mtype` with the
        given :param:`with_type_changes`.  See :func:`can_fit`
        """

        # filter arrays by type
        arrays = self.arrays[mtype]
        arrays = [a for a in arrays if not any(
            a in v for k, v in six.iteritems(with_type_changes) if k != mtype)]
        return arrays

    def can_fit(self, mtype=memory_type.m_constant, with_type_changes={}):
        """
        Determines whether the supplied :param:`arrays` of type :param:`type`
        can fit on the device

        Parameters
        ----------
        type: :class:`memory_type`
            The type of memory to use
        with_type_changes: dict
            Updates to apply to :prop:`arrays`

        Returns
        -------
        can_fit_ic: int
            The maximum number of times these arrays can be fit in memory.
            - For global memory, this determines the number of initial conditions
            that can be evaluated.
            - For shared and constant memory, this determines whether this data
            can be fit
        can_fit_wc: int
            The maximum number of times the arrays can be fit in working buffers.
            - For global memory, this determines the maximum work-size
            - For shared and constant memory, this determines whether this data
            can be fit (and should be identical to can_fit_ic)
        """

        # filter arrays by type
        arrays = self.arrays_with_type_changes(mtype, with_type_changes)
        if not arrays:
            # no arrays to process
            return (np.iinfo(np.int).max, np.iinfo(np.int).max)

        per_alloc_ic_limit = np.iinfo(np.int).max
        per_alloc_ws_limit = np.iinfo(np.int).max

        def __calculate_integer_limit(limit):
            old = limit
            limit = np.minimum(limit, self.integer_limited_problem_size(
                    array, self.dtype))
            if old != limit:
                stype = str(mtype)
                stype = stype[stype.index('.') + 3:]
                logger.info(
                    'Allocation of {} memory array {} '
                    'may result in integer overflow in indexing, and '
                    'cause failure on execution, limiting per-run size to {}.'
                    .format(stype, array.name, int(limit))
                    )
            return limit

        def __calculate_alloc_limit(limit):
            return np.minimum(limit, np.floor(
                self.limits[memory_type.m_alloc] / size))

        per_ic = 0
        per_ws = 0
        static = 0
        logger = logging.getLogger(__name__)
        for array in arrays:
            size = 1
            is_ic_dep = False
            is_ws_dep = False
            for s in array.shape:
                ss = next((x.search(str(s)) for x in self.string_strides
                           if x.search(str(s))), None)
                if ss:
                    if w_size.name in str(s):
                        # mark as dependent on the work size
                        is_ws_dep = True
                    else:
                        # mark as dependent on # of initial conditions
                        is_ic_dep = True
                    # get the floor div (if any)
                    floor_div = re.search(r'// (\d+)', str(s))
                    if floor_div:
                        floor_div = int(floor_div.group(1))
                        size /= floor_div
                    continue
                # update size
                size *= s
            # and convert size
            size *= array.dtype.itemsize

            # update counter
            if is_ic_dep:
                per_ic += size
            elif is_ws_dep:
                per_ws += size
            else:
                static += size

            assert not (is_ic_dep and is_ws_dep)

            # check for integer overflow -- this does not depend on any particular
            # memory limit
            if (is_ic_dep or is_ws_dep) and self.limit_int_overflow:
                if is_ic_dep:
                    per_alloc_ic_limit = __calculate_integer_limit(
                        per_alloc_ic_limit)
                elif is_ws_dep:
                    per_alloc_ws_limit = __calculate_integer_limit(
                        per_alloc_ws_limit)

            # also need to check the maximum allocation size for opencl
            if memory_type.m_alloc in self.limits:
                if is_ic_dep:
                    per_alloc_ic_limit = __calculate_alloc_limit(per_alloc_ic_limit)
                elif is_ws_dep:
                    per_alloc_ws_limit = __calculate_alloc_limit(per_alloc_ws_limit)
                else:
                    if static >= self.limits[memory_type.m_alloc]:
                        logger.warn(
                            'Allocation of constant memory array {}'
                            ' exceeds maximum allocation size, this will likely'
                            ' cause OpenCL to fail on execution.'
                            .format(array.name)
                            )

        # if no per_ic, just divide by 1
        if per_ic == 0:
            per_ic = 1
        if per_ws == 0:
            per_ws = 1

        # finally, check the number of times we can fit these array in the memory
        # limits of this type
        if mtype in self.limits:
            per_alloc_ic_limit = np.minimum(np.floor((
                self.limits[mtype] - static) / per_ic), per_alloc_ic_limit)
            per_alloc_ws_limit = np.minimum(np.floor((
                self.limits[mtype] - static) / per_ws), per_alloc_ws_limit)

        return int(np.maximum(per_alloc_ic_limit, -1)), int(
            np.maximum(per_alloc_ws_limit, -1))

    @staticmethod
    def get_limits(loopy_opts, arrays, input_file='',
                   string_strides=[p_size.name, w_size.name],
                   dtype=np.int32, limit_int_overflow=False):
        """
        Utility method to load shared / constant memory limits from a file or
        :mod:`pyopencl` as needed

        Parameters
        ----------
        loopy_opts: :class:`loopy_options`
            If loopy_opts.lang == 'opencl', pyopencl will be used to fill in any
            missing limits
        arrays: dict
            A mapping of :class:`memory_type` to :class:`loopy.TemporaryVariable` or
            :class:`loopy.GlobalArg`, representing the types of all arrays to be
            included
        input_file: str
            The path to a yaml file with specified limits (keys should include
            'local' and 'constant', and 'global')
        string_strides: str
            The strides of host & device buffers dependent on user input
            Need special handling in size determination
        dtype: np.dtype [np.int32]
            The index type of the kernel to be generated. Default is a 32-bit int
        limit_int_overflow: bool [False]
            If true, turn on limiting array sizes to avoid integer overflow.
            Currently only needed for Intel OpenCL

        Returns
        -------
        limits: :class:`memory_limits`
            An initialized :class:`memory_limits` that can determine the total
            'global', 'constant' and 'local' memory available on the device
        """

        limits = {}  # {memory_type.m_pagesize: align_size}
        if loopy_opts.lang == 'opencl':
            try:
                limits.update({
                    memory_type.m_global: loopy_opts.device.global_mem_size,
                    memory_type.m_constant:
                        loopy_opts.device.max_constant_buffer_size,
                    memory_type.m_local: loopy_opts.device.local_mem_size,
                    memory_type.m_alloc: loopy_opts.device.max_mem_alloc_size})
            except AttributeError:
                pass
        user = load_memory_limits(input_file)
        # find limit(s) that applies to us
        user = [u for u in user if 'platforms' not in u or
                loopy_opts.platform_name.lower() in u['platforms']]
        if len(user) > 1:
            # check that we don't have multiple limits with this platforms specified
            if len([u for u in user if 'platforms' in u]) > 1 or len(
                    [u for u in user if 'platforms' not in u]) > 1:
                logger = logging.getLogger(__name__)
                logger.error('Multiple memory-limits supplied by name in file ({}) '
                             'for platform {}.  Valid configurations are either one '
                             'default memory-limits specification for all platforms '
                             'with specific overrides for a platform, or a '
                             'platform-specific memory-limits only.'.format(
                                input_file, loopy_opts.platform_name))
                raise InvalidInputSpecificationException('memory-limits')
            assert len(user) <= 2

        if user:
            logger = logging.getLogger(__name__)
            for lim in sorted(user, key=lambda x: 'platforms' in x):
                # load from file
                mtype = utils.EnumType(memory_type)
                user_limits = {}
                for key, value in six.iteritems(lim):
                    if key == 'platforms':
                        continue
                    # check in memory type
                    key = 'm_' + key
                    # update with enum
                    logger.info('Overriding memory-limit for type {} from value '
                                '({}) to value ({}) from {} limits.'.format(
                                    key,
                                    limits[mtype(key)] if mtype(key) in limits
                                    else None, value,
                                    'per-platform' if 'platforms' in lim else
                                    'default'))
                    user_limits[mtype(key)] = value
                # and overwrite default limits w/ user
                limits.update(user_limits)

        return memory_limits(loopy_opts.lang, loopy_opts.order, arrays,
                             limits, string_strides, dtype, limit_int_overflow)


asserts = {'c': Template('cassert(${call}, "${message}");'),
           'opencl': Template('check_err(${call});')}
"""dict: the error checking macros for each language"""


def guarded_call(lang, call, message='Error'):
    """
    Returns a call guarded by error checking for the given :param:`lang`
    """
    return asserts[lang].safe_substitute(call=call, message=message)


host_langs = {'opencl': 'c',
              'c': 'c'}
"""dict: the host language that launches the kernel"""

host_prefix = 'h_'
""" the prefix for host arrays """
device_prefix = 'd_'
""" the prefix for device arrays """


class memory_strategy(object):
    def __init__(self, lang, alloc={}, sync={}, copy_in_1d={},
                 copy_in_2d={}, copy_out_1d={}, copy_out_2d={}, free={}, memset={},
                 alloc_flags={}, host_const_in={}):
        self.alloc_template = alloc
        self.sync_template = sync
        self.copy_in_1d = copy_in_1d
        self.copy_in_2d = copy_in_2d
        self.copy_out_1d = copy_out_1d
        self.copy_out_2d = copy_out_2d
        self.free_template = free
        self.memset_template = memset
        self.host_lang = host_langs[lang]
        self.device_lang = lang
        self.alloc_flags = alloc_flags
        self.host_const_in = host_const_in

    def lang(self, device):
        return self.host_lang if not device else self.device_lang

    def alloc(self, device, name, buff_size='', per_run_size='', readonly=False,
              host_constant=False, dtype='double', **kwargs):
        """
        Return an allocation for a buffer in the given :param:`lang`

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        name: str
            The desired name of the buffer
        buff_size: str ['']
            The actual size of the buffer.  May be overriden by :param:`per_run_size`
            if suppied
        per_run_size: str ['']
            The maximum allowable size for device allocation of this buffer
            If supplied, and :param:`device` is True, overrides :param:`buff_size`
        readonly: bool
            Changes memory flags / allocation type (e.g., for OpenCL)
        dtype: str ['double']
            The type of the array to allocate

        Returns
        -------
        alloc_str: str
            The resulting allocation instrucitons
        """

        flags = kwargs.pop('flags', '')
        if self.lang(device) in self.alloc_flags:
            flags = [self.alloc_flags[self.lang(device)][readonly]] \
                + flags.split(' | ')
            flags = ' | '.join([x for x in flags if x])

        # override
        if per_run_size and device and not host_constant:
            buff_size = per_run_size

        assert 'buff_size' not in kwargs

        return self.alloc_template[self.lang(device)].safe_substitute(
            name=name, buff_size=buff_size, memflag=flags,
            dtype=dtype, **kwargs)

    def copy(self, to_device, host_name, dev_name, buff_size, dim,
             host_constant=False, **kwargs):
        """
        Return a copy of a buffer to/from the "device"

        Parameters
        ----------
        to_device: bool
            If True, transfer to the "device"
        host_name: str
            The name of the host buffer
        dev_name: str
            The name of device buffer
        dim: int
            If dim == 1, can use a 1-d copy (e.g., a memcpy for C)
        host_constant: bool [False]
            If True, we are transferring a host constant, hence set any offset
            in the host buffer to zero
        Returns
        -------
        copy_str: str
            The resulting copy instructions
        """
        # get templates
        if host_constant:
            template = self.host_const_in
        elif dim <= 1:
            template = self.copy_in_1d if to_device else self.copy_out_1d
        else:
            template = self.copy_in_2d if to_device else self.copy_out_2d

        if not template:
            return ''

        return template.safe_substitute(
            host_name=host_name, dev_name=dev_name, buff_size=buff_size,
            host_offset='offset' if not host_constant else '0', **kwargs)

    def sync(self):
        """
        Synchronize host memory w/ device memory after kernel call

        Parameters
        ----------
        None

        Returns
        -------
        sync_instructions: str
            The synchronization instructions
        """
        if self.sync_template:
            return self.sync_template[self.device_lang]
        return ''

    def memset(self, device, name, buff_size='', per_run_size='', **kwargs):
        """
        Set the host / device memory

        Parameters
        ----------
        device: bool
            If true, set a device buffer, else a host buffer
        name: str
            The buffer name
        buff_size: str ['']
            The actual size of the buffer.  May be overriden by :param:`per_run_size`
            if suppied
        per_run_size: str ['']
            The maximum allowable size for device allocation of this buffer
            If supplied, and :param:`device` is True, overrides :param:`buff_size`

        Returns
        -------
        set_instructions: str
            The instructions to memset the buffer
        """

        # override
        if per_run_size and device:
            buff_size = per_run_size

        return self.memset_template[self.lang(device)].safe_substitute(
            name=name, buff_size=buff_size, **kwargs)

    def free(self, device, name):
        """
        Generates code to free a buffer

        Parameters
        ----------
        name: str
            The name of the buffer to free

        Returns
        -------
        free_str: str
            The resulting free instructions
        """
        return self.free_template[self.lang(device)].safe_substitute(name=name)


class mapped_memory(memory_strategy):
    def _get_2d_templates(self, lang, use_full=False, to_device=False,
                          ndim=1):
        """
        Returns a template to copy (part or all) of a 2D array from "host" to
        "device" arrays (note this is simulated on C)

        Parameters
        ----------
        use_full: bool [False]
            If true, get the copy template for the full array
        to_device: bool [False]
            If true, Write to device, else Read from device
        ndim: int [1]
            The number of dimensions of the array
        Notes
        -----

        Note that the OpenCL clEnqueueReadBufferRect/clEnqueueWriteBufferRect
        usage is somewhat tricky.  The documentation states that the
        region and row/slice pitch parameters should all be in bytes.

        What they actually mean is that the offsets comptuted by

            region[0] + region[1] * row_pitch + region[2] * slice_pitch
            origin[0] + origin[1] * row_pitch + origin[2] * slice_pitch

        for both the host and buffer should have units of bytes.  Hence we need
        to be careful about which values we multiply by the memory type size
        (i.e. sizeof(double), etc.)

        For reference, we will always select the first region/offset index to be
        in bytes, and the pitches to be in bytes as well (which seems to be
        required for all tested OpenCL implementations)

        Returns
        -------
        copy_str: Template
            The copy template to fill in for host/device I/O
        """

        order = self.order
        have_split = self.have_split

        if lang == 'opencl':
            rect_copy_template = Template(guarded_call(
                    lang,
                    'clEnqueue${ctype}BufferRect(queue, ${dev_name}, CL_TRUE, '
                    '(size_t[]) {0, 0, 0}, '  # buffer origin
                    '(size_t[]) ${host_origin}, '  # host origin
                    '(size_t[]) ${region}, '  # region
                    '${buffer_row_pitch}, '  # buffer_row_pitch
                    '${buffer_slice_pitch}, '  # buffer_slice_pitch
                    '${host_row_pitch}, '  # host_row_pitch
                    '${host_slice_pitch}, '  # host_slice_pitch
                    '${host_name}, 0, NULL, NULL)'
                ))
        elif lang == 'c':
            rect_copy_template = Template(
                'memcpy2D_${ctype}(${host_name}, ${dev_name}, '
                '(size_t[]) ${host_origin}, '  # host origin
                '(size_t[]) ${region}, '  # region
                '${buffer_row_pitch}, '  # buffer_row_pitch
                '${buffer_slice_pitch}, '  # buffer_slice_pitch
                '${host_row_pitch}, '  # host_row_pitch
                '${host_slice_pitch}'  # host_slice_pitch
                ');')

        def __f_split(ctype):
            # this us a F-split which requires a Rect Read/Write
            # with the given pitches / region / offsets
            # Note that this is actually a 3D (or 4D in the case of the)
            # Jacobian, but we can treat all subsequent dimensions 2 and
            # after as a flat 1-D array (as they are flattened in this order)

            return Template(rect_copy_template.safe_substitute(
                host_origin='{0, offset, 0}',
                region='{VECWIDTH * ${itemsize}, this_run, ${other_dim_size}}',
                buffer_row_pitch='VECWIDTH * ${itemsize}',
                buffer_slice_pitch='VECWIDTH * per_run * ${itemsize}',
                host_row_pitch='VECWIDTH * ${itemsize}',
                host_slice_pitch='VECWIDTH * problem_size * ${itemsize}',
                ctype=ctype
                ))

        def __f_unsplit(ctype):
            # this is a regular F-ordered array, which only requires a 2D copy
            return Template(rect_copy_template.safe_substitute(
                host_origin='{offset * ${itemsize}, 0, 0}',
                region='{this_run * ${itemsize}, ${non_ic_size}, 1}',
                buffer_row_pitch='per_run * ${itemsize}',
                buffer_slice_pitch='0',
                host_row_pitch='problem_size * ${itemsize}',
                host_slice_pitch='0',
                ctype=ctype
                ))

        if lang == 'opencl':
            # determine operation type
            ctype = 'Write' if to_device else 'Read'
            if use_full:
                return Template(guarded_call(lang, Template("""
            clEnqueue${ctype}Buffer(queue, ${dev_name}, CL_TRUE, 0,
                      ${buff_size}, &${host_name},
                      0, NULL, NULL)""").safe_substitute(ctype=ctype)))
            elif order == 'C' or ndim <= 1:
                # this is a simple opencl-copy
                return Template(guarded_call(lang, Template("""
            clEnqueue${ctype}Buffer(queue, ${dev_name}, CL_TRUE, 0,
                      ${this_run_size}, &${host_name}[${host_offset}*${non_ic_size}],
                      0, NULL, NULL)""").safe_substitute(ctype=ctype)))
            elif have_split:
                return __f_split(ctype)

            else:
                return __f_unsplit(ctype)

        elif lang == 'c':
            ctype = 'in' if to_device else 'out'
            host_arrays = ['${host_name}']
            dev_arrays = ['${dev_name}']

            def __combine(host, dev):
                arrays = dev + host if to_device else host + dev
                arrays = ', '.join(arrays)
                return arrays

            if use_full:
                arrays = __combine(host_arrays, dev_arrays)
                # don't put in the offset
                return Template(Template(
                    'memcpy(${arrays}, ${buff_size});').safe_substitute(
                        arrays=arrays))
            elif order == 'C' or ndim <= 1:
                host_arrays[0] = '&' + host_arrays[0] + \
                    '[offset * ${non_ic_size}]'
                arrays = __combine(host_arrays, dev_arrays)
                return Template(Template(
                    'memcpy(${arrays}, ${this_run_size});'
                    ).safe_substitute(arrays=arrays))
            elif order == 'F':
                if not isinstance(self, pinned_memory):
                    # not pinned memory -> no vectorized data ordering
                    # therefore, we only need the regular 2D-C copy, rather than
                    # the writeRect implementation
                    dev_arrays += ['per_run']
                    host_arrays += ['problem_size']
                    arrays = __combine(host_arrays, dev_arrays)
                    return Template(Template(
                        'memcpy2D_${ctype}(${arrays}, offset, '
                        'this_run * ${itemsize}, ${non_ic_size});'
                                             ).safe_substitute(
                                             ctype=ctype, arrays=arrays))
                if have_split:
                    return __f_split(ctype)
                else:
                    return __f_unsplit(ctype)

    def __init__(self, lang, order, have_split, **overrides):

        self.order = order
        self.have_split = have_split

        def __update(key, val):
            if key not in overrides:
                overrides[key] = val

        alloc = {'opencl': Template(Template(
                    '${name} = clCreateBuffer(context, ${memflag}, '
                    '${buff_size}, NULL, &return_code);\n'
                    '${guard}\n').safe_substitute(guard=guarded_call(
                        'opencl', 'return_code'))),
                 'c': Template(Template(
                    '${name} = (${dtype})malloc(${buff_size});\n'
                    '${guard}').safe_substitute(guard=guarded_call(
                        'c', '${name} != NULL', 'malloc failed')))}
        __update('alloc', alloc)

        # we use blocking read / writes here, so no need to sync currently
        # in the future this would be a convenient place to put in multiple-device
        # synchronizations
        sync = {}
        __update('sync', sync)

        __update('use_full', False)
        use_full = overrides['use_full']
        copy_in_2d = self._get_2d_templates(
            lang, to_device=True, ndim=2, use_full=use_full)
        __update('copy_in_2d', copy_in_2d)
        copy_out_2d = self._get_2d_templates(
            lang, to_device=False, ndim=2, use_full=use_full)
        __update('copy_out_2d', copy_out_2d)
        copy_in_1d = self._get_2d_templates(
            lang, to_device=True, use_full=use_full)
        __update('copy_in_1d', copy_in_1d)
        copy_out_1d = self._get_2d_templates(
            lang, to_device=False, use_full=use_full)
        __update('copy_out_1d', copy_out_1d)
        host_constant_template = self._get_2d_templates(
            lang, to_device=True, use_full=True)
        __update('host_const_in', host_constant_template)
        free = {'opencl': Template(guarded_call(
                         'opencl', 'clReleaseMemObject(${name})')),
                'c': Template('free(${name});')}
        __update('free', free)
        memset = {'opencl': Template(Template(
            """
            #if CL_LEVEL >= 120
                ${fill_call}
            #else
                ${write_call}
            #endif
            """
            ).safe_substitute(fill_call=guarded_call(
                            'opencl', 'clEnqueueFillBuffer(queue, ${name}, &zero, '
                            'sizeof(double), 0, ${buff_size}, 0, NULL, NULL)'),
                              write_call=guarded_call(
                            'opencl', 'clEnqueueWriteBuffer(queue, ${name}, CL_TRUE,'
                            ' 0, ${buff_size}, zero, 0, NULL, NULL)'))),
            'c': Template('memset(${name}, 0, ${buff_size});')
        }
        __update('memset', memset)

        alloc_flags = {'opencl': {
            False: 'CL_MEM_READ_WRITE',
            True: 'CL_MEM_READ_ONLY'}}
        __update('alloc_flags', alloc_flags)

        if 'use_full' in overrides:
            overrides.pop('use_full')
        super(mapped_memory, self).__init__(lang, **overrides)


class pinned_memory(mapped_memory):
    def __init__(self, lang, order, have_split, **kwargs):

        self.order = order
        self.have_split = have_split

        # copies in / out are not needed for pinned memory
        # If the value is an input or output, it will be be supplied as a host buffer
        copy_in_2d = self._get_2d_templates(
            host_langs[lang], to_device=True, ndim=2)
        copy_out_2d = self._get_2d_templates(
            host_langs[lang], to_device=False, ndim=2)
        copy_in_1d = self._get_2d_templates(
            host_langs[lang], to_device=True)
        copy_out_1d = self._get_2d_templates(
            host_langs[lang], to_device=False)
        host_const_in = self._get_2d_templates(
            host_langs[lang], to_device=True, use_full=True)
        copies = {
            'copy_in_2d': copy_in_2d,
            'copy_out_2d': copy_out_2d,
            'copy_in_1d': copy_in_1d,
            'copy_out_1d': copy_out_1d,
            'host_const_in': host_const_in
        }
        self.map_template = {'opencl': Template(
            'temp_${d_short} = (${dtype}*)clEnqueueMapBuffer(queue, ${dev_name}, '
            'CL_TRUE, ${map_flags}, 0, ${per_run_size}, 0, NULL, NULL, '
            '&return_code);\n'
            '${check}'
            ).safe_substitute(check=guarded_call(lang, 'return_code'))}
        self.unmap_template = {'opencl': guarded_call(
            lang, 'clEnqueueUnmapMemObject(queue, ${dev_name}, temp_${d_short}, 0, '
                  'NULL, NULL)')}
        self.map_flags = {'opencl': {True: 'CL_MAP_WRITE',
                                     False: 'CL_MAP_READ'}}

        dev_map_template = Template(Template(
            '// map to host address space for initialization\n'
            '${map}\n'
            '// set memory\n'
            '${host}\n'
            '// and unmap back to device\n'
            '${unmap}\n').safe_substitute(
            map=self.map_template[lang],
            unmap=self.unmap_template[lang]
        ))
        # now need to place a map around the copies
        copies = {k: Template(dev_map_template.safe_substitute(
            host=v.safe_substitute(dev_name='temp_${d_short}')))
            for k, v in six.iteritems(copies)}

        # and fixup host constant
        copies['host_const_in'] = Template(copies['host_const_in'].safe_substitute(
            this_run_size='${buff_size}'))

        # finally need to setup memset as a
        # map buffer -> write -> unmap combination

        memset = {
            'c': Template('memset(${name}, 0, ${buff_size});')
        }
        memset[lang] = Template(dev_map_template.safe_substitute(
            host=memset[host_langs[lang]].safe_substitute(
                name='temp_${d_short}'),
            dev_name='${name}',
            per_run_size='${buff_size}'))

        self.pinned_hostaloc_flags = {'opencl': 'CL_MEM_ALLOC_HOST_PTR'}
        self.pinned_hostbuff_flags = {'opencl': 'CL_MEM_USE_HOST_PTR'}

        # get the defaults from :class:`mapped_memory`
        super(pinned_memory, self).__init__(lang, order, have_split,
                                            memset=memset, **copies)

    def alloc(self, device, name, buff_size='', per_run_size='',
              readonly=False, host_constant=False, dtype='double', **kwargs):
        """
        Return an allocation for a buffer in the given :param:`lang`

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        name: str
            The desired name of the buffer
        buff_size: str ['']
            The actual size of the buffer.  May be overriden by :param:`per_run_size`
            if supplied, _unless_ :param:`host_ptr` is not 'NULL'
        per_run_size: str ['']
            The maximum allowable size for device allocation of this buffer
            If supplied If supplied, and :param:`device` is True,
            overrides :param:`buff_size` unless :param:`host_ptr` is not 'NULL'
        host_constant: bool [False]
            If True, this is a migrated host constant, hence we have no size
            restriction
        per_run_size: str ['']
            The maximum allowable size for this array, per-kernel-call
        dtype: str ['double']
            The type of the array to allocate
        Returns
        -------
        alloc_str: str
            The resulting allocation instrucitons
        """

        # fiddle with the flags
        flags = self.pinned_hostaloc_flags
        size = per_run_size if device else buff_size

        return super(pinned_memory, self).alloc(
            device, name, buff_size=size, readonly=readonly,
            flags=flags[self.device_lang], host_constant=host_constant,
            dtype=dtype, **kwargs)

    def copy(self, to_device, *args, **kwargs):
        """
        An override of the base :func:`copy` method to steal copies of the host
        buffer and place them in to a map / unmap for pinned memory

        Parameters
        ----------
        to_device: bool
            If True, transfer the data to the device buffer
        dtype: str ['']
            The dtype of this host constant, this is needed to select the correct
            temporary mapping array
        Note
        ----
        All other args and kwargs are passed directly to :func:`mapped_memory.copy`

        Returns
        -------
        copy_str: str
            The resulting copy instructions
        """

        map_flags = self.map_flags[self.lang(True)][to_device]
        dtype = kwargs.pop('dtype')
        return super(pinned_memory, self).copy(
            to_device, *args, dtype=dtype, d_short=dtype[0],
            map_flags=map_flags, **kwargs)

    def memset(self, device, *args, **kwargs):
        """
        An override of the base :func:`memset` to steal copies of the host buffer
        and place them in to a map / unmap for pinned memory

        Parameters
        ----------
        dtype: str ['']
            The dtype of this host constant, this is needed to select the correct
            temporary mapping array
        Note
        ----
        All other args and kwargs are passed directly to :func:`mapped_memory.memset`

        Returns
        -------
        copy_str: str
            The resulting copy instructions

        """

        map_flags = self.map_flags[self.lang(True)][device]
        dtype = kwargs.pop('dtype')
        return super(pinned_memory, self).memset(
            device, *args, dtype=dtype, d_short=dtype[0],
            map_flags=map_flags, **kwargs)


class memory_manager(object):

    """
    Aids in defining & allocating arrays for various languages
    """

    def __init__(self, lang, order, array_splitter,
                 dev_type=None, strided_c_copy=False):
        """
        Parameters
        ----------
        lang : str
            The language used in this memory initializer
        order: ['F', 'C']
            The order of the arrays used
        array_splitter: :class:`array_splitter`
            Used to determine whether copies to/from the device correspond to a
            C-split or F-split format (depending on :param:`order`)
        dev_type: :class:`pyopencl.device_type` [None]
            The device type.  If CPU, the host buffers will be used for input /
            output variables
        strided_c_copy: bool [False]
            Used in testing strided memory copies for c-targets
        """

        # no need to do 2-d copy if we're doing a wide-vectorized 'F' ordered SIMD
        have_split = False

        self.arrays = []
        self.in_arrays = []
        self.out_arrays = []
        self.host_constants = []
        self.lang = lang
        self.order = order
        self.memory_types = {np.dtype('float64'): {'c': 'double*',
                                                   'opencl': 'cl_mem'},
                             np.dtype('int32'): {'c': 'int*',
                                                 'opencl': 'cl_mem'}
                             }
        self.type_map = {np.dtype('int32'): 'int',
                         np.dtype('float64'): 'double'}
        self.dev_type = dev_type
        self.use_pinned = self.dev_type is not None and self.dev_type == DTYPE_CPU
        kwargs = {}
        if not utils.can_vectorize_lang[lang] and strided_c_copy:
            kwargs['use_full'] = False

        if self.use_pinned:
            self.mem = pinned_memory(lang, order, have_split, **kwargs)
        else:
            self.mem = mapped_memory(lang, order, have_split, **kwargs)
        self.host_constant_template = Template(
            'const ${type} h_${name} [${size}] = {${init}}'
            )

        self.string_strides, self.div_mod_strides = \
            memory_manager.get_string_strides()

    @staticmethod
    def get_string_strides():
        string_strides = [p_size.name, w_size.name]
        # convert string strides to regex, and include the div/mod form
        ss_size = len(string_strides)
        div_mod_strides = []
        for i in range(ss_size):
            name = string_strides[i]
            div_mod_re = re.compile(
                r'\(-1\)\*\(\(\(-1\)\*{}\) // (\d+)\)'.format(name))
            # convert the name to a regex
            string_strides[i] = re.compile(re.escape(name))
            # and add the divmod
            div_mod_strides.append(div_mod_re)

        return string_strides, div_mod_strides

    def add_arrays(self, arrays=[], in_arrays=[], out_arrays=[],
                   host_constants=[]):
        """
        Adds arrays to the manager

        Parameters
        ----------
        arrays : list of :class:`lp.GlobalArg`
            The arrays to declare
        in_arrays : list of str
            The array names that form the input to this kernel
        out_arrays : list of str
            The array names that form the output of this kernel
        host_constants: list of :class:`lp.GlobalArg`
            Host constants to transfer to a device

        Returns
        -------
        None
        """
        self.arrays.extend(
            [x for x in arrays if not isinstance(x, lp.ValueArg)])
        self.in_arrays.extend(in_arrays)
        self.out_arrays.extend(out_arrays)
        self.host_constants.extend(host_constants)
        self.arrays.extend([x for x in host_constants if x not in self.arrays])

    def fix_arrays(self, arrays):
        """
        Converts :attr:`in_arrays` and :attr:`out_arrays` to
        :class:`loopy.KernelArguments` by name

        Raises
        ------
        AssertionError:
            If any array in :attr:`in_arrays` + :attr:`out_arrays` not in
            :param:`arrays`

        Returns
        -------
        None
        """

        arr_dict = {x.name: x for x in arrays}

        for i, name in enumerate(self.in_arrays):
            assert name in arr_dict
            self.in_arrays[i] = arr_dict[name]

        for i, name in enumerate(self.out_arrays):
            assert name in arr_dict
            self.out_arrays[i] = arr_dict[name]

    @property
    def host_arrays(self):
        def _set_sort(arr):
            return sorted(set(arr), key=lambda x: arr.index(x))
        return _set_sort(self.in_arrays + self.out_arrays)

    @property
    def host_lang(self):
        return host_langs[self.lang]

    def get_defns(self):
        """
        Returns the definition strings for this memory manager's arrays

        Parameters
        ----------
        None

        Returns
        -------
        defn_str : str
            A string of global memory definitions
        """

        def __add(arraylist, lang, prefix, defn_list):
            for arr in arraylist:
                defn_list.append(self.memory_types[self._handle_type(arr)][lang] +
                                 ' ' + prefix + arr.name + utils.line_end[lang])

        defns = []
        # get all 'device' defns
        __add(self.arrays, self.lang, device_prefix, defns)

        # return defn string
        return '\n'.join(sorted(set(defns)))

    def _handle_type(self, arr):
        return to_loopy_type(arr.dtype).numpy_dtype

    def get_host_constants(self):
        """
        Returns allocations of initialized constant variables on the host.
        These result when we run out of __constant memory on the device, and must
        migrate to passing constant __global args.

        Parameters
        ----------
        None

        Returns
        -------
        alloc_str : str
            The string of memory allocations
        """

        def _stringify(arr):
            return ', '.join(['{}'.format(x) for x in arr.initializer.flatten(
                self.order)])

        return '\n'.join([self.host_constant_template.safe_substitute(
            name=x.name,
            type=self.type_map[self._handle_type(x)],
            size=str(int(np.prod(x.shape))),
            init=_stringify(x))
            + utils.line_end[self.lang] for x in self.host_constants])

    def get_mem_allocs(self, host=False, host_postfix='_local'):
        """
        Returns the allocation strings for this memory manager's arrays

        Parameters
        ----------
        host : bool
            If true, only define host arrays

        Returns
        -------
        alloc_str : str
            The string of memory allocations
        """

        def __get_alloc_and_memset(dev_arr, prefix):
            # if we're allocating inputs, call them something different
            # than the input args from python
            post_fix = host_postfix if host else ''

            in_host_const = any(dev_arr.name == y.name for y in self.host_constants)
            if host and in_host_const:
                return ''

            # get name
            name = prefix + dev_arr.name + post_fix
            # if it's opencl, we need to declare the buffer type
            buff_size = self._get_size(dev_arr)
            readonly = any(x.name == dev_arr.name for x in self.host_constants)
            host_ptr = 'NULL'
            per_run_size = self._get_size(dev_arr, subs_n='per_run')
            # check if buffer is input / output
            if not host and self.use_pinned\
                    and dev_arr.name in self.host_arrays\
                    and not in_host_const:
                # don't pass host constants this way, as it breaks some
                # pinned memory implementations -- should we always use the
                # HOST_MEM_ALLOC?

                # cast to void
                formatter = '(void*) {}'
                host_ptr = formatter.format(host_prefix + dev_arr.name)

            # generate allocs
            alloc = self.mem.alloc(not host,
                                   name=name,
                                   readonly=readonly,
                                   buff_size=buff_size,
                                   host_ptr=host_ptr,
                                   per_run_size=per_run_size,
                                   dtype=self.memory_types[
                                        self._handle_type(dev_arr)][self.host_lang])

            if host:
                # add a type
                alloc = self.memory_types[self._handle_type(dev_arr)][
                    self.host_lang] + ' ' + alloc

            # generate allocs
            return_list = [alloc]

            # don't reset constants or pinned host pointers
            if not in_host_const and host_ptr == 'NULL':
                # add the memset
                return_list.append(self.mem.memset(
                    not host, name=name, buff_size=buff_size,
                    per_run_size=per_run_size,
                    dtype=self.type_map[self._handle_type(dev_arr)]))
            # return
            return '\n'.join(return_list + ['\n'])

        to_alloc = [
            next(x for x in self.arrays if x.name == y) for y in self.host_arrays
            ] if host else self.arrays
        prefix = host_prefix if host else device_prefix
        alloc_list = [__get_alloc_and_memset(arr, prefix) for arr in to_alloc]

        return '\n'.join(alloc_list)

    def _get_size(self, arr, subs_n='problem_size', include_item_size=True,
                  return_as_dict=False):
        size = arr.shape
        nsize = []
        str_size = []
        skip = []
        # remove 'problem_size' from shape if present, as it's baked into the
        # various defns
        for i, x in enumerate(size):
            s = str(x)
            # check for non-integer sizes
            if any(x.search(s) for x in self.string_strides):
                str_size.append(subs_n)
                vsize = next(x.search(s) for x in self.div_mod_strides)
                if vsize:
                    # it's a floor division thing, need to do some cleanup here
                    # make sure we found the vector width
                    assert vsize is not None and len(vsize.groups()) > 0, (
                        'Unknown size for array {}, :{}'.format(arr.name, s))
                    vsize = int(vsize.group(1))
                    assert vsize in size, (
                        'Unknown vector-width: {}, found for array {}'.format(
                            vsize, arr.name))
                    # and eliminate it from the list
                    skip.append(size.index(vsize))
                # skip this entry
                skip.append(i)

        # find all numerical sizes
        nsize = [size[i] for i in range(len(size)) if i not in skip]
        if nsize:
            nsize = str(np.cumprod(nsize, dtype=arc.kint_type)[-1])
            str_size.append(nsize)
        # multiply by the item size
        item_size = ''
        if include_item_size:
            item_size = 'sizeof({})'.format(self.type_map[self._handle_type(arr)])
            str_size.append(item_size)

        if not any(x for x in str_size):
            # if empty, we're multiplying by this anyways...
            str_size.append(str(1))
        if return_as_dict:
            return {'item_size': item_size,
                    'str_size': [x for x in str_size if x != item_size]}
        return ' * '.join([x for x in str_size if x])

    def _mem_transfers(self, to_device=True, host_constants=False, host_postfix=''):
        if not host_constants:
            # get arrays
            arr_list = self.in_arrays if to_device else self.out_arrays
            # filter out host constants
            arr_list = [x for x in arr_list if not any(
                y.name == x for y in self.host_constants)]
            # put into map
            arr_maps = {a.name: a for a in self.arrays}

        else:
            arr_list = [x.name for x in self.host_constants]
            arr_maps = {x.name: x for x in self.host_constants}
            assert to_device

        copy_intructions = []
        for arr in arr_list:
            ret_size = self._get_size(arr_maps[arr], return_as_dict=True)
            ssize = ret_size['str_size']
            item_size = ret_size['item_size']

            def __stringify(subs_n=None, do_item_size=True):
                string_size = ssize[:]
                if subs_n is not None:
                    string_size = [x if x != p_size.name else subs_n
                                   for x in string_size]
                if do_item_size:
                    string_size.append(item_size)
                # filter empty
                string_size = [x for x in string_size if x]
                if not string_size:
                    string_size.append('1')
                return ' * '.join([x for x in string_size if x])

            copy_intructions.append(self.mem.copy(
                to_device,
                host_name=host_prefix + arr + host_postfix,
                dev_name=device_prefix + arr,
                buff_size=__stringify(),
                dim=len(arr_maps[arr].shape),
                this_run_size=__stringify(subs_n='this_run'),
                itemsize=item_size,
                other_dim_size=int(np.prod(arr_maps[arr].shape[2:])),
                per_run_size=__stringify(subs_n='per_run'),
                non_ic_size=__stringify(subs_n='', do_item_size=False),
                host_constant=host_constants,
                dtype=self.type_map[self._handle_type(arr_maps[arr])]
                ))

        return '\n'.join([x for x in copy_intructions if x])

    def get_mem_transfers_in(self):
        """
        Generates the memory transfers into the device before kernel execution

        Parameters
        ----------
        None

        Returns
        -------
        mem_transfer_in : str
            The string to perform the memory transfers before execution
        """

        return self._mem_transfers(to_device=True)

    def get_mem_transfers_out(self):
        """
        Generates the memory transfers into the back to the host after kernel
        execution

        Parameters
        ----------
        None

        Returns
        -------
        mem_transfer_out : str
            The string to perform the memory transfers back to the host after
            execution
        """

        return self._mem_transfers(to_device=False)

    def get_mem_strategy(self):
        """
        Returns the memory strategy MAPPED or PINNED used in memory creation

        Parameters
        ----------
        None

        Returns
        -------
        strat: str
            The memory strategy
        """

        return 'PINNED' if isinstance(self.mem, pinned_memory) else 'MAPPED'

    def get_host_constants_in(self):
        """
        Generates the memory transfers of the host constants
        into the device before kernel execution

        Parameters
        ----------
        None

        Returns
        -------
        mem_transfer_in : str
            The string to perform the memory transfers before execution
        """

        return self._mem_transfers(to_device=True, host_constants=True)

    def get_mem_frees(self, free_locals=False):
        """
        Returns code to free the allocated buffers

        Parameters
        ----------
        free_locals : bool
            If true, we're freeing the local versions of the host arrays

        Returns
        -------
        mem_free_str : str
            The generated code
        """

        if not free_locals:
            # device memory
            frees = [self.mem.free(not free_locals, name=device_prefix + arr.name)
                     for arr in self.arrays]
        else:
            frees = [self.mem.free(not free_locals,
                                   name=host_prefix + arr + '_local')
                     for arr in self.host_arrays if not any(
                        x.name == arr for x in self.host_constants)]

        return '\n'.join([x for x in frees if x])
