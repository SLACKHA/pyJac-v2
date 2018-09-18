"""
memory_limits.py - Handle limits on device memory

(C) Nicholas Curtis - 2018
"""

from __future__ import division

import six
import logging
import re

import numpy as np
from enum import Enum
from pytools import ImmutableRecord
#  import resource
#  align_size = resource.getpagesize()

from pyjac.core.array_creator import problem_size as p_size
from pyjac.core.array_creator import work_size as w_size
from pyjac import utils
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

        if not any(s.search(str(x)) for x in arry.shape
                   for s in self.string_strides):
            return np.iinfo(dtype).max

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
                logger.debug(
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
                if w_size.name in str(s):
                    # mark as dependent on the work size
                    is_ws_dep = True
                elif p_size.name in str(s):
                    # mark as dependent on # of initial conditions
                    is_ic_dep = True
                if is_ic_dep or is_ws_dep:
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
                   string_strides=[p_size.name],
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
                    logger.debug('Overriding memory-limit for type {} from value '
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


def get_string_strides():
    """
    Stride names we might see in variable indexing
    """
    string_strides = [p_size.name]
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
