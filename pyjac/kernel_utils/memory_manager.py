"""
memory_manager.py - generators for defining / allocating / transfering memory
for kernel creation
"""

from __future__ import division

from .. import utils

from string import Template
import numpy as np
import loopy as lp
from loopy.types import to_loopy_type
import re
import yaml
from enum import Enum
import six
import logging
from ..core import array_creator as arc
import resource
align_size = resource.getpagesize()

try:
    from pyopencl import device_type
    DTYPE_CPU = device_type.CPU
except:
    DTYPE_CPU = -1


class memory_type(Enum):
    m_constant = 0,
    m_local = 1,
    m_global = 2,
    m_alloc = 3
    m_pagesize = 4

    def __int__(self):
        return self.value


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
    """

    def __init__(self, lang, arrays, limits):
        """
        Initializes a :class:`memory_limits`
        """
        self.lang = lang
        self.arrays = arrays
        self.limits = limits

    def can_fit(self, type=memory_type.m_constant, with_type_changes={}):
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
        can_fit: int
            The maximum number of times these arrays can be fit in memory.
            - For global memory, this determines the number of initial conditions
            that can be evaluated.
            - For shared and constant memory, this determines whether this data
            can be fit
        """

        if self.lang == 'c':
            return True

        # filter arrays by type
        arrays = self.arrays[type]
        arrays = [a for a in arrays if not any(
            a in v for k, v in six.iteritems(with_type_changes) if k != type)]

        per_alloc_ic_limit = np.iinfo(np.int).max
        per_ic = 0
        static = 0
        for array in arrays:
            size = 1
            is_ic_dep = False
            for s in array.shape:
                if arc.problem_size.name in str(s):
                    # mark as dependent on # of initial conditions
                    is_ic_dep = True
                    # get the floor div (if any)
                    floor_div = re.search('// (\d+)', str(s))
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
            else:
                static += size

            # also need to check the maximum allocation size for opencl
            logger = logging.getLogger(__name__)
            if self.lang == 'opencl':
                if is_ic_dep:
                    per_alloc_ic_limit = np.minimum(
                        per_alloc_ic_limit,
                        np.floor(self.limits[memory_type.m_alloc] / per_ic))
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

        # return the number of times we can fit these array
        return int(np.maximum(np.minimum(
            np.floor((self.limits[type] - static) / per_ic), per_alloc_ic_limit), 0))

    @staticmethod
    def get_limits(loopy_opts, arrays, input_file=''):
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

        Returns
        -------
        limits: :class:`memory_limits`
            An initialized :class:`memory_limits` that can determine the total
            'global', 'constant' and 'local' memory available on the device
        """
        limits = {memory_type.m_pagesize: align_size}
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
        if input_file:
            with open(input_file, 'r') as file:
                # load from file
                lims = yaml.load(file.read())
                mtype = utils.EnumType(memory_type)
                limits = {}
                choices = [mt.name.lower()[2:] for mt in memory_type] + ['alloc']
                for key, value in six.iteritems(lims):
                    # check in memory type
                    if not key.lower() in choices:
                        msg = ', '.join(choices)
                        msg = '{0}: use one of {1}'.format(memory_type.name, msg)
                        raise Exception(msg)
                    key += 'm_'
                    # update with enum
                    limits[mtype(key)] = value

        return memory_limits(loopy_opts.lang, arrays,
                             {k: v for k, v in six.iteritems(limits)})


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
                 alloc_flags={}):
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

    def lang(self, device):
        return self.host_lang if not device else self.device_lang

    def alloc(self, device, name, buff_size, readonly=False, host_ptr='NULL',
              **kwargs):
        """
        Return an allocation for a buffer in the given :param:`lang`

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        name: str
            The desired name of the buffer
        readonly: bool
            Changes memory flags / allocation type (e.g., for OpenCL)
        host_ptr: str ['NULL']
            Unused by mapped memory (default)

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

        return self.alloc_template[self.lang(device)].safe_substitute(
            name=name, buff_size=buff_size, memflag=flags, host_ptr=host_ptr,
            **kwargs)

    def copy(self, to_device, host_name, dev_name, buff_size, dim, **kwargs):
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

        Returns
        -------
        copy_str: str
            The resulting copy instructions
        """
        # get templates
        if dim <= 1:
            template = self.copy_in_1d if to_device else self.copy_out_1d
        else:
            template = self.copy_in_2d if to_device else self.copy_out_2d

        if not template:
            return ''

        return template.safe_substitute(
            host_buff=host_name, dev_buff=dev_name, buff_size=buff_size, **kwargs)

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

    def memset(self, device, name, buff_size, **kwargs):
        """
        Set the host / device memory

        Parameters
        ----------
        device: bool
            If true, set a device buffer, else a host buffer
        name: str
            The buffer name
        buff_size: str
            The size of the buffer in bytes

        Returns
        -------
        set_instructions: str
            The instructions to memset the buffer
        """

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
    def __get_2d_templates(self, lang, use_full=False, to_device=False):
        """
        Returns a template to copy (part or all) of a 2D array from "host" to
        "device" arrays (note this is simulated on C)

        Parameters
        ----------
        use_full: bool [False]
            If true, get the copy template for the full array (useful for
            if/then guarded full copies)
        to_device: bool [False]
            If true, Write to device, else Read from device

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
            # determine operation type
            ctype = 'Write' if to_device else 'Read'
            if order == 'C' or use_full:
                # this is a simple opencl-copy
                return Template(guarded_call(lang, Template("""
            clEnqueue${ctype}Buffer(queue, ${dev_buff}, CL_TRUE, 0,
                      ${this_run_size}, &${host_buff}[offset * ${non_ic_size}],
                      0, NULL, NULL)""").safe_substitute(ctype=ctype)))
            elif have_split:
                # this us a F-split which requires a Rect Read/Write
                # with the given pitches / region / offsets
                # Note that this is actually a 3D (or 4D in the case of the)
                # Jacobian, but we can treat all subsequent dimensions 2 and
                # after as a flat 1-D array (as they are flattened in this order)
                return Template(guarded_call(lang, Template("""
            clEnqueue${ctype}BufferRect(queue, ${dev_buff}, CL_TRUE,
                (size_t[]) {0, 0, 0}, //buffer origin
                (size_t[]) {0, offset, 0}, //host origin
                (size_t[]) {VECWIDTH * ${itemsize}, this_run, ${other_dim_size}}\
, // region
                VECWIDTH * ${itemsize}, // buffer row pitch
                VECWIDTH * per_run * ${itemsize}, //buffer slice pitch
                VECWIDTH * ${itemsize}, //host row pitch,
                VECWIDTH * problem_size * ${itemsize}, //host slice pitch,
                ${host_buff}, 0, NULL, NULL)""").safe_substitute(ctype=ctype)))
            else:
                # this is a regular F-ordered array, which only requires a 2D
                # copy
                return Template(guarded_call(lang, Template("""
            clEnqueue${ctype}BufferRect(queue, ${dev_buff}, CL_TRUE,
                (size_t[]) {0, 0, 0}, //buffer origin
                (size_t[]) {offset * ${itemsize}, 0, 0}, //host origin
                (size_t[]) {this_run * ${itemsize}, ${non_ic_size}, 1}, // region
                per_run * ${itemsize}, // buffer row pitch
                0, //buffer slice pitch
                problem_size * ${itemsize}, //host row pitch,
                0, //host slice pitch,
                ${host_buff}, 0, NULL, NULL)""").safe_substitute(
                    ctype=ctype)))
        elif lang == 'c':
            ctype = 'in' if to_device else 'out'
            host_arrays = ['${host_buff}']
            dev_arrays = ['${dev_buff}']

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
            if order == 'C':
                host_arrays[0] += '&' + host_arrays[0] + \
                    '[offset * ${non_ic_size}]'
                arrays = __combine(host_arrays, dev_arrays)
                return Template(Template(
                    'memcpy(${arrays}, this_run * ${non_ic_size} * ${itemsize});'
                    ).safe_substitute(arrays=arrays))
            elif order == 'F':
                dev_arrays += ['per_run']
                host_arrays += ['problem_size']
                arrays = __combine(host_arrays(dev_arrays))
                return Template(Template(
                    'memcpy2D_${ctype}(${arrays}, offset, '
                    'this_run * ${itemsize}, ${non_ic_size});'
                                         ).safe_substitute(
                                         ctype=ctype, arrays=arrays))

    def __init__(self, lang, order, have_split, strided_c_copy=False,
                 **overrides):

        self.order = order
        self.have_split = have_split
        self.strided_c_copy = strided_c_copy

        def __update(key, val):
            if key not in overrides:
                overrides[key] = val

        alloc = {'opencl': Template(Template(
                    '${name} = clCreateBuffer(context, ${memflag}, ${buff_size},'
                    '${host_ptr}, &return_code);\n'
                    '${guard}\n').safe_substitute(guard=guarded_call(
                        'opencl', 'return_code'))),
                 'c': Template(Template(
                    '${name} = (double*)malloc(${buff_size});\n'
                    '${guard}').safe_substitute(guard=guarded_call(
                        'c', '${name} != NULL', 'malloc failed')))}
        __update('alloc', alloc)

        # we use blocking read / writes here, so no need to sync currently
        # in the future this would be a convenient place to put in multiple-device
        # synchronizations
        sync = {}
        __update('sync', sync)

        copy_in_2d = self.__get_2d_templates(
            lang, to_device=True, use_full=lang == 'c' and not strided_c_copy)
        __update('copy_in_2d', copy_in_2d)
        copy_out_2d = self.__get_2d_templates(
            lang, to_device=False, use_full=lang == 'c' and not strided_c_copy)
        __update('copy_out_2d', copy_out_2d)
        copy_in_1d = self.__get_2d_templates(lang, to_device=True, use_full=True)
        __update('copy_in_1d', copy_in_1d)
        copy_out_1d = self.__get_2d_templates(lang, to_device=False, use_full=True)
        __update('copy_out_1d', copy_out_1d)
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
                            'sizeof(double), 0,${buff_size}, 0, NULL, NULL)'),
                              write_call=guarded_call(
                            'opencl', 'clEnqueueWriteBuffer(queue, ${name}, CL_TRUE,'
                            ' 0, ${buff_size}, zero, 0, NULL, NULL)'))),
            'c': Template('memset(${name}, 0, ${buff_size});')
        }
        __update('memset', memset)

        alloc_flags = {'opencl': {
            True: 'CL_MEM_READ_WRITE',
            False: 'CL_MEM_READ_ONLY'}}
        __update('alloc_flags', alloc_flags)
        super(mapped_memory, self).__init__(lang, **overrides)


class pinned_memory(mapped_memory):
    def __init__(self, lang, order, have_split, align_size, strided_c_copy=False,
                 **kwargs):

        self.align_size = align_size
        # and modify as necessary
        alloc = {'c': Template(Template(
            '${name} = (double*)aligned_alloc(${align_size}, ${buff_size});\n'
            '${guard}').safe_substitute(guard=guarded_call(
                'c', '${name} != NULL', 'aligned_alloc failed.'),
                                        align_size=self.align_size)),
                 'opencl': Template(Template(
                    '${name} = clCreateBuffer(context, ${memflag}, '
                    '${buff_size}, ${host_ptr}, &return_code);\n'
                    '${guard}').safe_substitute(guard=guarded_call(
                        'opencl', 'return_code')))}
        # synchronizations
        sync = {'opencl': guarded_call('opencl', 'clFinish(queue)')}

        # copies in / out are not needed for pinned memory
        # If the value is an input or output, it will be be supplied as a host buffer
        copies = {'copy_out_1d': {},
                  'copy_in_1d': {},
                  'copy_out_2d': {},
                  'copy_in_2d': {}}

        self.map_template = {'opencl': Template(
            'temp = (double*)clEnqueueMapBuffer(queue, ${name}, CL_TRUE, '
            'CL_MAP_WRITE, 0, ${buff_size}, 0, NULL, NULL, &return_code);\n'
            '${check}'
            ).safe_substitute(check=guarded_call(lang, 'return_code'))}
        self.unmap_template = {'opencl': guarded_call(
            lang, 'clEnqueueUnmapMemObject(queue, ${name}, temp, 0, '
                  'NULL, NULL)')}

        # finally need to setup memset as a
        # map buffer -> write -> unmap combination

        memset = {
            'c': Template('memset(${name}, 0, ${buff_size});')
        }
        memset[lang] = Template(Template(
            '// map to host address space for initialization\n'
            '${map}\n'
            '// set memory\n'
            '${host_memset}\n'
            '// and unmap back to device\n'
            '${unmap}\n').safe_substitute(
            map=self.map_template[lang],
            host_memset=memset[host_langs[lang]].safe_substitute(name='temp'),
            unmap=self.unmap_template[lang]
        ))

        self.pinned_hostaloc_flags = {'opencl': 'CL_MEM_ALLOC_HOST_PTR'}
        self.pinned_hostbuff_flags = {'opencl': 'CL_MEM_USE_HOST_PTR'}

        # get the defaults from :class:`mapped_memory`
        super(pinned_memory, self).__init__(lang, order, have_split,
                                            strided_c_copy=strided_c_copy,
                                            alloc=alloc, sync=sync, memset=memset,
                                            **copies)

    def alloc(self, device, name, buff_size, readonly=False, host_ptr='NULL',
              **kwargs):
        """
        Return an allocation for a buffer in the given :param:`lang`

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        name: str
            The desired name of the buffer
        flags: str
            The memory flags (for OpenCL) to be used
        readonly: bool [False]
            Changes memory flags / allocation type (e.g., for OpenCL)
        host_buff: str ['']
            If supplied, use this as the base buffer for the pinned memory
        Returns
        -------
        alloc_str: str
            The resulting allocation instrucitons
        """

        # fiddle with the flags
        flags = self.pinned_hostaloc_flags
        if host_ptr != 'NULL':
            flags = self.pinned_hostbuff_flags

        return super(pinned_memory, self).alloc(
            device, name, buff_size, readonly=readonly,
            flags=flags[self.device_lang], host_ptr=host_ptr)


class memory_manager(object):

    """
    Aids in defining & allocating arrays for various languages
    """

    def __init__(self, lang, order, have_split, strided_c_copy=False,
                 dev_type=None, mem_limits_file=''):
        """
        Parameters
        ----------
        lang : str
            The language used in this memory initializer
        order: ['F', 'C']
            The order of the arrays used
        have_split: bool
            If true, the arrays in this manager correspond to a C-split or
            F-split format (depending on :param:`order`)
        strided_c_copy: bool [False]
            If true, enable strided copies for 'C'
        dev_type: :class:`pyopencl.device_type` [None]
            The device type.  If CPU, the host buffers will be used for input /
            output variables
        """
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
        if self.use_pinned:
            # check if user supplied alignement size
            # create dummy options
            loopy_opts = type('', (object,), {'lang': host_langs[lang]})
            align_size = memory_limits.get_limits(
                loopy_opts, {}, mem_limits_file).limits[memory_type.m_pagesize]
            self.mem = pinned_memory(lang, order, have_split, align_size)
        else:
            self.mem = mapped_memory(lang, order, have_split)
        self.host_constant_template = Template(
            'const ${type} h_${name} [${size}] = {${init}}'
            )

    def add_arrays(self, arrays=[], in_arrays=[], out_arrays=[]):
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

        Returns
        -------
        None
        """
        self.arrays.extend(
            [x for x in arrays if not isinstance(x, lp.ValueArg)])
        self.in_arrays.extend(in_arrays)
        self.out_arrays.extend(out_arrays)

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
            size=str(np.prod(x.shape)),
            init=_stringify(x))
            + utils.line_end[self.lang] for x in self.host_constants])

    def get_mem_allocs(self, alloc_locals=False):
        """
        Returns the allocation strings for this memory manager's arrays

        Parameters
        ----------
        alloc_locals : bool
            If true, only define host arrays

        Returns
        -------
        alloc_str : str
            The string of memory allocations
        """

        def __get_alloc_and_memset(dev_arr, prefix):
            # if we're allocating inputs, call them something different
            # than the input args from python
            post_fix = '_local' if alloc_locals else ''

            in_host_const = any(dev_arr.name == y.name for y in self.host_constants)
            if alloc_locals and in_host_const:
                return ''

            # get name
            name = prefix + dev_arr.name + post_fix
            # if it's opencl, we need to declare the buffer type
            buff_size = self._get_size(dev_arr)
            readonly = not any(
                    x.name == dev_arr.name for x in self.host_constants)
            host_ptr = 'NULL'
            # check if buffer is input / output
            if not alloc_locals and self.use_pinned\
                    and dev_arr.name in self.host_arrays:
                # cast to void
                formatter = '(void*) {}'
                if in_host_const:
                    # cast to const void
                    formatter = '(const void*) {}'
                host_ptr = formatter.format(host_prefix + dev_arr.name)
            # generate allocs
            alloc = self.mem.alloc(not alloc_locals,
                                   name=name,
                                   readonly=readonly,
                                   buff_size=buff_size,
                                   host_ptr=host_ptr)

            if alloc_locals:
                # add a type
                alloc = self.memory_types[self._handle_type(dev_arr)][
                    self.host_lang] + ' ' + alloc

            # generate allocs
            return_list = [alloc]

            # don't reset constants or pinned host pointers
            if not in_host_const and host_ptr == 'NULL':
                # add the memset
                return_list.append(self.mem.memset(
                    not alloc_locals, name=name, buff_size=buff_size))
            # return
            return '\n'.join(return_list + ['\n'])

        to_alloc = [
            next(x for x in self.arrays if x.name == y) for y in self.host_arrays
            ] if alloc_locals else self.arrays
        prefix = host_prefix if alloc_locals else device_prefix
        alloc_list = [__get_alloc_and_memset(arr, prefix) for arr in to_alloc]

        return '\n'.join(alloc_list)

    def _get_size(self, arr, subs_n='problem_size', include_item_size=True):
        size = arr.shape
        nsize = []
        str_size = []
        skip = []
        # remove 'problem_size' from shape if present, as it's baked into the
        # various defns
        for i, x in enumerate(size):
            s = str(x)
            # check for non-integer sizes
            if 'problem_size' in s:
                str_size.append(subs_n)
                if s != 'problem_size':
                    # it's a floor division thing, need to do some cleanup here
                    vsize = re.search(
                        r'\(-1\)\*\(\(\(-1\)\*problem_size\) // (\d+)\)', s)
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
            nsize = str(np.cumprod(nsize, dtype=np.int32)[-1])
            str_size.append(nsize)
        # multiply by the item size
        if include_item_size:
            str_size.append(str(arr.dtype.itemsize))
        if not any(x for x in str_size):
            # if empty, we're multiplying by this anyways...
            str_size.append(str(1))
        return ' * '.join([x for x in str_size if x])

    def _mem_transfers(self, to_device=True, host_constants=False, host_postfix='',
                       get_sync=False):
        if get_sync:
            # don't need to transfer as we use host pointers
            # instead we simply need to enqueue a barrier so that we
            # ensure we have consistent memory
            return self.mem.sync()
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

        copy_intructions = [self.mem.copy(
                to_device,
                host_name=host_prefix + arr + host_postfix,
                dev_name=device_prefix + arr,
                buff_size=self._get_size(arr_maps[arr]),
                dim=len(arr_maps[arr].shape),
                this_run_size=self._get_size(arr_maps[arr], subs_n='this_run'),
                itemsize=arr_maps[arr].dtype.itemsize,
                other_dim_size=np.prod(arr_maps[arr].shape[2:]),
                non_ic_size=self._get_size(arr_maps[arr], subs_n='',
                                           include_item_size=False)
                ) for arr in arr_list]
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

    def get_mem_sync(self):
        """
        Synchronizes OpenCL (or other) mapped host buffers

        Parameters
        ----------
        None

        Returns
        -------
        mem_sync : str
            The string to perform the host memory synchronization
        """
        return self._mem_transfers(get_sync=True)

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
