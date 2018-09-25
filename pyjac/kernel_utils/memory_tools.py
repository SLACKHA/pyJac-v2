"""
Standalone memory tools that can be easily imported into Cog / unit-tests
to generate copys / allocations / etc. for host and device memory.
"""

from string import Template

import six
import loopy as lp
import numpy as np

from pyjac.utils import indent, stdindent
from pyjac.core.enum_types import DeviceMemoryType
from pyjac.core.array_creator import problem_size


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


class Namer(object):
    def __init__(self, owner=None, prefix=None, postfix=None):
        if owner and prefix:
            prefix = '{}->{}'.format(owner, prefix)
        elif owner:
            prefix = '{}->'.format(owner)
        self.prefix = prefix if prefix is not None else ''
        self.postfix = postfix if postfix is not None else ''

    def __call__(self, name, **kwargs):
        prefix = kwargs.get('prefix', self.prefix)
        postfix = kwargs.get('postfix', self.postfix)
        return prefix + name + postfix


class HostNamer(Namer):
    def __init__(self, owner='', **kwargs):
        super(HostNamer, self).__init__(owner=owner, prefix=host_prefix,
                                        **kwargs)


class DeviceNamer(Namer):
    def __init__(self, owner='', **kwargs):
        super(DeviceNamer, self).__init__(owner=owner, prefix=device_prefix,
                                          **kwargs)


class StrideCalculator(object):
    """
    Convenience class to calculate mixed string / ValueArg / integer strides /
    shapes

    Attributes
    ----------
    type_map: dict
        A map of loopy types to c-types
    """

    def __init__(self, type_map):
        self.type_map = type_map.copy()

    def buffer_size(self, arr, subs={}, include_sizeof=True):
        """
        Returns the calculated buffer shape as a string (including dtype)
        """

        int_shape = 1
        str_shape = ''

        def __update_str(sshape, add):
            if add in subs:
                add = subs[add]
            if sshape and add:
                return sshape + ' * {}'.format(add)
            elif not add:
                return sshape
            else:
                return str(add)

        for x in arr.shape:
            # try as integer
            try:
                if isinstance(x, lp.ValueArg):
                    str_shape = __update_str(str_shape, x.name)
                else:
                    int_shape *= int(x)
            except TypeError:
                # it's a stringified value arg
                str_shape = __update_str(str_shape, str(x))

        # stitch together
        buff_size = __update_str(str(int_shape), str_shape) if int_shape != 1 else \
            str_shape
        if include_sizeof:
            buff_size = __update_str(buff_size, 'sizeof({})'.format(self.type_map[
                arr.dtype]))
        elif not buff_size:
            buff_size = '1'

        return buff_size

    def non_ic_size(self, arr, subs={problem_size.name: ''}):
        """
        Return the size in number of elements (Note: not bytes!) of the array
        dimensions that do no correspond to initial conditions axes
        """

        return self.buffer_size(arr, subs, include_sizeof=False)


class MemoryManager(object):
    def __init__(self, lang, order, type_map, alloc={}, sync={}, copy_in_1d={},
                 copy_in_2d={}, copy_out_1d={}, copy_out_2d={}, free={}, memset={},
                 alloc_flags={}, host_const_in={}, host_namer=None,
                 device_namer=None):
        self.order = order
        self.type_map = type_map.copy()
        self.def_map = {}
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
        self.host_namer = host_namer
        self.device_namer = device_namer
        self.definition = {'c': Template('${mem_type} ${name};'),
                           'opencl': Template('${mem_type} ${name};')}

        self.mem_type = {'c': self.determine_c_mem_type,
                         'opencl': self.determine_cl_mem_type}

    def determine_c_mem_type(self, arr, include_pointer=False):
        # array
        pointer = '*'
        temp = '{type}{pointer}'
        if isinstance(arr, lp.ValueArg):
            # int
            pointer = ''
        return temp.format(type=self.type_map[arr.dtype],
                           pointer=pointer)

    def determine_cl_mem_type(self, arr, include_pointer=False):
        if isinstance(arr, lp.ValueArg):
            if arr.dtype.is_integral():
                # hack to get unsigned ints
                temp = 'cl_u{}'
            else:
                temp = 'cl_{}'
            return temp.format(self.type_map[arr.dtype])
        # array
        return 'cl_mem'

    def dtype(self, device, arr, include_pointer=False):
        return self.mem_type[self.lang(device)](arr, include_pointer=include_pointer)

    def buffer_size(self, device, arr, num_ics='per_run', include_sizeof=True):
        calc = StrideCalculator(self.type_map)
        # setup substitution for device buffer # of initial conditions
        subs = {}
        if device:
            subs[problem_size.name] = num_ics
        return calc.buffer_size(arr, subs, include_sizeof=include_sizeof)

    def non_ic_size(self, arr, subs=None):
        kwargs = {}
        if subs:
            kwargs['subs'] = subs
        return StrideCalculator(self.type_map).non_ic_size(arr, **kwargs)

    def get_name(self, device, arr, **kwargs):
        namer = self.device_namer if device else self.host_namer
        try:
            name = arr.name
        except AttributeError:
            name = arr
        return name if not namer else namer(name, **kwargs)

    def get_signature(self, device, arr):
        """
        Returns the stringified version of :param:`arg`, for a use in a function
        signature definition.
        """

        return self.dtype(device, arr, True) + ' ' + self.get_name(device, arr)

    def lang(self, device):
        return self.host_lang if not device else self.device_lang

    def define(self, device, arr, host_constant=False, force_no_const=False):
        """
        Declare a host or device array

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        arr: :class:`loopy.ArrayArg`
            The buffer to allocate
        host_constant: bool [False]
            If true, define as a host constant (i.e., use an initializer)
        force_no_const: bool [False]
            Used for testing -- define a host constant w/o applying the const
            attribute
        """

        if host_constant:
            assert isinstance(arr, lp.TemporaryVariable) and isinstance(
                arr.initializer, np.ndarray)
            if device:
                raise Exception('Cannot directly define a host-constant on the '
                                'device')

        name = self.get_name(device, arr)
        dtype = self.mem_type[self.lang(device)](arr)

        if host_constant:
            init = arr.initializer.flatten(self.order)
            size = init.shape[0]
            precision = '{:d}' if arr.dtype.is_integral() else '{:.16e}'
            init = ', '.join([precision.format(x) for x in init])
            return '{const}{dtype} {name}[{size}] = {{{init}}};'.format(
                const='const ' if not force_no_const else '',
                dtype=self.type_map[arr.dtype], name=name, size=size, init=init)

        return self.definition[self.lang(device)].safe_substitute(
            mem_type=dtype, name=name)

    def alloc(self, device, arr, readonly=False, num_ics='per_run', **kwargs):
        """
        Return an allocation for a buffer in the given :param:`lang`

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        arr: :class:`loopy.ArrayArg`
            The buffer to allocate
        num_ics: str ['per_run']
            The number of initial conditions to evaluated per run
        readonly: bool
            Changes memory flags / allocation type (e.g., for OpenCL)

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

        name = self.get_name(device, arr)
        dtype = self.type_map[arr.dtype]
        buff_size = self.buffer_size(device, arr, num_ics=num_ics)

        return self.alloc_template[self.lang(device)].safe_substitute(
            name=name, buff_size=buff_size, memflag=flags,
            dtype=dtype, **kwargs)

    def copy(self, to_device, arr, host_constant=False, num_ics='per_run',
             num_ics_this_run='this_run', offset='offset', **kwargs):
        """
        Return a copy of a buffer to/from the "device"

        Parameters
        ----------
        to_device: bool
            If True, transfer to the "device"
        arr: :class:`loopy.ArrayArg`
            The buffer to use for transfering between host and device
        host_constant: bool [False]
            If True, we are transferring a host constant, hence set any offset
            in the host buffer to zero
        num_ics: str ['per_run']
            The number of initial conditions to evaluated per prun
        num_ics_this_run: str ['this_run']
            The number of initial conditions to evaluated _in this run_,
            should be <= :param:`num_ics`.
        offset: str ['offset']
            The initial condition offset
        Returns
        -------
        copy_str: str
            The resulting copy instructions
        """
        # get templates
        if host_constant:
            template = self.host_const_in
        elif len(arr.shape) <= 1:
            template = self.copy_in_1d if to_device else self.copy_out_1d
        else:
            template = self.copy_in_2d if to_device else self.copy_out_2d

        if not host_constant:
            kwargs['this_run'] = num_ics_this_run
            kwargs['per_run'] = num_ics
            kwargs['offset'] = offset
            # update special sizes in kwargs
            kwargs['non_ic_size'] = self.non_ic_size(arr)
            kwargs['per_run_size'] = self.buffer_size(True, arr,
                                                      num_ics=num_ics)
            kwargs['this_run_size'] = self.buffer_size(True, arr,
                                                       num_ics=num_ics_this_run)
            kwargs['itemsize'] = 'sizeof({})'.format(self.type_map[arr.dtype])
            buff_size = self.buffer_size(True, arr, num_ics=num_ics)
        else:
            kwargs['offset'] = '0'
            buff_size = self.buffer_size(False, arr)

        if not template:
            return ''

        host_name = self.get_name(False, arr)
        dev_name = self.get_name(True, arr)

        return template.safe_substitute(
            host_name=host_name, dev_name=dev_name, buff_size=buff_size,
            host_offset=offset if not host_constant else '0', **kwargs)

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

    def memset(self, device, arr, num_ics='per_run', **kwargs):
        """
        Set the host / device memory

        Parameters
        ----------
        device: bool
            If true, set a device buffer, else a host buffer
        arr: :class:`loopy.ArrayArg`
            The array to set
        num_ics: str ['per_run']
            The number of initial conditions to evaluated per prun

        Returns
        -------
        set_instructions: str
            The instructions to memset the buffer
        """

        name = self.get_name(device, arr)
        buff_size = self.buffer_size(device, arr, num_ics=num_ics)
        return self.memset_template[self.lang(device)].safe_substitute(
            name=name, buff_size=buff_size, **kwargs)

    def free(self, device, arr):
        """
        Generates code to free a buffer

        Parameters
        ----------
        name: :class:`loopy.ArrayArg`
            The buffer to free

        Returns
        -------
        free_str: str
            The resulting free instructions
        """
        return self.free_template[self.lang(device)].safe_substitute(
            name=self.get_name(device, arr))


class MappedMemory(MemoryManager):
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
            rect_copy_template = Template(
                '{\n' +
                indent('const size_t buffer_origin[3] = {0, 0, 0};\n', stdindent) +
                indent('const size_t host_origin[3] = ${host_origin};\n', stdindent)
                + indent('const size_t region[3] = ${region};\n', stdindent) +
                indent(guarded_call(
                    lang,
                    'clEnqueue${ctype}BufferRect(queue, ${dev_name}, CL_TRUE, '
                    '&buffer_origin[0], '  # buffer origin
                    '&host_origin[0], '  # host origin
                    '&region[0], '  # region
                    '${buffer_row_pitch}, '  # buffer_row_pitch
                    '${buffer_slice_pitch}, '  # buffer_slice_pitch
                    '${host_row_pitch}, '  # host_row_pitch
                    '${host_slice_pitch}, '  # host_slice_pitch
                    '${host_name}, 0, NULL, NULL)'
                ), stdindent) +
                '\n}\n')
        elif lang == 'c':
            rect_copy_template = Template(
                '{\n' +
                indent('const size_t host_origin[3] = ${host_origin};\n', stdindent)
                + indent('const size_t region[3] = ${region};\n', stdindent) +
                indent('memcpy2D_${ctype}(${host_name}, ${dev_name}, '
                       '&host_origin[0], '  # host origin
                       '&region[0], '  # region
                       '${buffer_row_pitch}, '  # buffer_row_pitch
                       '${buffer_slice_pitch}, '  # buffer_slice_pitch
                       '${host_row_pitch}, '  # host_row_pitch
                       '${host_slice_pitch}'  # host_slice_pitch
                       ');', stdindent) +
                '\n}\n')

        def __f_split(ctype):
            # this us a F-split which requires a Rect Read/Write
            # with the given pitches / region / offsets
            # Note that this is actually a 3D (or 4D in the case of the)
            # Jacobian, but we can treat all subsequent dimensions 2 and
            # after as a flat 1-D array (as they are flattened in this order)

            return Template(rect_copy_template.safe_substitute(
                host_origin='{0, ${offset}, 0}',
                region='{VECWIDTH * ${itemsize}, ${this_run}, ${other_dim_size}}',
                buffer_row_pitch='VECWIDTH * ${itemsize}',
                buffer_slice_pitch='VECWIDTH * ${per_run} * ${itemsize}',
                host_row_pitch='VECWIDTH * ${itemsize}',
                host_slice_pitch='VECWIDTH * {problem_size} * ${{itemsize}}'.format(
                    problem_size=problem_size.name),
                ctype=ctype
                ))

        def __f_unsplit(ctype):
            # this is a regular F-ordered array, which only requires a 2D copy
            return Template(rect_copy_template.safe_substitute(
                host_origin='{${offset} * ${itemsize}, 0, 0}',
                region='{${this_run} * ${itemsize}, ${non_ic_size}, 1}',
                buffer_row_pitch='${per_run} * ${itemsize}',
                buffer_slice_pitch='0',
                host_row_pitch='{problem_size} * ${{itemsize}}'.format(
                    problem_size=problem_size.name),
                host_slice_pitch='0',
                ctype=ctype
                ))

        if lang == 'opencl':
            # determine operation type
            ctype = 'Write' if to_device else 'Read'
            if use_full:
                return Template(guarded_call(lang, Template(
                    'clEnqueue${ctype}Buffer(queue, ${dev_name}, CL_TRUE, 0, '
                    '${buff_size}, &${host_name}, 0, NULL, NULL)').safe_substitute(
                    ctype=ctype)))
            elif order == 'C' or ndim <= 1:
                # this is a simple opencl-copy
                return Template(guarded_call(lang, Template(
                    'clEnqueue${ctype}Buffer(queue, ${dev_name}, CL_TRUE, 0, '
                    '${this_run_size}, &${host_name}[${host_offset}*${non_ic_size}],'
                    ' 0, NULL, NULL)').safe_substitute(ctype=ctype)))
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
                    '[${offset} * ${non_ic_size}]'
                arrays = __combine(host_arrays, dev_arrays)
                return Template(Template(
                    'memcpy(${arrays}, ${this_run_size});'
                    ).safe_substitute(arrays=arrays))
            elif order == 'F':
                if not isinstance(self, PinnedMemory):
                    # not pinned memory -> no vectorized data ordering
                    # therefore, we only need the regular 2D-C copy, rather than
                    # the writeRect implementation
                    dev_arrays += ['${per_run}']
                    host_arrays += [problem_size.name]
                    arrays = __combine(host_arrays, dev_arrays)
                    return Template(Template(
                        'memcpy2D_${ctype}(${arrays}, ${offset}, '
                        '${this_run} * ${itemsize}, ${non_ic_size});'
                                             ).safe_substitute(
                                             ctype=ctype, arrays=arrays))
                if have_split:
                    return __f_split(ctype)
                else:
                    return __f_unsplit(ctype)

    def __init__(self, lang, order, type_map, device_namer=None, host_namer=None,
                 **overrides):

        self.order = order
        self.type_map = type_map.copy()
        # currently only use non-split memory inputs / outputs
        self.have_split = False

        def __update(key, val):
            if key not in overrides:
                overrides[key] = val

        alloc = {'opencl': Template(Template(
                    '${name} = clCreateBuffer(context, ${memflag}, '
                    '${buff_size}, NULL, &return_code);\n'
                    '${guard}\n').safe_substitute(guard=guarded_call(
                        'opencl', 'return_code'))),
                 'c': Template(Template(
                    '${name} = (${dtype}*)malloc(${buff_size});\n'
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
        super(MappedMemory, self).__init__(lang, order, type_map,
                                           host_namer=host_namer,
                                           device_namer=device_namer, **overrides)


class PinnedMemory(MappedMemory):
    def __init__(self, lang, order, type_map, host_namer=None, device_namer=None,
                 **kwargs):

        self.order = order
        self.type_map = type_map.copy()
        # currently only use non-split memory inputs / outputs
        self.have_split = False

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
            '${temp_name} = (${dtype}*)clEnqueueMapBuffer(queue, ${dev_name}, '
            'CL_TRUE, ${map_flags}, 0, ${per_run_size}, 0, NULL, NULL, '
            '&return_code);\n'
            '${check}'
            ).safe_substitute(check=guarded_call(lang, 'return_code'))}
        self.unmap_template = {'opencl': guarded_call(
            lang, 'clEnqueueUnmapMemObject(queue, ${dev_name}, ${temp_name}, 0, '
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
            host=v.safe_substitute(dev_name='${temp_name}')))
            for k, v in six.iteritems(copies)}

        # and fixup host constant -- all sizes are fixed for host constant
        copies['host_const_in'] = Template(copies['host_const_in'].safe_substitute(
            this_run_size='${buff_size}',
            per_run_size='${buff_size}'))

        # finally need to setup memset as a
        # map buffer -> write -> unmap combination

        memset = {
            'c': Template('memset(${name}, 0, ${buff_size});')
        }
        memset[lang] = Template(dev_map_template.safe_substitute(
            host=memset[host_langs[lang]].safe_substitute(
                name='${temp_name}'),
            dev_name='${name}',
            per_run_size='${buff_size}'))

        self.pinned_hostaloc_flags = {'opencl': 'CL_MEM_ALLOC_HOST_PTR'}
        self.pinned_hostbuff_flags = {'opencl': 'CL_MEM_USE_HOST_PTR'}

        # get the defaults from :class:`mapped_memory`
        super(PinnedMemory, self).__init__(lang, order, type_map, memset=memset,
                                           host_namer=host_namer,
                                           device_namer=device_namer, **copies)

    def alloc(self, device, arr, readonly=False, num_ics='per_run',
              namer=None, **kwargs):
        """
        Return an allocation for a buffer in the given :param:`lang`

        Parameters
        ----------
        device: bool
            If true, allocate a device buffer, else a host buffer
        arr: :class:`loopy.ArrayArg`
            The desired name of the buffer
        readonly: bool
            Changes memory flags / allocation type (e.g., for OpenCL)
        namer: :class:`six.Callable` [None]
            A callable function to generate the name of the buffer to be allocated.
            If not specified, simply use :param:`arr.name`.

        Returns
        -------
        alloc_str: str
            The resulting allocation instrucitons
        """

        # fiddle with the flags
        flags = self.pinned_hostaloc_flags

        return super(PinnedMemory, self).alloc(
            device, arr, readonly=readonly, flags=flags[self.device_lang],
            namer=namer, num_ics=num_ics, **kwargs)

    def get_temp_name(self, dtype):
        # temp's are always host buffers!
        return self.get_name(False, 'temp_{}'.format(dtype[0]), postfix='')

    def copy(self, to_device, arr, **kwargs):
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

        map_flags = self.map_flags[self.lang(True)][to_device]
        dtype = self.type_map[arr.dtype]
        return super(PinnedMemory, self).copy(
            to_device, arr, dtype=dtype, temp_name=self.get_temp_name(dtype),
            map_flags=map_flags, **kwargs)

    def memset(self, device, arr, *args, **kwargs):
        """
        An override of the base :func:`memset` to steal copies of the host buffer
        and place them in to a map / unmap for pinned memory

        Parameters
        ----------
        device: bool
            If true, set a device buffer, else a host buffer
        arr: :class:`loopy.ArrayArg`
            The array to set
        num_ics: str ['per_run']
            The number of initial conditions to evaluated per prun
        namer: :class:`six.Callable` [None]
            A callable function to generate the name of the buffer to be allocated.
            If not specified, simply use :param:`arr.name`.

        Note
        ----
        All other args and kwargs are passed directly to :func:`mapped_memory.memset`

        Returns
        -------
        copy_str: str
            The resulting copy instructions

        """

        map_flags = self.map_flags[self.lang(True)][device]
        dtype = self.type_map[arr.dtype]
        return super(PinnedMemory, self).memset(
            device, arr, *args, dtype=dtype, temp_name=self.get_temp_name(dtype),
            map_flags=map_flags, **kwargs)


def get_memory(callgen, host_namer=None, device_namer=None):
    """
    Returns the appropriate memory manager given this :class:`pyjac.CallgenResult`
    object.

    Parameters
    ----------
    callgen: :class:`CallgenResult`
        The unpickled callgen result
    host_namer: :class:`Namer`
        If specified, an instance of the :class:`Namer` to transform the names of
        host arrays
    device_namer: :class:`Namer`
        If specified, an instance of the :class:`Namer` to transform the names of
        device arrays
    """

    if callgen.dev_mem_type == DeviceMemoryType.pinned:
        return PinnedMemory(callgen.lang, callgen.order, callgen.type_map,
                            host_namer=host_namer, device_namer=device_namer)
    else:
        return MappedMemory(callgen.lang, callgen.order, callgen.type_map,
                            host_namer=host_namer, device_namer=device_namer)
