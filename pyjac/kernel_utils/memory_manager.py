"""
memory_manager.py - generators for defining / allocating / transfering memory
for kernel creation
"""

from .. import utils

from string import Template
import numpy as np
import loopy as lp
import re


class memory_manager(object):

    """
    Aids in defining & allocating arrays for various languages
    """

    def __init__(self, lang):
        """
        Parameters
        ----------
        lang : str
            The language used in this memory initializer
        """
        self.arrays = []
        self.in_arrays = []
        self.out_arrays = []
        self.lang = lang
        self.memory_types = {'c': 'double*',
                             'opencl': 'cl_mem'}
        self.host_langs = {'opencl': 'c',
                           'c': 'c'}
        self.alloc_templates = {'opencl': Template(
            '${name} = clCreateBuffer(context, ${memflag}, '
            '${buff_size}, NULL, &return_code)'),
                                'c': Template(
            '${name} = (double*)malloc(${buff_size})')}
        self.copy_in_templates = {'opencl': Template(
            'clEnqueueWriteBuffer(queue, ${name}, CL_TRUE, 0, '
            '${buff_size}, ${host_buff}, 0, NULL, NULL)'),
                                  'c': Template(
            'memcpy(${name}, ${host_buff}, ${buff_size})')}
        self.copy_out_templates = {'opencl': Template(
            'clEnqueueReadBuffer(queue, ${name}, CL_TRUE, 0, '
            '${buff_size}, ${host_buff}, 0, NULL, NULL)'),
                                   'c': Template(
            'memcpy(${host_buff}, ${name}, ${buff_size})')}
        self.memset_templates = {'opencl': Template(
            """
            #if CL_LEVEL >= CL_VERSION_1_2
                clEnqueueFillBuffer(queue, ${name}, ${fill_value}, ${fill_size}, 0,
                    ${size}, 0, NULL, NULL)
            #else
                clEnqueueWriteBuffer(queue, ${name}, CL_TRUE, 0, ${fill_size},
                    zero, 0, NULL, NULL)
            #endif
            """
            ),
            'c': Template('memset(${name}, 0, ${size})')
        }
        self.free_template = {'opencl': Template('clReleaseMemObject(${name})'),
                              'c': Template('free(${name})')}

    def get_check_err_call(self, call, lang=None):
        if lang is None:
            lang = self.lang
        if lang == 'opencl':
            return Template('check_err(${call})').safe_substitute(call=call)
        else:
            return call

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
        return self.host_langs[self.lang]

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
                defn_list.append(self.memory_types[lang] + ' ' + prefix +
                                 arr + utils.line_end[lang])

        defns = []
        # get all 'device' defns
        __add([x.name for x in self.arrays], self.lang, 'd_', defns)

        # return defn string
        return '\n'.join(sorted(set(defns)))

    def get_mem_allocs(self, alloc_locals=False):
        """
        Returns the allocation strings for this memory manager's arrays

        Parameters
        ----------
        alloc_locals : bool
            If true, only define host arrays

        Returns
        alloc_str : str
            The string of memory allocations
        """

        def __get_alloc_and_memset(dev_arr, lang, prefix):
            # if we're allocating inputs, call them something different
            # than the input args from python
            post_fix = '_local' if alloc_locals else ''

            # get name
            name = prefix + dev_arr.name + post_fix
            # if it's opencl, we need to declare the buffer type
            memflag = None
            if lang == 'opencl':
                memflag = 'CL_MEM_READ_WRITE'

            # generate allocs
            alloc = self.alloc_templates[lang].safe_substitute(
                name=name,
                memflag=memflag,
                buff_size=self._get_size(dev_arr))

            if alloc_locals:
                # add a type
                alloc = self.memory_types[self.host_lang] + ' ' + alloc

            # generate allocs
            return_list = [alloc]

            # add error checking
            if lang == 'opencl':
                return_list.append(self.get_check_err_call('return_code'))

            # add the memset
            return_list.append(
                self.get_check_err_call(
                    self.memset_templates[lang].safe_substitute(
                        name=name,
                        buff_size=self._get_size(dev_arr),
                        fill_value='&zero',  # fill for OpenCL kernels
                        fill_size='sizeof(double)',  # fill type
                        size=self._get_size(dev_arr),
                        ), lang=lang))

            # return
            return '\n'.join([r + utils.line_end[lang] for r in return_list] +
                             ['\n'])

        to_alloc = [
            next(x for x in self.arrays if x.name == y) for y in self.host_arrays
            ] if alloc_locals else self.arrays
        prefix = 'h_' if alloc_locals else 'd_'
        lang = self.lang if not alloc_locals else self.host_lang
        alloc_list = [__get_alloc_and_memset(
            arr, lang, prefix) for arr in to_alloc]

        return '\n'.join(alloc_list)

    def _get_size(self, arr, subs_n='problem_size'):
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
        str_size.append(str(arr.dtype.itemsize))
        return ' * '.join(str_size)

    def _mem_transfers(self, to_device=True):
        arr_list = self.in_arrays if to_device else self.out_arrays
        arr_maps = {x: next(y for y in self.arrays if x == y.name)
                    for x in arr_list}
        templates = self.copy_in_templates if to_device else self.copy_out_templates

        return '\n'.join([
            self.get_check_err_call(templates[self.lang].safe_substitute(
                name='d_' + arr, host_buff='h_' + arr,
                buff_size=self._get_size(arr_maps[arr]))) +
            utils.line_end[self.lang] for arr in arr_list])

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
            frees = [self.get_check_err_call(
                self.free_template[self.lang].safe_substitute(name='d_' + arr.name))
                     for arr in self.arrays]
        else:
            frees = [self.free_template[self.host_lang].safe_substitute(
                name='h_' + arr + '_local') for arr in self.host_arrays]

        return '\n'.join([x + utils.line_end[self.lang] for x in sorted(set(frees))])
