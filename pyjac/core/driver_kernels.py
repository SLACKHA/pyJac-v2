# -*- coding: utf-8 -*-

"""
Generates driver functions that handle running pyJac kernels.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# system modules
import logging
from string import Template

# external modules
import six
import numpy as np

from pyjac.utils import listify, stringify_args
from pyjac.core.exceptions import InvalidInputSpecificationException
from pyjac.core import array_creator as arc
from pyjac.kernel_utils import kernel_gen as k_gen


def lockstep_driver(loopy_opts, namestore, inputs, outputs, driven,
                    test_size=None):
    """
    Implements a lockstep-based driver function for kernel evaluation.
    This allows pyJac to utilize a smaller working-buffer (sized to the
    global work size), and implements a static(like) scheduling algorithm

    Notes
    -----
    Currently Loopy doesn't have the machinery to enable native calling of other
    loopy kernels, so we have to fudge this a bit (and this can't be used for
    unit-tests).  Future versions will allow us to natively wrap test functions
    (i.e., once the new function calling interface is in place in Loopy)

    :see:`driver-function` for more information

    Parameters
    ----------
    loopy_opts: :class:`loopy_options`
        The loopy options specifying how to create this kernel
    namestore: :class:`NameStore`
        The namestore class that owns our arrays
    inputs: list of str
        The name of arrays that should be copied into internal working buffers
        before calling subfunctions
    outputs: list of str
        The name of arrays should be copied back into global memory after calling
        subfunctions
    driven: :class:`kernel_generator`
        The kernel generator to wrap in the driver

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    # we have to do some shennanigains here to get this to work in loopy:
    #
    # 1. Loopy currently doesn't allow you to alter the for-loop increment size,
    #    so for OpenCL where we must increment by the global work size, we have to
    #    put a dummy for-loop in, and teach the kernel generator to work around it
    #
    # 2. Additionally, the OpenMP target in Loopy is Coming Soon (TM), hence we need
    #    our own dynamic scheduling preamble for the driver loop (
    #    if we're operating in queue-form)
    #
    # 3. Finally, Loopy is just now supporting the ability to natively call other
    #    kernels, so for the moment we still need to utilize the dummy function
    #    calling we have set-up for the finite difference Jacobian

    # first, get our input / output arrays
    arrays = {}
    to_find = set(listify(inputs)) | set(listify(outputs))
    for arr in to_find:
        arr_creator = getattr(namestore, arr, None)
        if arr_creator is None:
            continue
        arrays[arr] = arr_creator

    if len(arrays) != len(to_find):
        missing = to_find - set(arrays.keys())
        logger = logging.getLogger(__name__)
        logger.debug('Input/output arrays for queue_driver kernel {} not found.'
                     .format(stringify_args(missing)))
        raise InvalidInputSpecificationException(missing)

    def arr_non_ic(array_input):
        return len(array_input.shape) > 1

    # ensure the inputs and output are all identically sized (among those that have)
    # a non-initial condition dimension

    def __check(check_input):
        shape = ()
        nameref = None
        desc = 'Input' if check_input else 'Output'
        for inp in [arrays[x] for x in (inputs if check_input else outputs)]:
            if not arr_non_ic(inp):
                # only the initial condition dimension, fine
                continue
            if shape:
                if inp.shape != shape:
                    logger = logging.getLogger(__name__)
                    logger.debug('{} array for queue_driver kernel {} does not '
                                 'match expected shape (from array {}).  '
                                 'Expected: ({}), got: ({})'.format(
                                    desc, inp.name, nameref,
                                    stringify_args(inp.shape), stringify_args(shape))
                                 )
                    raise InvalidInputSpecificationException(inp.name)
            else:
                nameref = inp.name
                shape = inp.shape[:]
        if not shape:
            logger = logging.getLogger(__name__)
            logger.debug('No {} arrays supplied to queue_driver that require '
                         'copying to working buffer!'.format(desc))
            raise InvalidInputSpecificationException('queue_driver')
    __check(True)
    __check(False)

    def create_interior_kernel(for_input):
        name = 'copy_{}'.format('in' if for_input else 'out')
        # get arrays
        arrs = [arrays[x] for x in (inputs if for_input else outputs)]
        # get shape and interior size
        shape = next(arr.shape for arr in arrs if arr_non_ic(arr))

        # create a dummy map and store
        map_shape = np.arange(shape[1], dtype=arc.kint_type)
        mapper = arc.creator(name, arc.kint_type, map_shape.shape, 'C',
                             initializer=map_shape)
        mapstore = arc.MapStore(loopy_opts, mapper, test_size)

        from pytools import UniqueNameGenerator
        # determine what other inames we need, if any
        namer = UniqueNameGenerator(set([mapstore.iname]))
        extra_inames = []
        for i in six.moves.range(2, len(shape)):
            iname = namer(mapstore.iname)
            extra_inames.append((iname, '0 <= {} < {}'.format(
                iname, shape[i])))

        indicies = [arc.global_ind, mapstore.iname] + [
            ex[0] for ex in extra_inames]

        def __build(arr, **kwargs):
            if arr_non_ic(arr):
                return mapstore.apply_maps(arr, *indicies, **kwargs)
            else:
                return mapstore.apply_maps(arr, arc.global_ind, **kwargs)

        # create working buffer version of arrays
        working_buffers = []
        working_strs = []
        for arr in arrs:
            arr_lp, arr_str = __build(arr, use_local_name=True)
            working_buffers.append(arr_lp)
            working_strs.append(arr_str)

        # create global versions of arrays
        buffers = []
        strs = []
        for arr in arrs:
            arr_lp, arr_str = __build(arr, is_input_or_output=True)
            buffers.append(arr_lp)
            strs.append(arr_str)

        # now create the instructions
        instruction_template = Template("""
            ${local_buffer} = ${global_buffer}
        """) if for_input else Template("""
            ${global_buffer} = ${local_buffer}
        """)

        instructions = []
        for i, arr in enumerate(arrs):
            instructions.append(instruction_template.substitute(
                local_buffer=working_strs[i],
                global_buffer=strs[i]))
        instructions = '\n'.join(instructions)

        # and return the kernel info
        return k_gen.knl_info(name=name,
                              instructions=instructions,
                              mapstore=mapstore,
                              var_name=arc.var_name,
                              extra_inames=extra_inames,
                              kernel_data=buffers + working_buffers + [
                                arc.work_size])

    copy_in = create_interior_kernel(True)
    # create a dummy kernel info that simply calls our internal function
    instructions = 'dummy()'
    func_call = k_gen.knl_info(name='driver_call',
                               instructions=instructions,
                               mapstore=copy_in.mapstore.copy(),
                               kernel_data=[x.copy() for x in copy_in.kernel_data],
                               var_name=arc.var_name,
                               extra_inames=copy_in.extra_inames[:])
    copy_out = create_interior_kernel(False)

    # and return
    return [copy_in, func_call, copy_out]


def queue_driver(loopy_opts, namestore, inputs, outputs, driven,
                 test_size=None):
    """
    Implements an atomic-queue based driver function for kernel evaluation.
    This allows pyJac to utilize a smaller working-buffer (sized to the
    global work size), and implements a dynamic(like) scheduling algorithm

    Notes
    -----
    Currently Loopy doesn't have the machinery to enable native calling of other
    loopy kernels, so we have to fudge this a bit (and this can't be used for
    unit-tests).  Future versions will allow us to natively wrap test functions
    (i.e., once the new function calling interface is in place in Loopy)

    :see:`driver-function` for more information

    Parameters
    ----------
    loopy_opts: :class:`loopy_options`
        The loopy options specifying how to create this kernel
    namestore: :class:`NameStore`
        The namestore class that owns our arrays
    inputs: list of str
        The name of arrays that should be copied into internal working buffers
        before calling subfunctions
    outputs: list of str
        The name of arrays should be copied back into global memory after calling
        subfunctions
    driven: :class:`kernel_generator`
        The kernel generator to wrap in the driver

    Returns
    -------
    knl_list : list of :class:`knl_info`
        The generated infos for feeding into the kernel generator

    """

    return lockstep_driver(loopy_opts, namestore, inputs, outputs, driven,
                           test_size=test_size)
