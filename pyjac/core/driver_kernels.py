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
import loopy as lp
from pytools import UniqueNameGenerator

from pyjac.utils import listify, stringify_args
from pyjac.core.exceptions import InvalidInputSpecificationException
from pyjac.core import array_creator as arc
from pyjac.kernel_utils import knl_info
from pyjac.loopy_utils import preambles_and_manglers as lp_pregen


driver_offset = lp.ValueArg('driver_offset', dtype=arc.kint_type)
"""
The offset to be used in the driver loop
"""


def get_problem_index(loopy_opts, global_ind=arc.global_ind, lane_iname='lane'):
    """
    Returns the correct unique identifier for a conditional expression limiting
    the problem size for driver copies
    """

    conditional_index = global_ind + ' + ' + driver_offset.name
    if loopy_opts.pre_split:
        conditional_index = '({} * {} + {} + {})'.format(
            global_ind + '_outer',
            loopy_opts.vector_width,
            global_ind + '_inner',
            driver_offset.name)

    return conditional_index


def get_driver(loopy_opts, namestore, inputs, outputs, driven,
               test_size=None):
    """
    Implements a driver function for kernel evaluation.
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
    inputs: list of :class:`lp.KernelArgument`
        The arrays that should be copied into internal working buffers
        before calling subfunctions
    outputs: list of :class:`lp.KernelArgument`
        The arrays should be copied back into global memory after calling
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
    # create mapping of array names
    array_names = {v.name: v for k, v in six.iteritems(vars(namestore))
                   if isinstance(v, arc.creator) and not (
                    v.fixed_indicies or v.affine)}
    for arr in to_find:
        arr_creator = next((array_names[x] for x in array_names if x == arr), None)
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

        def _raise(desc, inp, nameref, shape):
            logger = logging.getLogger(__name__)
            logger.debug('{} array for driver kernel {} does not '
                         'match expected shape (from array {}).  '
                         'Expected: ({}), got: ({})'.format(
                            desc, inp.name, nameref,
                            stringify_args(inp.shape),
                            stringify_args(shape))
                         )
            raise InvalidInputSpecificationException(inp.name)

        nameref = None
        desc = 'Input' if check_input else 'Output'
        for inp in [arrays[x] for x in (inputs if check_input else outputs)]:
            if not arr_non_ic(inp):
                # only the initial condition dimension, fine
                continue
            if shape:
                if inp.shape != shape and len(inp.shape) == len(shape):
                    # allow different shapes in the last index
                    if not all(x == y for x, y in zip(*(
                            inp.shape[:-1], shape[:-1]))):
                        _raise(desc, inp, nameref, shape)
                    # otherwise, take the maximum of the shape entry
                    shape = shape[:-1] + (max(shape[-1], inp.shape[-1]),)

                elif inp.shape != shape:
                    _raise(desc, inp, nameref, shape)
            else:
                nameref = inp.name
                shape = inp.shape[:]
        if not shape:
            logger = logging.getLogger(__name__)
            logger.debug('No {} arrays supplied to driver that require '
                         'copying to working buffer!'.format(desc))
            raise InvalidInputSpecificationException('Driver ' + desc + ' arrays')
        return shape

    def create_interior_kernel(for_input):
        shape = __check(for_input)
        name = 'copy_{}'.format('in' if for_input else 'out')
        # get arrays
        arrs = [arrays[x] for x in (inputs if for_input else outputs)]

        # create a dummy map and store
        map_shape = np.arange(shape[1], dtype=arc.kint_type)
        mapper = arc.creator(name, arc.kint_type, map_shape.shape, 'C',
                             initializer=map_shape)
        mapstore = arc.MapStore(loopy_opts, mapper, test_size)

        # determine what other inames we need, if any
        namer = UniqueNameGenerator(set([mapstore.iname]))
        extra_inames = []
        for i in six.moves.range(2, len(shape)):
            iname = namer(mapstore.iname)
            extra_inames.append((iname, '0 <= {} < {}'.format(
                iname, shape[i])))

        indicies = [arc.global_ind, mapstore.iname] + [
            ex[0] for ex in extra_inames]
        global_indicies = indicies[:]
        global_indicies[0] += ' + ' + driver_offset.name

        # bake in SIMD pre-split
        vec_spec = None
        split_spec = None
        conditional_index = get_problem_index(loopy_opts)

        def __build(arr, local, **kwargs):
            inds = global_indicies if not local else indicies
            if isinstance(arr, arc.jac_creator) and arr.is_sparse:
                # this is a sparse Jacobian, hence we have to override the default
                # indexing (as we're doing a straight copy)
                kwargs['ignore_lookups'] = True
            if arr_non_ic(arr):
                return mapstore.apply_maps(arr, *inds, **kwargs)
            else:
                return mapstore.apply_maps(arr, inds[0], **kwargs)

        # create working buffer version of arrays
        working_buffers = []
        working_strs = []
        for arr in arrs:
            arr_lp, arr_str = __build(arr, True, use_local_name=True)
            working_buffers.append(arr_lp)
            working_strs.append(arr_str)

        # create global versions of arrays
        buffers = []
        strs = []
        for arr in arrs:
            arr_lp, arr_str = __build(arr, False, reshape_to_working_buffer=False)
            buffers.append(arr_lp)
            strs.append(arr_str)

        # now create the instructions
        instruction_template = Template("""
            if ${ind} < ${problem_size} ${shape_check}
                ${local_buffer} = ${global_buffer} {id=copy_${name}}
            end
        """) if for_input else Template("""
            if ${ind} < ${problem_size} ${shape_check}
                ${global_buffer} = ${local_buffer} {id=copy_${name}}
            end
        """)

        warnings = []
        instructions = []
        for i, arr in enumerate(arrs):
            # get shape check
            shape_check = ''
            if arr.shape[-1] != shape[-1] and len(arr.shape) == len(shape):
                shape_check = ' and {} < {}'.format(
                    indicies[-1], arr.shape[-1])

            instructions.append(instruction_template.substitute(
                local_buffer=working_strs[i],
                global_buffer=strs[i],
                ind=conditional_index,
                problem_size=arc.problem_size.name,
                name=arr.name,
                shape_check=shape_check))
            warnings.append('write_race(copy_{})'.format(arr.name))
        if loopy_opts.is_simd:
            warnings.append('vectorize_failed')
            warnings.append('unrolled_vector_iname_conditional')
        instructions = '\n'.join(instructions)

        kwargs = {}
        if loopy_opts.lang == 'c':
            # override the number of copies in this function to 1
            # (i.e., 1 per-thread)
            kwargs['iname_domain_override'] = [(arc.global_ind, '0 <= {} < 1'.format(
                arc.global_ind))]

        priorities = ([arc.global_ind + '_outer'] if loopy_opts.pre_split else [
            arc.global_ind]) + [arc.var_name]
        # and return the kernel info
        return knl_info(name=name,
                        instructions=instructions,
                        mapstore=mapstore,
                        var_name=arc.var_name,
                        extra_inames=extra_inames,
                        kernel_data=buffers + working_buffers + [
                          arc.work_size, arc.problem_size, driver_offset],
                        silenced_warnings=warnings,
                        vectorization_specializer=vec_spec,
                        split_specializer=split_spec,
                        unrolled_vector=True,
                        loop_priority=set([tuple(priorities + [
                          iname[0] for iname in extra_inames])]),
                        **kwargs)

    copy_in = create_interior_kernel(True)
    # create a dummy kernel info that simply calls our internal function
    instructions = driven.name + '()'
    # create mapstore
    call_name = driven.name
    repeats = 1
    if loopy_opts.depth:
        # we need 'var_name' to have a non-unity size
        repeats = loopy_opts.vector_width

    map_shape = np.arange(repeats, dtype=arc.kint_type)
    mapper = arc.creator(call_name, arc.kint_type, map_shape.shape, 'C',
                         initializer=map_shape)
    mapstore = arc.MapStore(loopy_opts, mapper, test_size)
    mangler = lp_pregen.MangleGen(call_name, tuple(), tuple())
    kwargs = {}
    if loopy_opts.lang == 'c':
        # override the number of calls to the driven function in the driver, this
        # is currently fixed to 1 (i.e., 1 per-thread)
        kwargs['iname_domain_override'] = [(arc.global_ind, '0 <= {} < 1'.format(
            arc.global_ind))]

    func_call = knl_info(name='driver',
                         instructions=instructions,
                         mapstore=mapstore,
                         kernel_data=[arc.work_size, arc.problem_size],
                         var_name=arc.var_name,
                         extra_inames=copy_in.extra_inames[:],
                         manglers=[mangler],
                         **kwargs)
    copy_out = create_interior_kernel(False)

    # and return
    return [copy_in, func_call, copy_out]


def lockstep_driver_template(loopy_opts, driven):
    """
    Returns the appropriate template for a lockstep-based driver function for
    kernel evaluation.

    Parameters
    ----------
    loopy_opts: :class:`LoopyOptions`
        The kernel creation options
    driven: :class:`kernel_generator`
        The kernel to be driven

    Returns
    -------
    template: str
        The template to wrap the driver function in, with keyword insns
    """

    if loopy_opts.lang == 'c':
        template = Template("""
        #pragma omp parallel
        {
            // note: work_size unpacking _must_ be done in a parallel section in
            // order to get correct values from omp_get_num_threads()
            ${unpacks}
            #pragma omp for
            for (${dtype} ${driver_offset} = 0; ${driver_offset} < ${problem_size}; ${driver_offset} += 1)"""  # noqa
            """
            {
                ${insns}
            }
        }
        """)

    elif loopy_opts.lang == 'opencl':
        template = Template("""
        #if defined(WIDE) && !defined(EXPLICIT_SIMD)
            // each group processes get_global_size(0) condtions
            #define inc (get_global_size(0))
        #elif defined(WIDE) && defined(EXPLICIT_SIMD)
            // each group processes VECWIDTH condtions
            #define inc (VECWIDTH * get_global_size(0))
        #else
            // each group processes a single condtion
            #define inc (get_num_groups(0))
        #endif
        ${unpacks}
        for (${dtype} driver_offset = 0; ${driver_offset} < ${problem_size}; ${driver_offset} += inc)"""  # noqa
        """
        {
            ${insns}
        }
        """)

    from loopy.types import to_loopy_type
    return template.safe_substitute(
        dtype=driven.type_map[to_loopy_type(arc.kint_type)],
        driver_offset=driver_offset.name,
        problem_size=arc.problem_size.name,
        work_size=arc.work_size.name)


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

    raise NotImplementedError
