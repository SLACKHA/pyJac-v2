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
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.loopy_utils import preambles_and_manglers as lp_pregen


driver_index = lp.ValueArg('driver_index', dtype=arc.kint_type)
"""
The index to be used as an offset in the driver loop
"""


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

    del array_names

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
                    logger.debug('{} array for driver kernel {} does not '
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
            logger.debug('No {} arrays supplied to driver that require '
                         'copying to working buffer!'.format(desc))
            raise InvalidInputSpecificationException('Driver ' + desc + ' arrays')
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
        global_indicies[0] += ' + ' + driver_index.name

        # bake in SIMD pre-split
        vec_spec = None
        split_spec = None
        conditional_index = global_indicies[0]
        if loopy_opts.pre_split:
            conditional_index = '({} + {})'.format(
                indicies[0] + '_outer', driver_index.name, indicies[0] + '_inner')
            if loopy_opts.is_simd:
                # need put dependence of vector lane in
                extra_inames.append(('lane', '0 <= lane < {}'.format(
                    loopy_opts.vector_width)))
                global_indicies[0] += ' + lane'
                conditional_index = '({} + {}) + lane'.format(
                    indicies[0] + '_outer',
                    driver_index.name)

                def vectorization_specializer(knl):
                    # first, unroll lane
                    knl = lp.tag_inames(knl, {'lane': 'unr'})
                    return knl

                def split_specializer(knl):
                    # drop the vector iname and do a pure unroll
                    knl = lp.rename_iname(knl, indicies[0] + '_inner', 'lane',
                                          existing_ok=True)
                    priorities = set(list(knl.loop_priority)[0]) - \
                        set([indicies[0] + '_inner'])
                    priorities = set([tuple(priorities)])
                    return knl.copy(loop_priority=priorities)

                vec_spec = vectorization_specializer
                split_spec = split_specializer

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
            if ${ind} < ${problem_size}
                ${local_buffer} = ${global_buffer} {id=copy_${name}}
            end
        """) if for_input else Template("""
            if ${ind} < ${problem_size}
                ${global_buffer} = ${local_buffer} {id=copy_${name}}
            end
        """)

        warnings = []
        instructions = []
        for i, arr in enumerate(arrs):
            instructions.append(instruction_template.substitute(
                local_buffer=working_strs[i],
                global_buffer=strs[i],
                ind=conditional_index,
                problem_size=arc.problem_size.name,
                name=arr.name))
            warnings.append('write_race(copy_{})'.format(arr.name))
        if loopy_opts.is_simd:
            warnings.append('vectorize_failed')
            warnings.append('unrolled_vector_iname_conditional')
        instructions = '\n'.join(instructions)

        # and return the kernel info
        return k_gen.knl_info(name=name,
                              instructions=instructions,
                              mapstore=mapstore,
                              var_name=arc.var_name,
                              extra_inames=extra_inames,
                              kernel_data=buffers + working_buffers + [
                                arc.work_size, arc.problem_size, driver_index],
                              silenced_warnings=warnings,
                              vectorization_specializer=vec_spec,
                              split_specializer=split_spec)

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

    func_call = k_gen.knl_info(name='driver',
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
        #pragma omp parallel for
        ${unpacks}
        for (${dtype} ${driver_index} = 0; ${driver_index} < ${problem_size}; ${driver_index} += ${work_size})"""  # noqa
        """
        {
            ${insns}
        }
        """)

    elif loopy_opts.lang == 'opencl':
        template = Template("""
        #if defined(WIDE) && !defined(EXPLICIT_SIMD)
            // each group processes get_global_size(0) condtions
            #define inc (get_global_size(0))
            ${dtype} driver_index = get_global_id(0);
        #elif defined(WIDE) && defined(EXPLICIT_SIMD)
            // each group processes VECWIDTH condtions
            #define inc (VECWIDTH * get_num_groups(0))
            ${dtype} driver_index = get_global_id(0);
        #else
            // each group processes a single condtion
            #define inc (get_num_groups(0))
            ${dtype} driver_index = get_group_id(0);
        #endif
        ${unpacks}
        for (;${driver_index} < ${problem_size}; ${driver_index} += inc)"""  # noqa
        """
        {
            ${insns}
        }
        """)

    from loopy.types import to_loopy_type
    return template.safe_substitute(
        dtype=driven.type_map[to_loopy_type(arc.kint_type)],
        driver_index=driver_index.name,
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
