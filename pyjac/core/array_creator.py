# -*- coding: utf-8 -*-
"""Contains various utility classes for creating loopy arrays
and indexing / mapping
"""

import logging
import six
import copy
from string import Template

import loopy as lp
import numpy as np
from loopy.kernel.data import AddressSpace as scopes

from pyjac.core.enum_types import JacobianFormat, JacobianType
from pyjac.utils import listify, partition


class array_splitter(object):
    """
    A convenience object that handles splitting arrays to improve vectorized
    data-access patterns, etc.

    Can handle reshaping of both loopy and numpy arrays to the desired shape

    Properties
    ----------
    depth: int [None]
        If is not None, the vector-width to use for deep-vectorization
    wide: bool [False]
        If is not None, the vector-width to use for wide-vectorization
    data_order: ['C', 'F']
        The data ordering of the kernel
    """

    def __init__(self, loopy_opts):
        self.depth = loopy_opts.depth
        self.width = loopy_opts.width
        self.vector_width = loopy_opts.vector_width
        self.data_order = loopy_opts.order
        self.is_simd = loopy_opts.is_simd
        self.pre_split = None
        if loopy_opts.pre_split:
            if self.width:
                self.pre_split = global_ind
            else:
                self.pre_split = var_name

    @property
    def _is_simd_split(self):
        """
        Returns True IFF this array splitter has a split, and this split is a result
        of use of explicit SIMD data types
        """

        return self._have_split() and not self._have_split(with_simd_as=False)

    @staticmethod
    def __determine_split(data_order, vector_width, width, depth, is_simd):
        """
        The internal workings of the :func:`_have_split` methods, consolodated.
        Not intended to be called directly
        """

        if vector_width:
            if is_simd:
                return True
            return ((data_order == 'C' and width) or (data_order == 'F' and depth))
        return False

    def _have_split(self, with_simd_as=None):
        """
        Returns True if there is anything for this :class:`array_splitter` to do

        Parameters
        ----------
        with_simd_as: bool [None]
            If specified, calculate whether we have a split if :attr:`is_simd` was
            set to the given value

        Returns
        -------
        have_split: bool
            True IFF this vectorization pattern will result in a split for any
            array
        """

        is_simd = self.is_simd if with_simd_as is None else with_simd_as
        return self.__determine_split(
            self.data_order, self.vector_width, self.width, self.depth, is_simd)

    @staticmethod
    def _have_split_static(loopy_opts, with_simd_as=None):
        """
        Like :func:`_have_split`, but a static method for easy calling

        Parameters
        ----------
        loopy_opts: :class:`loopy_options`
            The options object that would be used to construct this splitter.
        with_simd_as: bool [None]
            If specified, calculate whether we have a split if :attr:`is_simd` was
            set to the given value

        Returns
        -------
        have_split: bool
            True IFF this vectorization pattern will result in a split for any
            array

        """

        is_simd = loopy_opts.is_simd if with_simd_as is None else with_simd_as
        return array_splitter.__determine_split(
            loopy_opts.order, loopy_opts.vector_width, loopy_opts.width,
            loopy_opts.depth, is_simd)

    def _should_split(self, array):
        """
        Returns True IFF this array should result in a split, the current criteria
        are:

        1) the array's at least 2-D
        2) we have a vector width, and we're using explicit SIMD

        Note: this assumes that :func:`_have_split` has been called previously (
        and passed)

        """

        return len(array.shape) >= 1 or self.is_simd

    def split_and_vec_axes(self, array):
        """
        Determine the axis that should be split, and the destination of the vector
        axis for the given array and :attr:`data_order`, :attr:`width` and
        :attr:`depth`

        Parameters
        ----------
        array: :class:`numpy.ndarray` or :class:`loopy.ArrayBase`
            The array to split

        Notes
        -----
        Does not take into account whether the array should be split, see:
            :func:`have_split`, :func:`should_split`

        Returns
        -------
        split_axis:
            the axis index to be split
        vec_axis:
            the destination index of the vector axis
        """

        if self.data_order == 'C' and self.width:
            split_axis = 0
            vec_axis = len(array.shape)
        elif self.data_order == 'C':
            split_axis = len(array.shape) - 1
            vec_axis = len(array.shape)
        elif self.data_order == 'F' and self.depth:
            split_axis = len(array.shape) - 1
            vec_axis = 0
        else:
            split_axis = 0
            vec_axis = 0

        return split_axis, vec_axis

    def grow_axis(self, array):
        """
        Returns
        -------
        grow_axis: int
            The integer index of the axis corresponding to the initial conditions,
            see :ref:`vector_split`
        """
        if self._should_split(array):
            # the grow axis is one IFF it's a F-ordered deep-split
            return 1 if self.data_order == 'F' else 0
        return 0

    def split_shape(self, array):
        """
        Returns the array shape that would result from splitting the supplied array

        Parameters
        ----------
        array: :class:`numpy.ndarray` (or object w/ attribute shape)

        Returns
        -------
        shape: tuple of int
            The resulting split array shape
        grow_axis: int
            The integer index of the axis corresponding to the initial conditions,
            see :ref:`vector_split`
        vec_axis: int
            The integer index of the vector axis, if present.
            If there is no split, this will be None, see :ref:`vector_split`
        split_axis: int
            The integer index of the axis that will be split (note: this is
            calculated _before_ the split is applied). If no split, this is None
        """

        grow_axis = 0
        vec_axis = None
        shape = tuple(x for x in array.shape)
        split_axis = None
        if not self._have_split():
            return shape, grow_axis, vec_axis, split_axis

        vector_width = None
        if self._should_split(array):
            # the grow axis is one IFF it's a F-ordered deep-split
            grow_axis = self.grow_axis(array)
            split_axis, vec_axis = self.split_and_vec_axes(array)
            vector_width = self.depth if self.depth else self.width
            assert vector_width
            new_shape = [-1] * (len(shape) + 1)

            # the vector axis is of size vector_width
            new_shape[vec_axis] = vector_width

            def __index(i):
                if i < vec_axis:
                    return i
                else:
                    return i + 1

            for i in range(len(shape)):
                if i == split_axis:
                    # the axis that is split is divided by the vector width
                    new_shape[__index(i)] = int(
                        np.ceil(shape[i] / float(vector_width)))
                else:
                    # copy old shape
                    new_shape[__index(i)] = shape[i]

            shape = tuple(new_shape)

        return shape, grow_axis, vec_axis, split_axis

    def _split_array_axis_inner(self, kernel, array_name, split_axis, dest_axis,
                                count, order='C', vec=False, **kwargs):
        if count == 1:
            return kernel

        # {{{ adjust arrays

        from loopy.kernel.tools import ArrayChanger
        from loopy.symbolic import SubstitutionRuleMappingContext
        from loopy.transform.padding import ArrayAxisSplitHelper

        achng = ArrayChanger(kernel, array_name)
        ary = achng.get()

        from pytools import div_ceil

        # {{{ adjust shape

        new_shape = ary.shape
        assert new_shape is not None, 'Cannot split auto-sized arrays'
        new_shape = list(new_shape)
        axis_len = new_shape[split_axis]
        # still need to reduce the global problem size in unit testing
        if self.pre_split and not isinstance(axis_len, int):
            outer_len = new_shape[split_axis]
        else:
            # todo: automatic detection of when axis_len % count == 0, and we can
            # replace with axis_len >> log2(count)
            outer_len = div_ceil(axis_len, count)

        new_shape[split_axis] = outer_len
        new_shape.insert(dest_axis, count)
        new_shape = tuple(new_shape)

        # }}}

        # {{{ adjust dim tags

        if ary.dim_tags is None:
            raise RuntimeError("dim_tags of '%s' are not known" % array_name)
        new_dim_tags = list(ary.dim_tags)

        old_dim_tag = ary.dim_tags[split_axis]

        from loopy.kernel.array import FixedStrideArrayDimTag
        if not isinstance(old_dim_tag, FixedStrideArrayDimTag):
            raise RuntimeError("axis %d of '%s' is not tagged fixed-stride".format(
                split_axis, array_name))

        tag = FixedStrideArrayDimTag(1)
        new_dim_tags.insert(dest_axis, tag)
        # fix strides
        toiter = reversed(list(enumerate(new_shape))) if order == 'C' \
            else enumerate(new_shape)

        stride = 1
        for i, shape in toiter:
            new_dim_tags[i] = new_dim_tags[i].copy(stride=stride)
            stride *= shape

        new_dim_tags = tuple(new_dim_tags)

        # }}}

        # {{{ adjust dim_names

        new_dim_names = ary.dim_names
        if new_dim_names is not None:
            new_dim_names = list(new_dim_names)
            existing_name = new_dim_names[split_axis]
            outer_name = existing_name + "_outer"
            new_dim_names[split_axis] = outer_name
            new_dim_names.insert(dest_axis, existing_name + "_inner")
            new_dim_names = tuple(new_dim_names)

        # }}}

        kernel = achng.with_changed_array(ary.copy(
            shape=new_shape, dim_tags=new_dim_tags, dim_names=new_dim_names))

        # }}}

        var_name_gen = kernel.get_var_name_generator()

        def split_access_axis(expr):
            idx = expr.index
            if not isinstance(idx, tuple):
                idx = (idx,)
            idx = list(idx)

            axis_idx = idx[split_axis]

            from loopy.symbolic import simplify_using_aff
            from pymbolic.primitives import Variable
            if self.pre_split:
                # no split, just add an axis
                outer_index = axis_idx
                inner_index = Variable(self.pre_split + '_inner')
            else:
                inner_index = simplify_using_aff(kernel, axis_idx % count)
                outer_index = simplify_using_aff(kernel, axis_idx // count)

            idx[split_axis] = outer_index
            idx.insert(dest_axis, inner_index)
            return expr.aggregate.index(tuple(idx))

        rule_mapping_context = SubstitutionRuleMappingContext(
                kernel.substitutions, var_name_gen)
        aash = ArrayAxisSplitHelper(rule_mapping_context,
                                    set([array_name]), split_access_axis)
        kernel = rule_mapping_context.finish_kernel(aash.map_kernel(kernel))

        if vec:
            achng = ArrayChanger(kernel, array_name)
            new_strides = [t.layout_nesting_level for t in achng.get().dim_tags]
            tag = ['N{}'.format(s) if i != dest_axis else 'vec'
                   for i, s in enumerate(new_strides)]
            kernel = lp.tag_array_axes(kernel, [array_name], tag)

        return kernel

    def __split_iname_access(self, knl, arrays):
        """
        Warning -- should be called _only_ on non-split arrays when :attr:`pre_split`
        is True

        This is a helper array to ensure that indexing of non-split arrays
        stay correct if we're using a pre-split.  Essentially, if we don't split
        the arrays, the pre-split index will never make it into the array, and hence
        the assumed iname splitting won't work.

        Parameters
        ----------
        knl: :class:`loopy.LoopKernel`
            The kernel to split iname access for
        arrays: :list of class:`loopy.ArrayArg`
            The array(s) to split iname access for

        Returns
        -------
        split_knl: :class:`loopy.LoopKernel`
            The kernel with iname access to the array's split

        Raises
        ------
        AssertionError
        """

        assert self.pre_split
        owner = self
        new_var = self.pre_split + '_inner'

        from pymbolic import var, substitute
        from pymbolic.mapper import IdentityMapper
        from loopy.symbolic import get_dependencies

        names = set(arrays)

        class SubstMapper(IdentityMapper):
            def map_subscript(self, expr, *args, **kwargs):
                if expr.aggregate.name in names:
                    # get old index
                    old = var(owner.pre_split + '_outer')
                    new = var(owner.pre_split + '_outer') * owner.vector_width + \
                        var(new_var)
                    expr.index = substitute(expr.index, {old: new})

                return super(SubstMapper, self).map_subscript(
                    expr, *args, **kwargs)

        insns = []
        mapper = SubstMapper()
        for insn in knl.instructions:
            try:
                if get_dependencies(insn.assignee) & names:
                    insn = insn.copy(assignee=mapper(insn.assignee),
                                     within_inames=insn.within_inames | set([
                                        new_var]))
                if get_dependencies(insn.expression) & names:
                    insn = insn.copy(expression=mapper(insn.expression),
                                     within_inames=insn.within_inames | set([
                                        new_var]))
            except AttributeError:
                pass
            insns.append(insn)

        return knl.copy(instructions=insns)

    def split_loopy_arrays(self, kernel, dont_split=[], **kwargs):
        """
        Splits the :class:`loopy.GlobalArg`'s that form the given kernel's arguements
        to conform to this split pattern

        Parameters
        ----------
        kernel : `loopy.LoopKernel`
            The kernel to apply the splits to
        dont_split: list of str
            List of array names that should not be split (typically representing
            global inputs / outputs)

        Keyword Arguments
        -----------------
        ignore_split_rename_errors: bool [False]
            If True, ignore errors that would result from the vector index not being
            the pre-split index, as expected.  Used in testing.

        Returns
        -------
        split_kernel : `loopy.LoopKernel`
            The kernel with the array splittings applied
        """

        if not self._have_split():
            return kernel

        arrays = [(x.name, x) for x in kernel.args
                  if isinstance(x, lp.ArrayArg)
                  and x.name not in dont_split]

        to_split, not_to_split = partition(
            arrays, lambda x: self._should_split(x[1]))

        # add to don't split list for iname access handling
        dont_split += [x[0] for x in not_to_split]
        if self.pre_split and dont_split:
            # we still have to split potential iname accesses in this array
            # to maintain correctness
            kernel = self.__split_iname_access(kernel, dont_split)

        for array_name, arr in to_split:
            split_axis, vec_axis = self.split_and_vec_axes(arr)
            kernel = self._split_array_axis_inner(
                kernel, array_name, split_axis, vec_axis,
                self.vector_width, self.data_order, self.is_simd,
                **kwargs)

        return kernel

    def _split_numpy_array(self, input_array):
        """
        Spits the supplied numpy array according to desired pattern

        Parameters
        ----------
        input_array : :class:`numpy.ndarray`
            The input array to split

        Returns
        -------
        output : :class:`numpy.ndarray`
            The properly split / resized numpy array
        """

        if not self._have_split() or not self._should_split(input_array):
            return input_array

        def _split_and_pad(arr, axis, width, dest_axis):
            # get the last split as the ceiling
            end = np.ceil(arr.shape[axis] / width) * width
            # create split indicies
            indicies = np.arange(width, end + 1, width, dtype=kint_type)
            # split array
            arr = np.split(arr, indicies, axis=axis)
            # filter out empties
            arr = [a for a in arr if a.size]
            # check for pad
            if arr[-1].shape[axis] != width:
                pads = [(0, 0) for x in arr[-1].shape]
                pads[axis] = (0, width - arr[-1].shape[axis])
                arr[-1] = np.pad(arr[-1], pads, 'constant')
            # get joined
            arr = np.stack(arr, axis=axis)
            # and move array axes
            # the created axis is at axis + 1, and should be moved to
            # the destination
            return np.moveaxis(arr, axis + 1, dest_axis).copy(order=self.data_order)

        # figure out split
        split_axis, vec_axis = self.split_and_vec_axes(input_array)
        return _split_and_pad(input_array, split_axis, self.vector_width,
                              vec_axis)

    def split_numpy_arrays(self, arrays):
        """
        Splits the provided numpy arrays

        See :func:`_split_numpy_array`

        Parameters
        ----------
        arrays: list of :class:`numpy.ndarray`
            The arrays to split

        Returns
        -------
        out_arrays: list of :class:`numpy.ndarray`
            The split arrays
        """

        if isinstance(arrays, np.ndarray):
            arrays = [arrays]
        elif isinstance(arrays, dict):
            return {k: self._split_numpy_array(v) for k, v in six.iteritems(arrays)}

        return [self._split_numpy_array(a) for a in arrays]


kint_type = np.int32
"""
    The integer type to use for kernel indicies, eventually I'd like to make this
    user-specifiable
"""


problem_size = lp.ValueArg('problem_size', dtype=kint_type)
"""
    The problem size variable for generated kernels, describes the size of
    input/output arrays used in drivers
"""

work_size = lp.ValueArg('work_size', dtype=kint_type)
"""
    The global work size of the generated kernel.
    Roughly speaking, this corresponds to:
        - The number of cores to utilize on the CPU
        - The number of blocks to launch on the GPU
    This may be specified at run-time or during kernel-generation.
"""

local_name_suffix = '_local'
"""
The suffix to append to 'local' versions of arrays (i.e., those in the working buffer
in use in the driver)
"""


global_ind = 'j'
"""str: The global initial condition index

This is the string index for the global condition loop in generated kernels
of :module:`rate_subs`
"""


var_name = 'i'
"""str: The inner loop index

This is the string index for the inner loops in generated kernels of
:module:`rate_subs`
"""


default_inds = (global_ind, var_name)
"""str: The default indicies used in main loops of :module:`rate_subs`

This is the string indicies for the main loops for generated kernels in
:module:`rate_subs`
"""


def initial_condition_dimension_vars(loopy_opts, test_size, is_driver_kernel=False):
    """
    Return the size to use for the initial condition dimension, considering whether
    we're in unit-testing, a driver kernel or simple kernel generation

    Parameters
    ----------
    loopy_opts: :class:`loopy_options`
        The loopy options to be used during kernel creation
    test_size: int or None
        The test size option, indicates whether we're in unit-testing
    is_driver_kernel: bool [False]
        If True, and not-unit testing, use the _full_ `problem_size` for the
        IC dim. size

    Returns
    -------
    size: list of :class:`loopy.ValueArg`
        The initial condition dimension size variables
    """

    if isinstance(test_size, int) or loopy_opts.unique_pointers:
        return []
    if is_driver_kernel:
        return [work_size, problem_size]
    else:
        return [work_size]


class tree_node(object):
    """
        A node in the :class:`MapStore`'s domain tree.
        Contains a base domain, a list of child domains, and a list of
        variables depending on this domain

    Parameters
    ----------
    owner: :class:`MapStore`
        The owning mapstore, used for iname creation
    parent : :class:`tree_node`
        The parent domain of this tree node
    domain : :class:`creator`
        The domain this :class:`tree_node` represents
    children : list of :class:`tree_node`
        The leaves of this node
    iname : str
        The iname of this :class:`tree_node`
    """

    def __add_to_owner(self, domain):
        assert domain not in self.owner.domain_to_nodes, (
            'Domain {} is already present in the tree!'.format(
                domain.name))
        self.owner.domain_to_nodes[self.domain] = self

    def __init__(self, owner, domain, children=[],
                 parent=None, iname=None):
        self.domain = domain
        self.owner = owner
        try:
            self.children = set(children)
        except:
            self.children = set([children])
        self.parent = parent
        self.transform = None
        self._iname = iname

        self.insn = None
        self.domain_transform = None

        # book keeping
        self.__add_to_owner(domain)
        for child in self.children:
            self.__add_to_owner(child)

    def is_leaf(self):
        return not self.children and self != self.owner.tree

    @property
    def iname(self):
        return self._iname

    @iname.setter
    def iname(self, value):
        if value is None:
            assert self.parent is not None, (
                "Can't set empty (untransformed) iname for root!")
            self._iname = self.parent.iname
        else:
            self._iname = value

    @property
    def name(self):
        return self.domain.name

    def set_transform(self, iname, insn, domain_transform):
        self.iname = iname
        self.insn = insn
        self.domain_transform = domain_transform

    def add_child(self, domain):
        """
        Adds a child domain (if not already present) to this node

        Parameters
        ----------
        domain : :class:`creator`
            The domain to create a the child node with

        Returns
        -------
        child : :class:`tree_node`
            The newly created tree node
        """

        # check for existing child
        child = next((x for x in self.children if x.domain == domain), None)

        if child is None:
            child = tree_node(self.owner, domain, parent=self)
            self.children.add(child)
            self.__add_to_owner(child)

        return child

    def has_children(self, arrays):
        """
        Checks whether an array is present in this :class:`tree_node`'s children

        Parameters
        ----------
        arrays: str, or :class:`creator`, or list of str/:class:`creator`
            The arrays to check for

        Returns
        -------
        present: list of bool
            True if array is present in children
        """

        arrays = [ary.name if isinstance(ary, creator) else ary
                  for ary in listify(arrays)]
        assert all(isinstance(x, str) for x in arrays)

        present = []
        for ary in arrays:
            child = next((x for x in self.children if x.name == ary), None)
            present.append(bool(child))
        return present

    def __repr__(self):
        return ', '.join(['{}'.format(x) for x in
                          (self.domain.name, self.iname, self.insn)])


def search_tree(root, arrays):
    """
    Searches the tree from the root supplied to see which tree node's (if any)
    the list of arrays are children of

    Parameters
    ----------
    root: :class:`tree_node`
        The root of the tree to search
    arrays: str, or :class:`creator`, or list of str/:class:`creator`
        The arrays to search for

    Returns
    -------
    parents: list of :class:`tree_node`
        A list of tree node's who are the parents of the supplied arrays
        If the array is not found, then the corresponding parent will be None
    """

    assert isinstance(root, tree_node), (
        "Can't start tree search from type {}, an instance of :class:`tree_node` "
        "expected.")
    # first, check the root level
    parents = root.has_children(arrays)
    parents = [root if p else None for p in parents]

    # now call recursively for the not-found arrays
    missing_inds, missing = list(zip(*enumerate(arrays)))
    for child in [c for c in root.children if isinstance(c, tree_node)]:
        found = search_tree(child, missing)
        for i, f in enumerate(found):
            if f is not None:
                parents[missing_inds[i]] = f

    return parents


class domain_transform(object):

    """
    Simple helper class to keep track of transform variables to test for
    equality
    """

    def __init__(self, mapping, affine):
        self.mapping = mapping
        self.affine = affine

    def __eq__(self, other):
        return self.mapping == other.mapping and self.affine == other.affine


class MapStore(object):

    """
    This class manages maps and masks for inputs / outputs in kernels

    Attributes
    ----------

    loopy_opts : :class:`LoopyOptions`
        The loopy options for kernel creation
    map_domain : :class:`creator`
        The domain of the iname to use for a mapped kernel
    is_unit_test : bool
        If true, we are generating arrays for a unit-test, and hence we should not
        convert anything to working-buffer form
    iname : str
        The loop index to work with
    have_input_map : bool
        If true, the input map domain needs a map for expression
    transformed_domains : set of :class:`tree_node`
        The nodes that have required transforms
    transform_insns : set
        A set of transform instructions generated for this :class:`MapStore`
    raise_on_final : bool
        If true, raise an exception if a variable / domain is added to the
        domain tree after this :class:`MapStore` has been finalized
    working_buffer_index : bool
        If True, use internal buffers for OpenCL/CUDA/etc. array creation
        where possible. If False, use full sized arrays.
    """

    def __init__(self, loopy_opts, map_domain, test_size, iname='i',
                 raise_on_final=True):
        self.loopy_opts = loopy_opts
        self.map_domain = map_domain
        self._check_is_valid_domain(self.map_domain)
        self.domain_to_nodes = {}
        self.is_unit_test = isinstance(test_size, int)
        self.transformed_domains = set()
        self.tree = tree_node(self, self._get_base_domain(), iname=iname)
        self.domain_to_nodes[self._get_base_domain()] = self.tree
        from pytools import UniqueNameGenerator
        self.taken_transform_names = UniqueNameGenerator(set([iname]))
        self.iname = iname
        self.have_input_map = False
        self.raise_on_final = raise_on_final
        self.is_finalized = False

        self.working_buffer_index = None
        self.reshape_to_working_buffer = False
        # if we're using a pre-split, find the WBI
        if self.loopy_opts.pre_split:
            from pyjac.kernel_utils.kernel_gen import kernel_generator
            specialization = kernel_generator.apply_specialization(
                self.loopy_opts, var_name, None,
                get_specialization=True)
            global_index = next((k for k, v in six.iteritems(specialization)
                                 if v == 'g.0'), global_ind)
            self.working_buffer_index = global_index

        # unit tests operate on the whole array
        self.initial_condition_dimension = None
        if not self.is_unit_test:
            self.initial_condition_dimension = \
                self.loopy_opts.initial_condition_dimsize

    def _is_map(self):
        """
        Return true if map kernel
        """

        return True

    @property
    def transform_insns(self):
        return set(val.insn for val in
                   self.transformed_domains if val.insn)

    def _add_input_map(self):
        """
        Adds an input map, and remakes the base domain
        """

        # copy the base map domain
        new_creator_name = self.map_domain.name + '_map'
        new_map_domain = self.map_domain.copy()

        # update new domain
        new_map_domain.name = new_creator_name
        new_map_domain.initializer = \
            np.arange(self.map_domain.initializer.size, dtype=kint_type)

        # change the base of the tree
        new_base = tree_node(self, new_map_domain, iname=self.iname,
                             children=self.tree)

        # and parent

        # to maintain consistency/sanity, we always consider the original tree
        # the 'base', even if the true base is being replaced
        # This will be accounted for in :meth:`finalize`
        self.tree.parent = new_base

        # reset iname
        self.tree.iname = None

        # update domain
        self.map_domain = new_map_domain

        # and finally set tree
        self.domain_to_nodes[new_map_domain] = new_base

        # finally, check the tree's offspring.  If they can be moved to the
        # new base without mapping, do so

        for child in list(self.tree.children):
            if not child.is_leaf():
                mapping, affine = self._get_map_transform(
                    new_map_domain, child.domain)
                if not mapping:
                    # set parent
                    child.parent = new_base
                    # remove from old parent's child list
                    self.tree.children.remove(child)
                    # and add to new parent's child list
                    new_base.children.add(child)

        self.have_input_map = True

    @property
    def absolute_root(self):
        """
        Returns the :class:`MapStore`'s :attr:`tree`, if the store has no input map,
        else the parent of the tree

        A convenience method to get the true root of the tree
        """

        return self.tree.parent if self.have_input_map else self.tree

    def _create_transform(self, node, transform, affine=None):
        """
        Creates a transform from the :class:`tree_node` node based on
        it's parent and any affine mapping supplied

        Parameters
        ----------
        node : :class:`tree_node`
            The node to create a transform for
        transform : :class:`domain_transform`
            The domain transform to store the base iname in
        affine : int or dict
            An integer or dictionary offset

        Returns
        -------
        new_iname : str
            The iname created for this transform
        transform_insn : str or None
            The loopy transform instruction to use.  If None, this is an
            affine transformation that doesn't require a separate instruction
        """

        assert node.parent is not None, (
            'Cannot create a transform for node'
            ' {} without parent'.format(node.domain.name))

        assert node.parent.iname, (
            'Cannot create a transform starting from parent node'
            ' {}, as it has no assigned iname'.format(node.parent.domain.name))

        # add the transformed inames, instruction and map
        new_iname = self._get_transform_iname(self.iname)

        # and use the parent's "iname" (i.e. with affine mapping)
        # to generate the transform
        transform_insn = self.generate_transform_instruction(
            node.parent.iname, new_iname, map_arr=node.domain.name,
            affine=affine
        )

        if affine:
            # store this as the new iname instead of issuing a new instruction
            new_iname = transform_insn
            transform_insn = None

        return new_iname, transform_insn

    def _get_transform_iname(self, iname):
        """Returns a new iname"""
        return self.taken_transform_names(iname)

    def _get_mask_transform(self, domain, new_domain):
        """
        Get the appropriate map transform between two given domains.
        Most likely, this will be a numpy array, but it may be an affine
        mapping

        Parameters
        ----------
        domain : :class:`creator`
            The domain to map
        new_domain: :class:`creator`
            The domain to map to

        Returns
        -------
        None if domains are equivalent
        Affine `str` map if an affine transform is possible
        :class:`creator` if a more complex map is required
        """

        try:
            dcheck = domain.initializer
        except AttributeError:
            dcheck = domain
        try:
            ncheck = new_domain.initializer
        except AttributeError:
            ncheck = new_domain

        # check equal
        if np.array_equal(dcheck, ncheck):
            return None, None

        # check for affine
        dset = np.where(dcheck != -1)[0]
        nset = np.where(ncheck != -1)[0]

        # must be same size for affine
        if dset.size == nset.size:
            # in order to be an affine mask transform, the set values should be
            # an affine transform
            diffs = nset - dset
            affine = diffs[0]
            if np.all(diffs == affine):
                # additionally, the affine mapped values should match the
                # original ones
                if np.array_equal(ncheck[nset], dcheck[dset]):
                    return new_domain, affine

        return new_domain, None

    def _get_map_transform(self, domain, new_domain):
        """
        Get the appropriate map transform between two given domains.
        Most likely, this will be a numpy array, but it may be an affine
        mapping

        Parameters
        ----------
        domain : :class:`creator`
            The domain to map
        new_domain: :class:`creator`
            The domain to map to

        Returns
        -------
        new_domain : :class:`creator`
        If not None, this is the mapping that must be used
            - None if domains are equivalent
            - `str` map if an affine transform is possible
            - :class:`creator` if a more complex map is required
        """

        try:
            dcheck = domain.initializer
        except AttributeError:
            dcheck = domain
        try:
            ncheck = new_domain.initializer
        except AttributeError:
            ncheck = new_domain

        # first, we need to make sure that the domains are the same size,
        # non-sensical otherwise

        if dcheck.shape != ncheck.shape:
            # Can't use affine map on domains of differing sizes
            return new_domain, None

        # check equal
        if np.array_equal(dcheck, ncheck):
            return None, None

        # check for affine map
        if np.all(ncheck - dcheck == (ncheck - dcheck)[0]):
            return new_domain, (ncheck - dcheck)[0]

        # finally return map
        return new_domain, None

    def _get_transform(self, base, domain):
        return self._get_map_transform(base, domain) if self._is_map()\
            else self._get_mask_transform(base, domain)

    def _check_is_valid_domain(self, domain):
        """Makes sure the domain passed is a valid :class:`creator`"""
        assert domain is not None, 'Invalid domain'
        assert isinstance(domain, creator), ('Domain'
                                             ' must be of type `creator`')
        assert domain.name is not None, ('Domain must have initialized name')
        assert domain.initializer is not None, (
            'Cannot use non-initialized creator {} as domain!'.format(
                domain.name))

    def _is_contiguous(self, domain):
        """Returns true if domain can be expressed with a simple for loop"""
        indicies = domain.initializer
        return indicies[0] + indicies.size - 1 == indicies[-1]

    def _check_create_transform(self, node):
        """
        Checks and creates a transform between the node and the
        parent node if necessary

        Parameters
        ----------
        node : :class:`tree_node`
            The domain to check

        Returns
        -------
        new_iname : str
            The iname created for this transform
        transform_insn : str or None
            The loopy transform instruction to use.  If None, this is an
            affine transformation that doesn't require a separate instruction
        transform : :class:`domain_transform`
            The representation of the transform used, for equality testing
        """

        domain = node.domain
        # check to see if root
        if node.parent is None:
            return None, None, None
        base = node.parent.domain

        # if this node should be treated as a variable,
        # don't create a transform
        if node.is_leaf():
            return None, None, None

        # get mapping
        mapping, affine = self._get_transform(base, domain)

        # if we actually need a mapping
        if mapping is not None:
            dt = domain_transform(mapping, affine)
            # see if this map already exists
            if node in self.transformed_domains:
                if dt == node.domain_transform:
                    return node.iname, node.insn, node.domain_transform

            # need a new map, so add
            iname, insn = self._create_transform(node, dt, affine=affine)
            return iname, insn, dt

        return None, None, None

    def _get_base_domain(self):
        """
        Conviencience method to get domain agnostic of map / mask type
        """
        return self.map_domain

    def check_and_add_transform(self, variable, domain, iname=None,
                                force_inline=False):
        """
        Check the domain of the variable given against the base domain of this
        kernel.  If not a match, a map / mask instruction will be generated
        as necessary--i.e. if no other variable with the transformed domain has
        already been specified--and the mapping will be stored for string
        creation

        Parameters
        ----------
        variable : :class:`creator` or list thereof
            The NameStore variable(s) to work with
        domain : :class:`creator`
            The domain of the variable to check.
            Note: this must be an initialized creator (i.e. temporary variable)
        iname : str
            The iname to transform.  If not specified, it will default to 'i'
            _or_ the iname for the transform of this domain
        force_inline : bool
            If True, the developed transform (if any) must be expressed as an
            inline transform.  If the transform is not affine, an exception
            will be raised

        Returns
        -------
        transform : :class:`domain_transform`
            The resulting transform, or None if not added
        """

        if self.is_finalized:
            if self.raise_on_final:
                raise Exception(
                    'Cannot add domain {} for variable {} to the tree as this'
                    ' mapstore is finalized, which may invalidate '
                    ' previously calculated transform data'.format(
                        domain.name,
                        variable.name))
            else:
                logger = logging.getLogger(__name__)
                logger.warn(
                    'Adding domain {} for variable {} to tree after'
                    ' finalization, this should not be used outside of unit'
                    ' testing'.format(domain.name, variable.name))

        # make sure this domain is valid
        self._check_is_valid_domain(domain)

        # check to see if this domain is already in the tree
        try:
            node = self.domain_to_nodes[domain]
            assert node.domain.initializer is not None, (
                "Can't use non-initialized creator {} as a transform domain".
                format(node.domain.name))
        except:
            # add the domain to the base of the tree
            node = self.tree.add_child(domain)

        # add variable to the new tree node
        node.add_child(variable)

    def finalize(self):
        """
        Turns the developed domain tree into transforms and instructions
        so that variables can begin to be created.  Called automatically on
        first use of :meth:`apply_maps`

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        # need to check the first level to see if we need an input map
        if self._is_map():
            # if it's not a contiguous name, we're forced to take a map
            if not self._is_contiguous(self.map_domain):
                self._add_input_map()
            # otherwise, test if we need one because of a child domain
            if not self.have_input_map:
                base = self.tree.domain
                for child in list(self.tree.children):
                    if not child.is_leaf() and not self.have_input_map:
                        mapping, affine = self._get_map_transform(
                            base, child.domain)
                        if mapping and not affine and base.initializer[0] != 0:
                            # need and input map
                            self._add_input_map()

        # next, we need to create our transforms
        # the goal here is to recursively transverse the tree checking whether
        # a transform is needed, such that any applied maps pick up the
        # right combination of transforms / inames

        base = self.tree if self.tree.parent is None else self.tree.parent
        branches = [[base]]

        while branches:
            # grab the current nodes under consideration
            branch = branches.pop()

            # for each sub domain in this branch
            for node in branch:
                if node != base:
                    # check for transform
                    iname, insn, dt = self._check_create_transform(node)
                    # and update node (empty transform will take the parent's)
                    # iname
                    node.set_transform(iname, insn, dt)

                    if not (iname is None and insn is None and dt is None):
                        # have a transform, add to the variable list
                        self.transformed_domains.add(node)
                    else:
                        # carry the parent's iname
                        node.iname = node.parent.iname

                # and update the branch lists
                branches.append(list(node.children))

        self.is_finalized = True

    def apply_maps(self, variable, *indicies, **kwargs):
        """
        Applies the developed iname mappings to the indicies supplied and
        returns the created loopy Arg/Temporary and the string version

        Parameters
        ----------
        variable : :class:`creator`
            The NameStore variable(s) to work with
        indices : list of str
            The inames to map
        affine : int
            An affine transformation to apply inline to this variable.
        Returns
        -------
        lp_var : :class:`loopy.GlobalArg` or :class:`loopy.TemporaryVariable`
            The generated variable
        lp_str : str
            The string indexed variable
        """

        if not self.is_finalized:
            self.finalize()

        affine = kwargs.pop('affine', None)

        var_affine = 0
        if variable.affine is not None:
            var_affine = variable.affine

        have_affine = var_affine or affine

        def __get_affine(iname):
            if not have_affine:
                return iname
            aff = 0
            if isinstance(affine, dict):
                if iname in affine:
                    aff = affine[iname]
            elif affine is not None:
                aff = affine
            if isinstance(aff, str):
                if var_affine:
                    aff += ' + {}'.format(var_affine)
                return iname + ' + {}'.format(aff)
            elif aff or var_affine:
                aff += var_affine
                return iname + ' {} {}'.format('+' if aff >= 0 else '-',
                                               np.abs(aff))
            return iname

        if variable in self.domain_to_nodes:
            # get the node this belongs to
            node = self.domain_to_nodes[variable]
        else:
            # ensure that any input map is picked up
            node = self.tree

        if have_affine and len(indicies) != 1 and not isinstance(affine, dict):
            raise Exception("Can't apply affine transformation to indicies, {}"
                            " as the index to apply to cannot be"
                            " determined".format(','.join(indicies)))

        # watch out for input maps w/ affine dicts
        if node == self.tree and isinstance(affine, dict)\
                and self.iname in affine and self.have_input_map:
            # copy to avoid any outside effects
            affine = affine.copy()
            # check
            assert node.iname not in affine, (
                "Can't resolve both input map {} and iname {} in affine map".format(
                    node.iname, self.iname))
            # copy over
            affine[node.iname] = affine[self.iname]
            del affine[self.iname]

        # pick up any mappings
        indicies = tuple(x if x != self.iname else
                         (node.iname if node.is_leaf() or node == self.tree
                          else node.parent.iname)
                         for x in indicies)

        # return affine mapping
        return variable(*tuple(__get_affine(i) for i in indicies),
                        working_buffer_index=kwargs.pop(
                            'working_buffer_index',
                            self.working_buffer_index),
                        reshape_to_working_buffer=kwargs.pop(
                            'reshape_to_working_buffer',
                            self.initial_condition_dimension),
                        **kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def generate_transform_instruction(self, oldname, newname, map_arr,
                                       affine='', force_inline=False):
        """
        Generates a loopy instruction that maps oldname -> newname via the
        mapping array (non-affine), or a simple affine transformation

        Parameters
        ----------
        oldname : str
            The old index to map from
        newname : str
            The new temporary variable to map to
        map_arr : str
            The array that holds the mappings
        affine : int, optional
            An optional affine mapping term that may be passed in
        force_inline : bool, optional
            If true, and affine simply return an inline transform rather than
            a separate instruction

        Returns
        -------
        map_inst : str
            A string to be used as a `loopy.Instruction`.
            By convention these will be given ids, id=index_new_iname
            to enable proper dependencies in loop sums
        """

        try:
            affine = ' + ' + str(int(affine))
            return oldname + affine
        except (TypeError, ValueError):
            if affine is None:
                affine = ''
            pass

        return ('<> {newname} = {mapper}[{oldname}]{affine} '
                '{{id=index_{newname}}}').format(
            newname=newname,
            mapper=map_arr,
            oldname=oldname,
            affine=affine)

    def get_iname_domain(self):
        """
        Get the final iname / domain for kernel generation

        Returns
        -------
        iname_tup : tuple of ('iname', 'range')
            The iname and range string to be fed to loopy
        """

        base = self._get_base_domain()
        fmt_str = '{start} <= {ind} <= {end}'

        if self._is_map():
            return (self.iname, fmt_str.format(
                    ind=self.iname,
                    start=base.initializer[0],
                    end=base.initializer[-1]))
        else:
            return (self.iname, fmt_str.format(
                    ind=self.iname,
                    start=0,
                    end=base.initializer.size - 1))


class creator(object):

    """
    The generic namestore interface, allowing easy access to
    loopy object creation, mapping, masking, etc.
    """

    def __init__(self, name, dtype, shape, order,
                 initializer=None, scope=scopes.GLOBAL,
                 fixed_indicies=None, is_temporary=False, affine=None):
        """
        Initializes the creator object

        Parameters
        ----------
        name : str
            The name of the loopy array to create
        dtype : :class:`numpy.dtype`
            The dtype of the array
        shape : tuple of (int, str)
            The shape of the array to create, parseable by loopy
        initializer : :class:`numpy.ndarray`
            If specified, the initializer of this array
        scope : :class:`loopy.AddressSpace`
            The scope of an initialized loopy array
        fixed_indicies : list of tuple
            If supplied, a list of index number, fixed values that
            specify indicies (e.g. for the Temperature/Phi array)
        order : ['C', 'F']
            The row/column-major data format to use in storage
        is_temporary : bool
            If true, this should be a temporary variable
        affine : int
            If supplied, this represents an offset that should be applied to
            the creator upon indexing
        """

        self.name = name
        self.dtype = dtype
        if not isinstance(shape, tuple):
            shape = (shape,)
        self.shape = shape
        self.scope = scope
        self.initializer = initializer
        self.fixed_indicies = None
        self.num_indicies = len(shape)
        self.order = order
        self.affine = affine
        if fixed_indicies is not None:
            self.fixed_indicies = fixed_indicies[:]
        if is_temporary or initializer is not None:
            self.creator = self.__temp_var_creator
            if initializer is not None:
                assert dtype == initializer.dtype, (
                    'Incorrect dtype specified for {}, got: {} expected: {}'
                    .format(name, initializer.dtype, dtype))
                assert shape == initializer.shape, (
                    'Incorrect shape specified for {}, got: {} expected: {}'
                    .format(name, initializer.shape, shape))
        else:
            self.creator = self.__glob_arg_creator

    @property
    def is_temporary(self):
        return isinstance(self.initializer, np.ndarray) \
            or self.initializer is not None

    @property
    def size(self):
        if self.initializer is None:
            raise NotImplementedError
        return self.initializer.size

    def __getitem__(self, key):
        if self.initializer is None:
            raise NotImplementedError
        return self.initializer[key]

    def __get_indicies(self, *indicies):
        if self.fixed_indicies:
            inds = [None for i in self.shape]
            for i, v in self.fixed_indicies:
                inds[i] = v
            empty = [i for i, x in enumerate(inds) if x is None]
            assert len(empty) == len(indicies), (
                'Wrong number of '
                'indicies supplied for {}: expected {} got {}'.format(
                    self.name, len(empty), len(indicies)))
            for i, ind in enumerate(empty):
                inds[ind] = indicies[i]
            return inds
        else:
            assert len(indicies) == self.num_indicies, (
                'Wrong number of indicies supplied for {}: expected {} got {}'
                .format(self.name, len(self.shape), len(indicies)))
            return indicies[:]

    def __temp_var_creator(self, **kwargs):
        # set default args
        arg_dict = {'shape': self.shape,
                    'dtype': self.dtype,
                    'order': self.order,
                    'initializer': self.initializer,
                    'scope': self.scope,
                    'read_only': self.initializer is not None}

        # and update any supplied overrides
        arg_dict.update(kwargs)
        return lp.TemporaryVariable(self.name, **arg_dict)

    def __glob_arg_creator(self, **kwargs):
        # set default args
        arg_dict = {'shape': self.shape,
                    'dtype': self.dtype,
                    'address_space': self.scope,
                    'order': self.order}
        # and update any supplied overrides
        arg_dict.update(kwargs)
        return lp.ArrayArg(self.name, **arg_dict)

    def __call__(self, *indicies, **kwargs):
        """
        Create a loopy array and corresponding string based on the supplied
        :param:`indicies` and :param:`kwargs`

        Parameters
        ----------
        indicies: tuple of str
            The indicies to apply for creation of the stringified array.
            For instance an array index of the form:
            ```
                a[i, j] = ...
            ```
            should be passed the indicies ['i', 'j']
        working_buffer_index: str
            If supplied, the :class:`creator` will replace access to the global index
            with the result of the vector specialization to enable pre-splitting of
            the loopy arrays / indicies (and avoid index hell). :see:`working-buffer`
        use_local_name: bool [False]
            If True, append '_local" to the created variable to avoid duplicate
            argument names in the driver functions.
        reshape_to_working_buffer: str [None]
            If True, and :param:`working_buffer_index` is supplied, the created array
            will be reshaped to the working buffer size.
            If False, the created array will not be reshaped (but the indicies may
            be changed).
            If not specified, defaults to `work_size`
        """
        # figure out whether to use private memory or not
        wbi = kwargs.pop('working_buffer_index', None)
        use_local_name = kwargs.pop('use_local_name', False)
        reshape = kwargs.pop('reshape_to_working_buffer',
                             work_size.name if wbi else None)
        inds = self.__get_indicies(*indicies)

        # handle working buffer request
        glob_ind = None

        def match(ind):
            try:
                return global_ind in ind
            except TypeError:
                # ind isn't iterable, hence not a str and therefore not global
                # ind
                return False

        # find the global ind if there
        if not self.is_temporary:
            glob_ind = next((i for i, ind in enumerate(inds) if match(ind)), None)
        if glob_ind is not None:
            if wbi:
                # convert index string to parallel iname only
                inds = tuple(s if i != glob_ind else s.replace(global_ind, wbi)
                             for i, s in enumerate(inds))
            # and reshape the array
            if reshape:
                shape = tuple(s if i != glob_ind else reshape
                              for i, s in enumerate(self.shape))
            else:
                shape = self.shape[:]
            lp_arr = self.__glob_arg_creator(shape=shape, **kwargs)
        else:
            lp_arr = self.creator(**kwargs)

        # check name
        name = lp_arr.name
        if use_local_name:
            name += local_name_suffix
            lp_arr = lp_arr.copy(name=name)

        return (lp_arr, name + '[{}]'.format(', '.join(
            str(x) for x in inds)))

    def copy(self, **kwargs):
        init = kwargs.get('initializer', self.initializer)
        init = init.copy() if isinstance(init, np.ndarray) else init
        fixed = kwargs.get('fixed_indicies', self.fixed_indicies)
        fixed = copy.deepcopy(fixed)
        return self.__class__(
            name=kwargs.get('name', self.name),
            dtype=kwargs.get('dtype', self.dtype),
            shape=kwargs.get('shape', self.shape),
            scope=kwargs.get('scope', self.scope),
            initializer=init,
            fixed_indicies=fixed,
            order=kwargs.get('order', self.order),
            affine=kwargs.get('affine', self.affine))


class jac_creator(creator):
    def __init__(self, *args, **kwargs):
        # store our row / column indicies
        self.row_inds = kwargs.pop('row_inds')
        self.col_inds = kwargs.pop('col_inds')
        # enable non-sparse guarded Jacobian access for FD-jacobian
        self.is_sparse = kwargs.pop('is_sparse', True)

        from pyjac.loopy_utils import preambles_and_manglers as lp_pregen
        self.lookup_call = Template(Template(
            '${lookup}(${start}, ${end}, ${match})').safe_substitute(
                lookup=lp_pregen.jac_indirect_lookup.name))
        super(jac_creator, self).__init__(*args, **kwargs)

    def __get_offset_and_lookup(self, *indicies):
        """
        Returns the correct sparse offset and lookup based on :param:`indicies` and
        our own :param:`order`
        """

        def __lookups(arr, lookup, match):
            def __lt(x):
                try:
                    x = int(x)
                except ValueError:
                    return False
                return x < 2

            # if we're in the first two rows in C-order, they are full
            can_skip = self.order == 'C' and __lt(lookup)
            # or if our match is less than two in any column in F-order
            can_skip = can_skip or self.order == 'F' and __lt(match)
            if can_skip:
                # this is a temperature or extra variable derivative
                # hence, we don't need to do an actual lookup (as all entries
                # are populated
                return str(match)

            def __add():
                if isinstance(lookup, int):
                    return lookup + 1
                return str(lookup) + ' + 1'
            # otherwise, we need to call the lookup function
            return self.lookup_call.safe_substitute(
                start=arr(lookup)[1],
                end=arr(__add())[1],
                match=match)

        if self.order == 'C':
            # looking at a CRS, hence we take the row index (indicies[-2])
            # and use that to get the row offset
            # and we need to do a lookup on the column ind
            lookup = __lookups(self.row_inds, indicies[-2], indicies[-1])
            # and use the row index to get the row offsets
            offset = self.row_inds(indicies[-2])[1]
        else:
            # looking at a CCS:
            # and use the column index to get the column offset
            offset = self.col_inds(indicies[-1])[1]
            # we need to do a lookup on the row ind
            lookup = __lookups(self.col_inds, indicies[-1], indicies[-2])
        return offset, lookup

    def __call__(self, *indicies, **kwargs):
        """
        Special keywords for :class:`jac_creator`

        ignore_lookups: bool [False]
            If True, do not call the indirect lookups.  Occasionally useful
            e.g. when resetting the Jacobian we don't need to lookup entries.
        plain_index: bool [False]
            If True, return index information instread of the
            :class:`loopy.GlobalArg` and access string returned by the parent
            :func:`creator.__call__`.  Useful when precomputing indicies
            to check to see if jacobian entry exists

            Returns
            -------
            replace_ind: int
                The index in the jacobian indicies that was should be replaced
                with a precomputed index
            computed_ind: str
                The computed lookup / sparse jacobian index -- this is the index
                that must be checked
            offset: str
                The offset string for sparse indicies -- ignored for non-sparse
                indicies
            lookup: str
                The lookup string

        """
        ignore_lookups = kwargs.pop('ignore_lookups', False)
        plain_index = kwargs.pop('plain_index', False)

        # all we have to do here is figure out the order, and add the row / column
        # indirect lookup accordingly
        if not ignore_lookups:
            indicies = list(indicies)
            offset, lookup = self.__get_offset_and_lookup(*indicies[:])
            if self.is_sparse:
                # add the offset to the lookup
                indicies = (indicies[0], ' + '.join([offset, lookup]))
                replace_ind = 1
                computed_ind = indicies[1]
            elif self.order == 'C':
                # only replace the column w/ lookup
                indicies[2] = lookup
                replace_ind = 2
                computed_ind = indicies[2]
            elif self.order == 'F':
                # only replace the row w/ lookup
                indicies[1] = lookup
                replace_ind = 1
                computed_ind = indicies[1]

        if plain_index:
            assert not ignore_lookups, "Can't do both."
            # return sparse / check index, offset & lookup
            return (replace_ind, computed_ind, offset, lookup)

        return super(jac_creator, self).__call__(*indicies, **kwargs)


def _make_mask(map_arr, mask_size):
    """
    Create a mask array from the given map and total mask size
    """

    assert len(map_arr.shape) == 1, "Can't make mask from 2-D array"

    mask = np.full(mask_size, -1, dtype=kint_type)
    mask[map_arr] = np.arange(map_arr.size, dtype=kint_type)
    return mask


# various names used by the outside world
pressure_array = 'P_arr'
"""
    The name of the pressure array
"""

volume_array = 'V_arr'
"""
    The name of the volume array
"""

state_vector = 'phi'
"""
    The name of the state vector array
"""

state_vector_rate_of_change = 'dphi'
"""
    The name of the state vector rate of change array
"""

jacobian_array = 'jac'
"""
    The name of the jacobian array
"""

enthalpy_array = 'h'
"""
    The name of the enthalpy array
"""

internal_energy_array = 'u'
"""
    The name of the enthalpy array
"""

constant_pressure_specific_heat = 'cp'
"""
    The name of the cp array
"""

constant_volume_specific_heat = 'cv'
"""
    The name of the cv array
"""

rate_const_thermo_coeff_array = 'b'
"""
    The name of the 'b' array utilized in calculation of the rate
    constants
"""

forward_rate_of_progress = 'rop_fwd'
"""
    The name of the forward ROP array
"""

reverse_rate_of_progress = 'rop_rev'
"""
    The name of the reverse ROP array
"""

net_rate_of_progress = 'rop_net'
"""
    The name of the net ROP array
"""

pressure_modification = 'pres_mod'
"""
    The name of the pressure modification of the ROP array
"""


class NameStore(object):

    """
    A convenience class that simplifies loopy array creation, indexing, mapping
    and masking

    Attributes
    ----------
    loopy_opts : :class:`LoopyOptions`
        The loopy options object describing the kernels
    rate_info : dict of reaction/species rate parameters
        Keys are 'simple', 'plog', 'cheb', 'fall', 'chem', 'thd'
        Values are further dictionaries including addtional rate info, number,
        offset, maps, etc.
    order : ['C', 'F']
        The row/column-major data format to use in storage
    conp : Boolean [True]
        If true, use the constant pressure formulation
    test_size : str or int
        Optional size used in testing.  If not supplied, this is a kernel arg
    """

    def __init__(self, loopy_opts, rate_info, conp=True, test_size='problem_size'):
        self.loopy_opts = loopy_opts
        self.rate_info = rate_info
        self.order = loopy_opts.order
        self.conp = conp
        self.jac_format = loopy_opts.jac_format
        self.jac_type = loopy_opts.jac_type
        self._add_arrays(rate_info, test_size)

    def __getattr__(self, name):
        """
        Override of getattr such that NameStore.nonexistantkey -> None
        """
        try:
            return super(NameStore, self).__getattr__(self, name)
        except AttributeError:
            return None

    def __make_offset(self, arr):
        """
        Creates an offset array from the given array
        """

        assert len(arr.shape) == 1, "Can't make offset from 2-D array"
        assert arr.dtype == kint_type, "Offset arrays should be integers!"

        return np.array(np.concatenate(
            (np.cumsum(arr) - arr, np.array([np.sum(arr)]))),
            dtype=kint_type)

    def _add_arrays(self, rate_info, test_size):
        """
        Initialize the various arrays needed for the namestore
        """

        # problem size
        if not isinstance(test_size, int):
            test_size = problem_size.name

        # generic ranges
        self.num_specs = creator('num_specs', shape=(rate_info['Ns'],),
                                 dtype=kint_type, order=self.order,
                                 initializer=np.arange(rate_info['Ns'],
                                                       dtype=kint_type))
        self.num_specs_no_ns = creator('num_specs_no_ns',
                                       shape=(rate_info['Ns'] - 1,),
                                       dtype=kint_type, order=self.order,
                                       initializer=np.arange(
                                           rate_info['Ns'] - 1,
                                           dtype=kint_type))
        self.num_reacs = creator('num_reacs', shape=(rate_info['Nr'],),
                                 dtype=kint_type, order=self.order,
                                 initializer=np.arange(rate_info['Nr'],
                                                       dtype=kint_type))
        self.num_rev_reacs = creator('num_rev_reacs',
                                     shape=(rate_info['rev']['num'],),
                                     dtype=kint_type, order=self.order,
                                     initializer=np.arange(
                                         rate_info['rev']['num'],
                                         dtype=kint_type))

        self.phi_inds = creator('phi_inds',
                                shape=(rate_info['Ns'] + 1),
                                dtype=kint_type, order=self.order,
                                initializer=np.arange(rate_info['Ns'] + 1,
                                                      dtype=kint_type))
        self.phi_spec_inds = creator('phi_spec_inds',
                                     shape=(rate_info['Ns'] - 1,),
                                     dtype=kint_type, order=self.order,
                                     initializer=np.arange(2,
                                                           rate_info['Ns'] + 1,
                                                           dtype=kint_type))

        # flat / dense jacobian
        if 'jac_inds' in rate_info:
            # if we're actually creating a jacobian
            name = 'flat_' + self.order
            flat_row_inds = rate_info['jac_inds'][name][:, 0]
            flat_col_inds = rate_info['jac_inds'][name][:, 1]
            self.num_nonzero_jac_inds = creator('num_jac_entries',
                                                shape=flat_row_inds.shape,
                                                dtype=kint_type,
                                                order=self.order,
                                                initializer=np.arange(
                                                    flat_row_inds.size,
                                                    dtype=kint_type))
            self.jac_size = creator('jac_size',
                                    shape=((rate_info['Ns'] + 1)**2),
                                    dtype=kint_type,
                                    order=self.order,
                                    initializer=np.arange((rate_info['Ns'] + 1)**2,
                                                          dtype=kint_type))
            self.flat_jac_row_inds = creator('jac_row_inds',
                                             shape=flat_row_inds.shape,
                                             dtype=kint_type,
                                             order=self.order,
                                             initializer=flat_row_inds)
            self.flat_jac_col_inds = creator('jac_col_inds',
                                             shape=flat_col_inds.shape,
                                             dtype=kint_type,
                                             order=self.order,
                                             initializer=flat_col_inds)

            # Compressed Row Storage jacobian
            crs_col_ind = rate_info['jac_inds']['crs']['col_ind']
            self.crs_jac_col_ind = creator('jac_col_inds',
                                           shape=crs_col_ind.shape,
                                           dtype=kint_type,
                                           order=self.order,
                                           initializer=crs_col_ind)
            crs_row_ptr = rate_info['jac_inds']['crs']['row_ptr']
            self.crs_jac_row_ptr = creator('jac_row_ptr',
                                           shape=crs_row_ptr.shape,
                                           dtype=kint_type,
                                           order=self.order,
                                           initializer=crs_row_ptr)

            # Compressed Column Storage jacobian
            ccs_row_ind = rate_info['jac_inds']['ccs']['row_ind']
            self.ccs_jac_row_ind = creator('jac_row_inds',
                                           shape=ccs_row_ind.shape,
                                           dtype=kint_type,
                                           order=self.order,
                                           initializer=ccs_row_ind)
            ccs_col_ptr = rate_info['jac_inds']['ccs']['col_ptr']
            self.ccs_jac_col_ptr = creator('jac_col_ptr',
                                           shape=ccs_col_ptr.shape,
                                           dtype=kint_type,
                                           order=self.order,
                                           initializer=ccs_col_ptr)

            if self.jac_format == JacobianFormat.sparse or \
                    self.jac_type == JacobianType.finite_difference:
                if self.order == 'C':
                    # use CRS
                    self.jac_row_inds = self.crs_jac_row_ptr
                    self.jac_col_inds = self.crs_jac_col_ind
                elif self.order == 'F':
                    # use CCS
                    self.jac_row_inds = self.ccs_jac_row_ind
                    self.jac_col_inds = self.ccs_jac_col_ptr

        # state arrays
        self.T_arr = creator(state_vector, shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 0)])

        # handle extra variable and P / V arrays
        self.E_arr = creator(state_vector, shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 1)])
        if self.conp:
            self.P_arr = creator(pressure_array, shape=(test_size,),
                                 dtype=np.float64, order=self.order)
            self.V_arr = self.E_arr
        else:
            self.P_arr = self.E_arr
            self.V_arr = creator(volume_array, shape=(test_size,),
                                 dtype=np.float64, order=self.order)

        self.n_arr = creator(state_vector, shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order)
        self.conc_arr = creator('conc', shape=(test_size, rate_info['Ns']),
                                dtype=np.float64, order=self.order)
        self.conc_ns_arr = creator('conc', shape=(test_size, rate_info['Ns']),
                                   dtype=np.float64, order=self.order,
                                   fixed_indicies=[(1, rate_info['Ns'] - 1)])
        self.n_dot = creator(state_vector_rate_of_change,
                             shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order)
        self.T_dot = creator(state_vector_rate_of_change,
                             shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 0)])
        self.E_dot = creator(state_vector_rate_of_change,
                             shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 1)])

        if self.jac_format == JacobianFormat.sparse and 'jac_inds' in rate_info:
            self.jac = jac_creator(jacobian_array,
                                   shape=(test_size, self.num_nonzero_jac_inds.size),
                                   order=self.order,
                                   dtype=np.float64,
                                   row_inds=self.jac_row_inds,
                                   col_inds=self.jac_col_inds)
        elif self.jac_type == JacobianType.finite_difference and \
                'jac_inds' in rate_info:
            self.jac = jac_creator(jacobian_array,
                                   shape=(test_size, rate_info['Ns'] + 1,
                                          rate_info['Ns'] + 1),
                                   order=self.order,
                                   dtype=np.float64,
                                   row_inds=self.jac_row_inds,
                                   col_inds=self.jac_col_inds,
                                   is_sparse=False)
        else:
            self.jac = creator(jacobian_array,
                               shape=(test_size, rate_info['Ns'] + 1,
                                      rate_info['Ns'] + 1),
                               order=self.order,
                               dtype=np.float64)

        self.spec_rates = creator('wdot', shape=(test_size, rate_info['Ns']),
                                  dtype=np.float64, order=self.order)

        # molecular weights
        self.mw_arr = creator('mw', shape=(rate_info['Ns'],),
                              initializer=rate_info['mws'],
                              dtype=np.float64,
                              order=self.order)

        self.mw_inv = creator('mw_inv', shape=(rate_info['Ns'] - 1,),
                              initializer=1. / rate_info['mws'][:-1],
                              dtype=np.float64,
                              order=self.order)

        self.mw_post_arr = creator('mw_factor', shape=(rate_info['Ns'] - 1,),
                                   initializer=rate_info['mw_post'],
                                   dtype=np.float64,
                                   order=self.order)

        self.mw_work = creator('mw_work', shape=(test_size,),
                               dtype=np.float64,
                               order=self.order)

        # net species rates data

        # per reaction
        self.rxn_to_spec = creator('rxn_to_spec',
                                   dtype=kint_type,
                                   shape=rate_info['net'][
                                       'reac_to_spec'].shape,
                                   initializer=rate_info[
                                       'net']['reac_to_spec'],
                                   order=self.order)
        off = self.__make_offset(rate_info['net']['num_reac_to_spec'])
        self.rxn_to_spec_offsets = creator('net_reac_to_spec_offsets',
                                           dtype=kint_type,
                                           shape=off.shape,
                                           initializer=off,
                                           order=self.order)
        self.rxn_to_spec_reac_nu = creator('reac_to_spec_nu',
                                           dtype=kint_type, shape=rate_info[
                                               'net']['nu'].shape,
                                           initializer=rate_info['net']['nu'],
                                           order=self.order,
                                           affine=1)
        self.rxn_to_spec_prod_nu = creator('reac_to_spec_nu',
                                           dtype=kint_type, shape=rate_info[
                                               'net']['nu'].shape,
                                           initializer=rate_info['net']['nu'],
                                           order=self.order)

        self.rxn_has_ns = creator('rxn_has_ns',
                                  dtype=kint_type,
                                  shape=rate_info['reac_has_ns'].shape,
                                  initializer=rate_info['reac_has_ns'],
                                  order=self.order)
        self.num_rxn_has_ns = creator('num_rxn_has_ns',
                                      dtype=kint_type,
                                      shape=rate_info['reac_has_ns'].shape,
                                      initializer=np.arange(
                                          rate_info['reac_has_ns'].size,
                                          dtype=kint_type),
                                      order=self.order)

        # per species
        net_nonzero_spec = rate_info['net_per_spec']['map'][np.where(
            rate_info['net_per_spec']['map'] != rate_info['Ns'] - 1)]
        self.net_nonzero_spec = creator(
            'net_nonzero_spec_no_ns', dtype=kint_type,
            shape=net_nonzero_spec.shape,
            initializer=net_nonzero_spec,
            order=self.order)
        self.net_nonzero_phi = creator(
            'net_nonzero_phi', dtype=kint_type,
            shape=(net_nonzero_spec.shape[0] + 2,),
            initializer=np.asarray(np.hstack(([0, 1], net_nonzero_spec + 2)),
                                   dtype=kint_type),
            order=self.order)

        self.spec_to_rxn = creator('spec_to_rxn', dtype=kint_type,
                                   shape=rate_info['net_per_spec'][
                                       'reacs'].shape,
                                   initializer=rate_info[
                                       'net_per_spec']['reacs'],
                                   order=self.order)
        off = self.__make_offset(rate_info['net_per_spec']['reac_count'])
        self.spec_to_rxn_offsets = creator('spec_to_rxn_offsets',
                                           dtype=kint_type,
                                           shape=off.shape,
                                           initializer=off,
                                           order=self.order)
        self.spec_to_rxn_nu = creator('spec_to_rxn_nu',
                                      dtype=kint_type, shape=rate_info[
                                          'net_per_spec']['nu'].shape,
                                      initializer=rate_info[
                                          'net_per_spec']['nu'],
                                      order=self.order)

        # rop's and fwd / rev / thd maps
        self.rop_net = creator(net_rate_of_progress,
                               dtype=np.float64,
                               shape=(test_size, rate_info['Nr']),
                               order=self.order)

        self.rop_fwd = creator(forward_rate_of_progress,
                               dtype=np.float64,
                               shape=(test_size, rate_info['Nr']),
                               order=self.order)

        if rate_info['rev']['num']:
            self.rop_rev = creator(reverse_rate_of_progress,
                                   dtype=np.float64,
                                   shape=(
                                       test_size, rate_info['rev']['num']),
                                   order=self.order)
            self.rev_map = creator('rev_map',
                                   dtype=kint_type,
                                   shape=rate_info['rev']['map'].shape,
                                   initializer=rate_info[
                                       'rev']['map'],
                                   order=self.order)

            mask = _make_mask(rate_info['rev']['map'],
                              rate_info['Nr'])
            self.rev_mask = creator('rev_mask',
                                    dtype=kint_type,
                                    shape=mask.shape,
                                    initializer=mask,
                                    order=self.order)

        if rate_info['thd']['num']:
            self.pres_mod = creator(pressure_modification,
                                    dtype=np.float64,
                                    shape=(
                                        test_size, rate_info['thd']['num']),
                                    order=self.order)
            self.thd_map = creator('thd_map',
                                   dtype=kint_type,
                                   shape=rate_info['thd']['map'].shape,
                                   initializer=rate_info[
                                       'thd']['map'],
                                   order=self.order)

            mask = _make_mask(rate_info['thd']['map'],
                              rate_info['Nr'])
            self.thd_mask = creator('thd_mask',
                                    dtype=kint_type,
                                    shape=mask.shape,
                                    initializer=mask,
                                    order=self.order)

            thd_inds = np.arange(rate_info['thd']['num'], dtype=kint_type)
            self.thd_inds = creator('thd_inds',
                                    dtype=kint_type,
                                    shape=thd_inds.shape,
                                    initializer=thd_inds,
                                    order=self.order)

        # reaction data (fwd / rev rates, KC)
        self.kf = creator('kf',
                          dtype=np.float64,
                          shape=(test_size, rate_info['Nr']),
                          order=self.order)

        # simple reaction parameters
        self.simple_A = creator('simple_A',
                                dtype=rate_info['simple']['A'].dtype,
                                shape=rate_info['simple']['A'].shape,
                                initializer=rate_info['simple']['A'],
                                order=self.order)
        self.simple_beta = creator('simple_beta',
                                   dtype=rate_info[
                                       'simple']['b'].dtype,
                                   shape=rate_info[
                                       'simple']['b'].shape,
                                   initializer=rate_info[
                                       'simple']['b'],
                                   order=self.order)
        self.simple_Ta = creator('simple_Ta',
                                 dtype=rate_info['simple']['Ta'].dtype,
                                 shape=rate_info['simple']['Ta'].shape,
                                 initializer=rate_info['simple']['Ta'],
                                 order=self.order)
        # reaction types
        self.simple_rtype = creator('simple_rtype',
                                    dtype=rate_info[
                                        'simple']['type'].dtype,
                                    shape=rate_info[
                                        'simple']['type'].shape,
                                    initializer=rate_info[
                                        'simple']['type'],
                                    order=self.order)

        # num simple
        num_simple = np.arange(rate_info['simple']['num'], dtype=kint_type)
        self.num_simple = creator('num_simple',
                                  dtype=kint_type,
                                  shape=num_simple.shape,
                                  initializer=num_simple,
                                  order=self.order)

        # simple map
        self.simple_map = creator('simple_map',
                                  dtype=kint_type,
                                  shape=rate_info['simple']['map'].shape,
                                  initializer=rate_info['simple']['map'],
                                  order=self.order)
        # simple mask
        simple_mask = _make_mask(rate_info['simple']['map'],
                                 rate_info['Nr'])
        self.simple_mask = creator('simple_mask',
                                   dtype=simple_mask.dtype,
                                   shape=simple_mask.shape,
                                   initializer=simple_mask,
                                   order=self.order)

        self.num_simple = creator('num_simple',
                                  dtype=kint_type,
                                  shape=num_simple.shape,
                                  initializer=num_simple,
                                  order=self.order)

        # rtype maps
        for rtype in np.unique(rate_info['simple']['type']):
            # find the map
            mapv = rate_info['simple']['map'][
                np.where(rate_info['simple']['type'] == rtype)[0]]
            setattr(self, 'simple_rtype_{}_map'.format(rtype),
                    creator('simple_rtype_{}_map'.format(rtype),
                            dtype=mapv.dtype,
                            shape=mapv.shape,
                            initializer=mapv,
                            order=self.order))
            # and the mask
            maskv = _make_mask(mapv, rate_info['Nr'])
            setattr(self, 'simple_rtype_{}_mask'.format(rtype),
                    creator('simple_rtype_{}_mask'.format(rtype),
                            dtype=maskv.dtype,
                            shape=maskv.shape,
                            initializer=maskv,
                            order=self.order))
            # and indicies inside of the simple parameters
            inds = np.where(
                np.in1d(rate_info['simple']['map'], mapv))[0].astype(
                dtype=kint_type)
            setattr(self, 'simple_rtype_{}_inds'.format(rtype),
                    creator('simple_rtype_{}_inds'.format(rtype),
                            dtype=inds.dtype,
                            shape=inds.shape,
                            initializer=inds,
                            order=self.order))
            # and num
            num = np.arange(inds.size, dtype=kint_type)
            setattr(self, 'simple_rtype_{}_num'.format(rtype),
                    creator('simple_rtype_{}_num'.format(rtype),
                            dtype=num.dtype,
                            shape=num.shape,
                            initializer=num,
                            order=self.order))

        if rate_info['rev']['num']:
            self.kr = creator('kr',
                              dtype=np.float64,
                              shape=(test_size, rate_info['rev']['num']),
                              order=self.order)

            self.Kc = creator('Kc',
                              dtype=np.float64,
                              shape=(test_size, rate_info['rev']['num']),
                              order=self.order)

            self.nu_sum = creator('nu_sum',
                                  dtype=rate_info['net'][
                                      'nu_sum'].dtype,
                                  shape=rate_info['net'][
                                      'nu_sum'].shape,
                                  initializer=rate_info[
                                      'net']['nu_sum'],
                                  order=self.order)

        # third body concs, maps, efficiencies, types, species
        if rate_info['thd']['num']:
            # third body concentrations
            self.thd_conc = creator('thd_conc',
                                    dtype=np.float64,
                                    shape=(
                                        test_size, rate_info['thd']['num']),
                                    order=self.order)

            # thd only indicies
            mapv = np.where(np.logical_not(np.in1d(rate_info['thd']['map'],
                                                   rate_info['fall']['map'])))[0]
            mapv = np.array(mapv, dtype=kint_type)
            self.thd_only_map = creator('thd_only_map',
                                        dtype=kint_type,
                                        shape=mapv.shape,
                                        initializer=mapv,
                                        order=self.order)
            self.num_thd_only = creator('num_thd_only',
                                        dtype=kint_type,
                                        shape=mapv.shape,
                                        initializer=np.arange(mapv.size,
                                                              dtype=kint_type),
                                        order=self.order)

            mask = _make_mask(mapv, rate_info['Nr'])
            self.thd_only_mask = creator('thd_only_mask',
                                         dtype=kint_type,
                                         shape=mask.shape,
                                         initializer=mask,
                                         order=self.order)

            num_specs = rate_info['thd']['spec_num'].astype(dtype=kint_type)
            spec_list = rate_info['thd']['spec'].astype(
                dtype=kint_type)
            thd_effs = rate_info['thd']['eff']

            # finally create arrays
            self.thd_eff = creator('thd_eff',
                                   dtype=thd_effs.dtype,
                                   shape=thd_effs.shape,
                                   initializer=thd_effs,
                                   order=self.order)
            num_thd = np.arange(rate_info['thd']['num'], dtype=kint_type)
            self.num_thd = creator('num_thd',
                                   dtype=num_thd.dtype,
                                   shape=num_thd.shape,
                                   initializer=num_thd,
                                   order=self.order)
            thd_only_ns_inds = np.where(
                np.in1d(
                    self.thd_only_map.initializer,
                    rate_info['thd']['has_ns']))[0].astype(kint_type)
            thd_only_ns_map = self.thd_only_map.initializer[
                thd_only_ns_inds]
            self.thd_only_ns_map = creator('thd_only_ns_map',
                                           dtype=thd_only_ns_map.dtype,
                                           shape=thd_only_ns_map.shape,
                                           initializer=thd_only_ns_map,
                                           order=self.order)

            self.thd_only_ns_inds = creator('thd_only_ns_inds',
                                            dtype=thd_only_ns_inds.dtype,
                                            shape=thd_only_ns_inds.shape,
                                            initializer=thd_only_ns_inds,
                                            order=self.order)
            self.thd_type = creator('thd_type',
                                    dtype=rate_info['thd']['type'].dtype,
                                    shape=rate_info['thd']['type'].shape,
                                    initializer=rate_info['thd']['type'],
                                    order=self.order)
            self.thd_spec = creator('thd_spec',
                                    dtype=spec_list.dtype,
                                    shape=spec_list.shape,
                                    initializer=spec_list,
                                    order=self.order)
            thd_offset = self.__make_offset(num_specs)
            self.thd_offset = creator('thd_offset',
                                      dtype=thd_offset.dtype,
                                      shape=thd_offset.shape,
                                      initializer=thd_offset,
                                      order=self.order)

        # falloff rxn rates, blending vals, reduced pressures, maps
        if rate_info['fall']['num']:
            # falloff reaction parameters
            self.kf_fall = creator('kf_fall',
                                   dtype=np.float64,
                                   shape=(test_size, rate_info['fall']['num']),
                                   order=self.order)
            self.fall_A = creator('fall_A',
                                  dtype=rate_info['fall']['A'].dtype,
                                  shape=rate_info['fall']['A'].shape,
                                  initializer=rate_info['fall']['A'],
                                  order=self.order)
            self.fall_beta = creator('fall_beta',
                                     dtype=rate_info[
                                         'fall']['b'].dtype,
                                     shape=rate_info[
                                         'fall']['b'].shape,
                                     initializer=rate_info[
                                         'fall']['b'],
                                     order=self.order)
            self.fall_Ta = creator('fall_Ta',
                                   dtype=rate_info['fall']['Ta'].dtype,
                                   shape=rate_info['fall']['Ta'].shape,
                                   initializer=rate_info['fall']['Ta'],
                                   order=self.order)
            # reaction types
            self.fall_rtype = creator('fall_rtype',
                                      dtype=rate_info[
                                          'fall']['type'].dtype,
                                      shape=rate_info[
                                          'fall']['type'].shape,
                                      initializer=rate_info[
                                          'fall']['type'],
                                      order=self.order)

            # fall mask
            fall_mask = _make_mask(rate_info['fall']['map'],
                                   rate_info['Nr'])
            self.fall_mask = creator('fall_mask',
                                     dtype=fall_mask.dtype,
                                     shape=fall_mask.shape,
                                     initializer=fall_mask,
                                     order=self.order)

            # rtype maps
            for rtype in np.unique(rate_info['fall']['type']):
                mapv = rate_info['fall']['map'][
                    np.where(rate_info['fall']['type'] == rtype)[0]]
                setattr(self, 'fall_rtype_{}_map'.format(rtype),
                        creator('fall_rtype_{}_map'.format(rtype),
                                dtype=mapv.dtype,
                                shape=mapv.shape,
                                initializer=mapv,
                                order=self.order))
                # create corresponding mask
                maskv = _make_mask(mapv, rate_info['Nr'])
                setattr(self, 'fall_rtype_{}_mask'.format(rtype),
                        creator('fall_rtype_{}_mask'.format(rtype),
                                dtype=maskv.dtype,
                                shape=maskv.shape,
                                initializer=maskv,
                                order=self.order))
                # and indicies inside of the falloff parameters
                inds = np.where(np.in1d(rate_info['fall']['map'], mapv))[0].astype(
                    dtype=kint_type)
                setattr(self, 'fall_rtype_{}_inds'.format(rtype),
                        creator('fall_rtype_{}_inds'.format(rtype),
                                dtype=inds.dtype,
                                shape=inds.shape,
                                initializer=inds,
                                order=self.order))
                # and num
                num = np.arange(inds.size, dtype=kint_type)
                setattr(self, 'fall_rtype_{}_num'.format(rtype),
                        creator('fall_rtype_{}_num'.format(rtype),
                                dtype=num.dtype,
                                shape=num.shape,
                                initializer=num,
                                order=self.order))

            # maps
            self.fall_map = creator('fall_map',
                                    dtype=kint_type,
                                    initializer=rate_info['fall']['map'],
                                    shape=rate_info['fall']['map'].shape,
                                    order=self.order)

            num_fall = np.arange(rate_info['fall']['num'], dtype=kint_type)
            self.num_fall = creator('num_fall',
                                    dtype=kint_type,
                                    initializer=num_fall,
                                    shape=num_fall.shape,
                                    order=self.order)

            # blending
            self.Fi = creator('Fi',
                              dtype=np.float64,
                              shape=(test_size, rate_info['fall']['num']),
                              order=self.order)

            # reduced pressure
            self.Pr = creator('Pr',
                              dtype=np.float64,
                              shape=(test_size, rate_info['fall']['num']),
                              order=self.order)

            # types
            self.fall_type = creator('fall_type',
                                     dtype=rate_info[
                                         'fall']['ftype'].dtype,
                                     shape=rate_info[
                                         'fall']['ftype'].shape,
                                     initializer=rate_info[
                                         'fall']['ftype'],
                                     order=self.order)

            # maps and masks
            fall_to_thd_map = np.array(
                np.where(
                    np.in1d(
                        rate_info['thd']['map'], rate_info['fall']['map'])
                )[0], dtype=kint_type)
            self.fall_to_thd_map = creator('fall_to_thd_map',
                                           dtype=kint_type,
                                           initializer=fall_to_thd_map,
                                           shape=fall_to_thd_map.shape,
                                           order=self.order)

            fall_to_thd_mask = _make_mask(fall_to_thd_map,
                                          rate_info['Nr'])
            self.fall_to_thd_mask = creator('fall_to_thd_mask',
                                            dtype=kint_type,
                                            initializer=fall_to_thd_mask,
                                            shape=fall_to_thd_mask.shape,
                                            order=self.order)

            if rate_info['fall']['troe']['num']:
                # Fcent, Atroe, Btroe
                self.Fcent = creator('Fcent',
                                     shape=(test_size,
                                            rate_info['fall']['troe']['num']),
                                     dtype=np.float64,
                                     order=self.order)

                self.Atroe = creator('Atroe',
                                     shape=(test_size,
                                            rate_info['fall']['troe']['num']),
                                     dtype=np.float64,
                                     order=self.order)

                self.Btroe = creator('Btroe',
                                     shape=(test_size,
                                            rate_info['fall']['troe']['num']),
                                     dtype=np.float64,
                                     order=self.order)

                # troe parameters
                self.troe_a = creator('troe_a',
                                      shape=rate_info['fall'][
                                          'troe']['a'].shape,
                                      dtype=rate_info['fall'][
                                          'troe']['a'].dtype,
                                      initializer=rate_info[
                                          'fall']['troe']['a'],
                                      order=self.order)
                self.troe_T1 = creator('troe_T1',
                                       shape=rate_info['fall'][
                                           'troe']['T1'].shape,
                                       dtype=rate_info['fall'][
                                           'troe']['T1'].dtype,
                                       initializer=rate_info[
                                           'fall']['troe']['T1'],
                                       order=self.order)
                self.troe_T3 = creator('troe_T3',
                                       shape=rate_info['fall'][
                                           'troe']['T3'].shape,
                                       dtype=rate_info['fall'][
                                           'troe']['T3'].dtype,
                                       initializer=rate_info[
                                           'fall']['troe']['T3'],
                                       order=self.order)
                self.troe_T2 = creator('troe_T2',
                                       shape=rate_info['fall'][
                                           'troe']['T2'].shape,
                                       dtype=rate_info['fall'][
                                           'troe']['T2'].dtype,
                                       initializer=rate_info[
                                           'fall']['troe']['T2'],
                                       order=self.order)

                # map and mask
                num_troe = np.arange(rate_info['fall']['troe']['num'],
                                     dtype=kint_type)
                self.num_troe = creator('num_troe',
                                        shape=num_troe.shape,
                                        dtype=num_troe.dtype,
                                        initializer=num_troe,
                                        order=self.order)
                self.troe_map = creator('troe_map',
                                        shape=rate_info['fall'][
                                            'troe']['map'].shape,
                                        dtype=rate_info['fall'][
                                            'troe']['map'].dtype,
                                        initializer=rate_info[
                                            'fall']['troe']['map'],
                                        order=self.order)
                troe_mask = _make_mask(rate_info['fall']['troe']['map'],
                                       rate_info['Nr'])
                self.troe_mask = creator('troe_mask',
                                         shape=troe_mask.shape,
                                         dtype=troe_mask.dtype,
                                         initializer=troe_mask,
                                         order=self.order)

                troe_inds = self.fall_to_thd_map.initializer[
                    self.troe_map.initializer]
                troe_ns_inds = np.where(
                    np.in1d(troe_inds, rate_info['thd']['has_ns']))[0].astype(
                        kint_type)
                troe_has_ns = self.troe_map.initializer[
                    troe_ns_inds]
                self.troe_has_ns = creator('troe_has_ns',
                                           shape=troe_has_ns.shape,
                                           dtype=troe_has_ns.dtype,
                                           initializer=troe_has_ns,
                                           order=self.order)
                self.troe_ns_inds = creator('troe_ns_inds',
                                            shape=troe_ns_inds.shape,
                                            dtype=troe_ns_inds.dtype,
                                            initializer=troe_ns_inds,
                                            order=self.order)

            if rate_info['fall']['sri']['num']:
                # X_sri
                self.X_sri = creator('X',
                                     shape=(test_size,
                                            rate_info['fall']['sri']['num']),
                                     dtype=np.float64,
                                     order=self.order)

                # sri parameters
                self.sri_a = creator('sri_a',
                                     shape=rate_info['fall'][
                                         'sri']['a'].shape,
                                     dtype=rate_info['fall'][
                                         'sri']['a'].dtype,
                                     initializer=rate_info[
                                         'fall']['sri']['a'],
                                     order=self.order)
                self.sri_b = creator('sri_b',
                                     shape=rate_info['fall'][
                                         'sri']['b'].shape,
                                     dtype=rate_info['fall'][
                                         'sri']['b'].dtype,
                                     initializer=rate_info[
                                         'fall']['sri']['b'],
                                     order=self.order)
                self.sri_c = creator('sri_c',
                                     shape=rate_info['fall'][
                                         'sri']['c'].shape,
                                     dtype=rate_info['fall'][
                                         'sri']['c'].dtype,
                                     initializer=rate_info[
                                         'fall']['sri']['c'],
                                     order=self.order)
                self.sri_d = creator('sri_d',
                                     shape=rate_info['fall'][
                                         'sri']['d'].shape,
                                     dtype=rate_info['fall'][
                                         'sri']['d'].dtype,
                                     initializer=rate_info[
                                         'fall']['sri']['d'],
                                     order=self.order)
                self.sri_e = creator('sri_e',
                                     shape=rate_info['fall'][
                                         'sri']['e'].shape,
                                     dtype=rate_info['fall'][
                                         'sri']['e'].dtype,
                                     initializer=rate_info[
                                         'fall']['sri']['e'],
                                     order=self.order)

                # map and mask
                num_sri = np.arange(rate_info['fall']['sri']['num'],
                                    dtype=kint_type)
                self.num_sri = creator('num_sri',
                                       shape=num_sri.shape,
                                       dtype=num_sri.dtype,
                                       initializer=num_sri,
                                       order=self.order)
                self.sri_map = creator('sri_map',
                                       shape=rate_info['fall'][
                                           'sri']['map'].shape,
                                       dtype=rate_info['fall'][
                                           'sri']['map'].dtype,
                                       initializer=rate_info[
                                           'fall']['sri']['map'],
                                       order=self.order)
                sri_mask = _make_mask(rate_info['fall']['sri']['map'],
                                      rate_info['Nr'])
                self.sri_mask = creator('sri_mask',
                                        shape=sri_mask.shape,
                                        dtype=sri_mask.dtype,
                                        initializer=sri_mask,
                                        order=self.order)

                sri_inds = self.fall_to_thd_map.initializer[
                    self.sri_map.initializer]
                sri_ns_inds = np.where(
                    np.in1d(sri_inds, rate_info['thd']['has_ns']))[0].astype(
                        kint_type)
                sri_has_ns = self.sri_map.initializer[
                    sri_ns_inds]
                self.sri_has_ns = creator('sri_has_ns_map',
                                          shape=sri_has_ns.shape,
                                          dtype=sri_has_ns.dtype,
                                          initializer=sri_has_ns,
                                          order=self.order)
                self.sri_ns_inds = creator('sri_ns_inds',
                                           shape=sri_ns_inds.shape,
                                           dtype=sri_ns_inds.dtype,
                                           initializer=sri_ns_inds,
                                           order=self.order)

            if rate_info['fall']['lind']['num']:
                # lind map / mask
                self.lind_map = creator('lind_map',
                                        shape=rate_info['fall'][
                                            'lind']['map'].shape,
                                        dtype=rate_info['fall'][
                                            'lind']['map'].dtype,
                                        initializer=rate_info[
                                            'fall']['lind']['map'],
                                        order=self.order)
                lind_mask = _make_mask(rate_info['fall']['lind']['map'],
                                       rate_info['Nr'])
                self.lind_mask = creator('lind_mask',
                                         shape=lind_mask.shape,
                                         dtype=lind_mask.dtype,
                                         initializer=lind_mask,
                                         order=self.order)
                num_lind = np.arange(rate_info['fall']['lind']['num'],
                                     dtype=kint_type)
                self.num_lind = creator('num_lind',
                                        shape=num_lind.shape,
                                        dtype=num_lind.dtype,
                                        initializer=num_lind,
                                        order=self.order)
                lind_inds = self.fall_to_thd_map.initializer[
                    self.lind_map.initializer]
                lind_ns_inds = np.where(
                    np.in1d(lind_inds, rate_info['thd']['has_ns']))[0].astype(
                        kint_type)
                lind_has_ns = self.lind_map.initializer[
                    lind_ns_inds]
                self.lind_has_ns = creator('lind_has_ns',
                                           shape=lind_has_ns.shape,
                                           dtype=lind_has_ns.dtype,
                                           initializer=lind_has_ns,
                                           order=self.order)
                self.lind_ns_inds = creator('lind_ns_inds',
                                            shape=lind_ns_inds.shape,
                                            dtype=lind_ns_inds.dtype,
                                            initializer=lind_ns_inds,
                                            order=self.order)

        # chebyshev
        if rate_info['cheb']['num']:
            self.cheb_numP = creator('cheb_numP',
                                     dtype=rate_info[
                                         'cheb']['num_P'].dtype,
                                     initializer=rate_info[
                                         'cheb']['num_P'],
                                     shape=rate_info[
                                         'cheb']['num_P'].shape,
                                     order=self.order)

            self.cheb_numT = creator('cheb_numT',
                                     dtype=rate_info[
                                         'cheb']['num_T'].dtype,
                                     initializer=rate_info[
                                         'cheb']['num_T'],
                                     shape=rate_info[
                                         'cheb']['num_T'].shape,
                                     order=self.order)

            # chebyshev parameters
            self.cheb_params = creator('cheb_params',
                                       dtype=rate_info['cheb'][
                                           'post_process']['params'].dtype,
                                       initializer=rate_info['cheb'][
                                           'post_process']['params'],
                                       shape=rate_info['cheb'][
                                           'post_process']['params'].shape,
                                       order=self.order)

            # limits for cheby polys
            self.cheb_Plim = creator('cheb_Plim',
                                     dtype=rate_info['cheb'][
                                         'post_process']['Plim'].dtype,
                                     initializer=rate_info['cheb'][
                                         'post_process']['Plim'],
                                     shape=rate_info['cheb'][
                                         'post_process']['Plim'].shape,
                                     order=self.order)
            self.cheb_Tlim = creator('cheb_Tlim',
                                     dtype=rate_info['cheb'][
                                         'post_process']['Tlim'].dtype,
                                     initializer=rate_info['cheb'][
                                         'post_process']['Tlim'],
                                     shape=rate_info['cheb'][
                                         'post_process']['Tlim'].shape,
                                     order=self.order)

            # workspace variables
            polymax = int(np.max(np.maximum(rate_info['cheb']['num_P'],
                                            rate_info['cheb']['num_T'])))
            self.cheb_pres_poly = creator('cheb_pres_poly',
                                          dtype=np.float64,
                                          shape=(polymax,),
                                          order=self.order,
                                          is_temporary=True,
                                          scope=scopes.PRIVATE)
            self.cheb_temp_poly = creator('cheb_temp_poly',
                                          dtype=np.float64,
                                          shape=(polymax,),
                                          order=self.order,
                                          is_temporary=True,
                                          scope=scopes.PRIVATE)

            # mask and map
            cheb_map = rate_info['cheb']['map'].astype(dtype=kint_type)
            self.cheb_map = creator('cheb_map',
                                    dtype=cheb_map.dtype,
                                    initializer=cheb_map,
                                    shape=cheb_map.shape,
                                    order=self.order)
            cheb_mask = _make_mask(cheb_map, rate_info['Nr'])
            self.cheb_mask = creator('cheb_mask',
                                     dtype=cheb_mask.dtype,
                                     initializer=cheb_mask,
                                     shape=cheb_mask.shape,
                                     order=self.order)
            num_cheb = np.arange(rate_info['cheb']['num'], dtype=kint_type)
            self.num_cheb = creator('num_cheb',
                                    dtype=num_cheb.dtype,
                                    initializer=num_cheb,
                                    shape=num_cheb.shape,
                                    order=self.order)

        # plog parameters, offsets, map / mask
        if rate_info['plog']['num']:
            self.plog_params = creator('plog_params',
                                       dtype=rate_info['plog'][
                                           'post_process']['params'].dtype,
                                       initializer=rate_info['plog'][
                                           'post_process']['params'],
                                       shape=rate_info['plog'][
                                           'post_process']['params'].shape,
                                       order=self.order)

            self.plog_num_param = creator('plog_num_param',
                                          dtype=rate_info['plog'][
                                              'num_P'].dtype,
                                          initializer=rate_info['plog'][
                                              'num_P'],
                                          shape=rate_info['plog'][
                                              'num_P'].shape,
                                          order=self.order)

            # mask and map
            plog_map = rate_info['plog']['map'].astype(dtype=kint_type)
            self.plog_map = creator('plog_map',
                                    dtype=plog_map.dtype,
                                    initializer=plog_map,
                                    shape=plog_map.shape,
                                    order=self.order)
            plog_mask = _make_mask(plog_map, rate_info['Nr'])
            self.plog_mask = creator('plog_mask',
                                     dtype=plog_mask.dtype,
                                     initializer=plog_mask,
                                     shape=plog_mask.shape,
                                     order=self.order)
            num_plog = np.arange(rate_info['plog']['num'], dtype=kint_type)
            self.num_plog = creator('num_plog',
                                    dtype=num_plog.dtype,
                                    initializer=num_plog,
                                    shape=num_plog.shape,
                                    order=self.order)

        # thermodynamic properties
        self.a_lo = creator('a_lo',
                            dtype=rate_info['thermo']['a_lo'].dtype,
                            initializer=rate_info['thermo']['a_lo'],
                            shape=rate_info['thermo']['a_lo'].shape,
                            order=self.order)
        self.a_hi = creator('a_hi',
                            dtype=rate_info['thermo']['a_hi'].dtype,
                            initializer=rate_info['thermo']['a_hi'],
                            shape=rate_info['thermo']['a_hi'].shape,
                            order=self.order)
        self.T_mid = creator('T_mid',
                             dtype=rate_info['thermo']['T_mid'].dtype,
                             initializer=rate_info['thermo']['T_mid'],
                             shape=rate_info['thermo']['T_mid'].shape,
                             order=self.order)
        for name in [constant_pressure_specific_heat, constant_volume_specific_heat,
                     internal_energy_array, enthalpy_array,
                     rate_const_thermo_coeff_array, 'dcp', 'dcv', 'db']:
            setattr(self, name, creator(name,
                                        dtype=np.float64,
                                        shape=(test_size, rate_info['Ns']),
                                        order=self.order))
        # thermo arrays
        self.spec_energy = self.h.copy() if self.conp else self.u.copy()
        self.spec_energy_ns = self.spec_energy.copy()
        self.spec_energy_ns.fixed_indicies = [(1, rate_info['Ns'] - 1)]
        self.spec_heat = self.cp if self.conp else self.cv
        self.spec_heat_ns = self.spec_heat.copy()
        self.spec_heat_ns.fixed_indicies = [(1, rate_info['Ns'] - 1)]
        self.spec_heat_total = creator(
            self.spec_heat.name + '_tot', shape=(test_size,),
            dtype=np.float64, order=self.order)
        self.dspec_heat = getattr(self, 'd' + self.spec_heat.name).copy()
