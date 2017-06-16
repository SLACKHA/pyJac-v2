# -*- coding: utf-8 -*-
"""Contains various utility classes for creating loopy arrays
and indexing / mapping
"""


import loopy as lp
import numpy as np
import copy
from loopy.kernel.data import temp_var_scope as scopes

problem_size = lp.ValueArg('problem_size', dtype=np.int32)
"""
    The problem size variable for non-testing
"""


class domain_transform(object):

    """
    A searchable representation of a domain transformation

    The idea here is that we have a current domain (A).
    If some other variable needs a different domain (B), we store:

        iname - The iname this transform operates on
        new_domain - the transformed domain
        insn - The transform instruction
        new_iname - The new transformed iname


    Parameters
    ----------
    iname : str
        The iname to map
    transform_insn : str
        The tranform instruction
    new_domain : :class:`creator`
        The new domain, to be transformed to
    new_iname : str
        The updated 'iname'
    """

    def __init__(self, iname, transform_insn, new_domain, new_iname):
        self.iname = iname
        self.transform_insn = transform_insn
        self.new_domain = new_domain
        self.new_iname = new_iname


class MapStore(object):

    """
    This class manages maps and masks for inputs / outputs in kernels

    Attributes
    ----------

    transformed_domains : list of :class:`domain_transform`
        used in string index / map intruction / mask instruction creation
    transformed_variables : dict
        Dictionary of :class:`creator` to
        :class:`domain_transform` that represents the developed maps
    loopy_opts : :class:`LoopyOptions`
        The loopy options for kernel creation
    knl_type : ['map', 'mask']
        The kernel mapping / masking type.  Controls whether this kernel should
        generate maps vs masks and the index ranges
    map_domain : :class:`creator`
        The domain of the iname to use for a mapped kernel
    mask_domain : :class:`creator`
        The domain of the iname to use for a masked kernel
    loop_index : str
        The loop index to work with
    have_input_map : bool
        If true, the input map domain needs a map for expression
    transform_insns : set
        A set of transform instructions generated for this :class:`MapStore`
    """

    def __init__(self, loopy_opts, map_domain, mask_domain, loop_index='i'):
        self.loopy_opts = loopy_opts
        self.knl_type = loopy_opts.knl_type
        self.map_domain = map_domain.copy()
        self.mask_domain = mask_domain.copy()
        self._check_is_valid_domain(self.map_domain)
        self._check_is_valid_domain(self.mask_domain)
        self.transformed_domains = []
        self.transformed_variables = {}
        from pytools import UniqueNameGenerator
        self.taken_transform_names = UniqueNameGenerator()
        self.loop_index = loop_index
        self.have_input_map = False
        self.transform_insns = set()

        if self._is_map() and not self._is_contiguous(self.map_domain):
            # need an input map
            self._add_input_map()

    def _is_map(self):
        """
        Return true if map kernel
        """

        return self.knl_type == 'map'

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
            np.arange(self.map_domain.initializer.size, dtype=np.int32)

        # update
        self.map_domain = new_map_domain

    def _add_transform(self, map_domain, iname, affine=None,
                       force_inline=False):
        """
        Convertes the map_domain (affine int or :class:`creator`)
        to a :class:`domain_transform` and adds it to this object
        """

        # add the transformed inames, instruction and map
        new_iname = self._get_transform_iname(self.loop_index +
                                              ('_map' if self._is_map()
                                               else '_mask'))

        if affine is not None:
            transform_insn = self.generate_transform_instruction(
                iname, new_iname, affine=affine, force_inline=force_inline)
        elif force_inline:
            raise Exception("Can't force inline for a non-affine"
                            " transformation.")
        else:
            transform_insn = self.generate_transform_instruction(
                iname, new_iname, map_domain.name)

        # update instruction list
        if not force_inline:
            self.transform_insns |= set([transform_insn])
        else:
            # directly place the transform in the new iname for inline access
            new_iname = transform_insn
            # and update transform insn
            transform_insn = ''

        # and store
        dt = domain_transform(
            iname, transform_insn, map_domain, new_iname)
        self.transformed_domains.append(dt)
        return dt

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

        assert dcheck.shape == ncheck.shape, (
            "Can't map domains of differing sizes")

        # check equal
        if np.array_equal(dcheck, ncheck):
            return None, None

        # check for affine map
        if np.all(ncheck - dcheck == (ncheck - dcheck)[0]):
            return new_domain, (ncheck - dcheck)[0]

        # finally return map
        return new_domain, None

    def _check_is_valid_domain(self, domain):
        """Makes sure the domain passed is a valid :class:`creator`"""
        assert domain is not None, 'Invalide domain'
        assert isinstance(domain, creator), ('Domain'
                                             ' must be of type `creator`')
        assert domain.name is not None, ('Domain must have initialized name')
        assert domain.initializer is not None, (
            'Cannot use non-initialized creator {} as domain!'.format(
                domain.name))

        if not self._is_map():
            # need to check that the maximum value is smaller than the base
            # mask domain size
            assert np.max(domain.initializer) < \
                self.mask_domain.initializer.size, (
                    "Mask entries for domain {} cannot be outside of "
                    "domain size {}".format(domain.name,
                                            self.mask_domain.initializer.size))

    def _is_contiguous(self, domain):
        """Returns true if domain can be expressed with a simple for loop"""
        indicies = domain.initializer
        return indicies[0] + indicies.size - 1 == indicies[-1]

    def _get_transform_if_exists(self, iname, domain):
        return next((x for x in self.transformed_domains if
                     np.array_equal(domain, x.new_domain) and
                     iname == x.iname), None)

    def _check_add_map(self, iname, domain, force_inline=False):
        """
        Checks and adds a map if necessary

        Parameters
        ----------
        iname : str
            The index to map
        domain : :class:`creator`
            The domain to check
        force_inline : bool
            If true, the resulting transform should be an inline, affine
            transformation

        Returns
        -------
        transform : :class:`domain_transform`
            The resulting transform, or None if not added
        """
        base = self._get_base_domain()

        # get mapping
        mapping, affine = self._get_map_transform(base, domain)

        # if we actually need a mapping
        if mapping is not None:
            # see if this map already exists
            mapv = self._get_transform_if_exists(iname, mapping)

            # if no map
            if mapv is None:
                # check if we need an input mapping
                if not self.have_input_map and affine is None and \
                        self.map_domain.initializer[0] != 0:
                    self._add_input_map()

                    # recheck map
                    mapv = self._get_transform_if_exists(iname, mapping)
                    if mapv is not None:
                        return mapv

                # need a new map, so add
                return self._add_transform(mapping, iname, affine=affine,
                                           force_inline=force_inline)
            return mapv

        return None

    def _check_add_mask(self, iname, domain, force_inline=False):
        """
        Checks and adds a mask if necessary

        Parameters
        ----------
        iname : str
            The index to map
        domain : :class:`creator`
            The domain to check
        force_inline : bool
            If true, the resulting transform should be an inline, affine
            transformation

        Returns
        -------
        transform : :class:`domain_transform`
            The resulting transform, or None if not added
        """
        base = self._get_base_domain()

        # get mask transform
        masking, affine = self._get_mask_transform(base, domain)

        # check if the masks match
        if masking is not None:
            # check if we have an matching mask
            maskv = self._get_transform_if_exists(iname, domain)

            if not maskv:
                # need to add a mask
                maskv = self._add_transform(domain, iname, affine=affine,
                                            force_inline=force_inline)
            return maskv

        return None

    def _get_base_domain(self):
        """
        Conviencience method to get domain agnostic of map / mask type
        """
        if self.knl_type == 'map':
            return self.map_domain
        elif self.knl_type == 'mask':
            return self.mask_domain
        else:
            raise NotImplementedError

    def check_and_add_transform(self, variable, domain, iname='i',
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
            The iname to transform
        force_inline : bool
            If True, the developed transform (if any) must be expressed as an
            inline transform.  If the transform is not affine, an exception
            will be raised

        Returns
        -------
        transform : :class:`domain_transform`
            The resulting transform, or None if not added
        """

        # make sure this domain is valid
        self._check_is_valid_domain(domain)

        transform = None
        if self.knl_type == 'map':
            transform = self._check_add_map(iname, domain,
                                            force_inline=force_inline)
        elif self.knl_type == 'mask':
            transform = self._check_add_mask(iname, domain,
                                             force_inline=force_inline)
        else:
            raise NotImplementedError

        if transform is not None:
            if isinstance(variable, list):
                for var in variable:
                    # add all variable mappings
                    self.transformed_variables[var] = transform
            else:
                # add this variable mapping
                self.transformed_variables[variable] = transform

        return transform

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

        affine = kwargs.pop('affine', None)

        var_affine = 0
        if variable.affine is not None:
            var_affine = variable.affine

        have_affine = var_affine or affine

        def __get_affine(iname):
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

        if variable in self.transformed_variables:
            indicies = tuple(__get_affine(x) if x !=
                             self.transformed_variables[variable].iname else
                             __get_affine(
                                 self.transformed_variables[variable].new_iname)
                             for x in indicies)
        elif have_affine and len(indicies) == 1:
            # if we don't have a map, but we do have an affine index
            # and it's obvious who to apply to
            indicies = (__get_affine(indicies[0]),)
        elif have_affine and isinstance(affine, dict):
            indicies = tuple(__get_affine(i) for i in indicies)
        elif have_affine:
            raise Exception("Can't apply affine transformation to indicies, {}"
                            " as the index to apply to cannot be"
                            " determined".format(indicies))

        return variable(*indicies, **kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def generate_transform_instruction(self, oldname, newname, map_arr='',
                                       affine='',
                                       force_inline=False):
        """
        Generates a loopy instruction that maps oldname -> newname via the
        mapping array

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
            A strings to be used `loopy.Instruction`'s) for
                    given mapping
        """

        try:
            affine = ' + ' + str(int(affine))
            if force_inline:
                return oldname + affine
        except:
            pass

        if not map_arr:
            return '<> {newname} = {oldname}{affine}'.format(
                newname=newname,
                oldname=oldname,
                affine=affine)

        return '<> {newname} = {mapper}[{oldname}]{affine}'.format(
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
            return (self.loop_index, fmt_str.format(
                    ind=self.loop_index,
                    start=base.initializer[0],
                    end=base.initializer[-1]))
        else:
            return (self.loop_index, fmt_str.format(
                    ind=self.loop_index,
                    start=0,
                    end=base.initializer.size - 1))


class creator(object):

    """
    The generic namestore interface, allowing easy access to
    loopy object creation, mapping, masking, etc.
    """

    def __init__(self, name, dtype, shape, order,
                 initializer=None,
                 scope=scopes.GLOBAL,
                 fixed_indicies=None,
                 is_temporary=False,
                 affine=None):
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
        scope : :class:`loopy.temp_var_scope`
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
        return lp.TemporaryVariable(self.name,
                                    shape=self.shape,
                                    initializer=self.initializer,
                                    scope=self.scope,
                                    read_only=self.initializer is not None,
                                    dtype=self.dtype,
                                    order=self.order,
                                    **kwargs)

    def __glob_arg_creator(self, **kwargs):
        return lp.GlobalArg(self.name,
                            shape=self.shape,
                            dtype=self.dtype,
                            order=self.order,
                            **kwargs)

    def __call__(self, *indicies, **kwargs):
        inds = self.__get_indicies(*indicies)
        lp_arr = self.creator(**kwargs)
        return (lp_arr, lp_arr.name + '[{}]'.format(', '.join(
            str(x) for x in inds)))

    def copy(self):
        return copy.deepcopy(self)


def _make_mask(map_arr, mask_size):
    """
    Create a mask array from the given map and total mask size
    """

    assert len(map_arr.shape) == 1, "Can't make mask from 2-D array"

    mask = np.full(mask_size, -1, dtype=np.int32)
    mask[map_arr] = np.arange(map_arr.size, dtype=np.int32)
    return mask


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

    def __init__(self, loopy_opts, rate_info, conp=True,
                 test_size='problem_size'):
        self.loopy_opts = loopy_opts
        self.rate_info = rate_info
        self.order = loopy_opts.order
        self.test_size = test_size
        self.conp = conp
        self._add_arrays(rate_info, test_size)

    def __getattr__(self, name):
        """
        Override of getattr such that NameStore.nonexistantkey -> None
        """
        try:
            return super(NameStore, self).__getattr__(self, name)
        except AttributeError:
            return None

    def __check(self, add_map=True):
        """ Ensures that maps are only added to map kernels etc. """
        if add_map:
            assert self.loopy_opts.knl_type == 'map', ('Cannot add'
                                                       ' map to mask kernel')
        else:
            assert self.loopy_opts.knl_type == 'mask', ('Cannot add'
                                                        ' mask to map kernel')

    def __make_offset(self, arr):
        """
        Creates an offset array from the given array
        """

        assert len(arr.shape) == 1, "Can't make offset from 2-D array"
        assert arr.dtype == np.int32, "Offset arrays should be integers!"

        return np.array(np.concatenate(
            (np.cumsum(arr) - arr, np.array([np.sum(arr)]))),
            dtype=np.int32)

    def _add_arrays(self, rate_info, test_size):
        """
        Initialize the various arrays needed for the namestore
        """

        # problem size
        if isinstance(test_size, str):
            self.problem_size = problem_size

        # generic ranges
        self.num_specs = creator('num_specs', shape=(rate_info['Ns'],),
                                 dtype=np.int32, order=self.order,
                                 initializer=np.arange(rate_info['Ns'],
                                                       dtype=np.int32))
        self.num_specs_no_ns = creator('num_specs_no_ns',
                                       shape=(rate_info['Ns'] - 1,),
                                       dtype=np.int32, order=self.order,
                                       initializer=np.arange(
                                           rate_info['Ns'] - 1,
                                           dtype=np.int32))
        self.num_reacs = creator('num_reacs', shape=(rate_info['Nr'],),
                                 dtype=np.int32, order=self.order,
                                 initializer=np.arange(rate_info['Nr'],
                                                       dtype=np.int32))
        self.phi_spec_inds = creator('phi_spec_inds',
                                     shape=(rate_info['Ns'] - 1,),
                                     dtype=np.int32, order=self.order,
                                     initializer=np.arange(2,
                                                           rate_info['Ns'] + 1,
                                                           dtype=np.int32))
        # state arrays
        self.T_arr = creator('phi', shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 0)])

        # handle extra variable and P / V arrays
        self.E_arr = creator('phi', shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 1)])
        if self.conp:
            self.P_arr = creator('P_arr', shape=(test_size,),
                                 dtype=np.float64, order=self.order)
            self.V_arr = self.E_arr
        else:
            self.P_arr = self.E_arr
            self.V_arr = creator('V_arr', shape=(test_size,),
                                 dtype=np.float64, order=self.order)

        self.n_arr = creator('phi', shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order)
        self.conc_arr = creator('conc', shape=(test_size, rate_info['Ns']),
                                dtype=np.float64, order=self.order)
        self.conc_ns_arr = creator('conc', shape=(test_size, rate_info['Ns']),
                                   dtype=np.float64, order=self.order,
                                   fixed_indicies=[(1, rate_info['Ns'] - 1)])
        self.n_dot = creator('dphi', shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order)
        self.T_dot = creator('dphi', shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 0)])
        self.E_dot = creator('dphi', shape=(test_size, rate_info['Ns'] + 1),
                             dtype=np.float64, order=self.order,
                             fixed_indicies=[(1, 1)])

        self.jac = creator('jac',
                           shape=(
                               test_size, rate_info['Ns'] + 1, rate_info['Ns'] + 1),
                           order=self.order,
                           dtype=np.float64)

        self.spec_rates = creator('wdot', shape=(test_size, rate_info['Ns']),
                                  dtype=np.float64, order=self.order)

        # molecular weights
        self.mw_arr = creator('mw', shape=(rate_info['Ns'],),
                              initializer=rate_info['mws'],
                              dtype=np.float64,
                              order=self.order)

        self.mw_post_arr = creator('mw_factor', shape=(rate_info['Ns'] - 1,),
                                   initializer=rate_info['mw_post'],
                                   dtype=np.float64,
                                   order=self.order)

        # thermo arrays
        self.h_arr = creator('h', shape=(test_size, rate_info['Ns']),
                             dtype=np.float64, order=self.order)
        self.u_arr = creator('u', shape=(test_size, rate_info['Ns']),
                             dtype=np.float64, order=self.order)
        self.cv_arr = creator('cv', shape=(test_size, rate_info['Ns']),
                              dtype=np.float64, order=self.order)
        self.cp_arr = creator('cp', shape=(test_size, rate_info['Ns']),
                              dtype=np.float64, order=self.order)
        self.b_arr = creator('b', shape=(test_size, rate_info['Ns']),
                             dtype=np.float64, order=self.order)

        # net species rates data

        # per reaction
        self.rxn_to_spec = creator('rxn_to_spec',
                                   dtype=np.int32,
                                   shape=rate_info['net'][
                                       'reac_to_spec'].shape,
                                   initializer=rate_info[
                                       'net']['reac_to_spec'],
                                   order=self.order)
        off = self.__make_offset(rate_info['net']['num_reac_to_spec'])
        self.rxn_to_spec_offsets = creator('net_reac_to_spec_offsets',
                                           dtype=np.int32,
                                           shape=off.shape,
                                           initializer=off,
                                           order=self.order)
        self.rxn_to_spec_reac_nu = creator('reac_to_spec_nu',
                                           dtype=np.int32, shape=rate_info[
                                               'net']['nu'].shape,
                                           initializer=rate_info['net']['nu'],
                                           order=self.order,
                                           affine=1)
        self.rxn_to_spec_prod_nu = creator('reac_to_spec_nu',
                                           dtype=np.int32, shape=rate_info[
                                               'net']['nu'].shape,
                                           initializer=rate_info['net']['nu'],
                                           order=self.order)

        self.rxn_has_ns = creator('rxn_has_ns',
                                  dtype=np.int32,
                                  shape=rate_info['reac_has_ns'].shape,
                                  initializer=rate_info['reac_has_ns'],
                                  order=self.order)

        # per species
        self.net_nonzero_spec = creator('net_nonzero_spec', dtype=np.int32,
                                        shape=rate_info['net_per_spec'][
                                            'map'].shape,
                                        initializer=rate_info[
                                            'net_per_spec']['map'],
                                        order=self.order)
        self.net_nonzero_phi = creator('net_nonzero_phi', dtype=np.int32,
                                       shape=rate_info['net_per_spec'][
                                           'map'].shape,
                                       initializer=rate_info[
                                           'net_per_spec']['map'] + 2,
                                       order=self.order)

        self.spec_to_rxn = creator('spec_to_rxn', dtype=np.int32,
                                   shape=rate_info['net_per_spec'][
                                       'reacs'].shape,
                                   initializer=rate_info[
                                       'net_per_spec']['reacs'],
                                   order=self.order)
        off = self.__make_offset(rate_info['net_per_spec']['reac_count'])
        self.spec_to_rxn_offsets = creator('spec_to_rxn_offsets',
                                           dtype=np.int32,
                                           shape=off.shape,
                                           initializer=off,
                                           order=self.order)
        self.spec_to_rxn_nu = creator('spec_to_rxn_nu',
                                      dtype=np.int32, shape=rate_info[
                                          'net_per_spec']['nu'].shape,
                                      initializer=rate_info[
                                          'net_per_spec']['nu'],
                                      order=self.order)

        # rop's and fwd / rev / thd maps
        self.rop_net = creator('rop_net',
                               dtype=np.float64,
                               shape=(test_size, rate_info['Nr']),
                               order=self.order)

        self.rop_fwd = creator('rop_fwd',
                               dtype=np.float64,
                               shape=(test_size, rate_info['Nr']),
                               order=self.order)

        if rate_info['rev']['num']:
            self.rop_rev = creator('rop_rev',
                                   dtype=np.float64,
                                   shape=(
                                       test_size, rate_info['rev']['num']),
                                   order=self.order)
            self.rev_map = creator('rev_map',
                                   dtype=np.int32,
                                   shape=rate_info['rev']['map'].shape,
                                   initializer=rate_info[
                                       'rev']['map'],
                                   order=self.order)

            mask = _make_mask(rate_info['rev']['map'],
                              rate_info['Nr'])
            self.rev_mask = creator('rev_mask',
                                    dtype=np.int32,
                                    shape=mask.shape,
                                    initializer=mask,
                                    order=self.order)

        if rate_info['thd']['num']:
            self.pres_mod = creator('pres_mod',
                                    dtype=np.float64,
                                    shape=(
                                        test_size, rate_info['thd']['num']),
                                    order=self.order)
            self.thd_map = creator('thd_map',
                                   dtype=np.int32,
                                   shape=rate_info['thd']['map'].shape,
                                   initializer=rate_info[
                                       'thd']['map'],
                                   order=self.order)

            mask = _make_mask(rate_info['thd']['map'],
                              rate_info['Nr'])
            self.thd_mask = creator('thd_mask',
                                    dtype=np.int32,
                                    shape=mask.shape,
                                    initializer=mask,
                                    order=self.order)

            thd_inds = np.arange(rate_info['thd']['num'], dtype=np.int32)
            self.thd_inds = creator('thd_inds',
                                    dtype=np.int32,
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
        num_simple = np.arange(rate_info['simple']['num'], dtype=np.int32)
        self.num_simple = creator('num_simple',
                                  dtype=np.int32,
                                  shape=num_simple.shape,
                                  initializer=num_simple,
                                  order=self.order)

        # simple map
        self.simple_map = creator('simple_map',
                                  dtype=np.int32,
                                  shape=rate_info['simple']['map'].shape,
                                  initializer=rate_info['simple']['map'],
                                  order=self.order)
        # simple mask
        simple_mask = _make_mask(rate_info['simple']['type'],
                                 rate_info['Nr'])
        self.simple_mask = creator('simple_mask',
                                   dtype=simple_mask.dtype,
                                   shape=simple_mask.shape,
                                   initializer=simple_mask,
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
                dtype=np.int32)
            setattr(self, 'simple_rtype_{}_inds'.format(rtype),
                    creator('simple_rtype_{}_inds'.format(rtype),
                            dtype=inds.dtype,
                            shape=inds.shape,
                            initializer=inds,
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
            mapv = np.array(mapv, dtype=np.int32)
            if not np.array_equal(mapv, rate_info['thd']['map']):
                self.thd_only_map = creator('thd_only_map',
                                            dtype=np.int32,
                                            shape=mapv.shape,
                                            initializer=mapv,
                                            order=self.order)

                mask = _make_mask(mapv, rate_info['Nr'])
                self.thd_only_mask = creator('thd_only_mask',
                                             dtype=np.int32,
                                             shape=mask.shape,
                                             initializer=mask,
                                             order=self.order)

            thd_eff_ns = rate_info['thd']['eff_ns']
            num_specs = rate_info['thd']['spec_num'].astype(dtype=np.int32)
            spec_list = rate_info['thd']['spec'].astype(
                dtype=np.int32)
            thd_effs = rate_info['thd']['eff']

            # finally create arrays
            self.thd_eff = creator('thd_eff',
                                   dtype=thd_effs.dtype,
                                   shape=thd_effs.shape,
                                   initializer=thd_effs,
                                   order=self.order)
            self.thd_eff_ns = creator('thd_eff_ns',
                                      dtype=thd_eff_ns.dtype,
                                      shape=thd_eff_ns.shape,
                                      initializer=thd_eff_ns,
                                      order=self.order)
            num_thd = np.arange(rate_info['thd']['num'], dtype=np.int32)
            self.num_thd = creator('num_thd',
                                   dtype=num_thd.dtype,
                                   shape=num_thd.shape,
                                   initializer=num_thd,
                                   order=self.order)
            self.thd_has_ns = creator('thd_has_ns',
                                      dtype=rate_info['thd']['has_ns'].dtype,
                                      shape=rate_info['thd']['has_ns'].shape,
                                      initializer=rate_info['thd']['has_ns'],
                                      order=self.order)
            num_thd_has_ns = np.arange(rate_info['thd']['has_ns'].size,
                                       dtype=np.int32)
            self.num_thd_has_ns = creator('num_thd_has_ns',
                                          dtype=num_thd_has_ns.dtype,
                                          shape=num_thd_has_ns.shape,
                                          initializer=num_thd_has_ns,
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
                # find the map in global reaction index
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
                inds = np.where(rate_info['fall']['map'] == mapv)[0].astype(
                    dtype=np.int32)
                setattr(self, 'fall_rtype_{}_inds'.format(rtype),
                        creator('fall_rtype_{}_inds'.format(rtype),
                                dtype=inds.dtype,
                                shape=inds.shape,
                                initializer=inds,
                                order=self.order))

            # maps
            self.fall_map = creator('fall_map',
                                    dtype=np.int32,
                                    initializer=rate_info['fall']['map'],
                                    shape=rate_info['fall']['map'].shape,
                                    order=self.order)

            num_fall = np.arange(rate_info['fall']['num'], dtype=np.int32)
            self.num_fall = creator('num_fall',
                                    dtype=np.int32,
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
                )[0], dtype=np.int32)
            self.fall_to_thd_map = creator('fall_to_thd_map',
                                           dtype=np.int32,
                                           initializer=fall_to_thd_map,
                                           shape=fall_to_thd_map.shape,
                                           order=self.order)

            fall_to_thd_mask = _make_mask(fall_to_thd_map,
                                          rate_info['Nr'])
            self.fall_to_thd_mask = creator('fall_to_thd_mask',
                                            dtype=np.int32,
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
                                     dtype=np.int32)
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
                                    dtype=np.int32)
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
            cheb_map = rate_info['cheb']['map'].astype(dtype=np.int32)
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
            num_cheb = np.arange(rate_info['cheb']['num'], dtype=np.int32)
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
            plog_map = rate_info['plog']['map'].astype(dtype=np.int32)
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
            num_plog = np.arange(rate_info['plog']['num'], dtype=np.int32)
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
        for name in ['cp', 'cv', 'u', 'h', 'b']:
            setattr(self, name, creator(name,
                                        dtype=np.float64,
                                        shape=(test_size, rate_info['Ns']),
                                        order=self.order))
