# -*- coding: utf-8 -*-
"""Contains various utility classes for creating loopy arrays
and indexing / mapping
"""


import loopy as lp
import numpy as np
import copy
from loopy.kernel.data import temp_var_scope as scopes

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
    """

    def __init__(self, loopy_opts, map_domain, mask_domain, loop_index):
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

        if not self._is_contiguous(self.map_domain):
            #need an input map
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

        # and update
        new_map_domain.name = new_creator_name
        new_map_domain.initializer = np.arange(self.map_domain.initializer.size,
            dtype=np.int32)

        # and update
        self.map_domain = new_map_domain


    def _add_transform(self, map_domain, iname, affine=None):
        """
        Convertes the map_domain (affine int or :class:`creator`)
        to a :class:`domain_transform` and adds it to this object
        """

        #add the transformed inames, instruction and map
        new_iname = self._get_transform_iname(self.loop_index +
            ('_map' if self._is_map() else '_mask'))

        if affine is not None:
            transform_insn = self.generate_transform_instruction(
                iname, new_iname, affine=affine)
        else:
            transform_insn = self.generate_transform_instruction(
                iname, new_iname, map_domain.name)

        #and store
        dt = domain_transform(
            iname, transform_insn, map_domain, new_iname)
        self.transformed_domains.append(dt)
        return dt


    def _get_transform_iname(self, iname):
        """Returns a new iname"""
        return self.taken_transform_names(iname)


    def _get_map_transform(self, domain, new_domain):
        """
        Get the appropriate map transform between two given domains.
        Most likely, this will be a numpy array, but it may be an affine
        mapping

        Parameters
        ----------
        domain : :class:`numpy.ndarray` or :class:`creator`
            The domain to map
        new_domain: :class:`numpy.ndarray` or :class:`creator`
            The domain to map to

        Returns
        -------
        None if domains are equivalent
        Affine `str` map if an affine transform is possible
        :class:`np.ndarray` if a more complex map is required
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
        assert isinstance(domain, creator), ('Domain'
            ' must be of type `creator`')
        assert domain.name is not None, ('Domain must have initialized name')
        assert domain.initializer is not None, ('Cannot use ',
            'non-initialized creator {} as domain!'.format(domain.name))

        if not self._is_map():
            # need to check that the maximum value is smaller than the base
            # mask domain size
            assert np.max(domain.initializer) < \
                self.mask_domain.initializer.size, ("Mask entries for domain "
                    "{} cannot be outside of domain size {}".format(domain.name,
                        self.mask_domain.initializer.size))


    def _is_contiguous(self, domain):
        """Returns true if domain can be expressed with a simple for loop"""
        indicies = domain.initializer
        return indicies[0] + indicies.size - 1 == indicies[-1]


    def _get_transform_if_exists(self, iname, domain):
        return next((x for x in self.transformed_domains if
                np.array_equal(domain, x.new_domain) and
                iname == x.iname), None)


    def _check_add_map(self, iname, domain):
        """
        Checks and adds a map if necessary

        Parameters
        ----------
        iname : str
            The index to map
        domain : :class:`creator`
            The domain to check

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

            #if no map
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
                return self._add_transform(mapping, iname, affine=affine)

        return None


    def _check_add_mask(self, iname, domain):
        """
        Checks and adds a mask if necessary

        Parameters
        ----------
        iname : str
            The index to map
        domain : :class:`creator`
            The domain to check

        Returns
        -------
        transform : :class:`domain_transform`
            The resulting transform, or None if not added
        """
        base = self._get_base_domain()

        #check if the masks match
        if not np.array_equal(base.initializer, domain.initializer):
            # check if we have an matching mask
            maskv = self._get_transform_if_exists(iname, domain)

            if not maskv:
                # need to add a mask
                maskv = self._add_transform(domain, iname)
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


    def check_and_add_transform(self, variable, iname, domain):
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
        iname : str
            The iname to transform
        domain : :class:`creator`
            The domain of the variable to check.
            Note: this must be an initialized creator (i.e. temporary variable)

        Returns
        -------
        None
        """

        # make sure this domain is valid
        self._check_is_valid_domain(domain)

        transform = None
        if self.knl_type == 'map':
            transform = self._check_add_map(iname, domain)
        elif self.knl_type == 'mask':
            transform = self._check_add_mask(iname, domain)
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


    def apply_maps(self, variable, *indicies):
        """
        Applies the developed iname mappings to the indicies supplied and
        returns the created loopy Arg/Temporary and the string version

        Parameters
        ----------
        variable : :class:`creator`
            The NameStore variable(s) to work with
        indices : list of str
            The inames to map

        Returns
        -------
        lp_var : :class:`loopy.GlobalArg` or :class:`loopy.TemporaryVariable`
            The generated variable
        lp_str : str
            The string indexed variable
        transform_str : str
            The transform instruction string
        """

        if variable in self.transformed_variables:
            indicies = tuple(x if x !=
                    self.transformed_variables[variable].iname else
                    self.transformed_variables[variable].new_iname
                for x in indicies)
            return (*variable(indicies), \
                self.transformed_variables[variable].transform_insn)

        return (*variable(*indicies), None)


    def generate_transform_instruction(self, oldname, newname, map_arr='',
            affine=''):
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

        Returns
        -------
        map_inst : str
            A strings to be used `loopy.Instruction`'s) for
                    given mapping
        """

        try:
            affine = ' + ' + str(int(affine))
        except:
            pass

        if not map_arr:
            return '<>{newname} = {oldname}{affine}'.format(
                newname=newname,
                oldname=oldname,
                affine=affine)

        return '<>{newname} = {mapper}[{oldname}]{affine}'.format(
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


class creator(object):
    """
    The generic namestore interface, allowing easy access to
    loopy object creation, mapping, masking, etc.
    """
    def __init__(self, name, dtype, shape, order,
            initializer=None,
            scope=scopes.GLOBAL,
            fixed_indicies=None):
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
        if fixed_indicies is not None:
            self.fixed_indicies = fixed_indicies[:]
        if initializer is not None:
            self.creator = self.__temp_var_creator
            assert dtype == initializer.dtype, ('Incorrect dtype specified'
                ' for {}, got: {} expected: {}'.format(
                    name, initializer.dtype, dtype))
            assert shape == initializer.shape, ('Incorrect shape specified'
                ' for {}, got: {} expected: {}'.format(
                    name, initializer.shape, shape))
        else:
            self.creator = self.__glob_arg_creator

    def __get_indicies(self, indicies):
        if self.fixed_indicies:
            inds = [None for i in self.shape]
            for i, v in self.fixed_indicies:
                inds[i] = v
            empty = [i for i, x in enumerate(inds) if x == None]
            assert len(empty) == len(indicies), (
                'Wrong number of '
                'indicies supplied for {}: expected {} got {}'.format(
                self.name, len(empty), len(indicies)))
            for i, ind in enumerate(empty):
                empty[ind] = indicies[i]
        else:
            assert len(indicies) == self.num_indicies, ('Wrong number of '
            'indicies supplied for {}: expected {} got {}'.format(
                self.name, len(self.shape), len(indicies)))
            return indicies[:]

    def __temp_var_creator(self, **kwargs):
        return lp.TemporaryVariable(self.name,
            shape=self.shape,
            initializer=self.initializer,
            scope=self.scope,
            read_only=True,
            dtype=self.dtype,
            **kwargs)

    def __glob_arg_creator(self, **kwargs):
        return lp.GlobalArg(self.name,
                    shape=self.shape,
                    dtype=self.dtype,
                    **kwargs)

    def __call__(self, indicies, **kwargs):
        inds = self.__get_indicies(indicies)
        lp_arr = self.creator(**kwargs)
        return (lp_arr, lp_arr.name + '[{}]'.format(','.join(
            str(x) for x in inds)))


    def copy(self):
        return copy.deepcopy(self)


def _make_mask(map_arr, mask_size):
    """
    Create a mask array from the given map and total mask size
    """

    mask = np.full(mask_size, -1, dtype=np.int32)
    mask[map_arr] = map_arr[:]
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
    test_size : str or int
        Optional size used in testing.  If not supplied, this is a kernel arg
    """

    def __init__(self, loopy_opts, rate_info, test_size='problem_size'):
        self.loopy_opts = loopy_opts
        self.rate_info = rate_info
        self.order = loopy_opts.order
        self.test_size = test_size
        self.__add_arrays(test_size)

    def __getattribute__(self, name):
        """
        Override of getattr such that NameStore.nonexistantkey -> None
        """
        if hasattr(self, name):
            return super(NameStore, self).__getattribute__(self, name)
        else:
            return None

    def __check(self, add_map=True):
        """ Ensures that maps are only added to map kernels etc. """
        if add_map:
            assert self.loopy_opts.knl_type == 'map', ('Cannot '
                'add map to mask kernel')
        else:
            assert self.loopy_opts.knl_type == 'mask', ('Cannot '
                'add mask to map kernel')

    def __make_offset(self, arr):
        """
        Creates an offset array from the given array
        """

        assert len(arr.shape) == 1, "Can't make offset from 2-D array"
        assert arr.dtype == np.int32, "Offset arrays should be integers!"

        return np.array(np.concatentate(
            (np.cumsum(arr), np.array([np.sum(arr) + arr[-1]]))),
        dtype=np.int32)

    def __add_arrays(self, test_size):
        """
        Initialize the various arrays needed for the namestore
        """

        #state arrays
        self.T_arr = creator('phi', shape=(test_size, self.rate_info['Ns'] + 1),
            dtype=np.float64, order=self.order, fixed_indicies=[(0, 0)])
        self.P_arr = creator('P_arr', shape=(test_size,),
            dtype=np.float64, order=self.order)
        self.conc_arr = creator('phi', shape=(test_size, self.rate_info['Ns'] + 1),
            dtype=np.float64, order=self.order)
        self.phi_dot = creator('dphi', shape=(test_size, self.rate_info['Ns'] + 1),
            dtype=np.float64, order=self.order)

        #thermo arrays
        self.h_arr = creator('h', shape=(test_size, self.rate_info['Ns']),
            dtype=np.float64, order=self.order)
        self.u_arr = creator('u', shape=(test_size, self.rate_info['Ns']),
            dtype=np.float64, order=self.order)
        self.cv_arr = creator('cv', shape=(test_size, self.rate_info['Ns']),
            dtype=np.float64, order=self.order)
        self.cp_arr = creator('cp', shape=(test_size, self.rate_info['Ns']),
            dtype=np.float64, order=self.order)
        self.b_arr = creator('b', shape=(test_size, self.rate_info['Ns']),
            dtype=np.float64, order=self.order)

        #net species rates data

        #per reaction
        self.net_reac_to_spec_map = creator('net_reac_to_spec', dtype=np.int32,
            shape=len(self.rate_info['net']['reac_to_spec']), order=self.order)
        off = self.__make_offset(self.rate_info['net']['num_reac_to_spec'])
        self.net_reac_to_spec_offsets = creator('net_reac_to_spec_offsets',
            dtype=np.int32, shape=off.shape, initializer=off, order=self.order)
        self.net_reac_to_spec_nu = creator('net_reac_to_spec_nu',
            dtype=np.int32, shape=self.rate_info['net']['nu'].shape,
            initializer=self.rate_info['net']['nu'], order=self.order)

        #per species
        self.net_spec_to_reac = creator('net_spec_to_reac', dtype=np.int32,
            shape=len(self.rate_info['net_per_spec']['reacs']),
            order=self.order)
        off = self.__make_offset(self.rate_info['net_per_spec']['reac_count'])
        self.net_spec_to_reac_offsets = creator('net_reac_to_spec_offsets',
            dtype=np.int32, shape=off.shape, initializer=off, order=self.order)
        self.net_spec_to_reac_nu = creator('net_spec_to_reac_nu',
            dtype=np.int32, shape=self.rate_info['net_per_spec']['nu'].shape,
            initializer=self.rate_info['net_per_spec']['nu'], order=self.order)

        #rop's and fwd / rev / thd maps
        self.rop_net = creator('rop_net',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['Nr']),
                    order=self.order)

        self.rop_fwd = creator('rop_fwd',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['Nr']),
                    order=self.order)

        if self.rate_info['rev']['num']:
            self.rop_rev = creator('rop_rev',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['rev']['num']),
                    order=self.order)
            self.rev_map = creator('rev_map',
                    dtype=np.int32,
                    shape=self.rate_info['rev']['map'].shape,
                    initializer=self.rate_info['rev']['map'].shape,
                    order=self.order)

            mask = _make_mask(self.rate_info['rev']['map'],
                self.rate_info['Nr'])
            self.rev_mask = creator('rev_mask',
                    dtype=np.int32,
                    shape=mask.shape,
                    initializer=mask.shape,
                    order=self.order)


        if self.rate_info['thd']['num']:
            self.pres_mod = creator('pres_mod',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['thd']['num']),
                    order=self.order)
            self.rev_map = creator('thd_map',
                    dtype=np.int32,
                    shape=self.rate_info['thd']['map'].shape,
                    initializer=self.rate_info['thd']['map'].shape,
                    order=self.order)

            mask = _make_mask(self.rate_info['thd']['map'],
                self.rate_info['Nr'])
            self.thd_mask = creator('thd_mask',
                    dtype=np.int32,
                    shape=mask.shape,
                    initializer=mask.shape,
                    order=self.order)

        #fwd / rev rop data
        self.nu_fwd = creator('nu_fwd',
                    dtype=np.int32,
                    shape=self.rate_info['fwd']['nu'].shape,
                    initializer=self.rate_info['fwd']['nu'],
                    order=self.order)

        if self.rate_info['rev']['num']:
            self.nu_rev = creator('nu_rev',
                    dtype=np.int32,
                    shape=self.rate_info['rev']['nu'].shape,
                    initializer=self.rate_info['rev']['nu'],
                    order=self.order)

        #reaction data (fwd / rev rates, KC)
        self.kf = creator('kf',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['Nr']),
                    order=self.order)

        #simple reaction parameters
        self.simple_A = creator('simple_A',
                    dtype=self.rate_info['simple']['A'].dtype,
                    shape=self.rate_info['simple']['A'].shape,
                    initializer=self.rate_info['simple']['A'],
                    order=self.order)
        self.simple_beta = creator('simple_beta',
                    dtype=self.rate_info['simple']['beta'].dtype,
                    shape=self.rate_info['simple']['beta'].shape,
                    initializer=self.rate_info['simple']['beta'],
                    order=self.order)
        self.simple_Ta = creator('simple_Ta',
                    dtype=self.rate_info['simple']['Ta'].dtype,
                    shape=self.rate_info['simple']['Ta'].shape,
                    initializer=self.rate_info['simple']['Ta'],
                    order=self.order)
        #reaction types
        self.simple_rtype = creator('simple_rtype',
                    dtype=self.rate_info['simple']['rtype'].dtype,
                    shape=self.rate_info['simple']['rtype'].shape,
                    initializer=self.rate_info['simple']['rtype'],
                    order=self.order)

        #simple mask
        simple_mask = _make_mask(rate_info['simple']['rtype'],
            self.rate_info['Nr'])
        self.simple_mask = creator('simple_mask',
                    dtype=simple_mask.dtype,
                    shape=simple_mask.shape,
                    initializer=simple_mask,
                    order=self.order)

        #rtype maps
        for rtype in np.unique(self.rate_info['simple']['rtype']):
            #find the map
            mapv = np.where(rate_info['simple']['type'] == rtype)[0].astype(
                dtype=np.int32)
            setattr(self, 'simple_rtype_{}_map'.format(rtype),
                creator('simple_rtype_{}_map'.format(rtype),
                    dtype=mapv.dtype,
                    shape=mapv.shape,
                    initializer=mapv,
                    order=self.order))

        if self.rate_info['rev']['num']:
            self.kr = creator('kr',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['Nr']),
                    order=self.order)

            self.Kc = creator('Kc',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['Nr']),
                    order=self.order)

            self.reac_to_spec_nu_sum = creator('reac_to_spec_nu_sum',
                    dtype=rate_info['net']['nu_sum'].dtype,
                    shape=rate_info['net']['nu_sum'].shape,
                    initializer=rate_info['net']['nu_sum'],
                    order=self.order)

        #third body concs, maps, efficiencies, types, species
        if self.rate_info['thd']['num']:
            #third body concentrations
            self.thd_conc = creator('thd_conc',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['thd']['num']),
                    order=self.order)

            #thd only indicies
            mapv = np.where(np.logical_not(np.in1d(self.rate_info['thd']['map'],
                        self.rate_info['fall']['map'])))[0]
            if not np.array_equal(mapv, self.rate_info['thd']['map']):
                self.thd_only_map = creator('thd_only_map',
                    dtype=np.int32,
                    shape=mapv.shape,
                    initializer=mapv,
                    order=self.order)

                mask = _make_mask(mapv, self.rate_info['Nr'])
                self.thd_only_mask = creator('thd_only_mask',
                    dtype=np.int32,
                    shape=mask.shape,
                    initializer=mask,
                    order=self.order)

            thd_eff_ns = self.rate_info['thd']['post_process']['eff_ns'].copy()
            num_specs = self.rate_info['thd']['post_process']['spec_num'].copy()
            spec_list = self.rate_info['thd']['post_process']['spec'].copy()
            thd_effs = self.rate_info['thd'['post_process']]['eff'].copy()

            #finally create arrays
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
            self.thd_type = creator('thd_type',
                dtype=self.rate_info['thd']['type'].dtype,
                shape=self.rate_info['thd']['type'].shape,
                initializer=self.rate_info['thd']['type'],
                order=self.order)
            self.thd_spec = creator('thd_spec',
                dtype=spec_list.dtype,
                shape=spec_list.shape,
                initializer=spec_list,
                order=self.order)
            thd_offset = self.__make_offset(thd_offset)
            self.thd_offset = creator('thd_offset',
                dtype=thd_offset.dtype,
                shape=thd_offset.shape,
                initializer=thd_offset,
                order=self.order)

        #falloff rxn rates, blending vals, reduced pressures, maps
        if self.rate_info['fall']['num']:
            #falloff reaction parameters
            self.fall_A = creator('fall_A',
                        dtype=self.rate_info['fall']['A'].dtype,
                        shape=self.rate_info['fall']['A'].shape,
                        initializer=self.rate_info['fall']['A'],
                        order=self.order)
            self.fall_beta = creator('fall_beta',
                        dtype=self.rate_info['fall']['beta'].dtype,
                        shape=self.rate_info['fall']['beta'].shape,
                        initializer=self.rate_info['fall']['beta'],
                        order=self.order)
            self.fall_Ta = creator('fall_Ta',
                        dtype=self.rate_info['fall']['Ta'].dtype,
                        shape=self.rate_info['fall']['Ta'].shape,
                        initializer=self.rate_info['fall']['Ta'],
                        order=self.order)
            #reaction types
            self.fall_rtype = creator('fall_rtype',
                        dtype=self.rate_info['fall']['rtype'].dtype,
                        shape=self.rate_info['fall']['rtype'].shape,
                        initializer=self.rate_info['fall']['rtype'],
                        order=self.order)

            #simple mask
            simple_mask = _make_mask(rate_info['fall']['rtype'],
                self.rate_info['Nr'])
            self.fall_mask = creator('fall_mask',
                        dtype=simple_mask.dtype,
                        shape=simple_mask.shape,
                        initializer=simple_mask,
                        order=self.order)

            #rtype maps
            for rtype in np.unique(self.rate_info['fall']['rtype']):
                #find the map
                mapv = np.where(rate_info['fall']['type'] == rtype)[0].astype(
                    dtype=np.int32)
                setattr(self, 'fall_rtype_{}_map'.format(rtype),
                    creator('fall_rtype_{}_map'.format(rtype),
                        dtype=mapv.dtype,
                        shape=mapv.shape,
                        initializer=mapv,
                        order=self.order))

            #maps
            self.fall_map = creator('fall_map',
                    dtype=np.int32,
                    initializer=self.rate_info['fall']['map'],
                    shape=self.rate_info['fall']['map'].shape,
                    order=self.order)

            #blending
            self.Fi = creator('Fi',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['fall']['num']),
                    order=self.order)

            #reduced pressure
            self.Pr = creator('Pr',
                    dtype=np.float64,
                    shape=(test_size, self.rate_info['fall']['num']),
                    order=self.order)

            #types
            self.fall_type = creator('fall_type',
                    dtype=self.rate_info['fall']['ftype'].dtype,
                    shape=self.rate_info['fall']['ftype'].shape,
                    initializer=self.rate_info['fall']['ftype'],
                    order=self.order)

            #maps and masks
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
                self.rate_info['Nr'])
            self.fall_to_thd_mask = creator('fall_to_thd_mask',
                    dtype=np.int32,
                    initializer=fall_to_thd_mask,
                    shape=fall_to_thd_mask.shape,
                    order=self.order)

            if self.rate_info['fall']['troe']['num']:
                #Fcent, Atroe, Btroe
                self.Fcent = self.creator('Fcent',
                        shape=(test_size,
                            self.rate_info['fall']['troe']['num']),
                        dtype=np.float64,
                        order=self.order)

                self.Atroe = self.creator('Atroe',
                        shape=(test_size,
                            self.rate_info['fall']['troe']['num']),
                        dtype=np.float64,
                        order=self.order)

                self.Btroe = self.creator('Btroe',
                        shape=(test_size,
                            self.rate_info['fall']['troe']['num']),
                        dtype=np.float64,
                        order=self.order)

                #troe parameters
                self.troe_a = self.creator('troe_a',
                        shape=self.rate_info['fall']['troe']['a'].shape,
                        dtype=self.rate_info['fall']['troe']['a'].dtype,
                        initializer=self.rate_info['fall']['troe']['a'],
                        order=self.order)
                self.troe_T1 = self.creator('troe_T1',
                        shape=self.rate_info['fall']['troe']['T1'].shape,
                        dtype=self.rate_info['fall']['troe']['T1'].dtype,
                        initializer=self.rate_info['fall']['troe']['T1'],
                        order=self.order)
                self.troe_T3 = self.creator('troe_T3',
                        shape=self.rate_info['fall']['troe']['T3'].shape,
                        dtype=self.rate_info['fall']['troe']['T3'].dtype,
                        initializer=self.rate_info['fall']['troe']['T3'],
                        order=self.order)
                self.troe_T2 = self.creator('troe_T2',
                        shape=self.rate_info['fall']['troe']['T2'].shape,
                        dtype=self.rate_info['fall']['troe']['T2'].dtype,
                        initializer=self.rate_info['fall']['troe']['T2'],
                        order=self.order)

                #map and mask
                self.troe_map = self.creator('troe_map',
                        shape=self.rate_info['fall']['troe']['map'].shape,
                        dtype=self.rate_info['fall']['troe']['map'].dtype,
                        initializer=self.rate_info['fall']['troe']['map'],
                        order=self.order)
                troe_mask = _make_mask(troe_map, self.rate_info['Nr'])
                self.troe_mask = self.creator('troe_mask',
                        shape=troe_mask.shape,
                        dtype=troe_mask.dtype,
                        initializer=troe_mask,
                        order=self.order)

            if self.rate_info['fall']['sri']['num']:
                #X_sri
                self.X_sri = self.creator('X',
                        shape=(test_size,
                            self.rate_info['fall']['sri']['num']),
                        dtype=np.float64,
                        order=self.order)

                #sri parameters
                self.sri_a = self.creator('sri_a',
                        shape=self.rate_info['fall']['sri']['a'].shape,
                        dtype=self.rate_info['fall']['sri']['a'].dtype,
                        initializer=self.rate['fall']['sri']['a'],
                        order=self.order)
                self.sri_b = self.creator('sri_b',
                        shape=self.rate_info['fall']['sri']['b'].shape,
                        dtype=self.rate_info['fall']['sri']['b'].dtype,
                        initializer=self.rate['fall']['sri']['b'],
                        order=self.order)
                self.sri_c = self.creator('sri_c',
                        shape=self.rate_info['fall']['sri']['c'].shape,
                        dtype=self.rate_info['fall']['sri']['c'].dtype,
                        initializer=self.rate['fall']['sri']['c'],
                        order=self.order)
                self.sri_d = self.creator('sri_d',
                        shape=self.rate_info['fall']['sri']['d'].shape,
                        dtype=self.rate_info['fall']['sri']['d'].dtype,
                        initializer=self.rate['fall']['sri']['d'],
                        order=self.order)
                self.sri_e = self.creator('sri_e',
                        shape=self.rate_info['fall']['sri']['e'].shape,
                        dtype=self.rate_info['fall']['sri']['e'].dtype,
                        initializer=self.rate['fall']['sri']['e'],
                        order=self.order)

                #map and mask
                self.sri_map = self.creator('sri_map',
                        shape=self.rate_info['fall']['sri']['map'].shape,
                        dtype=self.rate_info['fall']['sri']['map'].dtype,
                        initializer=self.rate_info['fall']['sri']['map'],
                        order=self.order)
                sri_mask = _make_mask(sri_map, self.rate_info['Nr'])
                self.sri_mask = self.creator('sri_mask',
                        shape=sri_mask.shape,
                        dtype=sri_mask.dtype,
                        initializer=sri_mask,
                        order=self.order)

            if self.rate_info['fall']['lind']['num']:
                #lind map / mask
                self.lind_map = self.creator('lind_map',
                        shape=self.rate_info['fall']['lind']['map'].shape,
                        dtype=self.rate_info['fall']['lind']['map'].dtype,
                        initializer=self.rate_info['fall']['lind']['map'],
                        order=self.order)
                lind_mask = _make_mask(lind_map, self.rate_info['Nr'])
                self.lind_mask = self.creator('lind_mask',
                        shape=lind_mask.shape,
                        dtype=lind_mask.dtype,
                        initializer=lind_mask,
                        order=self.order)

        #chebyshev
        if self.rate_info['cheb']['num']:
            self.cheb_numP = creator('cheb_numP',
                dtype=self.rate_info['cheb']['num_P'].dtype,
                initializer=self.rate_info['cheb']['num_P'],
                shape=self.rate_info['cheb']['num_P'].shape,
                order=self.order)

            self.cheb_numT = creator('cheb_numT',
                dtype=self.rate_info['cheb']['num_T'].dtype,
                initializer=self.rate_info['cheb']['num_T'],
                shape=self.rate_info['cheb']['num_T'].shape,
                order=self.order)

            #chebyshev parameters
            self.cheb_params = creator('cheb_params',
                dtype=self.rate_info['cheb']['post_process']['params'].dtype,
                initializer=self.rate_info['cheb']['post_process']['params'],
                shape=self.rate_info['cheb']['post_process']['params'].shape,
                order=self.order)

            # limits for cheby polys
            self.cheb_Plim = creator('cheb_Plim',
                dtype=self.rate_info['cheb']['post_process']['Plim'].dtype,
                initializer=Plim,
                shape=Plim.shape,
                order=self.order)
            self.cheb_Tlim = creator('cheb_Tlim',
                dtype=Tlim.dtype,
                initializer=Tlim,
                shape=Tlim.shape,
                order=self.order)

            #mask and map
            cheb_map = self.rate_info['cheb']['map'].astype(dtype=np.int32)
            self.cheb_map = creator('cheb_map',
                dtype=cheb_map.dtype,
                initializer=cheb_map,
                shape=cheb_map.shape,
                order=self.order)
            cheb_mask = _make_mask(cheb_map, self.rate_info['Nr'])
            self.cheb_mask = creator('cheb_mask',
                dtype=cheb_mask.dtype,
                initializer=cheb_mask,
                shape=cheb_mask.shape,
                order=self.order)

        #plog parameters, offsets, map / mask
        if self.rate_info['plog']['num']:
            self.plog_params = creator('plog_params',
                dtype=self.rate_info['plog']['post_process']['params'].dtype,
                initializer=self.rate_info['plog']['post_process']['params'],
                shape=self.rate_info['plog']['post_process']['params'].shape,
                order=self.order)

            plog_num_params = self.__make_offset(self.rate_info['plog'][
                'num_P'])
            self.plog_num_params = creator('plog_num_params',
                dtype=plog_num_params.dtype,
                initializer=plog_num_params,
                shape=plog_num_params.shape,
                order=self.order)

            #mask and map
            plog_map = self.rate_info['plog']['map'].astype(dtype=np.int32)
            self.plog_map = creator('plog_map',
                dtype=plog_map.dtype,
                initializer=plog_map,
                shape=plog_map.shape,
                order=self.order)
            plog_mask = _make_mask(plog_map, self.rate_info['Nr'])
            self.plog_mask = creator('plog_mask',
                dtype=plog_mask.dtype,
                initializer=plog_mask,
                shape=plog_mask.shape,
                order=self.order)