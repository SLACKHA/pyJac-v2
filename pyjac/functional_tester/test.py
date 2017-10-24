"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import subprocess


# Related modules
import numpy as np
import numpy.ma as ma

try:
    import cantera as ct
except ImportError:
    print('Error: Cantera must be installed.')
    raise

# Local imports
from ..core.mech_interpret import read_mech_ct

from ..tests.test_utils import parse_split_index, _run_mechanism_tests, runner
from ..tests import test_utils
from ..loopy_utils.loopy_utils import JacobianFormat, RateSpecialization
from ..libgen import build_type, generate_library
from ..core.create_jacobian import determine_jac_inds
from ..utils import EnumType

# turn off cache
import loopy as lp
lp.set_caching_enabled(False)


def getf(x):
    return os.path.basename(x)


class validation_runner(runner):
    def __init__(self, eval_class, rtype=build_type.jacobian):
        """Runs validation testing for pyJac for a mechanism

        Properties
        ----------
        eval_class: :class:`eval`
            Evaluate the answer and error for the current state, called on every
            iteration
        rtype: :class:`build_type` [build_type.jacobian]
            The type of test to run
        """
        super(validation_runner, self).__init__(rtype)
        self.eval_class = eval_class
        self.package_lang = {'opencl': 'ocl',
                             'c': 'c'}
        self.mod_test = test_utils.get_run_source()

    def check_file(self, filename):
        """Checks file for existing data, returns number of completed runs

        Parameters
        ----------
        filename : str
            Name of file with data

        Returns
        -------
        completed : bool
            True if the file is complete

        """

        Ns = self.gas.n_species
        Nr = self.gas.n_reactions
        Nrev = len([x for x in self.gas.reactions() if x.reversible])
        return self.helper.check_file(filename, Ns, Nr, Nrev, self.current_vecwidth)

    def get_filename(self, state):
        self.current_vecwidth = state['vecsize']
        desc = self.descriptor
        if self.rtype == build_type.jacobian:
            desc += '_sparse' if EnumType(JacobianFormat)(state['sparse'])\
                 == JacobianFormat.sparse else '_full'
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                desc, state['lang'], state['vecsize'], state['order'],
                'w' if state['wide'] else 'd' if state['deep'] else 'par',
                state['platform'], state['rate_spec'],
                'split' if state['split_kernels'] else 'single',
                state['num_cores'], 'conp' if state['conp'] else 'conv') + '_err.npz'

    def pre(self, gas, data, num_conditions, max_vec_width):
        """
        Initializes the validation runner

        Parameters
        ----------
        gas: :class:`cantera.Solution`
            The cantera object representing this mechanism
        data: dict
            A dictionary with keys T, P, V and moles, representing the test data
            for this mechanism
        num_conditions: int
            The number of conditions to test
        max_vec_width: int
            The maximum vector width considered for this test. The number of
            conditions per run must be a multiple of this for proper functioning
        """

        self.gas = gas
        self.gas.basis = 'molar'
        self.T = data['T']
        self.P = data['P']
        self.V = data['V']
        self.moles = data['moles']
        self.phi_cp = self.get_phi(self.T, self.P, self.V, self.moles)
        self.phi_cv = self.get_phi(self.T, self.V, self.P, self.moles)
        self.num_conditions = num_conditions
        self.max_vec_width = max_vec_width

        self.helper = self.eval_class(gas, num_conditions)
        if self.rtype != build_type.jacobian:
            # find the number of conditions per run (needed to avoid memory
            # issues with i-pentanol model)
            max_per_run = 100000
            self.cond_per_run = int(
                np.floor(max_per_run / max_vec_width) * max_vec_width)
        else:
            self.cond_per_run = num_conditions

    @property
    def max_per_run(self):
        return 100000 if self.rtype == build_type.species_rates else None

    def run(self, state, asplit, dirs, phi_path, data_output):
        """
        Run the validation test for the given state

        Parameters
        ----------
        state: dict
            A dictionary containing the state of the current optimization / language
            / vectorization patterns, etc.
        asplit: :class:`array_splitter`
            The array splitter to use in modifying state arrays
        dirs: dict
            A dictionary of directories to use for building / testing, etc.
            Has the keys "build", "test", "obj" and "run"
        phi_path: str
            The path expected by the generated kernel for the state vector
            phi to be saved in (as a binary file)
        data_output: str
            The file to output the results to

        Returns
        -------
        None
        """

        # get the answer
        phi = self.phi_cp if state['conp'] else self.phi_cv
        self.helper.eval_answer(phi, self.P, self.V, state)

        my_test = dirs['test']
        my_build = dirs['build']
        my_obj = dirs['obj']

        # compile library as executable
        lib = generate_library(state['lang'], my_build, obj_dir=my_obj,
                               out_dir=my_test, btype=self.rtype, shared=True,
                               as_executable=True)

        # now generate the per run data
        offset = 0
        # store the error dict
        err_dict = {}
        while offset < self.num_conditions:
            this_run = int(
                np.floor(np.minimum(self.cond_per_run, self.num_conditions - offset)
                         / self.max_vec_width) * self.max_vec_width)

            # get phi array
            phi = np.array(phi[offset:offset + this_run], order='C', copy=True)
            # save to file for input
            phi.flatten('C').tofile(phi_path)

            # get reference outputs
            out_names, ref_ans = self.helper.get_outputs(
                state, offset, this_run, asplit)

            # save for comparison
            outf = [os.path.join(my_test, '{}.bin'.format(name))
                    for name in out_names]

            # call
            subprocess.check_call([os.path.join(my_test, lib),
                                   str(this_run), str(state['num_cores'])],
                                  cwd=my_test)

            # get error
            err_dict = self.helper.eval_error(
                this_run, state['order'], outf, out_names, ref_ans, err_dict)

            # cleanup
            for x in outf:
                os.remove(x)

            # finally update the offset
            offset += self.cond_per_run

        # and write to file
        np.savez(data_output, **err_dict)


class eval(object):
    def eval_answer(self, phi, param, state):
        raise NotImplementedError

    def eval_error(self, my_test, offset, this_run):
        raise NotImplementedError

    def get_outputs(self, state, offset, this_run, asplit):
        raise NotImplementedError

    def _check_file(self, err, names, mods):
        try:
            return all(n + mod in err and np.all(np.isfinite(err[n + mod]))
                       for n in names for mod in mods)
        except:
            return False


class spec_rate_eval(eval):
    """
    Helper class for the species rates tester
    """
    def __init__(self, gas, num_conditions, atol=1e-10, rtol=1e-6):
        self.atol = atol
        self.rtol = rtol
        self.molar_rates = np.zeros((num_conditions, gas.n_species - 1))
        self.conp_temperature_rates = np.zeros((num_conditions, 1))
        self.conv_temperature_rates = np.zeros((num_conditions, 1))
        self.conp_extra_rates = np.zeros((num_conditions, 1))
        self.conv_extra_rates = np.zeros((num_conditions, 1))
        self.h = np.zeros((gas.n_species))
        self.u = np.zeros((gas.n_species))
        self.cp = np.zeros((gas.n_species))
        self.cv = np.zeros((gas.n_species))
        self.num_conditions = num_conditions

        # get mappings
        self.fwd_map = np.array(range(gas.n_reactions))
        self.rev_map = np.array(
            [x for x in range(gas.n_reactions) if gas.is_reversible(x)])
        self.thd_map = []
        for x in range(gas.n_reactions):
            try:
                gas.reaction(x).efficiencies
                self.thd_map.append(x)
            except:
                pass
        self.thd_map = np.array(self.thd_map, dtype=np.int32)
        self.rop_fwd_test = np.zeros((num_conditions, self.fwd_map.size))
        self.rop_rev_test = np.zeros((num_conditions, self.rev_map.size))
        self.rop_net_test = np.zeros((num_conditions, self.fwd_map.size))
        # need special maps for rev/thd
        self.rev_to_thd_map = np.where(np.in1d(self.rev_map, self.thd_map))[0]
        self.thd_to_rev_map = np.where(np.in1d(self.thd_map, self.rev_map))[0]
        # it's a pain to actually calcuate this
        # and we don't need it directly, since cantera computes
        # pdep terms in the forward / reverse ROP automatically
        # hence we create it as a placeholder for the testing script
        self.pres_mod_test = np.zeros((num_conditions, self.thd_map.size))

        # molecular weight fraction
        self.mw_frac = 1 - gas.molecular_weights[:-1] / gas.molecular_weights[-1]

        # predefines
        self.specs = gas.species()[:]
        self.gas = gas
        self.evaled = False
        self.name = 'spec'

    def eval_answer(self, phi, P, V, state):
        def __eval_cp(j, T):
            return self.specs[j].thermo.cp(T)
        eval_cp = np.vectorize(__eval_cp, cache=True)

        def __eval_h(j, T):
            return self.specs[j].thermo.h(T)
        eval_h = np.vectorize(__eval_h, cache=True)

        if not self.evaled:
            ns_range = np.arange(self.gas.n_species)

            T = phi[:, 0]
            P = phi[:, 1] if state['conp'] else phi[:, 2]
            V = phi[:, 2] if state['conp'] else phi[:, 1]
            # it's actually more accurate to set the density
            # (total concentration) due to the cantera internals
            D = P / (ct.gas_constant * T)

            # get the last species's concentrations as D - sum(other species)
            concs = phi[:, 3:] / V[:, np.newaxis]
            last_spec = np.expand_dims(D - np.sum(concs, axis=1), 1)
            concs = np.concatenate((concs, last_spec), axis=1)

            self.gas.basis = 'molar'
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(self.num_conditions):
                    if not i % 10000:
                        print(i)
                    # first, set T / D
                    self.gas.TD = T[i], D[i]
                    # now set concentrations
                    self.gas.concentrations = concs[i]
                    # assert allclose
                    assert np.allclose(self.gas.T, T[i], atol=1e-12)
                    assert np.allclose(self.gas.density, D[i], atol=1e-12)
                    assert np.allclose(self.gas.concentrations, concs[i], atol=1e-12)
                    # get molar species rates
                    spec_rates = self.gas.net_production_rates[:]
                    self.molar_rates[i, :] = spec_rates[:-1] * V[i]
                    # info vars
                    self.rop_fwd_test[i, :] = self.gas.forward_rates_of_progress[:]
                    self.rop_rev_test[i, :] = self.gas.reverse_rates_of_progress[:][
                        self.rev_map]
                    self.rop_net_test[i, :] = self.gas.net_rates_of_progress[:]

                    # find temperature rates
                    cp = eval_cp(ns_range, T[i])
                    h = eval_h(ns_range, T[i])
                    cv = cp - ct.gas_constant
                    u = h - T[i] * ct.gas_constant
                    np.divide(-np.dot(h, spec_rates), np.dot(cp, concs[i]),
                              out=self.conp_temperature_rates[i, :])
                    np.divide(-np.dot(u, spec_rates), np.dot(cv, concs[i]),
                              out=self.conv_temperature_rates[i, :])

                    # finally find extra variable rates
                    self.conp_extra_rates[i] = V[i] * (
                        T[i] * ct.gas_constant * np.sum(
                            self.mw_frac * spec_rates[:-1]) / P[i] +
                        self.conp_temperature_rates[i, :] / T[i])
                    self.conv_extra_rates[i] = (
                        P[i] / T[i]) * self.conv_temperature_rates[i, :] + \
                        T[i] * ct.gas_constant * np.sum(
                            self.mw_frac * spec_rates[:-1])

            self.evaled = True

    def get_outputs(self, state, offset, this_run, asplit):
        conp = state['conp']
        output_names = ['dphi', 'rop_fwd', 'rop_rev', 'pres_mod', 'rop_net']
        temperature_rates = self.conp_temperature_rates if conp \
            else self.conv_temperature_rates
        extra_rates = self.conp_extra_rates if conp else self.conv_extra_rates
        dphi = np.concatenate((temperature_rates[offset:offset + this_run, :],
                               extra_rates[offset:offset + this_run, :],
                               self.molar_rates[offset:offset + this_run, :]),
                              axis=1)
        out_arrays = [dphi,
                      self.rop_fwd_test[offset:offset + this_run, :],
                      self.rop_rev_test[offset:offset + this_run, :],
                      self.pres_mod_test[offset:offset + this_run, :],
                      self.rop_net_test[offset:offset + this_run, :]]

        return output_names, asplit.split_numpy_arrays(out_arrays)

    def eval_error(self, this_run, order, out_files, out_names, reference_answers,
                   err_dict):
        def __get_test(name):
            return reference_answers[out_names.index(name)]

        out_check = out_files[:]
        # load output arrays
        for i in range(len(out_files)):
            out_check[i] = np.fromfile(out_files[i], dtype=np.float64)
            # check finite
            assert np.all(np.isfinite(out_check[i]))
            # and reshape to match test array
            out_check[i] = np.reshape(out_check[i], __get_test(out_names[i]).shape,
                                      order=order)

        # multiply fwd/rev ROP by pressure rates to compare w/ Cantera
        pmod_ind = next(
            i for i, x in enumerate(out_files) if 'pres_mod' in x)
        fwd_ind = next(
            i for i, x in enumerate(out_files) if 'rop_fwd' in x)
        rev_ind = next(
            i for i, x in enumerate(out_files) if 'rop_rev' in x)
        # fwd
        fwd_masked = parse_split_index(out_check[fwd_ind], self.thd_map, order)
        out_check[fwd_ind][fwd_masked] *= out_check[pmod_ind][parse_split_index(
            out_check[pmod_ind], np.arange(self.thd_map.size, dtype=np.int32),
            order)]
        # rev
        rev_masked = parse_split_index(out_check[rev_ind], self.rev_to_thd_map,
                                       order)
        # thd to rev map already in thd index list, so don't need to do arange
        out_check[rev_ind][rev_masked] *= out_check[pmod_ind][parse_split_index(
            out_check[pmod_ind], self.thd_to_rev_map, order)]

        # simple test to get IC index
        mask_dummy = parse_split_index(reference_answers[0], np.array([this_run]),
                                       order=order, axis=(0,))
        IC_axis = next(i for i, ax in enumerate(mask_dummy) if ax != slice(None))

        # load output
        for name, out in zip(*(out_names, out_check)):
            if name == 'pres_mod':
                continue
            check_arr = __get_test(name)
            # get err
            err = np.abs(out - check_arr)
            err_compare = err / (self.atol + self.rtol * np.abs(check_arr))

            # find the split, if any
            def __get_locs_and_mask(arr, locs=None, inds=None):
                size = int(np.prod(arr.shape) / this_run)
                if inds is None:
                    inds = np.arange(size, dtype=np.int32)
                mask = parse_split_index(arr, inds, order, axis=(1,))
                # get maximum relative error locations
                if locs is None:
                    locs = np.argmax(err_compare[mask], axis=IC_axis)
                if locs.ndim >= 2:
                    # C-split, need to convert to two 1-d arrays
                    lrange = np.arange(locs[0].size, dtype=np.int32)
                    fixed = [np.zeros(size, dtype=np.int32),
                             np.zeros(size, dtype=np.int32)]
                    for i, x in enumerate(locs):
                        # find max in err_locs
                        ind = np.argmax(err_compare[x, [i], lrange])
                        fixed[0][i] = x[ind]
                        fixed[1][i] = ind
                    mask = (fixed[0], mask[1], fixed[1])
                else:
                    mask = tuple(
                        x if i != IC_axis else locs for i, x in enumerate(mask))
                return locs, mask

            err_locs, err_mask = __get_locs_and_mask(err_compare)

            # take err norm
            err_comp_store = err_compare[err_mask]
            err_inf = err[err_mask]
            if name == 'rop_net':
                # need to find the fwd / rop error at the max locations
                # here
                rop_fwd_err = np.abs(out_check[fwd_ind][err_mask] -
                                     reference_answers[fwd_ind][err_mask])

                rop_rev_err = np.zeros(rop_fwd_err.size)
                # get err locs for rev reactions
                rev_err_locs = err_locs[self.rev_map]
                # get reversible mask using the error locations for the reversible
                # reactions, and the rev_map size for the mask
                _, rev_mask = __get_locs_and_mask(
                    out_check[rev_ind], locs=rev_err_locs,
                    inds=np.arange(self.rev_map.size))
                # and finally update
                rop_rev_err[self.rev_map] = np.abs(
                    out_check[rev_ind][rev_mask] -
                    reference_answers[rev_ind][rev_mask])
                # now find maximum of error in fwd / rev ROP
                rop_component_error = np.maximum(rop_fwd_err, rop_rev_err)

            if name not in err_dict:
                err_dict[name] = np.zeros_like(err_inf)
                err_dict[name + '_value'] = np.zeros_like(err_inf)
                err_dict[name + '_store'] = np.zeros_like(err_inf)
                if name == 'rop_net':
                    err_dict['rop_component'] = np.zeros_like(err_inf)
            # get locations to update
            update_locs = np.where(
                err_comp_store >= err_dict[name + '_store'])
            # and update
            err_dict[name][update_locs] = err_inf[update_locs]
            err_dict[
                name + '_store'][update_locs] = err_comp_store[update_locs]
            # need to take max and update precision as necessary
            if name == 'rop_net':
                err_dict['rop_component'][
                    update_locs] = rop_component_error[update_locs]
            # update the values for normalization
            err_dict[name + '_value'][update_locs] = check_arr[err_mask][update_locs]

        del out_check
        return err_dict

    def _check_size(self, err, names, mods, size, vecwidth):
        non_conformant = [n + mod for n in names for mod in mods
                          if err[n + mod].size != size]
        if not non_conformant:
            return True
        return all(np.all(err[x][size:] == 0) and err[x].size % vecwidth == 0
                   for x in non_conformant)

    def check_file(self, filename, Ns, Nr, Nrev, current_vecwidth):
        """
        Checks a species validation file for completion

        Parameters
        ----------
        filename: str
            The file to check
        Ns: int
            The number of species in the mechanism
        Nr: int
            The number of reactions in the mechanism
        Nrev: int
            The number of reversible reactions in the mechanism
        current_vecwidth: int
            The curent vector width being used.  If the current state results in
            an array split, this may make the stored error arrays larger than
            expected, so we must check that they are divisible by current_vecwidth
            and the extra entries are identically zero

        Returns
        -------
        valid: bool
            If true, the test case is complete and can be skipped
        """

        try:
            err = np.load(filename)
            names = ['rop_fwd', 'rop_rev', 'rop_net', 'dphi']
            mods = ['', '_value', '_store']
            # check that we have all expected keys, and there is no nan's, etc.
            allclear = self._check_file(err, names, mods)
            # check Nr size
            allclear = allclear and self._check_size(
                err, [x for x in names if ('rop_fwd' in x or 'rop_net' in x)],
                mods, Nr, current_vecwidth)
            # check reversible
            allclear = allclear and self._check_size(
                err, [x for x in names if 'rop_rev' in x],
                mods, Nrev, current_vecwidth)
            # check Ns size
            allclear = allclear and self._check_size(
                err, [x for x in names if 'phi' in x], mods, Ns + 1,
                current_vecwidth)
            return allclear
        except:
            return False


class jacobian_eval(eval):
    """
    Helper class for the Jacobian tester
    """
    def __init__(self, gas, num_conditions, atol=1e-2, rtol=1e-6):
        self.atol = atol
        self.rtol = rtol
        self.evaled = False

        self.num_conditions = num_conditions
        # read mech
        _, self.specs, self.reacs = read_mech_ct(gas=gas)

        # predefines
        self.gas = gas
        self.evaled = False
        self.name = 'jac'
        self.inds = determine_jac_inds(self.reacs, self.specs,
                                       RateSpecialization.fixed)[
            'jac_inds']

    def __sparsify(self, jac, order, check=True):
        # get the sparse indicies
        inds = self.inds['flat_' + order]
        if check:
            # check that our sparse indicies make sense
            mask = np.zeros(jac.shape, dtype=np.bool)
            # create a masked jacobian that only looks at our indicies
            mask[:, inds[:, 0], inds[:, 1]] = True
            mask = ma.array(jac, mask=mask)
            # check that no entry not in the mask is non-zero
            assert not np.any(jac[~mask.mask])
            del mask
            # and finally return the sparse array
        return np.asarray(jac[:, inds[:, 0], inds[:, 1]], order=order)

    def __fast_jac(self, conp, sparse, order, check=True):
        jac = None
        if conp and hasattr(self, 'fd_jac_cp'):
            jac = self.fd_jac_cp
        elif not conp and hasattr(self, 'fd_jac_cv'):
            jac = self.fd_jac_cv

        if jac is None:
            return None

        if sparse == 'sparse':
            return self.__sparsify(jac, order, check=check)
        return jac

    def eval_answer(self, phi, P, V, state):
        jac = self.__fast_jac(state['conp'], state['sparse'], state['order'])
        if jac is not None:
            return jac

        # create the "store" for the AD-jacobian eval
        # mask phi to get rid of parameter stored in there for data input
        phi_mask = np.array([0] + list(range(2, phi.shape[1])))
        self.store = type('', (object,), {
            'reacs': self.reacs,
            'specs': self.specs,
            'phi_cp': phi[:, phi_mask].copy() if state['conp'] else None,
            'phi_cv': phi[:, phi_mask].copy() if not state['conp'] else None,
            'P': P,
            'V': V,
            'test_size': self.num_conditions
            })

        from ..tests.test_jacobian import _get_fd_jacobian
        jac = _get_fd_jacobian(self, self.num_conditions, state['conp'])
        if state['conp']:
            self.fd_jac_cp = jac.copy()
        else:
            self.fd_jac_cv = jac.copy()

        if state['sparse'] == 'sparse':
            jac = self.__sparsify(jac, state['order'])

        return jac

    def get_outputs(self, state, offset, this_run, asplit):
        output_names = ['jac']
        jac = self.__fast_jac(state['conp'], state['sparse'], state['order'])
        jac = jac[offset:offset + this_run, :]
        return output_names, asplit.split_numpy_arrays([jac])

    def eval_error(self, this_run, order, out_files, out_names, reference_answers,
                   err_dict):
        def __get_test(name):
            return reference_answers[out_names.index(name)]

        out_check = out_files[:]
        # load output arrays
        for i in range(len(out_files)):
            out_check[i] = np.fromfile(out_files[i], dtype=np.float64)
            # check finite
            assert np.all(np.isfinite(out_check[i]))
            # and reshape to match test array
            out_check[i] = np.reshape(out_check[i], __get_test(out_names[i]).shape,
                                      order=order)

        err_dict = {}
        # load output
        for name, out in zip(*(out_names, out_check)):
            check_arr = __get_test(name)
            # get err
            err = np.abs(out - check_arr)
            denom = np.abs(check_arr)
            non_zero = np.where(np.abs(check_arr) > 0)
            zero = np.where(np.abs(check_arr) == 0)
            # regular frobenieus norm, have to filter out zero entries for our
            # norm here
            err_dict['jac'] = np.linalg.norm(
                err[non_zero] / denom[non_zero])
            err_dict['jac_zero'] = np.linalg.norm(
                err[zero])
            assert np.allclose(err_dict['jac_zero'], 0)
            # norm suggested by lapack
            err_dict['jac_lapack'] = np.linalg.norm(err) / np.linalg.norm(
                denom)

            # thresholded error
            threshold = np.where(np.abs(out) > np.linalg.norm(out) / 1.e20)
            thresholded_err = err[threshold] / denom[threshold]
            amax = np.argmax(thresholded_err)
            err_dict['jac_thresholded_20'] = np.linalg.norm(thresholded_err)
            del thresholded_err
            err_dict['jac_thresholded_20_PJ_amax'] = out[threshold][amax]
            err_dict['jac_thresholded_20_AD_amax'] = check_arr[threshold][amax]

            threshold = np.where(np.abs(out) > np.linalg.norm(out) / 1.e15)
            thresholded_err = err[threshold] / denom[threshold]
            amax = np.argmax(thresholded_err)
            err_dict['jac_thresholded_15'] = np.linalg.norm(thresholded_err)
            del thresholded_err
            err_dict['jac_thresholded_15_PJ_amax'] = out[threshold][amax]
            err_dict['jac_thresholded_15_AD_amax'] = check_arr[threshold][amax]
            del threshold

            # largest relative errors for different absolute toleratnces
            for mul in [1, 10, 100, 1000]:
                atol = self.atol * mul
                err_weighted = err / (atol + self.rtol * denom)
                amax = np.argmax(err_weighted)
                err_dict['jac_weighted_{}'.format(atol)] = np.linalg.norm(
                    err_weighted)
                del err_weighted
                err_dict['jac_weighted_{}_PJ_amax'.format(atol)] = out.flat[amax]
                err_dict['jac_weighted_{}_AD_amax'.format(atol)] = check_arr.flat[
                    amax]

            # info values for lookup
            err_dict['jac_max_value'] = np.amax(out)
            err_dict['jac_threshold_value'] = np.linalg.norm(out)

            # info values for lookup
            err_dict['jac_max_value'] = np.amax(out)
            err_dict['jac_threshold_value'] = np.linalg.norm(out)

        del out_check
        return err_dict

    def check_file(self, filename, Ns, Nr, Nrev, current_vecwidth):
        """
        Checks a jacobian validation file for completion

        Parameters
        ----------
        filename: str
            The file to check
        Ns: int
            Unused
        Nr: int
            Unused
        Nrev: int
            Unused
        current_vecwidth: int
            Unused

        Returns
        -------
        valid: bool
            If true, the test case is complete and can be skipped
        """

        try:
            err = np.load(filename)
            # check basic error stats
            names = ['jac']
            mods = ['', '_zero', '_lapack']
            # check that we have all expected keys, and there is no nan's, etc.
            allclear = self._check_file(err, names, mods)
            # check thresholded error
            names = ['jac_threshold_15', 'jac_thresholded_20']
            mods = ['', 'PJ_amax', 'AD_amax']
            # check that we have all expected keys, and there is no nan's, etc.
            allclear = allclear and self._check_file(err, names, mods)
            # check that we have the weighted jacobian error
            names = ['jac_weighted_' + x for x in ['0.01', '0.1', '1.0', '10.0']]
            mods = ['', 'PJ_amax', 'AD_amax']
            allclear = allclear and self._check_file(err, names, mods)
            # check for max / threshold value
            names = ['jac_']
            mods = ['max_value', 'threshold_value']
            allclear = allclear and self._check_file(err, names, mods)
            return allclear
        except:
            return False


def species_rate_tester(work_dir='error_checking'):
    """Runs validation testing on pyJac's species_rate kernel, reading a series
    of mechanisms and datafiles from the :param:`work_dir`, and outputting
    a numpy zip file (.npz) with the error of various outputs (rhs vector, ROP, etc.)
    as compared to Cantera, for each configuration tested.

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """

    valid = validation_runner(spec_rate_eval, build_type.species_rates)
    _run_mechanism_tests(work_dir, valid)


def jacobian_tester(work_dir='error_checking'):
    """Runs validation testing on pyJac's jacobian kernel, reading a series
    of mechanisms and datafiles from the :param:`work_dir`, and outputting
    a numpy zip file (.npz) with the error of Jacobian as compared to a
    autodifferentiated reference answer, for each configuration tested.

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """

    valid = validation_runner(jacobian_eval, build_type.jacobian)
    _run_mechanism_tests(work_dir, valid)
