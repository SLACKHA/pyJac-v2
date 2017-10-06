"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess


# Related modules
import numpy as np

try:
    import cantera as ct
except ImportError:
    print('Error: Cantera must be installed.')
    raise

# Local imports
from ..core.mech_interpret import read_mech_ct
from ..pywrap.pywrap_gen import generate_wrapper

from ..tests.test_utils import parse_split_index, _run_mechanism_tests, runner
from ..tests import test_utils
from ..libgen import build_type

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
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                self.descriptor, state['lang'], state['vecsize'], state['order'],
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
        self.T = data['T']
        self.P = data['P']
        self.V = data['V']
        self.moles = data['moles']
        self.phi_cp = self.get_phi(self.T, self.V, self.moles)
        self.phi_cv = self.get_phi(self.T, self.P, self.moles)
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

    def run(self, state, asplit, dirs, data_output):
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
        data_output: str
            The file to output the results to

        Returns
        -------
        None
        """

        # get the answer
        phi = self.phi_cp if state['conp'] else self.phi_cv
        param = self.P if state['conp'] else self.V
        self.helper.eval_answer(phi, self.P, self.V, state)

        my_test = dirs['test']
        my_build = dirs['build']
        my_obj = dirs['obj']

        # save args to dir
        def __saver(arr, name, namelist=None):
            myname = os.path.join(my_test, name + '.npy')
            np.save(myname, arr)
            if namelist is not None:
                namelist.append(myname)

        # generate wrapper
        generate_wrapper(state['lang'], my_build, build_dir=my_obj,
                         out_dir=my_test, platform=str(state['platform']),
                         output_full_rop=self.rtype == build_type.species_rates,
                         btype=self.rtype)

        # now generate the per run data
        offset = 0
        # store the error dict
        err_dict = {}
        while offset < self.num_conditions:
            this_run = int(
                np.floor(np.minimum(self.cond_per_run, self.num_conditions - offset)
                         / self.max_vec_width) * self.max_vec_width)
            # get arrays
            # make sure to remove the last species in order to conform
            # to expected data
            myphi = np.array(phi[offset:offset + this_run, :],
                             order=state['order'], copy=True)

            myphi, = asplit.split_numpy_arrays(myphi)
            myphi = myphi.flatten(order=state['order'])

            args = []
            __saver(myphi, 'phi', args)
            __saver(param[offset:offset + this_run], 'param', args)
            del myphi

            # get reference outputs
            out_names, ref_ans = self.helper.get_outputs(
                state, offset, this_run, asplit)

            # save for comparison
            testfiles = []
            for i in range(len(ref_ans)):
                out = ref_ans[i]
                # and flatten in correct order
                out = out.flatten(order=state['order'])
                __saver(out, out_names[i], testfiles)
                del out
            outf = [os.path.join(my_test, '{}_rate.npy'.format(name))
                    for name in out_names]

            # write the module tester
            with open(os.path.join(my_test, 'test.py'), 'w') as file:
                file.write(self.mod_test.safe_substitute(
                    package='pyjac_{}'.format(self.package_lang[state['lang']]),
                    input_args=', '.join('"{}"'.format(x) for x in args),
                    test_arrays=', '.join('\'{}\''.format(x) for x in testfiles),
                    non_array_args='{}, {}'.format(this_run, state['num_cores']),
                    call_name=str(self.rtype)[str(self.rtype).index('.') + 1:],
                    output_files=', '.join('\'{}\''.format(x) for x in outf),
                    looser_tols=','.join('[]' for x in testfiles),
                    rtol=1e-5,
                    atol=1e-8)
                )

            # call
            subprocess.check_call(['python{}.{}'.format(
                sys.version_info[0], sys.version_info[1]),
                os.path.join(my_test, 'test.py'),
                '1' if offset != 0 else '0'])

            # get error
            err_dict = self.helper.eval_error(
                this_run, state['order'], outf, out_names, ref_ans, err_dict)

            # cleanup
            for x in args + outf:
                os.remove(x)
            for x in testfiles:
                os.remove(x)
            os.remove(os.path.join(my_test, 'test.py'))

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
            # it's actually more accurate to set the density
            # (total concentration) due to the cantera internals
            D = P / (ct.gas_constant * T)

            # get the last species's concentrations as D - sum(other species)
            last_spec = np.expand_dims(D - np.sum(phi[:, 2:], axis=1), 1)
            moles = np.concatenate((phi[:, 2:], last_spec), axis=1)

            self.gas.basis = 'molar'
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(self.num_conditions):
                    if not i % 10000:
                        print(i)
                    self.gas.TDX = T[i], D[i], moles[i]
                    # now, since cantera normalizes these concentrations
                    # let's read them back
                    concs = self.gas.concentrations[:]
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
                    np.divide(-np.dot(h, spec_rates), np.dot(cp, concs),
                              out=self.conp_temperature_rates[i, :])
                    np.divide(-np.dot(u, spec_rates), np.dot(cv, concs),
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
            del moles

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
            out_check[i] = np.load(out_files[i])
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
        fwd_masked = parse_split_index(out_check[fwd_ind], self.thd_map, order)
        out_check[fwd_ind][fwd_masked] *= out_check[pmod_ind][parse_split_index(
            out_check[pmod_ind], np.arange(
                self.thd_map.size, dtype=np.int32), order)]
        # rev
        rev_masked = parse_split_index(out_check[rev_ind], self.rev_to_thd_map,
                                       order)
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
            err_size = int(np.prod(out.shape) / this_run)
            err_mask = parse_split_index(err_compare,
                                         np.arange(err_size, dtype=np.int32), order,
                                         axis=(1,))
            # get maximum relative error locations
            err_locs = np.argmax(err_compare[err_mask], axis=IC_axis)
            if err_locs.ndim >= 2:
                # C-split, need to convert to two 1-d arrays
                lrange = np.arange(err_locs[0].size, dtype=np.int32)
                fixed = [np.zeros(err_size, dtype=np.int32),
                         np.zeros(err_size, dtype=np.int32)]
                for i, x in enumerate(err_locs):
                    # find max in err_locs
                    ind = np.argmax(err_compare[x, [i], lrange])
                    fixed[0][i] = x[ind]
                    fixed[1][i] = ind
                err_mask = (fixed[0], err_mask[1], fixed[1])
            else:
                err_mask = tuple(
                    x if i != IC_axis else err_locs for i, x in enumerate(err_mask))

            # take err norm
            err_comp_store = err_compare[err_mask]
            err_inf = err[err_mask]
            if name == 'rop_net':
                # need to find the fwd / rop error at the max locations
                # here
                rop_fwd_err = np.abs(out_check[fwd_ind][err_mask] -
                                     reference_answers[fwd_ind][err_mask])

                rop_rev_err = np.zeros(rop_fwd_err.size)
                rev_mask = tuple(x[self.rev_map] for x in err_mask)
                our_rev_mask = tuple((
                    x if not np.allclose(x, self.rev_map)
                    else np.arange(self.rev_map.size) for x in rev_mask))
                rop_rev_err[self.rev_map] = np.abs(
                    out_check[rev_ind][our_rev_mask] -
                    reference_answers[rev_ind][our_rev_mask])
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
                mods, Nr, current_vecwidth)
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
    def __init__(self, gas, num_conditions, atol=1e0, rtol=1e-8):
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

    def __fast_jac(self, conp):
        if conp and hasattr(self, 'fd_jac_cp'):
            return self.fd_jac_cp
        elif not conp and hasattr(self, 'fd_jac_cv'):
            return self.fd_jac_cv
        return None

    def eval_answer(self, phi, P, V, state):
        jac = self.__fast_jac(state['conp'])
        if jac is not None:
            return jac

        # need the sympy equations
        from ..sympy_utils.sympy_interpreter import load_equations
        _, conp_eqs = load_equations(True)
        _, conv_eqs = load_equations(False)

        # create the "store" for the AD-jacobian eval
        self.store = type('', (object,), {
            'reacs': self.reacs,
            'specs': self.specs,
            'conp_eqs': conp_eqs,
            'conv_eqs': conv_eqs,
            'phi_cp': phi.copy() if state['conp'] else None,
            'phi_cv': phi.copy() if not state['conp'] else None,
            'P': P,
            'V': V,
            'test_size': self.num_conditions
            })

        from ..tests.test_jacobian import _get_fd_jacobian
        jac = _get_fd_jacobian(self, self.num_conditions, state['conp'])
        if state['conp']:
            self.fd_jac_cp = jac
        else:
            self.fd_jac_cv = jac

        return jac

    def get_outputs(self, state, offset, this_run, asplit):
        output_names = ['jac']
        out_arrays = [self.__fast_jac(state['conp'])]
        return output_names, asplit.split_numpy_arrays(out_arrays)

    def eval_error(self, this_run, order, out_files, out_names, reference_answers,
                   err_dict):
        def __get_test(name):
            return reference_answers[out_names.index(name)]

        out_check = out_files[:]
        # load output arrays
        for i in range(len(out_files)):
            out_check[i] = np.load(out_files[i])
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
            assert np.isclose(err_dict['jac_zero'], 0)
            # norm suggested by lapack
            err_dict['jac_lapack'] = np.linalg.norm(err) / np.linalg.norm(
                denom)

            # thresholded error
            threshold = np.where(np.abs(out) > np.linalg.norm(out) / 1.e15)
            err_dict['jac_thresholded'] = np.linalg.norm(
                err[threshold] / denom[threshold])

            # try weighted
            err_dict['jac_weighted'] = np.linalg.norm(err / (
                self.atol + self.rtol * denom))

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
            names = ['jac']
            mods = ['', '_zero', '_lapack', '_thresholded', '_weighted']
            # check that we have all expected keys, and there is no nan's, etc.
            self._check_file(err, names, mods)
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
