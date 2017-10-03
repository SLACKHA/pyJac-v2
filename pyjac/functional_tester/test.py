"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
import shutil
import logging


# Related modules
import numpy as np

try:
    import cantera as ct
except ImportError:
    print('Error: Cantera must be installed.')
    raise

# Local imports
from .. import utils
from ..core.create_jacobian import create_jacobian, find_last_species
from ..core.array_creator import array_splitter
from ..core.mech_interpret import read_mech_ct
from ..pywrap.pywrap_gen import generate_wrapper

from ..tests.test_utils import data_bin_writer as dbw
from ..tests.test_utils import get_test_matrix as tm
from ..tests.test_utils import parse_split_index
from ..tests import test_utils
from ..libgen import build_type

# turn off cache
import loopy as lp
lp.set_caching_enabled(False)


def check_file(filename, Ns, Nr):
    """Checks file for existing data, returns number of completed runs

    Parameters
    ----------
    filename : str
        Name of file with data
    Ns : int
        The number of species in the mech
    Nr : int
        The number of reactions in the mech

    Returns
    -------
    completed : bool
        True if the file is complete

    """
    # new version
    try:
        np.load(filename)
        return True
    except:
        return False
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        complete = True
        for i, line in enumerate(lines):
            test = line[line.index(':') + 1:]
            filtered = [y.strip() for y in test.split(',') if y.strip()]
            complete = complete and len(filtered) == ((Ns + 1) if i < 2
                                                      else Nr)
            for x in filtered:
                float(x)
        return complete
    except:
        return False


def getf(x):
    return os.path.basename(x)


def __run_test(work_dir, eval_class, rtype=build_type.jacobian):
    """Runs validation testing for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    eval_class: :class:`eval`
        Evaluate the answer and error for the current state, called on every
        iteration
    rtype: :class:`build_type` [build_type.jacobian]
        The type of test to run

    Returns
    -------
    None

    """

    obj_dir = 'obj'
    build_dir = 'out'
    test_dir = 'test'

    work_dir = os.path.abspath(work_dir)
    mechanism_list, oploop, max_vec_width = tm.get_test_matrix(work_dir)

    if len(mechanism_list) == 0:
        print('No mechanisms found for performance testing in '
              '{}, exiting...'.format(work_dir)
              )
        sys.exit(-1)

    package_lang = {'opencl': 'ocl',
                    'c': 'c'}

    # load the module tester template
    mod_test = test_utils.get_run_source()

    for mech_name, mech_info in sorted(mechanism_list.items(),
                                       key=lambda x: x[1]['ns']):
        # ensure directory structure is valid
        this_dir = os.path.join(work_dir, mech_name)
        this_dir = os.path.abspath(this_dir)
        os.chdir(this_dir)
        my_obj = os.path.join(this_dir, obj_dir)
        my_build = os.path.join(this_dir, build_dir)
        my_test = os.path.join(this_dir, test_dir)
        utils.create_dir(my_obj)
        utils.create_dir(my_build)
        utils.create_dir(my_test)

        def __cleanup():
            # remove library
            test_utils.clean_dir(my_obj, False)
            # remove build
            test_utils.clean_dir(my_build, False)
            # clean sources
            test_utils.clean_dir(my_test, False)
            # clean dummy builder
            dist_build = os.path.join(os.getcwd(), 'build')
            if os.path.exists(dist_build):
                shutil.rmtree(dist_build)

        # get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))
        gas.basis = 'molar'

        # read our species for MW's
        _, specs, _ = read_mech_ct(gas=gas)

        # find the last species
        gas_map = find_last_species(specs, return_map=True)
        del specs
        # update the gas
        specs = gas.species()[:]
        gas = ct.Solution(thermo='IdealGas', kinetics='GasKinetics',
                          species=[specs[x] for x in gas_map],
                          reactions=gas.reactions())
        del specs

        # first load data to get species rates, jacobian etc.
        num_conditions, data = dbw.load(
            [], directory=os.path.join(work_dir, mech_name))

        # figure out the number of conditions to test
        num_conditions = int(
            np.floor(num_conditions / max_vec_width) * max_vec_width)
        # create the eval
        helper = eval_class(gas, num_conditions)
        if rtype != build_type.jacobian:
            # find the number of conditions per run (needed to avoid memory
            # issues with i-pentanol model)
            max_per_run = 100000
            cond_per_run = int(
                np.floor(max_per_run / max_vec_width) * max_vec_width)
        else:
            cond_per_run = num_conditions

        # set T / P arrays from data
        T = data[:num_conditions, 0].flatten()
        P = data[:num_conditions, 1].flatten()
        # set V = 1 such that concentrations == moles
        V = np.ones_like(P)

        # resize data
        moles = data[:num_conditions, 2:]
        # and reorder
        moles = moles[:, gas_map].copy()

        # set phi / params
        phi_cp = np.concatenate((np.reshape(T, (-1, 1)),
                                 np.reshape(V, (-1, 1)), moles), axis=1)
        phi_cv = np.concatenate((np.reshape(T, (-1, 1)),
                                 np.reshape(P, (-1, 1)), moles), axis=1)
        param_cp = P
        param_cv = V

        # begin iterations
        current_data_order = None
        done_parallel = False
        the_path = os.getcwd()
        op = oploop.copy()
        for i, state in enumerate(op):
            # remove any old builds
            __cleanup()
            lang = state['lang']
            vecsize = state['vecsize']
            order = state['order']
            wide = state['wide']
            deep = state['deep']
            platform = state['platform']
            rate_spec = state['rate_spec']
            split_kernels = state['split_kernels']
            num_cores = state['num_cores']
            conp = state['conp']
            if not (deep or wide) and done_parallel:
                # this is simple parallelization, don't need to repeat for
                # different vector sizes, simply choose one and go
                continue
            elif not (deep or wide):
                # mark done
                done_parallel = True

            if rate_spec == 'fixed' and split_kernels:
                continue  # not a thing!

            if deep and wide:
                # can't do both simultaneously
                continue

            data_output = ('{}_{}_{}_{}_{}_{}_{}_{}'.format(
                lang, vecsize, order, 'w' if wide else 'd' if deep else 'par',
                platform, rate_spec, 'split' if split_kernels else 'single',
                num_cores) + '_err.npz')

            # if already run, continue
            data_output = os.path.join(the_path, data_output)
            if check_file(data_output, gas.n_species, gas.n_reactions):
                continue

            # get an array splitter
            width = state['vecsize'] if state['wide'] else None
            depth = state['vecsize'] if state['deep'] else None
            order = state['order']
            asplit = array_splitter(type('', (object,), {
                'width': width, 'depth': depth, 'order': order}))

            # get the answer
            phi = phi_cp if conp else phi_cv
            param = param_cp if conp else param_cv
            helper.eval_answer(phi, P, V, state)

            if order != current_data_order:
                # rewrite data to file in 'C' order
                dbw.write(os.path.join(this_dir))

            # save args to dir
            def __saver(arr, name, namelist=None):
                myname = os.path.join(my_test, name + '.npy')
                np.save(myname, arr)
                if namelist is not None:
                    namelist.append(myname)

            #try:
            create_jacobian(lang,
                            gas=gas,
                            vector_size=vecsize,
                            wide=wide,
                            deep=deep,
                            data_order=order,
                            build_path=my_build,
                            skip_jac=True,
                            auto_diff=False,
                            platform=platform,
                            data_filename=os.path.join(this_dir, 'data.bin'),
                            split_rate_kernels=split_kernels,
                            rate_specialization=rate_spec,
                            split_rop_net_kernels=split_kernels,
                            output_full_rop=rtype == build_type.species_rates,
                            conp=conp,
                            use_atomics=state['use_atomics'])
            #except Exception as e:
            #    logging.exception(e)
            #    logging.warn('generation failed...')
            #    logging.warn(i, state)
            #    continue

            # generate wrapper
            generate_wrapper(lang, my_build, build_dir=obj_dir,
                             out_dir=my_test, platform=str(platform),
                             output_full_rop=rtype == build_type.species_rates,
                             btype=rtype)

            # now generate the per run data
            offset = 0
            # store the error dict
            err_dict = {}
            while offset < num_conditions:
                this_run = int(
                    np.floor(np.minimum(cond_per_run, num_conditions - offset)
                             / max_vec_width) * max_vec_width)
                # get arrays
                # make sure to remove the last species in order to conform
                # to expected data
                myphi = np.array(phi[offset:offset + this_run, :-1],
                                 order=order, copy=True)

                myphi, = asplit.split_numpy_arrays(myphi)
                myphi = myphi.flatten(order=order)

                args = []
                __saver(myphi, 'phi', args)
                __saver(param[offset:offset + this_run], 'param', args)
                del myphi

                # get reference outputs
                out_names, ref_ans = helper.get_outputs(state, offset, this_run,
                                                        asplit)

                # save for comparison
                testfiles = []
                for i in range(len(ref_ans)):
                    out = ref_ans[i]
                    # and flatten in correct order
                    out = out.flatten(order=order)
                    __saver(out, out_names[i], testfiles)
                    del out
                outf = [os.path.join(my_test, '{}_rate.npy'.format(name))
                        for name in out_names]

                # write the module tester
                with open(os.path.join(my_test, 'test.py'), 'w') as file:
                    file.write(mod_test.safe_substitute(
                        package='pyjac_{}'.format(package_lang[lang]),
                        input_args=', '.join('"{}"'.format(x) for x in args),
                        test_arrays=', '.join('\'{}\''.format(x) for x in testfiles),
                        non_array_args='{}, {}'.format(this_run, num_cores),
                        call_name=str(rtype)[str(rtype).index('.') + 1:],
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
                err_dict = helper.eval_error(
                    this_run, state['order'], outf, out_names, ref_ans, err_dict)

                # cleanup
                for x in args + outf:
                    os.remove(x)
                for x in testfiles:
                    os.remove(x)
                os.remove(os.path.join(my_test, 'test.py'))

                # finally update the offset
                offset += cond_per_run

            # and write to file
            np.savez(data_output, **err_dict)


class eval(object):
    def eval_answer(self, phi, param, state):
        raise NotImplementedError

    def eval_error(self, my_test, offset, this_run):
        raise NotImplementedError

    def get_outputs(self, state, offset, this_run, asplit):
        raise NotImplementedError


class spec_rate_eval(eval):
    """
    Helper class for the species rates tester
    """
    def __init__(self, gas, num_conditions, atol=1e-10, rtol=1e-6):
        self.atol = atol
        self.rtol = rtol
        self.evaled = False
        self.spec_rates = np.zeros((num_conditions, gas.n_species))
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

            self.gas.basis = 'molar'
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(self.num_conditions):
                    if not i % 10000:
                        print(i)
                    self.gas.TDX = T[i], D[i], phi[i, 2:]
                    # now, since cantera normalizes these concentrations
                    # let's read them back
                    concs = self.gas.concentrations[:]
                    # get molar species rates
                    self.spec_rates[i, :] = self.gas.net_production_rates[:] * V[i]
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
                    np.divide(-np.dot(h, self.spec_rates[i, :]), np.dot(cp, concs),
                              out=self.conp_temperature_rates[i, :])
                    np.divide(-np.dot(u, self.spec_rates[i, :]), np.dot(cv, concs),
                              out=self.conv_temperature_rates[i, :])

                    # finally find extra variable rates
                    self.conp_extra_rates[i] = V[i] * (
                        T[i] * ct.gas_constant * np.sum(
                            self.mw_frac * self.spec_rates[i, :-1]) / P[i] +
                        self.conp_temperature_rates[i, :] / T[i])
                    self.conv_extra_rates[i] = (
                        P[i] / T[i]) * self.conv_temperature_rates[i, :] + \
                        T[i] * ct.gas_constant * np.sum(
                            self.mw_frac * self.spec_rates[i, :-1])

            self.evaled = True

    def get_outputs(self, state, offset, this_run, asplit):
        conp = state['conp']
        output_names = ['dphi', 'rop_fwd', 'rop_rev', 'pres_mod', 'rop_net']
        temperature_rates = self.conp_temperature_rates if conp \
            else self.conv_temperature_rates
        extra_rates = self.conp_extra_rates if conp else self.conv_extra_rates
        dphi = np.concatenate((temperature_rates[offset:offset + this_run, :],
                               extra_rates[offset:offset + this_run, :],
                               self.spec_rates[offset:offset + this_run, :-1]),
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


def species_rate_tester(work_dir='error_checking'):
    """Runs performance testing for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """

    __run_test(work_dir, spec_rate_eval, build_type.species_rates)
