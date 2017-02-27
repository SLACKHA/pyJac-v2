"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import sys
import subprocess
import re
from argparse import ArgumentParser
import multiprocessing
import shutil
import gc
from collections import defaultdict, OrderedDict

from string import Template

# Related modules
import numpy as np

try:
    import cantera as ct
    from cantera import ck2cti
except ImportError:
    print('Error: Cantera must be installed.')
    raise

try:
    from optionloop import OptionLoop
except ImportError:
    print('Error: optionloop must be installed.')
    raise

# Local imports
from .. import utils
from ..core.create_jacobian import create_jacobian
from ..pywrap.pywrap_gen import generate_wrapper

from ..tests.test_utils import data_bin_writer as dbw
from ..tests.test_utils import get_test_matrix as tm
from ..tests import test_utils


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
    #new version
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


def functional_tester(work_dir):
    """Runs performance testing for pyJac, TChem, and finite differences.

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """
    obj_dir = 'obj'
    build_dir = 'out'
    test_dir = 'test'

    work_dir = os.path.abspath(work_dir)
    mechanism_list, ocl_params, max_vec_width = tm.get_test_matrix(work_dir)

    if len(mechanism_list) == 0:
        print('No mechanisms found for performance testing in '
              '{}, exiting...'.format(work_dir)
              )
        sys.exit(-1)

    repeats = 10
    conp = True

    script_dir = os.path.abspath(os.path.dirname(__file__))
    #load the module tester template
    mod_test = test_utils.get_import_source()

    for mech_name, mech_info in sorted(mechanism_list.items(),
                                       key=lambda x:x[1]['ns']
                                       ):
        #ensure directory structure is valid
        this_dir = os.path.join(work_dir, mech_name)
        this_dir = os.path.abspath(this_dir)
        os.chdir(this_dir)
        my_obj = os.path.join(this_dir, obj_dir)
        my_build = os.path.join(this_dir, build_dir)
        my_test = os.path.join(this_dir, test_dir)
        subprocess.check_call(['mkdir', '-p', obj_dir])
        subprocess.check_call(['mkdir', '-p', my_build])
        subprocess.check_call(['mkdir', '-p', my_test])

        def __cleanup():
            #remove library
            test_utils.clean_dir(my_obj, False)
            #remove build
            test_utils.clean_dir(my_build, False)
            #clean sources
            test_utils.clean_dir(my_test, False)
            #clean dummy builder
            dist_build = os.path.join(os.getcwd(), 'build')
            if os.path.exists(dist_build):
                shutil.rmtree(dist_build)

        #get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))
        gas.basis = 'molar'

        #first load data to get species rates, jacobian etc.
        num_conditions, data = dbw.load([], directory=os.path.join(work_dir, mech_name))
        #figure out the number of conditions to test
        num_conditions = int(np.floor(num_conditions / max_vec_width) * max_vec_width)
        #find the number of conditions per run (needed to avoid memory issues with i-pentanol model)
        max_per_run = 100000
        cond_per_run = int(np.floor(max_per_run / max_vec_width) * max_vec_width)

        #set T, P arrays
        T = data[:, 0].flatten()
        P = data[:, 1].flatten()

        #resize data
        data = data[:num_conditions, 2:]

        spec_rates = np.zeros((num_conditions, gas.n_species))
        conp_temperature_rates = np.zeros((num_conditions, 1))
        conv_temperature_rates = np.zeros((num_conditions, 1))
        h = np.zeros((gas.n_species))
        u = np.zeros((gas.n_species))
        cp = np.zeros((gas.n_species))
        cv = np.zeros((gas.n_species))

        #get mappings
        fwd_map = np.array(range(gas.n_reactions))
        rev_map = np.array([x for x in range(gas.n_reactions) if gas.is_reversible(x)])
        thd_map = []
        for x in range(gas.n_reactions):
            try:
                eff = gas.reaction(x).efficiencies
                thd_map.append(x)
            except:
                pass
        thd_map = np.array(thd_map)
        rop_fwd_test = np.zeros((num_conditions, fwd_map.size))
        rop_rev_test = np.zeros((num_conditions, rev_map.size))
        rop_net_test = np.zeros((num_conditions, fwd_map.size))
        #need special maps for rev/thd
        rev_to_thd_map = np.where(np.in1d(rev_map, thd_map))[0]
        thd_to_rev_map = np.where(np.in1d(thd_map, rev_map))[0]
        #it's a pain to actually calcuate this
        #and we don't need it directly, since cantera computes
        #pdep terms in the forward / reverse ROP automatically
        #hence we create it as a placeholder for the testing script
        pres_mod_test = np.zeros((num_conditions, thd_map.size))

        #predefines
        specs = gas.species()[:]

        def __eval_cp(j, T):
            return specs[j].thermo.cp(T)
        eval_cp = np.vectorize(__eval_cp, cache=True)
        def __eval_h(j, T):
            return specs[j].thermo.h(T)
        eval_h = np.vectorize(__eval_h, cache=True)
        ns_range = np.arange(gas.n_species)

        evaled = False
        def eval_state():
            with np.errstate(divide='ignore', invalid='ignore'):
                for i in range(num_conditions):
                    if not i % 10000:
                        print(i)
                    #it's actually more accurate to set the density (total concentration)
                    #due to the cantera internals
                    gas.TDX = T[i], P[i] / (ct.gas_constant * T[i]), data[i, :]
                    #now, since cantera normalizes these concentrations
                    #let's read them back
                    data[i, :] = gas.concentrations[:]
                    #get species rates
                    spec_rates[i, :] = gas.net_production_rates[:]
                    rop_fwd_test[i, :] = gas.forward_rates_of_progress[:]
                    rop_rev_test[i, :] = gas.reverse_rates_of_progress[:][rev_map]
                    rop_net_test[i, :] = gas.net_rates_of_progress[:]
                    cp[:] = eval_cp(ns_range, T[i])
                    h[:] = eval_h(ns_range, T[i])
                    cv[:] = cp - ct.gas_constant
                    u[:] = h - T[i] * ct.gas_constant

                    np.divide(-np.dot(h[:], spec_rates[i, :]), np.dot(cp[:], data[i, :]), out=conp_temperature_rates[i, :])
                    np.divide(-np.dot(u[:], spec_rates[i, :]), np.dot(cv[:], data[i, :]), out=conv_temperature_rates[i, :])

        current_data_order = None

        the_path = os.getcwd()
        first_run = True
        op = OptionLoop(ocl_params, lambda: False)

        for i, state in enumerate(op):
            #remove any old builds
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
            if not deep and not wide and vecsize != max_vec_width:
                continue #this is simple parallelization, don't need vector size
                #simpy choose one and go

            if rate_spec == 'fixed' and split_kernels:
                continue #not a thing!

            if num_cores < 8: #temporary
                continue

            data_output = ('{}_{}_{}_{}_{}_{}_{}_{}'.format(lang, vecsize, order,
                            'w' if wide else 'd' if deep else 'par',
                            platform, rate_spec, 'split' if split_kernels else 'single',
                            num_cores
                            ) +
                           '_err.npz'
                           )

            #if already run, continue
            data_output = os.path.join(the_path, data_output)
            if check_file(data_output, gas.n_species, gas.n_reactions):
                continue

            #eval if not done already
            if not evaled:
                eval_state()
                evaled = True
            #force garbage collection
            gc.collect()

            if order != current_data_order:
                #rewrite data to file in 'C' order
                dbw.write(os.path.join(this_dir))

            #save args to dir
            def __saver(arr, name, namelist=None):
                myname = os.path.join(my_test, name + '.npy')
                np.save(myname, arr)
                if namelist is not None:
                    namelist.append(myname)

            try:
                create_jacobian(lang,
                    mech_name=mech_info['mech'],
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
                    output_full_rop=True
                    )
            except:
                print('generation failed...')
                print(i, state)
                print()
                print()
                continue

            #generate wrapper
            generate_wrapper(lang, my_build, build_dir=obj_dir,
                         out_dir=my_test, platform=str(platform),
                         output_full_rop=True)

            #now generate the per run data
            offset = 0
            #store the error dict
            err_dict = {}
            while offset < num_conditions:
                this_run = int(np.floor(np.minimum(cond_per_run, num_conditions - offset) / max_vec_width) * max_vec_width)
                #get arrays
                concs = data[offset:offset + this_run, :].flatten(order=order)

                args = []
                __saver(T[offset:offset + this_run], 'T', args)
                __saver(P[offset:offset + this_run], 'P', args)
                __saver(concs, 'conc', args)
                del concs

                #put save outputs
                output_names = ['wdot', 'rop_fwd', 'rop_rev', 'pres_mod', 'rop_net']
                species_rates = np.concatenate((conp_temperature_rates[offset:offset + this_run, :] if conp
                    else conv_temperature_rates[offset:offset + this_run, :],
                        spec_rates[offset:offset + this_run, :]), axis=1)
                out_arrays = [species_rates,
                                rop_fwd_test[offset:offset + this_run, :],
                                rop_rev_test[offset:offset + this_run, :],
                                pres_mod_test[offset:offset + this_run, :],
                                rop_net_test[offset:offset + this_run, :]]
                for i in range(len(out_arrays)):
                    out = out_arrays[i]
                    #and flatten in correct order
                    out = out.flatten(order=order)
                    __saver(out, output_names[i])
                    del out

                outf = [os.path.join(my_test, '{}_rate.npy'.format(name))
                    for name in output_names]
                test_f = [os.path.join(my_test, '{}.npy'.format(name))
                    for name in output_names]

                #write the module tester
                with open(os.path.join(my_test, 'test.py'), 'w') as file:
                    file.write(mod_test.safe_substitute(
                        package='pyjac_ocl',
                        input_args=', '.join('"{}"'.format(x) for x in args),
                        test_arrays=', '.join('"{}"'.format(x) for x in test_f),
                        non_array_args='{}, 12'.format(this_run),
                        call_name='species_rates',
                        output_files=', '.join('\'{}\''.format(x) for x in outf)))

                #call
                subprocess.check_call([
                    'python{}.{}'.format(sys.version_info[0], sys.version_info[1]),
                    os.path.join(my_test, 'test.py')])

                def __get_test(name):
                    return out_arrays[output_names.index(name)]

                out_check = outf[:]
                #load output arrays
                for i in range(len(outf)):
                    out_check[i] = np.load(outf[i])
                    assert not np.any(np.logical_or(np.isnan(out_check[i]), np.isinf(out_check[i])))
                    #and reshape
                    out_check[i] = np.reshape(out_check[i], (this_run, -1),
                        order=order)

                #multiply pressure rates
                pmod_ind = next(i for i, x in enumerate(output_names) if x == 'pres_mod')
                #fwd
                out_check[1][:, thd_map] *= out_check[pmod_ind]
                #rev
                out_check[2][:, rev_to_thd_map] *= out_check[pmod_ind][:, thd_to_rev_map]

                fwd_ind = output_names.index('rop_fwd')
                rev_ind = output_names.index('rop_rev')
                atol = 1e-10
                rtol = 1e-6

                #load output
                for name, out in zip(*(output_names, out_check)):
                    if name == 'pres_mod':
                        continue
                    check_arr = __get_test(name)
                    #get err
                    err = np.abs(out - check_arr)
                    err_compare = err / (atol + rtol * np.abs(check_arr))
                    #get maximum relative error locations
                    err_locs = np.argmax(err_compare, axis=0)
                    #take err norm
                    err_comp_store = err_compare[err_locs, np.arange(err_locs.size)]
                    err_inf = err[err_locs, np.arange(err_locs.size)]
                    if name == 'rop_net':
                        #need to find the fwd / rop error at the max locations here
                        rop_fwd_err = np.abs(out_check[fwd_ind][err_locs, np.arange(err_locs.size)] - out_arrays[fwd_ind][err_locs, np.arange(err_locs.size)])
                        rop_rev_err = np.zeros(err_locs.size)
                        rev_errs = err_locs[rev_map]
                        rop_rev_err[rev_map] = np.abs(out_check[rev_ind][rev_errs, np.arange(rev_errs.size)] - out_arrays[rev_ind][rev_errs, np.arange(rev_errs.size)])
                        rop_component_error = np.maximum(rop_fwd_err, rop_rev_err)

                    if name not in err_dict:
                        err_dict[name] = np.zeros_like(err_inf)
                        err_dict[name + '_value'] = np.zeros_like(err_inf)
                        err_dict[name + '_store'] = np.zeros_like(err_inf)
                        if name == 'rop_net':
                            err_dict['rop_component'] = np.zeros_like(err_inf)
                    #get locations to update
                    update_locs = np.where(err_comp_store >= err_dict[name + '_store'])
                    #and update
                    err_dict[name][update_locs] = err_inf[update_locs]
                    err_dict[name + '_store'][update_locs] = err_comp_store[update_locs]
                    #need to take max and update precision as necessary
                    if name == 'rop_net':
                        err_dict['rop_component'][update_locs] = rop_component_error[update_locs]
                    #update the values for normalization
                    err_dict[name + '_value'][update_locs] = check_arr[err_locs, np.arange(err_inf.size)][update_locs]

                #cleanup
                for x in args + outf:
                    os.remove(x)
                for x in out_check:
                    del x
                os.remove(os.path.join(my_test, 'test.py'))

                #finally update the offset
                offset += cond_per_run

            #and write to file
            np.savez(data_output, **err_dict)


