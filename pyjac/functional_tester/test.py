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
from collections import defaultdict, OrderedDict

from string import Template
from decimal import *

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

from ..performance_tester import data_bin_writer as dbw
from .. get_test_matrix import get_test_matrix


def check_file(filename, Ns, num_conditions):
    """Checks file for existing data, returns number of completed runs

    Parameters
    ----------
    filename : str
        Name of file with data
    Ns : int
        The number of species in the mech

    Returns
    -------
    completed : bool
        True if the file has (Ns + 1) * num_conditions nums

    """
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
        complete = len(lines) == 1
        complete = complete and len(lines[0].split(',')) == (Ns + 1) * num_conditions
        for x in lines[0].split(','):
            float(x)
        return complete
    except:
        return False


def getf(x):
    return os.path.basename(x)


def functional_tester(work_dir, atol=1e-10, rtol=1e-6):
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
    mechanism_list, ocl_params, max_vec_width = get_test_matrix(work_dir)

    if len(mechanism_list) == 0:
        print('No mechanisms found for performance testing in '
              '{}, exiting...'.format(work_dir)
              )
        sys.exit(-1)

    repeats = 10
    conp = True

    script_dir = os.path.abspath(os.path.dirname(__file__))
    #load the module tester template
    with open(os.path.join(script_dir, os.pardir, 'tests', 'test_import.py.in'), 'r') as file:
        mod_test = Template(file.read())

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

        def __clean_dir(dirname, remove_dir=True):
            if not os.path.exists(dirname):
                return
            for file in os.listdir(dirname):
                if os.path.isfile(os.path.join(dirname, file)):
                    os.remove(os.path.join(dirname, file))
            if remove_dir:
                os.rmdir(dirname)

        def __cleanup():
            #remove library
            __clean_dir(my_obj, False)
            #remove build
            __clean_dir(my_build, False)
            #clean sources
            __clean_dir(my_test, False)
            #clean dummy builder
            dist_build = os.path.join(os.getcwd(), 'build')
            if os.path.exists(dist_build):
                shutil.rmtree(dist_build)

        #get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))
        gas.basis = 'molar'

        #first load data to get species rates, jacobian etc.
        num_conditions, data = dbw.load([], directory=os.path.join(work_dir, mech_name))

        #set T, P arrays
        T = data[:, 0].flatten()
        P = data[:, 1].flatten()

        #figure out the number of conditions to test
        num_conditions = int(np.floor(num_conditions / max_vec_width) * max_vec_width)

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
        #need special maps for rev/thd
        rev_to_thd_map = np.where(np.in1d(rev_map, thd_map))[0]
        thd_to_rev_map = np.where(np.in1d(thd_map, rev_map))[0]
        #it's a pain to actually calcuate this
        #and we don't need it directly, since cantera computes
        #pdep terms in the forward / reverse ROP automatically
        #hence we create it as a placeholder for the testing script
        pres_mod_test = np.zeros((num_conditions, thd_map.size))

        #precision loss measurement
        precision_loss = np.zeros((num_conditions, gas.n_reactions))
        #now we must evaluate the species rates
        for i in range(num_conditions):
            #it's actually more accurate to set the density (total concentration)
            #due to the cantera internals
            gas.TDX = T[i], P[i] / (ct.gas_constant * T[i]), data[i, 2:]
            #now, since cantera normalizes these concentrations
            #let's read them back
            data[i, 2:] = gas.concentrations[:]
            #get species rates
            spec_rates[i, :] = gas.net_production_rates[:]
            rop_fwd_test[i, :] = gas.forward_rates_of_progress[:]
            rop_rev_test[i, :] = gas.reverse_rates_of_progress[:]
            #get fwd / rev rop
            for j in range(gas.n_species):
                cps = gas.species(j).thermo.cp(T[i])
                hs = gas.species(j).thermo.h(T[i])
                h[j] = hs
                u[j] = hs - T[i] * ct.gas_constant
                cp[j] = cps
                cv[j] = cps - ct.gas_constant
                q = np.nan


            for j in range(gas.n_reactions):
                #try to estimate the loss of precision
                try:
                    ratio = (Decimal.from_float(gas.forward_rates_of_progress[j])
                        / Decimal.from_float(gas.reverse_rates_of_progress[j]))
                    mid = abs(Decimal('1') - ratio)
                    q = (-mid.ln() / Decimal('2').ln()).to_integral_value(rounding=ROUND_FLOOR)
                    precision_loss[i, j] = getcontext().power(Decimal('2'), -q) / Decimal.from_float(
                        gas.net_rates_of_progress[j]) * Decimal.from_float(100.0)
                except:
                    precision_loss[i, j] = q
                    pass

            conp_temperature_rates[i, :] = -np.dot(h[:], spec_rates[i, :]) / np.dot(cp[:], data[i, 2:])
            conv_temperature_rates[i, :] = -np.dot(u[:], spec_rates[i, :]) / np.dot(cv[:], data[i, 2:])

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

            data_output = ('{}_{}_{}_{}_{}_{}_{}_{}'.format(lang, vecsize, order,
                            'w' if wide else 'd' if deep else 'par',
                            platform, rate_spec, 'split' if split_kernels else 'single',
                            num_cores
                            ) +
                           '_err.txt'
                           )

            #if already run, continue
            data_output = os.path.join(the_path, data_output)
            if check_file(data_output, gas.n_species, num_conditions):
                continue

            if order != current_data_order:
                #rewrite data to file in 'C' order
                dbw.write(os.path.join(this_dir))

            #save args to dir
            def __saver(arr, name, namelist):
                myname = os.path.join(my_test, name + '.npy')
                np.save(myname, arr)
                namelist.append(myname)

            #get arrays
            concs = (data[:num_conditions, 2:].copy() if order == 'C' else
                        data[:num_conditions, 2:].T.copy()).flatten('K')

            #put together species rates
            species_rates = np.concatenate((conp_temperature_rates.copy() if conp else conv_temperature_rates.copy(),
                    spec_rates.copy()), axis=1)
            ropf = rop_fwd_test.copy()
            ropr = rop_rev_test.copy()
            ropp = pres_mod_test.copy()
            if order == 'F':
                species_rates = species_rates.T.copy()
                ropf = rop_fwd_test.T.copy()
                ropr = rop_rev_test.T.copy()
                ropp = pres_mod_test.T.copy()
            #and flatten in correct order
            species_rates = species_rates.flatten(order='K')

            args = []
            __saver(T, 'T', args)
            __saver(P, 'P', args)
            __saver(concs, 'conc', args)

            #and now the test values
            tests = []
            __saver(species_rates, 'wdot', tests)
            __saver(ropf, 'rop_fwd', tests)
            __saver(ropr, 'rop_rev', tests)
            __saver(ropp, 'pres_mod', tests)

            output_names = ['wdot', 'rop_fwd', 'rop_rev', 'pres_mod']
            outf = [os.path.join(my_test, '{}_rate.npy'.format(name))
                for name in output_names]
            #write the module tester
            with open(os.path.join(my_test, 'test.py'), 'w') as file:
                file.write(mod_test.safe_substitute(
                    package='pyjac_ocl',
                    input_args=', '.join('"{}"'.format(x) for x in args),
                    test_arrays=', '.join('"{}"'.format(x) for x in tests),
                    non_array_args='{}, 12'.format(num_conditions),
                    call_name='species_rates',
                    output_files=', '.join('\'{}\''.format(x) for x in outf)))

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
                         out_dir=my_test, platform=str(platform))

            #call
            subprocess.check_call([
                'python{}.{}'.format(sys.version_info[0], sys.version_info[1]),
                os.path.join(my_test, 'test.py')])

            def __get_test(name):
                if name == 'wdot':
                    return species_rates
                elif name == 'rop_fwd':
                    return rop_fwd_test
                elif name == 'rop_rev':
                    return rop_rev_test
                else:
                    return None

            #load output arrays
            for i in range(outf):
                outf[i] = np.load(outf[i])

            #multiply pressure rates
            #fwd
            outf[1][thd_map] *= outf[-1]
            #rev
            outf[2][rev_to_thd_map] *= outf[-1][thd_to_rev_map]

            #load output
            for name, out in zip(*(output_names, outf))[:-1]:
                check_arr = __get_test(name)
                err = np.abs(outv - check_arr) / (atol + rtol * np.abs(check_arr))
                #reshape
                err = np.reshape(err, (num_conditions, check_arr.shape[1]), order=order)
                #take err norm
                err = np.linalg.norm(err, ord=np.inf, axis=0)
                #save to output
                with open(data_output, 'a') as file:
                    file.write(', '.join(['{:.15e}'.format(x) for x in err]))
                #and print total max to screen
                print(name, np.linalg.norm(err, np.inf))

            #cleanup
            for x in args + tests:
                os.remove(x)
            os.remove(os.path.join(my_test, 'test.py'))


