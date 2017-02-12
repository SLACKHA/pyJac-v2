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


def functional_tester(home, work_dir):
    """Runs performance testing for pyJac, TChem, and finite differences.

    Parameters
    ----------
    home : str
        Directory of source code files
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """
    obj_dir = 'obj'
    build_dir = 'out'
    test_dir = 'test'

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
    with open(os.path.join(script_dir, os.par, 'tests', 'test_import.py.in'), 'r') as file:
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
            __clean_dir(my_obj)
            #remove build
            __clean_dir(my_build)
            #clean sources
            __clean_dir(my_test)
            #clean dummy builder
            dist_build = os.path.join(os.getcwd(), 'build')
            if os.path.exists(dist_build):
                shutil.rmtree(dist_build)

        #get the cantera object
        gas = ct.Solution(os.path.join(work_dir, mech_name, mech_info['mech']))

        #first load data to get species rates, jacobian etc.
        data = dbw.load([], directory=os.path.join(work_dir, mech_name))
        num_conditions = data.shape[0]

        #set T, P arrays
        T = data[:, 0].flatten()
        P = data[:, 1].flatten()

        #figure out the number of conditions to test
        num_conditions = np.floor(num_conditions / max_vec_width) * num_conditions

        spec_rates = np.zeros((num_conditions, gas.n_species))
        conp_temperature_rates = np.zeros((num_conditions, 1))
        conv_temperature_rates = np.zeros((num_conditions, 1))
        h = np.zeros((gas.n_species, 1))
        u = np.zeros((gas.n_species, 1))
        cp = np.zeros((gas.n_species, 1))
        cp = np.zeros((gas.n_species, 1))
        #now we must evaluate the species rates
        for i in range(num_conditions):
            #remove any old builds
            __cleanup()

            #set state
            gas.concentrations = data[i, 2:]
            gas.TP = T[i], P[i]
            #get species rates
            spec_rates[i, :] = gas.net_production_rates[:]
            for j in range(gas.n_species):
                cp = gas.species(j).thermo.cp(T[i])
                s = gas.species(j).thermo.s(T[i])
                h = gas.species(j).thermo.h(T[i])
                h[j] = h
                u[j] = h - T[i] * ct.gas_constant
                cp[j] = cp
                cv[j] = cp - ct.gas_constant
            conp_temperature_rates[i] = (-np.dot(h[:], spec_rates[i, :]) / np.dot(cp[i], data[i, 2:]))
            conv_temperature_rates[i] = (-np.dot(u[:], spec_rates[i, :]) / np.dot(cv[i], data[i, 2:]))

        current_data_order = None

        the_path = os.getcwd()
        first_run = True
        op = OptionLoop(ocl_params)

        for i, state in enumerate(op):
            lang = state['lang']
            vecsize = state['vecsize']
            order = state['order']
            wide = state['wide']
            deep = state['deep']
            platform = state['platform']
            rate_spec = state['rate_spec']
            split_kernels = state['split_kernels']
            num_cores = state['num_cores']
            if not deep and not wide and vecsize != vec_widths[0]:
                continue #this is simple parallelization, don't need vector size

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
                #rewrite data to file in correct order
                dbw.write(os.path.join(work_dir, mech_name),
                                            order=order)

            #save args to dir
            def __saver(arr, name, namelist):
                myname = os.path.join(my_test, name + '.npy')
                np.save(myname, arr)
                namelist.append(myname)

            #get arrays
            concs = (data[:, 2:].copy() if order == 'C' else
                        data[:, 2:].copy().T.copy()).flatten('K')

            #put together species rates
            spec_rates = np.concatenate((conp_temperature_rates.copy() if conp else conv_temperature_rates.copy(),
                    spec_rates.copy()))
            if opts.order == 'F':
                spec_rates = spec_rates.T.copy()
            #and flatten in correct order
            spec_rates = spec_rates.flatten(order='K')

            wdot = np.concatenate((Tdot_conp.copy() if conp else Tdot_conv,
                    self.store.species_rates.copy()))

            args = []
            __saver(T, 'T', args)
            __saver(P, 'P', args)
            __saver(concs, 'conc', args)

            #and now the test values
            tests = []
            __saver(spec_rates, 'wdot', tests)

            outf = 'wdot_rate_err.npy'
            #write the module tester
            with open(os.path.join(my_test, 'test.py'), 'w') as file:
                file.write(mod_test.safe_substitute(
                    package='pyjac_ocl',
                    input_args=', '.join('"{}"'.format(x) for x in args),
                    test_arrays=', '.join('"{}"'.format(x) for x in tests),
                    non_array_args='{}, 12'.format(self.store.test_size),
                    call_name='species_rates',
                    output_files='\'{}\''.format(outf)))

            try:
                create_jacobian(lang,
                    mech_name=mech_info['mech'],
                    vector_size=vecsize,
                    wide=wide,
                    deep=deep,
                    build_path=my_build,
                    skip_jac=True,
                    auto_diff=False,
                    platform=platform,
                    data_filename=os.path.join(work_dir, mech_name, 'data.bin'),
                    split_rate_kernels=split_kernels,
                    rate_specialization=rate_spec,
                    split_rop_net_kernels=split_kernels
                    )
            except:
                print('generation failed...')
                print(i, state)
                print()
                print()
                continue

            #generate wrapper
            generate_wrapper(opts.lang, my_build, build_dir=obj_dir,
                         out_dir=my_test, platform=str(opts.platform))

            #call
            subprocess.check_call([
                'python{}.{}'.format(sys.version_info[0], sys.version_info[1]),
                os.path.join(my_test, 'test.py')])

            #load err
            err = np.load(os.path.join(my_test, outf))
            #reshape
            err = np.reshape(err, (num_conditions, gas.n_species + 1), order=order)
            #take err norm
            err = np.linalg.norm(err, 'inf', axis=0)
            #save to output
            with open(data_output, 'a') as file:
                data_output.write(', '.join('%.15le'.format(x) for x in err))
            #and print total max to screen
            print(np.linalg.norm(err, 'inf'))

            #cleanup
            for x in args + tests:
                os.remove(x)
            os.remove(os.path.join(my_test, 'test.py'))


