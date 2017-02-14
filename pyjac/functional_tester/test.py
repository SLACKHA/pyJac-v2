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

from ..tests.utils import data_bin_writer as dbw
from ..tests.utils.test_matrix import get_test_matrix
from ..tests import utils as test_utils


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
    try:
        with open(filename, 'r') as file:
            lines = [line.strip() for line in file.readlines()]
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
        rop_net_test = np.zeros((num_conditions, fwd_map.size))
        #need special maps for rev/thd
        rev_to_thd_map = np.where(np.in1d(rev_map, thd_map))[0]
        thd_to_rev_map = np.where(np.in1d(thd_map, rev_map))[0]
        #it's a pain to actually calcuate this
        #and we don't need it directly, since cantera computes
        #pdep terms in the forward / reverse ROP automatically
        #hence we create it as a placeholder for the testing script
        pres_mod_test = np.zeros((num_conditions, thd_map.size))

        #set decimal context
        c = getcontext()
        c.traps[InvalidOperation] = 0
        c.traps[DivisionByZero] = 0
        setcontext(c)

        #predefines
        ln2 = Decimal('2').ln()
        d2 = Decimal('2')
        specs = gas.species()[:]
        ns_range = np.array(range(gas.n_species), dtype=np.int32)

        #precision loss measurement
        precision_loss_min = np.zeros((num_conditions, gas.n_reactions))
        precision_loss_max = np.zeros((num_conditions, gas.n_reactions))

        def __eval_cp(j, T):
            return specs[j].thermo.cp(T)
        def __eval_h(j, T):
            return specs[j].thermo.h(T)
        def __get_prec_max(x):
            x = (-x.ln() / ln2).to_integral_value(rounding=ROUND_FLOOR)
            return c.power(d2, -x)
        def __get_prec_min(x):
            x = (-x.ln() / ln2).to_integral_value(rounding=ROUND_CEILING)
            return c.power(d2, -x)

        evaled = False
        def eval_state(i):
            if not i % 10000:
                print(i)
            #it's actually more accurate to set the density (total concentration)
            #due to the cantera internals
            gas.TDX = T[i], P[i] / (ct.gas_constant * T[i]), data[i, 2:]
            #now, since cantera normalizes these concentrations
            #let's read them back
            data[i, 2:] = gas.concentrations[:]
            #get species rates
            spec_rates[i, :] = gas.net_production_rates[:]
            rop_fwd_test[i, :] = gas.forward_rates_of_progress[:]
            rop_rev_test[i, :] = gas.reverse_rates_of_progress[:][rev_map]
            rop_net_test[i, :] = gas.net_rates_of_progress[:]
            cp[:] = np.vectorize(__eval_cp, cache=True)(ns_range, T[i])
            h[:] = np.vectorize(__eval_h, cache=True)(ns_range, T[i])
            cv[:] = cp - ct.gas_constant
            u[:] = h - T[i] * ct.gas_constant

            #create find precisions
            fwd = np.array([Decimal.from_float(j) for j in gas.forward_rates_of_progress],
                dtype=np.dtype(Decimal))
            rev = np.array([Decimal.from_float(j) for j in gas.reverse_rates_of_progress],
                dtype=np.dtype(Decimal))
            ratio = fwd / rev
            mid = np.abs(Decimal('1') - ratio)
            precision_loss_min[i, :] = np.vectorize(__get_prec_min, cache=True)(mid)
            precision_loss_max[i, :] = np.vectorize(__get_prec_max, cache=True)(mid)

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
            if check_file(data_output, gas.n_species, gas.n_reactions):
                continue

            #eval if not done already
            if not evaled:
                np.vectorize(eval_state, cache=True)(np.arange(num_conditions))
                evaled = True

            if order != current_data_order:
                #rewrite data to file in 'C' order
                dbw.write(os.path.join(this_dir))

            #save args to dir
            def __saver(arr, name, namelist=None):
                myname = os.path.join(my_test, name + '.npy')
                np.save(myname, arr)
                if namelist is not None:
                    namelist.append(myname)

            #get arrays
            concs = (data[:num_conditions, 2:].copy() if order == 'C' else
                        data[:num_conditions, 2:].T.copy()).flatten('K')

            args = []
            __saver(T, 'T', args)
            __saver(P, 'P', args)
            __saver(concs, 'conc', args)

            #put save outputs
            out_arrays = []
            output_names = ['wdot', 'rop_fwd', 'rop_rev', 'pres_mod', 'rop_net']
            comp_arrays = []
            species_rates = np.concatenate((conp_temperature_rates.copy() if conp else conv_temperature_rates.copy(),
                    spec_rates.copy()), axis=1)
            ropf = rop_fwd_test.copy()
            ropr = rop_rev_test.copy()
            ropp = pres_mod_test.copy()
            ropnet = rop_net_test.copy()
            out_arrays = [species_rates, ropf, ropr, ropp, ropnet]
            comp_arrays = [x.copy() for x in out_arrays]
            for i in range(len(out_arrays)):
                if order == 'F':
                    out_arrays[i] = out_arrays[i].T.copy()
                #and flatten in correct order
                out_arrays[i] = out_arrays[i].flatten(order='K')
                __saver(out_arrays[i], output_names[i])

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
                         out_dir=my_test, platform=str(platform),
                         output_full_rop=True)

            #call
            subprocess.check_call([
                'python{}.{}'.format(sys.version_info[0], sys.version_info[1]),
                os.path.join(my_test, 'test.py')])

            def __get_test(name):
                return comp_arrays[output_names.index(name)]

            out_check = outf[:]
            #load output arrays
            for i in range(len(outf)):
                out_check[i] = np.load(outf[i])
                #and reshape
                out_check[i] = np.reshape(out_check[i], (num_conditions, -1),
                    order=order)

            #multiply pressure rates
            pmod_ind = next(i for i, x in enumerate(output_names) if x == 'pres_mod')
            #fwd
            out_check[1][:, thd_map] *= out_check[pmod_ind]
            #rev
            out_check[2][:, rev_to_thd_map] *= out_check[pmod_ind][:, thd_to_rev_map]

            #load output
            for name, out in zip(*(output_names, out_check)):
                if name == 'pres_mod':
                    continue
                check_arr = __get_test(name)
                #get err
                err_base = np.abs(out - check_arr)
                err = err_base / (atol + rtol * np.abs(check_arr))
                #get precision at each of these locs
                err_locs = np.argmax(err, axis=0)
                #take err norm
                err_inf = np.linalg.norm(err, ord=np.inf, axis=0)
                err_l2 = np.linalg.norm(err, ord=2, axis=0)
                #save to output
                with open(data_output, 'a') as file:
                    file.write(name + ' - linf: ' + ', '.join(['{:.15e}'.format(x) for x in err_inf]) + '\n')
                    file.write(name + ' - l2: ' + ', '.join(['{:.15e}'.format(x) for x in err_l2]) + '\n')
                    if name == 'rop_net':
                        def __prec_percent(precision):
                            precs = precision[err_locs, np.arange(err_inf.size)]
                            return 100.0 * err_base[err_locs, np.arange(err_inf.size)] / precs
                        #get precision at each of these locs
                        precs = __prec_percent(precision_loss_min)
                        file.write(name + '_precmin : ' + ', '.join(['{:.15e}'.format(x) for x in precs]) + '\n')
                        precs = __prec_percent(precision_loss_max)
                        file.write(name + '_precmax : ' + ', '.join(['{:.15e}'.format(x) for x in precs]) + '\n')

                #and print total max to screen
                print(name, np.linalg.norm(err_inf, np.inf))

            #cleanup
            for x in args + outf:
                os.remove(x)
            os.remove(os.path.join(my_test, 'test.py'))


