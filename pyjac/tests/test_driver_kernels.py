# system
from collections import OrderedDict
from os.path import join as pjoin
import subprocess
import sys
import logging

# external
from optionloop import OptionLoop
from nose.plugins.attrib import attr
import numpy as np

# internal
from pyjac.core import array_creator as arc
from pyjac.core.rate_subs import assign_rates
from pyjac.core.enum_types import RateSpecialization, kernel_type, JacobianType, \
    JacobianFormat
from pyjac.core.driver_kernels import lockstep_driver
from pyjac.core.mech_auxiliary import write_aux
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.tests import TestClass, get_test_langs
from pyjac.tests.test_utils import temporary_directory, get_run_source
from pyjac import utils
from pyjac.pywrap import generate_wrapper
from pyjac.core.exceptions import MissingDeviceError


def opts_loop(self,
              width=[4, None],
              depth=[4, None],
              order=['C', 'F'],
              lang=get_test_langs(),
              is_simd=[True, False]):

    oploop = OptionLoop(OrderedDict(
        [('width', width),
         ('depth', depth),
         ('order', order),
         ('use_working_buffers', [True]),
         ('lang', lang),
         ('order', order),
         ('is_simd', is_simd),
         ('unr', [None]),
         ('ilp', [None]),
         ('jac_type', [JacobianType.exact]),
         ('jac_format', [JacobianFormat.full]),
         ('platform', [self.store.test_platforms[:]])]))
    for state in oploop:
        if state['depth'] and state['width']:
            continue
        if (state['depth'] or state['width']) and not utils.can_vectorize_lang[
                state['lang']]:
            continue
        if not (state['width'] or state['depth']) and state['is_simd']:
            continue
        if state['lang'] == 'opencl':
            for i, platform in enumerate(platforms):
                state['platform'] = platform
                devices = platform.get_devices()
                if not devices and i == len(platforms) - 1:
                    raise MissingDeviceError('"any"', platform)
                state['device'] = devices[0]
                state['device_type'] = devices[0].type
                break

        yield type('dummy', (object,), state)


class SubTest(TestClass):
    @attr('long')
    def test_lockstep_driver(self):
        # get rate info
        rate_info = assign_rates(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)
        mod_test = get_run_source()
        for loopy_opts in opts_loop(self):
            # make namestore
            namestore = arc.NameStore(loopy_opts, rate_info,
                                      test_size=self.store.test_size)
            # create a dummy kernel that simply adds 1 to phi for easy testing
            inputs = ['phi', 'P_arr']
            outputs = ['phi']

            # make mapstore, arrays and kernel info
            mapstore = arc.MapStore(loopy_opts, namestore.phi_inds,
                                    namestore.phi_inds)
            phi_lp, phi_str = mapstore.apply_maps(namestore.n_arr,
                                                  arc.global_ind,
                                                  arc.var_name)
            P_lp, P_str = mapstore.apply_maps(namestore.P_arr,
                                              arc.global_ind)
            instructions = '{0} = {0} + 1'.format(phi_str)
            inner_kernel = k_gen.knl_info(
                name='inner',
                instructions=instructions,
                mapstore=mapstore,
                var_name=arc.var_name,
                kernel_data=[phi_lp])
            # put it in a generator
            generator = k_gen.make_kernel_generator(
                loopy_opts, 'inner', [inner_kernel], namestore,
                input_arrays=inputs[:],
                output_arrays=outputs[:],
                is_validation=True)

            # now get the driver
            driver = lockstep_driver(
              loopy_opts, namestore, inputs, outputs, generator,
              test_size=self.store.test_size)

            # and put in generator
            driver = k_gen.make_kernel_generator(
                loopy_opts, 'driver', [driver], namestore,
                input_arrays=inputs[:],
                output_arrays=outputs[:],
                depends_on=[generator.kernels[:]],
                is_validation=True)

            # and make
            with temporary_directory() as path:
                # make dirs
                out = pjoin(path, 'out')
                obj = pjoin(path, 'obj')
                lib = pjoin(path, 'lib')
                utils.create_dir(out)
                utils.create_dir(obj)
                utils.create_dir(lib)

                # write 'data'
                data = np.zeros(phi_lp.shape)
                # make it a simple range
                data.flat[:] = np.arange(np.prod(phi_lp.shape))
                # save
                myname = pjoin(path, 'phi.npy')
                # need to split inputs / answer
                np.save(myname, data.flatten('K'))
                # and a parameter
                param = np.zeros((self.store.test_size,))
                param[:] = np.arange(self.store.test_size)

                # build code
                driver.generate(path,
                                data_order=loopy_opts.order,
                                data_filename='data.bin',
                                for_validation=True)

                # write header
                write_aux(out, loopy_opts, self.store.specs, self.store.reacs)

                # generate wrapper
                generate_wrapper(loopy_opts.lang, out,
                                 obj_dir=obj, out_dir=lib,
                                 platform=str(loopy_opts.platform),
                                 btype=kernel_type.species_rates)
                # and python wrapper
                with open(pjoin(path, 'test.py'), 'w') as file:
                    file.write(mod_test.safe_substitute(
                        package='pyjac_{lang}'.format(
                            lang=utils.package_lang[loopy_opts.lang]),
                        input_args=utils.stringify_args(inputs, use_quotes=True),
                        test_arrays=utils.stringify_args(inputs, use_quotes=True),
                        looser_tols='[]',
                        loose_rtol=1e-5,
                        loose_atol=1e-8,
                        rtol=1e-5,
                        atol=1e-8,
                        non_array_args='{}, {}'.format(self.store.test_size, 1),
                        call_name='species_rates_kernel',
                        output_files=''))

                try:
                    subprocess.check_call([
                        'python{}.{}'.format(
                            sys.version_info[0], sys.version_info[1]), 'test.py'])
                except subprocess.CalledProcessError:
                    logger = logging.getLogger(__name__)
                    logger.debug(utils.stringify_args(vars(loopy_opts), kwd=True))
                    assert False, 'lockstep_driver error'
