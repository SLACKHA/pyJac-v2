# system
from os.path import join as pjoin
import subprocess
import logging

# external
from nose.plugins.attrib import attr
import numpy as np

# internal
from pyjac.core import array_creator as arc
from pyjac.core.rate_subs import assign_rates
from pyjac.core.enum_types import RateSpecialization, KernelType
from pyjac.core.driver_kernels import lockstep_driver
from pyjac.core.mech_auxiliary import write_aux
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.tests import TestClass, get_test_langs
from pyjac.tests.test_utils import get_run_source, OptionLoopWrapper
from pyjac import utils
from pyjac.utils import temporary_directory
from pyjac.pywrap import generate_wrapper


class SubTest(TestClass):
    @attr('long')
    def test_lockstep_driver(self):
        # get rate info
        rate_info = assign_rates(self.store.reacs, self.store.specs,
                                 RateSpecialization.fixed)
        mod_test = get_run_source()

        for i, loopy_opts in OptionLoopWrapper.from_get_oploop(
                self, do_ratespec=False, langs=get_test_langs(),
                do_vector=True, yield_index=True):
            # make namestore
            namestore = arc.NameStore(loopy_opts, rate_info)
            # create a dummy kernel that simply adds 1 to phi for easy testing
            inputs = ['n_arr', 'P_arr']
            outputs = ['n_arr']

            # make mapstore, arrays and kernel info
            mapstore = arc.MapStore(loopy_opts, namestore.phi_inds, None)
            base_phi_shape = namestore.n_arr.shape
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
                kernel_data=[phi_lp, P_lp, arc.work_size])
            # put it in a generator
            generator = k_gen.make_kernel_generator(
                loopy_opts, kernel_type=KernelType.dummy,
                name='inner', kernels=[inner_kernel], namestore=namestore,
                input_arrays=inputs[:],
                output_arrays=outputs[:],
                is_validation=True)

            # now get the driver
            driver = lockstep_driver(
              loopy_opts, namestore, inputs, outputs, generator)

            # and put in generator
            driver = k_gen.make_kernel_generator(
                loopy_opts, kernel_type=KernelType.dummy, name='driver',
                kernels=driver, namestore=namestore,
                input_arrays=inputs[:],
                output_arrays=outputs[:],
                depends_on=[generator],
                is_validation=True,
                fake_calls={'dummy', generator})

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
                data = np.zeros(base_phi_shape)
                # make it a simple range
                data.flat[:] = np.arange(np.prod(base_phi_shape))
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
                                 ktype=KernelType.species_rates)
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
                    utils.run_with_our_python(['test.py'])
                except subprocess.CalledProcessError:
                    logger = logging.getLogger(__name__)
                    logger.debug(utils.stringify_args(vars(loopy_opts), kwd=True))
                    assert False, 'lockstep_driver error'
