# system
from os.path import join as pjoin
import subprocess
import logging
from string import Template

# external
from nose.plugins.attrib import attr
import numpy as np
import psutil

# internal
from pyjac.core import array_creator as arc
from pyjac.core.create_jacobian import reset_arrays, determine_jac_inds
from pyjac.core import instruction_creator as ic
from pyjac.core.enum_types import RateSpecialization, KernelType, DriverType
from pyjac.core.mech_auxiliary import write_aux
from pyjac.kernel_utils import kernel_gen as k_gen
from pyjac.tests import TestClass, get_test_langs, _get_test_input
from pyjac.tests.test_utils import get_run_source, OptionLoopWrapper, \
    temporary_build_dirs
from pyjac import utils
from pyjac.pywrap import pywrap


class SubTest(TestClass):
    @attr('long')
    def test_lockstep_driver(self):
        # get rate info
        rate_info = determine_jac_inds(self.store.reacs, self.store.specs,
                                       RateSpecialization.fixed)
        mod_test = get_run_source()

        for kind, loopy_opts in OptionLoopWrapper.from_get_oploop(
                self, do_ratespec=False, langs=get_test_langs(),
                do_vector=True, yield_index=True):

            # make namestore
            namestore = arc.NameStore(loopy_opts, rate_info)

            # kernel 1 - need the jacobian reset kernel
            reset = reset_arrays(loopy_opts, namestore)
            # kernel 2 - incrementer
            # make mapstore, arrays and kernel info
            mapstore = arc.MapStore(loopy_opts, namestore.phi_inds, None)

            # use arrays of 2 & 3 dimensions to test the driver's copying
            base_phi_shape = namestore.n_arr.shape
            P_lp, P_str = mapstore.apply_maps(namestore.P_arr,
                                              arc.global_ind)
            phi_lp, phi_str = mapstore.apply_maps(namestore.n_arr,
                                                  arc.global_ind,
                                                  arc.var_name)
            inputs = [P_lp.name, phi_lp.name]
            base_jac_shape = namestore.jac.shape
            jac_lp, jac_str = mapstore.apply_maps(namestore.jac,
                                                  arc.global_ind,
                                                  arc.var_name,
                                                  arc.var_name)
            outputs = [jac_lp.name]
            kernel_data = [P_lp, phi_lp, jac_lp]
            kernel_data.extend(arc.initial_condition_dimension_vars(
                loopy_opts, None))
            instructions = Template("""
                ${phi_str} = ${phi_str} + ${P_str} {id=0, dep=*}
                ${jac_str} = ${jac_str} + ${phi_str} {id=1, dep=0, nosync=0}
            """).safe_substitute(**locals())

            # handle atomicity
            can_vec, vec_spec = ic.get_deep_specializer(
                loopy_opts, atomic_ids=['1'])
            barriers = []
            if loopy_opts.depth:
                # need a barrier between the reset & the kernel
                barriers = [(0, 1, 'global')]

            inner_kernel = k_gen.knl_info(
                name='inner',
                instructions=instructions,
                mapstore=mapstore,
                var_name=arc.var_name,
                kernel_data=kernel_data,
                silenced_warnings=['write_race(0)', 'write_race(1)'],
                can_vectorize=can_vec,
                vectorization_specializer=vec_spec)

            # put it in a generator
            generator = k_gen.make_kernel_generator(
                loopy_opts, kernel_type=KernelType.dummy,
                name='inner_kernel', kernels=[reset, inner_kernel],
                namestore=namestore,
                input_arrays=inputs[:],
                output_arrays=outputs[:],
                is_validation=True,
                driver_type=DriverType.lockstep,
                barriers=barriers)

            # use a "weird" (non-evenly divisibly by vector width) test-size to
            # properly test the copy-in / copy-out
            test_size = self.store.test_size - 37
            if test_size <= 0:
                test_size = self.store.test_size - 1
                assert test_size > 0
            # and make
            with temporary_build_dirs() as (build, obj, lib):

                numpy_arrays = []

                def __save(shape, name, zero=False):
                    data = np.zeros(shape)
                    if not zero:
                        # make it a simple range
                        data.flat[:] = np.arange(np.prod(shape))
                    # save
                    myname = pjoin(lib, name + '.npy')
                    # need to split inputs / answer
                    np.save(myname, data.flatten('K'))
                    numpy_arrays.append(data.flatten('K'))

                # write 'data'
                import loopy as lp
                for arr in kernel_data:
                    if not isinstance(arr, lp.ValueArg):
                        __save((test_size,) + arr.shape[1:], arr.name,
                               arr.name in outputs)

                # and a parameter
                param = np.zeros((test_size,))
                param[:] = np.arange(test_size)

                # build code
                generator.generate(build,
                                   data_order=loopy_opts.order,
                                   data_filename='data.bin',
                                   for_validation=True)

                # write header
                write_aux(build, loopy_opts, self.store.specs, self.store.reacs)

                # generate wrapper
                pywrap(loopy_opts.lang, build,
                       obj_dir=obj, out_dir=lib,
                       ktype=KernelType.dummy,
                       file_base=generator.name,
                       additional_inputs=inputs[:],
                       additional_outputs=outputs[:])

                # and calling script
                test = pjoin(lib, 'test.py')

                inputs = utils.stringify_args(
                    [pjoin(lib, inp + '.npy') for inp in inputs], use_quotes=True)
                str_outputs = utils.stringify_args(
                    [pjoin(lib, inp + '.npy') for inp in outputs], use_quotes=True)

                num_threads = _get_test_input(
                    'num_threads', psutil.cpu_count(logical=False))
                with open(test, 'w') as file:
                    file.write(mod_test.safe_substitute(
                        package='pyjac_{lang}'.format(
                            lang=utils.package_lang[loopy_opts.lang]),
                        input_args=inputs,
                        test_arrays=str_outputs,
                        output_files=str_outputs,
                        looser_tols='[]',
                        loose_rtol=0,
                        loose_atol=0,
                        rtol=0,
                        atol=0,
                        non_array_args='{}, {}'.format(
                            test_size, num_threads),
                        kernel_name=generator.name.title(),))

                try:
                    utils.run_with_our_python([test])
                except subprocess.CalledProcessError:
                    logger = logging.getLogger(__name__)
                    logger.debug(utils.stringify_args(vars(loopy_opts), kwd=True))
                    assert False, 'lockstep_driver error'

                # calculate answers
                ns = base_jac_shape[1]
                # pressure is added to phi
                phi = numpy_arrays[1].reshape((test_size, ns),
                                              order=loopy_opts.order)
                p_arr = numpy_arrays[0]
                phi = phi + p_arr[:, np.newaxis]
                jac = numpy_arrays[2].reshape((test_size, ns, ns),
                                              order=loopy_opts.order)
                # and the diagonal of the jacobian has the updated pressure added
                jac[:, range(ns), range(ns)] += phi[:, range(ns)]
                # and read in outputs
                test = np.load(pjoin(lib, outputs[0] + '.npy')).reshape(
                    jac.shape, order=loopy_opts.order)
                assert np.array_equal(test, jac)
