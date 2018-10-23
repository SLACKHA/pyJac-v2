import os
import re

import numpy as np
from nose.plugins.attrib import attr
from cogapp import Cog

from pyjac import utils
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.core.create_jacobian import get_jacobian_kernel, \
    finite_difference_jacobian, create_jacobian
from pyjac.core import array_creator as arc
from pyjac.core.enum_types import KernelType
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core.array_creator import work_size
from pyjac.kernel_utils.kernel_gen import knl_info, make_kernel_generator
from pyjac.libgen import generate_library
from pyjac.pywrap.pywrap_gen import pywrap
from pyjac.tests import TestClass, test_utils
from pyjac.tests.test_utils import temporary_build_dirs, OptionLoopWrapper, xfail


class SubTest(TestClass):

    def __run_test(self, method, test_python_wrapper=True,
                   ktype=KernelType.species_rates, **oploop_keywords):
        kwargs = {}
        if not test_python_wrapper:
            kwargs['shared'] = [True, False]
        oploop_keywords.update(kwargs)
        ignored_state_vals = ['conp'] + list(kwargs.keys())

        wrapper = OptionLoopWrapper.from_get_oploop(
            self, ignored_state_vals=ignored_state_vals,
            do_conp=False, **oploop_keywords)
        for opts in wrapper:
            with temporary_build_dirs() as (build_dir, obj_dir, lib_dir):
                # write files
                # write files
                conp = wrapper.state['conp']
                kgen = method(self.store.reacs, self.store.specs, opts,
                              conp=conp)
                # generate
                kgen.generate(build_dir, species_names=[
                    x.name for x in self.store.specs], rxn_strings=[
                    str(x) for x in self.store.reacs])
                # write header
                write_aux(build_dir, opts, self.store.specs, self.store.reacs)
                if test_python_wrapper:
                    package = 'pyjac_{}'.format(utils.package_lang[opts.lang])
                    # test wrapper generation
                    pywrap(opts.lang, build_dir, obj_dir=obj_dir, out_dir=lib_dir,
                           ktype=ktype)

                    imp = test_utils.get_import_source()
                    with open(os.path.join(lib_dir, 'test_import.py'), 'w') as file:
                        file.write(imp.substitute(
                            path=lib_dir, package=package,
                            kernel=utils.enum_to_string(ktype).title(),
                            nsp=len(self.store.specs), nrxn=len(self.store.reacs)))

                    utils.run_with_our_python([
                        os.path.join(lib_dir, 'test_import.py')])
                else:
                    # compile
                    generate_library(opts.lang, build_dir, obj_dir=obj_dir,
                                     out_dir=lib_dir, shared=wrapper.state['shared'],
                                     ktype=ktype)

    @attr('verylong')
    def test_specrates_compilation(self):
        self.__run_test(get_specrates_kernel, test_python_wrapper=True)

    @attr('verylong')
    def test_jacobian_compilation(self):
        self.__run_test(
            get_jacobian_kernel, ktype=KernelType.jacobian,
            # approximate doesn't change much about the code while sparse does!
            test_python_wrapper=True, do_approximate=False, do_sparse=True)

    @attr('verylong')
    @xfail(msg='Finite Difference Jacobian currently broken.')
    def test_fd_jacobian_compilation(self, state):
        self.__run_test(
            finite_difference_jacobian, test_python_wrapper=False,
            ktype=KernelType.jacobian, do_finite_difference=True)

    def test_unique_pointer_specification(self):
        with utils.temporary_directory() as build_dir:
            # test good fixed size
            create_jacobian('c', gas=self.store.gas, unique_pointers=True,
                            data_order='F', build_path=build_dir,
                            kernel_type=KernelType.species_rates)

            files = ['species_rates', 'chem_utils']
            files = [f + ext for f in files for ext in [utils.file_ext['c']]]
            for file in files:
                # read resulting file
                with open(os.path.join(build_dir, file), 'r') as file:
                    file = file.read()
                # and make sure we don't have 'work_size
                assert not re.search(r'\b{}\b'.format(work_size.name), file)

    def __write_with_subs(self, file, inpath, outpath, renamer=None, **subs):
        with open(os.path.join(inpath, file), 'r') as read:
            src = read.read()

        src = utils.subs_at_indent(src, **subs)

        if renamer:
            file = renamer(file)

        with open(os.path.join(outpath, file), 'w') as outfile:
            outfile.write(src)

        return os.path.join(outpath, file)

    def test_read_initial_conditions(self):
        setup = test_utils.get_read_ics_source()
        wrapper = OptionLoopWrapper.from_get_oploop(self, do_conp=True)
        for opts in wrapper:
            with temporary_build_dirs() as (build_dir, obj_dir, lib_dir):
                conp = wrapper.state['conp']

                # make a dummy generator
                insns = (
                    """
                        {spec} = {param} {{id=0}}
                    """
                )
                domain = arc.creator('domain', arc.kint_type, (10,), 'C',
                                     initializer=np.arange(10, dtype=arc.kint_type))
                mapstore = arc.MapStore(opts, domain, None)
                # create global args
                param = arc.creator(arc.pressure_array, np.float64,
                                    (arc.problem_size.name, 10), opts.order)
                spec = arc.creator(arc.state_vector, np.float64,
                                   (arc.problem_size.name, 10), opts.order)
                namestore = type('', (object,), {'jac': ''})
                # create array / array strings
                param_lp, param_str = mapstore.apply_maps(param, 'j', 'i')
                spec_lp, spec_str = mapstore.apply_maps(spec, 'j', 'i')

                # create kernel infos
                info = knl_info(
                    'spec_eval', insns.format(param=param_str, spec=spec_str),
                    mapstore, kernel_data=[spec_lp, param_lp, arc.work_size],
                    silenced_warnings=['write_race(0)'])
                # create generators
                kgen = make_kernel_generator(
                     opts, KernelType.dummy, [info],
                     namestore,
                     input_arrays=[param.name, spec.name],
                     output_arrays=[spec.name],
                     name='ric_tester')
                # make kernels
                kgen._make_kernels()
                # and generate RIC
                _, record, _ = kgen._generate_wrapping_kernel(build_dir)
                kgen._generate_common(build_dir, record)
                ric = os.path.join(build_dir, 'read_initial_conditions' +
                                   utils.file_ext[opts.lang])

                # write header
                write_aux(build_dir, opts, self.store.specs, self.store.reacs)
                with open(os.path.join(build_dir, 'setup.py'), 'w') as file:
                    file.write(setup.safe_substitute(buildpath=build_dir,
                                                     obj_dir=obj_dir))

                # and compile
                from pyjac.libgen import compile, get_toolchain
                toolchain = get_toolchain(opts.lang)
                compile(opts.lang, toolchain, [ric], obj_dir=obj_dir)

                # write wrapper
                self.__write_with_subs(
                    'read_ic_wrapper.pyx',
                    os.path.join(self.store.script_dir, 'test_utils'),
                    build_dir,
                    header_ext=utils.header_ext[opts.lang])
                # setup
                utils.run_with_our_python(
                    [os.path.join(build_dir, 'setup.py'),
                     'build_ext', '--build-lib', lib_dir])

                infile = os.path.join(self.store.script_dir, 'test_utils',
                                      'ric_tester.py.in')
                outfile = os.path.join(lib_dir, 'ric_tester.py')
                # cogify
                try:
                    Cog().callableMain([
                                'cogapp', '-e', '-d', '-Dconp={}'.format(conp),
                                '-o', outfile, infile])
                except Exception:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error('Error generating initial conditions reader:'
                                 ' {}'.format(outfile))
                    raise

                # save phi, param in correct order
                phi = (self.store.phi_cp if conp else self.store.phi_cv)
                savephi = phi.flatten(opts.order)
                param = self.store.P if conp else self.store.V
                savephi.tofile(os.path.join(lib_dir, 'phi_test.npy'))
                param.tofile(os.path.join(lib_dir, 'param_test.npy'))

                # save bin file
                out_file = np.concatenate((
                    np.reshape(phi[:, 0], (-1, 1)),  # temperature
                    np.reshape(param, (-1, 1)),  # param
                    phi[:, 1:]), axis=1  # species
                )
                out_file = out_file.flatten('K')
                with open(os.path.join(lib_dir, 'data.bin'), 'wb') as file:
                    out_file.tofile(file)

                # and run
                utils.run_with_our_python([outfile, opts.order,
                                           str(self.store.test_size)])
