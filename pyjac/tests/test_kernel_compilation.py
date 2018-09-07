import os
import shutil
import re

import numpy as np
from nose.tools import assert_raises
from nose.plugins.attrib import attr

from pyjac import utils
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.core.create_jacobian import get_jacobian_kernel, \
    finite_difference_jacobian, create_jacobian
from pyjac.tests import TestClass, test_utils
from pyjac.libgen import generate_library
from pyjac.core.enum_types import KernelType
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core.array_creator import array_splitter, work_size
from pyjac.core.exceptions import InvalidInputSpecificationException
from pyjac.pywrap.pywrap_gen import pywrap
from pyjac.tests.test_utils import temporary_build_dirs, OptionLoopWrapper, xfail


class SubTest(TestClass):

    def __run_test(self, method, test_python_wrapper=True,
                   ktype=KernelType.species_rates, **oploop_keywords):
        wrapper = OptionLoopWrapper.from_get_oploop(
            self, shared=[True, False], ignored_state_vals=['conp', 'shared'],
            **oploop_keywords)
        for opts in wrapper:
            with temporary_build_dirs() as (build_dir, obj_dir, lib_dir):
                # write files
                # write files
                conp = wrapper.state['conp']
                kgen = method(self.store.reacs, self.store.specs, opts,
                              conp=conp)
                # generate
                kgen.generate(build_dir)
                # write header
                write_aux(build_dir, opts, self.store.specs, self.store.reacs)
                if test_python_wrapper:
                    package = 'pyjac_{}'.format(utils.package_lang[opts.lang])
                    # test wrapper generation
                    pywrap(opts.lang, build_dir, obj_dir=obj_dir, out_dir=lib_dir,
                           ktype=ktype)

                    imp = test_utils.get_import_source()
                    with open(os.path.join(lib_dir, 'test_import.py'), 'w') as file:
                        file.write(imp.substitute(path=lib_dir, package=package))

                    utils.run_with_our_python([
                        os.path.join(lib_dir, 'test_import.py')])
                else:
                    # compile
                    generate_library(opts.lang, build_dir, obj_dir=obj_dir,
                                     out_dir=lib_dir, shared=wrapper.state['shared'],
                                     ktype=ktype)

    @attr('long')
    def test_specrates_compilation(self):
        self.__run_test(get_specrates_kernel, test_python_wrapper=True)

    @attr('verylong')
    def test_jacobian_compilation(self):
        self.__run_test(
            get_jacobian_kernel, ktype=KernelType.jacobian,
            test_python_wrapper=True, do_approximate=True)

    @attr('long')
    @xfail(msg='Finite Difference Jacobian currently broken.')
    def test_fd_jacobian_compilation(self, state):
        self.__run_test(
            finite_difference_jacobian, test_python_wrapper=False,
            ktype=KernelType.jacobian, do_finite_difference=True)

    def test_fixed_work_size(self):
        # test bad fixed size
        with assert_raises(InvalidInputSpecificationException):
            create_jacobian(
                'opencl', gas=self.store.gas, vector_size=4, wide=True, work_size=1)

        with utils.temporary_directory() as build_dir:
            # test good fixed size
            create_jacobian('c', gas=self.store.gas, work_size=1,
                            data_order='F', build_path=build_dir,
                            kernel_type=KernelType.species_rates)

            files = ['species_rates.c', 'species_rates.h', 'chem_utils.c',
                     'chem_utils.h']
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
        for opts in OptionLoopWrapper.from_get_oploop(self):
            with temporary_build_dirs() as (build_dir, obj_dir, lib_dir):
                # create dummy loopy opts
                asplit = array_splitter(opts)

                header_ext = utils.header_ext[opts.lang]
                file_ext = utils.file_ext[opts.lang]

                # write initial condition file
                ric = self.__write_with_subs(
                    'read_initial_conditions.c.in',
                    os.path.realpath(
                        os.path.join(self.store.script_dir, os.pardir,
                                     'kernel_utils', 'common')),
                    build_dir,
                    renamer=lambda x: x[:x.index('.')] + file_ext,
                    mechanism='mechanism' + header_ext,
                    vectorization='vectorization' + header_ext)

                # write header
                write_aux(build_dir, opts, self.store.specs, self.store.reacs)
                with open(os.path.join(build_dir, 'setup.py'), 'w') as file:
                    file.write(setup.safe_substitute(buildpath=build_dir,
                                                     obj_dir=obj_dir))

                # and compile
                from pyjac.libgen import compile, get_toolchain
                toolchain = get_toolchain(opts.lang)
                compile(opts.lang, toolchain, [ric], obj_dir=obj_dir)

                # copy read ics header to final dest
                read_ic_header = os.path.join(
                    self.store.script_dir, os.pardir,
                    'kernel_utils', 'common',
                    'read_initial_conditions.h')
                utils.copy_with_extension(
                    opts.lang, read_ic_header, build_dir, header=True)
                # write wrapper
                self.__write_with_subs(
                    'read_ic_wrapper.pyx',
                    os.path.join(self.store.script_dir, 'test_utils'),
                    build_dir,
                    header_ext=header_ext)
                # setup
                utils.run_with_our_python(
                    [os.path.join(build_dir, 'setup.py'),
                     'build_ext', '--build-lib', lib_dir])
                # copy in tester
                shutil.copyfile(os.path.join(self.store.script_dir, 'test_utils',
                                             'ric_tester.py'),
                                os.path.join(lib_dir, 'ric_tester.py'))

                # For simplicity (and really, lack of need) we test CONP only
                # hence, the extra variable is the volume, while the fixed parameter
                # is the pressure

                # save phi, param in correct order
                conp = True
                phi = (self.store.phi_cp if conp else self.store.phi_cv)
                save_phi, = asplit.split_numpy_arrays(phi)
                save_phi = save_phi.flatten(opts.order)
                param = self.store.P if conp else self.store.V
                save_phi.tofile(os.path.join(lib_dir, 'phi_test.npy'))
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
                utils.run_with_our_python([
                    os.path.join(lib_dir, 'ric_tester.py'), opts.order,
                    str(self.store.test_size)])
