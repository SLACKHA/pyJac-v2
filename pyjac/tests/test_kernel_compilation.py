import os
from collections import OrderedDict
import shutil
from string import Template
import sys
import subprocess

from optionloop import OptionLoop
import numpy as np
from parameterized import parameterized, param

from pyjac import utils
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.core.create_jacobian import get_jacobian_kernel, \
    finite_difference_jacobian
from pyjac.tests import TestClass, test_utils
from pyjac.loopy_utils.loopy_utils import loopy_options
from pyjac.libgen import generate_library, build_type
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core.array_creator import array_splitter
from pyjac.pywrap.pywrap_gen import generate_wrapper


class SubTest(TestClass):
    def __cleanup(self, remove_dirs=True):
        # remove library
        test_utils.clean_dir(self.store.lib_dir, remove_dirs)
        # remove build
        test_utils.clean_dir(self.store.obj_dir, remove_dirs)
        # clean dummy builder
        dist_build = os.path.join(self.store.build_dir, 'build')
        if os.path.exists(dist_build):
            shutil.rmtree(dist_build)
        # clean sources
        test_utils.clean_dir(self.store.build_dir, remove_dirs)

    def __get_spec_lib(self, state, opts):
        build_dir = self.store.build_dir
        conp = state['conp']
        kgen = get_specrates_kernel(self.store.reacs, self.store.specs, opts,
                                    conp=conp)
        # generate
        kgen.generate(build_dir)
        # write header
        write_aux(build_dir, opts, self.store.specs, self.store.reacs)

    def __get_objs(self, lang='opencl', depth=None, width=None, order='C'):
        opts = loopy_options(lang=lang,
                             width=width, depth=depth, ilp=False,
                             unr=None, order=order, platform='CPU')

        oploop = OptionLoop(OrderedDict([
            ('conp', [True]),
            ('shared', [True, False])]))
        return opts, oploop

    @parameterized.expand([('opencl',), ('c',)])
    def test_compile_specrates_knl(self, lang):
        opts, oploop = self.__get_objs(lang=lang)
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        for state in oploop:
            # clean old
            self.__cleanup()
            # create / write files
            self.__get_spec_lib(state, opts)
            # compile
            generate_library(opts.lang, build_dir, obj_dir=obj_dir,
                             out_dir=lib_dir, shared=state['shared'],
                             btype=build_type.species_rates)

    @parameterized.expand([('opencl',), ('c',)])
    def test_specrates_pywrap(self, lang):
        opts, oploop = self.__get_objs(lang=lang)
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        packages = {'c': 'pyjac_c', 'opencl': 'pyjac_ocl'}
        for state in oploop:
            # clean old
            self.__cleanup()
            # create / write files
            self.__get_spec_lib(state, opts)
            # test wrapper generation
            generate_wrapper(opts.lang, build_dir, obj_dir=obj_dir, out_dir=lib_dir,
                             btype=build_type.species_rates)

            # create the test importer, and run
            imp = test_utils.get_import_source()
            with open(os.path.join(lib_dir, 'test_import.py'), 'w') as file:
                file.write(imp.substitute(path=lib_dir, package=packages[lang]))

            python_str = 'python{}.{}'.format(
                sys.version_info[0], sys.version_info[1])
            subprocess.check_call([python_str,
                                   os.path.join(lib_dir, 'test_import.py')])

    def __test_cases():
        for state in OptionLoop(OrderedDict(
                [('lang', ['opencl', 'c']),
                 (('jac_type'), ['exact', 'approximate', 'finite_difference'])])):
            yield param(state)

    @parameterized.expand(__test_cases,
                          testcase_func_name=lambda x, y, z:
                          '{0}_{1}_{2}'.format(x.__name__, z[0][0]['lang'],
                                               z[0][0]['jac_type']))
    def test_compile_jacobian(self, state):
        lang = state['lang']
        jac_type = state['jac_type']
        opts, oploop = self.__get_objs(lang=lang)
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        packages = {'c': 'pyjac_c', 'opencl': 'pyjac_ocl'}
        for state in oploop:
            # clean old
            self.__cleanup()
            # create / write files
            build_dir = self.store.build_dir
            conp = state['conp']
            method = get_jacobian_kernel
            if jac_type == 'finite_difference':
                method = finite_difference_jacobian
            kgen = method(self.store.reacs, self.store.specs, opts,
                          conp=conp)
            # generate
            kgen.generate(build_dir)
            # write header
            write_aux(build_dir, opts, self.store.specs, self.store.reacs)
            # test wrapper generation
            generate_wrapper(opts.lang, build_dir, obj_dir=obj_dir, out_dir=lib_dir,
                             btype=build_type.jacobian)

            # create the test importer, and run
            imp = test_utils.get_import_source()
            with open(os.path.join(lib_dir, 'test_import.py'), 'w') as file:
                file.write(imp.substitute(path=lib_dir, package=packages[lang]))

            python_str = 'python{}.{}'.format(
                sys.version_info[0], sys.version_info[1])
            subprocess.check_call([python_str,
                                   os.path.join(lib_dir, 'test_import.py')])

    def test_read_initial_conditions(self):
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        setup = test_utils.get_read_ics_source()
        utils.create_dir(build_dir)
        utils.create_dir(obj_dir)
        utils.create_dir(lib_dir)
        oploop = OptionLoop(OrderedDict([
            # no need to test conv
            ('conp', [True]),
            ('order', ['C', 'F']),
            ('depth', [4, None]),
            ('width', [4, None]),
            ('lang', ['c'])]))
        for state in oploop:
            if state['depth'] and state['width']:
                continue
            self.__cleanup(False)
            # create dummy loopy opts
            opts = type('', (object,), state)()
            asplit = array_splitter(opts)

            # get source
            path = os.path.realpath(
                os.path.join(self.store.script_dir, os.pardir,
                             'kernel_utils', 'common',
                             'read_initial_conditions.c.in'))

            with open(path, 'r') as file:
                ric = Template(file.read())
            # subs
            ric = ric.safe_substitute(mechanism='mechanism.h',
                                      vectorization='vectorization.h')
            # write
            with open(os.path.join(
                    build_dir, 'read_initial_conditions.c'), 'w') as file:
                file.write(ric)
            # write header
            write_aux(build_dir, opts, self.store.specs, self.store.reacs)
            # write setup
            with open(os.path.join(build_dir, 'setup.py'), 'w') as file:
                file.write(setup.safe_substitute(buildpath=build_dir))
            # copy read ics header to final dest
            shutil.copyfile(os.path.join(self.store.script_dir, os.pardir,
                                         'kernel_utils', 'common',
                                         'read_initial_conditions.h'),
                            os.path.join(build_dir, 'read_initial_conditions.h'))
            # copy wrapper
            shutil.copyfile(os.path.join(self.store.script_dir, 'test_utils',
                                         'read_ic_wrapper.pyx'),
                            os.path.join(build_dir, 'read_ic_wrapper.pyx'))
            # setup
            python_str = 'python{}.{}'.format(
                sys.version_info[0], sys.version_info[1])
            call = [python_str, os.path.join(build_dir, 'setup.py'),
                    'build_ext', '--build-lib', lib_dir]
            subprocess.check_call(call)
            # copy in tester
            shutil.copyfile(os.path.join(self.store.script_dir, 'test_utils',
                                         'ric_tester.py'),
                            os.path.join(lib_dir, 'ric_tester.py'))

            # For simplicity (and really, lack of need) we test CONP only
            # hence, the extra variable is the volume, while the fixed parameter
            # is the pressure

            # save phi, param in correct order
            phi = (self.store.phi_cp if opts.conp else self.store.phi_cv)
            save_phi, = asplit.split_numpy_arrays(phi)
            save_phi = save_phi.flatten(opts.order)
            param = self.store.P if opts.conp else self.store.V
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
            subprocess.check_call(
                [python_str, os.path.join(lib_dir, 'ric_tester.py'), opts.order,
                 str(self.store.test_size)])
