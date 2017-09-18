import os
from ..core.rate_subs import write_specrates_kernel, write_chem_utils, assign_rates
from . import TestClass
from ..loopy_utils.loopy_utils import loopy_options, RateSpecialization
from ..libgen import generate_library
from ..core.mech_auxiliary import write_aux, write_mechanism_header
from ..pywrap.pywrap_gen import generate_wrapper
from ..core.array_creator import NameStore
from . import test_utils as test_utils
from optionloop import OptionLoop
from collections import OrderedDict
import importlib
import shutil
from string import Template
import sys
import subprocess
import numpy as np
from .. import utils
from parameterized import parameterized


class SubTest(TestClass):
    def __cleanup(self):
        # remove library
        test_utils.clean_dir(self.store.lib_dir)
        # remove build
        test_utils.clean_dir(self.store.obj_dir)
        # clean dummy builder
        dist_build = os.path.join(self.store.build_dir, 'build')
        if os.path.exists(dist_build):
            shutil.rmtree(dist_build)
        # clean sources
        test_utils.clean_dir(self.store.build_dir)

    def __get_spec_lib(self, state, eqs, opts):
        build_dir = self.store.build_dir
        conp = state['conp']
        kgen = write_specrates_kernel(eqs, self.store.reacs, self.store.specs, opts,
                                      conp=conp)
        # generate
        kgen.generate(build_dir)
        # write header
        write_aux(build_dir, opts, self.store.specs, self.store.reacs)

    def __get_objs(self, lang='opencl'):
        opts = loopy_options(lang=lang,
                             width=None, depth=None, ilp=False,
                             unr=None, order='C', platform='CPU')
        eqs = {'conp': self.store.conp_eqs, 'conv': self.store.conv_eqs}

        oploop = OptionLoop(OrderedDict([
            ('conp', [True, False]),
            ('shared', [True, False])]))
        return opts, eqs, oploop

    @parameterized.expand([('opencl',), ('c',)])
    def test_compile_specrates_knl(self, lang):
        opts, eqs, oploop = self.__get_objs(lang=lang)
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        for state in oploop:
            # clean old
            self.__cleanup()
            # create / write files
            self.__get_spec_lib(state, eqs, opts)
            # compile
            generate_library(opts.lang, build_dir, obj_dir=obj_dir,
                             out_dir=lib_dir, shared=state['shared'],
                             auto_diff=False)

    @parameterized.expand([('opencl',), ('c',)])
    def test_specrates_pywrap(self, lang):
        opts, eqs, oploop = self.__get_objs(lang=lang)
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        for state in oploop:
            # clean old
            self.__cleanup()
            # create / write files
            self.__get_spec_lib(state, eqs, opts)
            # test wrapper generation
            generate_wrapper(opts.lang, build_dir,
                             obj_dir=obj_dir, out_dir=lib_dir, auto_diff=False)
            # test import
            importlib.import_module('pyjac_ocl')

    def test_read_initial_conditions(self):
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        setup = test_utils.get_read_ics_source()
        utils.create_dir(build_dir)
        utils.create_dir(obj_dir)
        utils.create_dir(lib_dir)
        home = os.getcwd()
        for order in ['C', 'F']:
            os.chdir(home)
            test_utils.clean_dir(build_dir, False)
            test_utils.clean_dir(obj_dir, False)
            test_utils.clean_dir(lib_dir, False)
            if os.path.exists(os.path.join(os.getcwd(), 'build')):
                shutil.rmtree(os.path.join(os.getcwd(), 'build'))
            # get source
            path = os.path.realpath(
                os.path.join(self.store.script_dir, os.pardir,
                             'kernel_utils', 'common',
                             'read_initial_conditions.c.in'))

            with open(path, 'r') as file:
                ric = Template(file.read())
            # subs
            ric = ric.safe_substitute(mechanism='mechanism.h')
            # write
            with open(os.path.join(
                    build_dir, 'read_initial_conditions.c'), 'w') as file:
                file.write(ric)
            # write header
            write_mechanism_header(
                build_dir, 'c', self.store.specs, self.store.reacs)
            # write setup
            with open(os.path.join(build_dir, 'setup.py'), 'w') as file:
                file.write(setup.safe_substitute(
                    buildpath=build_dir))
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
            os.chdir(build_dir)
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

            # save phi, V in correct order
            save_phi = (self.store.phi_cp.copy() if order == 'C'
                        else self.store.phi_cp.T.copy()).flatten('K')
            save_phi.tofile(os.path.join(lib_dir, 'phi_test.npy'))
            self.store.P.tofile(os.path.join(lib_dir, 'param_test.npy'))

            # save bin file
            out_file = np.concatenate((
                np.reshape(self.store.phi_cp[:, 0], (-1, 1)),
                np.reshape(self.store.P, (-1, 1)),  # param
                self.store.phi_cp[:, 1:]), axis=1  # species & extra var
            )
            out_file = out_file.flatten('K')
            with open(os.path.join(lib_dir, 'data.bin'), 'wb') as file:
                out_file.tofile(file)

            # and run
            os.chdir(lib_dir)
            subprocess.check_call(
                [python_str, 'ric_tester.py', order, str(self.store.test_size)])
