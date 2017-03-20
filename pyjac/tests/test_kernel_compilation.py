import os
import filecmp
from ..core.rate_subs import write_specrates_kernel, write_chem_utils
from . import TestClass
from ..loopy_utils.loopy_utils import loopy_options
from ..libgen import generate_library
from ..core.mech_auxiliary import write_aux, write_mechanism_header
from ..pywrap.pywrap_gen import generate_wrapper
from .. import site_conf as site
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

class SubTest(TestClass):
    def __get_spec_lib(self, state, eqs, opts):
        build_dir = self.store.build_dir
        conp = state['conp']
        kgen = write_specrates_kernel(eqs, self.store.reacs, self.store.specs, opts,
                conp=conp)
        kgen2 = write_chem_utils(self.store.specs, eqs, opts)
        #add deps
        kgen.add_depencencies([kgen2])
        #generate
        kgen.generate(build_dir)
        #write header
        write_aux(build_dir, opts, self.store.specs, self.store.reacs)

    def __get_objs(self):
        opts = loopy_options(lang='opencl',
                    width=None, depth=None, ilp=False,
                    unr=None, order='C', platform='CPU')
        eqs = {'conp' : self.store.conp_eqs, 'conv' : self.store.conv_eqs}

        oploop = OptionLoop(OrderedDict([
            ('conp', [True, False]),
            ('shared', [True, False])]))
        return opts, eqs, oploop

    def test_compile_specrates_knl(self):
        opts, eqs, oploop = self.__get_objs()
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        for state in oploop:
            self.__get_spec_lib(state, eqs, opts)
            #compile
            generate_library(opts.lang, build_dir, obj_dir=obj_dir,
                         build_dir=obj_dir,
                         out_dir=lib_dir, shared=state['shared'],
                         finite_difference=False, auto_diff=False)

    def test_specrates_pywrap(self):
        opts, eqs, oploop = self.__get_objs()
        build_dir = self.store.build_dir
        obj_dir = self.store.obj_dir
        lib_dir = self.store.lib_dir
        for state in oploop:
            self.__get_spec_lib(state, eqs, opts)
            #test wrapper generation
            generate_wrapper(opts.lang, build_dir,
                         out_dir=lib_dir)
            #test import
            pywrap = importlib.import_module('pyjac_ocl')

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
            #get source
            path = os.path.realpath(os.path.join(self.store.script_dir, os.pardir,
                'kernel_utils', 'common', 'read_initial_conditions.c.in'))
            with open(path, 'r') as file:
                ric = Template(file.read())
            #subs
            ric = ric.safe_substitute(mechanism='mechanism.h')
            #write
            with open(os.path.join(build_dir, 'read_initial_conditions.c'), 'w') as file:
                file.write(ric)
            #write header
            write_mechanism_header(build_dir, 'c', self.store.specs, self.store.reacs)
            #write setup
            with open(os.path.join(build_dir, 'setup.py'), 'w') as file:
                file.write(setup.safe_substitute(
                    buildpath=build_dir))
            #copy read ics header to final dest
            shutil.copyfile(os.path.join(self.store.script_dir, os.pardir,
                'kernel_utils', 'common', 'read_initial_conditions.h'),
                os.path.join(build_dir, 'read_initial_conditions.h'))
            #copy wrapper
            shutil.copyfile(os.path.join(self.store.script_dir, 'test_utils',
                'read_ic_wrapper.pyx'),
                os.path.join(build_dir, 'read_ic_wrapper.pyx'))
            #setup
            os.chdir(build_dir)
            python_str = 'python{}.{}'.format(sys.version_info[0], sys.version_info[1])
            call = [python_str, os.path.join(build_dir, 'setup.py'),
                           'build_ext', '--build-lib', lib_dir]
            subprocess.check_call(call)
            #copy in tester
            shutil.copyfile(os.path.join(self.store.script_dir, 'test_utils',
                    'ric_tester.py'),
                os.path.join(lib_dir, 'ric_tester.py'))

            #save T, P, concs in correct order
            self.store.T.tofile(os.path.join(lib_dir, 'T_test.npy'))
            self.store.P.tofile(os.path.join(lib_dir, 'P_test.npy'))
            save_concs = (self.store.concs.copy() if order == 'F' else self.store.concs.T.copy()).flatten('K')
            save_concs.tofile(os.path.join(lib_dir, 'conc_test.npy'))

            #save bin file
            out_file = np.concatenate((
                    np.reshape(self.store.T, (-1, 1)),
                    np.reshape(self.store.P, (-1, 1)),
                    self.store.concs.T.copy()), axis=1
                )
            with open(os.path.join(lib_dir, 'data.bin'), 'wb') as file:
                out_file.tofile(file)

            #and run
            os.chdir(lib_dir)
            subprocess.check_call([python_str, 'ric_tester.py', order, str(self.store.test_size)])
