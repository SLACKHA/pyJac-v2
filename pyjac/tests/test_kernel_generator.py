import os
from collections import OrderedDict
import shutil

from optionloop import OptionLoop
import loopy as lp
from loopy.kernel.array import ArrayBase

from pyjac.core.create_jacobian import get_jacobian_kernel
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.utils import partition
from pyjac.tests import TestClass, test_utils, get_test_langs
from pyjac.tests.test_utils import OptionLoopWrapper
from pyjac.core.mech_auxiliary import write_aux


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

    def __get_oploop(self):
        oploop = OptionLoop(OrderedDict([
            ('conp', [True]),
            ('shared', [True, False]),
            ('lang', get_test_langs()),
            ('width', [4, None]),
            ('depth', [4, None]),
            ('order', ['C', 'F'])]))
        return oploop

    def test_process_args(self):
        oploop = OptionLoopWrapper.from_get_oploop(self)
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # and process the arguements
            args, local, readonly, constants, valueargs = kgen._process_args()

            # and check
            # 1) that all arguments in all kernels are in the args
            for kernel in kgen.kernels + [
                    knl for dep in kgen.depends_on for knl in dep.kernels]:
                arrays, values = partition(kernel.args, lambda x: isinstance(
                    x, ArrayBase))
                assert all(arg in args for arg in arrays)
                assert all(val in valueargs for val in values)
                assert not any(read in kernel.get_written_variables()
                               for read in readonly)

            # if deep vectorization, make sure we have a local argument
            if opts.depth:
                assert len(local)

            # check that all constants are temporary variables
            assert all(isinstance(x, lp.TemporaryVariable) for x in constants)
