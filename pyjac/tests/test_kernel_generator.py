import os
from collections import OrderedDict
import shutil
from tempfile import NamedTemporaryFile

from optionloop import OptionLoop
import loopy as lp
from loopy.kernel.array import ArrayBase
from nose.tools import assert_raises

from pyjac.core.create_jacobian import get_jacobian_kernel
from pyjac.core.enum_types import JacobianFormat
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.core.mech_auxiliary import write_aux
from pyjac.loopy_utils.preambles_and_manglers import jac_indirect_lookup
from pyjac.kernel_utils.memory_manager import memory_type
from pyjac.utils import partition
from pyjac.tests import TestClass, test_utils, get_test_langs
from pyjac.tests.test_utils import OptionLoopWrapper


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
        # we only really need to test OpenCL here, as we just want to ensure
        # deep vectorizations have local args
        oploop = OptionLoopWrapper.from_get_oploop(self, do_conp=False,
                                                   langs=['opencl'])
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # and process the arguements
            record = kgen._process_args()

            # and check
            # 1) that all arguments in all kernels are in the args
            for kernel in kgen.kernels + [
                    knl for dep in kgen.depends_on for knl in dep.kernels]:
                arrays, values = partition(kernel.args, lambda x: isinstance(
                    x, ArrayBase))
                assert all(any(kgen._compare_args(a1, a2)
                           for a2 in record.args) for a1 in arrays)
                assert all(val in record.valueargs for val in values)
                assert not any(read in kernel.get_written_variables()
                               for read in record.readonly)

            # if deep vectorization, make sure we have a local argument
            if opts.depth:
                assert len(record.local)

            # check that all constants are temporary variables
            assert all(isinstance(x, lp.TemporaryVariable) for x in record.constants)

            # now, insert a bad duplicate argument and make sure we get an error
            i_arg, arg = next((i, arg) for i, arg in enumerate(kgen.kernels[0].args)
                              if isinstance(arg, ArrayBase))
            new = arg.__class__(shape=tuple([x + 1 for x in arg.shape]),
                                name=arg.name,
                                order=opts.order,
                                address_space=arg.address_space)
            kgen.kernels[0] = kgen.kernels[0].copy(
                args=kgen.kernels[0].args + [new])

            with assert_raises(Exception):
                kgen._process_args()

    def test_process_memory(self):
        # test sparse in order to ensure the Jacobian preambles aren't removed
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=False,
                                                   do_sparse=True)
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record = kgen._process_args()

            # test that process memory works
            _, mem_limits = kgen._process_memory(record)

            limit = 0
            if opts.jac_format == JacobianFormat.sparse:
                # need to update the limit for the constant memory such that the
                # sparse indicies can fit
                preamble = next(x for x in kgen.extra_preambles
                                if isinstance(x, jac_indirect_lookup))
                limit = preamble.array.nbytes

            # next, write a dummy input file, such that we can force the constant
            # memory allocation to zero
            with NamedTemporaryFile(suffix='.yaml', mode='w') as temp:
                temp.write("""
                    memory-limits:
                        constant: {} B
                    """.format(limit))
                temp.seek(0)

                # set file
                kgen.mem_limits = temp.name

                # reprocesses
                noconst, mem_limits = kgen._process_memory(record)

                # and test that all constants (except potentially the jacobian
                # indicies) have been migrated
                if opts.jac_format == JacobianFormat.sparse:
                    assert len(mem_limits.arrays[memory_type.m_constant]) == 1
                    jac_inds = mem_limits.arrays[memory_type.m_constant][0]
                    assert all(x in record.constants for x in noconst.host_constants)
                    assert all(x in mem_limits.arrays[memory_type.m_global] for x in
                               record.constants if x != jac_inds)
                else:
                    assert not len(mem_limits.arrays[memory_type.m_constant])
                    assert all(x in record.constants for x in noconst.host_constants)
                    assert all(x in mem_limits.arrays[memory_type.m_global] for x in
                               record.constants)

                # and because we can, test the host constant migration at this point
                # get all kernels
                def __rec_kernel(gen, kernels=[]):
                    if not gen.depends_on:
                        return kernels + gen.kernels
                    kernels = kernels + gen.kernels
                    for dep in gen.depends_on:
                        kernels += __rec_kernel(dep)
                    return kernels

                kernels = __rec_kernel(kgen)
                kernels = kgen._migrate_host_constants(
                    kernels, noconst.host_constants)
                to_find = set([x.name for x in noconst.host_constants])
                for kernel in kernels:
                    # check temps
                    assert not any(x in kernel.temporary_variables.keys()
                                   for x in record.host_constants)
                    # and args
                    to_find = to_find - set([x.name for x in kernel.args])

                assert not len(to_find)
