import os
from collections import OrderedDict
import shutil
from tempfile import NamedTemporaryFile
import re

from optionloop import OptionLoop
import loopy as lp
from loopy.kernel.array import ArrayBase
from loopy.kernel.data import AddressSpace as scopes
from loopy.types import to_loopy_type
from nose.tools import assert_raises
import six

from pyjac.core.create_jacobian import get_jacobian_kernel
from pyjac.core.enum_types import JacobianFormat
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core import array_creator as arc
from pyjac.loopy_utils.preambles_and_manglers import jac_indirect_lookup
from pyjac.kernel_utils.memory_manager import memory_type
from pyjac.kernel_utils.kernel_gen import kernel_generator, TargetCheckingRecord
from pyjac.utils import partition
from pyjac.tests import TestClass, test_utils, get_test_langs
from pyjac.tests.test_utils import OptionLoopWrapper, temporary_directory


# get all kernels
def rec_kernel(gen, kernels=[]):
    if not gen.depends_on:
        return kernels + gen.kernels
    kernels = kernels + gen.kernels
    for dep in gen.depends_on:
        kernels += rec_kernel(dep)
    return kernels


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

                kernels = rec_kernel(kgen)
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

    def test_working_buffers(self):
        # test vector to ensure the various working buffer configurations work
        # (i.e., locals)
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=True,
                                                   do_sparse=False)
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record = kgen._process_args()

            # test that process memory works
            record, mem_limits = kgen._process_memory(record)

            # and generate working buffers
            recordnew, result = kgen._compress_to_working_buffer(record)

            if opts.depth:
                # check for local
                assert next((x for x in recordnew.kernel_data
                             if x.address_space == scopes.LOCAL), None)

            def __check_unpacks(unpacks, offsets, args):
                for arg in args:
                    # check that all args are in the unpacks
                    unpack = next((x for x in unpacks if re.search(
                        r'\b' + arg.name + r'\b', x)), None)
                    assert unpack
                    # next check the type
                    assert kgen.type_map[arg.dtype] in unpack
                    # and scope, if needed
                    if arg.address_space == scopes.LOCAL:
                        assert 'local' in unpack
                    # and in offset
                    assert arg.name in offsets

            def __check_local_unpacks(result, args):
                for i, arg in enumerate(args):
                    # get offset
                    offsets = result.pointer_offsets[arg.name][1]
                    new = kgen._get_local_unpacks(result, [arg])
                    if not new.pointer_unpacks:
                        assert isinstance(arg, lp.TemporaryVariable)
                    else:
                        # and check
                        assert re.search(r'\b' + re.escape(offsets) + r'\b',
                                         new.pointer_unpacks[0])

            # check that all args are in the pointer unpacks
            __check_unpacks(result.pointer_unpacks, result.pointer_offsets,
                            recordnew.args + recordnew.local
                            + recordnew.host_constants)
            # check unpacks for driver function (note: this isn't the 'local' scope
            # rather, local copies out of the working buffer)
            __check_local_unpacks(result, recordnew.args)
            # next, write a dummy input file, such that we can force the constant
            # memory allocation to zero
            with NamedTemporaryFile(suffix='.yaml', mode='w') as temp:
                temp.write("""
                    memory-limits:
                        constant: 0 B
                    """)
                temp.seek(0)

                # set file
                kgen.mem_limits = temp.name

                # reprocesses
                noconst, mem_limits = kgen._process_memory(record)
                noconst, result = kgen._compress_to_working_buffer(noconst)

                # check that we have an integer workspace
                int_type = to_loopy_type(arc.kint_type, target=kgen.target)
                assert next((x for x in noconst.kernel_data
                             if x.dtype == int_type), None)

                # and recheck pointer unpacks (including host constants)
                __check_unpacks(
                    result.pointer_unpacks, result.pointer_offsets,
                    recordnew.args + recordnew.local + record.constants)
                __check_local_unpacks(result, recordnew.args + recordnew.local +
                                      record.constants)

    def test_merge_kernels(self):
        # test vector to ensure the various working buffer configurations work
        # (i.e., locals)
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=True,
                                                   do_sparse=False)
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record = kgen._process_args()

            # test that process memory works
            record, mem_limits = kgen._process_memory(record)

            # and generate working buffers
            recordnew, result = kgen._compress_to_working_buffer(record)

            result = kgen._merge_kernels(record, result)

            # get ownership
            owner = kgen._get_kernel_ownership(kgen.kernels)

            # check we have generated our own kernels
            for kernel in kgen.kernels:
                if owner[kernel.name] == kgen:
                    assert re.search(
                        r'\b' + kernel.name + r'\b', result.extra_kernels)
                else:
                    assert not re.search(
                        r'\b' + kernel.name + r'\b', result.extra_kernels)

            # check that we have the instruction call to _all_ kernels
            all_kernels = rec_kernel(kgen)
            for kernel in all_kernels:
                assert re.search(
                    r'\b' + kernel.name + r'\b', result.instructions)

    def test_init_deduplication(self):
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=False,
                                                   do_sparse=False)
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record = kgen._process_args()

            # test that process memory works
            record, mem_limits = kgen._process_memory(record)

            # and generate working buffers
            recordnew, result = kgen._compress_to_working_buffer(record)

            result = kgen._merge_kernels(record, result)

            # and de-duplicate
            results = kgen._constant_deduplication(record, result)

            # check inits
            inits = {}
            for result in results:
                for k, v in six.iteritems(result.inits):
                    assert k not in inits
                    inits[k] = v

    def test_compilation_generator(self):
        # currently separate compiler code only exists for OpenCL
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   langs=['opencl'],
                                                   do_conp=False,
                                                   do_vector=False,
                                                   do_sparse=False)
        for opts in oploop:
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            with temporary_directory() as tdir:
                comp = kgen._generate_compiling_program(
                    tdir, ['adistinctivetestname', 'andyetanothertestname'])

                with open(comp, 'r') as file:
                    comp = file.read()
                # test filenames
                assert '"adistinctivetestname", "andyetanothertestname"' in comp
                # test build options
                assert kgen._get_cl_level() in comp
                # outname
                assert 'char* out_name = "{}";'.format(kgen.name + '.bin')
                # and platform
                assert 'char* platform = "{}";'.format(
                    opts.platform.vendor)


def test_remove_worksize():
    assert kernel_generator._remove_work_size(
        '(int const work_size, int dummy)') == '(int dummy)'
    assert kernel_generator._remove_work_size(
        '(int dummy, int const work_size)') == '(int dummy)'
    assert kernel_generator._remove_work_size(
        '(int dummy, int const work_size, int dummy2)') == '(int dummy, int dummy2)'
    assert kernel_generator._remove_work_size(
        'call(dummy, work_size)') == 'call(dummy)'
    assert kernel_generator._remove_work_size(
        'call(work_size, dummy)') == 'call(dummy)'
    assert kernel_generator._remove_work_size(
        'call(dummy, work_size, dummy)') == 'call(dummy, dummy)'


def test_target_record():
    # make bad argument (i.e, one without the target set)
    import numpy as np
    from loopy.types import to_loopy_type
    bad = lp.GlobalArg('bad', dtype=np.int32, shape=(1,), order='C')

    def __check(record):
        with assert_raises(AssertionError):
            record.__getstate__()

    # and check list
    __check(TargetCheckingRecord(kernel_data=[bad]))
    # dictionary
    __check(TargetCheckingRecord(kernel_data={'a': bad}))
    # dictionary of lists
    __check(TargetCheckingRecord(kernel_data={'a': [bad]}))
    # and plain value
    __check(TargetCheckingRecord(kernel_data=bad))
    # numpy dtype as dictionary key
    dtype = to_loopy_type(np.int32)
    __check(TargetCheckingRecord(kernel_data={dtype: 'bad'}))
