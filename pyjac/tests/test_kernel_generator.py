import os
from collections import OrderedDict
import shutil
from tempfile import NamedTemporaryFile
import re
import textwrap

import numpy as np
import loopy as lp
from optionloop import OptionLoop
from loopy.kernel.array import ArrayBase
from loopy.kernel.data import AddressSpace as scopes
from loopy.types import to_loopy_type
from nose.tools import assert_raises
import six

from pyjac.core.create_jacobian import get_jacobian_kernel
from pyjac.core.enum_types import JacobianFormat, KernelType
from pyjac.core.rate_subs import get_specrates_kernel
from pyjac.core.mech_auxiliary import write_aux
from pyjac.core import array_creator as arc
from pyjac.loopy_utils.preambles_and_manglers import jac_indirect_lookup, \
    PreambleGen
from pyjac.kernel_utils.memory_limits import memory_type
from pyjac.kernel_utils.kernel_gen import kernel_generator, TargetCheckingRecord, \
    knl_info, make_kernel_generator, CallgenResult, local_work_name, rhs_work_name
from pyjac.utils import partition, temporary_directory, clean_dir, \
    can_vectorize_lang, header_ext, file_ext
from pyjac.tests import TestClass, get_test_langs
from pyjac.tests.test_utils import OptionLoopWrapper


# get all kernels
def rec_kernel(gen, kernels=[]):
    if not gen.depends_on:
        return kernels + gen.kernels
    kernels = kernels + gen.kernels
    for dep in gen.depends_on:
        kernels += rec_kernel(dep)
    return kernels


class dummy_preamble(PreambleGen):
    def __init__(self):
        dtype = 'int' if arc.kint_type == np.int32 else 'long int'
        super(dummy_preamble, self).__init__(
            'dummy', textwrap.dedent("""
            {dtype} dummy({dtype} val, {dtype} one)
            {{
                return val;
            }}""").format(dtype=dtype),
            (arc.kint_type, arc.kint_type), arc.kint_type)

    def get_descriptor(self, func_match):
        return 'dummy_func'


class SubTest(TestClass):
    def __cleanup(self, remove_dirs=True):
        # remove library
        clean_dir(self.store.lib_dir, remove_dirs)
        # remove build
        clean_dir(self.store.obj_dir, remove_dirs)
        # clean dummy builder
        dist_build = os.path.join(self.store.build_dir, 'build')
        if os.path.exists(dist_build):
            shutil.rmtree(dist_build)
        # clean sources
        clean_dir(self.store.build_dir, remove_dirs)

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
            record, kernels = kgen._process_args()

            # and check
            # 1) that all arguments in all kernels are in the args
            for kernel in kernels:
                arrays, values = partition(kernel.args, lambda x: isinstance(
                    x, ArrayBase))
                local, arrays = partition(
                    arrays, lambda x: x.address_space == scopes.LOCAL)
                assert all(any(kgen._compare_args(a1, a2)
                           for a2 in record.args) for a1 in arrays)
                if local:
                    assert all(any(kgen._compare_args(a1, kgen._temporary_to_arg(a2))
                               for a2 in record.local) for a1 in local)
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
            # create a jacobian kernel generator for this state so that we have
            # a large number of arrays to test
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record, _ = kgen._process_args()

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
            # create a jacobian kernel generator for this state so that we have
            # a large number of arrays to test
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record, _ = kgen._process_args()

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
                        assert local_work_name in unpack
                        assert 'volatile' in unpack
                    else:
                        assert rhs_work_name in unpack
                    # and in offset
                    assert arg.name in offsets

            def __check_local_unpacks(result, args):
                for i, arg in enumerate(args):
                    # get offset
                    offsets = result.pointer_offsets[arg.name][2]
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
            def _get_kernel_gens():
                # two kernels (one for each generator)
                instructions0 = (
                    """
                        {arg} = 1
                    """
                )
                instructions1 = (
                    """
                        {arg} = 2
                    """
                )

                # create mapstore
                domain = arc.creator('domain', arc.kint_type, (10,), 'C',
                                     initializer=np.arange(10, dtype=arc.kint_type))
                mapstore = arc.MapStore(opts, domain, None)
                # create global arg
                arg = arc.creator('arg', np.float64, (arc.problem_size.name, 10),
                                  opts.order)
                # create array / array string
                arg_lp, arg_str = mapstore.apply_maps(arg, 'j', 'i')

                # create kernel infos
                knl0 = knl_info('knl0', instructions0.format(arg=arg_str), mapstore,
                                kernel_data=[arg_lp, arc.work_size])
                knl1 = knl_info('knl1', instructions1.format(arg=arg_str), mapstore,
                                kernel_data=[arg_lp, arc.work_size])
                # create generators
                gen0 = make_kernel_generator(
                     opts, KernelType.dummy, [knl0],
                     type('', (object,), {'jac': ''}),
                     name=knl0.name)
                gen1 = make_kernel_generator(
                     opts, KernelType.dummy, [knl0, knl1],
                     type('', (object,), {'jac': ''}), depends_on=[gen0],
                     name=knl1.name)
                return gen0, gen1

            # create a species rates kernel generator for this state
            kgen = _get_kernel_gens()[1]
            # make kernels
            kgen._make_kernels()

            # process the arguements
            record, _ = kgen._process_args()

            # test that process memory works
            record, mem_limits = kgen._process_memory(record)

            # and generate working buffers
            recordnew, result = kgen._compress_to_working_buffer(record)

            result = kgen._merge_kernels(record, result)

            # get ownership
            owner = kgen._get_kernel_ownership()

            # check we have generated our own kernels
            for kernel in kgen.kernels:
                extra_kernels = '\n'.join(result.extra_kernels)
                assert bool(re.search(
                    r'\b' + kernel.name + r'\b', extra_kernels)) == (
                    owner[kernel.name] == kgen)

            # check that we have the instruction call to _all_ kernels
            all_kernels = rec_kernel(kgen)
            for kernel in all_kernels:
                assert re.search(
                    r'\b' + kernel.name + r'\b', '\n'.join(result.instructions))

    def test_deduplication(self):
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=False,
                                                   do_sparse=False)
        # create some test kernels

        # first, make a shared temporary
        shared = np.arange(10, dtype=arc.kint_type)
        shared = lp.TemporaryVariable(
            'shared', shape=shared.shape, initializer=shared, read_only=True,
            address_space=scopes.GLOBAL)

        # test for issue where inits defined in the top kernel were accidentally
        # removed
        nonshared_top = np.arange(10, dtype=arc.kint_type)
        nonshared_top = lp.TemporaryVariable(
            'nonshared_top', shape=nonshared_top.shape, initializer=nonshared_top,
            read_only=True, address_space=scopes.GLOBAL)

        # and a non-shared temporary
        nonshared = np.arange(10, dtype=arc.kint_type)
        nonshared = lp.TemporaryVariable(
            'nonshared', shape=nonshared.shape, initializer=nonshared,
            read_only=True, address_space=scopes.GLOBAL)

        # and a preamble generator function -> fastpowi_PreambleGen
        pre = [dummy_preamble()]

        # and finally two kernels (one for each generator)
        # kernel 0 contains the non-shared temporary
        instructions0 = (
            """
                {arg} = shared[i] + dummy(nonshared[i], 1)
            """
        )
        instructions1 = (
            """
                {arg} = {arg} + shared[i] + dummy(nonshared_top[i], 1)
            """
        )

        for opts in oploop:
            # create mapstore
            domain = arc.creator('domain', arc.kint_type, (10,), 'C',
                                 initializer=np.arange(10, dtype=arc.kint_type))
            mapstore = arc.MapStore(opts, domain, None)
            # create global arg
            arg = arc.creator('arg', np.float64, (arc.problem_size.name, 10),
                              opts.order)
            # create array / array string
            arg_lp, arg_str = mapstore.apply_maps(arg, 'j', 'i')

            # create kernel infos
            knl0 = knl_info('knl0', instructions0.format(arg=arg_str), mapstore,
                            kernel_data=[arg_lp, shared, nonshared, arc.work_size],
                            preambles=pre)
            knl1 = knl_info('knl1', instructions1.format(arg=arg_str), mapstore,
                            kernel_data=[arg_lp, shared, nonshared_top,
                                         arc.work_size],
                            preambles=pre)
            # create generators
            gen0 = make_kernel_generator(
                 opts, KernelType.dummy, [knl0],
                 type('', (object,), {'jac': ''}),
                 name=knl0.name)
            gen1 = make_kernel_generator(
                 opts, KernelType.dummy, [knl1],
                 type('', (object,), {'jac': ''}), depends_on=[gen0],
                 name=knl1.name)

            def __get_result(gen, record=None):
                # make kernels
                gen._make_kernels()

                if not record:
                    # process the arguements
                    record, _ = gen._process_args()

                    # test that process memory works
                    record, mem_limits = gen._process_memory(record)

                # and generate working buffers
                recordnew, result = gen._compress_to_working_buffer(record)

                result = gen._merge_kernels(record, result)

                return result, record

            result1, record = __get_result(gen1)
            result2, _ = __get_result(gen0)
            results = [result1, result2]

            # and de-duplicate
            results = gen1._deduplicate(record, results)

            # check inits & preambles
            inits = {}
            preambles = []
            for result in results:
                for k, v in six.iteritems(result.inits):
                    assert k not in inits
                    inits[k] = v
                for p in result.preambles:
                    assert p not in preambles
                    preambles.append(p)

            assert 'shared' in inits
            # ensure that the init is defined in the lowest kernel
            k0 = next(x for x in results if x.name == knl0.name)
            assert 'shared' in k0.inits
            assert 'nonshared' in inits
            assert 'nonshared_top' in inits
            assert pre[0].code in preambles
            assert pre[0].code in k0.preambles

    def test_compilation_generator(self):
        # currently separate compiler code only exists for OpenCL
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   langs=['opencl'],
                                                   do_conp=False,
                                                   do_vector=False,
                                                   do_sparse=False)

        for opts in oploop:
            callgen = CallgenResult(source_names=[
                'adistinctivetestname', 'andyetanothertestname'])
            # create a species rates kernel generator for this state
            kgen = get_jacobian_kernel(self.store.reacs, self.store.specs, opts,
                                       conp=oploop.state['conp'])
            with temporary_directory() as tdir:
                comp = kgen._generate_compiling_program(tdir, callgen)

                file = os.path.join(
                    tdir, kgen.name + '_compiler' + file_ext[opts.lang])

                with open(file, 'r') as file:
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

    def __get_call_kernel_generator(self, opts):
        # create some test kernels

        # first, make a (potentially) host constant
        const = np.arange(10, dtype=arc.kint_type)
        const = lp.TemporaryVariable(
            'const', shape=const.shape, initializer=const, read_only=True,
            address_space=scopes.GLOBAL)

        # and finally two kernels (one for each generator)
        # kernel 0 contains the non-shared temporary
        jac_insns = (
            """
                {jac} = {jac} + {spec} {{id=0}}
            """
        )
        spec_insns = (
            """
                {spec} = {spec} + {chem} {{id=1}}
            """
        )
        chem_insns = (
            """
                {chem} = const[i] {{id=2}}
            """
        )

        # create mapstore
        domain = arc.creator('domain', arc.kint_type, (10,), 'C',
                             initializer=np.arange(10, dtype=arc.kint_type))
        mapstore = arc.MapStore(opts, domain, None)
        # create global args
        jac = arc.creator('jac', np.float64,
                          (arc.problem_size.name, 10), opts.order)
        spec = arc.creator('spec', np.float64,
                           (arc.problem_size.name, 10), opts.order)
        chem = arc.creator('chem', np.float64,
                           (arc.problem_size.name, 10), opts.order)
        namestore = type('', (object,),
                         {'jac': jac, 'spec': spec, 'chem': chem})
        # create array / array strings
        jac_lp, jac_str = mapstore.apply_maps(jac, 'j', 'i')
        spec_lp, spec_str = mapstore.apply_maps(spec, 'j', 'i')
        chem_lp, chem_str = mapstore.apply_maps(chem, 'j', 'i')

        # create kernel infos
        jac_info = knl_info(
            'jac_eval', jac_insns.format(jac=jac_str, spec=spec_str),
            mapstore, kernel_data=[jac_lp, spec_lp, arc.work_size],
            silenced_warnings=['write_race(0)'])
        spec_info = knl_info(
            'spec_eval', spec_insns.format(spec=spec_str, chem=chem_str),
            mapstore, kernel_data=[spec_lp, chem_lp, arc.work_size],
            silenced_warnings=['write_race(1)'])
        chem_info = knl_info(
            'chem_eval', chem_insns.format(
                chem=chem_str),
            mapstore, kernel_data=[chem_lp, const, arc.work_size],
            silenced_warnings=['write_race(2)'])

        # create generators
        chem_gen = make_kernel_generator(
             opts, KernelType.chem_utils, [chem_info],
             namestore,
             output_arrays=['chem'])
        spec_gen = make_kernel_generator(
             opts, KernelType.species_rates, [spec_info],
             namestore, depends_on=[chem_gen],
             input_arrays=['chem'], output_arrays=['spec'])
        jac_gen = make_kernel_generator(
             opts, KernelType.jacobian, [jac_info],
             namestore, depends_on=[spec_gen],
             input_arrays=['spec'], output_arrays=['jac'])

        return jac_gen

    def test_call_generator(self):
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=True,
                                                   do_sparse=False)

        for opts in oploop:
            with temporary_directory() as tdir:
                jac_gen = self.__get_call_kernel_generator(opts)

                jac_gen._make_kernels()
                callgen, record, result = jac_gen._generate_wrapping_kernel(tdir)
                callgen = jac_gen._generate_driver_kernel(
                    tdir, record, result, callgen)
                out, _ = jac_gen._generate_calling_program(
                    tdir, 'dummy.bin', callgen, record, for_validation=True)

                # check that 1) it runs
                with open(out, 'r') as file:
                    file_src = file.read()

                # check for local defn's
                for arr in jac_gen.in_arrays + jac_gen.out_arrays:
                    assert 'double* h_{}_local;'.format(arr) in file_src

                # check for operator defn
                assert ('void JacobianKernel::operator()(double* h_jac, '
                        'double* h_spec)') in file_src

                if opts.lang == 'c':
                    assert ('void Kernel::threadset(unsigned int num_threads)'
                            in file_src)
                elif opts.lang == 'opencl':
                    assert re.search(
                        r'build_options = [^\n]+-I{}'.format(tdir), file_src)
                    assert re.search(opts.platform.vendor, file_src)

                # and the validation output
                assert all(x in file_src for x in
                           """// write output to file if supplied
    char* output_files[1] = {"jac.bin"};
    size_t output_sizes[1] = {10 * problem_size * sizeof(double)};
    double* outputs[1] = {h_jac_local};
    for(int i = 0; i < 1; ++i)
    {
        write_data(output_files[i], outputs[i], output_sizes[i]);
    }

    kernel.finalize();""".split())

    def test_call_header_generator(self):
        oploop = OptionLoopWrapper.from_get_oploop(self,
                                                   do_conp=False,
                                                   do_vector=True,
                                                   do_sparse=False)
        for opts in oploop:
            with temporary_directory() as tdir:
                jac_gen = self.__get_call_kernel_generator(opts)
                jac_gen._make_kernels()
                callgen, record, result = jac_gen._generate_wrapping_kernel(tdir)
                callgen = jac_gen._generate_driver_kernel(
                    tdir, record, result, callgen)
                _, callgen = jac_gen._generate_calling_program(
                    tdir, 'dummy.bin', callgen, record, for_validation=True)
                file = jac_gen._generate_calling_header(tdir, callgen)
                with open(file, 'r') as file:
                    file_src = file.read()

                assert 'JacobianKernel();' in file_src
                assert ('JacobianKernel(size_t problem_size, size_t work_size, '
                        'bool do_not_compile=false);') in file_src
                assert 'void operator()(double* h_jac, double* h_spec);'

                headers = ['mechanism', 'error_check', 'timer']
                if can_vectorize_lang[opts.lang]:
                    headers += ['vectorization']
                assert all('#include "{}"'.format(header + header_ext[opts.lang])
                           in file_src for header in headers)

                if opts.lang == 'opencl':
                    # check build options
                    assert 'static const char* build_options;' in file_src
                    assert r'static const char* platform_check;' in file_src
                    assert r'static const unsigned int device_type;' in file_src

                    # check arguments
                    for x in jac_gen.in_arrays + jac_gen.out_arrays:
                        assert re.search(r'cl_mem d_{};'.format(x), file_src)
                    # and work arrays
                    for x in callgen.work_arrays:
                        if not isinstance(x, lp.ValueArg):
                            try:
                                name = x.name
                            except AttributeError:
                                name = x
                            assert re.search(
                                r'cl_mem d_{};'.format(name), file_src)
                else:
                    assert 'CL_LEVEL' not in file_src

                    for x in jac_gen.in_arrays + jac_gen.out_arrays:
                        assert re.search(r'double\* d_{};'.format(x), file_src)
                    # and work arrays
                    for x in callgen.work_arrays:
                        if not isinstance(x, lp.ValueArg):
                            try:
                                name = x.name
                            except AttributeError:
                                name = x
                            assert re.search(
                                r'double\* d_{};'.format(name), file_src)


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


def test_remove_work_array_consts():
    assert kernel_generator._remove_work_array_consts(
        '(double const *__restrict__ rwk)') == '(double *__restrict__ rwk)'
    assert kernel_generator._remove_work_array_consts(
        '(__local volatile double const *__restrict__ lwk)') == (
            '(__local volatile double *__restrict__ lwk)')
    assert kernel_generator._remove_work_array_consts(
        '(int const *__restrict__ iwk)') == '(int *__restrict__ iwk)'
    assert kernel_generator._remove_work_array_consts(
        '(long int const *__restrict__ iwk)') == '(long int *__restrict__ iwk)'


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
