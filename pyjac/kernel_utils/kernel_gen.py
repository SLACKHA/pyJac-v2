"""
kernel_gen.py - generators used for kernel creation
"""

import shutil
import textwrap
import os
import re
from string import Template
import logging
from collections import defaultdict

import six
from six.moves import reduce
import loopy as lp
from loopy.kernel.data import temp_var_scope as scopes
import pyopencl as cl
import numpy as np
import cgen

from . import file_writers as filew
from .memory_manager import memory_manager, memory_limits, memory_type, guarded_call
from .. import siteconf as site
from .. import utils
from ..loopy_utils import loopy_utils as lp_utils
from ..loopy_utils import preambles_and_manglers as lp_pregen
from ..core.array_creator import problem_size as p_size
from ..core.array_creator import global_ind
from ..core import array_creator as arc

script_dir = os.path.abspath(os.path.dirname(__file__))


class vecwith_fixer(object):

    """
    Simple utility class to force a constant vector width
    even when the loop being vectorized is shorted than the desired width

    clean : :class:`loopy.LoopyKernel`
        The 'clean' version of the kernel, that will be used for
        determination of the gridsize / vecwidth
    vecwidth : int
        The desired vector width
    """

    def __init__(self, clean, vecwidth):
        self.clean = clean
        self.vecwidth = vecwidth

    def __call__(self, insn_ids, ignore_auto=False):
        # fix for variable too small for vectorization
        grid_size, lsize = self.clean.get_grid_sizes_for_insn_ids(
            insn_ids, ignore_auto=ignore_auto)
        lsize = lsize if self.vecwidth is None else \
            self.vecwidth
        return grid_size, (lsize,)


def make_kernel_generator(loopy_opts, *args, **kw_args):
    """
    Factory generator method to return the appropriate
    :class:`kernel_generator` type based on the target language in the
    :param:`loopy_opts`

    Parameters
    ----------
    loopy_opts : :class:`LoopyOptions`
        The specified user options
    *args : tuple
        The other positional args to pass to the :class:`kernel_generator`
    **kw_args : dict
        The keyword args to pass to the :class:`kernel_generator`
    """
    if loopy_opts.lang == 'c':
        if not loopy_opts.auto_diff:
            return c_kernel_generator(loopy_opts, *args, **kw_args)
        if loopy_opts.auto_diff:
            return autodiff_kernel_generator(loopy_opts, *args, **kw_args)
    if loopy_opts.lang == 'opencl':
        return opencl_kernel_generator(loopy_opts, *args, **kw_args)
    if loopy_opts.lang == 'ispc':
        return ispc_kernel_generator(loopy_opts, *args, **kw_args)
    raise NotImplementedError()


class kernel_generator(object):

    """
    The base class for the kernel generators
    """

    def __init__(self, loopy_opts, name, kernels,
                 namestore,
                 external_kernels=[],
                 input_arrays=[],
                 output_arrays=[],
                 test_size=None,
                 auto_diff=False,
                 depends_on=[],
                 array_props={},
                 barriers=[],
                 extra_kernel_data=[],
                 extra_preambles=[],
                 is_validation=False,
                 fake_calls={},
                 mem_limits='',
                 for_testing=False):
        """
        Parameters
        ----------
        loopy_opts : :class:`LoopyOptions`
            The specified user options
        name : str
            The kernel name to use
        kernels : list of :class:`loopy.LoopKernel`
            The kernels / calls to wrap
        namestore: :class:`NameStore`
            The namestore object used in creation of this kernel.
            This is used to pull any extra data (e.g. the Jacobian row/col inds)
            as needed
        input_arrays : list of str
            The names of the input arrays of this kernel
        output_arrays : list of str
            The names of the output arrays of this kernel
        test_size : int
            If specified, the # of conditions to test
        auto_diff : bool
            If true, this will be used for automatic differentiation
        depends_on : list of :class:`kernel_generator`
            If supplied, this kernel depends on the supplied depencies
        array_props : dict
            Mapping of various switches to array names:
                doesnt_need_init
                    * Arrays in this list do not need initialization
                      [defined for host arrays only]
        barriers : list of tuples
            List of global memory barriers needed, (knl1, knl2, barrier_type)
        extra_kernel_data : list of :class:`loopy.ArrayBase`
            Extra kernel arguements to add to this kernel
        extra_preambles: list of :class:`PreambleGen`
            Preambles to add to subkernels
        is_validation: bool [False]
            If true, this kernel generator is being used to validate pyJac
            Hence we need to save our output data to a file
        fake_calls: dict of str -> kernel_generator
            In some cases, e.g. finite differnce jacobians, we need to place a dummy
            call in the kernel that loopy will accept as valid.  Then it needs to
            be substituted with an appropriate call to the kernel generator's kernel
        mem_limits: str ['']
            Path to a .yaml file indicating desired memory limits that control the
            desired maximum amount of global / local / or constant memory that
            the generated pyjac code may allocate.  Useful for testing, or otherwise
            limiting memory usage during runtime. The keys of this file are the
            members of :class:`pyjac.kernel_utils.memory_manager.mem_type`
        for_testing: bool [False]
            If true, this kernel generator will be used for unit testing
        """

        self.compiler = None
        self.loopy_opts = loopy_opts
        self.array_split = arc.array_splitter(loopy_opts)
        self.lang = loopy_opts.lang
        self.mem_limits = mem_limits

        # Used for pinned memory kernels to enable splitting evaluation over multiple
        # kernel calls
        self.arg_name_maps = {p_size: 'per_run'}

        self.mem = memory_manager(self.lang, self.loopy_opts.order,
                                  self.array_split._have_split(),
                                  dev_type=self.loopy_opts.device_type)
        self.name = name
        self.kernels = kernels
        self.namestore = namestore
        self.seperate_kernels = loopy_opts.seperate_kernels
        self.test_size = test_size
        self.auto_diff = auto_diff

        # update the memory manager
        self.mem.add_arrays(in_arrays=input_arrays, out_arrays=output_arrays)

        self.type_map = {}
        from loopy.types import to_loopy_type
        self.type_map[to_loopy_type(np.float64)] = 'double'
        self.type_map[to_loopy_type(np.int32)] = 'int'
        self.type_map[to_loopy_type(np.int64)] = 'long int'

        self.filename = ''
        self.bin_name = ''
        self.header_name = ''
        self.file_prefix = ''

        self.depends_on = depends_on[:]
        self.array_props = array_props.copy()
        self.all_arrays = []
        self.barriers = barriers[:]

        # the base skeleton for sub kernel creation
        self.skeleton = """
        for j
            ${pre}
            for ${var_name}
                ${main}
            end
            ${post}
        end
        """

        # list of inames added to sub kernels
        self.inames = [global_ind]

        # list of iname domains added to subkernels
        self.iname_domains = ['0<={}<{{}}'.format(global_ind)]

        # extra kernel parameters to be added to subkernels
        self.extra_kernel_data = extra_kernel_data[:]

        self.extra_preambles = extra_preambles[:]
        # check for Jacobian type
        if isinstance(namestore.jac, arc.jac_creator):
            # need to add the row / column inds
            self.extra_kernel_data.extend([self.namestore.jac_row_inds([''])[0],
                                           self.namestore.jac_col_inds([''])[0]])

            # and the preamble
            self.extra_preambles.append(lp_pregen.jac_indirect_lookup(
                self.namestore.jac_col_inds if self.loopy_opts.order == 'C'
                else self.namestore.jac_row_inds))

        # calls smuggled past loopy
        self.fake_calls = fake_calls.copy()
        if self.fake_calls:
            # compress into one kernel which we will call separately
            def __set(kgen):
                kgen.seperate_kernels = False
                for x in kgen.depends_on:
                    __set(x)
            __set(self)

        # set kernel attribute
        self.kernel = None
        # set testing
        self.for_testing = for_testing

    def apply_barriers(self, instructions, use_sub_barriers=True):
        """
        A method stud that can be overriden to apply synchonization barriers
        to vectorized code

        Parameters
        ----------

        instructions: list of str
            The instructions for this kernel

        use_sub_barriers: bool [True]
            If true, apply barriers from dependency kernel generators in
            :attr:`depends_on`

        Returns
        -------

        instructions : list of str
            The instructions passed in
        """
        return instructions

    def get_assumptions(self, test_size):
        """
        Returns a list of assumptions on the loop domains
        of generated subkernels

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel

        Returns
        -------

        assumptions : list of str
            List of assumptions to apply to the generated sub kernel
        """

        assumpt_list = ['{0} > 0'.format(test_size)]
        # get vector width
        vec_width = self.loopy_opts.depth if self.loopy_opts.depth \
            else self.loopy_opts.width
        if vec_width is not None:
            assumpt_list.append('{0} mod {1} = 0'.format(
                test_size, vec_width))
        return assumpt_list

    def get_inames(self, test_size):
        """
        Returns the inames and iname_ranges for subkernels created using
        this generator

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel

        Returns
        -------
        inames : list of str
            The string inames to add to created subkernels by default
        iname_domains : list of str
            The iname domains to add to created subkernels by default
        """

        return self.inames, [self.iname_domains[0].format(test_size)]

    def add_depencencies(self, k_gens):
        """
        Adds the supplied :class:`kernel_generator`s to this
        one's dependency list.  Functionally this means that this kernel
        generator will know how to compile and execute functions
        from the dependencies

        Parameters
        ----------
        k_gens : list of :class:`kernel_generator`
            The dependencies to add to this kernel
        """

        self.depends_on.extend(k_gens)

    def _make_kernels(self):
        """
        Turns the supplied kernel infos into loopy kernels,
        and vectorizes them!

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # TODO: need to update loopy to allow pointer args
        # to functions, in the meantime use a Template

        # now create the kernels!
        self.target = lp_utils.get_target(self.lang, self.loopy_opts.device,
                                          self.compiler)
        for i, info in enumerate(self.kernels):
            # if external, or already built
            if isinstance(info, lp.LoopKernel):
                continue
            # create kernel from k_gen.knl_info
            self.kernels[i] = self.make_kernel(info, self.target, self.test_size)
            # apply vectorization
            self.kernels[i] = self.apply_specialization(
                self.loopy_opts,
                info.var_name,
                self.kernels[i],
                vecspec=info.vectorization_specializer,
                can_vectorize=info.can_vectorize)

            # update the kernel args
            self.kernels[i] = self.array_split.split_loopy_arrays(
                self.kernels[i])

            # and add a mangler
            # func_manglers.append(create_function_mangler(kernels[i]))

            # set the editor
            self.kernels[i] = lp_utils.set_editor(self.kernels[i])

        # and finally register functions
        # for func in func_manglers:
        #    knl = lp.register_function_manglers(knl, [func])

        # need to call make_kernels on dependencies
        for x in self.depends_on:
            x._make_kernels()

    def __copy_deps(self, scan_path, out_path, change_extension=True):
        """
        Convenience function to copy the dependencies of this
        :class:`kernel_generator` to our own output path

        Parameters
        ----------

        scan_path : str
            The path the dependencies were written to
        out_path : str
            The path this generator is writing to
        change_ext : bool
            If True, any dependencies that do not end with the proper file
            extension, see :any:`utils.file_ext`

        """
        deps = [x for x in os.listdir(scan_path) if os.path.isfile(
            os.path.join(scan_path, x)) and not x.endswith('.in')]
        for dep in deps:
            dep_dest = dep
            dep_is_header = dep.endswith('.h')
            ext = (utils.file_ext[self.lang] if not dep_is_header
                   else utils.header_ext[self.lang])
            if change_extension and not dep.endswith(ext):
                dep_dest = dep[:dep.rfind('.')] + ext
            shutil.copyfile(os.path.join(scan_path, dep),
                            os.path.join(out_path, dep_dest))

    def generate(self, path, data_order=None, data_filename='data.bin',
                 for_validation=False):
        """
        Generates wrapping kernel, compiling program (if necessary) and
        calling / executing program for this kernel

        Parameters
        ----------
        path : str
            The output path
        data_order : {'C', 'F'}f
            If specified, the ordering of the binary input data
            which may differ from the loopy order
        data_filename : Optional[str]
            If specified, the path to the data file for reading / execution
            via the command line
        for_validation: bool [False]
            If True, this kernel is being generated to validate pyJac, hence we need
            to save output data to a file

        Returns
        -------
        None
        """
        utils.create_dir(path)
        self._make_kernels()
        max_per_run = self._generate_wrapping_kernel(path)
        self._generate_compiling_program(path)
        self._generate_calling_program(path, data_filename, max_per_run,
                                       for_validation=for_validation)
        self._generate_calling_header(path)
        self._generate_common(path)

        # finally, copy any dependencies to the path
        lang_dir = os.path.join(script_dir, self.lang)
        self.__copy_deps(lang_dir, path, change_extension=False)

    def _generate_common(self, path):
        """
        Creates the common files (used by all target languages) for this
        kernel generator

        Parameters
        ----------
        path : str
            The output path for the common files

        Returns
        -------
        None
        """

        common_dir = os.path.join(script_dir, 'common')
        # get the initial condition reader
        with open(os.path.join(common_dir,
                               'read_initial_conditions.c.in'), 'r') as file:
            file_src = Template(file.read())

        with filew.get_file(os.path.join(path, 'read_initial_conditions'
                                         + utils.file_ext[self.lang]),
                            self.lang,
                            use_filter=False) as file:
            file.add_lines(file_src.safe_substitute(
                mechanism='mechanism' + utils.header_ext[self.lang],
                vectorization='vectorization' + utils.header_ext[self.lang]))

        # and any other deps
        self.__copy_deps(common_dir, path)

    def _get_pass(self, argv, include_type=True, is_host=True, postfix=''):
        """
        Simple helper method to get the string for passing an arguement
        to a method (or for the method definition)

        Parameters
        ----------
        argv : :class:`loopy.KernelArgument`
            The arguement to pass
        include_type : bool
            If True, include the C-type in the pass string [Default:True]
        postfix : str
            Optional postfix to append to the variable name [Default:'']
        """
        prefix = 'h_' if is_host else 'd_'
        return '{type}{prefix}{name}'.format(
            type=self.type_map[argv.dtype] + '* ' if include_type else '',
            prefix=prefix,
            name=argv.name + postfix)

    def _generate_calling_header(self, path):
        """
        Creates the header file for this kernel

        Parameters
        ----------
        path : str
            The output path for the header file

        Returns
        -------
        None
        """
        assert self.filename or self.bin_name, ('Cannot generate calling '
                                                'header before wrapping kernel'
                                                ' is generated...')
        with open(os.path.join(script_dir, self.lang,
                               'kernel.h.in'), 'r') as file:
            file_src = Template(file.read())

        self.header_name = os.path.join(path, self.file_prefix + self.name + '_main'
                                        + utils.header_ext[self.lang])
        with filew.get_file(os.path.join(self.header_name), self.lang,
                            use_filter=False) as file:
            file.add_lines(file_src.safe_substitute(
                input_args=', '.join([self._get_pass(next(
                    x for x in self.mem.arrays if x.name == a))
                    for a in self.mem.host_arrays
                    if not any(x.name == a for x in self.mem.host_constants)]),
                knl_name=self.name))

    def _special_kernel_subs(self, file_src):
        """
        Substitutes kernel template parameters that are specific to a
        target languages, to be specialized by subclasses of the
        :class:`kernel_generator`

        Parameters
        ----------
        file_src : Template
            The kernel source template to substitute into

        Returns
        -------
        new_file_src : str
            An updated kernel source string to substitute general template
            parameters into
        """
        return file_src

    def _special_wrapper_subs(self, file_src):
        """
        Substitutes wrapper kernel template parameters that are specific to a
        target languages, to be specialized by subclasses of the
        :class:`kernel_generator`

        Parameters
        ----------
        file_src : Template
            The kernel source template to substitute into

        Returns:
        new_file_src : Template
            An updated kernel source template to substitute general template
            parameters into
        """
        return file_src

    def _special_kernel_fixes(self, extra_kernels, preamble, max_per_run):
        """
        An overrideable method that allows specific languages to "fix" issues
        with the preamble.

        Currently only used by OpenCL to deal with integer overflow issues in
        gid() / lid()

        Parameters
        ----------
        extra_kernels: str
            The extra_kernels to fix
        preamble: str
            The preamble to fix
        max_per_run: int
            The number of conditions allowed per run

        Returns
        -------
        extra_kernels: str
            The fixed instructions
        fixed_preamble: str
            The fixed preamble
        """
        return extra_kernels, preamble

    def _set_sort(self, arr):
        return sorted(set(arr), key=lambda x: arr.index(x))

    def _generate_calling_program(self, path, data_filename, max_per_run,
                                  for_validation=False):
        """
        Needed for all languages, this generates a simple C file that
        reads in data, sets up the kernel call, executes, etc.

        Parameters
        ----------
        path : str
            The output path to write files to
        data_filename : str
            The path to the data file for command line input
        max_per_run: int
            The maximum # of initial conditions that can be evaluated per kernel
            call based on memory limits
        for_validation: bool [False]
            If True, this kernel is being generated to validate pyJac, hence we need
            to save output data to a file

        Returns
        -------
        None
        """

        assert self.filename or self.bin_name, (
            'Cannot generate calling program before wrapping kernel '
            'is generated...')

        # find definitions
        mem_declares = self.mem.get_defns()

        # and input args

        # these are the args in the kernel defn
        knl_args = ', '.join([self._get_pass(
            next(x for x in self.mem.arrays if x.name == a))
            for a in self.mem.host_arrays
            if not any(x.name == a for x in self.mem.host_constants)])
        # these are the args passed to the kernel (exclude type)
        input_args = ', '.join([self._get_pass(
            next(x for x in self.mem.arrays if x.name == a),
            include_type=False) for a in self.mem.host_arrays
            if not any(x.name == a for x in self.mem.host_constants)])
        # these are passed from the main method (exclude type, add _local
        # postfix)
        local_input_args = ', '.join([self._get_pass(
            next(x for x in self.mem.arrays if x.name == a),
            include_type=False,
            postfix='_local') for a in self.mem.host_arrays
            if not any(x.name == a for x in self.mem.host_constants)])
        # create doc strings
        knl_args_doc = []
        knl_args_doc_template = Template(
            """
${name} : ${type}
    ${desc}
""")
        logger = logging.getLogger(__name__)
        for x in [y for y in self.mem.in_arrays if not any(
                z.name == y for z in self.mem.host_constants)]:
            if x == 'phi':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The state vector'))
            elif x == 'P_arr':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The array of pressures'))
            elif x == 'V_arr':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc='The array of volumes'))
            elif x == 'dphi':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc=('The time rate of change of'
                                                  'the state vector, in '
                                                  '{}-order').format(
                        self.loopy_opts.order)))
            elif x == 'jac':
                knl_args_doc.append(knl_args_doc_template.safe_substitute(
                    name=x, type='double*', desc=(
                        'The Jacobian of the time-rate of change of the state vector'
                        ' in {}-order').format(
                        self.loopy_opts.order)))
            else:
                logger.warn('Argument documentation not found for arg {}'.format(x))

        knl_args_doc = '\n'.join(knl_args_doc)
        # memory transfers in
        mem_in = self.mem.get_mem_transfers_in()
        # memory transfers out
        mem_out = self.mem.get_mem_transfers_out()
        # memory allocations
        mem_allocs = self.mem.get_mem_allocs()
        # input allocs
        local_allocs = self.mem.get_mem_allocs(True)
        # read args are those that aren't initalized elsewhere
        read_args = ', '.join(['h_' + x + '_local' for x in self.mem.in_arrays
                               if x in ['phi', 'P_arr', 'V_arr']])
        # memory frees
        mem_frees = self.mem.get_mem_frees()
        # input frees
        local_frees = self.mem.get_mem_frees(True)

        # get template
        with open(os.path.join(script_dir, self.lang,
                               'kernel.c.in'), 'r') as file:
            file_src = file.read()

        # specialize for language
        file_src = self._special_kernel_subs(file_src)

        # get data output
        if for_validation:
            num_outputs = len(self.mem.out_arrays)
            output_paths = ', '.join(['"{}"'.format(x + '.bin')
                                      for x in self.mem.out_arrays])
            outputs = ', '.join(['h_{}_local'.format(x)
                                 for x in self.mem.out_arrays])
            # get lp array map
            out_arrays = [next(x for x in self.mem.arrays if x.name == y)
                          for y in self.mem.out_arrays]
            output_sizes = ', '.join([str(self.mem._get_size(
                x, include_item_size=False)) for x in out_arrays])
        else:
            num_outputs = 0
            output_paths = ""
            outputs = ''
            output_sizes = ''

        with filew.get_file(os.path.join(path, self.name + '_main' + utils.file_ext[
                self.lang]), self.lang, use_filter=False) as file:
            file.add_lines(subs_at_indent(
                file_src,
                mem_declares=mem_declares,
                knl_args=knl_args,
                knl_args_doc=knl_args_doc,
                knl_name=self.name,
                input_args=input_args,
                local_input_args=local_input_args,
                mem_transfers_in=mem_in,
                mem_transfers_out=mem_out,
                mem_allocs=mem_allocs,
                mem_frees=mem_frees,
                read_args=read_args,
                order=self.loopy_opts.order,
                data_filename=data_filename,
                local_allocs=local_allocs,
                local_frees=local_frees,
                max_per_run=max_per_run,
                num_outputs=num_outputs,
                output_paths=output_paths,
                outputs=outputs,
                output_sizes=output_sizes
            ))

    def _generate_compiling_program(self, path):
        """
        Needed for some languages (e.g., OpenCL) this may be overriden in
        subclasses to generate a program that compilers the kernel

        Parameters
        ----------
        path : str
            The output path for the compiling program

        Returns
        -------
        None
        """

        pass

    def __migrate_locals(self, kernel, ldecls):
        """
        Migrates local variables in :param:`ldecls` to the arguements of the
        given :param:`kernel`

        Parameters
        ----------
        kernel: :class:`loopy.LoopKernel`
            The kernel to modify
        ldecls: list of :class:`loopy.TemporaryVariable` or :class:`cgen.CLLocal`
            The local variables to migrate

        Returns
        -------
        mod: :class:`loopy.LoopKernel`
            A modified kernel with the given local variables moved from the
            :attr:`loopy.LoopKernel.temporary_variables` to the kernel's
            :attr:`loopy.LoopKernel.args`

        """
        class TemporaryArg(lp.TemporaryVariable, lp.KernelArgument):
            # TODO: implement this in loopy
            # fix to avoid the is_written check in decl_info
            def decl_info(self, target, is_written, index_dtype,
                          shape_override=None):
                return super(TemporaryArg, self).decl_info(
                    target, index_dtype)

            # and sneak __local's past loopy
            def get_arg_decl(self, ast_builder, name_suffix, shape, dtype,
                             is_written):
                from cgen.opencl import CLLocal
                from loopy.target.opencl import OpenCLCASTBuilder
                if self.scope == scopes.GLOBAL:
                    return CLLocal(
                        super(OpenCLCASTBuilder, ast_builder).
                        get_global_arg_decl(
                            self.name + name_suffix, shape, dtype, is_written))
                else:
                    from loopy import LoopyError
                    raise LoopyError(
                        "unexpected request for argument declaration of "
                        "non-global temporary")

        def __argify(temp):
            if isinstance(temp, lp.TemporaryVariable):
                return TemporaryArg(
                    scopes=scopes.GLOBAL,
                    **{k: v for k, v in six.iteritems(vars(temp))
                       if k in ['name', 'shape', 'dtype']})
            # turn CLLocal or the like back into a temporary variable
            from loopy.target.c import POD
            # find actual decl for dtype and name
            while not isinstance(temp, POD):
                temp = temp.subdecl
            return TemporaryArg(temp.name, scope=scopes.GLOBAL, dtype=temp.dtype,
                                shape=(1,))
        # migrate locals to kernel args
        return kernel.copy(
            args=kernel.args[:] + [__argify(x) for x in ldecls],
            temporary_variables={
                key: val for key, val in six.iteritems(
                    kernel.temporary_variables) if not any(
                    key in str(l) for l in ldecls)})

    def __get_kernel_defn(self, knl=None, passed_locals=[]):
        """
        Returns the kernel definition string for this :class:`kernel_generator`,
        taking into account any migrated local variables

        Note: relies on building steps that occur in
        :func:`_generate_wrapping_kernel` -- will raise an error if called before
        this method

        Parameters
        ----------
        knl: None
            If supplied, this is used instead of the generated kernel
        passed_locals: list of :class:`cgen.CLLocal`
            __local variables declared in the wrapping kernel scope, that must
            be passed into this kernel, as __local defn's in subfunctions
            are not well defined, `function qualifiers in OpenCL <https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/functionQualifiers.html>` # noqa

        Returns
        -------
        defn: str
            The kernel definition
        """

        if self.kernel is None and knl is None:
            raise Exception('Must call _generate_wrapping_kernel first')

        if knl is None:
            knl = self.kernel
        if passed_locals:
            knl = self.__migrate_locals(knl, passed_locals)
        defn_str = lp_utils.get_header(knl)
        return defn_str[:defn_str.index(';')]

    def _get_kernel_call(self, knl=None, passed_locals=[]):
        """
        Returns a function call for the given kernel :param:`knl` to be used
        as an instruction.

        If :param:`knl` is None, returns the kernel call for
        this :class:`kernel_generator`

        Parameters
        ----------
        knl: :class:`loopy.LoopKernel`
            The loopy kernel to generate a call for
        passed_locals: list of :class:`cgen.CLLocal`
            __local variables declared in the wrapping kernel scope, that must
            be passed into this kernel, as __local defn's in subfunctions
            are not well defined, `function qualifiers in OpenCL <https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/functionQualifiers.html>` # noqa

        Returns
        -------
        call: str
            The resulting function call
        """

        # default is the generated kernel
        if knl is None:
            args = self.kernel_data + [x for x in self.extra_kernel_data
                                       if isinstance(x, lp.KernelArgument)]
            if passed_locals:
                # put a dummy object that we can reference the name of in the
                # arguements
                args += [type('', (object,), {'name': l.subdecl.name})
                         for l in passed_locals]
            name = self.name
        else:
            # otherwise used passed kernel
            if passed_locals:
                knl = self.__migrate_locals(knl, passed_locals)
            args = knl.args
            name = knl.name

        args = [x.name for x in args]

        return Template("${name}(${args});\n").substitute(
            name=name,
            args=', '.join(args)
            )

    def _generate_wrapping_kernel(self, path, instruction_store=None,
                                  as_dummy_call=False):
        """
        Generates a wrapper around the various subkernels in this
        :class:`kernel_generator` (rather than working through loopy's fusion)

        Parameters
        ----------
        path : str
            The output path to write files to
        instruction_store: dict [None]
            If supplied, store the generated instructions for this kernel
            in this store to avoid duplicate work
        as_dummy_call: bool [False]
            If True, this is being generated as a dummy call smuggled past loopy
            e.g., for a Finite Difference jacobian call to the species rates kernel
            Hence, we need to add any :attr:`extra_kernel_data` to our kernel defn

        Returns
        -------
        max_per_run: int
            The maximum number of initial conditions that can be executed per
            kernel call
        """

        from loopy.types import AtomicNumpyType, to_loopy_type

        assert all(
            isinstance(x, lp.LoopKernel) for x in self.kernels), (
            'Cannot generate wrapper before calling _make_kernels')

        sub_instructions = {} if instruction_store is None else instruction_store
        if self.depends_on:
            # generate wrappers for dependencies
            for x in self.depends_on:
                x._generate_wrapping_kernel(
                    path, instruction_store=sub_instructions,
                    as_dummy_call=x in self.fake_calls)

        self.file_prefix = ''
        if self.auto_diff:
            self.file_prefix = 'ad_'

        # first, load the wrapper as a template
        with open(os.path.join(
                script_dir,
                self.lang,
                'wrapping_kernel{}.in'.format(utils.file_ext[self.lang])),
                'r') as file:
            file_str = file.read()
            file_src = Template(file_str)

        # Find the list of all arguements needed for this kernel:
        # Scan through all our kernels and compile the args
        kernel_data = []
        defines = [arg for dummy in self.kernels for arg in dummy.args]

        # get read_only variables
        read_only = list(
            set(arg.name for dummy in self.kernels for arg in dummy.args
                if not any(
                    arg.name in d.get_written_variables() for d in self.kernels)
                and not isinstance(arg, lp.ValueArg)))

        # find problem_size
        problem_size = next(x for x in defines if x == p_size)

        # remove other value args
        defines = [x for x in defines if not isinstance(x, lp.ValueArg)]

        # check for dupicates
        nameset = sorted(set(d.name for d in defines))
        for name in nameset:
            same_name = []
            for x in defines:
                if x.name == name and not any(x == y for y in same_name):
                    same_name.append(x)
            if len(same_name) != 1:
                # need to see if differences are resolvable
                atomic = next((x for x in same_name if
                               isinstance(x.dtype, AtomicNumpyType)), None)

                def __raise():
                    raise Exception('Cannot resolve different arguements of '
                                    'same name: {}'.format(', '.join(
                                        str(x) for x in same_name)))

                if atomic is None or len(same_name) > 2:
                    # if we don't have an atomic, or we have multiple different
                    # args of the same name...
                    __raise()

                other = next(x for x in same_name if x != atomic)

                # check that all other properties are the same
                if other != atomic and other.copy(
                        dtype=to_loopy_type(other.dtype, for_atomic=True,
                                            target=self.target)) != atomic:
                    __raise()

                # otherwise, they're the same and the only difference is the
                # the atomic.
                # Next, we try to copy all the other kernels with this arg in it
                # with the atomic arg
                for i, knl in enumerate(self.kernels):
                    if other in knl.args:
                        self.kernels[i] = knl.copy(args=[
                            x if x != other else atomic for x in knl.args])

                same_name.remove(other)

            same_name = same_name.pop()
            kernel_data.append(same_name)

        # check (non-private) temporary variable duplicates
        temps = [arg for dummy in self.kernels
                 for arg in dummy.temporary_variables.values()
                 if isinstance(arg, lp.TemporaryVariable) and
                 arg.scope != lp.temp_var_scope.PRIVATE and
                 arg.scope != lp.auto]
        copy = temps[:]
        temps = []
        for name in sorted(set(x.name for x in copy)):
            same_names = [x for x in copy if x.name == name]
            if len(same_names) > 1:
                if not all(x == same_names[0] for x in same_names[1:]):
                    raise Exception('Cannot resolve different arguements of '
                                    'same name: {}'.format(', '.join(
                                        str(x) for x in same_names)))
            temps.append(same_names[0])

        # add problem size arg to front
        kernel_data.insert(0, problem_size)
        # and save
        self.kernel_data = kernel_data[:]

        # update memory args
        self.mem.add_arrays(kernel_data)

        # generate the kernel definition
        self.vec_width = self.loopy_opts.depth
        if self.vec_width is None:
            self.vec_width = self.loopy_opts.width
        if self.vec_width is None:
            self.vec_width = 0

        # keep track of local / global / constant memory allocations
        mem_types = defaultdict(lambda: list())
        # find if we need to pass constants in via global args
        for i, k in enumerate(self.kernels):
            # before generating, get memory types
            for a in (x for x in k.args if not isinstance(x, lp.ValueArg)):
                if a not in mem_types[memory_type.m_global]:
                    mem_types[memory_type.m_global].append(a)
            for a, v in six.iteritems(k.temporary_variables):
                # check scope to find type
                if v.scope == lp.temp_var_scope.LOCAL:
                    if v not in mem_types[memory_type.m_local]:
                        mem_types[memory_type.m_local].append(v)
                elif v.scope == lp.temp_var_scope.GLOBAL:
                    if v not in mem_types[memory_type.m_constant]:
                        # for opencl < 2.0, a constant global can only be a
                        # __constant
                        mem_types[memory_type.m_constant].append(v)

            # look for jacobian indirect lookup in preambles
            lookup = next((pre for pre in k.preamble_generators if isinstance(
                pre, lp_pregen.jac_indirect_lookup)), None)
            if lookup and lookup.array not in mem_types[memory_type.m_constant]:
                # also need to include the lookup array in consideration
                mem_types[memory_type.m_constant].append(lookup.array)

        # check if we're over our constant memory limit
        mem_limits = memory_limits.get_limits(
            self.loopy_opts, mem_types, string_strides=self.mem.string_strides,
            input_file=self.mem_limits)
        data_size = len(kernel_data)
        read_size = len(read_only)
        if not mem_limits.can_fit():
            # we need to convert our __constant temporary variables to
            # __global kernel args until we can fit
            type_changes = defaultdict(lambda: list())
            # we can't remove the sparse indicies as we can't pass pointers
            # to loopy preambles
            gtemps = [x for x in temps if 'sparse_jac' not in x.name]
            # sort by largest size
            gtemps = sorted(gtemps, key=lambda x: np.prod(x.shape), reverse=True)
            type_changes[memory_type.m_global].append(gtemps[0])
            gtemps = gtemps[1:]
            while not mem_limits.can_fit(with_type_changes=type_changes):
                if not gtemps:
                    logger = logging.getLogger(__name__)
                    logger.exception('Cannot fit kernel {} in memory'.format(
                        self.name))
                    break

                type_changes[memory_type.m_global].append(gtemps[0])
                gtemps = gtemps[1:]

            # once we've converted enough, we need to physically change these
            for x in [v for arrs in type_changes.values() for v in arrs]:
                kernel_data.append(
                    lp.GlobalArg(x.name, dtype=x.dtype, shape=x.shape))
                read_only.append(kernel_data[-1].name)
                self.mem.host_constants.append(x)
                self.kernel_data.append(kernel_data[-1])

            # and update the types
            for v in self.mem.host_constants:
                mem_types[memory_type.m_constant].remove(v)
                mem_types[memory_type.m_global].append(v)

            mem_limits = memory_limits.get_limits(
                self.loopy_opts, mem_types, string_strides=self.mem.string_strides,
                input_file=self.mem_limits)

        # update the memory manager with new args / input arrays
        if len(kernel_data) != data_size:
            self.mem.add_arrays(kernel_data[data_size:],
                                in_arrays=read_only[read_size:])

        # create a dummy kernel to get the defn string
        inames, _ = self.get_inames(0)

        # domains
        domains = []
        for iname in ['i'] + inames:
            domains.append('{{[{iname}]: 0 <= {iname} < {size}}}'.format(
                iname=iname,
                size=self.vec_width))

        # assign to non-readonly to prevent removal
        def _name_assign(arr, use_atomics=True):
            if arr.name not in read_only and not \
                    isinstance(arr, lp.ValueArg):
                return arr.name + '[{ind}] = 0 {atomic}'.format(
                    ind=', '.join(['0'] * len(arr.shape)),
                    atomic='{atomic}'
                           if isinstance(arr.dtype, AtomicNumpyType) and use_atomics
                           else '')
            return ''

        if as_dummy_call:
            # add extra kernel args
            kernel_data.extend([x for x in self.extra_kernel_data
                                if isinstance(x, lp.KernelArgument)])
        knl = lp.make_kernel(domains,
                             '\n'.join(_name_assign(arr)
                                       for arr in kernel_data),
                             kernel_data[:],
                             name=self.name,
                             target=self.target)
        # force vector width
        if self.vec_width != 0:
            ggs = vecwith_fixer(knl.copy(), self.vec_width)
            knl = knl.copy(overridden_get_grid_sizes_for_insn_ids=ggs)

        self.kernel = knl.copy()

        # and finally, generate the kernel code
        preambles = []
        extra_kernels = []
        inits = []
        instructions = []
        local_decls = []

        def _update_for_host_constants(kernel, return_new_args=False):
            """
            Moves temporary variables to global arguments based on the
            host constants for this kernel
            """
            transferred = set([const.name for const in self.mem.host_constants
                               if const.name in kernel.temporary_variables])
            # need to transfer these to arguments
            if transferred:
                # filter temporaries
                new_temps = {t: v for t, v in six.iteritems(
                             kernel.temporary_variables) if t not in transferred}
                # create new args
                new_args = [lp.GlobalArg(
                    t, shape=v.shape, dtype=v.dtype, order=v.order,
                    dim_tags=v.dim_tags)
                    for t, v in six.iteritems(kernel.temporary_variables)
                    if t in transferred]
                if return_new_args:
                    return new_args
                return kernel.copy(
                    args=kernel.args + new_args, temporary_variables=new_temps)
            elif not return_new_args:
                return kernel

        def _get_func_body(cgr, subs={}):
            """
            Returns the function declaration w/o initializers or preambles
            from a :class:`loopy.GeneratedProgram`
            """
            # get body
            if isinstance(cgr.ast, cgen.FunctionBody):
                body = str(cgr.ast)
            else:
                body = str(cgr.ast.contents[-1])

            # apply any substitutions
            for k, v in six.iteritems(subs):
                body = body.replace(k, v)

            # feed through get_code to get any corrections
            return lp_utils.get_code(body, self.loopy_opts)

        def _hoist_local_decls(k):
            # create kernel definition substitutions
            knl_defn = self.__get_kernel_defn(k)
            local_knl_defn = self.__get_kernel_defn(
                k, passed_locals=ldecls)
            subs = {knl_defn: local_knl_defn}
            subs.update(**{str(x): '' for x in ldecls})
            return subs

        if self.fake_calls:
            extra_fake_kernels = {x: [] for x in self.fake_calls}

        from cgen.opencl import CLLocal
        # split into bodies, preambles, etc.
        for i, k, in enumerate(self.kernels):
            if k.name in sub_instructions:
                # avoid regeneration if possible
                pre, init, extra, ldecls, insns = sub_instructions[k.name]
                if self.seperate_kernels:
                    # need to regenerate the call here, in the case that there was
                    # a difference in the host_constants in the sub kernel
                    k = _update_for_host_constants(k)
                    # get call w/ migrated locals
                    insns = self._get_kernel_call(k, passed_locals=ldecls)
                    # and generate code / func body
                    cgr = lp.generate_code_v2(k)
                    assert len(cgr.device_programs) == 1
                    subs = {}
                    if ldecls:
                        subs = _hoist_local_decls(k)
                    extra = _get_func_body(cgr.device_programs[0],
                                           subs)

                if self.fake_calls:
                    # update host constants in subkernel
                    new_args = _update_for_host_constants(k, True)
                    # find out which kernel this belongs to
                    sub = next(x for x in self.depends_on
                               if k.name in [y.name for y in x.kernels])
                    # update sub for host constants
                    if new_args:
                        sub.kernel = sub.kernel.copy(args=sub.kernel.args + [
                            x for x in new_args if x.name not in
                            set([y.name for y in sub.kernel.args])])
                    # and add the instructions to this fake kernel
                    extra_fake_kernels[sub].append(insns)
                    # and clear insns
                    insns = ''

                if insns:
                    instructions.append(insns)
                if pre and pre not in preambles:
                    preambles.extend(pre)
                if init:
                    # filter out any host constants in sub inits
                    init = [x for x in init if not any(n in x for n in read_only)]
                    inits.extend(init)
                if extra and extra not in extra_kernels:
                    # filter out any known extras
                    extra_kernels.append(extra)
                if ldecls:
                    ldecls = [x for x in ldecls if not any(
                                str(x) == str(l) for l in local_decls)]
                    local_decls.extend(ldecls)
                continue

            # check to see if any of our temporary variables are host constants
            if self.seperate_kernels:
                k = _update_for_host_constants(k)

            cgr = lp.generate_code_v2(k)
            # grab preambles
            preamble_list = []
            for _, preamble in cgr.device_preambles:
                if preamble not in preambles:
                    preamble_list.append(preamble)
            # and add to global list
            preambles.extend(preamble_list)

            # now scan device program
            assert len(cgr.device_programs) == 1
            cgr = cgr.device_programs[0]
            init_list = []
            if isinstance(cgr.ast, cgen.Collection):
                # look for preambles
                for item in cgr.ast.contents:
                    # initializers go in the preamble
                    if isinstance(item, cgen.Initializer):
                        def _rec_check_name(decl):
                            if 'name' in vars(decl):
                                return decl.name in read_only
                            elif 'subdecl' in vars(decl):
                                return _rec_check_name(decl.subdecl)
                            return False
                        # check for migrated constant
                        if _rec_check_name(item.vdecl):
                            continue
                        if str(item) not in inits:
                            init_list.append(str(item))

                    # blanklines and bodies can be ignored (as they will be added
                    # below)
                    elif not (isinstance(item, cgen.Line)
                              or isinstance(item, cgen.FunctionBody)):
                        raise NotImplementedError(type(item))
            # and add to inits
            inits.extend(init_list)

            # need to strip out any declaration of a __local variable as these
            # must be in the kernel scope

            # NOTE: this entails hoisting local declarations up to the wrapping
            # kernel for non-separated OpenCL kernels as __local variables in
            # sub-functions are not well defined in the standard:
            # https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/functionQualifiers.html # noqa
            def partition(l, p):
                return reduce(lambda x, y: x[
                    not p(y)].append(y) or x, l, ([], []))
            ldecls, body = partition(cgr.body_ast.contents,
                                     lambda x: isinstance(x, CLLocal))

            subs = {}
            if ldecls and self.seperate_kernels and not instruction_store:
                # need to check for any locals in top level kernels
                subs = _hoist_local_decls(k)

            # have to do a string compare for proper equality testing
            local_decls.extend([x for x in ldecls if not any(
                str(x) == str(y) for y in local_decls)])

            if not self.seperate_kernels:
                cgr.body_ast = cgr.body_ast.__class__(contents=body)
                # leave a comment to distinguish the name
                # and put the body in
                instructions.append('// {name}\n{body}\n'.format(
                    name=k.name, body=str(cgr.body_ast)))
            else:
                # we need to place the call in the instructions and the extra kernels
                # in their own array
                extra_kernels.append(_get_func_body(cgr, subs))
                # additionally, we need to hoist the local declarations to the call
                instructions.append(self._get_kernel_call(
                    k, passed_locals=ldecls))

            if instruction_store is not None:
                assert k.name not in instruction_store
                instruction_store[k.name] = (preamble_list, init_list,
                                             extra_kernels[-1][:] if extra_kernels
                                             else [], ldecls,
                                             instructions[-1][:])

        # fix extra fake kernels
        for gen in self.fake_calls:
            dep = self.fake_calls[gen]
            # update host constants in subkernel
            knl = _update_for_host_constants(gen.kernel)
            # replace call in instructions to call to kernel
            knl_call = gen._get_kernel_call(knl=knl, passed_locals=local_decls)
            instructions = [x.replace(dep, knl_call[:-2]) for x in instructions]
            # and put the kernel in the extra's
            sub_instructions = extra_fake_kernels[gen]
            # apply barriers
            sub_instructions = gen.apply_barriers(sub_instructions)
            code = subs_at_indent("""
${defn}
{
    ${insns}
}""", defn=gen.__get_kernel_defn(knl=knl,
                                 passed_locals=local_decls),
                                 insns='\n'.join(sub_instructions))
            # and place within a single extra kernel
            extra_kernels.append(lp_utils.get_code(code, self.loopy_opts))

        # insert barriers if any
        instructions = self.apply_barriers(instructions,
                                           use_sub_barriers=not self.fake_calls)

        # add local declaration to beginning of instructions
        instructions[0:0] = [str(x) for x in local_decls]

        # join to str
        instructions = '\n'.join(instructions)
        preamble = '\n'.join(textwrap.dedent(x) for x in preambles + inits)

        max_per_run = mem_limits.can_fit(memory_type.m_global)
        # normalize to divide evenly into vec_width
        if self.vec_width != 0:
            max_per_run = np.floor(max_per_run / self.vec_width) * self.vec_width

        extra_kernels, preamble = self._special_kernel_fixes(extra_kernels, preamble,
                                                             max_per_run)

        file_src = self._special_wrapper_subs(file_src)

        self.filename = os.path.join(
            path,
            self.file_prefix + self.name + utils.file_ext[self.lang])
        # create the file
        with filew.get_file(
                self.filename, self.lang, include_own_header=True) as file:
            instructions = _find_indent(file_str, 'body', instructions)
            preamble = _find_indent(file_str, 'preamble', preamble)
            lines = file_src.safe_substitute(
                defines='',
                preamble=preamble,
                func_define=self.__get_kernel_defn(),
                body=instructions,
                extra_kernels='\n'.join(extra_kernels)).split('\n')

            if self.auto_diff:
                lines = [x.replace('double', 'adouble') for x in lines]
            file.add_lines(lines)

        # and the header file (only include self now, as we're using embedded
        # kernels)
        headers = [self.__get_kernel_defn() + utils.line_end[self.lang]]
        with filew.get_header_file(
            os.path.join(path, self.file_prefix + self.name +
                         utils.header_ext[self.lang]), self.lang) as file:

            lines = '\n'.join(headers).split('\n')
            if self.auto_diff:
                file.add_headers('adept.h')
                file.add_lines('using adept::adouble;\n')
                lines = [x.replace('double', 'adouble') for x in lines]
            file.add_lines(lines)

        return int(max_per_run)

    def remove_unused_temporaries(self, knl):
        """
        Convenience method to remove unused temporary variables from created
        :class:`loopy.LoopKernel`'s

        ...with exception of the arrays used in the preambles
        """
        new_args = []

        exp_knl = lp.expand_subst(knl)

        refd_vars = set(knl.all_params())
        for insn in exp_knl.instructions:
            refd_vars.update(insn.dependency_names())

        from loopy.kernel.array import ArrayBase, FixedStrideArrayDimTag
        from loopy.symbolic import get_dependencies
        from itertools import chain

        def tolerant_get_deps(expr, parse=False):
            if expr is None or expr is lp.auto:
                return set()
            if parse and isinstance(expr, tuple):
                from loopy.kernel.array import _pymbolic_parse_if_necessary
                expr = tuple(_pymbolic_parse_if_necessary(x) for x in expr)
            return get_dependencies(expr)

        for ary in chain(knl.args, six.itervalues(knl.temporary_variables)):
            if isinstance(ary, ArrayBase):
                refd_vars.update(
                    tolerant_get_deps(ary.shape)
                    | tolerant_get_deps(ary.offset, parse=True))

                for dim_tag in ary.dim_tags:
                    if isinstance(dim_tag, FixedStrideArrayDimTag):
                        refd_vars.update(
                            tolerant_get_deps(dim_tag.stride))

        for arg in knl.temporary_variables:
            if arg in refd_vars:
                new_args.append(arg)

        return knl.copy(temporary_variables={arg: knl.temporary_variables[arg]
                                             for arg in new_args})

    def make_kernel(self, info, target, test_size):
        """
        Convience method to create loopy kernels from :class:`knl_info`'s

        Parameters
        ----------
        info : :class:`knl_info`
            The rate contstant info to generate the kernel from
        target : :class:`loopy.TargetBase`
            The target to generate code for
        test_size : int/str
            The integer (or symbolic) problem size

        Returns
        -------
        knl : :class:`loopy.LoopKernel`
            The generated loopy kernel
        """

        # and the skeleton kernel
        skeleton = self.skeleton[:]

        # convert instructions into a list for convienence
        instructions = info.instructions
        if isinstance(instructions, str):
            instructions = textwrap.dedent(info.instructions)
            instructions = [x for x in instructions.split('\n') if x.strip()]

        # load inames
        our_inames, our_iname_domains = self.get_inames(test_size)
        inames = [info.var_name] + our_inames

        # add map instructions
        instructions = list(info.mapstore.transform_insns) + instructions

        # look for extra inames, ranges
        iname_range = []

        assumptions = info.assumptions[:]

        # find the start index for 'i'
        iname, iname_domain = info.mapstore.get_iname_domain()

        # add to ranges
        iname_range.append(iname_domain)
        iname_range.extend(our_iname_domains)

        assumptions = []
        if isinstance(test_size, str):
            assumptions.extend(self.get_assumptions(test_size))

        for iname, irange in info.extra_inames:
            inames.append(iname)
            iname_range.append(irange)

        # construct the kernel args
        pre_instructions = info.pre_instructions[:]
        post_instructions = info.post_instructions[:]

        def subs_preprocess(key, value):
            # find the instance of ${key} in kernel_str
            result = _find_indent(skeleton, key, value)
            return Template(result).safe_substitute(var_name=info.var_name)

        kernel_str = Template(skeleton).safe_substitute(
            var_name=info.var_name,
            pre=subs_preprocess('${pre}', '\n'.join(pre_instructions)),
            post=subs_preprocess('${post}', '\n'.join(post_instructions)),
            main=subs_preprocess('${main}', '\n'.join(instructions)))

        # finally do extra subs
        if info.extra_subs:
            kernel_str = Template(kernel_str).safe_substitute(
                **info.extra_subs)

        iname_arr = []
        # generate iname strings
        for iname, irange in zip(*(inames, iname_range)):
            iname_arr.append(Template(
                '{[${iname}]:${irange}}').safe_substitute(
                iname=iname,
                irange=irange
            ))

        # get extra mapping data
        extra_kernel_data = [domain(node.iname)[0] for domain, node in
                             six.iteritems(info.mapstore.domain_to_nodes)
                             if not node.is_leaf()]

        extra_kernel_data += self.extra_kernel_data[:]

        # check for duplicate kernel data (e.g. multiple phi arguements)
        kernel_data = []
        for k in info.kernel_data + extra_kernel_data:
            if k not in kernel_data:
                kernel_data.append(k)

        # make the kernel
        knl = lp.make_kernel(iname_arr,
                             kernel_str,
                             kernel_data=kernel_data,
                             name=info.name,
                             target=target,
                             assumptions=' and '.join(assumptions),
                             default_offset=0,
                             **info.kwargs
                             )
        # fix parameters
        if info.parameters:
            knl = lp.fix_parameters(knl, **info.parameters)
        # prioritize and return
        knl = lp.prioritize_loops(knl, [y for x in inames
                                        for y in x.split(',')])
        # check manglers
        if info.manglers:
            knl = lp.register_function_manglers(knl, info.manglers)

        preambles = info.preambles + self.extra_preambles[:]
        # check preambles
        if preambles:
            # register custom preamble functions
            knl = lp.register_preamble_generators(knl, preambles)
            # also register their function manglers
            knl = lp.register_function_manglers(knl, [
                p.get_func_mangler() for p in preambles])

        return self.remove_unused_temporaries(knl)

    def apply_specialization(self, loopy_opts, inner_ind, knl, vecspec=None,
                             can_vectorize=True):
        """
        Applies wide / deep vectorization and/or ILP loop unrolling
        to a loopy kernel

        Parameters
        ----------
        loopy_opts : :class:`loopy_options` object
            A object containing all the loopy options to execute
        inner_ind : str
            The inner loop index variable
        knl : :class:`loopy.LoopKernel`
            The kernel to transform
        vecspec : :function:
            An optional specialization function that is applied after
            vectorization to fix hanging loopy issues
        can_vectorize : bool
            If False, cannot be vectorized in the normal manner, hence
            vecspec must be used to vectorize.

        Returns
        -------
        knl : :class:`loopy.LoopKernel`
            The transformed kernel
        """

        # before doing anything, find vec width
        # and split variable
        vec_width = None
        to_split = None
        i_tag = inner_ind
        j_tag = global_ind
        depth = loopy_opts.depth
        width = loopy_opts.width
        if depth:
            to_split = inner_ind
            vec_width = depth
            i_tag += '_outer'
        elif width:
            to_split = global_ind
            vec_width = width
            j_tag += '_outer'
        if not can_vectorize:
            assert vecspec is not None, ('Cannot vectorize a non-vectorizable '
                                         'kernel {} without a specialized '
                                         'vectorization function'.format(
                                             knl.name))

        # if we're splitting
        # apply specified optimizations
        if to_split and can_vectorize:
            # and assign the l0 axis to the correct variable
            knl = lp.split_iname(knl, to_split, vec_width, inner_tag='l.0')

        if utils.can_vectorize_lang[loopy_opts.lang]:
            # tag 'global_ind' as g0, use simple parallelism
            knl = lp.tag_inames(knl, [(j_tag, 'g.0')])

        # if we have a specialization
        if vecspec:
            knl = vecspec(knl)

        if vec_width is not None:
            # finally apply the vector width fix above
            ggs = vecwith_fixer(knl.copy(), vec_width)
            knl = knl.copy(overridden_get_grid_sizes_for_insn_ids=ggs)

        # now do unr / ilp
        if loopy_opts.unr is not None:
            knl = lp.split_iname(knl, i_tag, loopy_opts.unr, inner_tag='unr')
        elif loopy_opts.ilp:
            knl = lp.tag_inames(knl, [(i_tag, 'ilp')])

        return knl


class c_kernel_generator(kernel_generator):

    """
    A C-kernel generator that handles OpenMP parallelization
    """

    def __init__(self, *args, **kw_args):

        super(c_kernel_generator, self).__init__(*args, **kw_args)

        self.extern_defn_template = Template(
            'extern ${type}* ${name}' + utils.line_end[self.lang])

        if not self.for_testing:
            # add 'global_ind' to the list of extra kernel data to be added to
            # subkernels
            self.extra_kernel_data.append(lp.ValueArg(global_ind, dtype=np.int32))
            # clear list of inames added to sub kernels, as the OpenMP loop over
            # the states is implemented in the wrapping kernel
            self.inames = []
            # clear list of inames domains added to sub kernels, as the OpenMP loop
            # over the states is implemented in the wrapping kernel
            self.iname_domains = []
            # and modify the skeleton to remove the outer loop
            self.skeleton = """
        ${pre}
        for ${var_name}
            ${main}
        end
        ${post}
        """

    def get_inames(self, test_size):
        """
        Returns the inames and iname_ranges for subkernels created using
        this generator.

        This is an override of the base :func:`get_inames` that decides which form
        of the inames to return based on :attr:`for_testing`.  If False, the
        complete iname set will be returned.  If True, the outer loop iname
        will be removed from the inames / domain

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel

        Returns
        -------
        inames : list of str
            The string inames to add to created subkernels by default
        iname_domains : list of str
            The iname domains to add to created subkernels by default
        """

        if self.for_testing:
            return super(c_kernel_generator, self).get_inames(test_size)

        return self.inames, self.iname_domains

    def get_assumptions(self, test_size):
        """
        Returns a list of assumptions on the loop domains
        of generated subkernels

        For the C-kernels, the problem_size is abstracted out into the wrapper
        kernel's OpenMP loop.

        Additionally, there is no concept of a "vector width", hence
        we return an empty assumption set

        Parameters
        ----------
        test_size : int or str
            In testing, this should be the integer size of the test data
            For production, this should the 'test_size' (or the corresponding)
            for the variable test size passed to the kernel

        Returns
        -------

        assumptions : list of str
            List of assumptions to apply to the generated sub kernel
        """

        return []

    def _special_kernel_subs(self, file_src):
        """
        An override of the :method:`kernel_generator._special_wrapping_subs`
        that implements C-specific wrapping kernel arguement passing

        Parameters
        ----------
        file_src : Template
            The kernel source template to substitute into

        Returns
        -------
        new_file_src : str
            An updated kernel source string to substitute general template
            parameters into
        """

        # and input args

        # these are the args in the kernel defn
        full_kernel_args = ', '.join(self._set_sort(
            [self._get_pass(a, include_type=False, is_host=False)
             for a in self.mem.arrays]))

        return Template(file_src).safe_substitute(
            full_kernel_args=full_kernel_args)


class autodiff_kernel_generator(c_kernel_generator):

    """
    A C-Kernel generator specifically designed to work with the
    autodifferentiation scheme.  Handles adding jacobian, etc.
    """

    def __init__(self, *args, **kw_args):

        # no matter the 'testing' status, the autodiff always needs the outer loop
        # migrated out
        kw_args['for_testing'] = False
        super(autodiff_kernel_generator, self).__init__(*args, **kw_args)

        from ..loopy_utils.loopy_utils import AdeptCompiler
        self.compiler = AdeptCompiler()

    def add_jacobian(self, jacobian):
        """
        Adds the jacobian object to the extra kernel data for inclusion in
        generation (to be utilized during the edit / AD process)

        Parameters
        ----------

        jacobian : :class:`loopy.GlobalArg`
            The loopy arguement to add to the method signature

        Returns
        -------
        None
        """

        self.extra_kernel_data.append(jacobian)


class ispc_kernel_generator(kernel_generator):

    def __init__(self, *args, **kw_args):
        super(ispc_kernel_generator, self).__init__(*args, **kw_args)

    # TODO: fill in


class opencl_kernel_generator(kernel_generator):

    """
    An opencl specific kernel generator
    """

    def __init__(self, *args, **kw_args):
        super(opencl_kernel_generator, self).__init__(*args, **kw_args)

        # opencl specific items
        self.set_knl_arg_array_template = Template(
            guarded_call(self.lang, 'clSetKernelArg(kernel, ${arg_index}, '
                         '${arg_size}, ${arg_value})'))
        self.set_knl_arg_value_template = Template(
            guarded_call(self.lang, 'clSetKernelArg(kernel, ${arg_index}, '
                         '${arg_size}, ${arg_value})'))
        self.barrier_templates = {
            'global': 'barrier(CLK_GLOBAL_MEM_FENCE)',
            'local': 'barrier(CLK_LOCAL_MEM_FENCE)'
        }

        # add atomic types to typemap
        from loopy.types import to_loopy_type
        # these don't need to be volatile, as they are on the host side
        self.type_map[to_loopy_type(np.float64, for_atomic=True)] = 'double'
        self.type_map[to_loopy_type(np.int32, for_atomic=True)] = 'int'
        self.type_map[to_loopy_type(np.int64, for_atomic=True)] = 'long int'

    def _special_kernel_fixes(self, extra_kernels, preamble, max_per_run):
        """
        Deal with integer overflow issues in gid() / lid()

        Parameters
        ----------
        instructions: str
            The instructions to fix
        preamble: str
            The preamble to fix
        max_per_run: int
            The number of conditions allowed per run

        Returns
        -------
        fixed_instructions: str
            The fixed instructions
        fixed_preamble: str
            The fixed preamble
        """

        # the goal is to automatically promote the gid / lid calls in loopy
        # to long's if we can possibly go out of bounds.

        # Currently this only happens for the wide vectorized full Jacobian
        # for isopentanol, but it will likely happen for any large enough mechanism
        # (or alternatively, a large enough # of initial conditions)

        # The strategy is to check the maximum stride size, and see if that * the
        # maximum run size is > int32 max -- if so, trigger the promotion

        # A better strategy would be to allow the user to specify the desired
        # behaviour here (e.g., limiting the maximum per run size)

        # find maximum size of device arrays (that are allocated per-run)
        p_var = p_size.name
        # filter arrays to those depending on problem size
        arrays = [a for a in self.mem.arrays if any(
            p_var in str(x) for x in a.shape) and len(a.shape) >= 2]
        # next find maximum stride
        stride_ind = 0 if self.loopy_opts.order == 'C' else -1
        strides = [(a.dim_tags[stride_ind].stride, a.shape[stride_ind])
                   for a in arrays]

        def floatify(val):
            if not (isinstance(val, float) or isinstance(val, int)):
                ss = next((s for s in self.mem.string_strides if s.search(
                    str(val))), None)
                assert ss is not None, 'malformed strides'
                from pymbolic import parse
                val = parse(str(val).replace(p_var, str(max_per_run)))
                assert isinstance(val, float) or isinstance(val, int)
            return val
        # next convert problem_size -> max per run
        strides = [floatify(x[0]) * floatify(x[1]) for x in strides]
        # test for integer overflow
        if max(strides) >= np.iinfo(np.int32).max:
            logger = logging.getLogger(__name__)
            logger.warn('Promoting gid/lid type to long int to avoid integer '
                        'overflow')
            preamble = re.sub(r'#define (\w{3})\(N\) \(\(int\) ([\w_]+)\(N\)\)',
                              r'#define \1(N) ((long) \2(N))',
                              preamble)

            # finally, if F-ordered we need to change the problem-size from an int
            # to a long
            if self.loopy_opts.order == 'F':
                p_var = p_size.copy(dtype=np.int64)
                self.kernel = self.kernel.copy(args=[
                    a if a != p_size else p_var for a in self.kernel.args])
                # and replace in kernel data
                self.kernel_data = [a if a != p_size else p_var
                                    for a in self.kernel_data]
                extra_kernels = [re.sub(r'int const {}'.format(p_var.name),
                                        r'long const {}'.format(p_var.name),
                                        knl) for knl in extra_kernels]

        return extra_kernels, preamble

    def _special_kernel_subs(self, file_src):
        """
        An override of the :method:`kernel_generator._special_kernel_subs`
        that implements OpenCL specific kernel substitutions

        Parameters
        ----------
        file_src : Template
            The kernel source template to substitute into

        Returns
        -------
        new_file_src : str
            An updated kernel source string to substitute general template
            parameters into
        """

        # open cl specific
        # vec width
        vec_width = self.vec_width
        if not vec_width:
            # set to default
            vec_width = 1
        # platform
        platform_str = self.loopy_opts.platform.get_info(
            cl.platform_info.VENDOR)
        # build options
        build_options = self.build_options
        # kernel arg setting
        kernel_arg_set = self.get_kernel_arg_setting()
        # kernel list
        kernel_paths = [self.bin_name]
        kernel_paths = ', '.join('"{}"'.format(x)
                                 for x in kernel_paths if x.strip())

        # find maximum size of device arrays (that are allocated per-run)
        p_var = p_size.name
        # filter arrays to those depending on problem size
        arrays = [a for a in self.mem.arrays if any(
            p_var in str(x) for x in a.shape)]
        # next convert to size
        arrays = [np.prod(np.fromstring(
            self.mem._get_size(a, subs_n='1'), dtype=np.int32, sep=' * '))
            for a in arrays]
        # and get max size
        max_size = str(max(arrays)) + ' * {}'.format(
            self.arg_name_maps[p_size])

        # find converted constant variables -> global args
        host_constants = self.mem.get_host_constants()
        host_constants_transfers = self.mem.get_host_constants_in()

        # get host memory syncs if necessary
        mem_strat = self.mem.get_mem_strategy()

        return subs_at_indent(file_src,
                              vec_width=vec_width,
                              platform_str=platform_str,
                              build_options=build_options,
                              kernel_arg_set=kernel_arg_set,
                              kernel_paths=kernel_paths,
                              device_type=str(self.loopy_opts.device_type),
                              num_source=1,  # only 1 program / binary is built
                              CL_LEVEL=int(float(self._get_cl_level()) * 100),  # noqa -- CL standard level
                              max_size=max_size,  # max size for CL1.1 mem init
                              host_constants=host_constants,
                              host_constants_transfers=host_constants_transfers,
                              MEM_STRATEGY=mem_strat
                              )

    def get_kernel_arg_setting(self):
        """
        Needed for OpenCL, this generates the code that sets the kernel args

        Parameters
        ----------
        None

        Returns
        -------
        knl_arg_set_str : str
            The code that sets opencl kernel args
        """

        kernel_arg_sets = []
        for i, arg in enumerate(self.kernel_data):
            if not isinstance(arg, lp.ValueArg):
                kernel_arg_sets.append(
                    self.set_knl_arg_array_template.safe_substitute(
                        arg_index=i,
                        arg_size='sizeof({})'.format('d_' + arg.name),
                        arg_value='&d_' + arg.name)
                )
            else:
                # workaround for integer overflow of cl_uint
                # TODO: need to put in detection for integer overlflow here
                # or at least limits for maximum size of kernel before we switch
                # over to a 64bit integer for index type
                name = arg.name if arg not in self.arg_name_maps else \
                    self.arg_name_maps[arg]
                arg_set = self.set_knl_arg_value_template.safe_substitute(
                        arg_index=i,
                        arg_size='sizeof({})'.format(self.type_map[arg.dtype]),
                        arg_value='&{}'.format(name))
                kernel_arg_sets.append(arg_set)

        return '\n'.join(kernel_arg_sets)

    def _get_cl_level(self):
        """
        Searches the supplied platform for a OpenCL level.  If not found,
        uses the level from the site config

        Parameters
        ----------
        None

        Returns
        -------
        cl_level: str
            The stringified OpenCL standard level
        """

        # try get the platform's CL level
        try:
            device_level = self.loopy_opts.device.opencl_c_version.split()
            for d in device_level:
                try:
                    float(d)
                    return d
                    break
                except:
                    pass
        except:
            # default to the site level
            return site.CL_VERSION

    def _generate_compiling_program(self, path):
        """
        Needed for OpenCL, this generates a simple C file that
        compiles and stores the binary OpenCL kernel generated w/ the wrapper

        Parameters
        ----------
        path : str
            The output path to write files to

        Returns
        -------
        None
        """

        assert self.filename, (
            'Cannot generate compiler before wrapping kernel is generated...')
        if self.depends_on:
            assert [x.filename for x in self.depends_on], (
                'Cannot generate compiler before wrapping kernel '
                'for dependencies are generated...')

        self.build_options = ''
        if self.lang == 'opencl':
            with open(os.path.join(script_dir, self.lang,
                                   'opencl_kernel_compiler.c.in'),
                      'r') as file:
                file_str = file.read()
                file_src = Template(file_str)

            # get the platform from the options
            platform_str = self.loopy_opts.platform.get_info(
                cl.platform_info.VENDOR)

            cl_std = self._get_cl_level()

            # for the build options, we turn to the siteconf
            self.build_options = ['-I' + x for x in site.CL_INC_DIR + [path]]
            self.build_options.extend(site.CL_FLAGS)
            self.build_options.append('-cl-std=CL{}'.format(cl_std))
            self.build_options = ' '.join(self.build_options)

            file_list = [self.filename]
            file_list = ', '.join('"{}"'.format(x) for x in file_list)

            self.bin_name = self.filename[:self.filename.index(
                utils.file_ext[self.lang])] + '.bin'

            with filew.get_file(os.path.join(path, self.name + '_compiler'
                                             + utils.file_ext[self.lang]),
                                self.lang, use_filter=False) as file:
                file.add_lines(file_src.safe_substitute(
                    filenames=file_list,
                    outname=self.bin_name,
                    platform=platform_str,
                    build_options=self.build_options,
                    # compiler expects all source strings
                    num_source=1
                ))

    def apply_barriers(self, instructions, use_sub_barriers=True):
        """
        An override of :method:`kernel_generator.apply_barriers` that
        applies synchronization barriers to OpenCL kernels

        Parameters
        ----------

        instructions: list of str
            The instructions for this kernel

        use_sub_barriers: bool [True]
            If true, apply barriers from dependency kernel generators in
            :attr:`depends_on`

        Returns
        -------

        synchronized_instructions : list of str
            The instruction list with the barriers inserted
        """

        barriers = self.barriers[:]
        # include barriers from the sub-kernels
        if use_sub_barriers:
            for dep in self.depends_on:
                barriers = dep.barriers + barriers
        instructions = list(enumerate(instructions))
        for barrier in barriers:
            # find insert index (the second barrier ind)
            index = next(ind for ind, inst in enumerate(instructions)
                         if inst[0] == barrier[1])
            # check that we're inserting between the required barriers
            assert barrier[0] == instructions[index - 1][0]
            # and insert
            instructions.insert(index, (-1, self.barrier_templates[barrier[2]]
                                        + utils.line_end[self.lang]))
        # and get rid of indicies
        instructions = [inst[1] for inst in instructions]
        return instructions


class knl_info(object):

    """
    A composite class that contains the various parameters, etc.
    needed to create a simple kernel

    name : str
        The kernel name
    instructions : str or list of str
        The kernel instructions
    mapstore : :class:`array_creator.MapStore`
        The MapStore object containing map domains, indicies, etc.
    pre_instructions : list of str
        The instructions to execute before the inner loop
    post_instructions : list of str
        The instructions to execute after end of inner loop but before end
        of outer loop
    var_name : str
        The inner loop variable
    kernel_data : list of :class:`loopy.ArrayBase`
        The arguements / temporary variables for this kernel
    extra_inames : list of tuple
        A list of (iname, domain) tuples the form the extra loops in this kernel
    assumptions : list of str
        Assumptions to pass to the loopy kernel
    parameters : dict
        Dictionary of parameter values to fix in the loopy kernel
    extra subs : dict
        Dictionary of extra string substitutions to make in kernel generation
    can_vectorize : bool
        If False, the vectorization specializer must be used to vectorize this kernel
    vectorization_specializer : function
        If specified, use this specialization function to fix problems that would
        arise in vectorization
    preambles : :class:`preamble.PreambleGen`
        A list of preamble generators to insert code into loopy / opencl
    **kwargs: dict
        Any other keyword args to pass to :func:`loopy.make_kernel`
    """

    def __init__(self, name, instructions, mapstore, pre_instructions=[],
                 post_instructions=[],
                 var_name='i', kernel_data=None,
                 extra_inames=[],
                 assumptions=[], parameters={},
                 extra_subs={},
                 vectorization_specializer=None,
                 can_vectorize=True,
                 manglers=[],
                 preambles=[],
                 **kwargs):

        def __listify(arr):
            if isinstance(arr, str):
                return [arr]
            return arr
        self.name = name
        self.instructions = instructions
        self.mapstore = mapstore
        self.pre_instructions = __listify(pre_instructions)[:]
        self.post_instructions = __listify(post_instructions)[:]
        self.var_name = var_name
        if isinstance(kernel_data, set):
            kernel_data = list(kernel_data)
        self.kernel_data = kernel_data[:]
        self.extra_inames = extra_inames[:]
        self.assumptions = assumptions[:]
        self.parameters = parameters.copy()
        self.extra_subs = extra_subs
        self.can_vectorize = can_vectorize
        self.vectorization_specializer = vectorization_specializer
        self.manglers = manglers[:]
        self.preambles = preambles[:]
        self.kwargs = kwargs.copy()


def create_function_mangler(kernel, return_dtypes=()):
    """
    Returns a function mangler to interface loopy kernels with function calls
    to other kernels (e.g. falloff rates from the rate kernel, etc.)

    Parameters
    ----------
    kernel : :class:`loopy.LoopKernel`
        The kernel to create an interface for
    return_dtypes : list :class:`numpy.dtype` returned from the kernel, optional
        Most likely an empty list
    Returns
    -------
    func : :method:`MangleGen`.__call__
        A function that will return a :class:`loopy.kernel.data.CallMangleInfo` to
        interface with the calling :class:`loopy.LoopKernel`
    """
    from ..loopy_utils.preambles_and_manglers import MangleGen

    dtypes = []
    for arg in kernel.args:
        if not isinstance(arg, lp.TemporaryVariable):
            dtypes.append(arg.dtype)
    mg = MangleGen(kernel.name, tuple(dtypes), return_dtypes)
    return mg.__call__


def _find_indent(template_str, key, value):
    """
    Finds and returns a formatted value containing the appropriate
    whitespace to put 'value' in place of 'key' for template_str

    Parameters
    ----------
    template_str : str
        The string to sub into
    key : str
        The key in the template string
    value : str
        The string to format

    Returns
    -------
    formatted_value : str
        The properly indented value
    """

    # find the instance of ${key} in kernel_str
    whitespace = None
    for i, line in enumerate(template_str.split('\n')):
        if key in line:
            # get whitespace
            whitespace = re.match(r'\s*', line).group()
            break
    result = [line if i == 0 else whitespace + line for i, line in
              enumerate(textwrap.dedent(value).splitlines())]
    return '\n'.join(result)


def subs_at_indent(template_str, **kw_args):
    """
    Substitutes keys of :params:`kwargs` for values in :param:`template_str`
    ensuring that the indentation of the value is the same as that of the key
    for all lines present in the value

    Parameters
    ----------
    template_str : str
        The string to sub into
    kwargs: dict
        The dictionary of keys -> values to substituted into the template
    Returns
    -------
    formatted_value : str
        The formatted string
    """

    return Template(template_str).safe_substitute(
        **{key: _find_indent(template_str, '${{{key}}}'.format(key=key),
                             value if isinstance(value, str) else str(value))
            for key, value in six.iteritems(kw_args)})
