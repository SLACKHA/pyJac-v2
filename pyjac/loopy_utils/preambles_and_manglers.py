from string import Template
import numpy as np


class MangleGen(object):
    """
    A simple interface for Loopy to recognize user functions, custom preambles
    or just functions it doesn't know about
    """
    def __tuple_gen(self, vals):
        if isinstance(vals, tuple):
            return vals
        return (vals,)

    def __init__(self, name, arg_dtypes, result_dtypes):
        self.name = name
        self.arg_dtypes = self.__tuple_gen(arg_dtypes)
        self.result_dtypes = self.__tuple_gen(result_dtypes)

    def __call__(self, kernel, name, arg_dtypes):
        """
        A function that will return a :class:`loopy.kernel.data.CallMangleInfo`
        to interface with the calling :class:`loopy.LoopKernel`
        """
        if name != self.name:
            return None

        from loopy.types import to_loopy_type
        from loopy.kernel.data import CallMangleInfo

        def __compare(d1, d2):
            # compare dtypes ignoring atomic
            return to_loopy_type(d1, for_atomic=True) == \
                to_loopy_type(d2, for_atomic=True)

        # check types
        if len(arg_dtypes) != len(self.arg_dtypes):
            raise Exception('Unexpected number of arguements provided to mangler {},'
                            ' expected {}, got {}'.format(self.name,
                                                          len(self.arg_dtypes),
                                                          len(arg_dtypes)))

        for i, (d1, d2) in enumerate(zip(self.arg_dtypes, arg_dtypes)):
            if not __compare(d1, d2):
                raise Exception('Argument at index {} for mangler {} does not match'
                                'expected dtype.  Expected {}, got {}'.format(
                                    i, self.name, str(d1), str(d2)))

        # get target for creation
        target = arg_dtypes[0].target
        return CallMangleInfo(
            target_name=self.name,
            result_dtypes=tuple(to_loopy_type(x, target=target) for x in
                                self.result_dtypes),
            arg_dtypes=arg_dtypes)


class PreambleGen(object):
    """
    A base class to implement various preambles for OpenCL
    """

    def __tuple_gen(self, vals):
        if isinstance(vals, tuple):
            return vals
        return (vals,)

    def __init__(self, name, code, arg_dtypes, result_dtypes):
        self.func_mangler = MangleGen(name, arg_dtypes, result_dtypes)
        self.name = name
        self.code = code
        self.arg_dtypes = self.__tuple_gen(arg_dtypes)
        self.result_dtypes = self.__tuple_gen(result_dtypes)

    def generate_code(self, preamble_info):
        return self.code

    def get_descriptor(self, func_match):
        raise NotImplementedError

    def get_func_mangler(self):
        return self.func_mangler

    def match(self, func_sig):
        return func_sig.name == self.name

    def __call__(self, preamble_info):
        # find a function matching this name
        func_match = next(
            (x for x in preamble_info.seen_functions
             if self.match(x)), None)
        desc = self.get_descriptor(func_match)
        code = ''
        if func_match is not None:
            from loopy.types import to_loopy_type
            # check types
            if tuple(to_loopy_type(x) for x in self.arg_dtypes) == \
                    func_match.arg_dtypes:
                code = self.generate_code(preamble_info)
        # return code generator
        yield (desc, code)


class fastpowi_PreambleGen(PreambleGen):
    def __init__(self):
        # operators
        self.code = """
   inline double fast_powi(double val, int pow)
   {
        double retval = 1;
        for (int i = 0; i < pow; ++i)
            retval *= val;
        return retval;
   }
            """

        super(fastpowi_PreambleGen, self).__init__(
            'fast_powi', self.code,
            (np.float64, np.int32),
            (np.float64))

    def get_descriptor(self, func_match):
        return 'cust_funcs_fastpowi'


class fastpowf_PreambleGen(PreambleGen):
    def __init__(self):
        # operators
        self.code = """
   inline double fast_powf(double val, double pow)
   {
        double retval = 1;
        for (int i = 0; i < pow; ++i)
            retval *= val;
        if (pow != (int)pow)
        {
            retval *= powf(val, pow - (int) pow);
        }
        return retval;
   }
            """

        super(fastpowf_PreambleGen, self).__init__(
            'fast_powf', self.code,
            (np.float64, np.float64),
            (np.float64))

    def get_descriptor(self, func_match):
        return 'cust_funcs_fastpowf'


class fmax(MangleGen):
    def __init__(self, name='fmax', arg_dtypes=(np.float64, np.float64),
                 result_dtypes=np.float64):
        super(fmax, self).__init__(name, arg_dtypes, result_dtypes)


class jac_indirect_lookup(PreambleGen):
    name = 'jac_indirect'

    def __init__(self, array):
        self.code = Template("""
    int ${name}(int start, int end, int match)
    {
        int result = -1;
        for (int i = start; i < end; ++i)
        {
            if (${array}[i] == match)
                result = i - start;
        }
        return result;
    }
    """).safe_substitute(name=jac_indirect_lookup.name, array=array.name)
        from loopy.kernel.data import temp_var_scope as scopes
        from loopy.kernel.data import TemporaryVariable
        self.array = TemporaryVariable(array.name, shape=array.shape,
                                       dtype=array.dtype,
                                       initializer=array.initializer,
                                       scope=scopes.GLOBAL, read_only=True)

        super(jac_indirect_lookup, self).__init__(
            jac_indirect_lookup.name, self.code,
            (np.int32, np.int32, np.int32), (np.int32))

    def generate_code(self, preamble_info):
        from cgen import Initializer
        from loopy.target.c import generate_array_literal
        codegen_state = preamble_info.codegen_state.copy(
            is_generating_device_code=True)
        kernel = preamble_info.kernel
        ast_builder = codegen_state.ast_builder
        target = kernel.target
        decl_info, = self.array.decl_info(target, index_dtype=kernel.index_dtype)
        decl = ast_builder.wrap_global_constant(
                ast_builder.get_temporary_decl(
                    codegen_state, 1, self.array,
                    decl_info))
        if self.array.initializer is not None:
            decl = Initializer(decl, generate_array_literal(
                codegen_state, self.array, self.array.initializer))
        return '\n'.join([str(decl), self.code])

    def get_descriptor(self, func_match):
        return 'cust_funcs_jac_indirect'
