from string import Template
import numpy as np

from pyjac import utils as utils


class MangleGen(object):
    """
    A simple interface for Loopy to recognize user functions, custom preambles
    or just functions it doesn't know about
    """
    def __tuple_gen(self, vals):
        if isinstance(vals, tuple):
            return vals
        return (vals,)

    def __init__(self, name, arg_dtypes, result_dtypes, raise_on_fail=True):
        self.name = name
        self.arg_dtypes = self.__tuple_gen(arg_dtypes)
        self.result_dtypes = self.__tuple_gen(result_dtypes)
        self.raise_on_fail = raise_on_fail

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
            raise Exception('Unexpected number of arguments provided to mangler {},'
                            ' expected {}, got {}'.format(self.name,
                                                          len(self.arg_dtypes),
                                                          len(arg_dtypes)))

        for i, (d1, d2) in enumerate(zip(self.arg_dtypes, arg_dtypes)):
            if not __compare(d1, d2) and self.raise_on_fail:
                raise Exception('Argument at index {} for mangler {} does not match'
                                'expected dtype.  Expected {}, got {}'.format(
                                    i, self.name, str(d1), str(d2)))

        # get target for creation
        target = kernel.target
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
    def __init__(self, integer_dtype=np.int32):
        int_str = 'int' if integer_dtype == np.int32 else 'long'
        # operators
        self.code = Template("""
   inline double fast_powi(double val, ${int_str} pow)
   {
        // get sign
        double retval = (val > 0) - (val < 0);
        pow = abs(pow);
        switch(pow)
        {
            case 0:
                return 1;
            case 1:
                return retval * val;
            case 2:
                return retval * val * val;
            case 3:
                return retval * val * val * val;
            case 4:
                return retval * val * val * val * val;
            case 5:
                return retval * val * val * val * val * val;
            default:
                for (${int_str} i = 0; i < pow; ++i)
                {
                    retval *= val;
                }
                return retval;
        }
   }
            """).substitute(int_str=int_str)

        super(fastpowi_PreambleGen, self).__init__(
            'fast_powi', self.code,
            (np.float64, integer_dtype),
            (np.float64))

    def get_descriptor(self, func_match):
        return 'cust_funcs_fastpowi'


def power_function_preambles(loopy_opts, power_function):
    """
    Parameters
    ----------
    loopy_opts: :class:`loopy_options`
        unused, included for consistent interface
    power_function: str
        The power function used
    Returns
    -------
    gen: list of :class:`PreambleGen`
        power function preambles for  a given :param:`power_function`.
    """

    if power_function == 'fast_powi':
        return [fastpowi_PreambleGen()]
    return []


class pown(MangleGen):
    # turn off raise_on_fail, as multiple versions of this might be added
    def __init__(self, name='pown', arg_dtypes=(np.float64, np.int32),
                 result_dtypes=np.float64, raise_on_fail=False):
        super(pown, self).__init__(name, arg_dtypes, result_dtypes,
                                   raise_on_fail=raise_on_fail)


class powf(MangleGen):
    def __init__(self, name='pow', arg_dtypes=(np.float64, np.float64),
                 result_dtypes=np.float64, raise_on_fail=False):
        super(powf, self).__init__(name, arg_dtypes, result_dtypes,
                                   raise_on_fail=raise_on_fail)


class powr(MangleGen):
    def __init__(self, name='powr', arg_dtypes=(np.float64, np.float64),
                 result_dtypes=np.float64, raise_on_fail=False):
        super(powr, self).__init__(name, arg_dtypes, result_dtypes,
                                   raise_on_fail=raise_on_fail)


def power_function_manglers(loopy_opts, power_functions):
    """
    Parameters
    ----------
    loopy_opts: :class:`loopy_options`
        The code-generation options, used for determining vector-width / SIMD
    power_function: str or list of string
        The power function used
    Returns
    -------
    manglers: list of :class:`MangleGen`
        power function manglers for  a given :param:`power_function`.
    """

    def __manglers(power_function):
        if loopy_opts.lang == 'opencl' and 'pow' in power_function:
            # opencl only
            # create manglers
            manglers = []

            mangler_type = next((
                mangler for mangler in [pown, powf, powr]
                if mangler().name == power_function), None)
            if mangler_type is None:
                raise Exception('Unknown OpenCL power function {}'.format(
                    power_function))
            # 1) float and short integer
            manglers.append(mangler_type())
            # 2) float and long integer
            if power_function == 'pown':
                manglers.append(mangler_type(arg_dtypes=(np.float64, np.int64)))
            if loopy_opts.is_simd:
                from loopy.target.opencl import vec
                vfloat = vec.types[np.dtype(np.float64), loopy_opts.vector_width]
                vlong = vec.types[np.dtype(np.int64), loopy_opts.vector_width]
                vint = vec.types[np.dtype(np.int32), loopy_opts.vector_width]
                # 3) vector float and short integers
                # note: return type must be non-vector form (this will converted
                # by loopy in privatize)
                if power_function == 'pown':
                    manglers.append(mangler_type(arg_dtypes=(vfloat, np.int32),
                                                 result_dtypes=np.float64))
                    manglers.append(mangler_type(arg_dtypes=(vfloat, vint),
                                                 result_dtypes=np.float64))
                # 4) vector float and long integers
                    manglers.append(mangler_type(arg_dtypes=(vfloat, np.int64),
                                                 result_dtypes=np.float64))
                manglers.append(mangler_type(arg_dtypes=(vfloat, vlong),
                                             result_dtypes=np.float64))
            return manglers
        elif power_function == 'fast_powi':
            # skip, handled as preamble
            return []
        else:
            return [powf()]

    return [mangler for power_func in utils.listify(power_functions)
            for mangler in __manglers(power_func)]


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
