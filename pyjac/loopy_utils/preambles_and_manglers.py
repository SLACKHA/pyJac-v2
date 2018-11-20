from string import Template
import textwrap

import numpy as np

from pyjac import utils as utils
from pyjac.core import array_creator as arc
from loopy import to_loopy_type


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

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.name == other.name
                and self.arg_dtypes == other.arg_dtypes
                and self.result_dtypes == other.result_dtypes
                and self.raise_on_fail == other.raise_on_fail)

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
        self.name = name
        self.code = code
        self.arg_dtypes = self.__tuple_gen(arg_dtypes)
        self.result_dtypes = self.__tuple_gen(result_dtypes)

    def generate_code(self, preamble_info):
        return self.code

    def get_descriptor(self, func_match):
        raise NotImplementedError

    @property
    def func_mangler(self):
        return MangleGen(self.name, self.arg_dtypes, self.result_dtypes)

    def match(self, func_sig):
        return func_sig.name == self.name

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self.name == other.name
                and self.arg_dtypes == other.arg_dtypes
                and self.result_dtypes == other.result_dtypes)

    def __call__(self, preamble_info):
        # find a function matching this name
        func_match = next(
            (x for x in preamble_info.seen_functions
             if self.match(x)), None)
        desc = self.get_descriptor(func_match)
        code = ''
        if func_match is not None:
            # check types
            if tuple(to_loopy_type(x) for x in self.arg_dtypes) == \
                    func_match.arg_dtypes:
                code = self.generate_code(preamble_info)
        # return code generator
        yield (desc, code)


class fastpowi_PreambleGen(PreambleGen):
    def __init__(self, lang, integer_dtype=np.int32, vector=None,
                 name='fast_powi'):
        int_str = 'int' if integer_dtype == np.int32 else 'long'
        inline = 'static inline ' if lang == 'c' else ''
        double_str = 'double'
        if vector:
            double_str += str(vector)
        # operators
        code = Template("""
   ${inline}${double_str} ${name}(${double_str} val, ${int_str} pow)
   {
        // account for negatives
        if (pow < 0)
        {
            val = 1.0 / val;
            pow = -pow;
        }
        // switch for speed
        switch(pow)
        {
            case 0:
                return 1;
            case 1:
                return val;
            case 2:
                return val * val;
            case 3:
                return val * val * val;
            case 4:
                return val * val * val * val;
            case 5:
                return val * val * val * val * val;
        }
        ${double_str} retval = val * val * val * val * val * val;
        for (${int_str} i = 6; i < pow; ++i)
        {
            retval *= val;
        }
        return retval;
   }
            """).substitute(int_str=int_str, double_str=double_str,
                            name=name, inline=inline)

        super(fastpowi_PreambleGen, self).__init__(
            name, code,
            (np.float64, integer_dtype),
            (np.float64))

    def get_descriptor(self, func_match):
        return 'cust_funcs_{}'.format(self.name)


class fastpowiv_PreambleGen(fastpowi_PreambleGen):
    def __init__(self, lang, integer_dtype=np.int32, vector_width=None):
        assert vector_width is not None
        super(fastpowiv_PreambleGen, self).__init__(
            lang, integer_dtype, vector=vector_width, name='fast_powiv')


class signaware_limiter_PreambleGen(PreambleGen):
    def __init__(self, lang, limit, vector=None, name='limiter'):
        inline = 'static inline ' if lang == 'c' else ''
        double_str = 'double'
        if vector:
            double_str += str(vector)
        # operators
        code = Template("""
   ${inline}${double_str} ${name}(${double_str} val)
   {
        return (val < 0) ? (fmin(-${limit}, val) ? fmax(${limit}, val));
   }
   """).substitute(inline=inline, double_str=double_str,
                   name=name, limit=limit)

        super(signaware_limiter_PreambleGen, self).__init__(
            name, code,
            (np.float64),
            (np.float64))

    def get_descriptor(self, func_match):
        return 'cust_funcs_{}'.format(self.name)


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


class fmax(MangleGen):
    def __init__(self, name='fmax', arg_dtypes=(np.float64, np.float64),
                 result_dtypes=np.float64):
        super(fmax, self).__init__(name, arg_dtypes, result_dtypes)


class fmin(MangleGen):
    def __init__(self, name='fmin', arg_dtypes=(np.float64, np.float64),
                 result_dtypes=np.float64):
        super(fmin, self).__init__(name, arg_dtypes, result_dtypes)


class jac_indirect_lookup(PreambleGen):
    name = 'jac_indirect'

    def __init__(self, array, target):
        self.code = Template(textwrap.dedent("""
    static inline int ${name}(int start, int end, int match)
    {
        int result = -1;
        for (int i = start; i < end; ++i)
        {
            if (${array}[i] == match)
                result = i - start;
        }
        return result;
    }
    """)).safe_substitute(name=jac_indirect_lookup.name, array=array.name)
        from loopy.kernel.data import AddressSpace as scopes
        from loopy.kernel.data import TemporaryVariable

        int_dtype = to_loopy_type(array.dtype, target=target)
        self.array = TemporaryVariable(array.name, shape=array.shape,
                                       dtype=int_dtype,
                                       initializer=array.initializer,
                                       scope=scopes.GLOBAL, read_only=True)

        super(jac_indirect_lookup, self).__init__(
            jac_indirect_lookup.name, self.code,
            (int_dtype, int_dtype, int_dtype), (int_dtype))

    def __eq__(self, other):
        return (super(jac_indirect_lookup, self).__eq__(other)
                and self.code == other.code
                and self.array == other.array)

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
