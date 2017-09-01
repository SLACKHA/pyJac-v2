from string import Template
import re
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

    def __init__(self, name, arg_dtypes, result_dtypes):
        self.func_mangler = MangleGen(name, arg_dtypes, result_dtypes)
        self.name = name
        self.arg_dtypes = self.__tuple_gen(arg_dtypes)
        self.result_dtypes = self.__tuple_gen(result_dtypes)

    def generate_code(self, preamble_info):
        raise NotImplementedError

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
                code = self.generate_code(func_match)
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
            'fast_powi',
            (np.float64, np.int32),
            (np.float64))

    def generate_code(self, preamble_info):
        return self.code

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
            'fast_powf',
            (np.float64, np.float64),
            (np.float64))

    def generate_code(self, preamble_info):
        return self.code

    def get_descriptor(self, func_match):
        return 'cust_funcs_fastpowf'


class OpenCL_AtomicPreambleGen(PreambleGen):
    """
    A class to enable atomic adds / sums for OpenCL doubles
    """

    def __init__(self):
        # operators
        self.code = Template("""
   inline void atomic${op_name}_${mem_name}(volatile ${mem_type} double *addr, double val)
   {
       union{
           unsigned long u64;
           double        f64;
       } next, expected, current;
    current.f64    = *addr;
       do{
       expected.f64 = current.f64;
           next.f64     = expected.f64 ${operator} val;
        current.u64  = atom_cmpxchg( (volatile ${mem_type} unsigned long *)addr,
                               expected.u64, next.u64);
       } while( current.u64 != expected.u64 );
   }
            """)

        self.operators = {'ADD': ' + ',
                          'MUL': ' * ',
                          'DIV': ' / '}

        super(OpenCL_AtomicPreambleGen, self).__init__(
            'ocl_atomics',
            (np.float64, np.float64),
            (np.float64))

    def __params_from_name(self, name):
        match = re.search(r'^atomic(?P<op>\w+)_(?P<mem>\w+)$', name)
        op, mem = match.group('op', 'mem')
        return op, mem

    def match(self, func_match):
        # check function name
        operator, mem_short = self.__params_from_name(func_match.name)
        if operator and mem_short:
            operator = operator.upper()
            # check that operator is known
            assert operator in self.operators, (
                "Don't know how to generate "
                "OpenCL atomic for operator: {}".format(operator))
            return True
        return False

    def get_descriptor(self, func_match):
        # get parameters
        operator, mem_short = self.__params_from_name(func_match.name)

        return 'ocl_atomic_{op}_{ms}'.format(op=operator.lower(),
                                             ms=mem_short)

    def generate_code(self, func_match):
        # get parameters
        operator, mem_short = self.__params_from_name(func_match.name)
        mem_type = '__global' if mem_short == 'g' else '__local'

        return self.code.substitute(
            op_name=operator,
            operator=self.operators[operator.upper()],
            mem_type=mem_type,
            mem_name=mem_short
            )


class fmax(MangleGen):
    def __init__(self, name='fmax', arg_dtypes=(np.float64, np.float64),
                 result_dtypes=np.float64):
        super(fmax, self).__init__(name, arg_dtypes, result_dtypes)
