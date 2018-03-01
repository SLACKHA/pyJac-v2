# TODO way more tests here
from __future__ import division

from tempfile import NamedTemporaryFile
from collections import OrderedDict
import re

import loopy as lp
import numpy as np
from optionloop import OptionLoop
from parameterized import parameterized

from pyjac.core.array_creator import array_splitter, problem_size
from pyjac.kernel_utils.memory_manager import memory_limits, memory_type, memory_manager


def loopy_opts(langs=['opencl'],
               width=[4, None],
               depth=[4, None],
               order=['C', 'F']):

    oploop = OptionLoop(OrderedDict(
        [('lang', langs),
         ('width', width),
         ('depth', depth),
         ('order', order)]))
    for state in oploop:
        if state['depth'] and state['width']:
            continue
        yield type('', (object,), state)


@parameterized([(np.int32,), (np.int64,)])
def test_stride_limiter(dtype):
    # tests an issue, particularly for the Intel OpenCL runtime where integers in
    # array indexing that overflow the int32 max result in segfaults in kernel

    # The long term fix is probably to allow the user to specify the dtype via
    # command line or platform file, but for now we simply limit the maximum # of
    # conditions per run

    from pymbolic import parse
    arry_name = 'a'
    extractor = re.compile(r'{}\[(.+)\] = i'.format(arry_name))
    dim_size = 1000000
    for opt in loopy_opts():
        split = array_splitter(opt)
        # create a really big loopy array
        ary = lp.GlobalArg(arry_name, shape=(problem_size.name, dim_size),
                           dtype=dtype)
        # make a dummy kernel with this argument to populate dim tags
        knl = lp.make_kernel(['{{[i]: 0 <= i < {}}}'.format(dim_size),
                              '{{[j]: 0 <= j < {}}}'.format(problem_size.name)],
                             '{}[j, i] = i'.format(arry_name),
                             [ary, problem_size])
        # split said array
        knl = split.split_loopy_arrays(knl)
        ary = knl.args[0]
        # get limits object
        limits = None
        with NamedTemporaryFile(suffix='.yaml', mode='w') as temp:
            temp.write("""
                       alloc:
                          # some huge number such that this isn't the limiting factor
                          {0}
                       global:
                          {0}
                       """.format(
                        str(np.iinfo(dtype).max * 10),
                        str(np.iinfo(dtype).max * 10)))
            temp.seek(0)
            limits = memory_limits.get_limits(
                opt, {memory_type.m_global: [ary]}, temp.name,
                memory_manager.get_string_strides()[0],
                dtype=dtype)
        # and feed through stride limiter
        limit = limits.integer_limited_problem_size(ary, dtype=dtype)
        # get the intruction from the kernel
        knl = lp.generate_code_v2(knl).device_code()
        # regex the array indexing out
        index = extractor.search(knl).group(1)
        # sub out 'i', 'j' and 'problem_size'
        repl = {'i': str(dim_size - 1),
                'j': str(limit - 1),
                'problem_size': str(limit)}
        pattern = re.compile(r'\b(' + '|'.join(repl.keys()) + r')\b')
        index = pattern.sub(lambda x: repl[x.group()], index)
        index = re.sub('/', '//', index)
        max_index = parse(index)
        assert isinstance(max_index, (int, float))
        assert max_index < np.iinfo(dtype).max

        # finally, test that we get the same limit from can_fit
        assert limit == limits.can_fit(mtype=memory_type.m_global)
