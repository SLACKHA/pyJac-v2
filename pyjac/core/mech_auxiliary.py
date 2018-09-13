"""Writes mechanism header and output testing files
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os

# Local imports
from pyjac import utils
from pyjac.kernel_utils import file_writers as filew


def write_aux(path, loopy_opts, specs, reacs):
    write_mechanism_header(path, loopy_opts.lang, specs, reacs)
    write_vec_header(path, loopy_opts.lang, loopy_opts)


def write_mechanism_header(path, lang, specs, reacs):
    with filew.get_header_file(
            os.path.join(path, 'mechanism' + utils.header_ext[lang]), lang) as file:
        # define NR, NS, NN, etc.
        file.add_define('NS', len(specs))
        file.add_define('NR', len(reacs))
        file.add_define('NN', len(specs) + 1)


def write_vec_header(path, lang, loopy_opts):
    with filew.get_header_file(
            os.path.join(path, 'vectorization' + utils.header_ext[lang]),
            lang) as file:
        # define deep / wide / vecwidth
        if loopy_opts.width:
            file.add_define('WIDE')
            file.add_define('VECWIDTH', loopy_opts.width)
        elif loopy_opts.depth:
            file.add_define('DEEP')
            file.add_define('VECWIDTH', loopy_opts.depth)
        if loopy_opts.is_simd:
            file.add_define('EXPLICIT_SIMD')
