"""Main module for pywrap module.
"""
from argparse import ArgumentParser

from pyjac import utils
from pyjac.pywrap.pywrap_gen import generate_wrapper
from pyjac.libgen import build_type

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Generates a python wrapper for pyJac via Cython'
        )
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=utils.langs,
                        required=True,
                        help='Programming language for output '
                             'source files'
                        )
    parser.add_argument('-so', '--source_dir',
                        type=str,
                        required=True,
                        help='The folder that contains the generated pyJac '
                             'files.')
    parser.add_argument('-out', '--out_dir',
                        type=str,
                        required=False,
                        default=None,
                        help='The folder to place the generated library in')
    parser.add_argument('-bt', '--build_type',
                        required=False,
                        type=utils.EnumType(build_type),
                        default='jacobian',
                        help='The type of library to build: {type}'.format(
                            type=str(utils.EnumType(build_type))))

    args = parser.parse_args()
    generate_wrapper(args.lang, args.source_dir, args.out_dir, btype=args.build_type)
