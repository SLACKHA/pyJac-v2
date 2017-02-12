from argparse import ArgumentParser
from .test import functional_tester
from .. import utils
import os

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Tests pyJac versus a finite difference'
                    ' Cantera jacobian\n'
        )
    parser.add_argument('-w', '--working_directory',
                            type=str,
                            default='error_checking',
                            help='Directory storing the mechanisms / data.'
                            )
    args = parser.parse_args()
    functional_tester(args.working_directory)
