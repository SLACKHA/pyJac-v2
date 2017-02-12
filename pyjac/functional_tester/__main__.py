from argparse import ArgumentParser
from . import test
from .. import utils
import os

if __name__ == '__main__':
    parser = ArgumentParser(
        description='Tests pyJac versus a finite difference'
                    ' Cantera jacobian\n'
        )
    parser.add_argument('-w', '--working_directory',
                            type=str,
                            default='performance',
                            help='Directory storing the mechanisms / data.'
                            )
    args = parser.parse_args()
    test.test(os.path.dirname(os.path.abspath(test.__file__)),
                              args.working_directory
              )
