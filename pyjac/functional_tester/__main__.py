from argparse import ArgumentParser
from .test import species_rate_tester, jacobian_tester

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
    parser.add_argument('-t', '--type',
                        choices=['jac', 'spec'],
                        default='jac',
                        help='The type of validation test to run, Jacobian [jac]'
                             ' or species rates [spec].')
    args = parser.parse_args()
    method = jacobian_tester if args.type == 'jac' else species_rate_tester
    method(args.working_directory)
