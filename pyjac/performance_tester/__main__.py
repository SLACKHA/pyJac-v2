import sys
from argparse import ArgumentParser

import loopy as lp

from pyjac.performance_tester import species_performance_tester, \
    jacobian_performance_tester
from pyjac import utils


def main(args=None):
    lp.set_caching_enabled(False)
    utils.setup_logging()
    if args is None:
        # command line arguments
        parser = ArgumentParser(description='performance_tester.py: '
                                            'tests pyJac performance')
        parser.add_argument('-w', '--working_directory',
                            type=str,
                            default='performance',
                            help='Directory storing the mechanisms / data.'
                            )
        parser.add_argument('-t', '--test_matrix',
                            type=str,
                            help='The platforms / tests to run, as well as '
                                 'possible memory limits. For an example see'
                                 'the pyjac/examples/test_matrix.yaml included with'
                                 'pyJac'
                            )
        parser.add_argument('-r', '--runtype',
                            choices=['jac', 'spec', 'both'],
                            default='both',
                            help='The type of validation test to run, Jacobian [jac]'
                                 ' or species rates [spec], or [both].')
        parser.add_argument('-p', '--prefix',
                            type=str,
                            default='',
                            help='A prefix to store the output of this test in'
                                 'for each mechanism in the working_directory.'
                                 'This can be a helpful tool on a cluster to '
                                 'run multiple tests at once on different platforms')
        args = parser.parse_args()
        methods = []
        if args.runtype == 'jac':
            methods = [jacobian_performance_tester]
        elif args.runtype == 'spec':
            methods = [species_performance_tester]
        else:
            methods = [species_performance_tester, jacobian_performance_tester]

        for m in methods:
            m(args.working_directory, args.test_matrix, args.prefix)


if __name__ == '__main__':
    sys.exit(main())
