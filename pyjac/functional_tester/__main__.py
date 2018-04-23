from argparse import ArgumentParser
import sys
from pyjac.functional_tester.test import species_rate_tester, jacobian_tester
from pyjac import utils
# turn off cache
import loopy as lp


def main(args=None):
    lp.set_caching_enabled(False)
    utils.setup_logging()
    if args is None:
        # command line arguments
        parser = ArgumentParser(description='Tests pyJac versus an'
                                ' autodifferentiated jacobian\n')
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
            methods = [jacobian_tester]
        elif args.runtype == 'spec':
            methods = [species_rate_tester]
        else:
            methods = [species_rate_tester, jacobian_tester]

        for m in methods:
            m(args.working_directory, args.test_matrix, args.prefix)


if __name__ == '__main__':
    sys.exit(main())
