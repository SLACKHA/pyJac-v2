import sys
from .performance_tester import species_performance_tester, \
    jacobian_performance_tester
from argparse import ArgumentParser
from .. import utils


def main(args=None):
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
        parser.add_argument('-t', '--test_platforms',
                            type=str,
                            help='The platforms to test, for an example see'
                                 'the test_platforms_example.yaml included with'
                                 'pyJac'
                            )
        parser.add_argument('-m', '--memory_limits',
                            required=False,
                            type=str,
                            default='',
                            help='Path to a .yaml file indicating desired memory '
                                 'limits that control the desired maximum amount of '
                                 'global / local / or constant memory that the '
                                 'generated pyjac code may allocate.  Useful for '
                                 'testing, or otherwise limiting memory usage '
                                 'during runtime. The keys of this file are the '
                                 'members of :class:`pyjac.kernel_utils.'
                                 'memory_manager.mem_type`')
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
            m(args.working_directory, args.test_platforms, args.prefix,
              args.memory_limits)


if __name__ == '__main__':
    sys.exit(main())
