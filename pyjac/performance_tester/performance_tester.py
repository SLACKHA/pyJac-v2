"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import subprocess
from nose.tools import nottest
import six
# import io open to ignore any utf-8 characters in file output
# (e.g., from error'd OpenCL builds)
from io import open
from collections import defaultdict
import logging

# Local imports
from pyjac.libgen import build_type, generate_library
from pyjac.loopy_utils import JacobianFormat, JacobianType
from pyjac.utils import EnumType
from pyjac.tests.test_utils import _run_mechanism_tests, runner
from pyjac.tests import get_matrix_file, platform_is_gpu


class performance_runner(runner):
    def __init__(self, rtype=build_type.jacobian, repeats=10, steplist=[]):
        """
        Initialize the performance runner class

        Parameters
        ----------
        rtype: :class:`build_type`
            The type of run to test (jacobian or species_rates)
        repeats: int [10]
            The number of runs per state

        Returns
        -------
        None
        """
        super(performance_runner, self).__init__(rtype)
        self.repeats = repeats
        self.steplist = steplist

    def pre(self, gas, data, num_conditions, max_vec_size):
        """
        Initializes the performance runner for mechanism

        Parameters
        ----------
        gas: :class:`cantera.Solution`
            unused
        data: dict
            unused
        num_conditions: int
            The number of conditions to test
        max_vec_size: int
            unused
        """
        self.num_conditions = num_conditions
        self.steplist = []
        # initialize steplist
        step = max_vec_size
        self.max_vec_size = max_vec_size
        while step <= num_conditions:
            self.steplist.append(step)
            step *= 2

    def get_filename(self, state):
        self.current_vecsize = state['vecsize']
        desc = self.descriptor
        if self.rtype == build_type.jacobian:
            desc += '_sparse' if EnumType(JacobianFormat)(state['sparse'])\
                 == JacobianFormat.sparse else '_full'
        if EnumType(JacobianType)(state['jac_type']) == \
                JacobianType.finite_difference:
            desc = 'fd' + desc
        return '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                desc, state['lang'], state['vecsize'], state['order'],
                'w' if state['wide'] else 'd' if state['deep'] else 'par',
                state['platform'], state['rate_spec'],
                'split' if state['split_kernels'] else 'single',
                state['num_cores'], 'conp' if state['conp'] else 'conv') + '.txt'

    def check_file(self, filename, state, limits={}):
        """
        Checks file for existing data and determines the number of runs left
        for this file / state

        Parameters
        ----------
        filename : str
            Name of file with data
        state: dict
            The current state of the :class:`OptionLoop`, used in this context
            to provide the OpenCL platform (and determine which filetype to use)
        Returns
        -------
        valid: bool
            If true, the test case is complete and can be skipped
        """

        # get limited number of conditions, if available
        limited_num_conditions = self.have_limit(state, limits)
        num_conditions = self.num_conditions if limited_num_conditions is None else \
            limited_num_conditions

        # first, get platform
        if platform_is_gpu(state['platform']):
            self.todo = self.check_step_file(filename, num_conditions)
        else:
            num_completed = self.check_full_file(filename, num_conditions)
            self.todo = {num_conditions: self.repeats - num_completed}
        return not any(self.todo[x] > 0 for x in self.todo)

    def check_step_file(self, filename, _):
        """checks file for existing data and returns number of runs left to do
        for each step in :attr:`steplist`
        Parameters
        ----------
        filename : str
            Name of file with data
        steplist : list of int
            List of different numbers of steps
        Returns
        -------
        runs : dict
            Dictionary with number of runs left for each step
        """

        runs = defaultdict(lambda: self.repeats)
        for step in self.steplist:
            runs[step] = self.repeats

        try:
            with open(filename, 'r', encoding="utf8", errors='ignore') as file:
                lines = [line.strip() for line in file.readlines()]
            for line in lines:
                try:
                    vals = line.split(',')
                    if len(vals) == 4:
                        vals = [float(v) for v in vals]
                        runs[vals[0]] -= 1
                except ValueError:
                    pass
            return runs
        except:
            logger = logging.getLogger(__name__)
            logger.exception('Error reading performance file {}'.format(filename))
            return runs

    def check_full_file(self, filename, num_conditions):
        """Checks a file for existing data, returns number of completed runs

        Parameters
        ----------
        filename : str
            Name of file with data
        num_conditions: int
            The number of conditions (possibly limited) to check for.

        Returns
        -------
        num_completed : int
            Number of completed runs

        """
        try:
            with open(filename, 'r', encoding="utf8", errors='ignore') as file:
                lines = [line.strip() for line in file.readlines()]
            num_completed = 0
            to_find = 4
            for line in lines:
                try:
                    vals = line.split(',')
                    if len(vals) == to_find:
                        nc = int(vals[0])
                        if nc != num_conditions:
                            # TODO: remove file and return 0?
                            raise Exception(
                                'Wrong number of conditions in performance test')

                        float(vals[1])
                        float(vals[2])
                        float(vals[3])
                        num_completed += 1
                except ValueError:
                    pass
            return num_completed
        except:
            logger = logging.getLogger(__name__)
            logger.exception('Error reading performance file {}'.format(filename))
            return 0

    def run(self, state, asplit, dirs, phi_path, data_output, limits={}):
        """
        Run the validation test for the given state

        Parameters
        ----------
        state: dict
            A dictionary containing the state of the current optimization / language
            / vectorization patterns, etc.
        asplit: :class:`array_splitter`
            Not used
        dirs: dict
            A dictionary of directories to use for building / testing, etc.
            Has the keys "build", "test", "obj" and "run"
        phi_path: str
            Not used
        data_output: str
            The file to output the results to
        limits: dict
            If supplied, a limit on the number of conditions that may be tested
            at once. Important for larger mechanisms that may cause memory overflows

        Returns
        -------
        None
        """

        limited_num_conditions = self.have_limit(state, limits)
        if limited_num_conditions is not None:
            # remove any todo's over the maximum # of conditions
            self.todo = {k: v for k, v in six.iteritems(self.todo)
                         if k <= limited_num_conditions}
            if limited_num_conditions not in self.todo:
                self.todo[limited_num_conditions] = self.repeats

        # first create the executable (via libgen)
        tester = generate_library(state['lang'], dirs['build'],
                                  obj_dir=dirs['obj'], out_dir=dirs['test'],
                                  shared=True, btype=self.rtype, as_executable=True)

        # and do runs
        with open(data_output, 'a+') as file:
            for stepsize in self.todo:
                for i in range(self.todo[stepsize]):
                    print(i, "/", self.todo[stepsize])
                    subprocess.check_call([os.path.join(dirs['test'], tester),
                                           str(stepsize), str(state['num_cores'])],
                                          stdout=file)


@nottest
def species_performance_tester(work_dir='performance', test_matrix=None,
                               prefix=''):
    """Runs performance testing of the species rates kernel for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    test_matrix: str
        The testing matrix file, specifing the configurations to test
    prefix: str
        a prefix within the work directory to store the output of this run

    Returns
    -------
    None

    """

    raise_on_missing = True
    if not test_matrix:
        # pull default test platforms if available
        test_matrix = get_matrix_file()
        # and let the tester know we can pull default opencl values if not found
        raise_on_missing = False

    _run_mechanism_tests(work_dir, test_matrix, prefix,
                         performance_runner(build_type.species_rates),
                         mem_limits=test_matrix,
                         raise_on_missing=raise_on_missing)


@nottest
def jacobian_performance_tester(work_dir='performance',  test_matrix=None,
                                prefix=''):
    """Runs performance testing of the jacobian kernel for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    test_matrix: str
        The testing matrix file, specifing the configurations to test
    prefix: str
        a prefix within the work directory to store the output of this run

    Returns
    -------
    None

    """

    raise_on_missing = True
    if not test_matrix:
        # pull default test platforms if available
        test_matrix = get_matrix_file()
        # and let the tester know we can pull default opencl values if not found
        raise_on_missing = False

    _run_mechanism_tests(work_dir, test_matrix, prefix,
                         performance_runner(build_type.jacobian),
                         mem_limits=test_matrix,
                         raise_on_missing=raise_on_missing)
