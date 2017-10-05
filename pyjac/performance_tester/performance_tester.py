"""Module for performance testing of pyJac and related tools.
"""

# Python 2 compatibility
from __future__ import division
from __future__ import print_function

# Standard libraries
import os
import subprocess

# Local imports
from ..libgen import build_type, generate_library

from ..tests.test_utils import _run_mechanism_tests, runner, platform_is_gpu

import loopy as lp
lp.set_caching_enabled(False)


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

    def pre(self, gas, data, num_conditions, max_vec_width):
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
        max_vec_width: int
            unused
        """
        self.num_conditions = num_conditions

    def check_file(self, filename, state):
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

        # first, get platform
        if platform_is_gpu(state['platform']):
            self.todo = self.check_step_file(filename)
        else:
            num_completed = self.self.check_full_file(filename)
            self.todo = {self.num_conditions: self.repeats - num_completed}
        return not any(self.todo[x] > 0 for x in self.todo)

    def check_step_file(self, filename):
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

        runs = {}
        for step in self.steplist:
            runs[step] = 0

        try:
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            for line in lines:
                try:
                    vals = line.split(',')
                    if len(vals) == 2:
                        vals = [float(v) for v in vals]
                        runs[vals[0]] += 1
                except:
                    pass
            return runs
        except:
            return runs

    def check_full_file(self, filename):
        """Checks a file for existing data, returns number of completed runs

        Parameters
        ----------
        filename : str
            Name of file with data

        Returns
        -------
        num_completed : int
            Number of completed runs

        """
        try:
            with open(filename, 'r') as file:
                lines = [line.strip() for line in file.readlines()]
            num_completed = 0
            to_find = 4
            for line in lines:
                try:
                    vals = line.split(',')
                    if len(vals) == to_find:
                        nc = int(vals[0])
                        if nc != self.num_conditions:
                            # TODO: remove file and return 0?
                            raise Exception(
                                'Wrong number of conditions in performance test')

                        float(vals[1])
                        float(vals[2])
                        float(vals[3])
                        num_completed += 1
                except:
                    pass
            return num_completed
        except:
            return 0

    def run(self, state, asplit, dirs, data_output):
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
            Has the keys "build", "test" and "obj"
        data_output: str
            The file to output the results to

        Returns
        -------
        None
        """

        # first create the executable (via libgen)
        tester = generate_library(state['lang'], dirs['build'],
                                  build_dir=dirs['obj'], out_dir=dirs['test'],
                                  shared=True, btype=self.rtype, as_executable=True)

        # and do runs
        with open(data_output, 'a+') as file:
            for stepsize in self.todo:
                for i in range(self.todo[stepsize]):
                    print(i, "/", self.todo[stepsize])
                    subprocess.check_call([os.path.join(dirs['test'], tester),
                                           str(stepsize), str(state['num_cores'])],
                                          stdout=file)


def species_performance_tester(work_dir='performance'):
    """Runs performance testing of the species rates kernel for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """

    _run_mechanism_tests(work_dir, performance_runner(build_type.species_rates))


def jacobian_performance_tester(work_dir='performance'):
    """Runs performance testing of the jacobian kernel for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data

    Returns
    -------
    None

    """

    _run_mechanism_tests(work_dir, performance_runner(build_type.jacobian))
