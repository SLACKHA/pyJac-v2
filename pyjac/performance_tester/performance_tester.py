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

# Local imports
from ..libgen import build_type, generate_library
from ..loopy_utils.loopy_utils import JacobianFormat, JacobianType
from ..utils import EnumType
from ..tests.test_utils import _run_mechanism_tests, runner, platform_is_gpu
from ..tests import get_platform_file, get_mem_limits_file

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
        self.steplist = []
        # initialize steplist
        step = max_vec_width
        self.max_vec_width = max_vec_width
        while step <= num_conditions:
            self.steplist.append(step)
            step *= 2

    def get_filename(self, state):
        self.current_vecwidth = state['vecsize']
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
            num_completed = self.check_full_file(filename)
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
            with open(filename, 'r', encoding="utf8", errors='ignore') as file:
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

    def run(self, state, asplit, dirs, phi_path, data_output, limits={}):
        """
        Run the validation test for the given state

        Parameters
        ----------
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

        if limits and state['sparse'] in limits:
            num_conditions = limits[state['sparse']]
            # ensure it's divisible by the maximum vector width
            num_conditions = int((num_conditions // self.max_vec_width)
                                 * self.max_vec_width)
            # remove any todo's over the maximum # of conditions
            self.todo = {k: v for k, v in six.iteritems(self.todo)
                         if k <= num_conditions}
            if num_conditions not in self.todo:
                self.todo[num_conditions] = self.repeats

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
def species_performance_tester(work_dir='performance', test_platforms=None,
                               prefix='', mem_limits=''):
    """Runs performance testing of the species rates kernel for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    test_platforms: str
        The testing platforms file, specifing the configurations to test
    prefix: str
        a prefix within the work directory to store the output of this run

    Returns
    -------
    None

    """

    raise_on_missing = True
    if not test_platforms:
        # pull default test platforms if available
        test_platforms = get_platform_file()
        # and let the tester know we can pull default opencl values if not found
        raise_on_missing = False
    if not mem_limits:
        # pull user specified memory limits if available
        mem_limits = get_mem_limits_file()

    _run_mechanism_tests(work_dir, test_platforms, prefix,
                         performance_runner(build_type.species_rates),
                         mem_limits=mem_limits,
                         raise_on_missing=raise_on_missing)


@nottest
def jacobian_performance_tester(work_dir='performance',  test_platforms=None,
                                prefix='', mem_limits=''):
    """Runs performance testing of the jacobian kernel for pyJac

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    test_platforms: str
        The testing platforms file, specifing the configurations to test
    prefix: str
        a prefix within the work directory to store the output of this run

    Returns
    -------
    None

    """

    raise_on_missing = True
    if not test_platforms:
        # pull default test platforms if available
        test_platforms = get_platform_file()
        # and let the tester know we can pull default opencl values if not found
        raise_on_missing = False
    if not mem_limits:
        # pull user specified memory limits if available
        mem_limits = get_mem_limits_file()

    _run_mechanism_tests(work_dir, test_platforms, prefix,
                         performance_runner(build_type.jacobian),
                         raise_on_missing=raise_on_missing)
