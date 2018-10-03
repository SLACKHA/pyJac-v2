# Configuration file for nose-testconfig that sets:
# a) the test platforms file
# b) the chemical mechanism to test
# c) the maximum number of threads to test
# d) the relative / absolute tolerances
# e) the languages to test
# f) the reaction sorting method to use
# All test configuration variables  can be specified on the command line via
# ENV variables (prefixed with PYJAC_) if desired
# e.g. PYJAC_GAS=mymech.cti PYJAC_TEST_PLATFORM=my_platform.yaml nosetests ...
# or simply feel free to modify the below...
# NOTE: supplied enviroment variables with override variables set in this test config

import os
home = os.getcwd()
global config
config = {}
PLATFORM = 'test_platform.yaml'
gas = os.path.join(home, 'pyjac', 'tests', 'test.cti')
config['test_platform'] = os.path.join(home, PLATFORM)
config['gas'] = gas
# set test languages to opencl & c
config['test_langs'] = 'opencl,c'
# unused by default, sets maximum # of hardware threads for testing
# config['max_threads'] = None
# unused by default, but allows the user to specify relative tolerance for unit tests
# note that the default tolerances should work for the test mechanism, but you may
# need to adjust for other (real) mechanisms
# config['rtol'] = 1e-3
# unused by default, but allows the user to specify absolute tolerance for unit tests
# note that the default tolerances should work for the test mechanism, but you may
# need to adjust for other (real) mechanisms
# config['atol'] = 1
# Set the number of initial conditions for pyJac testing
# config['test_size'] = 8192
# Set the type of reaction sorting to utilize
# config['rxn_sort'] = 'simd'
