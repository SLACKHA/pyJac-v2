# a configuration file for nose-testconfig that sets:
# a) the test platforms file
# and
# b) the chemical mechanism to test
# note that both can be specified on the command line via ENV variables if desired
# e.g. GAS=mymech.cti TEST_PLATFORMS=my_platform.yaml nosetests ...
# or simply feel free to modify the below...

import os
home = os.getcwd()
global config
config = {}
PLATFORM = 'test_platform.yaml'
gas = os.path.join(home, 'pyjac', 'tests', 'test.cti')
config['test_platform'] = os.path.join(home, PLATFORM)
config['gas'] = gas
