import importlib
import numpy as np
import sys
import os
import six

home_dir = os.path.dirname(__file__)
read_ics = importlib.import_module('py_readics')
data = six.u(os.path.join(home_dir, 'data.bin')).encode('UTF-8')

phi_test = np.fromfile(os.path.join(home_dir, 'phi_test.npy'))
param_test = np.fromfile(os.path.join(home_dir, 'param_test.npy'))

order = str(sys.argv[1])
num = int(sys.argv[2])
assert order in ['C', 'F']

param_in = np.zeros_like(param_test)
phi_in = np.zeros_like(phi_test)

read_ics.read_ics(data, num, phi_in, param_in, order == 'C')

# check extra variable
allclear = np.allclose(param_in, param_test)

# and check
allclear = allclear and np.allclose(phi_in, phi_test)

sys.exit(not allclear)
