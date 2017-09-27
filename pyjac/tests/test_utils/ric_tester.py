import importlib
import numpy as np
import sys

read_ics = importlib.import_module('py_readics')

phi_test = np.fromfile('phi_test.npy')
param_test = np.fromfile('param_test.npy')

order = sys.argv[1]
num = int(sys.argv[2])
assert order in ['C', 'F']

param_in = np.zeros_like(param_test)
phi_in = np.zeros_like(phi_test)

read_ics.read_ics(num, phi_in, param_in, order == 'C')

# check extra variable
allclear = np.allclose(param_in, param_test)

# and check
allclear = allclear and np.allclose(phi_in, phi_test)

sys.exit(not allclear)
