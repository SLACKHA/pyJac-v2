import importlib
import numpy as np
import sys

read_ics = importlib.import_module('py_readics')

T_test = np.fromfile('T_test.npy')
P_test = np.fromfile('P_test.npy')
conc_test = np.fromfile('conc_test.npy')

order = sys.argv[1]
num = int(sys.argv[2])
assert order in ['C', 'F']

T_in = np.zeros_like(T_test)
P_in = np.zeros_like(P_test)
conc_in = np.zeros_like(conc_test)

read_ics.read_ics(num, T_in, P_in, conc_in, order=='C')

allclear = np.allclose(T_in, T_test)
allclear = allclear and np.allclose(P_in, P_test)
allclear = allclear and np.allclose(conc_in, conc_test)

sys.exit(not allclear)