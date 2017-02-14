import py_readics as read_ics
import numpy as np
import sys

T_test = np.load('T_test.npy')
P_test = np.load('P_test.npy')
conc_test = np.load('conc_test.npy')

order = sys.argv[1]
assert order in ['C', 'F']

T_in = np.zeros_like(T_test)
P_in = np.zeros_like(P_test)
conc_in = np.zeros_like(conc_test)

read_ics.read_ics(T_in, P_in)

allclear = np.allclose(T_in, T_test)
allclear = allclear and np.allclose(P_in, P_test)
allclear = allclear and np.allclose(conc_in, conc_test)

sys.exit(not allclear)