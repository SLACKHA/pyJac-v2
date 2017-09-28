"""A simple utility script that converts saved numpy files
from a Temperature, Pressure, Mass Fraction format to
Temperature, Pressure, Concentrations"""

import numpy as np
import cantera as ct
import os
import argparse


def main(input_dir='', output_dir='', mech=''):
    assert input_dir != output_dir, 'Cannot convert in same folder'
    gas = ct.Solution(mech)

    npy_files = [f for f in os.listdir(input_dir)]
    npy_files = [f for f in npy_files if f.endswith('.npy')
                 and os.path.isfile(os.path.join(input_dir, f))]
    for npy in sorted(npy_files):
        state_data = np.load(os.path.join(input_dir, npy))
        state_data = state_data.reshape(state_data.shape[0] *
                                        state_data.shape[1],
                                        state_data.shape[2]
                                        )
        out_data = np.zeros((state_data.shape[0], gas.n_species + 2))
        for i in range(state_data.shape[0]):
            # convert to T, P, C
            gas.TPY = state_data[i, 1], state_data[i, 2], state_data[i, 3:]
            out_data[i, 0] = gas.T
            out_data[i, 1] = gas.P
            out_data[i, 2:] = gas.concentrations[:]

        np.save(os.path.join(output_dir,
                             npy), out_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'A simple utility script that converts saved numpy files'
        'from a Temperature, Pressure, Mass Fraction format to'
        'Temperature, Pressure, Concentrations'))
    parser.add_argument('-i', '--input_dir',
                        type=str,
                        required=True,
                        help='The directory to scan for .npy files.'
                        )
    parser.add_argument('-o', '--output_dir',
                        type=str,
                        required=True,
                        help='The directory to place the converted .npy files.'
                        )
    parser.add_argument('-m', '--mech',
                        type=str,
                        required=True,
                        help='The Cantera format mechanism to use.')
    args = parser.parse_args()
    main(input_dir=args.input_dir,
         output_dir=args.output_dir,
         mech=args.mech)
