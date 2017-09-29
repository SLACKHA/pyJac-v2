import os
import multiprocessing
import sys
import cantera as ct
from collections import OrderedDict
from optionloop import OptionLoop
from .. import get_test_platforms


def get_test_matrix(work_dir):
    """Runs a set of mechanisms and an ordered dictionary for
    performance and functional testing

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    Returns
    -------
    mechanisms : dict
        A dictionary indicating which mechanism are available for testing,
        The structure is as follows:
            mech_name : {'mech' : file path
                         'chemkin' : chemkin file path
                         'ns' : number of species in the mechanism
                         'thermo' : the thermo file if avaialable}
    params  : OrderedDict
        The parameters to put in an oploop
    max_vec_width : int
        The maximum vector width to test

    """
    work_dir = os.path.abspath(work_dir)

    # find the mechanisms to test
    mechanism_list = {}
    if not os.path.exists(work_dir):
        print('Error: work directory {} for '.format(work_dir) +
              'testing not found, exiting...')
        sys.exit(-1)
    for name in os.listdir(work_dir):
        if os.path.isdir(os.path.join(work_dir, name)):
            # check for cti
            files = [f for f in os.listdir(os.path.join(work_dir, name)) if
                     os.path.isfile(os.path.join(work_dir, name, f))]
            for f in files:
                if f.endswith('.cti'):
                    mechanism_list[name] = {}
                    mechanism_list[name]['mech'] = f
                    mechanism_list[name]['chemkin'] = f.replace('.cti', '.dat')
                    gas = ct.Solution(os.path.join(work_dir, name, f))
                    mechanism_list[name]['ns'] = gas.n_species

                    thermo = next((tf for tf in files if 'therm' in tf), None)
                    if thermo is not None:
                        mechanism_list[name]['thermo'] = thermo

    rate_spec = ['fixed', 'hybrid']  # , 'full']
    vec_widths = [4, 8, 16]
    split_kernels = [False]
    num_cores = []
    nc = 1
    while nc < multiprocessing.cpu_count() / 2:
        num_cores.append(nc)
        nc *= 2

    def _get_key(params, key):
        for i, val in enumerate(params):
            if val[0] == key:
                return params[i][1][:]
        return [False]

    def _any_key(params, key):
        return any(x for x in _get_key(params, key))

    def _del_key(params, key):
        for i in range(len(params)):
            if params[i][0] == key:
                return params.pop(i)

    def _fix_params(params):
        for i in range(len(params)):
            platform = params[i][:]
            if _any_key(platform, 'width') or _any_key(platform, 'depth'):
                # set vec widths
                platform.append(('vecsize', vec_widths))
                # set wide flags
                if _any_key(platform, 'width'):
                    platform.append(('wide', [True, False]))
                else:
                    platform.append(('wide', [False]))
                _del_key('width')
                # set deep flags
                if _any_key(platform, 'depth'):
                    platform.append(('deep', [True, False]))
                else:
                    platform.append(('deep', [False]))
                _del_key('depth')

            cores = num_cores
            if _get_key('lang') == 'opencl':
                # test platform type
                import pyopencl as cl
                platform, = _get_key('platform')
                for p in cl.get_platforms():
                    if platform.lower() in p.name.lower():
                        # match, get device type
                        dtype = set(d.type for d in p.get_devices())
                        assert len(dtype) == 1, (
                            "Mixed device types on platform {}".format(p.name))
                        # fix cores for GPU
                        if cl.device_type.GPU in dtype:
                            cores = [1]

            platform += [('order', ['C', 'F']),
                         ('rate_spec', rate_spec),
                         ('split_kernels', split_kernels),
                         ('num_cores', cores),
                         ('conp', [True, False])]
            params[i] = platform[:]
        return params

    ocl_params = _fix_params(get_test_platforms())
    c_params = _fix_params(get_test_platforms(langs=['c']))

    oclloop = OptionLoop(OrderedDict(ocl_params), lambda: False)
    cloop = OptionLoop(OrderedDict(c_params), lambda: False)
    return mechanism_list, oclloop + cloop, vec_widths[-1]
