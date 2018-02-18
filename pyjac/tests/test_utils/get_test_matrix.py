import os
import psutil
import sys
import cantera as ct
from collections import OrderedDict, defaultdict
from optionloop import OptionLoop
from .. import get_test_platforms, _get_test_input, get_test_langs
from . import platform_is_gpu
from ...libgen import build_type
import logging
import yaml


def get_test_matrix(work_dir, test_type, test_platforms, for_validation,
                    raise_on_missing=False):
    """Runs a set of mechanisms and an ordered dictionary for
    performance and functional testing

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    test_type: :class:`build_type.jacobian`
        Controls some testing options (e.g., whether to do a sparse matrix or not)
    test_platforms: str ['']
        The platforms to test
    for_validation: bool
        If True, do not run finite difference Jacobians
    raise_on_missing: bool
        Raise an exception of the specified :param:`test_platforms` file is not found
    Returns
    -------
    mechanisms : dict
        A dictionary indicating which mechanism are available for testing,
        The structure is as follows:
            mech_name : {'mech' : file path
                         'chemkin' : chemkin file path
                         'ns' : number of species in the mechanism
                         'thermo' : the thermo file if avaialable
                         'limits' : {'full': XXX, 'sparse': XXX}}: a dictionary of
                            limits on the number of conditions that can be evaluated
                            for this mechanism (full & sparse jacobian respectively)
                            due to memory constraints
    params  : OrderedDict
        The parameters to put in an oploop
    max_vec_width : int
        The maximum vector width to test

    """
    work_dir = os.path.abspath(work_dir)

    # find the mechanisms to test
    mechanism_list = {}
    if not os.path.exists(work_dir):
        logger = logging.getLogger(__name__)
        logger.error('Work directory {} for '.format(work_dir) +
                     'testing not found, exiting...')
        sys.exit(-1)
    for name in os.listdir(work_dir):
        if os.path.isdir(os.path.join(work_dir, name)):
            # check for cti
            mech_file = [os.path.join(work_dir, name, f)
                         for f in os.listdir(os.path.join(work_dir, name)) if
                         os.path.isfile(os.path.join(work_dir, name, f))
                         and f.endswith('.yaml')]
            assert mech_file, 'YaML file describing mechanism {} not found'.format(
                name)
            with open(mech_file[0], 'r') as file:
                mech_file = yaml.load(file.read())
            mechanism_list[name] = defaultdict(lambda: None)
            mechanism_list[name]['mech'] = mech_file['mech']
            gas = ct.Solution(os.path.join(work_dir, name, mech_file['mech']))
            mechanism_list[name]['ns'] = gas.n_species
            del gas
            if 'limits' in mech_file:
                mechanism_list[name]['limits'] = mech_file['limits'].copy()

    assert isinstance(test_type, build_type)
    rate_spec = ['fixed', 'hybrid'] if test_type != build_type.jacobian \
        else ['fixed']
    sparse = ['sparse', 'full'] if test_type == build_type.jacobian else ['full']
    jtype = ['exact', 'finite_difference'] if (
        test_type == build_type.jacobian and not for_validation) else ['exact']
    vec_widths = [4, 8, 16]
    gpu_width = [64, 128]
    split_kernels = [False]
    num_cores = []
    nc = 1
    if _get_test_input('num_threads', None) is not None:
        num_cores = [_get_test_input('num_threads', None)]
    else:
        max_threads = int(_get_test_input('max_threads',
                                          psutil.cpu_count(logical=False)))
        while nc <= max_threads:
            num_cores.append(nc)
            nc *= 2

    def _get_key(params, key):
        for i, val in enumerate(params):
            if val[0] == key:
                try:
                    iter(params[i][1])
                    return params[i][1][:]
                except:
                    return (params[i][1],)
        return [False]

    def _any_key(params, key):
        return any(x for x in _get_key(params, key))

    def _del_key(params, key):
        for i in range(len(params)):
            if params[i][0] == key:
                return params.pop(i)

    def _fix_params(params):
        if params is None:
            return []
        out_params = []
        for i in range(len(params)):
            platform = params[i][:]
            cores = num_cores
            widths = vec_widths
            if _get_key(platform, 'lang') == 'opencl':
                # test platform type
                pname = _get_key(platform, 'platform')
                if platform_is_gpu(pname):
                    cores = [1]
                    widths = gpu_width

            if _any_key(platform, 'width') or _any_key(platform, 'depth'):
                # set vec widths
                platform.append(('vecsize', widths))
                # set wide flags
                if _any_key(platform, 'width'):
                    platform.append(('wide', [True, False]))
                else:
                    platform.append(('wide', [False]))
                _del_key(platform, 'width')
                # set deep flags
                if _any_key(platform, 'depth'):
                    platform.append(('deep', [True, False]))
                else:
                    platform.append(('deep', [False]))
                _del_key(platform, 'depth')

            # place cores as first changing thing in oploop so we can avoid
            # regenerating code if possible
            for jac_type in jtype:
                outplat = platform[:]
                if jac_type == 'finite_difference':
                    cores = [1]
                    # and go through platform to change vecsize to only the
                    # minimum as currently the FD jacobian doesn't vectorize
                    if (_get_key(platform, 'lang') == 'opencl' and not
                            platform_is_gpu(_get_key(platform, 'platform'))):
                        # get old vector widths
                        vws = _get_key(platform, 'vecsize')
                        # delete
                        _del_key(platform, 'vecsize')
                        # and add new
                        platform.append(('vecsize', [vws[0]]))

                outplat = [('num_cores', cores)] + outplat + \
                          [('order', ['C', 'F']),
                           ('rate_spec', rate_spec),
                           ('split_kernels', split_kernels),
                           ('conp', [True, False]),
                           ('sparse', sparse),
                           ('jac_type', jtype)]
                out_params.append(outplat[:])
        return out_params

    params = _fix_params(get_test_platforms(test_platforms, get_test_langs(),
                                            raise_on_missing=raise_on_missing))

    def reduce(params):
        out = []
        for p in params:
            val = OptionLoop(OrderedDict(p), lambda: False)
            if out == []:
                out = val
            else:
                out = out + val
        return out

    max_vec_width = max(max(dict(p)['vecsize']) for p in params
                        if 'vecsize' in dict(p))
    loop = reduce(params)
    return mechanism_list, loop, max_vec_width
