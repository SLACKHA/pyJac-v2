import os
from os.path import join, abspath, exists, isdir, isfile
import psutil
import sys
import cantera as ct
from collections import OrderedDict
from optionloop import OptionLoop
import logging
import re

from .. import get_test_platforms, _get_test_input
from . import platform_is_gpu
from ...libgen import build_type
from ...utils import enum_to_string
from ...loopy_utils.loopy_utils import JacobianType, JacobianFormat
from ...schemas import build_and_validate


allowed_override_keys = [enum_to_string(JacobianType.exact),
                         enum_to_string(JacobianType.finite_difference),
                         enum_to_string(JacobianFormat.sparse),
                         enum_to_string(JacobianFormat.full),
                         enum_to_string(build_type.species_rates)]

allowed_overrides = {'num_cores': int,
                     'order': ['C', 'F'],
                     'conp': ['conp', 'conv'],
                     'vecsize': int,
                     'vectype': ['par', 'w', 'd']}


def load_models(work_dir, matrix):
    """
    Load models from parsed test matrix

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    matrix: dict
        The parsed test matrix, i.e., output of :func:`build_and_validate`

    Returns
    -------
    models : dict
        A dictionary indicating which models are available for testing,
        The structure is as follows:
            mech_name : {'mech' : file path to the Cantera mechanism
                         'ns' : number of species in the mechanism
                         'limits' : {
                                'species_rates': XXX,
                                'jacobian': {
                                    'sparse': XXX
                                    'full' : XXX}
                                }
                            A dictionary of limits on the number of conditions that
                            can be evaluated for this mechanism for various
                            eval-types due to memory constraints
    """

    models = [matrix[x] for x in matrix if re.search(r'^models\.\d+$', x)]
    # find the mechanisms to test
    mechanism_list = {}

    for model in models:
        # load
        mech = model['mech']
        name = model['name']
        # default path
        path = model['path'] if 'path' in model else join(work_dir, name)

        # load mechanism
        if path is not None:
            mech = join(path, mech)
        gas = ct.Solution(mech)
        # get stats
        mechanism_list[name] = {}
        mechanism_list[name]['mech'] = mech
        # num species
        mechanism_list[name]['ns'] = gas.n_species
        del gas
        # if we have limits
        if 'limits' in model:
            mechanism_list[name]['limits'] = model['limits'].copy()

    return mechanism_list


def get_test_matrix(work_dir, test_type, test_matrix, for_validation,
                    raise_on_missing=True):
    """Runs a set of mechanisms and an ordered dictionary for
    performance and functional testing

    Parameters
    ----------
    work_dir : str
        Working directory with mechanisms and for data
    test_type: :class:`build_type.jacobian`
        Controls some testing options (e.g., whether to do a sparse matrix or not)
    test_matrix: str
        The test matrix file to load
    for_validation: bool
        If determines which test type to load from the test matrix,
        validation or performance
    raise_on_missing: bool
        Raise an exception of the specified :param:`test_matrix` file is not found
    Returns
    -------
    mechanisms : dict
        A dictionary indicating which mechanism are available for testing,
        The structure is as follows:
            mech_name : {'mech' : file path to the Cantera mechanism
                         'ns' : number of species in the mechanism
                         'limits' : {'full': XXX, 'sparse': XXX}}: a dictionary of
                            limits on the number of conditions that can be evaluated
                            for this mechanism (full & sparse jacobian respectively)
                            due to memory constraints
    params  : OrderedDict
        The parameters to put in an oploop
    max_vec_width : int
        The maximum vector width to test

    """
    work_dir = abspath(work_dir)

    # validate the test matrix
    test_matrix = build_and_validate('test_matrix_schema.yaml', test_matrix)

    # check that we have the working directory
    if not exists(work_dir):
        logger = logging.getLogger(__name__)
        logger.error('Work directory {} for '.format(work_dir) +
                     'testing not found, exiting...')
        sys.exit(-1)

    # load the models
    models = load_models(work_dir, test_matrix)
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
                conp = [True, False]
                if jac_type == 'finite_difference':
                    cores = [1]
                    # and go through platform to change vecsize to only the
                    # minimum as currently the FD jacobian doesn't vectorize
                    if (_get_key(outplat, 'lang') == 'opencl' and not
                            platform_is_gpu(_get_key(outplat, 'platform'))):
                        # get old vector widths
                        vws = _get_key(outplat, 'vecsize')
                        # delete
                        _del_key(outplat, 'vecsize')
                        # and add new
                        outplat.append(('vecsize', [vws[0]]))
                    # and change conp / conv to only conp, as again we don't really
                    # care
                    conp = [True]

                outplat = [('num_cores', cores)] + outplat + \
                          [('order', ['C', 'F']),
                           ('rate_spec', rate_spec),
                           ('split_kernels', split_kernels),
                           ('conp', conp),
                           ('sparse', sparse),
                           ('jac_type', [jac_type])]
                out_params.append(outplat[:])
        return out_params

    params = _fix_params(get_test_platforms(test_platforms,
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

    max_vec_width = 1
    vector_params = [max(dict(p)['vecsize']) for p in params if 'vecsize' in dict(p)]
    if vector_params:
        max_vec_width = max(max_vec_width, max(vector_params))
    loop = reduce(params)
    return models, loop, max_vec_width
