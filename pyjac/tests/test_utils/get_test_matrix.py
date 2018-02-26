from os.path import join, abspath, exists
import psutil
import cantera as ct
import logging
from collections import OrderedDict
from nose.tools import nottest

from .. import _get_test_input, get_test_langs
from .. import platform_is_gpu
from ...libgen import build_type
from ...utils import enum_to_string, can_vectorize_lang, listify, EnumType
from ...loopy_utils.loopy_utils import JacobianType, JacobianFormat
from ...schemas import build_and_validate
from ...core.exceptions import OverrideCollisionException

model_key = r'model-list'
platform_list_key = r'platform-list'
test_matrix_key = r'test-list'


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

    models = matrix[model_key]
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


def load_platforms(matrix, langs=get_test_langs(), raise_on_empty=False):
    try:
        # try to use user-specified platforms
        oploop = []
        platforms = matrix[platform_list_key]
        # put into oploop form, and make repeatable
        for p in sorted(platforms, key=lambda x: x['name']):

            # limit to supplied languages
            inner_loop = []
            allowed_langs = langs[:]
            if 'lang' in p:
                # pull from platform languages if possible
                allowed_langs = p['lang'] if p['lang'] in allowed_langs else []
            else:
                # can't use language
                continue

            # set lang
            inner_loop.append(('lang', allowed_langs))

            # get vectorization type and size
            vectype = listify('par' if (
                'vectype' not in p or not can_vectorize_lang[allowed_langs])
                else p['vectype'])
            # check if we have a vectorization
            if not (len(vectype) == 1 and vectype[0] == 'par'):
                # try load the vecwidth, fail on missing
                try:
                    vecwidth = [x for x in listify(p['vecwidth'])]
                except TypeError:
                    raise Exception(
                        'Platform {} has non-parallel vectype(s) {} but no supplied '
                        'vecwidth.'.format(
                            p['name'], [x for x in vectype if x != 'par']))

                add_none = 'par' in vectype
                for v in [x.lower() for x in vectype]:
                    def _get(add_none):
                        if add_none:
                            return vecwidth + [None]
                        return vecwidth
                    if v == 'wide':
                        inner_loop.append(('width', _get(add_none)))
                        add_none = False
                    elif v == 'deep':
                        inner_loop.append(('depth', _get(add_none)))
                        add_none = False
                    elif v != 'par':
                        raise Exception('Platform {} has invalid supplied vectype '
                                        '{}'.format(p['name'], v))

            # fill in missing vectypes
            for x in ['width', 'depth']:
                if next((y for y in inner_loop if y[0] == x), None) is None:
                    inner_loop.append((x, None))

            # check for atomics
            if 'atomics' in p:
                inner_loop.append(('use_atomics', p['atomics']))

            # and store platform
            inner_loop.append(('platform', p['name']))

            # finally check for seperate_kernels
            sep_knl = True
            if 'seperate_kernels' in p and not p['seperate_kernels']:
                sep_knl = False
            inner_loop.append(('seperate_kernels', sep_knl))

            # create option loop and add
            oploop += [inner_loop]
    except TypeError:
        if raise_on_empty:
            raise Exception('Supplied test matrix has no platforms.')
    finally:
        if not oploop and oploop is not None:
            # file not found, or no appropriate targets for specified languages
            for lang in langs:
                inner_loop = []
                vecwidths = [4, None] if can_vectorize_lang[lang] else [None]
                inner_loop = [('lang', lang)]
                if lang == 'opencl':
                    import pyopencl as cl
                    inner_loop += [('width', vecwidths[:]),
                                   ('depth', vecwidths[:])]
                    # add all devices
                    device_types = [cl.device_type.CPU, cl.device_type.GPU,
                                    cl.device_type.ACCELERATOR]
                    platforms = cl.get_platforms()
                    platform_list = []
                    for p in platforms:
                        for dev_type in device_types:
                            devices = p.get_devices(dev_type)
                            if devices:
                                plist = [('platform', p.name)]
                                use_atomics = False
                                if 'cl_khr_int64_base_atomics' in \
                                        devices[0].extensions:
                                    use_atomics = True
                                plist.append(('use_atomics', use_atomics))
                                platform_list.append(plist)
                    for p in platform_list:
                        # create option loop and add
                        oploop += [inner_loop + p]
                elif lang == 'c':
                    inner_loop += [('platform', 'OpenMP')]
                    oploop += [inner_loop]
    return oploop


allowed_override_keys = [enum_to_string(JacobianType.exact),
                         enum_to_string(JacobianType.finite_difference),
                         enum_to_string(JacobianFormat.sparse),
                         enum_to_string(JacobianFormat.full),
                         enum_to_string(build_type.species_rates)]
allowed_overrides = ['num_cores', 'order', 'conp', 'vecsize', 'vectype']


@nottest
def load_tests(matrix, filename):
    """
    Load the tests from the pre-parsed test matrix file and check for duplicates

    Parameters
    ----------
    matrix: dict
        The parsed test matrix, i.e., output of :func:`build_and_validate`

    Returns
    -------
    tests: list
        The list of valid tests
    """

    tests = matrix[test_matrix_key]
    dupes = set()

    def __getenumtype(ttype):
        from argparse import ArgumentTypeError
        try:
            EnumType(JacobianFormat)(ttype)
            return JacobianFormat
        except ArgumentTypeError:
            return JacobianType

    from collections import defaultdict
    for test in tests:
        # first check that the
        descriptors = [test['type'] + ' - ' + test['eval-type']]
        if test['eval-type'] == 'both':
            descriptors = [test['type'] + ' - ' + enum_to_string(
                                build_type.jacobian),
                           test['type'] + ' - ' + enum_to_string(
                                build_type.species_rates)]
        if set(descriptors) & dupes:
            raise Exception('Multiple test types of {} for evaluation type {} '
                            'detected in test matrix file {}'.format(
                                test['type'], test['eval-type'], filename))

        # overrides only need to be checked within a test
        overridedupes = defaultdict(lambda: [])
        dupes.update(descriptors)
        # now go through overrides
        for desc in descriptors:
            # loop through the allowed override keys
            for override_key in [x for x in allowed_override_keys if x in test]:
                # convert to enum
                override_type = __getenumtype(override_key)
                # next loop over the actual overides (
                # i.e., :attr:`allowed_overrides`)
                for override in test[override_key]:
                    # check for collisions
                    bad = next((ct for ct in overridedupes[override]
                                if __getenumtype(ct) != override_type),
                               None)
                    if bad is not None:
                        raise OverrideCollisionException(
                            override_key, bad, override_key)
                    # nad mark duplicate
                    overridedupes[override].append(override_key)

    return tests


@nottest
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
    matrix_name = test_matrix
    test_matrix = build_and_validate('test_matrix_schema.yaml', test_matrix)

    # check that we have the working directory
    if not exists(work_dir):
        raise Exception('Work directory {} for '.format(work_dir) +
                        'testing not found, exiting...')

    # load the models
    models = load_models(work_dir, test_matrix)
    assert isinstance(test_type, build_type)

    # load tests
    tests = load_tests(test_matrix, matrix_name)
    # filter those that match the test type
    valid_str = 'validation' if for_validation else 'performance'
    tests = [test for test in tests if test['type'] == valid_str]
    tests = [test for test in tests if test['eval-type'] == enum_to_string(
        test_type)]
    # and dictify
    tests = [OrderedDict(test) for test in tests]
    if not tests:
        raise Exception('No tests found in matrix {} for {} test of {}, '
                        'exiting...'.format(matrix_name, valid_str, enum_to_string(
                         test_type)))

    # get defaults we haven't migrated to schema yet
    rate_spec = ['fixed', 'hybrid'] if test_type != build_type.jacobian \
        else ['fixed']
    sparse = ['sparse', 'full'] if test_type == build_type.jacobian else ['full']
    jtype = ['exact', 'finite_difference'] if (
        test_type == build_type.jacobian and not for_validation) else ['exact']
    split_kernels = [False]

    # and default # of cores, this may be overriden
    default_num_cores = []
    nc = 1
    if _get_test_input('num_threads', None) is not None:
        can_override_cores = False
        default_num_cores = [_get_test_input('num_threads', None)]
    else:
        max_threads = int(_get_test_input('max_threads',
                                          psutil.cpu_count(logical=False)))
        while nc <= max_threads:
            default_num_cores.append(nc)
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

    # load platforms
    platforms = load_platforms(test_matrix, raise_on_missing=raise_on_missing)
    platform_lookups = [OrderedDict(platform) for platform in platforms]
    out_params = []
    for test in tests:
        # filter platforms
        plats = platforms.copy()
        lookups = platform_lookups.copy()
        if 'platforms' in test:
            plats, lookups = [(plats[i], lookups[i]) for i in range(len(plats))
                              if lookups['platform'] in test['platforms']]
        if not len(plats):
            logger = logging.getLogger(__name__)
            logger.warn('No platforms found for test {}, skipping...'.format(
                test))
            continue

        for p, plookup in zip(plats, lookups):
            cores = default_num_cores

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

    max_vec_width = 1
    vector_params = [max(dict(p)['vecsize']) for p in params if 'vecsize' in dict(p)]
    if vector_params:
        max_vec_width = max(max_vec_width, max(vector_params))
    loop = reduce(params)
    return models, loop, max_vec_width
