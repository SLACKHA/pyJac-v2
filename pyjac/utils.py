# -*- coding: utf-8 -*-
"""Module containing utility functions.
"""

# Standard libraries
import os
import errno
import argparse
import logging.config
import sys
import subprocess
from contextlib import contextmanager
import shutil
import tempfile
from string import Template
import re
import textwrap

import six
from six.moves import reduce
import yaml
import numpy as np

from pyjac.core import exceptions

__all__ = ['langs', 'file_ext', 'header_ext', 'line_end', 'exp_10_fun',
           'get_species_mappings', 'get_nu', 'read_str_num', 'split_str',
           'create_dir', 'reassign_species_lists', 'is_integer', 'get_parser',
           'platform_is_gpu', 'stringify_args', 'partition', 'enum_to_string',
           'listify']

langs = ['c', 'opencl']  # ispc' , 'cuda'
"""list(`str`): list of supported languages"""

package_lang = {'opencl': 'ocl',
                'c': 'c'}
"""dict: str->str
   short-names for the python wrappers for each language
"""

stdindent = ' ' * 4
"""
Standard indentation
"""


def get_env_val(key, default=''):
    try:
        from testconfig import config
    except ImportError:
        # not nose
        config = {}

    value = default
    in_config = False
    if key in config:
        logger = logging.getLogger(__name__)
        in_config = True
        logger.debug('Loading value {} = {} from testconfig'.format(
            key, config[key.lower()]))
        value = config[key.lower()]
    if 'PYJAC_' + key.upper() in os.environ:
        key = 'PYJAC_' + key.upper()
        logger = logging.getLogger(__name__)
        logger.debug('{}Loading value {} = {} from environment'.format(
            'OVERRIDE: ' if in_config else '', key, os.environ[key.upper()]))
        value = os.environ[key.upper()]
    if default is not None:
        return type(default)(value)
    else:
        return value


def indent(text, prefix, predicate=None):
    """Adds 'prefix' to the beginning of selected lines in 'text'.

    If 'predicate' is provided, 'prefix' will only be added to the lines
    where 'predicate(line)' is True. If 'predicate' is not provided,
    it will default to adding 'prefix' to all non-empty lines that do not
    consist solely of whitespace characters.

    Copied from https://github.com/python/cpython/blob/master/Lib/textwrap.py for
    py2 compat.
    """
    if predicate is None:
        def predicate(line):
            return line.strip()

    def prefixed_lines():
        for line in text.splitlines(True):
            yield (prefix + line if predicate(line) else line)
    return ''.join(prefixed_lines())


def platform_is_gpu(platform):
    """
    Attempts to determine if the given platform name corresponds to a GPU

    Parameters
    ----------
    platform_name: str or :class:`pyopencl.platform`
        The name of the platform to check

    Returns
    -------
    is_gpu: bool or None
        True if platform found and the device type is GPU
        False if platform found and the device type is not GPU
        None otherwise
    """
    # filter out C or other non pyopencl platforms
    if not platform:
        return False
    if isinstance(platform, six.string_types) and 'nvidia' in platform.lower():
        return True
    try:
        import pyopencl as cl
        if isinstance(platform, cl.Platform):
            return platform.get_devices()[0].type == cl.device_type.GPU

        for p in cl.get_platforms():
            if platform.lower() in p.name.lower():
                # match, get device type
                dtype = set(d.type for d in p.get_devices())
                assert len(dtype) == 1, (
                    "Mixed device types on platform {}".format(p.name))
                # fix cores for GPU
                if cl.device_type.GPU in dtype:
                    return True
                return False
    except ImportError:
        pass
    return None


def stringify_args(arglist, kwd=False, joiner=', ', use_quotes=False,
                   remove_empty=True):
    template = '{}' if not use_quotes else '"{}"'
    if kwd:
        template = template + '=' + template
        return joiner.join(template.format(str(k), str(v))
                           for k, v in six.iteritems(arglist))
    else:
        if remove_empty:
            arglist = [x for x in arglist if x]
        return joiner.join(template.format(str(a)) for a in arglist)


def partition(tosplit, predicate):
    """
    Splits the list :param:`tosplit` based on the :param:`predicate` applied to each
    list element and returns the two resulting lists

    Parameters
    ----------
    tosplit: list
        The list to split
    predicate: :class:`six.Callable`
        A callable predicate that takes as an argument a list element to test.

    Returns
    -------
    true_list: list
        The list of elements in :param:`tosplit` for which :param:`predicate` were
        True
    false_list: list
        The list of elements in :param:`tosplit` for which :param:`predicate` were
        False
    """
    return reduce(lambda x, y: x[not predicate(y)].append(y) or x, tosplit,
                  ([], []))


file_ext = dict(c='.cpp', cuda='.cu', opencl='.ocl')
"""dict: source code file extensions based on language"""


header_ext = dict(c='.hpp', cuda='.cuh', opencl='.oclh')
"""dict: header extensions based on language"""

line_end = dict(c=';', cuda=';',
                opencl=';'
                )
"""dict: line endings dependent on language"""

can_vectorize_lang = {'c': False,
                      'cuda': True,
                      'opencl': True,
                      'ispc': True}
"""dict: defines whether a language can be 'vectorized' in the loopy sense"""

exp_10_fun = dict(c='exp({log10} * {{val}})'.format(log10=np.log(10)),
                  cuda='exp10({val})',
                  opencl='exp10({val})')
"""dict: exp10 functions for various languages"""

log_10_fun = dict(c='log10({val})',
                  cuda='log10({val})',
                  opencl='log10({val})')
"""dict: log10 functions for various languages"""


def kernel_argument_ordering(args, kernel_type, for_validation=False,
                             dummy_args=None):
    """
    A convenience method to ensure that we have a consistent set of argument
    orderings throughout pyJac

    Parameters
    ----------
    args: list of str, or :class:`loopy.KernelArgument`'s
        The arguments to determine the order of
    kernel_type: :class:`KernelType`
        The type of kernel to use (to avoid spurious placements of non-arguments)
    for_validation: bool [False]
        If True, this kernel is being used for validation (affects which arguments
        are considered kernel arguments)
    dummy_args: list of str [None]
        The kernel arguments to be used if :param:`kernel_type` == `KernelType.dummy`

    Returns
    -------
    ordered_args: list of str or :class:`loopy.KernelArgument`
        The ordered kernel arguments
    """

    from pyjac.core import array_creator as arc
    from pyjac.core.enum_types import KernelType
    from pyjac.kernel_utils.kernel_gen import rhs_work_name, local_work_name, \
        int_work_name, time_array
    # first create a mapping of names -> original arguments
    mapping = {}
    for arg in args:
        try:
            mapping[arg.name] = arg
        except AttributeError:
            # str
            mapping[arg] = arg

    value_args = [arc.problem_size.name, arc.work_size.name, time_array.name]
    va, nva = partition(mapping.keys(), lambda x: x in value_args)

    # now sort ordered by name
    ordered = sorted(va, key=lambda x: value_args.index(x))

    # next, repeat with kernel arguments
    if kernel_type != KernelType.dummy:
        kernel_args = [arc.pressure_array, arc.volume_array, arc.state_vector]
        if kernel_type == KernelType.jacobian:
            kernel_args += [arc.jacobian_array]
        elif kernel_type == KernelType.species_rates:
            kernel_args += [arc.state_vector_rate_of_change]
            if for_validation:
                kernel_args += [
                    arc.forward_rate_of_progress, arc.reverse_rate_of_progress,
                    arc.pressure_modification, arc.net_rate_of_progress]
        elif kernel_type == KernelType.chem_utils:
            kernel_args += [arc.enthalpy_array, arc.internal_energy_array,
                            arc.constant_pressure_specific_heat,
                            arc.constant_volume_specific_heat]
    else:
        assert dummy_args
        kernel_args = listify(dummy_args)

    # and finally, add the work arrays
    kernel_args += [rhs_work_name, int_work_name, local_work_name]

    # sort non-kernel-data & append
    kd, nkd = partition(nva, lambda x: x in kernel_args)

    # add non-kernel-data
    ordered.extend(sorted(nkd))

    # and sorted kernel data
    ordered.extend(sorted(kd, key=lambda x: kernel_args.index(x)))

    return [mapping[x] for x in ordered]


inf_cutoff = 1e285
"""float: A cutoff above which values are considered infinite.
          Used in testing and validation to filter values that should only
          be compared as 'large numbers'"""

exp_max = 690.775527898
"""float: the maximum allowed exponential value (evaluates to ~2e130)
          useful to avoid FPE's / overflow if necessary.
          The actual IEEE-standard specifies 709.8, but we use a slightly smaller
          value for some wiggle-room
"""

small = 1e-300
"""float: A 'small' number used to bound values above zero (e.g., for logarithms)
"""


# https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
def setup_logging(
    default_path='logging.yaml',
    default_level=logging.INFO,
    env_key='LOG_CFG'
):
    """Setup logging configuration"""
    this_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(this_dir, default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
        # double check user-specified level
        if not logging.getLogger().isEnabledFor(default_level):
            logger = logging.getLogger('pyjac')
            logger.setLevel(default_level)
            for handler in logger.handlers:
                handler.setLevel(default_level)

    else:
        logging.basicConfig(level=default_level)

    # make set depenendencies logging levels to be less verbose
    logging.getLogger('loopy').setLevel(logging.WARNING)
    logging.getLogger('pyopencl').setLevel(logging.WARNING)
    logging.getLogger('pytools').setLevel(logging.WARNING)
    logging.getLogger('codepy').setLevel(logging.WARNING)


def clean_dir(dirname, remove_dir=True):
    if not os.path.exists(dirname):
        return
    for file in os.listdir(dirname):
        if os.path.isfile(os.path.join(dirname, file)):
            os.remove(os.path.join(dirname, file))
    if remove_dir:
        shutil.rmtree(dirname, ignore_errors=True)


@contextmanager
def temporary_directory(cleanup=True):
    dirpath = tempfile.mkdtemp()
    owd = os.getcwd()
    try:
        os.chdir(dirpath)
        yield dirpath
    finally:
        os.chdir(owd)
        if cleanup:
            clean_dir(dirpath, remove_dir=True)


class EnumType(object):
    """Factory for working with argparse for creating enum object types"""
    def __init__(self, enumclass):
        self.enums = enumclass

    def __call__(self, astring):
        name = self.enums.__name__
        try:
            return self.enums[astring.lower()]
        except KeyError:
            msg = ', '.join([t.name.lower() for t in self.enums])
            msg = '{0}: use one of {1}'.format(name, msg)
            raise argparse.ArgumentTypeError(msg)

    def __repr__(self):
        astr = ', '.join([t.name.lower() for t in self.enums])
        return '{0}({1})'.format(self.enums.__name__, astr)


def enum_to_string(enum):
    """
    Convenience method that converts an IntEnum/Enum to string

    Parameters
    ----------
    enum: Enum
        The enum to convert

    Returns
    -------
    name: str
        The stringified enum
    """

    enum = str(enum)
    return enum[enum.index('.') + 1:]


def to_enum(enum, enum_type):
    """
    Attempt to convert the :param:`enum` to type :param:`enum_type`.
    If :param:`estr` is already an Enum, no effect.

    Parameters
    ----------
    enum: str or instance of :class:`Enum`
        The string or (already converted) enum type to convert
    enum_type: :class:`Enum`
        The type of Enum to convert to

    Returns
    -------
    enum: :class:`Enum`
        The converted enum

    Raises
    ------
    argparse.ArgumentTypeError
        If an improper enum type is specified
    InvalidInputSpecificationException
        If the :param:`enum` is an Enum but not of the same type as
        :param:`enum_type`
    """

    try:
        return EnumType(enum_type)(enum)
    except AttributeError:
        # not a string
        if enum not in enum_type:
            logger = logging.getLogger()
            logger.error('Enum {} is not of type {}'.format(enum, enum_type))
            raise exceptions.InvalidInputSpecificationException(enum)
        return enum


def is_iterable(value):
    """
    Custom iterable test that excludes string types

    Parameters
    ----------
    value: object
        The value to test if iterable

    Returns
    -------
    iterable: bool
        True if the value is iterable and not a string, false otherwise
    """
    if isinstance(value, six.string_types):
        return False

    try:
        [vi for vi in value]
        return True
    except TypeError:
        return False


def listify(value):
    """
    Convert value to list

    Parameters
    ----------
    value: object
        The value to convert

    Returns
    -------
    listified: list
        If string, return [string]
        If tuple or other iterable, convert to lsit
        If not iterable, return [value]
    """
    if isinstance(value, six.string_types):
        return [value]

    try:
        return [vi for vi in value]
    except TypeError:
        return [value]


def get_species_mappings(num_specs, last_species):
    """
    Maps species indices around species moved to last position.

    Parameters
    ----------
    num_specs : int
        Number of species.
    last_species : int
        Index of species being moved to end of system.

    Returns
    -------
    fwd_species_map : list of `int`
        List of original indices in new order
    back_species_map : list of `int`
        List of new indicies in original order

    """

    fwd_species_map = list(range(num_specs))
    back_species_map = list(range(num_specs))

    # in the forward mapping process
    # last_species -> end
    # all entries after last_species are reduced by one
    back_species_map[last_species + 1:] = back_species_map[last_species:-1]
    back_species_map[last_species] = num_specs - 1

    # in the backwards mapping
    # end -> last_species
    # all entries with value >= last_species are increased by one
    ind = fwd_species_map.index(last_species)
    fwd_species_map[ind:-1] = fwd_species_map[ind + 1:]
    fwd_species_map[-1] = last_species

    return fwd_species_map, back_species_map


def get_nu(isp, rxn):
    """Returns the net nu of species isp for the reaction rxn

    Parameters
    ----------
    isp : int
        Species index
    rxn : `ReacInfo`
        Reaction

    Returns
    -------
    nu : int
        Overall stoichiometric coefficient of species ``isp`` in reaction ``rxn``

    """
    if isp in rxn.prod and isp in rxn.reac:
        nu = (rxn.prod_nu[rxn.prod.index(isp)] -
              rxn.reac_nu[rxn.reac.index(isp)])
        # check if net production zero
        if nu == 0:
            return 0
    elif isp in rxn.prod:
        nu = rxn.prod_nu[rxn.prod.index(isp)]
    elif isp in rxn.reac:
        nu = -rxn.reac_nu[rxn.reac.index(isp)]
    else:
        # doesn't participate in reaction
        return 0
    return nu


def read_str_num(string, sep=None):
    """Returns a list of floats pulled from a string.

    Delimiter is optional; if not specified, uses whitespace.

    Parameters
    ----------
    string : str
        String to be parsed.
    sep : str, optional
        Delimiter (default is None, which means consecutive whitespace).

    Returns
    -------
    list of `float`
        Floats separated by ``sep`` in ``string``.

    """

    # separate string into space-delimited strings of numbers
    num_str = string.split(sep)
    return [float(n) for n in num_str]


def split_str(seq, length):
    """Separate a string seq into length-sized pieces.

    Parameters
    ----------
    seq : str
        String containing sequence of smaller strings of constant length.
    length : int
        Length of individual sequences.

    Returns
    -------
    list of `str`
        List of strings of length ``length`` from ``seq``.

    """
    return [seq[i: i + length] for i in range(0, len(seq), length)]


def create_dir(path):
    """Creates a new directory based on input path.

    No error if path already exists, but other error is reported.

    Parameters
    ----------
    path : str
        Path of directory to be created

    Returns
    -------
    None

    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def reassign_species_lists(reacs, specs):
    """
    Given a list of `ReacInfo`, and `SpecInfo`, this method will update the
    `ReacInfo` reactants / products / third body list to integers
    representing the species' index in the list.

    Parameters
    ----------
    reacs : list of `ReacInfo`
        List of reactions to be updated.
    specs : list of `SpecInfo`
        List of species

    Returns
    -------
    None

    """

    species_map = {sp.name: i for i, sp in enumerate(specs)}
    for rxn in reacs:
        rxn.reac, rxn.reac_nu = zip(*[(species_map[sp], nu) for sp, nu in
                                      sorted(zip(rxn.reac, rxn.reac_nu),
                                             key=lambda x:species_map[x[0]])])
        rxn.prod, rxn.prod_nu = zip(*[(species_map[sp], nu) for sp, nu in
                                      sorted(zip(rxn.prod, rxn.prod_nu),
                                             key=lambda x:species_map[x[0]])])
        rxn.thd_body_eff = sorted([(species_map[thd[0]], thd[1])
                                   for thd in rxn.thd_body_eff], key=lambda x: x[0])
        if rxn.pdep_sp != '':
            rxn.pdep_sp = species_map[rxn.pdep_sp]
        else:
            rxn.pdep_sp = None


def is_integer(val):
    """Returns `True` if argument is an integer or whole number.

    Parameters
    ----------
    val : int, float
        Value to be checked.

    Returns
    -------
    bool
        ``True`` if ``val`` is `int` or whole number (if `float`).

    """
    try:
        return val.is_integer()
    except AttributeError:
        if isinstance(val, int):
            return True
        # last ditch effort
        try:
            return int(val) == float(val)
        except (ValueError, TypeError):
            return False


def run_with_our_python(command):
    """
    Run the given :param:`command` through subprocess, attempting as best as possible
    to utilize the same python intepreter as is currently running.

    Notes
    -----
    Does not perform any error checking, the calling code is responsible for this.

    Params
    ------
    command: list of str
        The subprocess command to run

    Returns
    -------
    None
    """

    cmd = [sys.executable]
    subprocess.check_call(cmd + command, env=os.environ.copy())


def check_lang(lang):
    """
    Checks that 'lang' is a valid identifier

    Parameters
    ----------
    lang : {'c', 'opencl', 'cuda'}
        The language to check

    Notes
    -----
    Raised NotImplementedError if incorrect lang given
    """
    if lang not in langs:
        raise NotImplementedError('Language {} not supported'.format(lang))


def check_order(order):
    """
    Checks that the :param:`order` is valid

    Parameters
    ----------
    order: ['C', 'F']
        The order to use, 'C' corresponds to a row-major data ordering, while
        'F' is a column-major data ordering.  See `row major`_ and `col major`_
        for more info

    .. _row major: https://docs.scipy.org/doc/numpy/glossary.html#term-row-major
    .. _col major: https://docs.scipy.org/doc/numpy/glossary.html#term-column-major

    Notes
    -----
    :class:`InvalidInputSpecificationException` raised if :param:`order` is not
        valid
    """

    if order not in ['C', 'F']:
        logger = logging.getLogger(__name__)
        logger.error("Invalid data-ordering ('{}') supplied, allowed values are 'C'"
                     " and 'F'".format(order))
        raise exceptions.InvalidInputSpecificationException('order')


def _find_indent(template_str, key, value):
    """
    Finds and returns a formatted value containing the appropriate
    whitespace to put 'value' in place of 'key' for template_str

    Parameters
    ----------
    template_str : str
        The string to sub into
    key : str
        The key in the template string
    value : str
        The string to format

    Returns
    -------
    formatted_value : str
        The properly indented value
    """

    # find the instance of ${key} in kernel_str
    whitespace = None
    for i, line in enumerate(template_str.split('\n')):
        if key in line:
            # get whitespace
            whitespace = re.match(r'\s*', line).group()
            break
    if whitespace is None:
        raise Exception('Key {} not found in template: {}'.format(key, template_str))
    result = [line if i == 0 else whitespace + line for i, line in
              enumerate(textwrap.dedent(value).splitlines())]
    return '\n'.join(result)


def subs_at_indent(template_str, **kwargs):
    """
    Substitutes keys of :params:`kwargs` for values in :param:`template_str`
    ensuring that the indentation of the value is the same as that of the key
    for all lines present in the value

    Parameters
    ----------
    template_str : str
        The string to sub into
    kwargs: dict
        The dictionary of keys -> values to substituted into the template
    Returns
    -------
    formatted_value : str
        The formatted string
    """

    return Template(template_str).safe_substitute(
        **{key: _find_indent(template_str, '${{{key}}}'.format(key=key),
                             value if isinstance(value, str) else str(value))
            for key, value in six.iteritems(kwargs)})


def copy_with_extension(lang, file, topath, frompath=None, header=False):
    """
    Copies :param:`file` to :param:`topath`, while changing the extension to
    match that of the given :param:`lang`

    Parameters
    ----------
    lang : str
        The language to determine the extension of
    file : str
        Either the full path to the file, or the filename to copy
    topath : str
        The path to copy the modified file to
    frompath : str [None]
        If specified, :param:`file` is a filename, and is located at
        :param:`frompath`
    header : bool [False]
        If true, this file is a header (and we should use the header extensions)


    Returns
    -------
    None
    """

    if frompath:
        file = os.path.join(frompath, file)

    base_name = os.path.basename(file)
    outfile = os.path.join(topath, base_name)

    # replace extension
    ext = header_ext[lang] if header else file_ext[lang]
    outfile = outfile[:outfile.rindex('.')] + ext

    # and copy
    shutil.copyfile(file, outfile)


def get_parser():
    """

    Parameters
    ----------
    None

    Returns
    -------
    args : `argparse.Namespace`
        Command line arguments for running pyJac.

    """

    # import enums
    from pyjac.core.enum_types import KernelType, RateSpecialization, \
        JacobianFormat, JacobianType

    # command line arguments
    parser = argparse.ArgumentParser(description='pyJac: Generates source code '
                                     'for analytical chemical Jacobians.')
    parser.add_argument('-l', '--lang',
                        type=str,
                        choices=langs,
                        required=True,
                        help='Programming language for output source files.'
                        )
    parser.add_argument('-i', '--input',
                        type=str,
                        required=True,
                        help='Input mechanism filename (e.g., mech.dat).'
                        )
    parser.add_argument('-t', '--thermo',
                        type=str,
                        default=None,
                        help='Thermodynamic database filename (e.g., '
                             'therm.dat), or nothing if in mechanism.'
                        )
    parser.add_argument('-w', '--width',
                        required=False,
                        type=int,
                        default=None,
                        help='Use a "wide" vectorization of vector-width "width".'
                        'The calculation of the Jacobian / source terms'
                        ' is vectorized over the entire set of thermo-chemical '
                        'states.  That is, each work-item (CUDA thread) '
                        'operates independently.')
    parser.add_argument('-d', '--depth',
                        required=False,
                        type=int,
                        default=None,
                        help='Use a "deep" vectorization of vector-width "depth".'
                        'The calculation of the Jacobian / source terms'
                        ' is vectorized over each individaul thermo-chemical '
                        'state.  That is, the various work-items (CUDA threads) '
                        'cooperate.')
    parser.add_argument('-se', '--explicit_simd',
                        required=False,
                        default=None,
                        action='store_true',
                        help='Use explicit-SIMD instructions in OpenCL if possible. '
                             'Note: currently available for wide-vectorizations '
                             'only.')
    parser.add_argument('-si', '--implicit_simd',
                        required=False,
                        action='store_false',
                        dest='explicit_simd',
                        help='Use implict-SIMD vectorization in OpenCL.')
    parser.add_argument('-unr', '--unroll',
                        type=int,
                        default=None,
                        required=False,
                        help='If supplied, a length to unroll the inner loops (e.g. '
                        'evaluation of the species rates for a single '
                        'thermo-chemical state) in the generated code. Turned off '
                        'by default.'
                        )
    parser.add_argument('-b', '--build_path',
                        required=False,
                        default='./out/',
                        help='The folder to generate the Jacobian and rate '
                             'subroutines in.'
                        )
    parser.add_argument('-ls', '--last_species',
                        required=False,
                        type=str,
                        default=None,
                        help='The name of the species to set as the last in '
                             'the mechanism. If not specifed, defaults to '
                             'the first of N2, AR, and HE in the mechanism.'
                        )
    parser.add_argument('-k', '--kernel_type',
                        required=False,
                        type=EnumType(KernelType),
                        default='jacobian',
                        help='The type of kernel to generate: {type}'.format(
                            type=str(EnumType(KernelType))))
    parser.add_argument('-p', '--platform',
                        required=False,
                        default='',
                        type=str,
                        help='The name (or subset thereof) of the OpenCL platform '
                             'to run on, e.g. "Intel", "nvidia", "pocl". '
                             'Must be supplied to properly generate the compilation '
                             'wrapper for OpenCL code, but may be ignored if not '
                             'using the OpenCL target.'),
    parser.add_argument('-o', '--data_order',
                        default='C',
                        type=str,
                        choices=['C', 'F'],
                        help="The data ordering, 'C' (row-major, recommended for "
                        "CPUs) or 'F' (column-major, recommended for GPUs)")
    parser.add_argument('-rs', '--rate_specialization',
                        default='hybrid',
                        type=EnumType(RateSpecialization),
                        help="The level of specialization in evaluating reaction "
                        "rates. 'Full' is the full form suggested by Lu et al. "
                        "(citation) 'Hybrid' turns off specializations in the "
                        "exponential term (Ta = 0, b = 0) 'Fixed' is a fixed"
                        " expression exp(logA + b logT + Ta / T).  Choices:"
                        ' {type}'.format(type=str(EnumType(RateSpecialization))))
    parser.add_argument('-rk', '--fused_rate_kernels',
                        default=False,
                        action='store_true',
                        help="If supplied, and the :param`rate_specialization` "
                        "is not 'Fixed', different rate evaluation will be evaluated"
                        " into in the same function.")
    parser.add_argument('-rn', '--split_rop_net_kernels',
                        default=False,
                        action='store_true',
                        help="If supplied, break evaluation of different rate of "
                        "progress values (fwd / back / pdep) into different "
                        "kernels. Note that for a deep vectorization this will "
                        "introduce additional synchronization requirements.")
    parser.add_argument('-conv', '--constant_volume',
                        required=False,
                        dest='conp',
                        action='store_false',
                        help='If supplied, use the constant volume assumption in '
                        'generation of the rate subs / Jacobian code. Otherwise, '
                        'use the constant pressure assumption [default].')
    parser.add_argument('-nad', '--no_atomic_doubles',
                        dest='use_atomic_doubles',
                        action='store_false',
                        required=False,
                        help='If supplied, the targeted language / platform is not'
                        'capable of using atomic instructions for double-precision '
                        'floating point types. This affects how deep vectorization '
                        'code is generated, and will force any potential data-races '
                        'to be run in serial/sequential form, resulting in '
                        'suboptimal deep vectorizations.'
                        )
    parser.add_argument('-nai', '--no_atomic_ints',
                        dest='use_atomic_ints',
                        action='store_false',
                        required=False,
                        help='If supplied, the targeted language / platform is not'
                        'capable of using atomic instructions for single-precision '
                        'integer types. This affects the generated driver kernel, '
                        'see "Driver Kernel Types" in the documentation.'
                        )
    parser.add_argument('-jt', '--jac_type',
                        type=EnumType(JacobianType),
                        required=False,
                        default='exact',
                        help='The type of Jacobian kernel to generate.  An '
                        'approximate Jacobian ignores derivatives of the last '
                        'species with respect to other species in the mechanism.'
                        'This can significantly increase sparsity for mechanisms '
                        'containing reactions that include the last species '
                        'directly, or as a third-body species with a non-unity '
                        'efficiency, but gives results in an approxmiate Jacobian, '
                        'and thus is more suitable to use with implicit integration '
                        'techniques. Choices: {type}'.format(
                            type=str(EnumType(JacobianType)))
                        )
    parser.add_argument('-jf', '--jac_format',
                        type=EnumType(JacobianFormat),
                        required=False,
                        default='sparse',
                        help='If "sparse", the Jacobian will be encoded using a '
                        'compressed row or column storage format (for a data order '
                        'of "C" and "F" respectively). Choices: {type}'.format(
                            type=str(EnumType(JacobianFormat)))
                        )
    parser.add_argument('-up', '--unique_pointers',
                        required=False,
                        default=False,
                        action='store_true',
                        help='If specified, this indicates that the pointers passed '
                             'to the generated pyJac methods will be unique (i.e., '
                             'distinct per OpenMP thread / OpenCL work-group). '
                             'This option is most useful for coupling to external '
                             'codes an that have already been parallelized.'
                        )
    parser.add_argument('-m', '--memory_limits',
                        required=False,
                        type=str,
                        default='',
                        help='Path to a .yaml file indicating desired memory limits '
                             'that control the desired maximum amount of global / '
                             'local / or constant memory that the generated pyjac '
                             'code may allocate.  Useful for testing, or otherwise '
                             'limiting memory usage during runtime. '
                             'The keys of this file are the members of '
                             ':class:`pyjac.kernel_utils.memory_limits.mem_type`')
    parser.add_argument('--verbose',
                        action='store_const',
                        const=logging.DEBUG,
                        dest='loglevel',
                        help='Increase verbosity of logging / output messages.',
                        default=logging.INFO)
    from pyjac.core.enum_types import reaction_sorting
    parser.add_argument('--reaction_sorting',
                        type=EnumType(reaction_sorting),
                        default=reaction_sorting.none,
                        help='Enable sorting of reactions [beta].')
    args = parser.parse_args()
    return args


def create(**kwargs):
    args = get_parser()
    vars(args).update(kwargs)
    setup_logging(default_level=args.loglevel)
    from pyjac.core.create_jacobian import create_jacobian
    create_jacobian(lang=args.lang,
                    mech_name=args.input,
                    therm_name=args.thermo,
                    width=args.width,
                    depth=args.depth,
                    unr=args.unroll,
                    build_path=args.build_path,
                    last_spec=args.last_species,
                    kernel_type=args.kernel_type,
                    platform=args.platform,
                    data_order=args.data_order,
                    rate_specialization=args.rate_specialization,
                    split_rate_kernels=not args.fused_rate_kernels,
                    split_rop_net_kernels=args.split_rop_net_kernels,
                    conp=args.conp,
                    use_atomic_doubles=args.use_atomic_doubles,
                    use_atomic_ints=args.use_atomic_ints,
                    jac_type=args.jac_type,
                    jac_format=args.jac_format,
                    mem_limits=args.memory_limits,
                    unique_pointers=args.unique_pointers,
                    explicit_simd=args.explicit_simd,
                    rsort=args.reaction_sorting
                    )
