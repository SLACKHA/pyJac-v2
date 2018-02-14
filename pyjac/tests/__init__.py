# modules
import cantera as ct
import numpy as np
import unittest
import loopy as lp
import yaml
from nose.tools import nottest

# system
import os

# local imports
from ..core.mech_interpret import read_mech_ct
from .. import utils
import logging

# various testing globals
test_size = 8192  # required to be a power of 2 for the moment
script_dir = os.path.abspath(os.path.dirname(__file__))
build_dir = os.path.join(script_dir, 'out')
obj_dir = os.path.join(script_dir, 'obj')
lib_dir = os.path.join(script_dir, 'lib')
utils.create_dir(build_dir)


@nottest
def get_test_platforms(test_platforms, do_vector=True, langs=['opencl'],
                       raise_on_missing=False):
    try:
        oploop = []
        # try to load user specified platforms
        with open(test_platforms, 'r') as file:
            platforms = yaml.load(file.read())

        # put into oploop form, and make repeatable
        for platform in sorted(platforms):
            p = platforms[platform]

            # limit to supplied languages
            inner_loop = []
            allowed_langs = langs[:]
            if 'lang' in p:
                # pull from platform languages if possible
                allowed_langs = p['lang'] if p['lang'] in allowed_langs else []
            else:
                allowed_langs = []

            if not allowed_langs:
                # empty
                continue

            if 'do_not_run' in p and p['do_not_run']:
                oploop = None
                return oploop

            # set lang
            inner_loop.append(('lang', allowed_langs))

            # get vectorization type and size
            vectype = None if (
                'vectype' not in p or not do_vector or not
                utils.can_vectorize_lang[allowed_langs]) \
                else p['vectype']
            vecsize = 4 if 'vecsize' not in p else int(p['vecsize'])
            if vectype is not None:
                for v in [x.lower() for x in vectype]:
                    if v == 'wide':
                        inner_loop.append(('width', [vecsize, None]))
                    if v == 'deep':
                        inner_loop.append(('depth', [vecsize, None]))

            # fill in missing vectypes
            for x in ['width', 'depth']:
                if next((y for y in inner_loop if y[0] == x), None) is None:
                    inner_loop.append((x, None))

            # check for atomics
            if 'atomics' in p:
                inner_loop.append(('use_atomics', p['atomics']))

            # and store platform
            inner_loop.append((
                'platform', None if 'type' not in p else p['type']))

            # finally check for seperate_kernels
            sep_knl = True
            if 'seperate_kernels' in p and not p['seperate_kernels']:
                sep_knl = False
            inner_loop.append(('seperate_kernels', sep_knl))

            # create option loop and add
            oploop += [inner_loop]
    except IOError:
        if raise_on_missing:
            raise Exception('Test platforms file {} not found'.format(
                test_platforms))

    finally:
        if not oploop and oploop is not None:
            # file not found, or no appropriate targets for specified languages
            for lang in langs:
                inner_loop = []
                vectypes = [4, None] if do_vector and \
                    utils.can_vectorize_lang[lang] else [None]
                inner_loop = [('lang', lang)]
                if lang == 'opencl':
                    import pyopencl as cl
                    inner_loop += [('width', vectypes[:]),
                                   ('depth', vectypes[:])]
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
                else:
                    oploop += [inner_loop]
    return oploop


def _get_test_input(key, default=''):
    try:
        from testconfig import config
    except ImportError:
        # not nose
        config = {}

    value = default
    if key in config:
        logger = logging.getLogger(__name__)
        logger.info('Loading value {} = {} from testconfig'.format(
            key, config[key.lower()]))
        value = config[key.lower()]
    if key.upper() in os.environ:
        logger = logging.getLogger(__name__)
        logger.info('Loading value {} = {} from environment'.format(
            key, os.environ[key.upper()]))
        value = os.environ[key.upper()]
    return value


def get_platform_file():
    """
    Returns the user specied or default test platform file.
    This can be set in :file:`test_setup.py` or via the command line

    For an example of this file format, see :file:`test_platforms_example.py`
    """
    return _get_test_input('test_platform', 'test_platforms.yaml')


def get_mem_limits_file():
    """
    Returns the user specied or empty memory limits file
    This can be set in :file:`test_setup.py` or via the command line

    For an example of this file format, see :file:`mem_limits_example.yaml`
    """
    return _get_test_input('mem_limits', '')


def get_mechanism_file():
    """
    Returns the user specied or default Cantera mechanism to test
    This can be set in :file:`test_setup.py` or via the command line
    """
    return _get_test_input('gas', 'test.cti')


def get_test_langs():
    """
    Returns the languages to use in unit testing, defaults to OpenCL & C
    """

    return [x.strip() for x in _get_test_input('test_langs', 'opencl,c').split(',')]


class storage(object):

    def __init__(self, test_platforms, gas, specs, reacs):
        self.test_platforms = test_platforms
        self.gas = gas
        self.specs = specs
        self.reacs = reacs
        self.test_size = test_size
        self.script_dir = script_dir
        self.build_dir = build_dir
        self.obj_dir = obj_dir
        self.lib_dir = lib_dir

        # clean out build dir
        for f in os.listdir(build_dir):
            if os.path.isfile(os.path.join(build_dir, f)):
                os.remove(os.path.join(build_dir, f))

        # info
        self.nspec = self.gas.n_species
        self.nrxn = self.gas.n_reactions
        # Ns - 1 + Temperature + Extra Variable
        self.jac_dim = self.gas.n_species - 1 + 2

        # create states
        np.random.seed(0)
        self.T = np.random.uniform(600, 2200, size=test_size)
        self.P = np.random.uniform(0.5, 50, size=test_size) * ct.one_atm
        self.V = np.random.uniform(1e-3, 1, size=test_size)
        self.Y = np.random.uniform(0, 1, size=(test_size, self.gas.n_species))
        # randomly set some zeros for each species
        self.Y[np.random.choice(self.Y.shape[0], size=gas.n_species),
               np.arange(gas.n_species)] = 0
        self.concs = np.empty_like(self.Y)
        self.n = np.empty((test_size, self.gas.n_species - 1))
        self.fwd_rate_constants = np.zeros((test_size, self.gas.n_reactions))
        self.fwd_rxn_rate = np.zeros((test_size, self.gas.n_reactions))

        self.spec_u = np.zeros((test_size, gas.n_species))
        self.spec_h = np.zeros((test_size, gas.n_species))
        self.spec_cv = np.zeros((test_size, gas.n_species))
        self.spec_cp = np.zeros((test_size, gas.n_species))
        self.spec_b = np.zeros((test_size, gas.n_species))
        self.conp_temperature_rates = np.zeros(test_size)
        self.conv_temperature_rates = np.zeros(test_size)

        # third body indicies
        self.thd_inds = np.array([i for i, x in enumerate(gas.reactions())
                                  if isinstance(x, ct.FalloffReaction) or
                                  isinstance(x, ct.ChemicallyActivatedReaction) or
                                  isinstance(x, ct.ThreeBodyReaction)])
        self.ref_thd = np.zeros((test_size, self.thd_inds.size))
        self.ref_pres_mod = np.zeros((test_size, self.thd_inds.size))
        self.species_rates = np.zeros((test_size, gas.n_species))
        self.rxn_rates = np.zeros((test_size, gas.n_reactions))
        thd_eff_maps = []
        for i in self.thd_inds:
            default = gas.reaction(i).default_efficiency
            # fill all missing with default
            thd_eff_map = [default if j not in gas.reaction(i).efficiencies
                           else gas.reaction(i).efficiencies[j]
                           for j in gas.species_names]
            thd_eff_maps.append(np.array(thd_eff_map))
        thd_eff_maps = np.array(thd_eff_maps)

        # various indicies and mappings
        self.rev_inds = np.array(
            [i for i in range(gas.n_reactions) if gas.is_reversible(i)],
            dtype=np.int32)
        self.rev_rate_constants = np.zeros((test_size, self.rev_inds.size))
        self.rev_rxn_rate = np.zeros((test_size, self.rev_inds.size))
        self.equilibrium_constants = np.zeros((test_size, self.rev_inds.size))

        self.fall_inds = np.array([i for i, x in enumerate(gas.reactions())
                                   if isinstance(x, ct.FalloffReaction)])
        self.sri_inds = np.array([i for i, x in enumerate(gas.reactions())
                                  if i in self.fall_inds and isinstance(
                                    x.falloff, ct.SriFalloff)])
        self.troe_inds = np.array([i for i, x in enumerate(gas.reactions())
                                   if i in self.fall_inds and isinstance(
                                    x.falloff, ct.TroeFalloff)])
        self.lind_inds = np.array([i for i, x in enumerate(gas.reactions())
                                   if i in self.fall_inds and not
                                   (i in self.troe_inds or i in self.sri_inds)])
        self.troe_to_pr_map = np.array(
            [np.where(self.fall_inds == j)[0][0] for j in self.troe_inds])
        self.sri_to_pr_map = np.array(
            [np.where(self.fall_inds == j)[0][0] for j in self.sri_inds])
        self.lind_to_pr_map = np.array(
            [np.where(self.fall_inds == j)[0][0] for j in self.lind_inds])
        self.ref_Pr = np.zeros((test_size, self.fall_inds.size))
        self.ref_Sri = np.zeros((test_size, self.sri_inds.size))
        self.ref_Troe = np.zeros((test_size, self.troe_inds.size))
        self.ref_Lind = np.ones((test_size, self.lind_inds.size))
        self.ref_Fall = np.ones((test_size, self.fall_inds.size))
        self.ref_B_rev = np.zeros((test_size, gas.n_species))
        # and the corresponding reactions
        fall_reacs = [gas.reaction(j) for j in self.fall_inds]
        sri_reacs = [gas.reaction(j) for j in self.sri_inds]
        troe_reacs = [gas.reaction(j) for j in self.troe_inds]

        # convenience method for reduced pressure evaluation
        arrhen_temp = np.zeros(self.fall_inds.size)

        def pr_eval(i, j):
            reac = fall_reacs[j]
            return reac.low_rate(self.T[i]) / reac.high_rate(self.T[i])

        thd_to_fall_map = np.where(np.in1d(self.thd_inds, self.fall_inds))[0]

        for i in range(test_size):
            self.gas.TPY = self.T[i], self.P[i], self.Y[i, :]
            self.concs[i, :] = self.gas.concentrations[:]

            # set moles
            self.n[i, :] = self.concs[i, :-1] * self.V[i]

            # ensure that n_ns is non-negative
            n_ns = self.P[i] * self.V[i] / (ct.gas_constant * self.T[i]) \
                - np.sum(self.n[i, :])
            assert n_ns >= 0 or np.isclose(n_ns, 0)

            # store various information
            self.fwd_rate_constants[i, :] = gas.forward_rate_constants[:]
            self.rev_rate_constants[
                i, :] = gas.reverse_rate_constants[self.rev_inds]
            self.equilibrium_constants[
                i, :] = gas.equilibrium_constants[self.rev_inds]
            self.fwd_rxn_rate[i, :] = gas.forward_rates_of_progress[:]
            self.rev_rxn_rate[
                i, :] = gas.reverse_rates_of_progress[self.rev_inds]
            self.rxn_rates[i, :] = gas.net_rates_of_progress[:]
            self.species_rates[i, :] = gas.net_production_rates[:]
            self.ref_thd[i, :] = np.dot(thd_eff_maps, self.concs[i, :])
            # species thermo props
            for j in range(gas.n_species):
                cp = gas.species(j).thermo.cp(self.T[i])
                s = gas.species(j).thermo.s(self.T[i])
                h = gas.species(j).thermo.h(self.T[i])
                self.spec_cv[i, j] = cp - ct.gas_constant
                self.spec_cp[i, j] = cp
                self.spec_b[i, j] = s / ct.gas_constant - h / \
                    (ct.gas_constant * self.T[i]) - np.log(self.T[i])
                self.spec_u[i, j] = h - self.T[i] * ct.gas_constant
                self.spec_h[i, j] = h

            self.conp_temperature_rates[i] = (
                -np.dot(self.spec_h[i, :], self.species_rates[i, :]) / np.dot(
                    self.spec_cp[i, :], self.concs[i, :]))
            self.conv_temperature_rates[i] = (
                -np.dot(self.spec_u[i, :], self.species_rates[i, :]) / np.dot(
                    self.spec_cv[i, :], self.concs[i, :]))
            for j in range(self.fall_inds.size):
                arrhen_temp[j] = pr_eval(i, j)
            self.ref_Pr[i, :] = self.ref_thd[i, thd_to_fall_map] * arrhen_temp
            for j in range(self.sri_inds.size):
                self.ref_Sri[i, j] = sri_reacs[j].falloff(
                    self.T[i], self.ref_Pr[i, self.sri_to_pr_map[j]])
            for j in range(self.troe_inds.size):
                self.ref_Troe[i, j] = troe_reacs[j].falloff(
                    self.T[i], self.ref_Pr[i, self.troe_to_pr_map[j]])
            if self.sri_inds.size:
                self.ref_Fall[i, self.sri_to_pr_map] = self.ref_Sri[i, :]
            if self.troe_inds.size:
                self.ref_Fall[i, self.troe_to_pr_map] = self.ref_Troe[i, :]
            for j in range(gas.n_species):
                self.ref_B_rev[i, j] = gas.species(j).thermo.s(
                    self.T[i]) / ct.gas_constant - gas.species(j).thermo.h(
                        self.T[i]) / (ct.gas_constant * self.T[i]) - np.log(
                        self.T[i])

        # set phi
        self.phi_cp = np.concatenate((self.T.reshape(-1, 1),
                                      self.V.reshape(-1, 1),
                                      self.n), axis=1)
        self.phi_cv = np.concatenate((self.T.reshape(-1, 1),
                                      self.P.reshape(-1, 1),
                                      self.n), axis=1)

        # get extra variable rates
        mws = self.gas.molecular_weights
        mws = (1 - mws[:-1] / mws[-1])
        self.V_dot = self.V * (self.T * ct.gas_constant / self.P *
                               np.dot(self.species_rates[:, :-1], mws) +
                               self.conp_temperature_rates / self.T)

        self.P_dot = (self.T * ct.gas_constant *
                      np.dot(self.species_rates[:, :-1], mws) +
                      self.conv_temperature_rates * self.P / self.T)

        self.dphi_cp = np.concatenate((
            self.conp_temperature_rates.reshape(-1, 1),
            self.V_dot.reshape(-1, 1),
            self.species_rates[:, :-1] * self.V[:, np.newaxis]), axis=1)
        self.dphi_cv = np.concatenate((
            self.conv_temperature_rates.reshape(-1, 1),
            self.P_dot.reshape(-1, 1),
            self.species_rates[:, :-1] * self.V[:, np.newaxis]), axis=1)

        # the pressure mod terms depend on the reaction type
        # for pure third bodies, it's just the third body conc:
        self.ref_pres_mod[:, :] = self.ref_thd[:, :]
        # now find the Pr poly
        Pr_poly = 1. / (1. + self.ref_Pr)
        # and multiply the falloff (i.e. non-chem activated terms)
        pure_fall_inds = np.array([i for i, x in enumerate(gas.reactions())
                                   if isinstance(x, ct.FalloffReaction) and
                                   not isinstance(
                                        x, ct.ChemicallyActivatedReaction)])
        pure_fall_inds = np.where(np.in1d(self.fall_inds, pure_fall_inds))[0]
        Pr_poly[:, pure_fall_inds] *= self.ref_Pr[:, pure_fall_inds]
        # finally find the product of the Pr poly and the falloff blending term
        Fall_pres_mod = Pr_poly * self.ref_Fall
        # and replace in the pressure mod
        replace_inds = np.where(np.in1d(self.thd_inds, self.fall_inds))[0]
        self.ref_pres_mod[:, replace_inds] = Fall_pres_mod[:, :]


class TestClass(unittest.TestCase):
    # global setup var
    _is_setup = False
    _store = None

    @nottest
    def runTest(self):
        pass

    @property
    def store(self):
        return TestClass._store

    @store.setter
    def store(self, val):
        TestClass._store = val

    @property
    def is_setup(self):
        return TestClass._is_setup

    @is_setup.setter
    def is_setup(self, val):
        TestClass._is_setup = val

    def setUp(self):
        lp.set_caching_enabled(False)
        if not self.is_setup:
            utils.setup_logging()
            # load equations
            self.dirpath = os.path.dirname(os.path.realpath(__file__))
            gasname = os.path.join(self.dirpath, 'test.cti')
            # first check test config
            gasname = get_mechanism_file()
            # load the gas
            gas = ct.Solution(gasname)
            # the mechanism
            elems, specs, reacs = read_mech_ct(gasname)
            # and finally check for a test platform
            platform = get_platform_file()
            if platform is None:
                logger = logging.getLogger(__name__)
                logger.warn('Warning: did not find a test platform file, using '
                            'default OpenCL platforms...')
            self.store = storage(platform, gas, specs, reacs)
            self.is_setup = True
