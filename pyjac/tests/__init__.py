#modules
import cantera as ct
import numpy as np
import unittest
import loopy as lp

#system
import os

#local imports
from ..sympy_utils.sympy_interpreter import load_equations
from ..core.mech_interpret import read_mech_ct
import logging

logging.getLogger('root').setLevel(logging.WARNING)

from .. import utils

#various testing globals
test_size = 8192 #required to be a power of 2 for the moment
script_dir = os.path.abspath(os.path.dirname(__file__))
build_dir = os.path.join(script_dir, 'out')
obj_dir = os.path.join(script_dir, 'obj')
lib_dir = os.path.join(script_dir, 'lib')
utils.create_dir(build_dir)

class storage(object):
    def __init__(self, conp_vars, conp_eqs, conv_vars,
        conv_eqs, gas, specs, reacs):
        self.conp_vars = conp_vars
        self.conp_eqs = conp_eqs
        self.conv_vars = conv_vars
        self.conv_eqs = conv_eqs
        self.gas = gas
        self.specs = specs
        self.reacs = reacs
        self.test_size = test_size
        self.script_dir = script_dir
        self.build_dir = build_dir
        self.obj_dir = obj_dir
        self.lib_dir = lib_dir

        #clean out build dir
        for f in os.listdir(build_dir):
            if os.path.isfile(os.path.join(build_dir, f)):
                os.remove(os.path.join(build_dir, f))

        #create states
        self.T = np.random.uniform(600, 2200, size=test_size)
        self.P = np.random.uniform(0.5, 50, size=test_size) * ct.one_atm
        self.Y = np.random.uniform(0, 1, size=(test_size, self.gas.n_species))
        #randomly set some zeros for each species
        self.Y[np.random.choice(self.Y.shape[0], size=gas.n_species),
               np.arange(gas.n_species)] = 0
        self.concs = np.empty_like(self.Y)
        self.fwd_rate_constants = np.zeros((test_size, self.gas.n_reactions))
        self.fwd_rxn_rate = np.zeros((test_size, self.gas.n_reactions))

        self.spec_u = np.zeros((test_size, gas.n_species))
        self.spec_h = np.zeros((test_size, gas.n_species))
        self.spec_cv = np.zeros((test_size, gas.n_species))
        self.spec_cp = np.zeros((test_size, gas.n_species))
        self.spec_b = np.zeros((test_size, gas.n_species))
        self.conp_temperature_rates = np.zeros(test_size)
        self.conv_temperature_rates = np.zeros(test_size)

        #third body indicies
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
            #fill all missing with default
            thd_eff_map = [default if j not in gas.reaction(i).efficiencies
                              else gas.reaction(i).efficiencies[j] for j in gas.species_names]
            thd_eff_maps.append(np.array(thd_eff_map))
        thd_eff_maps = np.array(thd_eff_maps)

        #various indicies and mappings
        self.rev_inds = np.array([i for i in range(gas.n_reactions) if gas.is_reversible(i)])
        self.rev_rate_constants = np.zeros((test_size, self.rev_inds.size))
        self.rev_rxn_rate = np.zeros((test_size, self.rev_inds.size))
        self.equilibrium_constants = np.zeros((test_size, self.rev_inds.size))

        self.fall_inds = np.array([i for i, x in enumerate(gas.reactions())
            if isinstance(x, ct.FalloffReaction)])
        self.sri_inds = np.array([i for i, x in enumerate(gas.reactions())
            if i in self.fall_inds and isinstance(x.falloff, ct.SriFalloff)])
        self.troe_inds = np.array([i for i, x in enumerate(gas.reactions())
            if i in self.fall_inds and isinstance(x.falloff, ct.TroeFalloff)])
        self.lind_inds = np.array([i for i, x in enumerate(gas.reactions())
            if i in self.fall_inds and not (i in self.troe_inds or i in self.sri_inds)])
        troe_to_pr_map = np.array([np.where(self.fall_inds == j)[0][0] for j in self.troe_inds])
        sri_to_pr_map = np.array([np.where(self.fall_inds == j)[0][0] for j in self.sri_inds])
        self.ref_Pr = np.zeros((test_size, self.fall_inds.size))
        self.ref_Sri = np.zeros((test_size, self.sri_inds.size))
        self.ref_Troe = np.zeros((test_size, self.troe_inds.size))
        self.ref_Lind = np.ones((test_size, self.lind_inds.size))
        self.ref_Fall = np.ones((test_size, self.fall_inds.size))
        self.ref_B_rev = np.zeros((test_size, gas.n_species))
        #and the corresponding reactions
        fall_reacs = [gas.reaction(j) for j in self.fall_inds]
        sri_reacs = [gas.reaction(j) for j in self.sri_inds]
        troe_reacs = [gas.reaction(j) for j in self.troe_inds]

        #convenience method for reduced pressure evaluation
        arrhen_temp = np.zeros(self.fall_inds.size)
        def pr_eval(i, j):
            reac = fall_reacs[j]
            return reac.low_rate(self.T[i]) / reac.high_rate(self.T[i])

        thd_to_fall_map = np.where(np.in1d(self.thd_inds, self.fall_inds))[0]

        for i in range(test_size):
            self.gas.TPY = self.T[i], self.P[i], self.Y[i, :]
            self.concs[i, :] = self.gas.concentrations[:]

            #store various information
            self.fwd_rate_constants[i, :] = gas.forward_rate_constants[:]
            self.rev_rate_constants[i, :] = gas.reverse_rate_constants[self.rev_inds]
            self.equilibrium_constants[i, :] = gas.equilibrium_constants[self.rev_inds]
            self.fwd_rxn_rate[i, :] = gas.forward_rates_of_progress[:]
            self.rev_rxn_rate[i, :] = gas.reverse_rates_of_progress[self.rev_inds]
            self.rxn_rates[i, :] = gas.net_rates_of_progress[:]
            self.species_rates[i, :] = gas.net_production_rates[:]
            self.ref_thd[i, :] = np.dot(thd_eff_maps, self.concs[i, :])
            #species thermo props
            for j in range(gas.n_species):
                cp = gas.species(j).thermo.cp(self.T[i])
                s = gas.species(j).thermo.s(self.T[i])
                h = gas.species(j).thermo.h(self.T[i])
                self.spec_cv[i, j] = cp - ct.gas_constant
                self.spec_cp[i, j] = cp
                self.spec_b[i, j] = s / ct.gas_constant - h / (ct.gas_constant * self.T[i]) - np.log(self.T[i])
                self.spec_u[i, j] = h - self.T[i] * ct.gas_constant
                self.spec_h[i, j] = h

            self.conp_temperature_rates[i] = (-np.dot(self.spec_h[i, :], self.species_rates[i, :])
                                                / np.dot(self.spec_cp[i, :], self.concs[i, :]))
            self.conv_temperature_rates[i] = (-np.dot(self.spec_u[i, :], self.species_rates[i, :])
                                                / np.dot(self.spec_cv[i, :], self.concs[i, :]))
            for j in range(self.fall_inds.size):
                arrhen_temp[j] = pr_eval(i, j)
            self.ref_Pr[i, :] = self.ref_thd[i, thd_to_fall_map] * arrhen_temp
            for j in range(self.sri_inds.size):
                self.ref_Sri[i, j] = sri_reacs[j].falloff(self.T[i], self.ref_Pr[i, sri_to_pr_map[j]])
            for j in range(self.troe_inds.size):
                self.ref_Troe[i, j] = troe_reacs[j].falloff(self.T[i], self.ref_Pr[i, troe_to_pr_map[j]])
            if self.sri_inds.size:
                self.ref_Fall[i, sri_to_pr_map] = self.ref_Sri[i, :]
            if self.troe_inds.size:
                self.ref_Fall[i, troe_to_pr_map] = self.ref_Troe[i, :]
            for j in range(gas.n_species):
                self.ref_B_rev[i, j] = gas.species(j).thermo.s(self.T[i]) / ct.gas_constant -\
                    gas.species(j).thermo.h(self.T[i]) / (ct.gas_constant * self.T[i]) - np.log(self.T[i])

        # set phi
        self.phi = np.concatenate((self.T.reshape(-1, 1), self.concs), axis=1)

        #the pressure mod terms depend on the reaction type
        #for pure third bodies, it's just the third body conc:
        self.ref_pres_mod[:, :] = self.ref_thd[:, :]
        #now find the Pr poly
        Pr_poly = 1. / (1. + self.ref_Pr)
        #and multiply the falloff (i.e. non-chem activated terms)
        pure_fall_inds = np.array([i for i, x in enumerate(gas.reactions())
            if isinstance(x, ct.FalloffReaction) and not isinstance(x, ct.ChemicallyActivatedReaction)])
        pure_fall_inds = np.where(np.in1d(self.fall_inds, pure_fall_inds))[0]
        Pr_poly[:, pure_fall_inds] *= self.ref_Pr[:, pure_fall_inds]
        #finally find the product of the Pr poly and the falloff blending term
        Fall_pres_mod = Pr_poly * self.ref_Fall
        #and replace in the pressure mod
        replace_inds = np.where(np.in1d(self.thd_inds, self.fall_inds))[0]
        self.ref_pres_mod[:, replace_inds] = Fall_pres_mod[:, :]


class TestClass(unittest.TestCase):
    #global setup var
    _is_setup = False
    _store = None
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
            #load equations
            conp_vars, conp_eqs = load_equations(True)
            conv_vars, conv_eqs = load_equations(False)
            self.dirpath = os.path.dirname(os.path.realpath(__file__))
            gasname = os.path.join(self.dirpath, 'test.cti')
            if 'GAS' in os.environ:
                gasname = os.environ['GAS']
            #load the gas
            gas = ct.Solution(gasname)
            #the mechanism
            elems, specs, reacs = read_mech_ct(gasname)
            self.store = storage(conp_vars, conp_eqs, conv_vars,
                conv_eqs, gas, specs, reacs)
            self.is_setup = True
