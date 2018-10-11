"""Module containing element dict, species and reaction classes, and constants.

"""

# Python 2 compatibility
from __future__ import division

# Standard libraries
import math
import logging
import numpy as np
from pyjac.core.enum_types import reaction_type, thd_body_type, falloff_form, \
    reversible_type
from pyjac.utils import is_integer

__all__ = ['RU', 'RUC', 'RU_JOUL', 'PA', 'get_elem_wt',
           'ReacInfo', 'SpecInfo', 'calc_spec_smh']

# universal gas constants, SI units
RU = 8314.4621  # J/(kmole * K)
RU_JOUL = 8.3144621
RUC = (RU / 4.18400)  # cal/(mole * K)

# Avogadro's number
AVAG = 6.0221367e23

# pressure of one standard atmosphere [Pa]
PA = 101325.0


def get_elem_wt():
    """Returns dict with built-in element names and atomic weights [kg/kmol].

    Attributes
    ----------
    None

    Returns
    -------
    elem_wt : dict
        Dictionary with element name keys and atomic weight [kg/kmol] values.
    """
    elem_wt = dict([
        ('h', 1.00794), ('he', 4.00260), ('li', 6.93900),
        ('be', 9.01220), ('b', 10.81100), ('c', 12.0110),
        ('n', 14.00674), ('o', 15.99940), ('f', 18.99840),
        ('ne', 20.18300), ('na', 22.98980), ('mg', 24.31200),
        ('al', 26.98150), ('si', 28.08600), ('p', 30.97380),
        ('s', 32.06400), ('cl', 35.45300), ('ar', 39.94800),
        ('k', 39.10200), ('ca', 40.08000), ('sc', 44.95600),
        ('ti', 47.90000), ('v', 50.94200), ('cr', 51.99600),
        ('mn', 54.93800), ('fe', 55.84700), ('co', 58.93320),
        ('ni', 58.71000), ('cu', 63.54000), ('zn', 65.37000),
        ('ga', 69.72000), ('ge', 72.59000), ('as', 74.92160),
        ('se', 78.96000), ('br', 79.90090), ('kr', 83.80000),
        ('rb', 85.47000), ('sr', 87.62000), ('y', 88.90500),
        ('zr', 91.22000), ('nb', 92.90600), ('mo', 95.94000),
        ('tc', 99.00000), ('ru', 101.07000), ('rh', 102.90500),
        ('pd', 106.40000), ('ag', 107.87000), ('cd', 112.40000),
        ('in', 114.82000), ('sn', 118.69000), ('sb', 121.75000),
        ('te', 127.60000), ('i', 126.90440), ('xe', 131.30000),
        ('cs', 132.90500), ('ba', 137.34000), ('la', 138.91000),
        ('ce', 140.12000), ('pr', 140.90700), ('nd', 144.24000),
        ('pm', 145.00000), ('sm', 150.35000), ('eu', 151.96000),
        ('gd', 157.25000), ('tb', 158.92400), ('dy', 162.50000),
        ('ho', 164.93000), ('er', 167.26000), ('tm', 168.93400),
        ('yb', 173.04000), ('lu', 174.99700), ('hf', 178.49000),
        ('ta', 180.94800), ('w', 183.85000), ('re', 186.20000),
        ('os', 190.20000), ('ir', 192.20000), ('pt', 195.09000),
        ('au', 196.96700), ('hg', 200.59000), ('tl', 204.37000),
        ('pb', 207.19000), ('bi', 208.98000), ('po', 210.00000),
        ('at', 210.00000), ('rn', 222.00000), ('fr', 223.00000),
        ('ra', 226.00000), ('ac', 227.00000), ('th', 232.03800),
        ('pa', 231.00000), ('u', 238.03000), ('np', 237.00000),
        ('pu', 242.00000), ('am', 243.00000), ('cm', 247.00000),
        ('bk', 249.00000), ('cf', 251.00000), ('es', 254.00000),
        ('fm', 253.00000), ('d', 2.01410), ('e', 5.48578e-4)
    ])
    return elem_wt


class ReacInfo(object):
    """Reaction class.

    Contains all information about a single reaction.

    Attributes
    ----------
    rev : bool
        True if reversible reaction, False if irreversible.
    reactants : list of str
        List of reactant species names.
    reac_nu : list of int/float
        List of reactant stoichiometric coefficients, either int or float.
    products : list of str
        List of product species names.
    prod_nu : list of int/float
        List of product stoichiometric coefficients, either int or float.
    A : float
        Arrhenius pre-exponential coefficient.
    b : float
        Arrhenius temperature exponent.
    E : float
        Arrhenius activation energy.
    rev_par : list of float, optional
        List of reverse Arrhenius coefficients (default empty).
    dup : bool, optional
        Duplicate reaction flag (default False).
    thd : bool, optional
        Third-body reaction flag (default False).
    thd_body : list of list of [str, float], optional
        List of third body names and efficiencies (default empty).
    pdep : bool, optional
        Pressure-dependence flag (default False).
    pdep_sp : str, optional
        Name of specific third-body or 'M' (default '').
    low : list of float, optional
        List of low-pressure-limit Arrhenius coefficients (default empty).
    high : list of float, optional
        List of high-pressure-limit Arrhenius coefficients (default empty).
    troe : bool, optional
        Troe pressure-dependence formulation flag (default False).
    troe_par : list of float, optional
        List of Troe formulation constants (default empty).
    sri : bool, optional
        SRI pressure-dependence formulation flag (default False).
    sri_par : list of float, optional
        List of SRI formulation constants (default empty).

    Notes
    -----
    `rev` does not require `rev_par`; if no explicit coefficients, the
    reverse reaction rate will be calculated through the equilibrium
    constant.
    Only one of [`low`,`high`] can be defined.
    If `troe` and `sri` are both False, then the Lindemann is assumed.

    """

    def __eq__(self, other):
        """
        Check for equality of reactions
        """

        try:
            # check type
            assert type(self) == type(other)
            # check reactants
            assert len(self.reac) == len(other.reac)
            for i in range(len(self.reac)):
                assert self.reac[i] in other.reac
                ind = other.reac.index(self.reac[i])
                assert self.reac_nu[i] == other.reac_nu[ind]
            # check products
            for i in range(len(self.prod)):
                assert self.prod[i] in other.prod
                ind = other.prod.index(self.prod[i])
                assert self.prod_nu[i] == other.prod_nu[ind]
            # check arrhenius
            assert np.isclose(self.A, other.A)
            assert np.isclose(self.b, other.b)
            assert np.isclose(self.E, other.E)
            # check rev
            assert self.rev == other.rev
            assert np.allclose(self.rev_par, other.rev_par)
            # check duplicate
            assert self.dup == other.dup
            # check third bodu
            assert self.thd_body == other.thd_body
            assert np.allclose(self.thd_body_eff, other.thd_body_eff)
            # check falloff
            assert self.pdep == other.pdep
            assert self.pdep_sp == other.pdep_sp
            assert np.allclose(self.low, other.low)
            assert np.allclose(self.high, other.high)
            # check troe
            assert self.troe == other.troe
            assert np.allclose(self.troe_par, other.troe_par)
            # check sri
            assert self.sri == other.sri

            def __optional_param_check(p1, p2, default):
                minsize = np.minimum(len(p1), len(p2))
                assert np.allclose(p1[:minsize], p2[:minsize])
                if len(p1) > minsize:
                    assert np.allclose(p1[minsize:], default)
                if len(p2) > minsize:
                    assert np.allclose(p2[minsize:], default)

            __optional_param_check(self.sri_par, other.sri_par, [1, 0])
            # check chebyshev
            assert self.cheb == other.cheb
            assert self.cheb_n_temp == other.cheb_n_temp
            assert self.cheb_n_pres == other.cheb_n_pres
            assert np.allclose(self.cheb_plim, other.cheb_plim)
            assert np.allclose(self.cheb_tlim, other.cheb_tlim)
            assert (self.cheb_par is None and other.cheb_par is None) or \
                np.allclose(self.cheb_par, other.cheb_par)
            # check plog
            assert self.plog == other.plog
            assert (self.plog_par is None and other.plog_par is None) or \
                np.allclose(self.plog_par, other.plog_par)
            return True
        except AssertionError:
            return False
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warn('Unknown exception occured in reaction equality testing,')
            logging.exception(e)
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __init__(self, rev, reactants, reac_nu, products, prod_nu, A, b, E):
        self.reac = reactants
        self.reac_nu = reac_nu
        self.prod = products
        self.prod_nu = prod_nu

        # Arrhenius coefficients
        # pre-exponential factor [m, kmol, s]
        self.A = A
        # Temperature exponent [-]
        self.b = b
        # Activation energy, stored as activation temperature [K]
        self.E = E

        # reversible reaction properties
        self.rev = rev
        self.rev_par = []  # reverse A, b, E

        # duplicate reaction
        self.dup = False

        # third-body efficiencies
        self.thd_body = False
        self.thd_body_eff = []  # in pairs with species and efficiency

        # pressure dependence
        self.pdep = False
        self.pdep_sp = ''
        self.low = []
        self.high = []

        self.troe = False
        self.troe_par = []

        self.sri = False
        self.sri_par = []

        # Parameters for pressure-dependent reaction parameterized by
        # bivariate Chebyshev polynomial in temperature and pressure.
        self.cheb = False
        # Number of temperature values over which fit computed.
        self.cheb_n_temp = 0
        # Number of pressure values over which fit computed.
        self.cheb_n_pres = 0
        # Pressure limits for Chebyshev fit [Pa]
        self.cheb_plim = [0.001 * PA, 100. * PA]
        # Temperature limits for Chebyshev fit [K]
        self.cheb_tlim = [300., 2500.]
        # 2D array of Chebyshev fit coefficients
        self.cheb_par = None

        # Parameters for pressure-dependent reaction parameterized by
        # logarithmically interpolating between Arrhenius rate expressions at
        # various pressures.
        self.plog = False
        # List of arrays with [pressure [Pa], A, b, E]
        self.plog_par = None
        # enums
        self.type = []

    def finalize(self, num_species):
        """

        Takes all the various options of the reaction, and turns it into
        :class:`reaction_types` enums

        Parameters
        ----------
        num_species : int
            Number of species in the mechanism

        """

        if self.rev:
            self.type.append(reversible_type.explicit if self.rev_par
                             else reversible_type.non_explicit)
        else:
            self.type.append(reversible_type.non_reversible)

        if self.thd_body:
            self.type.append(reaction_type.thd)
        elif self.pdep and self.low:
            self.type.append(reaction_type.fall)
        elif self.pdep and self.high:
            self.type.append(reaction_type.chem)
        elif self.plog:
            self.type.append(reaction_type.plog)
        elif self.cheb:
            self.type.append(reaction_type.cheb)
        else:
            self.type.append(reaction_type.elementary)

        if reaction_type.fall in self.type or \
                reaction_type.chem in self.type or\
                reaction_type.thd in self.type:

            # figure out the third body type
            if self.pdep_sp:  # single species
                self.type.append(thd_body_type.species)
            elif not self.thd_body_eff or \
                (len(self.thd_body_eff) == num_species  # check for all = 1
                 and all(thd[1] == 1.0 for thd in self.thd_body_eff)):
                self.type.append(thd_body_type.unity)
            else:  # mixture as third body
                self.type.append(thd_body_type.mix)
        else:
            self.type.append(thd_body_type.none)

        if reaction_type.fall in self.type or \
                reaction_type.chem in self.type:
            if self.troe:
                self.type.append(falloff_form.troe)
            elif self.sri:
                self.type.append(falloff_form.sri)
            else:
                self.type.append(falloff_form.lind)
        else:
            self.type.append(falloff_form.none)

        # cleanup Chemkin mechanisms that require PLOG / Chebysheb to have rate
        # parameters
        if self.plog or self.cheb:
            self.A = 0
            self.b = 0
            self.E = 0

        def __eqn_side(species, nu):
            estr = ''
            for (s, nu) in zip(*(species, nu)):
                if estr:
                    estr += ' + '

                nu_str = '{}'.format(int(nu) if is_integer(nu) else nu)
                estr += ('{} {}'.format(nu_str, s) if nu != 1 else s)

            if self.match([reaction_type.fall, reaction_type.chem]):
                if self.pdep_sp:
                    estr += ' (+{})'.format(self.pdep_sp)
                else:
                    estr += ' (+M)'
            elif self.match([reaction_type.thd]):
                estr += ' + M'
            return estr

        rxn_str = ''
        if self.match([reaction_type.fall]):
            rxn_str += 'FalloffReaction: '
        elif self.match([reaction_type.chem]):
            rxn_str += 'ChemicallyActivatedReaction: '
        elif self.match([reaction_type.thd]):
            rxn_str += 'ThreeBodyReaction: '
        elif self.match([reaction_type.plog]):
            rxn_str += 'PLOGReaction: '
        elif self.match([reaction_type.cheb]):
            rxn_str += 'ChebyshevReaction: '
        else:
            rxn_str += 'ElementaryReaction: '

        # finally create a rxn string
        rxn_str += __eqn_side(self.reac, self.reac_nu)
        if self.rev:
            rxn_str += ' <=> '
        else:
            rxn_str += ' => '
        rxn_str += __eqn_side(self.prod, self.prod_nu)
        self.rxn_str = rxn_str

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        assert self.type, 'Reaction must be finalized to stringify.'
        return self.rxn_str

    def get_type(self, reaction_enum_type):
        """
        Return all :class:`reaction_type` enums in our :attr:`types` that is
        and instance of the given :param:`reaction_enum_type` class, or None if
        not found.
        """
        return [x for x in self.type if isinstance(x, reaction_enum_type)]

    def match(self, reac_types):
        """
        Given a tuple of :class:`reaction_types` enums, for conditional equations
        this method returns true / false if the reaction falls under this equation

        Parameters
        ----------
        reac_types : tuple of `reaction_types`
            the conditions to check

        Notes
        -----
        The matching rules are as follows:

        Repeated `reaction_types` of the same type (e.g. `thd_body_type`)
            imply an OR; that is, this is a match if this reaction has
            any of the repeated reaction type

        A match is made if this reaction matches all `reaction types` given
            with the repeat rule given above

        An empty :param:`reac_types` will be matched by all reactions

        """

        if not reac_types:
            return True

        if not isinstance(reac_types, tuple):
            try:
                reac_types = tuple(reac_types)
            except TypeError:
                reac_types = (reac_types,)

        # get the types to a more managable form
        enum_types = set([type(rtype) for rtype in reac_types])
        enum_types = {etype: [x for x in reac_types if type(x) == etype]
                      for etype in enum_types}

        for rtype, enum in enum_types.items():
            if not any(x in self.type for x in enum):
                return False

        return True


class SpecInfo(object):
    """Species class.

    Contains all information about a single species.

    Attributes
    ----------
    name : str
        Name of species.
    elem : list of list of [str, float]
        Elemental composition in [element, number] pairs.
    mw : float
        Molecular weight.
    hi : list of float
        High-temperature range NASA thermodynamic coefficients.
    lo : list of float
        Low-temperature range NASA thermodynamic coefficients.
    Trange : list of float
        Temperatures defining ranges of thermodynamic polynomial fits
        (low, middle, high), default ([300, 1000, 5000]).

    """

    def __eq__(self, other):
        # check equality to other spec info
        try:
            # check elements
            assert type(self) == type(other)
            assert len(self.elem) == len(other.elem)
            for i in range(len(self.elem)):
                assert self.elem[i][0] == other.elem[i][0]
                assert self.elem[i][1] == other.elem[i][1]
            # check mw
            assert np.isclose(self.mw, other.mw)
            # check thermo params
            assert np.allclose(self.hi, other.hi)
            assert np.allclose(self.lo, other.lo)
            assert np.allclose(self.Trange, other.Trange)
            return True

        except AssertionError:
            return False
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warn('Unknown exception occured in species equality testing,')
            logging.exception(e)
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __init__(self, name):
        self.name = name

        # elemental composition
        self.elem = []
        # molecular weight [kg/kmol]
        self.mw = 0.0
        # high-temp range thermodynamic coefficients
        self.hi = np.zeros(7)
        # low-temp range thermodynamic coefficients
        self.lo = np.zeros(7)
        # temperature [K] range for thermodynamic coefficients
        self.Trange = [300.0, 1000.0, 5000.0]

    def finalize(self):
        """
        Cleans up the :class:`SpecInfo`, removing any unnecessary elements
        """

        # remove any zero elements
        self.elem = [x for x in self.elem if x[1] != 0]


def calc_spec_smh(T, specs):
    """Calculate standard-state entropies minus enthalpies for all species.

    Parameters
    ----------
    T : float
        Temperature of gas mixture.
    specs : list of SpecInfo
        List of species.

    Returns
    -------
    spec_smh : list of float
        List of species' standard-state entropies minus enthalpies.

    """

    spec_smh = []

    Tlog = math.log(T)
    T2 = T * T
    T3 = T2 * T
    T4 = T3 * T

    Thalf = T / 2.0
    T2 = T2 / 6.0
    T3 = T3 / 12.0
    T4 = T4 / 20.0

    for sp in specs:
        if T <= sp.Trange[1]:
            smh = (sp.lo[0] * (Tlog - 1.0) + sp.lo[1] * Thalf + sp.lo[2] *
                   T2 + sp.lo[3] * T3 + sp.lo[4] * T4 - (sp.lo[5] / T) +
                   sp.lo[6]
                   )
        else:
            smh = (sp.hi[0] * (Tlog - 1.0) + sp.hi[1] * Thalf + sp.hi[2] *
                   T2 + sp.hi[3] * T3 + sp.hi[4] * T4 - (sp.hi[5] / T) +
                   sp.hi[6]
                   )

        spec_smh.append(smh)

    return (spec_smh)
