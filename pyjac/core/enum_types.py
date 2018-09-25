from enum import Enum, IntEnum


class reaction_type(Enum):
    """
    The reaction type
    """
    elementary = 1
    thd = 2
    fall = 3
    chem = 4
    plog = 5
    cheb = 6

    def __int__(self):
        return self.value


class thd_body_type(Enum):
    """
    The form of the third body concentration modification
    """
    none = 0
    mix = 1
    species = 2
    unity = 3

    def __int__(self):
        return self.value

    def __long__(self):
        return self.value


class falloff_form(Enum):
    """
    The form of the falloff reaction type
    """
    none = 0
    lind = 1
    troe = 2
    sri = 3

    def __int__(self):
        return self.value


class reversible_type(Enum):
    """
    Whether the reaction is reversible or not
    """
    non_reversible = 1
    explicit = 2
    non_explicit = 3

    def __int__(self):
        return self.value


class reaction_sorting(Enum):
    """
    The reaction sorting scheme
    """
    none = 0,
    simd = 1


class RateSpecialization(IntEnum):
    """
    The form of reaction rate specialization, see
    :func:`pyjac.core.rate_subs.assign_rates`
    """
    fixed = 0,
    hybrid = 1,
    full = 2


class JacobianType(IntEnum):
    """
    The Jacobian type to be constructed.

    - An exact Jacobian has no approximations for reactions including the last
      species,
    - An approximate Jacobian ignores the derivatives of these reactions from
      species not directly involved (i.e. fwd/rev stoich coeff == 0, and not a third
      body species) while in a reaction including the last species
    - A finite differnce Jacobian is constructed from finite differences of the
      species rate kernel
    """
    exact = 0,
    approximate = 1,
    finite_difference = 2

    # TODO - provide an "approximate" FD?


class JacobianFormat(IntEnum):
    """
    The Jacobian format to use, full or sparse.

    A full Jacobian will include all zeros, while a sparse Jacobian will use either
    a Compressed Row/Column storage based format depending on the data-order ('C'
    and 'F' respectively)
    """
    full = 0,
    sparse = 1


class FiniteDifferenceMode(IntEnum):
    """
    The mode of finite differences--forwards, backwards or central--used to create
    the finite difference Jacobian
    """
    forward = 0,
    central = 1,
    backward = 2


class KernelType(Enum):
    """
    The kernel type being generated.
    """
    chem_utils = 1,
    species_rates = 2,
    jacobian = 3
    dummy = 4

    def __int__(self):
        return self.value


class DriverType(Enum):
    """
    The type of kernel driver to generate for evaluation of the kernel over many
    initial conditions
    """
    queue = 0,
    lockstep = 1


class DeviceMemoryType(Enum):
    """
    The type of memory to use for platforms with separate device memory pools
    (e.g., CUDA, OpenCL)
    """
    pinned = 0,
    mapped = 1

    def __int__(self):
        return self.value
