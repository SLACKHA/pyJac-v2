# -*- coding: utf-8 -*-
"""Contains various utility classes for creating update instructions
(e.g., handling updating rates of progress w/ reverse ROP, pressure mod term,
etc.)
"""

from __future__ import division

import inspect
import logging
import re
from string import Template
from functools import wraps

import loopy as lp
from loopy.types import AtomicType
from pytools import UniqueNameGenerator
import numpy as np
import six
from pytools import ImmutableRecord

from pyjac.core.array_creator import var_name, jac_creator, kint_type
from pyjac.loopy_utils import preambles_and_manglers as lp_pregen
from pyjac import utils


def use_atomics(loopy_opts):
    """
    Convenience method to detect whether atomic should be for double precision
    floating point operations.

    Useful in that we need to apply atomic modifiers to some instructions,
    but _not_ the sequential specializer

    Parameters
    ----------
    loopy_opts: :class:`loopy_utils.loopy_opts`
        The loopy options used to create this kernel.

    Returns
    -------
    use_atomics: bool
        Whether an atomic specializer would be returned by
        :meth:`get_deep_specializer`
    """

    return loopy_opts.depth and loopy_opts.use_atomic_doubles


def get_barrier(loopy_opts, local_memory=True, **loopy_kwds):
    """
    Returns the correct barrier type depending on the vectorization type / presence
    of atomics

    Parameters
    ----------
    loopy_opts: :class:`loopy_utils.loopy_opts`
        The loopy options used to create this kernel.
    local_memory: bool [True]
        If true, this barrier will be used for memory in the "local" address spaces.
        Only applicable to OpenCL
    loopy_kwds: dict
        Any other loopy keywords to put in the instruction options

    Returns
    -------
    barrier: str
        The built barrier instruction
    """

    mem_kind = ''
    barrier_kind = 'nop'
    if use_atomics(loopy_opts):
        mem_kind = 'local' if local_memory else 'global'
        barrier_kind = 'lbarrier'
        loopy_kwds['mem_kind'] = mem_kind

    return '...' + barrier_kind + '{' + ', '.join([
        '{}={}'.format(k, v) for k, v in six.iteritems(loopy_kwds)]) + '}'


def get_deep_specializer(loopy_opts, atomic_ids=[], split_ids=[], init_ids=[],
                         atomic=True, is_write_race=True,
                         split_size=None, **kwargs):
    """
    Returns a deep specializer to enable deep vectorization using either
    atomic updates or a sequential (single-lane/thread) "dummy" deep
    vectorizor (for implementations w/o 64-bit atomic instructions, e.g.
    intel's opencl)

    Parameters
    ----------
    loopy_opts: :class:`loopy_utils.loopy_opts`
        The loopy options used to create this kernel.  Determines the type
        of deep specializer to return
    atomic_ids: list of str
        A list of instruction id-names that require atomic updates for
        deep vectorization
    split_ids: list of str
        For instructions where (e.g.) a sum starts with a constant term,
        the easiest way to handle it is to split it over all the threads /
        lanes and have them contribute equally to the final sum.
        These instructions ids should be passed as split_ids
    split_size: int
        The size of the loop that is being split -- this is important in the case
        that the loop is smaller than the vector width, we must divide by only
        the number of vector lanes contributing
    init_ids: list of str
        List of instructions that initialize atomic variables
    atomic: bool [True]
        Use atomics for double precision floating point operations if available
    is_write_race: bool [True]
        If False, this kernel is guarenteed not to have a write race by nature
        of it's access pattern. Hence we only need to return a
        :class:`write_race_silencer` for non-atomic configurations
    use_atomics: bool [None]
        If supplied, override the results of :func:`use_atomics`

    Returns
    -------
    can_vectorize: bool [True]
        Whether the resulting kernel can be properly vectorized or not
    vectorization_specializer: :class:`Callable`(:class:`loopy.LoopKernel`)
        A function that takes as an argument the constructed kernel, and
        modifies it so as to enable the required vectorization
    """

    if not loopy_opts.depth:
        # no need to do anything
        return True, None

    have_atomics = kwargs.get('use_atomics', use_atomics(loopy_opts))

    if have_atomics and atomic:
        return True, atomic_deep_specialization(
            loopy_opts.depth, atomic_ids=atomic_ids,
            split_ids=split_ids, init_ids=init_ids,
            split_size=split_size)
    elif not is_write_race:
        return True, write_race_silencer(
            write_races=atomic_ids + split_ids + init_ids)
    else:
        return False, dummy_deep_specialization(
            write_races=atomic_ids + split_ids + init_ids)


class write_race_silencer(object):
    """
    Turns off warnings for loopy's detection of writing across a vectorized-iname
    as we handle his with either the :class:`dummy_deep_specialization` or
    the :class:`atomic_deep_specialization`
    """

    def __init__(self, write_races=[]):
        self.write_races = ['write_race({name})'.format(name=x) for x in write_races]

    def __call__(self, knl):
        return knl.copy(silenced_warnings=knl.silenced_warnings + self.write_races)


class within_inames_specializer(write_race_silencer):
    """
    A simple class designed to ensure all kernels are vectorizable
    by putting instructions that do not use the local hardware axis inside the
    correct loop.

    This should _not_ be used for anything but deep-vectorizations
    """
    def __init__(self, var_name=var_name, **kwargs):
        self.var_name = var_name
        super(within_inames_specializer, self).__init__(**kwargs)

    def __call__(self, knl):
        # get resulting tags
        in_tag = '{}_inner'.format(self.var_name)
        for insn in knl.instructions:
            if not insn.within_inames & frozenset([in_tag]):
                # add a fake dependency on the vectorized iname
                insn.within_inames |= frozenset([in_tag])

        return super(within_inames_specializer, self).__call__(
            knl.copy(instructions=knl.instructions[:]))


class atomic_deep_specialization(within_inames_specializer):
    """
    A class that turns write race instructions to atomics to enable deep
    vectorization

    vec_width: int
        The vector width to split over
    atomic_ids : list of str
        A list of instruction id-names that require atomic updates for
        deep vectorization
    split_ids : list of str
        For instructions where (e.g.) a sum starts with a constant term,
        the easiest way to handle it is to split it over all the threads /
        lanes and have them contribute equally to the final sum.
        These instructions ids should be passed as split_ids
    init_ids: list of str
        Lit of instruction id-names that require atomic inits for deep vectorization
    """

    def __init__(self, vec_width, atomic_ids=[], split_ids=[], init_ids=[],
                 split_size=None):
        def _listify(x):
            if not isinstance(x, list):
                return [x]
            return x
        # set parameters
        self.vec_width = vec_width
        self.atomic_ids = _listify(atomic_ids)[:]
        self.split_ids = _listify(split_ids)[:]
        self.init_ids = _listify(init_ids)[:]
        if self.split_ids:
            assert bool(split_size)
        self.split_size = split_size

        # and parent constructor
        super(atomic_deep_specialization, self).__init__(
            write_races=atomic_ids + split_ids + init_ids)

    def __call__(self, knl):
        insns = knl.instructions[:]
        data = knl.args[:]

        # we generally need to infer the dtypes for temporary variables at this stage
        # in case any of them are atomic
        from loopy.type_inference import infer_unknown_types
        from loopy.types import to_loopy_type
        from pymbolic.primitives import Sum
        knl = infer_unknown_types(knl, expect_completion=True)
        temps = knl.temporary_variables.copy()

        def _check_atomic_data(insn):
            # get the kernel arg written by this insn
            written = insn.assignee_var_names()[0]
            ind = next(
                (i for i, d in enumerate(data) if d.name == written), None)
            # make sure the dtype is atomic, if not update it
            if ind is not None and not isinstance(data[ind].dtype, AtomicType):
                assert data[ind].dtype is not None, (
                    "Change of dtype to atomic doesn't work if base dype is not"
                    " populated")
                data[ind] = data[ind].copy(for_atomic=True)
            elif ind is None:
                assert written in temps, (
                    'Cannot find written atomic variable: {}'.format(written))
                if not isinstance(temps[written].dtype, AtomicType):
                    temps[written] = temps[written].copy(dtype=to_loopy_type(
                        temps[written].dtype, for_atomic=True))
            return written

        for insn_ind, insn in enumerate(insns):
            if insn.id in self.atomic_ids:
                written = _check_atomic_data(insn)
                # and force the insn to an atomic update
                insns[insn_ind] = insn.copy(
                    atomicity=(lp.AtomicUpdate(written),))
            elif insn.id in self.init_ids:
                written = _check_atomic_data(insn)
                # setup an atomic init
                insns[insn_ind] = insn.copy(
                    atomicity=(lp.AtomicInit(written),))
            elif insn.id in self.split_ids:
                if isinstance(insn.expression, Sum) and \
                        insn.assignee in insn.expression.children:
                    written = _check_atomic_data(insn)
                    # get children that are not the assignee and re-sum
                    others = Sum(tuple(
                        x for x in insn.expression.children if x != insn.assignee))
                    # finally implement the split as a += sum(others) / vec_width
                    div_size = np.minimum(self.vec_width, self.split_size)
                    insns[insn_ind] = insn.copy(
                        expression=insn.assignee + others / div_size,
                        atomicity=(lp.AtomicUpdate(written),))
                else:
                    # otherwise can simply divide
                    div_size = np.minimum(self.vec_width, self.split_size)
                    insns[insn_ind] = insn.copy(
                        expression=insn.expression / div_size)

        # now force all instructions into inner loop
        return super(atomic_deep_specialization, self).__call__(
            knl.copy(instructions=insns, args=data, temporary_variables=temps))


class dummy_deep_specialization(within_inames_specializer):
    """
    A reusable-class to enable serialized deep vectorizations (i.e. reductions
    on a single OpenCL lane)
    """

    def __call__(self, knl):
        # do a dummy split
        knl = lp.split_iname(knl, self.var_name, 1, inner_tag='l.0')
        # now call the base to force all instructions inside the inner loop
        return super(dummy_deep_specialization, self).__call__(knl)


class PreambleMangler(ImmutableRecord):
    """
    An abstract class that can return it's own preambles and manglers
    """

    def _manglers(self):
        raise NotImplementedError

    def _preambles(self):
        raise NotImplementedError

    @property
    def preambles(self):
        return self._preambles()

    @property
    def manglers(self):
        return self._manglers()


class Guard(PreambleMangler):
    """
    A helper class to pass to a :class:`PrecomputedInstructions` that specifies
    min / max ranges for the variable in question

    Attributes
    ----------
    loopy_opts: :class:`LoopyOptions`
        The options to create this guard for
    min: float
        If specified, the minimum range of this guarded variable
    max: float
        If specified, the maximum range of this guarded variable
    """

    def __init__(self, loopy_opts, minv=None, maxv=None):
        self.min = minv
        self.max = maxv
        self.auto_diff = loopy_opts.auto_diff

    def __operation__(self, value):
        """
        An overridable operation, such that we may apply log's, exponentials, etc.
        """
        return value

    def __call__(self, varname):
        template = '${varname}'
        if self.min is not None and not self.auto_diff:
            template = 'fmax(${min}, ' + template + ')'
        if self.max is not None and not self.auto_diff:
            template = 'fmin(${max}, ' + template + ')'
        template = self.__operation__(template)
        return Template(template).safe_substitute(
            varname=varname, min=self.min, max=self.max)

    def _manglers(self):
        manglers = []
        if self.min:
            manglers += [lp_pregen.fmax()]
        if self.max:
            manglers += [lp_pregen.fmin()]

        return manglers

    def _preambles(self):
        return []


class NonzeroGuard(Guard):
    """
    A (potentially) sign-aware guard against non-zero numbers, may be employed
    on it's own, or as a component of another guard

    Attributes
    ----------
    is_positive: bool [None]
        If True, bound the value to be >= #small
        If False, bound the value to be <= -#small
        If None, use a sign aware context to determine the correct value
    """

    def __init__(self, loopy_opts, is_positive=None, limit=utils.small):
        if is_positive:
            super(NonzeroGuard, self).__init__(loopy_opts, minv=limit)
        elif is_positive is False:
            super(NonzeroGuard, self).__init__(loopy_opts, maxv=-limit)
        else:
            super(NonzeroGuard, self).__init__(loopy_opts)
        self.is_positive = is_positive
        self.limiter = lp_pregen.signaware_limiter_PreambleGen(
            loopy_opts.lang, limit=limit, vector=loopy_opts.vector_width)

    def __operation__(self, value):
        if self.is_positive is None:
            # need a sign-aware detection
            return '{}({})'.format(self.limiter.name, value)
        else:
            return super(NonzeroGuard, self).__operation__(value)

    def _manglers(self):
        manglers = []
        if self.is_positive is None:
            manglers += [self.limiter.func_mangler()]
        return manglers + super(Guard, self).manglers

    def _preambles(self):
        preambles = []
        if self.is_positive is None:
            preambles += [self.limiter]
        return preambles + super(Guard, self).preambles


class GuardedExp(Guard):
    """
    Take the guarded exponent of a value, i.e.:

        exp(fmin(exp_max, value))

    Attributes
    ----------
    maxv: float
        The maximum allowed exponential value

    """

    def __init__(self, loopy_opts, maxv=utils.exp_max, exptype='exp({val})'):
        self.exptype = exptype
        super(GuardedExp, self).__init__(loopy_opts, maxv=maxv)

    def __operation__(self, value):
        return self.exptype.format(val=value)


class GuardedLog(Guard):
    """
    Take the guarded logarithmn of a value, i.e.:

        log(fmax(1e-300d, value))

    Attributes
    ----------
    minv: float
        The minimum allowed logarithmic value

    """

    def __init__(self, loopy_opts, minv=utils.small, logtype='log({val})'):
        self.logtype = logtype
        super(GuardedLog, self).__init__(loopy_opts, minv=minv)

    def __operation__(self, value):
        return self.logtype.format(val=value)


class PowerFunction(PreambleMangler):
    """
    A simple wrapper that contains the name of a power function for a given language
    as well as any options
    """

    def __init__(self, loopy_opts, name, guard_nonzero=False):
        self.name = name
        self.lang = loopy_opts.lang
        self.vector_width = loopy_opts.vector_width
        self.is_simd = loopy_opts.is_simd
        self.guard_nonzero = guard_nonzero
        self.loopy_opts = loopy_opts

    def guard(self, value=None):
        g = Guard(self.loopy_opts, minv=utils.small)
        if value is not None:
            return g(value)
        return g

    def __call__(self, base, power):
        template = '{func}({base}, {power})'
        if self.guard_nonzero:
            base = self.guard(base)
        return template.format(func=self.name, base=base, power=power)

    def _manglers(self):
        manglers = []
        if self.guard_nonzero:
            manglers += self.guard().manglers
        if self.name in ['fast_powi', 'fast_powiv']:
            # skip, handled as preamble
            pass
        elif self.lang == 'opencl' and 'fast' not in self.name:
            # opencl only
            mangler_type = next((
                mangler for mangler in [lp_pregen.pown, lp_pregen.powf,
                                        lp_pregen.powr]
                if mangler().name == self.name), None)
            if mangler_type is None:
                raise Exception('Unknown OpenCL power function {}'.format(
                    self.name))
            # 1) float and short integer
            manglers.append(mangler_type())
            # 2) float and long integer
            if self.name == 'pown':
                manglers.append(mangler_type(arg_dtypes=(np.float64, np.int64)))
            if self.is_simd:
                from loopy.target.opencl import vec
                vfloat = vec.types[np.dtype(np.float64),
                                   self.vector_width]
                vlong = vec.types[np.dtype(np.int64), self.vector_width]
                vint = vec.types[np.dtype(np.int32), self.vector_width]
                # 3) vector float and short integers
                # note: return type must be non-vector form (this will converted
                # by loopy in privatize)
                if self.name == 'pown':
                    manglers.append(mangler_type(arg_dtypes=(vfloat, np.int32),
                                                 result_dtypes=np.float64))
                    manglers.append(mangler_type(arg_dtypes=(vfloat, vint),
                                                 result_dtypes=np.float64))
                    # 4) vector float and long integers
                    manglers.append(mangler_type(arg_dtypes=(vfloat, np.int64),
                                                 result_dtypes=np.float64))
                    manglers.append(mangler_type(arg_dtypes=(vfloat, vlong),
                                                 result_dtypes=np.float64))
        else:
            manglers += [lp_pregen.powf()]

        return manglers

    def _preambles(self):
        preambles = []
        if self.guard_nonzero:
            preambles += self.guard().preambles
        if 'fast_powi' == self.name:
            preambles += [lp_pregen.fastpowi_PreambleGen(
                self.lang, kint_type)]
        elif 'fast_powiv' == self.name:
            preambles += [lp_pregen.fastpowiv_PreambleGen(
                self.lang, kint_type, self.vector_width)]

        return preambles


def power_function(loopy_opts, is_integer_power=False, is_positive_power=False,
                   guard_nonzero=False, is_vector=False):
    """
    Returns the best power function to use for a given :param:`lang` and
    choice of :param:`is_integer_power` / :param:`is_positive_power` and
    the :param:`is_vector` status of the instruction in question
    """

    # 11/20/18 -> default to our unrolled power functions, when available
    if loopy_opts.lang == 'opencl' and is_positive_power and not is_integer_power:
        # opencl positive power function -- no need for guard
        return PowerFunction(loopy_opts, 'powr')
    elif is_integer_power:
        # 10/01/18 -- don't use OpenCL's pown -> VERY SLOW on intel
        # instead use internal integer power function
        if is_vector:
            return PowerFunction(loopy_opts, 'fast_powiv',
                                 guard_nonzero=guard_nonzero)
        else:
            return PowerFunction(loopy_opts, 'fast_powi',
                                 guard_nonzero=guard_nonzero)
    else:
        # use default
        return PowerFunction(loopy_opts, 'pow', guard_nonzero=guard_nonzero)


class PrecomputedInstructions(PreambleMangler):
    def __init__(self, loopy_opts, basename='precompute'):
        self.namer = UniqueNameGenerator()
        self.basename = basename
        self._mang = []
        self.loopy_opts = loopy_opts

    def __call__(self, result_name, var_str, INSN_KEY, guard=True):
        """
        Simple helper method to return a number of precomputes based off the passed
        instruction key

        Parameters
        ----------
        result_name : str
            The loopy temporary variable name to store in
        var_str : str
            The stringified representation of the variable to construct
        key : ['INV', 'LOG', 'VAL']
            The transform / value to precompute
        guard: :class:`Guard` [None]
            If specified, use a guard to avoid SigFPE's on the computed value

        Returns
        -------
        precompute : str
            A loopy instruction in the form:
                '<>result_name = fn(var_str)'
        """
        if guard:
            _guard = None
            if INSN_KEY == 'LOG':
                _guard = Guard(self.loopy_opts, minv=utils.small)
            elif INSN_KEY == 'LOG10':
                _guard = Guard(self.loopy_opts, minv=utils.small)
            if _guard:
                var_str = _guard(var_str)
                self._mang.extend(_guard.manglers)

        default_preinstructs = {'INV': '1 / {}'.format(var_str),
                                'LOG': 'log({})'.format(var_str),
                                'VAL': '{}'.format(var_str),
                                'LOG10': 'log10({})'.format(var_str)}

        return Template("<>${result} = ${value} {id=${id}}").safe_substitute(
            result=result_name,
            value=default_preinstructs[INSN_KEY],
            id=self.reserve_name())

    def reserve_name(self):
        """
        Let custom PrecomputedInstructions reserve their own name from the same
        object to avoid comflicts
        """
        return self.namer(self.basename)

    def _manglers(self):
        return self._mang

    def _preambles(self):
        return []


def get_update_instruction(mapstore, mask_arr, base_update_insn):
    """
    Handles updating a value by a possibly specified (masked value),
    for example this can be used to generate an instruction (or set thereof)
    to update the net rate of progress by the reverse or pressure modification
    term

    Parameters
    ----------
    mapstore: :class:`array_creator.MapStore`
        The base mapstore used in creation of this kernel
    mask_arr: :class:`array_creator.creator`
        The array to use as a mask to determine whether the :param:`base_value`
        should be updated
    base_update_insn: str
        The update instruction to use as a base.  This may be surrounded by
        and if statement or possibly discarded altogether depending on the
        :param:`mask_arr` and :param:`mapstore`

    Returns
    -------
    update_insn: str
        The update instruction string(s), possibly empty if no update is
        required
    """

    logger = logging.getLogger(__name__)
    if not mapstore.is_finalized:
        _, _, line_number, function_name, _, _ = inspect.stack()[1]
        logger.warn('Call to get_update_instruction() from {0}:{1} '
                    'used non-finalized mapstore, finalizing now...'.format(
                         function_name, line_number))

        mapstore.finalize()

    # empty mask
    if not mask_arr:
        # get id for noop anchor
        idx = re.search(r'id=([^,}]+)', base_update_insn)
        return '... nop {{id={id}}}'.format(id=idx.group(1))

    # ensure mask array in domains
    assert mask_arr in mapstore.domain_to_nodes, (
        'Cannot create update instruction - mask array '
        '{} not in mapstore domains'.format(
            mask_arr.name))

    # check to see if there are any empty mask entries
    mask = mapstore.domain_to_nodes[mask_arr]
    mask = mask if mask.parent in mapstore.transformed_domains \
        else None
    if mask:
        mask_iname = mask.iname
        # if so, return a guarded update insn
        return Template(
            """
    if ${mask_iname} >= 0
        ${base_update_insn}
    end
    """).safe_substitute(**locals())

    # else return the base update insn
    return base_update_insn


def wrap_instruction_on_condition(insn, condition, wrapper):
    """
    Utility function to wrap the :param:`insn` in the supplied :param:`wrapper`
    if :param:`condition` is True

    Parameters
    ----------
    insn: str
        The instruction to execute
    condition: bool or Callable
        If true, :param:`insn` will be wrapped in an if statement given by
        :param:`wrapper`
    wrapper: str
        The if statement condition to wrap :param:`insn` in if not :param:`condition`

    Returns
    -------
    If condition:
        `if wrapper
            insn
         end
        `
    else:
        insn
    """

    if condition:
        return Template("""
            if ${wrapper}
                ${insn}
            end
        """).safe_substitute(locals())
    return insn


def place_in_vectorization_loop(loopy_opts, insn, namer, vectorize=True):
    """
    Places a instruction (or set thereof) inside a "loop" who's sole purpose is
    to enable top-level vectorization -- taking into account whether atomics can be
    used.

    Note
    ----
    Non-deep vectorizations will not be affected

    Parameters
    ----------
    loopy_opts: :class:`loopy_options`
        The loopy options indicating whether this is a deep vectorization, and the
        availabilty of atomics
    insn: str
        The instructions to wrap
    namer: :class:`pytools.UniqueNameGenerator`
        The namer to use to ensure unique inames
    vectorize: [True]
        If true, this is intended to be a vectorized instruction.
        If false, this is a "dummy" vectorization
    needs_atomic: [False]
        If True, requires

    Returns
    -------
    wrapped_insn: str
        The wrapped instructions
    iname_spec: tuple of (str, str)
        The iname specification to add to the extra_inames.  If this is None, no
        iname need be added
    """

    if not loopy_opts.depth:
        return insn, None
    # otherwise
    if vectorize:
        iname = namer('full_vec')
        spec = '0 <= {} < {}'.format(iname, loopy_opts.depth)
    else:
        iname = namer('fake_vec')
        spec = '0 <= {} < 1'.format(iname)

    return Template("""
    for ${iname}
        ${insn}
    end
    """).safe_substitute(iname=iname, insn=insn), (iname, spec)


def with_conditional_jacobian(func):
    """
    A function wrapper that makes available the :func:`_conditional_jacobian`
    instruction to seemlessly enable checking for the existance of Jacobian entries
    and / or precomputing expensive Jacobian lookups
    """

    # define the jacobian wrapper logic
    def _conditional_jacobian(mapstore, jac, *jac_inds, **kwargs):
        """
        A method to handle updates / setting of the Jacobian for sparse and
        non-sparse matricies.  This helps ease the burden of checking to see if an
        entry is actually present in the Jacobian or not, and will automatically
        guard against out-of-bounds accesses to the sparse Jacobian (or just to the
        wrong entry)

        Parameters
        ----------
        loopy_opts: :class:`loopy_options`
            The loopy options indicating whether this jacobian is sparse or not
        mapstore: :class:`MapStore`
            The mapstore use in creation of the jacobian
        jac: :class:`creator`
            The Jacobian creator from the mapstore's :class:`NameStore`
        jac_inds: tuple of int/str
            The Jacobian indicies to use, of length 3
        insn: str ['']
            The update or Jacobian setting instruction to execute.
                -A template key form of the ${jac_str} is expected to substitute the
                 resulting Jacobian entry into
                -Additionaly, the instruction must have the key ${deps} in order
                 to insert any relevant pre-computed index depencency
            Ignored if not supplied
        deps: str ['']
            A colon separated list of dependencies for the insn.  Ignored if insn
            not supplied.  Taken as '' if not supplied
        index_insn: bool [True]
            If true, use an index instruction to precompute the Jacobian index
        entry_exists: bool [False]
            If True, this Jacobian entry exists so do not wrap in a conditonal
        return_arg: bool [True]
            If True, return the created :loopy:`GlobalArg`
        warn: bool [True]
            If True, warn if an indirect access will be made w/o supplying
            :param:`index_insn`
        **kwargs: dict
            Any other arguements will be passed to the :func:`mapstore.apply_maps`
            call
        Returns
        -------
        insn: str
            The jacobian lookup / access instructions
        """

        # check jacobian type
        logger = logging.getLogger(__name__)
        precompute = False
        is_sparse = False
        if isinstance(jac, jac_creator):
            is_sparse = jac.is_sparse
            precompute = True

        # Get defaults out of kwargs
        entry_exists = kwargs.pop('entry_exists', False)
        return_arg = kwargs.pop('return_arg', True)
        insn = kwargs.pop('insn', '')
        index_insn = kwargs.pop('index_insn', is_sparse) and insn != ''
        deps = kwargs.pop('deps', '')
        deps = deps.split(':')
        warn = kwargs.pop('warn', True)

        created_index = _conditional_jacobian.created_index
        if not index_insn and is_sparse:
            method = logger.warn if warn else logger.debug
            method('Using a sparse jacobian without precomputing the index'
                   ' will result in extra indirect lookups.')

        # if we want to precompute the index, do so
        if precompute:
            replace_ind, computed_ind, offset, _ = mapstore.apply_maps(
                jac, *jac_inds, plain_index=True, **kwargs)
            if index_insn:
                # get the index
                existing = sorted(_conditional_jacobian.id_namer.existing_names)
                if existing or deps:
                    dep_str = 'dep={}'.format(':'.join(deps + existing))
                else:
                    dep_str = ''

                name = '{}jac_index'.format('sparse_' if is_sparse else '')
                name = _conditional_jacobian.id_namer(name)
                index_insn = Template(
                    '${creation}jac_index = ${index_str} {id=${name}, ${dep_str}}'
                    ).safe_substitute(
                        creation='<> ' if not created_index else '',
                        index_str=computed_ind,
                        name=name,
                        dep_str=dep_str)
                # add dependency to all before me (and just added)
                # so that we don't get out of order
                deps += [name]
                if is_sparse:
                    # redefine the jac indicies
                    jac_inds = (jac_inds[0],) + ('jac_index',)
                    # and add conditional
                    conditional = 'jac_index >= {}'.format(offset)
                else:
                    # if not sparse, we don't replace the jacobian indicies at all
                    # this is simply a check to make sure this is present
                    conditional = 'jac_index >= 0'
                # we've now created the temporary
                _conditional_jacobian.created_index = True
            else:
                # otherwise we're conditional on the lookup
                index_insn = ''
                if is_sparse:
                    conditional = '{} >= {}'.format(computed_ind, offset)
                else:
                    conditional = '{} >= 0'.format(computed_ind)

        if precompute and insn:
            # need to wrap the instruction
            insn = wrap_instruction_on_condition(
                insn, not entry_exists, conditional)

        # and finally return the insn
        mykwargs = kwargs.copy()
        if precompute:
            # we can skip the lookup here _if_ we've precomputed it (and stored
            # in a temp) _or_ it's a full jacobian
            mykwargs.update({'ignore_lookups': index_insn != ''
                             or not is_sparse})
        # get jac_str
        jac_lp, jac_str = mapstore.apply_maps(
            jac, *jac_inds, **mykwargs)

        # find return value
        if insn:
            retv = Template(insn).safe_substitute(
                jac_str=jac_str, deps=':'.join(deps))
            if index_insn:
                retv = Template("""${index}
${retv}
""").safe_substitute(index=re.match(r"^\s*", retv).group() + index_insn,
                     retv=retv)
        else:
            retv = jac_str

        if return_arg:
            return jac_lp, retv
        return retv

    @wraps(func)
    def wrapper(*args, **kwargs):
        # pass into the function as a kwarg
        _conditional_jacobian.created_index = False
        _conditional_jacobian.id_namer = UniqueNameGenerator()
        return func(*args, jac_create=_conditional_jacobian, **kwargs)
    return wrapper
