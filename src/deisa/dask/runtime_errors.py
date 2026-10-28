# =============================================================================
# Copyright (C) 2026 Commissariat a l'energie atomique et aux energies alternatives (CEA)
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of CEA, nor the names of the contributors may be used
#   to endorse or promote products derived from this software without specific
#   prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# =============================================================================
"""
The v1 refusal taxonomy of the DEISA runtime reduction path.

A callback that uses the DEISA lazy collection runs against a closed operator surface: what the surface does not
support is REFUSED, never silently gathered. Every refusal in this module therefore carries three things, and each is
asserted by the test suite:

- the offending SYMBOL (the attribute, operator, reduction or keyword the callback used),
- WHY it cannot be served (the invariant it breaks),
- a REMEDY that names a working path -- always ``precompute=False``, which stays the legacy full-chunk scatter
  delivery mode (decision F3), so a refused callback is never left without an escape hatch.

Every class derives from :class:`~deisa.dask.precompute_errors.PrecomputeError`, the base of the precompute taxonomy,
so existing ``except PrecomputeError`` handlers keep catching refusals. The five classes that have a namesake in that
taxonomy ALSO subclass the namesake, so the two families are one ``isinstance`` graph: a handler written for the
legacy class catches the runtime refusal that replaced it. (The taxonomy lives in
:mod:`deisa.dask.precompute_errors`, not in the analyzer that used to define it: plan P4 deletes the analyzer, and
the class identity callers catch must outlive it.)

The raise sites are deliberately thin: build the exception through one of the module-level factory functions below
instead of hand-writing a message, so the wording (and therefore the tests over it) lives in exactly one place.
"""

from __future__ import annotations

from typing import Optional

import deisa.dask.precompute_errors as _precompute_errors

__all__ = [
    "MaterializationError",
    "NoPrecomputableReductionError",
    "NonAssociativeReductionError",
    "PrecomputeRuntimeError",
    "RawFieldReadError",
    "RuntimeRefusal",
    "UnsupportedReductionError",
    "ambiguous_truth",
    "cross_reduction",
    "ddof_unsupported",
    "dtype_out_unsupported",
    "iteration_materialization",
    "keepdims_unsupported",
    "mixed_root",
    "no_dispatch_transport",
    "no_precomputable_reduction",
    "non_associative",
    "numpy_materialization",
    "off_surface",
    "precompute_runtime_invariant",
    "raw_field_read",
]

# The remedy that must remain reachable from EVERY refusal: the legacy delivery mode ships the chunk itself, so it can
# serve any callback the closed surface refuses.
_REMEDY_PRECOMPUTE_OFF = (
    "register the callback with precompute=False to use the legacy full-chunk scatter delivery mode, which ships the "
    "chunk itself instead of a chunk-local reduction."
)

# The surface, quoted in the off-surface remedy so a refused callback knows what it may use instead.
_SURFACE_SUMMARY = (
    "the supported pointwise operators (+, -, *, /, //, %, **, comparisons, abs, neg, sqrt, exp, log, clip, where) on "
    "a single array/timestep root, and the supported reductions (sum, mean, var, std, min, max, prod, any, all, count)"
)


class RuntimeRefusal(_precompute_errors.PrecomputeError):
    """Base class of the v1 refusals: a message naming the offending symbol, plus a remedy.

    ``str(exception)`` is ``"<message> Remedy: <remedy>"`` and the two halves are also kept as the ``message`` and
    ``remedy`` attributes, so a caller can re-raise with its own remedy (or point at a closer alternative) without
    re-formatting the message.
    """

    remedy = _REMEDY_PRECOMPUTE_OFF

    def __init__(self, message: str, remedy: Optional[str] = None) -> None:
        self.message = str(message)
        if remedy is not None:
            self.remedy = remedy
        super().__init__(f"{self.message} Remedy: {self.remedy}")


class RawFieldReadError(RuntimeRefusal, _precompute_errors.RawFieldReadError):
    """The callback reads a registered array's raw data, which the reduction path never delivers.

    The precompute path ships per-bridge reduction partials only -- the chunk itself never crosses the network. A
    callback that consumes the raw data (computes the collection itself, indexes into it, plots it) asks for a value
    that does not exist on the analytics host. Raised at the FIRST compute boundary that resolves to a raw read, so
    the run fails fast on its first timestep instead of degrading into a logged error.
    """


class MaterializationError(RuntimeRefusal, _precompute_errors.MaterializationError):
    """A numpy conversion of a DEISA operand was attempted; that would gather the whole field.

    The hole this closes is silent: ``np.asarray(collection)`` leaves no graph artifact at all, so no analysis can see
    it, and it would quietly pull the entire registered array onto the analytics host. ``__array__`` and ``__iter__``
    on the collection and on every expression therefore refuse instead of returning data.
    """


class NonAssociativeReductionError(RuntimeRefusal):
    """A non-associative reduction was requested (argmax, argmin, median, quantile, percentile).

    These cannot be rebuilt from chunk-local partials: the answer depends on the GLOBAL ordering of the data, which no
    bridge can see. This class exists because the same call used to be refused as a raw field read, which named the
    wrong cause and pointed at a remedy that does not help.
    """

    remedy = (
        "use one of the associative reductions (sum, mean, var, std, min, max, prod, any, all, count), or register the "
        "callback with precompute=False to use the legacy full-chunk scatter delivery mode."
    )


class UnsupportedReductionError(RuntimeRefusal, _precompute_errors.UnsupportedReductionError):
    """The expression leaves the closed DEISA operator surface.

    Covers cross-array and mixed-timestep roots, a reduction used as an operand of another expression, slicing,
    rechunking, keeping dimensions, and any operator/attribute the surface does not implement.
    """

    remedy = f"rewrite the callback with {_SURFACE_SUMMARY}, or {_REMEDY_PRECOMPUTE_OFF}"


class NoPrecomputableReductionError(RuntimeRefusal, _precompute_errors.NoPrecomputableReductionError):
    """The callback contains no reduction the surface can dispatch.

    A callback that only builds pointwise expressions (or only reads metadata) asks for nothing a bridge could compute
    chunk-locally, so there is nothing to plan.
    """


class PrecomputeRuntimeError(RuntimeRefusal, _precompute_errors.PrecomputeRuntimeError):
    """A runtime invariant of the reduction path was violated (bridge side or combine side).

    Raised when the metadata, the composed request and the delivered partials disagree -- for example a chunk grid
    that contradicts the array metadata, or a branch callable executed without its program. Always preferred over
    silently delivering a wrong number.
    """

    remedy = (
        "inspect the array metadata and the registered reductions: this is a DEISA invariant violation, not a "
        "callback mistake. If it persists, register the callback with precompute=False to rule the reduction path out."
    )


def raw_field_read(symbol: str, *, array_name: str, timestep: int) -> RawFieldReadError:
    """Refusal for a callback that consumes the raw data of ``array_name`` at ``timestep`` via ``symbol``."""
    return RawFieldReadError(
        f"{symbol} reads the raw data of array {array_name!r} at timestep {timestep}, which the reduction path never "
        f"delivers: bridges ship chunk-local reduction partials, so the full field is not available on the analytics "
        f"host. Reductions on a DEISA collection return their value without materializing anything."
    )


def numpy_materialization(symbol: str) -> MaterializationError:
    """Refusal for a numpy conversion (``np.asarray``/``np.array``/matplotlib-style consumption) of ``symbol``."""
    return MaterializationError(
        f"{symbol} was converted to a numpy array. That materializes the whole registered field on the analytics host "
        f"and leaves no trace in any task graph, so it cannot be planned or refused later -- it is refused here "
        f"instead of silently gathering the chunk."
    )


def iteration_materialization(symbol: str) -> MaterializationError:
    """Refusal for iterating over ``symbol`` (``list(collection)``, ``iter(collection)``, ``for x in collection``)."""
    return MaterializationError(
        f"{symbol} was iterated over, which would yield the field's raw elements one at a time and therefore "
        f"materialize the whole registered field on the analytics host."
    )


def ambiguous_truth(symbol: str) -> UnsupportedReductionError:
    """Refusal for a truth test on ``symbol`` (``if collection:``), which has no single-element answer."""
    return UnsupportedReductionError(
        f"{symbol} was used as a boolean. A lazy DEISA operand has no truth value: testing it would either gather the "
        f"chunk or invent an answer. Reduce it first (for example ``.any()`` or ``.all()``) to get a value you can "
        f"branch on."
    )


def non_associative(op: str, symbol: str) -> NonAssociativeReductionError:
    """Refusal for the non-associative reduction ``op`` requested through ``symbol``."""
    return NonAssociativeReductionError(
        f"{symbol} requests the non-associative reduction {op!r}. It cannot be rebuilt from chunk-local partials "
        f"because its answer depends on the global ordering of the data, which no bridge can see."
    )


def off_surface(symbol: str, detail: str = "") -> UnsupportedReductionError:
    """Refusal for ``symbol``, which is outside the closed operator surface (``detail`` says how)."""
    reason = f"{symbol} is outside the closed DEISA operator surface"
    if detail:
        reason = f"{reason}: {detail}"
    return UnsupportedReductionError(f"{reason}.")


def mixed_root(symbol: str, root: str, other: str) -> UnsupportedReductionError:
    """Refusal for ``symbol``, which combined the two distinct roots ``root`` and ``other``."""
    return UnsupportedReductionError(
        f"{symbol} combined two different DEISA roots: {root} and {other}. One branch is rooted in exactly ONE "
        f"registered array AND one timestep, so a cross-array or mixed-timestep expression (for example "
        f"``window[-1] - window[-2]``) cannot be folded from chunk-local partials."
    )


def cross_reduction(symbol: str, detail: str = "") -> UnsupportedReductionError:
    """Refusal for ``symbol``, a reduction result used as an operand of another expression or reduction."""
    reason = f"{symbol} uses a completed reduction as an operand"
    if detail:
        reason = f"{reason}: {detail}"
    return UnsupportedReductionError(
        f"{reason}. A reduction collapses to a single delivered value and has no chunk-local partial a bridge could "
        f"fold into another expression; compute each reduction separately and combine the delivered values in the "
        f"callback."
    )


def keepdims_unsupported(symbol: str) -> UnsupportedReductionError:
    """Refusal for a ``keepdims=True`` reduction requested through ``symbol``."""
    return UnsupportedReductionError(
        f"{symbol} requested keepdims=True, which the v1 reduction set does not support: the combine step rebuilds "
        f"the reduced axes from the delivered partials, and keeping them at size 1 changes the delivered shape the "
        f"callback's other reductions are keyed by. Use keepdims=False and add the axis back with numpy on the "
        f"DELIVERED result."
    )


def dtype_out_unsupported(symbol: str, keyword: str) -> UnsupportedReductionError:
    """Refusal for the unsupported reduction keyword ``keyword`` on ``symbol``."""
    return UnsupportedReductionError(
        f"{symbol} passed {keyword}, which is not part of the closed DEISA reduction surface: the bridge folds its "
        f"chunk in the array's own dtype and the combine step owns the accumulation dtype, so a caller-chosen one "
        f"cannot be honoured consistently across bridges."
    )


def ddof_unsupported(op: str, ddof: object, symbol: str) -> UnsupportedReductionError:
    """Refusal for ``op`` requested with a non-zero ``ddof`` through ``symbol``."""
    return UnsupportedReductionError(
        f"{symbol} requested {op}(ddof={ddof!r}). The cross-bridge moment combine is fixed at ddof=0: applying a "
        f"per-bridge Bessel correction before the merge would correct each chunk by its own size and give a wrong "
        f"answer. Take the ddof=0 value and correct the DELIVERED scalar, or use ddof=0."
    )


def no_precomputable_reduction(symbol: str, detail: str = "") -> NoPrecomputableReductionError:
    """Refusal for a callback (``symbol``) that asks for no dispatchable reduction."""
    reason = f"{symbol} contains no reduction the DEISA surface can dispatch"
    if detail:
        reason = f"{reason}: {detail}"
    return NoPrecomputableReductionError(
        f"{reason}. A prefetch plan for this callback would carry nothing a bridge could compute chunk-locally. "
        f"Either add a reduction (a pointwise expression alone delivers no value) or register the callback with "
        f"precompute=False."
    )


def precompute_runtime_invariant(symbol: str, detail: str) -> PrecomputeRuntimeError:
    """Refusal for a violated runtime invariant, naming ``symbol`` and the ``detail`` that broke it."""
    return PrecomputeRuntimeError(f"{symbol}: {detail}.")


def no_dispatch_transport(symbol: str) -> PrecomputeRuntimeError:
    """Refusal for ``symbol``, a reduction whose collection carries no bridge transport.

    A collection is only able to dispatch because the delivery path attaches the transport of the array's bridges to
    it. A collection built by hand -- outside that path -- describes an array but owns no way to ask the bridges for
    anything, so the reduction has no partials to merge.
    """
    return PrecomputeRuntimeError(
        f"{symbol} cannot be dispatched: the collection it is rooted in carries no bridge transport. A reduction is "
        f"delivered by asking every bridge that owns a chunk for its partial, and the delivery path attaches that "
        f"transport when it hands the collection to the callback; a collection built outside that path has no bridges "
        f"to ask.",
        remedy=_REMEDY_PRECOMPUTE_OFF,
    )


_STALE_TIMESTEP_REMEDY = (
    "reduce window[-1] (the latest delivered timestep), or use a menu operation -- sum, min, max, mean, prod or std on "
    "the bare array -- which is served from the menu shipped with the delivery and works on any element of the window."
)


def stale_timestep(symbol: str, *, array_name: str, requested: int, latest: int) -> UnsupportedReductionError:
    """Refusal for ``symbol``, a reduction rooted in a window element older than the latest delivery.

    The on-demand path answers from the chunk a bridge LAST SENT (one retained chunk per array per bridge), so a plan
    stamped with an older timestep has no chunk to fold and the bridge can only answer with an error. Refusing at
    compose time keeps that out of the run: the same expression on ``window[-1]`` is served, and every menu operation
    is still answered for any retained window element, because the menu travels with the delivery.
    """
    return UnsupportedReductionError(
        f"{symbol} reduces array {array_name!r} at timestep {requested}, but the newest timestep delivered for that "
        f"array is {latest}. The bridges retain one chunk per array -- the one they last sent -- so an on-demand "
        f"reduction of an older window element has no chunk to fold on the bridge side, and asking for one would abort "
        f"the run.",
        remedy=_STALE_TIMESTEP_REMEDY,
    )


def rewrap(refusal: RuntimeRefusal, remedy: str) -> RuntimeRefusal:
    """Return a copy of ``refusal`` that keeps the message but carries ``remedy`` instead of its own."""
    return type(refusal)(refusal.message, remedy=remedy)
