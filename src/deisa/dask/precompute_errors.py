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
"""The precompute refusal taxonomy, kept as the base of the runtime refusal taxonomy.

This module holds the exception classes the precompute analyzer used to define. They live here, and not in the
analyzer, for one reason: :mod:`deisa.dask.runtime_errors` derives the runtime refusals from them, so
``except PrecomputeError`` -- a handler written against the precompute path -- keeps catching every refusal the
runtime reduction path raises. That ``isinstance`` graph outlives the analyzer it was named after (plan P4 removes the
analyzer; it does not remove the class identity callers may already catch).

The classes are unchanged in name and hierarchy. Their docstrings below describe the precompute analyzer that used to
raise them, which is now gone: the refusal FAMILY survives, the raise sites do not. The runtime path raises the
:mod:`deisa.dask.runtime_errors` subclasses; a caller should catch :class:`PrecomputeError` or
:class:`~deisa.dask.runtime_errors.RuntimeRefusal`, never the analyzer-era subclasses directly.
"""

from __future__ import annotations

__all__ = [
    "IncompatibleCallbackError",
    "MaterializationError",
    "NoComputeBoundaryError",
    "NoPrecomputableReductionError",
    "PrecomputeError",
    "PrecomputeRuntimeError",
    "RawFieldReadError",
    "UnsupportedReductionError",
]


class PrecomputeError(Exception):
    """Base class for precompute analysis errors."""


class UnsupportedReductionError(PrecomputeError):
    """A reduction is present but its input expression cannot be traced back to a dask array.

    Covers opaque operands (``da.sum(opaque_object)``), wrapped reductions (a custom function around a dask call), and
    untraceable parameters (``da.sum(f * v2)`` where ``v2`` is derived from a config object).
    """


class NoPrecomputableReductionError(PrecomputeError):
    """The callback contains no reductions we can precompute.

    Example: callback only does ``da.fft.fft2(arr)`` - FFTs don't reduce.
    """


class PrecomputeRuntimeError(PrecomputeError):
    """A runtime (post-registration) precompute invariant was violated.

    Raised on the Deisa side (topic handler, callback dispatch view) and the bridge side (executing a branch on the
    local chunk) when a precompute contract breaks -- e.g. a failed branch dropped a partial, a reduction's partials do
    not cover the full chunk grid, or a callback calls a reduction that the analyzer did not record. Always prefer
    raising this over silently delivering a wrong number.
    """


class MaterializationError(PrecomputeError):
    """A full data materialization was detected in the callback.

    Example: ``np.array(dask_array)`` - forces gathering the whole array, defeating the purpose.
    """


class RawFieldReadError(PrecomputeError):
    """The callback reads the registered array's raw data, which precompute does not deliver.

    The precompute path ships per-bridge reduction partials only -- never the full chunk. Any callback that consumes
    the raw data (binds it as a value, indexes it, plots it, or computes something that is not a recorded
    chunk-local reduction) breaks the precompute contract: the data it needs is not on the network.
    """


class IncompatibleCallbackError(PrecomputeError):
    """The callback pattern is not supported by the precompute system.

    Example: dynamic dispatch (``getattr``), closures, ``exec``/``eval``, etc.
    """


class NoComputeBoundaryError(PrecomputeError):
    """The callback contains dask operations but no compute boundaries.

    We can't tell what the user wants computed. Without a ``.compute()``, ``client.compute(...)``, or similar, the
    analyzer has no way to know which dask arrays to extract hints for.
    """
