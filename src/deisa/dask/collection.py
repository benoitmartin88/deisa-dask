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
The DEISA lazy collection a callback receives, one instance per registered array per timestep.

The collection is what the topic handler delivers (in P3) where it used to deliver a dask array. It carries the array's
identity and metadata -- name, timestep, global shape, dtype, chunk shape and the number of bridges that own a chunk --
and it exposes the closed operator surface of :mod:`deisa.dask.runtime_expr`. It is deliberately NOT a dask
collection and NOT an ``dask.array.Array`` subclass:

- a ``da.Array`` subclass loses its identity on the first derived expression, so it cannot keep a whole expression
  inspectable;
- a real dask collection invites the callback to compute it, which is exactly the full-chunk gather the reduction path
  exists to avoid.

So a raw read has no path that works: ``compute()``, ``np.asarray``, ``iter`` and an external dask graph are each
refused with a named cause and a remedy.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from deisa.dask import runtime_errors as errors
from deisa.dask.runtime_expr import _CHUNK_PROGRAM, LazyOperand, Root

__all__ = ["DeisaCollection"]


def _as_shape(value: Any, what: str) -> Tuple[int, ...]:
    """Normalize ``value`` into a non-empty tuple of positive ints, or raise a runtime-invariant refusal."""
    try:
        shape = tuple(int(size) for size in value)
    except (TypeError, ValueError):
        raise errors.precompute_runtime_invariant("DeisaCollection", f"{what} {value!r} is not a shape") from None
    if not shape or any(size <= 0 for size in shape):
        raise errors.precompute_runtime_invariant("DeisaCollection", f"{what} {shape} is not a non-empty shape")
    return shape


class DeisaCollection(LazyOperand):
    """One registered array as the callback sees it at one timestep.

    - ``:param name:`` Registered array name (e.g. ``"f"``). Its ``name`` is half of the root, so a reduction can never
      be served by another array's chunk.
    - ``:param timestep:`` Simulation timestep this instance was delivered for. The other half of the root: two
      collections of the same array at different timesteps are two different roots and cannot be combined.
    - ``:param shape:`` GLOBAL shape of the array (all bridges), e.g. ``(8, 8)``.
    - ``:param dtype:`` Element dtype of the array.
    - ``:param chunk_shape:`` Shape of ONE bridge's chunk (the harness invariant is one chunk per bridge). Defaults to
      the global shape, i.e. a single-bridge array.
    - ``:param bridge_count:`` Number of bridges that own a chunk. Auto-detected from ``shape`` and ``chunk_shape`` when
      omitted; when given it is CROSS-CHECKED against them, because a caller-supplied count that disagrees with the
      metadata would make every reduction's combine grid wrong.
    - ``:param dispatcher:`` The compute boundary that delivers this collection's reductions, attached by the delivery
      path. Left ``None`` by a collection built outside that path, whose reductions then refuse instead of gathering.
    """

    __slots__ = (
        "name",
        "timestep",
        "chunk_shape",
        "bridge_count",
        "_root",
        "_shape",
        "_dtype",
        "_program",
        "_dispatcher",
    )

    def __init__(
        self,
        name: str,
        timestep: int,
        shape: Sequence[int],
        dtype: Any = np.float64,
        chunk_shape: Optional[Sequence[int]] = None,
        bridge_count: Optional[int] = None,
        dispatcher: Any = None,
    ) -> None:
        self.name = str(name)
        self.timestep = int(timestep)
        self._shape = _as_shape(shape, "the global shape")
        self._dtype = np.dtype(dtype)
        self.chunk_shape = self._shape if chunk_shape is None else _as_shape(chunk_shape, "the chunk shape")
        if len(self.chunk_shape) != len(self._shape):
            raise errors.precompute_runtime_invariant(
                "DeisaCollection",
                f"the chunk shape {self.chunk_shape} and the global shape {self._shape} have different ranks",
            )
        if any(chunk > size for chunk, size in zip(self.chunk_shape, self._shape)):
            raise errors.precompute_runtime_invariant(
                "DeisaCollection",
                f"the chunk shape {self.chunk_shape} does not fit inside the global shape {self._shape}",
            )
        # Per-data-axis extent of the MPI chunk grid: the harness invariant that the cart dims map one-to-one onto the
        # array's data axes, so the product IS the number of bridges holding a chunk of this array.
        self.bridge_count = int(np.prod([max(1, size // chunk) for chunk, size in zip(self.chunk_shape, self._shape)]))
        if bridge_count is not None and int(bridge_count) != self.bridge_count:
            raise errors.precompute_runtime_invariant(
                "DeisaCollection",
                f"the metadata implies {self.bridge_count} bridge(s) for array {self.name!r} "
                f"(global shape {self._shape}, chunk shape {self.chunk_shape}) but {bridge_count} was declared",
            )
        self._root = Root(array_name=self.name, timestep=self.timestep)
        self._program = _CHUNK_PROGRAM
        self._dispatcher = dispatcher

    @classmethod
    def from_metadata(
        cls,
        name: str,
        timestep: int,
        metadata: Mapping[str, Any],
        bridge_count: Optional[int] = None,
        dispatcher: Any = None,
    ) -> "DeisaCollection":
        """Build a collection from the array metadata the handshake already exposes.

        The delivery site (the topic handler) has the metadata dict, not a shape: reading it here keeps the key spelling
        in one place instead of at every call site.

        - ``:param name:`` Registered array name.
        - ``:param timestep:`` Timestep the collection is delivered for.
        - ``:param metadata:`` Entry of the handshake's ``arrays_metadata`` for this array (``global_shape``,
          ``chunk_shape``, optional ``dtype``).
        - ``:param bridge_count:`` Optional cross-check, as in the constructor.
        - ``:param dispatcher:`` The compute boundary to attach, as in the constructor.
        """
        global_shape = metadata.get("global_shape")
        if not global_shape:
            raise errors.precompute_runtime_invariant(
                "DeisaCollection.from_metadata", f"the metadata for array {name!r} carries no ``global_shape``"
            )
        return cls(
            name=name,
            timestep=timestep,
            shape=global_shape,
            dtype=metadata.get("dtype", np.float64),
            chunk_shape=metadata.get("chunk_shape"),
            bridge_count=bridge_count,
            dispatcher=dispatcher,
        )

    def _symbol(self) -> str:
        return f"the collection {self.name!r} at timestep {self.timestep}"

    def __repr__(self) -> str:
        return (
            f"DeisaCollection(name={self.name!r}, timestep={self.timestep}, shape={self._shape}, "
            f"dtype={self._dtype}, chunk_shape={self.chunk_shape}, bridge_count={self.bridge_count})"
        )
