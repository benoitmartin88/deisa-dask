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
"""The retired ``BranchSpec`` dataclass, kept alive for the equivalence harness ONLY.

Nothing under ``src/`` may import this module.

``BranchSpec`` used to live in ``deisa/dask/branch.py`` and was the per-array handshake payload of the precompute
delivery path. The runtime planner replaced it with ``deisa.dask.runtime_expr`` / ``runtime_compose`` requests, and
nothing under ``src/`` builds or reads a ``BranchSpec`` any more. It survives here because the K6a equivalence oracle
(``_oracle/branch_specs.py``, imported by ``test_chain.py``, ``test_precompute.py`` and ``test/utils.py``) *builds*
real spec objects to compare the analyzer's plan against the runtime planner's.

The field set is frozen as it was at deletion: adding a field here is fine only if the oracle needs it, since the
runtime planner's own request types (``RuntimeRequest`` / ``PCARequest``) are the product-side contract now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple


@dataclass
class BranchSpec:
    """One chunk-local sub-expression the bridge can execute + combine.

    Attributes
    ----------
    output_key : str
        Stable identifier for this branch (e.g. ``"f-mean"``). The bridge uses it to namespace its scatter key and
        the Deisa topic handler uses it to route the per-bridge partials back to the same branch.
    input_name : str
        Registered array name the branch is rooted at (e.g. ``"a"``).
    output_kind : str
        One of ``"scalar"`` / ``"mean"`` / ``"moment"`` / ``"pca"``.
    branch_func : Callable
        Python callable that, given a numpy chunk, returns the branch's per-bridge partial value (a scalar /
        ndarray / dict). Pickled across the bridge process boundary.
    chunk_axis : Optional[Tuple[int, ...]]
        For reductions, the tuple of axes being reduced in the chunk (e.g. ``(0, 1)`` for full reduction on a 2-D
        chunk). ``None`` for pointwise-only branches.
    finalize : Optional[str]
        ``"sqrt"`` for std (apply ``np.sqrt`` after combining), else ``None``.
    partial_shape : Tuple[int, ...]
        Shape of the **per-bridge partial** (what the branch_func returns). For ``scalar`` reductions on a 2-D chunk
        with ``keepdims=False`` this is ``()``; with ``keepdims=True`` it is ``(1, 1)``. For axis reductions the
        partial keeps the un-reduced axes' full size (e.g. ``mean(axis=0)`` on ``(M, N)`` partial has shape ``(1,
        N)``).
    partial_dtype : str
        NumPy dtype string of the per-bridge partial.
    """

    output_key: str
    input_name: str
    output_kind: str
    branch_func: Callable[[Any], Any]
    chunk_axis: Optional[Tuple[int, ...]]
    finalize: Optional[str]
    partial_shape: Tuple[int, ...]
    partial_dtype: str
    op_name: str = ""
    deliver_direct: bool = True
    window_read: bool = False
    reduction_axes: Tuple[int, ...] = ()
    # For a "pca" branch: the recorded MergeablePCA configuration, applied at the ROOT of the merge tree on the Dask
    # side and deliberately NOT on the bridge. Empty for every reduction branch.
    summary_config: Dict[str, Any] = field(default_factory=dict)
