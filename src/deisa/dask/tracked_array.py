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
# * Neither the names of CEA, nor the names of the contributors may be used to
#   endorse or promote products derived from this software without specific
#   prior written  permission.
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
TrackedArray: Proxy dask array that intercepts .compute() on reduction results.

During callback registration, provides a real dask array interface that builds
task graphs, but captures operations when .compute() is called on reductions.
"""

import dask.array as da
import numpy as np
from dask.delayed import delayed
from typing import List, Dict, Any, Tuple, Callable


REDUCTION_OPS = {'sum', 'mean', 'std', 'var', 'max', 'min', 'prod', 'amax', 'amin'}


def _extract_operations_from_graph(darr: da.Array, array_name: str) -> List[Dict[str, Any]]:
    """
    Extract reduction operations from a dask array's task graph.

    - ``:param darr:`` The dask array.
    - ``:param array_name:`` The array name for hint keys.
    - ``:return:`` List of operation hints.
    """
    hints = []
    try:
        graph = darr.__dask_graph__()
        for layer_name, layer in graph.layers.items():
            for k, v in layer.items():
                if isinstance(v, tuple) and len(v) > 0:
                    func = v[0]
                    if hasattr(func, 'funcs'):
                        # Compose case (sum, max, min, prod)
                        for f in func.funcs:
                            if hasattr(f, 'func'):
                                np_name = f.func.__name__
                                if np_name == 'max':
                                    np_name = 'amax'
                                elif np_name == 'min':
                                    np_name = 'amin'
                                hints.append({
                                    'func_module': 'numpy',
                                    'func_name': np_name,
                                    'keywords': dict(f.keywords),
                                    'output_key': f"{array_name}-{np_name}",
                                    'type': 'reduction'
                                })
                    elif hasattr(func, 'func'):
                        # functools.partial case (mean, std, var)
                        func_name = func.func.__name__
                        if func_name == 'mean_agg':
                            op_name = 'mean'
                        elif func_name == 'moment_agg':
                            op_name = 'std'
                        else:
                            op_name = func_name
                        hints.append({
                            'func_module': 'numpy',
                            'func_name': op_name,
                            'keywords': dict(func.keywords),
                            'output_key': f"{array_name}-{op_name}",
                            'type': 'reduction'
                        })
                    break
    except Exception:
        pass
    return hints


class TrackedDeisaArray:
    """
    Proxy providing dask array interface with .compute() interception.

    Reduction methods (.sum, .mean, etc.) return wrapped dask arrays that
    capture operation hints when .compute() is called.
    """

    def __init__(self, array_name: str = "unknown"):
        self._array_name = array_name
        # The underlying dask array stub
        self._stub = da.from_delayed(
            delayed(lambda: np.array(0.0))(),
            shape=(1,),
            dtype=float
        )
        # Collected hints during analysis
        self._hints: List[Dict[str, Any]] = []
        # Track reduction results that have been computed
        self._computed_results: List[da.Array] = []

    def __getattr__(self, name):
        """Delegate to underlying stub, wrapping reduction results."""
        attr = getattr(self._stub, name)
        
        if callable(attr) and name in REDUCTION_OPS:
            # Wrap reduction method to intercept compute
            def wrapped_reduction(*args, op_name=name, **kwargs):
                result_darr = attr(*args, **kwargs)
                self._computed_results.append(result_darr)
                return TrackedResult(result_darr, self._array_name, self._hints)
            return wrapped_reduction
        
        # Non-reduction methods delegate directly
        if callable(attr):
            return attr
        return attr

    def get_reduction_hints(self) -> List[Dict[str, Any]]:
        """Return collected hints from computed reductions."""
        # Also extract from stub directly (for lazy patterns)
        stub_hints = _extract_operations_from_graph(self._stub, self._array_name)
        # Merge with computed hints, avoiding duplicates
        seen = {h['func_name'] for h in self._hints}
        for h in stub_hints:
            if h['func_name'] not in seen:
                self._hints.append(h)
        return self._hints


class TrackedResult:
    """Proxy for dask array that intercepts .compute()."""

    def __init__(self, darr: da.Array, array_name: str, hints: List[Dict[str, Any]]):
        self._darr = darr
        self._array_name = array_name
        self._hints = hints

    def compute(self, *args, **kwargs):
        """Intercept compute to capture operation hints."""
        hints = _extract_operations_from_graph(self._darr, self._array_name)
        self._hints.extend(hints)
        return self

    def __getattr__(self, name):
        return getattr(self._darr, name)


def make_tracked_deisa_stub(array_name: str = "unknown") -> Tuple[TrackedDeisaArray, Callable]:
    """
    Create a tracked stub for callback analysis.

    Returns a TrackedDeisaArray that builds real dask task graphs but
    intercepts .compute() calls on reductions to capture operations.

    - ``:param array_name:`` The name of the array.
    - ``:return:`` (tracked_stub, hints_extractor)
    """
    tracked = TrackedDeisaArray(array_name)
    return tracked, tracked.get_reduction_hints