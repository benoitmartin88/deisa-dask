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
Compute-boundary precompute analyzer (no AST pattern matching, no callback execution).

The analyzer finds "compute boundaries" in the user's callback source --
points where a dask array is forced to materialize (.compute(),
client.compute(...), client.submit(...), np.array(...), etc.) -- and walks
each dask array's task graph to discover the reductions that should run
locally on the bridge before the data is scattered.

The approach is generic: the analyzer never enumerates reduction methods
(arr.sum, da.sum, etc.). It just builds the dask graph lazily (dask
operations are not executed) and hands the resulting dask array to
:func:`deisa.dask.task_hints.extract_reduction_hints`, which walks the
graph.

Hard contract: **the user's callback is never invoked during analysis.**
All "evaluation" is symbolic: dask operations on dask arrays are lazy and
return new dask arrays, never running tasks.
"""

from __future__ import annotations

import ast
import inspect
import logging
import textwrap
from typing import Any, Callable, Dict, List, Optional

import numpy as np

import dask.array as da
from deisa.dask.task_hints import extract_reduction_hints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class PrecomputeError(Exception):
    """Base class for precompute analysis errors."""


class UnsupportedReductionError(PrecomputeError):
    """A reduction is present but its input expression cannot be traced back to a dask array.

    Example: ``da.sum(opaque_object)``, or a custom function wrapping a reduction.
    """


class OpaqueParameterError(PrecomputeError):
    """A reduction's input depends on a parameter that is not (and cannot be derived from) a dask array.

    Example: ``da.sum(f * v2)`` where ``v2`` is built from a config object we can't trace.
    """


class NoPrecomputableReductionError(PrecomputeError):
    """The callback contains no reductions we can precompute.

    Example: callback only does ``da.fft.fft2(arr)`` - FFTs don't reduce.
    """


class MaterializationError(PrecomputeError):
    """A full data materialization was detected in the callback.

    Example: ``np.array(dask_array)`` - forces gathering the whole array, defeating the purpose.
    """


class IncompatibleCallbackError(PrecomputeError):
    """The callback pattern is not supported by the precompute system.

    Example: dynamic dispatch (``getattr``), closures, ``exec``/``eval``, etc.
    """


class NoComputeBoundaryError(PrecomputeError):
    """The callback contains dask operations but no compute boundaries.

    We can't tell what the user wants computed. Without a ``.compute()``,
    ``client.compute(...)``, or similar, the analyzer has no way to know
    which dask arrays to extract hints for.
    """


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_callback(
    callback: Callable,
    registered_arrays: Dict[str, Any],
    helpers: Optional[Dict[str, Callable]] = None,
    force: bool = False,
) -> List[Dict[str, Any]]:
    """Analyze the callback's source to find all reducible operations.

    - ``:param callback:`` The user's callback function. **Not invoked.**
    - ``:param registered_arrays:`` Mapping of name -> value. Values that are
      :class:`dask.array.Array` are the actual arrays the callback receives;
      any other value is treated as an opaque helper (e.g. a config object)
      whose attributes can be read at analysis time.
    - ``:param helpers:`` Optional mapping of function name -> function object
      for helper functions defined in another module (or to disambiguate
      same-file helpers). Helpers not listed here are also resolved by
      walking the callback's source file.
    - ``:param force:`` If True, skip unresolvable reductions with a warning
      instead of raising.
    - ``:return:`` List of hint dicts (the schema from
      :mod:`deisa.dask.task_hints`).
    - ``:raises PrecomputeError:`` On any unresolvable reduction (unless ``force=True``).
    """
    try:
        hints, err = _analyze_callback(callback, registered_arrays, helpers, force)
    except PrecomputeError as e:
        if force:
            logger.warning("analyze_callback: %s (force=True, skipping)", e)
            return []
        raise

    if err is not None:
        if force:
            logger.warning("analyze_callback: %s (force=True, skipping)", err)
            return []
        raise err

    return hints


def _analyze_callback(
    callback: Callable,
    registered_arrays: Dict[str, Any],
    helpers: Optional[Dict[str, Callable]],
    force: bool,
) -> tuple[List[Dict[str, Any]], Optional[PrecomputeError]]:
    """Internal worker for :func:`analyze_callback` that never swallows errors.

    Returns ``(hints, last_error)``. The caller decides how to surface the
    error: ``force=True`` warnings or normal raises.
    """
    # 1. Parse callback source
    callback_src = _get_source(callback)
    callback_tree = ast.parse(callback_src)
    source_file = _SourceFile.from_tree(callback_tree)

    # 2. Merge helpers (same-file helpers are also auto-discovered)
    for name, fn in (helpers or {}).items():
        try:
            helper_src = _get_source(fn)
        except IncompatibleCallbackError:
            continue
        helper_tree = ast.parse(helper_src)
        source_file.merge(_SourceFile.from_tree(helper_tree))

    # 3. Locate the callback's FunctionDef
    callback_def = source_file.find_function(callback.__name__)
    if callback_def is None:
        raise IncompatibleCallbackError(f"Could not locate FunctionDef for {callback.__name__!r} in callback source.")

    # 4. Build initial scope: callback params bound to registered arrays.
    # The dict's first key is the "primary" registered array name (used as
    # the base for output keys in hints).
    primary_name: str = next(iter(registered_arrays)) if registered_arrays else "f"
    reg_values = list(registered_arrays.values())
    param_names = [a.arg for a in callback_def.args.args]

    scope = _Scope()
    # Pre-bind standard library aliases that callbacks commonly use.
    # These let callbacks reference ``da.sum(...)`` / ``np.array(...)``
    # without explicit imports; the dask/np operations are lazy and
    # never execute.
    scope.set("da", da)
    scope.set("dask_array", da)
    scope.set("dask", da)
    scope.set("np", np)

    for idx, pname in enumerate(param_names):
        if pname == "window":
            scope.set(pname, _WindowProxy(reg_values))
        elif idx < len(reg_values):
            scope.set(pname, reg_values[idx])
        else:
            scope.set(pname, _UnboundParam(pname))

    # 5. Walk the callback body and collect compute boundaries.
    walker = _BoundaryWalker(source_file=source_file, primary_name=primary_name)
    walker.walk_body(callback_def.body, scope)

    # 6. Materialization takes priority: if any np.array/asarray on a dask
    # array was found, the callback can't be precomputed at all.
    if walker.had_materialization:
        return [], MaterializationError(
            "Callback contains a full materialization (e.g. np.array(dask_array)) that defeats precomputation."
        )

    dask_arrays = walker.dask_arrays
    had_boundaries = bool(walker.boundaries)

    # 7. Walk the dask graphs to find reductions.
    hints: List[Dict[str, Any]] = []
    for arr_info in dask_arrays:
        darr = arr_info["array"]
        # Pick the array name: the first registered array, by default.
        # We don't try to track which specific registered array a chain
        # of dask ops descends from -- the primary name is used
        # uniformly so the bridge gets stable output keys.
        try:
            new_hints = extract_reduction_hints(darr, primary_name)
        except Exception as e:  # pragma: no cover - safety net
            logger.debug("extract_reduction_hints failed: %s", e)
            new_hints = []
        hints.extend(new_hints)

    # 8. Decide what (if anything) to raise.
    if not hints:
        if not had_boundaries:
            return [], NoComputeBoundaryError(
                f"Callback {callback.__name__!r} contains dask operations "
                "but no compute boundaries (.compute(), client.compute(), "
                "client.submit(), etc.). Cannot determine which arrays to precompute."
            )
        return [], NoPrecomputableReductionError(
            f"Callback {callback.__name__!r} contains compute boundaries but no reductions we can precompute."
        )

    return hints, None


# ---------------------------------------------------------------------------
# Source file: AST cache + helper lookup
# ---------------------------------------------------------------------------
class _SourceFile:
    """Holds the AST of a source file with name -> FunctionDef/ClassDef indices."""

    def __init__(self, tree: ast.AST):
        self.tree = tree
        self._functions: Dict[str, ast.FunctionDef] = {}
        self._index_body(tree.body)

    @classmethod
    def from_tree(cls, tree: ast.AST) -> "_SourceFile":
        return cls(tree)

    def merge(self, other: "_SourceFile") -> None:
        """Merge another source file's functions into this one."""
        for name, fn in other._functions.items():
            if name not in self._functions:
                self._functions[name] = fn

    def _index_body(self, body: List[ast.stmt]) -> None:
        for node in body:
            if isinstance(node, ast.FunctionDef):
                self._functions[node.name] = node

    def find_function(self, name: str) -> Optional[ast.FunctionDef]:
        return self._functions.get(name)


def _get_source(fn: Callable) -> str:
    """Get the source code for a function, dedented.

    Order of resolution:
    1. ``fn.__source__`` attribute (set by test helpers that compile via ``exec``)
    2. ``inspect.getsource`` (works for real source files)
    """
    src_attr = getattr(fn, "__source__", None)
    if src_attr is not None:
        return textwrap.dedent(src_attr)
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError) as e:
        raise IncompatibleCallbackError(
            f"Cannot read source of {getattr(fn, '__name__', fn)!r}: {e}. "
            "The AST-based analyzer requires a real Python function with source."
        ) from e
    return textwrap.dedent(src)


# ---------------------------------------------------------------------------
# Scope and value markers
# ---------------------------------------------------------------------------
class _Scope:
    """Tracks variable bindings during symbolic evaluation."""

    def __init__(self, parent: Optional["_Scope"] = None):
        self.bindings: Dict[str, Any] = {}
        self.parent = parent

    def get(self, name: str) -> Any:
        if name in self.bindings:
            return self.bindings[name]
        if self.parent is not None:
            return self.parent.get(name)
        return _Missing(name)

    def set(self, name: str, value: Any) -> None:
        self.bindings[name] = value

    def child(self) -> "_Scope":
        return _Scope(parent=self)


class _Missing:
    """Sentinel for unbound names.

    Behaves as a transparent placeholder in arithmetic and subscript
    operations: most ops return another ``_Missing`` so callers can chain
    through without erroring. Attribute/subscript access on a ``_Missing``
    returns another ``_Missing``. This lets callbacks with closure
    variables (e.g. a counter dict) be analyzed without raising.
    """

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __getattr__(self, attr: str) -> "_Missing":
        return _Missing(f"{self.name}.{attr}")

    def __getitem__(self, key: Any) -> "_Missing":
        return _Missing(f"{self.name}[{key!r}]")

    def __call__(self, *args: Any, **kwargs: Any) -> "_Missing":
        return _Missing(f"{self.name}()")

    def __bool__(self) -> bool:
        return False

    # Arithmetic: pass through as _Missing
    def _binop(self, other: Any) -> "_Missing":
        return _Missing(f"{self.name}")

    __add__ = __radd__ = _binop
    __sub__ = __rsub__ = _binop
    __mul__ = __rmul__ = _binop
    __truediv__ = __rtruediv__ = _binop
    __floordiv__ = __rfloordiv__ = _binop
    __mod__ = __rmod__ = _binop
    __pow__ = __rpow__ = _binop
    __lshift__ = __rlshift__ = _binop
    __rshift__ = __rrshift__ = _binop
    __and__ = __rand__ = _binop
    __or__ = __ror__ = _binop
    __xor__ = __rxor__ = _binop
    __matmul__ = __rmatmul__ = _binop

    def __neg__(self) -> "_Missing":
        return _Missing(self.name)

    def __pos__(self) -> "_Missing":
        return _Missing(self.name)

    def __invert__(self) -> "_Missing":
        return _Missing(self.name)

    def __repr__(self) -> str:
        return f"<Missing {self.name!r}>"


class _UnboundParam:
    """Marker for callback parameters the user did not pass to analyze_callback.

    Behaves as a Python scalar in arithmetic so dask operations still build
    a valid graph (the actual value is irrelevant - we only need the graph
    structure to extract reduction hints).
    """

    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    def __mul__(self, other):
        return other * 0.0

    def __rmul__(self, other):
        return other * 0.0

    def __add__(self, other):
        return other

    def __radd__(self, other):
        return other

    def __sub__(self, other):
        return other

    def __rsub__(self, other):
        return other

    def __truediv__(self, other):
        return 0.0

    def __rtruediv__(self, other):
        return 0.0

    def __pow__(self, other):
        return 0.0**other

    def __rpow__(self, other):
        return other**0.0

    def __neg__(self):
        return 0.0

    def __pos__(self):
        return 0.0

    def __repr__(self) -> str:
        return f"<UnboundParam {self.name!r}>"


class _WindowProxy:
    """List-like proxy used in place of the user's ``window`` parameter.

    Supports integer subscripting (positive or negative) to return one of
    the registered arrays. Reading any other attribute/method raises.
    """

    def __init__(self, arrays: List[Any]):
        self._arrays = list(arrays)

    def __len__(self) -> int:
        return len(self._arrays)

    def __getitem__(self, idx: int) -> Any:
        return self._arrays[idx]

    def __iter__(self):
        return iter(self._arrays)


# ---------------------------------------------------------------------------
# Boundary walker
# ---------------------------------------------------------------------------
# Functions we recognize as materialization (forbid precompute).
_MATERIALIZING_FUNCS = {"array", "asarray", "save", "savetxt", "savez", "savez_compressed"}

# Subset of dask/np submodules we recognize as returning a dask array.
_DASK_RECURSE_SUBMODULES = {"fft"}


class _BoundaryWalker:
    """Walks the callback's AST looking for compute boundaries.

    A compute boundary is a call that forces a dask array to materialize:
    - ``arr.compute()``
    - ``client.compute(arr)`` / ``client.compute([arr1, ...])``
    - ``client.submit(func, arr)``
    - ``np.array(darr)`` / ``np.asarray(darr)`` (materialization - error)

    When a boundary is found, the argument expression is symbolically
    evaluated to a dask array (lazy - no execution), and the array is
    queued for graph extraction. For lists, every element is queued.
    """

    def __init__(self, source_file: _SourceFile, primary_name: str = "f"):
        self.source_file = source_file
        self.primary_name = primary_name
        self.dask_arrays: List[Dict[str, Any]] = []
        self.boundaries: List[Dict[str, Any]] = []
        self.had_materialization: bool = False

    # -- Statement walking -------------------------------------------------
    def walk_body(self, body: List[ast.stmt], scope: _Scope) -> None:
        for stmt in body:
            self.walk_stmt(stmt, scope)

    def walk_stmt(self, stmt: ast.stmt, scope: _Scope) -> None:
        if isinstance(stmt, ast.Assign):
            value = self._eval(stmt.value, scope)
            for target in stmt.targets:
                self._assign_target(target, value, scope)
            return
        if isinstance(stmt, ast.AugAssign):
            current = self._eval(stmt.target, scope)
            rhs = self._eval(stmt.value, scope)
            new_value = self._binop(stmt.op, current, rhs)
            self._assign_target(stmt.target, new_value, scope)
            return
        if isinstance(stmt, ast.Expr):
            # Expression statement: evaluate, but ignore result.
            self._eval(stmt.value, scope)
            return
        if isinstance(stmt, ast.If):
            test = self._eval(stmt.test, scope)
            branch_value = _truthy(test)
            if branch_value is True:
                self.walk_body(stmt.body, scope)
            elif branch_value is False:
                self.walk_body(stmt.orelse, scope)
            else:
                # Both branches: walk them sequentially (defensive)
                self.walk_body(stmt.body, scope)
                self.walk_body(stmt.orelse, scope)
            return
        if isinstance(stmt, ast.For):
            self._walk_for(stmt, scope)
            return
        if isinstance(stmt, ast.Return):
            if stmt.value is not None:
                self._eval(stmt.value, scope)
            return
        if isinstance(stmt, (ast.Pass, ast.Break, ast.Continue)):
            return
        if isinstance(stmt, ast.Try):
            self.walk_body(stmt.body, scope)
            for handler in stmt.handlers:
                self.walk_body(handler.body, scope)
            self.walk_body(stmt.orelse, scope)
            self.walk_body(stmt.finalbody, scope)
            return
        if isinstance(stmt, ast.With):
            for item in stmt.items:
                ctx = self._eval(item.context_expr, scope)
                if item.optional_vars is not None:
                    self._assign_target(item.optional_vars, ctx, scope)
            self.walk_body(stmt.body, scope)
            return
        # Anything else: best-effort evaluation. We don't need to
        # surface IncompatibleCallbackError for rare AST shapes; the
        # tests that need them can be added explicitly.
        try:
            self._eval(stmt, scope)
        except IncompatibleCallbackError:
            raise
        except PrecomputeError:
            raise

    # -- For-loop: static-range unroll, else fail --------------------------
    def _walk_for(self, stmt: ast.For, scope: _Scope) -> None:
        values = self._try_unroll_iter(stmt.iter, scope)
        if values is None:
            raise IncompatibleCallbackError(
                f"For-loop over non-constant iterable at line {getattr(stmt, 'lineno', -1)}: "
                "only `for x in range(<constant>)` or `for x in [<literal>]` is supported."
            )
        target = stmt.target
        for value in values:
            inner_scope = scope.child()
            self._assign_target(target, value, inner_scope)
            self.walk_body(stmt.body, inner_scope)
            if stmt.orelse:
                self.walk_body(stmt.orelse, inner_scope)

    def _try_unroll_iter(self, iter_node: ast.AST, scope: _Scope) -> Optional[List[Any]]:
        if isinstance(iter_node, ast.Call):
            func = iter_node.func
            if isinstance(func, ast.Name) and func.id == "range":
                args = []
                for a in iter_node.args:
                    v = self._eval(a, scope)
                    if isinstance(v, _Missing):
                        return None
                    args.append(v)
                if not args:
                    return list(range(0))
                if len(args) == 1:
                    return list(range(int(args[0])))
                if len(args) == 2:
                    return list(range(int(args[0]), int(args[1])))
                if len(args) == 3:
                    return list(range(int(args[0]), int(args[1]), int(args[2])))
        if isinstance(iter_node, (ast.List, ast.Tuple)):
            return [self._eval(elt, scope) for elt in iter_node.elts]
        return None

    # -- Assignment --------------------------------------------------------
    def _assign_target(self, target: ast.AST, value: Any, scope: _Scope) -> None:
        if isinstance(target, ast.Name):
            scope.set(target.id, value)
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            if not isinstance(value, (list, tuple)):
                raise IncompatibleCallbackError(
                    f"Cannot unpack non-iterable value into tuple at line {getattr(target, 'lineno', -1)}"
                )
            if len(value) != len(target.elts):
                raise IncompatibleCallbackError(
                    f"Tuple/list assignment size mismatch at line {getattr(target, 'lineno', -1)}"
                )
            for elt, v in zip(target.elts, value):
                self._assign_target(elt, v, scope)
            return
        if isinstance(target, ast.Subscript):
            # obj[idx] = value -- not supported, but benign for closures.
            return
        if isinstance(target, ast.Attribute):
            obj = self._eval(target.value, scope)
            if isinstance(obj, (_Missing, _UnboundParam)):
                return  # benign
            setattr(obj, target.attr, value)
            return
        raise IncompatibleCallbackError(
            f"Unsupported assignment target: {type(target).__name__} at line {getattr(target, 'lineno', -1)}"
        )

    # -- Expression evaluation --------------------------------------------
    def _eval(self, node: ast.AST, scope: _Scope) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return scope.get(node.id)
        if isinstance(node, ast.BinOp):
            return self._binop(node.op, self._eval(node.left, scope), self._eval(node.right, scope))
        if isinstance(node, ast.UnaryOp):
            return self._unaryop(node.op, self._eval(node.operand, scope))
        if isinstance(node, ast.BoolOp):
            return self._boolop(node.op, node.values, scope)
        if isinstance(node, ast.Compare):
            return self._compare(node, scope)
        if isinstance(node, ast.Subscript):
            value = self._eval(node.value, scope)
            slc = self._slice(node.slice, scope)
            return self._apply_subscript(value, slc)
        if isinstance(node, ast.Call):
            return self._call(node, scope)
        if isinstance(node, ast.Attribute):
            return self._attr(node, scope)
        if isinstance(node, ast.IfExp):
            test = self._eval(node.test, scope)
            branch_value = _truthy(test)
            if branch_value is True:
                return self._eval(node.body, scope)
            if branch_value is False:
                return self._eval(node.orelse, scope)
            return self._eval(node.body, scope)
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval(elt, scope) for elt in node.elts]
        if isinstance(node, ast.Dict):
            return {self._eval(k, scope): self._eval(v, scope) for k, v in zip(node.keys, node.values)}
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(v.value)
                else:
                    parts.append(self._eval(v, scope))
            return "".join(str(p) for p in parts)
        if isinstance(node, ast.Starred):
            return self._eval(node.value, scope)
        raise IncompatibleCallbackError(
            f"Unsupported expression: {type(node).__name__} at line {getattr(node, 'lineno', -1)}"
        )

    def _slice(self, slc: ast.AST, scope: _Scope) -> Any:
        if isinstance(slc, ast.Slice):
            lower = self._eval(slc.lower, scope) if slc.lower is not None else None
            upper = self._eval(slc.upper, scope) if slc.upper is not None else None
            step = self._eval(slc.step, scope) if slc.step is not None else None
            return slice(lower, upper, step)
        if isinstance(slc, ast.Tuple):
            return tuple(self._slice(e, scope) for e in slc.elts)
        return self._eval(slc, scope)

    def _apply_subscript(self, value: Any, slc: Any) -> Any:
        if isinstance(value, _Missing):
            return _Missing(f"{value.name}[...]")
        if isinstance(value, da.Array):
            return value[slc]
        if isinstance(value, _WindowProxy):
            if isinstance(slc, int):
                return value[slc]
            raise IncompatibleCallbackError("window subscript must be an integer")
        if isinstance(value, (list, tuple, np.ndarray)):
            return value[slc]
        # Other types: try to subscript and hope for the best
        return value[slc]

    # -- Operators ---------------------------------------------------------
    def _binop(self, op: ast.AST, left: Any, right: Any) -> Any:
        if isinstance(op, ast.Add):
            return left + right
        if isinstance(op, ast.Sub):
            return left - right
        if isinstance(op, ast.Mult):
            return left * right
        if isinstance(op, ast.Div):
            return left / right
        if isinstance(op, ast.FloorDiv):
            return left // right
        if isinstance(op, ast.Mod):
            return left % right
        if isinstance(op, ast.Pow):
            return left**right
        if isinstance(op, ast.LShift):
            return left << right
        if isinstance(op, ast.RShift):
            return left >> right
        if isinstance(op, ast.BitOr):
            return left | right
        if isinstance(op, ast.BitXor):
            return left ^ right
        if isinstance(op, ast.BitAnd):
            return left & right
        if isinstance(op, ast.MatMult):
            return left @ right
        raise IncompatibleCallbackError(f"Unsupported binary operator: {type(op).__name__}")

    def _unaryop(self, op: ast.AST, operand: Any) -> Any:
        if isinstance(op, ast.USub):
            return -operand
        if isinstance(op, ast.UAdd):
            return +operand
        if isinstance(op, ast.Not):
            return not _truthy(operand)
        if isinstance(op, ast.Invert):
            return ~operand
        raise IncompatibleCallbackError(f"Unsupported unary operator: {type(op).__name__}")

    def _boolop(self, op: ast.AST, values: List[Any], scope: _Scope) -> Any:
        if isinstance(op, ast.And):
            for v in values:
                tv = _truthy(self._eval(v, scope))
                if tv is False:
                    return False
                if tv is True:
                    continue
            return True
        if isinstance(op, ast.Or):
            for v in values:
                tv = _truthy(self._eval(v, scope))
                if tv is True:
                    return True
            return False
        raise IncompatibleCallbackError(f"Unsupported boolean op: {type(op).__name__}")

    def _compare(self, node: ast.Compare, scope: _Scope) -> Any:
        left = self._eval(node.left, scope)
        for op, comp_node in zip(node.ops, node.comparators):
            right = self._eval(comp_node, scope)
            ok = self._apply_compare(op, left, right)
            if not ok:
                return False
            left = right
        return True

    def _apply_compare(self, op: ast.AST, left: Any, right: Any) -> bool:
        if isinstance(op, ast.Eq):
            return bool(left == right)
        if isinstance(op, ast.NotEq):
            return bool(left != right)
        if isinstance(op, ast.Lt):
            return bool(left < right)
        if isinstance(op, ast.LtE):
            return bool(left <= right)
        if isinstance(op, ast.Gt):
            return bool(left > right)
        if isinstance(op, ast.GtE):
            return bool(left >= right)
        if isinstance(op, ast.Is):
            return left is right
        if isinstance(op, ast.IsNot):
            return left is not right
        if isinstance(op, ast.In):
            return left in right
        if isinstance(op, ast.NotIn):
            return left not in right
        return False

    # -- Attribute access --------------------------------------------------
    def _attr(self, node: ast.Attribute, scope: _Scope) -> Any:
        obj = self._eval(node.value, scope)
        attr = node.attr
        if isinstance(obj, _Missing):
            return _Missing(f"{obj.name}.{attr}")
        return getattr(obj, attr)

    # -- Calls: this is where compute boundaries are detected --------------
    def _call(self, node: ast.Call, scope: _Scope) -> Any:
        func = node.func

        # Special-case: getattr(...) -> IncompatibleCallbackError
        if isinstance(func, ast.Name) and func.id == "getattr":
            raise IncompatibleCallbackError(f"getattr() is not supported (dynamic dispatch) at line {node.lineno}")
        if isinstance(func, ast.Name) and func.id in {"exec", "eval", "compile"}:
            raise IncompatibleCallbackError(
                f"{func.id}() is not supported in precompute analysis at line {node.lineno}"
            )

        # ---- Materialization detection (np.array/asarray/save on dask arrays) ----
        if isinstance(func, ast.Attribute):
            recv = func.value
            if isinstance(recv, ast.Name) and recv.id == "np" and func.attr in _MATERIALIZING_FUNCS:
                arg_vals = [self._eval(a, scope) for a in node.args]
                if any(isinstance(v, da.Array) for v in arg_vals):
                    self.had_materialization = True
                    self.boundaries.append({"kind": "materialize", "lineno": node.lineno, "func": f"np.{func.attr}"})
                    return _Missing(f"np.{func.attr}(...)")

        # ---- Compute boundary: client.compute(...) / client.submit(...) ----
        # We don't know the client's identity statically; the user
        # typically does ``client = get_client()``. We treat any
        # ``.compute``/``.submit`` method call on a non-dask-array
        # receiver as a compute boundary and register any dask arrays
        # found in the arguments.
        if isinstance(func, ast.Attribute) and func.attr in {"compute", "submit"}:
            recv_value = self._eval(func.value, scope)
            if not isinstance(recv_value, da.Array):
                args = [self._eval(a, scope) for a in node.args]
                for a in args:
                    self._register_args_as_dask_arrays(a, f"client.{func.attr}", node.lineno)
                self.boundaries.append({"kind": func.attr, "lineno": node.lineno, "func": f"client.{func.attr}"})
                return _Missing(f"client.{func.attr}(...)")
            # Otherwise it's ``arr.compute()`` -- fall through to the
            # dask array method branch below, which will register the
            # boundary and return _Missing.

        # ---- Compute boundary: arr.compute() (receiver is a dask array) ----
        if isinstance(func, ast.Attribute) and func.attr == "compute":
            recv_value = self._eval(func.value, scope)
            if isinstance(recv_value, da.Array):
                self._register_compute_boundary(recv_value, "compute", node.lineno)
            return _Missing(f"{func.value}.compute()")

        # ---- dask/np submodule calls (e.g. da.fft.fft2(arr)) ----
        if isinstance(func, ast.Attribute):
            recv_value = self._eval(func.value, scope)
            # da.fft, da.linalg, etc. - forward to the submodule
            if isinstance(recv_value, type(da)) and recv_value is da:
                if func.attr in _DASK_RECURSE_SUBMODULES:
                    return getattr(recv_value, func.attr)
            # dask array method calls
            if isinstance(recv_value, da.Array):
                attr = func.attr
                if attr in {"compute", "persist"}:
                    # Already handled above
                    return _Missing(f"arr.{attr}()")
                kwargs = self._eval_kwargs(node.keywords, scope)
                args = [self._eval(a, scope) for a in node.args]
                try:
                    return getattr(recv_value, attr)(*args, **kwargs)
                except Exception as e:
                    # Dask may refuse to build the graph (e.g. FFT on
                    # multi-chunk axes, slicing out of bounds, etc.).
                    # Treat as opaque -- the surrounding code keeps working
                    # and the absence of a compute boundary in this branch
                    # is reported as NoPrecomputableReductionError.
                    logger.debug("dask method call failed: %s", e)
                    return _Missing(f"arr.{attr}(...)")
            # Forward to the object (e.g. arr.shape, op.func, da.fft.fft2).
            # Dask operations like da.fft.fft2(arr) may raise on stub
            # arrays (e.g. multi-chunk axes); we catch and degrade to
            # _Missing so analysis can continue.
            kwargs = self._eval_kwargs(node.keywords, scope)
            args = [self._eval(a, scope) for a in node.args]
            try:
                return getattr(recv_value, func.attr)(*args, **kwargs)
            except Exception as e:
                logger.debug("call failed: %s", e)
                return _Missing(f"{recv_value}.{func.attr}(...)")

        # ---- bare-name calls ----
        if isinstance(func, ast.Name):
            name = func.id
            if name in {"da", "np", "dask_array"}:
                raise IncompatibleCallbackError(f"Calling {name}() directly is not supported at line {node.lineno}")
            if name in {"sum", "min", "max", "abs", "round", "len", "int", "float", "bool"}:
                args = [self._eval(a, scope) for a in node.args]
                kwargs = self._eval_kwargs(node.keywords, scope)
                if name == "sum" and args and isinstance(args[0], da.Array):
                    return args[0].sum(**kwargs)
                if name == "min" and args and isinstance(args[0], da.Array):
                    return args[0].min(**kwargs)
                if name == "max" and args and isinstance(args[0], da.Array):
                    return args[0].max(**kwargs)
                if name == "len":
                    return len(args[0]) if args else 0
                if name in {"int", "float", "bool"}:
                    return args[0] if args else 0
                # round, abs: forward
                return getattr(args[0] if args else None, name)() if args else None
            if name == "slice":
                args = [self._eval(a, scope) for a in node.args]
                return slice(*args)
            if name == "tuple":
                args = [self._eval(a, scope) for a in node.args]
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    return tuple(args[0])
                return tuple(args)
            if name == "list":
                args = [self._eval(a, scope) for a in node.args]
                if len(args) == 1 and isinstance(args[0], (list, tuple)):
                    return list(args[0])
                return list(args)
            if name == "getattr":
                raise IncompatibleCallbackError(f"getattr() is not supported at line {node.lineno}")

            # User-defined helper (same file or registered)
            helper_def = self.source_file.find_function(name)
            if helper_def is not None:
                return self._call_helper(helper_def, node, scope)
            raise IncompatibleCallbackError(
                f"Cannot resolve function {name!r} at line {node.lineno}: "
                "not defined in source and not a known dask/np op"
            )

        # obj.method() where obj is a complex expression
        if isinstance(func, ast.Attribute):
            obj = self._eval(func.value, scope)
            kwargs = self._eval_kwargs(node.keywords, scope)
            args = [self._eval(a, scope) for a in node.args]
            return getattr(obj, func.attr)(*args, **kwargs)

        raise IncompatibleCallbackError(f"Unsupported call form: {type(func).__name__} at line {node.lineno}")

    def _eval_kwargs(self, keywords: List[ast.keyword], scope: _Scope) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for kw in keywords:
            if kw.arg is None:
                raise IncompatibleCallbackError(f"**kwargs expansion is not supported at line {kw.lineno}")
            result[kw.arg] = self._eval(kw.value, scope)
        return result

    # -- Helpers -----------------------------------------------------------
    def _register_compute_boundary(self, darr: da.Array, kind: str, lineno: int) -> None:
        if isinstance(darr, da.Array):
            self.dask_arrays.append({"array": darr, "kind": kind, "lineno": lineno})
            self.boundaries.append({"kind": kind, "lineno": lineno, "func": "compute"})

    def _register_args_as_dask_arrays(self, value: Any, kind: str, lineno: int) -> None:
        """Recursively register dask arrays found in a boundary argument.

        Handles:
        - a single dask array
        - a list/tuple of dask arrays
        - other values (skipped silently)
        """
        if isinstance(value, da.Array):
            self.dask_arrays.append({"array": value, "kind": kind, "lineno": lineno})
            return
        if isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, da.Array):
                    self.dask_arrays.append({"array": v, "kind": kind, "lineno": lineno})
            return
        # _Missing / _UnboundParam / other: skip. We only need to find
        # the dask arrays being computed.

    def _call_helper(self, helper_def: ast.FunctionDef, node: ast.Call, scope: _Scope) -> Any:
        helper_scope = scope.child()
        args_nodes = list(node.args)
        kwargs_nodes = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}

        # Bind positional args
        for i, param in enumerate(helper_def.args.args):
            if i < len(args_nodes):
                value = self._eval(args_nodes[i], scope)
            elif param.arg in kwargs_nodes:
                value = self._eval(kwargs_nodes[param.arg], scope)
            elif param.arg in scope.bindings:
                value = scope.get(param.arg)
            else:
                raise IncompatibleCallbackError(
                    f"Helper {helper_def.name!r} parameter {param.arg!r} is not bound at line {node.lineno}"
                )
            helper_scope.set(param.arg, value)

        # *args, **kwargs in helper: not supported
        if helper_def.args.vararg or helper_def.args.kwarg or helper_def.args.kwonlyargs:
            raise IncompatibleCallbackError(
                f"Helper {helper_def.name!r} uses *args/**kwargs; not supported at line {node.lineno}"
            )

        # Defaults
        defaults = helper_def.args.defaults
        positional_args = helper_def.args.args
        for i, default in enumerate(defaults):
            param = positional_args[len(positional_args) - len(defaults) + i]
            if param.arg not in helper_scope.bindings:
                helper_scope.set(param.arg, self._eval(default, scope))

        # Execute the helper body in helper_scope
        return_value = None
        for stmt in helper_def.body:
            if isinstance(stmt, ast.Return):
                if stmt.value is not None:
                    return_value = self._eval(stmt.value, helper_scope)
                break
            self.walk_stmt(stmt, helper_scope)
        return return_value


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _truthy(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str, list, tuple, np.ndarray)):
        return bool(value)
    if isinstance(value, da.Array):
        # Don't materialize; assume truthy? Be safe and walk both branches
        return None
    if isinstance(value, _Missing):
        return None
    return None
