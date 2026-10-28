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
The DEISA lazy expression tree: pointwise nodes, reduction nodes and their metadata.

Every value a callback can build out of a DEISA collection is one of these objects. They are NOT dask collections and
NOT ``dask.array.Array`` subclasses: a ``da.Array`` subclass loses its identity on the first derived expression
(``(m + 1)`` is a plain ``Array``), so owning the operator surface is the only way the whole expression stays
inspectable. That is also why there is no ``__dask_graph__`` here: an external dask graph must fail fast instead of
silently materializing the chunk.

Two node kinds cover the closed v1 surface:

- :class:`PointwiseExpr` -- a pointwise tree over ONE root (``(collection * 2 - 1).sqrt()``). Constants are ordinary
  program leaves, because the tree is built through our own API and needs no extraction step.
- :class:`ReductionExpr` -- one associative reduction over a pointwise tree, carrying the ``op_name`` / ``axes`` /
  ``kind`` / ``finalize`` payload the wire request is composed from.

A node's ``program`` is a nested tuple of plain picklable values -- ``("chunk",)``, ``("const", value)`` or
``(op, child, ...)`` -- and :func:`eval_pointwise` executes it against a numpy chunk. That tuple is what crosses the
bridge boundary inside the composed ``branch_func``, so it must stay free of lambdas, closures and bound methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from deisa.dask import runtime_errors as errors

# Kinds of reduction payload. The first three are the values the existing combine algebra already understands; the last
# two are the F4 extension (any/all -> boolean, count -> count). The combine tables must gain the two new kinds before
# P2 delivers them -- P1 only composes and verifies the per-bridge partials.
KIND_SCALAR = "scalar"
KIND_MEAN = "mean"
KIND_MOMENT = "moment"
KIND_BOOLEAN = "boolean"
KIND_COUNT = "count"

# The closed v1 reduction whitelist and the payload each one produces.
_REDUCTION_KINDS = {
    "sum": KIND_SCALAR,
    "prod": KIND_SCALAR,
    "min": KIND_SCALAR,
    "max": KIND_SCALAR,
    "count": KIND_COUNT,
    "any": KIND_BOOLEAN,
    "all": KIND_BOOLEAN,
    "mean": KIND_MEAN,
    "var": KIND_MOMENT,
    "std": KIND_MOMENT,
}

# Post-combine finishing step, mirroring the surviving wire contract (``sqrt`` for a standard deviation).
_REDUCTION_FINALIZERS = {"std": "sqrt"}

# Reductions that are refused as NON-ASSOCIATIVE instead of as generic off-surface calls: they need the global data
# ordering, which no bridge can see. Naming them separately is what fixes today's misdiagnosis (they used to be refused
# as raw field reads, which pointed at a remedy that does not help).
NON_ASSOCIATIVE_OPS = frozenset({"argmax", "argmin", "median", "percentile", "quantile"})

# The pointwise whitelist. A name here is reachable through the collection/expression API and is executed on the bridge
# chunk by :func:`eval_pointwise`; a name that is NOT here is refused, never silently converted.
_POINTWISE_FUNCS = {
    # Binary arithmetic.
    "add": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "truediv": np.true_divide,
    "floordiv": np.floor_divide,
    "mod": np.mod,
    "pow": np.power,
    # Comparisons.
    "eq": np.equal,
    "ne": np.not_equal,
    "lt": np.less,
    "le": np.less_equal,
    "gt": np.greater,
    "ge": np.greater_equal,
    # Unary.
    "abs": np.abs,
    "neg": np.negative,
    "pos": np.positive,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "log2": np.log2,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "square": np.square,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "tanh": np.tanh,
    # Selection.
    "clip": np.clip,
    "where": np.where,
}

_COMPARISON_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge"})

# The program leaf that stands for "the chunk this branch_func is called with on the bridge".
_CHUNK_PROGRAM: Tuple[Any, ...] = ("chunk",)

# Public, enumerable views of the surface: the test suite asserts that every entry here is reachable and refused
# otherwise (risk R2 -- the surface must be genuinely closed, not accidentally narrow).
POINTWISE_OPS = frozenset(_POINTWISE_FUNCS)
REDUCTION_OPS = frozenset(_REDUCTION_KINDS)


@dataclass(frozen=True)
class Root:
    """Provenance of one lazy operand: the registered array AND the timestep it was delivered for.

    A branch is rooted in exactly one root, which is what makes ``window[-1] - window[-2]`` refusable: the two operands
    share an array name but not a timestep.
    """

    array_name: str
    timestep: int


def _node(operand: Any) -> Tuple[Any, ...]:
    """Encode one operand as a program node: a lazy operand contributes its own program, anything else a constant."""
    if isinstance(operand, LazyOperand):
        return operand.program
    if isinstance(operand, (bool, int, float, complex, np.generic, np.ndarray, list, tuple)):
        return ("const", operand)
    raise errors.off_surface(
        repr(operand), "a constant operand must be a number, a sequence or a numpy array the bridge can unpickle"
    )


def eval_pointwise(program: Tuple[Any, ...], chunk: Any) -> Any:
    """Execute a pointwise program against a numpy chunk.

    This is the whole bridge-side cost of a pointwise chain: one recursive walk, no graph, no analysis. Every operator
    it can reach is in :data:`_POINTWISE_FUNCS`.

    - ``:param program:`` Program node built by the expression tree (``("chunk",)``, ``("const", value)`` or
      ``(op, child, ...)``).
    - ``:param chunk:`` The bridge's own numpy chunk, which is what the ``("chunk",)`` leaf evaluates to.
    """
    head = program[0]
    if head == "chunk":
        return chunk
    if head == "const":
        return program[1]
    func = _POINTWISE_FUNCS.get(head)
    if func is None:
        raise errors.precompute_runtime_invariant(
            "eval_pointwise", f"unknown pointwise operator {head!r} in a composed branch program"
        )
    return func(*[eval_pointwise(child, chunk) for child in program[1:]])


def _operand_shape(operand: Any) -> Tuple[int, ...]:
    """Shape metadata of one operand (a lazy operand's shape, or the shape numpy would broadcast a constant to)."""
    if isinstance(operand, LazyOperand):
        return operand.shape
    return tuple(np.shape(operand))


def _operand_dtype(operand: Any) -> np.dtype:
    """dtype metadata of one operand, without materializing anything."""
    if isinstance(operand, LazyOperand):
        return operand.dtype
    return np.asarray(operand).dtype


def _pointwise_metadata(op: str, operands: Tuple[Any, ...]) -> Tuple[Tuple[int, ...], np.dtype]:
    """Derived ``(shape, dtype)`` of one pointwise node: broadcast shape, numpy's own result-type rule."""
    try:
        shape = np.broadcast_shapes(*[_operand_shape(operand) for operand in operands])
    except ValueError as exc:
        raise errors.off_surface(
            f"the pointwise operator {op!r}",
            f"the operand shapes {[_operand_shape(o) for o in operands]} do not broadcast ({exc})",
        ) from None
    if op in _COMPARISON_OPS:
        dtype = np.dtype(bool)
    elif op == "where":
        dtype = np.result_type(_operand_dtype(operands[1]), _operand_dtype(operands[2]))
    else:
        dtype = np.result_type(*[_operand_dtype(operand) for operand in operands])
    return tuple(int(size) for size in shape), np.dtype(dtype)


def _reduction_dtype(op_name: str, dtype: np.dtype) -> np.dtype:
    """dtype metadata of a reduction over ``dtype`` inputs.

    Deliberately conservative: the values that matter on the wire (the per-bridge partial's shape and dtype) are PROBED
    from the composed callable at composition time, never taken from here.
    """
    dtype = np.dtype(dtype)
    if op_name in ("any", "all"):
        return np.dtype(bool)
    if op_name == "count":
        return np.dtype(np.int64)
    if op_name in ("mean", "var", "std"):
        return np.dtype(np.result_type(dtype, np.float64))
    if op_name in ("sum", "prod") and dtype.kind in "biu":
        # numpy promotes integer/bool accumulation to the platform int.
        return np.dtype(np.result_type(dtype, np.int64))
    return dtype


def _normalize_axes(axis: Any, ndim: int, symbol: str) -> Tuple[int, ...]:
    """Normalize a reduction ``axis`` argument into the wire convention, where ``()`` means the FULL reduction.

    ``axis=None`` (full), a tuple of axes and negative axes all collapse onto one canonical form, so two callbacks
    writing ``arr.sum()`` and ``arr.sum(axis=(0, 1))`` on a 2-D array share one key instead of planning one branch
    twice. An axis subset that covers every axis IS the full reduction and normalizes to ``()``.
    """
    if axis is None:
        return ()
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    if not axes:
        raise errors.off_surface(symbol, "an empty axis tuple reduces no axis at all")
    normalized = []
    for item in axes:
        try:
            value = int(item)
        except (TypeError, ValueError):
            raise errors.off_surface(symbol, f"the axis {item!r} is not an integer") from None
        if value < 0:
            value += ndim
        if not 0 <= value < ndim:
            raise errors.off_surface(symbol, f"the axis {item!r} is out of range for the {ndim}-D root")
        normalized.append(value)
    unique = tuple(sorted(set(normalized)))
    return () if len(unique) == ndim else unique


class LazyOperand:
    """Shared behaviour of every lazy DEISA operand: provenance, metadata, the surface and the refusal guards.

    Subclasses supply ``_root`` / ``_shape`` / ``_dtype`` / ``_program``; everything a callback can DO lives here, so
    the collection and every expression offer exactly the same closed surface.
    """

    __slots__ = ()

    # Reflected operators need this: it puts our ``__radd__``/``__rmul__`` ahead of numpy's own attempt to coerce the
    # operand (which would otherwise trip the ``__array__`` guard on ``2 * collection``).
    __array_priority__ = 1000

    # Comparing two operands yields an expression, exactly as numpy arrays do, so a lazy operand is not hashable.
    __hash__ = None  # type: ignore[assignment]

    @property
    def root(self) -> Root:
        """The ``(array name, timestep)`` pair this operand is rooted in."""
        return self._root

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the value this operand would deliver."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """dtype of the value this operand would deliver."""
        return self._dtype

    @property
    def ndim(self) -> int:
        """Number of dimensions of the value this operand would deliver."""
        return len(self._shape)

    @property
    def program(self) -> Tuple[Any, ...]:
        """The picklable program node this operand contributes to a composed ``branch_func``."""
        return self._program

    def _symbol(self) -> str:
        """Short description of this operand used as the offending-symbol half of a refusal message."""
        return type(self).__name__

    # -- Materialization guards --------------------------------------------------------------------------------------
    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        """Refuse numpy conversion: it is the silent-gather hole, and it leaves no graph artifact."""
        raise errors.numpy_materialization(self._symbol())

    def __iter__(self):
        """Refuse iteration: yielding elements one at a time materializes the field just as ``np.asarray`` does."""
        raise errors.iteration_materialization(self._symbol())

    def __bool__(self) -> bool:
        """Refuse a truth test: a lazy operand has no single-element answer."""
        raise errors.ambiguous_truth(self._symbol())

    def __len__(self) -> int:
        """Refuse ``len()``: a length is metadata, but accepting it invites indexing a shape that is never delivered."""
        raise errors.off_surface("len()", "the number of elements is metadata, but every raw-data access is refused")

    def __getattr__(self, name: str):
        """Refuse any attribute outside the surface, instead of degrading into an ``AttributeError``.

        Underscore-prefixed names keep raising :class:`AttributeError` on purpose: protocols (``hasattr(obj,
        "__dask_graph__")``, ``copy``, ``pickle``) probe them, and a refusal there would break introspection instead of
        protecting the user's data.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        raise errors.off_surface(f".{name}", "the attribute is not part of the closed DEISA operator surface")

    def __getitem__(self, key: Any) -> Any:
        """Refuse indexing: a slice or an index of a registered field is a raw read.

        Window callbacks are unaffected -- ``window`` is an ordinary Python list, so ``window[-1]`` is list indexing
        and delivers the collection it holds.
        """
        raise errors.off_surface(
            f"the item access {key!r}",
            "slicing and indexing read the field's raw data, which the reduction path never ships; index the delivered "
            "``window`` list instead, or deliver a reduction of the whole collection",
        )

    # -- Compute boundary --------------------------------------------------------------------------------------------
    def compute(self) -> Any:
        """Refuse computing a lazy operand directly: that is a raw read of the registered field."""
        raise errors.raw_field_read(
            f".compute() on {self._symbol()}", array_name=self._root.array_name, timestep=self._root.timestep
        )

    # -- Pointwise surface -------------------------------------------------------------------------------------------
    def _pointwise(self, op: str, *operands: Any) -> "PointwiseExpr":
        """Build one pointwise node over this operand and ``operands``, after checking the whole closed-surface rule."""
        if op not in _POINTWISE_FUNCS:
            raise errors.off_surface(f"the operator {op!r}", "the operator is not in the pointwise whitelist")
        all_operands = (self, *operands)
        for operand in all_operands:
            if isinstance(operand, ReductionExpr):
                raise errors.cross_reduction(
                    self._symbol(), f"it is combined with the completed reduction {operand.key!r}"
                )
            if isinstance(operand, LazyOperand) and operand.root != self._root:
                raise errors.mixed_root(self._symbol(), repr(self._root), repr(operand.root))
            _node(operand)
        shape, dtype = _pointwise_metadata(op, all_operands)
        return PointwiseExpr(op=op, operands=all_operands, root=self._root, shape=shape, dtype=dtype)

    def __add__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("add", other)

    def __radd__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("add", other)

    def __sub__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("sub", other)

    def __rsub__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("sub", other, self)

    def __mul__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("mul", other)

    def __rmul__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("mul", other)

    def __truediv__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("truediv", other)

    def __rtruediv__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("truediv", other, self)

    def __floordiv__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("floordiv", other)

    def __rfloordiv__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("floordiv", other, self)

    def __mod__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("mod", other)

    def __rmod__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("mod", other, self)

    def __pow__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("pow", other)

    def __rpow__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("pow", other, self)

    def __abs__(self) -> "PointwiseExpr":
        return self._pointwise("abs")

    def __neg__(self) -> "PointwiseExpr":
        return self._pointwise("neg")

    def __pos__(self) -> "PointwiseExpr":
        return self._pointwise("pos")

    def __eq__(self, other: Any) -> "PointwiseExpr":  # type: ignore[override]
        return self._pointwise("eq", other)

    def __ne__(self, other: Any) -> "PointwiseExpr":  # type: ignore[override]
        return self._pointwise("ne", other)

    def __lt__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("lt", other)

    def __le__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("le", other)

    def __gt__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("gt", other)

    def __ge__(self, other: Any) -> "PointwiseExpr":
        return self._pointwise("ge", other)

    def abs(self) -> "PointwiseExpr":
        """Elementwise absolute value."""
        return self._pointwise("abs")

    def sqrt(self) -> "PointwiseExpr":
        """Elementwise square root."""
        return self._pointwise("sqrt")

    def exp(self) -> "PointwiseExpr":
        """Elementwise exponential."""
        return self._pointwise("exp")

    def log(self) -> "PointwiseExpr":
        """Elementwise natural logarithm."""
        return self._pointwise("log")

    def log10(self) -> "PointwiseExpr":
        """Elementwise base-10 logarithm."""
        return self._pointwise("log10")

    def log2(self) -> "PointwiseExpr":
        """Elementwise base-2 logarithm."""
        return self._pointwise("log2")

    def floor(self) -> "PointwiseExpr":
        """Elementwise floor."""
        return self._pointwise("floor")

    def ceil(self) -> "PointwiseExpr":
        """Elementwise ceiling."""
        return self._pointwise("ceil")

    def sign(self) -> "PointwiseExpr":
        """Elementwise sign."""
        return self._pointwise("sign")

    def square(self) -> "PointwiseExpr":
        """Elementwise square."""
        return self._pointwise("square")

    def sin(self) -> "PointwiseExpr":
        """Elementwise sine."""
        return self._pointwise("sin")

    def cos(self) -> "PointwiseExpr":
        """Elementwise cosine."""
        return self._pointwise("cos")

    def tan(self) -> "PointwiseExpr":
        """Elementwise tangent."""
        return self._pointwise("tan")

    def tanh(self) -> "PointwiseExpr":
        """Elementwise hyperbolic tangent."""
        return self._pointwise("tanh")

    def clip(self, min: Any = None, max: Any = None) -> "PointwiseExpr":
        """Elementwise clamp to ``[min, max]`` (the ``min``/``max`` names follow ``np.clip``).

        - ``:param min:`` Lower bound, a number or a lazy operand on the same root. ``None`` disables it.
        - ``:param max:`` Upper bound, a number or a lazy operand on the same root. ``None`` disables it.
        """
        return self._pointwise("clip", min, max)

    def where(self, x: Any, y: Any) -> "PointwiseExpr":
        """Elementwise selection ``np.where(self, x, y)``: ``x`` where ``self`` is true, ``y`` elsewhere.

        - ``:param x:`` Value (or lazy operand on the same root) taken where ``self`` is true.
        - ``:param y:`` Value (or lazy operand on the same root) taken elsewhere.
        """
        return self._pointwise("where", x, y)

    # -- Reduction surface -------------------------------------------------------------------------------------------
    def _reduce(self, op_name: str, axis: Any = None, **kwargs: Any) -> "ReductionExpr":
        """Build the reduction node for ``op_name`` over this operand, after the closed-surface keyword checks."""
        if op_name in NON_ASSOCIATIVE_OPS:
            raise errors.non_associative(op_name, f".{op_name}()")
        kind = _REDUCTION_KINDS.get(op_name)
        if kind is None:
            raise errors.off_surface(f".{op_name}()", "the reduction is not in the associative whitelist")
        for keyword, value in kwargs.items():
            if keyword == "keepdims" and value:
                raise errors.keepdims_unsupported(f".{op_name}()")
            if keyword in ("dtype", "out") and value is not None:
                raise errors.dtype_out_unsupported(f".{op_name}()", keyword)
            if keyword == "ddof" and value not in (0, None):
                raise errors.ddof_unsupported(op_name, value, f".{op_name}()")
        axes = _normalize_axes(axis, self.ndim, f".{op_name}(axis={axis!r})")
        reduced = range(self.ndim) if not axes else axes
        shape = tuple(size for position, size in enumerate(self._shape) if position not in reduced)
        return ReductionExpr(
            operand=self,
            op_name=op_name,
            axes=axes,
            kind=kind,
            finalize=_REDUCTION_FINALIZERS.get(op_name),
            shape=shape,
            dtype=_reduction_dtype(op_name, self._dtype),
        )

    def sum(self, axis: Any = None, keepdims: bool = False, dtype: Any = None, out: Any = None) -> "ReductionExpr":
        """Associative sum over ``axis`` (full reduction when ``axis`` is ``None``)."""
        return self._reduce("sum", axis, keepdims=keepdims, dtype=dtype, out=out)

    def prod(self, axis: Any = None, keepdims: bool = False, dtype: Any = None, out: Any = None) -> "ReductionExpr":
        """Associative product over ``axis``."""
        return self._reduce("prod", axis, keepdims=keepdims, dtype=dtype, out=out)

    def min(self, axis: Any = None, keepdims: bool = False, out: Any = None) -> "ReductionExpr":
        """Associative minimum over ``axis``."""
        return self._reduce("min", axis, keepdims=keepdims, out=out)

    def max(self, axis: Any = None, keepdims: bool = False, out: Any = None) -> "ReductionExpr":
        """Associative maximum over ``axis``."""
        return self._reduce("max", axis, keepdims=keepdims, out=out)

    def mean(self, axis: Any = None, keepdims: bool = False, dtype: Any = None, out: Any = None) -> "ReductionExpr":
        """Associative mean over ``axis`` (delivered as the surviving ``mean`` dict kind)."""
        return self._reduce("mean", axis, keepdims=keepdims, dtype=dtype, out=out)

    def var(
        self, axis: Any = None, ddof: int = 0, keepdims: bool = False, dtype: Any = None, out: Any = None
    ) -> "ReductionExpr":
        """Associative variance over ``axis`` (``ddof=0`` only: the cross-bridge moment combine is fixed at 0)."""
        return self._reduce("var", axis, keepdims=keepdims, dtype=dtype, out=out, ddof=ddof)

    def std(
        self, axis: Any = None, ddof: int = 0, keepdims: bool = False, dtype: Any = None, out: Any = None
    ) -> "ReductionExpr":
        """Associative standard deviation over ``axis`` (``ddof=0`` only; finalized with ``sqrt`` after the combine)."""
        return self._reduce("std", axis, keepdims=keepdims, dtype=dtype, out=out, ddof=ddof)

    def any(self, axis: Any = None, keepdims: bool = False, out: Any = None) -> "ReductionExpr":
        """Associative logical OR over ``axis`` (boolean kind)."""
        return self._reduce("any", axis, keepdims=keepdims, out=out)

    def all(self, axis: Any = None, keepdims: bool = False, out: Any = None) -> "ReductionExpr":
        """Associative logical AND over ``axis`` (boolean kind)."""
        return self._reduce("all", axis, keepdims=keepdims, out=out)

    def count(self, axis: Any = None, keepdims: bool = False) -> "ReductionExpr":
        """Count of NON-ZERO elements over ``axis`` (count kind; the fold is additive, so it merges by summation)."""
        return self._reduce("count", axis, keepdims=keepdims)

    def argmax(self, *args: Any, **kwargs: Any) -> "ReductionExpr":
        """Refused: ``argmax`` depends on the global data ordering."""
        raise errors.non_associative("argmax", ".argmax()")

    def argmin(self, *args: Any, **kwargs: Any) -> "ReductionExpr":
        """Refused: ``argmin`` depends on the global data ordering."""
        raise errors.non_associative("argmin", ".argmin()")

    def median(self, *args: Any, **kwargs: Any) -> "ReductionExpr":
        """Refused: ``median`` depends on the global data ordering."""
        raise errors.non_associative("median", ".median()")

    def quantile(self, *args: Any, **kwargs: Any) -> "ReductionExpr":
        """Refused: ``quantile`` depends on the global data ordering."""
        raise errors.non_associative("quantile", ".quantile()")

    def percentile(self, *args: Any, **kwargs: Any) -> "ReductionExpr":
        """Refused: ``percentile`` depends on the global data ordering."""
        raise errors.non_associative("percentile", ".percentile()")


class PointwiseExpr(LazyOperand):
    """One node of a pointwise expression tree rooted at a single DEISA root.

    - ``:param op:`` Operator name, a key of the pointwise whitelist.
    - ``:param operands:`` Operand tuple: lazy operands on the same root and/or constants.
    - ``:param root:`` The root every lazy operand in the tree shares.
    - ``:param shape:`` Broadcast shape of the node.
    - ``:param dtype:`` numpy result dtype of the node.
    """

    __slots__ = ("op", "operands", "_root", "_shape", "_dtype", "_program")

    def __init__(
        self,
        op: str,
        operands: Tuple[Any, ...],
        root: Root,
        shape: Tuple[int, ...],
        dtype: Any,
    ) -> None:
        self.op = op
        self.operands = tuple(operands)
        self._root = root
        self._shape = tuple(int(size) for size in shape)
        self._dtype = np.dtype(dtype)
        self._program = (op,) + tuple(_node(operand) for operand in self.operands)

    def _symbol(self) -> str:
        return f"the pointwise expression {self.op!r}"

    def __repr__(self) -> str:
        return (
            f"PointwiseExpr(op={self.op!r}, shape={self._shape}, dtype={self._dtype}, "
            f"root={self._root.array_name!r}@{self._root.timestep})"
        )


class ReductionExpr(LazyOperand):
    """One associative reduction over a pointwise expression, dispatchable as a single wire request.

    It is NOT an array: it delivers one value and has no chunk-local partial another expression could fold. That is
    exactly what the guards below refuse -- using it as a pointwise operand, or reducing it a second time.

    - ``:param operand:`` The pointwise expression (or the bare collection) being reduced.
    - ``:param op_name:`` Canonical reduction name, a key of the associative whitelist.
    - ``:param axes:`` Reduced data axes; ``()`` means the FULL reduction (the wire convention).
    - ``:param kind:`` Payload kind the combine step dispatches on.
    - ``:param finalize:`` Post-combine step (``"sqrt"`` for std, else ``None``).
    - ``:param shape:`` Shape of the delivered value.
    - ``:param dtype:`` dtype of the delivered value.
    """

    __slots__ = ("operand", "op_name", "axes", "kind", "finalize", "_root", "_shape", "_dtype", "_program")

    def __init__(
        self,
        operand: LazyOperand,
        op_name: str,
        axes: Tuple[int, ...],
        kind: str,
        finalize: Any,
        shape: Tuple[int, ...],
        dtype: Any,
    ) -> None:
        self.operand = operand
        self.op_name = op_name
        self.axes = tuple(axes)
        self.kind = kind
        self.finalize = finalize
        self._root = operand.root
        self._shape = tuple(int(size) for size in shape)
        self._dtype = np.dtype(dtype)
        self._program = ("reduce", op_name, self.axes, operand.program)

    @property
    def key(self) -> str:
        """Stable dedup key of this reduction: ``{array}-{op}`` plus an axis discriminator when it is needed.

        The rule is the surviving wire-naming contract: a full reduction of the BARE registered array keeps the plain
        ``{array}-{op}`` key, a full reduction of a derived expression appends ``-axisall``, and an axis subset appends
        ``-axis{a}x{b}``. Identical expressions therefore share a key (one branch, one bridge execution); different
        reduction shapes never do.

        Note for the dispatch step: the key names the array, the operator and the reduced axes, not the pointwise
        program, so two DIFFERENT derived expressions with the same op and axes share a key. Per-object dispatch (F2)
        sends one request per reduction object, so both are still served; a key that must namespace two simultaneous
        bridge payloads has to carry the program digest.
        """
        derived = self.operand.program != _CHUNK_PROGRAM
        if not self.axes:
            suffix = "-axisall" if derived else ""
        else:
            suffix = "-axis" + "x".join(str(axis) for axis in self.axes)
        return f"{self._root.array_name}-{self.op_name}{suffix}"

    def _symbol(self) -> str:
        return f"the reduction {self.key!r}"

    def compute(self) -> Any:
        """Refuse in P1: the dispatch that answers this call is P2's.

        - ``:raises NotImplementedError:`` always, with the reason and the path that will implement it.
        """
        raise NotImplementedError(
            f"{self._symbol()} cannot be computed yet: the per-object dispatch boundary (.compute() -> wire request -> "
            f"bridge partials -> combine) is P2 of the local-reduction redesign. Per-object dispatch is the locked "
            f"batching decision, so there is no registry to fall back on. Use compose_request() to inspect the exact "
            f"request this reduction will send."
        )

    def _pointwise(self, op: str, *operands: Any) -> "PointwiseExpr":
        raise errors.cross_reduction(self._symbol(), "it is used as an operand of a pointwise expression")

    def _reduce(self, op_name: str, axis: Any = None, **kwargs: Any) -> "ReductionExpr":
        raise errors.cross_reduction(self._symbol(), f"it is reduced a second time with {op_name!r}")

    def __repr__(self) -> str:
        return (
            f"ReductionExpr(key={self.key!r}, kind={self.kind!r}, axes={self.axes}, finalize={self.finalize!r}, "
            f"shape={self._shape}, dtype={self._dtype})"
        )
