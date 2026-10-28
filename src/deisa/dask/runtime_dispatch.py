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
Dispatch a reduction to the bridges, then merge the partials they send back.

This module is the compute boundary of the runtime reduction path. ``ReductionExpr.compute()`` composes the slim wire
request (:func:`deisa.dask.runtime_compose.compose_request`), hands it to :class:`PlanDispatcher`, and gets a concrete
value back. Nothing else happens: the dispatcher sends the request, matches one report per bridge, and merges the
partials with the combine algebra that already exists in :mod:`deisa.dask.branch` (imported, never copied).

**The chunk never crosses.** A bridge runs the composed callable on its OWN chunk and reports the partial it produced;
the analytics host only ever holds per-bridge partials.

Two messages carry the exchange:

:class:`PlanRequest`
    ``request_id``, ``array_name``, ``timestep`` and the slim request. The id is unique per dispatch, so two requests
    for the same reduction stay distinguishable on the wire.
:class:`PartialReport`
    one bridge's answer for one ``request_id``: the partial (with its probed shape/dtype) **or** the error that replaced
    it. A failing bridge never ships a partial.

Both the request id and the timestep are matched on the way back. That is the stale-timestep guard: a partial computed
for an earlier request, or for an earlier timestep of the same array, never satisfies a newer request -- it is refused,
because merging it would silently mix two timesteps of one field.

**How the two messages travel in production.** Over a pair of dask queues per array, named by convention
(:func:`plan_request_queue_name` / :func:`plan_report_queue_name`), which is the reverse channel the run already has:
the bridge side is a daemon thread per array's root bridge (:func:`serve_plan_queue`) that moves requests into a
:class:`PlanRequestBacklog`, and the array's next ``send`` boundary -- the one place where every rank of the array is
inside the same sub-comm collective -- folds the retained chunk of the requested timestep and posts the reports back
(:class:`QueuePlanExchange` is the host's half). Nothing is exchanged to agree on the names, and no bridge folds a
chunk from a thread: a second thread inside an MPI communicator is not a safe place to be.

:class:`EagerMenu` is the eager half of the transfer strategy: the six tiny scalar full folds a bridge computes once for
its chunk, so the common scalar reduction costs no round trip. A menu value IS the payload an on-demand request for the
same operation would return (same composed callable, same chunk) -- **for a reduction of the bare registered array**,
which is the program the menu is built from. A derived operand (``(collection * 2).sum()``) needs partials computed
from its own program, so it never reads the menu and always dispatches on demand; :class:`PlanDispatcher` gates the
menu path on the operand program for exactly that reason.

The same exchange carries the estimator-side PCA branch: :meth:`PlanDispatcher.compute_pca` dispatches a
:class:`~deisa.dask.runtime_compose.PCARequest`, whose per-bridge partial is a mergeable
:class:`~deisa.dask.mergeable_pca.PCASummary`, and merges those summaries at the root. A PCA payload cannot go through
the reduction algebra, which is why it has its own request type and its own merge entry point.
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from deisa.dask import runtime_errors as errors
from deisa.dask.branch import _BRANCH_KIND_PCA, _SCALAR_FOLDS, _combine_two_phase
from deisa.dask.constants import RUNTIME_PLAN_QUEUE_PREFIX, RUNTIME_REPORT_QUEUE_PREFIX
from deisa.dask.mergeable_pca import PCASummary, merge_tree
from deisa.dask.runtime_compose import PCARequest, RuntimeRequest, compose_request
from deisa.dask.runtime_expr import _CHUNK_PROGRAM, KIND_BOOLEAN, KIND_COUNT, KIND_SCALAR

__all__ = [
    "EAGER_MENU_OPS",
    "EagerMenu",
    "PartialReport",
    "PlanDispatcher",
    "PlanRequest",
    "PlanRequestBacklog",
    "PlanResponder",
    "QueuePlanExchange",
    "RUNTIME_REDUCTION_DELIVERY_ENABLED",
    "build_eager_menu",
    "dispatch_plan_over_comm",
    "eager_menu_requests",
    "merge_partials",
    "plan_report_queue_name",
    "plan_request_queue_name",
    "runtime_delivery_enabled",
    "serve_plan_queue",
    "serve_plan_requests",
]

# Internal delivery flag of the runtime reduction path: an environment variable read ONCE at import, deliberately not a
# user parameter. It selects the experimental delivery mode while the analyzer-driven one is still present; the default
# is off, so nothing changes until a deployment sets it.
RUNTIME_REDUCTION_DELIVERY_ENABLED: bool = os.environ.get("DEISA_DASK_RUNTIME_REDUCTION", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)


def runtime_delivery_enabled() -> bool:
    """Whether the runtime reduction delivery path is selected, read through the module so a test can flip it."""
    return RUNTIME_REDUCTION_DELIVERY_ENABLED


# The combine algebra ``_combine_two_phase`` is a ``dask.delayed`` function: a merge must not build a task graph, so the
# module calls the undecorated callable the decorator preserves on ``__wrapped__`` (the ``functools`` convention dask's
# ``delayed`` follows). Same algebra, no scheduler, no graph, no worker.
_combine_partials = getattr(_combine_two_phase, "__wrapped__", None)
if _combine_partials is None:  # pragma: no cover - only reachable if dask stops preserving the undecorated callable
    raise RuntimeError(
        "deisa.dask.runtime_dispatch: the combine algebra must stay callable without a task graph, but "
        "dask no longer exposes the undecorated _combine_two_phase."
    )
_COMBINE_PARTIALS: Callable[..., Any] = _combine_partials

# Kinds whose cross-bridge merge is a fold over one partial per bridge (the "fold kinds"). ``mean`` and ``moment`` are
# absent by design: their partials are the ``{n, total[, M]}`` dicts ``mean_agg``/``moment_agg`` consume.
_FOLD_KINDS = frozenset({KIND_SCALAR, KIND_BOOLEAN, KIND_COUNT})

# The post-combine step the wire request may ask for (``sqrt`` for a standard deviation).
_SQRT_FINALIZE = "sqrt"

# The eager scalar menu: the full folds small enough that a bridge can compute them once per timestep and ship them
# unprompted. ``mean``/``var`` travel as their dict partials, exactly as an on-demand request for them would.
EAGER_MENU_OPS: Tuple[str, ...] = ("sum", "min", "max", "mean", "var", "prod")


# ---------------------------------------------------------------------------------------------------------------------
# Protocol messages
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanRequest:
    """One on-demand branch request, addressed to every bridge that owns a chunk of ``array_name``.

    - ``:param request_id:`` Unique id of THIS dispatch. Every report is matched against it, so a late answer to an
      older request cannot satisfy this one.
    - ``:param array_name:`` Registered array the request is rooted in.
    - ``:param timestep:`` Timestep the bridge's chunk must belong to; a bridge holding another timestep refuses.
    - ``:param request:`` The slim wire request (a reduction's ``RuntimeRequest`` or a ``PCARequest``: ``key``,
      ``kind``, composed ``branch_func``, and the reduction's ``axes``/``finalize``).
    """

    request_id: str
    array_name: str
    timestep: int
    request: Union[RuntimeRequest, PCARequest]


@dataclass(frozen=True)
class PartialReport:
    """One bridge's answer to one :class:`PlanRequest`: the partial it produced, or the error that replaced it.

    - ``:param request_id:`` Id of the request this report answers (the stale-answer guard).
    - ``:param array_name:`` Registered array the partial belongs to.
    - ``:param timestep:`` Timestep the answering bridge's chunk belongs to.
    - ``:param rank:`` Rank of the answering bridge inside the array's communicator.
    - ``:param chunk_position:`` MPI coordinates of the answering bridge's chunk; the grid the merge walks.
    - ``:param partial:`` The per-bridge partial (scalar / array / ``{n, total[, M]}`` dict). ``None`` on an error.
    - ``:param shape:`` Shape of the partial's representative array.
    - ``:param dtype:`` dtype of the partial's representative array.
    - ``:param error:`` Description of the local failure that replaced the partial, else ``None``.
    """

    request_id: str
    array_name: str
    timestep: int
    rank: int
    chunk_position: Tuple[int, ...]
    partial: Any = None
    shape: Tuple[int, ...] = ()
    dtype: str = ""
    error: Optional[str] = None


@dataclass(frozen=True)
class EagerMenu:
    """The eager scalar menu of ONE bridge for ONE timestep (the zero-round-trip half of the transfer strategy).

    Every value is a full fold of the BARE registered array's chunk (the identity program), so the menu answers a
    reduction of the bare collection and nothing else: a reduction of a derived operand (``(c * 2).sum()``) must be
    dispatched, because its partials are folds of a different program.

    - ``:param array_name:`` Registered array the menu was computed for.
    - ``:param timestep:`` Timestep the bridge's chunk belonged to.
    - ``:param rank:`` Rank of the bridge inside the array's communicator.
    - ``:param chunk_position:`` MPI coordinates of the bridge's chunk.
    - ``:param values:`` ``op_name -> partial`` for every entry of :data:`EAGER_MENU_OPS`.
    """

    array_name: str
    timestep: int
    rank: int
    chunk_position: Tuple[int, ...]
    values: Mapping[str, Any]


def _partial_shape_dtype(value: Any) -> Tuple[Tuple[int, ...], str]:
    """Shape and dtype of the representative array inside one partial.

    Dict partials (``mean``/``moment``) are described by ``total`` (it always carries the reduction's output shape) or
    by ``M`` when ``total`` is absent. A PCA partial is described by its ``components`` array, the largest array a
    summary carries. Reading the shape off the value the bridge ACTUALLY produced is what keeps the payload metadata
    from drifting away from the payload.

    - ``:param value:`` The partial a bridge returned.
    """
    if isinstance(value, PCASummary):
        return _partial_shape_dtype(value.components)
    if isinstance(value, Mapping):
        for key in ("total", "M"):
            if key in value:
                return _partial_shape_dtype(value[key])
        if not value:
            raise errors.precompute_runtime_invariant(
                "_partial_shape_dtype", "a dict partial carried no entries to describe"
            )
        return _partial_shape_dtype(next(iter(value.values())))
    array = np.asarray(value)
    return tuple(int(size) for size in array.shape), str(array.dtype)


# ---------------------------------------------------------------------------------------------------------------------
# Bridge side
# ---------------------------------------------------------------------------------------------------------------------
class PlanResponder:
    """The bridge side of the plan protocol: it answers requests with the partial of ITS OWN chunk.

    One instance per bridge per array per timestep. ``branch_func(chunk)`` runs here and only what it returns is
    reported, so no code path in this class can ship the chunk.

    - ``:param array_name:`` Registered array this bridge holds a chunk of.
    - ``:param timestep:`` Timestep the held chunk belongs to.
    - ``:param chunk:`` The bridge's own numpy chunk.
    - ``:param rank:`` Rank of this bridge inside the array's communicator.
    - ``:param chunk_position:`` MPI coordinates of this bridge's chunk.
    """

    def __init__(
        self,
        array_name: str,
        timestep: int,
        chunk: Any,
        rank: int = 0,
        chunk_position: Sequence[int] = (),
    ) -> None:
        self.array_name = str(array_name)
        self.timestep = int(timestep)
        self.chunk = chunk
        self.rank = int(rank)
        self.chunk_position = tuple(int(value) for value in chunk_position)

    def respond(self, plan: PlanRequest) -> PartialReport:
        """Execute ``plan`` on the held chunk and report the partial -- or the error that replaced it.

        Fail-loud contract (the same one ``_execute_operations_on_chunk`` enforces): a branch that raises produces an
        ERROR report, never a partial. A dropped partial would silently corrupt the merge (a smaller sum, a mean
        missing a whole bridge's ``n``).

        - ``:param plan:`` The request to answer.
        """
        if plan.array_name != self.array_name:
            return self._error(
                plan,
                f"the request is rooted in array {plan.array_name!r}, but this bridge holds array {self.array_name!r}",
            )
        if plan.timestep != self.timestep:
            return self._error(
                plan,
                f"the request is for timestep {plan.timestep}, but the chunk this bridge holds belongs to timestep "
                f"{self.timestep}",
            )
        try:
            partial = plan.request.branch_func(self.chunk)
        except Exception as exc:
            return self._error(plan, f"branch {plan.request.key!r} failed on the local chunk: {exc!r}")
        shape, dtype = _partial_shape_dtype(partial)
        return PartialReport(
            request_id=plan.request_id,
            array_name=plan.array_name,
            timestep=plan.timestep,
            rank=self.rank,
            chunk_position=self.chunk_position,
            partial=partial,
            shape=shape,
            dtype=dtype,
        )

    def _error(self, plan: PlanRequest, detail: str) -> PartialReport:
        """Build the error report for ``plan``: the message that travels back instead of a partial."""
        return PartialReport(
            request_id=plan.request_id,
            array_name=plan.array_name,
            timestep=plan.timestep,
            rank=self.rank,
            chunk_position=self.chunk_position,
            error=detail,
        )


def build_eager_menu(
    collection: Any,
    chunk: Any,
    rank: int = 0,
    chunk_position: Sequence[int] = (),
) -> EagerMenu:
    """Compute the eager scalar menu on this bridge's own chunk.

    The menu composes the SAME slim requests an on-demand call would (one per :data:`EAGER_MENU_OPS` entry, full folds
    only) and runs them on the local chunk, so a menu value is byte-for-byte the partial the on-demand path returns and
    the merge cannot tell the two apart. The failure contract is the fail-loud one: a menu entry that raises aborts the
    menu instead of shipping a hole.

    - ``:param collection:`` The :class:`~deisa.dask.collection.DeisaCollection` describing this bridge's array.
    - ``:param chunk:`` The bridge's own numpy chunk.
    - ``:param rank:`` Rank of this bridge inside the array's communicator.
    - ``:param chunk_position:`` MPI coordinates of this bridge's chunk.
    """
    values = {}
    for op_name, request in zip(EAGER_MENU_OPS, eager_menu_requests(collection)):
        try:
            values[op_name] = request.branch_func(chunk)
        except Exception as exc:
            raise errors.precompute_runtime_invariant(
                "build_eager_menu", f"the {op_name!r} menu entry failed on the local chunk: {exc!r}"
            ) from exc
    return EagerMenu(
        array_name=collection.name,
        timestep=collection.timestep,
        rank=int(rank),
        chunk_position=tuple(int(value) for value in chunk_position),
        values=values,
    )


def eager_menu_requests(collection: Any) -> Tuple[RuntimeRequest, ...]:
    """The slim requests of the eager scalar menu, in :data:`EAGER_MENU_OPS` order.

    - ``:param collection:`` The :class:`~deisa.dask.collection.DeisaCollection` to compose the menu for.
    """
    return tuple(compose_request(getattr(collection, op_name)()) for op_name in EAGER_MENU_OPS)


def dispatch_plan_over_comm(comm: Any, plan: PlanRequest, responder: PlanResponder) -> List[PartialReport]:
    """Run one plan round from the array communicator's root: broadcast the request, collect every bridge's report.

    The collectives are the ones the branch filing already uses (a per-array sub-comm broadcast out, a gather back to
    rank 0), so the dispatch rides the communicator that is already wired instead of a new channel.

    - ``:param comm:`` The array's communicator (any ``ICommunicator``); the caller must be its root rank.
    - ``:param plan:`` The request to broadcast.
    - ``:param responder:`` The root bridge's responder (it answers with its own chunk, like every other bridge).
    """
    comm.bcast(plan, root=0)
    reports = comm.gather(responder.respond(plan), root=0)
    if reports is None:
        raise errors.precompute_runtime_invariant(
            "dispatch_plan_over_comm", f"the gather returned no reports for request {plan.request_id!r}"
        )
    return sorted(reports, key=lambda report: tuple(report.chunk_position))


def serve_plan_requests(comm: Any, responder: PlanResponder) -> None:
    """Answer every plan the root broadcasts, until it broadcasts the ``None`` stop sentinel.

    This is the non-root half of :func:`dispatch_plan_over_comm`: every participating bridge runs it, so no bridge needs
    to know which reductions a callback will ask for.

    - ``:param comm:`` The array's communicator (any ``ICommunicator``); the caller must NOT be its root rank.
    - ``:param responder:`` This bridge's responder.
    """
    while True:
        plan = comm.bcast(None, root=0)
        if plan is None:
            return
        comm.gather(responder.respond(plan), root=0)


# ---------------------------------------------------------------------------------------------------------------------
# Production transport: the run's dask queues
# ---------------------------------------------------------------------------------------------------------------------
def plan_request_queue_name(array_name: str) -> str:
    """Name of the queue carrying the analytics host's plan requests for one array.

    Named per array, like the feedback queues: the host and every bridge of the array derive the same name without
    exchanging it, so no configuration crosses the run.

    - ``:param array_name:`` Registered array the queue serves.
    """
    return f"{RUNTIME_PLAN_QUEUE_PREFIX}{array_name}"


def plan_report_queue_name(array_name: str) -> str:
    """Name of the queue carrying the bridges' per-bridge reports for one array.

    - ``:param array_name:`` Registered array the queue serves.
    """
    return f"{RUNTIME_REPORT_QUEUE_PREFIX}{array_name}"


# How long a bridge waits for another plan request before re-checking its stop flag, and how long the host waits for
# the reports of one request. The bridge bound is a poll interval (it costs one scheduler round trip); the host bound
# is the one that must tolerate the simulation taking a timestep to reach its next send boundary.
_PLAN_QUEUE_POLL_SECONDS: float = 0.5
_PLAN_REPORT_TIMEOUT_SECONDS: float = 300.0


class PlanRequestBacklog:
    """The hand-off between a bridge's plan-queue thread and its per-timestep ``send`` boundary.

    The queue thread appends, the send boundary drains, and the two never hold the lock at the same time as they touch
    a chunk. Plans are accumulated rather than answered on the spot because answering needs the array's other bridges:
    only the send boundary is a place where every rank of the array is parked in the same collective.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._plans: List[PlanRequest] = []

    def put(self, plan: PlanRequest) -> None:
        """Append one request for the next send boundary.

        - ``:param plan:`` The request the host's queue delivered.
        """
        with self._lock:
            self._plans.append(plan)

    def drain(self) -> List[PlanRequest]:
        """Take every request received so far, oldest first (an empty list when none arrived)."""
        with self._lock:
            plans, self._plans = self._plans, []
        return plans


class QueuePlanExchange:
    """The analytics host's transport: one request/reports round trip over the run's two dask queues.

    This is the production transport: the host puts the same request once per bridge (a dask queue hands every item
    to exactly ONE consumer, so a broadcast is one item per bridge), then blocks until every bridge answered. It is a
    plain callable, the shape :class:`PlanDispatcher` takes, so nothing on the host side knows about queues.

    The reports come back in whatever order the bridges answered; :func:`_validate_reports` and the merge both match
    on the request id and sort by chunk position, so the order is not part of the contract.

    - ``:param request_queue:`` The array's request queue (``put``-capable; in production a ``distributed.Queue``).
    - ``:param report_queue:`` The array's report queue (``get(timeout=...)``-capable).
    - ``:param bridge_count:`` Number of bridges that own a chunk of the array: one report is expected from each.
    - ``:param timeout:`` Seconds to wait for the full set of reports before refusing.
    """

    def __init__(
        self,
        request_queue: Any,
        report_queue: Any,
        bridge_count: int,
        timeout: float = _PLAN_REPORT_TIMEOUT_SECONDS,
    ) -> None:
        self._request_queue = request_queue
        self._report_queue = report_queue
        self._bridge_count = int(bridge_count)
        self._timeout = float(timeout)

    def __call__(self, plan: PlanRequest) -> List[PartialReport]:
        """Broadcast ``plan`` to every bridge and collect one report from each.

        - ``:param plan:`` The request to dispatch.
        """
        for _ in range(self._bridge_count):
            self._request_queue.put(plan)
        reports: List[PartialReport] = []
        try:
            while len(reports) < self._bridge_count:
                reports.append(self._report_queue.get(timeout=self._timeout))
        except TimeoutError as exc:
            raise errors.precompute_runtime_invariant(
                "QueuePlanExchange",
                f"request {plan.request_id!r} for array {plan.array_name!r} at timestep {plan.timestep} collected "
                f"{len(reports)} of {self._bridge_count} report(s) in {self._timeout}s; no bridge is serving plan "
                f"requests for that array. Refusing to merge a partial set with a hole in it",
            ) from exc
        return reports


def serve_plan_queue(
    request_queue: Any,
    backlog: PlanRequestBacklog,
    stop: Optional[threading.Event] = None,
    poll_seconds: float = _PLAN_QUEUE_POLL_SECONDS,
) -> None:
    """The bridge half of the transport: move plan requests off the host's queue into ``backlog``.

    Runs in a daemon thread on the array's root bridge and touches nothing but the queue and the backlog: no
    communicator call and no chunk. That is deliberate -- the chunk is folded by the bridge's MAIN thread at the next
    ``send`` boundary, because that is the only point where every rank of the array is inside the same collective, and
    because a second thread inside an MPI communicator is not a safe place to be.

    - ``:param request_queue:`` The array's request queue (``get(timeout=...)``-capable).
    - ``:param backlog:`` Where the requests are parked until the next send boundary.
    - ``:param stop:`` Set to end the loop; a ``None`` request on the queue also ends it.
    - ``:param poll_seconds:`` How long one ``get`` waits before the stop flag is re-checked.
    """
    while stop is None or not stop.is_set():
        try:
            plan = request_queue.get(timeout=poll_seconds)
        except TimeoutError:
            continue
        if plan is None:  # stop sentinel
            return
        backlog.put(plan)


# ---------------------------------------------------------------------------------------------------------------------
# Merge (the surviving combine algebra)
# ---------------------------------------------------------------------------------------------------------------------
def _grid_shape(reports: Sequence[PartialReport], ndim: int) -> Tuple[int, ...]:
    """Shape of the chunk grid the reports describe, after proving they cover it exactly once.

    Grid level ``i`` is data axis ``i`` (the one-chunk-per-bridge invariant), which is what makes an axis combine
    meaningful. A report whose coordinate rank disagrees with the array's rank, a duplicated chunk, or a hole in the
    grid all leave the grid<->data-axis mapping ambiguous -- refusing beats guessing.

    - ``:param reports:`` The reports to merge.
    - ``:param ndim:`` Number of data axes of the array.
    """
    coords = [tuple(int(value) for value in report.chunk_position) for report in reports]
    for coord in coords:
        if len(coord) != ndim:
            raise errors.precompute_runtime_invariant(
                "_grid_shape",
                f"a partial reports the chunk position {coord} but the array has {ndim} data axes; the "
                f"grid<->data-axis mapping is ambiguous",
            )
        if any(value < 0 for value in coord):
            raise errors.precompute_runtime_invariant("_grid_shape", f"the chunk position {coord} has a negative axis")
    if len(set(coords)) != len(coords):
        raise errors.precompute_runtime_invariant(
            "_grid_shape", f"two partials report the same chunk position: {sorted(coords)}"
        )
    grid_shape = tuple(max(coord[axis] for coord in coords) + 1 for axis in range(ndim))
    if int(np.prod(grid_shape)) != len(coords):
        raise errors.precompute_runtime_invariant(
            "_grid_shape",
            f"the chunk grid {grid_shape} describes {int(np.prod(grid_shape))} chunks but {len(coords)} partials were "
            f"delivered",
        )
    return grid_shape


def merge_partials(
    reports: Sequence[PartialReport],
    *,
    kind: str,
    op_name: str,
    axes: Tuple[int, ...],
    ndim: int,
    finalize: Optional[str],
) -> Any:
    """Merge the per-bridge partials of one reduction with the surviving combine algebra.

    The algebra itself is NOT reimplemented here: ``mean_agg``/``moment_agg`` dict combine, the ``_SCALAR_FOLDS``
    cross-bridge fold, the ``sqrt`` finalize and the grid<->data-axis validation all come from
    :mod:`deisa.dask.branch`. What this function adds is the ONE case that algebra cannot serve on this path: a full
    fold of a fold-kind ships the minimal scalar per bridge, and the tier-shaped combine squeezes reduced axes a 0-d
    partial does not have, so those scalars are folded directly.

    - ``:param reports:`` One report per bridge.
    - ``:param kind:`` Payload kind of the reduction.
    - ``:param op_name:`` Canonical reduction name.
    - ``:param axes:`` Reduced data axes; ``()`` means the full reduction.
    - ``:param ndim:`` Number of data axes of the array.
    - ``:param finalize:`` Post-combine step (``"sqrt"`` for a standard deviation), else ``None``.
    """
    if not reports:
        raise errors.precompute_runtime_invariant("merge_partials", "no per-bridge partial was delivered to merge")
    red_axes = tuple(range(ndim)) if not axes else tuple(int(axis) for axis in axes)
    kept_axes = tuple(axis for axis in range(ndim) if axis not in red_axes)
    grid_shape = _grid_shape(reports, ndim)
    out_dtype = reports[0].dtype or str(np.float64)
    if kind in _FOLD_KINDS and not kept_axes:
        fold = _SCALAR_FOLDS.get(op_name)
        if fold is None:
            raise errors.precompute_runtime_invariant(
                "merge_partials", f"no cross-bridge fold for the reduction {op_name!r}"
            )
        result = np.asarray(fold(np.stack([np.asarray(report.partial) for report in reports])))
        if finalize == _SQRT_FINALIZE:
            result = np.sqrt(result)
        return result
    combine_kind = "scalar" if kind in _FOLD_KINDS else kind
    return np.asarray(
        _COMBINE_PARTIALS(
            [(tuple(report.chunk_position), report.partial) for report in reports],
            kind=combine_kind,
            op_name=op_name,
            red_axes=red_axes,
            kept_axes=kept_axes,
            grid_shape=grid_shape,
            out_dtype=out_dtype,
            finalize=finalize,
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# Analytics side
# ---------------------------------------------------------------------------------------------------------------------
def _validate_reports(
    plan: PlanRequest, reports: Sequence[PartialReport], expected: Optional[int] = None
) -> List[PartialReport]:
    """Check every report answers THIS request, then trip on any error report.

    Two refusals live here and nowhere else: a report from another request id or another timestep is refused (never
    merged -- a stale partial would mix two timesteps of the same field), and a bridge error is re-raised on the
    analytics host so the callback sees it instead of a partial set with a hole in it.

    - ``:param plan:`` The request that was dispatched.
    - ``:param reports:`` The reports the exchange returned.
    - ``:param expected:`` Number of bridges the array has, when known; a short or long answer is refused, because a
      missing partial would silently shrink the merged value.
    """
    if not reports:
        raise errors.precompute_runtime_invariant(
            "PlanDispatcher", f"request {plan.request_id!r} returned no report from any bridge"
        )
    if expected is not None and len(reports) != expected:
        raise errors.precompute_runtime_invariant(
            "PlanDispatcher",
            f"request {plan.request_id!r} expected {expected} reports (one per bridge) but got {len(reports)}",
        )
    matched = []
    for report in reports:
        if report.request_id != plan.request_id:
            raise errors.precompute_runtime_invariant(
                "PlanDispatcher",
                f"request {plan.request_id!r} received a partial for request {report.request_id!r}; a result from "
                f"another request cannot satisfy this one",
            )
        if report.timestep != plan.timestep:
            raise errors.precompute_runtime_invariant(
                "PlanDispatcher",
                f"request {plan.request_id!r} is for timestep {plan.timestep} but bridge {report.rank} answered with a "
                f"partial of timestep {report.timestep}; refusing to mix two timesteps of array {plan.array_name!r}",
            )
        if report.array_name != plan.array_name:
            raise errors.precompute_runtime_invariant(
                "PlanDispatcher",
                f"request {plan.request_id!r} is rooted in array {plan.array_name!r} but bridge {report.rank} answered "
                f"for array {report.array_name!r}",
            )
        if report.error is not None:
            raise errors.precompute_runtime_invariant(
                "PlanDispatcher",
                f"bridge {report.rank} could not serve the reduction {plan.request.key!r} on its chunk: "
                f"{report.error}. Refusing to merge a partial set with a hole in it",
            )
        matched.append(report)
    return matched


def _finish(value: Any) -> Any:
    """Unwrap a 0-d result into its numpy scalar, the value ``dask`` itself delivers for a scalar reduction."""
    array = np.asarray(value)
    return array[()] if array.ndim == 0 else array


class PlanDispatcher:
    """The analytics-host side of the compute boundary: compose, dispatch (or read the menu), match, merge.

    ``exchange`` is the only transport knob: it takes one :class:`PlanRequest` and returns the reports every bridge
    produced for it. Keeping it a callable means this class owns no communicator dialect -- the same dispatcher is
    driven over the array's sub-comm in production and over a test communicator in the suite. It may be ``None``: a
    delivery that attaches the eager menu but no transport still serves every menu operation, and refuses the rest
    with the no-transport refusal instead of gathering anything.

    - ``:param exchange:`` Callable performing one request/reports round trip, or ``None`` when the delivery has no
      bridge transport.
    - ``:param bridge_count:`` Number of bridges that own a chunk of the array, when known. A round that returns another
      number of reports is refused: a missing partial would silently shrink the merged value.
    - ``:param menu:`` Optional eager menus already computed for the current timestep. A full fold of a menu operation
      is merged from them without touching ``exchange``.
    """

    def __init__(
        self,
        exchange: Optional[Callable[[PlanRequest], Sequence[PartialReport]]] = None,
        bridge_count: Optional[int] = None,
        menu: Optional[Sequence[EagerMenu]] = None,
    ) -> None:
        self._exchange = exchange
        self._bridge_count = None if bridge_count is None else int(bridge_count)
        self._menus: Dict[Tuple[str, int], List[EagerMenu]] = {}
        for eager in menu or ():
            self._menus.setdefault((eager.array_name, int(eager.timestep)), []).append(eager)

    def compute(self, reduction: Any) -> Any:
        """Deliver the concrete value of ``reduction``.

        - ``:param reduction:`` The :class:`~deisa.dask.runtime_expr.ReductionExpr` a callback computed.
        """
        request = compose_request(reduction)
        ndim = reduction.operand.ndim
        menus = self._menus.get((request.input_name, reduction.root.timestep))
        # The menu holds the BARE registered array's partials (``build_eager_menu`` composes the identity program), so
        # it may only answer a reduction whose operand IS the bare chunk program. A derived operand (``(c * 2).sum()``)
        # needs its own partials computed from the derived program: answering it from the menu would silently return
        # the un-derived field's value, with no round trip and no exception.
        if (
            menus
            and not request.axes
            and reduction.operand.program == _CHUNK_PROGRAM
            and all(reduction.op_name in eager.values for eager in menus)
        ):
            reports = self._menu_reports(menus, reduction.op_name)
            if self._bridge_count is not None and len(reports) != self._bridge_count:
                raise errors.precompute_runtime_invariant(
                    "PlanDispatcher",
                    f"the eager menu of array {request.input_name!r} at timestep {reduction.root.timestep} covers "
                    f"{len(reports)} bridge(s) but the array has {self._bridge_count}",
                )
        else:
            plan = PlanRequest(
                request_id=uuid.uuid4().hex,
                array_name=request.input_name,
                timestep=reduction.root.timestep,
                request=request,
            )
            reports = self._reports_for(plan)
        return _finish(
            merge_partials(
                reports,
                kind=request.kind,
                op_name=reduction.op_name,
                axes=request.axes,
                ndim=ndim,
                finalize=request.finalize,
            )
        )

    def compute_pca(self, request: PCARequest) -> PCASummary:
        """Dispatch one PCA branch request and merge the per-bridge summaries into the root summary.

        The bridge side runs the same module-level leaf a filed PCA branch runs, so each report carries a mergeable
        ``PCASummary`` -- ``(rank, d)`` components and ``(d,)`` mean, never a chunk. The root merge is the balanced
        pairwise tree the standalone estimator uses (``merge_pca`` is associative), so this path and a single-machine
        fit agree by construction rather than by reimplementation. ``n_components``/``whiten`` are NOT applied here:
        the estimator that dispatched the request is the root and applies them when it adopts the summary.

        - ``:param request:`` The composed :class:`~deisa.dask.runtime_compose.PCARequest`.
        """
        if getattr(request, "kind", None) != _BRANCH_KIND_PCA:
            raise errors.precompute_runtime_invariant(
                "PlanDispatcher",
                f"a PCA dispatch needs a PCA request, got kind {getattr(request, 'kind', None)!r}",
            )
        plan = PlanRequest(
            request_id=uuid.uuid4().hex,
            array_name=request.input_name,
            timestep=int(request.timestep),
            request=request,
        )
        reports = self._reports_for(plan)
        for report in reports:
            if not isinstance(report.partial, PCASummary):
                raise errors.precompute_runtime_invariant(
                    "PlanDispatcher",
                    f"bridge {report.rank} answered the PCA branch {request.key!r} with a "
                    f"{type(report.partial).__name__}; only a mergeable summary can be merged",
                )
        return merge_tree([report.partial for report in reports])

    def _reports_for(self, plan: PlanRequest) -> List[PartialReport]:
        """One exchange round for ``plan``, or the no-transport refusal when this delivery carries no way to ask.

        - ``:param plan:`` The request to send to the bridges.
        """
        if self._exchange is None:
            raise errors.no_dispatch_transport(
                f"the on-demand request {plan.request.key!r} for array {plan.array_name!r} at timestep {plan.timestep}"
            )
        return _validate_reports(plan, self._exchange(plan), expected=self._bridge_count)

    def _menu_reports(self, menus: Sequence[EagerMenu], op_name: str) -> List[PartialReport]:
        """Turn every bridge's ``op_name`` menu value into the report the on-demand path would have produced.

        No exchange happens: the values were computed on the bridges' own chunks and handed over with the timestep, so
        the staleness question is settled by the lookup that found these menus.

        - ``:param menus:`` The eager menus of ONE array at ONE timestep (one per bridge).
        - ``:param op_name:`` The menu operation the reduction asks for.
        """
        reports = []
        for eager in menus:
            partial = eager.values[op_name]
            shape, dtype = _partial_shape_dtype(partial)
            reports.append(
                PartialReport(
                    request_id=EAGER_MENU_REQUEST_ID,
                    array_name=eager.array_name,
                    timestep=int(eager.timestep),
                    rank=eager.rank,
                    chunk_position=tuple(eager.chunk_position),
                    partial=partial,
                    shape=shape,
                    dtype=dtype,
                )
            )
        return reports


# The synthetic request id every eager-menu value is reported under: a menu value answers the operation, not a request
# id, so it is deliberately not a uuid and can never collide with a dispatched request.
EAGER_MENU_REQUEST_ID = "eager-menu"
