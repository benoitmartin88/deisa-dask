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

This module is the compute boundary of the on-demand reduction path. ``ReductionExpr.compute()`` composes the slim wire
request (:func:`deisa.dask.ondemand_compose.compose_request`), hands it to :class:`PlanDispatcher`, and gets a concrete
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

**How the two messages travel in production.** The request rides the handshake actor the run ALREADY runs
(:meth:`deisa.dask.handshake.Handshake.put_plan` puts it in the actor's one slot for the array), and the reports ride
the topic event the array root publishes at each ``send``: the root takes the slot at its boundary -- the one place
where every rank of the array is inside the same sub-comm collective -- broadcasts what it took, folds the retained
chunk of the requested timestep, and attaches the gathered reports to the ``plan_reports`` entry of the NEXT event it
publishes (:class:`PlanEventExchange` is the host's half). Nothing is exchanged to agree on a name, no queue lives in
the scheduler, and no bridge folds a chunk off its main thread: a second thread inside an MPI communicator is not a
safe place to be.

**The one timing bound of that design.** A request is answered at the array's *next* ``send``, so a request must reach
the array root before the simulation sends that array again, and its reports come back on that send's event. The host
waits ``_PLAN_REPORT_TIMEOUT_SECONDS`` and then refuses loudly rather than merging a hole; it cannot do better,
because the bridge side has no way to tell "this array will never send again" from "this array is mid-timestep". The
consequence to know about: a reduction of the FINAL delivered timestep -- a callback that computes on the last
collection of a finished run, with no further ``send`` for that array -- waits the full timeout and then refuses.
Reducing any timestep the simulation still has to send is what is supported today; failing fast on the last one needs
an end-of-run signal the delivery does not carry yet.

**Which window element may be reduced on demand.** The bridges retain ONE chunk per array -- the timestep they last
sent -- so on-demand plans are served for the LATEST delivered timestep only. A callback's window keeps older
collections alive (and their aggregates still answer ``sum``/``min``/``max``/``mean``/``prod``/``std`` on the bare
array, because the aggregates travel with the delivery), but an on-demand reduction rooted in an older element is
refused at compose time by :func:`~deisa.dask.ondemand_errors.stale_timestep`, with the remedy in the message.
Serving the whole window on demand would need one retained chunk per window slot per bridge -- a v2 memory
decision, not a v1 one.

:class:`EagerAggregates` is the eager half of the transfer strategy: the six tiny scalar full folds a bridge computes
once for its chunk, so the common scalar reduction costs no round trip. An aggregate value IS the payload an
on-demand request for the same operation would return (same composed callable, same chunk) -- **for a reduction of
the bare registered array**, which is the program the aggregates are built from. A derived operand
(``(collection * 2).sum()``) needs partials computed from its own program, so it never reads the aggregates and
always dispatches on demand; :class:`PlanDispatcher` gates the aggregates path on the operand program for exactly
that reason.

The same exchange carries the estimator-side PCA branch: :meth:`PlanDispatcher.compute_pca` dispatches a
:class:`~deisa.dask.ondemand_compose.PCARequest`, whose per-bridge partial is a mergeable
:class:`~deisa.dask.mergeable_pca.PCASummary`, and merges those summaries at the root. A PCA payload cannot go through
the reduction algebra, which is why it has its own request type and its own merge entry point.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from deisa.dask import ondemand_errors as errors
from deisa.dask.branch import _BRANCH_KIND_PCA, _SCALAR_FOLDS, _combine_two_phase
from deisa.dask.mergeable_pca import PCASummary, merge_tree
from deisa.dask.ondemand_compose import OndemandRequest, PCARequest, compose_request
from deisa.dask.ondemand_expr import _CHUNK_PROGRAM, KIND_BOOLEAN, KIND_COUNT, KIND_SCALAR

__all__ = [
    "EAGER_AGGREGATES_OPS",
    "EagerAggregates",
    "PartialReport",
    "PlanDispatcher",
    "PlanEventExchange",
    "PlanReportCollector",
    "PlanRequest",
    "PlanRequestBacklog",
    "PlanResponder",
    "build_eager_aggregates",
    "dispatch_plan_over_comm",
    "eager_aggregates_requests",
    "merge_partials",
    "plan_report_timeout",
    "serve_plan_requests",
]


# The combine algebra ``_combine_two_phase`` is a ``dask.delayed`` function: a merge must not build a task graph, so the
# module calls the undecorated callable the decorator preserves on ``__wrapped__`` (the ``functools`` convention dask's
# ``delayed`` follows). Same algebra, no scheduler, no graph, no worker.
_combine_partials = getattr(_combine_two_phase, "__wrapped__", None)
if _combine_partials is None:  # pragma: no cover - only reachable if dask stops preserving the undecorated callable
    raise RuntimeError(
        "deisa.dask.ondemand_dispatch: the combine algebra must stay callable without a task graph, but "
        "dask no longer exposes the undecorated _combine_two_phase."
    )
_COMBINE_PARTIALS: Callable[..., Any] = _combine_partials

# Kinds whose cross-bridge merge is a fold over one partial per bridge (the "fold kinds"). ``mean`` and ``moment`` are
# absent by design: their partials are the ``{n, total[, M]}`` dicts ``mean_agg``/``moment_agg`` consume.
_FOLD_KINDS = frozenset({KIND_SCALAR, KIND_BOOLEAN, KIND_COUNT})

# The post-combine step the wire request may ask for (``sqrt`` for a standard deviation).
_SQRT_FINALIZE = "sqrt"

# The eager scalar aggregates: the full folds small enough that a bridge can compute them once per timestep and ship
# them unprompted. ``mean``/``var`` travel as their dict partials, exactly as an on-demand request for them would.
EAGER_AGGREGATES_OPS: Tuple[str, ...] = ("sum", "min", "max", "mean", "var", "prod")


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
    - ``:param request:`` The slim wire request (a reduction's ``OndemandRequest`` or a ``PCARequest``: ``key``,
      ``kind``, composed ``branch_func``, and the reduction's ``axes``/``finalize``).
    """

    request_id: str
    array_name: str
    timestep: int
    request: Union[OndemandRequest, PCARequest]


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

    def as_payload(self) -> Dict[str, Any]:
        """This report as a msgpack-safe mapping, i.e. the shape it travels in on the topic event.

        The bus is ``Client.log_event``: it carries msgpack primitives only, so the dataclass, a numpy partial AND a
        :class:`~deisa.dask.mergeable_pca.PCASummary` are all refused as they are. The partial is therefore encoded
        by :func:`_partial_payload`, which keeps every leaf's dtype and shape (see :func:`_value_payload` for why a
        widened float32 would change the merged dtype).
        """
        payload: Dict[str, Any] = {
            "request_id": self.request_id,
            "array_name": self.array_name,
            "timestep": int(self.timestep),
            "rank": int(self.rank),
            "chunk_position": [int(value) for value in self.chunk_position],
            "shape": [int(value) for value in self.shape],
            "dtype": self.dtype,
            "error": self.error,
        }
        if self.error is None:
            payload["partial"] = _partial_payload(self.partial)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PartialReport":
        """Rebuild the report a bridge encoded with :meth:`as_payload`.

        - ``:param payload:`` The decoded topic-event entry.
        """
        return cls(
            request_id=str(payload["request_id"]),
            array_name=str(payload["array_name"]),
            timestep=int(payload["timestep"]),
            rank=int(payload["rank"]),
            chunk_position=tuple(int(value) for value in payload.get("chunk_position") or ()),
            partial=_partial_from_payload(payload["partial"]) if payload.get("error") is None else None,
            shape=tuple(int(value) for value in payload.get("shape") or ()),
            dtype=str(payload.get("dtype") or ""),
            error=payload.get("error"),
        )


def _partial_payload(partial: Any) -> Dict[str, Any]:
    """Encode ONE bridge partial as a msgpack-safe tree: a mergeable PCA summary, or any value tree.

    - ``:param partial:`` The partial a :class:`~deisa.dask.ondemand_dispatch.PlanResponder` produced.
    """
    if isinstance(partial, PCASummary):
        return {
            "pca": {
                "n_samples": int(partial.n_samples),
                "mean": _value_payload(partial.mean),
                "components": _value_payload(partial.components),
                "singular_values": _value_payload(partial.singular_values),
            }
        }
    return {"value": _value_payload(partial)}


def _partial_from_payload(payload: Mapping[str, Any]) -> Any:
    """Rebuild the partial :func:`_partial_payload` encoded, PCA summary or value tree.

    - ``:param payload:`` One encoded partial.
    """
    if "pca" in payload:
        summary = payload["pca"]
        return PCASummary(
            n_samples=int(summary["n_samples"]),
            mean=_value_from_payload(summary["mean"]),
            components=_value_from_payload(summary["components"]),
            singular_values=_value_from_payload(summary["singular_values"]),
        )
    return _value_from_payload(payload["value"])


def _value_payload(value: Any) -> Dict[str, Any]:
    """Encode ONE aggregate value as a msgpack-safe tree, keeping its dtype and shape.

    The topic event bus is ``distributed.Client.log_event``, which carries msgpack primitives only: it rejects the
    dataclass, and it rejects a numpy scalar or array too. A aggregate value is either a numpy scalar or the
    ``{n, total[, M]}`` partial of the mean/moment chunk stages, so each leaf travels as
    ``{"dtype": ..., "shape": ..., "data": ...}`` and :func:`_value_from_payload` rebuilds it EXACTLY -- widening a
    float32 partial to a Python float would change the dtype the merge reports, and a list-valued total would make the
    aggregation algebra concatenate instead of add.

    - ``:param value:`` One aggregate value: a numpy scalar/array, or a mapping of them.
    """
    if isinstance(value, Mapping):
        return {"dict": {str(key): _value_payload(item) for key, item in value.items()}}
    array = np.asarray(value)
    return {"dtype": str(array.dtype), "shape": [int(size) for size in array.shape], "data": array.ravel().tolist()}


def _value_from_payload(payload: Mapping[str, Any]) -> Any:
    """Rebuild the numpy value :func:`_value_payload` encoded, dtype and shape included.

    - ``:param payload:`` One encoded aggregate value.
    """
    if "dict" in payload:
        return {key: _value_from_payload(item) for key, item in payload["dict"].items()}
    array = np.asarray(payload["data"], dtype=payload["dtype"]).reshape(tuple(payload["shape"]))
    return array[()] if not payload["shape"] else array


@dataclass(frozen=True)
class EagerAggregates:
    """The eager scalar aggregates of ONE bridge for ONE timestep (the zero-round-trip half of the transfer strategy).

    Every value is a full fold of the BARE registered array's chunk (the identity program), so the aggregates answer a
    reduction of the bare collection and nothing else: a reduction of a derived operand (``(c * 2).sum()``) must be
    dispatched, because its partials are folds of a different program.

    The aggregates ride the topic event, whose bus is msgpack (``Client.log_event``), so a bridge ships
    :meth:`as_payload` and the delivery rebuilds the aggregates with :meth:`from_payload`; the values a callback's
    reduction merges are then the objects this bridge computed, not a lossy rendering of them.

    - ``:param array_name:`` Registered array the aggregates were computed for.
    - ``:param timestep:`` Timestep the bridge's chunk belonged to.
    - ``:param rank:`` Rank of the bridge inside the array's communicator.
    - ``:param chunk_position:`` MPI coordinates of the bridge's chunk.
    - ``:param values:`` ``op_name -> partial`` for every entry of :data:`EAGER_AGGREGATES_OPS`.
    """

    array_name: str
    timestep: int
    rank: int
    chunk_position: Tuple[int, ...]
    values: Mapping[str, Any]

    def as_payload(self) -> Dict[str, Any]:
        """This aggregates as a msgpack-safe mapping, i.e. the shape it travels in on the topic event."""
        return {
            "array_name": self.array_name,
            "timestep": int(self.timestep),
            "rank": int(self.rank),
            "chunk_position": [int(value) for value in self.chunk_position],
            "values": {str(op_name): _value_payload(value) for op_name, value in self.values.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EagerAggregates":
        """Rebuild the aggregates a bridge encoded with :meth:`as_payload`.

        - ``:param payload:`` The decoded topic-event entry.
        """
        return cls(
            array_name=str(payload["array_name"]),
            timestep=int(payload["timestep"]),
            rank=int(payload["rank"]),
            chunk_position=tuple(int(value) for value in payload.get("chunk_position") or ()),
            values={op_name: _value_from_payload(value) for op_name, value in payload["values"].items()},
        )


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
            raise errors.ondemand_invariant("_partial_shape_dtype", "a dict partial carried no entries to describe")
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

        Fail-loud contract (the one the retired ``_execute_operations_on_chunk`` enforced, now carried by this responder
        and :meth:`PlanPump.serve`): a branch that raises produces an ERROR report, never a partial. A
        dropped partial would silently corrupt the merge (a smaller sum, a mean missing a whole bridge's ``n``).

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


def build_eager_aggregates(
    collection: Any,
    chunk: Any,
    rank: int = 0,
    chunk_position: Sequence[int] = (),
) -> EagerAggregates:
    """Compute the eager scalar aggregates on this bridge's own chunk.

    The aggregates compose the SAME slim requests an on-demand call would (one per :data:`EAGER_AGGREGATES_OPS`
    entry, full folds only) and run them on the local chunk, so an aggregate value is byte-for-byte the partial the
    on-demand path returns and the merge cannot tell the two apart. The failure contract is the fail-loud one: an
    aggregates entry that raises aborts the aggregates instead of shipping a hole.

    - ``:param collection:`` The :class:`~deisa.dask.deisa_array.OndemandDeisaArray` describing this bridge's array.
    - ``:param chunk:`` The bridge's own numpy chunk.
    - ``:param rank:`` Rank of this bridge inside the array's communicator.
    - ``:param chunk_position:`` MPI coordinates of this bridge's chunk.
    """
    values = {}
    for op_name, request in zip(EAGER_AGGREGATES_OPS, eager_aggregates_requests(collection)):
        try:
            values[op_name] = request.branch_func(chunk)
        except Exception as exc:
            raise errors.ondemand_invariant(
                "build_eager_aggregates", f"the {op_name!r} aggregates entry failed on the local chunk: {exc!r}"
            ) from exc
    return EagerAggregates(
        array_name=collection.name,
        timestep=collection.timestep,
        rank=int(rank),
        chunk_position=tuple(int(value) for value in chunk_position),
        values=values,
    )


def eager_aggregates_requests(collection: Any) -> Tuple[OndemandRequest, ...]:
    """The slim requests of the eager scalar aggregates, in :data:`EAGER_AGGREGATES_OPS` order.

    - ``:param collection:`` The :class:`~deisa.dask.deisa_array.OndemandDeisaArray` to compose the aggregates for.
    """
    return tuple(compose_request(getattr(collection, op_name)()) for op_name in EAGER_AGGREGATES_OPS)


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
        raise errors.ondemand_invariant(
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
# Production transport: the handshake actor (requests) and the topic event (reports)
# ---------------------------------------------------------------------------------------------------------------------
# How long the HOST waits for the reports of one plan request. There is no bridge-side bound: the array root TAKES the
# parked request from the handshake actor at each send boundary (one actor round trip, no wait), so a request that has
# not been taken by then simply stays parked and is taken at the NEXT boundary -- and its reports come back with that
# boundary's event. The host bound is the one that must tolerate the simulation taking a timestep to reach its next
# send boundary -- and it is also the bound a request that can never be served pays in full (see "The one timing bound
# of that design", above: a reduction of the final delivered timestep has no further send to be answered at). That is
# why it is cycle-tolerable rather than a five-minute stall, and why it is overridable.
# ``DEISA_ONDEMAND_PLAN_TIMEOUT`` is read by the HOST only: it bounds ``PlanEventExchange``'s wait, and no bridge waits
# on it, so a simulation whose timestep is longer than the host default needs the variable set in the analytics host's
# environment alone.
_PLAN_REPORT_TIMEOUT_SECONDS: float = 60.0
_PLAN_REPORT_TIMEOUT_ENV: str = "DEISA_ONDEMAND_PLAN_TIMEOUT"


def plan_report_timeout() -> float:
    """Seconds the host waits for one round of reports: ``DEISA_ONDEMAND_PLAN_TIMEOUT``, else the built-in default.

    A malformed value falls back to the default rather than refusing: the timeout is a diagnostic bound, and a typo in
    an environment variable must not become the reason a run cannot start.
    """
    raw = os.getenv(_PLAN_REPORT_TIMEOUT_ENV)
    if not raw:
        return _PLAN_REPORT_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _PLAN_REPORT_TIMEOUT_SECONDS
    return value if value > 0 else _PLAN_REPORT_TIMEOUT_SECONDS


class PlanRequestBacklog:
    """The array root bridge's plan requests, parked between the actor take and the boundary broadcast.

    Only the root ever puts: it takes the host's parked request from the handshake actor at the send boundary
    (``PlanPump.serve``) and drains the backlog in the same call, because answering a request needs every rank of the
    array and only the send boundary is a place where they are all parked in the same collective. Every other rank's
    backlog stays empty -- it receives the requests through the boundary broadcast.

    Nothing here is locked: the take and the drain are two statements of one ``serve`` call on one thread, and no other
    thread ever touches a backlog.
    """

    def __init__(self) -> None:
        self._plans: List[PlanRequest] = []

    def put(self, plan: PlanRequest) -> None:
        """Park one request for the fold that follows the take.

        - ``:param plan:`` The request the actor handed over.
        """
        self._plans.append(plan)

    def drain(self) -> List[PlanRequest]:
        """Take every parked request, oldest first (an empty list when the actor had none)."""
        plans, self._plans = self._plans, []
        return plans


class PlanReportCollector:
    """The analytics host's meeting point between a dispatched plan and the events that answer it.

    The reports cannot be handed to the dispatching callback directly: they arrive later, on the topic event of the
    array's NEXT ``send``, which is processed by the client's event loop while the callback is blocked inside its
    exchange. So the exchange registers what it is waiting for (``expect``), the topic handler feeds every event's
    ``plan_reports`` in (``deliver``), and the blocked exchange wakes on the report that answers ITS request id
    (``wait``). One collector serves the whole host: a request id is a uuid, so an entry can never collide across
    arrays or requests.

    The wait is a :class:`threading.Event` signalled by the loop -- NOT a thread, a poll, or a ``Future`` the loop
    would have to resolve for the callback to make progress. The callback already runs on a worker thread
    (``asyncio.to_thread`` in the topic handler), so blocking there costs nothing the run was not already paying, and
    the loop stays free to deliver the next event, which is the whole point: the answer to a plan IS a later event.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._waiting: Dict[str, List[PartialReport]] = {}
        self._arrived: Dict[str, threading.Event] = {}

    def expect(self, request_id: str) -> None:
        """Start waiting for the reports of ``request_id``.

        - ``:param request_id:`` Id of the request about to be put in the handshake actor.
        """
        with self._lock:
            self._waiting[request_id] = []
            self._arrived[request_id] = threading.Event()

    def deliver(self, reports: Sequence[PartialReport]) -> None:
        """File the reports one topic event carried, and wake whoever waits for them.

        A report whose request is no longer waited for is DROPPED: its host side either timed out and refused, or
        never asked (the event can carry the reports of a boundary whose requester went away). Keeping it would leak
        an entry per abandoned request for the life of the run.

        - ``:param reports:`` The ``plan_reports`` entry of one topic event, rebuilt into reports.
        """
        if not reports:
            return
        with self._lock:
            for report in reports:
                waiting = self._waiting.get(report.request_id)
                if waiting is None:
                    continue
                waiting.append(report)
                self._arrived[report.request_id].set()

    def wait(self, request_id: str, expected: int, timeout: float) -> List[PartialReport]:
        """Block until ``expected`` reports answered ``request_id``, or ``timeout`` seconds elapsed.

        Returns what arrived either way -- a short set is the caller's refusal to raise, so the message can say how
        many of how many landed.

        - ``:param request_id:`` Id of the dispatched request.
        - ``:param expected:`` Number of reports a complete round has (one per bridge).
        - ``:param timeout:`` Seconds to wait before returning the short set.
        """
        deadline = time.monotonic() + timeout
        with self._lock:
            arrived = self._arrived.get(request_id)
        while True:
            with self._lock:
                reports = list(self._waiting.get(request_id, ()))
            if len(reports) >= expected or arrived is None:
                return reports
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return reports
            arrived.wait(remaining)
            arrived.clear()

    def forget(self, request_id: str) -> None:
        """Stop waiting for ``request_id``: its round is over (answered or refused).

        - ``:param request_id:`` Id of the request whose entry is dropped.
        """
        with self._lock:
            self._waiting.pop(request_id, None)
            self._arrived.pop(request_id, None)


class PlanEventExchange:
    """The analytics host's transport: put the request in the handshake actor, take the reports off the topic events.

    One request/reports round trip with nothing spawned and nothing queued: the host puts the request in the actor's
    per-array slot (:meth:`deisa.dask.handshake.Handshake.put_plan`), the array root takes it at its next ``send``
    boundary, folds with every rank of the array, and attaches the gathered reports to the event that send publishes.
    The exchange returns as soon as THAT event delivered one report per bridge.

    A plan asked AFTER an event was published is served at the FOLLOWING event -- that is the boundary contract, not a
    race: the reports are matched on the request id, so a later event can never satisfy an earlier request by accident.

    The put happens ONCE per request. Putting once per bridge would be wrong for the same reason a broadcast is not a
    per-rank call: the array has exactly one root, it serves the request with all N ranks, and the event therefore
    carries exactly N reports -- N puts would park N requests (each replacing the previous one in the single slot) and
    still produce one round.

    The reports come back in the order the bridges answered; :func:`_validate_reports` and the merge both match on the
    request id and sort by chunk position, so the order is not part of the contract.

    - ``:param handshake:`` The host's handshake handle: ``put_plan`` is the whole request side of the transport.
    - ``:param reports:`` The collector the topic handler feeds (see :class:`PlanReportCollector`).
    - ``:param bridge_count:`` Number of bridges that own a chunk of the array: one report is expected from each.
    - ``:param timeout:`` Seconds to wait for the full set of reports before refusing; ``None`` takes
      :func:`plan_report_timeout` (the env-overridable default).
    """

    def __init__(
        self,
        handshake: Any,
        reports: PlanReportCollector,
        bridge_count: int,
        timeout: Optional[float] = None,
    ) -> None:
        self._handshake = handshake
        self._reports = reports
        self._bridge_count = int(bridge_count)
        self._timeout = plan_report_timeout() if timeout is None else float(timeout)

    def __call__(self, plan: PlanRequest) -> List[PartialReport]:
        """Park ``plan`` in the handshake actor once and collect one report from each bridge.

        - ``:param plan:`` The request to dispatch.
        """
        self._reports.expect(plan.request_id)
        try:
            self._handshake.put_plan(plan.array_name, plan)
            reports = self._reports.wait(plan.request_id, self._bridge_count, self._timeout)
        finally:
            self._reports.forget(plan.request_id)
        if len(reports) < self._bridge_count:
            raise errors.ondemand_invariant(
                "PlanEventExchange",
                f"request {plan.request_id!r} for array {plan.array_name!r} at timestep {plan.timestep} collected "
                f"{len(reports)} of {self._bridge_count} report(s) in {self._timeout}s; no bridge is serving plan "
                f"requests for that array. Refusing to merge a partial set with a hole in it. A request is answered at "
                f"the array's NEXT send, so this also happens when it names the simulation's final timestep (nothing "
                f"is sent after it) -- reduce the latest delivered timestep before the run ends, or raise "
                f"{_PLAN_REPORT_TIMEOUT_ENV} for a simulation whose timestep takes longer than {self._timeout}s",
            )
        return reports


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
            raise errors.ondemand_invariant(
                "_grid_shape",
                f"a partial reports the chunk position {coord} but the array has {ndim} data axes; the "
                f"grid<->data-axis mapping is ambiguous",
            )
        if any(value < 0 for value in coord):
            raise errors.ondemand_invariant("_grid_shape", f"the chunk position {coord} has a negative axis")
    if len(set(coords)) != len(coords):
        raise errors.ondemand_invariant("_grid_shape", f"two partials report the same chunk position: {sorted(coords)}")
    grid_shape = tuple(max(coord[axis] for coord in coords) + 1 for axis in range(ndim))
    if int(np.prod(grid_shape)) != len(coords):
        raise errors.ondemand_invariant(
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
        raise errors.ondemand_invariant("merge_partials", "no per-bridge partial was delivered to merge")
    red_axes = tuple(range(ndim)) if not axes else tuple(int(axis) for axis in axes)
    kept_axes = tuple(axis for axis in range(ndim) if axis not in red_axes)
    grid_shape = _grid_shape(reports, ndim)
    out_dtype = reports[0].dtype or str(np.float64)
    if kind in _FOLD_KINDS and not kept_axes:
        fold = _SCALAR_FOLDS.get(op_name)
        if fold is None:
            raise errors.ondemand_invariant("merge_partials", f"no cross-bridge fold for the reduction {op_name!r}")
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
        raise errors.ondemand_invariant(
            "PlanDispatcher", f"request {plan.request_id!r} returned no report from any bridge"
        )
    if expected is not None and len(reports) != expected:
        raise errors.ondemand_invariant(
            "PlanDispatcher",
            f"request {plan.request_id!r} expected {expected} reports (one per bridge) but got {len(reports)}",
        )
    matched = []
    for report in reports:
        if report.request_id != plan.request_id:
            raise errors.ondemand_invariant(
                "PlanDispatcher",
                f"request {plan.request_id!r} received a partial for request {report.request_id!r}; a result from "
                f"another request cannot satisfy this one",
            )
        if report.timestep != plan.timestep:
            raise errors.ondemand_invariant(
                "PlanDispatcher",
                f"request {plan.request_id!r} is for timestep {plan.timestep} but bridge {report.rank} answered with a "
                f"partial of timestep {report.timestep}; refusing to mix two timesteps of array {plan.array_name!r}",
            )
        if report.array_name != plan.array_name:
            raise errors.ondemand_invariant(
                "PlanDispatcher",
                f"request {plan.request_id!r} is rooted in array {plan.array_name!r} but bridge {report.rank} answered "
                f"for array {report.array_name!r}",
            )
        if report.error is not None:
            raise errors.ondemand_invariant(
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
    """The analytics-host side of the compute boundary: compose, dispatch (or read the aggregates), match, merge.

    ``exchange`` is the only transport knob: it takes one :class:`PlanRequest` and returns the reports every bridge
    produced for it. Keeping it a callable means this class owns no communicator dialect -- the same dispatcher is
    driven over the array's sub-comm in production and over a test communicator in the suite. It may be ``None``: a
    delivery that attaches the eager aggregates but no transport still serves every aggregates operation, and refuses
    the rest with the no-transport refusal instead of gathering anything.

    - ``:param exchange:`` Callable performing one request/reports round trip, or ``None`` when the delivery has no
      bridge transport.
    - ``:param bridge_count:`` Number of bridges that own a chunk of the array, when known. A round that returns another
      number of reports is refused: a missing partial would silently shrink the merged value.
    - ``:param aggregates:`` Optional eager aggregates already computed for the current timestep. A full fold of an
      aggregates operation is merged from them without touching ``exchange``.
    - ``:param latest_timestep:`` Callable returning the newest timestep delivered for the root array, or ``None`` when
      the delivery does not track it. A collection keeps living in a callback's window after its timestep, but the
      bridges only retain ONE chunk per array (the one they last sent), so an on-demand reduction of an older window
      element cannot be served; it is refused at compose time with a named remedy instead of being dispatched into an
      error report. A callable rather than a value on purpose: the dispatcher of an older delivery must see the
      timestep that has since become the latest, not the one current when it was built.
    """

    def __init__(
        self,
        exchange: Optional[Callable[[PlanRequest], Sequence[PartialReport]]] = None,
        bridge_count: Optional[int] = None,
        aggregates: Optional[Sequence[EagerAggregates]] = None,
        latest_timestep: Optional[Callable[[], Optional[int]]] = None,
    ) -> None:
        self._exchange = exchange
        self._bridge_count = None if bridge_count is None else int(bridge_count)
        self._latest_timestep = latest_timestep
        self._aggregates: Dict[Tuple[str, int], List[EagerAggregates]] = {}
        for eager in aggregates or ():
            self._aggregates.setdefault((eager.array_name, int(eager.timestep)), []).append(eager)

    def compute(self, reduction: Any) -> Any:
        """Deliver the concrete value of ``reduction``.

        - ``:param reduction:`` The :class:`~deisa.dask.ondemand_expr.ReductionExpr` a callback computed.
        """
        request = compose_request(reduction)
        ndim = reduction.operand.ndim
        aggregates = self._aggregates.get((request.input_name, reduction.root.timestep))
        # The aggregates hold the BARE registered array's partials (``build_eager_aggregates`` composes the identity
        # program), so they may only answer a reduction whose operand IS the bare chunk program. A derived operand
        # (``(c * 2).sum()``) needs its own partials computed from the derived program: answering it from the
        # aggregates would silently return the un-derived field's value, with no round trip and no exception.
        if (
            aggregates
            and not request.axes
            and reduction.operand.program == _CHUNK_PROGRAM
            and all(reduction.op_name in eager.values for eager in aggregates)
        ):
            reports = self._aggregates_reports(aggregates, reduction.op_name)
            if self._bridge_count is not None and len(reports) != self._bridge_count:
                raise errors.ondemand_invariant(
                    "PlanDispatcher",
                    f"the eager aggregates of array {request.input_name!r} at timestep "
                    f"{reduction.root.timestep} cover {len(reports)} bridge(s) but the array has "
                    f"{self._bridge_count}",
                )
        else:
            self._refuse_stale(reduction, request)
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

        - ``:param request:`` The composed :class:`~deisa.dask.ondemand_compose.PCARequest`.
        """
        if getattr(request, "kind", None) != _BRANCH_KIND_PCA:
            raise errors.ondemand_invariant(
                "PlanDispatcher",
                f"a PCA dispatch needs a PCA request, got kind {getattr(request, 'kind', None)!r}",
            )
        plan = PlanRequest(
            request_id=uuid.uuid4().hex,
            array_name=request.input_name,
            timestep=int(request.timestep),
            request=request,
        )
        self._refuse_stale_timestep(int(request.timestep), str(request.input_name), f"the PCA branch {request.key!r}")
        reports = self._reports_for(plan)
        for report in reports:
            if not isinstance(report.partial, PCASummary):
                raise errors.ondemand_invariant(
                    "PlanDispatcher",
                    f"bridge {report.rank} answered the PCA branch {request.key!r} with a "
                    f"{type(report.partial).__name__}; only a mergeable summary can be merged",
                )
        return merge_tree([report.partial for report in reports])

    def _refuse_stale(self, reduction: Any, request: Any) -> None:
        """Refuse an on-demand reduction whose operand is older than the newest delivered timestep.

        Only the on-demand path is guarded: the aggregates path is decided before this call and is served from the
        aggregates shipped WITH the delivery, so it stays valid for every element of a callback's window.

        - ``:param reduction:`` The :class:`~deisa.dask.ondemand_expr.ReductionExpr` being computed.
        - ``:param request:`` Its composed wire request (read for the root array's name).
        """
        self._refuse_stale_timestep(
            int(reduction.root.timestep),
            str(request.input_name),
            f"the reduction {reduction.key!r}",
        )

    def _refuse_stale_timestep(self, timestep: int, array_name: str, symbol: str) -> None:
        """Refuse a plan stamped with a timestep older than the newest delivery of ``array_name``.

        - ``:param timestep:`` Timestep of the root collection the request was composed from.
        - ``:param array_name:`` Registered array the request is rooted in.
        - ``:param symbol:`` What is being refused, for the message.
        """
        if self._latest_timestep is None:
            return
        latest = self._latest_timestep()
        if latest is None or timestep >= int(latest):
            return
        raise errors.stale_timestep(symbol, array_name=array_name, requested=timestep, latest=int(latest))

    def _reports_for(self, plan: PlanRequest) -> List[PartialReport]:
        """One exchange round for ``plan``, or the no-transport refusal when this delivery carries no way to ask.

        - ``:param plan:`` The request to send to the bridges.
        """
        if self._exchange is None:
            raise errors.no_dispatch_transport(
                f"the on-demand request {plan.request.key!r} for array {plan.array_name!r} at timestep {plan.timestep}"
            )
        return _validate_reports(plan, self._exchange(plan), expected=self._bridge_count)

    def _aggregates_reports(self, aggregates: Sequence[EagerAggregates], op_name: str) -> List[PartialReport]:
        """Turn every bridge's ``op_name`` aggregate value into the report the on-demand path would have produced.

        No exchange happens: the values were computed on the bridges' own chunks and handed over with the timestep, so
        the staleness question is settled by the lookup that found these aggregates.

        - ``:param aggregates:`` The eager aggregates of ONE array at ONE timestep (one per bridge).
        - ``:param op_name:`` The aggregates operation the reduction asks for.
        """
        reports = []
        for eager in aggregates:
            partial = eager.values[op_name]
            shape, dtype = _partial_shape_dtype(partial)
            reports.append(
                PartialReport(
                    request_id=EAGER_AGGREGATES_REQUEST_ID,
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


# The synthetic request id every eager-aggregates value is reported under: an aggregate value answers the operation,
# not a request id, so it is deliberately not a uuid and can never collide with a dispatched request.
EAGER_AGGREGATES_REQUEST_ID = "eager-aggregates"
