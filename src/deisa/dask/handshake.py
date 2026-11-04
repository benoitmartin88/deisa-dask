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
import logging
from typing import Any, List, Optional

from distributed import Client, Event, Future, get_client

from deisa.dask.constants import KEY_PREFIX
from deisa.dask.utils import _get_actor

logger = logging.getLogger(__name__)


class Handshake:
    _DEISA_HANDSHAKE_ACTOR_FUTURE_VARIABLE = "deisa_handshake_actor_future"
    _DEISA_WAIT_FOR_DONE_EVENT = "deisa_handshake_done"
    _DEISA_WAIT_FOR_BRIDGE_DONE_EVENT = "deisa_handshake_bridge_done"
    _DEISA_WAIT_FOR_GO_EVENT = "deisa_handshake_go"

    class HandshakeActor:
        def __init__(self):
            logger.debug("HandshakeActor.__init__()")
            self.analytics_ready = False
            self.client = get_client()
            self.nb_bridges = 0
            self.arrays_metadata = {}
            self.bridges_ready = False
            self.feedback_queue_size = 1024
            # ONE delivery fact per array: ``"plain"`` (chunk alone), ``"mixed"`` (aggregates AND chunk) or
            # ``"data_free"`` (aggregates only, nothing ships). The bridge reads this to decide whether to fold the
            # eager scalar aggregates and whether to scatter the chunk at all: for an array nobody reads, both costs
            # would be pure critical-path waste. An array the host never filed -- nobody registered a callback on it --
            # is ``"plain"``: the full-chunk scatter is the DEFAULT, not a registration fact.
            self.delivery_modes: dict = {}
            # The analytics host's on-demand plan requests, ONE slot per array: ``array_name -> PlanRequest``. The
            # root bridge of the array takes the slot at each ``send`` boundary and folds it there, which is why no
            # queue (and no polling thread) is needed to carry a request: this actor is already the one round trip
            # both sides make, so a request rides the mechanism every other handshake fact rides.
            self.plan_requests: dict = {}
            self.timestep: Optional[int] = None

        def set_bridges_ready(self, nb_bridges: int, arrays_metadata: dict) -> None:
            logger.debug(
                f"set_bridges_ready(): nb_bridges={nb_bridges}, arrays_metadata={arrays_metadata}, "
                f"analytics_ready={self.analytics_ready}"
            )
            self.nb_bridges = nb_bridges
            self.arrays_metadata = arrays_metadata
            self.bridges_ready = True
            if self.analytics_ready:
                self.__go()

        def set_bridges_done(self, timestep: int) -> None:
            logger.debug(f"set_bridges_done(): timestep={timestep}")
            self.timestep = timestep

        def set_analytics_ready(self, feedback_queue_size: int) -> None:
            logger.debug(f"set_analytics_ready(): bridges_ready={self.bridges_ready}")
            self.feedback_queue_size = feedback_queue_size
            self.analytics_ready = True
            if self.bridges_ready:
                self.__go()

        def get_arrays_metadata(self) -> dict | Future:
            return self.arrays_metadata

        def get_nb_bridges(self) -> int | Future:
            return self.nb_bridges

        def get_feedback_queue_size(self) -> int | Future:
            return self.feedback_queue_size

        def set_delivery_modes(self, modes: dict) -> None:
            logger.debug(f"set_delivery_modes(): modes={modes}")
            self.delivery_modes = dict(modes)

        def get_task_state(self, array_name: str) -> dict | Future:
            """Every per-array scheduling fact in one read: the ONE delivery mode of the array.

            The bridge fetches this over a single per-array broadcast: the mode must arrive in the same collective
            across the sub-comm, because a second, separate broadcast would have to be entered in the same order on
            every rank -- two maskable sources of a hang for one dict entry.

            ``"plain"`` is the default: an array the host never declared is served by the plain full-chunk scatter,
            so the bridge's path for an unknown array matches today's behaviour exactly.
            """
            return {"delivery": self.delivery_modes.get(array_name, "plain")}

        def put_plan(self, array_name: str, plan: Any) -> None:
            """Park the host's on-demand plan request for ONE array, replacing any request not yet taken.

            ONE slot per array, and that is a correctness statement, not a simplification: the host's
            ``PlanEventExchange`` blocks until the reports of the request it put have arrived, so it never has two
            requests for the same array in flight -- a second put can only follow a completed round. Where an older
            request is still parked, nobody is waiting for it any more (its host side timed out and refused), so the
            latest request is the only one worth serving. A queue here would instead let a request nobody waits for
            be served at a later boundary, which is work whose reports are dropped.

            - ``:param array_name:`` Registered array the request is rooted in; the array root's slot key.
            - ``:param plan:`` The composed :class:`~deisa.dask.ondemand_dispatch.PlanRequest`.
            """
            self.plan_requests[str(array_name)] = plan

        def take_plans(self, array_name: str) -> List[Any]:
            """Take the array's parked request, and answer at once: ``[]`` or ``[plan]``, never a wait.

            The bridge calls this at the array's ``send`` boundary. It is non-blocking by construction -- the actor
            answers with what is stored, or with an empty list when nothing was put since the last take -- so an idle
            boundary costs one actor round trip and no deadline is hidden in the scheduler. A list rather than a
            single request because it is what the boundary broadcasts to the array's peers: the root hands the
            broadcast exactly what it took, empty or not, so every rank enters the same collective every timestep.

            - ``:param array_name:`` Registered array whose slot is drained.
            """
            plan = self.plan_requests.pop(str(array_name), None)
            return [] if plan is None else [plan]

        def __go(self) -> None:
            logger.debug("Handshake go !")
            Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).set()

    def __init__(self, client: Optional[Client] = None):
        self.client = client
        self.__handshake_actor: Optional[Handshake.HandshakeActor] = None
        if self.client:
            self.__handshake_actor = _get_actor(
                self.client, Handshake.HandshakeActor, key=f"{KEY_PREFIX}-HandshakeActor"
            )
            assert self.__handshake_actor is not None

    def all_bridges_ready(self, nb_bridge: int, arrays_metadata: dict, wait_for_go=True) -> None:
        """
        Bridge must wait for analytics to be ready.
        """
        logger.debug(f"All bridges ready. nb_bridge={nb_bridge}, arrays_metadata={arrays_metadata}")
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_bridges_ready(nb_bridge, arrays_metadata).result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def deisa_ready(self, feedback_queue_size: int = 1024, wait_for_go=True, *args, **kwargs) -> None:
        """
        When analytics is ready, notify all Bridges
        """
        logger.debug("deisa ready.")
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_analytics_ready(feedback_queue_size).result()

        # wait for go
        if wait_for_go:
            self.__wait_for_go()

    def get_arrays_metadata(self) -> dict:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_arrays_metadata().result()

    def get_feedback_queue_size(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_feedback_queue_size().result()

    def get_nb_bridges(self) -> int:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_nb_bridges().result()

    def set_delivery_modes(self, modes: dict) -> None:
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_delivery_modes(modes).result()

    def get_task_state(self, array_name: str) -> dict:
        assert self.__handshake_actor is not None
        return self.__handshake_actor.get_task_state(array_name).result()

    def put_plan(self, array_name: str, plan: Any) -> None:
        """Hand ONE on-demand plan request to the array's root bridge, through the handshake actor.

        The host half of the plan transport: the request waits in the actor's per-array slot until the array root
        takes it at its next ``send`` boundary. One actor call, no queue, no subscription.

        - ``:param array_name:`` Registered array the request is rooted in.
        - ``:param plan:`` The composed :class:`~deisa.dask.ondemand_dispatch.PlanRequest`.
        """
        assert self.__handshake_actor is not None
        self.__handshake_actor.put_plan(array_name, plan).result()

    def take_plans(self, array_name: str) -> List[Any]:
        """Take the array's parked plan request, if any: ``[]`` or ``[plan]``, without waiting for one.

        The bridge half of the plan transport, called by the array root at its ``send`` boundary.

        - ``:param array_name:`` Registered array whose request slot is drained.
        """
        assert self.__handshake_actor is not None
        return self.__handshake_actor.take_plans(array_name).result()

    def __wait_for_go(self) -> None:
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()

    def wait_for_bridges(self):
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=self.client).wait()

    def set_bridges_done(self, timestep: int):
        logger.debug(f"set_bridges_done(): timestep={timestep}")
        assert self.__handshake_actor is not None
        self.__handshake_actor.set_bridges_done(timestep).result()
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).set()

    def wait_for_bridges_to_finish(self):
        Event(Handshake._DEISA_WAIT_FOR_BRIDGE_DONE_EVENT, client=self.client).wait()
