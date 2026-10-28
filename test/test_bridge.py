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
import asyncio
import gc
import logging
import os
import sys
import time
import weakref

import numpy as np
import pytest
from distributed import Client, LocalCluster
from utils import FakeCartComm, FakeComm, async_close_bridges, async_map

from deisa.dask import Bridge

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture(scope="function")
def env_setup():
    cluster = LocalCluster(
        n_workers=1, threads_per_worker=1, processes=True, dashboard_address=":0", worker_dashboard_address=":0"
    )
    os.environ["DEISA_DASK_SCHEDULER_ADDRESS"] = cluster.scheduler_address
    client = Client(cluster)
    client.wait_for_workers(1, timeout=10)
    yield client, cluster
    client.close()
    cluster.close()


class TestBridge:
    def get_new_bridge(self):
        arrays_metadata = {"temperature": {"global_shape": (1,), "chunk_shape": (1,), "chunk_position": (0,)}}
        comm_state = FakeComm.State(1)
        bridge = Bridge(comm=FakeComm(comm_state, 0), arrays_metadata=arrays_metadata, wait_for_go=False)
        return bridge, arrays_metadata

    def test_ctor(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata = self.get_new_bridge()
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])
        assert isinstance(bridge.comm, FakeComm)
        assert not bridge._has_close_been_called

    def test__del__(self, env_setup):
        client, cluster = env_setup
        bridge, arrays_metadata = self.get_new_bridge()
        assert bridge.id == 0
        assert bridge.arrays_metadata == arrays_metadata
        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])
        assert isinstance(bridge.comm, FakeComm)
        assert not bridge._has_close_been_called
        bridge.__del__()
        assert bridge._has_close_been_called

    def test_close(self, env_setup):
        client, _ = env_setup
        bridge, _ = self.get_new_bridge()
        assert not bridge._has_close_been_called
        bridge.close(timestep=42)
        assert bridge._has_close_been_called

    @pytest.mark.flaky(retries=3, delay=1)
    def test_send_update_workers(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])

        cluster.scale(2)
        cluster.wait_for_workers(2)

        bridge.send("temperature", np.ones(1), timestep=0, update_workers=True)

        assert bridge.workers is not None
        assert sorted(list(bridge.workers.keys())) == sorted([w.worker_address for w in cluster.workers.values()])

    @pytest.mark.flaky(retries=3, delay=1)
    def test_send_filter_workers_empty(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            return []

        with pytest.raises(TypeError) as _:
            bridge.send("temperature", np.ones(1), timestep=0, filter_workers=filter)

    @pytest.mark.parametrize("i", [1, np.array(1), np.array([1])])
    def test_send_timestep_type(self, env_setup, i):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        bridge.send("temperature", np.ones(1), timestep=i)

    def test_send_filter_workers_without_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            assert isinstance(workers, dict)
            for addr in workers.keys():
                assert isinstance(addr, str)
                assert addr in [w.worker_address for w in cluster.workers.values()]
            return list(workers.keys())

        bridge.send("temperature", np.ones(1), timestep=0, update_workers=False, filter_workers=filter)

    def test_send_filter_workers_with_update_workers_valid(self, env_setup):
        client, cluster = env_setup
        bridge, _ = self.get_new_bridge()

        def filter(workers):
            assert isinstance(workers, dict)
            return list(workers.keys())

        bridge.send("temperature", np.ones(1), timestep=0, update_workers=True, filter_workers=filter)

    def test_cart_comm(self, env_setup):
        client, cluster = env_setup

        arrays_metadata = {"temperature": {"global_shape": (8, 8), "chunk_shape": (4, 4), "chunk_position": (0, 0)}}
        comm_state = FakeComm.State(4)

        def make_bridge(rank):
            return Bridge(
                comm=FakeCartComm(comm_state, rank, dims=(2, 2)), arrays_metadata=arrays_metadata, wait_for_go=False
            )

        # Create bridges in parallel (Split is a collective op)
        bridges = async_map(range(4), make_bridge)

        async def _bridge_send():
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        bridge.send, "temperature", np.ones(arrays_metadata["temperature"]["chunk_shape"]), timestep=0
                    )
                    for i, bridge in enumerate(bridges)
                ]
            )

        asyncio.run(_bridge_send())

        event = client.get_events("temperature")
        assert len(event) == 1
        _, info = event[0]
        assert info["array_name"] == "temperature"
        assert info["iteration"] == 0
        assert len(info["futures"]) == 4
        for f in info["futures"]:
            assert f["chunk_position"] in [(0, 0), (0, 1), (1, 0), (1, 1)]

        async def _bridge_close():
            await asyncio.gather(*[asyncio.to_thread(bridge.close, 0) for i, bridge in enumerate(bridges)])

        asyncio.run(_bridge_close())

class TestPrecomputeRegressions:
    """Regression tests for bridge delivery and interpreter-shutdown teardown."""

    def _meta(self, array_name, chunk_pos, global_shape=(8,), chunk_shape=(4,)):
        return {array_name: {"global_shape": global_shape, "chunk_shape": chunk_shape, "chunk_position": chunk_pos}}

    def test_wait_for_go_false_skips_go_wait(self, env_setup):
        """Bridge.__init__ with wait_for_go=False must not block on the go event.

        The go event is set only by ``Deisa.execute_callbacks()``; when the bridge is constructed before callbacks are
        executed (e.g. a simulation that registers callbacks lazily or not at all), ``wait_for_go=False`` skips the
        wait and leaves branch fetching to the first ``send()``.
        """
        from distributed import Event

        from deisa.dask.handshake import Handshake

        client, cluster = env_setup
        Event(Handshake._DEISA_WAIT_FOR_GO_EVENT, client=client).set()
        start = time.monotonic()
        bridge = Bridge(
            comm=FakeComm(FakeComm.State(1), 0),
            arrays_metadata=self._meta("temperature", (0,)),
            wait_for_go=False,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 5, f"Bridge.__init__ blocked for {elapsed:.1f}s despite wait_for_go=False"
        # No go signal arrived and no prefetch happened; lazy fetch remains correct.
        bridge.close(timestep=0)

    def test_del_skips_close_at_interpreter_shutdown(self, env_setup, monkeypatch):
        """Teardown: __del__ must not run the blocking close() at shutdown.

        ``close()`` runs a world barrier which can never complete once peer ranks are gone -- ``__del__`` skips
        ``close()`` while ``sys.is_finalizing()``.
        """
        env_setup  # use fixture
        bridge, _ = self.get_plain_bridge()
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        calls = []
        orig_close = bridge.close

        def spy_close(timestep):
            calls.append(timestep)
            return orig_close(timestep)

        bridge.close = spy_close
        bridge.__del__()
        assert calls == [], "__del__ must skip close() during interpreter shutdown"

    def test_del_does_not_block_the_event_loop(self, env_setup, caplog):
        """Teardown: __del__ must not make blocking calls from inside an event-loop callback.

        The collector runs ``__del__`` on whichever thread dropped the last reference; for a leaked bridge
        that is a distributed "IO loop" thread. ``close()`` synchronizes with the scheduler through
        ``Client.sync()``, which blocks until *that* loop runs the callback -- from inside the loop the
        callback never runs, so the thread waits out the whole ``sync()`` timeout and, with an in-process
        cluster, freezes the scheduler, its workers and the client that share the loop.

        Deterministic sister of the intermittent serial teardown flake: the collection is forced onto the
        client's loop thread instead of waiting for the garbage collector to pick that thread by chance.
        """
        client, _ = env_setup
        bridge, _ = self.get_plain_bridge()
        # Keep the bridge alive through a reference cycle: refcounting alone cannot free it, so only the
        # gc.collect() below -- running on the client's loop thread -- can collect it (and run __del__).
        bridge._probe_cycle = [bridge]
        collected = weakref.ref(bridge)
        del bridge
        assert collected() is not None, "the reference cycle should keep the bridge alive"

        async def collect_on_the_loop():
            gc.collect()

        with caplog.at_level(logging.DEBUG, logger="deisa.dask.bridge"):
            start = time.monotonic()
            client.sync(collect_on_the_loop)
            elapsed = time.monotonic() - start

        assert collected() is None, "the bridge should have been collected inside the loop callback"
        assert elapsed < 30, f"__del__ blocked the event loop for {elapsed:.1f}s"
        assert any("event-loop callback" in record.getMessage() for record in caplog.records), (
            "close() must be skipped: it cannot complete while the loop that has to service its collectives "
            "is blocked by the very call waiting for it"
        )

    def test_close_skips_collectives_at_interpreter_shutdown(self, env_setup, monkeypatch):
        """Teardown: close() skips the barrier / sub-comm Free() at shutdown.

        At ``sys.is_finalizing()`` the blocking collectives are skipped; a hang is worse than an exception.
        """
        env_setup  # use fixture
        bridge, _ = self.get_plain_bridge()
        monkeypatch.setattr(sys, "is_finalizing", lambda: True)
        barriers = []
        bridge.comm.barrier = lambda: barriers.append(1)
        bridge.close(timestep=0)
        assert bridge._has_close_been_called
        assert barriers == [], "close() must skip the world barrier during interpreter shutdown"

    def test_send_non_participating_rank_skips_delivery_work(self, env_setup):
        """A rank whose sub-comm for the array is _COMM_NULL does no delivery work.

        The ``_COMM_NULL`` early return comes FIRST: such a rank never fetches the array's delivery state, never folds
        the eager menu on the chunk and never scatters it -- it owns nothing the workers asked for. (The per-rank fold
        the retired ``_scatter_partials`` used to guard lives on the same early return.)
        """
        client, cluster = env_setup
        state = FakeComm.State(2)
        meta0 = self._meta("temperature", (0,))
        meta1 = self._meta("pressure", (0,))
        metas = [meta0, meta1]

        def _make(rank):
            return Bridge(comm=FakeComm(state, rank), arrays_metadata=metas[rank], wait_for_go=False)

        b0, b1 = async_map([0, 1], _make)
        # Merged-metadata scenario: the array is known to this bridge but its
        # sub-comm for it is _COMM_NULL (it does not own it).
        b1.arrays_metadata["temperature"] = meta0["temperature"]
        calls = {"state": 0, "menu": 0, "scatter": 0}
        orig_state = b1._ensure_task_state
        orig_menu = b1._build_runtime_menu
        orig_scatter = b1._scatter_full

        def spy_state(array_name):
            calls["state"] += 1
            return orig_state(array_name)

        def spy_menu(array_name, chunk, timestep):
            calls["menu"] += 1
            return orig_menu(array_name, chunk, timestep)

        def spy_scatter(data, workers=None, hash=False):
            calls["scatter"] += 1
            return orig_scatter(data, workers=workers, hash=hash)

        b1._ensure_task_state = spy_state
        b1._build_runtime_menu = spy_menu
        b1._scatter_full = spy_scatter
        b1.send("temperature", np.ones(4), timestep=0)
        assert calls == {"state": 0, "menu": 0, "scatter": 0}

        async_close_bridges([b0, b1], 0)

    def get_plain_bridge(self):
        arrays_metadata = {"temperature": {"global_shape": (1,), "chunk_shape": (1,), "chunk_position": (0,)}}
        comm_state = FakeComm.State(1)
        bridge = Bridge(comm=FakeComm(comm_state, 0), arrays_metadata=arrays_metadata, wait_for_go=False)
        return bridge, arrays_metadata
