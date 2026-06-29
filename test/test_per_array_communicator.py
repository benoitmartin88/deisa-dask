# =============================================================================
# Tests for GitHub issues:
#   #17 — Use one communicator per array
#   #109 — Support sending data not present on all bridges (non-distributed arrays)
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
import os

import numpy as np
import pytest
from distributed import Client, LocalCluster

from deisa.dask import Bridge
from utils import FakeComm, FakeCartComm, async_map


class TestPerArrayCommunicator:
    @pytest.fixture(scope="function")
    def env_setup(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=True,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)
        yield client, cluster
        cluster.close()

    def get_single_bridge(self, arrays_metadata=None, comm_state=None):
        """Helper to create a single bridge (rank 0)."""
        if arrays_metadata is None:
            arrays_metadata = {
                'temperature': {
                    'global_shape': (1,),
                    'chunk_shape': (1,),
                    'chunk_position': (0,)
                }}
        if comm_state is None:
            comm_state = FakeComm.State(1)
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )
        return bridge, arrays_metadata

    def test_per_array_comm_isolation(self, env_setup):
        """Verify that _setup_array_comms creates per-array sub-communicators.

        The bridge layer uses comm.Split() to create sub-comms. For FakeComm,
        Split() returns self, so all sub-comms are identical. For real MPI,
        Split() would return distinct communicators.
        """
        client, cluster = env_setup

        comm_state = FakeComm.State(1)
        arrays_metadata = {
            'temperature': {
                'global_shape': (8, 8),
                'chunk_shape': (4, 4),
                'chunk_position': (0, 0)
            },
            'pressure': {
                'global_shape': (8, 8),
                'chunk_shape': (4, 4),
                'chunk_position': (0, 0)
            }}
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )

        # Bridge should have cached sub-communicators for each array
        assert 'temperature' in bridge._array_comms
        assert 'pressure' in bridge._array_comms

        # FakeComm.Split() returns a _SubComm (mimics MPI sub-comm creation)
        from utils import _SubComm
        assert isinstance(bridge._array_comms['temperature'], _SubComm)
        assert isinstance(bridge._array_comms['pressure'], _SubComm)
        assert bridge._array_comms['temperature'].Get_size() == 1
        assert bridge._array_comms['pressure'].Get_size() == 1

    def test_single_bridge_non_distributed(self, env_setup):
        """Verify single-bridge fast-path works without gather.

        When a bridge is the only participant (sub_comm_size == 1),
        it should skip the gather entirely and use _direct_send.
        """
        client, cluster = env_setup

        arrays_metadata = {
            'temperature': {
                'global_shape': (1,),
                'chunk_shape': (1,),
                'chunk_position': (0,),
            }}
        comm_state = FakeComm.State(1)
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )

        # Send data — should use fast-path (no gather)
        bridge.send('temperature', np.ones(1), timestep=0)

        # Verify the data was registered with the scheduler
        event = client.get_events('temperature')
        assert len(event) == 1
        _, info = event[0]
        assert info['array_name'] == 'temperature'
        assert info['iteration'] == 0
        assert len(info['futures']) == 1
        assert info['futures'][0]['chunk_position'] == (0,)

    def test_backward_compatibility(self, env_setup):
        """Verify existing all-bridges distribution still works.

        Multiple bridges with the same array (all participate).
        Bridges must be created in parallel because comm.Split() is a collective.
        """
        client, cluster = env_setup

        arrays_metadata = {
            'temperature': {
                'global_shape': (8, 8),
                'chunk_shape': (4, 4),
                'chunk_position': (0, 0)
            }}
        comm_state = FakeComm.State(4)

        def make_bridge(rank):
            return Bridge(comm=FakeCartComm(comm_state, rank, dims=(2, 2)),
                          arrays_metadata=arrays_metadata,
                          wait_for_go=False)

        # Create bridges in parallel (Split is a collective op)
        bridges = async_map(range(4), make_bridge)

        async def _bridge_send():
            await asyncio.gather(*[asyncio.to_thread(bridge.send, 'temperature',
                                                     np.ones(arrays_metadata['temperature']['chunk_shape']),
                                                     timestep=0)
                                   for i, bridge in enumerate(bridges)])

        asyncio.run(_bridge_send())

        event = client.get_events('temperature')
        assert len(event) == 1
        _, info = event[0]
        assert info['array_name'] == 'temperature'
        assert info['iteration'] == 0
        assert len(info['futures']) == 4
        for f in info['futures']:
            assert f['chunk_position'] in [(0, 0), (0, 1), (1, 0), (1, 1)]

        async def _bridge_close():
            await asyncio.gather(*[asyncio.to_thread(bridge.close, 0) for i, bridge in enumerate(bridges)])

        asyncio.run(_bridge_close())

    def test_comm_cleanup(self, env_setup):
        """Verify that bridge._array_comms is populated and can be cleaned up."""
        client, cluster = env_setup

        comm_state = FakeComm.State(1)
        arrays_metadata = {
            'temperature': {
                'global_shape': (1,),
                'chunk_shape': (1,),
                'chunk_position': (0,)
            }}
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )

        # Bridge should have per-array sub-comms cached
        assert 'temperature' in bridge._array_comms
        assert bridge._array_comms['temperature'] is not None

        # Free sub-comms manually (bridge.close() does this too)
        for sub_comm in bridge._array_comms.values():
            if sub_comm is not None:
                sub_comm.Free()
        bridge._array_comms.clear()
        assert len(bridge._array_comms) == 0

    def test_close_calls_cleanup(self, env_setup):
        """Verify that bridge.close() frees sub-communicators."""
        client, cluster = env_setup

        comm_state = FakeComm.State(1)
        arrays_metadata = {
            'temperature': {
                'global_shape': (1,),
                'chunk_shape': (1,),
                'chunk_position': (0,)
            }}
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )

        # Sub-comms should exist before close
        assert len(bridge._array_comms) > 0

        # Close should free sub-comms and clean up
        bridge.close(timestep=0)
        assert bridge._has_close_been_called
        assert len(bridge._array_comms) == 0

    def test_multi_bridge_mixed_participation(self, env_setup):
        """Two bridges, two arrays: one distributed on both, one on a subset.

        - temperature: both bridges declare it → sub-comm size 2 → gather
        - pressure: only bridge 0 declares it → bridge 0 fast-path (sub-comm size 1),
          bridge 1 is absent and sits out (_COMM_NULL for Split, never sends)
        """
        client, cluster = env_setup

        # Bridge 0 metadata: participates in both arrays
        arrays_metadata_0 = {
            'temperature': {
                'global_shape': (8,),
                'chunk_shape': (4,),
                'chunk_position': (0,),
            },
            'pressure': {
                'global_shape': (8,),
                'chunk_shape': (4,),
                'chunk_position': (0,),
            },
        }
        # Bridge 1 metadata: participates only in temperature
        # Pressure is absent — this bridge does not send it
        arrays_metadata_1 = {
            'temperature': {
                'global_shape': (8,),
                'chunk_shape': (4,),
                'chunk_position': (1,),
            },
        }
        comm_state = FakeComm.State(2)

        def make_bridge(rank, meta):
            return Bridge(comm=FakeComm(comm_state, rank),
                          arrays_metadata=meta,
                          wait_for_go=False)

        # Create bridges in parallel (Split is a collective)
        bridge0, bridge1 = async_map(
            [(0, arrays_metadata_0), (1, arrays_metadata_1)],
            lambda args: make_bridge(*args)
        )

        # Both bridges send temperature (gather path, sub-comm size 2)
        async def _send_temperature():
            await asyncio.gather(
                asyncio.to_thread(bridge0.send, 'temperature',
                                  np.ones(4), timestep=0),
                asyncio.to_thread(bridge1.send, 'temperature',
                                  np.ones(4) * 2, timestep=0),
            )

        asyncio.run(_send_temperature())

        # Only bridge 0 sends pressure (fast-path, sub-comm size 1)
        # Bridge 1 never sends pressure (not in its declared metadata)
        bridge0.send('pressure', np.ones(4) * 10, timestep=0)

        # Verify temperature: 2 futures (one from each bridge)
        event_temp = client.get_events('temperature')
        assert len(event_temp) == 1
        _, info_temp = event_temp[0]
        assert info_temp['iteration'] == 0
        assert len(info_temp['futures']) == 2
        positions_temp = {f['chunk_position'] for f in info_temp['futures']}
        assert positions_temp == {(0,), (1,)}

        # Verify pressure: 1 future (only bridge 0 sent)
        event_press = client.get_events('pressure')
        assert len(event_press) == 1
        _, info_press = event_press[0]
        assert info_press['iteration'] == 0
        assert len(info_press['futures']) == 1
        assert info_press['futures'][0]['chunk_position'] == (0,)

        # Clean up
        async def _close():
            await asyncio.gather(
                asyncio.to_thread(bridge0.close, 0),
                asyncio.to_thread(bridge1.close, 0),
            )

        asyncio.run(_close())
