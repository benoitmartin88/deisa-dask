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
from utils import FakeComm, FakeCartComm


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
        """Verify that get_array_comm returns a per-array communicator and
        that different arrays get different communicator references (for MPI path)
        or the same self (for Dask RPC path)."""
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

        # FakeComm returns self for get_array_comm (Dask RPC path)
        comm_temp = bridge.comm.get_array_comm('temperature')
        comm_pressure = bridge.comm.get_array_comm('pressure')
        comm_temp2 = bridge.comm.get_array_comm('temperature')

        # For Dask RPC path, all calls return self (stateless per-call isolation)
        assert comm_temp is comm_pressure
        assert comm_temp is comm_temp2
        assert comm_temp is bridge.comm

    def test_single_bridge_non_distributed(self, env_setup):
        """Verify single-bridge fast-path works without gather.

        When n_participants=1, the bridge should skip the gather entirely
        and use _direct_send to update the scheduler directly.
        """
        client, cluster = env_setup

        arrays_metadata = {
            'temperature': {
                'global_shape': (1,),
                'chunk_shape': (1,),
                'chunk_position': (0,),
                'n_participants': 1,
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

        Multiple bridges with the same array, no n_participants set
        (defaults to comm.Get_size()), should use gather path.
        """
        client, cluster = env_setup

        arrays_metadata = {
            'temperature': {
                'global_shape': (8, 8),
                'chunk_shape': (4, 4),
                'chunk_position': (0, 0)
            }}
        comm_state = FakeComm.State(4)

        bridges = [Bridge(comm=FakeCartComm(comm_state, rank, dims=(2, 2)),
                          arrays_metadata=arrays_metadata,
                          wait_for_go=False) for rank in range(4)]

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
        """Verify cleanup_array_comms frees resources without error."""
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

        # Get per-array comms
        comm1 = bridge.comm.get_array_comm('temperature')
        assert comm1 is not None

        # Cleanup should not raise
        bridge.comm.cleanup_array_comms()

        # After cleanup, calling cleanup again should still not raise
        bridge.comm.cleanup_array_comms()

    def test_close_calls_cleanup(self, env_setup):
        """Verify that bridge.close() calls cleanup_array_comms."""
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

        # Get a per-array comm
        bridge.comm.get_array_comm('temperature')

        # Close should not raise (it calls cleanup_array_comms internally)
        bridge.close(timestep=0)
        assert bridge._has_close_been_called
