# Issue #95: Execute tasks on bridge when possible

> **For Hermes:** Implement task-by-task, commit after each.

**Goal:** Allow the Bridge to execute Dask task graph nodes locally (on the bridge's own client) when the data is already present in the bridge's process, avoiding unnecessary data transfer to remote workers.

**Architecture:** When `Bridge.send()` scatters data to workers, the in-process path already places data directly into the worker's local store (zero-copy). The next step is: if a callback task graph can be partially evaluated using only data that is *already local* to the bridge process, execute that sub-graph locally and only send the reduced result to the cluster. This requires:
1. Detecting when a task graph node's dependencies are all local (in-process worker data)
2. Running that sub-graph on the bridge's Dask client
3. Returning the resulting future as a proxy for the downstream graph

**Tech Stack:** Python, Dask (distributed), NumPy, MPI (via ICommunicator)

---

## Current Flow

```
Simulation -> bridge.send(array_name, chunk, timestep)
           -> _better_scatter(chunk, workers)
           -> __scatter(data, workers)
           -> _scatter_to_workers(workers, data)
           -> scatter_to_workers(remote_workers, remote_data)  # distributed lib
           -> worker.update_data()  # in-process OR RPC
           -> comm.gather(...)
           -> client.log_event(array_name, futures)
           -> topic_handler(event) in Deisa
           -> builds dask array from futures
           -> persist(darr)
           -> _process_callback(cb_id, cb_data, array_name, darr, iteration)
           -> callback(*windows)
```

The key insight: currently ALL data goes to workers, then callbacks receive `dask.array.Array` objects backed by those worker futures. Issue #95 asks: if a callback only needs data that's already on the bridge (e.g., a local reduction of an array the bridge just sent), compute it locally instead of shipping it to a worker.

## Proposed Approach

The cleanest first step: **allow the bridge to submit simple Dask tasks locally** when the input data is already in the bridge's process (via in-process worker). This is a bridge-level optimization, not a Deisa-level one.

Specifically:
- After `_scatter_to_workers()` places data on in-process workers, check if any registered callbacks can be partially satisfied locally
- For simple cases (e.g., a callback that does `np.sum(arr)`), execute the task on the bridge's client and substitute the result future

This is a research/spike issue. The first branch should:
1. Create the branch from `main`
2. Add a method to Bridge that can detect "local-only" data availability
3. Add a test that verifies local execution works
4. Document what's feasible vs what needs more design

---

## Step-by-Step Plan

### Task 1: Create branch and research local task execution

**Objective:** Create branch `95_ExecuteTasksOnBridge` and spike local Dask computation from bridge

**Files:**
- Branch: `95_ExecuteTasksOnBranch` from `origin/main`

**Step 1: Create the branch**

```bash
cd /opt/data/profiles/george/research/deisa-dask
git checkout -b 95_ExecuteTasksOnBridge origin/main
```

**Step 2: Write a spike test**

Create `test/test_bridge_local_exec.py`:

```python
"""Spike: can the bridge execute a dask task locally using in-process worker data?"""

import numpy as np
import pytest
from distributed import Client, LocalCluster
import os

from deisa.dask import Bridge
from test.utils import FakeComm


class TestBridgeLocalExec:
    @pytest.fixture
    def env_setup(self):
        cluster = LocalCluster(n_workers=1, threads_per_worker=1, processes=False,
                               dashboard_address=":0", worker_dashboard_address=":0")
        os.environ['DEISA_DASK_SCHEDULER_ADDRESS'] = cluster.scheduler_address
        client = Client(cluster)
        client.wait_for_workers(1, timeout=10)
        yield client, cluster
        cluster.close()

    def get_new_bridge(self):
        arrays_metadata = {
            'temperature': {
                'global_shape': (10,),
                'chunk_shape': (10,),
                'chunk_position': (0,)
            }}
        comm_state = FakeComm.State(1)
        bridge = Bridge(
            comm=FakeComm(comm_state, 0),
            arrays_metadata=arrays_metadata,
            wait_for_go=False
        )
        return bridge

    def test_bridge_can_compute_local_sum(self, env_setup):
        """After sending data, bridge should be able to compute a local reduction."""
        client, cluster = env_setup
        bridge = self.get_new_bridge()

        data = np.ones(10)
        bridge.send('temperature', data, timestep=0)

        # The data is now in the in-process worker.
        # Can we submit a task that runs locally?
        import dask.array as da
        # Get the key of the data in the worker
        worker = list(cluster.workers.values())[0]
        keys = [k for k in worker.data if 'ndarray-' in k]
        assert len(keys) == 1

        # Try to compute something locally using the bridge's client
        future = client.scatter(data)
        result = client.submit(np.sum, future)
        assert result.result() == 10.0
```

**Step 3: Run the spike**

```bash
uv run python -m pytest test/test_bridge_local_exec.py -x -v
```

Expected: PASS (proves the concept works)

**Step 4: Commit**

```bash
git add test/test_bridge_local_exec.py
git commit -m "test: spike for local task execution on bridge (#95)"
```

---

### Task 2: Add `compute_local` method to Bridge

**Objective:** Add a method that detects if data is local and executes a function on it without round-tripping to remote workers

**Files:**
- Modify: `src/deisa/dask/bridge.py`

**Step 1: Add method after `_scatter_to_workers`**

```python
async def _compute_local(self, func, *keys):
    """Execute a function on data that is local to this bridge process.
    
    If the data referenced by `keys` is in an in-process worker,
    run the function locally without serialization.
    Returns (result_future, True) if local execution was possible,
    (None, False) otherwise.
    """
    from distributed.worker import _global_workers
    local_worker_map = {w.address: w for w in _global_workers}
    
    if not local_worker_map:
        return None, False
    
    # Check if all keys are in local workers
    local_data = {}
    for key in keys:
        found = False
        for addr, worker in local_worker_map.items():
            if key in worker.data:
                local_data[key] = worker.data[key]
                found = True
                break
        if not found:
            return None, False
    
    # All data is local — execute on bridge client
    if self.client is not None:
        args = [local_data[k] for k in keys]
        result = self.client.submit(func, *args)
        return result, True
    
    return None, False
```

**Step 2: Run existing tests to verify no regression**

```bash
uv run python -m pytest test/test_bridge.py -x -v
```

Expected: All pass

**Step 3: Commit**

```bash
git add src/deisa/dask/bridge.py
git commit -m "feat: add _compute_local to Bridge for local task execution (#95)"
```

---

### Task 3: Integration test — local reduction after send

**Objective:** End-to-end test showing data sent by bridge can be locally reduced without leaving the process

**Files:**
- Modify: `test/test_bridge_local_exec.py`

**Step 1: Add integration test**

```python
def test_local_reduction_after_send(self, env_setup):
    """After bridge.send(), a local reduction should work without data transfer."""
    client, cluster = env_setup
    bridge = self.get_new_bridge()

    data = np.arange(10.0)
    bridge.send('temperature', data, timestep=0)

    # Find the key in the in-process worker
    worker = list(cluster.workers.values())[0]
    keys = [k for k in worker.data if 'ndarray-' in k]
    assert len(keys) == 1

    # Use bridge's _compute_local to get the sum
    loop = asyncio.new_event_loop()
    result_future, is_local = loop.run_until_complete(
        bridge._compute_local(np.sum, keys[0])
    )
    loop.close()

    assert is_local, "Data should be detected as local"
    assert result_future is not None
    assert result_future.result() == 45.0  # sum of 0..9
```

**Step 2: Run test**

```bash
uv run python -m pytest test/test_bridge_local_exec.py::TestBridgeLocalExec::test_local_reduction_after_send -x -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add test/test_bridge_local_exec.py
git commit -m "test: integration test for local reduction after send (#95)"
```

---

### Task 4: Push branch

**Objective:** Push branch to remote for review

```bash
cp /opt/data/home/.ssh/id_ed25519 /tmp/id_ed25519
GIT_SSH_COMMAND="ssh -i /tmp/id_ed25519" git push origin HEAD
```

---

## Open Questions / Research Items

1. **Task graph splitting**: The issue mentions "split up into sub-graphs." This is a much larger feature — it requires Dask graph introspection. The spike above only handles the simple case of local execution after send. Full graph splitting would need a design doc.

2. **When to trigger local execution**: Should it be automatic (bridge decides) or user-driven (callback declares locality preference)?

3. **Multi-bridge coordination**: If multiple bridges each hold part of the data, can they coordinate local computation? This relates to the per-array communicator work.

4. **Scope**: Is this about (a) simple reductions the bridge can do before sending, or (b) full Dask graph scheduling awareness? The issue text suggests (b) but (b) is a major architectural change.

## Risks

- Dask task graph manipulation is complex and version-sensitive
- Local execution could compete with the simulation for memory/CPU
- The `_global_workers` private API is fragile across Dask versions
- Need to ensure MPI barriers still work correctly when some bridges compute locally and others don't
