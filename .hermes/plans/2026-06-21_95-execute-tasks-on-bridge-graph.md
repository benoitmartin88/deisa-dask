# Issue #95: Execute tasks on bridge when possible (Graph Splitting)

> **For Hermes:** Implement task-by-task, commit after each.

**Goal:** Allow the Bridge to execute parts of a Dask task graph locally (on the bridge's own client) when the input data is already in-process, splicing the resulting Future into the larger graph so downstream computation treats it as a regular scattered chunk.

**Architecture:** The bridge already has a Dask client (rank 0) connected to the same scheduler as Deisa. After `scatter_to_workers()` places data in an in-process worker, the bridge can submit Dask tasks that operate on that data locally (zero serialization). The result is a `distributed.Future` that can be wrapped with `da.from_delayed()` and used in the normal `topic_handler` → callback flow. The key mechanism:

```
raw data (local) → bridge sub-graph (local) → Future → da.from_delayed() → normal flow
```

vs. today:

```
raw data → scatter to worker → worker future → da.from_delayed() → normal flow
```

**Tech Stack:** Python, Dask (distributed), NumPy, MPI (via ICommunicator)

---

## Current Flow (what changes)

```
Simulation
  → bridge.send(array_name, chunk, timestep)
    → _better_scatter(chunk, workers)
      → __scatter(data, workers)
        → _scatter_to_workers(workers, data)
          → in-process: worker.update_data()  [zero-copy]
          → remote: scatter_to_workers() + RPC
    → comm.gather(...)
    → client.log_event(array_name, {futures: [...]})

Deisa.topic_handler(event)
  → futures = [Future(d['future']) for d in payload['futures']]
  → darr_chunks = [da.from_delayed(f, ...) for f in futures]
  → darr = tile_dask_blocks(darr_chunks)
  → persist(darr)
  → _process_callback(cb_id, cb_data, array_name, darr, iteration)
    → callback(*windows)
```

## Proposed Flow (with local execution)

```
Simulation
  → bridge.send(array_name, chunk, timestep, graph=task_graph)
    → _scatter_to_workers(workers, data)   [places data in workers]
    → _execute_local_subgraph(task_graph)  [NEW: compute local part]
      → identify nodes whose deps are all in-process
      → submit to bridge client
      → return Futures
    → comm.gather({..., local_futures: [...]})

Deisa.topic_handler(event)
  → futures = [Future(d['future']) for d in payload['futures']]
  → local_futures = [Future(d['future']) for d in payload.get('local_futures', [])]
  → darr_chunks = [da.from_delayed(f, ...) for f in all_futures]
  → ... (same as before)
```

The critical insight: **the bridge's client and Deisa's client connect to the same scheduler**. A Future computed by the bridge is visible to Deisa. `da.from_delayed(dask.delayed(future))` works identically whether the future came from a scattered worker or a local computation.

---

## Design Decisions

### 1. How does the bridge know what to compute locally?

The simulation passes an optional `graph` parameter to `send()`. This is a Dask delayed/graph object representing the computation the callback will perform. The bridge inspects the graph's tokenized keys and checks which input data is already in-process.

**Alternative (simpler):** The bridge always tries to execute a fixed set of "local-friendly" operations (reductions, normalizations) on in-process data before scattering. No graph parameter needed from the simulation.

**Preferred:** Start with a simple `local_transform` callable parameter — the simulation provides a function that the bridge runs locally on the data before scattering. This is explicit, testable, and doesn't require graph introspection. Can evolve to full graph splitting later.

### 2. How does the result enter the normal flow?

The bridge's `send()` already returns futures via `client.log_event()`. We extend the event payload to include locally-computed futures. The `topic_handler` in `deisa.py` already handles `da.from_delayed()` on futures — it just needs to include the local ones in the chunk list.

### 3. What about MPI coordination?

All bridges must agree on what was computed locally vs. scattered. The `comm.gather()` after scatter already aggregates metadata from all bridges. Local futures are included in this gather, so rank 0 sees the complete picture.

### 4. What about the `chunk_position` field?

Locally-computed results may represent a reduction (smaller shape) or a transformed chunk (same shape). The `placement` field already handles this — it's the bridge's MPI coordinate, not the array position. The shape/dtype in the future metadata tells Deisa how to use it.

---

## Step-by-Step Plan

### Task 1: Create branch and write spike test

**Objective:** Prove that a Future computed locally by the bridge can be used in `da.from_delayed()` within the normal Deisa flow.

**Files:**
- Branch: `95_ExecuteTasksOnBridge` from `origin/main`
- Create: `test/test_bridge_local_graph.py`

**Step 1: Create branch**

```bash
cd /opt/data/profiles/george/research/deisa-dask
git checkout -b 95_ExecuteTasksOnBridge origin/main
```

**Step 2: Write spike test**

```python
"""Spike: can a locally-computed Future be spliced into the normal dask array flow?"""

import asyncio
import numpy as np
import pytest
from distributed import Client, LocalCluster
import dask
import dask.array as da
import os

from deisa.dask import Bridge
from test.utils import FakeComm


class TestBridgeLocalGraph:
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

    def test_local_future_splices_into_dask_array(self, env_setup):
        """A Future computed locally by the bridge can be used in da.from_delayed()."""
        client, cluster = env_setup
        bridge = self.get_new_bridge()

        data = np.arange(10.0)
        bridge.send('temperature', data, timestep=0)

        # Simulate: bridge computes a local reduction
        # (in the real feature, this would be triggered by a graph parameter)
        worker = list(cluster.workers.values())[0]
        keys = [k for k in worker.data if 'ndarray-' in k]
        assert len(keys) == 1

        # The bridge's client submits a task using the in-process data
        raw = worker.data[keys[0]]
        local_result = client.submit(np.sum, raw)
        assert local_result.result() == 45.0

        # Can we use this Future in the normal flow?
        # i.e., build a dask array chunk from it?
        local_arr = da.from_delayed(
            dask.delayed(local_result),
            shape=(),
            dtype=float
        )
        assert local_arr.compute() == 45.0

        # Can we combine it with a 'normal' chunk?
        normal_future = client.scatter(np.ones(5))
        normal_arr = da.from_delayed(
            dask.delayed(normal_future),
            shape=(5,),
            dtype=float
        )
        combined = da.concatenate([local_arr[np.newaxis], normal_arr])
        result = combined.compute()
        assert result[0] == 45.0
        assert np.all(result[1:] == 1.0)
```

**Step 3: Run spike**

```bash
uv run python -m pytest test/test_bridge_local_graph.py -x -v
```

Expected: PASS

**Step 4: Commit**

```bash
git add test/test_bridge_local_graph.py
git commit -m "test: spike for local graph execution on bridge (#95)"
```

---

### Task 2: Add `local_transform` parameter to `send()`

**Objective:** Extend `send()` to accept an optional callable that transforms data locally before scattering.

**Files:**
- Modify: `src/deisa/dask/bridge.py` (lines 147-242, the `send()` method)
- Modify: `.venv/.../deisa/core/interface.py` (the `IBridge.send` signature — or skip if we don't control deisa-core)

**Step 1: Add parameter to send()**

```python
def send(self, array_name: str, chunk: np.ndarray, timestep: int, *args,
         local_transform=None, **kwargs):
    """
    ...
    :param local_transform: Optional callable(data) -> transformed_data.
        If provided and data is in-process, apply this transform locally
        before scattering. The transformed result is what gets sent to workers.
    """
    ...
    # After _scatter_to_workers, before comm.gather:
    local_futures = []
    if local_transform is not None and self.client is not None:
        local_futures = self._try_local_transform(array_name, chunk, timestep, local_transform)
    
    # Include local_futures in the gather payload
    to_send = {
        'future-info': res,
        'placement': ...,
        'local-futures': local_futures,  # NEW
    }
```

**Step 2: Add `_try_local_transform` method**

```python
def _try_local_transform(self, array_name, chunk, timestep, transform):
    """Apply a local transform to data if it's in-process.
    
    Returns a list of future-info dicts that can be included in the
    event payload and used by the topic_handler.
    """
    from distributed.worker import _global_workers
    
    local_workers = {w.address: w for w in _global_workers}
    if not local_workers:
        return []
    
    # Check if the scattered data landed on a local worker
    # (it did, since we use in-process path)
    try:
        result = self.client.submit(transform, chunk)
        # Register with scheduler
        self.client.sync(self.client.scheduler.update_data,
                         who_has={result.key: ['local']},
                         nbytes={result.key: chunk.nbytes})
        return [{
            'future': result.key,
            'shape': chunk.shape,  # may differ if transform changes shape
            'dtype': str(chunk.dtype),
            'placement': self.comm.Get_coords(self.id) if hasattr(self.comm, 'Get_coords') else self.id,
            'local': True,
        }]
    except Exception as e:
        logger.warning(f"[{self.id}] _try_local_transform failed: {e}")
        return []
```

**Step 3: Run all tests**

```bash
uv run python -m pytest test/test_bridge.py test/test_bridge_local_graph.py -x -v
```

Expected: All pass

**Step 4: Commit**

```bash
git add src/deisa/dask/bridge.py
git commit -m "feat: add local_transform parameter to bridge.send() (#95)"
```

---

### Task 3: Extend topic_handler to handle local futures

**Objective:** Modify `deisa.py`'s `_make_topic_handler` to include locally-computed futures in the dask array construction.

**Files:**
- Modify: `src/deisa/dask/deisa.py` (lines 320-366, the `_make_topic_handler` method)

**Step 1: Update topic_handler**

```python
async def topic_handler(event):
    ...
    futures = payload["futures"]
    local_futures_info = payload.get("local-futures", [])
    
    # Convert all to Future objects
    all_futures = tuple({**d, "future": Future(d["future"], client=_weak_self.client)} for d in futures)
    
    # Add local futures
    for lf in local_futures_info:
        all_futures = all_futures + ({
            **lf,
            "future": Future(lf["future"], client=_weak_self.client),
        },)
    
    _weak_self.__update_futures_ownership(all_futures)
    parts = sorted(all_futures, key=lambda p: p['placement'])
    darr_chunks = [da.from_delayed(p["future"], shape=p["shape"], dtype=p["dtype"]) for p in parts]
    ...
```

**Step 2: Run tests**

```bash
uv run python -m pytest test/ -x -v -k "not test_deisa"  # skip slow tests
```

Expected: All pass

**Step 3: Commit**

```bash
git add src/deisa/dask/deisa.py
git commit -m "feat: include local futures in topic_handler dask array construction (#95)"
```

---

### Task 4: Integration test — end-to-end with local_transform

**Objective:** Full test showing simulation → bridge.send() with local_transform → callback receives correct data.

**Files:**
- Modify: `test/test_bridge_local_graph.py`

**Step 1: Add integration test**

```python
def test_send_with_local_transform(self, env_setup):
    """bridge.send() with local_transform computes locally before scattering."""
    client, cluster = env_setup
    bridge = self.get_new_bridge()

    data = np.arange(10.0)
    
    # A local transform: compute sum locally
    def compute_sum(arr):
        return np.array([np.sum(arr)])
    
    bridge.send('temperature', data, timestep=0, local_transform=compute_sum)
    
    # Check that the event includes local-futures
    events = client.get_events('temperature')
    assert len(events) == 1
    _, payload = events[0]
    
    local_futures = payload.get('local-futures', [])
    assert len(local_futures) == 1
    
    # The local future should hold the sum
    from distributed import Future
    lf = Future(local_futures[0]['future'], client=client)
    assert lf.result() == 45.0
```

**Step 2: Run**

```bash
uv run python -m pytest test/test_bridge_local_graph.py::TestBridgeLocalGraph::test_send_with_local_transform -x -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add test/test_bridge_local_graph.py
git commit -m "test: end-to-end test for local_transform in send() (#95)"
```

---

### Task 5: Push branch

```bash
cp /opt/data/home/.ssh/id_ed25519 /tmp/id_ed25519
GIT_SSH_COMMAND="ssh -i /tmp/id_ed25519" git push origin HEAD
```

---

## Risks & Open Questions

1. **Graph introspection complexity**: The plan above uses a simple `local_transform` callable. True graph splitting (inspecting a Dask graph, identifying local-executable sub-graphs automatically) is significantly more complex and should be a follow-up.

2. **Shape changes**: If the local transform changes the shape (e.g., reduction), the `topic_handler` must handle mixed-shape chunks. The current `__tile_dask_blocks` assumes all chunks have the same shape. This needs testing.

3. **MPI coordination**: All bridges must agree on what was computed locally. The `comm.gather()` handles this, but if one bridge computes locally and another doesn't (different data placement), the gathered metadata must be consistent.

4. **Scheduler registration**: Locally-computed futures must be registered with the scheduler so they persist (not garbage collected). The `update_data` + `client-desires-keys` pattern from the existing flow must be replicated.

5. **The `local_transform` is a stepping stone**: It's explicit and simple. The full vision (automatic graph splitting) would require the bridge to understand the Dask graph structure and make scheduling decisions — that's a much bigger feature.

---

## What This Doesn't Do (Yet)

- **Automatic graph splitting**: The simulation must explicitly provide `local_transform`. Automatic detection of "local-friendly" sub-graphs is future work.
- **Multi-step local pipelines**: The current design handles a single transform. A chain of local operations would need the transform to encapsulate the full sub-graph.
- **Optimization heuristics**: No cost model for "should this be local or remote?" — that's a scheduler concern.
