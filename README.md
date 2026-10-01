# Deisa-Dask

**Dask-Enabled In Situ Analytics for HPC simulations.**

Deisa couples MPI-based scientific simulations with [Dask](https://www.dask.org/)
distributed computing, enabling real-time, scalable analysis of simulation data
as it is produced. Simulation data living on MPI ranks is streamed to Dask
workers, where callbacks run analytics (reductions, plotting, machine learning,
…) on the latest timestep without waiting for the simulation to finish.

This repository is the Dask integration layer of the Deisa project. It depends
on [`deisa-core`](https://github.com/deisa-project/deisa-core) for the shared
MPI-side protocol.

## How it works

Deisa runs two cooperating sides:

- **`Bridge`** — created inside the MPI simulation. Each rank builds a `Bridge`
  describing the arrays it owns (global shape, chunk shape, chunk position) and
  calls `bridge.send(name, local_data, timestep=ts)` each timestep to push its
  chunk to the Dask cluster.
- **`Deisa`** — runs on the analytics side (a Dask client). Callbacks are
  registered with `@deisa.register("array_name")` and are executed on the Dask
  workers whenever a new timestep for that array arrives. `deisa.execute_callbacks()`
  blocks until the simulation finishes and all callbacks have run.

The two sides connect through a Dask scheduler; the scheduler address is
communicated to the simulation via the `DEISA_DASK_SCHEDULER_ADDRESS`
environment variable.

## Key features

- **Per-array MPI communicators** — each distributed array gets its own
  communicator (`MPI.Comm.Dup()` / `MPI.Comm.Split()`), so arrays with different
  decomposition patterns coexist cleanly.
- **Non-distributed arrays** — arrays that live on a single rank are handled
  natively.
- **Round-robin worker scatter** — successive timesteps are scattered across
  workers so data is not piled onto a single worker.
- **Worker list control** — `update_worker` / `filter_worker` let a `Bridge`
  adjust which workers receive data before a `send()`.
- **MPI test harness** — the test suite runs both with a `FakeComm` mock (fast,
  `pytest-xdist`-parallel) and against real OpenMPI / MPICH.

## Installation

Requires Python ≥ 3.10.

```bash
# core
pip install deisa-dask

# with MPI support (for running simulations / MPI tests)
pip install "deisa-dask[mpi]"

# development / test dependencies
pip install "deisa-dask[test]"
```

## Getting started

A complete runnable example lives in [`example/getting-started/`](example/getting-started/).
It runs a 4-rank MPI simulation that pushes a 2-D Gaussian field each timestep,
and an analytics process that sums and plots the latest field.

**Simulation side** (`simulation.py`) — create a `Bridge`, describe your arrays,
and send each timestep:

```python
from mpi4py import MPI
from deisa.dask import Bridge

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

bridge = Bridge(
    comm=comm,
    arrays_metadata={
        "temperature": {
            "global_shape": (64, 64),
            "chunk_shape": (32, 32),
            "chunk_position": rank_coords,   # this rank's position in the cartesian grid
        }
    },
)

for ts in range(5):
    local_data = compute_local_field(ts)     # this rank's chunk
    bridge.send("temperature", local_data, timestep=ts)

bridge.close(timestep=5)
```

**Analytics side** (`analysis.py`) — register a callback and block until done:

```python
from deisa.dask import Deisa

deisa = Deisa()

@deisa.register("temperature")
def callback(temperatures):
    latest = temperatures[-1]
    print(f"t={latest.timestep} sum={latest.sum().compute()}")

deisa.execute_callbacks()
```

**Launch** (`launch.sh`) — start a Dask scheduler and worker, then the analytics
and simulation processes. The scheduler address is exported to the simulation
through `DEISA_DASK_SCHEDULER_ADDRESS`.

```bash
export DEISA_DASK_SCHEDULER_ADDRESS="tcp://localhost:8786"
mpirun -np 1 dask scheduler --scheduler-file scheduler.json &
mpirun -np 1 dask worker --scheduler-file scheduler.json &
mpirun -np 1 python3 analysis.py &
mpirun -np 4 python3 simulation.py
```

## Documentation

API documentation is published from the `site/` Hugo project. See the
[documentation website](https://deisa-project.github.io/deisa-dask/) for the
full reference.

## Development

```bash
git clone https://github.com/deisa-project/deisa-dask.git
cd deisa-dask
uv sync --extra test --extra mpi   # or: pip install -e ".[test,mpi]"

# run the non-MPI suite (FakeComm mock, parallel via pytest-xdist)
pytest -n auto

# run the MPI suite against a real MPI implementation
pytest test/test_mpi.py
```

Code style is enforced with [Ruff](https://docs.astral.sh/ruff/) (line length
120). The test suite covers the FakeComm harness, per-array communicators, handshake,
and MPI integration; CI runs MPI tests against both OpenMPI and MPICH and
tracks time-to-callback benchmarks.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) (Keep a Changelog, Semantic Versioning).

## License

[MIT](LICENSE) © deisa-project.