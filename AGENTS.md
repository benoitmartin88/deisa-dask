# AGENTS.md

Guidance for AI coding agents working in this repository, following the
[AGENTS.md](https://agents.md) convention. Human contributors should read
[CONTRIBUTING.md](CONTRIBUTING.md); this file is the agent-oriented digest of
the same rules plus the gotchas that bite in practice.

## Project overview

Deisa is an in-situ analytics framework for MPI-based simulations. Two sides
meet at a topic:

- `Bridge` (`src/deisa/dask/bridge.py`) — the simulation side. One bridge per
  MPI rank; sends array chunks to Dask workers and publishes topic events.
- `Deisa` (`src/deisa/dask/deisa.py`) — the analytics side. Connects to the
  Dask cluster, subscribes to topics, runs user callbacks.

Repository layout:

- `src/deisa/dask/` — the package (`bridge.py`, `deisa.py`, and support
  modules).
- `test/` — the test suite. `test/utils.py` provides `FakeComm` /
  `FakeCartComm`, so most logic is testable without an MPI runtime.
- `example/getting-started/` — a runnable 4-rank simulation + analytics
  example; the canonical client shape.
- `benchmark/` — benchmark scripts; result artifacts are untracked.
- `site/` — the Hugo documentation website.

## Setup

```bash
uv sync --extra test --extra mpi
# or: pip install -e ".[test,mpi]"
```

Requires Python ≥ 3.10.

## Commands

```bash
pytest test/ -n auto --dist loadgroup     # full non-MPI suite
pytest test/test_mpi.py -p no:xdist       # MPI tests (needs OpenMPI or MPICH)
ruff check src test benchmark example     # lint
ruff format --check src test benchmark example
```

Notes:

- Most tests run through the `FakeComm` mock; the non-MPI suite must pass
  without an MPI runtime installed.
- MPI tests are parametrized over process counts; some are `mpirun`-only and
  are skipped in plain runs. Run them serially (`-p no:xdist`), never under
  xdist.
- Each test has a 300 s timeout (`pytest.ini`). A test that hangs fails there;
  do not raise the timeout to make a test pass.

## Code style

- [Ruff](https://docs.astral.sh/ruff/) enforces style; the CI lint workflow
  (`.github/workflows/lint.yml`) is the source of truth.
- Line length **120**, everywhere — code, comments, docstrings, and the
  message literals of `raise` calls.
- Rules selected: `E`, `F`, `I`. Run both `ruff check` and `ruff format` before
  committing; a pre-commit hook runs both.
- Match the style of the surrounding code; when wrapping an underlying async
  method, the sync wrapper gets a different name than the async method it
  wraps (no shadowing).

## Testing instructions

- Add or update tests for any code you change.
- A change is not done until the full non-MPI suite passes (and
  `test/test_mpi.py` when the change touches bridge/communicator code and an
  MPI runtime is available).
- When fixing a regression, verify the new test fails without the fix and
  passes with it.
- Assert message **content**, not just exception type: a refusal/error contract
  test should also check the remedy the message documents.

## Conventions

- Branch names: `feature/<name>` for new work, `fix/<name>` for bug fixes.
- **Never push directly to `main`.** All changes land via pull request.
- One logical change per branch, based on the latest `main`.
- Keep diffs minimal and review-relevant. Do not reformat, reorder, or
  "clean up" code outside the issue scope — formatter churn on untouched code
  costs the reviewer a question per hunk. After any format pass, confirm every
  diff hunk is yours.
- Record user-visible changes in [CHANGELOG.md](CHANGELOG.md) (Keep a
  Changelog) under the appropriate `Unreleased` section, in the same PR.
- Do not reference `.md` files from code — comments, docstrings, and error
  messages must be self-contained.
- Commits must be authored with the contributor's real identity (`git config
  user.name` / `user.email`).

## Gotchas

- **Multi-bridge tests deadlock if bridges are created sequentially.** All
  bridges must exist before the first collective: construct them concurrently
  (`async_map`-style) over a single shared `FakeComm.State`, never one after
  another.
- **Close what you open.** A test fixture that creates a `LocalCluster` /
  `Client` must close both in teardown, and must not leave scheduler-address
  environment variables set — a leak surfaces as reconnect storms and flaky
  failures in unrelated tests far from the fixture.
- Internal `distributed.Client` instances must pass `set_as_default=False`; a
  client built with the default owns the process-global dask scheduler for its
  entire lifetime and breaks unrelated code on the same worker.
- Communicator splits that leave ranks out return a null communicator with
  `Get_size() == 0` — detect participation with `sub_comm.Get_size() == 0`,
  never by comparing against `None` or a `_COMM_NULL` sentinel.

## Pull requests

1. Push the branch to your fork.
2. Open the PR against `main` with a clear title and a description of the
   change and its motivation.
3. Ensure CI passes: lint, the unit-test matrix (Python 3.10–3.12 × several
   Dask versions), and the MPI tests (OpenMPI and MPICH).
4. Address review feedback; keep the branch rebased on `main` as needed.
