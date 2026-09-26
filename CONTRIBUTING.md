# Contributing to Deisa-Dask

Thanks for considering a contribution! This document covers how to set up the
project, the conventions to follow, and how to get changes reviewed and merged.

## Project layout

- `src/deisa/dask/` — the package. `Bridge` (MPI simulation side) and `Deisa`
  (Dask analytics side) are the two public entry points.
- `test/` — the test suite. `test/utils.py` provides `FakeComm` / `FakeCartComm`
  so most logic is testable without a real MPI runtime.
- `example/getting-started/` — a runnable 4-rank simulation + analytics example.
- `site/` — the Hugo documentation website.

## Setting up a development environment

Requires Python ≥ 3.10.

```bash
git clone https://github.com/deisa-project/deisa-dask.git
cd deisa-dask

# install the package with test and MPI extras
uv sync --extra test --extra mpi
# or, with pip:
pip install -e ".[test,mpi]"
```

## Branching and workflow

- Branch names follow `feature/<name>` for new work and `fix/<name>` for
  bug fixes.
- **Never push directly to `main`.** All changes land via pull request.
- Branch per concern: keep each branch focused on a single logical change so it
  stays easy to review. If you find yourself mixing unrelated edits, split them.
- Base your branch on the latest `main` before opening a PR.

The repository uses git worktrees for parallel branches; you may find

```bash
git worktree add -b feature/<name> ../deisa-dask-<name> main
```

convenient, but a normal branch in a single checkout works just as well.

## Code style

Style is enforced with [Ruff](https://docs.astral.sh/ruff/) and matches the CI
lint workflow (`.github/workflows/lint.yml`):

- Line length **120**.
- Rules selected: `E`, `F`, `I` (pycodestyle errors, pyflakes, isort).

Check both lint and format locally before committing:

```bash
ruff check src test benchmark example
ruff format --check src test benchmark example
```

A pre-commit hook runs both checks on every commit.

## Testing

Run the full non-MPI suite in parallel with [pytest-xdist](https://pytest-xdist.readthedocs.io/):

```bash
pytest test/ -n auto --dist loadgroup
```

The MPI integration tests exercise real MPI and are run with a compatible
OpenMPI or MPICH installation:

```bash
pytest test/test_mpi.py
```

Notes:

- Most logic is covered through the `FakeComm` mock, so the non-MPI suite
  should pass without an MPI runtime installed.
- MPI tests are parametrized over process counts; some are `mpirun`-only and
  are skipped in plain runs.
- CI runs the suite against Python 3.10–3.12 and several Dask versions, and
  runs `test_mpi.py` against both OpenMPI and MPICH.

## Changelog

All user-visible changes are recorded in [CHANGELOG.md](CHANGELOG.md), which
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Add an entry under
the appropriate `Unreleased` section (`Added`, `Changed`, `Fixed`, …) in the
same PR as your change.

## Commit identity

Commits should be authored as the contributor's real identity so the history
stays attributable:

```bash
git config user.name "Your Name"
git config user.email "you@example.com"
```

## Opening a pull request

1. Push your branch to your fork.
2. Open a PR against `main` with a clear title and a description of the change
   and its motivation.
3. Ensure CI passes: lint, the unit-test matrix, and the MPI tests.
4. Address review feedback; keep the branch rebased on `main` as needed.

## Reporting issues

Please open an issue at the
[issue tracker](https://github.com/deisa-project/deisa-dask/issues) with a
minimal reproduction when possible, including the Python, Dask, and MPI
versions you are using.

## License

By contributing you agree that your contributions are licensed under the
[MIT License](LICENSE).