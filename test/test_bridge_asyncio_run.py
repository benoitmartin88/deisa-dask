"""Regression test: ``asyncio.run`` must not be called with a running loop.

Before this fix, ``Bridge._scatter_full`` and ``Bridge._direct_send`` called ``asyncio.run(...)`` directly on the
no-client path. ``asyncio.run`` raises ``RuntimeError: asyncio.run() cannot be called from a running event loop`` when
the calling thread already has a running loop, which happens during bridge teardown.
That surfaced as ``ERROR:TestSimulator:Error while closing bridges: asyncio.run() cannot be called from a running event
loop`` and left the ``test_register_callback`` family reporting spurious teardown errors.

These tests exercise the helpers directly and through the real code paths, so a
regression fails here instead of surfacing as flaky teardown noise.
"""

import asyncio

import numpy as np
import pytest

from deisa.dask.bridge import Bridge
from deisa.dask.utils import run_coro_on_private_loop


async def _noop():
    return "done"


def test_run_async_without_a_running_loop_uses_asyncio_run():
    """No loop on this thread: the helper takes the cheap asyncio.run path."""
    bridge = Bridge.__new__(Bridge)  # no __init__: no cluster, no comm
    assert bridge._run_async(_noop()) == "done"


def test_run_async_with_a_running_loop_does_not_raise():
    """A loop IS running: asyncio.run would raise. The helper must still work."""
    bridge = Bridge.__new__(Bridge)
    result = {}

    async def main():
        # Deliberately call the SYNCHRONOUS helper from inside a running loop.
        # This is the situation that used to raise RuntimeError.
        result["value"] = bridge._run_async(_noop())

    asyncio.run(main())
    assert result["value"] == "done"


def test_scatter_full_no_client_path_under_a_running_loop():
    """The real call site: ``_scatter_full`` must survive a running loop.

    ``_scatter_full`` is what bridge teardown reaches on the no-client path, so it is where ``asyncio.run`` used to
    raise. Covering only ``_run_async`` is not enough: a change that fixed the helper while leaving a call site on bare
    ``asyncio.run`` would still pass and still fail in production.
    """
    bridge = Bridge.__new__(Bridge)  # no __init__: no cluster, no comm
    bridge.client = None  # force the no-client branch
    bridge.id = "test-bridge"  # _scatter_full logs f"[{self.id}]"
    bridge.workers = ["w-0"]  # workers=None path reads this

    captured = {}

    async def fake_scatter_blocking(workers, data, hash=False):
        captured["called"] = True
        return {"future": ["k"], "who_has": workers, "nbytes": {}}

    bridge._scatter_blocking = fake_scatter_blocking

    result = {}

    async def main():
        # Synchronous bridge call from inside a running loop: the failure condition.
        result["value"] = bridge._scatter_full(np.zeros(4), workers=None)

    asyncio.run(main())
    assert captured.get("called") is True, "the no-client scatter path was not taken"
    assert result["value"]["future"] == ["k"]


def test_private_loop_helper_propagates_exceptions():
    """An exception inside the coroutine must surface on the caller, not vanish."""

    async def boom():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_coro_on_private_loop(boom())


def test_private_loop_helper_returns_value():
    """Sanity check on the private-loop path outside a running loop too."""

    async def compute():
        await asyncio.sleep(0)
        return 42

    assert run_coro_on_private_loop(compute()) == 42
