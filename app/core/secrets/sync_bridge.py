"""Sync/async bridge for secret resolution in synchronous pipeline builds."""

from __future__ import annotations

import asyncio
from typing import TypeVar

T = TypeVar("T")


def run_async(coro) -> T:
    """Run an async coroutine from synchronous call sites (e.g. PipelineFactory.build)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    raise RuntimeError(
        "Cannot call synchronous PipelineFactory.build() from a running event loop; "
        "use await factory.build_async(...) instead."
    )
