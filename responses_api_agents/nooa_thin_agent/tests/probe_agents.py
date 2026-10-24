# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harness classes the budget-stop tests run inside the worker subprocess.

The worker imports them by ``module:Class``; the tests put this directory on the worker's
PYTHONPATH. Both make at least one model call before they misbehave, so a test can tell
"the probe ran" from "the probe failed to load".
"""

from __future__ import annotations

import asyncio

from nooa import strategy
from nooa_bench.gym import ShellAgent, ToolLoopStrategy


class SlowCleanupAgent(ShellAgent):
    """A cancelled run whose teardown takes longer than the supervisor's TERM grace."""

    cleanup_seconds: float = 15

    @strategy(ToolLoopStrategy(("shell",)))
    async def act(self, task: str) -> str:
        """Complete the task with the shell tool, then reply with the final answer as plain text."""
        ...

    async def solve(self, task: str) -> str:  # plain orchestrator around the generated method
        try:
            return await self.act(task)
        finally:
            # Heavy harnesses tear down child sessions here; the checkpoint must not wait for it.
            await asyncio.sleep(self.cleanup_seconds)


class SelfCancellingAgent(ShellAgent):
    """An internal cancellation of the worker's task, raised from inside a tool call."""

    async def shell(self, command: str) -> str:
        asyncio.current_task().cancel()
        await asyncio.sleep(0)
        return "unreachable"
