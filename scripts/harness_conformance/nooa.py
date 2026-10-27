# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOOA's borrowed-session fixture, backed by the suite's local subprocess provider.

In production the Resources server owns the task sandbox and grants it to the agent
(``sandbox_access``); the NOOA adapter only borrows it. This fixture gives the probe's
Resources side a local sandbox to grant, so the adapter runs its ordinary borrower path.
The worker inside it is ``nooa_bench.gym_worker`` on the probe's interpreter.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxHandle, SandboxSpec, register_provider
from nemo_gym.sandbox.providers.local.provider import LocalProvider

from .provider import Probe


class BorrowableLocalProvider(LocalProvider):
    """Reconnect to the Resources fixture's local workspace without acquiring ownership."""

    name = "conformance_local"

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id, "workspace": str(handle.raw["workspace"])}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        workspace = Path(descriptor["workspace"])
        if not workspace.is_dir():
            raise FileNotFoundError(workspace)
        return SandboxHandle(
            sandbox_id=descriptor["sandbox_id"],
            provider_name=self.name,
            raw={"workspace": workspace, "env": {}, "owned": False},
        )


@dataclass
class BorrowedSandboxProbe(Probe):
    """Seed and close the sandbox through Resources; NOOA runs its ordinary borrower path."""

    sandbox: AsyncSandbox | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        register_provider("conformance_local", BorrowableLocalProvider, override=True)

    async def seed_sandbox(self) -> dict:
        if self.sandbox is None:
            self.sandbox = AsyncSandbox(
                BorrowableLocalProvider(concurrency=1), SandboxSpec(workdir=str(self.directory / "workspace"))
            )
            await self.sandbox.start()
        return {
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "probe_sandbox",
                    "descriptor": await self.sandbox.serialize(),
                },
                "workdir": str(self.directory / "workspace"),
            }
        }

    async def close_sandbox(self) -> None:
        if self.sandbox is not None:
            await self.sandbox.stop()
