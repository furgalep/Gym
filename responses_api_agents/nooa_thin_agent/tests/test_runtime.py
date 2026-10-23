# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime staging picks the tarball that matches the sandbox image's libc."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from responses_api_agents.nooa_thin_agent import runtime


@dataclass
class _Result:
    return_code: int
    stdout: str = ""
    stderr: str = ""
    error_type: str | None = None


class _Sandbox:
    """Answers the libc probe, then reports every runtime as already staged and healthy."""

    def __init__(self, machine: str, libc: str):
        self.probe = f"{machine}\n{libc}\n"
        self.commands: list[str] = []

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float | None = None):
        self.commands.append(command)
        if command == runtime.LIBC_PROBE:
            return _Result(0, stdout=self.probe)
        return _Result(0)

    async def upload(self, src, dst):  # pragma: no cover - the healthy path never uploads
        raise AssertionError("no upload expected")


def _tarballs(tmp_path: Path) -> tuple[Path, Path]:
    gnu, musl = tmp_path / "gnu.tar.gz", tmp_path / "musl.tar.gz"
    gnu.write_bytes(b"gnu")
    musl.write_bytes(b"musl")
    return gnu, musl


def test_glibc_image_gets_the_glibc_runtime(tmp_path):
    gnu, musl = _tarballs(tmp_path)
    sandbox = _Sandbox("x86_64", "gnu")
    python = asyncio.run(runtime.stage_runtime(sandbox, gnu, musl_tarball=musl))
    assert python.startswith(f"{runtime.REMOTE_ROOT}/{runtime.file_sha256(gnu)[:16]}/")


def test_musl_image_gets_the_musl_runtime(tmp_path):
    gnu, musl = _tarballs(tmp_path)
    sandbox = _Sandbox("x86_64", "musl")
    python = asyncio.run(runtime.stage_runtime(sandbox, gnu, musl_tarball=musl))
    assert python.startswith(f"{runtime.REMOTE_ROOT}/{runtime.file_sha256(musl)[:16]}/")


def test_musl_image_without_a_musl_runtime_fails_before_upload(tmp_path):
    gnu, _ = _tarballs(tmp_path)
    sandbox = _Sandbox("x86_64", "musl")
    with pytest.raises(RuntimeError, match="musl"):
        asyncio.run(runtime.stage_runtime(sandbox, gnu))
    assert sandbox.commands == [runtime.LIBC_PROBE]


def test_build_targets_are_pinned():
    assert set(runtime.PYTHON_SHA256) >= {
        "x86_64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
    }
    assert runtime.python_url("x86_64-unknown-linux-musl").endswith(
        "x86_64-unknown-linux-musl-install_only_stripped.tar.gz"
    )
