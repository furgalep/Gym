# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime staging picks the tarball that matches the sandbox image's libc."""

from __future__ import annotations

import asyncio
import shlex
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
        if command.endswith(runtime.LIBC_PROBE):
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
    assert len(sandbox.commands) == 1 and sandbox.commands[0].endswith(runtime.LIBC_PROBE)


def test_staging_commands_put_system_dirs_first_on_path(tmp_path):
    gnu, _ = _tarballs(tmp_path)
    sandbox = _Sandbox("x86_64", "gnu")
    asyncio.run(runtime.stage_runtime(sandbox, gnu))
    assert all(command.startswith(runtime.SANE_PATH) for command in sandbox.commands), sandbox.commands


class _ColdSandbox(_Sandbox):
    """Nothing staged yet: the health check fails until the staging script has run."""

    def __init__(self, machine: str, libc: str, *, staging_exit: int = 0):
        super().__init__(machine, libc)
        self.staging_exit = staging_exit
        self.uploads: list[tuple[Path, str]] = []

    async def exec(self, command: str, *, cwd: str | None = None, timeout_s: float | None = None):
        self.commands.append(command)
        if command.endswith(runtime.LIBC_PROBE):
            return _Result(0, stdout=self.probe)
        if "echo '+ clean'" in command:  # the staging script
            return _Result(
                self.staging_exit,
                stderr="+ clean\n+ mkdir\n+ extract\nstep extract failed (exit 2)\n--- diagnostics\nPATH=/usr/bin",
            )
        if command.endswith(shlex.quote(runtime.HEALTH_CHECK)):  # pre-staging health check
            return _Result(1, stderr="ModuleNotFoundError: nooa")
        return _Result(0)

    async def upload(self, src, dst):
        self.uploads.append((Path(src), dst))


def test_cold_sandbox_uploads_extracts_and_health_checks(tmp_path):
    gnu, _ = _tarballs(tmp_path)
    sandbox = _ColdSandbox("x86_64", "gnu")
    python = asyncio.run(runtime.stage_runtime(sandbox, gnu))
    digest = runtime.file_sha256(gnu)[:16]
    assert sandbox.uploads == [(gnu, f"{runtime.REMOTE_ROOT}/{digest}.tar.gz")]
    script = sandbox.commands[-1]
    for step in ("clean", "mkdir", "extract", "discard-archive", "rename", "health-check"):
        assert f"echo '+ {step}'" in script, step
    assert script.index("+ extract") < script.index("+ rename") < script.index("+ health-check")
    assert "mv -T" not in script  # busybox
    assert python == f"{runtime.REMOTE_ROOT}/{digest}/python/bin/python3"


def test_staging_failure_reports_the_step_and_diagnostics(tmp_path):
    gnu, _ = _tarballs(tmp_path)
    sandbox = _ColdSandbox("x86_64", "gnu", staging_exit=2)
    with pytest.raises(RuntimeError, match=r"(?s)staging failed \(exit 2\).*step extract failed.*diagnostics") as info:
        asyncio.run(runtime.stage_runtime(sandbox, gnu))
    assert "PATH=/usr/bin" in str(info.value)


def test_probe_failure_is_reported_before_any_upload(tmp_path):
    gnu, _ = _tarballs(tmp_path)

    class _Broken(_ColdSandbox):
        async def exec(self, command, *, cwd=None, timeout_s=None):
            self.commands.append(command)
            return _Result(127, stderr="sh: uname: not found")

    sandbox = _Broken("x86_64", "gnu")
    with pytest.raises(RuntimeError, match="could not probe.*exit 127.*uname"):
        asyncio.run(runtime.stage_runtime(sandbox, gnu))
    assert sandbox.uploads == []


def test_build_targets_are_pinned():
    assert set(runtime.PYTHON_SHA256) >= {
        "x86_64-unknown-linux-gnu",
        "x86_64-unknown-linux-musl",
        "aarch64-unknown-linux-gnu",
    }
    assert runtime.python_url("x86_64-unknown-linux-musl").endswith(
        "x86_64-unknown-linux-musl-install_only_stripped.tar.gz"
    )
