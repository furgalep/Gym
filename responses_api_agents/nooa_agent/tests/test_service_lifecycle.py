# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-process lifecycle: retained services outlive the worker until session close, never on failure."""

import asyncio
import os
import shutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from responses_api_agents.nooa_agent.tests.test_sandbox_entrypoint import payload
from responses_api_agents.nooa_agent.tests.test_sandbox_runner import artifact, runner


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux subreaper contract")

DRIVER = """
import sys
from pathlib import Path
from responses_api_agents.nooa_agent import sandbox_supervisor
worker, mode = sys.argv[1], sys.argv[2]
real = sandbox_supervisor.process_supervisor._supervise
def fake_worker(command, **kwargs):
    return real([sys.executable, worker, command[-1], mode], **kwargs)
sandbox_supervisor.process_supervisor._supervise = fake_worker
directory = Path(sys.argv[3])
raise SystemExit(sandbox_supervisor.supervise(directory, retain_services="--retain-services" in sys.argv[4:]))
"""

WORKER = """
import pathlib, shutil, subprocess, sys
result, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
# A detached service as an agent would start it with nohup ... & setsid.
service = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(120)'],
    start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
(result.parent / 'service.pid').write_text(str(service.pid))
shutil.copyfile(result.parent / 'result.src', result)
raise SystemExit(0 if mode == 'ok' else 7)
"""


class LocalShellSandbox:
    """Run sandbox commands as real local bash processes."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def exec(self, command: str, **kwargs):
        self.calls.append((command, kwargs))
        process = await asyncio.create_subprocess_exec(
            "bash", "-c", command, cwd=kwargs.get("cwd") or "/", stdin=asyncio.subprocess.DEVNULL
        )
        return SimpleNamespace(return_code=await asyncio.wait_for(process.wait(), timeout=60))

    async def upload(self, source: Path, destination: str) -> None:
        shutil.copyfile(source, destination)

    async def download(self, source: str, destination: Path) -> None:
        shutil.copyfile(source, destination)

    async def disconnect(self) -> None:
        pass


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie would also answer kill(0); the subreaper must have reaped it.
    return Path(f"/proc/{pid}/stat").read_text().split(") ")[-1][0] != "Z"


def setup(tmp_path: Path, mode: str):
    r, _ = runner()
    sandbox = LocalShellSandbox()
    r.sandbox = sandbox
    r.workdir = str(tmp_path)
    r.directory = str(tmp_path / "session")
    r.retain_task_services = True
    r.poll_interval_s = 0.05
    (tmp_path / "driver.py").write_text(DRIVER)
    (tmp_path / "worker.py").write_text(WORKER)
    python = tmp_path / "python"
    # The runner calls "<python> -I -m <module> <dir> [--retain-services]"; drop "-I -m <module>".
    python.write_text(
        f'#!/bin/bash\nshift 3\nexec {sys.executable} {tmp_path / "driver.py"} {tmp_path / "worker.py"} {mode} "$@"\n'
    )
    python.chmod(0o755)
    r.python = str(python)
    return r, sandbox


@pytest.mark.asyncio
async def test_successful_worker_keeps_service_alive_until_close(tmp_path: Path) -> None:
    r, sandbox = setup(tmp_path, "ok")
    await r.prepare()
    (tmp_path / "session" / "result.src").write_text(artifact().model_dump_json())
    service = None
    try:
        result = await asyncio.wait_for(r.run(payload().request), timeout=30)
        assert result.resource_cookies == {"resource": "new"}
        service = int((tmp_path / "session" / "service.pid").read_text())
        await asyncio.sleep(0.5)
        assert alive(service), "service must survive the worker exit for verification"
        assert not (tmp_path / "session" / "cleanup.json").exists()
        await asyncio.wait_for(r.close(), timeout=30)
        assert r.stopped
        assert not alive(service), "session close must reap the retained service"
        assert not (tmp_path / "session").exists()
    finally:
        if service is not None and alive(service):
            os.kill(service, signal.SIGKILL)


@pytest.mark.asyncio
async def test_failed_worker_reaps_service_before_returning(tmp_path: Path) -> None:
    r, sandbox = setup(tmp_path, "fail")
    await r.prepare()
    (tmp_path / "session" / "result.src").write_text(artifact().model_dump_json())
    service = None
    try:
        with pytest.raises(RuntimeError, match="did not complete successfully"):
            await asyncio.wait_for(r.run(payload().request), timeout=30)
        service = int((tmp_path / "session" / "service.pid").read_text())
        assert r.stopped
        assert not alive(service)
        await r.close()
    finally:
        if service is not None and alive(service):
            os.kill(service, signal.SIGKILL)


@pytest.mark.asyncio
async def test_cancelled_run_is_reaped_by_close(tmp_path: Path) -> None:
    r, sandbox = setup(tmp_path, "ok")
    await r.prepare()
    # Never produce the result: the worker blocks, as a still-running agent would.
    (tmp_path / "worker.py").write_text(
        "import pathlib, subprocess, sys, time\n"
        "s = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'], start_new_session=True)\n"
        "pathlib.Path(sys.argv[1]).parent.joinpath('service.pid').write_text(str(s.pid))\n"
        "time.sleep(120)\n"
    )
    task = asyncio.create_task(r.run(payload().request))
    pid_file = tmp_path / "session" / "service.pid"
    for _ in range(200):
        if pid_file.exists():
            break
        await asyncio.sleep(0.05)
    service = int(pid_file.read_text())
    try:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert alive(service)
        await asyncio.wait_for(r.close(), timeout=30)
        assert not alive(service)
    finally:
        if alive(service):
            os.kill(service, signal.SIGKILL)
