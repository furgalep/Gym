# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker stopped at its time budget by Gym's real process supervisor is graded, not dropped.

The chain under test, with the real pieces at every step except the model server and the
sandbox transport: process_supervisor (SIGTERM at the budget, TERM grace, SIGKILL) ->
nooa_bench.gym_worker (checkpoints result.json) -> SandboxNOOARunner.run (collect validates
the result against the supervisor's receipt and grades it incomplete) ->
single_agent_turn's outcome policy (VERIFY). The negatives: a stop with no grace, a
cancellation nobody asked for, and SIGINT leave no gradable result.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import signal
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nooa_bench import gym_worker

from environment_servers.single_agent_turn.app import OutcomeAction, agent_outcome_policy
from nemo_gym.agent_utils import process_supervisor
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent import sandbox_runner as runner_module
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunRequest
from responses_api_agents.nooa_agent.sandbox_runner import SandboxNOOARunner
from responses_api_agents.nooa_agent.worker import Episode


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the sandbox supervisor is Linux-only")

if "snapshot" not in inspect.getsource(gym_worker):
    pytest.skip(
        "the installed nooa_bench.gym_worker predates the SIGTERM checkpoint (NOOA harness 64aaafc0+)",
        allow_module_level=True,
    )

TESTS_DIR = Path(__file__).parent
# Probes run with NOOA's ambient tracing off: an unreachable OTLP endpoint and a short probe.
QUIET_TRACING = {"OTLP_ENDPOINT": "http://127.0.0.1:9/v1/traces", "OTLP_PROBE_TIMEOUT": "0.2"}


def _stalling_app(log: list[dict], release: asyncio.Event) -> FastAPI:
    """First turn: a shell call; every later request stalls until ``release`` is set."""
    app = FastAPI()

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        log.append(await request.json())
        if len(log) > 1:
            await asyncio.wait_for(release.wait(), 120)
        return JSONResponse(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gym-policy",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "shell",
                        "status": "completed",
                        "arguments": json.dumps({"command": "echo hi > hi.txt"}),
                    }
                ],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )

    return app


async def _serve(app: FastAPI):
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, sock.getsockname()[1]


def _worker_env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(TESTS_DIR), env.get("PYTHONPATH")]))
    env.update(QUIET_TRACING)
    return env


def _write_input(tmp_path: Path, port: int, agent: str) -> tuple[Path, Path]:
    inputs, result = tmp_path / "input.json", tmp_path / "result.json"
    inputs.write_text(
        json.dumps(
            {
                "task": "write hi",
                "agent": agent,
                "model_url": f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
                "workdir": str(tmp_path),
            }
        )
    )
    return inputs, result


@dataclass
class _Supervised:
    receipt: dict
    code: int
    started_at: float
    finished_at: float
    result: Path


async def _supervised_worker(
    tmp_path: Path, port: int, *, budget: float, grace: float, agent: str = "nooa_bench.gym:ShellAgent"
) -> _Supervised:
    """Run the worker under the real supervisor CLI with a budget it cannot meet."""
    inputs, result = _write_input(tmp_path, port, agent)
    receipt = tmp_path / "receipt.json"
    started_at = time.time()
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        process_supervisor.__file__,
        "--timeout",
        str(budget),
        "--cleanup-timeout",
        str(grace),
        "--receipt",
        str(receipt),
        "--",
        sys.executable,
        "-m",
        "nooa_bench.gym_worker",
        str(inputs),
        str(result),
        env=_worker_env(),
    )
    async with asyncio.timeout(budget + 4 * grace + 60):
        code = await process.wait()
    return _Supervised(json.loads(receipt.read_text()), code, started_at, time.time(), result)


@dataclass
class _Session:
    """Stands in for SandboxSession over a finished supervised run: the real receipt and result file."""

    run: _Supervised
    session_dir: str = "/tmp/nemo-gym-nooa/x"
    workdir: str = "/app"

    @property
    def cleanup(self):
        return self.run.receipt

    async def read_output_log(self):
        return ""

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        return await collect()

    async def close(self, *, timeout):
        pass


async def _collect_through_the_runner(run: _Supervised, monkeypatch) -> tuple[SandboxNOOARunner, object]:
    """Drive SandboxNOOARunner.run over the supervised run's artifacts, as the agent server would."""

    async def upload_text(sandbox, *, path, text):
        pass

    async def read_text(sandbox, *, path):
        return run.result.read_text()  # FileNotFoundError when the worker left nothing, as in a sandbox

    monkeypatch.setattr(runner_module, "upload_text", upload_text)
    monkeypatch.setattr(runner_module, "read_text", read_text)
    runner = SandboxNOOARunner(
        sandbox=object(),
        workdir="/app",
        python=sys.executable,
        invocation=NOOAInvocationConfig(agent_class="nooa_bench.gym:ShellAgent"),
        model_base_url="http://h:1",
        max_policy_calls=None,
    )
    runner.session = _Session(run)
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [{"role": "user", "content": "write hi"}]})
    result = await runner.run(
        NOOARunRequest(responses_create_params=body, model_url_path="/ng-rollout/r1/v1/responses")
    )
    return runner, result.episode.response


async def test_supervisor_budget_stop_is_checkpointed_collected_and_verified(tmp_path, monkeypatch):
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        # The worker needs a few seconds to import NOOA and make its first call before the stop.
        run = await _supervised_worker(tmp_path, port, budget=15, grace=10)
    finally:
        release.set()
        server.should_exit = True
        await task
    # The supervisor stopped the worker at the budget and confirmed cleanup.
    assert run.code == 0 and run.receipt["timed_out"] is True and run.receipt["cleanup_confirmed"] is True
    assert len(log) == 2  # one finished turn, one stalled at the stop
    # The worker left the sandbox's work in place and a complete checkpoint behind.
    assert (tmp_path / "hi.txt").read_text() == "hi\n"
    episode = Episode.model_validate_json(run.result.read_text())
    assert episode.stop_reason == "timeout_exceeded" and episode.runtime["stop_signal"] == "SIGTERM"
    assert [r.status_code for r in episode.model_calls] == [200, None]
    assert [t.name for t in episode.tool_calls] == ["shell"]
    # collect() accepts it because the receipt vouches for the stop; the response is graded incomplete.
    runner, response = await _collect_through_the_runner(run, monkeypatch)
    assert runner.failure is None and runner.episode is not None
    assert response.status == "incomplete"
    assert response.usage is not None and response.usage.total_tokens == 2  # the finished call only
    assert response.metadata["harness_stop_signal"] == "SIGTERM"
    # ... and the environment's policy sends it to the verifier.
    assert agent_outcome_policy(response) is OutcomeAction.VERIFY
    assert runner.observations.gaps[0].code == "timeout_exceeded"


async def test_slow_agent_cleanup_cannot_defeat_the_checkpoint(tmp_path, monkeypatch):
    """The checkpoint is written in the SIGTERM handler, before the cancelled agent's teardown; a
    teardown longer than the TERM grace ends in SIGKILL with the result already on disk."""
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = await _supervised_worker(tmp_path, port, budget=15, grace=5, agent="probe_agents:SlowCleanupAgent")
    finally:
        release.set()
        server.should_exit = True
        await task
    assert len(log) == 2  # the probe ran a full turn before the stop
    assert run.receipt["timed_out"] is True and run.receipt["cleanup_confirmed"] is True
    assert run.receipt["return_code"] not in (0, None)  # killed during its 15 s teardown
    # The result landed at the budget, not after the teardown.
    written_after = run.result.stat().st_mtime - run.started_at
    assert 15 <= written_after < 15 + 3, written_after
    episode = Episode.model_validate_json(run.result.read_text())
    assert episode.stop_reason == "timeout_exceeded" and [r.status_code for r in episode.model_calls] == [200, None]
    runner, response = await _collect_through_the_runner(run, monkeypatch)
    assert response.status == "incomplete" and agent_outcome_policy(response) is OutcomeAction.VERIFY


async def test_a_stop_before_the_run_starts_leaves_no_result_and_fails(tmp_path, monkeypatch):
    """A worker stopped while it is still importing NOOA has no handler installed yet and no
    episode to render: nothing is left behind, and Gym keeps the infrastructure failure (504).
    (With the checkpoint written inside the SIGTERM handler, a stop during the run leaves a
    result within milliseconds, so "no result" needs a stop before the run starts.)"""
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = await _supervised_worker(tmp_path, port, budget=1, grace=10)
    finally:
        release.set()
        server.should_exit = True
        await task
    assert log == []  # the stop came before the first model call
    assert run.receipt["timed_out"] is True and run.receipt["cleanup_confirmed"] is True
    assert not run.result.exists()
    with pytest.raises(TimeoutError):  # the agent server reports this as a 504
        await _collect_through_the_runner(run, monkeypatch)


async def test_internal_cancellation_is_not_a_budget_stop(tmp_path, monkeypatch):
    """A cancellation the supervisor did not ask for leaves no result; the receipt says the worker
    exited on its own, so collect() reports a crash, not a timeout."""
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = await _supervised_worker(tmp_path, port, budget=60, grace=10, agent="probe_agents:SelfCancellingAgent")
    finally:
        release.set()
        server.should_exit = True
        await task
    assert len(log) == 1  # the probe made its first call, then cancelled itself from the tool
    assert run.receipt["timed_out"] is False and run.receipt["return_code"] not in (0, None)
    assert not run.result.exists()
    with pytest.raises(RuntimeError, match="exited without a result"):
        await _collect_through_the_runner(run, monkeypatch)


async def test_sigint_leaves_no_result(tmp_path):
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    inputs, result = _write_input(tmp_path, port, "nooa_bench.gym:ShellAgent")
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "nooa_bench.gym_worker", str(inputs), str(result), env=_worker_env()
    )
    try:
        async with asyncio.timeout(60):
            while len(log) < 2:
                await asyncio.sleep(0.05)
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(15):
            code = await process.wait()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        release.set()
        server.should_exit = True
        await task
    assert code != 0 and not result.exists()


@pytest.mark.parametrize(
    ("agent", "teardown"),
    [
        ("probe_agents:SwallowingAgent", "completed: None"),
        ("probe_agents:RaisingCleanupAgent", "error: RuntimeError: teardown failed"),
    ],
)
async def test_the_stop_classifies_the_episode_whatever_the_teardown_reports(tmp_path, monkeypatch, agent, teardown):
    """An agent that swallows the cancellation, or whose teardown raises, cannot turn a budget stop
    into a completed or errored episode: the final write keeps timeout_exceeded and records what the
    teardown reported as a diagnostic."""
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        run = await _supervised_worker(tmp_path, port, budget=15, grace=10, agent=agent)
    finally:
        release.set()
        server.should_exit = True
        await task
    assert len(log) == 2 and run.receipt["timed_out"] is True and run.code == 0
    episode = Episode.model_validate_json(run.result.read_text())
    assert episode.stop_reason == "timeout_exceeded" and episode.runtime["stop_signal"] == "SIGTERM"
    assert episode.runtime["teardown"].startswith(teardown)
    runner, response = await _collect_through_the_runner(run, monkeypatch)
    assert response.status == "incomplete"
    assert response.metadata["harness_teardown"].startswith(teardown)
    assert agent_outcome_policy(response) is OutcomeAction.VERIFY
