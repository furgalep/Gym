# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A worker stopped at its time budget by Gym's real process supervisor is graded, not dropped.

The chain under test, with the real pieces at every step except the model server and the
sandbox transport: process_supervisor (SIGTERM at the budget, TERM grace, SIGKILL) ->
nooa_bench.gym_worker (checkpoints result.json) -> NOOAThinAgent.collect/_response
(status incomplete) -> single_agent_turn's outcome policy (VERIFY).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import socket
import sys
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from nooa_bench import gym_worker
from nooa_bench.gym import GymEpisode

from environment_servers.single_agent_turn.app import OutcomeAction, agent_outcome_policy
from nemo_gym.agent_utils import process_supervisor
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_thin_agent.app import NOOAThinAgent, NOOAThinAgentConfig


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="the sandbox supervisor is Linux-only")

if "SIGTERM" not in inspect.getsource(gym_worker):
    pytest.skip(
        "the installed nooa_bench.gym_worker predates the SIGTERM checkpoint (NOOA harness a4331732+)",
        allow_module_level=True,
    )


def _stalling_app(log: list[dict], release: asyncio.Event) -> FastAPI:
    """First turn: a shell call; every later request stalls until ``release`` is set."""
    app = FastAPI()

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        log.append(await request.json())
        if len(log) > 1:
            await asyncio.wait_for(release.wait(), 60)
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


def _agent() -> NOOAThinAgent:
    config = NOOAThinAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="nooa_thin_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        execution_mode="sandboxed",
    )
    return NOOAThinAgent.model_construct(config=config)


async def _supervised_worker(tmp_path: Path, port: int, *, budget: float, grace: float) -> tuple[dict, int]:
    """Run the worker under the real supervisor CLI with a budget it cannot meet."""
    inputs, result, receipt = tmp_path / "input.json", tmp_path / "result.json", tmp_path / "receipt.json"
    inputs.write_text(
        json.dumps(
            {
                "task": "write hi",
                "agent": "nooa_bench.gym:ShellAgent",
                "model_url": f"http://127.0.0.1:{port}/ng-rollout/r1/v1",
                "workdir": str(tmp_path),
            }
        )
    )
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
        env=dict(os.environ),
    )
    async with asyncio.timeout(budget + 4 * grace + 30):
        code = await process.wait()
    return json.loads(receipt.read_text()), code


async def test_supervisor_budget_stop_is_checkpointed_and_verified(tmp_path):
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        # The worker needs a few seconds to import NOOA and make its first call before the stop.
        receipt, code = await _supervised_worker(tmp_path, port, budget=15, grace=10)
    finally:
        release.set()
        server.should_exit = True
        await task
    # The supervisor stopped the worker at the budget and confirmed cleanup.
    assert code == 0 and receipt["timed_out"] is True and receipt["cleanup_confirmed"] is True
    assert len(log) == 2  # one finished turn, one stalled at the stop
    # The worker left the sandbox's work in place and a complete checkpoint behind.
    assert (tmp_path / "hi.txt").read_text() == "hi\n"
    episode = GymEpisode.from_json((tmp_path / "result.json").read_text())
    assert episode.stop_reason == "timeout_exceeded"
    assert [r.status_code for r in episode.model_calls] == [200, None]
    assert [t.name for t in episode.tool_calls] == ["shell"]
    # The thin agent grades it as incomplete, and the environment's policy sends it to the verifier.
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate({"input": [{"role": "user", "content": "write hi"}]})
    response = _agent()._response(episode, body, "r1")
    assert response.status == "incomplete" and response.error is None
    assert response.usage is not None and response.usage.total_tokens == 2  # the finished call only
    assert agent_outcome_policy(response) is OutcomeAction.VERIFY


async def test_supervisor_kill_without_checkpoint_stays_a_failure(tmp_path):
    """No TERM grace: the worker is killed before it can checkpoint and nothing is salvaged."""
    log: list[dict] = []
    release = asyncio.Event()
    server, task, port = await _serve(_stalling_app(log, release))
    try:
        receipt, code = await _supervised_worker(tmp_path, port, budget=4, grace=0.001)
    finally:
        release.set()
        server.should_exit = True
        await task
    assert receipt["timed_out"] is True
    assert not (tmp_path / "result.json").exists()  # collect() then raises TimeoutError -> 504, as tested in test_app
