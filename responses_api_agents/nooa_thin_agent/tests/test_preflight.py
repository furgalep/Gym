# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-flight: every request the NOOA harness puts on the wire must survive Gym's model servers.

A chat-backed policy server (``vllm_model`` without Responses passthrough) downconverts each
Responses request to Chat Completions and refuses fields it cannot represent. Catching that
here costs seconds; catching it in a cluster canary costs a 35-minute cycle (it did, twice:
``prompt_cache_options`` and ``truncation``).
"""

from __future__ import annotations

import asyncio
import json
import socket

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.responses_converter import ResponsesConverter


nooa_gym = pytest.importorskip("nooa_bench.gym", reason="NOOA harness not installed in this venv")


def _policy_app(bodies: list[dict]) -> FastAPI:
    """One tool turn, then a final message: the two request shapes a tool loop produces."""
    app = FastAPI()
    state = {"n": 0}

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        bodies.append(await request.json())
        state["n"] += 1
        if state["n"] == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "shell",
                    "status": "completed",
                    "arguments": json.dumps({"command": "echo hi"}),
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "DONE", "annotations": []}],
                }
            ]
        return JSONResponse(
            {
                "id": f"resp_{state['n']}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "error": None,
                "incomplete_details": None,
                "model": "gym-policy",
                "output": output,
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
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


async def _harness_requests() -> list[dict]:
    bodies: list[dict] = []
    server, task, port = await _serve(_policy_app(bodies))

    async def run_command(command: str) -> str:
        return "hi\n[exit 0]"

    try:
        episode = await nooa_gym.run_gym_episode(
            lambda llm: nooa_gym.ShellAgent(llm=llm, run_command=run_command),
            lambda agent: agent.solve("say hi"),
            model_url=f"http://127.0.0.1:{port}/ng-rollout/preflight/v1",
            max_policy_calls=5,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "completed", episode.error
    assert len(bodies) == 2
    return bodies


def test_harness_requests_downconvert_to_chat_completions():
    bodies = asyncio.run(_harness_requests())
    converter = ResponsesConverter(return_token_id_information=False)
    for body in bodies:
        params = NeMoGymResponseCreateParamsNonStreaming(**body)
        # Strict mode: what a vllm_model with drop_unrepresentable_request_fields=false does.
        chat = converter.responses_to_chat_completion_create_params(params)
        assert chat.messages, "downconversion produced no chat messages"
    # The second request replays the tool call and its output; both must become chat messages.
    second = converter.responses_to_chat_completion_create_params(
        NeMoGymResponseCreateParamsNonStreaming(**bodies[1])
    )
    roles = [m["role"] if isinstance(m, dict) else m.role for m in second.messages]
    assert "tool" in roles and "assistant" in roles
