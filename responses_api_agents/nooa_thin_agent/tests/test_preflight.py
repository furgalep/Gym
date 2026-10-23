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


def _call(name: str, **arguments) -> dict:
    return {"type": "function_call", "name": name, "arguments": json.dumps(arguments)}


def _message(text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


SHELL_SCRIPT = [[_call("shell", command="echo hi")], [_message("DONE")]]

# Atom: a reasoning item plus a python_cell, then the cell that ends the task. The second request
# therefore replays reasoning, function calls and their outputs: the richest shape Atom produces.
ATOM_SCRIPT = [
    [
        {"type": "reasoning", "summary": [{"type": "summary_text", "text": "Write the file, then verify."}]},
        _call("python_cell", code="from pathlib import Path\nPath('hello.txt').write_text('hello gym')\nprint('ok')"),
    ],
    [
        _call(
            "python_cell",
            code="return_result(Done(explanation='wrote hello.txt', result=TaskResult("
            "solution_description='Created hello.txt', evidence='print ok', how_to_verify='cat hello.txt', "
            "report='hello.txt written')))",
        )
    ],
]


def _policy_app(bodies: list[dict], script: list[list[dict]]) -> FastAPI:
    """Answer request N with script[N], recording every request body."""
    app = FastAPI()
    state = {"n": 0}

    @app.post("/ng-rollout/{rollout_id}/v1/responses")
    async def responses(rollout_id: str, request: Request):
        bodies.append(await request.json())
        state["n"] += 1
        if state["n"] > len(script):
            return JSONResponse(
                {"error": {"message": "script exhausted", "type": "test", "code": "409"}}, status_code=409
            )
        output = []
        for index, item in enumerate(script[state["n"] - 1]):
            item = {"id": f"item_{state['n']}_{index}", "status": "completed", **item}
            if item["type"] == "function_call":
                item.setdefault("call_id", f"call_{state['n']}_{index}")
            output.append(item)
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


async def _harness_requests(agent_factory, invoke, script) -> tuple[list[dict], object]:
    bodies: list[dict] = []
    server, task, port = await _serve(_policy_app(bodies, script))
    try:
        episode = await nooa_gym.run_gym_episode(
            agent_factory,
            invoke,
            model_url=f"http://127.0.0.1:{port}/ng-rollout/preflight/v1",
            max_policy_calls=len(script) + 1,
        )
    finally:
        server.should_exit = True
        await task
    assert episode.stop_reason == "completed", episode.error
    assert len(bodies) == len(script)
    return bodies, episode


def _downconvert_all(bodies: list[dict]) -> list:
    """Strict mode: what a vllm_model with drop_unrepresentable_request_fields=false does."""
    converter = ResponsesConverter(return_token_id_information=False)
    converted = []
    for body in bodies:
        chat = converter.responses_to_chat_completion_create_params(NeMoGymResponseCreateParamsNonStreaming(**body))
        assert chat.messages, "downconversion produced no chat messages"
        converted.append(chat)
    return converted


def _roles(chat) -> list[str]:
    return [m["role"] if isinstance(m, dict) else m.role for m in chat.messages]


def test_shell_loop_requests_downconvert_to_chat_completions():
    async def run_command(command: str) -> str:
        return "hi\n[exit 0]"

    bodies, _ = asyncio.run(
        _harness_requests(
            lambda llm: nooa_gym.ShellAgent(llm=llm, run_command=run_command),
            lambda agent: agent.solve("say hi"),
            SHELL_SCRIPT,
        )
    )
    first, second = _downconvert_all(bodies)
    # The second request replays the tool call and its output; both must become chat messages.
    assert "tool" in _roles(second) and "assistant" in _roles(second)


def test_atom_requests_downconvert_to_chat_completions(tmp_path, monkeypatch):
    atom_gym = pytest.importorskip("nooa_atom.gym", reason="nooa-atom not installed in this venv")
    workdir = tmp_path / "task"
    workdir.mkdir()
    monkeypatch.chdir(workdir)  # the worker runs with cwd = task dir; Atom's cells resolve paths against it
    bodies, episode = asyncio.run(
        _harness_requests(
            lambda llm: atom_gym.AtomHarness(
                llm, turn_method="handle_batch", workdir=str(workdir), sessions_dir=str(tmp_path / "sessions")
            ),
            lambda agent: agent.solve("Create hello.txt containing 'hello gym', then finish."),
            ATOM_SCRIPT,
        )
    )
    converted = _downconvert_all(bodies)
    # Atom's second request replays two python_cell rounds. The policy's reasoning item is not
    # replayed (run_gym_episode default), matching the thick nooa_agent; the strict downconversion
    # below is what a chat-backed vllm_model applies to the shape that does go out.
    types = [item.get("type") or item.get("role") for item in bodies[1]["input"]]
    assert "reasoning" not in types and types.count("function_call") >= 2, types
    assert "tool" in _roles(converted[1]) and "assistant" in _roles(converted[1])
    assert (workdir / "hello.txt").read_text() == "hello gym"
    # Every executed cell is a tool record; the finishing one is also recorded as return_result.
    names = {t.name for t in episode.tool_calls}
    assert names == {"python_cell", "return_result"}, names
