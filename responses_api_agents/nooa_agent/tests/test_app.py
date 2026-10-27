# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response and observation projection of the thin NOOA agent, without servers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest, AgentSessionSetupError
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent import app as app_module
from responses_api_agents.nooa_agent.app import (
    Episode,
    ModelCall,
    NOOAAgent,
    NOOAAgentConfig,
    NOOASession,
    ToolCall,
    _task_text,
)


TRAIN = {"prompt_token_ids": [1], "generation_token_ids": [2], "generation_log_probs": [-0.1]}
MESSAGE = {
    "type": "message",
    "id": "m",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "DONE", "annotations": []}],
    **TRAIN,
}
CALL = {
    "type": "function_call",
    "id": "f",
    "call_id": "c1",
    "name": "shell",
    "arguments": "{}",
    "status": "completed",
    **TRAIN,
}
USAGE = {
    "input_tokens": 10,
    "output_tokens": 2,
    "total_tokens": 12,
    "input_tokens_details": {"cached_tokens": 4},
    "output_tokens_details": {"reasoning_tokens": 1},
}


def _agent() -> NOOAAgent:
    config = NOOAAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="nooa_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return NOOAAgent.model_construct(config=config)


def _episode(stop_reason="completed", calls=2, error=None) -> Episode:
    records = [
        ModelCall(
            invocation_id="inv",
            request={"input": []},
            response={"id": f"r{i}", "output": [CALL if i < calls - 1 else MESSAGE], "usage": USAGE},
            status_code=200,
            started_at=1.0,
            completed_at=2.0,
        )
        for i in range(calls)
    ]
    return Episode(
        output_items=[CALL, {"type": "function_call_output", "call_id": "c1", "output": "ok"}, MESSAGE]
        if calls
        else [],
        conversation=[{"role": "user", "content": "task"}, CALL, MESSAGE],
        model_calls=records,
        tool_calls=[
            ToolCall(
                invocation_id="inv",
                tool_call_id="c1",
                name="shell",
                arguments={"command": "ls"},
                output="ok",
                started_at=1.0,
                completed_at=1.5,
                status="completed",
            )
        ],
        return_value="DONE" if calls else "ok",
        stop_reason=stop_reason,
        error=error,
        terminal_response_id=f"r{calls - 1}" if calls else None,
    )


def _body() -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])


def test_response_passes_training_fields_and_sums_usage():
    response = _agent()._response(_episode(), _body(), "r-0")
    assert response.status == "completed" and response.error is None
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert response.output[0].generation_token_ids == [2]
    assert response.usage.input_tokens == 20 and response.usage.input_tokens_details.cached_tokens == 8


def test_policy_limits_are_incomplete_not_failed():
    response = _agent()._response(_episode(stop_reason="policy_budget_exceeded"), _body(), "r-0")
    assert response.status == "incomplete" and response.error is None


def test_errors_fail_with_zero_usage_when_nothing_finished():
    response = _agent()._response(_episode(stop_reason="error", calls=0, error="boom"), _body(), "r-0")
    assert response.status == "failed" and response.error.message == "boom"
    assert response.usage.total_tokens == 0
    # No synthetic message: the output holds only what the policy produced (nothing here); the
    # return value travels in metadata so the attribution witnesses stay intact.
    assert response.output == []
    assert response.metadata["harness_return_value"] == '"ok"'


def test_response_declares_the_terminal_response_for_attribution():
    response = _agent()._response(_episode(), _body(), "r-0")
    assert response.metadata["terminal_response_id"] == "r1"
    assert response.output[-1].type == "message" and response.output[-1].id == "m"


def test_evidence_has_one_invocation_per_id_and_no_model_call_refs():
    state = NOOASession(
        request=AgentSeedSessionRequest.model_validate(
            {
                "agent_session_id": "s",
                "episode_id": {"rollout_id": "r-0"},
                "task_id": {"task_id": "t", "taskset": "probe"},
            }
        )
    )
    state.episode = _episode()
    bundle = _agent()._evidence(state)
    invocations = [r for r in bundle.records if r.kind == "agent_invocation"]
    tools = [r for r in bundle.records if r.kind == "tool_call"]
    assert [i.invocation_id for i in invocations] == ["inv"]
    assert invocations[0].model_calls == []  # Gym joins the capture by x-session-id
    assert invocations[0].conversation[0].role == "user"
    assert [(t.tool_name, t.status) for t in tools] == [("shell", "completed")]
    assert bundle.gaps == []


def test_evidence_without_an_episode_records_a_gap():
    state = NOOASession(
        request=AgentSeedSessionRequest.model_validate(
            {
                "agent_session_id": "s",
                "episode_id": {"rollout_id": "r-0"},
                "task_id": {"task_id": "t", "taskset": "probe"},
            }
        )
    )
    state.failure = "timeout_exceeded"
    bundle = _agent()._evidence(state)
    assert [g.code for g in bundle.gaps] == ["observation_capture_failed"]


def test_response_leaves_usage_unknown_when_the_provider_omitted_it():
    episode = _episode()
    del episode.model_calls[0].response["usage"]
    assert _agent()._response(episode, _body(), "r-0").usage is None


# -- task text ------------------------------------------------------------------------------------


def _params(**kwargs) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(**kwargs)


def test_task_text_keeps_instructions_and_preamble_in_order():
    body = _params(
        instructions="Be brief.",
        input=[
            {"role": "system", "content": "You fix bugs."},
            {"role": "developer", "content": [{"type": "input_text", "text": "Use pytest."}]},
            {"role": "user", "content": "Fix the failing test."},
            {"role": "user", "content": [{"type": "input_text", "text": "Then stop."}]},
        ],
    )
    assert _task_text(body) == "Be brief.\n\nYou fix bugs.\n\nUse pytest.\n\nFix the failing test.\n\nThen stop."
    assert _task_text(_params(input="plain")) == "plain"


@pytest.mark.parametrize(
    "input",
    [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": [{"type": "output_text", "text": "b"}]}],
        [{"type": "function_call_output", "call_id": "c", "output": "x"}],
        [{"role": "system", "content": "only a preamble"}],
        [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}],
    ],
)
def test_task_text_refuses_what_nooa_cannot_replay(input):
    with pytest.raises(HTTPException) as info:
        _task_text(_params(input=input))
    assert info.value.status_code == 422


# -- seed cleanup ---------------------------------------------------------------------------------


@dataclass
class _Provider:
    name: str = "opensandbox"
    closed: bool = False

    async def aclose(self):
        self.closed = True


@dataclass
class _Sandbox:
    disconnected: bool = False
    stopped: bool = False
    fail_disconnect: bool = False
    commands: list[str] = field(default_factory=list)

    async def exec(self, command, *, cwd=None, timeout_s=None):
        self.commands.append(command)
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": "", "error_type": None})()

    async def disconnect(self):
        if self.fail_disconnect:
            raise ConnectionError("provider gone")
        self.disconnected = True

    async def stop(self):
        self.stopped = True


def _seed_request(borrowed: bool = True) -> AgentSeedSessionRequest:
    data = {"agent_session_id": "s", "episode_id": {"rollout_id": "r-0"}, "task_id": {"task_id": "t", "taskset": "p"}}
    if borrowed:
        data["sandbox_access"] = {
            "connection": {"provider_config_ref": "opensandbox", "descriptor": {"id": "sb-1"}},
            "workdir": "/app",
        }
    return AgentSeedSessionRequest.model_validate(data)


@pytest.fixture
def seeding(monkeypatch, tmp_path):
    """A sandboxed-mode agent whose provider, sandbox and model URL are fakes."""
    tarball = tmp_path / "rt.tar.gz"
    tarball.write_bytes(b"rt")
    config = NOOAAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="nooa_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        runtime_tarball=str(tarball),
        sandbox_provider="opensandbox",
    )
    agent = NOOAAgent.model_construct(config=config)
    provider, sandbox = _Provider(), _Sandbox()
    urls = {"model": "http://10.1.2.3:8000/v1"}

    class _FakeAsyncSandbox:
        def __init__(self, provider):
            self.provider = provider

        async def start(self, spec):
            if urls.get("start_fails"):
                raise RuntimeError("no capacity")
            return sandbox

        @classmethod
        async def connect(cls, descriptor, *, provider):
            return sandbox

    monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {provider.name: {}})
    monkeypatch.setattr(app_module, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app_module, "create_provider", lambda cfg: provider)
    monkeypatch.setattr(app_module, "AsyncSandbox", _FakeAsyncSandbox)
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: urls["model"])
    monkeypatch.setattr(NOOAAgent, "effective_tool_accesses", lambda self, body: [])
    return agent, provider, sandbox, urls, monkeypatch


def test_seed_releases_a_borrowed_sandbox_when_staging_fails(seeding):
    agent, provider, sandbox, _, monkeypatch = seeding

    async def failing_stage(*args, **kwargs):
        raise RuntimeError("sandbox image uses musl libc")

    monkeypatch.setattr(app_module, "stage_runtime", failing_stage)
    with pytest.raises(RuntimeError, match="musl"):
        asyncio.run(agent._seed_agent_session_state(_seed_request()))
    assert sandbox.disconnected and not sandbox.stopped


def test_seed_hands_unreleasable_state_to_the_base_server(seeding):
    agent, provider, sandbox, _, monkeypatch = seeding
    sandbox.fail_disconnect = True

    async def failing_stage(*args, **kwargs):
        raise RuntimeError("tar: not found")

    monkeypatch.setattr(app_module, "stage_runtime", failing_stage)
    with pytest.raises(AgentSessionSetupError) as info:
        asyncio.run(agent._seed_agent_session_state(_seed_request()))
    assert isinstance(info.value.error, RuntimeError) and isinstance(info.value.state, NOOASession)


def test_seed_closes_the_provider_when_an_owned_sandbox_fails_to_start(seeding):
    agent, provider, sandbox, urls, _ = seeding
    urls["start_fails"] = True
    with pytest.raises(RuntimeError, match="no capacity"):
        asyncio.run(agent._seed_agent_session_state(_seed_request(borrowed=False)))
    assert provider.closed


def test_seed_refuses_a_loopback_model_url_before_touching_the_sandbox(seeding):
    agent, provider, sandbox, urls, _ = seeding
    urls["model"] = "http://127.0.0.1:8000/v1"
    with pytest.raises(HTTPException) as info:
        asyncio.run(agent._seed_agent_session_state(_seed_request()))
    assert info.value.status_code == 422 and "loopback" in info.value.detail
    assert sandbox.commands == [] and not sandbox.disconnected


def test_seed_stages_the_runtime_and_opens_a_session(seeding):
    agent, provider, sandbox, _, monkeypatch = seeding

    async def staged(*args, **kwargs):
        return "/tmp/nemo-gym-nooa-runtime/abcd/python/bin/python3"

    monkeypatch.setattr(app_module, "stage_runtime", staged)
    state = asyncio.run(agent._seed_agent_session_state(_seed_request()))
    assert state.runtime_python == "/tmp/nemo-gym-nooa-runtime/abcd/python/bin/python3"
    assert state.session is not None and state.session.session_dir.startswith("/tmp/nemo-gym-nooa/")


# -- activation outcome and close -----------------------------------------------------------------


@dataclass
class _Session:
    """Stands in for SandboxSession: runs stage then collect and exposes the supervisor's receipt.

    It propagates collect()'s exception directly. The real session records it as capture_error and
    re-raises it from execute() (the supervisor exits 0 on a timed-out worker), so these tests cover
    how collect() classifies a missing result, not SandboxSession's propagation.
    """

    sandbox: object
    session_dir: str = "/tmp/nemo-gym-nooa/x"
    timed_out: bool = False
    cleanup_confirmed: bool = True
    artifacts: object = None
    closed: bool = False
    log: str = "Traceback: ModuleNotFoundError: nooa_atom"

    @property
    def cleanup(self):
        return {"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": self.timed_out}

    async def read_output_log(self):
        return self.log

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        await stage_activation()
        return await collect()

    async def close(self, *, timeout):
        self.closed = True


def _sandboxed_state(session: _Session) -> NOOASession:
    state = NOOASession(
        request=_seed_request(), sandbox=session.sandbox, workdir="/app", runtime_python="/rt/python/bin/python3"
    )
    state.session = session
    return state


@pytest.fixture
def worker_without_result(monkeypatch):
    async def upload_text(sandbox, *, path, text):
        pass

    async def read_text(sandbox, *, path):
        raise FileNotFoundError(path)

    monkeypatch.setattr(app_module, "upload_text", upload_text)
    monkeypatch.setattr(app_module, "read_text", read_text)
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: "http://h:1/v1")


def test_a_timed_out_worker_is_a_504_with_the_log_tail(worker_without_result):
    state = _sandboxed_state(_Session(_Sandbox(), timed_out=True))
    with pytest.raises(HTTPException) as info:
        asyncio.run(_agent()._execute(state, "task", _body(), "/ng-rollout/r-0/v1/responses"))
    assert info.value.status_code == 504 and "ModuleNotFoundError" in info.value.detail
    assert state.failure == "timeout_exceeded"


def test_a_timed_out_worker_that_checkpointed_is_graded_as_incomplete(monkeypatch):
    """The supervisor's TERM grace lets the worker write its partial episode; Gym then verifies
    the sandbox instead of failing the rollout (Terminal-Bench grades after an agent timeout)."""
    checkpoint = _episode(stop_reason="timeout_exceeded", calls=1, error="the agent was stopped before it finished")

    async def upload_text(sandbox, *, path, text):
        pass

    async def read_text(sandbox, *, path):
        return checkpoint.model_dump_json()

    monkeypatch.setattr(app_module, "upload_text", upload_text)
    monkeypatch.setattr(app_module, "read_text", read_text)
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: "http://h:1/v1")
    state = _sandboxed_state(_Session(_Sandbox(), timed_out=True))
    response = asyncio.run(_agent()._execute(state, "task", _body(), "/ng-rollout/r-0/v1/responses"))
    assert response.status == "incomplete" and response.error is None
    assert state.failure is None
    evidence = _agent()._evidence(state)
    assert {r.status for r in evidence.records if hasattr(r, "stop_reason")} == {"incomplete"}
    assert evidence.gaps[0].code == "timeout_exceeded"


def test_a_crashed_worker_surfaces_its_log_and_is_recorded_as_the_failure(worker_without_result):
    state = _sandboxed_state(_Session(_Sandbox(), timed_out=False))
    with pytest.raises(RuntimeError, match="without a result.*ModuleNotFoundError"):
        asyncio.run(_agent()._execute(state, "task", _body(), "/ng-rollout/r-0/v1/responses"))
    assert state.failure.startswith("RuntimeError: NOOA worker exited without a result")
    assert _agent()._evidence(state).gaps[0].detail == state.failure


def test_close_reports_the_supervisors_confirmation():
    session = _Session(_Sandbox(), cleanup_confirmed=False)
    state = _sandboxed_state(session)
    response = asyncio.run(_agent()._close_agent_session_state(state))
    assert response.cleanup_confirmed is False and session.closed and state.closing


def test_close_without_a_supervisor_reports_no_confirmation_and_releases_the_sandbox():
    sandbox = _Sandbox()
    state = NOOASession(request=_seed_request(), sandbox=sandbox, workdir="/app")  # seed failed before the session
    response = asyncio.run(_agent()._close_agent_session_state(state))
    assert response.cleanup_confirmed is None and sandbox.disconnected


def test_responses_fences_closing_sessions_and_foreign_retries(monkeypatch):
    agent = _agent()
    state = _sandboxed_state(_Session(_Sandbox()))
    monkeypatch.setattr(NOOAAgent, "_agent_session_id_from_request", lambda self, request: "s")
    monkeypatch.setattr(NOOAAgent, "_require_agent_session", lambda self, session_id: state)
    monkeypatch.setattr(NOOAAgent, "url_path_for_request", lambda self, path, request: "/ng-rollout/r-0/v1/responses")
    request = type("Req", (), {"path_params": {"rollout_id": "r-0"}})()

    async def never(*args, **kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(NOOAAgent, "_execute", never)

    async def scenario():
        first = asyncio.create_task(agent.responses(request, _body()))
        await asyncio.sleep(0)
        with pytest.raises(HTTPException) as info:
            await agent.responses(request, _params(input="a different task"))
        assert info.value.status_code == 409 and "another activation" in info.value.detail
        async with state.lock:
            state.closing = True
        with pytest.raises(HTTPException) as info:
            await agent.responses(request, _body())
        assert info.value.status_code == 409 and "closing" in info.value.detail
        first.cancel()
        await asyncio.gather(first, return_exceptions=True)

    asyncio.run(scenario())


def test_media_inputs_config_is_validated():
    base = {
        "host": "127.0.0.1",
        "port": 1,
        "entrypoint": "app.py",
        "name": "nooa_agent",
        "model_server": {"type": "responses_api_models", "name": "policy_model"},
    }
    assert app_module.NOOAAgentConfig.model_validate(base).media_inputs == ()
    assert app_module.NOOAAgentConfig.model_validate({**base, "media_inputs": ["image"]}).media_inputs == ("image",)
    with pytest.raises(Exception, match="gif"):
        app_module.NOOAAgentConfig.model_validate({**base, "media_inputs": ["gif"]})


def test_a_timeout_result_the_supervisor_does_not_vouch_for_is_a_failure(monkeypatch):
    """A `timeout_exceeded` result without a timed-out receipt is a stop nobody asked for (or a forged
    file), not a gradable budget stop."""
    forged = _episode(stop_reason="timeout_exceeded", calls=1, error="the agent was stopped before it finished")

    async def upload_text(sandbox, *, path, text):
        pass

    async def read_text(sandbox, *, path):
        return forged.model_dump_json()

    monkeypatch.setattr(app_module, "upload_text", upload_text)
    monkeypatch.setattr(app_module, "read_text", read_text)
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: "http://h:1/v1")
    state = _sandboxed_state(_Session(_Sandbox(), timed_out=False))
    with pytest.raises(RuntimeError, match="supervisor did not stop it at the budget"):
        asyncio.run(_agent()._execute(state, "task", _body(), "/ng-rollout/r-0/v1/responses"))
    assert state.failure.startswith("RuntimeError: NOOA worker reported a budget stop")
    assert _agent()._evidence(state).gaps[0].code == "observation_capture_failed"
