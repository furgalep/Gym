# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandbox runner over the worker boundary: payload, result collection, budget stops, evidence."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from environment_servers.single_agent_turn.app import OutcomeAction, agent_outcome_policy
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.nooa_agent import app as app_module
from responses_api_agents.nooa_agent import sandbox_runner as runner_module
from responses_api_agents.nooa_agent.app import NOOAAgent, NOOASessionState
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import NOOARunRequest
from responses_api_agents.nooa_agent.sandbox_runner import SandboxNOOARunner
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.tests.test_config import agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_worker import episode


@dataclass
class _Sandbox:
    commands: list[str] = field(default_factory=list)
    disconnected: bool = False

    async def exec(self, command, *, cwd=None, timeout_s=None):
        self.commands.append(command)
        return type("R", (), {"return_code": 0, "stdout": "", "stderr": "", "error_type": None})()

    async def disconnect(self):
        self.disconnected = True


@dataclass
class _Session:
    """Stands in for SandboxSession: runs stage then collect and exposes the supervisor's receipt.

    It propagates collect()'s exception directly. The real session records it as capture_error and
    re-raises it from execute(), so these tests cover how collect() classifies a missing result, not
    SandboxSession's propagation.
    """

    session_dir: str = "/tmp/nemo-gym-nooa/x"
    workdir: str = "/app"
    timed_out: bool = False
    artifacts: object = None
    closed: bool = False
    log: str = "Traceback: ModuleNotFoundError: nooa_atom"
    staged: list = field(default_factory=list)
    budgets: list = field(default_factory=list)

    @property
    def cleanup(self):
        return {"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": self.timed_out}

    async def read_output_log(self):
        return self.log

    async def execute(self, *, stage_activation, collect, timeout, close_timeout):
        self.budgets.append(timeout)
        self.staged.append(await stage_activation())
        return await collect()

    async def close(self, *, timeout):
        self.closed = True


def _runner(session: _Session, **kwargs) -> SandboxNOOARunner:
    runner = SandboxNOOARunner(
        sandbox=_Sandbox(),
        workdir="/app",
        python="/rt/python/bin/python3",
        invocation=NOOAInvocationConfig(agent_class="nooa_bench.gym:ShellAgent", turn_method="solve"),
        model_base_url="http://h:1",
        max_policy_calls=7,
        **kwargs,
    )
    runner.session = session
    return runner


def _request(**overrides) -> NOOARunRequest:
    return NOOARunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}]),
        model_url_path="/ng-rollout/r-0/v1/responses",
        rollout_id="r-0",
        **overrides,
    )


@pytest.fixture
def worker_result(monkeypatch):
    """What the worker left in result.json: a function of the test, read by collect()."""
    store: dict = {"text": None, "uploaded": []}

    async def upload_text(sandbox, *, path, text):
        store["uploaded"].append(json.loads(text))

    async def read_text(sandbox, *, path):
        if store["text"] is None:
            raise FileNotFoundError(path)
        return store["text"]

    monkeypatch.setattr(runner_module, "upload_text", upload_text)
    monkeypatch.setattr(runner_module, "read_text", read_text)
    return store


def test_stage_writes_the_worker_payload_and_command(worker_result):
    worker_result["text"] = episode().model_dump_json()
    session = _Session()
    runner = _runner(session)
    result = asyncio.run(runner.run(_request()))
    payload = worker_result["uploaded"][0]
    assert payload["agent"] == "nooa_bench.gym:ShellAgent" and payload["turn_method"] == "solve"
    assert payload["task"] == "task" and payload["model_url"] == "http://h:1/ng-rollout/r-0/v1"
    assert payload["max_policy_calls"] == 7 and payload["workdir"] == "/app"
    command = session.staged[0]
    assert command.argv == [
        "/rt/python/bin/python3",
        "-I",
        "-m",
        "nooa_bench.gym_worker",
        "/tmp/nemo-gym-nooa/x/input.json",
        "/tmp/nemo-gym-nooa/x/result.json",
    ]
    assert result.termination_reason is None and result.episode.response.status == "completed"
    assert result.episode.response.metadata["terminal_response_id"] == "r1"
    assert result.return_value == "DONE" and runner.episode is not None


def test_a_timed_out_worker_without_a_result_is_a_timeout_with_the_log_tail(worker_result):
    runner = _runner(_Session(timed_out=True))
    with pytest.raises(TimeoutError, match="ModuleNotFoundError"):
        asyncio.run(runner.run(_request()))
    assert runner.failure == "timeout_exceeded"
    assert runner.observations.gaps[0].code == "observation_capture_failed"


def test_a_timed_out_worker_that_checkpointed_is_graded_as_incomplete(worker_result):
    """The supervisor's TERM grace lets the worker write its partial episode; Gym then verifies
    the sandbox instead of failing the rollout."""
    worker_result["text"] = episode(stop_reason="timeout_exceeded", calls=1, error="stopped").model_dump_json()
    runner = _runner(_Session(timed_out=True))
    result = asyncio.run(runner.run(_request()))
    assert result.termination_reason is NOOATerminationReason.TIMEOUT_EXCEEDED
    assert result.episode.response.status == "incomplete"
    assert agent_outcome_policy(result.episode.response) is OutcomeAction.VERIFY
    assert runner.failure is None
    assert {r.status for r in runner.observations.records if r.kind == "agent_invocation"} == {"incomplete"}


def test_a_crashed_worker_surfaces_its_log_and_is_recorded_as_the_failure(worker_result):
    runner = _runner(_Session(timed_out=False))
    with pytest.raises(RuntimeError, match="without a result.*ModuleNotFoundError"):
        asyncio.run(runner.run(_request()))
    assert runner.failure.startswith("RuntimeError: NOOA worker exited without a result")
    assert runner.observations.gaps[0].detail == runner.failure


def test_a_timeout_result_the_supervisor_does_not_vouch_for_is_a_failure(worker_result):
    """A `timeout_exceeded` result without a timed-out receipt is a stop nobody asked for (or a forged
    file), not a gradable budget stop."""
    worker_result["text"] = episode(stop_reason="timeout_exceeded", calls=1, error="stopped").model_dump_json()
    runner = _runner(_Session(timed_out=False))
    with pytest.raises(RuntimeError, match="supervisor did not stop it at the budget"):
        asyncio.run(runner.run(_request()))
    assert runner.failure.startswith("RuntimeError: NOOA worker reported a budget stop")


def test_policy_stops_keep_their_reason_and_resource_cookies_flow_back(worker_result):
    e = episode(stop_reason="policy_budget_exceeded", error="budget")
    e.resource_cookies = {"sid": "9"}
    worker_result["text"] = e.model_dump_json()
    request = _request(resource_cookies={"old": "1"})
    result = asyncio.run(_runner(_Session()).run(request))
    assert result.termination_reason is NOOATerminationReason.POLICY_BUDGET_EXCEEDED
    assert result.episode.response.status == "incomplete"
    assert request.resource_cookies == {"old": "1", "sid": "9"} == result.resource_cookies


def test_close_keeps_a_checkpoint_the_session_collected_while_stopping():
    session = _Session(artifacts=episode(stop_reason="timeout_exceeded", calls=1, error="stopped"))
    runner = _runner(session)
    asyncio.run(runner.close())
    assert session.closed and runner.episode is session.artifacts


# -- the agent server over the runner ----------------------------------------------------------------


def _seed_request() -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest.model_validate(
        {
            "agent_session_id": "s",
            "episode_id": {"rollout_id": "r-0"},
            "task_id": {"task_id": "t", "taskset": "p"},
            "sandbox_access": {
                "connection": {"provider_config_ref": "probe", "descriptor": {"id": "sb-1"}},
                "workdir": "/app",
            },
        }
    )


def _sandboxed_agent(**config_overrides) -> NOOAAgent:
    config = agent_config(nooa=invocation_config(execution_mode="sandboxed"), **config_overrides)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return NOOAAgent(config=config, server_client=client)


def test_seed_uses_runtime_python_without_preparing_a_runtime(monkeypatch):
    agent = _sandboxed_agent(runtime_python="/rt/python/bin/python3")
    sandbox = _Sandbox()

    async def connect(descriptor, *, provider):
        return sandbox

    async def never_prepare(*args, **kwargs):
        raise AssertionError("runtime preparation must be skipped")

    monkeypatch.setattr(app_module.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {"probe": {}})
    monkeypatch.setattr(app_module, "create_provider", lambda cfg: object())
    monkeypatch.setattr(app_module, "stage_runtime", never_prepare)
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: "http://h:1/v1")
    monkeypatch.setattr(NOOAAgent, "effective_tool_accesses", lambda self, body: [])
    state = asyncio.run(agent._seed_agent_session_state(_seed_request()))
    assert isinstance(state.runner, SandboxNOOARunner) and state.runner.python == "/rt/python/bin/python3"
    assert any(command.startswith("mkdir -p") for command in sandbox.commands)


def _state_with(runner: SandboxNOOARunner) -> NOOASessionState:
    state = NOOASessionState(request=_seed_request())
    state.runner = runner
    return state


def test_responses_refuses_what_the_worker_cannot_replay_before_any_activation(monkeypatch):
    agent = _sandboxed_agent()
    state = _state_with(_runner(_Session()))
    monkeypatch.setattr(NOOAAgent, "_agent_session_id_from_request", lambda self, request: "s")
    monkeypatch.setattr(NOOAAgent, "_require_agent_session", lambda self, session_id: state)
    request = type("Req", (), {"path_params": {"rollout_id": "r-0"}})()
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": [{"type": "output_text", "text": "b"}]},
        ]
    )
    with pytest.raises(HTTPException) as info:
        asyncio.run(agent.responses(request, body))
    assert info.value.status_code == 422 and state.execution is None


def test_a_budget_stop_without_a_checkpoint_is_a_504(worker_result):
    agent = _sandboxed_agent()
    state = _state_with(_runner(_Session(timed_out=True)))
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])
    with pytest.raises(HTTPException) as info:
        asyncio.run(agent._execute_session(state, body, "/ng-rollout/r-0/v1/responses"))
    assert info.value.status_code == 504 and "ModuleNotFoundError" in info.value.detail


def test_the_worker_response_is_reported_as_is_without_a_synthetic_message(worker_result):
    worker_result["text"] = episode(stop_reason="error", calls=0, error="boom").model_dump_json()
    agent = _sandboxed_agent()
    state = _state_with(_runner(_Session()))
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])
    response = asyncio.run(agent._execute_session(state, body, "/ng-rollout/r-0/v1/responses"))
    assert response.status == "failed" and response.output == []
    assert response.metadata["harness_return_value"] == '"ok"'


def test_close_reports_worker_observations_and_a_late_checkpoint(worker_result):
    agent = _sandboxed_agent()
    session = _Session(artifacts=episode(stop_reason="timeout_exceeded", calls=1, error="stopped"))
    state = _state_with(_runner(session))
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}])
    state.activation_key = (body, "/ng-rollout/r-0/v1/responses")
    response = asyncio.run(agent._close_agent_session_state(state))
    assert session.closed and state.result is not None
    assert state.result.termination_reason is NOOATerminationReason.TIMEOUT_EXCEEDED
    assert response.agent_observations.gaps[0].code == "timeout_exceeded"
    assert {r.status for r in response.agent_observations.records if r.kind == "agent_invocation"} == {"incomplete"}


def test_seed_stages_the_configured_archives_when_no_interpreter_is_given(monkeypatch, tmp_path):
    gnu, musl = tmp_path / "gnu.tar.gz", tmp_path / "musl.tar.gz"
    agent = _sandboxed_agent(runtime_tarball=str(gnu), runtime_tarball_musl=str(musl))
    sandbox = _Sandbox()
    staged: list = []

    async def connect(descriptor, *, provider):
        return sandbox

    async def stage(sb, tarball, *, musl_tarball=None):
        staged.append((sb, tarball, musl_tarball))
        return "/tmp/nemo-gym-nooa-runtime/abcd/python/bin/python3"

    monkeypatch.setattr(app_module.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {"probe": {}})
    monkeypatch.setattr(app_module, "create_provider", lambda cfg: object())
    monkeypatch.setattr(app_module, "stage_runtime", stage)
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: "http://h:1/v1")
    monkeypatch.setattr(NOOAAgent, "effective_tool_accesses", lambda self, body: [])
    state = asyncio.run(agent._seed_agent_session_state(_seed_request()))
    assert staged == [(sandbox, str(gnu), str(musl))]
    assert state.runner.python == "/tmp/nemo-gym-nooa-runtime/abcd/python/bin/python3"


def test_seed_without_an_archive_or_interpreter_is_a_422_before_the_sandbox_is_touched(monkeypatch):
    agent = _sandboxed_agent()
    sandbox = _Sandbox()

    async def connect(descriptor, *, provider):
        return sandbox

    monkeypatch.setattr(app_module.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda ref, cfg: {"probe": {}})
    monkeypatch.setattr(app_module, "create_provider", lambda cfg: object())
    monkeypatch.setattr(NOOAAgent, "resolve_model_base_url", lambda self, name, rollout_id=None: "http://h:1/v1")
    monkeypatch.setattr(NOOAAgent, "effective_tool_accesses", lambda self, body: [])
    monkeypatch.delenv("NOOA_GYM_RUNTIME_TARBALL", raising=False)
    agent.config.runtime_tarball = None
    with pytest.raises(Exception) as info:
        asyncio.run(agent._seed_agent_session_state(_seed_request()))
    detail = getattr(getattr(info.value, "error", info.value), "detail", str(info.value))
    assert "runtime_tarball" in str(detail)


# -- traces and trajectories -------------------------------------------------------------------------


class _ArchivingSandbox(_Sandbox):
    """Records the archive command and the download the runner asks for."""

    def __init__(self, *, tar_exit: int = 0):
        super().__init__()
        self.tar_exit = tar_exit
        self.downloads: list[tuple[str, str]] = []

    async def exec(self, command, *, cwd=None, timeout_s=None):
        self.commands.append(command)
        code = self.tar_exit if "tarfile" in command else 0
        return type("R", (), {"return_code": code, "stdout": "", "stderr": "disk full", "error_type": None})()

    async def download(self, remote, local):
        self.downloads.append((remote, str(local)))
        Path(local).write_bytes(b"tgz")


def _tracing_runner(session: _Session, tmp_path, *, tar_exit: int = 0, trace: dict | None = None) -> SandboxNOOARunner:
    runner = _runner(
        session,
        trace={"atif": True, "trace_jsonl": True, "otlp_endpoint": "http://c:4318/v1/traces"}
        if trace is None
        else trace,
        artifacts_output=tmp_path / "out",
    )
    runner.sandbox = _ArchivingSandbox(tar_exit=tar_exit)
    return runner


@pytest.mark.parametrize(
    ("metadata", "budget"),
    [
        (None, 12000.0),  # no row budget: the configured ceiling
        ({"agent_timeout_sec": "900"}, 900.0),  # canonical task.toml budget, carried as a string
        ({"agent_timeout_sec": "40000"}, 12000.0),  # a row budget above the ceiling is capped
    ],
)
def test_the_row_agent_budget_bounds_the_supervised_run(worker_result, metadata, budget):
    worker_result["text"] = episode().model_dump_json()
    session = _Session()
    runner = _runner(session, timeout=12000)
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "task"}], metadata=metadata)
    request = NOOARunRequest(
        responses_create_params=body, model_url_path="/ng-rollout/r-0/v1/responses", rollout_id="r-0"
    )
    asyncio.run(runner.run(request))
    assert session.budgets == [budget]


def test_tracing_flags_reach_the_worker_and_artifacts_are_archived_per_rollout(worker_result, tmp_path):
    worker_result["text"] = episode().model_dump_json()
    runner = _tracing_runner(_Session(), tmp_path)
    result = asyncio.run(runner.run(_request()))
    payload = worker_result["uploaded"][0]
    assert payload["atif"] is True and payload["trace_jsonl"] is True
    assert payload["otlp_endpoint"] == "http://c:4318/v1/traces"
    assert payload["artifacts_dir"] == "/tmp/nemo-gym-nooa/x/artifacts"
    sandbox = runner.sandbox
    # The archive is built by the staged interpreter from the session directory (see the script tests).
    assert sandbox.commands[-1] == runner_module.archive_command(
        "/rt/python/bin/python3", "/tmp/nemo-gym-nooa/x", "/tmp/nemo-gym-nooa/x/artifacts.tar.gz"
    )
    assert sandbox.downloads == [("/tmp/nemo-gym-nooa/x/artifacts.tar.gz", str(tmp_path / "out" / "r-0.tar.gz"))]
    assert runner.artifacts == tmp_path / "out" / "r-0.tar.gz" and runner.artifacts.read_bytes() == b"tgz"
    assert result.episode.response.metadata["harness_artifacts_archive"] == str(tmp_path / "out" / "r-0.tar.gz")
    assert runner.observations.gaps == []


def _archive(tmp_path, files: dict[str, str]) -> list[str]:
    """Run the archive command on a real session directory holding ``files``; return the member names."""
    import subprocess
    import sys
    import tarfile

    root = tmp_path / "session"
    root.mkdir(parents=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    out = tmp_path / "artifacts.tar.gz"
    subprocess.run(runner_module.archive_command(sys.executable, str(root), str(out)), shell=True, check=True)
    with tarfile.open(out) as tar:
        return sorted(tar.getnames())


def test_the_archive_takes_whatever_evidence_exists(tmp_path):
    complete = {
        "artifacts/atif/root.json": "{}",
        "result.json": "{}",
        "result.json.atom-policy.json": "{}",
        "input.json": "{}",  # the payload is not evidence
    }
    assert _archive(tmp_path / "a", complete) == [
        "artifacts",
        "artifacts/atif",
        "artifacts/atif/root.json",
        "result.json",
        "result.json.atom-policy.json",
    ]
    # A cancelled worker leaves exports and a sidecar but no Episode: still archived, no error.
    without_result = {"artifacts/traces/t.jsonl": "", "result.json.atom-policy.json": "{}"}
    assert _archive(tmp_path / "b", without_result) == [
        "artifacts",
        "artifacts/traces",
        "artifacts/traces/t.jsonl",
        "result.json.atom-policy.json",
    ]
    # Names with spaces are plain file names, not shell words.
    assert _archive(tmp_path / "c", {"result.json.extra sidecar": "{}", "result.json": "{}"}) == [
        "result.json",
        "result.json.extra sidecar",
    ]
    assert _archive(tmp_path / "d", {}) == []  # nothing left behind: an empty archive, not a failure


def test_artifacts_are_collected_when_the_worker_leaves_no_result(worker_result, tmp_path):
    """A stopped or crashed worker's exports are evidence for the failed row; collection runs before close."""
    worker_result["text"] = "not json"
    runner = _tracing_runner(_Session(), tmp_path)
    with pytest.raises(RuntimeError, match="exited without a result"):
        asyncio.run(runner.run(_request()))
    assert runner.failure.startswith("RuntimeError")
    assert runner.artifacts == tmp_path / "out" / "r-0.tar.gz" and runner.artifacts.read_bytes() == b"tgz"
    assert runner.sandbox.downloads == [
        ("/tmp/nemo-gym-nooa/x/artifacts.tar.gz", str(tmp_path / "out" / "r-0.tar.gz"))
    ]


def test_a_failed_artifact_collection_is_a_gap_not_a_failed_episode(worker_result, tmp_path):
    worker_result["text"] = episode().model_dump_json()
    runner = _tracing_runner(_Session(), tmp_path, tar_exit=1)
    result = asyncio.run(runner.run(_request()))
    assert result.termination_reason is None and result.episode.response.status == "completed"
    assert runner.artifacts is None and "disk full" in runner.artifacts_error
    assert [g.code for g in runner.observations.gaps] == ["artifacts_capture_failed"]
    assert "harness_artifacts_archive" not in result.episode.response.metadata


def test_live_export_alone_archives_nothing(worker_result, tmp_path):
    worker_result["text"] = episode().model_dump_json()
    runner = _tracing_runner(_Session(), tmp_path, trace={"otlp_endpoint": "http://c:4318/v1/traces"})
    asyncio.run(runner.run(_request()))
    assert not runner.collects_artifacts and runner.sandbox.downloads == []
    assert "artifacts_dir" in worker_result["uploaded"][0]  # harmless; the worker only uses it with atif/jsonl


def test_tracing_config_defaults_save_everything_and_follow_gyms_collector(monkeypatch):
    from responses_api_agents.nooa_agent.config import NOOATracingConfig

    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracing = NOOATracingConfig()
    assert tracing.atif and tracing.trace_jsonl and tracing.otlp_endpoint is None
    assert tracing.artifacts_output == Path("nooa-artifacts") and tracing.collects_artifacts
    assert tracing.worker_payload == {"atif": True, "trace_jsonl": True}
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318")
    assert NOOATracingConfig().otlp_endpoint == "http://collector:4318/v1/traces"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/v1/traces/")
    assert NOOATracingConfig().otlp_endpoint == "http://collector:4318/v1/traces"
    off = NOOATracingConfig(atif=False, trace_jsonl=False, artifacts_output=None, otlp_endpoint=None)
    assert off.worker_payload == {} and not off.collects_artifacts
    assert agent_config().tracing.atif is True  # the server config carries the defaults


def test_a_concurrent_close_archives_before_the_session_is_released(worker_result, tmp_path):
    """Close while the activation still waits on provider I/O: the archive is taken inside the shared
    finalization, before the session directory is removed, and the run's failure is unchanged."""
    worker_result["text"] = "not json"
    order: list[str] = []

    class _ClosingSession(_Session):
        def __init__(self):
            super().__init__()
            self.collect = None
            self.released = asyncio.Event()

        async def execute(self, *, stage_activation, collect, timeout, close_timeout):
            self.collect = collect
            self.staged.append(await stage_activation())
            await self.released.wait()  # the controller task returns only after finalization
            raise RuntimeError("activation stopped by close")

        async def close(self, *, timeout):
            # SandboxSession._finalize: capture once after the harness stopped, then release.
            try:
                self.artifacts = await self.collect()
            except Exception:
                pass
            order.append("released")
            self.closed = True
            self.released.set()

    class _OrderedSandbox(_ArchivingSandbox):
        async def exec(self, command, *, cwd=None, timeout_s=None):
            if "tarfile" in command:
                order.append("archive")
            return await super().exec(command, cwd=cwd, timeout_s=timeout_s)

    session = _ClosingSession()
    runner = _tracing_runner(session, tmp_path)
    runner.sandbox = _OrderedSandbox()

    async def scenario():
        run = asyncio.create_task(runner.run(_request()))
        while not session.staged:
            await asyncio.sleep(0)
        await runner.close()
        with pytest.raises(RuntimeError, match="stopped by close"):
            await run

    asyncio.run(scenario())
    assert order == ["archive", "released"]
    assert runner.artifacts == tmp_path / "out" / "r-0.tar.gz" and runner.artifacts.read_bytes() == b"tgz"
    assert runner.failure.startswith("RuntimeError")
    assert "artifacts_capture_failed" not in [g.code for g in runner.observations.gaps]
