# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOOA agent server for NeMo Gym.

Any NOOA agent runs inside the task sandbox as ``python -I -m nooa_bench.gym_worker``, on
a prebuilt NOOA runtime this server stages at seed time, under Gym's shared
``SandboxSession`` supervisor. The model seam is NOOA's own Responses client pointed at the
rollout-prefixed Gym model server, so Gym's model server holds the authoritative capture of
every model call and a lost worker loses no training evidence.

This server never imports NOOA. The boundary is the worker's JSON payload and result
(:class:`Episode`); the only vocabulary the two sides share is the Responses API. The
agent is reached by configuration: ``agent`` names a ``module:Class`` and ``turn_method``
the coroutine awaited with the task text, so a new NOOA agent needs no code here.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from typing import Any, ClassVar, Literal
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from pydantic import BaseModel, Field

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agent_utils.task_timeout import resolve_agent_timeout
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionSetupError,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
    assert_model_url_reachable_from_sandbox,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ObservationGap,
    ToolCallObservation,
)
from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, create_provider
from nemo_gym.sandbox.access import DirectSandboxConnection
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.utils import read_text, upload_text
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent.runtime import stage_runtime


class NOOAAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    # Any NOOA agent class, as ``module:Class``; constructed with ``llm=`` and ``agent_kwargs``.
    agent: str = "nooa_bench.gym:ShellAgent"
    # The coroutine method awaited with the task text.
    turn_method: str = "solve"
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)
    # ``module:callable(agent, tools)`` replacing the default translation of Gym tools into the
    # NOOA interface (a Skill of callables on the agent plus function tools for tool-call loops).
    tool_binding: str | None = None
    # NOOA's own exports, written in the sandbox and collected after the episode (see artifacts_output).
    artifacts: tuple[Literal["atif", "trace_jsonl"], ...] = ()
    # Local directory receiving one ``<rollout_id>.tar.gz`` of the artifacts per episode.
    artifacts_output: str | None = None
    # Policy context window, handed to NOOA for its context budgeting; None leaves NOOA's default.
    context_window: int | None = Field(default=None, gt=0)
    # Replay prior reasoning items (summary text) to the policy. Off by default: Gym's chat-backed
    # servers render them as <think> text, which changes the model's view of its own history.
    replay_reasoning: bool = False
    # Media the policy accepts, as NOOA's media_inputs. Empty is text-only: NOOA replaces captured
    # images with a short note. Images reach the policy in a user turn, which Gym's chat-backed
    # servers downconvert to image_url parts; tool outputs stay text.
    media_inputs: tuple[Literal["image", "audio", "video", "pdf"], ...] = ()
    # Prebuilt relocatable NOOA runtime (see runtime.py), staged into the sandbox at seed time.
    runtime_tarball: str | None = Field(default_factory=lambda: os.environ.get("NOOA_GYM_RUNTIME_TARBALL") or None)
    # Alpine-style task images need a musl build; staging probes the image's libc and picks it.
    runtime_tarball_musl: str | None = Field(
        default_factory=lambda: os.environ.get("NOOA_GYM_RUNTIME_TARBALL_MUSL") or None
    )
    # Interpreter inside the sandbox that already has NOOA installed; used when no tarball is staged.
    # Only meaningful with a local sandbox provider (tests, conformance).
    runtime_python: str | None = None
    timeout: float = Field(default=900, gt=0)
    close_timeout: float = Field(default=30, gt=0)
    max_policy_calls: int | None = Field(default=None, gt=0)
    shell_timeout_seconds: int = Field(default=180, gt=0)
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)


# -- the worker's result: the contract with nooa_bench.gym_worker -----------------------------------


class ModelCall(BaseModel):
    """One wire request/response pair the worker sent, verbatim."""

    invocation_id: str
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    status_code: int | None = None
    started_at: float = 0.0
    completed_at: float | None = None


class ToolCall(BaseModel):
    invocation_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    output: Any = None
    started_at: float
    completed_at: float
    status: str  # completed | failed | incomplete


class Episode(BaseModel):
    """What the worker leaves in ``result.json``."""

    output_items: list[dict[str, Any]]
    conversation: list[dict[str, Any]]
    model_calls: list[ModelCall]
    tool_calls: list[ToolCall]
    return_value: Any = None
    stop_reason: str
    error: str | None = None
    # Served ``response.id`` of the root agent's last completed call: the main chain's terminal.
    terminal_response_id: str | None = None
    resource_cookies: dict[str, str] = Field(default_factory=dict)
    # Where the agent ran (interpreter, version, host, artifact paths), filled in by the worker.
    runtime: dict[str, str] = Field(default_factory=dict)


@dataclass
class NOOASession(AgentSessionState):
    sandbox: AsyncSandbox | None = None
    session: SandboxSession[Episode] | None = None
    owns_sandbox: bool = False
    workdir: str = "/app"
    tool_access: DirectHTTPToolAccess | None = None
    runtime_python: str | None = None
    model_cookies: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    closing: bool = False
    # The request that owns the single activation; a retry must carry the same one.
    activation_key: tuple[NeMoGymResponseCreateParamsNonStreaming, str] | None = None
    execution: asyncio.Task[NeMoGymResponse] | None = None
    episode: Episode | None = None
    failure: str | None = None
    artifacts: str | None = None  # local archive of NOOA's exports, once collected


def _text(content: Any, role: str) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        part = part if isinstance(part, dict) else part.model_dump()
        if part.get("type") != "input_text":
            raise HTTPException(422, f"NOOA accepts text content only; {role} message carries {part.get('type')!r}")
        parts.append(part.get("text", ""))
    return "\n".join(parts)


def _task_text(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """One task string from the request: instructions, system/developer preamble, then the user turn(s).

    The turn method takes a single prompt, so nothing is dropped silently: the preamble is prepended
    in order and any item NOOA cannot represent (assistant history, tool items) is a 422.
    """
    items = [{"type": "message", "role": "user", "content": body.input}] if isinstance(body.input, str) else body.input
    sections: list[str] = [body.instructions] if body.instructions else []
    users: list[str] = []
    for index, item in enumerate(items):
        data = item if isinstance(item, dict) else item.model_dump()
        kind, role = data.get("type", "message"), data.get("role")
        if kind != "message":
            raise HTTPException(422, f"NOOA expects message items, but input[{index}] is {kind!r}")
        if role in {"system", "developer"}:
            if users:
                raise HTTPException(422, f"NOOA expects {role} messages before the first user message")
            sections.append(_text(data.get("content"), role))
        elif role == "user":
            users.append(_text(data.get("content"), role))
        else:
            raise HTTPException(422, f"NOOA cannot replay {role!r} messages; input[{index}] is unsupported")
    if not users:
        raise HTTPException(422, "NOOA task input must contain a user message")
    return "\n\n".join(section for section in [*sections, *users] if section)


# Stops that leave a gradable sandbox: the policy ran out of budget, or the supervisor stopped
# the worker at the agent's time budget and it checkpointed its episode. Gym grades these as
# incomplete; every other non-completed stop is an error.
_INCOMPLETE_STOP_REASONS = frozenset({"policy_budget_exceeded", "max_output_tokens_exceeded", "timeout_exceeded"})


def _zero_usage() -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
    )


class NOOAAgent(SimpleResponsesAPIAgent):
    config: NOOAAgentConfig
    # The worker inside the task container calls the model server itself.
    calls_model_from_sandbox: ClassVar[bool] = True

    async def run(self, body: object) -> None:
        raise NotImplementedError("Use the single_agent_turn Environment Server")

    # -- sessions ---------------------------------------------------------------------------------

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> NOOASession:
        owns = body.sandbox_access is None
        if owns:
            if not self.config.sandbox_provider:
                raise HTTPException(422, "NOOA requires sandbox_access or a configured sandbox_provider")
            spec = SandboxSpec(**{"workdir": "/app", **self.config.sandbox_config})
            workdir, provider_ref = spec.workdir, self.config.sandbox_provider
        else:
            if not isinstance(body.sandbox_access.connection, DirectSandboxConnection):
                raise HTTPException(422, "NOOA requires direct SandboxAccess")
            workdir, provider_ref = body.sandbox_access.workdir, body.sandbox_access.connection.provider_config_ref
        # Everything that can be refused without a sandbox is refused first: nothing to clean up.
        accesses = self.effective_tool_accesses(body)
        direct = [a for a in accesses if isinstance(a, DirectHTTPToolAccess)]
        if any(a.required for a in accesses if not isinstance(a, DirectHTTPToolAccess)) or len(direct) > 1:
            raise HTTPException(422, "NOOA supports at most one direct HTTP tool grant")
        if not (self.config.runtime_tarball or self.config.runtime_python):
            raise HTTPException(422, "NOOA requires runtime_tarball (or runtime_python for a local sandbox)")
        provider_config = resolve_provider_config(provider_ref, get_global_config_dict())
        # The worker in the container dials the model server: a loopback URL would make every
        # call fail inside a remote sandbox and the episode score 0 without a visible error.
        try:
            assert_model_url_reachable_from_sandbox(
                self.resolve_model_base_url(self.config.model_server.name),
                provider_name=next(iter(provider_config), None),
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        provider = create_provider(provider_config)
        try:
            if owns:
                sandbox = AsyncSandbox(provider)
                await sandbox.start(spec)
            else:
                sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
        except BaseException:
            await provider.aclose()
            raise
        state = NOOASession(
            request=body,
            sandbox=sandbox,
            owns_sandbox=owns,
            workdir=workdir,
            tool_access=direct[0] if direct else None,
        )
        try:
            await self._prepare(state)
        except BaseException as error:
            # The container is up; release it. If that fails too, hand the state to the base
            # server so its close path retries the release instead of leaking the sandbox.
            try:
                await self._release(state)
            except BaseException:
                raise AgentSessionSetupError(state, error=error) from error
            raise
        return state

    async def _prepare(self, state: NOOASession) -> None:
        """Stage the runtime and the session directory in a sandbox that is already running."""
        sandbox = state.sandbox
        assert sandbox is not None
        if state.owns_sandbox:
            await sandbox.exec(f"mkdir -p -- {shlex.quote(state.workdir)}", cwd="/", timeout_s=30)
        if self.config.runtime_tarball:
            state.runtime_python = await stage_runtime(
                sandbox, self.config.runtime_tarball, musl_tarball=self.config.runtime_tarball_musl
            )
        state.session = SandboxSession(
            sandbox=sandbox,
            session_dir=f"/tmp/nemo-gym-nooa/{uuid4().hex}",
            workdir=state.workdir,
            harness="NOOA",
            owns_sandbox=state.owns_sandbox,
        )
        await sandbox.exec(f"mkdir -p -- {shlex.quote(state.session.session_dir)}", cwd="/", timeout_s=30)

    async def _release(self, state: NOOASession) -> None:
        """Stop the harness if one ran, then stop an owned sandbox or disconnect a borrowed one."""
        if state.execution is not None and not state.execution.done():
            state.execution.cancel()
            await asyncio.gather(state.execution, return_exceptions=True)
        if state.session is not None:
            await state.session.close(timeout=self.config.close_timeout)
            if state.episode is None and isinstance(state.session.artifacts, Episode):
                state.episode = state.session.artifacts
        elif state.sandbox is not None:
            await (state.sandbox.stop() if state.owns_sandbox else state.sandbox.disconnect())

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, NOOASession)
        async with state.lock:
            state.closing = True
        await self._release(state)
        cookies = None
        if state.tool_access is not None:
            cookies = (
                dict(state.episode.resource_cookies) if state.episode is not None else dict(state.tool_access.cookies)
            )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=self._evidence(state),
            resources_cookies=cookies,
            # A session that never got a supervisor (seed failed early) reports nothing rather than a guess.
            cleanup_confirmed=state.session.cleanup_confirmed if state.session is not None else None,
        )

    # -- activation -------------------------------------------------------------------------------

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "NOOA requires an agent session")
        state = self._require_agent_session(session_id)
        assert isinstance(state, NOOASession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Agent session does not match the rollout route")
        if body.tools and state.tool_access is None:
            raise HTTPException(422, "Resource tools require a direct HTTP grant")
        task = _task_text(body)  # a request NOOA cannot run is a terminal 422, never a cached activation
        route = self.url_path_for_request("/v1/responses", request)
        async with state.lock:
            if state.closing:
                raise HTTPException(409, "Agent session is closing")
            key = (body.model_copy(deep=True), route)
            if state.activation_key is not None and state.activation_key != key:
                raise HTTPException(409, "Session already has another activation")
            if state.execution is None:
                state.activation_key = key
                state.execution = asyncio.create_task(self._execute(state, task, body, route))
            execution = state.execution
        return await asyncio.shield(execution)

    def _model_url(self, route: str) -> str:
        base = self.resolve_model_base_url(self.config.model_server.name).removesuffix("/v1")
        return base + route.removesuffix("/responses")

    async def _execute(
        self, state: NOOASession, task: str, body: NeMoGymResponseCreateParamsNonStreaming, route: str
    ) -> NeMoGymResponse:
        budget = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        try:
            state.episode = await self._run_sandboxed(state, task, self._model_url(route), budget, body)
        except TimeoutError as error:
            state.failure = "timeout_exceeded"
            raise HTTPException(504, f"NOOA agent exceeded its time budget: {error}") from error
        except Exception as error:
            state.failure = f"{type(error).__name__}: {error}"
            raise
        if self.config.artifacts and self.config.artifacts_output:
            await self._collect_artifacts(state)
        return self._response(state.episode, body, state.request.episode_id.capture_key)

    def _payload(self, state: NOOASession, task: str, model_url: str, body: NeMoGymResponseCreateParamsNonStreaming):
        session = state.session
        assert session is not None
        return {
            "task": task,
            "agent": self.config.agent,
            "turn_method": self.config.turn_method,
            "agent_kwargs": self.config.agent_kwargs,
            "tool_binding": self.config.tool_binding,
            "artifacts_dir": f"{session.session_dir}/artifacts" if self.config.artifacts else None,
            **{name: True for name in self.config.artifacts},
            "context_window": self.config.context_window,
            "replay_reasoning": self.config.replay_reasoning,
            "media_inputs": list(self.config.media_inputs),
            "model_url": model_url,
            "max_policy_calls": self.config.max_policy_calls,
            "shell_timeout_seconds": self.config.shell_timeout_seconds,
            "workdir": state.workdir,
            "cookies": state.model_cookies,
            "tools": [t.model_dump(mode="json", exclude_none=True) for t in body.tools or []],
            "tool_access": {
                "base_url": str(state.tool_access.base_url),
                "headers": dict(state.tool_access.headers),
                "cookies": dict(state.tool_access.cookies),
            }
            if state.tool_access
            else None,
        }

    async def _run_sandboxed(
        self,
        state: NOOASession,
        task: str,
        model_url: str,
        budget: float,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> Episode:
        session = state.session
        assert session is not None and state.sandbox is not None
        python = state.runtime_python or self.config.runtime_python
        if python is None:
            raise RuntimeError("no NOOA interpreter in the sandbox: configure runtime_tarball or runtime_python")
        inputs, result = f"{session.session_dir}/input.json", f"{session.session_dir}/result.json"

        async def stage() -> SandboxCommand:
            await upload_text(state.sandbox, path=inputs, text=json.dumps(self._payload(state, task, model_url, body)))
            return SandboxCommand(argv=[python, "-I", "-m", "nooa_bench.gym_worker", inputs, result], python=python)

        async def collect() -> Episode:
            # A worker stopped at its budget checkpoints the episode it has (stop reason
            # `timeout_exceeded`) during the supervisor's TERM grace; one that left nothing behind
            # is an infrastructure failure, and the supervisor's receipt says which.
            try:
                episode = Episode.model_validate_json(await read_text(state.sandbox, path=result))
            except Exception as error:
                log = (await session.read_output_log())[-4000:]
                if session.cleanup["timed_out"]:
                    raise TimeoutError(f"NOOA worker was stopped at its {budget:.0f}s budget; tail: {log}") from error
                raise RuntimeError(f"NOOA worker exited without a result; output log tail: {log}") from error
            # The supervisor's receipt is the authority on budget stops: a `timeout_exceeded` result it
            # does not vouch for (a stop the worker was not asked for, or a forged file) is not gradable.
            if episode.stop_reason == "timeout_exceeded" and not session.cleanup["timed_out"]:
                raise RuntimeError(
                    "NOOA worker reported a budget stop, but the supervisor did not stop it at the budget"
                )
            return episode

        return await session.execute(
            stage_activation=stage, collect=collect, timeout=budget, close_timeout=self.config.close_timeout
        )

    async def _collect_artifacts(self, state: NOOASession) -> None:
        """Archive NOOA's exports (ATIF, spans) out of the sandbox, one tarball per rollout."""
        session = state.session
        assert session is not None and state.sandbox is not None and self.config.artifacts_output
        remote = f"{session.session_dir}/artifacts.tar.gz"
        result = await state.sandbox.exec(
            f"tar -czf {shlex.quote(remote)} -C {shlex.quote(session.session_dir)} artifacts", cwd="/", timeout_s=120
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"could not archive NOOA artifacts (exit {result.return_code}): {result.stderr or ''}")
        output = Path(self.config.artifacts_output)
        output.mkdir(parents=True, exist_ok=True)
        local = output / f"{state.request.episode_id.capture_key}.tar.gz"
        await state.sandbox.download(remote, local)
        state.artifacts = str(local)

    # -- projection -------------------------------------------------------------------------------

    def _response(
        self, episode: Episode, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: str
    ) -> NeMoGymResponse:
        completed = [c for c in episode.model_calls if c.response and (c.status_code or 500) < 400]
        usages = [c.response["usage"] for c in completed if c.response.get("usage")]
        if not completed:
            usage = _zero_usage()  # a rollout with no finished response has a known zero total
        elif len(usages) == len(completed):
            usage = NeMoGymResponseUsage.sum_from_list([NeMoGymResponseUsage.model_validate(u) for u in usages])
        else:
            usage = None  # the provider omitted usage: counts are unknown, never invented
        status, error = "completed", None
        if episode.stop_reason in _INCOMPLETE_STOP_REASONS:
            status = "incomplete"
        elif episode.stop_reason != "completed":
            status, error = "failed", {"code": "server_error", "message": episode.error or episode.stop_reason}
        metadata = {"harness_execution": "sandboxed", **{f"harness_{k}": v for k, v in episode.runtime.items()}}
        if episode.return_value is not None:
            metadata["harness_return_value"] = json.dumps(episode.return_value, default=str)[:4000]
        if episode.terminal_response_id:
            # Gym attributes the main chain to this served response; the output is the policy's own
            # items, never a synthetic message, so the attribution witnesses stay intact.
            metadata["terminal_response_id"] = episode.terminal_response_id
        return NeMoGymResponse.model_validate(
            {
                "id": f"nooa-{rollout_id}",
                "created_at": time(),
                "model": body.model or "nooa",
                "object": "response",
                "output": list(episode.output_items),
                "status": status,
                "error": error,
                "tools": [t.model_dump(mode="json") for t in body.tools or []],
                "tool_choice": body.tool_choice,
                "parallel_tool_calls": body.parallel_tool_calls,
                "usage": usage.model_dump() if usage is not None else None,
                "metadata": metadata,
            }
        )

    def _evidence(self, state: NOOASession) -> AgentObservationBundle:
        """Observations only. Gym joins the model server's capture to these invocations by
        the x-session-id each call carried and derives the turns from it, so the harness
        neither restates model calls nor builds a trajectory of its own."""
        episode = state.episode
        gaps: list[ObservationGap] = []
        if episode is None:
            gaps.append(ObservationGap(code="observation_capture_failed", detail=state.failure or "no activation ran"))
            return AgentObservationBundle(source="nooa", gaps=gaps)
        if episode.error:
            gaps.append(ObservationGap(code=episode.stop_reason, detail=episode.error))
        status = (
            "completed"
            if episode.stop_reason == "completed"
            else "incomplete"
            if episode.stop_reason in _INCOMPLETE_STOP_REASONS
            else "failed"
        )
        ordered = [
            *(c.invocation_id for c in episode.model_calls),
            *(t.invocation_id for t in episode.tool_calls),
        ] or ["root"]
        invocation_ids = list(dict.fromkeys(ordered))
        invocations = {
            i: AgentInvocation(invocation_id=i, status=status, stop_reason=episode.stop_reason) for i in invocation_ids
        }
        invocations[invocation_ids[0]].conversation = episode.conversation
        tools = [
            ToolCallObservation(
                invocation_id=t.invocation_id,
                tool_call_id=t.tool_call_id,
                tool_name=t.name,
                started_at=t.started_at,
                completed_at=t.completed_at or None,
                duration_ms=max(0.0, (t.completed_at - t.started_at) * 1000) if t.completed_at else None,
                timing_source="harness",
                status=t.status if t.status in {"completed", "failed"} else "incomplete",
            )
            for t in episode.tool_calls
        ]
        return AgentObservationBundle(source="nooa", records=[*invocations.values(), *tools], gaps=gaps)


if __name__ == "__main__":
    NOOAAgent.run_webserver()
