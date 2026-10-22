# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin NOOA agent server for NeMo Gym.

Two execution modes share one evidence path:

* ``embedded``: NOOA runs in this process and shells into the task sandbox
  through ``AsyncSandbox.exec``.
* ``sandboxed``: NOOA runs inside the task sandbox as ``python -m
  nooa_bench.gym_worker`` under Gym's shared ``SandboxSession``.

Either way the model seam is NOOA's own Responses client pointed at the
rollout-prefixed Gym model server, and the agent returns the verbatim wire
record that ``nooa_bench.gym.run_gym_episode`` produced. Gym's model server
holds the authoritative capture, so a lost worker loses no training evidence.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from time import time
from typing import Any, Literal
from uuid import uuid4

from fastapi import Body, HTTPException, Request
from nooa_bench.gym import GymEpisode, resource_tools, run_gym_episode
from pydantic import Field

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agent_utils.task_timeout import resolve_agent_timeout
from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
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
from responses_api_agents.nooa_thin_agent.runtime import stage_runtime


class NOOAThinAgentConfig(BaseResponsesAPIAgentConfig):
    resources_server: ResourcesServerRef | None = None
    model_server: ModelServerRef
    agent: str = "nooa_bench.gym:ShellAgent"
    # Extra constructor arguments for the agent class (for AtomHarness: turn_method).
    agent_kwargs: dict[str, Any] = Field(default_factory=dict)
    # Policy context window, handed to NOOA for its context budgeting; None leaves NOOA's default.
    context_window: int | None = Field(default=None, gt=0)
    execution_mode: Literal["embedded", "sandboxed"] = "sandboxed"
    # Interpreter inside the sandbox that has NOOA installed. None uses this
    # server's own interpreter, which is only meaningful with a local sandbox.
    runtime_python: str | None = None
    # Prebuilt relocatable NOOA runtime (see runtime.py). Staged into the sandbox at seed time
    # when set; it then supplies ``runtime_python`` for the session.
    runtime_tarball: str | None = Field(default_factory=lambda: os.environ.get("NOOA_GYM_RUNTIME_TARBALL") or None)
    timeout: float = Field(default=900, gt=0)
    close_timeout: float = Field(default=30, gt=0)
    max_policy_calls: int | None = Field(default=None, gt=0)
    shell_timeout_seconds: int = Field(default=180, gt=0)
    sandbox_provider: str | None = None
    sandbox_config: dict[str, Any] = Field(default_factory=dict)


@dataclass
class NOOAThinSession(AgentSessionState):
    sandbox: AsyncSandbox | None = None
    session: SandboxSession[GymEpisode] | None = None
    owns_sandbox: bool = False
    workdir: str = "/app"
    tool_access: DirectHTTPToolAccess | None = None
    runtime_python: str | None = None
    model_cookies: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    execution: asyncio.Task[NeMoGymResponse] | None = None
    episode: GymEpisode | None = None
    failure: str | None = None


def _task_text(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    if isinstance(body.input, str):
        return body.input
    for item in reversed(body.input):
        data = item.model_dump() if hasattr(item, "model_dump") else dict(item)
        if data.get("role") == "user":
            content = data.get("content")
            if isinstance(content, str):
                return content
            return "\n".join(p.get("text", "") for p in content or [] if isinstance(p, dict))
    raise HTTPException(422, "NOOA task input must contain a user message")


def _zero_usage() -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
    )


class NOOAThinAgent(SimpleResponsesAPIAgent):
    config: NOOAThinAgentConfig

    async def run(self, body: object) -> None:
        raise NotImplementedError("Use the single_agent_turn Environment Server")

    # -- sessions ---------------------------------------------------------------------------------

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> NOOAThinSession:
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
        provider = create_provider(resolve_provider_config(provider_ref, get_global_config_dict()))
        if owns:
            sandbox = AsyncSandbox(provider)
            await sandbox.start(spec)
            await sandbox.exec(f"mkdir -p -- {shlex.quote(workdir)}", cwd="/", timeout_s=30)
        else:
            sandbox = await AsyncSandbox.connect(body.sandbox_access.connection.descriptor, provider=provider)
        accesses = self.effective_tool_accesses(body)
        direct = [a for a in accesses if isinstance(a, DirectHTTPToolAccess)]
        if any(a.required for a in accesses if not isinstance(a, DirectHTTPToolAccess)) or len(direct) > 1:
            raise HTTPException(422, "NOOA supports at most one direct HTTP tool grant")
        state = NOOAThinSession(
            request=body,
            sandbox=sandbox,
            owns_sandbox=owns,
            workdir=workdir,
            tool_access=direct[0] if direct else None,
        )
        if self.config.execution_mode == "sandboxed":
            if self.config.runtime_tarball:
                state.runtime_python = await stage_runtime(sandbox, self.config.runtime_tarball)
            state.session = SandboxSession(
                sandbox=sandbox,
                session_dir=f"/tmp/nemo-gym-nooa/{uuid4().hex}",
                workdir=workdir,
                harness="NOOA",
                owns_sandbox=owns,
            )
            await sandbox.exec(f"mkdir -p -- {shlex.quote(state.session.session_dir)}", cwd="/", timeout_s=30)
        return state

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        assert isinstance(state, NOOAThinSession)
        if state.session is not None:
            await state.session.close(timeout=self.config.close_timeout)
            if state.execution is not None and not state.execution.done():
                state.execution.cancel()
                await asyncio.gather(state.execution, return_exceptions=True)
            if state.episode is None and isinstance(state.session.artifacts, GymEpisode):
                state.episode = state.session.artifacts
        else:
            if state.execution is not None and not state.execution.done():
                state.execution.cancel()
                await asyncio.gather(state.execution, return_exceptions=True)
            if state.sandbox is not None:
                await (state.sandbox.stop() if state.owns_sandbox else state.sandbox.disconnect())
        cookies = None
        if state.tool_access is not None:
            cookies = (
                dict(state.episode.resource_cookies) if state.episode is not None else dict(state.tool_access.cookies)
            )
        return AgentCloseSessionResponse(
            agent_session_id=state.request.agent_session_id,
            agent_observations=self._evidence(state),
            resources_cookies=cookies,
            cleanup_confirmed=True,
        )

    # -- activation -------------------------------------------------------------------------------

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        session_id = self._agent_session_id_from_request(request)
        if session_id is None:
            raise HTTPException(409, "NOOA requires an agent session")
        state = self._require_agent_session(session_id)
        assert isinstance(state, NOOAThinSession)
        if request.path_params.get("rollout_id") != state.request.episode_id.capture_key:
            raise HTTPException(409, "Agent session does not match the rollout route")
        if body.tools and state.tool_access is None:
            raise HTTPException(422, "Resource tools require a direct HTTP grant")
        route = self.url_path_for_request("/v1/responses", request)
        async with state.lock:
            if state.execution is None:
                state.execution = asyncio.create_task(self._execute(state, body, route))
            execution = state.execution
        return await asyncio.shield(execution)

    def _model_url(self, route: str) -> str:
        base = self.resolve_model_base_url(self.config.model_server.name).removesuffix("/v1")
        return base + route.removesuffix("/responses")

    async def _execute(
        self, state: NOOAThinSession, body: NeMoGymResponseCreateParamsNonStreaming, route: str
    ) -> NeMoGymResponse:
        task = _task_text(body)
        budget = resolve_agent_timeout(body.metadata, ceiling=self.config.timeout)
        try:
            if state.session is not None:
                state.episode = await self._run_sandboxed(state, task, self._model_url(route), budget, body)
            else:
                async with asyncio.timeout(budget):
                    state.episode = await self._run_embedded(state, task, self._model_url(route), body)
        except TimeoutError:
            state.failure = "timeout_exceeded"
            raise HTTPException(504, "NOOA agent exceeded its time budget")
        return self._response(state.episode, body, state.request.episode_id.capture_key)

    async def _run_embedded(
        self, state: NOOAThinSession, task: str, model_url: str, body: NeMoGymResponseCreateParamsNonStreaming
    ) -> GymEpisode:
        assert state.sandbox is not None
        sandbox, workdir = state.sandbox, state.workdir

        async def run_command(command: str) -> str:
            result = await sandbox.exec(command, cwd=workdir, timeout_s=self.config.shell_timeout_seconds)
            text = ((result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")).rstrip()
            if result.error_type or result.return_code != 0:
                raise RuntimeError(f"{text}\n[exit {result.return_code}]")
            return f"{text}\n[exit 0]"

        module_name, _, class_name = self.config.agent.partition(":")
        agent_cls = getattr(importlib.import_module(module_name), class_name)
        resource_cookies = dict(state.tool_access.cookies) if state.tool_access else {}
        extra_tools = (
            resource_tools(
                [t.model_dump(mode="json", exclude_none=True) for t in body.tools],
                base_url=str(state.tool_access.base_url),
                headers=dict(state.tool_access.headers),
                cookies=resource_cookies,
            )
            if body.tools and state.tool_access
            else []
        )
        episode = await run_gym_episode(
            lambda llm: agent_cls(
                llm=llm, run_command=run_command, extra_tools=extra_tools, workdir=workdir, **self.config.agent_kwargs
            ),
            lambda agent: agent.solve(task),
            model_url=model_url,
            max_policy_calls=self.config.max_policy_calls,
            cookies=state.model_cookies,
            resource_cookies=resource_cookies,
            context_window=self.config.context_window,
        )
        episode.runtime = {"python": sys.executable, "version": sys.version.split()[0]}
        return episode

    async def _run_sandboxed(
        self,
        state: NOOAThinSession,
        task: str,
        model_url: str,
        budget: float,
        body: NeMoGymResponseCreateParamsNonStreaming,
    ) -> GymEpisode:
        session = state.session
        assert session is not None and state.sandbox is not None
        python = state.runtime_python or self.config.runtime_python or sys.executable
        inputs, result = f"{session.session_dir}/input.json", f"{session.session_dir}/result.json"

        async def stage() -> SandboxCommand:
            payload = {
                "task": task,
                "agent": self.config.agent,
                "agent_kwargs": self.config.agent_kwargs,
                "context_window": self.config.context_window,
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
            await upload_text(state.sandbox, path=inputs, text=json.dumps(payload))
            return SandboxCommand(argv=[python, "-I", "-m", "nooa_bench.gym_worker", inputs, result], python=python)

        async def collect() -> GymEpisode:
            return GymEpisode.from_json(await read_text(state.sandbox, path=result))

        return await session.execute(
            stage_activation=stage, collect=collect, timeout=budget, close_timeout=self.config.close_timeout
        )

    # -- projection -------------------------------------------------------------------------------

    def _response(
        self, episode: GymEpisode, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: str
    ) -> NeMoGymResponse:
        output = list(episode.output_items)
        last = output[-1] if output else None
        if episode.return_value is not None and not (isinstance(last, dict) and last.get("type") == "message"):
            output.append(
                NeMoGymResponseOutputMessage(
                    id="nooa_fallback",
                    content=[NeMoGymResponseOutputText(annotations=[], text=str(episode.return_value))],
                ).model_dump(mode="json")
            )
        completed = [c for c in episode.model_calls if c.response and (c.status_code or 500) < 400]
        usages = [c.response["usage"] for c in completed if c.response.get("usage")]
        if not completed:
            usage = _zero_usage()  # a rollout with no finished response has a known zero total
        elif len(usages) == len(completed):
            usage = NeMoGymResponseUsage.sum_from_list([NeMoGymResponseUsage.model_validate(u) for u in usages])
        else:
            usage = None  # the provider omitted usage: counts are unknown, never invented
        status, error = "completed", None
        if episode.stop_reason in {"policy_budget_exceeded", "max_output_tokens_exceeded"}:
            status = "incomplete"
        elif episode.stop_reason != "completed":
            status, error = "failed", {"code": "server_error", "message": episode.error or episode.stop_reason}
        return NeMoGymResponse.model_validate(
            {
                "id": f"nooa-{rollout_id}",
                "created_at": time(),
                "model": body.model or "nooa",
                "object": "response",
                "output": output,
                "status": status,
                "error": error,
                "tools": [t.model_dump(mode="json") for t in body.tools or []],
                "tool_choice": body.tool_choice,
                "parallel_tool_calls": body.parallel_tool_calls,
                "usage": usage.model_dump() if usage is not None else None,
                "metadata": {
                    "harness_execution": self.config.execution_mode,
                    **{f"harness_{key}": value for key, value in episode.runtime.items()},
                },
            }
        )

    def _evidence(self, state: NOOAThinSession) -> AgentObservationBundle:
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
        status = {
            "completed": "completed",
            "policy_budget_exceeded": "incomplete",
            "max_output_tokens_exceeded": "incomplete",
        }.get(episode.stop_reason, "failed")
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
    NOOAThinAgent.run_webserver()
