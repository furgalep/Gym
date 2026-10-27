# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym's side of the contract with ``nooa_bench.gym_worker``, the NOOA episode runner in the sandbox.

Gym writes a JSON payload, starts the worker on the staged NOOA runtime, and reads back an
:class:`Episode`. Nothing here imports NOOA: the two sides share only this file format and the
Responses API. The worker's model calls go straight to Gym's rollout-prefixed model server, which
holds the authoritative capture, so the episode carries observations and never a trajectory.
"""

from __future__ import annotations

import json
from time import time
from typing import Any

from pydantic import BaseModel, Field

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
from nemo_gym.tool_access import DirectHTTPToolAccess
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.termination import NOOATerminationReason


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
    # completed | policy_budget_exceeded | max_output_tokens_exceeded | max_iterations_exceeded |
    # loop_detected | retry_limit_exceeded | timeout_exceeded | error
    stop_reason: str
    error: str | None = None
    # Served ``response.id`` of the root agent's last completed call: the main chain's terminal.
    terminal_response_id: str | None = None
    resource_cookies: dict[str, str] = Field(default_factory=dict)
    # Where the agent ran (interpreter, version, host, artifact paths), filled in by the worker.
    runtime: dict[str, str] = Field(default_factory=dict)


def _text(content: Any, role: str) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for part in content or []:
        part = part if isinstance(part, dict) else part.model_dump()
        if part.get("type") != "input_text":
            raise ValueError(f"NOOA accepts text content only; {role} message carries {part.get('type')!r}")
        parts.append(part.get("text", ""))
    return "\n".join(parts)


def task_text(body: NeMoGymResponseCreateParamsNonStreaming) -> str:
    """One task string from the request: instructions, system/developer preamble, then the user turn(s).

    The turn method takes a single prompt, so nothing is dropped silently: the preamble is prepended
    in order and any item NOOA cannot represent (assistant history, tool items) is a ``ValueError``.
    """
    items = [{"type": "message", "role": "user", "content": body.input}] if isinstance(body.input, str) else body.input
    sections: list[str] = [body.instructions] if body.instructions else []
    users: list[str] = []
    for index, item in enumerate(items):
        data = item if isinstance(item, dict) else item.model_dump()
        kind, role = data.get("type", "message"), data.get("role")
        if kind != "message":
            raise ValueError(f"NOOA expects message items, but input[{index}] is {kind!r}")
        if role in {"system", "developer"}:
            if users:
                raise ValueError(f"NOOA expects {role} messages before the first user message")
            sections.append(_text(data.get("content"), role))
        elif role == "user":
            users.append(_text(data.get("content"), role))
        else:
            raise ValueError(f"NOOA cannot replay {role!r} messages; input[{index}] is unsupported")
    if not users:
        raise ValueError("NOOA task input must contain a user message")
    return "\n\n".join(section for section in [*sections, *users] if section)


def _tool_dicts(body: NeMoGymResponseCreateParamsNonStreaming) -> list[dict[str, Any]]:
    # Tool params arrive as TypedDicts or models depending on how the request was built.
    return [dict(t) if isinstance(t, dict) else t.model_dump(mode="json", exclude_none=True) for t in body.tools or []]


def worker_payload(
    *,
    task: str,
    invocation: NOOAInvocationConfig,
    model_url: str,
    body: NeMoGymResponseCreateParamsNonStreaming,
    workdir: str,
    tool_access: DirectHTTPToolAccess | None,
    model_cookies: dict[str, str],
    max_policy_calls: int | None,
    context_window: int | None,
    shell_timeout_seconds: int,
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The JSON the worker reads: agent entry, task, model seam, tools and their grant."""
    return {
        "task": task,
        "agent": invocation.agent_class,
        "turn_method": invocation.turn_method,
        "agent_kwargs": invocation.init_kwargs,
        "tool_binding": invocation.tool_binding,
        "model_url": model_url,
        "max_policy_calls": max_policy_calls,
        "context_window": context_window,
        "shell_timeout_seconds": shell_timeout_seconds,
        "workdir": workdir,
        "cookies": model_cookies,
        "tools": _tool_dicts(body),
        "tool_access": {
            "base_url": str(tool_access.base_url),
            "headers": dict(tool_access.headers),
            "cookies": dict(tool_access.cookies),
        }
        if tool_access
        else None,
        **(trace or {}),
    }


_STOP_REASONS = {
    "policy_budget_exceeded": NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
    "max_output_tokens_exceeded": NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
    "max_iterations_exceeded": NOOATerminationReason.MAX_ITERATIONS_EXCEEDED,
    "loop_detected": NOOATerminationReason.LOOP_DETECTED,
    "retry_limit_exceeded": NOOATerminationReason.RETRY_LIMIT_EXCEEDED,
    "timeout_exceeded": NOOATerminationReason.TIMEOUT_EXCEEDED,
}


def model_status_code(episode: Episode) -> int | None:
    """The HTTP status of the last model call that did not succeed, if the episode ended on one."""
    if not episode.model_calls:
        return None
    last = episode.model_calls[-1]
    return last.status_code if last.status_code is not None and last.status_code >= 400 else None


def termination(episode: Episode) -> tuple[NOOATerminationReason | None, str | None]:
    """The worker's stop reason in the terms the rest of this adapter uses.

    An episode that ended on the model server rejecting the request (400, 413, 422) is a model
    rejection, which the environment's bounded retry policy handles; every other unnamed stop is an
    infrastructure error.
    """
    if episode.stop_reason == "completed":
        return None, None
    reason = _STOP_REASONS.get(episode.stop_reason)
    if reason is None:
        rejected = model_status_code(episode) in {400, 413, 422}
        reason = NOOATerminationReason.MODEL_REJECTED if rejected else NOOATerminationReason.INFRASTRUCTURE_ERROR
    return reason, episode.error or episode.stop_reason


def _zero_usage() -> NeMoGymResponseUsage:
    return NeMoGymResponseUsage(
        input_tokens=0,
        output_tokens=0,
        total_tokens=0,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
    )


def project_response(
    episode: Episode, body: NeMoGymResponseCreateParamsNonStreaming, rollout_id: str
) -> NeMoGymResponse:
    """The policy's own output items, summed usage, and the attribution metadata.

    No synthetic final message: ``metadata.terminal_response_id`` names the root agent's last
    completed model response, so token-id capture attributes the main chain from the served
    response alone. The return value travels in metadata. Status and error are the caller's to
    set from :func:`termination` (``set_response_lifecycle``).
    """
    completed = [c for c in episode.model_calls if c.response and (c.status_code or 500) < 400]
    usages = [c.response["usage"] for c in completed if c.response.get("usage")]
    if not completed:
        usage = _zero_usage()  # a rollout with no finished response has a known zero total
    elif len(usages) == len(completed):
        usage = NeMoGymResponseUsage.sum_from_list([NeMoGymResponseUsage.model_validate(u) for u in usages])
    else:
        usage = None  # the provider omitted usage: counts are unknown, never invented
    metadata = {"harness_execution": "sandboxed", **{f"harness_{k}": v for k, v in episode.runtime.items()}}
    if episode.return_value is not None:
        metadata["harness_return_value"] = json.dumps(episode.return_value, default=str)[:4000]
    if episode.terminal_response_id:
        metadata["terminal_response_id"] = episode.terminal_response_id
    return NeMoGymResponse.model_validate(
        {
            "id": f"nooa-{rollout_id}",
            "created_at": time(),
            "model": body.model or "nooa",
            "object": "response",
            "output": list(episode.output_items),
            "status": "completed",
            "error": None,
            "tools": _tool_dicts(body),
            "tool_choice": body.tool_choice,
            "parallel_tool_calls": body.parallel_tool_calls,
            "usage": usage.model_dump() if usage is not None else None,
            "metadata": metadata,
        }
    )


def evidence(episode: Episode | None, failure: str | None) -> AgentObservationBundle:
    """Observations only. Gym joins the model server's capture to these invocations by the
    x-session-id each call carried and derives the turns from it, so the harness neither
    restates model calls nor builds a trajectory of its own."""
    gaps: list[ObservationGap] = []
    if episode is None:
        gaps.append(ObservationGap(code="observation_capture_failed", detail=failure or "no activation ran"))
        return AgentObservationBundle(source="nooa", gaps=gaps)
    reason, error = termination(episode)
    if error:
        gaps.append(ObservationGap(code=str(reason), detail=error))
    status = (
        "completed"
        if reason is None
        else "incomplete"
        if reason
        in {
            NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
            NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
            NOOATerminationReason.TIMEOUT_EXCEEDED,
        }
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
