# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gym's side of the worker contract: task text, response projection, evidence, termination."""

from __future__ import annotations

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import set_response_lifecycle
from responses_api_agents.nooa_agent.termination import NOOATerminationReason
from responses_api_agents.nooa_agent.worker import (
    Episode,
    ModelCall,
    ToolCall,
    evidence,
    model_status_code,
    project_response,
    task_text,
    termination,
    worker_payload,
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


def episode(stop_reason: str = "completed", calls: int = 2, error: str | None = None) -> Episode:
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


def body(**kwargs) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(**({"input": [{"role": "user", "content": "task"}]} | kwargs))


def finalized(e: Episode):
    reason, error = termination(e)
    return set_response_lifecycle(project_response(e, body(), "r-0"), reason, error)


def test_response_passes_training_fields_sums_usage_and_declares_the_terminal():
    response = finalized(episode())
    assert response.status == "completed" and response.error is None
    assert [item.type for item in response.output] == ["function_call", "function_call_output", "message"]
    assert response.output[0].generation_token_ids == [2]
    assert response.usage.input_tokens == 20 and response.usage.input_tokens_details.cached_tokens == 8
    # Attribution comes from the served response id, never from a synthetic final message.
    assert response.metadata["terminal_response_id"] == "r1"
    assert response.metadata["harness_return_value"] == '"DONE"'
    assert response.output[-1].id == "m"


def test_policy_limits_are_incomplete_not_failed():
    assert finalized(episode(stop_reason="policy_budget_exceeded")).status == "incomplete"
    assert finalized(episode(stop_reason="max_iterations_exceeded")).status == "incomplete"


def test_a_supervised_budget_stop_is_incomplete_with_its_reason():
    response = finalized(episode(stop_reason="timeout_exceeded", calls=1, error="stopped"))
    assert response.status == "incomplete" and response.error.message == "stopped"


def test_errors_fail_with_zero_usage_and_no_invented_output():
    response = finalized(episode(stop_reason="error", calls=0, error="boom"))
    assert response.status == "failed" and response.error.message == "boom"
    assert response.usage.total_tokens == 0
    assert response.output == []  # no synthetic message
    assert "terminal_response_id" not in response.metadata


def test_usage_is_unknown_when_the_provider_omitted_it():
    e = episode()
    del e.model_calls[0].response["usage"]
    assert project_response(e, body(), "r-0").usage is None


@pytest.mark.parametrize(
    ("stop", "reason"),
    [
        ("completed", None),
        ("policy_budget_exceeded", NOOATerminationReason.POLICY_BUDGET_EXCEEDED),
        ("loop_detected", NOOATerminationReason.LOOP_DETECTED),
        ("retry_limit_exceeded", NOOATerminationReason.RETRY_LIMIT_EXCEEDED),
        ("timeout_exceeded", NOOATerminationReason.TIMEOUT_EXCEEDED),
        ("error", NOOATerminationReason.INFRASTRUCTURE_ERROR),
        ("something_new", NOOATerminationReason.INFRASTRUCTURE_ERROR),
    ],
)
def test_termination_maps_every_worker_stop_reason(stop, reason):
    assert termination(episode(stop_reason=stop, error="why"))[0] is reason


def test_evidence_has_one_invocation_per_id_and_no_model_call_refs():
    bundle = evidence(episode(), None)
    invocations = [r for r in bundle.records if r.kind == "agent_invocation"]
    tools = [r for r in bundle.records if r.kind == "tool_call"]
    assert [i.invocation_id for i in invocations] == ["inv"]
    assert invocations[0].model_calls == []  # Gym joins the capture by x-session-id
    assert invocations[0].conversation[0].role == "user"
    assert [(t.tool_name, t.status) for t in tools] == [("shell", "completed")]
    assert bundle.gaps == []


def test_evidence_without_an_episode_records_the_failure_as_a_gap():
    bundle = evidence(None, "timeout_exceeded")
    assert [(g.code, g.detail) for g in bundle.gaps] == [("observation_capture_failed", "timeout_exceeded")]


def test_evidence_marks_incomplete_and_failed_stops():
    incomplete = evidence(episode(stop_reason="timeout_exceeded", error="stopped"), None)
    assert {r.status for r in incomplete.records if r.kind == "agent_invocation"} == {"incomplete"}
    assert incomplete.gaps[0].code == "timeout_exceeded"
    failed = evidence(episode(stop_reason="error", error="boom"), None)
    assert {r.status for r in failed.records if r.kind == "agent_invocation"} == {"failed"}


def test_task_text_keeps_instructions_and_preamble_in_order():
    request = body(
        instructions="Be brief.",
        input=[
            {"role": "system", "content": "You fix bugs."},
            {"role": "developer", "content": [{"type": "input_text", "text": "Use pytest."}]},
            {"role": "user", "content": "Fix the failing test."},
            {"role": "user", "content": [{"type": "input_text", "text": "Then stop."}]},
        ],
    )
    assert task_text(request) == "Be brief.\n\nYou fix bugs.\n\nUse pytest.\n\nFix the failing test.\n\nThen stop."
    assert task_text(body(input="plain")) == "plain"


@pytest.mark.parametrize(
    "input",
    [
        [{"role": "user", "content": "a"}, {"role": "assistant", "content": [{"type": "output_text", "text": "b"}]}],
        [{"type": "function_call_output", "call_id": "c", "output": "x"}],
        [{"role": "system", "content": "only a preamble"}],
        [{"role": "user", "content": "a"}, {"role": "system", "content": "late"}],
    ],
)
def test_task_text_refuses_what_the_worker_cannot_replay(input):
    with pytest.raises(ValueError):
        task_text(body(input=input))


def test_worker_payload_carries_the_agent_entry_and_the_tool_grant():
    invocation = NOOAInvocationConfig(
        agent_class="nooa_bench.gym:ShellAgent",
        turn_method="go",
        tool_binding="probes:bind",
        init_kwargs={"depth": 2},
    )
    payload = worker_payload(
        task="t",
        invocation=invocation,
        model_url="http://m/ng-rollout/r/v1",
        body=body(
            tools=[
                {
                    "type": "function",
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {"type": "object", "properties": {}},
                    "strict": False,
                }
            ]
        ),
        workdir="/app",
        tool_access=None,
        model_cookies={"s": "1"},
        max_policy_calls=5,
        context_window=4096,
        shell_timeout_seconds=30,
        trace={"otlp_endpoint": "http://c:4318/v1/traces", "trace_jsonl": True, "artifacts_dir": "/tmp/a"},
    )
    assert payload["agent"] == "nooa_bench.gym:ShellAgent" and payload["turn_method"] == "go"
    assert payload["tool_binding"] == "probes:bind" and payload["agent_kwargs"] == {"depth": 2}
    assert payload["tools"][0]["name"] == "lookup" and payload["tool_access"] is None
    assert payload["otlp_endpoint"].endswith("/v1/traces") and payload["artifacts_dir"] == "/tmp/a"


def test_an_episode_that_ended_on_a_model_rejection_is_classified_for_the_retry_policy():
    e = episode(stop_reason="error", calls=1, error="BadRequestError: controlled model failure")
    e.model_calls.append(
        ModelCall(invocation_id="inv", request={"input": []}, response={"error": {}}, status_code=400)
    )
    assert termination(e)[0] is NOOATerminationReason.MODEL_REJECTED
    assert model_status_code(e) == 400
    e.model_calls[-1].status_code = 503
    assert termination(e)[0] is NOOATerminationReason.INFRASTRUCTURE_ERROR
    assert model_status_code(episode()) is None  # ended on a success
