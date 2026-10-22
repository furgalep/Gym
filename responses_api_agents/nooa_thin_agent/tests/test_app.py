# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Response and observation projection of the thin NOOA agent, without servers."""

from __future__ import annotations

from nooa_bench.gym import GymEpisode, ModelCallRecord, ToolCallRecord

from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.nooa_thin_agent.app import NOOAThinAgent, NOOAThinAgentConfig, NOOAThinSession


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


def _agent() -> NOOAThinAgent:
    config = NOOAThinAgentConfig(
        host="127.0.0.1",
        port=1,
        entrypoint="app.py",
        name="nooa_thin_agent",
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return NOOAThinAgent.model_construct(config=config)


def _episode(stop_reason="completed", calls=2, error=None) -> GymEpisode:
    records = [
        ModelCallRecord(
            "inv",
            {"input": []},
            {"id": f"r{i}", "output": [CALL if i < calls - 1 else MESSAGE], "usage": USAGE},
            200,
            1.0,
            2.0,
        )
        for i in range(calls)
    ]
    return GymEpisode(
        output_items=[CALL, {"type": "function_call_output", "call_id": "c1", "output": "ok"}, MESSAGE]
        if calls
        else [],
        conversation=[{"role": "user", "content": "task"}, CALL, MESSAGE],
        model_calls=records,
        tool_calls=[ToolCallRecord("inv", "c1", "shell", {"command": "ls"}, "ok", 1.0, 1.5, "completed")],
        return_value="DONE",
        stop_reason=stop_reason,
        error=error,
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
    # The verifier still sees the return value as a final message.
    assert response.output[-1].type == "message"


def test_evidence_has_one_invocation_per_id_and_no_model_call_refs():
    state = NOOAThinSession(
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
    state = NOOAThinSession(
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
