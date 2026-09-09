# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from http.cookies import SimpleCookie
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from nooa import Agent, strategy
from pydantic import BaseModel, ConfigDict

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.config import NOOAInvocationConfig
from responses_api_agents.nooa_agent.runner import ArgumentMappingError, EmbeddedNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


class ValidAgent(Agent):
    def __init__(self, *, llm: Any, label: str) -> None:
        super().__init__(llm=llm)
        self.label = label

    async def analyze(self, text: str, customer_id: str) -> str: ...


class AliasedAgent(Agent):
    async def analyze(self, text: str) -> list[str]:
        primary = await self.primary(text)
        helper = await self.helper(text)
        child = type(self)(llm=self._llm)
        return [primary, helper, await child.helper(text)]

    async def primary(self, text: str) -> str:
        """Answer the question."""
        ...

    @strategy(llm="helper")
    async def helper(self, text: str) -> str:
        """Answer the question."""
        ...


class UnknownAliasAgent(Agent):
    @strategy(llm="not-configured")
    async def analyze(self, text: str) -> str:
        """Answer the question."""
        ...


class FakeAgent:
    instances = 0
    get_weather: Any

    def __init__(self, *, llm: Any, label: str) -> None:
        FakeAgent.instances += 1
        self.llm = llm
        self.label = label
        self.event_manager = FakeEventManager()

    async def analyze(self, text: str, customer_id: str) -> str:
        weather = await self.get_weather(city=customer_id)
        return f"{text}: {weather['weather']}"


class FakeEventManager:
    def on(self, event_type: str, handler: Any) -> Any:
        return lambda: None


class Row(BaseModel):
    model_config = ConfigDict(extra="allow")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming
    agent_inputs: dict[str, str]


class FakeContent:
    async def read(self) -> bytes:
        return json.dumps({"weather": "cold"}).encode()


class FakeResponse:
    status = 200
    content = FakeContent()
    cookies = SimpleCookie()


def make_runner() -> tuple[EmbeddedNOOARunner, MagicMock]:
    invocation = NOOAInvocationConfig.model_validate(
        {
            "agent_class": f"{__name__}:ValidAgent",
            "entrypoint": "analyze",
            "init_kwargs": {"label": "configured"},
            "allowed_tools": ["get_weather"],
            "arguments": {
                "text": {
                    "source": "responses_create_params.input",
                    "transform": "latest_user_text",
                },
                "customer_id": {"source": "agent_inputs.customer_id"},
            },
        }
    )
    client = MagicMock()
    client.post = AsyncMock(return_value=FakeResponse())
    runner = EmbeddedNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="policy_model",
        resources_server_name="weather_resources",
        max_steps=3,
    )
    runner._agent_class = FakeAgent
    return runner, client


def row(customer_id: str) -> Row:
    return Row(
        agent_inputs={"customer_id": customer_id},
        responses_create_params={
            "input": [{"role": "user", "content": "Check delivery"}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
    )


@pytest.mark.asyncio
async def test_embedded_runner_maps_full_row_and_attaches_resource_methods() -> None:
    runner, client = make_runner()

    result = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            model_url_path="/ng-rollout/rollout-1/v1/responses",
            resource_cookies={"session": "one"},
        )
    )

    assert result.episode.response.output == []
    assert result.return_value == "Check delivery: cold"
    assert result.episode.observations.source == "nooa"
    assert result.episode.observations.gaps == []
    assert client.post.await_args.kwargs["json"] == {"city": "Paris"}


@pytest.mark.asyncio
async def test_row_cannot_widen_configured_tools() -> None:
    runner, client = make_runner()
    runner._invocation.allowed_tools = []
    with pytest.raises(ValueError, match="configured allowed_tools"):
        await runner.run(NOOARunRequest(row=row("Paris"), model_url_path="/v1/responses"))
    client.post.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedded_runner_classifies_argument_mapping_errors() -> None:
    runner, _ = make_runner()
    incomplete_row = row("Paris").model_copy(update={"agent_inputs": {}})

    with pytest.raises(ArgumentMappingError, match="customer_id") as error:
        await runner.run(
            NOOARunRequest(
                row=incomplete_row,
                model_url_path="/v1/responses",
            )
        )

    assert isinstance(error.value.__cause__, ValueError)


@pytest.mark.asyncio
async def test_constructs_a_fresh_agent_for_every_rollout() -> None:
    runner, _ = make_runner()
    FakeAgent.instances = 0

    first = await runner.run(
        NOOARunRequest(
            row=row("Paris"),
            model_url_path="/one/v1/responses",
        )
    )
    second = await runner.run(
        NOOARunRequest(
            row=row("Berlin"),
            model_url_path="/two/v1/responses",
        )
    )

    assert FakeAgent.instances == 2
    assert first.episode is not second.episode
    assert first.resource_cookies is not second.resource_cookies


def alias_runner(agent_class: type[Agent] = AliasedAgent) -> tuple[EmbeddedNOOARunner, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []

    async def post(*, server_name: str, cookies: dict, **kwargs: Any) -> FakeHTTPResponse:
        calls.append((server_name, dict(cookies)))
        output = NeMoGymResponseFunctionToolCall(
            id=f"fc-{len(calls)}",
            call_id=f"call-{len(calls)}",
            name="return_result",
            arguments=json.dumps({"result": server_name}),
        )
        jar = SimpleCookie()
        jar["session"] = server_name
        return FakeHTTPResponse(model_response(output, response_id=f"response-{len(calls)}"), jar)

    client = MagicMock()
    client.post = AsyncMock(side_effect=post)
    invocation = NOOAInvocationConfig(
        agent_class=f"{__name__}:{agent_class.__name__}",
        entrypoint="analyze",
        arguments={"text": {"source": "responses_create_params.input", "transform": "latest_user_text"}},
        model_aliases={"helper": "helper_model"},
    )
    return EmbeddedNOOARunner(
        invocation=invocation,
        server_client=client,
        model_server_name="primary_model",
        resources_server_name="resources",
        max_steps=3,
    ), calls


@pytest.mark.asyncio
async def test_actual_alias_dispatch_and_children_share_only_their_rollouts_clients() -> None:
    runner, calls = alias_runner()
    for _ in range(2):
        result = await runner.run(
            NOOARunRequest(
                row=Row(responses_create_params={"input": "question"}, agent_inputs={}),
                model_url_path="/v1/responses",
                model_cookies={"session": "inbound"},
            )
        )
        assert result.return_value == ["primary_model", "helper_model", "helper_model"]
        assert result.model_cookies == {"session": "primary_model"}
        assert result.termination_reason is None
    assert (
        calls
        == [
            ("primary_model", {"session": "inbound"}),
            ("helper_model", {"session": "inbound"}),
            ("helper_model", {"session": "helper_model"}),
        ]
        * 2
    )
    assert not hasattr(AliasedAgent, "_strategy_llm_alias_cache")


@pytest.mark.asyncio
async def test_unknown_method_alias_never_consults_nooa_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MagicMock(side_effect=AssertionError("external registry must not be consulted"))
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", registry)
    runner, calls = alias_runner(UnknownAliasAgent)
    with pytest.raises(ValueError, match="not in configured model_aliases"):
        await runner.run(
            NOOARunRequest(
                row=Row(responses_create_params={"input": "question"}, agent_inputs={}),
                model_url_path="/v1/responses",
            )
        )
    registry.assert_not_called()
    assert calls == []
