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

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from responses_api_agents.nooa_agent.invocation import (
    NOOAInvocationAdapter as NOOAInvocationAdapter,
)
from responses_api_agents.nooa_agent.invocation import (
    NOOAInvocationConfig as NOOAInvocationConfig,
)
from responses_api_agents.nooa_agent.invocation import (
    load_agent_class as load_agent_class,
)
from responses_api_agents.nooa_agent.invocation import (
    load_invocation_adapter as load_invocation_adapter,
)
from responses_api_agents.nooa_agent.invocation import (
    validate_invocation as validate_invocation,
)


def _otlp_traces_endpoint() -> str | None:
    """NOOA's OTLP exporter wants the traces URL; Gym's telemetry names the collector's base URL."""
    base = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not base:
        return None
    base = base.rstrip("/")
    return base if base.endswith("/v1/traces") else f"{base}/v1/traces"


class NOOATracingConfig(BaseModel):
    """NOOA's own instrumentation inside the sandbox, and where it ends up.

    ATIF trajectories and OTLP spans are written by the worker; after the episode the agent server
    archives them out of the sandbox, one ``<rollout_id>.tar.gz`` per episode under ``artifacts_output``.
    ``otlp_endpoint`` additionally ships the spans live to a collector, by default the one Gym's
    telemetry points at (``OTEL_EXPORTER_OTLP_ENDPOINT``); the JSONL spool keeps the run's own copy.
    """

    model_config = ConfigDict(extra="forbid")

    atif: bool = Field(default=True, description="Export ATIF trajectories (nooa.atif) from the worker.")
    trace_jsonl: bool = Field(default=True, description="Spool NOOA's OTLP spans as JSONL files in the sandbox.")
    otlp_endpoint: str | None = Field(
        default_factory=_otlp_traces_endpoint,
        description="Collector URL for live span export from the sandbox; the pod must be able to reach it.",
    )
    artifacts_output: Path | None = Field(
        default=Path("nooa-artifacts"),
        description="Local directory receiving one archive of the sandbox artifacts per rollout; null discards them.",
    )

    @property
    def worker_payload(self) -> dict[str, object]:
        """The keys nooa_bench.gym_worker reads; ``artifacts_dir`` is added by the runner."""
        payload: dict[str, object] = {}
        if self.atif:
            payload["atif"] = True
        if self.trace_jsonl:
            payload["trace_jsonl"] = True
        if self.otlp_endpoint:
            payload["otlp_endpoint"] = self.otlp_endpoint
        return payload

    @property
    def collects_artifacts(self) -> bool:
        return (self.atif or self.trace_jsonl) and self.artifacts_output is not None


class NOOAAgentConfig(BaseResponsesAPIAgentConfig):
    """Gym server configuration for the NOOA adapter."""

    model_config = ConfigDict(extra="forbid")

    tracing: NOOATracingConfig = Field(default_factory=NOOATracingConfig)

    resources_server: ResourcesServerRef | None = Field(
        default=None,
        description="Dataset/schema discovery metadata; runtime access comes from environment session grants.",
    )
    model_server: ModelServerRef
    runtime_tarball: str | None = Field(
        default_factory=lambda: os.environ.get("NOOA_GYM_RUNTIME_TARBALL") or None,
        description="Prebuilt relocatable NOOA runtime for glibc task images (see runtime.py build), staged into the "
        "sandbox at seed time; nothing is resolved or installed inside the sandbox.",
    )
    runtime_tarball_musl: str | None = Field(
        default_factory=lambda: os.environ.get("NOOA_GYM_RUNTIME_TARBALL_MUSL") or None,
        description="The musl build for Alpine-style task images; staging probes the image's libc and picks it.",
    )
    runtime_tarball_required: bool = Field(
        default=False,
        description="Require a prebuilt runtime before starting this agent (network-restricted sandboxes).",
    )
    nooa: NOOAInvocationConfig
    max_policy_calls: int | None = Field(
        default=None, gt=0, description="Optional shared model-call limit per rollout; null disables the limit."
    )
    context_window: int | None = Field(
        default=None, gt=0, description="Known policy context window used by NOOA's context planner."
    )

    runtime_python: str | None = Field(
        default=None,
        description="Interpreter inside the sandbox that already has NOOA installed; skips runtime preparation. "
        "Only meaningful with a local sandbox provider (tests, conformance).",
    )
    shell_timeout_seconds: int = Field(
        default=180, gt=0, description="Per-command limit for the worker's shell tool, when the agent has one."
    )
    sandbox_runner_timeout_seconds: float = Field(
        default=21420,
        gt=0,
        allow_inf_nan=False,
        description="Finite sandbox worker ceiling in seconds; separate from the environment's episode timeout.",
    )
    session_close_timeout_seconds: float = Field(
        default=30,
        gt=0,
        allow_inf_nan=False,
        description="Budget in seconds for each sandbox close phase and for joining the remaining activation.",
    )

    @field_validator("runtime_tarball", "runtime_tarball_musl")
    @classmethod
    def normalize_runtime_tarball(cls, value: str | None) -> str | None:
        if value is None:
            return None
        expanded = Path(value).expanduser()
        if not expanded.is_absolute():
            raise ValueError(
                "runtime tarballs must be absolute paths (or ~/path); relative server paths are ambiguous"
            )
        return str(expanded.resolve())

    @model_validator(mode="after")
    def require_prepared_runtime(self) -> NOOAAgentConfig:
        if self.runtime_tarball_required and self.runtime_tarball is None and self.runtime_python is None:
            raise ValueError(
                "This recipe requires a prebuilt NOOA runtime. Set NOOA_GYM_RUNTIME_TARBALL (and "
                "NOOA_GYM_RUNTIME_TARBALL_MUSL for Alpine images) to archives built with "
                "python -m responses_api_agents.nooa_agent.runtime build before starting servers."
            )
        return self
