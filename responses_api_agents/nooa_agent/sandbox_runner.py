# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Borrow a task sandbox and supervise one NOOA activation run by ``nooa_bench.gym_worker``."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from shlex import quote
from typing import Any
from uuid import uuid4

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.agent_utils.task_timeout import resolve_agent_timeout
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentEpisode, AgentObservationBundle, ObservationGap
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.utils import read_text, upload_text
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
from responses_api_agents.nooa_agent.result import set_response_lifecycle
from responses_api_agents.nooa_agent.runner import NOOARunRequest, NOOARunResult
from responses_api_agents.nooa_agent.worker import (
    Episode,
    evidence,
    model_status_code,
    project_response,
    task_text,
    termination,
    worker_payload,
)


LOGGER = logging.getLogger(__name__)

# Archives the worker's evidence out of its session directory: the exports directory, the Episode
# file and every sidecar beside it (result.json.*). Runs on the staged interpreter so it behaves the
# same on glibc and musl images, takes whatever exists (a stopped or cancelled worker leaves no
# result.json), handles any file name, and stores symlinks as links rather than following them.
ARCHIVE_SCRIPT = """\
import os, sys, tarfile
root, out = sys.argv[1], sys.argv[2]
with tarfile.open(out, "w:gz") as tar:
    for name in sorted(os.listdir(root)):
        if name in ("artifacts", "result.json") or name.startswith("result.json."):
            tar.add(os.path.join(root, name), arcname=name)
"""


def archive_command(python: str, directory: str, archive: str) -> str:
    """Shell command that archives the evidence in ``directory`` into ``archive`` with ``python``."""
    return f"{quote(python)} -I -c {quote(ARCHIVE_SCRIPT)} {quote(directory)} {quote(archive)}"


class SandboxNOOARunner:
    """Stage the worker's payload, let the shared session supervise the process, read the episode back."""

    def __init__(
        self,
        *,
        sandbox: AsyncSandbox,
        workdir: str,
        python: str,
        invocation: NOOAInvocationConfig,
        model_base_url: str,
        max_policy_calls: int | None,
        context_window: int | None = None,
        timeout: float = 21600,
        close_timeout: float = 30,
        shell_timeout_seconds: int = 180,
        trace: dict[str, Any] | None = None,
        artifacts_output: Path | None = None,
    ) -> None:
        self.sandbox = sandbox
        self.python = python
        self.invocation = invocation
        self.model_base_url = model_base_url
        self.max_policy_calls = max_policy_calls
        self.context_window = context_window
        self.timeout = timeout
        self.close_timeout = close_timeout
        self.shell_timeout_seconds = shell_timeout_seconds
        self.trace = dict(trace or {})
        self.artifacts_output = artifacts_output
        self.artifacts: Path | None = None  # local archive of NOOA's exports, once collected
        # Construct before prepare so failed setup can release the borrowed client.
        self.session = SandboxSession[Episode](
            sandbox=sandbox,
            session_dir=f"/tmp/nemo-gym-nooa/{uuid4().hex}",
            workdir=workdir,
            harness="NOOA",
            owns_sandbox=False,
        )
        self.episode: Episode | None = None
        self.failure: str | None = None
        self.artifacts_error: str | None = None

    @property
    def directory(self) -> str:
        """Location of activation files outside the graded repository."""
        return self.session.session_dir

    @property
    def observations(self) -> AgentObservationBundle:
        bundle = evidence(self.episode, self.failure)
        if self.artifacts_error:
            bundle.gaps.append(ObservationGap(code="artifacts_capture_failed", detail=self.artifacts_error))
        return bundle

    async def prepare(self) -> None:
        """Create session files outside the graded repository."""
        result = await self.sandbox.exec(f"mkdir -p {quote(self.directory)}", cwd="/", timeout_s=30)
        if result.return_code != 0 or result.error_type:
            raise RuntimeError("Could not prepare NOOA session directory")

    async def run(self, request: NOOARunRequest) -> NOOARunResult:
        """Execute once. A budget stop the supervisor vouches for is a gradable episode; a missing result is not."""
        body = request.responses_create_params
        task = task_text(body)
        model_url = self.model_base_url + request.model_url_path.removesuffix("/responses")
        budget = resolve_agent_timeout(body.metadata, ceiling=self.timeout)
        inputs, result = f"{self.directory}/input.json", f"{self.directory}/result.json"

        async def stage() -> SandboxCommand:
            payload = worker_payload(
                task=task,
                invocation=self.invocation,
                model_url=model_url,
                body=body,
                workdir=self.session.workdir,
                tool_access=request.tool_access,
                model_cookies=request.model_cookies,
                max_policy_calls=self.max_policy_calls,
                context_window=self.context_window,
                shell_timeout_seconds=self.shell_timeout_seconds,
                trace={**self.trace, "artifacts_dir": self.artifacts_dir} if self.trace else None,
            )
            await upload_text(self.sandbox, path=inputs, text=json.dumps(payload))
            return SandboxCommand(
                python=self.python, argv=[self.python, "-I", "-m", "nooa_bench.gym_worker", inputs, result]
            )

        async def collect() -> Episode:
            # Runs once inside the session's shared finalization, after the worker is stopped and
            # before the session directory is released, whether run() or a concurrent close() gets
            # there first. Archive first: a stopped or crashed worker's exports and sidecars are
            # evidence for the row even when it left no gradable Episode.
            if self.collects_artifacts:
                await self._collect_artifacts(request.rollout_id)
            # A worker stopped at its budget checkpoints the episode it has (stop reason
            # `timeout_exceeded`) during the supervisor's TERM grace; one that left nothing behind
            # is an infrastructure failure, and the supervisor's receipt says which.
            try:
                episode = Episode.model_validate_json(await read_text(self.sandbox, path=result))
            except Exception as error:
                log = (await self.session.read_output_log())[-4000:]
                if self.session.cleanup["timed_out"]:
                    raise TimeoutError(f"NOOA worker was stopped at its {budget:.0f}s budget; tail: {log}") from error
                raise RuntimeError(f"NOOA worker exited without a result; output log tail: {log}") from error
            # The supervisor's receipt is the authority on budget stops: a `timeout_exceeded` result it
            # does not vouch for (a stop the worker was not asked for, or a forged file) is not gradable.
            if episode.stop_reason == "timeout_exceeded" and not self.session.cleanup["timed_out"]:
                raise RuntimeError(
                    "NOOA worker reported a budget stop, but the supervisor did not stop it at the budget"
                )
            return episode

        try:
            episode = await self.session.execute(
                stage_activation=stage, collect=collect, timeout=budget, close_timeout=self.close_timeout
            )
        except TimeoutError:
            self.failure = "timeout_exceeded"
            raise
        except Exception as error:
            self.failure = f"{type(error).__name__}: {error}"
            raise
        self.episode = episode
        request.resource_cookies.update(episode.resource_cookies)
        if self.artifacts is not None:
            episode.runtime["artifacts_archive"] = str(self.artifacts)
        return self.run_result(episode, body, request.rollout_id, request.model_cookies, request.resource_cookies)

    @property
    def artifacts_dir(self) -> str:
        """Where the worker writes NOOA's exports (ATIF, OTLP JSONL), inside the session directory."""
        return f"{self.directory}/artifacts"

    @property
    def collects_artifacts(self) -> bool:
        return self.artifacts_output is not None and any(k in self.trace for k in ("atif", "trace_jsonl"))

    async def _collect_artifacts(self, rollout_id: str) -> None:
        """Archive NOOA's exports out of the sandbox, one tarball per rollout.

        The archive holds whatever of the worker's evidence exists: its exports directory, the
        Episode it wrote (`result.json`) and every sidecar beside it (`result.json.*`, such as a
        benchmark policy receipt), so the retained evidence is the worker's own files, not a
        projection of them. A failure here is logged and recorded as a gap, never a failed episode:
        the exports are evidence beside the rollout, and Gym's model server already holds the
        training capture.
        """
        assert self.artifacts_output is not None
        remote = f"{self.directory}/artifacts.tar.gz"
        local = Path(self.artifacts_output) / f"{rollout_id}.tar.gz"
        try:
            result = await self.sandbox.exec(
                archive_command(self.python, self.directory, remote), cwd="/", timeout_s=120
            )
            if result.return_code != 0 or result.error_type:
                raise RuntimeError(f"tar exited {result.return_code}: {(result.stderr or '')[-500:]}")
            local.parent.mkdir(parents=True, exist_ok=True)
            await self.sandbox.download(remote, local)
        except Exception as error:
            LOGGER.warning("Could not collect NOOA artifacts for rollout %s: %s", rollout_id, error)
            self.artifacts_error = f"{type(error).__name__}: {error}"
            return
        self.artifacts = local

    @staticmethod
    def run_result(
        episode: Episode,
        body: NeMoGymResponseCreateParamsNonStreaming,
        rollout_id: str,
        model_cookies: dict[str, str],
        resource_cookies: dict[str, str],
    ) -> NOOARunResult:
        """The episode in the result contract the rest of the adapter consumes."""
        reason, error = termination(episode)
        response = set_response_lifecycle(project_response(episode, body, rollout_id), reason, error)
        return NOOARunResult(
            episode=AgentEpisode(response=response, observations=evidence(episode, None)),
            return_value=episode.return_value,
            model_cookies=dict(model_cookies),
            resource_cookies=dict(resource_cookies),
            termination_reason=reason,
            termination_error=error,
            model_status_code=model_status_code(episode),
        )

    async def close(self) -> None:
        """Join shared finalization; a checkpoint the worker left while being stopped is kept as evidence."""
        await self.session.close(timeout=self.close_timeout)
        if self.episode is None and isinstance(self.session.artifacts, Episode):
            self.episode = self.session.artifacts
            if self.artifacts is not None:
                self.episode.runtime["artifacts_archive"] = str(self.artifacts)
