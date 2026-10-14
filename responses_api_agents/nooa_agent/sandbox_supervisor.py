# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the NOOA worker under Gym's unchanged Linux descendant supervisor."""

import json
import math
import os
import sys
from pathlib import Path

from nemo_gym.agent_utils import process_supervisor


def supervise(directory: Path, *, retain_services: bool = False) -> int:
    """Keep one episode clock and publish the shared supervisor's cleanup receipt.

    This is the sole dependency on Gym's private ``_supervise`` helper. Its CLI
    requires a finite deadline; infinity here leaves cancellation to the NOOA
    environment without changing the shared supervisor or copying its reaper.

    ``retain_services`` publishes ``completion.json`` when the worker exits and,
    only after a successful exit, keeps the subreaper (and so the agent's detached
    task services) alive until ``runner.stop`` appears or SIGTERM arrives.
    """
    extra = {"completion_path": directory / "completion.json"} if retain_services else {}
    (directory / "runner.pid").write_text(str(os.getpid()))
    receipt = process_supervisor._supervise(
        [
            sys.executable,
            "-I",
            "-m",
            "responses_api_agents.nooa_agent.sandbox_entrypoint",
            str(directory / "input.json"),
            str(directory / "result.json"),
        ],
        timeout=math.inf,
        cleanup_timeout=5,
        stop_path=directory / "runner.stop",
        **extra,
    )
    temporary = directory / "cleanup.tmp"
    temporary.write_text(json.dumps(receipt))
    temporary.replace(directory / "cleanup.json")
    return 0 if receipt["cleanup_confirmed"] and receipt["error"] is None else 1


if __name__ == "__main__":
    raise SystemExit(supervise(Path(sys.argv[1]), retain_services="--retain-services" in sys.argv[2:]))
