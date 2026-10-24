# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare the 89 pinned Terminal-Bench 2.1 tasks for the thin NOOA agent.

Same flat rows as ``prepare.py`` (so ``resources_servers/terminal_bench_2_1`` reads them), plus
each task's canonical budgets from ``task.toml`` as request metadata: ``agent_timeout_sec`` is
what the thin agent caps its worker by (``resolve_agent_timeout``), ``verifier_timeout_sec`` is
recorded for the manifest. The task checkout is pinned to the revision the thick NOOA runs used.
"""

from __future__ import annotations

import json
import math
import tomllib
from pathlib import Path
from subprocess import run


BENCHMARK_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BENCHMARK_DIR / "data" / "nooa_thin.jsonl"
TASK_REPOSITORY = "https://github.com/harbor-framework/terminal-bench-2-1"
TASK_REVISION = "7131e4375048a0e408a8fb404b5f499d726b695b"  # pragma: allowlist secret
TASK_COUNT = 89


def task_row(task_dir: Path, *, root: Path) -> dict:
    """One flat dataset row; ``root`` is the Gym checkout the task folder is made relative to."""
    with (task_dir / "task.toml").open("rb") as reader:
        task = tomllib.load(reader)
    budgets: dict[str, str] = {}
    for section in ("agent", "verifier"):
        value = task[section]["timeout_sec"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid {section} timeout in {task_dir}")
        budgets[f"{section}_timeout_sec"] = str(int(value)) if float(value).is_integer() else str(float(value))
    return {
        "responses_create_params": {
            "input": [{"role": "user", "content": (task_dir / "instruction.md").read_text()}],
            # Responses metadata values are strings.
            "metadata": budgets,
        },
        "task_id": task_dir.name,
        "task_name": task["task"]["name"],
        "docker_image": task["environment"]["docker_image"],
        "task_folder": str(task_dir.relative_to(root)),
    }


def prepare() -> Path:
    repo = BENCHMARK_DIR / "terminal-bench-2-1"
    if not repo.exists():
        run(["git", "clone", TASK_REPOSITORY, str(repo)], check=True)
    head = run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != TASK_REVISION:
        run(["git", "-C", str(repo), "fetch", "--quiet", "origin", TASK_REVISION], check=True)
        run(["git", "-C", str(repo), "checkout", "--quiet", TASK_REVISION], check=True)
    root = BENCHMARK_DIR.parents[1]
    rows = [task_row(task_dir, root=root) for task_dir in sorted((repo / "tasks").iterdir()) if task_dir.is_dir()]
    if len(rows) != TASK_COUNT or len({row["task_id"] for row in rows}) != TASK_COUNT:
        raise ValueError(f"expected {TASK_COUNT} unique tasks, found {len(rows)}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w") as writer:
        for row in rows:
            writer.write(json.dumps(row) + "\n")
    return OUTPUT_PATH


if __name__ == "__main__":
    print(prepare())
