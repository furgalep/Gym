# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Terminal-Bench rows for the thin agent carry each task's canonical budgets as metadata."""

from benchmarks.terminal_bench_2_1.prepare_nooa_thin import task_row
from nemo_gym.agent_utils.task_timeout import resolve_agent_timeout
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.task_materialization import materialize_task


def test_task_row_carries_canonical_budgets_as_request_metadata(tmp_path):
    task_dir = tmp_path / "tasks" / "build-pov-ray"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Build it.")
    (task_dir / "task.toml").write_text(
        '[task]\nname = "build-pov-ray"\n[environment]\ndocker_image = "img:1"\n'
        "[agent]\ntimeout_sec = 12000.0\n[verifier]\ntimeout_sec = 9000\n"
    )
    row = task_row(task_dir, root=tmp_path)
    assert row["task_id"] == "build-pov-ray" and row["task_folder"] == "tasks/build-pov-ray"
    # The legacy environment materializes the flat row and hands responses_create_params to the agent.
    task = materialize_task(row, taskset="tb")
    params = NeMoGymResponseCreateParamsNonStreaming.model_validate(task["task_input"]["responses_create_params"])
    assert params.metadata == {"agent_timeout_sec": "12000", "verifier_timeout_sec": "9000"}
    # The thin agent caps its worker by the task budget, under the configured ceiling.
    assert resolve_agent_timeout(params.metadata, ceiling=12000) == 12000
    assert resolve_agent_timeout({"agent_timeout_sec": "900"}, ceiling=12000) == 900
