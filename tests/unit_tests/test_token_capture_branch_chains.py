# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test branch-chain delivery: a sub-agent's calls ride beside the main chain.

A sub-agent's first prompt carries no model-authored history, so capture admits it as a
root of its own, and its chain never extends the main chain's tokens. The builder tags such
a chain ``independent_root``; the consumer projects it as its own Responses object when
``deliver_branch_chains`` is set and the main chain is attributed; the finalizer puts the
projections on the record under ``_ng_token_capture_branches``. A fork off the main chain's
root, a branch through an ambiguous retry, and a branch rooted at an unresolved fragment are
counted and withheld.
"""

import asyncio
from pathlib import Path

from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.token_id_capture import (
    ParentResolutionStatus,
    TokenCaptureStore,
    TokenEntry,
    TokenIdCaptureConfig,
    deliver_branch_chains_from_config,
    project_branch_chain_responses,
    stamp_lineage,
)
from nemo_gym.token_id_capture.builder import assert_prefix_contiguity, run_builder
from nemo_gym.token_id_capture.consumer import _assemble
from nemo_gym.token_id_capture.delivery import (
    BRANCH_RESPONSES_KEY,
    MASK_SAMPLE_KEY,
    TOKEN_CAPTURE_KEY,
    finalize_rollout_token_capture,
)


def _assistant_item(text: str) -> dict:
    return {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _entry(
    call_id: str,
    prompt: list[int],
    gen: list[int],
    *,
    text: str,
    parent_call_id: str | None = None,
    unresolved: bool = False,
    created_at: float = 0.0,
    **extra,
) -> TokenEntry:
    entry = TokenEntry(
        rollout_id="r",
        model_call_id=call_id,
        prompt_token_ids=prompt,
        generation_token_ids=gen,
        generation_log_probs=[-0.1] * len(gen),
        output_items=[_assistant_item(text)],
        token_item_index=0,
        created_at=created_at,
        **extra,
    )
    if unresolved:
        status = ParentResolutionStatus.UNRESOLVED
    elif parent_call_id:
        status = ParentResolutionStatus.RESOLVED
    else:
        status = ParentResolutionStatus.ROOT
    stamp_lineage(entry, parent_call_id, parent_resolution=status)
    return entry


def _main_entries() -> list[TokenEntry]:
    """Two chained calls on the main agent; ``call2`` is the terminal."""
    return [
        _entry("call1", [1, 2], [3, 4], text="step one", response_id="resp_1", created_at=1.0),
        _entry(
            "call2",
            [1, 2, 3, 4, 5, 6],
            [7, 8],
            text="final answer",
            response_id="resp_2",
            created_at=4.0,
            parent_call_id="call1",
        ),
    ]


def _sub_agent_entries(finish_reason: str | None = None) -> list[TokenEntry]:
    """A sub-agent's two chained calls: its root prompt shares nothing with the main chain."""
    return [
        _entry("sub1", [100, 101], [102], text="sub step", created_at=2.0),
        _entry(
            "sub2",
            [100, 101, 102, 103],
            [104, 105],
            text="sub result",
            created_at=3.0,
            parent_call_id="sub1",
            finish_reason=finish_reason,
        ),
    ]


def _fork_entry() -> TokenEntry:
    """A continuation of ``call1`` the harness did not keep: it leaves the main path."""
    return _entry("call3", [1, 2, 3, 4, 9], [10], text="abandoned", created_at=2.5, parent_call_id="call1")


# --- builder ------------------------------------------------------------------------


def test_builder_tags_a_sub_agent_chain_as_an_independent_root_and_a_fork_as_withheld():
    out = run_builder(_main_entries() + _sub_agent_entries() + [_fork_entry()], terminal_call_id="call2")

    assert out.notes.terminal_chain == "delivered"
    by_relation = {chain.relation: chain for chain in out.chains}
    assert set(by_relation) == {"main", "independent_root", "fork"}
    main = by_relation["main"]
    assert (main.chain_id, main.exclusion, main.root_call_id, main.terminal_call_id) == ("main", "", "call1", "call2")
    sub = by_relation["independent_root"]
    assert (sub.exclusion, sub.root_call_id, sub.terminal_call_id) == ("", "sub1", "sub2")
    assert sub.chain_id.startswith("branch-")
    fork = by_relation["fork"]
    assert (fork.exclusion, fork.root_call_id, fork.terminal_call_id) == ("fork", "call1", "call3")

    assert out.notes.branch_chains == {"independent_root": 1, "fork": 1}
    assert out.notes.branches_excluded == {"fork": 1}
    assert out.notes.branch_generated_tokens == 3
    # ``response.output`` still carries only the main chain, and the fraction says so.
    assert out.notes.generated_tokens_delivered == 4
    assert out.notes.delivered_fraction == round(4 / 8, 4)


def test_builder_withholds_a_branch_kept_over_a_retry_twin_by_call_id_alone():
    twins = _sub_agent_entries() + [
        _entry("sub2b", [100, 101, 102, 103], [106, 107], text="sub result again", parent_call_id="sub1"),
    ]
    out = run_builder(_main_entries() + twins, terminal_call_id="call2")

    # The retry group is off the terminal path, so it does not mask the rollout (unchanged).
    assert out.notes.terminal_chain == "delivered"
    assert out.notes.unresolved_retries == []
    assert out.quarantined == ["sub2b"]
    [branch] = [chain for chain in out.chains if chain.relation != "main"]
    assert branch.exclusion == "retry_ambiguous"
    assert out.notes.branches_excluded == {"retry_ambiguous": 1}
    assert out.notes.branch_generated_tokens == 0


def test_builder_withholds_a_branch_rooted_at_an_unresolved_fragment():
    fragment = _entry("frag", [200, 201, 202], [203], text="after a lost parent", unresolved=True)
    out = run_builder(_main_entries() + [fragment], terminal_call_id="call2")

    [branch] = [chain for chain in out.chains if chain.relation != "main"]
    assert (branch.relation, branch.exclusion) == ("independent_root", "unresolved_root")
    assert out.notes.branches_excluded == {"unresolved_root": 1}
    assert out.notes.unresolved_parent_calls == ["frag"]


def test_builder_leaves_a_single_chain_rollout_without_branches():
    out = run_builder(_main_entries(), terminal_call_id="call2")
    assert out.notes.branch_chains == {} and out.notes.branches_excluded == {}
    assert out.notes.branch_generated_tokens == 0
    assert project_branch_chain_responses("r", out) == []


# --- projection ---------------------------------------------------------------------


def test_project_branch_chain_responses_projects_only_deliverable_branches():
    out = run_builder(
        _main_entries() + _sub_agent_entries(finish_reason="length") + [_fork_entry()], terminal_call_id="call2"
    )

    [branch] = project_branch_chain_responses("r", out, model="m")
    assert_prefix_contiguity(branch)
    assert branch["id"].startswith("proj-r-branch-") and branch["object"] == "response" and branch["model"] == "m"
    assert branch["_ng_chain"] == {
        "chain_id": branch["id"].removeprefix("proj-r-"),
        "relation": "independent_root",
        "root_call_id": "sub1",
        "terminal_call_id": "sub2",
        "n_calls": 2,
    }
    generated = [item for item in branch["output"] if item.get("generation_token_ids") is not None]
    assert [item["prompt_token_ids"] for item in generated] == [[100, 101], [100, 101, 102, 103]]
    assert [item["generation_token_ids"] for item in generated] == [[102], [104, 105]]
    assert [item["generation_log_probs"] for item in generated] == [[-0.1], [-0.1, -0.1]]
    assert branch["usage"] == {"input_tokens": 2, "output_tokens": 3}
    # A branch's truncation verdict is its own terminal call's finish reason.
    assert branch["status"] == "incomplete" and branch["incomplete_details"] == {"reason": "max_output_tokens"}


# --- consumer -----------------------------------------------------------------------


def _scored_response() -> dict:
    return {
        "id": "resp_2",
        "model": "m",
        "object": "response",
        "output": [_assistant_item("step one"), _assistant_item("final answer")],
    }


def test_assemble_counts_branches_always_and_delivers_them_only_when_asked():
    entries = _main_entries() + _sub_agent_entries() + [_fork_entry()]

    built = _assemble("r", entries, "prefix_merging", "m", verified_response=_scored_response())
    assert built[MASK_SAMPLE_KEY] is False
    assert built["branch_responses"] == []
    assert built["metrics"]["branch_chains"] == {
        "by_relation": {"independent_root": 1, "fork": 1},
        "excluded": {"fork": 1},
        "generated_tokens": 3,
        "delivered": 0,
    }

    built = _assemble(
        "r", entries, "prefix_merging", "m", verified_response=_scored_response(), deliver_branch_chains=True
    )
    assert built[MASK_SAMPLE_KEY] is False
    [branch] = built["branch_responses"]
    assert branch["_ng_chain"]["root_call_id"] == "sub1"
    assert built["metrics"]["branch_chains"]["delivered"] == 1
    # The main response is the verified chain alone.
    delivered = [
        i["generation_token_ids"] for i in built["rebuilt_response"]["output"] if i.get("generation_token_ids")
    ]
    assert delivered == [[3, 4], [7, 8]]


def test_assemble_delivers_no_branches_without_terminal_attribution():
    # Two roots and no witness: the strict single-chain policy masks the rollout, and a branch
    # of a masked rollout has nothing to ride beside.
    entries = _main_entries() + _sub_agent_entries()
    built = _assemble("r", entries, "prefix_merging", "m", verified_response=None, deliver_branch_chains=True)
    assert built[MASK_SAMPLE_KEY] is True
    assert built["branch_responses"] == []
    assert built["metrics"]["branch_chains"]["by_relation"] == {"independent_root": 1}
    assert built["metrics"]["branch_chains"]["delivered"] == 0


# --- delivery -----------------------------------------------------------------------


def _record() -> dict:
    return {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": _scored_response()}


def _store_with_sub_agent(tmp_path: Path) -> TokenCaptureStore:
    store = TokenCaptureStore(tmp_path)
    for entry in _main_entries() + _sub_agent_entries():
        store.append(entry.model_copy(update={"rollout_id": "0-0"}))
    return store


def test_finalize_puts_branch_responses_on_the_record_when_configured(tmp_path: Path):
    store = _store_with_sub_agent(tmp_path)

    result = _record()
    built = asyncio.run(finalize_rollout_token_capture(result, store))
    assert built is not None and built[MASK_SAMPLE_KEY] is False
    assert BRANCH_RESPONSES_KEY not in result
    assert result[TOKEN_CAPTURE_KEY]["branch_chains"]["delivered"] == 0

    result = _record()
    built = asyncio.run(finalize_rollout_token_capture(result, store, deliver_branch_chains=True))
    assert built is not None and built[MASK_SAMPLE_KEY] is False
    [branch] = result[BRANCH_RESPONSES_KEY]
    assert branch["_ng_chain"]["terminal_call_id"] == "sub2"
    assert_prefix_contiguity(branch)
    assert result[TOKEN_CAPTURE_KEY]["branch_chains"]["delivered"] == 1
    assert result[TOKEN_CAPTURE_KEY]["terminal_attribution"]["chain"] == "delivered"
    # ``response.output`` is unchanged by the branches: it carries the main chain alone.
    main_tokens = [i["generation_token_ids"] for i in result["response"]["output"] if i.get("generation_token_ids")]
    assert main_tokens == [[3, 4], [7, 8]]
    assert result["reward"] == 1.0


def test_finalize_keeps_branches_off_a_masked_record(tmp_path: Path):
    store = _store_with_sub_agent(tmp_path)
    # An uncommitted call could be on any chain, so an incomplete snapshot masks and delivers nothing.
    asyncio.run(store.mark_incomplete("0-0", "lost"))

    result = _record()
    built = asyncio.run(finalize_rollout_token_capture(result, store, deliver_branch_chains=True))
    assert built is not None and built[MASK_SAMPLE_KEY] is True
    assert result[MASK_SAMPLE_KEY] is True
    assert BRANCH_RESPONSES_KEY not in result
    assert built["branch_responses"] == []
    assert result[TOKEN_CAPTURE_KEY]["branch_chains"]["delivered"] == 0


# --- configuration ------------------------------------------------------------------


def test_deliver_branch_chains_setting_defaults_off_and_reads_from_config(tmp_path: Path):
    base = {"token_id_capture": {"enabled": True, "dir": str(tmp_path)}}
    assert TokenIdCaptureConfig.model_validate(base).token_id_capture.deliver_branch_chains is False
    assert deliver_branch_chains_from_config(base) is False
    enabled = {"token_id_capture": {**base["token_id_capture"], "deliver_branch_chains": True}}
    assert deliver_branch_chains_from_config(enabled) is True
