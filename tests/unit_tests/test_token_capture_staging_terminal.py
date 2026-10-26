# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Heuristic terminal-call selection over token-free manifest rows."""

from __future__ import annotations

from nemo_gym.token_id_capture.staging.digest import EMPTY_EXTRAS_DIGEST
from nemo_gym.token_id_capture.staging.records import TERMINAL_AMBIGUOUS as AMBIGUOUS_TERMINAL
from nemo_gym.token_id_capture.staging.records import TERMINAL_ORPHANED_ROW as ORPHANED_ROW
from nemo_gym.token_id_capture.staging.records import CallRecord
from nemo_gym.token_id_capture.staging.terminal import (
    BranchSelection,
    TerminalSelection,
    select_branch_terminals,
    select_terminal_call,
)


DIGEST = "b" * 64


def _row(
    model_call_id: str,
    *,
    parent_call_id: str | None = None,
    prev_len: int = 0,
    admitted_at: float | None = None,
) -> CallRecord:
    return CallRecord(
        model_call_id=model_call_id,
        parent_call_id=parent_call_id,
        prev_len=prev_len,
        delta_len=10,
        cum_len=prev_len + 10,
        weight_version=1,
        digest=DIGEST,
        extras_digest=EMPTY_EXTRAS_DIGEST,
        staging_key=f"r1/{model_call_id}",
        mode="text" if parent_call_id is None else "token_in",
        chain_hash=DIGEST,
        cumulative_hash=DIGEST,
        response_id=f"chatcmpl-{model_call_id}",
        admitted_at=admitted_at,
    )


def _chain(*call_ids: str, start_at: float = 100.0) -> list[CallRecord]:
    rows = []
    parent = None
    prev_len = 0
    for offset, call_id in enumerate(call_ids):
        rows.append(_row(call_id, parent_call_id=parent, prev_len=prev_len, admitted_at=start_at + offset))
        parent = call_id
        prev_len += 10
    return rows


def test_empty_manifest_selects_nothing():
    assert select_terminal_call([]) == TerminalSelection(None, "no_records")


def test_single_linear_chain_selects_the_leaf():
    rows = _chain("c1", "c2", "c3")
    assert select_terminal_call(rows) == TerminalSelection("c3", "selected")


def test_abandoned_mid_rollout_retry_is_eliminated():
    rows = _chain("c1", "c2", "c3")
    # A retry sibling of c2 that the harness abandoned: same parent, no child.
    rows.append(_row("c2-retry", parent_call_id="c1", prev_len=10, admitted_at=101.5))
    assert select_terminal_call(rows) == TerminalSelection("c3", "selected")


def test_retry_of_the_final_call_is_ambiguous():
    rows = [
        _row("c1", admitted_at=100.0),
        _row("c2", parent_call_id="c1", prev_len=10, admitted_at=101.0),
        _row("c2-retry", parent_call_id="c1", prev_len=10, admitted_at=101.5),
    ]
    assert select_terminal_call(rows) == TerminalSelection(None, "ambiguous_terminal")


def test_two_abandoned_siblings_under_the_last_fork_are_ambiguous():
    rows = _chain("c1", "c2", "c3")
    rows.append(_row("c4-a", parent_call_id="c3", prev_len=30, admitted_at=104.0))
    rows.append(_row("c4-b", parent_call_id="c3", prev_len=30, admitted_at=105.0))
    assert select_terminal_call(rows) == TerminalSelection(None, "ambiguous_terminal")


def test_two_roots_earliest_admission_wins():
    main = _chain("m1", "m2", start_at=100.0)
    aux = _chain("a1", "a2", start_at=200.0)
    assert select_terminal_call(main + aux) == TerminalSelection("m2", "selected")
    # Order in the manifest does not matter; the timestamps do.
    assert select_terminal_call(aux + main) == TerminalSelection("m2", "selected")


def test_extended_root_beats_an_abandoned_earlier_root():
    abandoned_root = [_row("r0", admitted_at=50.0)]
    main = _chain("m1", "m2", start_at=100.0)
    assert select_terminal_call(abandoned_root + main) == TerminalSelection("m2", "selected")


def test_root_timestamp_tie_breaks_by_manifest_order():
    first = _chain("m1", "m2", start_at=100.0)
    second = _chain("n1", "n2", start_at=100.0)
    assert select_terminal_call(first + second) == TerminalSelection("m2", "selected")
    assert select_terminal_call(second + first) == TerminalSelection("n2", "selected")


def test_all_unstamped_rows_fall_back_to_manifest_order():
    main = [_row("m1"), _row("m2", parent_call_id="m1", prev_len=10)]
    aux = [_row("a1"), _row("a2", parent_call_id="a1", prev_len=10)]
    assert select_terminal_call(main + aux) == TerminalSelection("m2", "selected")


def test_mixed_stamped_and_unstamped_roots_prefer_the_stamped_root():
    unstamped = [_row("u1")]
    stamped = _chain("s1", start_at=999.0)
    # The unstamped root sorts last even though it appears first.
    assert select_terminal_call(unstamped + stamped) == TerminalSelection("s1", "selected")


def test_deep_chain_with_dead_side_branch_selects_the_main_leaf():
    rows = _chain("c1", "c2", "c3", "c4")
    rows.append(_row("side", parent_call_id="c2", prev_len=20, admitted_at=150.0))
    assert select_terminal_call(rows) == TerminalSelection("c4", "selected")


def test_two_extended_branches_are_ambiguous():
    rows = _chain("c1", "c2")
    rows.append(_row("d2", parent_call_id="c1", prev_len=10, admitted_at=101.5))
    rows.append(_row("d3", parent_call_id="d2", prev_len=20, admitted_at=102.5))
    rows.append(_row("c3", parent_call_id="c2", prev_len=20, admitted_at=102.0))
    assert select_terminal_call(rows) == TerminalSelection(None, "ambiguous_terminal")


def test_orphaned_row_selects_nothing():
    rows = _chain("c1", "c2")
    rows.append(_row("lost", parent_call_id="ghost", prev_len=10, admitted_at=103.0))
    assert select_terminal_call(rows) == TerminalSelection(None, "orphaned_row")


def test_cyclic_rows_select_nothing():
    rows = [
        _row("c1", parent_call_id="c2", prev_len=10),
        _row("c2", parent_call_id="c1", prev_len=10),
    ]
    assert select_terminal_call(rows) == TerminalSelection(None, "no_root")


def test_duplicate_call_ids_select_nothing():
    rows = _chain("c1", "c2") + _chain("c1", start_at=300.0)
    assert select_terminal_call(rows) == TerminalSelection(None, "duplicate_call_id")


# --- select_branch_terminals ------------------------------------------------------


def test_branch_terminals_name_the_leaf_of_each_root_beside_the_terminal_chain():
    rows = _chain("root", "main") + _chain("sub1", "sub2", "sub3", start_at=200.0) + _chain("aux", start_at=50.0)
    selection = select_branch_terminals(rows, "main")
    # Roots are visited in admission order; the terminal chain's own root is not a branch.
    assert selection.terminal_model_call_ids == ["aux", "sub3"]
    assert selection.excluded == {}


def test_a_fork_off_the_terminal_root_is_not_a_branch():
    rows = _chain("root", "main") + [_row("fork", parent_call_id="root", prev_len=10, admitted_at=101.5)]
    selection = select_branch_terminals(rows, "main")
    assert selection.terminal_model_call_ids == [] and selection.excluded == {}


def test_an_abandoned_retry_inside_a_branch_is_eliminated_but_a_final_retry_is_ambiguous():
    rows = _chain("root", "main") + _chain("sub1", "sub2", "sub3", start_at=200.0)
    rows.append(_row("sub2-retry", parent_call_id="sub1", prev_len=10, admitted_at=201.5))
    assert select_branch_terminals(rows, "main").terminal_model_call_ids == ["sub3"]

    rows.append(_row("sub3-retry", parent_call_id="sub2", prev_len=20, admitted_at=202.5))
    selection = select_branch_terminals(rows, "main")
    assert selection.terminal_model_call_ids == []
    assert selection.excluded == {AMBIGUOUS_TERMINAL: 1}


def test_branch_selection_fails_closed_on_a_malformed_manifest():
    rows = _chain("root", "main") + [_row("orphan", parent_call_id="ghost", prev_len=10)]
    assert select_branch_terminals(rows, "main") == BranchSelection([], {ORPHANED_ROW: 1})
    assert select_branch_terminals(_chain("root", "main"), "ghost") == BranchSelection([], {"missing_terminal": 1})
    assert select_branch_terminals([], "main") == BranchSelection([], {"missing_terminal": 1})
