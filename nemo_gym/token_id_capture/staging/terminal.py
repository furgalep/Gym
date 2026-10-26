# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Select a terminal call when the harness did not identify one.

The selector follows ``parent_call_id`` links in the manifest.
It ignores an abandoned leaf when a sibling has descendants.
It returns no result when more than one terminal call remains possible.

This function does not inspect token data.
``verify_and_linearize`` validates the selected chain before training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from nemo_gym.token_id_capture.staging.records import (
    TERMINAL_AMBIGUOUS,
    TERMINAL_DUPLICATE_CALL_ID,
    TERMINAL_NO_RECORDS,
    TERMINAL_NO_ROOT,
    TERMINAL_ORPHANED_ROW,
    TERMINAL_SELECTED,
    CallRecord,
)


# Local aliases: the selection vocabulary lives with the wire schemas.
SELECTED = TERMINAL_SELECTED
NO_RECORDS = TERMINAL_NO_RECORDS
DUPLICATE_CALL_ID = TERMINAL_DUPLICATE_CALL_ID
ORPHANED_ROW = TERMINAL_ORPHANED_ROW
NO_ROOT = TERMINAL_NO_ROOT
AMBIGUOUS_TERMINAL = TERMINAL_AMBIGUOUS


@dataclass(frozen=True)
class TerminalSelection:
    """The inferred terminal call, or the reason none could be chosen."""

    terminal_model_call_id: str | None
    reason: str


@dataclass(frozen=True)
class BranchSelection:
    """The terminal calls of branch chains, and the roots whose chain could not be followed.

    ``terminal_model_call_ids`` lists one leaf per deliverable branch root in root admission
    order. ``excluded`` counts, by reason, the branch roots whose subtree offered no single leaf.
    """

    terminal_model_call_ids: list[str]
    excluded: dict[str, int]


def _root_order_key(records: Sequence[CallRecord]):
    """Order candidate roots by admission time, unstamped rows last.

    ``admitted_at`` is absent on rows written before the column existed and on
    any path that fails to thread it; those rows sort after every stamped row.
    Manifest position breaks exact-timestamp ties (ledger append order is the
    commit order), which also makes the all-unstamped case deterministic.
    """
    index_by_id = {record.model_call_id: position for position, record in enumerate(records)}

    def key(record: CallRecord) -> tuple:
        return (
            record.admitted_at is None,
            record.admitted_at if record.admitted_at is not None else 0.0,
            index_by_id[record.model_call_id],
        )

    return key


def select_terminal_call(records: Sequence[CallRecord]) -> TerminalSelection:
    """Select a terminal call from parent links when the harness did not provide one.

    Records may contain multiple roots and branches.
    A root with children is preferred over roots without children.
    Remaining roots are ordered by admission time and then manifest position.
    At each subsequent level, children with descendants are preferred over childless siblings.
    Selection continues when exactly one preferred child remains.
    If multiple preferred children remain, no terminal call is returned.
    """
    if not records:
        return TerminalSelection(None, NO_RECORDS)

    by_id = {record.model_call_id: record for record in records}
    if len(by_id) != len(records):
        return TerminalSelection(None, DUPLICATE_CALL_ID)
    for record in records:
        if record.parent_call_id is not None and record.parent_call_id not in by_id:
            return TerminalSelection(None, ORPHANED_ROW)

    children: dict[str, list[CallRecord]] = {}
    roots: list[CallRecord] = []
    for record in records:
        if record.parent_call_id is None:
            roots.append(record)
        else:
            children.setdefault(record.parent_call_id, []).append(record)
    if not roots:
        # Every row names a present parent: the graph is cyclic.
        return TerminalSelection(None, NO_ROOT)

    key = _root_order_key(records)

    def survivors(candidates: list[CallRecord]) -> list[CallRecord]:
        extended = [record for record in candidates if children.get(record.model_call_id)]
        return extended or candidates

    root_pool = survivors(roots)
    node = min(root_pool, key=key)
    while True:
        candidates = children.get(node.model_call_id) or []
        if not candidates:
            return TerminalSelection(node.model_call_id, SELECTED)
        pool = survivors(candidates)
        if len(pool) > 1:
            return TerminalSelection(None, AMBIGUOUS_TERMINAL)
        node = pool[0]


def select_branch_terminals(records: Sequence[CallRecord], terminal_model_call_id: str) -> BranchSelection:
    """Select the terminal call of each branch chain beside the terminal chain.

    A branch chain is rooted at a call the terminal chain does not share, which is how a
    sub-agent's calls appear: its first prompt carries no model-authored history, so it is
    admitted as a root. The walk from each such root follows the rule ``select_terminal_call``
    applies, preferring children with descendants over childless siblings and stopping when
    more than one preferred child remains. Roots are visited in admission order.

    A fork off the terminal chain's own root is not a branch chain: the harness did not keep
    its output on the scored path, so it is never selected.
    """
    by_id = {record.model_call_id: record for record in records}
    if len(by_id) != len(records) or terminal_model_call_id not in by_id:
        return BranchSelection([], {DUPLICATE_CALL_ID if len(by_id) != len(records) else "missing_terminal": 1})
    for record in records:
        if record.parent_call_id is not None and record.parent_call_id not in by_id:
            return BranchSelection([], {ORPHANED_ROW: 1})

    children: dict[str, list[CallRecord]] = {}
    roots: list[CallRecord] = []
    for record in records:
        if record.parent_call_id is None:
            roots.append(record)
        else:
            children.setdefault(record.parent_call_id, []).append(record)

    terminal_root: CallRecord | None = by_id[terminal_model_call_id]
    visited: set[str] = set()
    while terminal_root is not None and terminal_root.parent_call_id is not None:
        if terminal_root.model_call_id in visited:
            return BranchSelection([], {NO_ROOT: 1})
        visited.add(terminal_root.model_call_id)
        terminal_root = by_id.get(terminal_root.parent_call_id)
    if terminal_root is None:
        return BranchSelection([], {NO_ROOT: 1})

    def survivors(candidates: list[CallRecord]) -> list[CallRecord]:
        extended = [record for record in candidates if children.get(record.model_call_id)]
        return extended or candidates

    terminals: list[str] = []
    excluded: dict[str, int] = {}
    for root in sorted(roots, key=_root_order_key(records)):
        if root.model_call_id == terminal_root.model_call_id:
            continue
        node = root
        seen: set[str] = set()
        reason = ""
        while True:
            if node.model_call_id in seen:
                reason = NO_ROOT
                break
            seen.add(node.model_call_id)
            candidates = children.get(node.model_call_id) or []
            if not candidates:
                break
            pool = survivors(candidates)
            if len(pool) > 1:
                reason = AMBIGUOUS_TERMINAL
                break
            node = pool[0]
        if reason:
            excluded[reason] = excluded.get(reason, 0) + 1
        else:
            terminals.append(node.model_call_id)
    return BranchSelection(terminals, excluded)
