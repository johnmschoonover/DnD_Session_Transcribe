from __future__ import annotations

import pytest

from dnd_session_transcribe.features.decision_bookmarks import (
    BookmarkKind,
    BookmarkStatus,
    DecisionBookmarkingWorkflow,
    TaskStatus,
)


def test_bookmark_creation_and_resolution():
    workflow = DecisionBookmarkingWorkflow()
    bookmark_id = workflow.create_bookmark(
        42.0,
        "Decide who keeps the wand",
        "Party needs to vote next session",
        kind=BookmarkKind.DECISION,
        tags=("loot", "vote"),
    )

    bookmark_entries = dict(workflow.iter_bookmarks())
    assert bookmark_id in bookmark_entries
    assert bookmark_entries[bookmark_id].status == BookmarkStatus.OPEN
    workflow.resolve_bookmark(bookmark_id)
    assert bookmark_entries[bookmark_id].status == BookmarkStatus.RESOLVED


def test_promote_bookmark_to_task_and_roll_forward():
    workflow = DecisionBookmarkingWorkflow()
    bookmark_id = workflow.create_bookmark(12.0, "Follow up with NPC", "Ask for the missing map", kind=BookmarkKind.NPC_THREAD)
    task_id = workflow.promote_to_task(bookmark_id)

    tasks = dict(workflow.iter_tasks())
    assert task_id in tasks
    assert tasks[task_id].source_bookmark == bookmark_id

    workflow.update_task_status(task_id, TaskStatus.DEFERRED)
    carry_over = workflow.roll_forward_agenda()
    assert carry_over[0].identifier == task_id
    assert carry_over[0].carried_forward is True


def test_promoting_unknown_bookmark_raises_error():
    workflow = DecisionBookmarkingWorkflow()
    with pytest.raises(KeyError):
        workflow.promote_to_task("missing")

