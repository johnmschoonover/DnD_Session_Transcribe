"""Bookmark and task-tracking helpers for long-form session reviews."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Iterator, List, MutableMapping, Optional


class BookmarkKind(str, Enum):
    """Types of bookmarks a DM can drop while listening back to a session."""

    DECISION = "decision"
    LOOT = "loot"
    FORESHADOW = "foreshadow"
    NPC_THREAD = "npc_thread"


class BookmarkStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"


class TaskStatus(str, Enum):
    OPEN = "open"
    COMPLETED = "completed"
    DEFERRED = "deferred"


@dataclass(slots=True)
class Bookmark:
    timestamp: float
    title: str
    note: str
    kind: BookmarkKind = BookmarkKind.DECISION
    status: BookmarkStatus = BookmarkStatus.OPEN
    tags: tuple[str, ...] = ()


@dataclass(slots=True)
class Task:
    identifier: str
    title: str
    detail: str
    source_bookmark: str
    status: TaskStatus = TaskStatus.OPEN
    carried_forward: bool = False


class DecisionBookmarkingWorkflow:
    """Capture bookmarks and promote them into actionable tasks."""

    def __init__(self) -> None:
        self._bookmarks: MutableMapping[str, Bookmark] = {}
        self._tasks: MutableMapping[str, Task] = {}
        self._bookmark_counter = itertools.count(1)
        self._task_counter = itertools.count(1)

    def create_bookmark(
        self,
        timestamp: float,
        title: str,
        note: str,
        *,
        kind: BookmarkKind = BookmarkKind.DECISION,
        tags: Iterable[str] | None = None,
    ) -> str:
        bookmark_id = f"bm-{next(self._bookmark_counter)}"
        bookmark = Bookmark(
            timestamp=timestamp,
            title=title,
            note=note,
            kind=kind,
            tags=tuple(tag.lower() for tag in (tags or ()))
            if tags
            else (),
        )
        self._bookmarks[bookmark_id] = bookmark
        return bookmark_id

    def resolve_bookmark(self, bookmark_id: str) -> None:
        bookmark = self._bookmarks.get(bookmark_id)
        if bookmark is None:
            raise KeyError(f"Unknown bookmark id: {bookmark_id}")
        bookmark.status = BookmarkStatus.RESOLVED

    def promote_to_task(
        self,
        bookmark_id: str,
        *,
        title: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> str:
        bookmark = self._bookmarks.get(bookmark_id)
        if bookmark is None:
            raise KeyError(f"Unknown bookmark id: {bookmark_id}")

        task_id = f"task-{next(self._task_counter)}"
        task = Task(
            identifier=task_id,
            title=title or bookmark.title,
            detail=detail or bookmark.note,
            source_bookmark=bookmark_id,
        )
        self._tasks[task_id] = task
        return task_id

    def update_task_status(self, task_id: str, status: TaskStatus) -> None:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Unknown task id: {task_id}")
        task.status = status

    def roll_forward_agenda(self) -> list[Task]:
        """Return tasks that should carry into the next session."""

        carry_over: List[Task] = []
        for task in self._tasks.values():
            if task.status in {TaskStatus.OPEN, TaskStatus.DEFERRED}:
                task.carried_forward = True
                carry_over.append(task)
        return sorted(carry_over, key=lambda task: task.identifier)

    def iter_bookmarks(self) -> Iterator[tuple[str, Bookmark]]:
        return iter(self._bookmarks.items())

    def iter_tasks(self) -> Iterator[tuple[str, Task]]:
        return iter(self._tasks.items())


__all__ = [
    "Bookmark",
    "BookmarkKind",
    "BookmarkStatus",
    "DecisionBookmarkingWorkflow",
    "Task",
    "TaskStatus",
]

