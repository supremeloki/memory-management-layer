from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable, Sequence


class MemoryLayerError(Exception):
    pass


class EmptyMemoryError(MemoryLayerError):
    pass


class MemoryTier(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class MemoryEntry:
    key: str
    content: str
    tier: MemoryTier
    created_at: float
    importance: float = 1.0
    last_accessed_at: float = 0.0
    access_count: int = 0

    def with_access(self, accessed_at: float) -> "MemoryEntry":
        return MemoryEntry(
            key=self.key,
            content=self.content,
            tier=self.tier,
            created_at=self.created_at,
            importance=self.importance,
            last_accessed_at=accessed_at,
            access_count=self.access_count + 1,
        )

    @property
    def retrieval_score(self) -> float:
        recency_bonus = 1.0 / (1.0 + max(0.0, self.last_accessed_at - self.created_at))
        return self.importance * (1.0 + 0.1 * self.access_count) * (0.5 + recency_bonus)


@dataclass(frozen=True)
class WorkingWindow:
    entries: tuple[MemoryEntry, ...]
    capacity: int

    @property
    def is_full(self) -> bool:
        return len(self.entries) >= self.capacity


@dataclass(frozen=True)
class PromotionReport:
    promoted_keys: tuple[str, ...]
    evicted_keys: tuple[str, ...]


class MemoryManager:
    def __init__(self,
                 working_capacity: int = 8,
                 episodic_capacity: int = 100,
                 semantic_capacity: int = 500,
                 clock: Callable[[], float] | None = None) -> None:
        if working_capacity < 1 or episodic_capacity < 1 or semantic_capacity < 1:
            raise MemoryLayerError("capacities must be >= 1")
        self._clock = clock or time.time
        self._working: deque[MemoryEntry] = deque(maxlen=working_capacity)
        self._episodic: dict[str, MemoryEntry] = {}
        self._semantic: dict[str, MemoryEntry] = {}
        self._episodic_capacity = episodic_capacity
        self._semantic_capacity = semantic_capacity

    @property
    def now(self) -> float:
        return self._clock()

    def store(self, key: str, content: str, tier: MemoryTier = MemoryTier.WORKING,
              importance: float = 1.0) -> MemoryEntry:
        entry = MemoryEntry(
            key=key,
            content=content,
            tier=tier,
            created_at=self.now,
            importance=importance,
        )
