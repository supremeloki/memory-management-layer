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
        if tier is MemoryTier.WORKING:
            self._working.append(entry)
        elif tier is MemoryTier.EPISODIC:
            self._store_episodic(entry)
        else:
            self._store_semantic(entry)
        return entry

    def _store_episodic(self, entry: MemoryEntry) -> None:
        self._episodic[entry.key] = entry
        while len(self._episodic) > self._episodic_capacity:
            weakest = min(
                self._episodic.values(),
                key=lambda e: (e.importance, e.created_at),
            )
            del self._episodic[weakest.key]

    def _store_semantic(self, entry: MemoryEntry) -> None:
        self._semantic[entry.key] = entry
        while len(self._semantic) > self._semantic_capacity:
            weakest = min(
                self._semantic.values(),
                key=lambda e: (e.retrieval_score, -e.access_count),
            )
            del self._semantic[weakest.key]

    def recall(self, key: str) -> MemoryEntry:
        for window_entry in reversed(self._working):
            if window_entry.key == key:
                refreshed = window_entry.with_access(self.now)
                return refreshed
        found = self._episodic.get(key) or self._semantic.get(key)
        if found is None:
            raise EmptyMemoryError(f"no memory under {key!r}")
        refreshed = found.with_access(self.now)
        if found.tier is MemoryTier.EPISODIC and refreshed.access_count >= 3:
            self._promote_to_semantic(refreshed)
            del self._episodic[refreshed.key]
        elif found.tier is MemoryTier.EPISODIC:
            self._episodic[key] = refreshed
        else:
            self._semantic[key] = refreshed
        return refreshed

    def _promote_to_semantic(self, entry: MemoryEntry) -> None:
        consolidated = MemoryEntry(
            key=entry.key,
            content=entry.content,
            tier=MemoryTier.SEMANTIC,
            created_at=entry.created_at,
            importance=max(1.5, entry.importance),
            last_accessed_at=entry.last_accessed_at,
            access_count=entry.access_count,
        )
        self._store_semantic(consolidated)

    def working_window(self) -> WorkingWindow:
        return WorkingWindow(entries=tuple(self._working),
                             capacity=self._working.maxlen or 0)

    def search_by_prefix(self, prefix: str) -> list[MemoryEntry]:
        hits: list[MemoryEntry] = []
        seen: set[str] = set()
        pools: Sequence[Iterable[MemoryEntry]] = (
            reversed(self._working), self._episodic.values(), self._semantic.values(),
        )
        for pool in pools:
            for entry in pool:
                if entry.key.startswith(prefix) and entry.key not in seen:
                    seen.add(entry.key)
                    hits.append(entry)
        hits.sort(key=lambda e: e.retrieval_score, reverse=True)
        return hits

    def consolidate(self) -> PromotionReport:
        promoted: list[str] = []
        evicted: list[str] = []
        overflow = max(
            0, len(self._working) - (self._working.maxlen // 2 if self._working.maxlen else 0)
        )
        for _ in range(min(overflow, 3)):
            candidate = min(
                self._working,
                key=lambda e: (e.importance, -e.created_at),
            )
            promoted.append(candidate.key)
            self.store(candidate.key, candidate.content,
                       tier=MemoryTier.EPISODIC, importance=candidate.importance)
            self._working.remove(candidate)
        while len(self._episodic) > self._episodic_capacity:
            weakest = min(self._episodic.values(), key=lambda e: e.created_at)
            del self._episodic[weakest.key]
            evicted.append(weakest.key)
        return PromotionReport(promoted_keys=tuple(promoted), evicted_keys=tuple(evicted))

    @property
    def size_report(self) -> dict[str, int]:
        return {
            "working": len(self._working),
            "episodic": len(self._episodic),
            "semantic": len(self._semantic),
        }
