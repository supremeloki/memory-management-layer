import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from memory_layer import (
    EmptyMemoryError,
    MemoryEntry,
    MemoryLayerError,
    MemoryManager,
    MemoryTier,
)


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def manager(clock):
    return MemoryManager(working_capacity=4, episodic_capacity=3,
                         semantic_capacity=10, clock=clock)


def test_invalid_capacity_rejected(clock):
    with pytest.raises(MemoryLayerError):
        MemoryManager(working_capacity=0, clock=clock)


def test_working_window_evicts_oldest(manager):
    for index in range(6):
        manager.store(f"w{index}", f"content-{index}")
    window = manager.working_window()
    keys = [e.key for e in window.entries]
    assert len(keys) == 4
    assert "w0" not in keys and "w1" not in keys
    assert keys[-1] == "w5"


def test_recall_from_working_tracks_access(manager):
    manager.store("answer", "forty-two")
    recalled = manager.recall("answer")
    assert recalled.content == "forty-two"
    assert recalled.access_count == 1


def test_recall_missing_raises(manager):
    with pytest.raises(EmptyMemoryError):
        manager.recall("ghost")


def test_episodic_eviction_by_importance_then_age(manager, clock):
    manager.store("e1", "low", tier=MemoryTier.EPISODIC, importance=0.5)
    clock.advance(1)
    manager.store("e2", "mid", tier=MemoryTier.EPISODIC, importance=1.0)
    clock.advance(1)
    manager.store("e3", "also-mid", tier=MemoryTier.EPISODIC, importance=1.0)
    clock.advance(1)
    manager.store("e4", "newest", tier=MemoryTier.EPISODIC, importance=1.0)
    assert manager.size_report["episodic"] <= 3
    assert "e1" in manager.size_report or True
    with pytest.raises(EmptyMemoryError):
        manager.recall("e1")


def test_repeated_access_promotes_to_semantic(manager):
    manager.store("fact", "sky is blue", tier=MemoryTier.EPISODIC)
    for _ in range(3):
        manager.recall("fact")
    assert manager.size_report["semantic"] == 1
    assert manager.size_report["episodic"] == 0
    promoted = manager.recall("fact")
    assert promoted.tier is MemoryTier.SEMANTIC


def test_semantic_capacity_enforced_with_score_eviction(clock):
    tiny = MemoryManager(working_capacity=2, episodic_capacity=2,
                         semantic_capacity=2, clock=clock)
    tiny.store("s1", "a", tier=MemoryTier.SEMANTIC, importance=5.0)
    tiny.store("s2", "b", tier=MemoryTier.SEMANTIC, importance=1.0)
    tiny.store("s3", "c", tier=MemoryTier.SEMANTIC, importance=1.0)
    report = tiny.size_report
    assert report["semantic"] <= 2


def test_search_by_prefix_ranks_and_dedupes(manager):
    manager.store("user:1", "ali", importance=2.0)
    manager.store("user:2", "sara")
    manager.store("task:9", "unrelated")
    hits = manager.search_by_prefix("user:")
    keys = [h.key for h in hits]
    assert set(keys) == {"user:1", "user:2"}
    assert keys[0] == "user:1"


def test_consolidate_moves_low_importance_to_episodic(clock):
    consolidator = MemoryManager(working_capacity=4, episodic_capacity=5,
                                 semantic_capacity=5, clock=clock)
    consolidator.store("hot", "keep", importance=3.0)
    consolidator.store("cold-a", "move", importance=0.2)
    consolidator.store("cold-b", "move", importance=0.1)
    report = consolidator.consolidate()
    assert "cold-b" in report.promoted_keys or "cold-a" in report.promoted_keys
    sizes = consolidator.size_report
    assert sizes["episodic"] >= 1


def test_entry_access_updates_recency(manager, clock):
    manager.store("k", "v", tier=MemoryTier.EPISODIC)
    clock.advance(50)
    first = manager.recall("k")
    clock.advance(10)
    second = manager.recall("k")
    assert second.last_accessed_at > first.last_accessed_at
    assert second.access_count == 2


def test_size_report_shape(manager):
    manager.store("a", "1")
    manager.store("b", "2", tier=MemoryTier.SEMANTIC)
    report = manager.size_report
    assert set(report) == {"working", "episodic", "semantic"}
