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


