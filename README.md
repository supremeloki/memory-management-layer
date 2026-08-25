# memory-layer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A three-tier agent memory system: working window, episodic store with importance eviction, and semantic long-term memory with access-based promotion — the cognitive memory model for agents that remember.

## 🚀 Overview

Flat key-value stores don't model how agents should remember. `memory-layer` implements the three-tier cognitive pattern: a bounded **working** deque for the current context, an **episodic** dict evicted by importance-then-age, and a **semantic** tier ranked by retrieval score. Memories recalled repeatedly (3+ accesses) are automatically promoted episodic→semantic with boosted importance. A pluggable clock makes all time-dependent behavior deterministic under test.

## ✨ Features

- **Working window:** fixed-capacity deque; oldest silently drops
- **Episodic tier:** capacity-bounded; weakest `(importance, age)` evicted first
- **Semantic tier:** evicted by lowest retrieval score (importance × access × recency)
- **Automatic promotion:** 3 accesses on an episodic entry consolidates it into semantic
- **Access tracking:** every recall refreshes `last_accessed_at` and increments `access_count`
- **Prefix search:** deduplicated hits across all tiers, ranked by retrieval score
- **Consolidation pass:** moves low-importance working entries to episodic in bulk
- **Injectable clock:** fully deterministic tests via `FakeClock`
- **Zero dependencies**

## 🚧 Structure

```
memory-management-layer/
├── src/memory_layer/
│   ├── __init__.py
│   └── core.py
├── tests/
│   └── test_core.py
├── README.md
└── pyproject.toml
```

## 📦 Installation

```bash
git clone https://github.com/supremeloki/memory-management-layer.git
cd memory-management-layer
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## 📋 Requirements

- Python 3.11+
- No runtime dependencies

## 🏃 Quick Start

```python
from memory_layer import MemoryManager, MemoryTier

manager = MemoryManager(working_capacity=8, episodic_capacity=100,
                        semantic_capacity=500)
manager.store("user:name", "Kooroush", importance=2.0)
manager.store("session:step1", "searched docs", tier=MemoryTier.EPISODIC)

print(manager.recall("user:name").content)
print(manager.search_by_prefix("session:"))
print(manager.size_report)
```

## 🔧 Error Handling

```text
MemoryLayerError   # invalid capacities at construction
EmptyMemoryError   # recall on a key that was never stored / already evicted
```

## 🧪 Testing

```bash
pytest tests/ -v
```

## 📝 Code Quality

- Full type hints (`X | None` style), frozen entries/windows/reports
- Zero comments — names carry the meaning
- Promotion, eviction ordering, and recency updates all explicitly tested against a fake clock

## 📄 License

MIT — see [LICENSE](LICENSE).

## 👤 Author

**Kooroush Masoumi** - [kooroushmasoumi@gmail.com](mailto:kooroushmasoumi@gmail.com)

---

⭐ Star this repo if you find it useful!
