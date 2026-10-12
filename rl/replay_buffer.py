"""Bounded experience replay for (state, search policy, outcome) tuples."""
from __future__ import annotations

from collections import deque
import pickle
from pathlib import Path

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int = 20_000):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.data: deque[tuple[np.ndarray, np.ndarray, float]] = deque(maxlen=capacity)

    def extend(self, examples) -> None:
        for state, policy, value in examples:
            self.data.append((np.asarray(state, dtype=np.float32), np.asarray(policy, dtype=np.float32), float(value)))

    def append(self, example) -> None:
        self.extend([example])

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.data:
            raise ValueError("Cannot sample an empty replay buffer")
        count = min(batch_size, len(self.data))
        indices = np.random.choice(len(self.data), count, replace=False)
        batch = [self.data[int(i)] for i in indices]
        states, policies, values = zip(*batch)
        return np.stack(states), np.stack(policies), np.asarray(values, dtype=np.float32)

    def save(self, path: str | Path) -> None:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump({"capacity": self.capacity, "data": list(self.data)}, handle)

    @classmethod
    def load(cls, path: str | Path) -> "ReplayBuffer":
        with Path(path).open("rb") as handle:
            payload = pickle.load(handle)
        buffer = cls(payload["capacity"])
        buffer.extend(payload["data"])
        return buffer

    def __len__(self) -> int:
        return len(self.data)

