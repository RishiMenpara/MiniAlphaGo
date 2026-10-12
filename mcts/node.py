"""Tree node and edge statistics used by PUCT."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    prior: float = 0.0
    to_play: int = 1
    parent: "MCTSNode | None" = None
    action: int | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "MCTSNode"] = field(default_factory=dict)

    @property
    def q_value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def expanded(self) -> bool:
        return bool(self.children)

