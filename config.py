"""Central configuration for MiniAlphaGo."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch


@dataclass
class Config:
    board_size: int = 5
    action_size: int = 26
    komi: float = 2.5
    num_channels: int = 64
    num_res_blocks: int = 2
    c_puct: float = 1.5
    mcts_simulations: int = 50
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    replay_buffer_size: int = 20_000
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    iterations: int = 10
    self_play_games: int = 25
    training_steps: int = 200
    evaluation_games: int = 20
    promotion_threshold: float = 0.55
    temperature: float = 1.0
    temperature_drop_move: int = 10
    max_game_length: int = 200
    random_seed: int = 42
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def validate(self) -> None:
        if self.board_size != 5 or self.action_size != self.board_size**2 + 1:
            raise ValueError("This project requires a 5x5 board and 26 actions")
        if self.mcts_simulations < 1 or self.batch_size < 1:
            raise ValueError("MCTS simulations and batch size must be positive")
        if not 0 <= self.dirichlet_epsilon <= 1:
            raise ValueError("dirichlet_epsilon must be in [0, 1]")
        if not 0 <= self.promotion_threshold <= 1:
            raise ValueError("promotion_threshold must be in [0, 1]")

    def ensure_directories(self) -> None:
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return asdict(self)


BOARD_SIZE = 5
ACTION_SIZE = 26
DEFAULT_CONFIG = Config()


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

