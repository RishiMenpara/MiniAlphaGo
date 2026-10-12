"""Residual policy-value CNN for 5x5 Go."""
from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 5, action_size: int = 26, channels: int = 64, residual_blocks: int = 2):
        super().__init__()
        self.board_size = board_size
        self.action_size = action_size
        self.stem = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(residual_blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True), nn.Flatten(),
            nn.Linear(2 * board_size * board_size, action_size),
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True), nn.Flatten()
        )
        self.value_fc = nn.Sequential(nn.Linear(board_size * board_size, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(self.stem(x))
        return self.policy_head(features), self.value_fc(self.value_conv(features))

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

