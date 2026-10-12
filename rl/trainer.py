"""Policy + value optimization and device-safe checkpoint helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F


class Trainer:
    def __init__(self, model, replay_buffer, config):
        self.model = model.to(config.device)
        self.buffer = replay_buffer
        self.config = config
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )

    def train_steps(self, steps: int | None = None) -> list[dict[str, float]]:
        if not self.buffer:
            raise ValueError("Replay buffer is empty")
        history = []
        self.model.train()
        for _ in range(steps if steps is not None else self.config.training_steps):
            states, target_policies, target_values = self.buffer.sample(self.config.batch_size)
            states_t = torch.from_numpy(states).to(self.config.device)
            policies_t = torch.from_numpy(target_policies).to(self.config.device)
            values_t = torch.from_numpy(target_values).to(self.config.device)
            logits, values = self.model(states_t)
            policy_loss = -(policies_t * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
            value_loss = F.mse_loss(values.squeeze(1), values_t)
            loss = policy_loss + value_loss
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            history.append({
                "total_loss": float(loss.item()), "policy_loss": float(policy_loss.item()),
                "value_loss": float(value_loss.item()), "learning_rate": self.optimizer.param_groups[0]["lr"],
                "replay_samples": float(len(self.buffer)),
            })
        return history


def save_checkpoint(path, model, config, *, optimizer=None, iteration=0, metrics=None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None,
        "iteration": iteration, "config": config.to_dict(), "metrics": metrics or {},
    }, path)


def load_checkpoint(path, model, *, device="cpu", optimizer=None) -> dict[str, Any]:
    payload = torch.load(path, map_location=device, weights_only=False)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    if optimizer is not None and payload.get("optimizer_state"):
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload

