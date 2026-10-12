"""Policy-value-guided Monte Carlo Tree Search with PUCT."""
from __future__ import annotations

import math

import numpy as np
import torch

from game.encoding import encode_state
from game.go_env import GoEnv
from .node import MCTSNode


class MCTS:
    def __init__(self, model, simulations: int = 50, c_puct: float = 1.5, device: str | torch.device = "cpu",
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = torch.device(device)
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    @torch.no_grad()
    def _evaluate(self, env: GoEnv) -> tuple[np.ndarray, float]:
        self.model.eval()
        state = torch.from_numpy(encode_state(env)).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        policy = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.float64)
        legal = env.legal_action_mask().astype(np.float64)
        policy *= legal
        total = policy.sum()
        if not np.isfinite(total) or total <= 1e-12:
            policy = legal / legal.sum()
        else:
            policy /= total
        return policy, float(value.item())

    @staticmethod
    def _expand(node: MCTSNode, env: GoEnv, priors: np.ndarray) -> None:
        for action in env.legal_actions():
            node.children[action] = MCTSNode(
                prior=float(priors[action]), to_play=-env.current_player, parent=node, action=action
            )

    def _select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        sqrt_parent = math.sqrt(max(1, node.visit_count))
        best_score, best_action, best_child = -float("inf"), -1, None
        for action, child in node.children.items():
            # child Q is from child's player perspective; negate for the parent.
            score = -child.q_value + self.c_puct * child.prior * sqrt_parent / (1 + child.visit_count)
            if score > best_score:
                best_score, best_action, best_child = score, action, child
        assert best_child is not None
        return best_action, best_child

    @staticmethod
    def _backup(path: list[MCTSNode], value: float, leaf_player: int) -> None:
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value if node.to_play == leaf_player else -value

    def run(self, root_env: GoEnv, add_noise: bool = False) -> MCTSNode:
        root = MCTSNode(to_play=root_env.current_player)
        if root_env.done:
            root.visit_count = 1
            root.value_sum = root_env.terminal_value(root_env.current_player)
            return root
        priors, _ = self._evaluate(root_env)
        self._expand(root, root_env, priors)
        if add_noise and root.children:
            actions = list(root.children)
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
            for action, sample in zip(actions, noise):
                child = root.children[action]
                child.prior = (1 - self.dirichlet_epsilon) * child.prior + self.dirichlet_epsilon * float(sample)

        for _ in range(self.simulations):
            env = root_env.clone()
            node = root
            path = [node]
            while node.expanded and not env.done:
                action, node = self._select_child(node)
                env.step(action)
                path.append(node)
            if env.done:
                value = env.terminal_value(env.current_player)
            else:
                child_priors, value = self._evaluate(env)
                self._expand(node, env, child_priors)
            self._backup(path, value, env.current_player)
        return root

    @staticmethod
    def policy_from_root(root: MCTSNode, temperature: float = 1.0, action_size: int = 26) -> np.ndarray:
        visits = np.zeros(action_size, dtype=np.float64)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if temperature <= 1e-8:
            policy = np.zeros_like(visits)
            if visits.sum() > 0:
                policy[int(np.argmax(visits))] = 1.0
            elif root.children:
                policy[next(iter(root.children))] = 1.0
            return policy.astype(np.float32)
        scaled = np.power(visits, 1.0 / temperature)
        if not np.isfinite(scaled).all() or scaled.sum() <= 0:
            legal = np.array([a in root.children for a in range(action_size)], dtype=np.float64)
            scaled = legal
        return (scaled / scaled.sum()).astype(np.float32)

    def search(self, env: GoEnv, temperature: float = 1.0, add_noise: bool = False) -> tuple[np.ndarray, int]:
        root = self.run(env, add_noise=add_noise)
        policy = self.policy_from_root(root, temperature, env.action_size)
        action = int(np.random.choice(env.action_size, p=policy)) if temperature > 1e-8 else int(np.argmax(policy))
        return policy, action

