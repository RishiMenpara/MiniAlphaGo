import numpy as np
import torch
from torch import nn

from game.go_env import GoEnv
from mcts.search import MCTS


class UniformNet(nn.Module):
    def forward(self, x):
        return torch.zeros(len(x), 26, device=x.device), torch.zeros(len(x), 1, device=x.device)


def test_mcts_policy_is_legal_finite_and_normalized():
    env = GoEnv()
    env.step(0)
    mcts = MCTS(UniformNet(), simulations=8)
    policy, action = mcts.search(env, temperature=1)
    assert policy.shape == (26,)
    assert np.isfinite(policy).all()
    assert np.isclose(policy.sum(), 1)
    assert env.is_legal(action)
    assert policy[0] == 0


def test_terminal_root_does_not_evaluate_network():
    env = GoEnv(); env.step(25); env.step(25)
    root = MCTS(UniformNet(), simulations=2).run(env)
    assert not root.children and root.visit_count == 1
