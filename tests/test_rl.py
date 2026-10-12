import numpy as np
import torch

from config import Config
from model.policy_value_net import PolicyValueNet
from rl.replay_buffer import ReplayBuffer
from rl.self_play import play_self_game
from rl.trainer import Trainer


def test_replay_and_training_step():
    cfg = Config(num_channels=8, num_res_blocks=1, batch_size=2, training_steps=1, device="cpu")
    buffer = ReplayBuffer(5)
    policy = np.ones(26, dtype=np.float32) / 26
    buffer.extend([(np.zeros((4, 5, 5), dtype=np.float32), policy, 1)] * 3)
    history = Trainer(PolicyValueNet(channels=8, residual_blocks=1), buffer, cfg).train_steps(1)
    assert len(buffer) == 3 and np.isfinite(history[0]["total_loss"])


def test_tiny_self_play_integrates_all_components():
    cfg = Config(num_channels=8, num_res_blocks=1, mcts_simulations=1, max_game_length=4, device="cpu")
    model = PolicyValueNet(channels=8, residual_blocks=1)
    examples, winner, env = play_self_game(model, cfg, add_noise=False)
    # Two passes may correctly finish an untrained model's game before the
    # four-move safety cap, so assert the invariant rather than an exact length.
    assert 1 <= len(examples) <= cfg.max_game_length
    assert len(examples) == env.move_count
    assert examples[0][0].shape == (4, 5, 5)
    assert examples[0][1].shape == (26,) and np.isclose(examples[0][1].sum(), 1)
    assert winner in (-1, 0, 1)
