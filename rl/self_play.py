"""Generate AlphaZero-style self-play training examples."""
from __future__ import annotations

import numpy as np

from config import Config
from game.encoding import encode_state
from game.go_env import GoEnv
from mcts.search import MCTS


def play_self_game(model, config: Config, *, add_noise: bool = True):
    env = GoEnv(config.board_size, config.komi)
    search = MCTS(model, config.mcts_simulations, config.c_puct, config.device,
                  config.dirichlet_alpha, config.dirichlet_epsilon)
    trajectory: list[tuple[np.ndarray, np.ndarray, int]] = []
    while not env.done and env.move_count < config.max_game_length:
        temperature = config.temperature if env.move_count < config.temperature_drop_move else 0.0
        policy, action = search.search(env, temperature=temperature, add_noise=add_noise)
        trajectory.append((encode_state(env), policy, env.current_player))
        env.step(action)
    winner = env.winner()
    examples = [(state, policy, float(winner * player)) for state, policy, player in trajectory]
    return examples, winner, env

