"""Current-player-relative state encoding."""
import numpy as np

from .go_env import GoEnv


def encode_state(env: GoEnv) -> np.ndarray:
    legal = env.legal_action_mask()[:-1].reshape(env.size, env.size)
    indicator = np.full((env.size, env.size), float(env.current_player == GoEnv.BLACK), dtype=np.float32)
    return np.stack(
        [
            (env.board == env.current_player).astype(np.float32),
            (env.board == -env.current_player).astype(np.float32),
            legal.astype(np.float32),
            indicator,
        ],
        axis=0,
    )

