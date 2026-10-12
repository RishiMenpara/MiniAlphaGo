import numpy as np

from game.encoding import encode_state
from game.go_env import GoEnv


def test_encoding_shape_and_perspective():
    env = GoEnv()
    env.step(0)
    state = encode_state(env)
    assert state.shape == (4, 5, 5)
    assert state.dtype == np.float32
    assert state[1, 0, 0] == 1  # black is opponent after black's move
    assert np.all(state[3] == 0)  # current player is white

