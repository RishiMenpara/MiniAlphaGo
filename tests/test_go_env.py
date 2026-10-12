import numpy as np
import pytest

from game.go_env import GoEnv, IllegalMoveError


def play(env, *actions):
    for action in actions:
        env.step(action)


def test_simple_capture():
    env = GoEnv(komi=0)
    env.board[1, 1] = env.WHITE
    env.board[0, 1] = env.board[1, 0] = env.board[1, 2] = env.BLACK
    env.current_player = env.BLACK
    env.position_history = {env._board_key(env.board)}
    env.step(11)
    assert env.board[1, 1] == 0


def test_group_capture():
    env = GoEnv(komi=0)
    env.board[1, 1] = env.WHITE
    env.board[1, 2] = env.WHITE
    env.board[0, 1:3] = env.BLACK
    env.board[2, 1:3] = env.BLACK
    env.board[1, 0] = env.BLACK
    env.current_player = env.BLACK
    env.position_history = {env._board_key(env.board)}
    env.step(8)
    assert env.board[1, 1] == env.board[1, 2] == 0


def test_group_and_liberties():
    env = GoEnv()
    env.board[2, 2] = env.board[2, 3] = env.BLACK
    group, liberties = env.get_group_and_liberties(2, 2)
    assert group == {(2, 2), (2, 3)}
    assert len(liberties) == 6


def test_suicide_is_illegal():
    env = GoEnv()
    env.board[0, 1] = env.board[1, 0] = env.WHITE
    env.current_player = env.BLACK
    env.position_history = {env._board_key(env.board)}
    assert not env.is_legal(0)
    with pytest.raises(IllegalMoveError):
        env.step(0)


def test_positional_superko():
    env = GoEnv()
    # A legal empty-board placement is prohibited if its resulting board occurred before.
    future = env.board.copy(); future[0, 0] = env.BLACK
    env.position_history.add(env._board_key(future))
    assert not env.is_legal(0)


def test_pass_and_two_pass_end():
    env = GoEnv()
    env.step(env.pass_action)
    assert not env.done and env.consecutive_passes == 1
    env.step(env.pass_action)
    assert env.done


def test_chinese_area_scoring():
    env = GoEnv(komi=0.5)
    env.board[:, :] = env.BLACK
    env.board[2, 2] = env.EMPTY
    score = env.score()
    assert score.black == 25
    assert score.white == 0.5
    assert env.winner() == env.BLACK

