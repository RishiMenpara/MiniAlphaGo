"""Correct, compact 5x5 Go environment with positional superko."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np


class IllegalMoveError(ValueError):
    """Raised when an action is illegal in the current position."""


@dataclass(frozen=True)
class Score:
    black: float
    white: float


class GoEnv:
    EMPTY, BLACK, WHITE = 0, 1, -1

    def __init__(self, size: int = 5, komi: float = 2.5):
        if size != 5:
            raise ValueError("MiniAlphaGo currently supports exactly a 5x5 board")
        self.size = size
        self.action_size = size * size + 1
        self.pass_action = size * size
        self.komi = float(komi)
        self.reset()

    def reset(self) -> np.ndarray:
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = self.BLACK
        self.consecutive_passes = 0
        self.move_count = 0
        self.done = False
        self.last_move: int | None = None
        self.history: list[tuple[np.ndarray, int | None, int]] = []
        self.position_history: set[bytes] = {self._board_key(self.board)}
        return self.board.copy()

    def clone(self) -> "GoEnv":
        other = GoEnv(self.size, self.komi)
        other.board = self.board.copy()
        other.current_player = self.current_player
        other.consecutive_passes = self.consecutive_passes
        other.move_count = self.move_count
        other.done = self.done
        other.last_move = self.last_move
        other.history = [(b.copy(), a, p) for b, a, p in self.history]
        other.position_history = set(self.position_history)
        return other

    copy = clone

    def _board_key(self, board: np.ndarray) -> bytes:
        return board.tobytes()

    def _neighbors(self, row: int, col: int) -> Iterable[tuple[int, int]]:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def get_group_and_liberties(
        self, row: int, col: int, board: np.ndarray | None = None
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        board = self.board if board is None else board
        color = int(board[row, col])
        if color == self.EMPTY:
            return set(), set()
        group: set[tuple[int, int]] = set()
        liberties: set[tuple[int, int]] = set()
        queue = deque([(row, col)])
        while queue:
            point = queue.popleft()
            if point in group:
                continue
            group.add(point)
            for neighbor in self._neighbors(*point):
                value = int(board[neighbor])
                if value == self.EMPTY:
                    liberties.add(neighbor)
                elif value == color and neighbor not in group:
                    queue.append(neighbor)
        return group, liberties

    def _simulate_placement(self, action: int) -> np.ndarray | None:
        if not 0 <= action < self.pass_action:
            return None
        row, col = divmod(action, self.size)
        if self.board[row, col] != self.EMPTY:
            return None
        board = self.board.copy()
        board[row, col] = self.current_player
        for nr, nc in self._neighbors(row, col):
            if board[nr, nc] == -self.current_player:
                group, liberties = self.get_group_and_liberties(nr, nc, board)
                if not liberties:
                    for gr, gc in group:
                        board[gr, gc] = self.EMPTY
        _, own_liberties = self.get_group_and_liberties(row, col, board)
        if not own_liberties:  # Suicide is illegal after captures are resolved.
            return None
        if self._board_key(board) in self.position_history:  # Positional superko.
            return None
        return board

    def is_legal(self, action: int) -> bool:
        if self.done or not 0 <= action < self.action_size:
            return False
        if action == self.pass_action:
            return True
        return self._simulate_placement(action) is not None

    def legal_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=np.float32)
        if not self.done:
            for action in range(self.action_size):
                mask[action] = float(self.is_legal(action))
        return mask

    def legal_actions(self) -> list[int]:
        return np.flatnonzero(self.legal_action_mask()).tolist()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if not self.is_legal(action):
            raise IllegalMoveError(f"Illegal action {action} for player {self.current_player}")
        player = self.current_player
        self.history.append((self.board.copy(), action, player))
        if action == self.pass_action:
            self.consecutive_passes += 1
        else:
            board = self._simulate_placement(action)
            assert board is not None
            self.board = board
            self.position_history.add(self._board_key(self.board))
            self.consecutive_passes = 0
        self.last_move = action
        self.move_count += 1
        if self.consecutive_passes >= 2:
            self.done = True
        self.current_player = -player
        reward = 0.0 if not self.done else float(self.winner())
        return self.board.copy(), reward, self.done, {"player": player}

    def _territory(self) -> tuple[int, int]:
        visited: set[tuple[int, int]] = set()
        black, white = 0, 0
        for r in range(self.size):
            for c in range(self.size):
                if self.board[r, c] != self.EMPTY or (r, c) in visited:
                    continue
                region: set[tuple[int, int]] = set()
                borders: set[int] = set()
                queue = deque([(r, c)])
                while queue:
                    point = queue.popleft()
                    if point in region:
                        continue
                    region.add(point)
                    visited.add(point)
                    for neighbor in self._neighbors(*point):
                        value = int(self.board[neighbor])
                        if value == self.EMPTY and neighbor not in region:
                            queue.append(neighbor)
                        elif value != self.EMPTY:
                            borders.add(value)
                if borders == {self.BLACK}:
                    black += len(region)
                elif borders == {self.WHITE}:
                    white += len(region)
        return black, white

    def score(self) -> Score:
        """Chinese area: living stones plus exclusively surrounded empty points."""
        black_territory, white_territory = self._territory()
        black = int(np.count_nonzero(self.board == self.BLACK)) + black_territory
        white = int(np.count_nonzero(self.board == self.WHITE)) + white_territory + self.komi
        return Score(float(black), float(white))

    def winner(self) -> int:
        score = self.score()
        if score.black > score.white:
            return self.BLACK
        if score.white > score.black:
            return self.WHITE
        return 0

    def terminal_value(self, perspective_player: int) -> float:
        if not self.done:
            raise ValueError("terminal_value is only defined for finished games")
        return float(self.winner() * perspective_player)

