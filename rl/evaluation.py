"""Deterministic alternating-color model and baseline evaluation."""
from __future__ import annotations

from dataclasses import dataclass

from agents.mcts_agent import MCTSAgent
from game.go_env import GoEnv


@dataclass
class EvaluationResult:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def games(self): return self.wins + self.losses + self.draws

    @property
    def win_rate(self): return self.wins / self.games if self.games else 0.0

    def to_dict(self):
        return {"wins": self.wins, "losses": self.losses, "draws": self.draws, "win_rate": self.win_rate}


def play_match(black_agent, white_agent, config) -> int:
    env = GoEnv(config.board_size, config.komi)
    while not env.done and env.move_count < config.max_game_length:
        agent = black_agent if env.current_player == GoEnv.BLACK else white_agent
        env.step(agent.select_action(env))
    return env.winner()


def evaluate_agents(candidate_agent, opponent_agent, config, games: int | None = None) -> EvaluationResult:
    result = EvaluationResult()
    for game in range(games or config.evaluation_games):
        candidate_black = game % 2 == 0
        winner = play_match(candidate_agent if candidate_black else opponent_agent,
                            opponent_agent if candidate_black else candidate_agent, config)
        candidate_color = GoEnv.BLACK if candidate_black else GoEnv.WHITE
        if winner == candidate_color: result.wins += 1
        elif winner == 0: result.draws += 1
        else: result.losses += 1
    return result


def evaluate_models(candidate, best, config, games=None) -> EvaluationResult:
    return evaluate_agents(MCTSAgent(candidate, config), MCTSAgent(best, config), config, games)

