"""Small terminal interface using the real model and MCTS."""
import argparse

from agents.mcts_agent import MCTSAgent
from config import Config
from evaluate import load_model
from game.go_env import GoEnv, IllegalMoveError


def show(env):
    symbols = {0: ".", 1: "X", -1: "O"}
    print("  " + " ".join(map(str, range(5))))
    for r in range(5): print(r, " ".join(symbols[int(x)] for x in env.board[r]))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--color", choices=("black", "white"), default="black"); parser.add_argument("--simulations", type=int, default=100)
    args = parser.parse_args(); cfg = Config(mcts_simulations=args.simulations)
    env = GoEnv(komi=cfg.komi); ai = MCTSAgent(load_model(args.checkpoint, cfg), cfg)
    human = 1 if args.color == "black" else -1
    while not env.done:
        show(env)
        if env.current_player == human:
            raw = input("move 'row col' or 'pass': ").strip().lower()
            action = 25 if raw == "pass" else int(raw.split()[0]) * 5 + int(raw.split()[1])
            try: env.step(action)
            except (IllegalMoveError, ValueError, IndexError): print("Illegal move; try again.")
        else:
            action = ai.select_action(env); print(f"AI: {'pass' if action == 25 else divmod(action, 5)}"); env.step(action)
    show(env); print(f"Score={env.score()} winner={env.winner():+d}")


if __name__ == "__main__": main()

