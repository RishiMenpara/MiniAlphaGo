"""Evaluate a checkpoint against RandomAgent or another checkpoint."""
import argparse

from agents.mcts_agent import MCTSAgent
from agents.random_agent import RandomAgent
from config import Config, set_global_seeds
from model.policy_value_net import PolicyValueNet
from rl.evaluation import evaluate_agents
from rl.trainer import load_checkpoint
import torch


def load_model(path, cfg):
    payload = torch.load(path, map_location=cfg.device, weights_only=False)
    saved = payload.get("config", {}) if isinstance(payload, dict) else {}
    channels = int(saved.get("num_channels", cfg.num_channels))
    blocks = int(saved.get("num_res_blocks", cfg.num_res_blocks))
    model = PolicyValueNet(cfg.board_size, cfg.action_size, channels, blocks).to(cfg.device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state); model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=100)
    args = parser.parse_args()
    cfg = Config(evaluation_games=args.games, mcts_simulations=args.simulations); set_global_seeds(cfg.random_seed)
    candidate = MCTSAgent(load_model(args.checkpoint, cfg), cfg)
    opponent = MCTSAgent(load_model(args.opponent_checkpoint, cfg), cfg) if args.opponent_checkpoint else RandomAgent()
    result = evaluate_agents(candidate, opponent, cfg, args.games)
    print(f"Device: {cfg.device} | Results: {result.to_dict()}")


if __name__ == "__main__": main()

