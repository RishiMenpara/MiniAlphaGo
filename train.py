"""Iterative AlphaZero-style training entry point."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt

from agents.random_agent import RandomAgent
from config import Config, set_global_seeds
from model.policy_value_net import PolicyValueNet
from rl.evaluation import evaluate_agents, evaluate_models
from agents.mcts_agent import MCTSAgent
from rl.replay_buffer import ReplayBuffer
from rl.self_play import play_self_game
from rl.trainer import Trainer, load_checkpoint, save_checkpoint


def make_model(cfg: Config) -> PolicyValueNet:
    return PolicyValueNet(cfg.board_size, cfg.action_size, cfg.num_channels, cfg.num_res_blocks).to(cfg.device)


def save_plots(metrics: list[dict], output_dir: str) -> None:
    if not metrics:
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    steps = range(len(metrics))
    for key in ("total_loss", "policy_loss", "value_loss"):
        axes[0].plot(steps, [m[key] for m in metrics], label=key)
    axes[0].set(title="Training losses", xlabel="optimizer step"); axes[0].legend()
    iteration_rows = [m for m in metrics if "candidate_win_rate" in m]
    if iteration_rows:
        axes[1].plot([m["iteration"] for m in iteration_rows], [m["candidate_win_rate"] for m in iteration_rows], marker="o", label="candidate vs best")
        axes[1].plot([m["iteration"] for m in iteration_rows], [m.get("random_win_rate", 0) for m in iteration_rows], marker="o", label="best vs random")
    axes[1].set(title="Evaluation", xlabel="iteration", ylim=(0, 1)); axes[1].legend()
    fig.tight_layout(); fig.savefig(Path(output_dir) / "training_metrics.png", dpi=140); plt.close(fig)


def run_training(cfg: Config, resume: str | None = None) -> None:
    cfg.validate(); cfg.ensure_directories(); set_global_seeds(cfg.random_seed)
    if resume:
        import torch
        resume_payload = torch.load(resume, map_location="cpu", weights_only=False)
        saved_cfg = resume_payload.get("config", {})
        cfg.num_channels = int(saved_cfg.get("num_channels", cfg.num_channels))
        cfg.num_res_blocks = int(saved_cfg.get("num_res_blocks", cfg.num_res_blocks))
    best = make_model(cfg)
    best_path = Path(cfg.checkpoint_dir) / "best_model.pt"
    start_iteration = 1
    if resume:
        payload = load_checkpoint(resume, best, device=cfg.device)
        start_iteration = int(payload.get("iteration", 0)) + 1
        print(f"Resumed model from {resume} at iteration {start_iteration}")
    elif best_path.exists():
        load_checkpoint(best_path, best, device=cfg.device)
        print(f"Loaded existing best model: {best_path}")
    else:
        save_checkpoint(best_path, best, cfg)
    print(f"Device: {cfg.device} | Parameters: {best.parameter_count:,}")

    replay_path = Path(cfg.checkpoint_dir) / "replay_buffer.pkl"
    replay = ReplayBuffer.load(replay_path) if resume and replay_path.exists() else ReplayBuffer(cfg.replay_buffer_size)
    all_metrics: list[dict] = []
    for iteration in range(start_iteration, start_iteration + cfg.iterations):
        results = {1: 0, -1: 0, 0: 0}
        for game in range(cfg.self_play_games):
            examples, winner, _ = play_self_game(best, cfg, add_noise=True)
            replay.extend(examples); results[winner] += 1
            print(f"Iteration {iteration} self-play {game + 1}/{cfg.self_play_games}: winner={winner:+d}, positions={len(examples)}")
        replay.save(replay_path)

        candidate = copy.deepcopy(best).to(cfg.device)
        trainer = Trainer(candidate, replay, cfg)
        train_metrics = trainer.train_steps()
        all_metrics.extend(train_metrics)
        candidate_path = Path(cfg.checkpoint_dir) / "candidate_model.pt"
        save_checkpoint(candidate_path, candidate, cfg, optimizer=trainer.optimizer, iteration=iteration)

        evaluation = evaluate_models(candidate, best, cfg)
        promoted = evaluation.win_rate >= cfg.promotion_threshold
        if promoted:
            best.load_state_dict(candidate.state_dict())
        checkpoint_metrics = {**evaluation.to_dict(), "promoted": promoted}
        # Persist the accepted incumbent every iteration. This makes both
        # best_model.pt and iteration_NNN.pt safe resume points even when the
        # newly trained candidate was rejected.
        save_checkpoint(best_path, best, cfg, iteration=iteration, metrics=checkpoint_metrics)
        snapshot = Path(cfg.checkpoint_dir) / f"iteration_{iteration:03d}.pt"
        save_checkpoint(snapshot, best, cfg, iteration=iteration, metrics=checkpoint_metrics)

        random_result = evaluate_agents(MCTSAgent(best, cfg), RandomAgent(), cfg, max(2, cfg.evaluation_games // 2))
        row = {"iteration": iteration, "candidate_win_rate": evaluation.win_rate,
               "random_win_rate": random_result.win_rate, "promoted": promoted, **train_metrics[-1]}
        all_metrics.append(row)
        print(f"Evaluation candidate vs best: {evaluation.to_dict()} | promoted={promoted}")
        print(f"Best vs random: {random_result.to_dict()} | replay={len(replay)}")
        Path(cfg.output_dir, "metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
        save_plots(all_metrics, cfg.output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Run a tiny end-to-end training iteration")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--self-play-games", type=int)
    parser.add_argument("--mcts-simulations", type=int)
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--evaluation-games", type=int)
    return parser.parse_args()


def main():
    args = parse_args(); cfg = Config()
    for key in ("iterations", "self_play_games", "mcts_simulations", "training_steps", "evaluation_games"):
        value = getattr(args, key)
        if value is not None: setattr(cfg, key, value)
    if args.smoke_test:
        cfg.iterations = 1; cfg.self_play_games = 1; cfg.mcts_simulations = 2
        cfg.training_steps = 2; cfg.evaluation_games = 2; cfg.batch_size = 16
        cfg.num_channels = 16; cfg.num_res_blocks = 1; cfg.max_game_length = 30
    run_training(cfg, args.resume)


if __name__ == "__main__":
    main()

