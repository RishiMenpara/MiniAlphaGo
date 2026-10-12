# MiniAlphaGo — AlphaZero-Inspired 5×5 Go AI

MiniAlphaGo is a complete educational self-play reinforcement-learning system. It learns a policy and value function from its own games and improves move selection with PUCT Monte Carlo Tree Search (MCTS). It is **AlphaZero-inspired**; it is not a reproduction of DeepMind's full AlphaGo/AlphaZero training infrastructure and it does not use a pretrained Go engine.

## What is included

- Correct 5×5 Go: captures, connected groups/liberties, suicide prevention, positional superko, pass, two-pass termination, Chinese area scoring, and configurable komi.
- Four-plane, current-player-relative encoding with shape `[4, 5, 5]`.
- Configurable residual PyTorch policy-value CNN. The policy head emits 26 logits and the value head emits one `tanh` value.
- PUCT MCTS with illegal-action masking, pass support, root Dirichlet noise, temperature control, terminal backup, and alternating value perspectives.
- Self-play examples `(s, π, z)`, bounded serializable replay, and batched AdamW policy/value training.
- Candidate-vs-best evaluation with alternating colors, deterministic search, and threshold-based promotion.
- Random baseline, resumable device-safe checkpoints, JSON metrics, and training plots.
- Terminal play and a Gradio UI for Human-vs-AI or stepwise AI-vs-AI visualization.

## Learning pipeline

```text
Go State
   ↓
State Encoder [4, 5, 5]
   ↓
Residual Policy-Value CNN
   ↓
Policy P + Value V
   ↓
MCTS / PUCT
   ↓
Improved Policy π → Action → Go Environment
   ↓
Self-Play → (s, π, z) → Replay Buffer
   ↓
Policy loss + Value loss → Candidate Network
   ↓
Candidate vs Best → Promote or retain Best Model
```

The policy target `π` is the normalized root visit distribution, not the raw network policy. After a game, `z` is `+1`, `0`, or `-1` from the current player's perspective at each stored state. Training minimizes:

```text
Lpolicy = -Σa π(a) log softmax(logits)(a)
Lvalue  = MSE(v, z)
Ltotal  = Lpolicy + Lvalue
```

PUCT selects an edge using approximately `Q + c_puct × P × sqrt(Nparent) / (1 + Nedge)`. A child's stored value is from the child-to-play perspective, so it is negated when its parent compares actions and is sign-corrected during backup.

## Rules and representation

Black (`1`) moves first; White is `-1`; empty is `0`. Actions `0..24` are row-major board intersections and action `25` is pass. Two consecutive passes finish the game. Chinese area score is stones plus surrounded empty area; komi is added to White. Positional superko rejects a placement if its resulting board position has occurred earlier. Passing is always legal before game end.

The input planes are current-player stones, opponent stones, legal placement mask, and a plane filled with `1` when Black is to move (otherwise `0`). Batches have shape `[B, 4, 5, 5]`; outputs are policy logits `[B, 26]` and values `[B, 1]`.

## Install and verify

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
pytest
python train.py --smoke-test
```

`requirements.txt` contains the core training/test dependencies and is the correct file for Kaggle. The optional local web interface has its own dependency file so it does not disturb Kaggle's preinstalled UI packages:

```bash
pip install -r requirements-ui.txt
python app.py
```

The smoke run uses a small network, two MCTS simulations per move, one self-play game, two optimizer steps, and two evaluation games. It verifies the pipeline; it does not produce a strong player.

## Training

Default hyperparameters live in `config.py` and can be edited centrally. Common command-line overrides are available:

```bash
python train.py
python train.py --iterations 20 --self-play-games 50 --mcts-simulations 100 --training-steps 400 --evaluation-games 40
python train.py --resume checkpoints/iteration_005.pt
```

Each iteration performs self-play with the current best model, trains a copied candidate, evaluates candidate vs best without root noise and at zero temperature, promotes it when its decisive win rate reaches the threshold, and evaluates the resulting best model against RandomAgent. Files are written to:

```text
checkpoints/best_model.pt
checkpoints/candidate_model.pt
checkpoints/iteration_001.pt
checkpoints/replay_buffer.pkl
outputs/metrics.json
outputs/training_metrics.png
```

Checkpoint loading uses `map_location` and reads the saved channel/block configuration, so CPU, CUDA, and smoke-sized checkpoints are portable. Resume restores model progress and replay data. A newly created trainer initializes a fresh AdamW optimizer for the next candidate; candidate checkpoints themselves also contain optimizer state.

## Evaluation and play

```bash
python evaluate.py --checkpoint checkpoints/best_model.pt --games 20 --simulations 100
python evaluate.py --checkpoint checkpoints/candidate_model.pt --opponent-checkpoint checkpoints/best_model.pt
python play.py --checkpoint checkpoints/best_model.pt --color white --simulations 100
pip install -r requirements-ui.txt
python app.py
```

Open the local URL printed by Gradio. Select a checkpoint, human color, mode, and search budget, then press **New Game**. If the human is White, the AI automatically makes Black's opening move. Board intersections and PASS use the same `GoEnv`, trained `PolicyValueNet`, and `MCTS` implementation used during training. In AI-vs-AI mode, **AI Step** advances one move so the search can be observed without blocking the UI for a full game. The last move is shown in brackets.

## Kaggle Tesla T4 quick start

Upload this directory as a Kaggle dataset or clone/copy it under `/kaggle/working/mini-alphago`, enable a GPU accelerator, and run:

```python
import os, torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
os.chdir("/kaggle/working/mini-alphago")
!pip install -q -r requirements.txt
!pytest -q
!python train.py --smoke-test
```

For a larger run:

```python
!python train.py --iterations 20 --self-play-games 50 --mcts-simulations 100 --training-steps 400
!python train.py --resume /kaggle/working/mini-alphago/checkpoints/iteration_005.pt
```

Everything under `/kaggle/working/mini-alphago/checkpoints` is downloadable from notebook outputs. Gradio may launch with `python app.py`, but inbound interactive access depends on Kaggle's current networking policy. See `notebooks/kaggle_quickstart.ipynb` for executable cells.

CUDA is detected automatically. GPU acceleration is strongest for batched training. This implementation performs sequential Python tree search and single-state inference, so it does **not** fully utilize a T4 during MCTS; batching or parallel actors would be needed for that.

## Project layout

```text
config.py                 central validated configuration
game/                     Go rules and encoding
model/                    residual policy-value network
mcts/                     node statistics and PUCT search
rl/                       self-play, replay, training, evaluation
agents/                   MCTS and random agents
train.py                  iterative training/promotion loop
evaluate.py               checkpoint evaluation
play.py                   terminal Human-vs-AI
app.py                    Gradio Human-vs-AI / AI-vs-AI UI
tests/                    rule, shape, search, RL, and loading tests
notebooks/                Kaggle quick-start notebook
```

## Important limitations

- 5×5 only. The compact environment intentionally does not implement handicap stones, Japanese scoring, dead-stone adjudication, or Go protocol support.
- Scoring treats surrounded empty regions by border color. As with most compact Go environments, players should capture dead groups before passing; there is no post-game dead-stone agreement phase.
- Python MCTS is correct and readable rather than distributed or highly optimized. There is no inference batching, transposition table, symmetry augmentation, or parallel self-play worker pool.
- Strength depends on compute. The default settings are a practical starting point, not a promise of expert play; the smoke checkpoint is only a functionality check.
- Replay serialization uses Python pickle and should only load files you trust.

