# MiniAlphaGo

An **AlphaZero-inspired 5×5 Go AI** built with Python, PyTorch, Monte Carlo Tree Search (MCTS), and a residual policy-value neural network.

MiniAlphaGo plays real 5×5 Go using the trained model included with this project. It supports Human vs AI and AI vs AI games through an interactive Gradio interface.

## 🎮 Live Demo

**[Play MiniAlphaGo on Hugging Face](https://rishimenpara-minialphago.hf.space)**

> The free hosted demo may take a moment to start. Select 5–25 MCTS simulations for faster moves.

## ✨ Features

- Human vs AI gameplay
- AI vs AI visualization
- Play as Black or White
- Policy-value residual CNN implemented in PyTorch
- Neural-network-guided MCTS with PUCT
- Complete 5×5 Go rules
- Captures and multi-stone captures
- Liberty detection and suicide prevention
- Positional superko
- Pass moves and two-pass game termination
- Chinese-style area scoring with configurable komi
- Legal-move validation and clear error messages
- Checkpoint loading on CPU or CUDA
- Interactive Gradio web interface

## 📊 Model Performance

The trained checkpoint achieved **95 wins in 100 games against the included RandomAgent baseline**.

This result measures performance against a random opponent and does not represent professional Go strength.

## 🧠 Architecture

```text
5×5 Go position
      ↓
State encoder [4, 5, 5]
      ↓
Residual policy-value network
      ↓
Policy logits + position value
      ↓
MCTS with PUCT
      ↓
Selected legal move
```

The action space contains **26 actions**: 25 board intersections and one PASS action. The four input planes represent the current player's stones, the opponent's stones, legal placements, and the current-player indicator.

## 🚀 Run Locally

### Requirements

- Python 3.10 or 3.11
- Windows, Linux, or macOS
- GPU optional

### Installation

```bash
git clone https://github.com/RishiMenpara/MiniAlphaGo.git
cd MiniAlphaGo
python -m venv .venv
```

Activate the environment.

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-ui.txt
```

Place the trained checkpoint at:

```text
checkpoints/best_model.pt
```

Launch the interface:

```bash
python app.py
```

Open the address printed in the terminal, normally `http://127.0.0.1:7860`.

## 🕹️ How to Play

1. Select Human vs AI or AI vs AI.
2. Choose whether the human plays Black or White.
3. Select the trained checkpoint.
4. Choose the number of MCTS simulations.
5. Start a new game.
6. Click a legal board intersection to place a stone.
7. Use PASS when you do not want to place a stone.

Black moves first. If the human selects White, the AI automatically makes the opening Black move.

## 📁 Project Structure

```text
MiniAlphaGo/
├── app.py                 # Gradio web interface
├── play.py                # Local gameplay entry point
├── evaluate.py            # Model evaluation utility
├── config.py              # Central configuration
├── game/                  # Go rules and state encoding
├── model/                 # Residual policy-value network
├── mcts/                  # MCTS and PUCT search
├── rl/                    # Reinforcement-learning components
├── agents/                # Baseline agents
├── tests/                 # Automated tests
├── checkpoints/           # Trained model checkpoints
└── outputs/               # Evaluation results and plots
```

## ✅ Tests

```bash
pytest
```

The tests cover Go rules, state encoding, model shapes, MCTS behavior, and integration between the main components.

## 🛠️ Technology

- Python
- PyTorch
- NumPy
- Gradio
- Matplotlib
- Hugging Face Spaces

## ⚠️ Limitations

- The project uses a 5×5 board and is not intended for full-size 19×19 Go.
- MCTS is sequential Python code, so higher simulation counts increase response time.
- The free hosted demo has limited compute resources.
- Results against RandomAgent do not establish strength against experienced players or established Go engines.

## Disclaimer

This is an educational **AlphaZero-inspired MiniAlphaGo project**. It is not a reproduction of DeepMind's complete AlphaGo or AlphaZero systems and does not use a pretrained external Go engine.

## 👨‍💻 Author

Developed by **Rishi Menpara**.

- [GitHub Repository](https://github.com/RishiMenpara/MiniAlphaGo)
- [Live Demo](https://rishimenpara-minialphago.hf.space)

If you find the project useful, please consider giving the repository a ⭐.
