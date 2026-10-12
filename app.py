"""Gradio board for Human-vs-AI and stepwise AI-vs-AI visualization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # Core training/tests remain usable before UI extras are installed.
    gr = None

from agents.mcts_agent import MCTSAgent
from config import Config
from evaluate import load_model
from game.go_env import GoEnv, IllegalMoveError


@dataclass
class GameSession:
    env: GoEnv
    ai: MCTSAgent
    human_color: int
    mode: str


def available_checkpoints():
    paths = sorted(Path("checkpoints").glob("*.pt"))
    return [str(p) for p in paths] or ["checkpoints/best_model.pt"]


def _require_gradio():
    if gr is None:
        raise RuntimeError("Gradio is not installed. Run: pip install -r requirements.txt")


def create_session(checkpoint, human_color, mode, simulations):
    _require_gradio()
    cfg = Config(mcts_simulations=int(simulations))
    if not Path(checkpoint).exists():
        raise gr.Error(f"Checkpoint not found: {checkpoint}. Run training first.")
    session = GameSession(GoEnv(komi=cfg.komi), MCTSAgent(load_model(checkpoint, cfg), cfg),
                          1 if human_color == "Black" else -1, mode)
    if mode == "Human vs AI" and session.human_color == -1:
        session.env.step(session.ai.select_action(session.env))
    return session


def render(session: GameSession, message=""):
    _require_gradio()
    env = session.env; updates = []
    for action in range(25):
        r, c = divmod(action, 5); value = int(env.board[r, c])
        label = "●" if value == 1 else "○" if value == -1 else "·"
        if env.last_move == action: label = f"[{label}]"
        human_turn = session.mode == "Human vs AI" and env.current_player == session.human_color
        updates.append(gr.update(value=label, interactive=not env.done and human_turn and env.is_legal(action)))
    score = env.score(); current = "Game over" if env.done else ("Black" if env.current_player == 1 else "White")
    winner = "—" if not env.done else ("Black" if env.winner() == 1 else "White" if env.winner() == -1 else "Draw")
    status = message or ("Your turn." if session.mode == "Human vs AI" and env.current_player == session.human_color else "AI ready.")
    return updates + [session, status, current, f"{score.black:g}", f"{score.white:g}", f"{env.komi:g}", winner]


def new_game(checkpoint, color, mode, simulations):
    return render(create_session(checkpoint, color, mode, simulations), "New game started.")


def human_action(session, action):
    if session is None: raise gr.Error("Start a new game first.")
    env = session.env
    if session.mode != "Human vs AI" or env.current_player != session.human_color: return render(session, "It is not the human turn.")
    try: env.step(action)
    except IllegalMoveError as exc: return render(session, str(exc))
    if not env.done: env.step(session.ai.select_action(env))
    return render(session, "Move complete." if not env.done else "Game finished.")


def ai_step(session):
    if session is None: raise gr.Error("Start a new game first.")
    if session.mode != "AI vs AI": return render(session, "AI Step is for AI vs AI mode.")
    if not session.env.done: session.env.step(session.ai.select_action(session.env))
    return render(session, "AI played one move.")


def build_app():
    _require_gradio()
    css = ".board button {font-size: 24px !important; min-width: 54px !important; background:#d8a95b !important}"
    with gr.Blocks(css=css, title="MiniAlphaGo") as demo:
        gr.Markdown("# MiniAlphaGo — AlphaZero-Inspired 5×5 Go")
        session = gr.State(None)
        with gr.Row():
            checkpoint = gr.Dropdown(available_checkpoints(), value=available_checkpoints()[0], label="AI checkpoint", allow_custom_value=True)
            color = gr.Radio(["Black", "White"], value="Black", label="Human color")
            mode = gr.Radio(["Human vs AI", "AI vs AI"], value="Human vs AI", label="Mode")
            simulations = gr.Slider(1, 800, value=50, step=1, label="MCTS simulations")
        with gr.Column(elem_classes="board"):
            buttons = []
            for row in range(5):
                with gr.Row():
                    for col in range(5): buttons.append(gr.Button("·", min_width=54))
        with gr.Row():
            new = gr.Button("New Game", variant="primary"); pass_btn = gr.Button("PASS"); step_btn = gr.Button("AI Step")
        status = gr.Textbox(label="Status", interactive=False)
        with gr.Row():
            current = gr.Textbox(label="Current player", interactive=False)
            black_score = gr.Textbox(label="Black score", interactive=False)
            white_score = gr.Textbox(label="White score", interactive=False)
            komi = gr.Textbox(label="Komi", interactive=False)
            winner = gr.Textbox(label="Winner", interactive=False)
        outputs = buttons + [session, status, current, black_score, white_score, komi, winner]
        new.click(new_game, [checkpoint, color, mode, simulations], outputs)
        pass_btn.click(lambda s: human_action(s, 25), session, outputs)
        step_btn.click(ai_step, session, outputs)
        for action, button in enumerate(buttons):
            button.click(lambda s, a=action: human_action(s, a), session, outputs)
    return demo


if __name__ == "__main__":
    build_app().launch()

