"""Chess.com Light & Colorful modern interface for MiniAlphaGo 5×5 Go AI Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import gradio as gr
except ImportError:  # Core training/tests remain usable before UI extras are installed.
    gr = None

from agents.mcts_agent import MCTSAgent
from config import Config
from evaluate import load_model
from game.go_env import GoEnv, IllegalMoveError

COORD_NAMES = [f"{'ABCDE'[c]}{5 - r}" for r in range(5) for c in range(5)] + ["PASS"]


@dataclass
class GameSession:
    env: GoEnv
    ai: MCTSAgent
    human_color: int
    mode: str
    move_log: list[str] = field(default_factory=list)


def available_checkpoints():
    paths = sorted(Path("checkpoints").glob("*.pt"))
    return [str(p) for p in paths] or ["checkpoints/best_model.pt"]


def _require_gradio():
    if gr is None:
        raise RuntimeError("Gradio is not installed. Run: pip install -r requirements-ui.txt")


def create_session(checkpoint, human_color, mode, simulations):
    _require_gradio()
    cfg = Config(mcts_simulations=int(simulations))
    if not Path(checkpoint).exists():
        raise gr.Error(f"Checkpoint not found: {checkpoint}. Ensure checkpoints/best_model.pt is present.")
    session = GameSession(
        GoEnv(komi=cfg.komi),
        MCTSAgent(load_model(checkpoint, cfg), cfg),
        1 if human_color == "Black" else -1,
        mode,
        [],
    )
    if mode == "Human vs AI" and session.human_color == -1:
        # AI plays first as Black
        action = session.ai.select_action(session.env)
        session.env.step(action)
        coord = COORD_NAMES[action] if action < 25 else "PASS"
        session.move_log.append(f"#1 Black (AI) → {coord}")
    return session


def render(session: GameSession | None, message=""):
    _require_gradio()
    if session is None:
        updates = [
            gr.update(
                value="✦" if i == 12 else "·",
                elem_classes=["empty-cell", "tengen"] if i == 12 else ["empty-cell"],
                interactive=True,
            )
            for i in range(25)
        ]
        sidebar_html = """
        <div class="chess-sidebar-card">
            <div class="player-header">
                <div class="player-avatar ai-avatar">🤖</div>
                <div class="player-meta">
                    <div class="player-name">MiniAlphaGo AI</div>
                    <div class="player-sub">White • 2.5 pts (Komi)</div>
                </div>
            </div>
            <div class="eval-bar-wrapper">
                <div class="eval-bar-label"><span>Black 50%</span><span>White 50%</span></div>
                <div class="eval-bar"><div class="eval-black" style="width: 50%;"></div><div class="eval-white" style="width: 50%;"></div></div>
            </div>
            <div class="player-header" style="margin-top: 10px;">
                <div class="player-avatar human-avatar">👤</div>
                <div class="player-meta">
                    <div class="player-name">You</div>
                    <div class="player-sub">Black • 0.0 pts</div>
                </div>
                <div class="turn-indicator-badge active">Your Turn</div>
            </div>
        </div>
        """
        history_html = '<div class="chess-move-table"><div class="empty-table-msg">Game ready. Click any square to begin.</div></div>'
        return updates + [None, sidebar_html, history_html, "Ready to start.", "Black", "0", "2.5", "2.5", "—"]

    env = session.env
    updates = []
    for action in range(25):
        r, c = divmod(action, 5)
        value = int(env.board[r, c])
        is_last = (env.last_move == action)

        if value == 1:
            label = "●"
            classes = ["stone-black"]
            if is_last:
                classes.append("last-move")
        elif value == -1:
            label = "○"
            classes = ["stone-white"]
            if is_last:
                classes.append("last-move")
        else:
            label = "✦" if action == 12 else "·"
            classes = ["empty-cell"]
            if action == 12:
                classes.append("tengen")

        human_turn = session.mode == "Human vs AI" and env.current_player == session.human_color
        interactive = not env.done and (human_turn or session.mode == "AI vs AI") and env.is_legal(action)
        updates.append(gr.update(value=label, elem_classes=classes, interactive=interactive))

    score = env.score()
    current_player_name = "Black" if env.current_player == 1 else "White"
    human_color_name = "Black" if session.human_color == 1 else "White"
    ai_color_name = "White" if session.human_color == 1 else "Black"
    human_turn = session.mode == "Human vs AI" and env.current_player == session.human_color

    winner_name = "—"
    if env.done:
        w = env.winner()
        winner_name = "Black" if w == 1 else "White" if w == -1 else "Draw"
        current_display = "Game over"
    else:
        current_display = current_player_name

    status_text = message or ("Your turn." if human_turn else "AI ready.")

    total_score = max(score.black + score.white, 0.1)
    black_pct = min(max((score.black / total_score) * 100, 5), 95)
    white_pct = 100 - black_pct

    if env.done:
        margin = abs(score.black - score.white)
        ai_badge = ""
        human_badge = f'<div class="turn-indicator-badge winner">🏆 {winner_name} won (+{margin:.1f})</div>'
    elif human_turn:
        human_badge = '<div class="turn-indicator-badge active">Your Turn</div>'
        ai_badge = '<div class="turn-indicator-badge idle">Waiting</div>'
    else:
        human_badge = '<div class="turn-indicator-badge idle">Waiting</div>'
        ai_badge = '<div class="turn-indicator-badge active">Thinking...</div>'

    sidebar_html = f"""
    <div class="chess-sidebar-card">
        <!-- AI Card (White) -->
        <div class="player-header">
            <div class="player-avatar ai-avatar">🤖</div>
            <div class="player-meta">
                <div class="player-name">MiniAlphaGo AI</div>
                <div class="player-sub">{ai_color_name} • {score.white:.1f} pts (incl. +{env.komi:g} Komi)</div>
            </div>
            {ai_badge}
        </div>

        <!-- Evaluation Bar -->
        <div class="eval-bar-wrapper">
            <div class="eval-bar-label">
                <span style="color:#1e293b; font-weight:800;">Black {score.black:.1f} ({black_pct:.0f}%)</span>
                <span style="color:#0284c7; font-weight:800;">White {score.white:.1f} ({white_pct:.0f}%)</span>
            </div>
            <div class="eval-bar">
                <div class="eval-black" style="width: {black_pct:.1f}%;"></div>
                <div class="eval-white" style="width: {white_pct:.1f}%;"></div>
            </div>
        </div>

        <!-- Human Card (Black) -->
        <div class="player-header">
            <div class="player-avatar human-avatar">👤</div>
            <div class="player-meta">
                <div class="player-name">You</div>
                <div class="player-sub">{human_color_name} • {score.black:.1f} pts</div>
            </div>
            {human_badge}
        </div>
    </div>
    """

    rows = []
    moves = session.move_log
    for i in range(0, len(moves), 2):
        move_num = (i // 2) + 1
        b_move = moves[i].split("→")[-1].strip() if i < len(moves) else ""
        w_move = moves[i + 1].split("→")[-1].strip() if i + 1 < len(moves) else ""
        rows.append(f"""
        <div class="move-row">
            <div class="move-idx">{move_num}.</div>
            <div class="move-val black-val">{b_move}</div>
            <div class="move-val white-val">{w_move}</div>
        </div>
        """)

    move_rows_html = "".join(rows) if rows else '<div class="empty-table-msg">No moves played yet.</div>'
    history_html = f"""
    <div class="chess-move-table">
        <div class="move-table-header">
            <div class="move-idx">#</div>
            <div class="move-val">Black</div>
            <div class="move-val">White</div>
        </div>
        <div class="move-table-body">
            {move_rows_html}
        </div>
    </div>
    """

    return updates + [
        session,
        sidebar_html,
        history_html,
        status_text,
        current_display,
        f"{score.black:g}",
        f"{score.white:g}",
        f"{env.komi:g}",
        winner_name,
    ]


def new_game(checkpoint, color, mode, simulations):
    session = create_session(checkpoint, color, mode, simulations)
    return render(session, "New game started! Black plays first.")


def human_action(session, action, checkpoint, color, mode, simulations):
    if session is None:
        session = create_session(checkpoint, color, mode, simulations)
    env = session.env
    if session.mode != "Human vs AI" or env.current_player != session.human_color:
        return render(session, "It is not the human player's turn.")

    color_name = "Black" if session.human_color == 1 else "White"
    coord_name = COORD_NAMES[action] if action < 25 else "PASS"
    move_num = len(session.move_log) + 1

    try:
        env.step(action)
        session.move_log.append(f"#{move_num} {color_name} → {coord_name}")
    except IllegalMoveError as exc:
        return render(session, f"Illegal move: {exc}")

    if not env.done:
        ai_color = "White" if session.human_color == 1 else "Black"
        ai_action = session.ai.select_action(env)
        env.step(ai_action)
        ai_coord = COORD_NAMES[ai_action] if ai_action < 25 else "PASS"
        session.move_log.append(f"#{len(session.move_log) + 1} {ai_color} → {ai_coord}")

    return render(session, "Move complete." if not env.done else "Game finished.")


def ai_step(session, checkpoint, color, mode, simulations):
    if session is None:
        session = create_session(checkpoint, color, mode, simulations)
    if session.mode != "AI vs AI":
        return render(session, "AI Step is for AI vs AI mode.")
    if not session.env.done:
        color_name = "Black" if session.env.current_player == 1 else "White"
        action = session.ai.select_action(session.env)
        session.env.step(action)
        coord_name = COORD_NAMES[action] if action < 25 else "PASS"
        session.move_log.append(f"#{len(session.move_log) + 1} {color_name} → {coord_name}")
        return render(session, f"AI played {coord_name}.")
    return render(session, "Game is already finished.")


CUSTOM_CSS = """
/* ==========================================================================
   MiniAlphaGo — Chess.com COLORFUL (Zero Grey Anywhere)
   ========================================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box !important; }

/* ── Page background ── */
body, .gradio-container, .gradio-container > .main {
    max-width: 1420px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    background: linear-gradient(145deg, #eef2ff 0%, #f0fdf4 50%, #fdf4ff 100%) !important;
    color: #1e293b !important;
    min-height: 100vh !important;
}

/* ── Suppress Gradio dark defaults ── */
.gr-panel, .gr-block, .gr-box, .gr-group, .gradio-group,
[class*="block"], [class*="panel"], .contain, .wrap,
footer { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── Labels / Titles ── */
label, span.label-text, .gr-form label, [data-testid="block-label"],
.svelte-1gfkn6j {
    background: transparent !important; color: #6d28d9 !important;
    font-size: 11px !important; font-weight: 800 !important;
    letter-spacing: 0.7px !important; text-transform: uppercase !important;
    border: none !important; box-shadow: none !important; }

/* ── Inputs / Dropdowns — indigo bordered ── */
input, select, textarea, .gr-input, .gr-dropdown,
[data-testid="dropdown"], .svelte-select {
    background: #f5f3ff !important;
    color: #3730a3 !important;
    border: 1.5px solid #a5b4fc !important;
    border-radius: 10px !important;
    font-weight: 700 !important; font-size: 13px !important; }
input:focus, select:focus, textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.18) !important;
    outline: none !important; }

/* ── Number input spin (e.g. simulations box) ── */
input[type="number"] {
    background: #ede9fe !important;
    border: 1.5px solid #c4b5fd !important;
    color: #4338ca !important; font-weight: 800 !important; }

/* ── Slider track ── */
input[type="range"] {
    accent-color: #6366f1 !important;
    background: #ede9fe !important; }

/* ── Radio buttons ── */
input[type="radio"] { accent-color: #6366f1 !important; }

/* ── Accordion ── */
details, .gr-accordion {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe) !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 14px !important; margin-top: 12px !important; }
details summary, .gr-accordion summary {
    color: #4338ca !important; font-weight: 800 !important;
    font-size: 13px !important; }

/* ──────────────────────────────────────────
   HEADER BANNER
   ────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 40%, #059669 100%) !important;
    border-radius: 18px !important; padding: 18px 28px !important; margin-bottom: 22px !important;
    display: flex !important; justify-content: space-between !important; align-items: center !important;
    flex-wrap: wrap !important; gap: 14px !important;
    box-shadow: 0 12px 30px -5px rgba(79,70,229,0.45), 0 4px 12px rgba(0,0,0,0.10) !important; }

.header-title-box { display: flex !important; align-items: center !important; gap: 16px !important; }

.header-logo {
    font-size: 26px !important; background: #81b64c !important;
    width: 52px !important; height: 52px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: 12px !important; color: #fff !important; font-weight: 900 !important;
    box-shadow: 0 5px 0 #457524, 0 8px 20px rgba(0,0,0,0.25) !important; }

.header-title { font-size: 26px !important; font-weight: 900 !important; color: #fff !important;
    margin: 0 !important; letter-spacing: -0.5px !important; }
.header-subtitle { font-size: 13px !important; color: #c7d2fe !important;
    margin: 2px 0 0 !important; font-weight: 600 !important; }

.header-badges { display: flex !important; gap: 8px !important; flex-wrap: wrap !important; }
.badge {
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.35) !important;
    padding: 6px 14px !important; border-radius: 9999px !important;
    font-size: 12px !important; font-weight: 800 !important;
    color: #fff !important; backdrop-filter: blur(6px) !important; }

/* ──────────────────────────────────────────
   LEFT CONTROL CARD — Lavender tint
   ────────────────────────────────────────── */
.control-card {
    background: linear-gradient(160deg, #faf5ff 0%, #f5f3ff 100%) !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 18px !important; padding: 22px !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.12), 0 2px 6px rgba(0,0,0,0.04) !important; }

/* ── Section headings inside card ── */
.control-card h3, .control-card .gr-markdown h3 {
    color: #4338ca !important; font-size: 17px !important;
    font-weight: 900 !important; margin-bottom: 14px !important; }

/* ──────────────────────────────────────────
   BUTTONS
   ────────────────────────────────────────── */

/* Primary — chess.com GREEN */
.chess-btn-green, button.chess-btn-green {
    background: linear-gradient(180deg, #95c862 0%, #81b64c 100%) !important;
    color: #fff !important; font-size: 15px !important; font-weight: 800 !important;
    border-radius: 10px !important; border: none !important;
    box-shadow: 0 5px 0 #4a7c1a, 0 8px 20px rgba(129,182,76,0.40) !important;
    padding: 13px 22px !important; cursor: pointer !important;
    transition: transform 0.08s, box-shadow 0.08s !important; width: 100% !important; }
.chess-btn-green:hover { background: linear-gradient(180deg, #a8d877 0%, #95c862 100%) !important; }
.chess-btn-green:active { transform: translateY(4px) !important; box-shadow: 0 1px 0 #4a7c1a !important; }

/* Secondary — indigo/violet */
.chess-btn-secondary, button.chess-btn-secondary {
    background: linear-gradient(180deg, #818cf8 0%, #6366f1 100%) !important;
    color: #fff !important; font-size: 13px !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important;
    box-shadow: 0 4px 0 #3730a3, 0 6px 16px rgba(99,102,241,0.35) !important;
    padding: 10px 16px !important; cursor: pointer !important;
    transition: transform 0.08s, box-shadow 0.08s !important; width: 100% !important; }
.chess-btn-secondary:hover { background: linear-gradient(180deg, #a5b4fc 0%, #818cf8 100%) !important; }
.chess-btn-secondary:active { transform: translateY(3px) !important; box-shadow: 0 1px 0 #3730a3 !important; }

/* ──────────────────────────────────────────
   GOBAN FRAME & BOARD
   ────────────────────────────────────────── */
.goban-frame {
    background: linear-gradient(135deg, #fdf8f0 0%, #fff8ee 100%) !important;
    border: 1.5px solid #fde68a !important; border-radius: 20px !important;
    padding: 24px 18px !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    box-shadow: 0 10px 30px rgba(245,158,11,0.15), 0 3px 8px rgba(0,0,0,0.06) !important; }

.goban-wood {
    background: radial-gradient(ellipse at 40% 35%, #ffe599 0%, #f6c244 35%, #d4860a 75%, #b05e00 100%) !important;
    border: 10px solid #7c3f0a !important; border-radius: 18px !important;
    box-shadow: 0 16px 40px rgba(124,63,10,0.40),
                inset 0 4px 8px rgba(255,255,255,0.55),
                inset 0 -4px 10px rgba(0,0,0,0.30) !important;
    padding: 16px 22px !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; margin: 0 auto !important; }

.coord-letters { display: flex !important; justify-content: center !important;
    gap: 8px !important; margin: 5px 0 !important; }
.coord-letters span { width: 52px !important; text-align: center !important;
    font-weight: 900 !important; font-size: 14px !important; color: #5c2e00 !important;
    user-select: none !important; }

.goban-row { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
    justify-content: center !important; align-items: center !important;
    gap: 8px !important; margin: 4px 0 !important; }

.goban-row button {
    width: 52px !important; height: 52px !important;
    min-width: 52px !important; max-width: 52px !important;
    min-height: 52px !important; max-height: 52px !important;
    padding: 0 !important; margin: 0 !important; border-radius: 50% !important;
    font-size: 22px !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
    transition: all 0.12s ease !important; border: none !important; }

.goban-row button.empty-cell {
    background: rgba(185, 115, 20, 0.16) !important;
    border: 1.5px dashed rgba(140, 75, 15, 0.50) !important;
    color: #7c3f0a !important; font-weight: 800 !important; }
.goban-row button.empty-cell.tengen {
    border: 2.5px solid rgba(140, 75, 15, 0.9) !important;
    color: #5c2e00 !important; font-weight: 900 !important; }
.goban-row button.empty-cell:hover:not(:disabled) {
    background: rgba(129, 182, 76, 0.45) !important;
    border: 2.5px solid #81b64c !important; transform: scale(1.15) !important;
    box-shadow: 0 0 18px rgba(129,182,76,0.65) !important; cursor: pointer !important; }

/* Black stone — deep indigo */
.goban-row button.stone-black {
    background: radial-gradient(circle at 32% 28%, #6366f1 0%, #312e81 50%, #1e1b4b 100%) !important;
    border-radius: 50% !important;
    box-shadow: 3px 7px 16px rgba(30,27,75,0.60), inset 0 2px 3px rgba(255,255,255,0.40) !important;
    color: transparent !important; }

/* White stone */
.goban-row button.stone-white {
    background: radial-gradient(circle at 32% 28%, #ffffff 0%, #f1f5f9 50%, #cbd5e1 100%) !important;
    border-radius: 50% !important;
    box-shadow: 3px 7px 14px rgba(0,0,0,0.22), inset 0 -2px 4px rgba(0,0,0,0.10), inset 0 1px 3px #fff !important;
    color: transparent !important; }

/* Last-move ring */
.goban-row button.last-move {
    outline: 3.5px solid #f59e0b !important; outline-offset: 3px !important;
    box-shadow: 0 0 16px rgba(245,158,11,0.75) !important; }

/* ──────────────────────────────────────────
   RIGHT SIDEBAR
   ────────────────────────────────────────── */
.chess-sidebar-card {
    background: linear-gradient(160deg, #faf5ff 0%, #f0f9ff 100%) !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 18px !important; padding: 18px !important; margin-bottom: 16px !important;
    box-shadow: 0 6px 24px rgba(124,58,237,0.10) !important; }

.player-header { display: flex !important; align-items: center !important;
    gap: 12px !important; padding: 8px 4px !important; }

.player-avatar { width: 44px !important; height: 44px !important; border-radius: 12px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-size: 22px !important; }
.player-avatar.ai-avatar { background: #ede9fe !important; border: 2px solid #a78bfa !important; }
.player-avatar.human-avatar { background: #d1fae5 !important; border: 2px solid #6ee7b7 !important; }

.player-meta { flex: 1 !important; }
.player-name { font-size: 15px !important; font-weight: 800 !important;
    color: #1e293b !important; line-height: 1.2 !important; }
.player-sub { font-size: 12px !important; color: #6d28d9 !important; font-weight: 600 !important; }

/* Turn badges */
.turn-indicator-badge { padding: 5px 12px !important; border-radius: 8px !important;
    font-size: 12px !important; font-weight: 800 !important; white-space: nowrap !important; }
.turn-indicator-badge.active {
    background: linear-gradient(135deg, #81b64c, #4ade80) !important;
    color: #fff !important; box-shadow: 0 3px 10px rgba(129,182,76,0.45) !important; }
/* Waiting — violet instead of grey */
.turn-indicator-badge.idle {
    background: linear-gradient(135deg, #ede9fe, #ddd6fe) !important;
    color: #6d28d9 !important; border: 1px solid #c4b5fd !important; }
.turn-indicator-badge.winner {
    background: linear-gradient(135deg, #fef08a, #fbbf24) !important;
    color: #713f12 !important; border: 1px solid #f59e0b !important; font-weight: 900 !important; }

/* Eval bar */
.eval-bar-wrapper {
    margin: 14px 0 !important; padding: 12px !important;
    background: linear-gradient(135deg, #f5f3ff, #f0f9ff) !important;
    border-radius: 12px !important; border: 1px solid #c4b5fd !important; }
.eval-bar-label { display: flex !important; justify-content: space-between !important;
    font-size: 12px !important; font-weight: 800 !important;
    color: #4338ca !important; margin-bottom: 7px !important; }
.eval-bar { height: 14px !important; border-radius: 8px !important;
    overflow: hidden !important; display: flex !important; border: 1px solid #a5b4fc !important; }
.eval-black { background: linear-gradient(90deg, #312e81, #4f46e5) !important;
    height: 100% !important; transition: width 0.35s ease !important; }
.eval-white { background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important;
    height: 100% !important; transition: width 0.35s ease !important; }

/* Move table */
.chess-move-table {
    background: #faf5ff !important;
    border: 1.5px solid #c4b5fd !important;
    border-radius: 16px !important; overflow: hidden !important;
    box-shadow: 0 6px 20px rgba(124,58,237,0.08) !important; }
.move-table-header {
    display: grid !important; grid-template-columns: 46px 1fr 1fr !important;
    background: linear-gradient(135deg, #ede9fe, #ddd6fe) !important;
    padding: 10px 14px !important; font-size: 11px !important; font-weight: 800 !important;
    color: #5b21b6 !important; border-bottom: 1.5px solid #c4b5fd !important;
    text-transform: uppercase !important; letter-spacing: 0.7px !important; }
.move-table-body { max-height: 200px !important; overflow-y: auto !important; }
.move-row { display: grid !important; grid-template-columns: 46px 1fr 1fr !important;
    padding: 8px 14px !important; font-size: 13px !important;
    border-bottom: 1px solid #ede9fe !important; }
.move-row:nth-child(even) { background: #f5f3ff !important; }
.move-idx { color: #7c3aed !important; font-weight: 700 !important; }
.move-val { font-weight: 800 !important; color: #1e293b !important; }
.move-val.black-val { color: #4338ca !important; }
.move-val.white-val { color: #0284c7 !important; }
.empty-table-msg { padding: 22px !important; text-align: center !important;
    color: #a78bfa !important; font-size: 13px !important; font-weight: 600 !important; }




/* ── Suppress Gradio dark panel defaults ── */
.gr-panel, .gr-block, .gr-box, .gr-group, .gradio-group,
[class*="block"], [class*="panel"], .contain, .wrap,
footer { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── Labels ── */
label, span.label-text, .gr-form label, [data-testid="block-label"],
.svelte-1gfkn6j { background: transparent !important; color: #475569 !important;
    font-size: 12px !important; font-weight: 700 !important; letter-spacing: 0.5px !important;
    text-transform: uppercase !important; border: none !important; box-shadow: none !important; }

/* ── Gradio inputs/selects/dropdowns ── */
input, select, textarea, .gr-input, .gr-dropdown, [data-testid="dropdown"] {
    background: #ffffff !important; color: #0f172a !important;
    border: 1.5px solid #c7d2fe !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 13px !important; }
input:focus, select:focus, textarea:focus { border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important; outline: none !important; }

/* ── Accordion ── */
details, .gr-accordion { background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important; margin-top: 12px !important; }

/* ──────────────────────────────────────────
   HEADER BANNER
   ────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 40%, #059669 100%) !important;
    border-radius: 18px !important; padding: 18px 28px !important; margin-bottom: 22px !important;
    display: flex !important; justify-content: space-between !important; align-items: center !important;
    flex-wrap: wrap !important; gap: 14px !important;
    box-shadow: 0 12px 30px -5px rgba(79,70,229,0.40), 0 4px 12px rgba(0,0,0,0.08) !important; }

.header-title-box { display: flex !important; align-items: center !important; gap: 16px !important; }

.header-logo {
    font-size: 26px !important; background: #81b64c !important; width: 52px !important; height: 52px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: 12px !important; color: #fff !important; font-weight: 900 !important;
    box-shadow: 0 5px 0 #457524, 0 8px 20px rgba(0,0,0,0.25) !important; }

.header-title { font-size: 26px !important; font-weight: 900 !important; color: #fff !important;
    margin: 0 !important; letter-spacing: -0.5px !important; text-shadow: 0 2px 8px rgba(0,0,0,0.2) !important; }
.header-subtitle { font-size: 13px !important; color: #c7d2fe !important; margin: 2px 0 0 !important; font-weight: 600 !important; }

.header-badges { display: flex !important; gap: 8px !important; flex-wrap: wrap !important; }
.badge { background: rgba(255,255,255,0.18) !important; border: 1px solid rgba(255,255,255,0.35) !important;
    padding: 6px 14px !important; border-radius: 9999px !important; font-size: 12px !important;
    font-weight: 800 !important; color: #ffffff !important; backdrop-filter: blur(6px) !important; }

/* ──────────────────────────────────────────
   LEFT CONTROL CARD
   ────────────────────────────────────────── */
.control-card {
    background: #ffffff !important; border: 1.5px solid #e0e7ff !important;
    border-radius: 18px !important; padding: 22px !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.08), 0 2px 6px rgba(0,0,0,0.04) !important; }

/* ──────────────────────────────────────────
   BUTTONS  (Chess.com style)
   ────────────────────────────────────────── */

/* Primary — chess.com GREEN */
.chess-btn-green, button.chess-btn-green {
    background: linear-gradient(180deg, #95c862 0%, #81b64c 100%) !important;
    color: #fff !important; font-size: 15px !important; font-weight: 800 !important;
    border-radius: 10px !important; border: none !important;
    box-shadow: 0 5px 0 #4a7c1a, 0 8px 20px rgba(129,182,76,0.35) !important;
    padding: 13px 22px !important; cursor: pointer !important;
    transition: transform 0.08s, box-shadow 0.08s !important; width: 100% !important; }
.chess-btn-green:hover { background: linear-gradient(180deg, #a8d877 0%, #95c862 100%) !important; }
.chess-btn-green:active { transform: translateY(4px) !important; box-shadow: 0 1px 0 #4a7c1a !important; }

/* Secondary — colorful indigo/violet */
.chess-btn-secondary, button.chess-btn-secondary {
    background: linear-gradient(180deg, #818cf8 0%, #6366f1 100%) !important;
    color: #ffffff !important; font-size: 13px !important; font-weight: 700 !important;
    border-radius: 10px !important; border: none !important;
    box-shadow: 0 4px 0 #3730a3, 0 6px 16px rgba(99,102,241,0.35) !important;
    padding: 10px 16px !important;
    cursor: pointer !important; transition: transform 0.08s, box-shadow 0.08s !important; width: 100% !important; }
.chess-btn-secondary:hover { background: linear-gradient(180deg, #a5b4fc 0%, #818cf8 100%) !important; box-shadow: 0 4px 0 #3730a3, 0 8px 20px rgba(99,102,241,0.45) !important; }
.chess-btn-secondary:active { transform: translateY(3px) !important; box-shadow: 0 1px 0 #3730a3 !important; }

/* ──────────────────────────────────────────
   GOBAN BOARD
   ────────────────────────────────────────── */
.goban-frame {
    background: linear-gradient(135deg, #fdf8f0 0%, #fff8ee 100%) !important;
    border: 1.5px solid #fde68a !important; border-radius: 20px !important;
    padding: 24px 18px !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important;
    box-shadow: 0 10px 30px rgba(245,158,11,0.15), 0 3px 8px rgba(0,0,0,0.06) !important; }

.goban-wood {
    background: radial-gradient(ellipse at 40% 35%, #ffe599 0%, #f6c244 35%, #d4860a 75%, #b05e00 100%) !important;
    border: 10px solid #7c3f0a !important; border-radius: 18px !important;
    box-shadow: 0 16px 40px rgba(124,63,10,0.40), inset 0 4px 8px rgba(255,255,255,0.55),
                inset 0 -4px 10px rgba(0,0,0,0.30) !important;
    padding: 16px 22px !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; margin: 0 auto !important; }

.coord-letters { display: flex !important; justify-content: center !important;
    gap: 8px !important; margin: 5px 0 !important; }
.coord-letters span { width: 52px !important; text-align: center !important;
    font-weight: 900 !important; font-size: 14px !important; color: #5c2e00 !important;
    user-select: none !important; }

.goban-row { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
    justify-content: center !important; align-items: center !important;
    gap: 8px !important; margin: 4px 0 !important; }

.goban-row button {
    width: 52px !important; height: 52px !important; min-width: 52px !important;
    max-width: 52px !important; min-height: 52px !important; max-height: 52px !important;
    padding: 0 !important; margin: 0 !important; border-radius: 50% !important;
    font-size: 22px !important; display: flex !important; align-items: center !important;
    justify-content: center !important; transition: all 0.12s ease !important; border: none !important; }

/* Empty cell */
.goban-row button.empty-cell {
    background: rgba(185, 115, 20, 0.16) !important;
    border: 1.5px dashed rgba(140, 75, 15, 0.50) !important;
    color: #7c3f0a !important; font-weight: 800 !important; }
.goban-row button.empty-cell.tengen {
    border: 2.5px solid rgba(140, 75, 15, 0.9) !important;
    color: #5c2e00 !important; font-weight: 900 !important; }
.goban-row button.empty-cell:hover:not(:disabled) {
    background: rgba(129, 182, 76, 0.45) !important;
    border: 2.5px solid #81b64c !important; transform: scale(1.15) !important;
    box-shadow: 0 0 18px rgba(129,182,76,0.65) !important; cursor: pointer !important; }

/* Black stone — deep indigo (not pitch black) */
.goban-row button.stone-black {
    background: radial-gradient(circle at 32% 28%, #6366f1 0%, #312e81 50%, #1e1b4b 100%) !important;
    border-radius: 50% !important;
    box-shadow: 3px 7px 16px rgba(30,27,75,0.60), inset 0 2px 3px rgba(255,255,255,0.40) !important;
    color: transparent !important; }

/* White stone */
.goban-row button.stone-white {
    background: radial-gradient(circle at 32% 28%, #ffffff 0%, #f1f5f9 50%, #cbd5e1 100%) !important;
    border-radius: 50% !important;
    box-shadow: 3px 7px 14px rgba(0,0,0,0.22), inset 0 -2px 4px rgba(0,0,0,0.10), inset 0 1px 3px #fff !important;
    color: transparent !important; }

/* Last-move ring */
.goban-row button.last-move {
    outline: 3.5px solid #f59e0b !important; outline-offset: 3px !important;
    box-shadow: 0 0 16px rgba(245,158,11,0.75) !important; }

/* ──────────────────────────────────────────
   RIGHT SIDEBAR CARDS
   ────────────────────────────────────────── */
.chess-sidebar-card {
    background: #ffffff !important; border: 1.5px solid #e0e7ff !important;
    border-radius: 18px !important; padding: 18px !important; margin-bottom: 16px !important;
    box-shadow: 0 6px 24px rgba(99,102,241,0.08) !important; }

.player-header { display: flex !important; align-items: center !important;
    gap: 12px !important; padding: 8px 4px !important; }

.player-avatar { width: 44px !important; height: 44px !important; border-radius: 12px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-size: 22px !important; }
.player-avatar.ai-avatar { background: #ede9fe !important; border: 2px solid #a78bfa !important; }
.player-avatar.human-avatar { background: #d1fae5 !important; border: 2px solid #6ee7b7 !important; }

.player-meta { flex: 1 !important; }
.player-name { font-size: 15px !important; font-weight: 800 !important; color: #1e293b !important; line-height: 1.2 !important; }
.player-sub { font-size: 12px !important; color: #64748b !important; font-weight: 600 !important; }

.turn-indicator-badge { padding: 5px 12px !important; border-radius: 8px !important;
    font-size: 12px !important; font-weight: 800 !important; white-space: nowrap !important; }
.turn-indicator-badge.active { background: linear-gradient(135deg, #81b64c, #4ade80) !important;
    color: #fff !important; box-shadow: 0 3px 10px rgba(129,182,76,0.45) !important; }
.turn-indicator-badge.idle { background: #f1f5f9 !important; color: #94a3b8 !important; }
.turn-indicator-badge.winner { background: linear-gradient(135deg, #fef08a, #fbbf24) !important;
    color: #713f12 !important; border: 1px solid #f59e0b !important; font-weight: 900 !important; }

/* Eval bar */
.eval-bar-wrapper { margin: 14px 0 !important; padding: 12px !important;
    background: linear-gradient(135deg, #f8fafc, #f0f9ff) !important;
    border-radius: 12px !important; border: 1px solid #e0e7ff !important; }
.eval-bar-label { display: flex !important; justify-content: space-between !important;
    font-size: 12px !important; font-weight: 800 !important; color: #475569 !important; margin-bottom: 7px !important; }
.eval-bar { height: 14px !important; background: #e2e8f0 !important; border-radius: 8px !important;
    overflow: hidden !important; display: flex !important; border: 1px solid #cbd5e1 !important; }
.eval-black { background: linear-gradient(90deg, #312e81, #4f46e5) !important;
    height: 100% !important; transition: width 0.35s ease !important; }
.eval-white { background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important;
    height: 100% !important; transition: width 0.35s ease !important; }

/* Move table */
.chess-move-table { background: #ffffff !important; border: 1.5px solid #e0e7ff !important;
    border-radius: 16px !important; overflow: hidden !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.07) !important; }
.move-table-header { display: grid !important; grid-template-columns: 46px 1fr 1fr !important;
    background: linear-gradient(135deg, #f0f4ff, #f8fafc) !important;
    padding: 10px 14px !important; font-size: 11px !important; font-weight: 800 !important;
    color: #6366f1 !important; border-bottom: 1.5px solid #e0e7ff !important;
    text-transform: uppercase !important; letter-spacing: 0.6px !important; }
.move-table-body { max-height: 200px !important; overflow-y: auto !important; }
.move-row { display: grid !important; grid-template-columns: 46px 1fr 1fr !important;
    padding: 8px 14px !important; font-size: 13px !important;
    border-bottom: 1px solid #f1f5f9 !important; }
.move-row:nth-child(even) { background: #fafbff !important; }
.move-idx { color: #94a3b8 !important; font-weight: 700 !important; }
.move-val { font-weight: 800 !important; color: #1e293b !important; }
.move-val.black-val { color: #4338ca !important; }
.move-val.white-val { color: #0284c7 !important; }
.empty-table-msg { padding: 22px !important; text-align: center !important;
    color: #94a3b8 !important; font-size: 13px !important; font-weight: 600 !important; }
"""


def build_app():
    _require_gradio()
    theme = gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="purple",
    ).set(
        body_background_fill="#eef2ff",
        block_background_fill="#faf5ff",
        block_border_width="1.5px",
        block_border_color="#c4b5fd",
        button_secondary_background_fill="#6366f1",
        button_secondary_text_color="#ffffff",
        slider_color="#6366f1",
    )

    with gr.Blocks(title="MiniAlphaGo", theme=theme, css=CUSTOM_CSS) as demo:
        # Colorful Chess.com Header
        gr.HTML(
            """
            <div class="app-header">
                <div class="header-title-box">
                    <div class="header-logo">碁</div>
                    <div>
                        <h1 class="header-title">MiniAlphaGo</h1>
                        <p class="header-subtitle">AlphaZero Deep RL • 5×5 Go AI Engine</p>
                    </div>
                </div>
                <div class="header-badges">
                    <span class="badge">● Engine Ready</span>
                    <span class="badge">Neural MCTS</span>
                    <span class="badge">Komi 7.5</span>
                </div>
            </div>
            """
        )

        session = gr.State(None)

        with gr.Row():
            # LEFT COLUMN: Match Settings & Actions
            with gr.Column(scale=3):
                with gr.Group(elem_classes=["control-card"]):
                    gr.Markdown("### Match Settings")
                    checkpoint = gr.Dropdown(
                        available_checkpoints(),
                        value=available_checkpoints()[0],
                        label="AI Checkpoint",
                        allow_custom_value=True,
                    )
                    mode = gr.Radio(
                        ["Human vs AI", "AI vs AI"],
                        value="Human vs AI",
                        label="Game Mode",
                    )
                    color = gr.Radio(
                        ["Black", "White"],
                        value="Black",
                        label="Play As (Black moves first)",
                    )
                    simulations = gr.Slider(
                        minimum=5,
                        maximum=400,
                        value=50,
                        step=5,
                        label="Engine Simulations",
                        info="50 = Fast, 150+ = Master strength",
                    )

                    with gr.Row():
                        new_btn = gr.Button("Play Game", variant="primary", elem_classes=["chess-btn-green"])
                        pass_btn = gr.Button("Pass Turn", elem_classes=["chess-btn-secondary"])
                    step_btn = gr.Button("AI Move (Step)", elem_classes=["chess-btn-secondary"])

                # Quick Rules
                with gr.Accordion("5×5 Go Rules", open=False):
                    gr.Markdown(
                        """
                        - **Objective**: Control more territory + living stones than opponent.
                        - **Black moves first**: White receives **7.5 Komi** compensation.
                        - **Captures**: Stones surrounded with 0 liberties are removed.
                        - **Passing**: Two consecutive passes conclude the match.
                        """
                    )

            # CENTER COLUMN: The Goban (Go Board)
            with gr.Column(scale=5):
                with gr.Group(elem_classes=["goban-frame"]):
                    with gr.Group(elem_classes=["goban-wood"]):
                        # Top Coordinates
                        gr.HTML('<div class="coord-letters"><span>A</span><span>B</span><span>C</span><span>D</span><span>E</span></div>')

                        buttons = []
                        for row in range(5):
                            with gr.Row(elem_classes=["goban-row"]):
                                for col in range(5):
                                    action = row * 5 + col
                                    init_label = "✦" if action == 12 else "·"
                                    classes = ["empty-cell", "tengen"] if action == 12 else ["empty-cell"]
                                    buttons.append(gr.Button(init_label, elem_classes=classes))

                        # Bottom Coordinates
                        gr.HTML('<div class="coord-letters"><span>A</span><span>B</span><span>C</span><span>D</span><span>E</span></div>')

                    gr.HTML('<div style="margin-top: 10px; font-size: 13px; color: #64748b; text-align: center; font-weight: 600;">Click any square to play. Black moves first.</div>')

            # RIGHT COLUMN: Chess.com Style Match Hub
            with gr.Column(scale=4):
                sidebar_html = gr.HTML()
                history_html = gr.HTML()

                with gr.Accordion("Raw Match Stats", open=False):
                    status_txt = gr.Textbox(label="Status", interactive=False)
                    current_txt = gr.Textbox(label="Current player", interactive=False)
                    black_score_txt = gr.Textbox(label="Black score", interactive=False)
                    white_score_txt = gr.Textbox(label="White score", interactive=False)
                    komi_txt = gr.Textbox(label="Komi", interactive=False)
                    winner_txt = gr.Textbox(label="Winner", interactive=False)

        # Output collection
        outputs = buttons + [
            session,
            sidebar_html,
            history_html,
            status_txt,
            current_txt,
            black_score_txt,
            white_score_txt,
            komi_txt,
            winner_txt,
        ]

        cfg_inputs = [checkpoint, color, mode, simulations]

        # Bind events
        new_btn.click(new_game, cfg_inputs, outputs)
        pass_btn.click(lambda s, cp, cl, md, sm: human_action(s, 25, cp, cl, md, sm), [session] + cfg_inputs, outputs)
        step_btn.click(ai_step, [session] + cfg_inputs, outputs)

        for action, button in enumerate(buttons):
            button.click(
                lambda s, cp, cl, md, sm, a=action: human_action(s, a, cp, cl, md, sm),
                [session] + cfg_inputs,
                outputs,
            )

        # Auto-initialize game on initial load so the board is immediately playable
        demo.load(new_game, cfg_inputs, outputs)

    return demo


if __name__ == "__main__":
    build_app().launch()
