"""Chess.com Light & Colorful modern interface for MiniAlphaGo 5×5 Go AI Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

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
    best_pt = Path("checkpoints/best_model.pt")
    zip_path = Path("checkpoints/minialphago-trained.zip")
    if not best_pt.exists() and zip_path.exists():
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(".")
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
    is_ai_vs_ai = (session.mode == "AI vs AI")
    current_player_name = "Black" if env.current_player == 1 else "White"
    human_color_name = "Black" if session.human_color == 1 else "White"
    ai_color_name = "White" if session.human_color == 1 else "Black"
    human_turn = (not is_ai_vs_ai) and (env.current_player == session.human_color)

    winner_name = "—"
    if env.done:
        w = env.winner()
        winner_name = "Black" if w == 1 else "White" if w == -1 else "Draw"
        current_display = "Game over"
    else:
        current_display = f"{current_player_name} (AI)" if is_ai_vs_ai else current_player_name

    if message:
        status_text = message
    elif env.done:
        status_text = f"Game over. Winner: {winner_name}"
    elif is_ai_vs_ai:
        status_text = f"AI vs AI: {current_player_name}'s turn. Click 'AI Step' or 'Auto-Play Match'."
    elif human_turn:
        status_text = "Your turn."
    else:
        status_text = "AI ready / thinking."

    total_score = max(score.black + score.white, 0.1)
    black_pct = min(max((score.black / total_score) * 100, 5), 95)
    white_pct = 100 - black_pct

    if env.done:
        margin = abs(score.black - score.white)
        ai_badge = ""
        human_badge = f'<div class="turn-indicator-badge winner">🏆 {winner_name} won (+{margin:.1f})</div>'
    elif is_ai_vs_ai:
        if env.current_player == -1:  # White's turn
            ai_badge = '<div class="turn-indicator-badge active">Thinking...</div>'
            human_badge = '<div class="turn-indicator-badge idle">Waiting</div>'
        else:  # Black's turn
            ai_badge = '<div class="turn-indicator-badge idle">Waiting</div>'
            human_badge = '<div class="turn-indicator-badge active">Thinking...</div>'
    elif human_turn:
        human_badge = '<div class="turn-indicator-badge active">Your Turn</div>'
        ai_badge = '<div class="turn-indicator-badge idle">Waiting</div>'
    else:
        human_badge = '<div class="turn-indicator-badge idle">Waiting</div>'
        ai_badge = '<div class="turn-indicator-badge active">Thinking...</div>'

    top_name = "MiniAlphaGo AI (White)" if is_ai_vs_ai else "MiniAlphaGo AI"
    top_sub = f"White • {score.white:.1f} pts (incl. +{env.komi:g} Komi)" if is_ai_vs_ai else f"{ai_color_name} • {score.white:.1f} pts (incl. +{env.komi:g} Komi)"

    bottom_avatar = "🤖" if is_ai_vs_ai else "👤"
    bottom_name = "MiniAlphaGo AI (Black)" if is_ai_vs_ai else "You"
    bottom_sub = f"Black • {score.black:.1f} pts" if is_ai_vs_ai else f"{human_color_name} • {score.black:.1f} pts"

    sidebar_html = f"""
    <div class="chess-sidebar-card">
        <!-- Top Player Card (White) -->
        <div class="player-header">
            <div class="player-avatar ai-avatar">🤖</div>
            <div class="player-meta">
                <div class="player-name">{top_name}</div>
                <div class="player-sub">{top_sub}</div>
            </div>
            {ai_badge}
        </div>

        <!-- Evaluation Bar -->
        <div class="eval-bar-wrapper">
            <div class="eval-bar-label">
                <span style="color:#1a1a1a; font-weight:800;">Black {score.black:.1f} ({black_pct:.0f}%)</span>
                <span style="color:#555555; font-weight:800;">White {score.white:.1f} ({white_pct:.0f}%)</span>
            </div>
            <div class="eval-bar">
                <div class="eval-black" style="width: {black_pct:.1f}%;"></div>
                <div class="eval-white" style="width: {white_pct:.1f}%;"></div>
            </div>
        </div>

        <!-- Bottom Player Card (Black / Human) -->
        <div class="player-header">
            <div class="player-avatar human-avatar">{bottom_avatar}</div>
            <div class="player-meta">
                <div class="player-name">{bottom_name}</div>
                <div class="player-sub">{bottom_sub}</div>
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
    if mode == "AI vs AI":
        return render(session, "New AI vs AI game ready! Click 'AI Move (Step)' or 'Auto-Play Match' to begin.")
    return render(session, "New game started! Black plays first.")


def human_action(session, action, checkpoint, color, mode, simulations):
    if session is None or session.mode != mode:
        session = create_session(checkpoint, color, mode, simulations)

    if session.mode == "AI vs AI":
        # In AI vs AI mode, clicking any square advances an AI move
        return ai_step(session, checkpoint, color, mode, simulations)

    env = session.env
    if env.current_player != session.human_color:
        return render(session, "It is not your turn.")

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
    if session is None or session.mode != "AI vs AI":
        session = create_session(checkpoint, color, "AI vs AI", simulations)
    session.mode = "AI vs AI"

    if not session.env.done:
        color_name = "Black" if session.env.current_player == 1 else "White"
        action = session.ai.select_action(session.env)
        session.env.step(action)
        coord_name = COORD_NAMES[action] if action < 25 else "PASS"
        session.move_log.append(f"#{len(session.move_log) + 1} {color_name} (AI) → {coord_name}")
        msg = f"AI {color_name} played {coord_name}." if not session.env.done else "Game finished!"
        return render(session, msg)
    return render(session, "Game is already finished.")


def ai_autoplay(session, checkpoint, color, mode, simulations):
    if session is None or session.mode != "AI vs AI":
        session = create_session(checkpoint, color, "AI vs AI", simulations)
    session.mode = "AI vs AI"

    if session.env.done:
        session = create_session(checkpoint, color, "AI vs AI", simulations)
        session.mode = "AI vs AI"

    yield render(session, "AI vs AI match in progress...")
    max_steps = 60
    while not session.env.done and len(session.move_log) < max_steps:
        color_name = "Black" if session.env.current_player == 1 else "White"
        action = session.ai.select_action(session.env)
        session.env.step(action)
        coord_name = COORD_NAMES[action] if action < 25 else "PASS"
        session.move_log.append(f"#{len(session.move_log) + 1} {color_name} (AI) → {coord_name}")
        yield render(session, f"AI {color_name} played {coord_name}.")
        time.sleep(0.12)

    if session.env.done:
        score = session.env.score()
        w = session.env.winner()
        winner_name = "Black" if w == 1 else "White" if w == -1 else "Draw"
        yield render(session, f"Match finished! Winner: {winner_name} (Black {score.black:.1f} vs White {score.white:.1f}).")


def pass_action(session, checkpoint, color, mode, simulations):
    if session is None or session.mode != mode:
        session = create_session(checkpoint, color, mode, simulations)
    if session.mode == "AI vs AI":
        if not session.env.done:
            color_name = "Black" if session.env.current_player == 1 else "White"
            session.env.step(25)
            session.move_log.append(f"#{len(session.move_log) + 1} {color_name} (AI) → PASS")
            return render(session, f"AI {color_name} passed.")
        return render(session, "Game is already finished.")
    return human_action(session, 25, checkpoint, color, mode, simulations)


CUSTOM_CSS = """
/* ==========================================================================
   MiniAlphaGo — Authentic Go Board Style (matches reference image)
   ========================================================================== */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

*, *::before, *::after { box-sizing: border-box !important; }

/* ── Page background — clean white like reference image ── */
body, .gradio-container, .gradio-container > .main {
    max-width: 1420px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    background: #ffffff !important;
    background-color: #ffffff !important;
    color: #1a1a1a !important;
    min-height: 100vh !important;
}

/* ── Suppress Gradio dark defaults ── */
.gr-panel, .gr-block, .gr-box, .gr-group, .gradio-group,
[class*="block"], [class*="panel"], .contain, .wrap,
footer { background: transparent !important; border: none !important; box-shadow: none !important; }

/* ── Labels ── */
label, span.label-text, .gr-form label, [data-testid="block-label"],
.svelte-1gfkn6j {
    background: transparent !important; color: #555 !important;
    font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 0.5px !important; text-transform: uppercase !important;
    border: none !important; box-shadow: none !important; }

/* ── Inputs / Dropdowns ── */
input, select, textarea, .gr-input, .gr-dropdown,
[data-testid="dropdown"], .svelte-select {
    background: #fff !important;
    color: #1a1a1a !important;
    border: 1.5px solid #ccc !important;
    border-radius: 8px !important;
    font-weight: 600 !important; font-size: 13px !important; }
input:focus, select:focus, textarea:focus {
    border-color: #e8a000 !important;
    box-shadow: 0 0 0 3px rgba(232,160,0,0.18) !important;
    outline: none !important; }

input[type="number"] {
    background: #fff !important;
    border: 1.5px solid #ccc !important;
    color: #1a1a1a !important; font-weight: 700 !important; }

input[type="range"] { accent-color: #e8a000 !important; }
input[type="radio"] { accent-color: #e8a000 !important; }

/* ── Accordion ── */
details, .gr-accordion {
    background: #fff !important;
    border: 1px solid #ddd !important;
    border-radius: 10px !important; margin-top: 12px !important; }
details summary, .gr-accordion summary {
    color: #333 !important; font-weight: 700 !important; font-size: 13px !important; }

/* ──────────────────────────────────────────
   HEADER BANNER
   ────────────────────────────────────────── */
.app-header {
    background: #1a1a1a !important;
    border-radius: 14px !important; padding: 16px 24px !important; margin-bottom: 20px !important;
    display: flex !important; justify-content: space-between !important; align-items: center !important;
    flex-wrap: wrap !important; gap: 12px !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.30) !important; }

.header-title-box { display: flex !important; align-items: center !important; gap: 14px !important; }

.header-logo {
    font-size: 24px !important;
    background: #e8a000 !important;
    width: 48px !important; height: 48px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    border-radius: 10px !important; color: #fff !important; font-weight: 900 !important;
    box-shadow: 0 3px 0 #a06c00 !important; }

.header-title { font-size: 24px !important; font-weight: 900 !important; color: #fff !important;
    margin: 0 !important; letter-spacing: -0.3px !important; }
.header-subtitle { font-size: 12px !important; color: #aaa !important;
    margin: 2px 0 0 !important; font-weight: 500 !important; }

.header-badges { display: flex !important; gap: 8px !important; flex-wrap: wrap !important; }
.badge {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    padding: 5px 12px !important; border-radius: 9999px !important;
    font-size: 12px !important; font-weight: 700 !important;
    color: #eee !important; }

/* ──────────────────────────────────────────
   LEFT CONTROL CARD — clean white
   ────────────────────────────────────────── */
.control-card,
.control-card > *,
.control-card .form,
.control-card .block,
.control-card .gr-form,
.control-card .gr-panel,
.control-card .gr-box,
.control-card .gr-group {
    background: #ffffff !important;
    background-color: #ffffff !important;
}

.control-card {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 14px !important; padding: 20px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important; }

.control-card h3,
.control-card .prose,
.control-card .markdown {
    background: transparent !important;
    background-color: transparent !important;
    color: #1a1a1a !important; font-size: 16px !important;
    font-weight: 800 !important; margin-bottom: 14px !important; }

/* ──────────────────────────────────────────
   BUTTONS — simple & clean like reference
   ────────────────────────────────────────── */

/* Primary — amber/gold (matches board colour) */
.chess-btn-green, button.chess-btn-green {
    background: #e8a000 !important;
    color: #fff !important; font-size: 15px !important; font-weight: 800 !important;
    border-radius: 8px !important; border: none !important;
    box-shadow: 0 3px 0 #a06c00 !important;
    padding: 12px 20px !important; cursor: pointer !important;
    transition: transform 0.08s, box-shadow 0.08s !important; width: 100% !important;
    letter-spacing: 0.3px !important; }
.chess-btn-green:hover { background: #f5b000 !important; }
.chess-btn-green:active { transform: translateY(3px) !important; box-shadow: 0 0 0 #a06c00 !important; }

/* Secondary — dark/charcoal like "PASS" / "RESIGN" in reference */
.chess-btn-secondary, button.chess-btn-secondary {
    background: #333 !important;
    color: #fff !important; font-size: 13px !important; font-weight: 700 !important;
    border-radius: 8px !important; border: none !important;
    box-shadow: 0 3px 0 #111 !important;
    padding: 10px 16px !important; cursor: pointer !important;
    transition: transform 0.08s, box-shadow 0.08s !important; width: 100% !important;
    letter-spacing: 0.5px !important; }
.chess-btn-secondary:hover { background: #444 !important; }
.chess-btn-secondary:active { transform: translateY(3px) !important; box-shadow: 0 0 0 #111 !important; }

/* ──────────────────────────────────────────
   GOBAN FRAME & BOARD — matches reference exactly
   ────────────────────────────────────────── */
.goban-frame {
    background: #ffffff !important;
    background-color: #ffffff !important;
    border: none !important; border-radius: 16px !important;
    padding: 20px !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; justify-content: center !important; }

/* The golden amber board — exactly like reference image */
.goban-wood {
    background: #e8a000 !important;
    background-color: #e8a000 !important;
    border: none !important; border-radius: 4px !important;
    box-shadow: 0 6px 24px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.20) !important;
    padding: 20px !important; display: flex !important; flex-direction: column !important;
    align-items: center !important; margin: 0 auto !important; }

.coord-letters { display: flex !important; justify-content: center !important;
    gap: 8px !important; margin: 4px 0 !important; }
.coord-letters span { width: 52px !important; text-align: center !important;
    font-weight: 800 !important; font-size: 13px !important; color: #1a1a1a !important;
    user-select: none !important; }

.goban-row { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
    justify-content: center !important; align-items: center !important;
    gap: 0px !important; margin: 0 !important;
    background: transparent !important; background-color: transparent !important; }

.goban-row button,
.goban-wood button,
button.empty-cell,
.empty-cell {
    width: 52px !important; height: 52px !important;
    min-width: 52px !important; max-width: 52px !important;
    min-height: 52px !important; max-height: 52px !important;
    padding: 0 !important; margin: 0 !important; border-radius: 0 !important;
    font-size: 22px !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
    transition: all 0.10s ease !important;
    background: #e8a000 !important;
    background-color: #e8a000 !important;
    border: 1px solid rgba(0,0,0,0.35) !important; }

/* Empty intersections — amber board */
.goban-row button.empty-cell,
.goban-wood button.empty-cell {
    background: #e8a000 !important;
    background-color: #e8a000 !important;
    border: 1px solid rgba(0,0,0,0.35) !important;
    color: transparent !important; }
.goban-row button.empty-cell.tengen,
.goban-wood button.empty-cell.tengen {
    border: 1px solid rgba(0,0,0,0.35) !important; }
.goban-row button.empty-cell:hover:not(:disabled),
.goban-wood button.empty-cell:hover:not(:disabled) {
    background: #d49200 !important;
    background-color: #d49200 !important;
    cursor: pointer !important; }

/* Black stone — pure black like real Go */
.goban-row button.stone-black {
    background: radial-gradient(circle at 35% 30%, #555 0%, #111 50%, #000 100%) !important;
    border-radius: 50% !important;
    border: none !important;
    box-shadow: 2px 5px 12px rgba(0,0,0,0.55), inset 0 1px 2px rgba(255,255,255,0.25) !important;
    color: transparent !important; }

/* White stone — pure white like real Go */
.goban-row button.stone-white {
    background: radial-gradient(circle at 35% 30%, #ffffff 0%, #e8e8e8 60%, #ccc 100%) !important;
    border-radius: 50% !important;
    border: 1px solid #bbb !important;
    box-shadow: 2px 5px 10px rgba(0,0,0,0.25), inset 0 -1px 3px rgba(0,0,0,0.10) !important;
    color: transparent !important; }

/* Last-move marker */
.goban-row button.last-move {
    outline: 2.5px solid rgba(255,255,255,0.8) !important; outline-offset: 2px !important; }

/* ──────────────────────────────────────────
   RIGHT SIDEBAR CARDS — white, clean
   ────────────────────────────────────────── */
.chess-sidebar-card {
    background: #ffffff !important;
    border: 1px solid #ddd !important;
    border-radius: 14px !important; padding: 16px !important; margin-bottom: 14px !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07) !important; }

.player-header { display: flex !important; align-items: center !important;
    gap: 12px !important; padding: 6px 2px !important; }

.player-avatar { width: 42px !important; height: 42px !important; border-radius: 10px !important;
    display: flex !important; align-items: center !important; justify-content: center !important;
    font-size: 20px !important; }
.player-avatar.ai-avatar { background: #f5f5f5 !important; border: 1.5px solid #ccc !important; }
.player-avatar.human-avatar { background: #f5f5f5 !important; border: 1.5px solid #ccc !important; }

.player-meta { flex: 1 !important; }
.player-name { font-size: 15px !important; font-weight: 800 !important;
    color: #1a1a1a !important; line-height: 1.2 !important; }
.player-sub { font-size: 12px !important; color: #666 !important; font-weight: 500 !important; }

/* Turn badges */
.turn-indicator-badge { padding: 5px 12px !important; border-radius: 6px !important;
    font-size: 12px !important; font-weight: 800 !important; white-space: nowrap !important; }
.turn-indicator-badge.active {
    background: #e8a000 !important;
    color: #fff !important; box-shadow: 0 2px 8px rgba(232,160,0,0.40) !important; }
.turn-indicator-badge.idle {
    background: #f0f0f0 !important;
    color: #888 !important; border: 1px solid #ddd !important; }
.turn-indicator-badge.winner {
    background: #1a1a1a !important;
    color: #e8a000 !important; font-weight: 900 !important; }

/* Eval bar */
.eval-bar-wrapper {
    margin: 12px 0 !important; padding: 10px !important;
    background: #f5f5f5 !important;
    border-radius: 10px !important; border: 1px solid #ddd !important; }
.eval-bar-label { display: flex !important; justify-content: space-between !important;
    font-size: 12px !important; font-weight: 700 !important;
    color: #333 !important; margin-bottom: 6px !important; }
.eval-bar { height: 12px !important; border-radius: 6px !important;
    overflow: hidden !important; display: flex !important; border: 1px solid #ccc !important; }
.eval-black { background: #1a1a1a !important;
    height: 100% !important; transition: width 0.35s ease !important; }
.eval-white { background: #e8e8e8 !important; border-left: 1px solid #ccc !important;
    height: 100% !important; transition: width 0.35s ease !important; }

/* Move table */
.chess-move-table {
    background: #ffffff !important;
    border: 1px solid #ddd !important;
    border-radius: 12px !important; overflow: hidden !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important; }
.move-table-header {
    display: grid !important; grid-template-columns: 46px 1fr 1fr !important;
    background: #f5f5f5 !important;
    padding: 9px 14px !important; font-size: 11px !important; font-weight: 800 !important;
    color: #333 !important; border-bottom: 1px solid #ddd !important;
    text-transform: uppercase !important; letter-spacing: 0.6px !important; }
.move-table-body { max-height: 200px !important; overflow-y: auto !important; }
.move-row { display: grid !important; grid-template-columns: 46px 1fr 1fr !important;
    padding: 8px 14px !important; font-size: 13px !important;
    border-bottom: 1px solid #f0f0f0 !important; }
.move-row:nth-child(even) { background: #fafafa !important; }
.move-idx { color: #999 !important; font-weight: 700 !important; }
.move-val { font-weight: 700 !important; color: #1a1a1a !important; }
.move-val.black-val { color: #1a1a1a !important; }
.move-val.white-val { color: #666 !important; }
.empty-table-msg { padding: 20px !important; text-align: center !important;
    color: #aaa !important; font-size: 13px !important; font-weight: 500 !important; }
"""


def build_app():
    _require_gradio()
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="stone",
        neutral_hue="stone",
    ).set(
        body_background_fill="#ffffff",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_border_color="#e5e7eb",
        button_secondary_background_fill="#333333",
        button_secondary_text_color="#ffffff",
        slider_color="#e8a000",
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
                    with gr.Row():
                        step_btn = gr.Button("AI Move (Step)", elem_classes=["chess-btn-secondary"])
                        autoplay_btn = gr.Button("Auto-Play Match", variant="primary", elem_classes=["chess-btn-green"])

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
        pass_btn.click(pass_action, [session] + cfg_inputs, outputs)
        step_btn.click(ai_step, [session] + cfg_inputs, outputs)
        autoplay_btn.click(ai_autoplay, [session] + cfg_inputs, outputs)
        mode.change(new_game, cfg_inputs, outputs)

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
    import os
    port = int(os.environ.get("PORT", 7860))
    share = os.environ.get("SHARE", "false").lower() in ("true", "1")
    build_app().launch(server_name="0.0.0.0", server_port=port, share=share)
