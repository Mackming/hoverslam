"""
A proper menu UI tying the whole project together: play manually, watch a
trained run play, watch a run evolve across every saved checkpoint, or kick
off a brand-new training run from scratch -- all as clickable buttons in one
pygame app, instead of remembering separate CLI incantations across
rocket_lander.py / watch_agent.py / train.py.

Usage:
    python launcher.py
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import pygame
except ImportError:
    pygame = None

from rocket_lander import WINDOW_H, WINDOW_W, play_human
from watch_agent import watch, watch_checkpoint_sequence

MODELS_DIR = Path("models")
LOGS_DIR = Path("logs/launcher")

BG_COLOR = (10, 12, 24)
BUTTON_COLOR = (40, 46, 74)
BUTTON_HOVER = (60, 68, 104)
BUTTON_BORDER = (90, 100, 140)
TEXT_COLOR = (225, 228, 235)
MUTED_TEXT = (150, 155, 175)
ACCENT = (90, 200, 255)


def list_runs():
    """Every subdirectory of models/ that looks like a training run,
    newest first, with enough info to watch/play it or show its status."""
    if not MODELS_DIR.exists():
        return []
    runs = []
    for d in MODELS_DIR.iterdir():
        if not d.is_dir():
            continue
        final_model = d / f"{d.name}_final.zip"
        final_vecnorm = d / "vecnormalize.pkl"
        best_model = d / "best_model" / "best_model.zip"
        ckpt_dir = d / "checkpoints"
        checkpoints = list(ckpt_dir.glob("*_steps.zip")) if ckpt_dir.exists() else []
        if not final_model.exists() and not checkpoints:
            continue  # not a real run directory
        status = "complete" if final_model.exists() else f"in progress ({len(checkpoints)} checkpoint(s))"
        runs.append({
            "name": d.name,
            "final_model": final_model if final_model.exists() else None,
            "final_vecnorm": final_vecnorm if final_vecnorm.exists() else None,
            "best_model": best_model if best_model.exists() else None,
            "has_checkpoints": bool(checkpoints),
            "status": status,
            "mtime": d.stat().st_mtime,
        })
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


class Launcher:
    def __init__(self):
        if pygame is None:
            raise RuntimeError("pygame is required. Install it with `pip install pygame`.")
        pygame.init()
        pygame.display.set_caption("Hoverslam -- Launcher")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("consolas", 36, bold=True)
        self.font = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("consolas", 14, bold=True)
        self.running = True

    def run(self):
        while self.running:
            action = self.show_main_menu()
            if not self.running or action is None:
                break
            if action == "play":
                play_human()
                self._reinit_window()
            elif action == "watch":
                self.do_watch_ai()
            elif action == "evolve":
                self.do_watch_checkpoints()
            elif action == "train":
                self.do_train_from_scratch()
        pygame.quit()

    # ---------------- main menu ----------------

    def show_main_menu(self):
        buttons = {
            "play": pygame.Rect(WINDOW_W // 2 - 220, 260, 440, 70),
            "watch": pygame.Rect(WINDOW_W // 2 - 220, 350, 440, 70),
            "evolve": pygame.Rect(WINDOW_W // 2 - 220, 440, 440, 70),
            "train": pygame.Rect(WINDOW_W // 2 - 220, 530, 440, 70),
        }
        labels = {
            "play": ("Play Manually", "Fly it yourself with the keyboard"),
            "watch": ("Watch AI Play", "Pick a trained run and watch it fly"),
            "evolve": ("Watch Checkpoints Evolve", "See a run improve, checkpoint by checkpoint"),
            "train": ("Train From Scratch", "Start a brand-new training run"),
        }

        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.running = False
                    return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for name, rect in buttons.items():
                        if rect.collidepoint(mouse_pos):
                            return name

            self.screen.fill(BG_COLOR)
            title = self.font_title.render("HOVERSLAM", True, TEXT_COLOR)
            self.screen.blit(title, title.get_rect(center=(WINDOW_W // 2, 150)))
            subtitle = self.font_small.render("Choose an option", True, MUTED_TEXT)
            self.screen.blit(subtitle, subtitle.get_rect(center=(WINDOW_W // 2, 195)))

            for name, rect in buttons.items():
                hovered = rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.screen, BUTTON_HOVER if hovered else BUTTON_COLOR, rect, border_radius=8)
                pygame.draw.rect(self.screen, BUTTON_BORDER, rect, width=2, border_radius=8)
                label, sublabel = labels[name]
                label_surf = self.font.render(label, True, TEXT_COLOR)
                self.screen.blit(label_surf, (rect.left + 18, rect.top + 12))
                sub_surf = self.font_small.render(sublabel, True, MUTED_TEXT)
                self.screen.blit(sub_surf, (rect.left + 18, rect.top + 40))

            pygame.display.flip()
            self.clock.tick(60)
        return None

    # ---------------- run picker ----------------

    def pick_run(self, title):
        """Returns a chosen run dict, or None if the user went back (Esc)
        or quit the app (self.running is set to False in that case)."""
        runs = list_runs()
        if not runs:
            self.show_message(f"{title}\n\nNo training runs found yet.\nTrain one first!")
            return None

        row_h = 64
        top = 160
        rows = [(r, pygame.Rect(WINDOW_W // 2 - 320, top + i * (row_h + 8), 640, row_h))
                for i, r in enumerate(runs)]
        max_visible = max(1, (WINDOW_H - top - 60) // (row_h + 8))
        scroll = 0

        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return None
                if event.type == pygame.MOUSEWHEEL:
                    scroll = max(0, min(scroll - event.y, max(0, len(rows) - max_visible)))
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for r, rect in rows[scroll: scroll + max_visible]:
                        if rect.collidepoint(mouse_pos):
                            return r

            self.screen.fill(BG_COLOR)
            header = self.font.render(title, True, TEXT_COLOR)
            self.screen.blit(header, (WINDOW_W // 2 - 320, 90))
            hint = self.font_small.render("Esc to go back", True, MUTED_TEXT)
            self.screen.blit(hint, (WINDOW_W // 2 - 320, 120))

            for r, rect in rows[scroll: scroll + max_visible]:
                hovered = rect.collidepoint(mouse_pos)
                pygame.draw.rect(self.screen, BUTTON_HOVER if hovered else BUTTON_COLOR, rect, border_radius=8)
                pygame.draw.rect(self.screen, BUTTON_BORDER, rect, width=2, border_radius=8)
                name_surf = self.font.render(r["name"], True, TEXT_COLOR)
                self.screen.blit(name_surf, (rect.left + 16, rect.top + 10))
                status_surf = self.font_small.render(r["status"], True, MUTED_TEXT)
                self.screen.blit(status_surf, (rect.left + 16, rect.top + 36))

            pygame.display.flip()
            self.clock.tick(60)
        return None

    def show_message(self, message):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_RETURN):
                    return
            self.screen.fill(BG_COLOR)
            for i, line in enumerate(message.split("\n")):
                surf = self.font.render(line, True, TEXT_COLOR)
                self.screen.blit(surf, surf.get_rect(center=(WINDOW_W // 2, 280 + i * 34)))
            hint = self.font_small.render("Esc / Enter to go back", True, MUTED_TEXT)
            self.screen.blit(hint, hint.get_rect(center=(WINDOW_W // 2, WINDOW_H - 60)))
            pygame.display.flip()
            self.clock.tick(60)

    # ---------------- actions ----------------

    def do_watch_ai(self):
        run = self.pick_run("Watch AI Play -- choose a run")
        if not self.running:
            return
        if run is None:
            self._reinit_window()
            return
        model_path = run["best_model"] or run["final_model"]
        if model_path is None or run["final_vecnorm"] is None:
            self.show_message(f"{run['name']}\n\nNo finished model yet for this run\n(still training -- no _final.zip saved).")
            if self.running:
                self._reinit_window()
            return
        watch(str(model_path), str(run["final_vecnorm"]), episodes=20, deterministic=True)
        self._reinit_window()

    def do_watch_checkpoints(self):
        run = self.pick_run("Watch Checkpoints Evolve -- choose a run")
        if not self.running:
            return
        if run is None:
            self._reinit_window()
            return
        if not run["has_checkpoints"]:
            self.show_message(f"{run['name']}\n\nNo checkpoints saved yet for this run.")
            if self.running:
                self._reinit_window()
            return
        watch_checkpoint_sequence(str(MODELS_DIR), run["name"], episodes_per_checkpoint=3, deterministic=True)
        self._reinit_window()

    def do_train_from_scratch(self):
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        run_name = f"ppo_rocket_lander_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        log_path = LOGS_DIR / f"{run_name}.log"
        log_file = open(log_path, "w")
        # Deliberately no --init-model/--init-vecnormalize: a real from-zero run.
        proc = subprocess.Popen(
            [sys.executable, "train.py", "--timesteps", "1500000", "--run-name", run_name],
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        self.show_training_progress(run_name, log_path, proc)
        log_file.close()
        if self.running:
            self._reinit_window()

    def show_training_progress(self, run_name, log_path, proc):
        pattern_steps = re.compile(r"total_timesteps\s*\|\s*(\d+)")
        pattern_reward = re.compile(r"mean_reward\s*\|\s*([-\d.]+)")
        last_steps, last_reward = 0, None
        last_check = 0.0

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return  # training keeps running in the background

            now = time.time()
            if now - last_check > 1.0:
                last_check = now
                try:
                    text = log_path.read_text(errors="ignore")
                except OSError:
                    text = ""
                steps_matches = pattern_steps.findall(text)
                reward_matches = pattern_reward.findall(text)
                if steps_matches:
                    last_steps = int(steps_matches[-1])
                if reward_matches:
                    last_reward = float(reward_matches[-1])

            alive = proc.poll() is None

            self.screen.fill(BG_COLOR)
            title = self.font.render(f"Training: {run_name}", True, TEXT_COLOR)
            self.screen.blit(title, (WINDOW_W // 2 - 300, 140))
            status_surf = self.font.render(
                f"Status: {'running...' if alive else 'finished (or stopped)'}",
                True, ACCENT if alive else MUTED_TEXT)
            self.screen.blit(status_surf, (WINDOW_W // 2 - 300, 190))
            steps_surf = self.font.render(f"Steps: {last_steps:,}", True, TEXT_COLOR)
            self.screen.blit(steps_surf, (WINDOW_W // 2 - 300, 240))
            reward_text = f"{last_reward:.1f}" if last_reward is not None else "--"
            reward_surf = self.font.render(f"Latest eval reward: {reward_text}", True, TEXT_COLOR)
            self.screen.blit(reward_surf, (WINDOW_W // 2 - 300, 280))
            hint = self.font_small.render(
                "Esc: back to menu (training keeps running in the background)", True, MUTED_TEXT)
            self.screen.blit(hint, (WINDOW_W // 2 - 300, WINDOW_H - 80))
            pygame.display.flip()
            self.clock.tick(30)

    def _reinit_window(self):
        # play_human()/watch()/watch_checkpoint_sequence() close pygame
        # entirely on exit (via env.close()) -- bring the menu's own window
        # back since pygame.init()/set_mode() are safely idempotent.
        pygame.init()
        pygame.display.set_caption("Hoverslam -- Launcher")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))


def main():
    Launcher().run()


if __name__ == "__main__":
    main()
