"""
Watches a trained PPO agent fly the rocket, using the exact same pygame
rendering as play_human() in rocket_lander.py -- just driven by the policy
instead of the keyboard.

Usage:
    python watch_agent.py
    python watch_agent.py --model models/ppo_rocket_lander/best_model/best_model.zip
    python watch_agent.py --episodes 5 --stochastic

    # Watch how a run's policy evolved: plays a few episodes on every saved
    # checkpoint in order (earliest -> final), so you can SEE it improve
    # instead of just reading the eval-reward numbers.
    python watch_agent.py --run-name ppo_rocket_lander_v7_padvel --episodes-per-checkpoint 3
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rocket_lander import RocketLanderEnv

try:
    import pygame
except ImportError:
    pygame = None

# Frames to keep rendering after an episode ends, so the crash fireball/shake
# or the landing dust actually play out instead of freezing on the terminal
# frame (mirrors play_human()'s POST_EPISODE_HOLD_FRAMES).
POST_EPISODE_HOLD_FRAMES = 90


def discover_checkpoints(models_dir: str, run_name: str):
    """Returns [(label, model_path, vecnorm_path), ...] sorted earliest ->
    final, for every checkpoint of a training run that has BOTH the model
    and its matching vecnormalize stats saved (CheckpointCallback's
    save_vecnormalize=True writes them side by side)."""
    run_dir = Path(models_dir) / run_name
    ckpt_dir = run_dir / "checkpoints"
    found = []

    if ckpt_dir.exists():
        for model_path in ckpt_dir.glob(f"{run_name}_*_steps.zip"):
            m = re.search(r"_(\d+)_steps\.zip$", model_path.name)
            if not m:
                continue
            steps = int(m.group(1))
            vecnorm_path = ckpt_dir / f"{run_name}_vecnormalize_{steps}_steps.pkl"
            if vecnorm_path.exists():
                found.append((steps, model_path, vecnorm_path))
    found.sort(key=lambda t: t[0])

    labeled = [(f"{steps:,} steps", str(mp), str(vp)) for steps, mp, vp in found]

    final_model = run_dir / f"{run_name}_final.zip"
    final_vecnorm = run_dir / "vecnormalize.pkl"
    if final_model.exists() and final_vecnorm.exists():
        labeled.append(("final", str(final_model), str(final_vecnorm)))

    return labeled


def watch(model_path: str, vecnorm_path: str, episodes: int, deterministic: bool,
          window_label: str = None, checkpoint_progress: str = None):
    if pygame is None:
        raise RuntimeError("pygame is required to watch the agent. Install it with `pip install pygame`.")

    venv = DummyVecEnv([lambda: RocketLanderEnv(render_mode="human")])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    model = PPO.load(model_path, device="cpu")
    underlying_env = venv.venv.envs[0]  # the real RocketLanderEnv, for post-episode rendering

    landed = 0
    ep = 0
    hold = 0
    running = True

    def set_caption():
        parts = ["Hoverslam"]
        if window_label:
            parts.append(window_label)
        parts.append(f"episode {min(ep + 1, episodes)}/{episodes}")
        if checkpoint_progress:
            parts.append(checkpoint_progress)
        pygame.display.set_caption("  |  ".join(parts))

    obs = venv.reset()
    set_caption()
    while running and ep < episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        if hold > 0:
            underlying_env._render_frame()
            hold -= 1
            if hold == 0:
                obs = venv.reset()
                set_caption()
            continue

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = venv.step(action)

        if done[0]:
            ep += 1
            outcome = info[0]
            if outcome.get("landed"):
                landed += 1
                tag = "LANDED"
            elif outcome.get("crashed"):
                tag = "CRASHED"
            else:
                tag = "TIME UP"
            label_prefix = f"[{window_label}] " if window_label else ""
            print(f"{label_prefix}Episode {ep}/{episodes}: {tag}  (landed {landed}/{ep} so far)")
            hold = POST_EPISODE_HOLD_FRAMES
            set_caption()

    venv.close()
    if ep:
        print(f"{'[' + window_label + '] ' if window_label else ''}"
              f"Subtotal: {landed}/{ep} landed ({100 * landed / ep:.0f}%)")
    return running  # False if the user closed the window / hit Escape


def watch_checkpoint_sequence(models_dir: str, run_name: str, episodes_per_checkpoint: int,
                               deterministic: bool):
    checkpoints = discover_checkpoints(models_dir, run_name)
    if not checkpoints:
        print(f"No checkpoints with matching vecnormalize stats found for run "
              f"'{run_name}' under {models_dir}/{run_name}/checkpoints")
        return

    print(f"Found {len(checkpoints)} checkpoints for '{run_name}': "
          f"{', '.join(label for label, _, _ in checkpoints)}\n"
          f"Playing {episodes_per_checkpoint} episode(s) on each, earliest -> final. "
          f"Close the window or press Escape to stop early.\n")

    for i, (label, model_path, vecnorm_path) in enumerate(checkpoints):
        progress = f"checkpoint {i + 1}/{len(checkpoints)}"
        print(f"--- {label} ({progress}) ---")
        keep_going = watch(model_path, vecnorm_path, episodes_per_checkpoint, deterministic,
                            window_label=label, checkpoint_progress=progress)
        if not keep_going:
            print("Stopped early by user.")
            break


def main():
    default_model = "models/ppo_rocket_lander_v12_master/best_model/best_model.zip"
    default_vecnorm = "models/ppo_rocket_lander_v12_master/vecnormalize.pkl"
    if not Path(default_model).exists():
        default_model = "models/ppo_rocket_lander/ppo_rocket_lander_final.zip"
        default_vecnorm = "models/ppo_rocket_lander/vecnormalize.pkl"

    p = argparse.ArgumentParser(description="Watch a trained rocket-lander agent.")
    p.add_argument("--model", default=default_model)
    p.add_argument("--vecnormalize", default=default_vecnorm)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--stochastic", action="store_true",
                    help="Sample actions instead of using the deterministic policy mean.")
    p.add_argument("--run-name", default=None,
                   help="If set, ignores --model/--vecnormalize and instead plays every saved "
                        "checkpoint of this run in order (earliest -> final), so you can watch "
                        "the policy's actual behavior evolve over training.")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--episodes-per-checkpoint", type=int, default=3,
                   help="Only used with --run-name: how many episodes to play per checkpoint.")
    args = p.parse_args()

    if args.run_name:
        watch_checkpoint_sequence(args.models_dir, args.run_name, args.episodes_per_checkpoint,
                                   deterministic=not args.stochastic)
    else:
        watch(args.model, args.vecnormalize, args.episodes, deterministic=not args.stochastic)


if __name__ == "__main__":
    main()
