"""
Trains a PPO agent (Stable-Baselines3) to land the rocket in RocketLanderEnv.

Usage:
    python train.py                          # train with sane defaults
    python train.py --timesteps 2000000       # train longer
    python train.py --vec-env subproc         # force a specific vec-env backend

Efficiency choices, and why:
  - Vectorized envs so PPO collects many episodes of experience per policy
    update instead of one at a time.
  - The env is cheap pure-Python physics (no Box2D), so whether many-envs-
    in-one-process (DummyVecEnv) or many-envs-in-separate-processes
    (SubprocVecEnv) is faster depends on whether IPC pickling overhead beats
    the physics cost itself -- that's not obvious in advance, so by default
    this script benchmarks both for a second on THIS machine and picks
    whichever is actually faster, instead of assuming.
  - VecNormalize (running mean/std for obs and reward) is standard practice
    for fast, stable PPO convergence on continuous-control tasks like this.
  - Hyperparameters are adapted from SB3-Zoo's tuned LunarLanderContinuous
    config, the closest published analogue to this task, rather than
    untuned library defaults.
  - device="cpu": for a small MLP policy, GPU transfer overhead dominates
    any compute benefit, so CPU is faster (true even if a GPU is present).
  - EvalCallback stops training early once the agent is reliably landing,
    so compute isn't wasted after convergence. CheckpointCallback means a
    long run is never lost to a crash/interrupt.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rocket_lander import RocketLanderEnv

VEC_ENV_BACKENDS = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}


def _env_fn():
    return RocketLanderEnv(render_mode=None)


def _benchmark_backend(vec_env_cls, n_envs: int, steps: int = 500) -> float:
    """Returns steps/sec throughput for this vec-env backend on this machine."""
    env = make_vec_env(_env_fn, n_envs=n_envs, vec_env_cls=vec_env_cls)
    env.reset()
    actions = np.stack([env.action_space.sample() for _ in range(n_envs)])
    start = time.perf_counter()
    for _ in range(steps):
        env.step(actions)
    elapsed = time.perf_counter() - start
    env.close()
    return (steps * n_envs) / elapsed


def pick_vec_env_backend(n_envs: int):
    """Benchmarks DummyVecEnv vs SubprocVecEnv on this machine and picks the
    faster one, instead of assuming which is better for this env."""
    if n_envs <= 1:
        return DummyVecEnv, "dummy"

    print(f"Benchmarking vec-env backends with n_envs={n_envs} ...")
    dummy_sps = _benchmark_backend(DummyVecEnv, n_envs)
    subproc_sps = _benchmark_backend(SubprocVecEnv, n_envs)
    print(f"  DummyVecEnv:   {dummy_sps:>10,.0f} steps/sec")
    print(f"  SubprocVecEnv: {subproc_sps:>10,.0f} steps/sec")

    if subproc_sps > dummy_sps:
        print("  -> using SubprocVecEnv (faster on this machine)\n")
        return SubprocVecEnv, "subproc"
    print("  -> using DummyVecEnv (faster on this machine)\n")
    return DummyVecEnv, "dummy"


def parse_args():
    p = argparse.ArgumentParser(description="Train a PPO agent on RocketLanderEnv.")
    p.add_argument("--timesteps", type=int, default=1_000_000, help="Total env steps to train for.")
    p.add_argument("--n-envs", type=int, default=32, help="Number of parallel envs.")
    p.add_argument("--vec-env", choices=["auto", "dummy", "subproc"], default="auto")
    p.add_argument("--target-reward", type=float, default=80.0,
                   help="Mean eval reward (over --n-eval-episodes) that counts as 'solved' and stops training early.")
    p.add_argument("--n-eval-episodes", type=int, default=20)
    p.add_argument("--eval-freq", type=int, default=20_000, help="Total env steps between evaluations.")
    p.add_argument("--checkpoint-freq", type=int, default=100_000, help="Total env steps between checkpoints.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default="ppo_rocket_lander")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--logs-dir", default="runs")
    p.add_argument("--init-model", default=None,
                   help="Path to a .zip to warm-start from (transfer learning), instead of "
                        "training a fresh policy from scratch. Pairs with --init-vecnormalize.")
    p.add_argument("--init-vecnormalize", default=None,
                   help="Path to the matching vecnormalize.pkl for --init-model, so obs/reward "
                        "running stats carry over too instead of restarting cold.")
    return p.parse_args()


def main():
    args = parse_args()

    models_dir = Path(args.models_dir) / args.run_name
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if args.vec_env == "auto":
        vec_env_cls, _ = pick_vec_env_backend(args.n_envs)
    else:
        vec_env_cls = VEC_ENV_BACKENDS[args.vec_env]

    train_env = make_vec_env(_env_fn, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_env_cls)
    if args.init_vecnormalize:
        # Carry over the running obs/reward normalization stats from the
        # warm-start source instead of restarting them cold -- the source
        # model's policy was trained against those stats, so feeding it
        # freshly-reset (mean 0, var 1) normalization would effectively
        # hand it out-of-distribution observations on step 1.
        train_env = VecNormalize.load(args.init_vecnormalize, train_env)
        train_env.training = True
    else:
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # A few parallel eval envs (not just one) so the n_eval_episodes requested
    # below finish in a fraction of the wall-clock time -- otherwise, early in
    # training when the policy rarely lands/crashes quickly, each evaluation
    # can serialize up to n_eval_episodes * MAX_EPISODE_STEPS single-env steps.
    eval_n_envs = min(4, args.n_envs)
    eval_env = make_vec_env(_env_fn, n_envs=eval_n_envs, seed=args.seed + 10_000, vec_env_cls=DummyVecEnv)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    stop_on_solved = StopTrainingOnRewardThreshold(reward_threshold=args.target_reward, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_on_solved,
        best_model_save_path=str(models_dir / "best_model"),
        log_path=str(models_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(models_dir / "checkpoints"),
        name_prefix=args.run_name,
        save_vecnormalize=True,
    )

    if args.init_model:
        # Warm-start: keep the already-learned policy (e.g. a model that
        # reliably gets close/slow/upright) and continue training it under
        # this run's reward/env, instead of re-learning that from scratch
        # under whatever new reward pressure was just added.
        model = PPO.load(args.init_model, env=train_env, device="cpu",
                          tensorboard_log=str(logs_dir))
    else:
        # Hyperparameters adapted from SB3-Zoo's tuned LunarLanderContinuous
        # config -- the closest published task to this one.
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=1024,
            batch_size=64,
            gae_lambda=0.98,
            gamma=0.999,
            n_epochs=4,
            # Back to the SB3-Zoo LunarLanderContinuous default. Raising this
            # to 0.03 was a stopgap for the old, nearly-flat reward shaping
            # (which gave the agent no gradient toward "almost landed") --
            # it just pushed action noise (std ~1.67) high enough to be
            # indistinguishable from random bang-bang, which is not the same
            # as learning. Now that the reward itself rewards braking and
            # grades crashes by how close they came, the agent doesn't need
            # help from raw noise to find it.
            ent_coef=0.01,
            learning_rate=3e-4,
            clip_range=0.2,
            device="cpu",
            tensorboard_log=str(logs_dir),
            seed=args.seed,
            verbose=1,
        )

    final_model_path = models_dir / f"{args.run_name}_final.zip"
    vecnorm_path = models_dir / "vecnormalize.pkl"

    print(f"Training for up to {args.timesteps:,} steps "
          f"({args.n_envs} envs, target reward {args.target_reward})...")
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name=args.run_name,
            progress_bar=True,
        )
    finally:
        model.save(str(final_model_path))
        train_env.save(str(vecnorm_path))
        print(f"\nSaved final model to {final_model_path}")
        print(f"Saved normalization stats to {vecnorm_path}")
        print(f"Best model (if any beat the eval baseline): {models_dir / 'best_model' / 'best_model.zip'}")
        print(f"\nWatch it fly:\n"
              f"  python watch_agent.py --model {final_model_path} --vecnormalize {vecnorm_path}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
