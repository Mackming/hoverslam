<div align="center">

# 🚀 Hoverslam

**An Autonomous Continuous-Control Rocket Landing Gymnasium Environment & PPO RL Agent**

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![CI Status](https://img.shields.io/github/actions/workflow/status/Mackming/hoverslam/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/Mackming/hoverslam/actions/workflows/ci.yml)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29+-blueviolet?style=for-the-badge&logo=openai&logoColor=white)](https://gymnasium.farama.org/)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-PPO-brightgreen?style=for-the-badge)](https://stable-baselines3.readthedocs.io/)
[![Pygame](https://img.shields.io/badge/Pygame-Rendering-yellow?style=for-the-badge&logo=python)](https://www.pygame.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-orange?style=for-the-badge)](LICENSE)

*Inspired by SpaceX Falcon 9 autonomous booster landings on autonomous spaceport drone ships.*

[Key Features](#-key-features) •
[Quickstart](#-quickstart) •
[Interactive Launcher](#-interactive-launcher) •
[Environment Specs](#-environment-specification) •
[Physics Model](#-physics--simulation-dynamics) •
[Training](#-training-the-agent) •
[Pre-Trained Models](#-pre-trained-models)

---

</div>

## 📖 Overview

**Hoverslam** is a continuous-control reinforcement learning environment adhering to the **Farama Gymnasium** standard. Unlike traditional Box2D-dependent lunar lander environments, Hoverslam features:

- A **pure-Python kinematic & rigid-body contact physics engine** (zero external C++ dependencies).
- A **drifting autonomous drone ship** requiring predictive trajectory and velocity matching.
- **"Rocking-block" ground contact dynamics**, decoupling impact tolerance from angular stability (landing hard causes bounces; landing unbalanced causes topples).
- **Sub-pixel Pygame telemetry and particle visualization**, with realistic thrust plumes, RCS cold-gas puffs, smoke, debris fireballs, and screen shake.
- A **complete PPO training pipeline** powered by [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) featuring observation normalization, checkpoint evolution tracking, and automatic vectorized environment benchmarking.

---

## ✨ Key Features

- **🎮 Dual-Mode Playability**: Fly manually with full keyboard controls or delegate control to a trained deep neural network policy.
- **🎯 Dynamic Moving Platform**: The drone ship wanders horizontally across the landing zone, forcing the RL agent to learn non-stationary target interception.
- **⚙️ Realistic Touchdown Physics**:
  - Distinguishes between structural failure ($v > v_{\text{max}}$) and catastrophic destruction ($v > v_{\text{cat}}$).
  - Restitution-damped ground bounces with impact-asymmetry spin transfer.
  - Foot-pivot mechanics where the rocket rocks on one leg under gravity and RCS torque before settling or toppling.
- **📊 Real-time Telemetry HUD**: Displays horizontal/vertical velocities, altitude, attitude angle, fuel gauge, throttle meters, and landing gear sensor states.
- **🎬 Checkpoint Evolution Mode**: Step through training iterations from step 0 to convergence to visualize the policy learning in real-time.
- **⚡ Automated Benchmarking**: Benchmarks `DummyVecEnv` vs `SubprocVecEnv` on your specific CPU architecture to maximize training throughput.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Mackming/hoverslam.git
cd hoverslam
```

### 2. Set Up a Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell):
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional) Install in editable mode with development tools:*
```bash
pip install -e ".[dev]"
```

---

## 🕹️ Quickstart

### 🖥️ Interactive Launcher (Recommended)
Launch the graphical dashboard to play manually, watch pre-trained AI checkpoints, or start training:
```bash
python launcher.py
```

### 🧑‍🚀 Human Manual Flight
Test your piloting skills directly using keyboard controls:
```bash
python rocket_lander.py
```

| Key | Action | Description |
| :---: | :---: | :--- |
| <kbd>▲ Up Arrow</kbd> | **Main Engine** | Fires main thruster upward (proportional throttle) |
| <kbd>◄ Left Arrow</kbd> | **Counter-Clockwise RCS** | Fires right-side cold gas thruster to rotate CCW |
| <kbd>► Right Arrow</kbd> | **Clockwise RCS** | Fires left-side cold gas thruster to rotate CW |
| <kbd>R</kbd> | **Reset / Fly Again** | Restarts the flight episode |
| <kbd>Esc</kbd> | **Quit** | Exits to menu or closes the window |

### 🤖 Watch the AI Agent
Watch the best pre-trained PPO policy land the rocket on the drifting drone ship:
```bash
python watch_agent.py --model models/ppo_rocket_lander_v12_master/best_model/best_model.zip
```

#### Watch Policy Evolution Across Checkpoints:
Observe how the agent learns over 3,000,000 steps—starting from uncoordinated tumbling to precision touchdown:
```bash
python watch_agent.py --run-name ppo_rocket_lander_v12_master --episodes-per-checkpoint 2
```

---

## 🔬 Environment Specification

The environment inherits from `gymnasium.Env` and implements the standard `step()` and `reset()` contract.

### Observation Space: `Box(9,)`
All values are normalized float32 scalars:

| Index | Name | Range | Normalized Bounds | Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | $\Delta x$ | $[-400, 400]$ m | `[-2.0, 2.0]` | Rocket $x$ relative to pad center, scaled by half-world width |
| `1` | $\Delta y$ | $[0, 600]$ m | `[-2.5, 2.5]` | Rocket $y$ relative to pad deck, scaled by half-world height |
| `2` | $v_x$ | $[-\infty, \infty]$ m/s | `[-20.0, 20.0]` | Horizontal velocity, scaled by $10.0$ |
| `3` | $v_y$ | $[-\infty, \infty]$ m/s | `[-20.0, 20.0]` | Vertical velocity, scaled by $10.0$ |
| `4` | $\theta$ | $[-\pi, \pi]$ rad | `[-π, π]` | Attitude angle ($0 = \text{upright}$, $>0 = \text{clockwise}$) |
| `5` | $\omega$ | $[-\infty, \infty]$ rad/s | `[-8.0, 8.0]` | Angular velocity |
| `6` | $c_{\text{left}}$ | $\{0, 1\}$ | `[0.0, 1.0]` | Left landing foot ground contact sensor |
| `7` | $c_{\text{right}}$ | $\{0, 1\}$ | `[0.0, 1.0]` | Right landing foot ground contact sensor |
| `8` | $v_{\text{pad}, x}$ | $\pm 15.0$ m/s | `[-2.0, 2.0]` | Drone ship horizontal drift velocity, scaled by $10.0$ |

### Action Space: `Box(2,)` (Continuous)
Continuous controls mapped in range `[-1.0, 1.0]`:

| Index | Control | Input Range | Mapped Output | Effect |
| :---: | :--- | :---: | :---: | :--- |
| `0` | **Main Throttle** | `[-1.0, 1.0]` | `[0.0, 1.0]` | $0\% \to 100\%$ linear thrust ($126.0\text{ m/s}^2$ max) |
| `1` | **RCS Thrusters** | `[-1.0, 1.0]` | `[-1.0, 1.0]` | Attitude torque ($\pm 12.0\text{ rad/s}^2$ angular accel) |

### Reward Function
The reward formulation combines potential-based shaping with terminal bonuses:

$$\mathcal{R}_{\text{step}} = \Delta \Phi - 0.2 \cdot u_{\text{main}} - 0.05 \cdot |u_{\text{rcs}}| - 0.1$$

Where:
- $\Phi = -100 \sqrt{\Delta x^2 + \Delta y^2} - 100 \sqrt{v_x^2 + v_y^2} - 100 |\theta| + 10(c_{\text{left}} + c_{\text{right}})$
- **Time penalty**: $-0.1$ per frame to discourage indefinite hovering.
- **Terminal Crash**: $-100.0$
- **Terminal Success**: $+100.0 + 50.0 \cdot Q_{\text{touchdown}}$
  - $Q_{\text{touchdown}} \in [0, 1]$ is a graded quality score based on residual touchdown velocity, attitude uprightness, and centeredness on the pad.

---

## 📐 Physics & Simulation Dynamics

Hoverslam's physics engine operates at 60 FPS ($dt = 1/60\text{s}$) with analytical kinematic equations:

```
                  ▲ +Y
                  │           [ RCS Port ]
                  │             ◄── ──►
              ┌───┴───┐
              │   ▲   │
              │   │   │  Rocket Body (34m x 14m)
              │   θ   │
              └───┬───┘
               /  │  \
              /   ▼   \
      Foot L ○  [MAIN] ○ Foot R
    ══════════════╤══════════════   ◄── Raised Drone Ship Deck (y = 12m)
            ▲     │     ▲
            └─────┴─────┘ Pad Pylons
```

1. **Thrust & Gravity**:
   - Gravity: $g = -45.0\text{ m/s}^2$ (weighty cinematic gravity).
   - Main Engine: Up to $126.0\text{ m/s}^2$ vector-aligned along the rocket hull axis.
   - Atmospheric drag and angular damping ($0.94\text{ per step}$) prevent unrecoverable spin runaway.
2. **Rocking-Block Contact Mechanics**:
   - Ground contact is modeled around the landing gear footprint ($22\text{m}$ span, $23\text{m}$ pivot-to-center offset).
   - The geometric stability limit angle is $\theta_{\text{tip}} = \arctan(11 / 23) \approx 25.6^\circ$.
   - If impact speed $v < 63.0\text{ m/s}$, the rocket absorbs the shock or rebounds with restitution ($e = 0.4$) and pivot torque.
   - If $\theta > \theta_{\text{tip}}$ or impact exceeds structural thresholds, the vehicle enters topple mode and crashes.

---

## 🏋️ Training the Agent

To train a new PPO agent from scratch:

```bash
python train.py --timesteps 2000000 --run-name my_ppo_run
```

### CLI Arguments:
- `--timesteps`: Total environment steps (default: `1,500,000`).
- `--n-envs`: Number of parallel vectorized environments (default: `8`).
- `--vec-env`: Force backend (`dummy`, `subproc`, or `auto` benchmarking).
- `--init-model`: Path to an existing model checkpoint to warm-start training.
- `--tb-log`: Directory for TensorBoard metrics (default: `runs/`).

### Real-Time Monitoring:
Launch TensorBoard to visualize training metrics (reward curves, episode length, policy entropy):
```bash
tensorboard --logdir runs/
```

---

## 🏆 Pre-Trained Models

The repository includes pre-trained model weights ready to run:

| Model Version | Training Steps | Mean Eval Reward | Features |
| :--- | :---: | :---: | :--- |
| **`ppo_rocket_lander_v12_master`** | 3.0M | **+235.4** | Full rocking-block physics, moving drone ship target, time penalty shaping |
| `ppo_rocket_lander_v7_padvel` | 1.5M | +180.2 | Initial moving-pad velocity state adaptation |
| `ppo_rocket_lander` (Baseline) | 1.0M | +142.0 | Static platform baseline |

---

## 📁 Repository Structure

```
hoverslam/
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md
│   │   └── feature_request.md
│   ├── workflows/
│   │   └── ci.yml               # Multi-OS & Multi-Python CI pipeline
│   └── pull_request_template.md
├── models/                      # Saved policies & checkpoint series
│   └── ppo_rocket_lander_v12_master/
│       ├── best_model/best_model.zip
│       ├── checkpoints/         # Step-by-step checkpoints for evolution viewer
│       └── vecnormalize.pkl
├── tests/
│   ├── __init__.py
│   └── test_env.py              # Headless environment unit & integration tests
├── .gitattributes               # Line-ending normalization
├── .gitignore                   # Python, bytecode, and log exclusions
├── CODE_OF_CONDUCT.md           # Contributor Covenant v2.1
├── CONTRIBUTING.md              # Contribution and PR guidelines
├── LICENSE                      # MIT License
├── launcher.py                  # Pygame graphical menu & launcher
├── pyproject.toml               # Modern PEP 621 package metadata
├── requirements.txt             # Direct dependencies
├── rocket_lander.py             # Gymnasium environment & Pygame rendering
├── setup.py                     # Legacy editable install support
├── train.py                     # PPO training with auto-benchmarking
└── watch_agent.py               # AI visualizer & evolution player
```

---

## 🧪 Testing

Hoverslam comes with a headless unit test suite verifying Gymnasium API compliance, space limits, RNG determinism, and vector environment integration:

```bash
python -m unittest discover tests
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Mackming/hoverslam/issues).
For detailed guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 License

This project is open-source and distributed under the [MIT License](LICENSE).
