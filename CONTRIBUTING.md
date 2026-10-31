# Contributing to Hoverslam

Thank you for your interest in contributing to **Hoverslam**! Whether you are fixing bugs, improving the simulation physics, tweaking the reward shaping, or experimenting with new reinforcement learning algorithms, contributions are welcome.

---

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Development Setup

1. **Fork and clone** the repository:
   ```bash
   git clone https://github.com/Mackming/hoverslam.git
   cd hoverslam
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

4. **Verify the environment**:
   Run the headless test suite to make sure everything passes:
   ```bash
   python -m unittest discover tests
   ```

---

## Workflow & Guidelines

### Branching Strategy
- Branch off `main` using descriptive names:
  - `feat/wind-gusts`
  - `fix/foot-restitution-clamp`
  - `docs/telemetry-guide`

### Coding Standards
- Follow **PEP 8** style guidelines.
- Keep the physics engine pure Python / NumPy (avoid adding heavy C++ dependencies like Box2D to maintain zero-hassle portability).
- Keep rendering isolated to `Pygame` routines so headless execution remains lightning fast for RL training.
- Add or update unit tests in `tests/` for any new physics features or environment state space changes.

### Submitting a Pull Request
1. Ensure all unit tests pass locally:
   ```bash
   python -m unittest discover tests
   ```
2. Open a Pull Request on GitHub with a clear description of the problem solved or feature added.
3. Reference any related issues (e.g., `Closes #12`).

---

## Feature Requests & Bug Reports
Please use the provided [Issue Templates](.github/ISSUE_TEMPLATE) when opening tickets on GitHub.
