"""
Unit and integration tests for RocketLanderEnv.
Runs headless without requiring a display device or Pygame window.
"""

import math
import unittest
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from rocket_lander import (
    RocketLanderEnv,
    WORLD_W,
    PAD_MOVE_RANGE,
    PAD_MOVE_SPEED,
    MAX_EPISODE_STEPS,
)


class TestRocketLanderEnv(unittest.TestCase):
    def setUp(self):
        self.env = RocketLanderEnv(render_mode=None)

    def tearDown(self):
        self.env.close()

    def test_spaces(self):
        """Verify observation and action spaces match specification."""
        self.assertEqual(self.env.observation_space.shape, (9,))
        self.assertEqual(self.env.action_space.shape, (2,))
        self.assertTrue(np.all(self.env.action_space.low == -1.0))
        self.assertTrue(np.all(self.env.action_space.high == 1.0))

    def test_reset(self):
        """Verify reset returns valid observation and info dictionary."""
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (9,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIsInstance(info, dict)

    def test_deterministic_seeding(self):
        """Identical seeds should generate identical initial states."""
        obs1, _ = self.env.reset(seed=123)
        obs2, _ = self.env.reset(seed=123)
        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_step_execution(self):
        """Verify step returns 5-tuple with finite values and correct types."""
        self.env.reset(seed=42)
        action = np.array([0.5, -0.2], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertEqual(obs.shape, (9,))
        self.assertTrue(np.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)
        self.assertIn("landed", info)
        self.assertIn("crashed", info)

    def test_observation_clipping(self):
        """Observation values must remain strictly within declared space bounds."""
        self.env.reset(seed=42)
        for _ in range(50):
            action = self.env.action_space.sample()
            obs, _, term, trunc, _ = self.env.step(action)
            self.assertTrue(self.env.observation_space.contains(obs))
            if term or trunc:
                self.env.reset()

    def test_pad_drift_bounds(self):
        """Landing pad position must remain within allowable wander range."""
        self.env.reset(seed=42)
        center_x = WORLD_W / 2.0
        min_x = center_x - PAD_MOVE_RANGE - 1e-4
        max_x = center_x + PAD_MOVE_RANGE + 1e-4

        for _ in range(120):
            self.env.step(np.array([0.0, 0.0], dtype=np.float32))
            self.assertGreaterEqual(self.env.pad_x, min_x)
            self.assertLessEqual(self.env.pad_x, max_x)
            if self.env.landed or self.env.crashed:
                break

    def test_episode_truncation(self):
        """Episodes should truncate at MAX_EPISODE_STEPS if no contact occurs."""
        self.env.reset(seed=42)
        # Hold in air by freezing downward acceleration artificially or checking step cap
        self.env.steps = MAX_EPISODE_STEPS - 1
        _, _, _, truncated, _ = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertTrue(truncated)

    def test_vector_env_compatibility(self):
        """Verify compatibility with Stable-Baselines3 vector environments."""
        venv = DummyVecEnv([lambda: RocketLanderEnv(render_mode=None)])
        obs = venv.reset()
        self.assertEqual(obs.shape, (1, 9))
        actions = np.array([[0.0, 0.0]], dtype=np.float32)
        obs, rewards, dones, infos = venv.step(actions)
        self.assertEqual(obs.shape, (1, 9))
        self.assertEqual(len(rewards), 1)
        self.assertEqual(len(dones), 1)
        venv.close()


if __name__ == "__main__":
    unittest.main()
