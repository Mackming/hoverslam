"""
Hoverslam: a 2D continuous-control rocket landing environment.

Implements the Gymnasium API (gymnasium.Env) with lightweight kinematic
physics written in plain Python (no Box2D). Pygame is used only for
rendering. The bottom of the file has a human-playable loop driven by
the keyboard so you can fly the rocket yourself before training an agent.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:  # pygame is only required for rendering / human play
    pygame = None


# --------------------------------------------------------------------------
# World / physics constants
# --------------------------------------------------------------------------

WORLD_W = 800.0          # world width in "meters"
WORLD_H = 600.0          # world height

GRAVITY = -45.0          # m/s^2, downward (negative y). Numerically already
                         # ~2.5x Earth's 9.8, but the 600-"meter" world is
                         # squeezed into ~580 screen pixels, so free-fall
                         # still looked slow -- this makes it look weighty.

MAIN_ENGINE_POWER = 126.0    # m/s^2 of thrust at full main throttle. Scaled
                              # by the same 1.8x as GRAVITY so the braking
                              # margin (net decel = thrust - gravity) feels
                              # exactly as forgiving as before, just heavier.
SIDE_ENGINE_POWER = 12.0     # rad/s^2 of angular accel at full side throttle

ANGULAR_DAMPING = 0.94       # per-step damping so rotation doesn't run away
LINEAR_DAMPING = 0.999       # tiny air drag

ROCKET_HEIGHT = 34.0
ROCKET_WIDTH = 14.0
LEG_SPAN = 22.0              # horizontal distance between the two landing feet
FOOT_DROP = 6.0              # how far the foot pad extends below the hull base
FOOT_OFFSET = ROCKET_HEIGHT / 2.0 + FOOT_DROP   # center-to-foot distance (=23.0)

PAD_HALF_WIDTH = 60.0
PAD_HEIGHT = 12.0            # the landing pad is a raised deck, not flush with the ground
PAD_MOVE_SPEED = 15.0        # world-units/sec the pad drifts horizontally while airborne
PAD_MOVE_RANGE = 200.0       # how far the pad's center can wander from world-center

FPS = 60
DT = 1.0 / FPS

# Ground-contact ("rocking block") physics: how HARD you land is judged
# separately from how you BALANCE. A landing fails if either: (a) the raw
# impact speed exceeds what the landing gear can structurally absorb
# (MAX_IMPACT_SPEED -- a real lander can't just free-fall onto the pad at
# terminal velocity and call it fine), or (b) it topples over once resting
# (rotates past the point where its center of mass swings outside its own
# footprint -- a fin/nose/nozzle touching the ground IS that topple), or
# (c) it ends up off the landing platform.
TIP_ANGLE = math.atan2(LEG_SPAN / 2.0, FOOT_OFFSET)  # ~25.6 deg: the real
                                                       # geometric stability limit
MAX_IMPACT_SPEED = 63.0      # m/s: harder than this and the legs can't absorb
                              # it cleanly, but it still BOUNCES rather than
                              # instantly failing. Scaled by the same 1.8x as
                              # GRAVITY, so free gravity alone still reaches
                              # it in the same 1.4s reaction window as before.
CATASTROPHIC_IMPACT_SPEED = 120.0  # m/s: past this, no bounce even gets a
                                    # chance -- straight to real destruction.
                                    # A full uncontrolled free-fall (~209 m/s)
                                    # is always well past this.
ROCKET_INERTIA = 250.0       # body's own rotational inertia (about its own center)
GROUND_ANGULAR_DAMPING = 0.995   # lighter than flight damping -- energy is lost
                                  # mainly at each foot-plant impact, not every frame
GROUND_RCS_FACTOR = 0.25     # side-thruster authority while grounded (can save a
                              # marginal tip, can't just flip yourself upright at will)
PLANT_OMEGA = 0.15           # rad/s: below this, a rebound is small enough to call it settled
PLANT_EPS = 0.25             # world-units: feet within this height of each other count as "together"
MAX_ROCK_SECONDS = 1.5       # a single-foot rock can (rarely) find a stable
                              # near-equilibrium and balance indefinitely;
                              # force a real settle/topple decision past this

# Newton's third law: a too-hard impact doesn't just delete the rocket, the
# ground pushes back. Not a full rigid-body restitution derivation (kept
# intentionally simple) -- reflects vertical speed with energy loss, damps
# horizontal speed like friction, and blends in some of the same
# impact-asymmetry spin used for a normal foot-plant, so it looks like a
# real (if a little chaotic) bounce rather than a perfect mirror.
BOUNCE_RESTITUTION = 0.4          # fraction of vertical impact speed reflected back up
BOUNCE_HORIZONTAL_DAMPING = 0.85  # fraction of horizontal speed kept through a bounce
BOUNCE_SPIN_TRANSFER = 0.5        # how much impact-asymmetry spin blends in per bounce
MAX_BOUNCES = 3                   # this many too-hard impacts in a row and the legs finally give out

# A topple is a real, continued fall: once past the point of no return, the
# rocket keeps pivoting (now on whatever hit -- a fin, the nose) and
# rotating under gravity until it has actually fallen to lying down, rather
# than freezing the instant the condition is first detected.
TOPPLE_DONE_ANGLE = math.radians(85.0)
TOPPLE_ANGULAR_ACCEL = 5.0    # rad/s^2: a committed topple's fall rate (see
                              # _begin_topple for why this ISN'T the same
                              # gravity-torque formula rocking uses)

MAX_EPISODE_STEPS = 1000

# --------------------------------------------------------------------------
# Rendering / window constants
# --------------------------------------------------------------------------

WINDOW_W = 960
WINDOW_H = 720
HUD_H = 100          # reserved band at the top for telemetry
VIEW_MARGIN = 20     # side/bottom padding around the game viewport


@dataclass
class RocketState:
    x: float
    y: float
    vx: float
    vy: float
    angle: float        # radians, 0 = upright, positive = tilted clockwise
    angular_vel: float
    fuel: float


class RocketLanderEnv(gym.Env):
    """A 2D rocket landing environment with simple kinematic physics.

    Observation (Box, shape (8,), float32):
        [x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact]
        x, y are normalized relative to the landing pad and world scale.

    Action (Box, shape (2,), float32), continuous control:
        action[0]: main engine throttle, in [-1, 1] -> mapped to [0, 1]
        action[1]: side thruster, in [-1, 1] -> negative rotates counter-
                   clockwise, positive rotates clockwise (RCS torque
                   proportional to magnitude)

    Reward:
        Shaped reward that encourages moving toward the pad, descending
        slowly, and staying upright. Large terminal bonus for landing on
        the pad without toppling over, large penalty for crashing.

    Landing/crashing uses a simplified rigid-body "rocking block" model,
    plus a separate impact-speed limit -- two independent ways to fail:
    landing too HARD (raw touchdown speed exceeding MAX_IMPACT_SPEED, a
    structural failure regardless of angle), or landing unbalanced (a foot
    plants and the rocket pivots/rocks on it under gravity and thruster
    torque; it crashes if it topples over -- a fin/nose/nozzle touches the
    ground -- or ends up off the landing platform).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()

        self.render_mode = render_mode
        self.screen = None
        self.clock = None

        self.observation_space = spaces.Box(
            # x-bound widened from +/-1.5: with a moving pad and a wider
            # randomized start position, the rocket-to-pad offset can now
            # exceed the old margin (pad wanders +/-PAD_MOVE_RANGE from
            # center while the rocket can be anywhere in the world).
            # Last entry is the pad's own horizontal velocity: this is a
            # plain MLP with no memory of past frames, so without this it
            # can only ever perceive the pad's CURRENT position, never that
            # it's moving or which way -- it has no way to infer velocity
            # from a single instantaneous snapshot.
            low=np.array([-2.0, -2.5, -20.0, -20.0, -math.pi, -8.0, 0.0, 0.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.5, 20.0, 20.0, math.pi, 8.0, 1.0, 1.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )

        # [main throttle, side throttle], both in [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.pad_x = WORLD_W / 2.0
        self.pad_y = PAD_HEIGHT
        self._pad_direction = 1.0  # +/-1: which way the pad is currently drifting

        self.state: Optional[RocketState] = None
        self.steps = 0
        self.landed = False
        self.crashed = False

        self._last_main_throttle = 0.0
        self._last_side_throttle = 0.0

        # Set once ground contact resolves into a crash; freezes the engine
        # flame/particles and swaps in a charred palette for the wreck.
        self._wrecked = False
        self._impact_point: Optional[tuple] = None
        self._blast_scale = 1.0  # scales crash fx (fireball/shake) with impact speed
        self._prev_shaping = 0.0

        # Snapshot of the touch that led to (or almost led to) a settle,
        # used to grade the landing bonus -- see _resolve_contact.
        self._touchdown_speed = 0.0
        self._touchdown_angle = 0.0
        self._touchdown_dx = 0.0

        # Rocking-block ground-contact state: while _grounded, the rocket is
        # pivoting on a fixed world point (_pivot_local is its body-space
        # coordinates, _pivot_x/_pivot_y its fixed world position). Normally
        # that's a foot; _toppling means it's pivoting on whatever hit
        # instead (a fin, the nose) and is genuinely falling over -- not
        # recoverable, just playing out until TOPPLE_DONE_ANGLE. _planted
        # means it came to rest flat.
        self._grounded = False
        self._planted = False
        self._toppling = False
        self._topple_sign = 1.0
        self._bounce_count = 0
        self._rock_timer = 0.0
        self._pivot_local = (0.0, 0.0)
        self._pivot_x = 0.0
        self._pivot_y = 0.0

        # Cosmetic-only state (never affects physics/reward/observations).
        self._particles: list = []
        self._flash_color = (0, 0, 0)
        self._flash_alpha = 0.0
        self._fireball_timer = 0.0
        self._shake = 0.0
        self._shake_offset = (0.0, 0.0)
        self._terminal_fx_done = False
        self._frame = 0

        self._rng = np.random.default_rng()

    # ----------------------------------------------------------------
    # Gymnasium API
    # ----------------------------------------------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Random position in the air: both x and y vary now, not just x.
        start_x = self._rng.uniform(100.0, WORLD_W - 100.0)
        start_y = self._rng.uniform(350.0, WORLD_H - 40.0)
        start_angle = self._rng.uniform(-0.15, 0.15)

        self.pad_x = WORLD_W / 2.0 + self._rng.uniform(-PAD_MOVE_RANGE, PAD_MOVE_RANGE)
        self._pad_direction = 1.0 if self._rng.uniform() < 0.5 else -1.0

        self.state = RocketState(
            x=start_x,
            y=start_y,
            vx=self._rng.uniform(-2.0, 2.0),
            vy=0.0,
            angle=start_angle,
            angular_vel=0.0,
            fuel=100.0,
        )
        self.steps = 0
        self.landed = False
        self.crashed = False
        self._last_main_throttle = 0.0
        self._last_side_throttle = 0.0

        self._wrecked = False
        self._impact_point = None
        self._blast_scale = 1.0

        self._touchdown_speed = 0.0
        self._touchdown_angle = 0.0
        self._touchdown_dx = 0.0

        self._grounded = False
        self._planted = False
        self._toppling = False
        self._topple_sign = 1.0
        self._bounce_count = 0
        self._rock_timer = 0.0
        self._pivot_local = (0.0, 0.0)
        self._pivot_x = 0.0
        self._pivot_y = 0.0

        self._particles = []
        self._flash_color = (0, 0, 0)
        self._flash_alpha = 0.0
        self._fireball_timer = 0.0
        self._shake = 0.0
        self._shake_offset = (0.0, 0.0)
        self._terminal_fx_done = False

        self._prev_shaping = self._shaping()

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self._render_frame()

        return obs, info

    def step(self, action: np.ndarray):
        assert self.state is not None, "Call reset() before step()."

        action = np.clip(action, self.action_space.low, self.action_space.high)
        main_throttle = (float(action[0]) + 1.0) / 2.0   # -> [0, 1]
        side_throttle = float(action[1])                  # -> [-1, 1]

        # _simulate returns the *effective* throttles (zeroed out if the tank
        # ran dry mid-step), so the flame/particles and the fuel-burn reward
        # penalty never show thrust that didn't actually happen.
        main_throttle, side_throttle = self._simulate(main_throttle, side_throttle)
        self._last_main_throttle = main_throttle
        self._last_side_throttle = side_throttle

        # _simulate() resolves ground contact internally (rocking-block
        # model): it sets self.landed/self.crashed directly once the
        # outcome is decided, rather than returning it, since a topple can
        # also occur mid-rock, several steps after the initial touchdown.
        terminated = self.landed or self.crashed

        s = self.state
        if not terminated and s.y > WORLD_H * 1.2:
            self.crashed = True
            self._wrecked = True
            self._impact_point = (s.x, s.y)
            terminated = True

        self.steps += 1
        obs = self._get_obs()
        reward = self._compute_reward(terminated)
        truncated = self.steps >= MAX_EPISODE_STEPS

        if terminated and not self._terminal_fx_done:
            self._terminal_fx_done = True
            if self.landed:
                self._trigger_terminal_fx((100, 235, 150))
            elif self.crashed:
                self._trigger_terminal_fx((245, 90, 80), fireball=True)

        info = {"landed": self.landed, "crashed": self.crashed}

        if self.render_mode == "human":
            self._render_frame()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        return None

    def close(self):
        if self.screen is not None and pygame is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None

    # ----------------------------------------------------------------
    # Physics
    # ----------------------------------------------------------------

    def _simulate(self, main_throttle: float, side_throttle: float):
        main_throttle, side_throttle = self._burn_fuel(main_throttle, side_throttle)

        if self.landed or self.crashed:
            return main_throttle, side_throttle  # episode already resolved

        if self._grounded:
            lifted_off = self._grounded_step(main_throttle, side_throttle)
            if not lifted_off:
                return main_throttle, side_throttle

        # The pad only drifts while genuinely airborne -- freezing it once
        # grounded avoids the platform sliding out from under a rocket
        # that's mid-rock/toppling/resting on a now-fixed pivot point.
        self._update_pad_position()
        self._flight_step(main_throttle, side_throttle)
        if not self._grounded:
            self._resolve_contact()

        return main_throttle, side_throttle

    def _update_pad_position(self):
        """Drifts the pad at a constant speed, reversing at the edges of
        its allowed range (a simple, predictable ping-pong -- easier for a
        player to track than a varying-speed sine drift)."""
        center = WORLD_W / 2.0
        self.pad_x += self._pad_direction * PAD_MOVE_SPEED * DT
        if self.pad_x > center + PAD_MOVE_RANGE:
            self.pad_x = center + PAD_MOVE_RANGE
            self._pad_direction = -1.0
        elif self.pad_x < center - PAD_MOVE_RANGE:
            self.pad_x = center - PAD_MOVE_RANGE
            self._pad_direction = 1.0

    def _burn_fuel(self, main_throttle: float, side_throttle: float):
        s = self.state
        fuel_use = (main_throttle * 0.12 + abs(side_throttle) * 0.03)
        fuel_use = min(fuel_use, s.fuel)
        s.fuel -= fuel_use
        if s.fuel <= 0.0:
            main_throttle = 0.0
            side_throttle = 0.0
        return main_throttle, side_throttle

    def _flight_step(self, main_throttle: float, side_throttle: float):
        s = self.state

        # Main engine thrust acts along the rocket's own "up" axis, which is
        # rotated by the current angle away from world-up.
        thrust = main_throttle * MAIN_ENGINE_POWER
        ax = math.sin(s.angle) * thrust
        ay = math.cos(s.angle) * thrust + GRAVITY

        s.vx = (s.vx + ax * DT) * LINEAR_DAMPING
        s.vy = (s.vy + ay * DT) * LINEAR_DAMPING

        s.x += s.vx * DT
        s.y += s.vy * DT

        angular_accel = side_throttle * SIDE_ENGINE_POWER
        s.angular_vel = (s.angular_vel + angular_accel * DT) * ANGULAR_DAMPING
        s.angle += s.angular_vel * DT

        # Keep angle in [-pi, pi] for a stable observation range.
        s.angle = (s.angle + math.pi) % (2 * math.pi) - math.pi

        # World bounds: side walls are solid; ground/pad contact and the
        # "flew off the top" case are handled elsewhere.
        if s.x < 0.0:
            s.x = 0.0
            s.vx = 0.0
        elif s.x > WORLD_W:
            s.x = WORLD_W
            s.vx = 0.0

    def _surface_height(self, x: float) -> float:
        """Terrain height at world-x: the raised pad deck, or flat ground."""
        if abs(x - self.pad_x) <= PAD_HALF_WIDTH:
            return PAD_HEIGHT
        return 0.0

    def _body_points(self):
        """Single source of truth for every body-space point used by BOTH
        collision (_contact_points) and rendering (_draw_rocket), so the two
        can never drift apart the way the original pad/foot mismatch did.

        The fin tips and nozzle bottom are deliberately kept a hair above
        the feet so the feet are always the geometrically lowest point on
        an upright rocket -- otherwise the fins/nozzle would register
        ground contact a moment before the feet do. That margin is also
        what makes "a fin touches down" a reliable signal for "toppled
        past the point of no return": TIP_ANGLE (~25.6 deg) is where the
        feet alone go unstable, and the fins sit just beyond that, so a
        fin touch always means recovery is already impossible.
        """
        half_w, half_h = ROCKET_WIDTH / 2.0, ROCKET_HEIGHT / 2.0
        underside = FOOT_OFFSET - 2.0
        return {
            "shoulder_l": (-half_w, half_h * 0.55),
            "shoulder_r": (half_w, half_h * 0.55),
            "base_l": (-half_w, -half_h),
            "base_r": (half_w, -half_h),
            "nose": (0.0, half_h + 12.0),
            "fin_l_tip": (-half_w - 8, -underside),
            "fin_r_tip": (half_w + 8, -underside),
            "fin_l_root": (-half_w, -half_h + 11),
            "fin_r_root": (half_w, -half_h + 11),
            "nozzle_bl": (-half_w * 0.35, -underside),
            "nozzle_br": (half_w * 0.35, -underside),
            "foot_l": (-LEG_SPAN / 2.0, -FOOT_OFFSET),
            "foot_r": (LEG_SPAN / 2.0, -FOOT_OFFSET),
        }

    def _contact_points(self):
        """[(name, world_x, world_y, penetration), ...] for every point in
        _body_points(). Single source of truth for both first-contact
        classification and the ongoing topple check while grounded."""
        result = []
        for name, (lx, ly) in self._body_points().items():
            wx, wy = self._local_to_world(lx, ly)
            pen = self._surface_height(wx) - wy
            result.append((name, wx, wy, pen))
        return result

    def _check_contact(self):
        """Returns (feet_touching, hull_touching, max_penetration) against
        whatever terrain (ground or pad deck) is under each point."""
        feet_touch = False
        hull_touch = False
        max_pen = 0.0
        for name, wx, wy, pen in self._contact_points():
            if pen > 1e-6:
                if name in ("foot_l", "foot_r"):
                    feet_touch = True
                else:
                    hull_touch = True
            max_pen = max(max_pen, pen)
        return feet_touch, hull_touch, max_pen

    def _pivot_omega(self, lx: float, ly: float) -> float:
        """The closed-form post-impact spin about the body-local point
        (lx, ly), assuming an inelastic, non-slipping plant there: angular
        momentum about that point is conserved through the impulsive
        contact force. Derived directly from this file's own rotation
        convention (see _local_dir), not a generic textbook formula. Works
        for ANY body point, not just a foot -- used both for a normal
        foot-plant and for the spin a topple picks up when a fin/nose
        catches the ground."""
        s = self.state
        fdx, fdy = self._local_dir(lx, ly)  # point's offset from COM
        return (ROCKET_INERTIA * s.angular_vel + fdx * s.vy - fdy * s.vx) / (
            ROCKET_INERTIA + fdx * fdx + fdy * fdy
        )

    def _place_on_pivot(self):
        """Re-derives x, y, vx, vy from the fixed world pivot point plus
        the current angle, so the planted point never visibly slides (the
        bug in a naive "rotate about the center, then snap y" approach)."""
        s = self.state
        fdx, fdy = self._local_dir(*self._pivot_local)
        s.x = self._pivot_x - fdx
        s.y = self._pivot_y - fdy
        s.vx = -s.angular_vel * fdy
        s.vy = s.angular_vel * fdx

    def _set_pivot(self, lx: float, ly: float, omega: float):
        """Begins (or switches to) pivoting on the given body point: anchors
        it at its current world position and gives the body the post-impact
        spin about it."""
        wx, _ = self._local_to_world(lx, ly)
        self._grounded = True
        self._pivot_local = (lx, ly)
        self._pivot_x = wx
        self._pivot_y = self._surface_height(wx)
        self.state.angular_vel = omega
        self._place_on_pivot()

    def _resolve_flat_stance(self):
        """Called whenever both feet are at the surface at once (either on
        first touchdown, or mid-rock when the other foot catches up).
        Decides whether this settles flat, or keeps rocking onto one foot
        or the other. A flat settle is only a successful landing if it's
        actually over the pad -- missing the platform is a real, if
        undramatic, way to fail."""
        s = self.state
        s.angle = 0.0
        half = LEG_SPAN / 2.0
        w_r = self._pivot_omega(half, -FOOT_OFFSET)
        w_l = self._pivot_omega(-half, -FOOT_OFFSET)
        if w_r > PLANT_OMEGA and w_r >= -w_l:
            self._set_pivot(half, -FOOT_OFFSET, w_r)
        elif -w_l > PLANT_OMEGA:
            self._set_pivot(-half, -FOOT_OFFSET, w_l)
        else:
            self._grounded = False
            self._planted = True
            s.vx = s.vy = s.angular_vel = 0.0
            s.y = self._surface_height(s.x) + FOOT_OFFSET
            self._impact_point = (s.x, s.y)
            if abs(s.x - self.pad_x) <= PAD_HALF_WIDTH:
                self.landed = True
                self._spawn_landing_dust()
            else:
                self._finish_crash()

    def _begin_topple(self, contact_name: str, world_x: float):
        """Transitions from normal foot-rocking (or straight from first
        touchdown, if it arrived already badly oriented) into a real,
        continued fall, instead of freezing the instant this is detected.

        Deliberately does NOT reuse the foot-pivot gravity-torque formula
        with the new contact point as pivot: that point (a fin, the nose)
        sits FARTHER from center than a foot, so its own stability cone is
        actually WIDER than the foot's -- pivoting the rotation physics on
        it can spuriously restore back toward vertical instead of
        continuing to fall (verified this the hard way with a 35 degree
        touchdown that rocked itself back up). Since we only ever get here
        after already crossing the foot's own point-of-no-return, the fix
        is to keep committing to the direction it's already falling with a
        simple, guaranteed-monotonic fall rate, and use the new contact
        point only to anchor its POSITION (so it doesn't slide/teleport),
        not to derive the rotation."""
        lx, ly = self._body_points()[contact_name]
        s = self.state
        self._toppling = True
        self._pivot_local = (lx, ly)
        self._pivot_x = world_x
        self._pivot_y = self._surface_height(world_x)
        self._grounded = True
        direction = s.angular_vel if abs(s.angular_vel) > 1e-3 else (s.angle if s.angle != 0.0 else 1.0)
        self._topple_sign = 1.0 if direction >= 0.0 else -1.0
        self._place_on_pivot()

    def _finish_crash(self):
        s = self.state
        impact_speed = math.hypot(s.vx, s.vy)
        self._blast_scale = min(2.5, max(1.0, impact_speed / MAX_IMPACT_SPEED))
        self.crashed = True
        self._wrecked = True
        self._toppling = False
        self._grounded = False
        self._impact_point = (s.x, s.y)

    def _bounce(self, name: str, pen: float):
        """A too-hard impact: the ground pushes back (Newton's third law)
        instead of instantly ending the episode. Reflects/damps velocity
        and blends in some impact-asymmetry spin, then lets it fly back
        into ordinary free flight to come down again. Capped at
        MAX_BOUNCES -- repeated hard impacts are what finally breaks the
        landing gear for good."""
        s = self.state
        s.y += pen
        self._bounce_count += 1

        if self._bounce_count > MAX_BOUNCES:
            self._finish_crash()
            return

        lx, ly = self._body_points()[name]
        kick = self._pivot_omega(lx, ly)

        s.vy = -s.vy * BOUNCE_RESTITUTION
        s.vx *= BOUNCE_HORIZONTAL_DAMPING
        s.angular_vel += (kick - s.angular_vel) * BOUNCE_SPIN_TRANSFER

        self._shake = max(self._shake, 6.0)
        self._emit_particles(
            s.x, max(s.y, 0.5), 0.0, 1.0,
            count=14, speed_range=(15.0, 50.0), spread=1.2,
            life_range=(0.2, 0.45), size_range=(1.5, 3.0), color=(190, 185, 170),
        )

    def _grounded_step(self, main_throttle: float, side_throttle: float) -> bool:
        """Advances the rocking-on-one-point physics by one step. Returns
        True if enough main thrust broke it back out into free flight this
        step (only possible while NOT toppling -- a genuine topple is past
        the point of no return, thrust can no longer save it)."""
        s = self.state

        if not self._toppling and main_throttle * MAIN_ENGINE_POWER * math.cos(s.angle) > -GRAVITY:
            self._grounded = False
            return True

        if self._toppling:
            # Committed: keep falling in the direction already picked at
            # _begin_topple, at a simple guaranteed rate (see that method's
            # docstring for why this ISN'T the foot-pivot torque formula).
            angular_accel = self._topple_sign * TOPPLE_ANGULAR_ACCEL
            s.angular_vel = (s.angular_vel + angular_accel * DT) * GROUND_ANGULAR_DAMPING
            s.angle += s.angular_vel * DT
            self._place_on_pivot()
            if abs(s.angle) >= TOPPLE_DONE_ANGLE:
                self._finish_crash()
            return False

        lx, ly = self._pivot_local
        fdx, fdy = self._local_dir(lx, ly)
        i_pivot = ROCKET_INERTIA + fdx * fdx + fdy * fdy

        # Gravity's torque about the pivot: restores toward vertical while
        # the center of mass is still over the pivot, reinforces the tilt
        # once it has swung past it (this sign flip *is* TIP_ANGLE/topple).
        angular_accel = (fdx * GRAVITY) / i_pivot
        # Side thrusters can still fight a marginal tip.
        angular_accel += side_throttle * SIDE_ENGINE_POWER * GROUND_RCS_FACTOR

        s.angular_vel = (s.angular_vel + angular_accel * DT) * GROUND_ANGULAR_DAMPING
        s.angle += s.angular_vel * DT
        self._place_on_pivot()

        # Toppled? Anything other than the pivot foot touching now means a
        # fin/nose/nozzle has hit the ground -- past the point of no
        # return. Keep falling for real instead of freezing here.
        for name, wx, wy, pen in self._contact_points():
            if name in ("foot_l", "foot_r"):
                continue
            if pen > 1e-6:
                self._begin_topple(name, wx)
                return False

        # Has the other foot reached the surface? -> settle/keep-rocking.
        other_lx = -lx
        ox, oy = self._local_to_world(other_lx, -FOOT_OFFSET)
        if self._surface_height(ox) - oy > -PLANT_EPS:
            self._resolve_flat_stance()
            return False

        # A single-foot rock can (rarely) find a stable near-equilibrium
        # and balance indefinitely without ever bringing the other foot
        # down within PLANT_EPS. Force a real settle/topple decision past
        # MAX_ROCK_SECONDS rather than let it hover out the whole episode.
        self._rock_timer += DT
        if self._rock_timer >= MAX_ROCK_SECONDS:
            self._resolve_flat_stance()

        return False

    def _resolve_contact(self):
        """Called once per step while airborne (not yet self._grounded) to
        detect first ground/pad contact. Classifies by whichever point
        penetrated deepest -- not "any hull point touched" -- since a fast
        fall can push the feet AND the fins/nozzle past the surface within
        the same single physics step, which would otherwise misfire as an
        instant crash even for a perfectly upright touchdown."""
        s = self.state
        pts = self._contact_points()
        name, wx, wy, pen = max(pts, key=lambda p: p[3])
        if pen <= 1e-6:
            return  # still airborne

        # Bare ground -- not the pad. There is nothing to land on here: any
        # contact, however gentle or upright, is instant, unconditional
        # destruction. No bounce, no rocking, no topple grace period --
        # otherwise the agent can (and did) learn a perfectly controlled
        # soft landing on ordinary ground, only to be told after the fact
        # it doesn't count (the earlier design only checked pad-vs-ground
        # once, at the very end, inside _resolve_flat_stance).
        if self._surface_height(wx) <= 0.0:
            s.y += pen
            self._finish_crash()
            return

        # A penetration far deeper than this step's own descent means the
        # rocket approached from the side and hit the raised pad's edge,
        # not its landing surface -- e.g. sliding in below deck height.
        # There's no ground to rest on there, so this stays instant.
        descent_this_step = max(0.0, -s.vy * DT)
        if pen > descent_this_step + 0.5:
            s.y += pen
            self._finish_crash()
            return

        if name not in ("foot_l", "foot_r"):
            # Arrived already badly oriented -- straight into the same
            # visible topple-fall as a mid-rock topple, not an instant
            # freeze.
            self._begin_topple(name, wx)
            return

        impact_speed = math.hypot(s.vx, s.vy)

        # Snapshot this touch's quality (speed/angle/centering) for grading
        # the landing bonus, since a successful settle zeroes velocity and
        # angle in _resolve_flat_stance -- by the time self.landed is set,
        # the state that made it a *good* landing is already gone. A bounce
        # re-enters here on its next touch, so this naturally reflects
        # whichever touch actually ends up sticking, not just the first one.
        self._touchdown_speed = impact_speed
        self._touchdown_angle = s.angle
        self._touchdown_dx = abs(s.x - self.pad_x)

        if impact_speed > CATASTROPHIC_IMPACT_SPEED:
            # Too fast for even a bounce to matter -- real destruction, no
            # reflected-velocity animation.
            s.y += pen
            self._finish_crash()
            return
        if impact_speed > MAX_IMPACT_SPEED:
            self._bounce(name, pen)
            return

        other_name = "foot_r" if name == "foot_l" else "foot_l"
        other_pen = next(p[3] for p in pts if p[0] == other_name)

        if abs(pen - other_pen) <= PLANT_EPS:
            self._resolve_flat_stance()
        else:
            self._rock_timer = 0.0  # fresh grounded period starting now
            pivot_lx = LEG_SPAN / 2.0 if name == "foot_r" else -LEG_SPAN / 2.0
            self._set_pivot(pivot_lx, -FOOT_OFFSET, self._pivot_omega(pivot_lx, -FOOT_OFFSET))

    def _spawn_landing_dust(self):
        for side in (-1.0, 1.0):
            wx, wy = self._local_to_world(side * LEG_SPAN / 2.0, -FOOT_OFFSET)
            self._emit_particles(
                wx, wy, side, 0.3,
                count=10, speed_range=(10.0, 40.0), spread=0.5,
                life_range=(0.3, 0.7), size_range=(2.0, 4.0), color=(180, 175, 160),
            )

    def _leg_contacts(self):
        """Returns (left_leg_touching, right_leg_touching), used for the
        observation and for color-coding the feet in the renderer."""
        pts = self._body_points()
        lx, ly = self._local_to_world(*pts["foot_l"])
        rx, ry = self._local_to_world(*pts["foot_r"])
        return ly <= self._surface_height(lx) + 0.5, ry <= self._surface_height(rx) + 0.5

    # ----------------------------------------------------------------
    # Reward / termination
    # ----------------------------------------------------------------

    def _shaping(self) -> float:
        """Potential function: higher (less negative) when closer to the
        pad, slower, and more upright. Scaled to match LunarLander's
        proportions (dist/speed/angle coefficients ~100/~100/~100, no extra
        0.01 flattening) so that one step of braking (which costs a real,
        immediate fuel penalty) pays back more in shaping than it costs --
        a much smaller speed coefficient made braking a net loss on every
        single step, so the only viable strategy was to barely thrust at
        all (never actually slow down) and just eat the terminal penalty."""
        s = self.state
        dx = (s.x - self.pad_x) / (WORLD_W / 2.0)
        dy = (s.y - self.pad_y) / (WORLD_H / 2.0)
        dist = math.hypot(dx, dy)
        speed = math.hypot(s.vx, s.vy)
        return -100.0 * dist - 0.2 * speed - 30.0 * abs(s.angle)

    def _compute_reward(self, terminated: bool) -> float:
        # Potential-based shaping: reward is the *change* in potential, so a
        # long careful descent doesn't rack up a large negative total the
        # way an absolute per-step penalty would (which would otherwise
        # teach an agent that crashing immediately is cheaper than flying).
        shaping = self._shaping()
        reward = shaping - self._prev_shaping
        self._prev_shaping = shaping

        # Small continuous penalty for firing engines, so the agent doesn't
        # burn fuel needlessly. Uses the effective (post fuel-cutoff)
        # throttles, so a dry tank never gets penalized for "phantom" thrust.
        # Kept small relative to the now-much-larger shaping terms above: at
        # full throttle one step's braking shaping gain (~0.36) still clears
        # this fuel cost (0.06) several times over, so thrust actually pays
        # for itself instead of "never really slow down" being optimal.
        reward -= self._last_main_throttle * 0.06
        reward -= abs(self._last_side_throttle) * 0.01

        # Small constant per-step cost. Without this, a policy that gets
        # close/slow/upright and then just loiters there for the full 1000
        # steps nets almost as much reward as actually landing (the shaping
        # swing from "far spawn" to "hovering near the pad" is worth up to
        # ~150-200 on its own, dwarfing fuel costs), with none of the risk
        # of a botched final touchdown -- confirmed directly: a trained
        # model's deterministic timeout episodes scored +90 to +155, often
        # more than a crash and comparable to landing. At 0.1/step this
        # costs -100 over a full-length timeout (enough to make "commit and
        # land" clearly better than "hover forever"), but only -20 to -70
        # over the ~200-700 steps a real landing or a quick crash takes, so
        # it doesn't reintroduce "crash immediately to end the episode
        # cheaply" (a crash's own -100 terminal penalty still dominates).
        reward -= 0.1

        if terminated:
            if self.landed:
                # Base +100 always (a landing is never worse than not
                # landing), plus up to +50 more graded by how clean the
                # touch that stuck actually was -- otherwise a landing at
                # 62 m/s right at the pad's edge scores identically to a
                # dead-center, dead-slow, perfectly upright one, giving no
                # gradient to refine *how well* it lands once it can land
                # at all.
                speed_frac = 1.0 - np.clip(self._touchdown_speed / MAX_IMPACT_SPEED, 0.0, 1.0)
                angle_frac = 1.0 - np.clip(abs(self._touchdown_angle) / TIP_ANGLE, 0.0, 1.0)
                center_frac = 1.0 - np.clip(self._touchdown_dx / PAD_HALF_WIDTH, 0.0, 1.0)
                quality = (speed_frac + angle_frac + center_frac) / 3.0
                reward += 100.0 + 50.0 * float(quality)
            else:
                reward -= 100.0

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        s = self.state
        left_leg, right_leg = self._leg_contacts()
        # The pad only actually drifts while airborne (see _update_pad_position /
        # _simulate) -- frozen once grounded, so its velocity is genuinely 0 then.
        pad_vx = 0.0 if self._grounded else self._pad_direction * PAD_MOVE_SPEED
        obs = np.array(
            [
                (s.x - self.pad_x) / (WORLD_W / 2.0),
                (s.y - self.pad_y) / (WORLD_H / 2.0),
                s.vx / 10.0,
                s.vy / 10.0,
                s.angle,
                s.angular_vel,
                float(left_leg),
                float(right_leg),
                pad_vx / 10.0,
            ],
            dtype=np.float32,
        )
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    # ----------------------------------------------------------------
    # Geometry helpers (shared by physics and rendering so the two never
    # drift out of sync)
    # ----------------------------------------------------------------

    def _local_dir(self, lx: float, ly: float):
        """Rotates a body-space direction vector into world space."""
        s = self.state
        cos_a, sin_a = math.cos(s.angle), math.sin(s.angle)
        return lx * cos_a + ly * sin_a, -lx * sin_a + ly * cos_a

    def _local_to_world(self, lx: float, ly: float):
        """Maps a body-space point (relative to the rocket's center) to world space."""
        dx, dy = self._local_dir(lx, ly)
        return self.state.x + dx, self.state.y + dy

    def _world_to_screen(self, x: float, y: float):
        return (
            self._vx + self._extra_x + x * self._scale + self._shake_offset[0],
            self._vy + self._extra_y + (WORLD_H - y) * self._scale + self._shake_offset[1],
        )

    def _poly_screen(self, local_pts):
        return [self._world_to_screen(*self._local_to_world(lx, ly)) for lx, ly in local_pts]

    # ----------------------------------------------------------------
    # Rendering setup
    # ----------------------------------------------------------------

    def _ensure_pygame(self):
        if pygame is None:
            raise RuntimeError(
                "pygame is required for rendering. Install it with `pip install pygame`."
            )
        if self.screen is not None:
            return

        pygame.init()
        pygame.display.set_caption("Hoverslam")
        self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        self.clock = pygame.time.Clock()

        self.font_small = pygame.font.SysFont("consolas", 14, bold=True)
        self.font = pygame.font.SysFont("consolas", 18, bold=True)
        self.font_big = pygame.font.SysFont("consolas", 24, bold=True)
        self.font_banner = pygame.font.SysFont("consolas", 52, bold=True)

        self._vx, self._vy = VIEW_MARGIN, HUD_H + VIEW_MARGIN
        self._vw = WINDOW_W - 2 * VIEW_MARGIN
        self._vh = WINDOW_H - HUD_H - 2 * VIEW_MARGIN
        self._scale = min(self._vw / WORLD_W, self._vh / WORLD_H)
        self._extra_x = (self._vw - WORLD_W * self._scale) / 2.0
        self._extra_y = (self._vh - WORLD_H * self._scale) / 2.0

        self._stars = [
            (
                random.uniform(0, WINDOW_W),
                random.uniform(0, self._vy + self._vh * 0.9),
                random.uniform(0.6, 1.8),
                random.uniform(0, math.tau),
            )
            for _ in range(110)
        ]

        self._bg_surface = self._build_background_surface()
        self._vignette_surface = self._build_vignette_surface()

    def _build_background_surface(self):
        surf = pygame.Surface((WINDOW_W, WINDOW_H))
        top = (8, 10, 26)
        mid = (24, 19, 44)
        horizon = (55, 36, 52)
        ground_y = self._vy + self._extra_y + WORLD_H * self._scale

        for y in range(WINDOW_H):
            t = min(1.0, y / max(ground_y, 1.0))
            if t < 0.6:
                tt = t / 0.6
                color = tuple(int(top[i] + (mid[i] - top[i]) * tt) for i in range(3))
            else:
                tt = (t - 0.6) / 0.4
                color = tuple(int(mid[i] + (horizon[i] - mid[i]) * tt) for i in range(3))
            pygame.draw.line(surf, color, (0, y), (WINDOW_W, y))
        return surf

    def _build_vignette_surface(self):
        try:
            yy, xx = np.mgrid[0:WINDOW_H, 0:WINDOW_W]
            cx, cy = WINDOW_W / 2.0, WINDOW_H / 2.0
            dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
            alpha = np.clip((dist - 0.55) * 200.0, 0, 130).astype(np.uint8).T

            surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            rgb = np.zeros((WINDOW_W, WINDOW_H, 3), dtype=np.uint8)
            pygame.surfarray.blit_array(surf, rgb)
            alpha_view = pygame.surfarray.pixels_alpha(surf)
            alpha_view[:, :] = alpha
            del alpha_view
            return surf
        except Exception:
            return None

    # ----------------------------------------------------------------
    # Particle effects (purely cosmetic)
    # ----------------------------------------------------------------

    def _emit_particles(self, x, y, dir_x, dir_y, count, speed_range, spread, life_range, size_range, color):
        base_angle = math.atan2(dir_y, dir_x)
        for _ in range(count):
            ang = base_angle + random.uniform(-spread, spread)
            spd = random.uniform(*speed_range)
            life = random.uniform(*life_range)
            self._particles.append(
                {
                    "x": x,
                    "y": y,
                    "vx": math.cos(ang) * spd,
                    "vy": math.sin(ang) * spd,
                    "life": life,
                    "max_life": life,
                    "size": random.uniform(*size_range),
                    "color": color,
                }
            )

    def _spawn_engine_particles(self):
        if self._wrecked or self._toppling:
            return

        half_h = ROCKET_HEIGHT / 2.0
        half_w = ROCKET_WIDTH / 2.0

        if self._last_main_throttle > 0.03:
            wx, wy = self._local_to_world(0.0, -half_h - 6.0)
            dx, dy = self._local_dir(0.0, -1.0)
            self._emit_particles(
                wx, wy, dx, dy,
                count=int(2 + 4 * self._last_main_throttle),
                speed_range=(30.0, 70.0 + 50.0 * self._last_main_throttle),
                spread=0.35,
                life_range=(0.25, 0.55),
                size_range=(2.0, 4.5),
                color=(255, 170, 60),
            )

        if abs(self._last_side_throttle) > 0.03:
            # Positive side_throttle spins the rocket clockwise, which needs
            # the reaction thrust pushing the nose right -> exhaust vents left.
            side = -1.0 if self._last_side_throttle > 0 else 1.0
            wx, wy = self._local_to_world(side * half_w, half_h * 0.5)
            dx, dy = self._local_dir(side, 0.0)
            self._emit_particles(
                wx, wy, dx, dy,
                count=2,
                speed_range=(15.0, 35.0),
                spread=0.3,
                life_range=(0.2, 0.4),
                size_range=(1.5, 3.0),
                color=(140, 200, 255),
            )

    def _trigger_terminal_fx(self, color, fireball=False):
        self._flash_color = color
        self._flash_alpha = 170.0
        if fireball:
            self._fireball_timer = 0.8
            self._shake = 14.0 * self._blast_scale

        if self.state is None:
            return

        origin = self._impact_point if self._impact_point is not None else (self.state.x, max(self.state.y, 0.5))
        self._emit_particles(
            origin[0], origin[1], 0.0, 1.0,
            count=55 if fireball else 36,
            speed_range=(30.0, 140.0) if fireball else (20.0, 90.0),
            spread=math.pi,
            life_range=(0.5, 1.3) if fireball else (0.4, 1.0),
            size_range=(2.0, 5.0),
            color=color,
        )
        if fireball:
            # Drifting smoke plume on top of the fiery debris burst above.
            self._emit_particles(
                origin[0], origin[1], 0.0, 1.0,
                count=20, speed_range=(5.0, 25.0), spread=0.6,
                life_range=(0.8, 1.6), size_range=(3.0, 7.0), color=(90, 90, 95),
            )

    def _update_and_draw_particles(self):
        alive = []
        for p in self._particles:
            p["x"] += p["vx"] * DT
            p["y"] += p["vy"] * DT
            p["life"] -= DT
            if p["life"] > 0:
                alive.append(p)
        self._particles = alive
        if not self._particles:
            return

        surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        for p in self._particles:
            t = p["life"] / p["max_life"]
            alpha = max(0, min(255, int(255 * t)))
            r, g, b = p["color"]
            sx, sy = self._world_to_screen(p["x"], p["y"])
            radius = max(1, int(p["size"] * self._scale * t + 1))
            pygame.draw.circle(surf, (r, g, b, alpha), (int(sx), int(sy)), radius)
        self.screen.blit(surf, (0, 0))

    # ----------------------------------------------------------------
    # Scene drawing
    # ----------------------------------------------------------------

    def _draw_stars(self):
        t = self._frame * DT
        for x, y, r, phase in self._stars:
            b = 140 + int(100 * (0.5 + 0.5 * math.sin(t * 2.0 + phase)))
            b = max(60, min(255, b))
            pygame.draw.circle(self.screen, (b, b, min(255, b + 15)), (int(x), int(y)), max(1, int(r)))

    def _draw_ground_and_pad(self):
        ground_y = int(self._world_to_screen(0, 0)[1])
        pygame.draw.rect(self.screen, (46, 41, 37), pygame.Rect(0, ground_y, WINDOW_W, WINDOW_H - ground_y))
        pygame.draw.line(self.screen, (92, 82, 72), (0, ground_y), (WINDOW_W, ground_y), 3)
        for hx in range(0, WINDOW_W + 40, 26):
            pygame.draw.line(self.screen, (58, 52, 46), (hx, ground_y + 4), (hx - 16, WINDOW_H), 1)

        # The pad is a raised deck standing on two support pylons above the
        # ground line. Its drawn TOP EDGE must exactly equal the true
        # collision surface (PAD_HEIGHT) -- the rocket's feet rest there --
        # with the deck's visual thickness extending downward from it, not
        # straddling it (that mismatch was what made the rocket look like
        # it was sinking into the platform instead of standing on it).
        pad_left_w = self.pad_x - PAD_HALF_WIDTH
        pad_right_w = self.pad_x + PAD_HALF_WIDTH
        plx, deck_top_y = self._world_to_screen(pad_left_w, PAD_HEIGHT)
        prx, _ = self._world_to_screen(pad_right_w, PAD_HEIGHT)
        deck_rect = pygame.Rect(int(plx), int(deck_top_y), int(prx - plx), 12)

        pylon_w = 10
        for px in (deck_rect.left + 6, deck_rect.right - 6 - pylon_w):
            pygame.draw.rect(self.screen, (55, 58, 66), (px, deck_rect.bottom, pylon_w, max(0, ground_y - deck_rect.bottom)))
        pygame.draw.line(self.screen, (55, 58, 66), (deck_rect.left + 6, deck_rect.bottom), (deck_rect.right - 6, ground_y), 2)
        pygame.draw.line(self.screen, (55, 58, 66), (deck_rect.right - 6, deck_rect.bottom), (deck_rect.left + 6, ground_y), 2)

        for i in range(deck_rect.height):
            tt = i / max(deck_rect.height - 1, 1)
            c = tuple(int(148 + (88 - 148) * tt) for _ in range(3))
            pygame.draw.line(self.screen, c, (deck_rect.left, deck_rect.top + i), (deck_rect.right, deck_rect.top + i))
        pygame.draw.rect(self.screen, (20, 20, 24), deck_rect, width=2)

        stripe_w = 14
        x = deck_rect.left
        flip = False
        while x < deck_rect.right:
            w = min(stripe_w, deck_rect.right - x)
            color = (30, 30, 20) if flip else (232, 190, 40)
            pygame.draw.rect(self.screen, color, (x, deck_rect.top - 5, w, 5))
            flip = not flip
            x += stripe_w

        blink = (self._frame // 15) % 2 == 0
        light_color = (250, 90, 90) if blink else (90, 220, 250)
        for lx in (deck_rect.left, deck_rect.right):
            pygame.draw.circle(self.screen, light_color, (lx, deck_rect.top), 4)

    def _draw_rocket(self):
        half_w, half_h = ROCKET_WIDTH / 2.0, ROCKET_HEIGHT / 2.0

        if self._wrecked:
            hull_color, shade_color, outline_color = (74, 70, 66), (52, 49, 46), (28, 26, 24)
            nose_color, nose_outline = (96, 54, 46), (40, 24, 20)
            fin_color, nozzle_color = (46, 43, 40), (48, 46, 44)
        else:
            hull_color, shade_color, outline_color = (212, 216, 224), (164, 168, 180), (110, 114, 126)
            nose_color, nose_outline = (222, 92, 72), (120, 50, 45)
            fin_color, nozzle_color = (58, 62, 74), (66, 68, 78)

        pts = self._body_points()
        shoulder_l, shoulder_r = pts["shoulder_l"], pts["shoulder_r"]
        base_l, base_r = pts["base_l"], pts["base_r"]
        nose = pts["nose"]

        # Fins (drawn first so the hull overlaps their inner edge). Same
        # points _check_contact uses, so the fins never silently poke below
        # the feet again (that was the bug: fins/nozzle registering ground
        # contact before the landing legs did).
        fin_l = [base_l, pts["fin_l_tip"], pts["fin_l_root"]]
        fin_r = [base_r, pts["fin_r_tip"], pts["fin_r_root"]]
        pygame.draw.polygon(self.screen, fin_color, self._poly_screen(fin_l))
        pygame.draw.polygon(self.screen, fin_color, self._poly_screen(fin_r))

        # Engine nozzle
        nozzle = [
            (-half_w * 0.5, -half_h), (half_w * 0.5, -half_h),
            pts["nozzle_br"], pts["nozzle_bl"],
        ]
        pygame.draw.polygon(self.screen, nozzle_color, self._poly_screen(nozzle))

        # Main hull + cylindrical shading + outline
        body = [shoulder_l, shoulder_r, base_r, base_l]
        body_screen = self._poly_screen(body)
        pygame.draw.polygon(self.screen, hull_color, body_screen)
        shade = [shoulder_l, (-half_w * 0.2, half_h * 0.55), (-half_w * 0.2, -half_h), base_l]
        pygame.draw.polygon(self.screen, shade_color, self._poly_screen(shade))
        pygame.draw.polygon(self.screen, outline_color, body_screen, width=2)

        # Nose cone
        nose_poly = [shoulder_l, nose, shoulder_r]
        pygame.draw.polygon(self.screen, nose_color, self._poly_screen(nose_poly))
        pygame.draw.polygon(self.screen, nose_outline, self._poly_screen(nose_poly), width=2)

        # Porthole window
        wsx, wsy = self._world_to_screen(*self._local_to_world(0.0, half_h * 0.15))
        r = max(2, int(4 * self._scale))
        window_color = (70, 70, 75) if self._wrecked else (125, 195, 235)
        pygame.draw.circle(self.screen, window_color, (int(wsx), int(wsy)), r)
        pygame.draw.circle(self.screen, (25, 25, 35), (int(wsx), int(wsy)), r, width=1)

        # Landing legs, color-coded by ground contact
        left_contact, right_contact = self._leg_contacts()
        leg_color = (90, 88, 86) if self._wrecked else (150, 154, 166)
        for side, contact, foot_pt in ((-1.0, left_contact, pts["foot_l"]), (1.0, right_contact, pts["foot_r"])):
            hip = self._world_to_screen(*self._local_to_world(side * half_w * 0.7, -half_h * 0.3))
            knee = self._world_to_screen(*self._local_to_world(side * (LEG_SPAN / 2.0) * 0.8, -half_h * 0.9))
            foot = self._world_to_screen(*self._local_to_world(*foot_pt))
            pygame.draw.line(self.screen, leg_color, hip, knee, 3)
            pygame.draw.line(self.screen, leg_color, knee, foot, 3)
            foot_color = (110, 230, 140) if (contact and not self._wrecked) else leg_color
            pygame.draw.circle(self.screen, foot_color, (int(foot[0]), int(foot[1])), 4)

        if self._wrecked or self._toppling:
            return  # no working engines once wrecked, and a topple can't be thrust-recovered

        # Main engine flame
        if self._last_main_throttle > 0.03:
            t = self._last_main_throttle
            flicker = 1.0 + random.uniform(-0.08, 0.08)
            flame_len = (14.0 + t * 46.0) * flicker
            outer = [
                (-half_w * 0.35, -half_h - 8),
                (0.0, -half_h - 8 - flame_len),
                (half_w * 0.35, -half_h - 8),
            ]
            pygame.draw.polygon(self.screen, (255, 140, 40), self._poly_screen(outer))
            inner = [
                (-half_w * 0.18, -half_h - 8),
                (0.0, -half_h - 8 - flame_len * 0.55),
                (half_w * 0.18, -half_h - 8),
            ]
            pygame.draw.polygon(self.screen, (255, 235, 160), self._poly_screen(inner))

        # Side thruster puff
        if abs(self._last_side_throttle) > 0.03:
            # Positive side_throttle spins the rocket clockwise, which needs
            # the reaction thrust pushing the nose right -> exhaust vents left.
            side = -1.0 if self._last_side_throttle > 0 else 1.0
            strength = abs(self._last_side_throttle)
            puff = [
                (side * half_w, half_h * 0.5 + 4),
                (side * (half_w + 10 + 18 * strength), half_h * 0.5),
                (side * half_w, half_h * 0.5 - 4),
            ]
            pygame.draw.polygon(self.screen, (120, 190, 255), self._poly_screen(puff))

    def _draw_hud(self):
        hud_surf = pygame.Surface((WINDOW_W, HUD_H), pygame.SRCALPHA)
        hud_surf.fill((12, 14, 24, 220))
        pygame.draw.line(hud_surf, (90, 200, 255, 180), (0, HUD_H - 2), (WINDOW_W, HUD_H - 2), 2)
        self.screen.blit(hud_surf, (0, 0))

        s = self.state

        def tier_color(value, safe, warn):
            if value <= safe:
                return (110, 230, 140)
            if value <= warn:
                return (250, 210, 90)
            return (250, 90, 90)

        altitude = max(0.0, s.y - self.pad_y)
        speed = math.hypot(s.vx, s.vy)
        # Two independent ways to fail: landing too HARD (speed, vs.
        # MAX_IMPACT_SPEED) and landing unbalanced (tilt, vs. TIP_ANGLE) --
        # both get colored so either danger is visible before you touch down.
        speed_color = tier_color(speed, MAX_IMPACT_SPEED * 0.6, MAX_IMPACT_SPEED)
        angle_color = tier_color(abs(s.angle), TIP_ANGLE * 0.6, TIP_ANGLE)

        self.screen.blit(self.font_big.render(f"ALT {altitude:6.1f} m", True, (230, 235, 245)), (24, 12))
        self.screen.blit(
            self.font.render(f"VX {s.vx:+5.1f}  VY {s.vy:+5.1f} m/s", True, speed_color), (24, 42)
        )
        self.screen.blit(self.font_small.render(f"SPD {speed:5.1f} m/s", True, (160, 175, 195)), (24, 66))

        self.screen.blit(
            self.font_big.render(f"TILT {math.degrees(s.angle):+5.1f} deg", True, angle_color), (300, 12)
        )
        self.screen.blit(
            self.font.render(f"SPIN {math.degrees(s.angular_vel):+5.1f} deg/s", True, (160, 175, 195)),
            (300, 42),
        )
        self.screen.blit(self.font_small.render(f"STEP {self.steps:4d}", True, (120, 130, 150)), (300, 66))

        fuel_frac = max(0.0, s.fuel / 100.0)
        bar_x, bar_y, bar_w, bar_h = WINDOW_W - 260, 30, 200, 18
        pygame.draw.rect(self.screen, (40, 44, 56), (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        fuel_color = (110, 230, 140) if fuel_frac > 0.4 else ((250, 210, 90) if fuel_frac > 0.15 else (250, 90, 90))
        pygame.draw.rect(self.screen, fuel_color, (bar_x, bar_y, int(bar_w * fuel_frac), bar_h), border_radius=6)
        pygame.draw.rect(self.screen, (90, 100, 120), (bar_x, bar_y, bar_w, bar_h), width=1, border_radius=6)
        self.screen.blit(self.font_small.render("FUEL", True, (160, 175, 195)), (bar_x, bar_y - 18))

    def _draw_fireball(self):
        if self._fireball_timer <= 0 or self._impact_point is None:
            return
        t = 1.0 - (self._fireball_timer / 0.8)
        alpha = max(0, int(255 * (1.0 - t)))
        if alpha <= 0:
            return
        radius_world = 8.0 + t * 55.0 * self._blast_scale
        sx, sy = self._world_to_screen(*self._impact_point)
        r_px = max(1, int(radius_world * self._scale))
        surf = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
        pygame.draw.circle(surf, (255, 150, 40, alpha), (int(sx), int(sy)), r_px, width=max(2, int(r_px * 0.25)))
        pygame.draw.circle(surf, (255, 220, 120, int(alpha * 0.6)), (int(sx), int(sy)), max(1, int(r_px * 0.55)))
        self.screen.blit(surf, (0, 0))

    def _draw_banner(self, hint: str = ""):
        if not (self.landed or self.crashed):
            return
        text = "LANDING SUCCESSFUL" if self.landed else "CRASH"
        color = (110, 230, 140) if self.landed else (250, 90, 90)
        text_surf = self.font_banner.render(text, True, color)

        box_w = text_surf.get_width() + 40
        box_h = text_surf.get_height() + 24
        hint_surf = None
        if hint:
            hint_surf = self.font.render(hint, True, (210, 215, 225))
            box_w = max(box_w, hint_surf.get_width() + 40)
            box_h += hint_surf.get_height() + 10

        box = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        box.fill((10, 10, 20, 190))
        box.blit(text_surf, (20, 12))
        if hint_surf is not None:
            box.blit(hint_surf, (20, text_surf.get_height() + 22))
        rect = box.get_rect(center=(WINDOW_W // 2, self._vy + self._vh // 2))
        self.screen.blit(box, rect)

    def _render_frame(self, banner_hint: str = ""):
        self._ensure_pygame()
        self._frame += 1

        # Screen shake and the fireball timer are driven by wall-clock
        # frames (not self.steps), so they keep animating even after the
        # episode has ended and physics stepping has stopped.
        if self._shake > 0.15:
            self._shake_offset = (
                random.uniform(-1.0, 1.0) * self._shake,
                random.uniform(-1.0, 1.0) * self._shake,
            )
            self._shake *= 0.85
        else:
            self._shake = 0.0
            self._shake_offset = (0.0, 0.0)

        self.screen.blit(self._bg_surface, (0, 0))
        self._draw_stars()
        self._draw_ground_and_pad()
        self._spawn_engine_particles()
        self._draw_rocket()
        self._update_and_draw_particles()
        self._draw_fireball()
        if self._fireball_timer > 0:
            self._fireball_timer = max(0.0, self._fireball_timer - DT)

        if self._vignette_surface is not None:
            self.screen.blit(self._vignette_surface, (0, 0))

        if self._flash_alpha > 0:
            alpha = int(min(255, self._flash_alpha))
            overlay = pygame.Surface((WINDOW_W, WINDOW_H), pygame.SRCALPHA)
            overlay.fill((*self._flash_color, alpha))
            self.screen.blit(overlay, (0, 0))
            self._flash_alpha = max(0.0, self._flash_alpha - 260.0 * DT)

        self._draw_hud()
        self._draw_banner(hint=banner_hint)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(FPS)
            return None
        else:
            arr = pygame.surfarray.array3d(self.screen)
            return np.transpose(arr, (1, 0, 2))


# --------------------------------------------------------------------------
# Human-playable loop
# --------------------------------------------------------------------------

def play_human():
    """Run the environment with keyboard controls.

    Controls:
        Up arrow     -> main engine thrust
        Left arrow   -> rotate counter-clockwise (fires the right-side RCS thruster)
        Right arrow  -> rotate clockwise (fires the left-side RCS thruster)
        R            -> reset episode (also how you leave the platform after landing/crashing)
        Esc / close  -> quit
    """
    if pygame is None:
        raise RuntimeError("pygame is required to play. Install it with `pip install pygame`.")

    env = RocketLanderEnv(render_mode="human")
    obs, info = env.reset()

    running = True
    total_reward = 0.0
    # Once True, the rocket stays exactly where it landed/crashed -- the
    # episode does NOT auto-reset. Only pressing R starts a new attempt.
    episode_over = False

    while running:
        want_reset = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    want_reset = True

        if want_reset:
            obs, info = env.reset()
            total_reward = 0.0
            episode_over = False
            continue

        if episode_over:
            # The episode is over: hold here (rocket stays resting exactly
            # where it landed/crashed) until the player presses R.
            env._render_frame(banner_hint="Press R to fly again")
            continue

        main = 0.0
        side = 0.0
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            main = 1.0
        if keys[pygame.K_LEFT]:
            side = -1.0
        if keys[pygame.K_RIGHT]:
            side = 1.0

        # Map throttle-style input [0,1] main back into the env's [-1,1]
        # action convention (env maps action[0] from [-1,1] -> [0,1]).
        action = np.array([main * 2.0 - 1.0, side], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            state = "LANDED" if info.get("landed") else ("CRASHED" if info.get("crashed") else "TIME UP")
            print(f"Episode finished: {state} | total_reward={total_reward:.2f}")
            if terminated:
                # Ground contact (landed or crashed): stay put until R is
                # pressed. A bare time-up (truncated only) still auto-resets
                # since there's nothing resting in place to look at.
                episode_over = True
            else:
                obs, info = env.reset()
                total_reward = 0.0

    env.close()


if __name__ == "__main__":
    play_human()
