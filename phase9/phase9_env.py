"""
================================================================================
PHASE 9 — RL COLLISION-AVOIDANCE ENVIRONMENT
================================================================================
A Gymnasium environment where an RL agent learns to resolve a conjunction with
MINIMUM fuel by choosing 3-axis impulsive burns at a sequence of decision points
before TCA. This is the novel "decision intelligence" layer: instead of the
rule-based single along-track burn (Phase 8), the policy learns WHEN and IN WHICH
DIRECTION to burn for the best fuel/safety trade.

PHYSICS — Clohessy-Wiltshire (Hill's) equations, the standard linearized model
for relative motion near a circular orbit. The displacement of the primary at
TCA produced by an impulsive Δv = (dvR, dvT, dvN) applied a time t earlier:

    θ = n·t,  s = sinθ,  c = cosθ      (n = mean motion)
    Δx(t) =  (s/n)·dvR + (2/n)(1−c)·dvT                 (radial)
    Δy(t) = −(2/n)(1−c)·dvR + (1/n)(4s − 3θ)·dvT        (along-track, secular)
    Δz(t) =  (s/n)·dvN                                  (cross-track)

The along-track secular term (−3θ/n)·dvT → −3·t·dvT for large θ, which matches
Phase 8's first-order along-track drift (3·dv·dt). CW adds the correct radial and
cross-track responses, so the agent can exploit all three axes. This is a
recognised, defensible model — not an ad-hoc rule.

EPISODE: K decision points at decreasing time-to-TCA. At each, the agent applies
a Δv (bounded). Displacements accumulate (linear superposition under CW). After
the last point, the resulting miss + the Phase-7 Pc engine give the final Pc.

REWARD: large terminal bonus for resolving (Pc < PC_SAFE) minus total fuel; per-
step fuel penalty; failure penalty scaled by residual risk. Among policies that
resolve, lower total Δv scores higher → the agent learns fuel-optimal avoidance.

Catalog re-screening (new-conjunction safety) is NOT done inside the fast env —
it is enforced afterward by the Phase-10 safety layer on the policy's output.
Train fast here; validate rigorously there.
================================================================================
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from phase7_collision import (project_to_plane, pc_2d_quadrature, rtn_to_eci_cov,
                              secondary_covariance_rtn)

MU = 398600.4418
PC_SAFE = 1e-6
PC_THRESHOLD = 1e-4
G0 = 9.80665e-3       # km/s^2
ISP_S = 220.0


def cw_displacement(dv_rtn, t_s, n):
    """Clohessy-Wiltshire position displacement (km, RTN) from impulse dv (km/s)
       applied t_s seconds before the evaluation epoch. n = mean motion (rad/s)."""
    dvR, dvT, dvN = dv_rtn
    th = n * t_s
    s, c = np.sin(th), np.cos(th)
    dx = (s / n) * dvR + (2.0 / n) * (1 - c) * dvT
    dy = -(2.0 / n) * (1 - c) * dvR + (1.0 / n) * (4 * s - 3 * th) * dvT
    dz = (s / n) * dvN
    return np.array([dx, dy, dz])


class ConjunctionEnv(gym.Env):
    """RL environment for fuel-optimal collision avoidance."""
    metadata = {"render_modes": []}

    def __init__(self, n_decision_points=4, max_dv_ms=0.5, hbr_km=0.020,
                 sat_mass_kg=2300.0, seed=None):
        super().__init__()
        self.K = n_decision_points
        self.max_dv = max_dv_ms / 1000.0           # km/s per axis per step
        self.hbr = hbr_km
        self.mass = sat_mass_kg
        # action: 3-axis Δv in [-1,1] (scaled by max_dv).  burn-now is implicit per step
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        # obs: [miss_R, miss_T, miss_N (norm km), vrel_norm, sig_maj, sig_min,
        #       log10Pc, time_frac, fuel_frac]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(9,), dtype=np.float32)
        self.rng = np.random.default_rng(seed)

    # ---- scenario sampling ----
    def _sample_conjunction(self):
        # primary near-circular LEO ~700 km
        alt = self.rng.uniform(500, 800)
        r = 6378.137 + alt
        self.n = np.sqrt(MU / r**3)                 # mean motion
        self.period = 2 * np.pi / self.n
        speed = np.sqrt(MU / r)
        # primary state in a simple frame: position along x, velocity along y
        self.rp = np.array([r, 0.0, 0.0])
        self.vp = np.array([0.0, speed, 0.0])
        # secondary: crossing course, small miss in RTN
        miss = self.rng.uniform(0.01, 0.30)         # km
        ang = self.rng.uniform(0, 2 * np.pi)
        self.miss0 = np.array([miss * np.cos(ang), 0.0, miss * np.sin(ang)])
        v2dir = self.rng.normal(size=3); v2dir /= np.linalg.norm(v2dir)
        self.vrel = speed * v2dir                    # high relative speed
        # covariances (RTN) -> combined, projected later
        self.Cp_rtn = np.diag([0.04, 0.20, 0.05])**2
        self.C2_rtn = np.diag([0.05, 0.30, 0.08])**2
        self.tca_total = self.rng.uniform(4, 12) * 3600.0   # seconds to TCA

    def _pc_now(self, disp):
        """Pc given accumulated primary displacement (RTN km)."""
        rel = self.miss0 + disp                      # secondary - primary, RTN
        C = self.Cp_rtn + self.C2_rtn                # combined RTN cov (approx)
        # encounter-plane projection in RTN using relative velocity direction
        miss2d, C2d = project_to_plane(rel, C, self.vrel)
        return pc_2d_quadrature(miss2d, C2d, self.hbr), float(np.linalg.norm(rel))

    def _obs(self):
        rel = self.miss0 + self.disp
        pc, _ = self._pc_now(self.disp)
        sig = np.sqrt(np.linalg.eigvalsh(self.Cp_rtn + self.C2_rtn))
        return np.array([rel[0], rel[1], rel[2], np.linalg.norm(self.vrel),
                         sig.max(), sig.min(), np.log10(max(pc, 1e-30)),
                         self.step_i / self.K, self.fuel_frac], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._sample_conjunction()
        self.disp = np.zeros(3)
        self.total_dv = 0.0
        self.step_i = 0
        self.fuel_frac = 1.0
        self.pc0, _ = self._pc_now(self.disp)
        return self._obs(), {}

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1, 1)
        dv = a * self.max_dv                          # km/s this step, RTN
        # time from this decision point to TCA
        frac_remaining = 1.0 - self.step_i / self.K
        t_to_tca = self.tca_total * frac_remaining
        # accumulate CW displacement contribution of this burn
        self.disp = self.disp + cw_displacement(dv, t_to_tca, self.n)
        dv_mag = float(np.linalg.norm(dv)) * 1000.0   # m/s
        self.total_dv += dv_mag
        self.fuel_frac = max(0.0, self.fuel_frac - dv_mag / 2.0)   # crude budget

        self.step_i += 1
        pc, miss = self._pc_now(self.disp)

        # per-step shaping: fuel penalty (meaningful weight)
        reward = -0.5 * dv_mag
        done = False
        if self.step_i >= self.K:
            done = True
            if pc < PC_SAFE:
                # resolved. fixed success bonus + strong fuel economy incentive:
                # the less total Δv used, the higher the reward.
                reward += 50.0 - 8.0 * self.total_dv
            else:
                # failed: penalty scaled by how far above safe (log decades)
                reward += -25.0 * (np.log10(pc) - np.log10(PC_SAFE))
        return self._obs(), float(reward), done, False, {
            "pc": pc, "miss_km": miss, "total_dv_ms": self.total_dv}
