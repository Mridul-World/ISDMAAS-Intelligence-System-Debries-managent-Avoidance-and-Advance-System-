# Phase 9 — RL Maneuver Optimizer

Reinforcement-learning policy (PPO) that proposes fuel-optimal 3-axis avoidance
burns, wrapped so it plugs into the Phase-11 API behind propose_maneuver(), with
a rule-based (Phase-8) fallback. Every RL output is still validated by Phase-10
safety — RL optimizes, deterministic logic guarantees safety.

## Files
- phase9_env.py      Gymnasium env, Clohessy-Wiltshire dynamics + Pc engine
- phase9_train.py    PPO training (stable-baselines3)
- phase9_policy.py   policy wrapper + rule-based fallback (API drop-in)
- phase9_selftest.py env checks + RL-vs-rule-based fuel benchmark
- phase7_collision.py, phase8_maneuver.py   (copied dependencies)

## Install
pip install gymnasium stable-baselines3

## Train (run like the orbit model — overnight for best policy)
python phase9_train.py --steps 300000 --envs 8     # full
python phase9_train.py --steps 30000               # quick smoke
# saves model/phase9_ppo.zip

## Test + benchmark
python phase9_selftest.py

## Plug into the API (phase11_api.py)
Replace inside propose_maneuver():
    return plan_maneuver(...)
with:
    from phase9_policy import propose_maneuver_rl
    return propose_maneuver_rl(rp, vp, Cp, r2, v2, C2, tca_s, sat_mass_kg)
The console / autonomous mode / Phase-10 validation all work unchanged; the
telemetry will show proposer = "rl_ppo" or a fallback label.

## Design notes (for paper / pitch)
- Dynamics: Clohessy-Wiltshire (Hill's) linearised relative motion — standard,
  consistent with Phase-8's along-track first-order term, adds radial/cross-track.
- Reward: resolve (Pc < 1e-6) with minimum total Δv; failure penalised by residual
  risk decades. Among resolving policies, lower Δv scores higher -> fuel economy.
- Safety: the fast env omits catalog re-screen for speed; the Phase-10 layer
  enforces no-new-conjunction / fuel / orbit safety on the policy's output.
- Fallback: if RL fails to resolve or the rule-based burn is cheaper, the wrapper
  returns the rule-based maneuver. RL can only improve, never degrade, safety.
