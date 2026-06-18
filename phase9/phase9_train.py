"""
PHASE 9 — train the RL maneuver-optimizer policy (PPO via stable-baselines3).

    python phase9_train.py --steps 300000        # full train (run like the orbit model)
    python phase9_train.py --steps 20000         # quick smoke

Saves model/phase9_ppo.zip. The policy learns to resolve conjunctions (Pc below
PC_SAFE) using minimum total Δv, exploiting all three burn axes and burn timing
(earlier decision points have more time-to-TCA leverage under CW dynamics).

Trains in the fast ConjunctionEnv. The resulting policy is wrapped by
phase9_policy.py and validated by the Phase-10 safety layer before any use.
"""
import os, sys, argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from phase9_env import ConjunctionEnv, PC_SAFE


class ProgressCb(BaseCallback):
    """Log resolve-rate and mean Δv periodically."""
    def __init__(self, every=10000):
        super().__init__()
        self.every = every

    def _on_step(self):
        if self.num_timesteps % self.every == 0:
            rr, dv = evaluate(self.model, n=120, quiet=True)
            print(f"  [{self.num_timesteps:>7}] resolve_rate={rr:.1%}  mean_Δv={dv:.3f} m/s")
        return True


def evaluate(model, n=300, quiet=False):
    """Run n episodes greedily; return (resolve_rate, mean_dv_of_resolved)."""
    env = ConjunctionEnv(seed=12345)
    resolved, dvs = 0, []
    for ep in range(n):
        obs, _ = env.reset(seed=10000 + ep)
        done = False; info = {}
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            obs, r, done, _, info = env.step(a)
        if info["pc"] < PC_SAFE:
            resolved += 1; dvs.append(info["total_dv_ms"])
    rr = resolved / n
    mdv = float(np.mean(dvs)) if dvs else float("nan")
    if not quiet:
        print(f"eval: resolve_rate={rr:.1%}  mean_Δv(resolved)={mdv:.3f} m/s  (n={n})")
    return rr, mdv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--envs", type=int, default=8)
    args = ap.parse_args()

    venv = make_vec_env(lambda: ConjunctionEnv(), n_envs=args.envs)
    model = PPO("MlpPolicy", venv, verbose=0, n_steps=512, batch_size=256,
                gae_lambda=0.95, gamma=0.99, ent_coef=0.005, learning_rate=3e-4,
                policy_kwargs=dict(net_arch=[128, 128]))
    print(f"training PPO for {args.steps} steps on {args.envs} envs ...")
    model.learn(total_timesteps=args.steps, callback=ProgressCb(), progress_bar=False)
    os.makedirs("model", exist_ok=True)
    model.save("model/phase9_ppo")
    print("saved model/phase9_ppo.zip")
    evaluate(model, n=300)


if __name__ == "__main__":
    main()
