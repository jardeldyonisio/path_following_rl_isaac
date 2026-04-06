import argparse
import os
import sys
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect env action/observation space")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper

sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import TurtlebotNavEnvCfg

cfg = TurtlebotNavEnvCfg()
cfg.scene.num_envs = args.num_envs
cfg.seed = 42

env = ManagerBasedRLEnv(cfg=cfg)
wrapped = SkrlVecEnvWrapper(env, ml_framework="torch")

print("[raw env] action_space:", env.action_space)
print("[raw env] obs_space:", env.observation_space)
print("[wrapped] action_space:", wrapped.action_space)
print("[wrapped] obs_space:", wrapped.observation_space)

for name, space in [("raw", env.action_space), ("wrapped", wrapped.action_space)]:
    low = np.asarray(space.low)
    high = np.asarray(space.high)
    print(f"[{name}] low finite={np.isfinite(low).all()} min={np.nanmin(low)} max={np.nanmax(low)}")
    print(f"[{name}] high finite={np.isfinite(high).all()} min={np.nanmin(high)} max={np.nanmax(high)}")

print("[inspect] wrapped type:", type(wrapped))
print("[inspect] raw type:", type(env))
print("[inspect] wrapped has _action_space:", hasattr(wrapped, "_action_space"))
print("[inspect] raw has _action_space:", hasattr(env, "_action_space"))
if hasattr(wrapped, "_action_space"):
    print("[inspect] wrapped._action_space:", wrapped._action_space)
if hasattr(env, "_action_space"):
    print("[inspect] raw._action_space:", env._action_space)

for attr_name in ["_env", "_unwrapped", "unwrapped", "env"]:
    if hasattr(wrapped, attr_name):
        obj = getattr(wrapped, attr_name)
        print(f"[inspect] wrapped.{attr_name}:", type(obj))
        if hasattr(obj, "action_space"):
            print(f"[inspect] wrapped.{attr_name}.action_space:", obj.action_space)
        if hasattr(obj, "_action_space"):
            print(f"[inspect] wrapped.{attr_name}._action_space:", obj._action_space)

simulation_app.close()
