"""
hello_robot.py
--------------
Minimal test — uses the EXACT same setup as train.py but with no TD3.
Just spawns the robot, resets, and steps forever.

Run:
    cd ~/IsaacLab
    ./isaaclab.sh -p /home/lognav/Jardel/path_following_rl_isaac/hello_robot.py
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv

import sys
sys.path.insert(0, "/home/lognav/Jardel/path_following_rl_isaac")
from env_cfg import TurtlebotNavEnvCfg

# Same config as training — just 1 env
env_cfg = TurtlebotNavEnvCfg()
env_cfg.scene.num_envs = 1

env = ManagerBasedRLEnv(cfg=env_cfg)
obs, _ = env.reset()

print("\n[INFO] Environment ready. Robot spawned.")
print(f"[INFO] Obs shape: {obs['policy'].shape}")
print("[INFO] Stepping forever — watch the viewport.\n")

step = 0
while simulation_app.is_running():
    # Send zero action — robot stays still
    action = torch.zeros(1, 2, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)
    step += 1

    if step % 100 == 0:
        robot = env.scene["robot"]
        pos = robot.data.root_pos_w[0]
        print(f"[Step {step:5d}] pos: x={pos[0]:.3f} y={pos[1]:.3f} z={pos[2]:.3f}  reward={reward[0]:.3f}")

    if terminated[0] or truncated[0]:
        obs, _ = env.reset()
        print(f"[Step {step}] Episode ended — resetting.")

env.close()
simulation_app.close()