import argparse
import copy
import torch
import os
import sys
import shutil

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TurtleBot3 navigation with TD3")
parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
parser.add_argument("--max_iterations", type=int, default=5000, help="Number of iterations.")
parser.add_argument("--seed", type=int, default=42, help="Seed.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import TurtlebotNavEnvCfg

set_seed(args.seed)

# --- Environment ---
env_cfg = TurtlebotNavEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env = ManagerBasedRLEnv(cfg=env_cfg)
env = SkrlVecEnvWrapper(env, ml_framework="torch")
device = env.device

# ---------------------------------------------------------------------------
# STABLE MODELS (With LayerNorm and Prints)
# ---------------------------------------------------------------------------

class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 256),
            torch.nn.LayerNorm(256), # THE ARMOR
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, action_space.shape[0]),
            torch.nn.Tanh(),
        )

    def compute(self, inputs, role):
        states = torch.nan_to_num(inputs["states"], nan=0.0, posinf=10.0, neginf=-10.0)
        output = self.net(states)
        if torch.isnan(output).any():
            print(f"🚨 [ACTOR NaN] States: {states[0].detach().cpu().numpy()}")
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        return output, {}

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0] + action_space.shape[0], 256),
            torch.nn.LayerNorm(256), # THE ARMOR
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        states = torch.nan_to_num(inputs["states"], nan=0.0, posinf=10.0, neginf=-10.0)
        actions = torch.nan_to_num(inputs["taken_actions"], nan=0.0, posinf=1.0, neginf=-1.0)
        sa = torch.cat([states, actions], dim=1)
        q_value = self.net(sa)
        if torch.isnan(q_value).any():
            print(f"🚨 [CRITIC NaN] Q-value exploded!")
        q_value = torch.nan_to_num(q_value, nan=0.0, posinf=1e3, neginf=-1e3)
        return q_value, {}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

if os.path.exists("runs/turtlebot_nav"):
    shutil.rmtree("runs/turtlebot_nav")

os.makedirs("runs/turtlebot_nav/td3_final/checkpoints", exist_ok=True)

models = {
    "policy": Actor(env.observation_space, env.action_space, device),
    "target_policy": Actor(env.observation_space, env.action_space, device),
    "critic_1": Critic(env.observation_space, env.action_space, device),
    "critic_2": Critic(env.observation_space, env.action_space, device),
    "target_critic_1": Critic(env.observation_space, env.action_space, device),
    "target_critic_2": Critic(env.observation_space, env.action_space, device),
}

td3_cfg = copy.deepcopy(TD3_DEFAULT_CONFIG)
# IsaacLab manager env exposes Box(-inf, inf) action space by default.
# skrl random warmup samples from env.action_space and can yield NaNs.
# Start directly from policy actions (already bounded by tanh) instead.
td3_cfg["random_timesteps"] = 0
td3_cfg["actor_learning_rate"] = 1e-4
td3_cfg["critic_learning_rate"] = 1e-4
td3_cfg["grad_norm_clip"] = 0.5
td3_cfg["exploration"]["noise"] = GaussianNoise(mean=0.0, std=0.1, device=device)
td3_cfg["smooth_regularization_noise"] = GaussianNoise(mean=0.0, std=0.2, device=device)
td3_cfg["smooth_regularization_clip"] = 0.5

td3_cfg["experiment"]["directory"] = "runs/turtlebot_nav"
td3_cfg["experiment"]["experiment_name"] = "td3_final"

agent = TD3(models=models, memory=RandomMemory(1000000, env.num_envs, device), 
            cfg=td3_cfg, observation_space=env.observation_space, 
            action_space=env.action_space, device=device)

trainer = SequentialTrainer(cfg={"timesteps": args.max_iterations * args.num_envs}, env=env, agents=agent)

trainer.train()
simulation_app.close()