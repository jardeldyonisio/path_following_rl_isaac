"""
train.py
--------
Entry point to train the navigation policy using TD3 (Twin Delayed DDPG).

Why TD3 over vanilla DDPG:
  1. Twin critics: two Q-networks, always use the minimum estimate.
     This prevents the overestimation that makes vanilla DDPG diverge.
  2. Delayed actor updates: the policy updates every 2 critic steps,
     not every step. Lets the Q-values stabilise before the actor follows.
  3. Target policy smoothing: adds small noise to the target action
     during critic updates. Prevents the actor from exploiting
     narrow Q-value spikes — the same instability your current
     DDPG code works around with DrQv2Noise.

Architecture vs your current code (ddpg.py / models.py):
  - Actor:          same 256-256 MLP + tanh output (deterministic)
  - Critic:         TWO independent 256-256 MLPs instead of one
  - Target nets:    same soft update (tau=0.005)
  - Replay buffer:  same random experience replay
  - Noise:          Gaussian on target actions (internal to TD3)
                    + exploration noise on behaviour actions (like DrQv2Noise)

How to run:
    cd ~/IsaacLab
    ./isaaclab.sh -p /path/to/isaac_nav_env/train.py

    # With options:
    ./isaaclab.sh -p /path/to/isaac_nav_env/train.py --num_envs 16
    ./isaaclab.sh -p /path/to/isaac_nav_env/train.py --num_envs 64 --headless

    # Monitor:
    tensorboard --logdir runs/
"""

import argparse
import torch

# --- Isaac Lab bootstrap (MUST be first, before any other Isaac imports) ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TurtleBot3 navigation with TD3")
parser.add_argument("--num_envs", type=int, default=16,
                    help="Number of parallel environments.")
parser.add_argument("--headless", action="store_true", default=False,
                    help="Run without GUI. Much faster for training.")
parser.add_argument("--max_iterations", type=int, default=1000,
                    help="Number of TD3 update iterations.")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility.")

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Safe to import Isaac/Omniverse modules now ---
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper

import skrl
from skrl.agents.torch.td3 import TD3, TD3_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import skrl.resources.noises.torch as noises

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import TurtlebotNavEnvCfg

# ---------------------------------------------------------------------------
# Reproducibility
# Mirrors your original set_seed() from train.py
# ---------------------------------------------------------------------------
set_seed(args.seed)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
env_cfg = TurtlebotNavEnvCfg()
env_cfg.scene.num_envs = args.num_envs

env = ManagerBasedRLEnv(cfg=env_cfg)
env = SkrlVecEnvWrapper(env)

device = env.device

# ---------------------------------------------------------------------------
# Neural Network Models
# ---------------------------------------------------------------------------
# TD3 uses THREE networks:
#   - Actor            (deterministic policy, like your Actor in models.py)
#   - Critic 1 + 2     (twin Q-networks, both used during training)
# Plus their target copies (handled internally by SKRL).

class Actor(DeterministicMixin, Model):
    """
    Deterministic policy network.
    Identical architecture to your current Actor in models.py:
      Linear(obs, 256) -> ReLU -> Linear(256, 256) -> ReLU -> Linear(256, act) -> Tanh
    Output is in [-1, 1] and scaled by DifferentialDriveAction inside env_cfg.py.
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, act_dim),
            torch.nn.Tanh(),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class Critic(DeterministicMixin, Model):
    """
    Q-value network: estimates Q(state, action).
    Same architecture as your Critic in models.py.
    TD3 instantiates this TWICE (critic_1 and critic_2).

    Concatenates [state, action] as input, exactly like your:
        sa = torch.cat([state, action], 1)
    """
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        self.net = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 1),
        )

    def compute(self, inputs, role):
        # SKRL passes states and actions separately — concatenate here
        sa = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        return self.net(sa), {}


models = {
    "policy":          Actor(env.observation_space, env.action_space, device),
    "target_policy":   Actor(env.observation_space, env.action_space, device),
    "critic_1":        Critic(env.observation_space, env.action_space, device),
    "critic_2":        Critic(env.observation_space, env.action_space, device),
    "target_critic_1": Critic(env.observation_space, env.action_space, device),
    "target_critic_2": Critic(env.observation_space, env.action_space, device),
}

# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------
# Same concept as your ReplayBuffer in utils.py.
# Size 1_000_000 matches your buffer_size=1000000.
memory = RandomMemory(
    memory_size=1_000_000,
    num_envs=env.num_envs,
    device=device,
)

# ---------------------------------------------------------------------------
# TD3 Configuration
# ---------------------------------------------------------------------------
td3_cfg = TD3_DEFAULT_CONFIG.copy()

# --- Core hyperparameters (matched to your ddpg.py where equivalent) ---
td3_cfg["discount_factor"] = 0.99          # gamma — same as your DDPGAgent
td3_cfg["polyak"] = 0.005                  # tau for soft target update — same as yours

td3_cfg["actor_learning_rate"]  = 1e-4     # same as your actor_learning_rate
td3_cfg["critic_learning_rate"] = 1e-3     # same as your critic_learning_rate

td3_cfg["batch_size"] = 256                # same as your batch_size in train.py

# --- TD3-specific parameters (the three fixes over vanilla DDPG) ---

# How many steps to collect before starting updates.
# Mirrors your 'seed_steps = 2000' — fills the replay buffer with
# random actions before the network starts learning.
td3_cfg["random_timesteps"] = 2000

# Exploration noise added to BEHAVIOUR actions during data collection.
# Equivalent to your DrQv2Noise. std=0.1 keeps exploration alive
# without drowning the signal.
td3_cfg["exploration"]["noise"] = noises.GaussianNoise(
    mean=0.0, std=0.1, device=device
)

# Target policy smoothing: noise added to TARGET actions during the
# critic update. This is what vanilla DDPG lacks.
# Prevents the critic from learning to exploit narrow action-value peaks.
td3_cfg["smooth_regularization_noise"] = noises.GaussianNoise(
    mean=0.0, std=0.2, device=device
)
td3_cfg["smooth_regularization_clip"] = 0.5   # clip to [-0.5, 0.5]

# Delayed actor update: update policy every 2 critic steps.
# Lets Q-values stabilise before the actor follows them.
td3_cfg["policy_delay"] = 2

# Gradient steps per environment step
td3_cfg["gradient_steps"] = 1

# --- Logging ---
td3_cfg["experiment"]["directory"]           = "runs/turtlebot_nav"
td3_cfg["experiment"]["experiment_name"]     = "td3_waypoint_nav"
td3_cfg["experiment"]["write_interval"]      = 100    # TensorBoard every 100 steps
td3_cfg["experiment"]["checkpoint_interval"] = 5000   # save model every 5000 steps

agent = TD3(
    models=models,
    memory=memory,
    cfg=td3_cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
# Total timesteps = iterations * num_envs.
# With 16 envs and 1000 iterations = 16,000 env steps.
# For real training increase max_iterations to 100_000+.
total_timesteps = args.max_iterations * args.num_envs

trainer_cfg = {
    "timesteps": total_timesteps,
    "headless":  args.headless,
}

trainer = SequentialTrainer(cfg=trainer_cfg, env=env, agents=agent)

print(f"\n{'='*60}")
print(f"  TurtleBot3 Navigation — TD3 Training")
print(f"  Algorithm     : TD3 (Twin Delayed DDPG)")
print(f"  Parallel envs : {args.num_envs}")
print(f"  Obs dim       : {env.observation_space.shape[0]}")
print(f"  Action dim    : {env.action_space.shape[0]}")
print(f"  Total steps   : {total_timesteps:,}")
print(f"  Seed steps    : {td3_cfg['random_timesteps']:,}  (random actions, like seed_steps)")
print(f"  Batch size    : {td3_cfg['batch_size']}")
print(f"  Policy delay  : every {td3_cfg['policy_delay']} critic updates")
print(f"  Device        : {device}")
print(f"{'='*60}\n")

trainer.train()

# ---------------------------------------------------------------------------
# Save final model
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
agent.save("models/turtlebot_nav_td3_final.pt")
print("Training complete. Model saved to models/turtlebot_nav_td3_final.pt")

simulation_app.close()
