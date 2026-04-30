import argparse
import copy
import torch
import os
import sys
from collections import deque
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convoy navigation.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--max_episodes", type=int, default=100000, help="Number of training episodes (simple env default).")
parser.add_argument("--max_steps", type=int, default=1000, help="Max steps per episode (simple env default).")
parser.add_argument("--seed_steps", type=int, default=2000, help="Number of warmup (random) steps before learning starts.")
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
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import ConvoyNavigationEnvCgf
from noise import DrQv2Noise, FixedGaussianNoise

set_seed(args.seed)

env_cfg = ConvoyNavigationEnvCgf()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
env = ManagerBasedRLEnv(cfg=env_cfg)
env = SkrlVecEnvWrapper(env, ml_framework="torch")
device = env.device

class EpisodeTrackerWrapper:
    def __init__(self, env):
        self.env = env
        self.episode_rewards = torch.zeros(env.num_envs, device=env.device)
        self.episode_steps = torch.zeros(env.num_envs, device=env.device)
        self.last_10_rewards = deque(maxlen=10)
        self.best_avg_reward = float("-inf")
        self.agent = None
        self.best_dir = None

    def set_agent(self, agent, best_dir):
        self.agent = agent
        self.best_dir = best_dir

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(actions)

        if torch.is_tensor(rewards):
            rewards_flat = rewards.float().view(-1)
            self.episode_rewards += rewards_flat
        else:
            try:
                self.episode_rewards += float(rewards)
            except (TypeError, ValueError):
                pass
        self.episode_steps += 1

        if torch.is_tensor(terminated) or torch.is_tensor(truncated):
            done = (terminated | truncated).view(-1)
            if done.any():
                done_indices = torch.nonzero(done).squeeze(-1)
                for idx in done_indices.tolist():
                    ep_reward = float(self.episode_rewards[idx].item())
                    self.last_10_rewards.append(ep_reward)
                    if self.agent is not None and self.best_dir is not None and len(self.last_10_rewards) > 0:
                        avg_last_10 = sum(self.last_10_rewards) / len(self.last_10_rewards)
                        self.agent.track_data("Reward / Average reward last 10 episodes", avg_last_10)
                        if avg_last_10 >= self.best_avg_reward:
                            self.best_avg_reward = avg_last_10
                            best_path = os.path.join(self.best_dir, "best_avg_model.pt")
                            print("[INFO] New best avg reward. Saving model to", best_path)
                            self.agent.save(best_path)
                self.episode_rewards[done] = 0.0
                self.episode_steps[done] = 0

        return obs, rewards, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)

class Actor(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(observation_space.shape[0], 256),
            torch.nn.LayerNorm(256),
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
            torch.nn.LayerNorm(256),
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

run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
td3_cfg["experiment"]["write_interval"] = 1000
td3_cfg["experiment"]["checkpoint_interval"] = 20000
total_timesteps = max(1, int(args.max_episodes * args.max_steps))
td3_cfg["exploration"]["noise"] = DrQv2Noise(
    action_dim=env.action_space.shape[0],
    device=device,
    initial_std=0.3,
    final_std=0.05,
    decay_steps=max(1, int(total_timesteps * 0.8)),
)
td3_cfg["smooth_regularization_noise"] = FixedGaussianNoise(mean=0.0, std=0.2, device=device)
td3_cfg["smooth_regularization_clip"] = 0.5

base_dir = os.path.dirname(os.path.abspath(__file__))
run_root = os.path.join(base_dir, "runs", "simple_env")
td3_cfg["experiment"]["directory"] = run_root
td3_cfg["experiment"]["experiment_name"] = run_timestamp
run_dir = os.path.join(run_root, run_timestamp)
checkpoints_dir = os.path.join(run_dir, "checkpoints")
best_dir = os.path.join(run_dir, "best")
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(best_dir, exist_ok=True)

agent = TD3(
    models=models,
    memory=RandomMemory(1000000, env.num_envs, device),
    cfg=td3_cfg,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=device,
)

env = EpisodeTrackerWrapper(env)
env.set_agent(agent, best_dir)

trainer = SequentialTrainer(cfg={"timesteps": total_timesteps}, env=env, agents=agent)

trainer.train()
simulation_app.close()