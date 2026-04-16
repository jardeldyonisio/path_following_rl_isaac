import argparse
import copy
import torch
import os
import sys
import numpy as np

from datetime import datetime
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TurtleBot3 navigation with TD3")
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
from skrl.utils import set_seed
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import TurtlebotNavEnvCfg
from noise import DrQv2Noise, FixedGaussianNoise

set_seed(args.seed)

# --- Environment ---
env_cfg = TurtlebotNavEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
device = env.device

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
            print(f"[ACTOR NaN] States: {states[0].detach().cpu().numpy()}")
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
            print(f"[CRITIC NaN] Q-value exploded!")
        q_value = torch.nan_to_num(q_value, nan=0.0, posinf=1e3, neginf=-1e3)
        return q_value, {}

# Timestamped directories to keep each run separate (matches simple env pattern)
project_dir     = os.path.dirname(os.path.abspath(__file__))
timestamp       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

runs_dir        = os.path.join(project_dir, "runs",   "td3_path_following", timestamp)
models_dir      = os.path.join(project_dir, "models", timestamp)
checkpoints_dir = os.path.join(models_dir, "checkpoints")
best_dir        = os.path.join(models_dir, "best")
final_dir       = os.path.join(models_dir, "final")

os.makedirs(runs_dir,        exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)
os.makedirs(best_dir,        exist_ok=True)
os.makedirs(final_dir,       exist_ok=True)

print(f"TensorBoard : {runs_dir}")
print(f"Checkpoints : {checkpoints_dir}")
print(f"Best model  : {best_dir}")
print(f"Final model  : {final_dir}")

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

# Disable skrl's internal checkpoint/tensorboard saving entirely —
# we handle all saves manually to control exact paths and filenames.
td3_cfg["experiment"]["directory"]           = ""
td3_cfg["experiment"]["experiment_name"]     = ""
td3_cfg["experiment"]["write_interval"]      = 0     # disable skrl log writes
td3_cfg["experiment"]["checkpoint_interval"] = 0     # disable skrl auto-checkpoints
td3_cfg["experiment"]["write_tensorboard"]   = False

# TensorBoard event files go into runs_dir only
tb_writer = SummaryWriter(log_dir=runs_dir)
print(f"TensorBoard logs will be written to: {runs_dir}")

agent = TD3(models=models, memory=RandomMemory(1000000, env.num_envs, device), 
            cfg=td3_cfg, observation_space=env.observation_space, 
            action_space=env.action_space, device=device)

# Inject writer into agent experiment tracking
agent.writer = tb_writer

# ---------------------------------------------------------------------------
# Manual training loop (mirrors simple env) — enables best model tracking
# ---------------------------------------------------------------------------

last_best_avg_reward = -float("inf")
episode_rewards      = []
step                 = 0

agent.init()  # initialise skrl internal state before stepping manually

from mdp import reset_path_state, reset_robot_pose, reset_obstacles

for episode in range(args.max_episodes):

    obs, _ = env.reset()
    
    # CRITICAL: Manually trigger reset event functions since external env.reset()
    # does NOT automatically invoke the event manager's reset events.
    # We must explicitly call these to regenerate paths and reposition the robot.
    all_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    reset_path_state(raw_env, all_env_ids)
    reset_obstacles(raw_env, all_env_ids)
    reset_robot_pose(raw_env, all_env_ids)
    
    # After manual resets, get fresh observation
    obs, _ = env.reset()
    
    # DEBUG: Verify reset worked
    if episode > 0:
        robot_pos = raw_env.scene["robot"].data.root_pos_w[0, :2]
        wp0 = raw_env.waypoints[0, 0, :2] if hasattr(raw_env, 'waypoints') else torch.zeros(2)
        dist = float(torch.norm(robot_pos - wp0).item())
        wp_idx = int(raw_env.waypoint_idx[0].item()) if hasattr(raw_env, 'waypoint_idx') else -1
        print(f"[EP {episode} AFTER RESET] pos=({robot_pos[0]:.3f},{robot_pos[1]:.3f}), "
              f"wp0=({wp0[0]:.3f},{wp0[1]:.3f}), dist={dist:.4f}, wp_idx={wp_idx}")
    
    episode_reward = 0.0

    for _ in range(args.max_steps):

        # Warm-up: random actions before seed_steps
        if step < args.seed_steps:
            action = torch.tensor(
                np.array([env.action_space.sample() for _ in range(env.num_envs)]),
                device=device, dtype=torch.float32,
            )
        else:
            with torch.no_grad():
                action = agent.act(obs, timestep=step, timesteps=total_timesteps)[0]

        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated.any() or truncated.any()

        # DEBUG: Print when termination/truncation fires
        if step % 500 == 0 or done:
            print(f"[STEP {step}] term={terminated}, trunc={truncated}, done={done}")

        # Manually trigger resets for truncated (timed-out) environments
        # since time_out=True prevents the event manager from auto-resetting.
        if truncated.any():
            from mdp import reset_path_state, reset_robot_pose, reset_obstacles
            # Handle both scalar and (num_envs,) shaped tensors
            if truncated.dim() == 0:
                truncated_ids = torch.tensor([0], device=device)
            elif truncated.dim() == 1:
                truncated_ids = torch.where(truncated)[0]
            else:
                truncated_ids = torch.where(truncated.any(dim=1))[0]
            
            print(f"[TRUNCATED RESET] truncated_ids={truncated_ids}")
            
            if truncated_ids.numel() > 0:
                # Access the raw ManagerBasedRLEnv via the wrapper
                raw_env = env.env
                print(f"[TRUNCATED RESET] Calling reset functions for envs {truncated_ids.tolist()}...")
                reset_path_state(raw_env, truncated_ids)
                reset_obstacles(raw_env, truncated_ids)
                reset_robot_pose(raw_env, truncated_ids)
                print(f"[TRUNCATED RESET] Reset complete")

        # Store transition in replay buffer
        agent.record_transition(
            states=obs,
            actions=action,
            rewards=reward,
            next_states=next_obs,
            terminated=terminated,
            truncated=truncated,
            infos=info,
            timestep=step,
            timesteps=total_timesteps,
        )

        # Learning step — log losses per step (mirrors simple env pattern)
        if step >= args.seed_steps:
            agent.post_interaction(timestep=step, timesteps=total_timesteps)

            tracking = getattr(agent, "tracking_data", {})
            if "Loss / Critic loss" in tracking and tracking["Loss / Critic loss"]:
                tb_writer.add_scalar("Loss/Critic", float(np.mean(tracking["Loss / Critic loss"])), step)
            if "Exploration / Exploration noise (mean)" in tracking and tracking["Exploration / Exploration noise (mean)"]:
                tb_writer.add_scalar("Exploration/Noise_mean", float(np.mean(tracking["Exploration / Exploration noise (mean)"])), step)
            if "Q-network / Q1 (mean)" in tracking and tracking["Q-network / Q1 (mean)"]:
                tb_writer.add_scalar("Q-network/Q1_mean", float(np.mean(tracking["Q-network / Q1 (mean)"])), step)
            if "Target / Target (mean)" in tracking and tracking["Target / Target (mean)"]:
                tb_writer.add_scalar("Target/Target_mean", float(np.mean(tracking["Target / Target (mean)"])), step)

        episode_reward += float(reward.mean().item())
        obs = next_obs
        step += 1

        if done:
            # IsaacLab auto-resets terminated envs inside env.step() and
            # returns the fresh first observation in next_obs — use it directly.
            obs = next_obs
            break

    # ---- Per-episode bookkeeping ----
    episode_rewards.append(episode_reward)
    avg_reward_last_10 = float(np.mean(episode_rewards[-10:]))

    tb_writer.add_scalar("Reward/Episode",              episode_reward,     episode)
    tb_writer.add_scalar("Reward/Avg_last_10_episodes", avg_reward_last_10, episode)
    tb_writer.add_scalar("Train/Total_steps",           step,               episode)

    sys.stdout.write(
        f"episode: {episode:6d} | reward: {episode_reward:8.2f} | "
        f"avg(10): {avg_reward_last_10:8.2f} | steps: {step}\n"
    )
    sys.stdout.flush()

    # ---- Periodic checkpoint ----
    if episode > 0 and episode % 1000 == 0:
        ckpt_path = os.path.join(checkpoints_dir, f"agent_{episode}.pt")
        agent.save(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    # ---- Best model (mirrors simple env pattern) ----
    if avg_reward_last_10 >= last_best_avg_reward:
        last_best_avg_reward = avg_reward_last_10
        agent.save(os.path.join(best_dir, "best_agent.pt"))
        print(f"New best avg reward {avg_reward_last_10:.2f} — model saved to: {best_dir}")

# Save final model
agent.save(os.path.join(final_dir, "final_agent.pt"))
print(f"Final model saved to: {final_dir}")

tb_writer.close()
simulation_app.close()