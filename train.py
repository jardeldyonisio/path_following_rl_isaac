import argparse
import torch
import os
import sys
import numpy as np

from datetime import datetime
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train TurtleBot3 navigation with DDPG")
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
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(__file__))
from env_cfg import TurtlebotNavEnvCfg
from noise import DrQv2Noise
from mdp import reset_path_state, reset_robot_pose, reset_obstacles
from agent.ddpg import DDPGAgent
from agent.replay_buffer import ReplayBuffer

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)

# --- Environment ---
env_cfg = TurtlebotNavEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env_cfg.seed = args.seed
if args.num_envs != 1:
    raise ValueError("Current training loop reset/termination logic supports only --num_envs=1")
raw_env = ManagerBasedRLEnv(cfg=env_cfg)
env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
device = env.device

# Timestamped directories to keep each run separate (matches simple env pattern)
project_dir     = os.path.dirname(os.path.abspath(__file__))
timestamp       = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

runs_dir        = os.path.join(project_dir, "runs",   "ddpg_path_following", timestamp)
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

total_timesteps = max(1, int(args.max_episodes * args.max_steps))
batch_size = 256
action_dim = int(env.action_space.shape[0])

noise = DrQv2Noise(
    action_dim=env.action_space.shape[0],
    device=device,
    initial_std=1.0,
    final_std=0.1,
    decay_steps=max(1, int(total_timesteps * 0.8)),
)

# TensorBoard event files go into runs_dir only
tb_writer = SummaryWriter(log_dir=runs_dir)
print(f"TensorBoard logs will be written to: {runs_dir}")

agent = DDPGAgent(
    observation_dim=int(env.observation_space.shape[0]),
    action_dim=action_dim,
    max_action=1.0,
    gamma=0.99,
    tau=0.005,
    buffer_size=1000000,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-4,
    device=device,
    seed=args.seed,
)
replay_buffer = ReplayBuffer(capacity=1000000)

last_best_avg_reward = -float("inf")
episode_rewards      = []
step                 = 0

def _obs_to_numpy(obs):
    if isinstance(obs, dict):
        if "policy" in obs:
            obs = obs["policy"]
        else:
            obs = next(iter(obs.values()))
    if torch.is_tensor(obs):
        arr = obs.detach().cpu().numpy()
    else:
        arr = np.asarray(obs)
    if arr.ndim >= 2 and arr.shape[0] == 1:
        return arr[0].astype(np.float32)
    return arr.astype(np.float32)

for episode in range(args.max_episodes):
    
    # CRITICAL: Manually trigger reset event functions since external env.reset()
    # does NOT automatically invoke the event manager's reset events.
    # We must explicitly call these to regenerate paths and reposition the robot.
    all_env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    reset_path_state(raw_env, all_env_ids)
    reset_obstacles(raw_env, all_env_ids)
    reset_robot_pose(raw_env, all_env_ids)
    
    # After manual resets, get fresh observation
    obs, _ = env.reset()
    obs_np = _obs_to_numpy(obs)
    
    # # DEBUG: Verify reset worked
    # if episode > 0:
    #     robot_pos = raw_env.scene["robot"].data.root_pos_w[0, :2]
    #     wp0 = raw_env.waypoints[0, 0, :2] if hasattr(raw_env, 'waypoints') else torch.zeros(2)
    #     dist = float(torch.norm(robot_pos - wp0).item())
    #     wp_idx = int(raw_env.waypoint_idx[0].item()) if hasattr(raw_env, 'waypoint_idx') else -1
    #     print(f"[EP {episode} AFTER RESET] pos=({robot_pos[0]:.3f},{robot_pos[1]:.3f}), "
    #           f"wp0=({wp0[0]:.3f},{wp0[1]:.3f}), dist={dist:.4f}, wp_idx={wp_idx}")
    
    episode_reward = 0.0

    for _ in range(args.max_steps):

        # Warm-up: random actions before seed_steps
        if step < args.seed_steps:
            action_np = np.random.uniform(-1.0, 1.0, size=(action_dim,)).astype(np.float32)
        else:
            action_np = agent.get_action(obs_np)
            noise_np = noise.sample((1, action_dim)).detach().cpu().numpy()[0]
            action_np = np.clip(action_np + noise_np, -1.0, 1.0).astype(np.float32)

        action = torch.tensor(action_np, device=device, dtype=torch.float32).unsqueeze(0)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs_np = _obs_to_numpy(next_obs)

        done = bool(torch.as_tensor(terminated).any().item() or torch.as_tensor(truncated).any().item())
        reward_value = float(torch.as_tensor(reward).reshape(-1)[0].item())

        # DEBUG: Print when termination/truncation fires
        if step % 500 == 0 or done:
            print(f"[STEP {step}] term={terminated}, trunc={truncated}, done={done}")

        # Store transition in replay buffer
        replay_buffer.add(obs_np, action_np, reward_value, next_obs_np, done)

        # Learning step — log losses per step (mirrors simple env pattern)
        if step >= args.seed_steps and len(replay_buffer) >= batch_size:
            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = replay_buffer.sample(batch_size)
            critic_loss, actor_loss = agent.update(
                batch_obs,
                batch_actions,
                batch_rewards,
                batch_next_obs,
                batch_dones,
            )
            tb_writer.add_scalar("Loss/Critic", critic_loss, step)
            tb_writer.add_scalar("Loss/Actor", actor_loss, step)
            tb_writer.add_scalar("Exploration/Noise_mean", float(noise_np.mean()) if step >= args.seed_steps else 0.0, step)

        episode_reward += reward_value
        obs = next_obs
        obs_np = next_obs_np
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
        agent.save_model(ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")

    # ---- Best model (mirrors simple env pattern) ----
    if avg_reward_last_10 >= last_best_avg_reward:
        last_best_avg_reward = avg_reward_last_10
        agent.save_model(os.path.join(best_dir, "best_agent.pt"))
        print(f"New best avg reward {avg_reward_last_10:.2f} — model saved to: {best_dir}")

# Save final model
agent.save_model(os.path.join(final_dir, "final_agent.pt"))
print(f"Final model saved to: {final_dir}")

tb_writer.close()
simulation_app.close()