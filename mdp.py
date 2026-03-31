"""
mdp.py
------
Contains ALL the functions that the managers in env_cfg.py call.
Each function receives (env, ...) as arguments — Isaac Lab injects
the environment automatically.

This is equivalent to the logic spread across _get_obs(), _rewards(),
_is_terminated(), etc. in your original PathMultiObstaclesLidarEnv.

Structure:
  1. Custom Action Term (DifferentialDriveAction)
  2. Observation functions
  3. Reward functions
  4. Termination functions
  5. Event (reset) functions
"""

from __future__ import annotations

import math
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab.managers import ActionTermCfg, ActionTerm, SceneEntityCfg
from isaaclab.utils import configclass
from dataclasses import MISSING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ===========================================================================
# 1. CUSTOM ACTION TERM — Differential Drive
# ===========================================================================
# Isaac Lab's action system works in two parts:
#   a) ActionTermCfg  — the config dataclass (goes in env_cfg.py)
#   b) ActionTerm     — the actual logic class

class DifferentialDriveAction(ActionTerm):
    """
    Converts [linear_vel, angular_vel] network output into
    individual left/right wheel velocity commands.

    Kinematics:
        v_left  = (v - w * L/2) / r
        v_right = (v + w * L/2) / r
    where L = wheel_base, r = wheel_radius
    """

    cfg: DifferentialDriveActionCfg

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        # Resolve which asset this action applies to
        self._asset = env.scene[cfg.asset_name]

        # Find joint indices by name inside the articulation
        self._left_idx = self._asset.find_joints(cfg.left_joint_name)[0][0]
        self._right_idx = self._asset.find_joints(cfg.right_joint_name)[0][0]

        # Store kinematics params
        self._r = cfg.wheel_radius
        self._L = cfg.wheel_base
        self._lin_scale = cfg.linear_vel_scale
        self._ang_scale = cfg.angular_vel_scale

    @property
    def action_dim(self) -> int:
        return 2  # [v, w]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Called once per policy step. actions shape: (num_envs, 2)."""
        self._raw_actions = actions.clone()

        # Scale from [-1, 1] to actual velocity ranges
        v = actions[:, 0] * self._lin_scale   # linear  vel  [m/s]
        w = actions[:, 1] * self._ang_scale   # angular vel  [rad/s]

        # Differential drive inverse kinematics
        v_left  = (v - w * self._L / 2.0) / self._r   # [rad/s]
        v_right = (v + w * self._L / 2.0) / self._r   # [rad/s]

        self._processed_actions = torch.stack([v_left, v_right], dim=1)

    def apply_actions(self):
        """Called every physics step. Applies wheel velocities to the sim."""
        # Build a full joint velocity tensor (all joints, only wheels get set)
        vel_targets = torch.zeros(
            self._env.num_envs,
            self._asset.num_joints,
            device=self._env.device,
        )
        vel_targets[:, self._left_idx]  = self._processed_actions[:, 0]
        vel_targets[:, self._right_idx] = self._processed_actions[:, 1]

        self._asset.set_joint_velocity_target(vel_targets)


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """Config for the differential drive action term."""

    class_type: type = DifferentialDriveAction

    asset_name: str = MISSING
    left_joint_name: str = MISSING
    right_joint_name: str = MISSING
    wheel_radius: float = 0.033
    wheel_base: float = 0.160
    linear_vel_scale: float = 0.25
    angular_vel_scale: float = 0.50


# ===========================================================================
# 2. OBSERVATION FUNCTIONS
# ===========================================================================

def goal_observation(env: ManagerBasedRLEnv, lookahead: int = 1) -> torch.Tensor:
    """
    Returns [distance_to_next_waypoint, angle_to_next_waypoint].
    Shape: (num_envs, 2)

    The angle is relative to the robot's current heading (yaw), so
    the network learns heading-agnostic navigation.
    """
    robot = env.scene["robot"]
    num_envs = env.num_envs
    device = env.device

    # Robot world position (x, y)
    robot_pos = robot.data.root_pos_w[:, :2]  # (num_envs, 2)

    # Robot yaw angle (rotation around Z)
    # root_quat_w is [w, x, y, z]
    quat = robot.data.root_quat_w  # (num_envs, 4)
    yaw = _quat_to_yaw(quat)       # (num_envs,)

    # Current goal for each environment
    # env.waypoints: (num_envs, num_waypoints, 2) — set during reset
    # env.waypoint_idx: (num_envs,) — index of the active waypoint
    goals = _get_current_waypoint(env)  # (num_envs, 2)

    # Vector from robot to goal in world frame
    delta = goals - robot_pos  # (num_envs, 2)

    # Distance
    dist = torch.norm(delta, dim=1, keepdim=True)  # (num_envs, 1)

    # Angle to goal in world frame
    goal_angle_world = torch.atan2(delta[:, 1], delta[:, 0])  # (num_envs,)

    # Angle RELATIVE to robot heading (what the robot "sees")
    rel_angle = _wrap_angle(goal_angle_world - yaw)  # (num_envs,)

    return torch.cat([dist, rel_angle.unsqueeze(1)], dim=1)  # (num_envs, 2)


def velocity_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns [linear_velocity, angular_velocity] of the robot base.
    Shape: (num_envs, 2)

    Velocities are normalized to [-1, 1] using the action scales defined
    in the config so the network always receives similar magnitude inputs.
    """
    robot = env.scene["robot"]

    # root_lin_vel_b: linear velocity in the robot's body frame (x=forward)
    lin_vel = robot.data.root_lin_vel_b[:, 0]  # (num_envs,) — forward component

    # root_ang_vel_b: angular velocity in body frame (z=yaw rate)
    ang_vel = robot.data.root_ang_vel_b[:, 2]  # (num_envs,) — yaw rate

    # Normalize using the scales from the action config
    lin_scale = env.cfg.actions.robot_vel.linear_vel_scale
    ang_scale = env.cfg.actions.robot_vel.angular_vel_scale

    lin_norm = torch.clamp(lin_vel / lin_scale, -1.0, 1.0)
    ang_norm = torch.clamp(ang_vel / ang_scale, -1.0, 1.0)

    return torch.stack([lin_norm, ang_norm], dim=1)  # (num_envs, 2)


# ===========================================================================
# 3. REWARD FUNCTIONS
# ===========================================================================

def progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Rewards the agent for getting closer to the current waypoint.
    This is a dense reward — the agent gets feedback every step.

    reward = previous_distance - current_distance
    (positive when moving closer, negative when moving away)
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)

    current_dist = torch.norm(goals - robot_pos, dim=1)

    # prev_dist is stored from the last step in env.extras
    prev_dist = env.extras.get("prev_dist_to_goal", current_dist.clone())
    env.extras["prev_dist_to_goal"] = current_dist.clone()

    return prev_dist - current_dist  # (num_envs,)


def waypoint_reached_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Sparse reward: +1 each time a waypoint is cleared.
    Also advances the waypoint index for that environment.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)

    dist = torch.norm(goals - robot_pos, dim=1)  # (num_envs,)
    reached = dist < 0.2  # 0.2m threshold — matches your goal_threshold

    # Advance waypoint index for environments that reached the goal
    _advance_waypoint(env, reached)

    return reached.float()  # (num_envs,)


def out_of_bounds_penalty(env: ManagerBasedRLEnv, max_dist: float = 2.0) -> torch.Tensor:
    """
    Penalty when the robot strays too far from the current waypoint.
    Returns -1.0 for out-of-bounds environments, 0.0 otherwise.
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)

    dist = torch.norm(goals - robot_pos, dim=1)
    out = dist > max_dist

    return -out.float()


def alive_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Small constant penalty per step to encourage efficiency."""
    return -torch.ones(env.num_envs, device=env.device)


# ===========================================================================
# 4. TERMINATION FUNCTIONS
# ===========================================================================

def out_of_bounds_termination(env: ManagerBasedRLEnv, max_dist: float = 2.0) -> torch.Tensor:
    """Terminate if robot is too far from the current waypoint."""
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)

    dist = torch.norm(goals - robot_pos, dim=1)
    return dist > max_dist  # (num_envs,) bool tensor


def all_waypoints_reached_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate (with success) when all waypoints have been visited."""
    num_waypoints = env.waypoints.shape[1]
    # waypoint_idx >= num_waypoints means all were cleared
    return env.waypoint_idx >= num_waypoints  # (num_envs,) bool tensor


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Built-in Isaac Lab termination.
    Returns True when episode_length_buf >= max_episode_length.
    Isaac Lab handles this automatically when time_out=True in the config.
    """
    return env.episode_length_buf >= env.max_episode_length


# ===========================================================================
# 5. EVENT (RESET) FUNCTIONS
# ===========================================================================

def reset_robot_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    Called on reset for the specified environment IDs.
    Places the robot at the START of the new waypoint path.
    env_ids: 1D tensor of environment indices being reset.
    """
    robot = env.scene["robot"]

    # Safety initialiser: if waypoints don't exist yet (first ever reset),
    # create zero-filled tensors so the rest of this function doesn't crash.
    # reset_waypoints will overwrite them with real values immediately after.
    if not hasattr(env, 'waypoints'):
        num_waypoints = 15
        env.waypoints = torch.zeros(env.num_envs, num_waypoints, 2, device=env.device)
        env.waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    start_pos = env.waypoints[env_ids, 0, :]  # (len(env_ids), 2)

    # Build full pose tensors. Isaac Lab expects:
    #   root_state: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z, vel_x...vel_z, ang_x...ang_z]
    num_reset = len(env_ids)
    root_state = torch.zeros(num_reset, 13, device=env.device)

    # Position: start of path, z slightly above ground
    root_state[:, 0] = start_pos[:, 0]
    root_state[:, 1] = start_pos[:, 1]
    root_state[:, 2] = 0.01

    # Orientation: identity quaternion [w=1, x=0, y=0, z=0]
    # Point robot toward the first waypoint for a clean start
    if env.waypoints.shape[1] > 1:
        direction = env.waypoints[env_ids, 1, :] - start_pos
        yaw = torch.atan2(direction[:, 1], direction[:, 0])
        # Convert yaw to quaternion
        root_state[:, 3] = torch.cos(yaw / 2)  # w
        root_state[:, 6] = torch.sin(yaw / 2)  # z
    else:
        root_state[:, 3] = 1.0  # identity

    # Zero all velocities at reset
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    # Reset joint states (wheels)
    joint_pos = robot.data.default_joint_pos[env_ids]
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_waypoints(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    """
    Generates a new random waypoint path for each resetting environment.

    Path generation logic (mirrors your _create_path / _switch_path):
      - Start near origin with small random offset
      - Each subsequent waypoint is placed 1m ahead with random lateral offset
      - 15 waypoints total (matches your num_goals_window)
    """
    num_reset = len(env_ids)
    num_waypoints = 15       # Matches your num_goals_window
    step_size = 1.0          # meters between waypoints (your goal_step)
    lateral_noise = 0.3      # max lateral deviation per step

    device = env.device

    waypoints = torch.zeros(num_reset, num_waypoints, 2, device=device)

    # Random starting position near origin
    start_x = torch.zeros(num_reset, device=device)
    start_y = torch.zeros(num_reset, device=device)

    # Build path waypoint by waypoint
    current_x = start_x.clone()
    current_y = start_y.clone()
    # Random heading for this episode
    heading = torch.rand(num_reset, device=device) * 2 * math.pi

    for i in range(num_waypoints):
        waypoints[:, i, 0] = current_x
        waypoints[:, i, 1] = current_y

        # Advance forward + add lateral noise
        dx = torch.cos(heading) * step_size
        dy = torch.sin(heading) * step_size
        lateral = (torch.rand(num_reset, device=device) * 2 - 1) * lateral_noise

        # Perpendicular direction
        perp_x = -torch.sin(heading)
        perp_y =  torch.cos(heading)

        current_x = current_x + dx + perp_x * lateral
        current_y = current_y + dy + perp_y * lateral

        # Slightly update heading to make curved paths
        heading = heading + (torch.rand(num_reset, device=device) * 0.4 - 0.2)

    # Store on the environment object so other functions can access it
    if not hasattr(env, 'waypoints'):
        # First reset: allocate full tensors for ALL environments
        env.waypoints = torch.zeros(env.num_envs, num_waypoints, 2, device=device)
        env.waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    env.waypoints[env_ids] = waypoints
    env.waypoint_idx[env_ids] = 1  # Index 0 is the start (robot spawns there), so next goal is index 1

    # Reset the distance tracker used by progress_reward
    if "prev_dist_to_goal" in env.extras:
        goals = waypoints[:, 1, :]
        starts = waypoints[:, 0, :]
        env.extras["prev_dist_to_goal"][env_ids] = torch.norm(goals - starts, dim=1)


# ===========================================================================
# HELPERS (private, not called by managers directly)
# ===========================================================================

def _get_current_waypoint(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Returns the current active waypoint for each environment.
    Clamps index to avoid out-of-bounds once all waypoints are reached.
    Shape: (num_envs, 2)
    """
    if not hasattr(env, 'waypoints'):
        # Fallback: return origin if waypoints not yet initialized
        return torch.zeros(env.num_envs, 2, device=env.device)

    num_waypoints = env.waypoints.shape[1]
    idx = torch.clamp(env.waypoint_idx, 0, num_waypoints - 1)  # (num_envs,)

    # Fancy indexing: for each env, pick waypoints[env_i, idx[env_i], :]
    return env.waypoints[torch.arange(env.num_envs, device=env.device), idx]


def _advance_waypoint(env: ManagerBasedRLEnv, reached: torch.Tensor):
    """
    Increments waypoint_idx for environments where 'reached' is True.
    reached: (num_envs,) bool tensor
    """
    if hasattr(env, 'waypoint_idx'):
        env.waypoint_idx[reached] += 1


def _quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    """
    Extracts yaw (rotation around Z) from a [w, x, y, z] quaternion.
    Shape: (N, 4) → (N,)
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    # Standard formula for yaw from quaternion
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return yaw


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wraps angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi