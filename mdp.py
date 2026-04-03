"""
mdp.py
------
Contains the logic functions for the Managers.
Upgraded to protect the Neural Network from Infinite (inf) physics shocks!
"""

from __future__ import annotations
import math
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import ActionTermCfg, ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ===========================================================================
# 1. ACTION TERM - Differential Drive
# ===========================================================================
class DifferentialDriveAction(ActionTerm):
    cfg: DifferentialDriveActionCfg

    def __init__(self, cfg: DifferentialDriveActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._asset = env.scene[cfg.asset_name]
        self._left_idx = self._asset.find_joints(cfg.left_joint_name)[0][0]
        self._right_idx = self._asset.find_joints(cfg.right_joint_name)[0][0]
        
        self._joint_ids = [self._left_idx, self._right_idx]
        self._r = cfg.wheel_radius
        self._L = cfg.wheel_base
        self._lin_scale = cfg.linear_vel_scale
        self._ang_scale = cfg.angular_vel_scale

    @property
    def action_dim(self) -> int: return 2
    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # Neural Network Safety Net
        if not torch.isfinite(actions).all():
            actions = torch.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
            
        self._raw_actions = actions.clone()
        v = actions[:, 0] * self._lin_scale
        w = actions[:, 1] * self._ang_scale
        v_left  = (v - w * self._L / 2.0) / self._r
        v_right = (v + w * self._L / 2.0) / self._r
        self._processed_actions = torch.stack([v_left, v_right], dim=1)

    def apply_actions(self):
        self._asset.set_joint_velocity_target(self._processed_actions, joint_ids=self._joint_ids)

@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    class_type: type = DifferentialDriveAction
    asset_name: str = "robot"
    left_joint_name: str = "wheel_left_joint"
    right_joint_name: str = "wheel_right_joint"
    wheel_radius: float = 0.033
    wheel_base: float = 0.160
    linear_vel_scale: float = 0.22  
    angular_vel_scale: float = 2.84 

# ===========================================================================
# 2. OBSERVATIONS AND REWARDS
# ===========================================================================
def goal_observation(env: ManagerBasedRLEnv, lookahead: int = 1) -> torch.Tensor:
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    yaw = _quat_to_yaw(robot.data.root_quat_w)
    goals = _get_current_waypoint(env)
    
    delta = goals - robot_pos
    dist = torch.norm(delta, dim=1, keepdim=True)
    goal_angle_world = torch.atan2(delta[:, 1], delta[:, 0])
    rel_angle = _wrap_angle(goal_angle_world - yaw)
    
    obs = torch.cat([dist, rel_angle.unsqueeze(1)], dim=1)
    return torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0) # Safety clamp

def velocity_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_b[:, 0]
    ang_vel = robot.data.root_ang_vel_b[:, 2]
    
    lin_norm = torch.clamp(lin_vel / 0.22, -1.0, 1.0)
    ang_norm = torch.clamp(ang_vel / 2.84, -1.0, 1.0)
    
    obs = torch.stack([lin_norm, ang_norm], dim=1)
    return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0) # Safety clamp

def progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)
    current_dist = torch.norm(goals - robot_pos, dim=1)
    prev_dist = env.extras.get("prev_dist_to_goal", current_dist.clone())
    env.extras["prev_dist_to_goal"] = current_dist.clone()
    
    reward = prev_dist - current_dist
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)

def waypoint_reached_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)
    reached = torch.norm(goals - robot_pos, dim=1) < 0.2
    _advance_waypoint(env, reached)
    return reached.float()

def out_of_bounds_penalty(env: ManagerBasedRLEnv, max_dist: float = 2.0) -> torch.Tensor:
    return -(torch.norm(_get_current_waypoint(env) - env.scene["robot"].data.root_pos_w[:, :2], dim=1) > max_dist).float()

def alive_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return -torch.ones(env.num_envs, device=env.device)

def out_of_bounds_termination(env: ManagerBasedRLEnv, max_dist: float = 2.0) -> torch.Tensor:
    return torch.norm(_get_current_waypoint(env) - env.scene["robot"].data.root_pos_w[:, :2], dim=1) > max_dist

def all_waypoints_reached_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, 'waypoints'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.waypoint_idx >= env.waypoints.shape[1]

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.episode_length_buf >= env.max_episode_length

# ===========================================================================
# 5. EVENTS (RESET)
# ===========================================================================
def reset_robot_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    robot = env.scene["robot"]
    if not hasattr(env, 'waypoints'):
        return
    
    start_pos = env.waypoints[env_ids, 0, :]
    root_state = robot.data.default_root_state[env_ids].clone()
    
    root_state[:, 0:2] = start_pos
    # --- Soft Landing: Spawn just 2cm above ground ---
    root_state[:, 2] = env.scene.env_origins[env_ids, 2] + 0.02 
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:13] = 0.0 
    
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    
    joint_pos = torch.zeros_like(robot.data.default_joint_pos[env_ids])
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def reset_waypoints(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    num_reset, num_pts, device = len(env_ids), 15, env.device
    waypoints = torch.zeros(num_reset, num_pts, 2, device=device)
    
    env_origins = env.scene.env_origins[env_ids]
    curr_x = env_origins[:, 0].clone()
    curr_y = env_origins[:, 1].clone()
    heading = torch.rand(num_reset, device=device) * 2 * math.pi
    
    for i in range(num_pts):
        waypoints[:, i, 0] = curr_x
        waypoints[:, i, 1] = curr_y
        heading += (torch.rand(num_reset, device=device) * 0.4 - 0.2)
        curr_x += torch.cos(heading) * 1.0
        curr_y += torch.sin(heading) * 1.0
        
    if not hasattr(env, 'waypoints'):
        env.waypoints = torch.zeros(env.num_envs, num_pts, 2, device=device)
        env.waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    
    env.waypoints[env_ids] = waypoints
    env.waypoint_idx[env_ids] = 1

# ===========================================================================
# HELPERS
# ===========================================================================
def _get_current_waypoint(env: ManagerBasedRLEnv):
    if not hasattr(env, 'waypoint_idx'):
        return torch.zeros(env.num_envs, 2, device=env.device)
    idx = torch.clamp(env.waypoint_idx, 0, env.waypoints.shape[1] - 1)
    return env.waypoints[torch.arange(env.num_envs, device=env.device), idx]

def _advance_waypoint(env: ManagerBasedRLEnv, reached: torch.Tensor):
    if hasattr(env, 'waypoint_idx'):
        env.waypoint_idx[reached] += 1

def _quat_to_yaw(quat: torch.Tensor):
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def _wrap_angle(angle: torch.Tensor):
    return (angle + math.pi) % (2 * math.pi) - math.pi