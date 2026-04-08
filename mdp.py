'''
@brief Defines the action processing, observation computation, reward calculation, termination conditions, and reset events 
for a differential drive robot navigation task in IsaacLab. The robot is expected to follow a series of waypoints, and the 
code includes utilities for handling waypoints, computing relative angles, and visualizing the waypoints in the simulation 
for debugging purposes.
'''

from __future__ import annotations

import math
import torch
import numpy as np
import gymnasium as gym

from typing import TYPE_CHECKING
from isaaclab.utils import configclass
from isaaclab.managers import ActionTermCfg, ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def _get_current_waypoint(env: ManagerBasedRLEnv):
    '''
    @brief Get the current target waypoint for each environment instance.
    '''
    if not hasattr(env, 'waypoint_idx'):
        return torch.zeros(env.num_envs, 2, device=env.device)
    idx = torch.clamp(env.waypoint_idx, 0, env.waypoints.shape[1] - 1)
    return env.waypoints[torch.arange(env.num_envs, device=env.device), idx]

def _goal_distance_and_yaw_error(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    '''
    @brief Compute distance and relative yaw error from robot to current waypoint.

    @return: (distance, yaw_error), both shaped (num_envs,).
    '''
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    yaw = _quat_to_yaw(robot.data.root_quat_w)
    goals = _get_current_waypoint(env)

    delta = goals - robot_pos
    dist = torch.norm(delta, dim=1)
    goal_angle_world = torch.atan2(delta[:, 1], delta[:, 0])
    rel_angle = _wrap_angle(goal_angle_world - yaw)
    return dist, rel_angle

def _get_current_raw_actions(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Try to recover current normalized raw actions in [-1, 1] from action manager.

    Returns zeros if action access is unavailable.
    '''
    zeros = torch.zeros((env.num_envs, 2), device=env.device)

    if not hasattr(env, "action_manager"):
        return zeros

    action_manager = env.action_manager

    # Common manager-level fields across IsaacLab versions.
    for name in ("action", "actions", "_action", "_actions"):
        val = getattr(action_manager, name, None)
        if torch.is_tensor(val) and val.ndim == 2 and val.shape[0] == env.num_envs and val.shape[1] >= 2:
            return torch.clamp(val[:, :2], -1.0, 1.0)

    # Term-level fallback: search for the differential-drive term that exposes raw_actions.
    for terms_attr in ("terms", "_terms"):
        terms = getattr(action_manager, terms_attr, None)
        if isinstance(terms, dict):
            for term in terms.values():
                raw = getattr(term, "raw_actions", None)
                if torch.is_tensor(raw) and raw.ndim == 2 and raw.shape[0] == env.num_envs and raw.shape[1] >= 2:
                    return torch.clamp(raw[:, :2], -1.0, 1.0)

    return zeros

def _advance_waypoint(env: ManagerBasedRLEnv, reached: torch.Tensor):
    '''
    @brief Advance the waypoint index for each environment instance.

    @param reached: A boolean tensor indicating which environments have reached their current waypoint.
    '''
    if hasattr(env, 'waypoint_idx'):
        env.waypoint_idx[reached] += 1

def _advance_to_secondary_if_closer(
    env: ManagerBasedRLEnv,
    robot_pos: torch.Tensor,
    main_dist: torch.Tensor,
    reached: torch.Tensor,
):
    '''
    @brief Advance current goal index to a secondary goal when it is closer than the main goal.

    This mirrors the classic env behavior where passing/being closer to a secondary goal should
    update the current main goal target.
    '''
    if not hasattr(env, "waypoint_idx") or not hasattr(env, "waypoints"):
        return

    goal_step = int(getattr(env, "goal_step", 1))
    if goal_step <= 0:
        goal_step = 1
    window_size = int(getattr(env, "num_goals_window", 15))
    if window_size <= 0:
        window_size = 15

    num_pts = int(env.waypoints.shape[1])
    idx = env.waypoint_idx.clone()

    # Preserve normal advancement for genuinely reached goals.
    idx[reached] = idx[reached] + goal_step

    # For non-reached envs, jump to nearest secondary if closer than current main goal.
    for e in range(env.num_envs):
        if bool(reached[e].item()):
            continue

        curr = int(idx[e].item())
        if curr >= num_pts - 1:
            continue

        sec_start = curr + goal_step
        sec_end = min(num_pts, curr + goal_step * window_size)
        if sec_start >= sec_end:
            continue

        sec_indices = torch.arange(sec_start, sec_end, goal_step, device=env.device, dtype=torch.long)
        if sec_indices.numel() == 0:
            continue

        sec_pts = env.waypoints[e, sec_indices, :]
        sec_dist = torch.norm(sec_pts - robot_pos[e].unsqueeze(0), dim=1)
        best_j = int(torch.argmin(sec_dist).item())
        best_dist = float(sec_dist[best_j].item())
        curr_dist = float(main_dist[e].item())

        if best_dist < curr_dist:
            idx[e] = sec_indices[best_j]

    env.waypoint_idx = torch.clamp(idx, 0, num_pts - 1)

def _quat_to_yaw(quat: torch.Tensor):
    '''
    @brief Convert quaternion to yaw angle.
    
    @param quat: A tensor of shape (num_envs, 4) representing the quaternion (w, x, y, z) for each environment instance.
    '''
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

def _wrap_angle(angle: torch.Tensor):
    '''
    @brief Wrap angle to [-pi, pi].
    
    @param angle: A tensor of shape (num_envs,) representing the angle for each environment instance.
    '''
    return (angle + math.pi) % (2 * math.pi) - math.pi

def _switch_path(num_paths: int, device: torch.device) -> torch.Tensor:
    '''
    @brief Sample a path type per environment, similar to the classic env's _switch_path behavior.

    Path IDs:
      0 -> straight
      1 -> left_curve
      2 -> right_curve
      3 -> sine
    '''
    return torch.randint(low=0, high=4, size=(num_paths,), device=device)

def _create_path(start_x: float,
                 start_y: float,
                 path_type: int,
                 num_pts: int,
                 step_len: float = 1.0,) -> np.ndarray:
    '''
    @brief Create one 2D path (N x 2) similarly to the classic env's _create_path abstraction.
    '''
    pts = np.zeros((num_pts, 2), dtype=np.float32)
    x = float(start_x)
    y = float(start_y)

    # Per-path local state.
    heading = 0.0
    base_y = y

    for i in range(num_pts):
        pts[i, 0] = x
        pts[i, 1] = y

        if path_type == 0:
            x += step_len
        elif path_type == 1:
            # left curve
            heading += 0.12
            x += math.cos(heading) * step_len
            y += math.sin(heading) * step_len
        elif path_type == 2:
            # right curve
            heading -= 0.12
            x += math.cos(heading) * step_len
            y += math.sin(heading) * step_len
        else:
            # sine-like path
            x += step_len
            y = base_y + 1.5 * math.sin(0.35 * (i + 1))

    return pts

def _analytic_lidar_for_env(
    env: "ManagerBasedRLEnv",
    env_id: int,
    num_rays: int,
    fov_deg: float,
    max_distance: float,
):
    '''
    @brief Analytic LiDAR fallback using ray-circle intersection against obstacle objects.
    '''
    # Anchor fallback rays to LiDAR frame (base_scan) to match mounted sensor location.
    lidar = env.scene["lidar"]
    sensor_pos = lidar.data.pos_w[env_id]
    sensor_xy = sensor_pos[:2]
    sensor_z = float(sensor_pos[2].item())
    yaw = float(_quat_to_yaw(lidar.data.quat_w[env_id:env_id + 1])[0].item())

    if num_rays <= 1:
        rel_angles = np.array([0.0], dtype=np.float32)
    else:
        rel_angles = np.linspace(-0.5 * np.deg2rad(fov_deg), 0.5 * np.deg2rad(fov_deg), num_rays, dtype=np.float32)

    obstacle_names = [f"obstacle_{i}" for i in range(5)]
    obstacle_radii = getattr(env, "obstacle_radii", [0.25, 0.22, 0.20, 0.18, 0.16])

    centers = []
    radii = []
    for i, name in enumerate(obstacle_names):
        try:
            pos_xy = env.scene[name].data.root_pos_w[env_id, :2]
            centers.append(pos_xy)
            radii.append(float(obstacle_radii[i]))
        except Exception:
            continue

    starts = []
    hits = []
    dists = []
    origin_np = sensor_xy.detach().cpu().numpy()

    for rel in rel_angles:
        ang = yaw + float(rel)
        direction = np.array([math.cos(ang), math.sin(ang)], dtype=np.float32)

        best_t = float(max_distance)
        for c_xy_t, r in zip(centers, radii):
            c = c_xy_t.detach().cpu().numpy().astype(np.float32)
            d = origin_np - c
            a = 1.0
            b = 2.0 * float(np.dot(d, direction))
            ccoef = float(np.dot(d, d) - r * r)
            disc = b * b - 4.0 * a * ccoef
            if disc < 0.0:
                continue
            sqrt_disc = math.sqrt(disc)
            t1 = (-b - sqrt_disc) / (2.0 * a)
            t2 = (-b + sqrt_disc) / (2.0 * a)
            candidates = [t for t in (t1, t2) if t >= 0.0]
            if candidates:
                t = min(candidates)
                if t < best_t:
                    best_t = t

        hit_xy = origin_np + direction * best_t
        starts.append((float(origin_np[0]), float(origin_np[1]), sensor_z))
        hits.append((float(hit_xy[0]), float(hit_xy[1]), sensor_z))
        dists.append(float(best_t))

    return starts, hits, torch.tensor(dists, device=env.device, dtype=torch.float32)

def _visual_markers(env: "ManagerBasedRLEnv"):
    '''
    @brief Visualize the goals and path.
    '''
    if not hasattr(env, "waypoints") or env.num_envs == 0:
        return

    # Lazy init of debug draw interface (fallback-friendly across Isaac versions)
    if not hasattr(env, "_debug_draw"):
        env._debug_draw = None
        try:
            from omni.isaac.debug_draw import _debug_draw

            env._debug_draw = _debug_draw.acquire_debug_draw_interface()
        except Exception:
            try:
                from isaacsim.util.debug_draw import _debug_draw

                env._debug_draw = _debug_draw.acquire_debug_draw_interface()
            except Exception as e:
                print(f"[DEBUG] Could not initialize debug draw interface: {e}")

    debug_draw = env._debug_draw
    if debug_draw is None:
        return

    # Draw only env_0 for clarity.
    env_id = 0
    wp = env.waypoints[env_id]
    if wp.numel() == 0:
        return

    active_idx = int(env.waypoint_idx[env_id].item()) if hasattr(env, "waypoint_idx") else 0
    active_idx = max(0, min(active_idx, wp.shape[0] - 1))

    # Reproduce matplotlib logic:
    # - Main goal: current waypoint index (green)
    # - Secondary goals: next goals in window (pink)
    # - Path: continuous line over all waypoints
    goal_step = int(getattr(env, "goal_step", 1))
    if goal_step <= 0:
        goal_step = 1
    window_size = int(getattr(env, "num_goals_window", min(15, int(wp.shape[0]))))
    if window_size <= 0:
        window_size = min(15, int(wp.shape[0]))

    secondary_start = active_idx + 1
    secondary_end = min(int(wp.shape[0]), active_idx + goal_step * window_size)
    secondary_indices = list(range(secondary_start, secondary_end, goal_step))

    z = 0.08

    # Clear previous draw from this frame.
    try:
        debug_draw.clear_lines()
        debug_draw.clear_points()
    except Exception:
        pass

    # Draw full path as continuous line.
    if wp.shape[0] > 1:
        p0 = []
        p1 = []
        colors = []
        widths = []
        for i in range(int(wp.shape[0]) - 1):
            a = wp[i]
            b = wp[i + 1]
            p0.append((float(a[0]), float(a[1]), z))
            p1.append((float(b[0]), float(b[1]), z))
            colors.append((0.1, 0.8, 1.0, 1.0))
            widths.append(2.5)
        try:
            debug_draw.draw_lines(p0, p1, colors, widths)
        except Exception as e:
            print(f"[DEBUG] draw_lines failed: {e}")

    # Draw main and secondary goals.
    points = []
    point_colors = []
    point_sizes = []

    main_goal = wp[active_idx]
    points.append((float(main_goal[0]), float(main_goal[1]), z + 0.02))
    point_colors.append((0.0, 1.0, 0.0, 1.0))  # green
    point_sizes.append(24.0)

    for idx in secondary_indices:
        if 0 <= idx < int(wp.shape[0]):
            p = wp[idx]
            points.append((float(p[0]), float(p[1]), z + 0.01))
            point_colors.append((1.0, 0.0, 1.0, 1.0))  # pink/magenta
            point_sizes.append(16.0)

    if points:
        try:
            debug_draw.draw_points(points, point_colors, point_sizes)
        except Exception as e:
            print(f"[DEBUG] draw_points failed: {e}")

    # Draw LiDAR rays (env_0) if sensor is available and visualization enabled.
    if bool(getattr(env, "lidar_debug_vis", True)):
        try:
            lidar = env.scene["lidar"]
            ray_hits = lidar.data.ray_hits_w[env_id]

            # IsaacLab version compatibility: RayCasterData exposes sensor position (pos_w)
            # and hit points (ray_hits_w), but not ray_starts_w in this release.
            sensor_pos = lidar.data.pos_w[env_id]
            ray_starts = sensor_pos.unsqueeze(0).repeat(ray_hits.shape[0], 1)

            finite_mask = torch.isfinite(ray_hits).all(dim=-1)
            if finite_mask.any():
                lidar_p0 = [tuple(v.tolist()) for v in ray_starts[finite_mask]]
                lidar_p1 = [tuple(v.tolist()) for v in ray_hits[finite_mask]]
                lidar_colors = [(1.0, 1.0, 0.0, 0.65)] * len(lidar_p0)  # yellow
                lidar_widths = [1.0] * len(lidar_p0)
                debug_draw.draw_lines(lidar_p0, lidar_p1, lidar_colors, lidar_widths)
            else:
                # Fallback: draw analytic LiDAR rays based on obstacle circles.
                num_rays = int(getattr(env, "lidar_num_rays", 24))
                fov_deg = float(getattr(env, "lidar_fov_deg", 180.0))
                max_dist = float(getattr(env, "lidar_max_distance", 2.5))
                lidar_p0, lidar_p1, _ = _analytic_lidar_for_env(env, env_id, num_rays, fov_deg, max_dist)
                lidar_colors = [(1.0, 0.8, 0.0, 0.65)] * len(lidar_p0)  # orange/yellow fallback
                lidar_widths = [1.0] * len(lidar_p0)
                debug_draw.draw_lines(lidar_p0, lidar_p1, lidar_colors, lidar_widths)

        except Exception:
            # Keep silent to avoid flooding if sensor isn't initialized in early steps.
            pass

def lidar_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute LiDAR observation as normalized ray distances in [0, 1].
    '''
    try:
        lidar = env.scene["lidar"]
    except Exception:
        # Fallback shape: [num_envs, 1] if sensor is unavailable.
        return torch.zeros((env.num_envs, 1), device=env.device)

    ray_hits = lidar.data.ray_hits_w

    # Determine expected LiDAR dimension from config for robust init-time fallback.
    pattern_cfg = lidar.cfg.pattern_cfg
    fov_min, fov_max = float(pattern_cfg.horizontal_fov_range[0]), float(pattern_cfg.horizontal_fov_range[1])
    horiz_res = max(1e-6, float(pattern_cfg.horizontal_res))
    channels = max(1, int(getattr(pattern_cfg, "channels", 1)))
    rays_per_channel = max(1, int(round((fov_max - fov_min) / horiz_res)) + 1)
    expected_dim = channels * rays_per_channel

    if ray_hits is None:
        # Analytic fallback for this sensor API/runtime state.
        num_rays = int(getattr(env, "lidar_num_rays", expected_dim))
        fov_deg = float(getattr(env, "lidar_fov_deg", 180.0))
        max_dist = float(getattr(env, "lidar_max_distance", 2.5))
        fallback = torch.zeros((env.num_envs, num_rays), device=env.device)
        for e in range(env.num_envs):
            _, _, d = _analytic_lidar_for_env(env, e, num_rays, fov_deg, max_dist)
            fallback[e] = torch.clamp(1.0 - (d / max_dist), 0.0, 1.0)
        return fallback

    sensor_pos = lidar.data.pos_w.unsqueeze(1).expand(-1, ray_hits.shape[1], -1)
    dists = torch.norm(ray_hits - sensor_pos, dim=-1)

    max_dist = float(getattr(lidar.cfg, "max_distance", 2.5))
    dists = torch.nan_to_num(dists, nan=max_dist, posinf=max_dist, neginf=0.0)
    dists = torch.clamp(dists, 0.0, max_dist)

    # Match classic env convention: 0.0 -> far, 1.0 -> near obstacle.
    norm = 1.0 - (dists / max_dist)
    norm = torch.clamp(norm, 0.0, 1.0)

    # If no finite hits from ray caster, use analytic obstacle fallback.
    finite_hits_any = torch.isfinite(ray_hits).all(dim=-1).any(dim=1)
    if (~finite_hits_any).any():
        num_rays = int(getattr(env, "lidar_num_rays", norm.shape[1]))
        fov_deg = float(getattr(env, "lidar_fov_deg", 180.0))
        fallback_max_dist = float(getattr(env, "lidar_max_distance", max_dist))
        for e in range(env.num_envs):
            if not bool(finite_hits_any[e].item()):
                _, _, d = _analytic_lidar_for_env(env, e, num_rays, fov_deg, fallback_max_dist)
                norm[e, :num_rays] = torch.clamp(1.0 - (d / fallback_max_dist), 0.0, 1.0)

    return norm

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
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    @property
    def raw_actions(self) -> torch.Tensor: return self._raw_actions
    
    @property
    def processed_actions(self) -> torch.Tensor: return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        '''
        @brief Process the raw actions from the RL agent, which are expected to be in the range [-1, 1], and convert them
        to wheel velocity targets for the left and right wheels of the differential drive robot.

        @param actions: A tensor of shape (num_envs, 2) where actions[:, 0] is the linear velocity command and actions[:, 1]
        is the angular velocity command, both normalized to [-1, 1].
        '''
        
        self._raw_actions = actions.clone()
        v = actions[:, 0] * self._lin_scale
        w = actions[:, 1] * self._ang_scale
        v_left  = (v - w * self._L / 2.0) / self._r
        v_right = (v + w * self._L / 2.0) / self._r
        self._processed_actions = torch.stack([v_left, v_right], dim=1)

    def apply_actions(self):
        '''
        @brief Apply the processed actions to the robot's wheel joints by setting their velocity targets.
        '''
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

def goal_observation(env: ManagerBasedRLEnv, lookahead: int = 1) -> torch.Tensor:
    '''
    @brief Compute the goal observation, which consists of the distance and relative angle to the current target waypoint.

    @param lookahead: How many waypoints ahead to look when computing the goal observation. Default is 1 (the current target waypoint).
    '''
    dist, rel_angle = _goal_distance_and_yaw_error(env)

    # Update debug visualization of full waypoint path + current goal.
    _visual_markers(env)
    
    obs = torch.stack([dist, rel_angle], dim=1)
    return torch.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)

def goal_distance_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Distance to current main goal. Shape: (num_envs, 1).
    '''
    dist, _ = _goal_distance_and_yaw_error(env)
    return torch.nan_to_num(dist.unsqueeze(1), nan=0.0, posinf=10.0, neginf=0.0)

def yaw_error_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Relative yaw error to current main goal. Shape: (num_envs, 1).
    '''
    _, rel_angle = _goal_distance_and_yaw_error(env)

    # Keep debug drawing tied to navigation observables.
    _visual_markers(env)

    return torch.nan_to_num(rel_angle.unsqueeze(1), nan=0.0, posinf=math.pi, neginf=-math.pi)

def previous_action_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Previous normalized action [linear_cmd, angular_cmd]. Shape: (num_envs, 2).

    Mirrors the simple simulation, which exposes the action from the prior step.
    '''
    prev = env.extras.get("prev_action_obs", None)
    if prev is None or not torch.is_tensor(prev) or prev.shape != (env.num_envs, 2):
        prev = torch.zeros((env.num_envs, 2), device=env.device)

    output = prev.clone()
    current = _get_current_raw_actions(env)
    env.extras["prev_action_obs"] = current.clone()

    return torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)

def subgoal_window_distance_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Distances to secondary goals in current window. Shape: (num_envs, num_goals_window).

    Behavior mirrors the simple env:
    - Collect distances to available secondary goals only.
    - If fewer than window size are available, repeat the last available distance.
    - If none are available, fill with current main-goal distance.
    '''
    window_size = int(getattr(env, "num_goals_window", 15))
    goal_step = int(getattr(env, "goal_step", 1))
    window_size = max(1, window_size)
    goal_step = max(1, goal_step)

    out = torch.zeros((env.num_envs, window_size), device=env.device)
    if not hasattr(env, "waypoints") or not hasattr(env, "waypoint_idx"):
        return out

    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    main_dist, _ = _goal_distance_and_yaw_error(env)
    num_pts = int(env.waypoints.shape[1])

    for e in range(env.num_envs):
        curr = int(torch.clamp(env.waypoint_idx[e], 0, num_pts - 1).item())
        sec_start = curr + 1
        sec_end = min(num_pts, curr + goal_step * window_size)

        if sec_start >= sec_end:
            out[e, :] = main_dist[e]
            continue

        sec_indices = torch.arange(sec_start, sec_end, goal_step, device=env.device, dtype=torch.long)
        if sec_indices.numel() == 0:
            out[e, :] = main_dist[e]
            continue

        sec_pts = env.waypoints[e, sec_indices, :]
        d = torch.norm(sec_pts - robot_pos[e].unsqueeze(0), dim=1)

        n = min(window_size, int(d.shape[0]))
        out[e, :n] = d[:n]
        if n < window_size:
            out[e, n:] = d[n - 1] if n > 0 else main_dist[e]

    return torch.nan_to_num(out, nan=0.0, posinf=1e3, neginf=0.0)

def velocity_observation(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute the velocity observation, which consists of the linear and angular velocity of the robot, normalized to [-1, 1] 
    based on expected max values.
    '''
    robot = env.scene["robot"]
    lin_vel = robot.data.root_lin_vel_b[:, 0]
    ang_vel = robot.data.root_ang_vel_b[:, 2]
    
    lin_norm = torch.clamp(lin_vel / 0.22, -1.0, 1.0)
    ang_norm = torch.clamp(ang_vel / 2.84, -1.0, 1.0)
    
    obs = torch.stack([lin_norm, ang_norm], dim=1)
    return torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

def progress_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute the progress reward, which is the decrease in distance to the current target waypoint.
    '''
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)
    current_dist = torch.norm(goals - robot_pos, dim=1)
    prev_dist = env.extras.get("prev_dist_to_goal", current_dist.clone())
    env.extras["prev_dist_to_goal"] = current_dist.clone()
    
    reward = prev_dist - current_dist
    return torch.nan_to_num(reward, nan=0.0, posinf=1.0, neginf=-1.0)

def waypoint_reached_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute the waypoint reached reward, which gives a positive reward when the robot is within a 
    certain distance of the current target waypoint.
    '''
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]
    goals = _get_current_waypoint(env)
    main_dist = torch.norm(goals - robot_pos, dim=1)
    reached = main_dist < 0.2
    _advance_to_secondary_if_closer(env, robot_pos, main_dist, reached)
    return reached.float()

def out_of_bounds_penalty(env: ManagerBasedRLEnv, max_dist: float = 2.0) -> torch.Tensor:
    '''
    @brief Compute the out-of-bounds penalty, which is a negative reward when the robot moves too far from the current target waypoint.
    '''
    return -(torch.norm(_get_current_waypoint(env) - env.scene["robot"].data.root_pos_w[:, :2], dim=1) > max_dist).float()

def alive_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute the alive penalty, which is a small negative reward for each time step the robot remains alive.
    '''
    return -torch.ones(env.num_envs, device=env.device)

def out_of_bounds_termination(env: ManagerBasedRLEnv, max_dist: float = 2.0) -> torch.Tensor:
    '''
    @brief Compute the out-of-bounds termination condition, which is True when the robot moves too far from the current target waypoint.
    '''
    return torch.norm(_get_current_waypoint(env) - env.scene["robot"].data.root_pos_w[:, :2], dim=1) > max_dist

def all_waypoints_reached_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute the termination condition for when all waypoints have been reached.
    '''
    if not hasattr(env, 'waypoints'):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return env.waypoint_idx >= env.waypoints.shape[1]

def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    '''
    @brief Compute the time-out termination condition, which is True when the episode length exceeds the maximum allowed length.
    '''
    return env.episode_length_buf >= env.max_episode_length

def reset_robot_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    '''
    @brief Reset the robot's position and rotation to the starting pose for the given environment IDs. 
    The starting position is based on the first waypoint for each environment instance.

    @param env_ids: A tensor of environment IDs to reset the robot pose for.
    '''
    robot = env.scene["robot"]
    if not hasattr(env, 'waypoints'):
        return
    
    start_pos = env.waypoints[env_ids, 0, :]
    root_state = robot.data.default_root_state[env_ids].clone()
    
    root_state[:, 0:2] = start_pos
    root_state[:, 2] = env.scene.env_origins[env_ids, 2] + 0.02 
    root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    root_state[:, 7:13] = 0.0 
    
    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    
    joint_pos = torch.zeros_like(robot.data.default_joint_pos[env_ids])
    joint_vel = torch.zeros_like(robot.data.default_joint_vel[env_ids])
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

def reset_path_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    '''
    @brief Reset path-related state for the given env IDs:
    regenerate waypoints and reset current path index.
    '''
    num_reset, num_pts, device = len(env_ids), 50, env.device
    waypoints = torch.zeros(num_reset, num_pts, 2, device=device)

    # Keep classic-env goal window defaults available to rendering/logic.
    if not hasattr(env, "goal_step"):
        env.goal_step = 1
    if not hasattr(env, "num_goals_window"):
        env.num_goals_window = 15
    if not hasattr(env, "out_of_bound_threshold"):
        env.out_of_bound_threshold = (env.goal_step * env.num_goals_window) + env.goal_step

    env_origins = env.scene.env_origins[env_ids]
    path_types = _switch_path(num_reset, device)

    for k in range(num_reset):
        start_x = float(env_origins[k, 0].item())
        start_y = float(env_origins[k, 1].item())
        path_np = _create_path(
            start_x=start_x,
            start_y=start_y,
            path_type=int(path_types[k].item()),
            num_pts=num_pts,
            step_len=1.0,
        )
        waypoints[k] = torch.tensor(path_np, device=device)
        
    if not hasattr(env, 'waypoints'):
        env.waypoints = torch.zeros(env.num_envs, num_pts, 2, device=device)
        env.waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    
    env.waypoints[env_ids] = waypoints
    env.waypoint_idx[env_ids] = 1

    # Reset previous-action observation buffer for fresh episodes.
    if "prev_action_obs" not in env.extras or not torch.is_tensor(env.extras.get("prev_action_obs")):
        env.extras["prev_action_obs"] = torch.zeros((env.num_envs, 2), device=device)
    env.extras["prev_action_obs"][env_ids] = 0.0

def reset_obstacles(env: ManagerBasedRLEnv, env_ids: torch.Tensor):
    '''
    @brief Reset obstacles for the given env IDs, similar in spirit to path_multiple_obstacles_lidar.py.

    - Randomly enables 0..N obstacles per env.
    - Places active obstacles near path points with lateral offsets.
    - Moves inactive obstacles far away.
    '''
    obstacle_names = [f"obstacle_{i}" for i in range(5)]
    obstacle_radii = [0.25, 0.22, 0.20, 0.18, 0.16]

    # Resolve obstacle assets once.
    obstacles = []
    for name in obstacle_names:
        try:
            obstacles.append(env.scene[name])
        except Exception:
            return

    # Ensure path-related state exists.
    if not hasattr(env, "waypoints"):
        return

    device = env.device

    for env_id in env_ids.tolist():
        wp = env.waypoints[env_id]
        if wp.shape[0] < 3:
            continue

        # Random number of obstacles [0, 5]
        num_obs = int(torch.randint(low=0, high=len(obstacles) + 1, size=(1,), device=device).item())

        # Sample candidate indices away from very start/end.
        idx_low = 2
        idx_high = max(idx_low + 1, int(wp.shape[0]) - 2)

        if idx_high <= idx_low:
            chosen_indices = []
        else:
            num_choices = min(num_obs, idx_high - idx_low)
            perm = torch.randperm(idx_high - idx_low, device=device)[:num_choices] + idx_low
            chosen_indices = perm.tolist()

        for i, obstacle in enumerate(obstacles):
            if i < len(chosen_indices):
                p_idx = int(chosen_indices[i])
                p = wp[p_idx]

                # Compute local tangent using neighboring points.
                p_prev = wp[max(0, p_idx - 1)]
                p_next = wp[min(int(wp.shape[0]) - 1, p_idx + 1)]
                tangent = p_next - p_prev
                tangent_norm = torch.norm(tangent) + 1e-6
                tangent = tangent / tangent_norm

                # Perpendicular direction and random offset under 1m.
                perp = torch.tensor([-tangent[1], tangent[0]], device=device)
                offset = torch.empty(1, device=device).uniform_(-1.0, 1.0)[0]
                pos_xy = p + perp * offset

                # Keep safe distance from start pose.
                start_xy = wp[0]
                if torch.norm(pos_xy - start_xy) < 1.0:
                    pos_xy = pos_xy + perp * 1.0

                pose = torch.tensor(
                    [[float(pos_xy[0]), float(pos_xy[1]), 0.2, 1.0, 0.0, 0.0, 0.0]],
                    device=device,
                    dtype=torch.float32,
                )

                try:
                    obstacle.write_root_pose_to_sim(pose, env_ids=torch.tensor([env_id], device=device, dtype=torch.long))
                except Exception:
                    # Fallback path for API variants that only expose root state writes.
                    root_state = obstacle.data.default_root_state[torch.tensor([env_id], device=device, dtype=torch.long)].clone()
                    root_state[:, 0] = pose[:, 0]
                    root_state[:, 1] = pose[:, 1]
                    root_state[:, 2] = pose[:, 2]
                    root_state[:, 3:7] = pose[:, 3:7]
                    root_state[:, 7:13] = 0.0
                    obstacle.write_root_state_to_sim(root_state, env_ids=torch.tensor([env_id], device=device, dtype=torch.long))
            else:
                # Disable unused obstacles by moving them far away.
                far_x = float(env.scene.env_origins[env_id, 0].item() + 50.0 + i * 2.0)
                far_y = float(env.scene.env_origins[env_id, 1].item() + 50.0)
                pose = torch.tensor(
                    [[far_x, far_y, 0.2, 1.0, 0.0, 0.0, 0.0]],
                    device=device,
                    dtype=torch.float32,
                )
                try:
                    obstacle.write_root_pose_to_sim(pose, env_ids=torch.tensor([env_id], device=device, dtype=torch.long))
                except Exception:
                    root_state = obstacle.data.default_root_state[torch.tensor([env_id], device=device, dtype=torch.long)].clone()
                    root_state[:, 0] = pose[:, 0]
                    root_state[:, 1] = pose[:, 1]
                    root_state[:, 2] = pose[:, 2]
                    root_state[:, 3:7] = pose[:, 3:7]
                    root_state[:, 7:13] = 0.0
                    obstacle.write_root_state_to_sim(root_state, env_ids=torch.tensor([env_id], device=device, dtype=torch.long))