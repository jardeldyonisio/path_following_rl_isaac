'''
@brief Defined managers for actions, observations, rewards, terminations, and events for a robot navigation task.
'''

import mdp

from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
)
from scene_cfg import NavSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg

@configclass
class ActionsCfg:
    '''
    @brief Action configuration for the robot navigation task.
    '''
    robot_vel = mdp.DifferentialDriveActionCfg()

@configclass
class ObservationsCfg:
    '''
    @brief Observation configuration for the robot navigation task.
    '''
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        goal_distance_obs = ObservationTermCfg(func=mdp.goal_distance_observation)
        previous_action_obs = ObservationTermCfg(func=mdp.previous_action_observation)
        yaw_error_obs = ObservationTermCfg(func=mdp.yaw_error_observation)
        lidar_obs = ObservationTermCfg(func=mdp.lidar_observation)
        subgoal_distances_obs = ObservationTermCfg(func=mdp.subgoal_window_distance_observation)

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    '''
    @brief Reward configuration for the robot navigation task.
    '''
    reward_goal_reached = RewardTermCfg(func=mdp.reward_goal_reached, weight=1.0)
    reward_subgoal_reached = RewardTermCfg(func=mdp.reward_subgoal_reached, weight=1.0)
    reward_success = RewardTermCfg(func=mdp.reward_success, weight=1.0)
    reward_truncated = RewardTermCfg(func=mdp.reward_truncated, weight=1.0)
    reward_direction_penalty = RewardTermCfg(func=mdp.reward_direction_penalty, weight=1.0)

@configclass
class TerminationsCfg:
    '''
    @brief Termination configuration for the robot navigation task.
    '''
    out_of_bounds = TerminationTermCfg(func=mdp.out_of_bounds_termination)
    obstacle_collision = TerminationTermCfg(func=mdp.obstacle_collision_termination)
    all_waypoints_reached = TerminationTermCfg(func=mdp.all_waypoints_reached_termination)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

@configclass
class EventsCfg:
    '''
    @brief Event configuration (resets).
    '''
    reset_path_state = EventTermCfg(
        func=mdp.reset_path_state,
        mode="reset",
    )

    reset_obstacles = EventTermCfg(
        func=mdp.reset_obstacles,
        mode="reset",
    )
    
    reset_robot_pose = EventTermCfg(
        func=mdp.reset_robot_pose,
        mode="reset",
    )

@configclass
class TurtlebotNavEnvCfg(ManagerBasedRLEnvCfg):
    '''
    @brief Environment configuration for a robot navigation task.
    '''
    
    # Scene settings 
    scene: NavSceneCfg = NavSceneCfg(num_envs=1, env_spacing=4.0)
    
    # Register the managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    # Episode settings
    episode_length_s: float = 1000.0
    decimation: int = 4

    # LiDAR settings
    lidar_num_rays: int = 24
    lidar_fov_deg: float = 180.0
    lidar_max_distance: float = 20.0
    lidar_debug_vis: bool = True
    
    def __post_init__(self):
        super().__post_init__()

        # Apply LiDAR settings to scene sensor config
        half_fov = 0.5 * float(self.lidar_fov_deg)
        # For N rays across FOV, step is ~FOV/(N-1) (min 1 ray safeguard)
        ray_count = max(1, int(self.lidar_num_rays))
        horiz_res = float(self.lidar_fov_deg) / max(1, ray_count - 1)
        self.scene.lidar.pattern_cfg.horizontal_fov_range = (-half_fov, half_fov)
        self.scene.lidar.pattern_cfg.horizontal_res = horiz_res
        self.scene.lidar.max_distance = float(self.lidar_max_distance)
        # Keep IsaacLab internal RayCaster debug visualization disabled.
        # We render LiDAR safely through mdp._visual_markers (lidar_debug_vis).
        self.scene.lidar.debug_vis = False
        
        # Viewer camera settings
        self.viewer.eye = [5.0, 5.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        
        # Physics step sizes
        self.sim.dt = 0.02
        # Syncing render interval with decimation to remove the terminal warning
        self.sim.render_interval = self.decimation