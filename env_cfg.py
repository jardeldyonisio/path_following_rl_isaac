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
from scene_cfg import (
    NavSceneCfg,
    TURTLEBOT3_USD_PATH,
    GLR_USD_PATH,
    GLR_TUGGER_USD_PATH,
)
from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedRLEnvCfg


ROBOT_PROFILES = {
    "turtlebot3": {
        "usd_path": TURTLEBOT3_USD_PATH,
        "lidar_prim_path": "{ENV_REGEX_NS}/Robot/base_footprint/base_link/base_scan",
        "left_joint_name": "wheel_left_joint",
        "right_joint_name": "wheel_right_joint",
        "wheel_radius": 0.033,
        "wheel_base": 0.16,
        "reset_quat": (1.0, 0.0, 0.0, 0.0),
        "robot_radius": 0.105,
        "linear_vel_scale": 0.22,
        "angular_vel_scale": 2.84,
        "min_linear_velocity": -0.22,
        "max_linear_velocity": 0.22,
        "min_angular_velocity": -2.84,
        "max_angular_velocity": 2.84,
    },
    "glr": {
        "usd_path": GLR_USD_PATH,
        "lidar_prim_path": "{ENV_REGEX_NS}/Robot/base_link/chassibigga/lidar_link",
        "left_joint_name": "left_wheel_joint",
        "right_joint_name": "right_wheel_joint",
        "wheel_radius": 0.1,
        "wheel_base": 0.5,
        "reset_quat": (0.7071068, 0.0, 0.0, 0.7071068),
        "robot_radius": 0.4,
        "linear_vel_scale": 1.0,
        "angular_vel_scale": 0.5,
        "min_linear_velocity": -1.0,
        "max_linear_velocity": 1.0,
        "min_angular_velocity": -0.5,
        "max_angular_velocity": 0.5,
    },
    "glr_tugger": {
        "usd_path": GLR_TUGGER_USD_PATH,
        "lidar_prim_path": "{ENV_REGEX_NS}/Robot/base_link/chassibigga/lidar_link",
        "left_joint_name": "left_wheel_joint",
        "right_joint_name": "right_wheel_joint",
        "wheel_radius": 0.1,
        "wheel_base": 0.5,
        "reset_quat": (0.7071068, 0.0, 0.0, 0.7071068),
        "robot_radius": 0.4,
        "linear_vel_scale": 1.0,
        "angular_vel_scale": 0.5,
        "min_linear_velocity": -0.2,
        "max_linear_velocity": 1.0,
        "min_angular_velocity": -0.5,
        "max_angular_velocity": 0.5,
    },
}

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
    @brief Reward and penalty for the agent.
    '''

    # Rewards
    goal_reached_reward = RewardTermCfg(func=mdp.goal_reached_reward, weight=1.0)
    subgoal_reached_reward = RewardTermCfg(func=mdp.subgoal_reached_reward, weight=1.0)
    success_reward = RewardTermCfg(func=mdp.success_reward, weight=1.0)
    progress_reward = RewardTermCfg(func=mdp.progress_reward, weight=1.0)

    # Penalties
    # angular_velocity_penalty = RewardTermCfg(func=mdp.angular_velocity_penalty, weight=1.0)
    direction_penalty = RewardTermCfg(func=mdp.direction_penalty, weight=1.0)
    truncated_penalty = RewardTermCfg(func=mdp.truncated_penalty, weight=1.0)
    # alive_penalty = RewardTermCfg(func=mdp.alive_penalty, weight=1.0)
    # reverse_penalty = RewardTermCfg(func=mdp.reverse_penalty, weight=1.0)

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
class ConvoyNavigationEnvCgf(ManagerBasedRLEnvCfg):
    '''
    @brief Environment configuration for a robot navigation task.
    '''

    robot: str = "turtlebot3"
    
    # Scene settings 
    scene: NavSceneCfg = NavSceneCfg(num_envs=1, env_spacing=4.0)
    
    # Register the managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    # Episode settings
    episode_length_s: float = 100.0
    decimation: int = 4

    # LiDAR settings
    lidar_num_rays: int = 24
    lidar_fov_deg: float = 180.0
    lidar_max_distance: float = 20.0
    lidar_debug_vis: bool = True
    
    def __post_init__(self):
        super().__post_init__()

        if self.robot not in ROBOT_PROFILES:
            raise ValueError(f"Unsupported robot '{self.robot}'. Expected one of: {sorted(ROBOT_PROFILES)}")

        robot_profile = ROBOT_PROFILES[self.robot]
        self.robot_name = self.robot

        self.scene.robot.spawn.usd_path = robot_profile["usd_path"]
        self.scene.robot.init_state.rot = robot_profile["reset_quat"]
        self.scene.robot.actuators["drive_wheels"].joint_names_expr = [
            robot_profile["left_joint_name"],
            robot_profile["right_joint_name"],
        ]
        self.scene.lidar.prim_path = robot_profile["lidar_prim_path"]

        self.actions.robot_vel.left_joint_name = robot_profile["left_joint_name"]
        self.actions.robot_vel.right_joint_name = robot_profile["right_joint_name"]
        self.actions.robot_vel.wheel_radius = robot_profile["wheel_radius"]
        self.actions.robot_vel.wheel_base = robot_profile["wheel_base"]
        linear_min = robot_profile.get("min_linear_velocity")
        if linear_min is None:
            linear_min = -robot_profile["linear_vel_scale"]
        linear_max = robot_profile.get("max_linear_velocity")
        if linear_max is None:
            linear_max = robot_profile["linear_vel_scale"]

        angular_min = robot_profile.get("min_angular_velocity")
        if angular_min is None:
            angular_min = -robot_profile["angular_vel_scale"]
        angular_max = robot_profile.get("max_angular_velocity")
        if angular_max is None:
            angular_max = robot_profile["angular_vel_scale"]

        self.actions.robot_vel.min_linear_velocity = float(linear_min)
        self.actions.robot_vel.max_linear_velocity = float(linear_max)
        self.actions.robot_vel.min_angular_velocity = float(angular_min)
        self.actions.robot_vel.max_angular_velocity = float(angular_max)

        self.actions.robot_vel.linear_vel_scale = max(abs(float(linear_min)), abs(float(linear_max)))
        self.actions.robot_vel.angular_vel_scale = max(abs(float(angular_min)), abs(float(angular_max)))

        self.min_linear_velocity = float(linear_min)
        self.max_linear_velocity = float(linear_max)
        self.min_angular_velocity = float(angular_min)
        self.max_angular_velocity = float(angular_max)

        self.linear_vel_scale = self.actions.robot_vel.linear_vel_scale
        self.angular_vel_scale = self.actions.robot_vel.angular_vel_scale
        self.robot_radius = robot_profile["robot_radius"]
        self._robot_reset_quat = robot_profile["reset_quat"]

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