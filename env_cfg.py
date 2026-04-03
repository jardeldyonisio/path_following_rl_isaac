"""
env_cfg.py
----------
Environment configuration for TurtleBot3 Navigation.
Defines managers for actions, observations, rewards, terminations, and events.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
    EventTermCfg,
)
from isaaclab.utils import configclass

# Import the scene and custom MDP functions
from scene_cfg import NavSceneCfg
import mdp

# ===========================================================================
# Managers Configuration
# ===========================================================================

@configclass
class ActionsCfg:
    """Action configuration."""
    robot_vel = mdp.DifferentialDriveActionCfg()

@configclass
class ObservationsCfg:
    """Observation configuration."""
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        goal_obs = ObservationTermCfg(func=mdp.goal_observation, params={"lookahead": 1})
        velocity_obs = ObservationTermCfg(func=mdp.velocity_observation)

    policy: PolicyCfg = PolicyCfg()

@configclass
class RewardsCfg:
    """Reward configuration."""
    progress_reward = RewardTermCfg(func=mdp.progress_reward, weight=5.0)
    waypoint_reached = RewardTermCfg(func=mdp.waypoint_reached_reward, weight=10.0)
    out_of_bounds_penalty = RewardTermCfg(func=mdp.out_of_bounds_penalty, weight=-1.0)
    alive_penalty = RewardTermCfg(func=mdp.alive_penalty, weight=-0.01)

@configclass
class TerminationsCfg:
    """Termination configuration."""
    out_of_bounds = TerminationTermCfg(func=mdp.out_of_bounds_termination)
    all_waypoints_reached = TerminationTermCfg(func=mdp.all_waypoints_reached_termination)
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)

@configclass
class EventsCfg:
    """Event configuration (resets)."""
    # 1. Reset the waypoints when an episode ends
    reset_waypoints = EventTermCfg(
        func=mdp.reset_waypoints,
        mode="reset",
    )
    
    # 2. Reset the robot's position and rotation (This was the missing piece!)
    reset_robot_pose = EventTermCfg(
        func=mdp.reset_robot_pose,
        mode="reset",
    )

# ===========================================================================
# Environment Configuration
# ===========================================================================

@configclass
class TurtlebotNavEnvCfg(ManagerBasedRLEnvCfg):
    """Main environment configuration."""
    
    # Scene settings (We default to 16, but your train.py overrides this if you pass --num_envs 1)
    scene: NavSceneCfg = NavSceneCfg(num_envs=16, env_spacing=4.0)
    
    # Register the managers
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    
    # Episode settings
    episode_length_s: float = 20.0
    decimation: int = 4
    
    def __post_init__(self):
        super().__post_init__()
        
        # Viewer camera settings
        self.viewer.eye = [5.0, 5.0, 5.0]
        self.viewer.lookat = [0.0, 0.0, 0.0]
        
        # Physics step sizes
        self.sim.dt = 0.02
        # Syncing render interval with decimation to remove the terminal warning
        self.sim.render_interval = self.decimation