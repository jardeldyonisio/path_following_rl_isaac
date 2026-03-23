"""
env_cfg.py
----------
Defines HOW the RL task works on top of the scene:
  - Observation space
  - Action space
  - Reward terms
  - Termination / reset conditions
  - Episode length

This is the "brain" of the environment. Isaac Lab calls these
"managers" — each aspect of the RL loop is handled by a dedicated
manager class, configured here as dataclasses.
"""

import math
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils import configclass

from scene_cfg import NavSceneCfg

# We import our custom manager functions defined in mdp.py
import mdp


# ---------------------------------------------------------------------------
# Observation Space
# ---------------------------------------------------------------------------
@configclass
class ObservationsCfg:
    """
    Defines what the agent SEES at each timestep.

    Isaac Lab organises observations into "groups". The most important
    group is 'policy' — it is what gets passed to the neural network.
    You can have a second group (e.g., 'critic') with privileged info
    that only the value function sees (useful for teacher-student later!).
    """

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        """Observations visible to the policy network."""

        # --- Goal information ---
        # Distance and angle to the NEXT waypoint.
        # shape: (2,)
        goal_obs = ObservationTermCfg(
            func=mdp.goal_observation,
            params={"lookahead": 1},  # How many waypoints ahead to look
        )

        # --- Robot velocity ---
        # Linear (v) and angular (w) velocities.
        # shape: (2,)
        velocity_obs = ObservationTermCfg(
            func=mdp.velocity_observation,
        )

        # Concatenated shape: (4,) = 2 + 2
        # Future: add LiDAR (+24), articulation angle (+1), more waypoints (+N)

    policy: PolicyCfg = PolicyCfg()


# ---------------------------------------------------------------------------
# Action Space
# ---------------------------------------------------------------------------
@configclass
class ActionsCfg:
    """
    Defines what the agent DOES.

    We use a DifferentialControllerActionCfg to map the network output
    directly to wheel velocities. The network outputs [v, w] and Isaac
    Lab converts them to left/right wheel velocities automatically.
    """
    # We keep it simple for now: the action is [linear_vel, angular_vel]
    # handled in mdp.py via a custom action term.
    robot_vel = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        # TurtleBot3 Burger wheel joint names from the USD
        left_joint_name="wheel_left_joint",
        right_joint_name="wheel_right_joint",
        wheel_radius=0.033,       # meters — TurtleBot3 spec
        wheel_base=0.16,          # meters — TurtleBot3 spec (distance between wheels)
        # Action scale: maps [-1, 1] output to actual velocity range
        # linear:  [-1, 1] → [-0.25, 0.25] m/s  (matches your original env)
        # angular: [-1, 1] → [-0.5,  0.5]  rad/s (matches your original env)
        linear_vel_scale=0.25,
        angular_vel_scale=0.5,
    )


# ---------------------------------------------------------------------------
# Reward Terms
# ---------------------------------------------------------------------------
@configclass
class RewardsCfg:
    """
    Each field is one reward term. They are summed at every step.
    'weight' scales how important each term is.
    Positive weight = reward, negative weight = penalty.
    """

    # +1 for moving closer to the next waypoint
    progress_reward = RewardTermCfg(
        func=mdp.progress_reward,
        weight=5.0,
    )

    # +10 for reaching a waypoint (clearing it from the queue)
    waypoint_reached = RewardTermCfg(
        func=mdp.waypoint_reached_reward,
        weight=10.0,
    )

    # -1 per step if too far from the path (out-of-bounds penalty)
    out_of_bounds_penalty = RewardTermCfg(
        func=mdp.out_of_bounds_penalty,
        params={"max_dist": 2.0},
        weight=-1.0,
    )

    # Small penalty for being alive (encourages the agent to be efficient)
    alive_penalty = RewardTermCfg(
        func=mdp.alive_penalty,
        weight=-0.01,
    )


# ---------------------------------------------------------------------------
# Termination Conditions
# ---------------------------------------------------------------------------
@configclass
class TerminationsCfg:
    """
    Episode ends when ANY of these conditions is True.
    'time_out' is handled automatically by Isaac Lab via max_episode_length.
    """

    # Episode ends if the robot is too far from the path
    out_of_bounds = TerminationTermCfg(
        func=mdp.out_of_bounds_termination,
        params={"max_dist": 2.0},
    )

    # Episode ends when ALL waypoints are reached (success!)
    all_waypoints_reached = TerminationTermCfg(
        func=mdp.all_waypoints_reached_termination,
    )

    # Built-in Isaac Lab: ends episode after max_episode_length steps
    time_out = TerminationTermCfg(
        func=mdp.time_out,
        time_out=True,  # Marks this as a timeout (not a failure) for bootstrapping
    )


# ---------------------------------------------------------------------------
# Reset / Randomization Events
# ---------------------------------------------------------------------------
@configclass
class EventsCfg:
    """
    Events are triggered at specific moments (reset, startup, interval).
    This is also where Domain Randomization lives in the future.
    """

    # On every episode reset: randomize robot starting pose
    reset_robot_pose = EventTermCfg(
        func=mdp.reset_robot_pose,
        mode="reset",  # Triggers on every env.reset()
    )

    # On every episode reset: generate a new random waypoint path
    reset_waypoints = EventTermCfg(
        func=mdp.reset_waypoints,
        mode="reset",
    )


# ---------------------------------------------------------------------------
# Main Environment Config
# ---------------------------------------------------------------------------
@configclass
class TurtlebotNavEnvCfg(ManagerBasedRLEnvCfg):
    """
    Top-level config. Combines scene + all managers.
    This is what you pass to ManagerBasedRLEnv() to create the environment.
    """

    # Wire up all managers defined above
    scene: NavSceneCfg = NavSceneCfg(num_envs=1, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        """Called after dataclass init. Use to set derived parameters."""
        super().__post_init__()

        # Simulation timestep (seconds). TurtleBot3 dynamics are stable at 0.02s.
        self.sim.dt = 0.02          # 50 Hz physics
        self.decimation = 4         # Policy runs at 50/4 = ~12.5 Hz (like your original env)

        # Episode length: 800 ticks matches your original max_tick
        self.episode_length_s = self.decimation * self.sim.dt * 800  # ~12.8 seconds

        # Viewer camera position (only used in interactive mode, ignored during headless training)
        self.viewer.eye = (5.0, 5.0, 5.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
