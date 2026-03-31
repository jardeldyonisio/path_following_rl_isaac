"""
scene_cfg.py
------------
Defines WHAT exists in the simulation world:
  - Ground plane
  - Lighting
  - The TurtleBot3 robot
  - (future) walls, obstacles, trailer

Isaac Lab uses a declarative approach: you describe the scene in a
SceneCfg dataclass, and the framework handles spawning + physics.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

# ---------------------------------------------------------------------------
# Robot Asset
# ---------------------------------------------------------------------------
TURTLEBOT3_CFG = ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/lognav/turtlebot3_ws/src/turtlebot3/turtlebot3_description/urdf/turtlebot3_burger/turtlebot3_burger.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),
        joint_pos={".*": 0.0},
    ),
    # ---------------------------------------------------------------------------
    # Actuators block — REQUIRED by Isaac Lab.
    # Tells the physics engine how to control each joint.
    #
    # ImplicitActuatorCfg is the simplest option: the physics engine handles
    # the motor model internally. You set a velocity target and PhysX drives
    # the joint towards it. This is perfect for wheel velocity control.
    #
    # ".*wheel.*" is a regex that matches both wheel joints regardless of
    # their exact name (wheel_left_joint, wheel_right_joint, etc.)
    # ---------------------------------------------------------------------------
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            effort_limit=1.0,         # Nm — TurtleBot3 Burger motor limit
            velocity_limit=10.0,      # rad/s — well above our max needed
            stiffness=0.0,            # 0 stiffness = velocity control mode
            damping=1.0,              # resistance that stabilises the wheel
        ),
    },
)


# ---------------------------------------------------------------------------
# Scene Configuration
# ---------------------------------------------------------------------------
@configclass
class NavSceneCfg(InteractiveSceneCfg):
    """
    Everything that exists in the simulation world.

    InteractiveSceneCfg is the Isaac Lab base class for scenes.
    Each field becomes a prim (entity) in the USD stage.
    """

    # --- Ground plane ---
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # --- Lighting ---
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    # --- The Robot ---
    # {ENV_REGEX_NS} expands to /World/envs/env_0, env_1, etc.
    # enabling multiple parallel environments automatically.
    robot: ArticulationCfg = TURTLEBOT3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )