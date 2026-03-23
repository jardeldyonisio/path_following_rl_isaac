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
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# ---------------------------------------------------------------------------
# Robot Asset
# ---------------------------------------------------------------------------
# Isaac Lab ships a TurtleBot3 Burger USD on the Nucleus server.
# ArticulationCfg wraps the USD and tells Isaac how to treat it as a
# physics articulation (wheels, joints, etc.).
TURTLEBOT3_CFG = ArticulationCfg(
    # Path on the Omniverse Nucleus (downloaded automatically on first run).
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Clearpath/TurtleBot3/turtlebot3_burger.usd",
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
    # Where to place the robot at startup (before reset() is called).
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.01),  # Slightly above ground to avoid z-fighting
        joint_pos={".*": 0.0},  # All joints start at 0
    ),
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
    # AssetBaseCfg is used for static, non-articulated objects.
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # --- Lighting ---
    # A distant light simulates sunlight. Required for rendering.
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    # --- The Robot ---
    # The 'robot' attribute name is important: we reference it later as
    # scene["robot"] inside the environment.
    robot: ArticulationCfg = TURTLEBOT3_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
        # {ENV_REGEX_NS} is a special Isaac Lab token that expands to
        # /World/envs/env_0, /World/envs/env_1, etc. when you run
        # multiple parallel environments. Very useful for vectorized training.
    )
