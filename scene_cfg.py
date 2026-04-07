'''
@brief Defines what exists in the scene, including the robot, the ground plane, and the light source.
'''
import os
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

ASSETS_DIR = "/home/lognav/Jardel/path_following_rl_isaac/assets"
PHYSICS_USD = os.path.join(ASSETS_DIR, "turtlebot3_burger_fixed", "configuration", "turtlebot3_burger_fixed_physics.usd")
READY_USD = os.path.join(ASSETS_DIR, "turtlebot3_burger_ready.usd")
FIXED_USD = os.path.join(ASSETS_DIR, "turtlebot3_burger_fixed", "turtlebot3_burger_fixed.usd")

# `turtlebot3_burger_ready.usd` currently contains nested articulation roots
# (/World and /World/turtlebot3_burger), which crashes IsaacLab when spawned.
# The physics layer USD has a single articulation root and is safe for ArticulationCfg.
if os.path.exists(PHYSICS_USD):
    TURTLEBOT3_USD_PATH = PHYSICS_USD
elif os.path.exists(READY_USD):
    TURTLEBOT3_USD_PATH = READY_USD
else:
    TURTLEBOT3_USD_PATH = FIXED_USD

TURTLEBOT3_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TURTLEBOT3_USD_PATH,
        # NOTE:
        # The TurtleBot USD already carries articulation metadata.
        # Applying articulation/rigid-root modifiers at this spawn level can
        # create nested articulation roots (e.g. /Robot and /Robot/turtlebot3_burger).
        # Keep spawn minimal and configure articulation internals in the USD itself.
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            effort_limit=400.0,   # Let the physics engine use enough torque to move the 1kg robot
            velocity_limit=100.0, # The absolute safety limit for wheel spin
            stiffness=0.0,        # 0.0 means we are strictly using Velocity Control, not Position
            damping=10000.0,      # In PhysX, high damping is REQUIRED to lock in the target velocity
        ),
    },
)

@configclass
class NavSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    robot: ArticulationCfg = TURTLEBOT3_CFG