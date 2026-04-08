'''
@brief Defines what exists in the scene, including the robot, the ground plane, and the light source.
'''
import os
import isaaclab.sim as sim_utils

from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
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

    # 2D LiDAR (defaults to 180°, can be changed in env_cfg.__post_init__)
    lidar: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint/base_link/base_scan",
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-90.0, 90.0),
            horizontal_res=7.5,
        ),
        max_distance=2.5,
        drift_range=(0.0, 0.0),
        debug_vis=False,
        mesh_prim_paths=["/World"],
    )

    # Obstacles (will be randomized on reset in mdp.reset_obstacles)
    obstacle_0: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_0",
        spawn=sim_utils.CylinderCfg(
            radius=0.25,
            height=0.4,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(50.0, 50.0, 0.2)),
    )

    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_1",
        spawn=sim_utils.CylinderCfg(
            radius=0.22,
            height=0.4,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(52.0, 50.0, 0.2)),
    )

    obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_2",
        spawn=sim_utils.CylinderCfg(
            radius=0.20,
            height=0.4,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(54.0, 50.0, 0.2)),
    )

    obstacle_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_3",
        spawn=sim_utils.CylinderCfg(
            radius=0.18,
            height=0.4,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(56.0, 50.0, 0.2)),
    )

    obstacle_4: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_4",
        spawn=sim_utils.CylinderCfg(
            radius=0.16,
            height=0.4,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(58.0, 50.0, 0.2)),
    )