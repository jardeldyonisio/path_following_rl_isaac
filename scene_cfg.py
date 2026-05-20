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

PROJECT_DIR = "/home/lognav/Jardel/path_following_rl_isaac"
ROBOTS_DIR = os.path.join(PROJECT_DIR, "robots")

# Robot paths
TURTLEBOT3_USD_PATH = os.path.join(ROBOTS_DIR, "turtlebot3_burger_fixed", "configuration", "turtlebot3_burger_fixed_physics.usd")
GLR_USD_PATH = os.path.join(ROBOTS_DIR, "glr", "configuration", "glr_physics.usd")
GLR_TUGGER_USD_PATH = os.path.join(ROBOTS_DIR, "glr_tugger", "configuration", "glr_tugger_physics.usd")

if not os.path.exists(TURTLEBOT3_USD_PATH):
    raise FileNotFoundError(f"TurtleBot3 robot USD not found: {TURTLEBOT3_USD_PATH}")

if not os.path.exists(GLR_USD_PATH):
    raise FileNotFoundError(f"GLR robot USD not found: {GLR_USD_PATH}")

if not os.path.exists(GLR_TUGGER_USD_PATH):
    raise FileNotFoundError(f"GLR Tugger robot USD not found: {GLR_TUGGER_USD_PATH}")

ROBOT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TURTLEBOT3_USD_PATH,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.0),
        # rot=(0.7071068, 0.0, 0.0, 0.7071068), # GLR
        rot=(0.0, 0.0, 0.0, 0.0), # TURTLEBOT3
        joint_pos={".*": 0.0},
    ),
    actuators={
        "drive_wheels": ImplicitActuatorCfg(
            # joint_names_expr=["left_wheel_joint", "right_wheel_joint"], # GLR
            joint_names_expr=["wheel_left_joint", "wheel_right_joint"], # TURTLEBOT3
            effort_limit=400.0,
            velocity_limit=15.0, # rad/s, not m/s, since the action will be wheel angular velocity
            stiffness=0.0,
            damping=10000.0, 
        ),
        # "passive_wheels": ImplicitActuatorCfg(
        #     # Pega todas as rodas direcionais/rolagem do reboque ("eixo...") e casters do rebocador
        #     joint_names_expr=["eixo.*", ".*caster.*"], 
        #     effort_limit=0.0,
        #     velocity_limit=0.0,
        #     stiffness=0.0, # Zero rigidez = solto
        #     damping=0.0,   # Zero resistência = rola livremente
        # ),
        # "cambao_joints": ImplicitActuatorCfg(
        #     # Pega as duas juntas do cambão
        #     joint_names_expr=[".*cambao.*"],
        #     effort_limit=0.0,
        #     velocity_limit=0.0,
        #     stiffness=0.0,
        #     damping=1000.0, # Damping para estabilizar a articulação e não dobrar como papel
        # ),
    },
)

@configclass
class NavSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=2.0,
            )
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )

    robot: ArticulationCfg = ROBOT_CFG

    # 2D LiDAR (defaults to 180°, can be changed in env_cfg.__post_init__)
    lidar: RayCasterCfg = RayCasterCfg(
        # prim_path="{ENV_REGEX_NS}/Robot/base_link/chassibigga/lidar_link", # GLR
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint/base_link/base_scan", # TURTLEBOT
        update_period=0.0,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0), 
            rot=(0.7071068, 0.0, 0.0, 0.7071068)
        ),
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