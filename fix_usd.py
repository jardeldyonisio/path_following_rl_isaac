"""
fix_usd.py
----------
Fixes the TurtleBot3 USD exported by Isaac Sim's GUI importer by:
  1. Adding UsdPhysics.ArticulationRootAPI to the root prim
  2. Removing the instanceable flag that breaks visual rendering
  3. Saving a clean fixed USD ready for Isaac Lab

Run ONCE before training:
    cd ~/IsaacLab
    ./isaaclab.sh -p /home/lognav/Jardel/path_following_rl_isaac/fix_usd.py
"""

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# --- Safe to import now ---
from pxr import Usd, UsdPhysics, UsdGeom, PhysxSchema, Sdf
import omni.usd

INPUT_USD  = "/home/lognav/Jardel/path_following_rl_isaac/assets/turtlebot3_burger_fixed/turtlebot3_burger_fixed.usd"
OUTPUT_USD = "/home/lognav/Jardel/path_following_rl_isaac/assets/turtlebot3_burger_ready.usd"

print(f"\n[INFO] Loading USD: {INPUT_USD}")

# Open the stage
stage = Usd.Stage.Open(INPUT_USD)
if not stage:
    print("[ERROR] Could not open USD file.")
    simulation_app.close()
    exit(1)

# ---------------------------------------------------------------------------
# Step 1 — Find the root prim
# ---------------------------------------------------------------------------
root_prim = stage.GetDefaultPrim()
if not root_prim:
    # Try common names
    for name in ["turtlebot3_burger", "turtlebot3_burger_fixed", "base_footprint", "Robot"]:
        prim = stage.GetPrimAtPath(f"/{name}")
        if prim.IsValid():
            root_prim = prim
            break

print(f"[INFO] Root prim: {root_prim.GetPath()}")
print(f"[INFO] Root prim type: {root_prim.GetTypeName()}")

# ---------------------------------------------------------------------------
# Step 2 — Apply ArticulationRootAPI to the root prim
# ---------------------------------------------------------------------------
if not root_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    print("[INFO] Applied ArticulationRootAPI to root prim.")
else:
    print("[INFO] ArticulationRootAPI already present.")

# Apply PhysxArticulationAPI for solver settings
if not root_prim.HasAPI(PhysxSchema.PhysxArticulationAPI):
    physx_art = PhysxSchema.PhysxArticulationAPI.Apply(root_prim)
    physx_art.CreateEnabledSelfCollisionsAttr(False)
    physx_art.CreateSolverPositionIterationCountAttr(4)
    physx_art.CreateSolverVelocityIterationCountAttr(0)
    print("[INFO] Applied PhysxArticulationAPI to root prim.")

# ---------------------------------------------------------------------------
# Step 3 — Remove instanceable flag from ALL prims
# Instanceable prims break visual rendering when cloned across envs.
# ---------------------------------------------------------------------------
instanceable_count = 0
for prim in stage.Traverse():
    if prim.IsInstanceable():
        prim.SetInstanceable(False)
        instanceable_count += 1

print(f"[INFO] Removed instanceable flag from {instanceable_count} prims.")

# ---------------------------------------------------------------------------
# Step 4 — Ensure RigidBodyAPI is on the base link
# ---------------------------------------------------------------------------
for prim in stage.Traverse():
    name = prim.GetName().lower()
    if "base_footprint" in name or "base_link" in name:
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI.Apply(prim)
            print(f"[INFO] Applied RigidBodyAPI to: {prim.GetPath()}")

# ---------------------------------------------------------------------------
# Step 5 — Print all joints found (for verifying wheel joint names)
# ---------------------------------------------------------------------------
print("\n[INFO] Joints found in USD:")
for prim in stage.Traverse():
    if prim.GetTypeName() in ["PhysicsRevoluteJoint", "PhysicsJoint", "RevoluteJoint"]:
        print(f"  - {prim.GetName()}  ({prim.GetPath()})")

# ---------------------------------------------------------------------------
# Step 6 — Export clean USD
# ---------------------------------------------------------------------------
stage.Export(OUTPUT_USD)
print(f"\n[SUCCESS] Fixed USD saved to: {OUTPUT_USD}")

import os
size_kb = os.path.getsize(OUTPUT_USD) / 1024
print(f"[INFO] File size: {size_kb:.1f} KB")

simulation_app.close()