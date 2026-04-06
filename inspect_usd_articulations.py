import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Inspect articulation roots in USD files")
parser.add_argument("--usd", type=str, required=True, help="USD file path")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Usd, UsdPhysics

stage = Usd.Stage.Open(args.usd)
if stage is None:
    print(f"[ERROR] Could not open USD: {args.usd}")
    simulation_app.close()
    raise SystemExit(1)

roots = []
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        roots.append(prim)

print(f"[INFO] USD: {args.usd}")
print(f"[INFO] defaultPrim: {stage.GetDefaultPrim().GetPath() if stage.GetDefaultPrim() else 'None'}")
print(f"[INFO] articulation_root_count: {len(roots)}")
for i, prim in enumerate(roots, start=1):
    parent = prim.GetParent()
    parent_path = parent.GetPath() if parent and parent.IsValid() else "None"
    print(f"  [{i}] {prim.GetPath()} (parent={parent_path})")

if len(roots) > 1:
    print("[INFO] nested_pairs:")
    root_paths = [str(p.GetPath()) for p in roots]
    for a in root_paths:
        for b in root_paths:
            if a != b and b.startswith(a + "/"):
                print(f"  - {a} -> {b}")

simulation_app.close()
