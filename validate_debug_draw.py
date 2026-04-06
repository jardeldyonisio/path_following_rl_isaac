"""
Script to validate that the debug draw for waypoints and path is working in IsaacLab.
This script will:
- Import the environment and mdp modules
- Reset the environment
- Call the debug draw function directly (if possible)
- Step the environment and check for errors
- Print a message if the debug draw function is called

Usage:
  python validate_debug_draw.py
"""

import sys
import importlib

try:
    import mdp
except ImportError:
    print("Could not import mdp.py. Make sure you are running from the project root.")
    sys.exit(1)

# Try to find the debug draw function
if hasattr(mdp, '_draw_waypoints_debug'):
    print("Found _draw_waypoints_debug in mdp.py. Calling it with dummy data...")
    # Create dummy data for waypoints and robot pose
    waypoints = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    robot_pose = (0.5, 0.5)
    try:
        mdp._draw_waypoints_debug(waypoints, robot_pose)
        print("_draw_waypoints_debug executed without error.")
    except Exception as e:
        print(f"Error calling _draw_waypoints_debug: {e}")
else:
    print("_draw_waypoints_debug not found in mdp.py. Please check your implementation.")

print("\nNow attempting to step the environment (if available)...")

try:
    import train
    if hasattr(train, 'make_env'):
        env = train.make_env()
        obs = env.reset()
        print("Environment reset. Stepping once...")
        obs, reward, done, info = env.step(env.action_space.sample())
        print("Step completed. No errors detected.")
    else:
        print("No make_env() found in train.py. Skipping environment step.")
except Exception as e:
    print(f"Error during environment step: {e}")

print("\nValidation script finished.")
