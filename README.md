# Isaac Lab Navigation Environment
## TurtleBot3 Waypoint Following — Starter Environment

---

## File Structure

```
isaac_nav_env/
├── scene_cfg.py   ← WHAT exists in the world (robot, ground, lights)
├── env_cfg.py     ← HOW the RL task works (obs, actions, rewards, resets)
├── mdp.py         ← The actual logic for each manager function
└── train.py       ← Training entry point (PPO via SKRL)
```

---

## How Isaac Lab Works (Mental Model)

Your original environment had everything in ONE class:  
`_get_obs()`, `_rewards()`, `_is_terminated()`, `reset()` etc.

Isaac Lab **splits these responsibilities** into separate "Manager" classes.
Each manager handles one concern:

| Your original code         | Isaac Lab manager         | Config class           |
|---------------------------|---------------------------|------------------------|
| `__init__` (spawn assets) | `SceneManager`            | `NavSceneCfg`          |
| `_get_obs()`              | `ObservationManager`      | `ObservationsCfg`      |
| `step(action)`            | `ActionManager`           | `ActionsCfg`           |
| `_rewards()`              | `RewardManager`           | `RewardsCfg`           |
| `_is_terminated()`        | `TerminationManager`      | `TerminationsCfg`      |
| `reset()`                 | `EventManager`            | `EventsCfg`            |

The managers call the **functions in mdp.py** at the right moment.
You never call these functions yourself — Isaac Lab orchestrates everything.

---

## Installation & First Run

### Step 1 — Install Isaac Lab
```bash
# Clone Isaac Lab alongside your Isaac Sim installation
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# The install script creates symlinks to your Isaac Sim Python
./isaaclab.sh --install
```

### Step 2 — Verify installation
```bash
# This should print Isaac Lab version info without errors
./isaaclab.sh -p -c "import isaaclab; print(isaaclab.__version__)"
```

### Step 3 — Run training
```bash
# From the IsaacLab directory:
./isaaclab.sh -p /path/to/isaac_nav_env/train.py --num_envs 16

# For headless (faster, no GUI):
./isaaclab.sh -p /path/to/isaac_nav_env/train.py --num_envs 64 --headless

# Monitor training:
tensorboard --logdir runs/
```

---

## Observation Space (Current: 4 dimensions)

| Index | Value                        | Range      |
|-------|------------------------------|------------|
| 0     | Distance to next waypoint    | [0, inf]   |
| 1     | Angle to next waypoint (rel) | [-π, π]    |
| 2     | Linear velocity (normalized) | [-1, 1]    |
| 3     | Angular velocity (normalized)| [-1, 1]    |

**Planned expansions (in order of priority):**
1. LiDAR scan: +24 rays → obs dim becomes 28
2. More waypoints lookahead: +N*2 → obs dim becomes 28 + N*2
3. Articulation angle γ: +1 → obs dim becomes 29 + N*2

---

## Reward Structure

| Term                  | Weight | Description                                    |
|-----------------------|--------|------------------------------------------------|
| `progress_reward`     | +5.0   | Dense: Δdistance to waypoint per step          |
| `waypoint_reached`    | +10.0  | Sparse: +1 each time a waypoint is cleared     |
| `out_of_bounds`       | -1.0   | Per step penalty if > 2m from waypoint         |
| `alive_penalty`       | -0.01  | Tiny per-step cost to encourage speed          |

---

## Planned Features (Thesis Roadmap)

### Phase 2 — Add LiDAR
In `scene_cfg.py`, add a `RayCasterCfg` sensor to the robot.
In `mdp.py`, add a `lidar_observation()` function.
In `env_cfg.py`, add it to `PolicyCfg`.

### Phase 3 — Add Trailer
1. Create a trailer USD with a hitch point
2. Add a `RevoluteJointCfg` between robot chassis and trailer drawbar
3. Add `articulation_angle_observation()` to read joint angle γ
4. The γ goes into the **critic** observations only during training (teacher)
5. During deployment (student), γ is NOT in the observation

### Phase 4 — Domain Randomization
In `env_cfg.py`, add to `EventsCfg`:
```python
randomize_trailer = EventTermCfg(
    func=mdp.randomize_trailer_params,
    mode="reset",
    params={
        "length_range": (0.3, 1.2),    # meters
        "mass_range": (5.0, 50.0),     # kg
    }
)
```

### Phase 5 — Teacher-Student Network
Use Isaac Lab's asymmetric actor-critic:
- Teacher (critic) sees: obs + γ + privileged info
- Student (policy) sees: obs only
- Distill teacher into student using DAgger or behavior cloning

---

## Common Errors & Fixes

**Error:** `ModuleNotFoundError: No module named 'isaaclab'`  
**Fix:** You're running with system Python. Always use `./isaaclab.sh -p`

**Error:** `URDF/USD not found on Nucleus`  
**Fix:** Run Isaac Sim once with GUI to download assets, or set up a local Nucleus server

**Error:** `CUDA out of memory`  
**Fix:** Reduce `--num_envs`. Start with 4 to verify the environment works.

**Error:** `joint 'wheel_left_joint' not found`  
**Fix:** Open the TurtleBot3 USD in Isaac Sim GUI and check exact joint names in the Stage panel.
