"""Script to generate a backflip motion trajectory file for motion tracking tasks."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

try:
  import mujoco.viewer
except ImportError:
  mujoco.viewer = None  # type: ignore

from mjlab import MJLAB_SRC_PATH


@dataclass
class GenerateBackflipTrajConfig:
  """Configuration for generating backflip motion trajectory.

  Args:
    robot: Robot name (e.g., "go1", "g1") or path to robot XML file.
      If robot name is provided, will look for XML in asset_zoo/robots/{robot}/xmls/{robot}.xml
    output: Output path for the motion.npz file. Defaults to "backflip_motion.npz"
    duration: Duration of the backflip trajectory in seconds. Defaults to 3.0
    timestep: Simulation timestep in seconds. Defaults to 0.005 (200 Hz)
    show_viewer: If True, opens the MuJoCo viewer and syncs every step.
  """

  robot: str
  output: str = "backflip_motion.npz"
  duration: float = 3.0
  timestep: float = 0.005
  show_viewer: bool = False


def find_robot_xml(robot: str) -> Path:
  """Find robot XML file by name or return path if already a path.

  Args:
    robot: Robot name (e.g., "go1") or path to XML file.

  Returns:
    Path to robot XML file.
  """
  robot_path = Path(robot)
  if robot_path.exists() and robot_path.suffix == ".xml":
    return robot_path

  # Try to find in asset zoo
  xml_path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / robot / "xmls" / f"{robot}.xml"
  if xml_path.exists():
    return xml_path

  raise FileNotFoundError(
    f"Robot XML not found. Tried:\n"
    f"  - {robot_path} (if provided as path)\n"
    f"  - {xml_path} (if provided as robot name)"
  )


def generate_reference(
  t: float, duration: float
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
  """User hook: define BOTH body pose and half-joint angles for the reference.

  Returns:
  - body position (x,y,z) in world frame
  - body orientation quaternion (w,x,y,z)
  - joint angles for ONE symmetric half of the robot (dict joint_name->rad).
    The generator will mirror:
      FR_* <-> FL_*
      RR_* <-> RL_*
    and will flip sign for mirrored `*_hip_joint` by default.
  """

  # -------------------------
  # Body pose trajectory
  # -------------------------
  # Simple 5-phase backflip reference:
  # 0) hold:    0.0s-0.5s   hold still (0 deg, z at ground)
  # 1) prep:    0.5s-0.6s   0  -> -20 deg
  # 2) preload: 0.6s-0.8s   -20 -> +15 deg
  # 3) flight:  0.8s-1.25s  +15 -> 330 deg (with ballistic z arc)
  # 4) land:    1.25s+      330 -> 360 deg

  # Ground height
  z_ground = 0.29
  # Takeoff velocity (for parabolic arc)
  v_z0 = 3.0  # m/s upward
  g = 9.81  # gravity

  fr_hip = 0.05
  fr_thigh = 1.0
  fr_calf = -1.9
  rr_hip = 0.05
  rr_thigh = 1.0
  rr_calf = -1.9


  z_1 = 0.2
  fr_thigh_1 = 1.5
  fr_calf_1 = -2.6
  rr_thigh_1 = 1.0
  rr_calf_1 = -1.9

  z_2 = 0.43
  fr_thigh_2 = 0.4
  fr_calf_2 = -0.7
  rr_thigh_2 = 0.7
  rr_calf_2 = -1.2


  # Phase 0: Hold (0.0 ≤ t < 0.5)
  if t < 0.5:
    theta_deg = 0.0
    x = 0.0
    z = z_ground


  # Phase 1: Preparation (0.5 ≤ t < 0.6)
  elif t < 0.6:
    alpha = (t - 0.5) / 0.1  # 0 to 1
    s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep for joints
    theta_deg = -20.0 * alpha  # 0° to -20°
    x = 0.0
    z = z_ground + (z_1 - z_ground) * s

    fr_thigh = fr_thigh + (fr_thigh_1 - fr_thigh) * s
    fr_calf = fr_calf + (fr_calf_1 - fr_calf) * s
    rr_thigh = rr_thigh + (rr_thigh_1 - rr_thigh) * s
    rr_calf = rr_calf + (rr_calf_1 - rr_calf) * s

  elif t < 0.8:
    # Pre-takeoff: θ from -20° to +15°, z constant
    alpha = (t - 0.6) / 0.2  # 0 to 1
    s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep for joints
    theta_deg = -20.0 + alpha * 35.0  # -20° to +15°
    x = 0.0
    z = z_1 + (z_2 - z_1) * s

    fr_thigh = fr_thigh_1 + (fr_thigh_2 - fr_thigh_1) * s
    fr_calf = fr_calf_1 + (fr_calf_2 - fr_calf_1) * s
    rr_thigh = rr_thigh_1 + (rr_thigh_2 - rr_thigh_1) * s
    rr_calf = rr_calf_1 + (rr_calf_2 - rr_calf_1) * s

  # Phase 3: Flight (0.8 ≤ t ≤ 1.25)
  elif t <= 1.25:
    # Pitch: +15° to 330° (linear)
    flight_time = t - 0.8  # Time since takeoff
    theta_deg = 15.0 + (flight_time / 0.45) * 315.0  # +15° to 330°
    # x: linear advance
    x = -0.5 * flight_time  # Forward motion
    # z: parabolic arc (ballistic)
    z = z_2 + v_z0 * flight_time - 0.5 * g * flight_time * flight_time


    fr_thigh = fr_thigh_2
    fr_calf = fr_calf_2
    rr_thigh = rr_thigh_2
    rr_calf = rr_calf_2
  # Phase 4: Landing (t > 1.25)
  else:
    # Pitch: 330° → 360° (complete rotation)
    landing_time = min((t - 1.25) / 0.3, 1.0)  # 0.3s landing phase
    # Faster joint convergence: blend joints over first half of landing window.
    land_blend = min(landing_time / 0.5, 1.0)
    s_land = land_blend * land_blend * (3.0 - 2.0 * land_blend)  # smoothstep
    theta_deg = 330.0 + landing_time * 30.0  # 330° to 360°
    # x: continue forward from flight end
    flight_time = 0.45  # Flight duration
    x = -0.5 * flight_time  # Forward distance from flight
    # z: return to ground
    z_apex = z_2 + v_z0 * flight_time - 0.5 * g * flight_time * flight_time
    z = z_ground + (z_apex - z_ground) * (1.0 - landing_time)

    # Fade joints back toward initial pose
    fr_thigh = fr_thigh_2 + (fr_thigh - fr_thigh_2) * s_land
    fr_calf = fr_calf_2 + (fr_calf - fr_calf_2) * s_land
    rr_thigh = rr_thigh_2 + (rr_thigh - rr_thigh_2) * s_land
    rr_calf = rr_calf_2 + (rr_calf - rr_calf_2) * s_land

  # Convert pitch angle to quaternion (rotation around Y-axis)
  theta_rad = np.deg2rad(theta_deg)
  quat = np.array([
    np.cos(theta_rad / 2.0),  # w
    0.0,                       # x
    -np.sin(theta_rad / 2.0),   # y (pitch axis)
    0.0                        # z
  ])

  pos = np.array([x, 0.0, z])

  # Assemble dict for ONE side; mirroring happens later.
  half_joints: dict[str, float] = {
    "FR_hip_joint": fr_hip,
    "FR_thigh_joint": fr_thigh,
    "FR_calf_joint": fr_calf,
    "RR_hip_joint": rr_hip,
    "RR_thigh_joint": rr_thigh,
    "RR_calf_joint": rr_calf,
  }

  return pos, quat, half_joints


def generate_backflip_joint_traj(
  model: mujoco.MjModel, T: int, timestep: float, duration: float
) -> np.ndarray:
  """Generate joint trajectory with neutral/default pose.

  Sets all joints to a neutral/standing pose. The body pose trajectory
  should be defined in `generate_body_pose_traj()` function.

  Args:
    model: MuJoCo model to get joint information from.
    T: Number of time steps.
    timestep: Simulation timestep in seconds.
    duration: Total duration of trajectory in seconds.

  Returns:
    Joint positions array of shape (T, n_joints) with neutral pose.
  """
  n_joints = model.nv - 6  # Exclude floating base
  q = np.zeros((T, n_joints))

  # Map joints to their positions in qpos array (excluding floating base)
  import re

  # Build mapping: joint name -> index in q array (not qpos)
  joint_name_to_qidx = {}
  qidx = 0
  for i in range(model.njnt):
    joint = model.joint(i)
    if joint.name != "floating_base_joint":
      joint_name_to_qidx[joint.name] = qidx
      qidx += 1

  # Helper to find joint indices in q array by pattern
  def find_joint_indices(pattern: str) -> list[int]:
    regex = re.compile(pattern)
    indices = []
    for name, idx in joint_name_to_qidx.items():
      if regex.match(name):
        indices.append(idx)
    return indices

  # Set neutral/default pose based on joint patterns
  # For Go1: thigh=0.9, calf=-1.8, hip varies
  # For other robots, use 0.0 as default
  hip_indices = find_joint_indices(r".*_hip_joint")
  thigh_indices = find_joint_indices(r".*_thigh_joint")
  calf_indices = find_joint_indices(r".*_calf_joint")

  # Set neutral pose (standing pose for quadrupeds)
  # All joints set to neutral - you can customize these values
  for i in hip_indices:
    q[:, i] = 0.0  # Neutral hip angle
  for i in thigh_indices:
    q[:, i] = 0.9  # Slightly bent thigh (standing pose)
  for i in calf_indices:
    q[:, i] = -1.8  # Extended calf (standing pose)

  # Optionally override joints from user-defined half-pose (mirrored to other side).
  mirror_prefix = {"FR_": "FL_", "FL_": "FR_", "RR_": "RL_", "RL_": "RR_"}

  def _mirror_name(name: str) -> str | None:
    for src, dst in mirror_prefix.items():
      if name.startswith(src):
        return dst + name[len(src) :]
    return None

  def _mirror_value(name: str, value: float) -> float:
    # Flip sign for hip abduction joints on the mirrored side.
    return -value if name.endswith("_hip_joint") else value

  for ti in range(T):
    _, _, half = generate_reference(ti * timestep, duration)
    for jname, jval in half.items():
      if jname in joint_name_to_qidx:
        q[ti, joint_name_to_qidx[jname]] = float(jval)
      mname = _mirror_name(jname)
      if mname is not None and mname in joint_name_to_qidx:
        q[ti, joint_name_to_qidx[mname]] = float(_mirror_value(jname, jval))

  return q


def generate_backflip_motion(
  robot_xml: Path,
  output_path: str,
  duration: float,
  timestep: float,
  show_viewer: bool,
) -> None:
  """Generate backflip motion.npz file from robot model.

  Args:
    robot_xml: Path to robot XML file.
    output_path: Output path for motion.npz file.
    duration: Duration of trajectory in seconds.
    timestep: Simulation timestep in seconds.
  """
  # Load robot model and add ground plane
  print(f"[INFO] Loading robot model from: {robot_xml}")
  spec = mujoco.MjSpec.from_file(str(robot_xml))

  # Checkerboard floor + simple directional light.
  from mjlab.utils import spec_config as spec_cfg

  spec_cfg.TextureCfg(
    name="groundplane",
    type="2d",
    builtin="checker",
    mark="edge",
    rgb1=(0.2, 0.3, 0.4),
    rgb2=(0.1, 0.2, 0.3),
    markrgb=(0.8, 0.8, 0.8),
    width=300,
    height=300,
  ).edit_spec(spec)
  spec_cfg.MaterialCfg(
    name="groundplane",
    texuniform=True,
    texrepeat=(4, 4),
    reflectance=0.2,
    texture="groundplane",
  ).edit_spec(spec)
  spec.worldbody.add_body(name="terrain").add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
    size=(0, 0, 0.01),
    material="groundplane",
  )
  spec_cfg.LightCfg(pos=(0, 0, 1.5), type="directional").edit_spec(spec)

  # Compile the model
  model = spec.compile()
  data = mujoco.MjData(model)

  nq = model.nq
  nv = model.nv
  n_bodies = model.nbody - 1  # skip world body (body 0)

  # Calculate number of time steps
  T = int(duration / timestep)
  print(f"[INFO] Generating trajectory: {T} steps ({duration}s at {1/timestep:.0f} Hz)")

  # Determine number of joints (assuming 6-DoF floating base)
  n_joints = nv - 6
  if n_joints <= 0:
    raise ValueError(f"Expected at least 6 DoF floating base, got nv={nv}")

  print(f"[INFO] Robot has {n_joints} joints, {n_bodies} bodies")

  # Generate joint trajectory (neutral pose)
  print("[INFO] Generating joint trajectory with neutral pose...")
  joint_pos = generate_backflip_joint_traj(model, T, timestep, duration)

  # Initialize arrays for body states
  # Velocities set to zero - only body positions/orientations matter
  joint_vel = np.zeros_like(joint_pos)
  body_pos_w = np.zeros((T, n_bodies, 3))
  body_quat_w = np.zeros((T, n_bodies, 4))
  body_lin_vel_w = np.zeros((T, n_bodies, 3))  # Set to zero
  body_ang_vel_w = np.zeros((T, n_bodies, 3))  # Set to zero

  # Compute forward kinematics for each time step (generate trajectory once)
  print("[INFO] Computing forward kinematics with user-defined body pose...")
  for t in range(T):
    # Get desired body pose from user-defined function
    current_time = t * timestep
    root_pos, root_quat, _ = generate_reference(current_time, duration)

    # Set root body pose (floating base: 3 pos + 4 quat)
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat  # (w, x, y, z)

    # Set joint positions to neutral
    data.qpos[7:] = joint_pos[t]
    data.qvel[:] = 0.0

    # Forward kinematics to compute body states
    mujoco.mj_forward(model, data)

    # Collect body states (skip body 0 = world)
    for b in range(1, model.nbody):
      idx = b - 1
      body_pos_w[t, idx] = data.xpos[b]
      body_quat_w[t, idx] = data.xquat[b]  # (w, x, y, z)
      body_lin_vel_w[t, idx] = data.cvel[b, :3]
      body_ang_vel_w[t, idx] = data.cvel[b, 3:]

  # Velocities are set to zero - only body positions/orientations are used
  # The RL agent will optimize joint angles to match the desired body poses
  print("[INFO] Velocities set to zero (only body poses matter)")

  # Save to npz file (trajectory generated once)
  print(f"[INFO] Saving motion file to: {output_path}")
  np.savez(
    output_path,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
  )
  print(f"[INFO] Successfully generated motion file: {output_path}")

  # If viewer enabled, loop the animation continuously
  if show_viewer:
    print("[INFO] Launching MuJoCo viewer in world frame...")
    print("[INFO] Animation will loop continuously. Close viewer window to exit.")
    viewer = mujoco.viewer.launch_passive(model, data)
    # Configure camera to be in world frame (not tracking robot)
    if viewer is not None:
      # Set frame to WORLD (not robot frame)
      viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD.value
      # Set camera type to FREE (not tracking any body)
      viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
      viewer.cam.trackbodyid = -1  # Don't track any body
      viewer.cam.fixedcamid = -1  # Not using fixed camera
      # Set camera to look at origin from a fixed position
      viewer.cam.lookat[:] = [0.0, 0.0, 0.5]  # Look at center, slightly above ground
      viewer.cam.distance = 4.0  # Distance from lookat point
      viewer.cam.azimuth = 45.0  # Angle around vertical axis
      viewer.cam.elevation = -20.0  # Angle above/below horizontal

    # Loop the animation continuously
    import time

    while viewer is not None and viewer.is_running():
      for t in range(T):
        if not viewer.is_running():
          break

        # Set base pose from trajectory (root body is index 0)
        data.qpos[:3] = body_pos_w[t, 0]
        data.qpos[3:7] = body_quat_w[t, 0]

        # Set joint positions from pre-computed trajectory
        data.qpos[7:] = joint_pos[t]
        # Set joint velocities (skip floating base velocities at indices 0-5)
        data.qvel[6:] = joint_vel[t]
        data.qvel[:6] = 0.0  # Floating base velocities remain zero

        # Forward kinematics
        mujoco.mj_forward(model, data)

        # Sync viewer
        viewer.sync()

        # Small delay to control playback speed (optional)
        time.sleep(timestep)
        # time.sleep(0.01)



def main():
  """Main entry point for the script."""
  cfg = tyro.cli(GenerateBackflipTrajConfig)
  robot_xml = find_robot_xml(cfg.robot)
  generate_backflip_motion(
    robot_xml,
    cfg.output,
    cfg.duration,
    cfg.timestep,
    cfg.show_viewer,
  )


if __name__ == "__main__":
  main()