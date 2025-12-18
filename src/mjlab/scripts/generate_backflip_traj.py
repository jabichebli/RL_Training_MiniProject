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
    include_pauses: If True, includes pauses between phases in both the motion file and viewer.
  """

  robot: str
  output: str = "backflip_motion.npz"
  duration: float = 3.0
  timestep: float = 0.005
  show_viewer: bool = False
  include_pauses: bool = False


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
  t: float, duration: float, use_pauses: bool = True
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
  """User hook: define BOTH body pose and half-joint angles for the reference.

  Args:
    t: Current time in seconds
    duration: Total duration of the trajectory
    use_pauses: If True, includes pauses between phases. If False, continuous transitions.

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
  # ============================================================================
  # TRAJECTORY PARAMETERS - Modify these to tune the backflip
  # ============================================================================

  # --- Flight dynamics ---
  v_z0 = 3.0             # Vertical takeoff velocity (m/s)
  g = 9.81               # Gravity (m/s²)
  
  # --- Pause duration between phases (seconds) ---
  pause_duration = 0.3 if use_pauses else 0.0  # How long to hold at the end of each phase
  
  # --- Phase 0: Initial stance (neutral) ---
  x_0 = 0.0              # Phase 0: Forward position
  z_0 = 0.29             # Phase 0: Ground level / landing height
  theta_hold = 0.0       # Phase 0: Initial hold
  t_0_transition = 0.5   # Phase 0: Transition end (before pause)
  t_0_pause_end = t_0_transition + pause_duration  # Phase 0: End (including pause)
  fr_hip = 0.05
  fr_thigh = 1.0
  fr_calf = -1.9
  rr_hip = 0.05
  rr_thigh = 1.0
  rr_calf = -1.9
  

  # --- Phase 1: Preparation joint angles ---
  x_1 = -0.05              # Phase 1: Forward position
  z_1 = 0.29             # Phase 1: Vertical position
  theta_prep = 10.0      # Phase 1: Preparation (initial crouch)
  t_1_transition = t_0_pause_end + 0.2  # Phase 1: Transition end (before pause)
  t_1_pause_end = t_1_transition + pause_duration  # Phase 1: End (including pause)
  fr_thigh_1 = 0.7
  fr_calf_1 = -1.4
  rr_thigh_1 = 0.3
  rr_calf_1 = -1.8
  

  # --- Phase 2: Loading joint angles ---
  x_2 = -0.2              # Phase 2: Forward position
  z_2 = 0.43             # Phase 2: Vertical position
  theta_loading = 65.0   # Phase 2: Loading (deep crouch)
  t_2_transition = t_1_pause_end + 0.1  # Phase 2: Transition end (before pause)
  t_2_pause_end = t_2_transition + pause_duration  # Phase 2: End (including pause)
  fr_thigh_2 = 1.3
  fr_calf_2 = -1.2
  rr_thigh_2 = 1
  rr_calf_2 = -1.2
  

  # --- Phase 3: Squat joint angles ---
  x_squat = -0.3          # Phase 3: Forward position
  z_squat = 0.48          # Phase 3: Vertical position
  theta_squat = 85.0      # Phase 3: Squat (deepest position)
  t_3_transition = t_2_pause_end + 0.05  # Phase 3: Transition end (before pause)
  t_3_pause_end = t_3_transition + pause_duration  # Phase 3: End (including pause)
  fr_thigh_squat = 1.3
  fr_calf_squat = -1.2
  rr_thigh_squat = 1.4
  rr_calf_squat = -0.8
  

  # --- Phase 4: Takeoff joint angles ---
  x_4 = -0.4              # Phase 4: Forward position
  z_4 = 0.55             # Phase 4: Vertical position
  theta_takeoff = 110.0   # Phase 4: Takeoff (push off)
  t_4_transition = t_3_pause_end + 0.05  # Phase 4: Transition end (before pause)
  t_4_pause_end = t_4_transition + pause_duration  # Phase 4: End (including pause)
  fr_thigh_4 = 0.6
  fr_calf_4 = -1.6
  rr_thigh_4 = 1.8
  rr_calf_4 = -0.5

  # --- Phase 5: Flight parameters ---
  t_flight_end = 1.25    # Phase 5: End of flight
  theta_flight_end = 330.0  # Phase 5: End of flight (nearly upright after rotation)
  
  # --- Phase 6: Landing parameters ---
  z_ground = 0.29        # Landing ground height (same as z_0)
  t_landing_duration = 0.3  # Phase 6: Landing duration
  theta_final = 360.0    # Phase 6: Landing (complete rotation)
  
  # ============================================================================


  # Phase 0: Hold (transition)
  if t < t_0_transition:
    theta_deg = theta_hold
    x = x_0
    z = z_0

  # Phase 0: Pause
  elif t < t_0_pause_end:
    theta_deg = theta_hold
    x = x_0
    z = z_0

  # Phase 1: Preparation (transition)
  elif t < t_1_transition:
    phase_duration = t_1_transition - t_0_pause_end
    alpha = (t - t_0_pause_end) / phase_duration
    s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep for joints
    theta_deg = theta_hold + alpha * (theta_prep - theta_hold)
    x = x_0 + (x_1 - x_0) * s
    z = z_0 + (z_1 - z_0) * s

    fr_thigh = fr_thigh + (fr_thigh_1 - fr_thigh) * s
    fr_calf = fr_calf + (fr_calf_1 - fr_calf) * s
    rr_thigh = rr_thigh + (rr_thigh_1 - rr_thigh) * s
    rr_calf = rr_calf + (rr_calf_1 - rr_calf) * s

  # Phase 1: Pause
  elif t < t_1_pause_end:
    theta_deg = theta_prep
    x = x_1
    z = z_1
    fr_thigh = fr_thigh_1
    fr_calf = fr_calf_1
    rr_thigh = rr_thigh_1
    rr_calf = rr_calf_1

  # Phase 2: Loading (transition)
  elif t < t_2_transition:
    phase_duration = t_2_transition - t_1_pause_end
    alpha = (t - t_1_pause_end) / phase_duration
    s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep for joints
    theta_deg = theta_prep + alpha * (theta_loading - theta_prep)
    x = x_1 + (x_2 - x_1) * s
    z = z_1 + (z_2 - z_1) * s

    fr_thigh = fr_thigh_1 + (fr_thigh_2 - fr_thigh_1) * s
    fr_calf = fr_calf_1 + (fr_calf_2 - fr_calf_1) * s
    rr_thigh = rr_thigh_1 + (rr_thigh_2 - rr_thigh_1) * s
    rr_calf = rr_calf_1 + (rr_calf_2 - rr_calf_1) * s

  # Phase 2: Pause
  elif t < t_2_pause_end:
    theta_deg = theta_loading
    x = x_2
    z = z_2
    fr_thigh = fr_thigh_2
    fr_calf = fr_calf_2
    rr_thigh = rr_thigh_2
    rr_calf = rr_calf_2

  # Phase 3: Squat (transition)
  elif t < t_3_transition:
    phase_duration = t_3_transition - t_2_pause_end
    alpha = (t - t_2_pause_end) / phase_duration
    s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep for joints
    theta_deg = theta_loading + alpha * (theta_squat - theta_loading)
    x = x_2 + (x_squat - x_2) * s
    z = z_2 + (z_squat - z_2) * s

    fr_thigh = fr_thigh_2 + (fr_thigh_squat - fr_thigh_2) * s
    fr_calf = fr_calf_2 + (fr_calf_squat - fr_calf_2) * s
    rr_thigh = rr_thigh_2 + (rr_thigh_squat - rr_thigh_2) * s
    rr_calf = rr_calf_2 + (rr_calf_squat - rr_calf_2) * s

  # Phase 3: Pause
  elif t < t_3_pause_end:
    theta_deg = theta_squat
    x = x_squat
    z = z_squat
    fr_thigh = fr_thigh_squat
    fr_calf = fr_calf_squat
    rr_thigh = rr_thigh_squat
    rr_calf = rr_calf_squat

  # Phase 4: Takeoff (transition)
  elif t < t_4_transition:
    phase_duration = t_4_transition - t_3_pause_end
    alpha = (t - t_3_pause_end) / phase_duration
    s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep for joints
    theta_deg = theta_squat + alpha * (theta_takeoff - theta_squat)
    x = x_squat + (x_4 - x_squat) * s
    z = z_squat + (z_4 - z_squat) * s

    fr_thigh = fr_thigh_squat + (fr_thigh_4 - fr_thigh_squat) * s
    fr_calf = fr_calf_squat + (fr_calf_4 - fr_calf_squat) * s
    rr_thigh = rr_thigh_squat + (rr_thigh_4 - rr_thigh_squat) * s
    rr_calf = rr_calf_squat + (rr_calf_4 - rr_calf_squat) * s

  # Phase 4: Pause and hold position after takeoff
  else:
    theta_deg = theta_takeoff
    x = x_4
    z = z_4

    fr_thigh = fr_thigh_4
    fr_calf = fr_calf_4
    rr_thigh = rr_thigh_4
    rr_calf = rr_calf_4

  # TODO: Add Phase 5 (Flight) and Phase 6 (Landing) here when ready
  # Phase 5: Flight
  # elif t <= t_flight_end:
  #   flight_time = t - t_takeoff_end
  #   flight_duration = t_flight_end - t_takeoff_end
  #   theta_deg = theta_takeoff + (flight_time / flight_duration) * (theta_flight_end - theta_takeoff)
  #   x = -0.5 * flight_time
  #   z = z_4 + v_z0 * flight_time - 0.5 * g * flight_time * flight_time
  #   fr_thigh = fr_thigh_4
  #   fr_calf = fr_calf_4
  #   rr_thigh = rr_thigh_4
  #   rr_calf = rr_calf_4
  #
  # Phase 6: Landing
  # else:
  #   landing_time = min((t - t_flight_end) / t_landing_duration, 1.0)
  #   land_blend = min(landing_time / 0.5, 1.0)
  #   s_land = land_blend * land_blend * (3.0 - 2.0 * land_blend)
  #   theta_deg = theta_flight_end + landing_time * (theta_final - theta_flight_end)
  #   flight_duration = t_flight_end - t_takeoff_end
  #   x = -0.5 * flight_duration
  #   z_apex = z_4 + v_z0 * flight_duration - 0.5 * g * flight_duration * flight_duration
  #   z = z_ground + (z_apex - z_ground) * (1.0 - landing_time)
  #   fr_thigh = fr_thigh_4 + (fr_thigh - fr_thigh_4) * s_land
  #   fr_calf = fr_calf_4 + (fr_calf - fr_calf_4) * s_land
  #   rr_thigh = rr_thigh_4 + (rr_thigh - rr_thigh_4) * s_land
  #   rr_calf = rr_calf_4 + (rr_calf - rr_calf_4) * s_land

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
  model: mujoco.MjModel, T: int, timestep: float, duration: float, include_pauses: bool
) -> np.ndarray:
  """Generate joint trajectory with neutral/default pose.

  Sets all joints to a neutral/standing pose. The body pose trajectory
  should be defined in `generate_body_pose_traj()` function.

  Args:
    model: MuJoCo model to get joint information from.
    T: Number of time steps.
    timestep: Simulation timestep in seconds.
    duration: Total duration of trajectory in seconds.
    include_pauses: If True, includes pauses between phases.

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
    _, _, half = generate_reference(ti * timestep, duration, use_pauses=include_pauses)
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
  include_pauses: bool,
) -> None:
  """Generate backflip motion.npz file from robot model.

  Args:
    robot_xml: Path to robot XML file.
    output_path: Output path for motion.npz file.
    duration: Duration of trajectory in seconds.
    timestep: Simulation timestep in seconds.
    show_viewer: If True, opens the MuJoCo viewer.
    include_pauses: If True, includes pauses in the generated motion file.
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
  print(f"[INFO] Generating joint trajectory (pauses={'enabled' if include_pauses else 'disabled'})...")
  joint_pos = generate_backflip_joint_traj(model, T, timestep, duration, include_pauses)

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
    root_pos, root_quat, _ = generate_reference(current_time, duration, use_pauses=include_pauses)

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
    print(f"[INFO] Viewer showing trajectory {'WITH pauses' if include_pauses else 'WITHOUT pauses'}.")
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

        # Compute poses on-the-fly matching the motion file setting
        current_time = t * timestep
        root_pos, root_quat, half_joints = generate_reference(current_time, duration, use_pauses=include_pauses)

        # Set base pose from on-the-fly computation
        data.qpos[:3] = root_pos
        data.qpos[3:7] = root_quat

        # Set joint positions from on-the-fly computation
        # Note: We're using the pre-computed joint_pos which includes mirroring
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
    cfg.include_pauses,
  )


if __name__ == "__main__":
  main()
  main()