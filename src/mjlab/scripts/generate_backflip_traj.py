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


def generate_backflip_joint_traj(
  model: mujoco.MjModel, T: int, timestep: float
) -> np.ndarray:
  """Generate a backflip joint trajectory using keyframe animation.

  Creates a simple backflip motion with these phases:
  1. Crouch/preparation (0-20%)
  2. Jump/extend (20-30%)
  3. Tuck and rotate (30-70%)
  4. Extend for landing (70-85%)
  5. Landing/cushion (85-100%)

  Args:
    model: MuJoCo model to get joint information from.
    T: Number of time steps.
    timestep: Simulation timestep in seconds.

  Returns:
    Joint positions array of shape (T, n_joints).
  """
  n_joints = model.nv - 6  # Exclude floating base
  q = np.zeros((T, n_joints))

  # Map joints to their positions in qpos array (excluding floating base)
  # qpos has: [pos(3), quat(4), joint1, joint2, ...]
  # So joint positions start at index 7 in qpos, but at index 0 in our q array
  import re

  # Build mapping: joint name -> index in q array (not qpos)
  joint_name_to_qidx = {}
  qidx = 0
  for i in range(model.njnt):
    joint = model.joint(i)
    if joint.name != "floating_base_joint":
      # This joint is in our q array
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

  # Find joint indices for each leg type
  hip_indices = find_joint_indices(r".*_hip_joint")
  thigh_indices = find_joint_indices(r".*_thigh_joint")
  calf_indices = find_joint_indices(r".*_calf_joint")

  # Keyframe poses (as fractions of total time)
  keyframes = {
    # (time_fraction, hip_angle, thigh_angle, calf_angle)
    0.0: (0.0, 0.9, -1.8),  # Standing pose
    0.2: (0.0, 0.5, -1.2),  # Crouch
    0.3: (0.0, 1.2, -2.0),  # Jump/extend
    0.5: (0.0, 2.5, -0.5),  # Tuck (mid-flip)
    0.7: (0.0, 2.5, -0.5),  # Still tucked
    0.85: (0.0, 1.0, -1.8),  # Extend for landing
    1.0: (0.0, 0.9, -1.8),  # Landing/standing
  }

  # Create time array
  times = np.linspace(0, 1, T)

  # Interpolate each joint type
  keyframe_times = np.array(list(keyframes.keys()))
  keyframe_hips = np.array([keyframes[t][0] for t in keyframe_times])
  keyframe_thighs = np.array([keyframes[t][1] for t in keyframe_times])
  keyframe_calves = np.array([keyframes[t][2] for t in keyframe_times])

  # Use interpolation for smooth motion
  try:
    from scipy.interpolate import interp1d

    hip_interp = interp1d(
      keyframe_times, keyframe_hips, kind="cubic", fill_value="extrapolate"
    )
    thigh_interp = interp1d(
      keyframe_times, keyframe_thighs, kind="cubic", fill_value="extrapolate"
    )
    calf_interp = interp1d(
      keyframe_times, keyframe_calves, kind="cubic", fill_value="extrapolate"
    )
    # scipy interp1d (cubic interpolation)
    hip_traj = hip_interp(times)
    thigh_traj = thigh_interp(times)
    calf_traj = calf_interp(times)
  except ImportError:
    # Fallback to numpy interpolation if scipy not available
    hip_traj = np.interp(times, keyframe_times, keyframe_hips)
    thigh_traj = np.interp(times, keyframe_times, keyframe_thighs)
    calf_traj = np.interp(times, keyframe_times, keyframe_calves)

  # Assign to joint arrays
  for i in hip_indices:
    q[:, i] = hip_traj
  for i in thigh_indices:
    q[:, i] = thigh_traj
  for i in calf_indices:
    q[:, i] = calf_traj

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

#   # Add ground plane to worldbody
#   if show_viewer:
#     print("[INFO] Adding ground plane to model...")
#     # Add a simple ground plane geom
#     spec.worldbody.add_geom(
#       name="ground",
#       type=mujoco.mjtGeom.mjGEOM_PLANE,
#       size=(10.0, 10.0, 0.01),  # Large plane
#       rgba=(0.2, 0.3, 0.4, 1.0),  # Gray-blue color
#     )
#     # Add some lighting
#     spec.worldbody.add_light(
#       pos=(0, 0, 3),
#       dir=(0, 0, -1),
#       diffuse=(0.7, 0.7, 0.7),
#     )

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

  # Generate joint trajectory
  print("[INFO] Generating backflip joint trajectory...")
  joint_pos = generate_backflip_joint_traj(model, T, timestep)

  # Initialize arrays for body states
  joint_vel = np.zeros_like(joint_pos)
  body_pos_w = np.zeros((T, n_bodies, 3))
  body_quat_w = np.zeros((T, n_bodies, 4))
  body_lin_vel_w = np.zeros((T, n_bodies, 3))
  body_ang_vel_w = np.zeros((T, n_bodies, 3))

  # Compute forward kinematics for each time step (generate trajectory once)
  print("[INFO] Computing forward kinematics and generating trajectory...")
  for t in range(T):
    # Set base pose; start at origin, neutral orientation
    data.qpos[:3] = 0.0
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

    # Set joint positions
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

  # Compute joint velocities from finite differences
  print("[INFO] Computing joint velocities...")
  joint_vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / timestep
  joint_vel[0] = joint_vel[1]

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

        # Set base pose
        data.qpos[:3] = 0.0
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])

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