"""Generate backflip motion trajectory for tracking tasks."""

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro

try:
    import mujoco.viewer
except ImportError:
    mujoco.viewer = None

from mjlab import MJLAB_SRC_PATH


@dataclass
class Config:
    robot: str
    output: str = "backflip.npz"
    duration: float = 5.0
    timestep: float = 0.02
    show_viewer: bool = False


def find_robot_xml(robot: str) -> Path:
    """Find robot XML by name or path."""
    path = Path(robot)
    if path.exists() and path.suffix == ".xml":
        return path
    xml = MJLAB_SRC_PATH / "asset_zoo" / "robots" / robot / "xmls" / f"{robot}.xml"
    if xml.exists():
        return xml
    raise FileNotFoundError(f"Robot XML not found: {robot}")


def backflip_trajectory(t: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Define backflip trajectory: body pose + joint angles."""
    
    # ============================================================================
    # TRAJECTORY PARAMETERS - Modify these to tune the backflip
    # ============================================================================
    
    # --- Flight dynamics ---
    v_z0 = 3.0    # Vertical takeoff velocity (m/s)
    g = 9.81      # Gravity (m/s²)
    
    # --- Hip angles (constant throughout) ---
    hip = 0.05
    
    # --- Phase 0: Initial stance ---
    x_0 = 0.0
    z_0 = 0.29
    theta_0 = 0.0
    t_0 = 0.5
    fr_thigh_0 = 1.0
    fr_calf_0 = -1.9
    rr_thigh_0 = 1.0
    rr_calf_0 = -1.9
    
    # --- Phase 1: Crouch ---
    x_1 = -0.05
    z_1 = 0.31
    theta_1 = 20.0
    t_1 = 0.8
    fr_thigh_1 = 0.4
    fr_calf_1 = -0.7
    rr_thigh_1 = 0.3
    rr_calf_1 = -1.8
    
    # --- Phase 2: Loading ---
    x_2 = -0.2
    z_2 = 0.43
    theta_2 = 65.0
    t_2 = 0.9
    fr_thigh_2 = 1.3
    fr_calf_2 = -1.2
    rr_thigh_2 = 1.0
    rr_calf_2 = -1.2
    
    # --- Phase 3: Takeoff ---
    x_3 = -0.4
    z_3 = 0.55
    theta_3 = 110.0
    t_3 = 0.95
    fr_thigh_3 = 0.6
    fr_calf_3 = -1.6
    rr_thigh_3 = 2.4
    rr_calf_3 = -0.8
    
    # --- Phase 4: Flight ---
    theta_4 = 330.0
    t_4 = t_3 + 0.45
    
    # --- Phase 5: Landing ---
    x_5 = -0.7
    z_5 = 0.29
    theta_5 = 360.0
    t_5 = t_4 + 0.3
    
    # ============================================================================
    
    # Phase 0: Hold
    if t < t_0:
        x, z, theta = x_0, z_0, theta_0
        fr_thigh, fr_calf = fr_thigh_0, fr_calf_0
        rr_thigh, rr_calf = rr_thigh_0, rr_calf_0
    
    # Phase 1: Crouch
    elif t < t_1:
        s = (t - t_0) / (t_1 - t_0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep
        x = x_0 + (x_1 - x_0) * s
        z = z_0 + (z_1 - z_0) * s
        theta = theta_0 + (theta_1 - theta_0) * s
        fr_thigh = fr_thigh_0 + (fr_thigh_1 - fr_thigh_0) * s
        fr_calf = fr_calf_0 + (fr_calf_1 - fr_calf_0) * s
        rr_thigh = rr_thigh_0 + (rr_thigh_1 - rr_thigh_0) * s
        rr_calf = rr_calf_0 + (rr_calf_1 - rr_calf_0) * s
    
    # Phase 2: Loading
    elif t < t_2:
        s = (t - t_1) / (t_2 - t_1)
        s = s * s * (3.0 - 2.0 * s)
        x = x_1 + (x_2 - x_1) * s
        z = z_1 + (z_2 - z_1) * s
        theta = theta_1 + (theta_2 - theta_1) * s
        fr_thigh = fr_thigh_1 + (fr_thigh_2 - fr_thigh_1) * s
        fr_calf = fr_calf_1 + (fr_calf_2 - fr_calf_1) * s
        rr_thigh = rr_thigh_1 + (rr_thigh_2 - rr_thigh_1) * s
        rr_calf = rr_calf_1 + (rr_calf_2 - rr_calf_1) * s
    
    # Phase 3: Takeoff
    elif t < t_3:
        s = (t - t_2) / (t_3 - t_2)
        s = s * s * (3.0 - 2.0 * s)
        x = x_2 + (x_3 - x_2) * s
        z = z_2 + (z_3 - z_2) * s
        theta = theta_2 + (theta_3 - theta_2) * s
        fr_thigh = fr_thigh_2 + (fr_thigh_3 - fr_thigh_2) * s
        fr_calf = fr_calf_2 + (fr_calf_3 - fr_calf_2) * s
        rr_thigh = rr_thigh_2 + (rr_thigh_3 - rr_thigh_2) * s
        rr_calf = rr_calf_2 + (rr_calf_3 - rr_calf_2) * s
    
    # Phase 4: Flight (ballistic)
    elif t < t_4:
        dt = t - t_3
        alpha = dt / (t_4 - t_3)
        x = x_3 - 0.5 * dt
        z = z_3 + v_z0 * dt - 0.5 * g * dt * dt
        theta = theta_3 + alpha * (theta_4 - theta_3)
        fr_thigh = fr_thigh_3
        fr_calf = fr_calf_3
        rr_thigh = rr_thigh_3
        rr_calf = rr_calf_3
    
    # Phase 5: Landing
    elif t < t_5:
        s = (t - t_4) / (t_5 - t_4)
        s = s * s * (3.0 - 2.0 * s)
        x_4_end = x_3 - 0.5 * (t_4 - t_3)
        z_4_end = z_3 + v_z0 * (t_4 - t_3) - 0.5 * g * (t_4 - t_3) ** 2
        x = x_4_end + (x_5 - x_4_end) * s
        z = z_4_end + (z_5 - z_4_end) * s
        theta = theta_4 + (theta_5 - theta_4) * s
        fr_thigh = fr_thigh_3 + (fr_thigh_0 - fr_thigh_3) * s
        fr_calf = fr_calf_3 + (fr_calf_0 - fr_calf_3) * s
        rr_thigh = rr_thigh_3 + (rr_thigh_0 - rr_thigh_3) * s
        rr_calf = rr_calf_3 + (rr_calf_0 - rr_calf_3) * s
    
    # Hold final
    else:
        x, z, theta = x_5, z_5, theta_5
        fr_thigh, fr_calf = fr_thigh_0, fr_calf_0
        rr_thigh, rr_calf = rr_thigh_0, rr_calf_0
    
    # Convert to pose
    theta_rad = np.deg2rad(theta)
    pos = np.array([x, 0.0, z])
    quat = np.array([np.cos(theta_rad / 2), 0.0, -np.sin(theta_rad / 2), 0.0])
    joints = {
        "FR_hip_joint": hip,
        "FR_thigh_joint": fr_thigh,
        "FR_calf_joint": fr_calf,
        "RR_hip_joint": hip,
        "RR_thigh_joint": rr_thigh,
        "RR_calf_joint": rr_calf,
    }
    return pos, quat, joints


def main():
    cfg = tyro.cli(Config)
    xml = find_robot_xml(cfg.robot)
    spec = mujoco.MjSpec.from_file(str(xml))
    
    # Add ground
    spec.worldbody.add_body(name="terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE, size=(0, 0, 0.01)
    )
    
    model = spec.compile()
    data = mujoco.MjData(model)
    
    T = int(cfg.duration / cfg.timestep)
    n_joints = model.nv - 6
    n_bodies = model.nbody - 1
    
    # Generate trajectory
    joint_pos = np.zeros((T, n_joints))
    body_pos_w = np.zeros((T, n_bodies, 3))
    body_quat_w = np.zeros((T, n_bodies, 4))
    
    # Joint mapping
    joint_names = [model.joint(i).name for i in range(model.njnt) 
                   if model.joint(i).name != "floating_base_joint"]
    joint_idx = {name: i for i, name in enumerate(joint_names)}
    
    for t in range(T):
        pos, quat, half_joints = backflip_trajectory(t * cfg.timestep)
        
        # Set robot state
        data.qpos[:3] = pos
        data.qpos[3:7] = quat
        
        # Mirror joints FR<->FL, RR<->RL
        for name, val in half_joints.items():
            if name in joint_idx:
                joint_pos[t, joint_idx[name]] = val
            mirror = name.replace("FR_", "FL_").replace("RR_", "RL_") if "FR_" in name or "RR_" in name else None
            if mirror and mirror in joint_idx:
                joint_pos[t, joint_idx[mirror]] = -val if "_hip_joint" in name else val
        
        data.qpos[7:] = joint_pos[t]
        mujoco.mj_forward(model, data)
        
        # Record body states
        for b in range(n_bodies):
            body_pos_w[t, b] = data.xpos[b + 1]
            body_quat_w[t, b] = data.xquat[b + 1]
    
    # Save
    np.savez(
        cfg.output,
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=np.zeros((T, n_bodies, 3)),
        body_ang_vel_w=np.zeros((T, n_bodies, 3)),
    )
    print(f"Saved: {cfg.output} ({T} frames @ {1/cfg.timestep:.0f}Hz)")
    
    # Viewer
    if cfg.show_viewer and mujoco.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0, 0, 0.5]
        viewer.cam.distance = 4.0
        import time
        while viewer.is_running():
            for t in range(T):
                if not viewer.is_running():
                    break
                pos, quat, _ = backflip_trajectory(t * cfg.timestep)
                data.qpos[:3] = pos
                data.qpos[3:7] = quat
                data.qpos[7:] = joint_pos[t]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(cfg.timestep)


if __name__ == "__main__":
    main()
