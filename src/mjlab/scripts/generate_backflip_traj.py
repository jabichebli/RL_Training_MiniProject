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
    include_pauses: bool = False


def find_robot_xml(robot: str) -> Path:
    """Find robot XML by name or path."""
    path = Path(robot)
    if path.exists() and path.suffix == ".xml":
        return path
    xml = MJLAB_SRC_PATH / "asset_zoo" / "robots" / robot / "xmls" / f"{robot}.xml"
    if xml.exists():
        return xml
    raise FileNotFoundError(f"Robot XML not found: {robot}")


def backflip_trajectory(t: float, use_pauses: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """Define backflip trajectory: body pose + joint angles."""
    
    # ============================================================================
    # TRAJECTORY PARAMETERS - Modify these to tune the backflip
    # ============================================================================
    
    # --- Flight dynamics ---
    v_z0 = 3.0    # Vertical takeoff velocity (m/s)
    g = 9.81      # Gravity (m/s²)
    
    # --- Hip angles (constant throughout) ---
    hip = 0.05
    
    # --- Pause duration ---
    pause = 0.3 if use_pauses else 0.0
    
    # --- Phase 0: Initial stance ---
    x_0 = 0.0
    z_0 = 0.29
    theta_0 = 0.0
    t_0_transition = 0.5
    t_0 = t_0_transition + pause
    fr_thigh_0 = 1.0
    fr_calf_0 = -1.9
    rr_thigh_0 = 1.0
    rr_calf_0 = -1.9
    
    # # --- Phase 1: Preparation ---
    # x_1 = -0.02
    # z_1 = 0.30
    # theta_1 = 10.0
    # t_1_transition = t_0 + 0.1
    # t_1 = t_1_transition + pause
    # fr_thigh_1 = 0.6
    # fr_calf_1 = -1.0
    # rr_thigh_1 = 0.55
    # rr_calf_1 = -2.1

      # --- Phase 1: Preparation ---
    x_1 = -0.02
    z_1 = 0.30
    theta_1 = 10.0
    t_1_transition = t_0 + 0.15
    t_1 = t_1_transition + pause
    fr_thigh_1 = 1.4
    fr_calf_1 = -2.3
    rr_thigh_1 = 0.1
    rr_calf_1 = -1.9
    
    # --- Phase 2: Pushing with arms ---
    x_2 = -0.05
    z_2 = 0.31
    theta_2 = 20.0
    t_2_transition = t_1 + 0.12
    t_2 = t_2_transition + pause
    fr_thigh_2 = 0.8
    fr_calf_2 = -0.3
    rr_thigh_2 = 0.6
    rr_calf_2 = -1.8
    
    # --- Phase 3:  ---
    x_3 = -0.1
    z_3 = 0.39
    theta_3 = 65.0
    t_3_transition = t_2 + 0.11
    t_3 = t_3_transition + pause
    fr_thigh_3 = 1.3
    fr_calf_3 = -1.6
    rr_thigh_3 = 1.6
    rr_calf_3 = -1.7
    
    # --- Phase 4: Final push with legs ---
    x_4 = -0.4
    z_4 = 0.55
    theta_4 = 110.0
    t_4_transition = t_3 + 0.13
    t_4 = t_4_transition + pause
    fr_thigh_4 = 0.6
    fr_calf_4 = -1.6
    rr_thigh_4 = 2.9
    rr_calf_4 = -0.7
    
    # --- Phase 5a: Flight (first half with leg retraction) ---
    theta_5a = 200
    t_5a_duration = 0.15  # Duration of first half of flight
    t_5a = t_4 + t_5a_duration
    fr_thigh_5a = 0.6
    fr_calf_5a = -1.6
    rr_thigh_5a = 1.0
    rr_calf_5a = -1.9
    
    # --- Phase 5b: Flight (second half, continuing ballistic) ---
    theta_5b = 300.0
    t_5b_duration = 0.3  # Duration of second half of flight
    t_5b = t_5a + t_5b_duration
    
    # --- Phase 6: Landing ---
    x_6 = -0.7
    z_6 = 0.29
    theta_6 = 360.0
    t_6 = t_5b + 0.45
    
    # ============================================================================
    
    # Phase 0: Hold (transition)
    if t < t_0_transition:
        x, z, theta = x_0, z_0, theta_0
        fr_thigh, fr_calf = fr_thigh_0, fr_calf_0
        rr_thigh, rr_calf = rr_thigh_0, rr_calf_0
    
    # Phase 0: Pause
    elif t < t_0:
        x, z, theta = x_0, z_0, theta_0
        fr_thigh, fr_calf = fr_thigh_0, fr_calf_0
        rr_thigh, rr_calf = rr_thigh_0, rr_calf_0
    
    # Phase 1: Preparation (transition)
    elif t < t_1_transition:
        s = (t - t_0) / (t_1_transition - t_0)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep
        x = x_0 + (x_1 - x_0) * s
        z = z_0 + (z_1 - z_0) * s
        theta = theta_0 + (theta_1 - theta_0) * s
        fr_thigh = fr_thigh_0 + (fr_thigh_1 - fr_thigh_0) * s
        fr_calf = fr_calf_0 + (fr_calf_1 - fr_calf_0) * s
        rr_thigh = rr_thigh_0 + (rr_thigh_1 - rr_thigh_0) * s
        rr_calf = rr_calf_0 + (rr_calf_1 - rr_calf_0) * s
    
    # Phase 1: Pause
    elif t < t_1:
        x, z, theta = x_1, z_1, theta_1
        fr_thigh, fr_calf = fr_thigh_1, fr_calf_1
        rr_thigh, rr_calf = rr_thigh_1, rr_calf_1
    
    # Phase 2: Crouch (transition)
    elif t < t_2_transition:
        s = (t - t_1) / (t_2_transition - t_1)
        s = s * s * (3.0 - 2.0 * s)  # smoothstep
        x = x_1 + (x_2 - x_1) * s
        z = z_1 + (z_2 - z_1) * s
        theta = theta_1 + (theta_2 - theta_1) * s
        fr_thigh = fr_thigh_1 + (fr_thigh_2 - fr_thigh_1) * s
        fr_calf = fr_calf_1 + (fr_calf_2 - fr_calf_1) * s
        rr_thigh = rr_thigh_1 + (rr_thigh_2 - rr_thigh_1) * s
        rr_calf = rr_calf_1 + (rr_calf_2 - rr_calf_1) * s
    
    # Phase 2: Pause
    elif t < t_2:
        x, z, theta = x_2, z_2, theta_2
        fr_thigh, fr_calf = fr_thigh_2, fr_calf_2
        rr_thigh, rr_calf = rr_thigh_2, rr_calf_2
    
    # Phase 3: Loading (transition)
    elif t < t_3_transition:
        s = (t - t_2) / (t_3_transition - t_2)
        s = s * s * (3.0 - 2.0 * s)
        x = x_2 + (x_3 - x_2) * s
        z = z_2 + (z_3 - z_2) * s
        theta = theta_2 + (theta_3 - theta_2) * s
        fr_thigh = fr_thigh_2 + (fr_thigh_3 - fr_thigh_2) * s
        fr_calf = fr_calf_2 + (fr_calf_3 - fr_calf_2) * s
        rr_thigh = rr_thigh_2 + (rr_thigh_3 - rr_thigh_2) * s
        rr_calf = rr_calf_2 + (rr_calf_3 - rr_calf_2) * s
    
    # Phase 3: Pause
    elif t < t_3:
        x, z, theta = x_3, z_3, theta_3
        fr_thigh, fr_calf = fr_thigh_3, fr_calf_3
        rr_thigh, rr_calf = rr_thigh_3, rr_calf_3
    
    # Phase 4: Takeoff (transition)
    elif t < t_4_transition:
        s = (t - t_3) / (t_4_transition - t_3)
        s = s * s * (3.0 - 2.0 * s)
        x = x_3 + (x_4 - x_3) * s
        z = z_3 + (z_4 - z_3) * s
        theta = theta_3 + (theta_4 - theta_3) * s
        fr_thigh = fr_thigh_3 + (fr_thigh_4 - fr_thigh_3) * s
        fr_calf = fr_calf_3 + (fr_calf_4 - fr_calf_3) * s
        rr_thigh = rr_thigh_3 + (rr_thigh_4 - rr_thigh_3) * s
        rr_calf = rr_calf_3 + (rr_calf_4 - rr_calf_3) * s
    
    # Phase 4: Pause
    elif t < t_4:
        x, z, theta = x_4, z_4, theta_4
        fr_thigh, fr_calf = fr_thigh_4, fr_calf_4
        rr_thigh, rr_calf = rr_thigh_4, rr_calf_4
    
    # Phase 5a: Flight (first half with leg retraction, ballistic)
    elif t < t_5a:
        dt = t - t_4
        alpha = dt / t_5a_duration
        x = x_4 - 0.5 * dt
        z = z_4 + v_z0 * dt - 0.5 * g * dt * dt
        theta = theta_4 + alpha * (theta_5a - theta_4)
        # Transition legs to retracted position
        s = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
        fr_thigh = fr_thigh_4 + (fr_thigh_5a - fr_thigh_4) * s
        fr_calf = fr_calf_4 + (fr_calf_5a - fr_calf_4) * s
        rr_thigh = rr_thigh_4 + (rr_thigh_5a - rr_thigh_4) * s
        rr_calf = rr_calf_4 + (rr_calf_5a - rr_calf_4) * s
    
    # Phase 5b: Flight (second half, continuing ballistic with retracted legs)
    elif t < t_5b:
        dt = t - t_4
        dt_5a = t_5a - t_4
        alpha = (dt - dt_5a) / t_5b_duration
        x = x_4 - 0.5 * dt
        z = z_4 + v_z0 * dt - 0.5 * g * dt * dt
        theta = theta_5a + alpha * (theta_5b - theta_5a)
        # Keep legs in retracted position
        fr_thigh = fr_thigh_5a
        fr_calf = fr_calf_5a
        rr_thigh = rr_thigh_5a
        rr_calf = rr_calf_5a
    
    # Phase 6: Landing
    elif t < t_6:
        s = (t - t_5b) / (t_6 - t_5b)
        s = s * s * (3.0 - 2.0 * s)
        x_5b_end = x_4 - 0.5 * (t_5b - t_4)
        z_5b_end = z_4 + v_z0 * (t_5b - t_4) - 0.5 * g * (t_5b - t_4) ** 2
        x = x_5b_end + (x_6 - x_5b_end) * s
        z = z_5b_end + (z_6 - z_5b_end) * s
        theta = theta_5b + (theta_6 - theta_5b) * s
        fr_thigh = fr_thigh_5a + (fr_thigh_0 - fr_thigh_5a) * s
        fr_calf = fr_calf_5a + (fr_calf_0 - fr_calf_5a) * s
        rr_thigh = rr_thigh_5a + (rr_thigh_0 - rr_thigh_5a) * s
        rr_calf = rr_calf_5a + (rr_calf_0 - rr_calf_5a) * s
    
    # Hold final
    else:
        x, z, theta = x_6, z_6, theta_6
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
        pos, quat, half_joints = backflip_trajectory(t * cfg.timestep, cfg.include_pauses)
        
        # Debug key frames
        if t in [0, 25, 50, 75, 100]:
            print(f"\n[DEBUG] Frame {t} (t={t*cfg.timestep:.2f}s) - Half-body trajectory:")
            for name, val in half_joints.items():
                if 'calf' in name.lower():
                    print(f"  {name:20s}: {val:7.3f}")
        
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
        
        # Debug after mirroring
        if t in [0, 25, 50, 75, 100]:
            print(f"  After mirroring to full body:")
            for idx, name in enumerate(joint_names):
                if 'calf' in name.lower():
                    print(f"    Joint[{idx}] {name:20s}: {joint_pos[t, idx]:7.3f}")
        
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
    pauses_str = "WITH pauses" if cfg.include_pauses else "WITHOUT pauses"
    print(f"Saved: {cfg.output} ({T} frames @ {1/cfg.timestep:.0f}Hz, {pauses_str})")
    
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
                pos, quat, _ = backflip_trajectory(t * cfg.timestep, cfg.include_pauses)
                data.qpos[:3] = pos
                data.qpos[3:7] = quat
                data.qpos[7:] = joint_pos[t]
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(cfg.timestep)


if __name__ == "__main__":
    main()
