#!/usr/bin/env python3
"""Simulate motion.npz with physics-based PD control."""

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from mjlab.asset_zoo.robots import get_go1_robot_cfg, get_go2_robot_cfg
from mjlab.entity import Entity


@dataclass
class Config:
    motion_file: Path
    robot: str = "go2"
    loop: bool = True
    save_output: bool = False
    output_file: str = "motion_physics.npz"
    use_controller: bool = True
    """Use realistic PD gains. If False, use very high gains for kinematic replay."""
    gain_scale: float = 1.0
    """Scale factor for PD gains (applied to both kp and kd)."""
    gravity_scale: float = 1.0
    """Scale factor for gravity (1.0 = normal, 0.5 = half gravity, etc.)."""


def main():
    cfg = tyro.cli(Config)
    motion = dict(np.load(cfg.motion_file))
    n_frames, n_joints = motion["joint_pos"].shape

    # Load robot and add ground
    robot_cfg = get_go1_robot_cfg() if cfg.robot == "go1" else get_go2_robot_cfg()
    spec = Entity(robot_cfg)._spec
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.5, 0.5, 0.5, 1.0],
        friction=[1.0, 0.005, 0.0001],
    )
    
    model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.timestep = 0.002
    # Apply gravity scale
    if cfg.gravity_scale != 1.0:
        model.opt.gravity[2] *= cfg.gravity_scale  # Scale z-component (vertical gravity)
    motion_timestep = 0.02
    steps_per_frame = int(motion_timestep / model.opt.timestep)

    # Build joint-to-actuator mapping
    # Use same filtering as generate_backflip_traj.py: exclude floating_base_joint by name
    joint_names = [model.joint(i).name for i in range(model.njnt) 
                   if model.joint(i).name != "floating_base_joint"]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    actuator_to_joint = np.full(model.nu, -1, dtype=int)  # Initialize to -1 to mark unmapped
    
    for joint_idx, jname in enumerate(joint_names):
        for act_idx in range(model.nu):
            trnid = model.actuator(act_idx).trnid[0]
            if trnid >= 0 and trnid < model.njnt:
                if model.joint(trnid).name == jname:
                    actuator_to_joint[act_idx] = joint_idx
                    break

    # PD gains
    if cfg.use_controller:
        # Calculated from go2_constants.py:
        # NATURAL_FREQ = 10.0 * 2.0 * π ≈ 62.83 rad/s
        # DAMPING_RATIO = 2.0
        # HIP: reflected_inertia = 0.000111842 * 6² = 0.004026
        #      kp = 0.004026 * 62.83² ≈ 15.89, kd = 2*2*0.004026*62.83 ≈ 1.01
        # KNEE: reflected_inertia = 0.000111842 * 12² = 0.016105
        #       kp = 0.016105 * 62.83² ≈ 63.55, kd = 2*2*0.016105*62.83 ≈ 4.05
        kp = np.array([63.55 if 'calf' in model.joint(model.actuator(i).trnid[0]).name.lower() 
                       else 15.89 for i in range(model.nu)])
        kd = np.array([4.05 if 'calf' in model.joint(model.actuator(i).trnid[0]).name.lower() 
                       else 1.01 for i in range(model.nu)])
        kp *= cfg.gain_scale
        kd *= cfg.gain_scale
    else:
        # Very high gains for kinematic replay
        kp = np.full(model.nu, 10000.0)
        kd = np.full(model.nu, 100.0)

    # Check motion file range for calf joints and adjust joint limits if needed
    print("\nCalf joint motion file range:")
    for i, jname in enumerate(joint_names):
        if 'calf' in jname.lower():
            min_val = motion["joint_pos"][:, i].min()
            max_val = motion["joint_pos"][:, i].max()
            print(f"  {jname:20s}: min={min_val:.3f}, max={max_val:.3f}")
            
            # Find corresponding joint in model and check/adjust limits
            for jnt_id in range(model.njnt):
                if model.joint(jnt_id).name == jname:
                    jnt_range = model.jnt_range[jnt_id]
                    if jnt_range[0] > min_val or jnt_range[1] < max_val:
                        print(f"    Adjusting limits from [{jnt_range[0]:.3f}, {jnt_range[1]:.3f}] to [{min_val - 0.1:.3f}, {max_val + 0.1:.3f}]")
                        model.jnt_range[jnt_id, 0] = min_val - 0.1
                        model.jnt_range[jnt_id, 1] = max_val + 0.1
                    break

    # Initialize
    data.qpos[2] = 0.29
    data.qpos[7:] = motion["joint_pos"][0]
    data.qvel[6:] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)

    # Recording arrays
    n_bodies = model.nbody - 1
    rec: dict[str, np.ndarray] = {}
    if cfg.save_output:
        rec = {
            "joint_pos": np.zeros((n_frames, n_joints)),
            "joint_vel": np.zeros((n_frames, n_joints)),
            "body_pos_w": np.zeros((n_frames, n_bodies, 3)),
            "body_quat_w": np.zeros((n_frames, n_bodies, 4)),
            "body_lin_vel_w": np.zeros((n_frames, n_bodies, 3)),
            "body_ang_vel_w": np.zeros((n_frames, n_bodies, 3)),
        }

    frame_idx, physics_step = 0, 0
    paused, recorded, should_exit = False, False, False

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = [0, 0, 0.3]
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -20

            while viewer.is_running() and not should_exit:
                if not paused:
                    ref_pos = motion["joint_pos"][frame_idx]
                    ref_vel = motion["joint_vel"][frame_idx]
                    
                    # PD control with joint-to-actuator mapping
                    pos_error = ref_pos - data.qpos[7:]
                    vel_error = ref_vel - data.qvel[6:]
                    ctrl = np.zeros(model.nu)
                    for act_idx in range(model.nu):
                        joint_idx = actuator_to_joint[act_idx]
                        if joint_idx >= 0 and joint_idx < len(ref_pos):
                            ctrl[act_idx] = kp[act_idx] * pos_error[joint_idx] + kd[act_idx] * vel_error[joint_idx]
                    
                    # Clipping occurs here
                    data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
                    
                    # Monitor when reference is near -2.3 for rear calf joints
                    if physics_step == 0:
                        for i, jname in enumerate(joint_names):
                            if 'calf' in jname.lower() and ('RR' in jname or 'RL' in jname):
                                ref_val = ref_pos[i]
                                curr_val = data.qpos[7 + i]
                                if ref_val < -2.0 and abs(ref_val - curr_val) > 0.1:  # Ref near -2.3 but not reaching it
                                    act_idx = next((a for a in range(model.nu) if actuator_to_joint[a] == i), None)
                                    if act_idx is not None:
                                        ctrl_val = ctrl[act_idx]
                                        ctrl_applied = data.ctrl[act_idx]
                                        print(f"Frame {frame_idx}: {jname} ref={ref_val:.3f} curr={curr_val:.3f} err={ref_val-curr_val:.3f} ctrl={ctrl_val:.1f} applied={ctrl_applied:.1f}")
                    
                    mujoco.mj_step(model, data)
                    physics_step += 1

                    # Record at frame boundaries
                    if cfg.save_output and rec and physics_step >= steps_per_frame and not recorded:
                        rec["joint_pos"][frame_idx] = data.qpos[7:]
                        rec["joint_vel"][frame_idx] = data.qvel[6:]
                        for b in range(n_bodies):
                            rec["body_pos_w"][frame_idx, b] = data.xpos[b + 1]
                            rec["body_quat_w"][frame_idx, b] = data.xquat[b + 1]
                            rec["body_lin_vel_w"][frame_idx, b] = data.cvel[b + 1, :3]
                            rec["body_ang_vel_w"][frame_idx, b] = data.cvel[b + 1, 3:]

                    # Advance frame
                    if physics_step >= steps_per_frame:
                        physics_step = 0
                        frame_idx += 1
                        if frame_idx >= n_frames:
                            if cfg.save_output and rec and not recorded:
                                np.savez(
                                    cfg.output_file,
                                    joint_pos=rec["joint_pos"],
                                    joint_vel=rec["joint_vel"],
                                    body_pos_w=rec["body_pos_w"],
                                    body_quat_w=rec["body_quat_w"],
                                    body_lin_vel_w=rec["body_lin_vel_w"],
                                    body_ang_vel_w=rec["body_ang_vel_w"],
                                )
                                print(f"Saved: {cfg.output_file}")
                                recorded = True
                            frame_idx = 0 if cfg.loop else n_frames - 1
                            if not cfg.loop:
                                should_exit = True

                if viewer.is_running() and not should_exit:
                    viewer.sync()
                    time.sleep(model.opt.timestep)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Save output if requested and not already saved
        if cfg.save_output and rec and not recorded:
            np.savez(
                cfg.output_file,
                joint_pos=rec["joint_pos"],
                joint_vel=rec["joint_vel"],
                body_pos_w=rec["body_pos_w"],
                body_quat_w=rec["body_quat_w"],
                body_lin_vel_w=rec["body_lin_vel_w"],
                body_ang_vel_w=rec["body_ang_vel_w"],
            )
            print(f"Saved: {cfg.output_file}")
    except Exception as e:
        print(f"\nError: {e}")
        raise


if __name__ == "__main__":
    main()
