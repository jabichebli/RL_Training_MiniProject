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
    motion_timestep = 0.02
    steps_per_frame = int(motion_timestep / model.opt.timestep)

    # Build joint-to-actuator mapping
    joint_names = [model.joint(i).name for i in range(model.njnt) 
                   if model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
    actuator_to_joint = np.zeros(model.nu, dtype=int)
    
    for joint_idx, jname in enumerate(joint_names):
        for act_idx in range(model.nu):
            trnid = model.actuator(act_idx).trnid[0]
            if trnid >= 0 and trnid < model.njnt:
                if model.joint(trnid).name == jname:
                    actuator_to_joint[act_idx] = joint_idx
                    break

    # PD gains
    if cfg.use_controller:
        # Go2-specific: knee=32/2.04, hip/thigh=16/1.02
        kp = np.array([32.0 if 'calf' in model.joint(model.actuator(i).trnid[0]).name.lower() 
                       else 16.0 for i in range(model.nu)])
        kd = np.array([2.04 if 'calf' in model.joint(model.actuator(i).trnid[0]).name.lower() 
                       else 1.02 for i in range(model.nu)])
    else:
        # Very high gains for kinematic replay
        kp = np.full(model.nu, 10000.0)
        kd = np.full(model.nu, 100.0)

    # Initialize
    data.qpos[2] = 0.29
    data.qpos[7:] = motion["joint_pos"][0]
    data.qvel[6:] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)
    
    print(f"Mode: {'PD controller' if cfg.use_controller else 'Kinematic replay (high gains)'}")

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
    paused, recorded = False, False

    def key_callback(keycode):
        nonlocal paused, frame_idx, physics_step
        if keycode == 32:
            paused = not paused
        elif keycode in (82, 114):
            frame_idx, physics_step = 0, 0
            data.qpos[7:] = motion["joint_pos"][0]
            data.qvel[6:] = motion["joint_vel"][0]
            mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.3]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20

        while viewer.is_running():
            if not paused:
                ref_pos = motion["joint_pos"][frame_idx]
                ref_vel = motion["joint_vel"][frame_idx]
                
                # PD control with joint-to-actuator mapping
                pos_error = ref_pos - data.qpos[7:]
                vel_error = ref_vel - data.qvel[6:]
                ctrl = np.zeros(model.nu)
                for act_idx in range(model.nu):
                    joint_idx = actuator_to_joint[act_idx]
                    ctrl[act_idx] = kp[act_idx] * pos_error[joint_idx] + kd[act_idx] * vel_error[joint_idx]
                data.ctrl[:] = np.clip(ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
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
                            break

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
