#!/usr/bin/env python3
"""Simulate a motion.npz trajectory with physics enabled to verify feasibility.

Minimal script that uses PD control to track reference joint positions with
realistic physics constraints. When --no-use-controller is set, uses very high
PD gains to effectively enforce kinematic motion while keeping body physics active.
"""

from __future__ import annotations

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
class SimulateMotionConfig:
    """Configuration for motion physics simulation."""

    motion_file: Path
    """Path to the motion.npz file to simulate."""

    robot: str = "go2"
    """Robot to use: 'go1' or 'go2'."""

    kp: float = 40
    """PD controller proportional gain (Nm/rad). Realistic range: Hip~16, Knee~64."""

    kd: float = 4
    """PD controller derivative gain (Nm/(rad/s)). Realistic range: Hip~1, Knee~4."""

    loop: bool = True
    """Whether to loop the motion continuously. If False, plays once, holds final position for 5s, then exits."""
    
    hold_duration: float = 5.0
    """Duration (seconds) to hold final position before exiting when loop=False."""

    timestep: float = 0.002
    """Physics timestep in seconds."""

    no_gravity: bool = False
    """Disable gravity for testing joint tracking only."""

    init_height: float = 0.29
    """Initial height to suspend robot (meters)."""

    use_controller: bool = True
    """Use custom PD controller. If False, use very high gains for kinematic replay with physics."""

    save_output: bool = False
    """Save the physics-simulated trajectory to a new motion file. Recommended: use --no-loop with this."""

    output_file: str = "backflip_motion_physics.npz"
    """Output filename for the physics-simulated motion file. Will contain actual simulated joint/body states."""


def load_motion_data(motion_file: Path) -> dict[str, np.ndarray]:
    """Load motion data from .npz file."""
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    data = np.load(motion_file)
    required_keys = ["joint_pos", "joint_vel"]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"Motion file missing required key: {key}")

    return {key: data[key] for key in required_keys}


def simulate_motion(cfg: SimulateMotionConfig):
    """Run physics simulation with motion tracking."""
    print(f"[INFO] Loading motion file: {cfg.motion_file}")
    motion = load_motion_data(cfg.motion_file)

    n_frames = motion["joint_pos"].shape[0]
    n_joints = motion["joint_pos"].shape[1]
    print(f"[INFO] Motion: {n_frames} frames, {n_joints} joints")

    # Load robot
    print(f"[INFO] Loading robot: {cfg.robot}")
    if cfg.robot == "go1":
        robot_cfg = get_go1_robot_cfg()
    elif cfg.robot == "go2":
        robot_cfg = get_go2_robot_cfg()
    else:
        raise ValueError(f"Unknown robot: {cfg.robot}")

    entity = Entity(robot_cfg)
    spec = entity._spec

    # Add ground plane with collision
    spec.worldbody.add_geom(
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.05],
        rgba=[0.5, 0.5, 0.5, 1.0],
        contype=1,  # Ground collision type
        conaffinity=1,  # Ground collides with robot (contype=1)
        friction=[1.0, 0.005, 0.0001],  # Friction coefficients
    )

    # Compile model
    model = spec.compile()
    data = mujoco.MjData(model)
    model.opt.timestep = cfg.timestep

    # Disable gravity if requested
    if cfg.no_gravity:
        model.opt.gravity[:] = [0, 0, 0]
        print("[INFO] Gravity disabled - robot suspended in midair")

    # Get joint and actuator names early for diagnostics
    joint_names = [model.joint(i).name for i in range(model.njnt) if model.joint(i).type != mujoco.mjtJoint.mjJNT_FREE]
    actuator_names = [model.actuator(i).name for i in range(model.nu)]
    
    # Set per-joint gains based on Go2 specifications
    # Hip/Thigh: kp=16.0, kd=1.02
    # Knee (calf): kp=32.0, kd=2.04
    kp_array = np.zeros(model.nu)
    kd_array = np.zeros(model.nu)
    
    for i in range(model.nu):
        actuator = model.actuator(i)
        trnid = actuator.trnid[0]
        if trnid >= 0 and trnid < model.njnt:
            joint_name = model.joint(trnid).name
            # Check if this is a knee/calf joint
            if 'calf' in joint_name.lower():
                kp_array[i] = 32.0  # Knee stiffness
                kd_array[i] = 2.04  # Knee damping
            else:
                kp_array[i] = 16.0  # Hip/Thigh stiffness
                kd_array[i] = 1.02  # Hip/Thigh damping
    
    print(f"\n[INFO] Using Go2-specific gains:")
    for i in range(model.nu):
        print(f"  [{i}] {actuator_names[i]:20s}: kp={kp_array[i]:5.2f} Nm/rad, kd={kd_array[i]:5.2f} Nm/(rad/s)")
    
    # Initialize robot at first frame
    print(f"[INFO] Model DOF: qpos={model.nq}, qvel={model.nv}, actuators={model.nu}")
    print(f"[INFO] Motion file: joint_pos shape={motion['joint_pos'][0].shape}, joint_vel shape={motion['joint_vel'][0].shape}")
    
    if motion["joint_pos"][0].shape[0] != model.nu:
        print(f"[WARN] Motion has {motion['joint_pos'][0].shape[0]} joints, but model has {model.nu} actuators")
    
    data.qpos[2] = cfg.init_height  # Set z position


    # Set initial position (suspended in air if no_gravity)
    if cfg.no_gravity:
        data.qpos[3:7] = [1, 0, 0, 0]  # Set body orientation to upright (identity quaternion)
        print(f"[INFO] Robot suspended at height {cfg.init_height}m with fixed upright orientation")
    
    data.qpos[7:] = motion["joint_pos"][0]
    data.qvel[6:] = motion["joint_vel"][0]
    mujoco.mj_forward(model, data)
    
    print(f"[INFO] Initial joint angles (rad): {np.array2string(motion['joint_pos'][0], precision=3, suppress_small=True)}")
    
    # Sanity check: verify motion file has correct number of joints
    if motion["joint_pos"].shape[1] != len(joint_names):
        print(f"[ERROR] Motion file has {motion['joint_pos'].shape[1]} joints, but model has {len(joint_names)} joints!")
        print(f"[ERROR] This will cause control signal misalignment!")

    ctrl_range = model.actuator_ctrlrange
    
    print(f"\n[INFO] Number of joints: {len(joint_names)}, Number of actuators: {len(actuator_names)}")
    print(f"[INFO] Joint order: {joint_names}")
    print(f"[INFO] Actuator order: {actuator_names}")
    print(f"[INFO] Control range shape: {ctrl_range.shape}, data.ctrl shape: {data.ctrl.shape}")
    
    # Verify joint-actuator correspondence
    print("\n[INFO] Joint-Actuator Mapping:")
    mismatch_warning = False
    for i in range(min(len(joint_names), len(actuator_names))):
        jname = joint_names[i]
        aname = actuator_names[i]
        # Check if actuator targets this joint
        actuator = model.actuator(i)
        trnid = actuator.trnid[0]  # First transmission ID
        if trnid >= 0 and trnid < model.njnt:
            target_joint_name = model.joint(trnid).name
            match_symbol = "✓" if target_joint_name == jname else "✗ MISMATCH"
            print(f"  [{i}] Joint: {jname:20s} <-> Actuator: {aname:20s} (targets: {target_joint_name}) {match_symbol}")
            if target_joint_name != jname:
                mismatch_warning = True
        else:
            print(f"  [{i}] Joint: {jname:20s} <-> Actuator: {aname:20s}")
    
    # Create bidirectional mapping between joint and actuator indices
    joint_to_actuator = np.zeros(len(joint_names), dtype=int)
    actuator_to_joint = np.zeros(model.nu, dtype=int)
    
    for joint_idx, jname in enumerate(joint_names):
        # Find which actuator controls this joint
        for act_idx in range(model.nu):
            actuator = model.actuator(act_idx)
            trnid = actuator.trnid[0]
            if trnid >= 0 and trnid < model.njnt:
                target_joint_name = model.joint(trnid).name
                if target_joint_name == jname:
                    joint_to_actuator[joint_idx] = act_idx
                    actuator_to_joint[act_idx] = joint_idx
                    break
    
    print(f"\n[INFO] Joint-to-Actuator Index Mapping: {joint_to_actuator}")
    print(f"[INFO] Actuator-to-Joint Index Mapping: {actuator_to_joint}")
    
    if mismatch_warning:
        print("\n[WARNING] Joint-Actuator order mismatch detected!")
        print("[WARNING] Using remapping to fix control signals...")
    
    if cfg.use_controller:
        print(f"[INFO] Mode: PD Controller with Go2-specific gains")
        print(f"  Hip/Thigh: kp=16.0 Nm/rad, kd=1.02 Nm/(rad/s)")
        print(f"  Knee:      kp=32.0 Nm/rad, kd=2.04 Nm/(rad/s)")
    else:
        print(f"[INFO] Mode: Kinematic Replay with Physics (very high gains: kp=10000, kd=100)")
    print(f"[INFO] Gravity: {'OFF (midair test)' if cfg.no_gravity else 'ON'}")
    if cfg.save_output:
        print(f"[INFO] Will save physics-simulated trajectory to: {cfg.output_file}")
    if cfg.loop:
        print("[INFO] Looping: Motion will repeat continuously")
    else:
        print(f"[INFO] Single play: Will hold final frame for {cfg.hold_duration}s then exit")
    print("[INFO] Controls:")
    print("  - SPACE: Pause/Resume")
    print("  - R: Reset to frame 0")
    print("  - ESC: Exit\n")

    frame_idx = 0
    paused = False
    holding_final_frame = False
    hold_start_time = None

    # Arrays for recording simulated trajectory
    recording_complete = False
    # Exclude world body (body 0) - only record robot bodies for training compatibility
    n_bodies = model.nbody - 1
    recorded_joint_pos = np.zeros((n_frames, n_joints)) if cfg.save_output else None
    recorded_joint_vel = np.zeros((n_frames, n_joints)) if cfg.save_output else None
    recorded_body_pos_w = np.zeros((n_frames, n_bodies, 3)) if cfg.save_output else None
    recorded_body_quat_w = np.zeros((n_frames, n_bodies, 4)) if cfg.save_output else None
    recorded_body_lin_vel_w = np.zeros((n_frames, n_bodies, 3)) if cfg.save_output else None
    recorded_body_ang_vel_w = np.zeros((n_frames, n_bodies, 3)) if cfg.save_output else None
    
    if cfg.save_output:
        print(f"[INFO] Recording simulation data for output...")

    def key_callback(keycode):
        nonlocal paused, frame_idx, holding_final_frame, hold_start_time
        if keycode == 32:  # Space bar
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        elif keycode == 82 or keycode == 114:  # 'R' or 'r'
            frame_idx = 0
            holding_final_frame = False
            hold_start_time = None
            # Reset robot state to first frame
            if cfg.no_gravity:
                data.qpos[2] = cfg.init_height  # Maintain suspended height
                data.qpos[3:7] = [1, 0, 0, 0]  # Reset body orientation to upright
            data.qpos[7:] = motion["joint_pos"][0]
            data.qvel[6:] = motion["joint_vel"][0]
            mujoco.mj_forward(model, data)
            print("[INFO] Reset to frame 0")

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    ) as viewer:
        viewer.cam.lookat[:] = [0, 0, 0.3]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -20

        while viewer.is_running():
            if not paused:
                # Check if we're holding the final frame
                if holding_final_frame:
                    if hold_start_time is None:
                        hold_start_time = time.time()
                        print(f"[INFO] Holding final frame for {cfg.hold_duration}s...")
                    
                    # Check if hold duration has elapsed
                    if time.time() - hold_start_time >= cfg.hold_duration:
                        print("[INFO] Hold duration complete. Exiting.")
                        break
                    
                    # Use last frame's reference
                    ref_pos = motion["joint_pos"][-1]
                    ref_vel = motion["joint_vel"][-1]
                else:
                    # Normal playback
                    ref_pos = motion["joint_pos"][frame_idx]
                    ref_vel = motion["joint_vel"][frame_idx]
                
                if cfg.use_controller:
                    # PD control mode: tau = kp * (q_ref - q) + kd * (qd_ref - qd)
                    curr_pos = data.qpos[7:]  # Skip root (7 DOF)
                    curr_vel = data.qvel[6:]  # Skip root (6 DOF)

                    pos_error = ref_pos - curr_pos
                    vel_error = ref_vel - curr_vel
                    
                    # Compute control directly in actuator order using per-actuator gains
                    ctrl = np.zeros(model.nu)
                    for act_idx in range(model.nu):
                        joint_idx = actuator_to_joint[act_idx]
                        ctrl[act_idx] = (
                            kp_array[act_idx] * pos_error[joint_idx] + 
                            kd_array[act_idx] * vel_error[joint_idx]
                        )
                    
                    # Apply control with actuator limits
                    ctrl_clipped = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
                    data.ctrl[:] = ctrl_clipped

                    # Step physics
                    mujoco.mj_step(model, data)
                    
                    # Fix body orientation if no gravity (prevent drift)
                    if cfg.no_gravity:
                        data.qpos[2] = cfg.init_height  # Maintain height
                        data.qpos[3:7] = [1, 0, 0, 0]  # Fix orientation to upright
                        data.qvel[0:6] = 0.0  # Zero out root velocities

                    # Print tracking error and joint angles periodically
                    joint_error = np.linalg.norm(pos_error)
                    
                    # Detailed output for first few frames to debug control
                    if frame_idx < 3:
                        print(f"\n=== Frame {frame_idx} (DETAILED) ===")
                        print(f"Reference pos (joint order): {np.array2string(ref_pos, precision=4, suppress_small=True)}")
                        print(f"Actual pos (joint order):    {np.array2string(curr_pos, precision=4, suppress_small=True)}")
                        print(f"Pos error (joint order):     {np.array2string(pos_error, precision=4, suppress_small=True)}")
                        print(f"Reference vel (joint order): {np.array2string(ref_vel, precision=4, suppress_small=True)}")
                        print(f"Actual vel (joint order):    {np.array2string(curr_vel, precision=4, suppress_small=True)}")
                        print(f"Vel error (joint order):     {np.array2string(vel_error, precision=4, suppress_small=True)}")
                        print(f"Actuator gains (kp):         {np.array2string(kp_array, precision=2, suppress_small=True)}")
                        print(f"Actuator gains (kd):         {np.array2string(kd_array, precision=2, suppress_small=True)}")
                        print(f"Control signal (act order):  {np.array2string(ctrl, precision=4, suppress_small=True)}")
                        print(f"Control clipped (act order): {np.array2string(ctrl_clipped, precision=4, suppress_small=True)}")
                        print(f"Ctrl limits:   min={np.array2string(ctrl_range[:, 0], precision=1)}")
                        print(f"               max={np.array2string(ctrl_range[:, 1], precision=1)}")
                    elif frame_idx % 50 == 0:
                        print(f"\nFrame {frame_idx}/{n_frames}  Joint Error: {joint_error:.3f} rad")
                        print(f"  Reference joints: {np.array2string(ref_pos, precision=3, suppress_small=True)}")
                        print(f"  Actual joints:    {np.array2string(curr_pos, precision=3, suppress_small=True)}")
                        print(f"  Error:            {np.array2string(pos_error, precision=3, suppress_small=True)}")
                        print(f"  Control signal:   {np.array2string(ctrl_clipped, precision=3, suppress_small=True)}")
                        print(f"  Joint velocities: {np.array2string(curr_vel, precision=3, suppress_small=True)}")
                else:
                    # Kinematic replay mode with physics:
                    # Use very high PD gains to effectively lock joints while allowing body physics
                    curr_pos = data.qpos[7:]
                    curr_vel = data.qvel[6:]
                    
                    # Very high gains to enforce kinematic motion
                    kp_kinematic = 10000.0
                    kd_kinematic = 100.0
                    
                    pos_error = ref_pos - curr_pos
                    vel_error = ref_vel - curr_vel
                    
                    # Compute control directly in actuator order
                    ctrl = np.zeros(model.nu)
                    for act_idx in range(model.nu):
                        joint_idx = actuator_to_joint[act_idx]
                        ctrl[act_idx] = kp_kinematic * pos_error[joint_idx] + kd_kinematic * vel_error[joint_idx]
                    
                    # Apply control with actuator limits
                    ctrl_clipped = np.clip(ctrl, ctrl_range[:, 0], ctrl_range[:, 1])
                    data.ctrl[:] = ctrl_clipped
                    
                    # Step physics (body is affected by gravity/contact, joints track reference)
                    mujoco.mj_step(model, data)
                    
                    if cfg.no_gravity:
                        data.qpos[2] = cfg.init_height  # Maintain height
                        data.qpos[3:7] = [1, 0, 0, 0]  # Set orientation to upright
                        data.qvel[0:6] = 0.0  # Zero out root velocities
                    
                    if frame_idx % 50 == 0:
                        joint_error = np.linalg.norm(pos_error)
                        print(f"\nFrame {frame_idx}/{n_frames}  (kinematic replay with physics)")
                        print(f"  Joint Error: {joint_error:.4f} rad")
                        print(f"  Reference joints: {np.array2string(ref_pos, precision=3, suppress_small=True)}")
                        print(f"  Actual joints:    {np.array2string(curr_pos, precision=3, suppress_small=True)}")

                # Record simulation data if saving output
                if cfg.save_output and not recording_complete and recorded_joint_pos is not None:
                    # Record physics-simulated joint angles
                    recorded_joint_pos[frame_idx] = data.qpos[7:]
                    recorded_joint_vel[frame_idx] = data.qvel[6:]  # type: ignore
                    
                    # Record physics-simulated body states (skip world body at index 0)
                    for body_idx in range(n_bodies):
                        # MuJoCo body index starts at 1 (skip world at 0), array index starts at 0
                        mujoco_body_idx = body_idx + 1
                        recorded_body_pos_w[frame_idx, body_idx] = data.xpos[mujoco_body_idx]  # type: ignore
                        recorded_body_quat_w[frame_idx, body_idx] = data.xquat[mujoco_body_idx]  # type: ignore
                        recorded_body_lin_vel_w[frame_idx, body_idx] = data.cvel[mujoco_body_idx, :3]  # type: ignore
                        recorded_body_ang_vel_w[frame_idx, body_idx] = data.cvel[mujoco_body_idx, 3:]  # type: ignore
                    
                    # Debug: Print recording status (body 0 in array is trunk, MuJoCo body 1)
                    if frame_idx == 0:
                        print(f"[DEBUG] Recording started - trunk body pos: {data.xpos[1]} (saved to array index 0)")
                    elif frame_idx % 100 == 0:
                        trunk_height = data.xpos[1][2]
                        status = "⚠ UNDERGROUND!" if trunk_height < 0 else "OK"
                        print(f"[DEBUG] Frame {frame_idx} - trunk pos: {data.xpos[1]} {status}")

                # Advance frame (only if not holding)
                if not holding_final_frame:
                    frame_idx += 1
                    if frame_idx >= n_frames:
                        if cfg.save_output and not recording_complete and recorded_joint_pos is not None:
                            recording_complete = True
                            print(f"\n[INFO] Recording complete! Saving to {cfg.output_file}")
                            print(f"[INFO] Data shapes:")
                            print(f"  joint_pos: {recorded_joint_pos.shape}")
                            if recorded_joint_vel is not None:
                                print(f"  joint_vel: {recorded_joint_vel.shape}")
                            if recorded_body_pos_w is not None:
                                print(f"  body_pos_w: {recorded_body_pos_w.shape}")
                            if recorded_body_quat_w is not None:
                                print(f"  body_quat_w: {recorded_body_quat_w.shape}")
                            if recorded_body_lin_vel_w is not None:
                                print(f"  body_lin_vel_w: {recorded_body_lin_vel_w.shape}")
                            if recorded_body_ang_vel_w is not None:
                                print(f"  body_ang_vel_w: {recorded_body_ang_vel_w.shape}")
                            
                            np.savez(
                                cfg.output_file,
                                joint_pos=recorded_joint_pos,
                                joint_vel=recorded_joint_vel,
                                body_pos_w=recorded_body_pos_w,
                                body_quat_w=recorded_body_quat_w,
                                body_lin_vel_w=recorded_body_lin_vel_w,
                                body_ang_vel_w=recorded_body_ang_vel_w,
                            )
                            print(f"[INFO] Physics-simulated motion saved to: {cfg.output_file}")
                            
                            # Verify saved file
                            verify_data = np.load(cfg.output_file)
                            print(f"[INFO] Verification - saved file contains keys: {list(verify_data.keys())}")
                            if 'body_pos_w' in verify_data:
                                print(f"[INFO] Verification - body_pos_w shape: {verify_data['body_pos_w'].shape}")
                        
                        if cfg.loop:
                            frame_idx = 0
                        else:
                            # Enter holding mode instead of breaking immediately
                            holding_final_frame = True
                            frame_idx = n_frames - 1  # Stay on last frame

            viewer.sync()
            time.sleep(cfg.timestep)

    print("[INFO] Done.")


def main():
    """Entry point."""
    cfg = tyro.cli(SimulateMotionConfig)
    simulate_motion(cfg)


if __name__ == "__main__":
    main()

