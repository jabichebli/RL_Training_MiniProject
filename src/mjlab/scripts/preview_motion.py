#!/usr/bin/env python3
"""Preview a motion.npz file in the MuJoCo viewer.

Simple kinematic replay of any motion file without physics simulation.
Just visualizes the trajectory as defined in the file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from mjlab.asset_zoo.robots import get_go2_robot_cfg
from mjlab.entity import Entity


@dataclass
class PreviewMotionConfig:
    """Configuration for motion preview."""

    motion_file: Path
    """Path to the motion.npz file to preview."""

    loop: bool = True
    """Whether to loop the motion continuously."""

    playback_speed: float = 1.0
    """Playback speed multiplier (1.0 = normal, 0.5 = half speed, 2.0 = double speed)."""

    timestep: float = 0.005
    """Timestep for playback (seconds). Should match the motion file's timestep."""


def load_motion_file(motion_file: Path) -> dict[str, np.ndarray]:
    """Load motion data from .npz file."""
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    data = np.load(motion_file)
    
    # Check for required keys
    required_keys = ["joint_pos", "joint_vel"]
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        raise ValueError(f"Motion file missing required keys: {missing_keys}")
    
    # Load body data if available
    result = {
        "joint_pos": data["joint_pos"],
        "joint_vel": data["joint_vel"],
    }
    
    # Optional body data
    if "body_pos_w" in data:
        result["body_pos_w"] = data["body_pos_w"]
    if "body_quat_w" in data:
        result["body_quat_w"] = data["body_quat_w"]
    
    return result


def preview_motion(cfg: PreviewMotionConfig) -> None:
    """Preview motion file in MuJoCo viewer."""
    print(f"[INFO] Loading motion file: {cfg.motion_file}")
    motion = load_motion_file(cfg.motion_file)
    
    n_frames = motion["joint_pos"].shape[0]
    n_joints = motion["joint_pos"].shape[1]
    duration = n_frames * cfg.timestep
    
    print(f"[INFO] Motion: {n_frames} frames, {n_joints} joints")
    print(f"[INFO] Duration: {duration:.2f}s at {cfg.timestep}s per frame")
    print(f"[INFO] Playback speed: {cfg.playback_speed}x")
    
    # Load robot (hardcoded to Go2)
    print(f"[INFO] Loading robot: Unitree Go2")
    robot_cfg = get_go2_robot_cfg()
    entity = Entity(robot_cfg)
    spec = entity._spec
    
    # Add ground plane with checkerboard texture
    # from mjlab.utils import spec_config as spec_cfg
    
    # spec_cfg.TextureCfg(
    #     name="groundplane",
    #     type="2d",
    #     builtin="checker",
    #     mark="edge",
    #     rgb1=(0.2, 0.3, 0.4),
    #     rgb2=(0.1, 0.2, 0.3),
    #     markrgb=(0.8, 0.8, 0.8),
    #     width=300,
    #     height=300,
    # ).func(spec)
    
    # spec_cfg.MaterialCfg(
    #     name="groundplane",
    #     texrepeat=(5, 5),
    #     texuniform=True,
    #     reflectance=0.2,
    # ).func(spec, texture="groundplane")
    
    # spec.worldbody.add_geom(
    #     type=mujoco.mjtGeom.mjGEOM_PLANE,
    #     size=[0, 0, 0.05],
    #     material="groundplane",
    # )
    
    # # Add light
    # spec.worldbody.add_light(
    #     directional=True,
    #     pos=[0, 0, 3],
    #     dir=[0, 0, -1],
    #     castshadow=True,
    # )
    
    # Compile model
    model = spec.compile()
    data = mujoco.MjData(model)
    
    # Validate joint count
    model_joints = model.nv - 6  # Exclude floating base DOF
    if n_joints != model_joints:
        print(f"[WARNING] Motion has {n_joints} joints but model has {model_joints} joints!")
        print(f"[WARNING] This may cause visualization errors.")
    
    print(f"\n[INFO] Starting preview...")
    if cfg.loop:
        print("[INFO] Looping: Press ESC to exit")
    else:
        print("[INFO] Single play: Will exit after one playthrough")
    print("[INFO] Controls:")
    print("  - SPACE: Pause/Resume")
    print("  - R: Reset to frame 0")
    print("  - ESC: Exit\n")
    
    frame_idx = 0
    paused = False
    use_body_data = "body_pos_w" in motion and "body_quat_w" in motion
    
    if use_body_data:
        print("[INFO] Motion file contains body pose data - using it for visualization")
        print(f"[INFO] Body data shape: {motion['body_pos_w'].shape} (frames, bodies, xyz)")
        print(f"[INFO] Using body 1 (robot trunk) for base position/orientation")
    
    def key_callback(keycode):
        nonlocal paused, frame_idx
        if keycode == 32:  # Space bar
            paused = not paused
            print(f"[INFO] {'Paused' if paused else 'Resumed'}")
        elif keycode == 82 or keycode == 114:  # 'R' or 'r'
            frame_idx = 0
            print("[INFO] Reset to frame 0")
    
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False, key_callback=key_callback
    ) as viewer:
        # Set camera to world frame
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD.value
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
        viewer.cam.trackbodyid = -1
        viewer.cam.fixedcamid = -1
        viewer.cam.lookat[:] = [0.0, 0.0, 0.5]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -20.0
        
        actual_timestep = cfg.timestep / cfg.playback_speed
        
        while viewer.is_running():
            if not paused:
                # Set robot state from motion file
                if use_body_data:
                    # Use body pose from motion file (body 1 is robot trunk, body 0 is world)
                    data.qpos[:3] = motion["body_pos_w"][frame_idx, 1]
                    data.qpos[3:7] = motion["body_quat_w"][frame_idx, 1]
                
                # Set joint positions and velocities
                data.qpos[7:] = motion["joint_pos"][frame_idx]
                data.qvel[6:] = motion["joint_vel"][frame_idx]
                
                # Forward kinematics only (no physics)
                mujoco.mj_forward(model, data)
                
                # Display frame info periodically
                if frame_idx % 50 == 0:
                    time_in_motion = frame_idx * cfg.timestep
                    if use_body_data:
                        trunk_pos = motion["body_pos_w"][frame_idx, 1]
                        print(f"\rFrame {frame_idx}/{n_frames} ({time_in_motion:.2f}s/{duration:.2f}s) - Trunk pos: [{trunk_pos[0]:.2f}, {trunk_pos[1]:.2f}, {trunk_pos[2]:.2f}]", end="", flush=True)
                    else:
                        print(f"\rFrame {frame_idx}/{n_frames} ({time_in_motion:.2f}s/{duration:.2f}s)", end="", flush=True)
                
                # Advance frame
                frame_idx += 1
                if frame_idx >= n_frames:
                    if cfg.loop:
                        frame_idx = 0
                        print("\n[INFO] Looping back to start")
                    else:
                        print("\n[INFO] Motion complete. Exiting.")
                        break
            
            viewer.sync()
            time.sleep(actual_timestep)
    
    print("[INFO] Preview complete.")


def main():
    """Main entry point for the script."""
    cfg = tyro.cli(PreviewMotionConfig)
    preview_motion(cfg)


if __name__ == "__main__":
    main()

