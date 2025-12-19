#!/usr/bin/env python3
"""Synthesize a motion file by combining physics body poses with kinematic joint angles.

This script takes two motion files:
1. Kinematic motion file (from generate_backflip_traj.py) - provides ideal joint angles
2. Physics motion file (from simulate_motion_physics.py) - provides realistic body poses

Output: Hybrid motion with:
- Body poses (position, orientation, velocities): from PHYSICS simulation
- Joint angles (positions, velocities): from KINEMATIC reference
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


@dataclass
class SynthesizeMotionConfig:
    """Configuration for motion synthesis."""

    kinematic_file: Path
    """Path to kinematic motion.npz file (from generate_backflip_traj.py)."""

    physics_file: Path
    """Path to physics motion.npz file (from simulate_motion_physics.py)."""

    output_file: str = "backflip_motion_synthesized.npz"
    """Output filename for the synthesized motion file."""


def load_motion_file(motion_file: Path) -> dict[str, np.ndarray]:
    """Load motion data from .npz file."""
    if not motion_file.exists():
        raise FileNotFoundError(f"Motion file not found: {motion_file}")

    data = np.load(motion_file)
    
    # Check for required keys
    required_keys = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    missing_keys = [key for key in required_keys if key not in data]
    
    if missing_keys:
        raise ValueError(f"Motion file missing required keys: {missing_keys}")
    
    return {key: data[key] for key in required_keys}


def synthesize_motion(cfg: SynthesizeMotionConfig) -> None:
    """Synthesize motion by combining kinematic joints with physics body poses."""
    print("[INFO] Loading motion files...")
    
    # Load both motion files
    kinematic = load_motion_file(cfg.kinematic_file)
    physics = load_motion_file(cfg.physics_file)
    
    # Get dimensions
    n_frames_kin = kinematic["joint_pos"].shape[0]
    n_frames_phys = physics["joint_pos"].shape[0]
    n_joints_kin = kinematic["joint_pos"].shape[1]
    n_joints_phys = physics["joint_pos"].shape[1]
    
    print(f"[INFO] Kinematic motion: {n_frames_kin} frames, {n_joints_kin} joints")
    print(f"[INFO] Physics motion: {n_frames_phys} frames, {n_joints_phys} joints")
    
    # Validate compatibility
    if n_joints_kin != n_joints_phys:
        raise ValueError(f"Joint count mismatch: kinematic={n_joints_kin}, physics={n_joints_phys}")
    
    # Use the minimum number of frames
    n_frames = min(n_frames_kin, n_frames_phys)
    if n_frames_kin != n_frames_phys:
        print(f"[WARNING] Frame count mismatch. Using {n_frames} frames (minimum of both).")
    
    # Create synthesized motion
    print(f"[INFO] Synthesizing motion...")
    print(f"[INFO]   - Joint angles/velocities: from KINEMATIC file")
    print(f"[INFO]   - Body poses/velocities: from PHYSICS file")
    
    synthesized = {
        # Joint data from kinematic (ideal joint angles)
        "joint_pos": kinematic["joint_pos"][:n_frames],
        "joint_vel": kinematic["joint_vel"][:n_frames],
        
        # Body data from physics (realistic body motion)
        "body_pos_w": physics["body_pos_w"][:n_frames],
        "body_quat_w": physics["body_quat_w"][:n_frames],
        "body_lin_vel_w": physics["body_lin_vel_w"][:n_frames],
        "body_ang_vel_w": physics["body_ang_vel_w"][:n_frames],
    }
    
    # Save synthesized motion
    print(f"[INFO] Saving synthesized motion to: {cfg.output_file}")
    np.savez(
        cfg.output_file,
        joint_pos=synthesized["joint_pos"],
        joint_vel=synthesized["joint_vel"],
        body_pos_w=synthesized["body_pos_w"],
        body_quat_w=synthesized["body_quat_w"],
        body_lin_vel_w=synthesized["body_lin_vel_w"],
        body_ang_vel_w=synthesized["body_ang_vel_w"],
    )
    
    print(f"[INFO] Successfully synthesized motion file: {cfg.output_file}")
    print(f"[INFO] Output contains {n_frames} frames")
    print("\n[INFO] Usage for training:")
    print(f"    uv run train-tracking --task=Mjlab-Tracking-Flat-Unitree-Go2 --motion-file={cfg.output_file}")


def main():
    """Main entry point for the script."""
    cfg = tyro.cli(SynthesizeMotionConfig)
    synthesize_motion(cfg)


if __name__ == "__main__":
    main()

