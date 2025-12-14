"""Script to run a velocity demo with a pretrained policy.

This demo uses a local checkpoint and runs the go1 robot with constant forward motion.
"""

import os
from dataclasses import asdict
from pathlib import Path

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig, _apply_play_env_overrides
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer


def main() -> None:
  """Run demo with pretrained velocity policy."""
  print("🎮 Setting up MJLab demo with pretrained velocity policy...")

  # Use local checkpoint - looks for checkpoint.pt in common locations
  possible_paths = [
    Path("checkpoint.pt"),
    Path("logs/rsl_rl/go1_velocity/2025-11-30_01-39-44/checkpoint.pt"),
    Path("logs/rsl_rl/go1_velocity/2025-11-30_01-31-18/checkpoint.pt"),
    Path("logs/rsl_rl/go1_velocity/2025-11-30_01-39-44/model_999.pt"),  # Fallback
    Path("logs/rsl_rl/go1_velocity/2025-11-30_01-31-18/model_999.pt"),  # Fallback
  ]
  
  checkpoint_path = None
  for path in possible_paths:
    if path.exists():
      checkpoint_path = path
      break
  
  if checkpoint_path is None:
    print("❌ Checkpoint not found. Please ensure checkpoint.pt exists in one of:")
    for path in possible_paths:
      print(f"   - {path}")
    return

  checkpoint_path = str(checkpoint_path.resolve())
  print(f"✅ Using checkpoint: {checkpoint_path}")

  args = tyro.cli(
    PlayConfig,
    default=PlayConfig(
      checkpoint_file=checkpoint_path,
      num_envs=1,
      viewer="viser",
    ),
  )
  
  # Setup environment and policy (similar to run_play)
  configure_torch_backends()
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  
  task = "Mjlab-Velocity-Flat-Unitree-Go1"
  env_cfg = load_env_cfg(task)
  _apply_play_env_overrides(env_cfg, args.motion_command_sampling_mode)
  agent_cfg = load_rl_cfg(task)
  
  if args.num_envs is not None:
    env_cfg.scene.num_envs = args.num_envs
  
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  
  # Load policy
  log_dir = Path(checkpoint_path).parent
  runner = OnPolicyRunner(
    env, asdict(agent_cfg), log_dir=str(log_dir), device=device
  )
  runner.load(checkpoint_path, map_location=device)
  policy = runner.get_inference_policy(device=device)
  
  # Set constant forward velocity command
  if env_cfg.commands is not None and "twist" in env_cfg.commands:
    twist_term = env.unwrapped.command_manager._terms.get("twist")
    if twist_term is not None:
      # Disable automatic command resampling and updates
      original_compute = twist_term.compute
      original_update = twist_term._update_command
      
      # Constant forward velocity: 0.6 m/s
      forward_vel = 0.6
      
      def compute_constant_forward(dt: float):
        # Only update metrics, set constant forward command
        twist_term._update_metrics()
        twist_term.time_left[:] = 1e6  # Prevent resampling
        # Set constant forward velocity
        twist_term.vel_command_b[:, 0] = forward_vel  # Forward
        twist_term.vel_command_b[:, 1] = 0.0  # No lateral
        twist_term.vel_command_b[:, 2] = 0.0  # No turning

      def update_no_op():
        # Don't modify commands
        pass

      twist_term.compute = compute_constant_forward
      twist_term._update_command = update_no_op
      
      # Set initial command
      twist_term.vel_command_b[:, 0] = forward_vel
      twist_term.vel_command_b[:, 1] = 0.0
      twist_term.vel_command_b[:, 2] = 0.0
      
      print(f"✅ Configured for constant forward motion ({forward_vel} m/s)")
  
  # Run viewer
  viewer = ViserPlayViewer(env, policy)
  viewer.run()
  env.close()


if __name__ == "__main__":
  main()
