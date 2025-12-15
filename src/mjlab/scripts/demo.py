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
    Path("/content/mjlab/logs/rsl_rl/go1_velocity/2025-12-14_22-08-52/model_650.pt"),
    Path("/content/mjlab/logs/rsl_rl/go1_velocity/part1/model_650.pt"),
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
  
  # Set time-based velocity commands
  if env_cfg.commands is not None and "twist" in env_cfg.commands:
    twist_term = env.unwrapped.command_manager._terms.get("twist")
    if twist_term is not None:
      # Disable automatic command resampling and updates
      original_compute = twist_term.compute
      original_update = twist_term._update_command
      
      # Define command sequence based on timesteps
      # Format: (start_step, end_step, (vx, vy, wz))
      command_sequence = [
        (0, 125, (0.6, 0.0, 0.0)),      # Forward for 125 steps
        (125, 250, (0.0, 0.4, 0.0)),   # Lateral for 125 steps
        (250, 375, (0.0, 0.0, 0.4)),   # Turning for 125 steps
        (375, 500, (0.5, 0.0, 0.3)),   # Mixed for 125 steps
        (500, float('inf'), (0.0, 0.0, 0.0)),  # Forward after 500 steps
      ]
      
      # Track step counter ourselves since episode_length_buf resets
      step_counter = [0]
      
      def compute_time_based(dt: float):
        # Only update metrics
        twist_term._update_metrics()
        twist_term.time_left[:] = 1e6  # Prevent resampling
        
        # Use our own step counter (increments each time compute is called)
        current_step = step_counter[0]
        step_counter[0] += 1
        
        # Find the command for current timestep
        vx, vy, wz = 0.0, 0.0, 0.0
        for start_step, end_step, (cmd_vx, cmd_vy, cmd_wz) in command_sequence:
          if start_step <= current_step < end_step:
            vx, vy, wz = cmd_vx, cmd_vy, cmd_wz
            break
        
        # Set velocity command for all environments
        twist_term.vel_command_b[:, 0] = vx  # Forward
        twist_term.vel_command_b[:, 1] = vy  # Lateral
        twist_term.vel_command_b[:, 2] = wz  # Turning
        
        # Debug print (only every 25 steps to avoid spam)
        if current_step % 25 == 0:
          print(f"Step {current_step}: Command vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}")

      def update_no_op():
        # Don't modify commands - this prevents heading/standing logic from overriding
        # The original _update_command would modify ang_vel_z for heading and zero commands for standing
        pass
      
      # Reset step counter when environment resets
      original_reset = twist_term.reset
      def reset_with_counter(env_ids):
        result = original_reset(env_ids)
        step_counter[0] = 0  # Reset our counter on environment reset
        return result
      twist_term.reset = reset_with_counter

      twist_term.compute = compute_time_based
      twist_term._update_command = update_no_op
      
      # Set initial command before first step
      # Reset environment to initialize step counter
      env.reset()
      # Now set initial command based on step 0
      initial_vx, initial_vy, initial_wz = 0.6, 0.0, 0.0
      for start_step, end_step, (cmd_vx, cmd_vy, cmd_wz) in command_sequence:
        if start_step <= 0 < end_step:
          initial_vx, initial_vy, initial_wz = cmd_vx, cmd_vy, cmd_wz
          break
      twist_term.vel_command_b[:, 0] = initial_vx
      twist_term.vel_command_b[:, 1] = initial_vy
      twist_term.vel_command_b[:, 2] = initial_wz
      print(f"Initial command set: vx={initial_vx:.2f}, vy={initial_vy:.2f}, wz={initial_wz:.2f}")
      
      print("✅ Configured for time-based commands:")
      for start, end, (vx, vy, wz) in command_sequence:
        if end == float('inf'):
          print(f"   Steps {start}+: vx={vx:.1f}, vy={vy:.1f}, wz={wz:.1f}")
        else:
          print(f"   Steps {start}-{end}: vx={vx:.1f}, vy={vy:.1f}, wz={wz:.1f}")
  
  # Run viewer
  viewer = ViserPlayViewer(env, policy)
  viewer.run()
  env.close()


if __name__ == "__main__":
  main()