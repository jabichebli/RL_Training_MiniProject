"""Script to evaluate RL agent command tracking performance."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
  checkpoint_file: str
  """Path to the model checkpoint file."""
  device: str = "cuda:0"
  """Device to run evaluation on."""
  output_dir: str | None = None
  """Output directory for plots. If None, uses checkpoint directory."""


def run_eval(task_id: str, cfg: EvalConfig) -> None:
  """Run command tracking evaluation."""
  configure_torch_backends()

  device = cfg.device

  # Load configs
  env_cfg = load_env_cfg(task_id)
  env_cfg.sim.device = device
  env_cfg.scene.num_envs = 1

  agent_cfg = load_rl_cfg(task_id)

  # Create environment
  env = ManagerBasedRlEnv(cfg=env_cfg)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  # Create runner
  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  log_dir = os.path.dirname(cfg.checkpoint_file)
  runner = runner_cls(env, asdict(agent_cfg), log_dir=log_dir, device=device)
  runner.load(cfg.checkpoint_file)
  policy = runner.alg.actor_critic.actor
  policy.eval()

  obs, _ = env.reset()
  robot = env.unwrapped.scene["robot"]
  twist_term = env.unwrapped.command_manager._terms["twist"]

  # Command sequence: (lin_vel_x, lin_vel_y, ang_vel_z) for 125 steps each
  commands = [
    (0.5, 0.0, 0.0),   # Forward
    (0.0, 0.0, 0.5),   # Turn right
    (-0.5, 0.0, 0.0),  # Backward
    (0.0, 0.0, -0.5),  # Turn left
    (0.5, 0.3, 0.0),   # Forward + lateral
    (0.0, 0.0, 0.0),   # Stop
  ]

  # Collect data
  cmd_x, cmd_y, cmd_z = [], [], []
  actual_x, actual_y, actual_z = [], [], []

  print(f"Running evaluation with {len(commands)} command segments...")

  for cmd_idx, (cmd_x_val, cmd_y_val, cmd_z_val) in enumerate(commands):
    print(
      f"  Command {cmd_idx + 1}/{len(commands)}: lin_vel=({cmd_x_val:.1f}, {cmd_y_val:.1f}), ang_vel={cmd_z_val:.1f}"
    )

    for _ in range(125):
      twist_term.vel_command_b[0, 0] = cmd_x_val
      twist_term.vel_command_b[0, 1] = cmd_y_val
      twist_term.vel_command_b[0, 2] = cmd_z_val

      with torch.no_grad():
        actions = policy(obs)
      obs, _, _, _, _ = env.step(actions)

      cmd_x.append(cmd_x_val)
      cmd_y.append(cmd_y_val)
      cmd_z.append(cmd_z_val)

      vel = robot.data.root_link_lin_vel_b[0, :].cpu().numpy()
      ang = robot.data.root_link_ang_vel_b[0, :].cpu().numpy()
      actual_x.append(vel[0])
      actual_y.append(vel[1])
      actual_z.append(ang[2])

  env.close()

  # Plot
  fig, axes = plt.subplots(3, 1, figsize=(14, 10))
  time = np.arange(len(cmd_x))

  axes[0].plot(time, cmd_x, "k--", label="Commanded", linewidth=2, alpha=0.5)
  axes[0].plot(time, actual_x, "b-", label="Actual", linewidth=1.5)
  axes[0].set_ylabel("Linear Velocity X (m/s)")
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)

  axes[1].plot(time, cmd_y, "k--", label="Commanded", linewidth=2, alpha=0.5)
  axes[1].plot(time, actual_y, "g-", label="Actual", linewidth=1.5)
  axes[1].set_ylabel("Linear Velocity Y (m/s)")
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)

  axes[2].plot(time, cmd_z, "k--", label="Commanded", linewidth=2, alpha=0.5)
  axes[2].plot(time, actual_z, "r-", label="Actual", linewidth=1.5)
  axes[2].set_xlabel("Time Steps")
  axes[2].set_ylabel("Angular Velocity Z (rad/s)")
  axes[2].legend()
  axes[2].grid(True, alpha=0.3)

  plt.tight_layout()

  # Determine output directory
  if cfg.output_dir:
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
  else:
    output_dir = Path(cfg.checkpoint_file).parent

  output_path = output_dir / "command_tracking.png"
  plt.savefig(output_path, dpi=150)
  print(f"✅ Saved plot to {output_path}")

  # Print errors
  error_x = np.mean(np.abs(np.array(cmd_x) - np.array(actual_x)))
  error_y = np.mean(np.abs(np.array(cmd_y) - np.array(actual_y)))
  error_z = np.mean(np.abs(np.array(cmd_z) - np.array(actual_z)))
  print(f"\nTracking Errors (MAE):")
  print(f"  Lin Vel X: {error_x:.4f} m/s")
  print(f"  Lin Vel Y: {error_y:.4f} m/s")
  print(f"  Ang Vel Z: {error_z:.4f} rad/s")


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments
  args = tyro.cli(
    EvalConfig,
    args=remaining_args,
    default=EvalConfig(checkpoint_file=""),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  if not args.checkpoint_file or not os.path.exists(args.checkpoint_file):
    print(f"Error: Checkpoint file not found: {args.checkpoint_file}")
    sys.exit(1)

  run_eval(chosen_task, args)


if __name__ == "__main__":
  main()
