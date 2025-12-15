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
    import matplotlib.pyplot as plt
    import numpy as np

    # Lists to store data for plotting
    command_history = []
    measured_history = []
    possible_paths = [
        Path("/content/mjlab/logs/rsl_rl/go1_velocity/part1/model_650.pt"),
    ]

    checkpoint_path = None
    for path in possible_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("Checkpoint not found. Please ensure checkpoint.pt exists in one of:")
        for path in possible_paths:
            print(f"   - {path}")
        return

    checkpoint_path = str(checkpoint_path.resolve())
    print(f"Using checkpoint: {checkpoint_path}")

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
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
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
                (0, 125, (0.6, 0.0, 0.0)),
                (125, 250, (0.0, 0.4, 0.0)),
                (250, 375, (0.0, 0.0, 0.4)),
                (375, 500, (0.5, 0.0, 0.3)),
                (500, float("inf"), (0.0, 0.0, 0.0)),
            ]

            step_counter = [0]

            def compute_time_based(dt: float):
                twist_term._update_metrics()
                twist_term.time_left[:] = 1e6

                current_step = step_counter[0]
                step_counter[0] += 1

                # Find the command for current timestep
                vx, vy, wz = 0.0, 0.0, 0.0
                for i, (start_step, end_step, (cmd_vx, cmd_vy, cmd_wz)) in enumerate(
                    command_sequence
                ):
                    if start_step <= current_step < end_step:
                        if i == 0:
                            progress = float(current_step - start_step) / (
                                end_step - start_step
                            )
                            vx = cmd_vx * progress
                            vy = cmd_vy
                            wz = cmd_wz
                        else:
                            vx, vy, wz = cmd_vx, cmd_vy, cmd_wz
                        break

                # Set velocity command for all environments
                twist_term.vel_command_b[:, 0] = vx  # Forward
                twist_term.vel_command_b[:, 1] = vy  # Lateral
                twist_term.vel_command_b[:, 2] = wz  # Turning

                # Log commanded and measured velocities for plotting
                command_history.append((vx, vy, wz))
                lin_vel_b = env.unwrapped.scene["robot"].data.root_link_lin_vel_b[0].cpu().numpy()
                ang_vel_b = env.unwrapped.scene["robot"].data.root_link_ang_vel_b[0].cpu().numpy()
                measured_history.append((lin_vel_b[0], lin_vel_b[1], ang_vel_b[2]))

                if current_step % 25 == 0:
                    print(
                        f"Step {current_step}: Command vx={vx:.2f}, vy={vy:.2f}, wz={wz:.2f}"
                    )

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
            env.reset()
            initial_vx, initial_vy, initial_wz = 0.0, 0.0, 0.0
            twist_term.vel_command_b[:, 0] = initial_vx
            twist_term.vel_command_b[:, 1] = initial_vy
            twist_term.vel_command_b[:, 2] = initial_wz
            print(
                f"Initial command set: vx={initial_vx:.2f}, vy={initial_vy:.2f}, wz={initial_wz:.2f}"
            )

            print("Configured for time-based commands:")
            for start, end, (vx, vy, wz) in command_sequence:
                if end == float("inf"):
                    print(f"   Steps {start}+: vx={vx:.1f}, vy={vy:.1f}, wz={wz:.1f}")
                else:
                    print(
                        f"   Steps {start}-{end}: vx={vx:.1f}, vy={vy:.1f}, wz={wz:.1f}"
                    )

    num_steps = 500
    print(f"\nRunning simulation for {num_steps} steps to collect data...")

    obs, _ = env.reset()

    for step in range(num_steps):
        action = policy(obs)
        obs, _, _, _ = env.step(action)
        if (step + 1) % 50 == 0:
            print(f"  Simulating step {step + 1}/{num_steps}")

    env.close()

    command_history = np.array(command_history)
    measured_history = np.array(measured_history)

    # Create plots
    time_steps = np.arange(len(command_history))
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Test-Time Command Tracking Performance", fontsize=16)

    # Linear Velocity X
    axs[0].plot(time_steps, command_history[:, 0], "r--", label="Commanded vx")
    axs[0].plot(time_steps, measured_history[:, 0], "b-", label="Measured vx")
    axs[0].set_ylabel("Velocity (m/s)")
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_title("Forward Velocity (vx)")

    # Linear Velocity Y
    axs[1].plot(time_steps, command_history[:, 1], "r--", label="Commanded vy")
    axs[1].plot(time_steps, measured_history[:, 1], "b-", label="Measured vy")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_title("Lateral Velocity (vy)")

    # Angular Velocity Z (Yaw)
    axs[2].plot(time_steps, command_history[:, 2], "r--", label="Commanded ωz")
    axs[2].plot(time_steps, measured_history[:, 2], "b-", label="Measured ωz")
    axs[2].set_xlabel("Time Steps")
    axs[2].set_ylabel("Velocity (rad/s)")
    axs[2].legend()
    axs[2].grid(True)
    axs[2].set_title("Yaw Velocity (ωz)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot to a file
    plot_filename = "command_tracking_plot.png"
    plt.savefig(plot_filename)
    print(f"Plot saved to {plot_filename}")


if __name__ == "__main__":
    main()
