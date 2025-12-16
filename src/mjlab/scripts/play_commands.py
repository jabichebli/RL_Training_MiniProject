import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import PlayConfig, _apply_play_env_overrides
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import ViserPlayViewer, NativeMujocoViewer


def main() -> None:
    possible_paths = [
        Path("/content/mjlab/logs/rsl_rl/go1_velocity/part1/model_650.pt"),
    ]

    checkpoint_path = None
    for path in possible_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("Checkpoint not found. Please ensure a trained model exists in one of:")
        for path in possible_paths:
            print(f"   - {path}")
        return

    checkpoint_path = str(checkpoint_path.resolve())
    print(f"Using checkpoint: {checkpoint_path}")

    @dataclass
    class ScriptConfig:
        checkpoint_file: str
        device: str | None = None
        num_envs: int = 1
        viewer: Literal["native", "viser"] = "viser"
        motion_command_sampling_mode: Literal["start", "uniform"] = "start"

    args = tyro.cli(
        ScriptConfig,
        default=ScriptConfig(
            checkpoint_file=checkpoint_path,
        ),
    )

    # Setup environment and policy
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
            original_update = twist_term._update_command

            # Define command sequence based on timesteps
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

            def update_no_op():
                pass

            original_reset = twist_term.reset

            def reset_with_counter(env_ids):
                result = original_reset(env_ids)
                step_counter[0] = 0
                return result

            twist_term.reset = reset_with_counter
            twist_term.compute = compute_time_based
            twist_term._update_command = update_no_op

            env.reset()
            twist_term.vel_command_b[:, 0] = 0.0
            twist_term.vel_command_b[:, 1] = 0.0
            twist_term.vel_command_b[:, 2] = 0.0

    print(f"\nRunning in visualization mode (viewer='{args.viewer}')...")
    if args.viewer == "native":
        viewer = NativeMujocoViewer(env, policy)
        viewer.run()
    elif args.viewer == "viser":
        viewer = ViserPlayViewer(env, policy)
        viewer.run()

    env.close()


if __name__ == "__main__":
    main()
