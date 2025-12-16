import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro
from rsl_rl.runners import OnPolicyRunner
import numpy as np
import matplotlib.pyplot as plt

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.play import _apply_play_env_overrides
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor


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
        motion_command_sampling_mode: Literal["start", "uniform"] = "start"

    args = tyro.cli(
        ScriptConfig,
        default=ScriptConfig(
            checkpoint_file=checkpoint_path,
        ),
    )

    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    task = "Mjlab-Velocity-Flat-Unitree-Go1"
    env_cfg = load_env_cfg(task)
    _apply_play_env_overrides(env_cfg, "start")
    agent_cfg = load_rl_cfg(task)

    env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    log_dir = Path(checkpoint_path).parent
    runner = OnPolicyRunner(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
    runner.load(checkpoint_path, map_location=device)
    policy = runner.get_inference_policy(device=device)

    command_sequence = [
        (0, 125, (0.6, 0.0, 0.0)),
        (125, 250, (0.0, 0.4, 0.0)),
        (250, 375, (0.0, 0.0, 0.4)),
        (375, 500, (0.5, 0.0, 0.3)),
        (500, float("inf"), (0.0, 0.0, 0.0)),
    ]

    feet_sensor: ContactSensor = env.unwrapped.scene["feet_ground_contact"]
    tracked_geom_names = sorted(
        [s.primary_name for s in feet_sensor._slots if s.field_name == "found"]
    )

    def geom_to_site_name(geom_name):
        return geom_name.split("_")[0]

    foot_site_names_ordered = [geom_to_site_name(g) for g in tracked_geom_names]
    robot_entity = env.unwrapped.scene["robot"]
    all_site_names = list(robot_entity.site_names)
    foot_site_indices = torch.tensor(
        [all_site_names.index(name) for name in foot_site_names_ordered],
        device=device,
        dtype=torch.long,
    )
    print(f"Dynamically mapped contact geoms to sites: {foot_site_names_ordered}")
    fl_foot_idx = all_site_names.index("FL")
    print(f"Tracking site 'FL' at index {fl_foot_idx} for foot trajectory.")

    linear_error_history = []
    slip_velocity_history = []
    fl_foot_pos_history = []

    num_steps = 500
    print(f"\nRunning simulation for {num_steps} steps to collect analysis data...")

    obs, _ = env.reset()

    for step in range(num_steps):
        current_step = step
        vx, vy, wz = 0.0, 0.0, 0.0
        for start_step, end_step, cmd in command_sequence:
            if start_step <= current_step < end_step:
                if start_step == 0:
                    progress = float(current_step - start_step) / (
                        end_step - start_step
                    )
                    vx = cmd[0] * progress
                    vy = cmd[1]
                    wz = cmd[2]
                else:
                    vx, vy, wz = cmd
                break

        twist_term = env.unwrapped.command_manager._terms.get("twist")
        if twist_term:
            twist_term.vel_command_b[:, 0] = vx
            twist_term.vel_command_b[:, 1] = vy
            twist_term.vel_command_b[:, 2] = wz

        action = policy(obs)
        obs, _, _, _ = env.step(action)

        robot: Entity = env.unwrapped.scene["robot"]

        command_lin_vel = torch.tensor([vx, vy], device=device)
        measured_lin_vel = robot.data.root_link_lin_vel_b[0, :2]
        error = torch.norm(command_lin_vel - measured_lin_vel).item()
        linear_error_history.append(error)

        in_contact = (feet_sensor.data.found[0] > 0).float()
        all_site_vel_xy = robot.data.site_lin_vel_w[0, :, :2]
        foot_vel_xy = all_site_vel_xy[foot_site_indices]
        vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
        num_in_contact = torch.sum(in_contact)
        mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
            num_in_contact, min=1.0
        )
        slip_velocity_history.append(mean_slip_vel.item())

        fl_foot_pos = robot.data.site_pos_w[0, fl_foot_idx, :].cpu().numpy()
        fl_foot_pos_history.append(fl_foot_pos)

        if (step + 1) % 50 == 0:
            print(f"  Simulating step {step + 1}/{num_steps}")

    env.close()

    print("Simulation finished. Generating analysis plots...")

    fl_foot_pos_history = np.array(fl_foot_pos_history)
    time_steps = np.arange(num_steps)

    plt.figure(figsize=(8, 6))
    forward_walk_data = fl_foot_pos_history[:125]
    plt.plot(forward_walk_data[:, 0], forward_walk_data[:, 2])
    plt.title("Front-Left Foot Trajectory")
    plt.xlabel("X Position")
    plt.ylabel("Z Position")
    plt.grid(True)
    plt.axis("equal")
    plot1_filename = "part2_foot_trajectory.png"
    plt.savefig(plot1_filename)
    print(f"Foot trajectory plot saved to {plot1_filename}")

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Controller Performance Metrics", fontsize=16)

    axs[0].plot(
        time_steps, linear_error_history, label="Linear Velocity Tracking Error"
    )
    axs[0].set_ylabel("Error (m/s)")
    axs[0].grid(True)
    axs[0].set_title("Linear Velocity Tracking Error")

    axs[1].plot(
        time_steps,
        slip_velocity_history,
        label="Mean Foot Slip Velocity",
        color="orange",
    )
    axs[1].set_xlabel("Time Steps")
    axs[1].set_ylabel("Velocity (m/s)")
    axs[1].grid(True)
    axs[1].set_title("Mean Foot Slip Velocity")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot2_filename = "part2_slip_vel_error.png"
    plt.savefig(plot2_filename)
    print(f"Performance metrics plot saved to {plot2_filename}")


if __name__ == "__main__":
    main()

