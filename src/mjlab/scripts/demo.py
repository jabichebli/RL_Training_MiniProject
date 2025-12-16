"""
Script to run a demo with a pretrained TRACKING policy.

Key properties:
- Uses ONLY the motion reference (motion.npz)
- No velocity / twist / time-based commands
- Policy behavior matches training exactly
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
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from mjlab.viewer import ViserPlayViewer


def main() -> None:
    import numpy as np
    import matplotlib.pyplot as plt

    # ---------------------------------------------------------------------
    # Locate checkpoint
    # ---------------------------------------------------------------------
    possible_paths = [
        Path("/content/mjlab/logs/rsl_rl/go2_tracking/2025-12-16_00-35-34/model_500.pt"),
    ]

    checkpoint_path = None
    for path in possible_paths:
        if path.exists():
            checkpoint_path = path
            break

    if checkpoint_path is None:
        print("Checkpoint not found. Checked:")
        for p in possible_paths:
            print(f"  - {p}")
        return

    checkpoint_path = str(checkpoint_path.resolve())
    print(f"Using checkpoint: {checkpoint_path}")

    # ---------------------------------------------------------------------
    # CLI arguments
    # ---------------------------------------------------------------------
    args = tyro.cli(
        PlayConfig,
        default=PlayConfig(
            checkpoint_file=checkpoint_path,
            num_envs=1,
            viewer="viser",
        ),
    )

    # ---------------------------------------------------------------------
    # Torch + device
    # ---------------------------------------------------------------------
    configure_torch_backends()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Load task (MUST be a tracking task)
    # ---------------------------------------------------------------------
    task = os.environ.get(
        "MJLAB_DEMO_TASK",
        "Mjlab-Tracking-Flat-Unitree-Go2",
    )

    env_cfg = load_env_cfg(task)
    _apply_play_env_overrides(env_cfg, args.motion_command_sampling_mode)
    agent_cfg = load_rl_cfg(task)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    # ---------------------------------------------------------------------
    # Ensure motion file is provided
    # ---------------------------------------------------------------------
    if (
        env_cfg.commands is None
        or "motion" not in env_cfg.commands
        or not isinstance(env_cfg.commands["motion"], MotionCommandCfg)
    ):
        raise RuntimeError(
            "This demo requires a TRACKING task with MotionCommandCfg."
        )

    if args.motion_file is None or args.motion_file == "":
        raise ValueError(
            "Tracking demo requires --motion_file /path/to/motion.npz"
        )

    motion_cmd: MotionCommandCfg = env_cfg.commands["motion"]
    motion_cmd.motion_file = str(
        Path(args.motion_file).expanduser().resolve()
    )

    print(f"Using motion file: {motion_cmd.motion_file}")

    # ---------------------------------------------------------------------
    # Environment
    # ---------------------------------------------------------------------
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ---------------------------------------------------------------------
    # Load policy
    # ---------------------------------------------------------------------
    log_dir = Path(checkpoint_path).parent
    runner = OnPolicyRunner(
        env,
        asdict(agent_cfg),
        log_dir=str(log_dir),
        device=device,
    )
    runner.load(checkpoint_path, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # ---------------------------------------------------------------------
    # Data logging (measured vs reference body pose)
    # ---------------------------------------------------------------------
    measured_pose = []
    reference_pose = []

    # ---------------------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------------------
    num_steps = 2000
    print(f"Running tracking demo for {num_steps} steps...")

    obs, _ = env.reset()

    for step in range(num_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env.step(action)

        robot_data = env.unwrapped.scene["robot"].data
        motion_term = env.unwrapped.command_manager._terms["motion"]

        # measured_pose.append(
        #     robot_data.root_link_pose_w[0].cpu().numpy()
        # )
        # reference_pose.append(
        #     motion_term.motion_ref.root_pose_w[0].cpu().numpy()
        # )

        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}/{num_steps}")

    viewer = ViserPlayViewer(env, policy)
    viewer.run()
    env.close()

    # measured_pose = np.asarray(measured_pose)
    # reference_pose = np.asarray(reference_pose)

    # ---------------------------------------------------------------------
    # Plot body height and pitch tracking
    # ---------------------------------------------------------------------
    # time = np.arange(len(measured_pose))

    # fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # fig.suptitle("Tracking Performance (Body Pose)", fontsize=16)

    # # Z position
    # axs[0].plot(time, reference_pose[:, 2], "r--", label="Reference z")
    # axs[0].plot(time, measured_pose[:, 2], "b-", label="Measured z")
    # axs[0].set_ylabel("Height (m)")
    # axs[0].legend()
    # axs[0].grid(True)
    # axs[0].set_title("Body Height")

    # # Pitch (quaternion → approximate pitch)
    # def quat_to_pitch(q):
    #     qw, qx, qy, qz = q
    #     return np.arctan2(
    #         2 * (qw * qy - qx * qz),
    #         1 - 2 * (qy * qy + qz * qz),
    #     )

    # ref_pitch = np.array([quat_to_pitch(p[3:7]) for p in reference_pose])
    # meas_pitch = np.array([quat_to_pitch(p[3:7]) for p in measured_pose])

    # axs[1].plot(time, ref_pitch, "r--", label="Reference pitch")
    # axs[1].plot(time, meas_pitch, "b-", label="Measured pitch")
    # axs[1].set_ylabel("Pitch (rad)")
    # axs[1].set_xlabel("Time steps")
    # axs[1].legend()
    # axs[1].grid(True)
    # axs[1].set_title("Body Pitch")

    # plt.tight_layout(rect=[0, 0, 1, 0.96])

    # plot_file = "tracking_pose_plot.png"
    # plt.savefig(plot_file)
    # print(f"Saved plot to {plot_file}")


if __name__ == "__main__":
    main()
