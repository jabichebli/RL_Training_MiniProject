"""Script to train RL agent with RSL-RL for tracking tasks using local motion files."""

import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import dump_yaml, get_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainTrackingConfig:
  env: Any
  agent: RslRlOnPolicyRunnerCfg
  motion_file: str
  pose_only_rewards: bool = False
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  distributed: bool = False


def run_train_tracking(task_id: str, cfg: TrainTrackingConfig) -> None:
  configure_torch_backends()

  # Multi-GPU training configuration.
  device = cfg.device
  if cfg.distributed:
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"

    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank
    cfg.env.seed = seed
    cfg.agent.seed = seed

    print(
      f"[INFO] Multi-GPU training enabled: local_rank={local_rank}, device={device}, seed={seed}"
    )

  # Verify this is a tracking task
  if cfg.env.commands is None or "motion" not in cfg.env.commands:
    raise ValueError(
      f"Task '{task_id}' is not a tracking task. This script only supports tracking tasks."
    )

  motion_cmd = cfg.env.commands["motion"]
  if not isinstance(motion_cmd, MotionCommandCfg):
    raise ValueError(
      f"Task '{task_id}' does not have a MotionCommandCfg. This script only supports tracking tasks."
    )

  # Set motion file from local path
  motion_file_path = Path(cfg.motion_file)
  if not motion_file_path.exists():
    raise FileNotFoundError(f"Motion file not found: {motion_file_path}")
  if not motion_file_path.is_file():
    raise ValueError(f"Motion file path is not a file: {motion_file_path}")

  print(f"[INFO] Using local motion file: {motion_file_path}")
  motion_cmd.motion_file = str(motion_file_path.resolve())

  # Optional: filter rewards based on pose_only_rewards flag.
  if cfg.pose_only_rewards:
    # Keep only body position + orientation (no joint tracking, no velocities).
    if cfg.env.rewards is None:
      raise ValueError("env.rewards is None; cannot apply pose_only_rewards.")
    keep = {"motion_body_pos", "motion_body_ori"}
    cfg.env.rewards = {k: v for k, v in cfg.env.rewards.items() if k in keep}
    print(f"[INFO] pose_only_rewards enabled. Keeping rewards: {sorted(cfg.env.rewards.keys())}")
  else:
    # Default: include all tracking rewards (body pose + joint positions + velocities).
    print(f"[INFO] Using all tracking rewards including joint positions.")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  # Remove strict termination conditions that cause frequent resets during training.
  if cfg.env.terminations is not None:
    removed = []
    for k in ("anchor_pos", "anchor_ori", "ee_body_pos"):
      if k in cfg.env.terminations:
        cfg.env.terminations.pop(k)
        removed.append(k)
    if removed:
      print(f"[INFO] Removed termination conditions: {removed}")

  # Specify directory for logging experiments.
  log_root_path = Path("logs") / "rsl_rl" / cfg.agent.experiment_name
  log_root_path.resolve()
  print(f"[INFO] Logging experiment in directory: {log_root_path}")
  log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if cfg.agent.run_name:
    log_dir += f"_{cfg.agent.run_name}"
  log_dir = log_root_path / log_dir

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  resume_path = (
    get_checkpoint_path(log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if cfg.agent.resume
    else None
  )

  if cfg.video:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  # For tracking tasks, we don't pass registry_name since we're using local files
  runner = runner_cls(env, agent_cfg, str(log_dir), device)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
  dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  # Filter to only tracking tasks
  tracking_tasks = [
    task
    for task in all_tasks
    if task.startswith("Mjlab-Tracking-")
  ]

  if not tracking_tasks:
    print("[ERROR] No tracking tasks found in registry.")
    print(f"Available tasks: {all_tasks}")
    sys.exit(1)

  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(tracking_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  env_cfg = load_env_cfg(chosen_task)
  agent_cfg = load_rl_cfg(chosen_task)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  args = tyro.cli(
    TrainTrackingConfig,
    args=remaining_args,
    default=TrainTrackingConfig(env=env_cfg, agent=agent_cfg, motion_file=""),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  # Validate motion_file is provided
  if not args.motion_file:
    raise ValueError(
      "Must provide --motion-file with path to local motion.npz file.\n"
      f"Example: {sys.argv[0]} {chosen_task} --motion-file ./backflip_motion.npz"
    )

  del env_cfg, agent_cfg, remaining_args

  run_train_tracking(chosen_task, args)


if __name__ == "__main__":
  main()

