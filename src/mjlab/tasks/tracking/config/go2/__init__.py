from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  UNITREE_GO2_FLAT_TRACKING_ENV_CFG,
  UNITREE_GO2_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
)
from .rl_cfg import UNITREE_GO2_TRACKING_PPO_RUNNER_CFG

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-Go2",
  env_cfg=UNITREE_GO2_FLAT_TRACKING_ENV_CFG,
  rl_cfg=UNITREE_GO2_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-Go2-No-State-Estimation",
  env_cfg=UNITREE_GO2_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG,
  rl_cfg=UNITREE_GO2_TRACKING_PPO_RUNNER_CFG,
  runner_cls=MotionTrackingOnPolicyRunner,
)

