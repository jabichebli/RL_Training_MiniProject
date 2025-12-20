"""Unitree Go2 flat terrain tracking configuration.

This module provides factory functions that create complete ManagerBasedRlEnvCfg
instances for the Go2 robot tracking task on flat terrain.
"""

from copy import deepcopy

from mjlab.asset_zoo.robots import (
  GO2_ACTION_SCALE,
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.tracking_env_cfg import create_tracking_env_cfg
from mjlab.utils.retval import retval


@retval
def UNITREE_GO2_FLAT_TRACKING_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain tracking configuration."""
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="trunk", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  # Foot contact sensor for ground reaction force observations
  # Match the foot collision geoms (FR_foot_collision, etc.) against terrain
  foot_contact_cfg = ContactSensorCfg(
    name="foot_contact",
    primary=ContactMatch(mode="geom", pattern=r"^[FR][LR]_foot_collision$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=4,  # One slot per foot
  )
  # Scale action scale for difficult maneuvers (backflip)
  # Increase multiplier to allow more force generation for takeoff
  action_scale_multiplier = 1.0  # 1.0 = default, 2.0 = double the action range
  scaled_action_scale = {
    k: v * action_scale_multiplier for k, v in GO2_ACTION_SCALE.items()
  }
  return create_tracking_env_cfg(
    robot_cfg=get_go2_robot_cfg(),
    action_scale=scaled_action_scale,
    viewer_body_name="trunk",
    motion_file="",
    anchor_body_name="trunk",
    body_names=(
      "trunk",
      "FR_hip",
      "FR_thigh",
      "FR_calf",
      "FL_hip",
      "FL_thigh",
      "FL_calf",
      "RR_hip",
      "RR_thigh",
      "RR_calf",
      "RL_hip",
      "RL_thigh",
      "RL_calf",
    ),
    foot_friction_geom_names=(r"^[FR][LR]_foot_collision$",),
    ee_body_names=(
      "FR_calf",
      "FL_calf",
      "RR_calf",
      "RL_calf",
    ),
    base_com_body_name="trunk",
    sensors=(self_collision_cfg, foot_contact_cfg),
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
  )


@retval
def UNITREE_GO2_FLAT_TRACKING_NO_STATE_ESTIMATION_ENV_CFG() -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain tracking config without state estimation.

  This variant disables motion_anchor_pos_b and base_lin_vel observations,
  simulating the lack of state estimation.
  """
  cfg = deepcopy(UNITREE_GO2_FLAT_TRACKING_ENV_CFG)
  assert "policy" in cfg.observations
  cfg.observations["policy"].terms.pop("motion_anchor_pos_b")
  cfg.observations["policy"].terms.pop("base_lin_vel")
  return cfg

