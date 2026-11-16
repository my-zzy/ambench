# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Privileged teacher for regrasping a slider at its current rail position."""

from __future__ import annotations

from typing import Any

import torch

from ..utils import compute_wall_aligned_quat, compute_wall_axes
from .base import BasePolicy, Waypoint


def interpolate_command(start: torch.Tensor, target: torch.Tensor, fraction: float) -> torch.Tensor:
    """Smooth position and interpolate rotation/gripper between absolute EE targets."""
    position, quaternion, gripper = BasePolicy.interpolate(
        Waypoint(0, start[:3], start[3:7], float(start[7])),
        Waypoint(1, target[:3], target[3:7], float(target[7])),
        min(max(fraction, 0.0), 1.0),
    )
    return torch.cat([position, quaternion, start.new_tensor([gripper])])


class PushSliderRecoveryPolicy(BasePolicy):
    """Open, align, approach, regrasp, and push without changing task success or PID.

    The teacher uses simulator object geometry and contact measurements. They
    are label-generation inputs, not additional observations for a learned VLA.
    Call ``start`` after externally induced contact loss, then ``advance`` at 30 Hz.
    """

    def __init__(self, env: Any, contact_sensor: Any, *, target_hz: int = 30) -> None:
        super().__init__(env, inject_noise=False)
        self.contact_sensor = contact_sensor
        self.target_dt = 1.0 / target_hz
        self.slider_body_index = self.env.slider.find_bodies("Slider")[0][0]
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.phase = "idle"
        self.elapsed = 0.0
        self.duration = 0.0
        self.grasp_steps = 0
        self.loss_steps = 0
        self.retries = 0
        self.regrasp_count = 0
        self.command: torch.Tensor | None = None
        self.start_command: torch.Tensor | None = None
        self.target_command: torch.Tensor | None = None

    def start(self, command: torch.Tensor, env_id: int) -> None:
        """Begin a recovery from the current command, preserving command continuity."""
        self.command = command.clone()
        self.normal, wall_y, _ = compute_wall_axes(self.env.wall.data.root_quat_w[env_id])
        self.direction = -wall_y
        self.orientation = compute_wall_aligned_quat(self.normal)
        target = command.clone()
        target[7] = 1.0
        self.set_stage("open", target, 0.5)

    def set_stage(self, phase: str, target: torch.Tensor, duration: float) -> None:
        """Start one bounded motion segment from the preceding command."""
        if self.command is None:
            raise RuntimeError("Call start before advancing recovery")
        self.phase = phase
        self.start_command = self.command.clone()
        self.target_command = target.clone()
        self.duration = duration
        self.elapsed = 0.0
        self.grasp_steps = 0

    def advance(self, obs: dict, env_id: int) -> torch.Tensor:
        if self.command is None or self.phase == "idle":
            raise RuntimeError("Recovery has not been started")
        state = obs["policy"][env_id]
        slider = self.env.slider.data.body_link_pos_w[env_id, self.slider_body_index]
        slider = slider - self.env.scene.env_origins[env_id]
        separation = float(torch.dot(slider - state["ee_pos"], self.normal))
        force = float(torch.linalg.vector_norm(self.contact_sensor.data.net_forces_w[env_id, 0]))
        width = float(state["gripper_width"].item())
        near_handle = abs(float(state["ee_pos"][2] - slider[2])) < 0.025
        near_handle &= abs(float(torch.dot(state["ee_pos"] - slider, self.direction))) < 0.04
        enclosed = width > 0.025 and 0.20 < separation < 0.26 and near_handle
        # Opposing finger forces can cancel in the net slider force at rest.
        # Enclosure only permits a small probe; object motion verifies the grasp.
        self.grasp_steps = self.grasp_steps + 1 if enclosed else 0

        if self.phase == "probe":
            moved = float(state["slider_pos"].item()) - self.probe_start_position >= 0.015
            if moved and force > 0.1 and enclosed:
                self.regrasp_count += 1
                remaining = max(0.0, self.env.cfg.slider_pushed_threshold - float(state["slider_pos"].item()))
                target = self.command.clone()
                target[:3] = slider - 0.23 * self.normal + self.direction * (remaining + 0.05)
                self.set_stage("push", target, max(1.0, (remaining + 0.05) / 0.18))

        if self.phase == "push":
            lost = force < 0.1 and (width < 0.015 or separation > 0.28)
            self.loss_steps = self.loss_steps + 1 if lost else 0
            if self.loss_steps >= 6:
                self.retries += 1
                self.loss_steps = 0
                self.start(self.command, env_id)

        if self.elapsed >= self.duration:
            target = self.command.clone()
            target[3:7] = self.orientation
            if self.phase == "open":
                standoff = max(0.33, float(torch.dot(slider - self.command[:3], self.normal)))
                target[:3] = slider - standoff * self.normal
                target[7] = 1.0
                duration = max(0.6, float(torch.linalg.vector_norm(target[:3] - self.command[:3])) / 0.15)
                self.set_stage("align", target, duration)
            elif self.phase == "align":
                target[:3] = slider - 0.23 * self.normal
                self.set_stage("approach", target, max(1.0, abs(separation - 0.23) / 0.10))
            elif self.phase == "approach":
                target[7] = -1.0
                self.set_stage("close", target, 0.8)
            elif self.phase == "close":
                if self.grasp_steps >= 5:
                    self.probe_start_position = float(state["slider_pos"].item())
                    target[:3] += self.direction * 0.06
                    self.set_stage("probe", target, 0.8)
                elif self.elapsed > 1.5:
                    self.retries += 1
                    self.start(self.command, env_id)
            elif self.phase == "probe" and self.elapsed > 1.4:
                self.retries += 1
                self.start(self.command, env_id)
            elif self.phase == "push":
                # Recompute remaining travel if the object lagged behind the smooth segment.
                remaining = max(0.0, self.env.cfg.slider_pushed_threshold - float(state["slider_pos"].item()))
                target[:3] = slider - 0.23 * self.normal + self.direction * (remaining + 0.05)
                self.set_stage("push", target, max(1.0, (remaining + 0.05) / 0.18))

        if self.retries > 2:
            self.phase = "failed"
            return self.command.unsqueeze(0)
        self.elapsed += self.target_dt
        self.command = interpolate_command(self.start_command, self.target_command, self.elapsed / self.duration)
        return self.command.unsqueeze(0)
