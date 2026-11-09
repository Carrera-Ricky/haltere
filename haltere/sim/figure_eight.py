"""Headless two-post figure-eight task for the differentiable quad simulator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from .quad import QuadState
from .tasks import HoverTask, HoverTaskConfig, observe_from_sensors


def load_route(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if data.get("pattern") != "two-post-figure-eight":
        raise ValueError(f"{path} is not a two-post figure-eight route")
    waypoints = np.asarray(data.get("waypoints"), dtype=np.float32)
    posts = np.asarray(data.get("posts"), dtype=np.float32)
    if waypoints.ndim != 2 or waypoints.shape[1] != 3 or len(waypoints) < 2:
        raise ValueError("route must contain at least two 3D waypoints")
    if posts.shape != (2, 3):
        raise ValueError("route must contain exactly two 3D post centres")
    return {**data, "waypoints": waypoints, "posts": posts}


class FigureEightTask:
    """Follow a known route while recording clearance from two posts."""

    channels = HoverTask.channels

    def __init__(self, cfg: HoverTaskConfig, sim, route: dict, B: int, device,
                 speed: float = 1.5, post_height: float = 3.0,
                 crossbar_width: float = 2.0, tpost_radius: float = 0.35):
        self.cfg, self.sim, self.B, self.device = cfg, sim, B, device
        self.speed = float(speed)
        self.post_height = float(post_height)
        self.crossbar_width = float(crossbar_width)
        self.tpost_radius = float(tpost_radius)
        self.points = torch.as_tensor(route["waypoints"], dtype=torch.float32, device=device)
        self.posts = torch.as_tensor(route["posts"], dtype=torch.float32, device=device)
        vertical_start = self.posts.clone()
        vertical_start[:, 2] = 0.0
        vertical_end = self.posts.clone()
        vertical_end[:, 2] = torch.maximum(
            torch.full_like(self.posts[:, 2], self.post_height), self.posts[:, 2] + 0.25
        )
        bar_start = self.posts.clone()
        bar_start[:, 0] -= self.crossbar_width / 2.0
        bar_end = self.posts.clone()
        bar_end[:, 0] += self.crossbar_width / 2.0
        self.segment_start = torch.cat([vertical_start, bar_start], dim=0)
        self.segment_end = torch.cat([vertical_end, bar_end], dim=0)
        seg = (self.points[1:] - self.points[:-1]).norm(dim=1)
        self.arc = torch.cat([torch.zeros(1, device=device), torch.cumsum(seg, dim=0)])
        self.length = float(self.arc[-1])
        self.progress = torch.zeros(B, device=device)
        self.target = self.points[0].expand(B, -1).clone()
        self.prev_action = torch.zeros(B, 4, device=device)
        self.t = torch.zeros(B, dtype=torch.long, device=device)
        self.min_clearance = torch.full((B,), float("inf"), device=device)
        self.hit_post = torch.zeros(B, dtype=torch.bool, device=device)
        self.distance_sum = torch.zeros(B, device=device)
        self.speed_sum = torch.zeros(B, device=device)
        self.within_sum = torch.zeros(B, device=device)
        self.cross_track_sum = torch.zeros(B, device=device)
        self.max_cross_track = torch.zeros(B, device=device)
        self.samples = 0

    def _point_at(self, progress: torch.Tensor) -> torch.Tensor:
        s = progress.clamp(0.0, self.length)
        idx = torch.bucketize(s, self.arc[1:]).clamp(max=len(self.points) - 2)
        s0 = self.arc[idx]
        f = ((s - s0) / (self.arc[idx + 1] - s0).clamp_min(1e-6))[:, None]
        return self.points[idx] * (1.0 - f) + self.points[idx + 1] * f

    def reset_all(self, difficulty: float | None = None) -> QuadState:
        p = self.points[0].expand(self.B, -1).clone()
        quat = torch.zeros(self.B, 4, device=self.device)
        quat[:, 0] = 1.0
        motor = torch.full((self.B, 4), self.sim.hover_command(), device=self.device)
        self.progress.zero_()
        self.target = p.clone()
        self.prev_action.zero_()
        self.t.zero_()
        self.min_clearance.fill_(float("inf"))
        self.hit_post.zero_()
        self.distance_sum.zero_()
        self.speed_sum.zero_()
        self.within_sum.zero_()
        self.cross_track_sum.zero_()
        self.max_cross_track.zero_()
        self.samples = 0
        return QuadState(p, torch.zeros_like(p), quat, torch.zeros_like(p), motor,
                         torch.zeros(self.B, dtype=torch.bool, device=self.device))

    def observe(self, st: QuadState) -> dict[str, torch.Tensor]:
        sensors = self.sim.sensors(st)
        return observe_from_sensors(sensors, self.target, st.motor.mean(dim=1, keepdim=True), self.cfg)

    def _clearance(self, pos: torch.Tensor) -> torch.Tensor:
        """Surface clearance from the nearest vertical pole or crossbar."""
        segment = self.segment_end - self.segment_start
        offset = pos[:, None, :] - self.segment_start[None, :, :]
        fraction = (offset * segment[None, :, :]).sum(dim=2)
        fraction = fraction / segment.square().sum(dim=1).clamp_min(1e-6)[None, :]
        fraction = fraction.clamp(0.0, 1.0)
        closest = self.segment_start[None, :, :] + fraction[:, :, None] * segment[None, :, :]
        centreline_distance = (pos[:, None, :] - closest).norm(dim=2).min(dim=1).values
        return centreline_distance - self.tpost_radius

    def _cross_track_distance(self, pos: torch.Tensor) -> torch.Tensor:
        """Shortest 3D distance from each drone position to the waypoint polyline."""
        start, end = self.points[:-1], self.points[1:]
        segment = end - start
        offset = pos[:, None, :] - start[None, :, :]
        fraction = (offset * segment[None, :, :]).sum(dim=2)
        fraction = fraction / segment.square().sum(dim=1).clamp_min(1e-6)[None, :]
        fraction = fraction.clamp(0.0, 1.0)
        closest = start[None, :, :] + fraction[:, :, None] * segment[None, :, :]
        return (pos[:, None, :] - closest).norm(dim=2).min(dim=1).values

    def cost(self, st: QuadState, action: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        sensors = self.sim.sensors(st, noise=False)
        pos_err = (st.pos - self.target).pow(2).sum(1)
        vel = st.vel.pow(2).sum(1)
        rate = st.omega.pow(2).sum(1)
        tilt = 1.0 - sensors["up"]
        act = action.pow(2).sum(1)
        dact = (action - self.prev_action).pow(2).sum(1)
        floor = (c.min_height - st.pos[:, 2]).clamp(min=0.0).pow(2)
        clearance = self._clearance(st.pos)
        cross_track = self._cross_track_distance(st.pos)
        obstacle = (-clearance).clamp(min=0.0).pow(2)
        self.min_clearance = torch.minimum(self.min_clearance, clearance.detach())
        self.hit_post |= clearance < 0.0
        distance = (st.pos - self.target).norm(dim=1)
        self.distance_sum += distance.detach()
        self.speed_sum += st.vel.norm(dim=1).detach()
        self.within_sum += (distance < 0.5).float().detach()
        self.cross_track_sum += cross_track.detach()
        self.max_cross_track = torch.maximum(self.max_cross_track, cross_track.detach())
        self.samples += 1
        self.prev_action = action.detach()
        return (c.w_pos * pos_err + c.w_vel * vel + c.w_rate * rate + c.w_up * tilt
                + c.w_act * act + c.w_dact * dact + c.w_floor * floor
                + c.w_crash * st.crashed.float() + 20.0 * obstacle)

    def tick(self, st: QuadState) -> torch.Tensor:
        self.t += 1
        self.progress = (self.progress + self.speed * self.cfg.control_dt).clamp(max=self.length)
        self.target = self._point_at(self.progress)
        clearance = self._clearance(st.pos)
        self.min_clearance = torch.minimum(self.min_clearance, clearance.detach())
        self.hit_post |= clearance < 0.0
        return st.crashed | self.hit_post | (self.t >= self.cfg.episode_steps)

    def metrics(self, st: QuadState) -> dict[str, float]:
        d = (st.pos - self.target).norm(dim=1)
        final_cross_track = self._cross_track_distance(st.pos)
        n = max(self.samples, 1)
        return {
            "dist_mean": float((self.distance_sum / n).mean()),
            "within_0.5m": float((self.within_sum / n).mean()),
            "crashed": float(st.crashed.float().mean()),
            "speed": float((self.speed_sum / n).mean()),
            "cross_track_error": float((self.cross_track_sum / n).mean()),
            "max_cross_track_error": float(self.max_cross_track.max()),
            "final_dist": float(d.mean()),
            "final_speed": float(st.vel.norm(dim=1).mean()),
            "final_cross_track_error": float(final_cross_track.mean()),
            "target_speed": self.speed,
            "route_progress_m": float(self.progress.mean()),
            "route_length_m": self.length,
            "min_post_clearance_m": float(self.min_clearance.min()),
            "hit_post": float(self.hit_post.float().mean()),
        }
