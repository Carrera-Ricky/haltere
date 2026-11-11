"""File-based client for the GraphicsEngine synthetic camera."""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

from .camera import quat_wxyz_to_mat
from .tposts import (TPostParams, add_body_observations, add_camera_directions,
                     add_distance_estimates, detect_image,
                     focal_length_from_vertical_fov, observation_from_detections)


def sim_to_renderer(vector: np.ndarray) -> np.ndarray:
    """Map Haltere FLU world coordinates to the renderer's right/up/depth axes."""
    x, y, z = np.asarray(vector, dtype=np.float64)
    return np.asarray([-y, z, -x], dtype=np.float64)


class SyntheticCameraClient:
    """Request camera frames from the pose-file protocol used by GraphicsEngine."""

    def __init__(self, directory: str | Path, *, tilt_deg: float = 30.0,
                 timeout_s: float = 2.0, poll_s: float = 0.005,
                 params: TPostParams | None = None):
        self.directory = Path(directory)
        self.pose_path = self.directory / "haltere_camera_pose.txt"
        self.frame_path = self.directory / "haltere_camera_frame.ppm"
        self.metadata_path = self.directory / "haltere_camera_frame.json"
        self.tilt_deg = float(tilt_deg)
        self.timeout_s = float(timeout_s)
        self.poll_s = float(poll_s)
        self.params = params or TPostParams()

    def request_frame(self, position_sim: np.ndarray, forward_sim: np.ndarray) -> Path:
        """Send one camera pose and wait for the renderer to refresh its frame."""
        position = sim_to_renderer(position_sim)
        forward = sim_to_renderer(forward_sim)
        if np.linalg.norm(forward) < 1e-8:
            raise ValueError("camera forward vector must be non-zero")
        before = self.frame_path.stat().st_mtime_ns if self.frame_path.exists() else -1
        temporary = self.pose_path.with_name(self.pose_path.name + ".tmp")
        temporary.write_text(
            " ".join(f"{value:.9g}" for value in (*position, *forward)) + "\n",
            encoding="ascii")
        os.replace(temporary, self.pose_path)

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if self.frame_path.exists() and self.frame_path.stat().st_mtime_ns > before:
                return self.frame_path
            time.sleep(self.poll_s)
        raise TimeoutError(f"synthetic camera did not refresh {self.frame_path}")

    def observe(self, position_sim: np.ndarray, quat_wxyz: np.ndarray) -> dict:
        """Render and detect the nearest T-post from a Haltere drone pose."""
        rotation = quat_wxyz_to_mat(np.asarray(quat_wxyz, dtype=np.float64))
        camera_forward_body = np.asarray([
            np.cos(np.deg2rad(self.tilt_deg)), 0.0,
            np.sin(np.deg2rad(self.tilt_deg))], dtype=np.float64)
        camera_forward_sim = rotation @ camera_forward_body
        image = self.request_frame(position_sim, camera_forward_sim)
        metadata = __import__("json").loads(self.metadata_path.read_text(encoding="utf-8"))
        focal_length = focal_length_from_vertical_fov(
            int(metadata["image_height"]), float(metadata["camera_vertical_fov_deg"]))
        detections = detect_image(image, params=self.params)
        detections = add_distance_estimates(
            detections, float(metadata["post_height_m"]), focal_length)
        detections = add_camera_directions(
            detections, int(metadata["image_width"]), int(metadata["image_height"]), focal_length)
        detections = add_body_observations(detections, self.tilt_deg)
        return observation_from_detections(detections)
