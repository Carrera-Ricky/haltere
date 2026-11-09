"""Generate a conservative two-post figure-eight waypoint route.

The route is deliberately geometry-only: it does not inspect camera frames and
does not provide course geometry to the runtime pilot.  The generated post
locations are metadata for course construction and offline scoring; the pilot
only consumes the waypoint polyline.
"""
from __future__ import annotations

import math

import numpy as np


def generate_two_post_figure_eight(
    *,
    post_spacing: float = 12.0,
    orbit_radius: float = 3.0,
    altitude: float = 2.0,
    points_per_orbit: int = 48,
    transition_points: int = 12,
    repetitions: int = 1,
) -> dict:
    """Return a waypoint route that alternates opposite-direction orbits.

    Posts lie on the x axis in the simulator frame, with y being the lateral
    direction and z altitude.  The route starts at the inner side of the first
    post, circles it, crosses to the inner side of the second post, circles it
    in the opposite direction, and repeats.

    ``post_spacing`` must exceed twice ``orbit_radius`` so the two protected
    orbit regions do not overlap.  The route is open by default: callers can
    repeat it explicitly with ``repetitions``.
    """
    for name, value in (
        ("post_spacing", post_spacing),
        ("orbit_radius", orbit_radius),
        ("altitude", altitude),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a positive finite number")
    if post_spacing <= 2.0 * orbit_radius:
        raise ValueError("post_spacing must be greater than twice orbit_radius")
    if points_per_orbit < 8 or transition_points < 2 or repetitions < 1:
        raise ValueError("use at least 8 orbit points, 2 transition points, and 1 repetition")

    half = post_spacing / 2.0
    posts = np.array([[-half, 0.0, altitude], [half, 0.0, altitude]], dtype=float)
    route: list[np.ndarray] = []

    def add_orbit(center: np.ndarray, start_angle: float, direction: float) -> None:
        angles = start_angle + direction * np.linspace(
            0.0, 2.0 * math.pi, points_per_orbit, endpoint=False
        )
        for angle in angles:
            route.append(center + np.array([
                orbit_radius * math.cos(angle),
                orbit_radius * math.sin(angle),
                0.0,
            ]))

    def add_transition(start: np.ndarray, end: np.ndarray) -> None:
        for f in np.linspace(0.0, 1.0, transition_points + 1)[1:]:
            route.append(start * (1.0 - f) + end * f)

    for repetition in range(repetitions):
        first_start = posts[0] + np.array([orbit_radius, 0.0, 0.0])
        second_start = posts[1] - np.array([orbit_radius, 0.0, 0.0])
        add_orbit(posts[0], 0.0, 1.0)
        add_transition(first_start, second_start)
        add_orbit(posts[1], math.pi, -1.0)
        if repetition + 1 < repetitions:
            add_transition(second_start, first_start)

    waypoints = np.asarray(route, dtype=float)
    return {
        "schema": 1,
        "pattern": "two-post-figure-eight",
        "frame": "sim (x forward, y left, z up), relative to reset point",
        "post_spacing_m": float(post_spacing),
        "orbit_radius_m": float(orbit_radius),
        "altitude_m": float(altitude),
        "repetitions": int(repetitions),
        "posts": posts.tolist(),
        "waypoints": waypoints.tolist(),
    }
