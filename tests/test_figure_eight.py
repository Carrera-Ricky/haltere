import numpy as np
import pytest

from haltere.liftoff.figure_eight import generate_two_post_figure_eight


def test_two_post_route_has_opposite_orbits_and_clearance():
    route = generate_two_post_figure_eight(
        post_spacing=12.0, orbit_radius=3.0, points_per_orbit=32, transition_points=4
    )
    posts = np.asarray(route["posts"])
    points = np.asarray(route["waypoints"])
    assert points.shape == (68, 3)
    assert np.allclose(points[:, 2], 2.0)
    assert np.min(np.linalg.norm(points[:, None, :2] - posts[None, :, :2], axis=2)) >= 3.0 - 1e-9
    assert np.linalg.norm(points[0] - posts[0]) == pytest.approx(3.0)
    assert np.linalg.norm(points[36] - posts[1]) == pytest.approx(3.0)


def test_two_post_route_rejects_overlapping_orbits():
    with pytest.raises(ValueError, match="greater than twice"):
        generate_two_post_figure_eight(post_spacing=6.0, orbit_radius=3.0)
