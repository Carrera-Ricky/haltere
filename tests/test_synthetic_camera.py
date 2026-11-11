import numpy as np

from haltere.vision.synthetic_camera import sim_to_renderer


def test_sim_to_renderer_maps_flu_axes_to_right_up_depth():
    assert np.array_equal(sim_to_renderer(np.array([1.0, 0.0, 0.0])), [0.0, 0.0, -1.0])
    assert np.array_equal(sim_to_renderer(np.array([0.0, 1.0, 0.0])), [-1.0, 0.0, 0.0])
    assert np.array_equal(sim_to_renderer(np.array([0.0, 0.0, 1.0])), [0.0, 1.0, 0.0])
