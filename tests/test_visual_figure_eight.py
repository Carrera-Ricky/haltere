import torch

import pytest

from haltere.sim.visual_figure_eight import (body_observation_to_tensor, search_yaw_for_frame,
                                             select_visual_post)


def test_select_visual_post_chooses_nearest_route_post():
    target = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    posts = torch.tensor([[-3.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    chosen = select_visual_post(target, posts)

    assert torch.equal(chosen, posts[[0, 1]])


def test_body_observation_to_tensor_preserves_body_axes():
    observation = {
        "visible": True,
        "offset_forward_m": 8.0,
        "offset_left_m": -2.0,
        "offset_up_m": 1.5,
    }

    result = body_observation_to_tensor(observation, B=2, device="cpu")

    assert torch.equal(result, torch.tensor([[8.0, -2.0, 1.5], [8.0, -2.0, 1.5]]))


def test_body_observation_to_tensor_rejects_missing_post():
    with pytest.raises(ValueError, match="no visible post"):
        body_observation_to_tensor({"visible": False}, B=1, device="cpu")


def test_search_yaw_alternates_directions():
    assert search_yaw_for_frame(0, period=3, magnitude=0.2) == -0.2
    assert search_yaw_for_frame(2, period=3, magnitude=0.2) == -0.2
    assert search_yaw_for_frame(3, period=3, magnitude=0.2) == 0.2
