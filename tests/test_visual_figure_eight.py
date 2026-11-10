import torch

from haltere.sim.visual_figure_eight import select_visual_post


def test_select_visual_post_chooses_nearest_route_post():
    target = torch.tensor([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    posts = torch.tensor([[-3.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

    chosen = select_visual_post(target, posts)

    assert torch.equal(chosen, posts[[0, 1]])
