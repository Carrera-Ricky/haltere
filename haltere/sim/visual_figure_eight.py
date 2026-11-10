"""Figure-eight task using the visual goal-input contract.

The first implementation uses known post geometry to synthesize the same body-frame
observation that the image detector will eventually provide. This is an oracle
visual bridge for interface testing, not a claim of image-based autonomy.
"""
from __future__ import annotations

import torch

from .figure_eight import FigureEightTask
from .tasks import observe_from_sensors
from .quad import rotate_inv


def select_visual_post(target: torch.Tensor, posts: torch.Tensor) -> torch.Tensor:
    """Select the post nearest the current route target for the bridge sensor."""
    index = (target[:, None, :] - posts[None, :, :]).pow(2).sum(dim=2).argmin(dim=1)
    return posts[index]


class VisualFigureEightTask(FigureEightTask):
    """Use a post-relative body-frame goal instead of the route target vector."""

    def observe(self, st):
        sensors = self.sim.sensors(st)
        post = select_visual_post(self.target, self.posts)
        rel_b = rotate_inv(st.quat, post - st.pos)
        return observe_from_sensors(
            sensors, self.target, st.motor.mean(dim=1, keepdim=True), self.cfg, rel_b=rel_b)
