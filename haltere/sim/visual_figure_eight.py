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


def body_observation_to_tensor(observation: dict, B: int, device) -> torch.Tensor:
    """Convert one detector observation into batched ``(forward,left,up)``."""
    from ..vision.tposts import body_offset_from_observation

    rel_b = body_offset_from_observation(observation)
    if rel_b is None:
        raise ValueError("cannot control from a visual observation with no visible post")
    return torch.as_tensor(rel_b, dtype=torch.float32, device=device).expand(B, -1).clone()


class VisualFigureEightTask(FigureEightTask):
    """Use a post-relative body-frame goal instead of the route target vector."""

    def observe(self, st, rel_b: torch.Tensor | None = None):
        sensors = self.sim.sensors(st)
        if rel_b is None:
            post = select_visual_post(self.target, self.posts)
            rel_b = rotate_inv(st.quat, post - st.pos)
        return observe_from_sensors(
            sensors, self.target, st.motor.mean(dim=1, keepdim=True), self.cfg, rel_b=rel_b)
