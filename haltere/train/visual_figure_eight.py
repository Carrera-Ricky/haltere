"""Fine-tune a brain on the body-frame visual figure-eight observation contract."""
from __future__ import annotations

from collections import deque
from pathlib import Path

import torch

from .bptt import load_checkpoint, make_world, save_checkpoint
from ..sim.figure_eight import load_route
from ..sim.visual_figure_eight import VisualFigureEightTask


def fine_tune(checkpoint: str | Path, route: str | Path, out: str | Path,
              *, iters: int = 100, steps: int = 600, batch: int = 8,
              lr: float = 1e-5, device: str = "cpu", speed: float = 1.5,
              post_height: float = 3.0, crossbar_width: float = 2.0,
              tpost_radius: float = 0.35) -> Path:
    """Fine-tune from a checkpoint using oracle visual observations.

    The oracle supplies the same body-frame vector as the camera pipeline, but
    keeps training differentiable and fast. Camera evaluation remains separate.
    """
    brain, cfg, _ = load_checkpoint(checkpoint, device)
    device = brain.device
    vehicle, _ = make_world(cfg, batch, device)
    task = VisualFigureEightTask(
        cfg.task, vehicle.sim, load_route(route), batch, device,
        speed=speed, post_height=post_height, crossbar_width=crossbar_width,
        tpost_radius=tpost_radius)
    optimizer = torch.optim.Adam(
        [parameter for parameter in brain.parameters() if parameter.requires_grad], lr=lr)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for iteration in range(1, iters + 1):
        vs = vehicle.wrap(task.reset_all())
        state = brain.init_state(batch)
        weights = brain.weight_matrix()
        delay = deque([torch.zeros(batch, brain.n_actions, device=device)
                       for _ in range(cfg.train.delay_steps)])
        total = torch.zeros((), device=device)
        aux = None
        for _ in range(steps):
            observation = task.observe(vs.quad)
            action, state, aux = brain(observation, state, weights)
            delay.append(action)
            applied = delay.popleft()
            vs = vehicle.step(vs, applied)
            total = total + task.cost(vs.quad, applied).mean()
            task.tick(vs.quad)

        loss = total / steps + brain.regularization(aux)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), cfg.train.grad_clip)
        optimizer.step()

        value = float(loss.detach())
        print(f"iter {iteration:4d}/{iters} visual_loss {value:.5f}", flush=True)
        save_checkpoint(out_dir / "last.pt", brain, optimizer, cfg, iteration, cfg.train.graph)
        if value < best_loss:
            best_loss = value
            save_checkpoint(out_dir / "best.pt", brain, optimizer, cfg, iteration, cfg.train.graph)

    return out_dir
