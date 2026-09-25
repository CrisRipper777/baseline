from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models import build_model
from src.tasks.common import build_optimizer
from src.utils.summary import count_parameters


MODELS = ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
NUM_NODES = 16
TEXT_DIM = 8
VISUAL_DIM = 8


def _inputs(device: torch.device):
    x = torch.randn(NUM_NODES, TEXT_DIM + VISUAL_DIM, device=device)
    nodes = torch.arange(NUM_NODES, device=device)
    edge_index = torch.stack(
        [torch.cat([nodes, (nodes + 1) % NUM_NODES]),
         torch.cat([(nodes + 1) % NUM_NODES, nodes])]
    ).long()
    return x, edge_index


@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_formal_nc_baseline_forward_backward_inference_finite(name: str, device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_name)
    model_cfg = OmegaConf.load(Path("configs/model") / f"{name}.yaml")
    cfg = OmegaConf.create({
        "model": model_cfg,
        "task": {"optimizer": "adamw", "lr": 1e-3, "weight_decay": 1e-4},
    })
    model = build_model(cfg, {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": NUM_NODES,
        "num_classes": 4,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }).to(device)
    assert model.out_dim > 0
    assert count_parameters(model) > 0
    optimizer = build_optimizer(model.parameters(), cfg, model=model)
    expected_lr = float(model_cfg.get("lr", 1e-3))
    expected_wd = float(model_cfg.get("weight_decay", 1e-4))
    assert {group["lr"] for group in optimizer.param_groups} == {expected_lr}
    assert all(group["weight_decay"] in (expected_wd, 0.0) for group in optimizer.param_groups)

    x, edge_index = _inputs(device)
    model.train()
    z, _, _, aux_loss, _ = model(x, edge_index)
    assert z.shape == (NUM_NODES, model.out_dim)
    loss = z.square().mean() + aux_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )

    model.eval()
    with torch.no_grad():
        inferred = model.inference(x.cpu(), edge_index.cpu(), device=device, batch_size=4)
    assert inferred.shape == z.shape
    assert torch.isfinite(inferred).all()
