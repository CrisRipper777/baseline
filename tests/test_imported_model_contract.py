from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from src.models import build_model


TEXT_DIM = 3
VISUAL_DIM = 4
NUM_NODES = 8


def _cfg(name: str):
    values = {
        "dgf": {
            "hidden_dim": 8,
            "alpha": 1.0,
            "beta": 1.0,
            "num_layers": 10,
            "dropout": 0.0,
            "lr": 5e-3,
            "weight_decay": 1e-5,
        },
        "dmgc": {
            "hidden_dim": 8,
            "num_layers": 1,
            "dropout": 0.2,
            "tau": 1.0,
            "lambda_cr": 0.001,
            "lambda_cm": 1.0,
            "lr": 5e-3,
            "weight_decay": 1e-5,
        },
        "lgmrec": {
            "hidden_dim": 8,
            "num_layers": 3,
            "dropout": 0.2,
            "hyper_num": 4,
            "alpha": 0.1,
            "lr": 5e-3,
            "weight_decay": 1e-5,
        },
    }
    return OmegaConf.create({"model": {"name": name, **values[name]}})


def _data_info():
    return {
        "input_dim": TEXT_DIM + VISUAL_DIM,
        "num_nodes": NUM_NODES,
        "num_classes": 3,
        "text_dim": TEXT_DIM,
        "visual_dim": VISUAL_DIM,
    }


def _inputs(device: torch.device):
    generator = torch.Generator().manual_seed(31)
    x = torch.randn(
        NUM_NODES, TEXT_DIM + VISUAL_DIM, generator=generator
    ).to(device)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 0],
            [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 6, 0, 7],
        ],
        dtype=torch.long,
        device=device,
    )
    return x, edge_index


@pytest.mark.parametrize("name", ["dgf", "dmgc", "lgmrec"])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_imported_model_forward_inference_and_lp_batch_contract(
    name: str, device_name: str
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_name)
    model = build_model(_cfg(name), _data_info()).to(device).eval()
    x, edge_index = _inputs(device)

    with torch.no_grad():
        z, _, _, aux_loss, aux_info = model(x, edge_index)
        inferred = model.inference(
            x.cpu(), edge_index.cpu(), device=device, batch_size=3
        )

    assert model.out_dim == 8
    assert z.shape == (NUM_NODES, model.out_dim)
    assert inferred.shape == z.shape
    assert torch.isfinite(z).all()
    assert torch.isfinite(inferred).all()
    assert torch.allclose(z.cpu(), inferred.cpu(), atol=1e-4, rtol=1e-4)
    assert aux_loss.ndim == 0 and aux_loss.item() == 0.0
    assert aux_info == {}

    # LinkNeighborLoader supplies an induced local graph and its feature rows;
    # every imported encoder must accept the same sampled-batch forward API.
    sampled_x = x[:6]
    sampled_edges = edge_index[:, (edge_index[0] < 6) & (edge_index[1] < 6)]
    with torch.no_grad():
        sampled_z, _, _, sampled_aux, _ = model(sampled_x, sampled_edges)
    assert sampled_z.shape == (6, model.out_dim)
    assert torch.isfinite(sampled_z).all()
    assert sampled_aux.item() == 0.0


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_dmgc_v3_inference_device_fix_preserves_full_forward_math(device_name: str) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device(device_name)
    model = build_model(_cfg("dmgc"), _data_info()).to(device).eval()
    x, edge_index = _inputs(device)
    with torch.no_grad():
        full_z = model(x, edge_index)[0]
        inferred_z = model.inference(
            x.cpu(), edge_index.cpu(), device=device, batch_size=3
        )
    assert torch.isfinite(full_z).all()
    assert torch.isfinite(inferred_z).all()
    assert torch.allclose(full_z.cpu(), inferred_z.cpu(), atol=1e-4, rtol=1e-4)
