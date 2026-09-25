from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data

import src.tasks.nc as nc
import src.tasks.lp as lp
from src.data import MAGData
from src.models.predictor import LinkPredictor
from src.tasks.common import build_optimizer, resolve_num_neighbors
from src.utils.summary import mean_std


def _tiny_nc_data() -> MAGData:
    return MAGData(
        name="tiny",
        source="test",
        task="nc",
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        num_nodes=4,
        y=torch.tensor([0, 0, 1, 1]),
        num_classes=2,
        train_idx=torch.tensor([0, 2]),
        val_idx=torch.tensor([1]),
        test_idx=torch.tensor([3]),
    )


def test_unified_nc_graph_models_cannot_fall_back_to_neighbor_loader() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "gcn", "full_graph_training": False},
            "task": {
                "protocol_version": "unified_full_graph_nc_v1",
                "training_mode": "full_graph",
            },
        }
    )
    model = SimpleNamespace(full_graph_training=False)
    assert nc._resolve_training_mode(cfg, model) == "full_graph"

    cfg.task.training_mode = "sampled"
    with pytest.raises(ValueError, match="requires task.training_mode"):
        nc._resolve_training_mode(cfg, model)


def test_unified_nc_keeps_mlp_on_feature_only_path() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"name": "mlp", "full_graph_training": False},
            "task": {"training_mode": "full_graph"},
        }
    )
    assert nc._resolve_training_mode(cfg) == "feature_only"


def test_multi_run_dispatch_calls_set_seed_42_43_44(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "num_runs": 3,
            "model": {"name": "gcn"},
            "task": {
                "training_mode": "full_graph",
                "evaluate_test": True,
            },
        }
    )
    seeds: list[int] = []
    monkeypatch.setattr(nc, "set_seed", seeds.append)

    def fake_run(cfg, data, device, logger, run_id, seed, eval_labels):
        return {
            "val_acc": 0.5,
            "val_macro_f1": 0.5,
            "test_acc": 0.5,
            "test_macro_f1": 0.5,
        }

    monkeypatch.setattr(nc, "_run_single_nc", fake_run)
    result = nc.run_nc(
        cfg,
        _tiny_nc_data(),
        torch.device("cpu"),
        logging.getLogger("test.protocol"),
    )
    assert seeds == [42, 43, 44]
    assert result["test_acc"] == (0.5, 0.0)


def test_nc_run_metrics_sidecar_records_internal_seeds_and_metadata(tmp_path, monkeypatch) -> None:
    import json

    cfg = OmegaConf.create({
        "seed": 42,
        "num_runs": 3,
        "model": {"name": "gcn"},
        "task": {
            "training_mode": "full_graph",
            "protocol_version": "unified_full_graph_nc_v1",
            "evaluate_test": True,
            "run_metrics_path": str(tmp_path / "runs.json"),
        },
    })
    monkeypatch.setattr(nc, "set_seed", lambda _: None)
    monkeypatch.setattr(nc, "_run_single_nc", lambda cfg, data, device, logger, run_id, seed, eval_labels: {
        "val_acc": 0.5, "val_macro_f1": 0.4, "test_acc": 0.45, "test_macro_f1": 0.35,
        "_run_metadata": {"model_parameters": 123, "optimizer": "AdamW"},
    })
    nc.run_nc(cfg, _tiny_nc_data(), torch.device("cpu"), logging.getLogger("test.sidecar"))
    payload = json.loads((tmp_path / "runs.json").read_text())
    assert payload["run_seeds"] == [42, 43, 44]
    assert payload["runs"][1]["seed"] == 43
    assert payload["runs"][1]["metadata"]["model_parameters"] == 123


def test_mean_std_uses_population_standard_deviation() -> None:
    mean, std = mean_std([0.0, 1.0, 1.0])
    assert mean == pytest.approx(2.0 / 3.0)
    assert std == pytest.approx((2.0 / 9.0) ** 0.5)


def test_protocol_configs_pin_nc_lp_contracts() -> None:
    root_cfg = OmegaConf.load("configs/config.yaml")
    assert root_cfg.seed == 42
    assert root_cfg.num_runs == 3

    nc_cfg = OmegaConf.load("configs/task/nc.yaml")
    assert nc_cfg.protocol_version == "unified_full_graph_nc_v1"
    assert nc_cfg.training_mode == "full_graph"
    assert nc_cfg.optimizer == "adamw"
    assert (nc_cfg.epochs, nc_cfg.lr, nc_cfg.weight_decay) == (300, 1e-3, 1e-4)
    assert (nc_cfg.eval_every, nc_cfg.patience) == (1, 30)
    assert nc_cfg.early_stop_min_epoch == 30
    assert nc_cfg.early_stop_min_delta == pytest.approx(1e-4)
    assert nc_cfg.grad_clip == pytest.approx(1.0)
    assert nc_cfg.inference_mode == "full"
    assert nc_cfg.inference_batch_size == 4096
    assert nc_cfg.scheduler is None

    lp_cfg = OmegaConf.load("configs/task/lp.yaml")
    assert lp_cfg.protocol_version == "unified_sampled_lp_v1"
    assert lp_cfg.training_mode == "sampled"
    assert lp_cfg.optimizer == "adam"
    assert (lp_cfg.epochs, lp_cfg.lr, lp_cfg.weight_decay) == (150, 1e-3, 1e-5)
    assert lp_cfg.num_neighbors == [5, 5, 5]
    assert lp_cfg.subgraph_type == "bidirectional"
    assert lp_cfg.positive_edge_mask_backend == "global_eid"
    assert lp_cfg.loader_num_workers == 4
    assert lp_cfg.loader_prefetch_factor == 2
    assert lp_cfg.torch_threads == 8
    assert lp_cfg.eval_every == 2
    assert lp_cfg.eval_edge_batch_size == 512
    assert lp_cfg.decoder.proj_dim == 128


def test_lp_three_fanout_setting_and_model_depth_resolution() -> None:
    cfg = OmegaConf.create(
        {"model": {"num_layers": 10}, "task": {"num_neighbors": [5, 5, 5]}}
    )
    # DGF's ten filtering iterations do not turn the frozen LP sampler into
    # ten hops; the official loader fanout remains exactly three hops.
    assert lp._resolve_lp_num_neighbors(cfg) == [5, 5, 5]

    cfg.model.num_layers = 3
    model = SimpleNamespace(requires_full_lp_sampler_depth=True)
    assert lp._resolve_lp_num_neighbors(cfg, model) == [5, 5, 5]
    cfg.model.num_layers = 2
    assert lp._resolve_lp_num_neighbors(cfg, model) == [5, 5]
    assert resolve_num_neighbors(cfg) == [5, 5]


def test_link_neighbor_loader_receives_frozen_three_hop_settings(monkeypatch) -> None:
    cfg = OmegaConf.create(
        {
            "model": {"num_layers": 3},
            "task": {
                "num_neighbors": [5, 5, 5],
                "batch_size": 2048,
                "subgraph_type": "bidirectional",
                "loader_num_workers": 4,
                "loader_prefetch_factor": 2,
            },
        }
    )
    captured = {}
    monkeypatch.setattr(
        lp,
        "LinkNeighborLoader",
        lambda *args, **kwargs: captured.update(kwargs) or object(),
    )
    generator = torch.Generator().manual_seed(23)
    loader = lp._build_link_loader(
        cfg,
        Data(x=torch.ones(3, 2), edge_index=torch.tensor([[0, 1], [1, 0]])),
        torch.tensor([[0], [1]]),
        torch.tensor([1.0]),
        generator,
        1234,
    )
    assert loader is not None
    assert captured["num_neighbors"] == [5, 5, 5]
    assert captured["subgraph_type"] == "bidirectional"
    assert captured["num_workers"] == 4
    assert captured["prefetch_factor"] == 2
    assert captured["generator"] is generator


def test_global_eid_positive_mask_matches_local_key_audit() -> None:
    global_edge_index = torch.tensor(
        [[0, 1, 0, 2, 3, 4], [1, 0, 2, 0, 4, 3]], dtype=torch.long
    )
    # Deliberately permute local node order to require n_id mapping.
    n_id = torch.tensor([4, 3, 2, 1, 0], dtype=torch.long)
    inverse = torch.empty_like(n_id)
    inverse[n_id] = torch.arange(n_id.numel())
    local_edges = inverse[global_edge_index]
    supervision = torch.tensor([[inverse[0]], [inverse[1]]], dtype=torch.long)
    labels = torch.tensor([1.0])
    expected = lp._exclude_positive_label_edges_from_message_graph(
        local_edges, supervision, labels, num_nodes=n_id.numel()
    )

    batch = Data(
        edge_index=local_edges.clone(),
        e_id=torch.arange(global_edge_index.size(1)),
        n_id=n_id,
        edge_label_index=supervision,
        edge_label=labels,
    )
    lookup = lp._build_message_edge_lookup(global_edge_index, num_nodes=5)
    removed = lp._exclude_positive_label_edges_by_global_eid(batch, lookup)
    assert removed == 2
    assert torch.equal(batch.edge_index, expected)
    assert batch.e_id.tolist() == [2, 3, 4, 5]


def test_lp_validation_ties_use_v3_pessimistic_ranking() -> None:
    predictor = LinkPredictor(in_dim=2, hidden_dim=2, num_layers=1, dropout=0.0)
    for parameter in predictor.parameters():
        torch.nn.init.zeros_(parameter)
    metrics = lp._evaluate_split(
        torch.zeros(3, 2),
        predictor,
        {
            "source_node": torch.tensor([0]),
            "target_node": torch.tensor([1]),
            "target_node_neg": torch.tensor([[2]]),
        },
        torch.device("cpu"),
        batch_size=1,
    )
    assert metrics["mrr"] == pytest.approx(0.5)
    assert metrics["hits@1"] == 0.0


def test_lp_multi_run_dispatch_calls_set_seed_42_43_44(monkeypatch) -> None:
    from src.data import EdgeSplit

    cfg = OmegaConf.create(
        {
            "seed": 42,
            "num_runs": 3,
            "model": {"name": "gcn"},
            "task": {"training_mode": "sampled", "evaluate_test": True},
        }
    )
    data = SimpleNamespace(
        edge_split=EdgeSplit(
            train={"source_node": torch.tensor([0]), "target_node": torch.tensor([1])},
            valid={
                "source_node": torch.tensor([0]),
                "target_node": torch.tensor([2]),
                "target_node_neg": torch.tensor([[3]]),
            },
            test={
                "source_node": torch.tensor([1]),
                "target_node": torch.tensor([3]),
                "target_node_neg": torch.tensor([[2]]),
            },
        )
    )
    seeds: list[int] = []
    monkeypatch.setattr(lp, "set_seed", seeds.append)

    def fake_run(cfg, data, device, logger, run_id, seed):
        return {
            "val_mrr": 0.5,
            "test_mrr": 0.5,
            "test_hits@1": 0.0,
            "test_hits@3": 1.0,
            "test_hits@10": 1.0,
        }

    monkeypatch.setattr(lp, "_run_single_lp", fake_run)
    result = lp.run_lp(
        cfg,
        data,
        torch.device("cpu"),
        logging.getLogger("test.protocol.lp"),
    )
    assert seeds == [42, 43, 44]
    assert result["test_mrr"] == (0.5, 0.0)


def test_lp_epoch_random_streams_use_separate_frozen_offsets() -> None:
    cfg = OmegaConf.create(
        {
            "task": {
                "negative_seed_offset": 10000,
                "shuffle_seed_offset": 20000,
                "neighbor_seed_offset": 30000,
            }
        }
    )
    seeds = lp._epoch_random_seeds(cfg, run_seed=42, epoch=1)
    assert seeds == (10043, 20043, 30043)
    assert len(set(seeds)) == 3
    assert seeds == lp._epoch_random_seeds(cfg, run_seed=42, epoch=1)


def test_model_optimizer_preset_overrides_task_lr_and_weight_decay() -> None:
    cfg = OmegaConf.create(
        {
            "model": {"lr": 5e-3, "weight_decay": 1e-5},
            "task": {
                "optimizer": "adamw",
                "lr": 1e-3,
                "weight_decay": 1e-4,
            },
        }
    )
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(model.parameters(), cfg, model=model)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-5)



def test_formal_dataset_and_model_scopes_and_mob_capacity_are_frozen() -> None:
    from scripts.run_mob_factorial_nc import DATASETS as MOB_DATASETS, FUSIONS, READOUTS, _jobs
    from scripts.run_nc_baselines import DATASETS as NC_DATASETS, MODELS as NC_MODELS

    assert MOB_DATASETS == ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")
    assert NC_DATASETS == MOB_DATASETS
    assert NC_MODELS == ("mlp", "gcn", "sage", "mmgcn", "mgat", "dip", "dgf", "dmgc", "lgmrec")
    assert len(list(_jobs(MOB_DATASETS))) == 30
    assert READOUTS == ("terminal", "uniform", "gpr")
    assert FUSIONS == ("plain_mlp", "residual")

    mob_cfg = OmegaConf.load("configs/model/multi_order_bank.yaml")
    assert mob_cfg.hidden_dim == 256
    assert mob_cfg.max_order == mob_cfg.num_layers == 3
    assert mob_cfg.dropout == pytest.approx(0.2)
    assert mob_cfg.fusion_mode == "plain_mlp"
    assert all("books" not in dataset.lower() for dataset in MOB_DATASETS)
