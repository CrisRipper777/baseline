from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from scripts.build_degree_preserving_rewires import _degree, _rewire
from scripts.run_problem_deep_dive_nc import _jobs
from src.analysis.problem_deep_dive import (
    canonical_pairs,
    context_novelty,
    fixed_norm_mask,
    fit_train_only_logistic,
    four_regimes,
    matched_random_edge_sets,
    normalized_physical_operator,
    remove_pairs_vectorized,
    source_matched_compatibility,
    structural_edge_descriptors,
    train_edge_quantile_bins,
)
from src.models.operator_control_gcn import Model as GCNControl
from src.models.operator_control_sage import Model as SAGEControl


def _cfg(readout="deep_only"):
    return OmegaConf.create({"model": {"hidden_dim": 8, "num_layers": 3, "dropout": 0.0,
                                         "activation": "relu", "norm": "layernorm", "readout": readout}})


def _cycle_graph(n=12):
    edges = [(i, (i + 1) % n) for i in range(n)]
    edges += [(i, (i + 3) % n) for i in range(n)]
    pairs = torch.tensor(sorted({tuple(sorted(e)) for e in edges}), dtype=torch.long)
    return torch.cat([pairs.T, pairs.flip(1).T], dim=1)


@pytest.mark.parametrize("model_cls", [GCNControl, SAGEControl])
@pytest.mark.parametrize("readout", ["deep_only", "anchored25"])
def test_operator_controls_keep_modalities_independent_and_use_exact_readout(model_cls, readout):
    torch.manual_seed(5)
    model = model_cls(_cfg(readout), {"input_dim": 4, "text_dim": 2, "visual_dim": 2}).eval()
    edge = _cycle_graph()
    x = torch.randn(12, 4)
    first = model.analyze(x, edge)
    changed_visual = x.clone(); changed_visual[:, 2:] += 100
    second = model.analyze(changed_visual, edge)
    assert torch.equal(first["H0_text"], second["H0_text"])
    assert torch.equal(first["H3_text"], second["H3_text"])
    assert torch.equal(first["Z_text"], second["Z_text"])
    changed_text = x.clone(); changed_text[:, :2] -= 50
    third = model.analyze(changed_text, edge)
    assert torch.equal(first["H0_visual"], third["H0_visual"])
    assert torch.equal(first["H3_visual"], third["H3_visual"])
    assert torch.equal(first["Z_visual"], third["Z_visual"])
    if readout == "deep_only":
        assert torch.equal(first["Z_text"], first["H3_text"])
        assert torch.equal(first["Z_visual"], first["H3_visual"])
    else:
        assert torch.equal(first["Z_text"], .25 * first["H0_text"] + .75 * first["H3_text"])
        assert torch.equal(first["Z_visual"], .25 * first["H0_visual"] + .75 * first["H3_visual"])
    assert not any("alpha" in name.lower() for name, _ in model.named_parameters())
    assert torch.isfinite(first["fused_z"]).all()


def test_fixed_norm_mask_only_zeros_target_pair_and_keeps_loops_and_other_weights():
    edge = torch.tensor([[0, 1, 1, 2, 2, 0, 2, 3, 3, 2],
                         [1, 0, 2, 1, 0, 2, 3, 2, 3, 3]])
    base = normalized_physical_operator(edge, 4)
    changed = fixed_norm_mask(base, torch.tensor([[0, 1]]))
    base_d, changed_d = base.to_dense(), changed.to_dense()
    assert changed_d[0, 1] == 0 and changed_d[1, 0] == 0
    assert changed_d[0, 0] == base_d[0, 0]
    non_target = torch.ones_like(base_d, dtype=torch.bool)
    non_target[0, 1] = non_target[1, 0] = False
    assert torch.equal(base_d[non_target], changed_d[non_target])


def test_renormalized_delete_recomputes_neighbor_weights():
    from src.analysis.mechanism_discovery import remove_undirected_pairs
    edge = _cycle_graph(6)
    before = normalized_physical_operator(edge, 6).to_dense()
    deleted = remove_undirected_pairs(edge, torch.tensor([[0, 1]]))
    after = normalized_physical_operator(deleted, 6).to_dense()
    assert not torch.equal(before[0, 3], after[0, 3])
    assert after[0, 1] == 0 and after[1, 0] == 0


def test_novelty_is_nan_for_degree_one_endpoints_and_label_field_is_descriptive_only():
    pairs = torch.tensor([[0, 1], [1, 2]])
    states = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    novelty = context_novelty(states, pairs[:1])
    assert torch.isnan(novelty[0])
    edge = torch.cat([pairs.T, pairs.flip(1).T], 1)
    labels = torch.tensor([0, 0, 1, 7])
    d1 = structural_edge_descriptors(edge, 4, labels, torch.tensor([0, 1]))
    labels[2] = 0
    d2 = structural_edge_descriptors(edge, 4, labels, torch.tensor([0, 1]))
    assert d1["same_label_edge"][0] == 1
    assert torch.isnan(d1["same_label_edge"][1])
    for key in d1:
        if key != "same_label_edge":
            assert torch.allclose(d1[key], d2[key], equal_nan=True)


def test_four_regimes_use_train_thresholds_and_leave_middle_unassigned():
    sim = torch.tensor([.1, .9, .1, .9, .5])
    nov = torch.tensor([.1, .1, .9, .9, .5])
    bins = four_regimes(sim, nov, {"q25": .2, "q75": .8}, {"q25": .2, "q75": .8})
    assert bins.tolist() == [3, 0, 2, 1, -1]


def test_matched_controls_exactly_match_marginals_or_fail_explicitly():
    pairs = torch.tensor([[i, i + 1] for i in range(12)])
    degree_bins = torch.tensor([[0, 0]] * 12)
    weight_bins = torch.tensor([0] * 12)
    controls, diagnostics = matched_random_edge_sets(pairs, torch.tensor([0, 1]), degree_bins,
                                                      weight_bins, repeats=20, seed=4)
    assert diagnostics["status"] == "MATCHED"
    assert diagnostics["verified_exact_marginals"]
    assert len(controls) == 20 and all(len(x) == 2 for x in controls)
    failed, diag = matched_random_edge_sets(pairs[:2], torch.tensor([0, 1]), degree_bins[:2],
                                           weight_bins[:2], repeats=20)
    assert failed == []
    assert diag["status"] == "MATCHING_FAILED"


def test_rewire_preserves_degree_edge_count_and_simple_undirected_graph():
    edge = _cycle_graph(16)
    pairs = canonical_pairs(edge)
    rewired, successful, attempts, ratio = _rewire(pairs, 16, seed=2026, target_factor=2)
    assert successful >= 2 * len(pairs)
    assert attempts >= successful
    assert ratio >= 2
    assert torch.equal(_degree(pairs, 16), _degree(rewired, 16))
    assert torch.unique(rewired, dim=0).size(0) == len(pairs)
    assert not (rewired[:, 0] == rewired[:, 1]).any()


def test_training_job_counts_and_seed_aggregation_shape():
    assert len(list(_jobs("preflight", ("ele-fashion",)))) == 5
    assert len(list(_jobs("operator", ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")))) == 20
    assert len(list(_jobs("rewired", ("Movies", "Toys", "Grocery", "ele-fashion", "Reddit-S")))) == 5


def test_vectorized_source_matched_compatibility_matches_frozen_definition():
    from src.analysis.mechanism_discovery import edge_percentile_compatibility
    torch.manual_seed(19)
    edge = _cycle_graph(18)
    states = torch.randn(18, 7)
    old = edge_percentile_compatibility(states, edge, seed=2026, max_sources=100000, nonedges_per_source=8)
    new = source_matched_compatibility(states, edge, seed=2026, max_sources=100000, nonedges_per_source=8)
    assert torch.equal(old[0], new[0])
    assert torch.equal(old[1], new[1])
    assert old[2] == new[2]


def test_train_quantile_thresholds_ignore_validation_edges():
    pairs = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    train = torch.tensor([0, 1, 2, 3])
    values = torch.tensor([0.1, 0.2, 100.0, 200.0])
    bins_a, thresholds_a = train_edge_quantile_bins(values, pairs, train)
    values[2:] = 1e8
    bins_b, thresholds_b = train_edge_quantile_bins(values, pairs, train)
    assert thresholds_a == thresholds_b
    assert thresholds_a["train_edge_count"] == 2
    assert bins_a[:2].tolist() == bins_b[:2].tolist()
    assert bins_b[2:].tolist() == [3, 3]


def test_train_only_logistic_fit_does_not_read_validation_targets():
    torch.manual_seed(8)
    features = torch.randn(12, 3)
    target_a = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    target_b = target_a.clone(); target_b[8:] = torch.tensor([1, 1, 1, 1])
    train, val = torch.arange(8), torch.arange(8, 12)
    first = fit_train_only_logistic(features, target_a, train, val, random_state=4)
    second = fit_train_only_logistic(features, target_b, train, val, random_state=4)
    assert torch.equal(torch.as_tensor(first["val_pred"]), torch.as_tensor(second["val_pred"]))
    assert (first["model"].coef_ == second["model"].coef_).all()
    assert first["train_indices_used"].tolist() == train.tolist()


def test_selector_feature_contract_excludes_label_derived_fields():
    from scripts.problem_deep_dive_analysis import (
        MODALITY_SELECTOR_FEATURES, SELF_STRUCTURE_SELECTOR_FEATURES, UTILITY_FEATURE_GROUPS,
    )
    allowed = SELF_STRUCTURE_SELECTOR_FEATURES + MODALITY_SELECTOR_FEATURES
    allowed += [name for group in UTILITY_FEATURE_GROUPS.values() for name in group]
    forbidden = {"same_label_edge", "homophily", "label", "same_label_edge_descriptive_only"}
    assert not forbidden.intersection(allowed)
    assert "same_label_edge_descriptive_only" not in UTILITY_FEATURE_GROUPS["ALL"]


def test_full_graph_operator_control_cannot_use_neighbor_loader_nc_path():
    from src.tasks.nc import _resolve_training_mode
    cfg = OmegaConf.create({"task": {"training_mode": "full_graph"},
                            "model": {"name": "operator_control_gcn"}})
    assert _resolve_training_mode(cfg) == "full_graph"
    cfg.model.name = "operator_control_sage"
    assert _resolve_training_mode(cfg) == "full_graph"


def test_intervention_sparse_forward_matches_model_full_graph_forward():
    from torch import nn
    from src.analysis.problem_deep_dive import normalized_physical_operator
    from src.models.multi_order_bank import Model as MultiOrderBank
    from scripts.problem_deep_dive_analysis import _model_logits_from_operator

    cfg = OmegaConf.create({"model": {"hidden_dim": 8, "max_order": 3, "dropout": 0.0,
                                        "readout": "uniform", "modality_mode": "both",
                                        "fusion_mode": "plain_mlp"}})
    torch.manual_seed(17)
    model = MultiOrderBank(cfg, {"input_dim": 4, "text_dim": 2, "visual_dim": 2}).eval()
    edge = _cycle_graph(12)
    x = torch.randn(12, 4)
    head = nn.Linear(8, 3).eval()
    val = torch.tensor([0, 3, 6, 9])
    expected = head(model.analyze(x, edge)["fused_z"])[val]
    operator = normalized_physical_operator(edge, 12)
    actual = _model_logits_from_operator(model, head, x, operator, val)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_two_target_pairs_are_not_transposed_when_masked_or_deleted():
    edge = torch.tensor([[0, 1, 2, 3, 1, 0, 3, 2, 4, 4],
                         [1, 0, 3, 2, 4, 4, 4, 4, 1, 1]])
    targets = torch.tensor([[0, 1], [2, 3]])
    base = normalized_physical_operator(edge, 5)
    masked = fixed_norm_mask(base, targets).to_dense()
    assert masked[0, 1] == 0 and masked[1, 0] == 0
    assert masked[2, 3] == 0 and masked[3, 2] == 0
    deleted = remove_pairs_vectorized(edge, targets, num_nodes=5)
    remain = canonical_pairs(deleted)
    assert not any(tuple(pair.tolist()) in {(0, 1), (2, 3)} for pair in remain)


def test_selector_and_utility_probe_contracts_run_on_split_specific_logits():
    from scripts.problem_deep_dive_analysis import (
        SELF_STRUCTURE_SELECTOR_FEATURES, MODALITY_SELECTOR_FEATURES,
        UTILITY_FEATURE_GROUPS, _selector_analysis_one, _utility_predictability_one,
    )
    torch.manual_seed(31)
    n, n_train = 24, 16
    train, val = torch.arange(n_train), torch.arange(n_train, n)
    known_y = torch.full((n,), -1, dtype=torch.long)
    known_y[:n_train] = torch.randint(0, 3, (n_train,))
    known_y[n_train:] = torch.randint(0, 3, (n - n_train,))
    names = ("uniform", "gpr", "self_only", "propagated_uniform", "text_self",
             "text_uniform", "visual_self", "visual_uniform")
    logits = {name: {"train": torch.randn(n_train, 3), "val": torch.randn(n - n_train, 3)}
              for name in names}
    features = sorted(set(SELF_STRUCTURE_SELECTOR_FEATURES + MODALITY_SELECTOR_FEATURES +
                          [x for group in UTILITY_FEATURE_GROUPS.values() for x in group]))
    node_features = {name: torch.randn(n) for name in features}
    node_features["degree"] = torch.randint(0, 8, (n,)).float()
    node_features["degree_quantile"] = torch.rand(n)
    cache = {"dataset": "toy", "seed": 42, "num_nodes": n, "train_idx": train,
             "val_idx": val, "known_y": known_y, "class_ids": [0, 1, 2],
             "logits": logits, "node_features": node_features}
    oracle, learned = _selector_analysis_one(cache)
    probe = _utility_predictability_one(cache)
    assert len(oracle) == 8 and len(learned) == 2
    assert {row["family"] for row in learned} == {"self_structure", "modality"}
    assert len(probe) == 3 * len(UTILITY_FEATURE_GROUPS)
    assert all(row["fit_split"] == "train" and row["evaluation_split"] == "validation" for row in probe)


@pytest.mark.parametrize("model_cls", [GCNControl, SAGEControl])
def test_operator_control_full_graph_forward_backward_and_inference_are_finite(model_cls):
    model = model_cls(_cfg("anchored25"), {"input_dim": 4, "text_dim": 2, "visual_dim": 2})
    edge = _cycle_graph(12)
    x = torch.randn(12, 4)
    output, _, _, aux_loss, _ = model(x, edge)
    (output.square().mean() + aux_loss).backward()
    assert torch.isfinite(output).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    inferred = model.inference(x, edge, device=torch.device("cpu"))
    assert inferred.shape == output.shape
    assert torch.isfinite(inferred).all()
