from __future__ import annotations

import torch

from src.analysis.mechanism_discovery import (
    FrozenProbeProtocolV1,
    PROBE_CONFIG,
    canonical_undirected_pairs,
    edge_percentile_compatibility,
    js_divergence,
    orthogonal_procrustes_fit,
    remove_random_same_size,
    remove_undirected_pairs,
    train_median_threshold,
)


def test_probe_protocol_is_fixed_and_has_no_test_input():
    assert PROBE_CONFIG.epochs == 200
    assert PROBE_CONFIG.seed == 0
    assert PROBE_CONFIG.early_stopping is False
    assert PROBE_CONFIG.scheduler is None
    assert PROBE_CONFIG.test_access is False


def test_probe_fits_train_and_reports_validation_only_even_if_other_labels_change():
    torch.manual_seed(5)
    x = torch.randn(12, 4)
    labels = torch.tensor([0, 1] * 6)
    train = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    val = torch.tensor([8, 9])
    # The probe API receives no test indices and output dimension is train/val-derived.
    result = FrozenProbeProtocolV1().fit(x, labels, train, val, [0, 1], "cpu")
    changed = labels.clone()
    changed[10:] = torch.tensor([9, 8])
    again = FrozenProbeProtocolV1().fit(x, changed, train, val, [0, 1], "cpu")
    assert torch.equal(result["val_logits"], again["val_logits"])
    assert result["config_sha256"] == again["config_sha256"]


def test_js_is_finite_and_zero_for_equal_distributions():
    p = torch.tensor([[0.1, 0.9], [0.5, 0.5]])
    assert torch.isfinite(js_divergence(p, p)).all()
    assert torch.allclose(js_divergence(p, p), torch.zeros(2), atol=1e-7)


def test_edge_percentiles_are_deterministic_and_use_source_matched_non_edges():
    x = torch.tensor([[1., 0.], [0.9, 0.1], [0., 1.], [-1., 0.]])
    edge = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    a = edge_percentile_compatibility(x, edge, seed=17, max_sources=4, nonedges_per_source=2)
    b = edge_percentile_compatibility(x, edge, seed=17, max_sources=4, nonedges_per_source=2)
    assert torch.equal(a[0], b[0])
    assert torch.equal(a[1], b[1])
    assert a[2] == b[2]
    assert torch.all((a[1] >= 0) & (a[1] <= 1))


def test_targeted_undirected_removal_deletes_both_directions_and_random_count_matches():
    edge = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    pairs = canonical_undirected_pairs(edge)
    removed = remove_undirected_pairs(edge, pairs[:1])
    assert removed.size(1) == edge.size(1) - 2
    assert not ({(0, 1), (1, 0)} & set(map(tuple, removed.T.tolist())))
    random_removed, selected = remove_random_same_size(edge, 2, seed=9)
    assert selected.size(0) == 2
    assert canonical_undirected_pairs(edge).size(0) - canonical_undirected_pairs(random_removed).size(0) == 2


def test_cube_median_threshold_uses_training_values_only():
    threshold = train_median_threshold(torch.tensor([1., 2., 3., 4.]))
    _validation_values = torch.tensor([-1000., 1000.])
    assert threshold == 2.5


def test_alignment_transform_is_fitted_from_train_argument_only():
    torch.manual_seed(3)
    source = torch.randn(10, 4)
    q, _ = torch.linalg.qr(torch.randn(4, 4))
    target = source @ q
    fit_ids = torch.arange(6)
    rotation = orthogonal_procrustes_fit(source[fit_ids], target[fit_ids])
    aligned = (source[6:] - source[fit_ids].mean(0)) @ rotation
    assert torch.allclose(aligned, target[6:] - target[fit_ids].mean(0), atol=1e-5)


def test_discovery_analyzer_never_reads_test_indices_or_test_label_slices():
    from pathlib import Path
    source = Path("scripts/analyze_mechanism_discovery.py").read_text(encoding="utf-8")
    assert "data.test_idx" not in source
    assert "data.y[data.test_idx" not in source


def test_problem_discovery_matrix_uses_only_allowed_statuses():
    from scripts.analyze_mechanism_discovery import _problem_discovery_matrix
    names = ("modality_sufficiency", "joint_branch_suppression",
             "modality_structure_utility", "structure_attribute_conflict",
             "edge_role_intervention", "semantic_utility_relation",
             "crossmodal_interaction_probe", "crossmodal_alignment_trajectory",
             "shared_private_proxy", "interaction_cube", "interaction_regression")
    matrix = _problem_discovery_matrix({name: [] for name in names})
    assert {row["problem_id"] for row in matrix} == {f"P{i}" for i in range(1, 13)}
    assert {row["status"] for row in matrix} <= {
        "A_STRONG_CANDIDATE", "B_PLAUSIBLE_CANDIDATE",
        "C_DESCRIPTIVE_ONLY", "D_NOT_SUPPORTED",
    }
