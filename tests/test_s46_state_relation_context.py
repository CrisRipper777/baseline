from __future__ import annotations

import inspect

import numpy as np
import torch

from scripts import analyze_s46_state_relation_context as s46


def test_p_is_fixed_mean_of_s1_s2_s3() -> None:
    states = [torch.zeros(3, 4)] + [torch.randn(3, 4) for _ in range(3)]
    expected = (states[1] + states[2] + states[3]) / 3.0
    actual = s46.p_state(states)
    assert torch.equal(actual, expected)


def test_alpha_quarter_is_bit_identical_to_historical_uniform_readout() -> None:
    states = [torch.randn(5, 7) for _ in range(4)]
    uniform = sum(states) / 4.0
    alpha_quarter = s46.composition_state(states, .25, uniform)
    expected_mixture = .25 * states[0] + .75 * s46.p_state(states)
    assert torch.allclose(expected_mixture, uniform, atol=1e-5, rtol=1e-6)
    assert torch.equal(alpha_quarter, uniform)


def test_composition_suite_emits_landscape_pairs_and_node_oracle(monkeypatch) -> None:
    rng = torch.Generator().manual_seed(7)
    states = {name: [torch.randn(12, 3, generator=rng) for _ in range(4)]
              for name in ("text", "visual")}
    uniform = {name: sum(values) / 4.0 for name, values in states.items()}
    train = torch.arange(8)
    val = torch.arange(8, 12)
    y_train = np.array([0, 1] * 4)
    y_val = np.array([0, 1, 0, 1])

    def fake_fit(x_train, labels_train, x_val, labels_val, classes):
        probs = np.full((len(labels_val), 2), 0.5)
        pred = np.zeros(len(labels_val), dtype=np.int64)
        return {"probs": probs, "pred": pred, "accuracy": float((pred == labels_val).mean()),
                "macro_f1": 0.3, "true_label_ce": float(np.log(2)),
                "converged": True, "n_iter": 1}

    monkeypatch.setattr(s46, "fit_logistic", fake_fit)
    landscape, modality, oracle = [], [], []
    s46.composition_suite("unit", 42, s46.VARIANTS[0], "IDENTITY", states,
                          train, val, y_train, y_val, [0, 1], uniform,
                          landscape, modality, oracle)
    assert len(landscape) == 27
    assert len(modality) == 6
    assert len(oracle) == 27
    assert {(row["alpha_text"], row["alpha_visual"]) for row in modality} == set(s46.MODALITY_ALPHA_PAIRS)


def test_standardizer_uses_train_statistics_only() -> None:
    train = np.array([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]], dtype=np.float32)
    validation = np.array([[100.0, -100.0]], dtype=np.float32)
    x_train, x_val, scaler = s46.standardize_pair(train, validation)
    np.testing.assert_allclose(scaler.mean_, train.mean(0))
    np.testing.assert_allclose(x_train.mean(0), 0.0, atol=1e-7)
    np.testing.assert_allclose(x_val[0], (validation[0] - train.mean(0)) / train.std(0))
    assert not np.allclose(scaler.mean_, np.vstack((train, validation)).mean(0))


def test_residual_fit_is_invariant_to_validation_targets() -> None:
    rng = np.random.default_rng(0)
    source_train = rng.normal(size=(24, 5)).astype(np.float32)
    source_val = rng.normal(size=(8, 5)).astype(np.float32)
    target_train = (source_train @ rng.normal(size=(5, 4))).astype(np.float32)
    target_val = rng.normal(size=(8, 4)).astype(np.float32)
    train_a, val_a = s46.residualize_train_only(source_train, source_val, target_train, target_val)
    train_b, _ = s46.residualize_train_only(source_train, source_val, target_train, target_val + 500)
    np.testing.assert_array_equal(train_a, train_b)
    assert not np.allclose(val_a, s46.residualize_train_only(
        source_train, source_val, target_train, target_val + 500)[1])


def test_oof_utility_has_one_prediction_for_each_train_node() -> None:
    rng = np.random.default_rng(4)
    train_nodes = np.arange( ninety := 90)
    val_nodes = np.arange(90, 120)
    y_train = np.repeat(np.arange(3), 30)
    y_val = np.tile(np.arange(3), 10)
    self_features = rng.normal(size=(120, 8)).astype(np.float32)
    struct_features = self_features + rng.normal(scale=.2, size=(120, 8)).astype(np.float32)
    g_train, g_val, self_ce, struct_ce = s46.fit_utility_oof(
        self_features, struct_features, y_train, y_val, [0, 1, 2],
        train_nodes, val_nodes, "unit", 42, "text")
    assert len(g_train) == len(train_nodes) == len(self_ce) == len(struct_ce)
    assert len(g_val) == len(val_nodes)
    assert np.isfinite(g_train).all() and np.isfinite(g_val).all()


def test_degree_bin_shuffle_preserves_each_bin_marginal() -> None:
    degree = np.array([0, 0, 1, 1, 2, 2, 10, 10, 11, 11])
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    shuffled = s46.shuffle_within_degree_bins(values, degree, np.random.default_rng(13))
    bins = s46.degree_bins(degree)
    for b in np.unique(bins):
        ids = np.flatnonzero(bins == b)
        np.testing.assert_array_equal(np.sort(shuffled[ids, 0]), np.sort(values[ids, 0]))
        np.testing.assert_array_equal(np.sort(shuffled[ids, 1]), np.sort(values[ids, 1]))


def test_relation_features_are_finite_for_degree_zero_and_one() -> None:
    class Dummy:
        calibrated = True
        mass_preserving = False

    indices = torch.tensor([[0, 1, 1], [1, 0, 2]])
    values = torch.tensor([.25, .25, .5], dtype=torch.float32)
    p_rel = torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()
    result = {
        "P_rel": p_rel,
        "raw_gates": {"text": torch.tensor([[1.0], [1.5], [.5]]),
                      "visual": torch.tensor([[.8], [1.2], [1.0]])},
    }
    features, degree = s46.extract_relation_features(Dummy(), result, 4, np.array([0, 1, 2]))
    assert degree.tolist() == [1, 2, 0, 0]
    assert all(np.isfinite(item).all() for item in features.values())
    assert features["text_dosage"][2] == 1.0
    assert features["text_normalized_entropy"][0] == 0.0
    assert features["text_max_contribution_share"][0] == 1.0
    assert features["text_effective_neighbor_count"][0] == 1.0
    assert features["text_effective_neighbor_count"][2] == 0.0


def test_relation_utility_rejects_mass_preserving_gate_scale() -> None:
    class Dummy:
        calibrated = True
        mass_preserving = True

    p_rel = torch.sparse_coo_tensor(torch.tensor([[0], [1]]), torch.tensor([1.0]), (2, 2)).coalesce()
    result = {"P_rel": p_rel, "raw_gates": {"text": torch.ones(1, 1), "visual": torch.ones(1, 1)}}
    try:
        s46.extract_relation_features(Dummy(), result, 2, np.array([0, 1]))
    except ValueError as error:
        assert "unconstrained" in str(error)
    else:
        raise AssertionError("Mass-preserving raw gate magnitudes must not be used for utility")


def test_train_validation_loader_never_indexes_test_labels() -> None:
    source = inspect.getsource(s46.load_dataset)
    assert "test_idx" not in source
    assert "data.y[" in source
    assert "data.y[data.test_idx]" not in source


def test_toys_are_architecture_holdout_and_test_lp_are_out_of_scope() -> None:
    assert "Toys" not in s46.DATASETS
    assert s46.VARIANTS == (
        "s45_identity_uniform", "s45_unconstrained_entry_uniform", "s45_masspres_entry_uniform")
    assert s46.PROTOCOL == "unified_full_graph_nc_v1"

