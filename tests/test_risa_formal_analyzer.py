from scripts.analyze_risa_p0_formal import _optional_mechanism_summary


def test_formal_analyzer_skips_router_summary_for_non_routed_variants():
    result_without_router_fields = {"fused_z": object()}
    for variant in (
        "p0_identity", "p0_masspres_scalar", "p0_operator_uniform", "p0_operator_global",
    ):
        assert _optional_mechanism_summary(variant, result_without_router_fields) is None


def test_macro_f1_labels_use_train_and_validation_only():
    import torch
    from types import SimpleNamespace

    from scripts.analyze_risa_p0_formal import _validation_label_ids

    class SplitData(SimpleNamespace):
        @property
        def test_idx(self):
            raise AssertionError("test split must not be read")

    data = SplitData(
        y=torch.tensor([0, 1, 2, 5, 5, 0]),
        train_idx=torch.tensor([0, 1, 2]),
        val_idx=torch.tensor([4, 5]),
        num_classes=6,
    )
    assert _validation_label_ids(data) == [0, 1, 2, 5]
