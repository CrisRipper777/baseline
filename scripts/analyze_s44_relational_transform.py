from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "results/s43_h15_h1r_h2ab_v1"
PATCH = ROOT / "results/s44_relational_transform_v1/phase2_analysis_patch"
DATASETS = ("Movies", "Grocery", "ele-fashion", "Reddit-S")
SEEDS = (42, 43, 44)
MODALITIES = ("text", "visual")
H2A = ("h2a_global_scalar", "h2a_node_scalar", "h2a_global_group", "h2a_node_group")
H2B = ("h2b_global_agg_correction", "h2b_global_diff_correction",
       "h2b_node_agg_correction", "h2b_node_diff_correction")


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _patch_table(source: Path, target: Path, variants: tuple[str, ...]) -> int:
    fields, rows = _read(source)
    if not {"dataset", "seed", "modality"}.issubset(fields):
        raise ValueError(f"unexpected historical schema: {source}")
    expected = [
        (dataset, str(seed), modality, variant)
        for dataset in DATASETS for variant in variants for seed in SEEDS for modality in MODALITIES
    ]
    if len(rows) != len(expected):
        raise ValueError(f"historical row count mismatch for {source}: {len(rows)} != {len(expected)}")
    output: list[dict[str, str]] = []
    for index, (row, key) in enumerate(zip(rows, expected, strict=True)):
        dataset, seed, modality, variant = key
        observed = (row["dataset"], row["seed"], row["modality"])
        if observed != (dataset, seed, modality):
            raise ValueError(f"row {index + 2} ordering mismatch in {source}: {observed} != {(dataset, seed, modality)}")
        output.append({"variant": variant, **row})
    fields = ["variant", *fields]
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    return len(output)


def run_patch() -> dict[str, int]:
    PATCH.mkdir(parents=True, exist_ok=True)
    counts = {
        "h2a_gate_statistics_v2.csv": _patch_table(
            OLD / "h2a_gate_statistics.csv", PATCH / "h2a_gate_statistics_v2.csv", H2A),
        "h2b_lambda_statistics_v2.csv": _patch_table(
            OLD / "h2b_lambda_statistics.csv", PATCH / "h2b_lambda_statistics_v2.csv", H2B),
    }
    report = """# S4.4 Phase 2 analysis patch\n\nThis analysis-only patch restores the omitted `variant` column in copies of the S4.3 H2a/H2b statistics. Historical CSVs and model outputs were read only; no training was run. Variant labels are assigned using the historical writer order (dataset, variant, seed, modality), and each source row's dataset/seed/modality is checked against that order before writing.\n\n| Table | Historical rows | Patched rows |\n|---|---:|---:|\n"""
    for filename, count in counts.items():
        report += f"| `{filename}` | {count} | {count} |\n"
    report += "\nScope: Movies, Grocery, ele-fashion, Reddit-S; seeds 42–44; modalities text and visual.\n"
    (PATCH / "phase2_gate_lambda_patch_report.md").write_text(report, encoding="utf-8")
    return counts



# Analysis helpers below are intentionally read-only with respect to all historical outputs.
import hashlib
import json
import math
import statistics
from collections import defaultdict
from functools import lru_cache
from typing import Any

OUTPUT_ROOT = ROOT / "outputs/s44_relational_transform_v1"
RESULT_ROOT = ROOT / "results/s44_relational_transform_v1"
FORMAL_VARIANTS = (
    "s44_scalar_global", "s44_scalar_raw", "s44_scalar_rel", "s44_scalar_rel_multi",
    "s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi",
    "s44_expert_uniform", "s44_expert_rel", "s44_expert_rel_multi",
)
SOURCE_COMMIT = "7b98b31a78fb978fadd96448bbcfd3ffa295d82f"


def _config(dataset: str, model_name: str, variant: str, seed: int):
    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", f"model={model_name}", f"model.variant={variant}",
            f"seed={seed}", "num_runs=1", "device=cpu", "task.evaluate_test=false",
            "task.training_mode=full_graph", "task.protocol_version=unified_full_graph_nc_v1",
        ])


@lru_cache(maxsize=1)
def _load_data(dataset: str):
    from src.data import load_mag_data
    # The NC launcher loads one split at base seed 42, then varies model RNG seeds.
    cfg = _config(dataset, "relational_transform_pilot", "s44_scalar_global", 42)
    return load_mag_data(cfg, "nc", 42)


def _data_info(data) -> dict[str, int]:
    return {"input_dim": data.input_dim, "num_nodes": data.num_nodes,
            "num_classes": data.num_classes, "text_dim": int(data.x_t.size(1)),
            "visual_dim": int(data.x_i.size(1))}


def _load_run(dataset: str, seed: int, variant: str, data, device: torch.device):
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels
    checkpoint = OUTPUT_ROOT / "formal" / dataset / variant / f"best_run{SEEDS.index(seed) + 1}.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != "unified_full_graph_nc_v1":
        raise ValueError(f"invalid NC protocol in {checkpoint}")
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"invalid seed or selection in {checkpoint}")
    if any(key.startswith("test_") for key in payload.get("metrics", {})):
        raise ValueError(f"test evaluation found in {checkpoint}")
    cfg = _config(dataset, "relational_transform_pilot", variant, seed)
    model = build_model(cfg, _data_info(data)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval(); head.eval()
    labels = _resolve_nc_eval_labels(data)
    return model, head, payload, labels


def _load_historical_p0(dataset: str, seed: int, data, device: torch.device):
    from src.models import build_model
    from src.tasks.nc import _resolve_nc_eval_labels
    ckpt = ROOT / "outputs/s43_p0_h1_v1/formal" / dataset / "p0_residual" / f"best_run{SEEDS.index(seed) + 1}.pt"
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if int(payload.get("seed", -1)) != seed or payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"invalid historical P0 checkpoint {ckpt}")
    cfg = _config(dataset, "relation_basis_pilot", "p0_residual", seed)
    model = build_model(cfg, _data_info(data)).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    head = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    head.load_state_dict(payload["head_state"], strict=True)
    model.eval(); head.eval()
    return model, head, payload, _resolve_nc_eval_labels(data)


def _metrics(logits: torch.Tensor, data, eval_labels: list[int]) -> tuple[dict[str, float], torch.Tensor]:
    val_cpu = data.val_idx.detach().cpu().long()
    val = val_cpu.to(logits.device)
    target = data.y[val_cpu].to(logits.device, dtype=torch.long)
    val_logits = logits[val]
    pred = val_logits.argmax(dim=-1)
    metrics = {
        "val_acc": float((pred == target).float().mean()),
        "val_macro_f1": float(f1_score(target.cpu().numpy(), pred.cpu().numpy(), labels=eval_labels,
                                        average="macro", zero_division=0)),
        "true_label_ce": float(F.cross_entropy(val_logits, target)),
    }
    return metrics, pred.detach().cpu()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def _std(values: list[float]) -> float:
    return statistics.pstdev(values) if values else math.nan


def _intervention_row(dataset: str, seed: int, variant: str, name: str,
                      metrics: dict[str, float], normal: dict[str, float],
                      pred: torch.Tensor, normal_pred: torch.Tensor, **extra) -> dict[str, Any]:
    row: dict[str, Any] = {"dataset": dataset, "seed": seed, "variant": variant, "intervention": name,
                           **metrics,
                           "delta_val_acc": metrics["val_acc"] - normal["val_acc"],
                           "delta_val_macro_f1": metrics["val_macro_f1"] - normal["val_macro_f1"],
                           "delta_true_label_ce": metrics["true_label_ce"] - normal["true_label_ce"],
                           "prediction_flip_rate": float((pred != normal_pred).float().mean())}
    row.update(extra)
    return row


def _target_edge_shuffle(controller: torch.Tensor, rows: torch.Tensor, val_mask: torch.Tensor,
                          shuffle_seed: int) -> torch.Tensor:
    """Shuffle a controller only among each validation target's incoming edges."""
    source = controller.detach().cpu()
    shuffled = source.clone()
    row_cpu = rows.detach().cpu().long()
    values, counts = torch.unique_consecutive(row_cpu, return_counts=True)
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    offset = 0
    for target, count in zip(values.tolist(), counts.tolist(), strict=True):
        end = offset + count
        if val_mask[target] and count > 1:
            permutation = torch.randperm(count, generator=generator)
            shuffled[offset:end] = source[offset:end][permutation]
        offset = end
    return shuffled


def _run_controller_intervention(model, head, data, x_gpu, edge_gpu, val_gpu, labels,
                                 controller: dict[str, torch.Tensor], controller_rows: torch.Tensor,
                                 cross_mode: str = "normal", full_eval: bool = False):
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[data.val_idx.detach().cpu().long()] = True
    edge_mask = (torch.ones(controller_rows.numel(), dtype=torch.bool) if full_eval
                 else val_mask[controller_rows.detach().cpu().long()])
    local_controller = {}
    for key, values in controller.items():
        if values.size(0) == controller_rows.numel():
            values = values[edge_mask.to(values.device)]
        local_controller[key] = values
    with torch.no_grad():
        analysis = model.analyze(x_gpu, edge_gpu, controller_override=local_controller,
                                 cross_modal_mode=cross_mode,
                                 target_nodes=None if full_eval else val_gpu)
        logits = head(analysis["fused_z"])
        return _metrics(logits, data, labels)


def _add_controller_audits(dataset, seed, variant, model, head, data, x_gpu, edge_gpu,
                           val_gpu, labels, normal_metrics, normal_pred, normal_analysis,
                           intervention_rows):
    controllers = normal_analysis["controllers"]
    edge_rows = normal_analysis["edge_row"]
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask[data.val_idx.cpu().long()] = True
    val_edge = val_mask[edge_rows.detach().cpu().long()]
    if not bool(val_edge.any()):
        raise RuntimeError(f"no validation-adjacent directed edges in {dataset} seed={seed}")
    val_rows = edge_rows[val_edge.to(edge_rows.device)]
    val_controllers = {key: value[val_edge.to(value.device)].detach().clone()
                       for key, value in controllers.items()}
    mean_override = {key: value.mean(dim=0, keepdim=True).expand_as(value).clone()
                     for key, value in val_controllers.items()}
    metrics, pred = _run_controller_intervention(model, head, data, x_gpu, edge_gpu,
                                                  val_gpu, labels, mean_override, val_rows)
    intervention_rows.append(_intervention_row(dataset, seed, variant, "MEAN_REPLACE", metrics,
                                               normal_metrics, pred, normal_pred,
                                               validation_target_edges=int(val_edge.sum())))

    shuffle_rows = []
    variant_code = FORMAL_VARIANTS.index(variant) + 1
    for replicate in range(10):
        shuffled = {
            modality: _target_edge_shuffle(values, val_rows, val_mask,
                                           seed * 1000003 + variant_code * 10007 + replicate)
            for modality, values in val_controllers.items()
        }
        shuffled = {key: value.to(x_gpu.device) for key, value in shuffled.items()}
        metrics, pred = _run_controller_intervention(model, head, data, x_gpu, edge_gpu,
                                                      val_gpu, labels, shuffled, val_rows)
        row = _intervention_row(dataset, seed, variant, "EDGE_SHUFFLE_REPLICATE", metrics,
                                normal_metrics, pred, normal_pred,
                                shuffle_replicate=replicate + 1,
                                shuffle_seed=seed * 1000003 + variant_code * 10007 + replicate,
                                validation_target_edges=int(val_edge.sum()))
        intervention_rows.append(row)
        shuffle_rows.append(row)
    summary = {"dataset": dataset, "seed": seed, "variant": variant,
               "intervention": "EDGE_SHUFFLE_MEAN_STD", "replicates": len(shuffle_rows),
               "validation_target_edges": int(val_edge.sum())}
    for metric in ("val_acc", "val_macro_f1", "true_label_ce", "delta_val_acc",
                   "delta_val_macro_f1", "delta_true_label_ce", "prediction_flip_rate"):
        vals = [float(row[metric]) for row in shuffle_rows]
        summary[f"mean_{metric}"] = _mean(vals)
        summary[f"std_{metric}"] = _std(vals)
    intervention_rows.append(summary)

    if variant in {"s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi"}:
        zeros = {key: torch.zeros_like(value) for key, value in controllers.items()}
        # Exact nested P0 carrier check at the same frozen checkpoint parameters.
        with torch.no_grad():
            ht, hv = normal_analysis["H_text"], normal_analysis["H_visual"]
            tt, tv = model.transform_a_text, model.transform_a_visual
            rt = model._tail(tt, normal_analysis["base_pre_text"])
            rv = model._tail(tv, normal_analysis["base_pre_visual"])
            zt = model.output_norm_text(ht + rt)
            zv = model.output_norm_visual(hv + rv)
            base_logits = head(model.plain_fusion(torch.cat((zt, zv), dim=-1)))
            if variant == "s44_lowrank_global":
                saved = {name: getattr(model, f"global_a_{name}").detach().clone() for name in ("text", "visual")}
                try:
                    for name in ("text", "visual"):
                        getattr(model, f"global_a_{name}").zero_()
                    zero_analysis = model.analyze(x_gpu, edge_gpu)
                finally:
                    for name in ("text", "visual"):
                        getattr(model, f"global_a_{name}").copy_(saved[name])
            else:
                zero_analysis = model.analyze(x_gpu, edge_gpu, controller_override=zeros)
            zero_logits = head(zero_analysis["fused_z"])
            max_error = float((base_logits[val_gpu] - zero_logits[val_gpu]).abs().max())
            # The algebraic carrier is identical; allow only standard GPU GEMM roundoff.
            zero_dynamic_atol, zero_dynamic_rtol = 1e-5, 1e-6
            if not torch.allclose(base_logits[val_gpu], zero_logits[val_gpu],
                                  atol=zero_dynamic_atol, rtol=zero_dynamic_rtol):
                raise AssertionError(f"ZERO_DYNAMIC differs from P0 branch beyond tolerance: max error={max_error}")
            metrics, pred = _metrics(zero_logits, data, labels)
        zero_row = _intervention_row(dataset, seed, variant, "ZERO_DYNAMIC", metrics,
                                     normal_metrics, pred, normal_pred)
        zero_row["max_abs_logit_difference_from_p0_branch"] = max_error
        zero_row["p0_branch_atol"] = zero_dynamic_atol
        zero_row["p0_branch_rtol"] = zero_dynamic_rtol
        intervention_rows.append(zero_row)

    if variant in {"s44_expert_rel", "s44_expert_rel_multi"}:
        uniform = {key: torch.full_like(value, 0.5) for key, value in controllers.items()}
        metrics, pred = _run_controller_intervention(model, head, data, x_gpu, edge_gpu,
                                                      val_gpu, labels, uniform, edge_rows)
        intervention_rows.append(_intervention_row(dataset, seed, variant, "ROUTE_UNIFORMIZE",
                                                   metrics, normal_metrics, pred, normal_pred))


def _add_multimodal_audits(dataset, seed, variant, model, head, data, x_gpu, edge_gpu,
                           val_gpu, labels, normal_metrics, normal_pred, rows):
    for label, mode in (("CROSS_MODAL_REMOVE", "remove"), ("MODALITY_RELATION_SWAP", "swap")):
        with torch.no_grad():
            analysis = model.analyze(x_gpu, edge_gpu, cross_modal_mode=mode, target_nodes=val_gpu)
            logits = head(analysis["fused_z"])
            metrics, pred = _metrics(logits, data, labels)
        rows.append(_intervention_row(dataset, seed, variant, label, metrics, normal_metrics,
                                      pred, normal_pred, cross_modal_mode=mode))


def _add_expert_diagnostics(dataset, seed, variant, model, analysis, rows):
    for modality in ("text", "visual"):
        pi = analysis["controllers"][modality].detach().float()
        p1, p2 = pi[:, 0], pi[:, 1]
        entropy = -(pi.clamp_min(1e-12) * pi.clamp_min(1e-12).log()).sum(-1)
        h = analysis[f"H_{modality}"]
        with torch.no_grad():
            e1 = model.get_submodule(f"expert_u1_{modality}")(model.get_submodule(f"expert_v1_{modality}")(h)).float()
            e2 = model.get_submodule(f"expert_u2_{modality}")(model.get_submodule(f"expert_v2_{modality}")(h)).float()
            cosine = F.cosine_similarity(e1, e2, dim=-1, eps=1e-8)
            norm_ratio = torch.linalg.vector_norm(e1, dim=-1) / torch.linalg.vector_norm(e2, dim=-1).clamp_min(1e-8)
        q = torch.quantile(entropy.cpu(), torch.tensor([.1, .25, .5, .75, .9]))
        rows.append({
            "dataset": dataset, "seed": seed, "variant": variant, "modality": modality,
            "mean_pi1": float(p1.mean()), "mean_pi2": float(p2.mean()),
            "mean_entropy": float(entropy.mean()), "std_entropy": float(entropy.std(unbiased=False)),
            **{f"entropy_q{n}": float(v) for n, v in zip((10, 25, 50, 75, 90), q)},
            "fraction_pi_max_gt_0_75": float((pi.max(-1).values > .75).float().mean()),
            "fraction_pi_max_gt_0_90": float((pi.max(-1).values > .90).float().mean()),
            "expert1_utilization": float(p1.mean()), "expert2_utilization": float(p2.mean()),
            "mean_expert_output_cosine": float(cosine.mean()),
            "std_expert_output_cosine": float(cosine.std(unbiased=False)),
            "mean_expert1_to_expert2_norm_ratio": float(norm_ratio.mean()),
            "directed_edge_count": int(pi.size(0)),
        })


def _semantic_audit_rows(dataset: str, seed: int, variant: str, analysis: dict[str, Any], data) -> list[dict[str, Any]]:
    from scipy.stats import spearmanr
    from src.analysis.problem_deep_dive import train_edge_quantile_bins

    cache_path = ROOT / "outputs/problem_deep_dive_v1/checkpoint_cache" / f"{dataset}_seed{seed}.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if cache.get("dataset") != dataset or int(cache.get("seed", -1)) != seed:
        raise ValueError(f"D3 semantic cache identity mismatch: {cache_path}")
    if not torch.equal(cache["edge_index"].long(), data.edge_index.detach().cpu().long()):
        raise ValueError(f"D3 physical graph differs from S4.4 graph: {cache_path}")
    n = int(cache["num_nodes"])
    pairs = cache["edge_cache"]["pairs"].long()
    edge_rows = analysis["edge_row"].detach().cpu().long()
    edge_cols = analysis["edge_col"].detach().cpu().long()
    val_mask = torch.zeros(n, dtype=torch.bool)
    val_mask[data.val_idx.detach().cpu().long()] = True
    keep = val_mask[edge_rows]
    edge_rows, edge_cols = edge_rows[keep], edge_cols[keep]
    keys = pairs[:, 0] * n + pairs[:, 1]
    directed_keys = torch.minimum(edge_rows, edge_cols) * n + torch.maximum(edge_rows, edge_cols)
    index = torch.searchsorted(keys, directed_keys)
    if index.numel() and (int(index.max()) >= keys.numel() or not torch.equal(keys[index], directed_keys)):
        raise RuntimeError("S4.4 directed P_rel edges do not match cached D3 canonical pairs")
    controllers = {key: value.detach().float().cpu()[keep] for key, value in analysis["controllers"].items()}

    edge_cache = cache["edge_cache"]
    modalities = ("text", "visual")
    semantic_values = {
        "text": edge_cache["compat"]["text_uniform"].float(),
        "visual": edge_cache["compat"]["visual_uniform"].float(),
    }
    # D3 contributes its exact per-seed compatibility percentiles; quartiles are
    # recomputed with the actual S4.4 training split (the shared base-seed split).
    train_idx = data.train_idx.detach().cpu().long()
    quartiles, thresholds_by_modality = {}, {}
    for modality in modalities:
        q, thresholds = train_edge_quantile_bins(semantic_values[modality], pairs, train_idx)
        quartiles[modality] = q[index]
        thresholds_by_modality[modality] = thresholds
    selected_semantic = {modality: semantic_values[modality][index] for modality in modalities}

    rows: list[dict[str, Any]] = []
    for controller_modality, values in controllers.items():
        if values.dim() == 1:
            values = values[:, None]
        for component in range(values.size(1)):
            control = values[:, component]
            for semantic_modality in modalities:
                semantic = selected_semantic[semantic_modality]
                finite = torch.isfinite(control) & torch.isfinite(semantic)
                if int(finite.sum()) > 1 and float(control[finite].max()) > float(control[finite].min()) and float(semantic[finite].max()) > float(semantic[finite].min()):
                    rho = float(spearmanr(control[finite].numpy(), semantic[finite].numpy()).statistic)
                else:
                    rho = math.nan
                total_var = float(control[finite].var(unbiased=False)) if int(finite.sum()) else math.nan
                thresholds = thresholds_by_modality[semantic_modality]
                rows.append({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "controller_modality": controller_modality,
                    "controller_component": "g" if variant.startswith("s44_scalar_")
                    else (f"a{component}" if "lowrank" in variant else f"pi{component + 1}"),
                    "semantic_modality": semantic_modality, "semantic_quartile": "ALL",
                    "validation_adjacent_directed_edges": int(finite.sum()), "spearman_rho": rho,
                    "controller_mean": float(control[finite].mean()) if int(finite.sum()) else math.nan,
                    "controller_std": float(control[finite].std(unbiased=False)) if int(finite.sum()) else math.nan,
                    "controller_variance": total_var, "within_quartile_variance_fraction": math.nan,
                    "train_semantic_q25": thresholds["q25"], "train_semantic_q50": thresholds["q50"],
                    "train_semantic_q75": thresholds["q75"],
                    "semantic_source": "D3 cached per-seed modality-uniform H0; source_matched_compatibility(seed*101+7)",
                    "validation_split_seed": 42, "train_threshold_split_seed": 42,
                })
                q_values = quartiles[semantic_modality]
                for qid in range(4):
                    mask = (q_values == qid) & finite
                    part = control[mask]
                    rows.append({
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "controller_modality": controller_modality,
                        "controller_component": "g" if variant.startswith("s44_scalar_")
                        else (f"a{component}" if "lowrank" in variant else f"pi{component + 1}"),
                        "semantic_modality": semantic_modality, "semantic_quartile": f"Q{qid}",
                        "validation_adjacent_directed_edges": int(mask.sum()), "spearman_rho": math.nan,
                        "controller_mean": float(part.mean()) if part.numel() else math.nan,
                        "controller_std": float(part.std(unbiased=False)) if part.numel() else math.nan,
                        "controller_variance": float(part.var(unbiased=False)) if part.numel() else math.nan,
                        "within_quartile_variance_fraction":
                            float(part.var(unbiased=False)) / total_var if part.numel() and total_var > 0 else math.nan,
                        "train_semantic_q25": thresholds["q25"], "train_semantic_q50": thresholds["q50"],
                        "train_semantic_q75": thresholds["q75"],
                        "semantic_source": "D3 cached per-seed modality-uniform H0; source_matched_compatibility(seed*101+7)",
                    "validation_split_seed": 42, "train_threshold_split_seed": 42,
                    })
    return rows


def _paired_contrasts(metric_map: dict[tuple[str, int, str], dict[str, float]], datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    specifications = (
        ("scalar_rel_minus_raw", "s44_scalar_raw", "s44_scalar_rel", "PRIMARY_PAIRED"),
        ("scalar_rel_minus_global", "s44_scalar_global", "s44_scalar_rel", "PRIMARY_PAIRED"),
        ("scalar_rel_multi_minus_rel", "s44_scalar_rel", "s44_scalar_rel_multi", "PRIMARY_PAIRED"),
        ("lowrank_rel_minus_global", "s44_lowrank_global", "s44_lowrank_rel", "PRIMARY_PAIRED"),
        ("lowrank_rel_multi_minus_rel", "s44_lowrank_rel", "s44_lowrank_rel_multi", "PRIMARY_PAIRED"),
        ("expert_rel_minus_uniform", "s44_expert_uniform", "s44_expert_rel", "PRIMARY_PAIRED"),
        ("expert_rel_multi_minus_rel", "s44_expert_rel", "s44_expert_rel_multi", "PRIMARY_PAIRED"),
        ("lowrank_global_minus_p0", "p0_residual_historical", "s44_lowrank_global", "DESCRIPTIVE_ONLY"),
        ("lowrank_rel_minus_p0", "p0_residual_historical", "s44_lowrank_rel", "DESCRIPTIVE_ONLY"),
        ("lowrank_rel_multi_minus_p0", "p0_residual_historical", "s44_lowrank_rel_multi", "DESCRIPTIVE_ONLY"),
    )
    rows = []
    for name, left, right, label in specifications:
        group = []
        for dataset in datasets:
            for seed in SEEDS:
                a, b = metric_map.get((dataset, seed, left)), metric_map.get((dataset, seed, right))
                if a is None or b is None:
                    continue
                row = {"dataset": dataset, "seed": seed, "contrast": name,
                       "left_variant": left, "right_variant": right, "comparison_label": label,
                       "delta_val_acc": b["val_acc"] - a["val_acc"],
                       "delta_val_macro_f1": b["val_macro_f1"] - a["val_macro_f1"],
                       "delta_true_label_ce": b["true_label_ce"] - a["true_label_ce"]}
                rows.append(row); group.append(row)
        aggregate = {"dataset": "ALL", "seed": "ALL", "contrast": name,
                     "left_variant": left, "right_variant": right, "comparison_label": label}
        for key in ("delta_val_acc", "delta_val_macro_f1", "delta_true_label_ce"):
            vals = [float(item[key]) for item in group]
            aggregate[f"mean_{key}"] = _mean(vals)
            aggregate[f"population_std_{key}"] = _std(vals)
            aggregate[f"positive_seed_pairs_{key}"] = sum(v > 0 for v in vals)
            aggregate[f"total_seed_pairs_{key}"] = len(vals)
        rows.append(aggregate)
    return rows


def _complexity_rows(datasets: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for dataset in datasets:
        p0_path = ROOT / "outputs/s43_p0_h1_v1/formal" / dataset / "p0_residual" / "complete.json"
        p0 = json.loads(p0_path.read_text(encoding="utf-8"))
        p0_epoch = float(p0["mean_epoch_seconds_proxy"])
        for variant in FORMAL_VARIANTS:
            path = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
            complete = json.loads(path.read_text(encoding="utf-8"))
            epoch = float(complete["mean_epoch_seconds_proxy"])
            peak = int(complete["peak_cuda_memory_mb"])
            rows.append({
                "dataset": dataset, "variant": variant, "trainable_params": complete["trainable_params"],
                "model_trainable_params": complete["model_trainable_params"],
                "classifier_trainable_params": complete["classifier_trainable_params"],
                "peak_cuda_memory_mb": peak, "wall_clock_seconds": complete["wall_clock_seconds"],
                "mean_epoch_seconds_proxy": epoch, "historical_p0_mean_epoch_seconds_proxy": p0_epoch,
                "epoch_time_ratio_to_p0": epoch / p0_epoch if p0_epoch > 0 else math.nan,
                "best_epochs": ";".join(map(str, complete["best_epochs"])),
                "complexity_status": "COMPUTE_HEAVY" if peak >= 20 * 1024 or epoch > 2.5 * p0_epoch else "WITHIN_REFERENCE",
                "peak_memory_source": "nvidia-smi process job delta; see complete.json for absolute device baseline/peak",
            })
    return rows


_SHA256_CACHE: dict[str, str] = {}


def _file_sha256(path: Path) -> str:
    cache_key = str(path.resolve())
    if cache_key in _SHA256_CACHE:
        return _SHA256_CACHE[cache_key]
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


def _dataset_hashes(dataset: str, seed: int) -> dict[str, str]:
    from src.data.loaders import resolve_path
    cfg = _config(dataset, "relational_transform_pilot", "s44_scalar_global", 42)
    ds = cfg.dataset
    keys = ("graph_path", "text_feat_path", "image_feat_path", "joint_feat_path", "edge_path",
            "label_path", "node_split_path", "nc_split_path")
    hashes = {}
    for key in keys:
        value = ds.get(key)
        if value is None:
            continue
        path = resolve_path(str(value))
        hashes[str(path)] = _file_sha256(path)
    return hashes


def _write_report(summary: dict[str, Any], contrasts: list[dict[str, Any]],
                  intervention_rows: list[dict[str, Any]], expert_rows: list[dict[str, Any]],
                  complexity: list[dict[str, Any]]) -> str:
    lines = [
        "# S4.4 Relational Transformation Pilot", "",
        "Scope: full-graph node classification on Movies, Grocery, ele-fashion and Reddit-S; seeds 42–44; test evaluation disabled. No LP, Toys, H3, H4 or final-model claim is included.", "",
        f"- Training commit: `{summary['provenance']['training_commit']}`",
        f"- Analysis commit: `{summary['provenance']['analysis_commit']}`",
        f"- Source branch/commit: `{summary['provenance']['source_branch']}` / `{summary['provenance']['source_commit']}`",
        f"- Formal runs: {summary['completed_runs']} / {summary['expected_runs']}",
        "- Selection: best validation accuracy; no test metrics were evaluated.", "",
        "## Primary paired contrasts", "",
        "Positive accuracy and macro-F1 deltas favor the right-hand variant; negative CE favors the right-hand variant.",
        "| Contrast | Mean Δ val acc | Positive pairs | Mean Δ macro-F1 | Mean Δ CE | Label |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in ("scalar_rel_minus_raw", "scalar_rel_minus_global", "scalar_rel_multi_minus_rel",
                 "lowrank_rel_minus_global", "lowrank_rel_multi_minus_rel",
                 "expert_rel_minus_uniform", "expert_rel_multi_minus_rel"):
        row = next(r for r in contrasts if r.get("contrast") == name and r.get("dataset") == "ALL" and "mean_delta_val_acc" in r)
        lines.append(f"| {name} | {row['mean_delta_val_acc']:.5f} | {row['positive_seed_pairs_delta_val_acc']}/{row['total_seed_pairs_delta_val_acc']} | {row['mean_delta_val_macro_f1']:.5f} | {row['mean_delta_true_label_ce']:.5f} | {row['comparison_label']} |")
    lines += ["", "## Interpretation statuses", ""]
    for name, record in summary["interpretation_statuses"].items():
        lines.append(f"- **{name}: {record['status']}** — {record['evidence']}")
    lines += [
        "", "## Functional audits", "",
        "Frozen validation interventions measure sensitivity of these trained checkpoints. They do not establish retrained causal utility. Router weights are not causal effects. Expert-routing evidence requires routed-vs-uniform benefit, edge-shuffle sensitivity and no severe collapse together. Multimodal-context evidence requires multi-vs-single paired results and remove/swap interventions.",
        "", f"- Frozen intervention rows: {len(intervention_rows)} (replicate-level edge shuffles and their mean/std are included).",
        f"- Multimodal intervention rows: {summary.get('multimodal_intervention_rows', 0)}.",
        f"- Expert diagnostic rows: {len(expert_rows)}.",
        "- Low semantic correlation alone is not evidence of task utility; inspect the paired edge-shuffle results alongside the semantic audit.",
        "", "## Complexity", "",
        "COMPUTE_HEAVY marks peak process GPU use ≥20 GiB or a mean epoch proxy >2.5× historical P0 for that dataset. Epoch proxy includes setup and validation overhead.",
        "| Dataset | Variant | Params | Peak job GPU MiB | Epoch proxy s | P0 ratio | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in complexity:
        lines.append(f"| {row['dataset']} | {row['variant']} | {row['trainable_params']} | {row['peak_cuda_memory_mb']} | {row['mean_epoch_seconds_proxy']:.3f} | {row['epoch_time_ratio_to_p0']:.2f} | {row['complexity_status']} |")
    lines += [
        "", "## Interpretation boundary", "",
        "These are validation-selected, frozen-checkpoint mechanism probes in the stated NC protocol. Frozen interventions are sensitivity analyses, not retrained counterfactuals. The model outputs do not imply a universal architecture winner or a final recommendation.",
    ]
    return "\n".join(lines) + "\n"


def _interpretation_statuses(contrasts, interventions, multimodal, experts) -> dict[str, dict[str, str]]:
    def contrast(name):
        return next(row for row in contrasts if row.get("contrast") == name and row.get("dataset") == "ALL"
                    and "mean_delta_val_acc" in row)
    def category(mean: float, positives: int, count: int, secondary: bool = False) -> str:
        if count and mean > 0 and positives > count / 2:
            return "STRONG_SUPPORT"
        if mean > 0 or secondary:
            return "MECHANISM_SUPPORT"
        if mean < 0 and positives == 0:
            return "UNSUPPORTED"
        return "MIXED"

    relation = contrast("scalar_rel_minus_raw")
    relation_status = category(float(relation["mean_delta_val_acc"]),
                               int(relation["positive_seed_pairs_delta_val_acc"]),
                               int(relation["total_seed_pairs_delta_val_acc"]))
    shuffles = [r for r in interventions if r.get("intervention") == "EDGE_SHUFFLE_MEAN_STD"]
    shuffle_deltas = [float(r["mean_delta_val_acc"]) for r in shuffles]
    shuffle_harm = -_mean(shuffle_deltas)
    shuffle_positive = sum(v < 0 for v in shuffle_deltas)
    shuffle_status = category(shuffle_harm, shuffle_positive, len(shuffle_deltas))
    lowrank = contrast("lowrank_rel_minus_global")
    zero_rows = [r for r in interventions if r.get("intervention") == "ZERO_DYNAMIC"]
    exact_zero = all(float(r.get("max_abs_logit_difference_from_p0_branch", math.inf)) <= 1e-5 for r in zero_rows)
    dynamic_status = category(float(lowrank["mean_delta_val_acc"]),
                              int(lowrank["positive_seed_pairs_delta_val_acc"]),
                              int(lowrank["total_seed_pairs_delta_val_acc"]), secondary=exact_zero)
    multi = contrast("scalar_rel_multi_minus_rel")
    multi_rows = list(multimodal)
    multi_harm = sum(float(r["delta_val_acc"]) < 0 for r in multi_rows)
    multi_status = category(float(multi["mean_delta_val_acc"]),
                            int(multi["positive_seed_pairs_delta_val_acc"]),
                            int(multi["total_seed_pairs_delta_val_acc"]), secondary=multi_harm > len(multi_rows) / 2)
    expert_pair = contrast("expert_rel_minus_uniform")
    expert_shuffle = [r for r in interventions if r.get("variant") in {"s44_expert_rel", "s44_expert_rel_multi"}
                      and r.get("intervention") == "EDGE_SHUFFLE_MEAN_STD"]
    expert_shuffle_harm = sum(float(r["mean_delta_val_acc"]) < 0 for r in expert_shuffle)
    routed_experts = [row for row in experts if row["variant"] in {"s44_expert_rel", "s44_expert_rel_multi"}]
    collapse = _mean([float(row["fraction_pi_max_gt_0_90"]) for row in routed_experts])
    expert_supported = (float(expert_pair["mean_delta_val_acc"]) > 0 and
                        expert_shuffle_harm > len(expert_shuffle) / 2 and collapse < 0.5)
    expert_status = "STRONG_SUPPORT" if expert_supported else (
        "MECHANISM_SUPPORT" if expert_shuffle_harm > len(expert_shuffle) / 2 else
        ("MIXED" if float(expert_pair["mean_delta_val_acc"]) >= 0 else "UNSUPPORTED"))
    return {
        "RelationState": {"status": relation_status,
                          "evidence": f"scalar relation minus raw mean Δaccuracy={relation['mean_delta_val_acc']:.5f}; positive paired seeds={relation['positive_seed_pairs_delta_val_acc']}/{relation['total_seed_pairs_delta_val_acc']}"},
        "EdgeConditionality": {"status": shuffle_status,
                               "evidence": f"across edge-conditioned variants, mean loss under target-wise edge shuffle={shuffle_harm:.5f}; affected variant-seed rows={shuffle_positive}/{len(shuffles)}"},
        "DynamicTransformation": {"status": dynamic_status,
                                  "evidence": f"low-rank relation minus global mean Δaccuracy={lowrank['mean_delta_val_acc']:.5f}; ZERO_DYNAMIC=P0 branch within 1e-5 absolute tolerance={exact_zero}"},
        "MultimodalRelationContext": {"status": multi_status,
                                      "evidence": f"scalar multi-relation minus relation mean Δaccuracy={multi['mean_delta_val_acc']:.5f}; remove/swap interventions lower accuracy in {multi_harm}/{len(multi_rows)} rows"},
        "ExpertRouting": {"status": expert_status,
                          "evidence": f"routed minus uniform mean Δaccuracy={expert_pair['mean_delta_val_acc']:.5f}; routing shuffle harms {expert_shuffle_harm}/{len(expert_shuffle)} rows; mean edge fraction pi_max>0.90={collapse:.3f}"},
    }


def run_analysis(datasets: tuple[str, ...] = DATASETS) -> dict[str, Any]:
    global torch, nn, F, compose, initialize_config_dir, f1_score
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from hydra import compose, initialize_config_dir
    from sklearn.metrics import f1_score
    from src.tasks.nc import _resolve_nc_eval_labels

    requested = tuple(dataset for dataset in DATASETS if dataset in datasets)
    if requested != datasets:
        raise ValueError(f"datasets must be a subset of {DATASETS} in canonical order")
    if not requested:
        raise ValueError("at least one dataset is required")
    for dataset in requested:
        for variant in FORMAL_VARIANTS:
            complete = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
            if not complete.is_file():
                raise FileNotFoundError(f"formal run is incomplete: {complete}")
            record = json.loads(complete.read_text(encoding="utf-8"))
            if record.get("test_evaluation") is not False or record.get("phase") != "formal":
                raise ValueError(f"formal job metadata violates S4.4 protocol: {complete}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    table_rows: list[dict[str, Any]] = []
    p0_rows: list[dict[str, Any]] = []
    metric_map: dict[tuple[str, int, str], dict[str, float]] = {}
    intervention_rows: list[dict[str, Any]] = []
    multimodal_rows: list[dict[str, Any]] = []
    expert_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    run_hashes = {}
    # Loop dataset/seed first to reuse the read-only graph and feature tensors.
    for dataset in requested:
        for seed in SEEDS:
            data = _load_data(dataset)
            labels = _resolve_nc_eval_labels(data)
            x_gpu = data.x.to(device)
            edge_gpu = data.edge_index.to(device)
            val_gpu = data.val_idx.to(device=device, dtype=torch.long)

            p0_model, p0_head, p0_payload, _ = _load_historical_p0(dataset, seed, data, device)
            with torch.no_grad():
                p0_analysis = p0_model.analyze(x_gpu, edge_gpu)
                p0_logits = p0_head(p0_analysis["fused_z"])
                p0_metrics, _ = _metrics(p0_logits, data, labels)
            p0_row = {"dataset": dataset, "seed": seed, "variant": "p0_residual_historical",
                      **p0_metrics, "checkpoint_best_epoch": int(p0_payload["epoch"]),
                      "trainable_params": "historical", "is_historical_reference": True,
                      "test_evaluation": False}
            p0_rows.append(p0_row)
            metric_map[(dataset, seed, "p0_residual_historical")] = p0_metrics
            del p0_model, p0_head, p0_analysis

            for variant in FORMAL_VARIANTS:
                model, head, payload, eval_labels = _load_run(dataset, seed, variant, data, device)
                with torch.no_grad():
                    analysis = model.analyze(x_gpu, edge_gpu, return_edge_state=True)
                    logits = head(analysis["fused_z"])
                    metrics, normal_pred = _metrics(logits, data, eval_labels)
                for key in ("val_acc", "val_macro_f1"):
                    if abs(metrics[key] - float(payload["metrics"][key])) > 1e-6:
                        raise RuntimeError(f"analysis failed to reproduce checkpoint {key}: {dataset}/{seed}/{variant}")
                complete_path = OUTPUT_ROOT / "formal" / dataset / variant / "complete.json"
                complete = json.loads(complete_path.read_text(encoding="utf-8"))
                base_row = {"dataset": dataset, "seed": seed, "variant": variant, **metrics,
                            "checkpoint_best_epoch": int(payload["epoch"]),
                            "trainable_params": complete["trainable_params"],
                            "is_historical_reference": False, "test_evaluation": False}
                table_rows.append(base_row)
                metric_map[(dataset, seed, variant)] = metrics
                run_hashes[f"{dataset}/{seed}/{variant}"] = {
                    "training_commit": complete["training_commit"],
                    "checkpoint": str(complete["checkpoint_paths"][SEEDS.index(seed)]),
                }

                if variant in {"s44_scalar_raw", "s44_scalar_rel", "s44_scalar_rel_multi",
                               "s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi",
                               "s44_expert_rel", "s44_expert_rel_multi"}:
                    _add_controller_audits(dataset, seed, variant, model, head, data, x_gpu, edge_gpu,
                                           val_gpu, eval_labels, metrics, normal_pred, analysis,
                                           intervention_rows)
                if variant in {"s44_scalar_rel_multi", "s44_lowrank_rel_multi", "s44_expert_rel_multi"}:
                    _add_multimodal_audits(dataset, seed, variant, model, head, data, x_gpu, edge_gpu,
                                           val_gpu, eval_labels, metrics, normal_pred, multimodal_rows)
                if variant.startswith("s44_expert_"):
                    _add_expert_diagnostics(dataset, seed, variant, model, analysis, expert_rows)
                if variant in {"s44_scalar_global", "s44_scalar_raw", "s44_scalar_rel", "s44_scalar_rel_multi",
                               "s44_lowrank_global", "s44_lowrank_rel", "s44_lowrank_rel_multi",
                               "s44_expert_rel", "s44_expert_rel_multi"}:
                    semantic_rows.extend(_semantic_audit_rows(dataset, seed, variant, analysis, data))

                del model, head, analysis, logits
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del x_gpu, edge_gpu, val_gpu
            if device.type == "cuda":
                torch.cuda.empty_cache()

    table_rows.extend(p0_rows)
    contrasts = _paired_contrasts(metric_map, requested)
    complexity = _complexity_rows(requested)
    interpretation = _interpretation_statuses(contrasts, intervention_rows, multimodal_rows, expert_rows)

    _write_csv(RESULT_ROOT / "s44_table.csv", table_rows)
    _write_csv(RESULT_ROOT / "s44_paired_contrasts.csv", contrasts)
    _write_csv(RESULT_ROOT / "scalar_family.csv", [row for row in table_rows if row["variant"].startswith("s44_scalar_")])
    _write_csv(RESULT_ROOT / "lowrank_family.csv", [row for row in table_rows if row["variant"].startswith("s44_lowrank_")])
    _write_csv(RESULT_ROOT / "expert_family.csv", [row for row in table_rows if row["variant"].startswith("s44_expert_")])
    _write_csv(RESULT_ROOT / "dynamic_interventions.csv", intervention_rows)
    _write_csv(RESULT_ROOT / "multimodal_interventions.csv", multimodal_rows)
    _write_csv(RESULT_ROOT / "expert_diagnostics.csv", expert_rows)
    _write_csv(RESULT_ROOT / "semantic_leakage_audit.csv", semantic_rows)
    _write_csv(RESULT_ROOT / "complexity_table.csv", complexity)

    training_commits = sorted({entry["training_commit"] for entry in run_hashes.values()})
    if len(training_commits) != 1:
        raise ValueError(f"formal checkpoints span multiple training commits: {training_commits}")
    config_hashes = {
        str(path.relative_to(ROOT)): _file_sha256(path)
        for path in [ROOT / "configs/model/relational_transform_pilot.yaml", ROOT / "configs/task/nc.yaml"]
        + [ROOT / f"configs/dataset/{dataset}.yaml" for dataset in requested]
    }
    data_hashes = {dataset: _dataset_hashes(dataset, seed)
                   for dataset in requested for seed in SEEDS}
    summary = {
        "experiment": "S4.4 Relational Transformation Pilot",
        "protocol": {"task": "NC", "protocol_version": "unified_full_graph_nc_v1",
                     "training_mode": "full_graph", "selection": "best_val_accuracy", "data_split_seed": 42,
                     "test_evaluation": False, "lp_evaluation": False,
                     "datasets": list(requested), "seeds": list(SEEDS),
                     "variants": list(FORMAL_VARIANTS), "expected_runs": len(requested) * len(SEEDS) * len(FORMAL_VARIANTS)},
        "completed_runs": len(table_rows) - len(p0_rows),
        "multimodal_intervention_rows": len(multimodal_rows),
        "expected_runs": len(requested) * len(SEEDS) * len(FORMAL_VARIANTS),
        "interpretation_statuses": interpretation,
        "provenance": {
            "source_branch": "s43_h15_h1r_h2ab", "source_commit": SOURCE_COMMIT,
            "training_branch": "s44_relational_transform", "training_commit": training_commits[0],
            "analysis_commit": __import__("subprocess").check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "config_sha256": config_hashes, "dataset_sha256_by_dataset_seed": data_hashes,
            "run_checkpoint_map": run_hashes,
            "d3_semantic_cache_sha256": {
                f"{dataset}/{seed}": _file_sha256(
                    ROOT / "outputs/problem_deep_dive_v1/checkpoint_cache" / f"{dataset}_seed{seed}.pt")
                for dataset in requested for seed in SEEDS
            },
            "semantic_source": "D3 checkpoint_cache per-seed modality-uniform H0 compatibility from source_matched_compatibility(seed*101+7); validation adjacency and train-edge quartiles use the actual shared S4.4 base split seed 42",
            "formulas": {
                "P_rel": "historical relation_basis_pilot.Model._build_operators: remove raw self-loops; symmetrize; add one self-loop; symmetric normalize; remove normalized diagonal; no renormalization",
                "edge_relation_R0": "[q_i,k_j,0,0] -> Linear(128,32), ReLU, Linear(32,32)",
                "edge_relation_R1": "[q_i,k_j,abs(q_i-k_j),q_i*k_j] -> same pair MLP; no semantic scalar input",
                "multimodal_relation": "LN(r_m + W_cross_out_m(Linear_cross_T(r_T)*Linear_cross_V(r_V)))",
                "lowrank": "Delta_i=sum_j P_rel[i,j] U(a_ij * V(H_j)); rank=8; controller preactivation zero initialized",
                "expert": "pi1*U1(V1(H_j)) + pi2*U2(V2(H_j)); K=2; routed logits zero initialized",
            },
        },
        "evidence_boundaries": [
            "Frozen interventions are checkpoint sensitivity analyses, not retrained causal effects.",
            "Router weights do not have causal interpretation.",
            "Expert routing requires routed-vs-uniform benefit, edge-shuffle sensitivity and no severe collapse together.",
            "Multimodal context requires multi-vs-single paired results plus remove/swap interventions.",
            "Low semantic correlation alone does not demonstrate task utility.",
            "Conclusions are restricted to node classification; LP was not evaluated.",
        ],
    }
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "s44_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    report = _write_report(summary, contrasts, intervention_rows, expert_rows, complexity)
    (RESULT_ROOT / "s44_report.md").write_text(report, encoding="utf-8")
    return {"completed_runs": summary["completed_runs"], "expected_runs": summary["expected_runs"],
            "training_commit": training_commits[0], "analysis_commit": summary["provenance"]["analysis_commit"],
            "status": {key: value["status"] for key, value in interpretation.items()},
            "result_directory": str(RESULT_ROOT)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase2-patch", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    if args.phase2_patch:
        print(json.dumps(run_patch(), indent=2))
    elif args.analyze:
        print(json.dumps(run_analysis(tuple(args.datasets)), indent=2))
    else:
        raise SystemExit("select --phase2-patch or --analyze")


if __name__ == "__main__":
    main()
