from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_risa_v05 import (
    MAIN_ABLATION_VARIANTS, NC_DATASETS, OUTPUT_ROOT, PROTOCOL, SEEDS, VARIANTS,
)

RESULT_DIR = ROOT / "results/risa_v05/main_ablation"


def _config(dataset: str, device: str, variant: str):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str((ROOT / "configs").resolve()), version_base=None):
        return compose(config_name="config", overrides=[
            f"dataset={dataset}", "task=nc", "model=risa_v05",
            f"model.variant={variant}", "seed=42", "num_runs=1", f"device={device}",
            "task.evaluate_test=false", "task.training_mode=full_graph",
            f"task.protocol_version={PROTOCOL}",
        ])


def _data_info(data) -> dict[str, int]:
    return {
        "input_dim": int(data.input_dim), "num_nodes": int(data.num_nodes),
        "num_classes": int(data.num_classes), "text_dim": int(data.x_t.size(1)),
        "visual_dim": int(data.x_i.size(1)),
    }


def validation_view(data, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Read only val_idx and its labels; this helper never accesses another split."""
    val_idx_cpu = data.val_idx.detach().cpu().long().reshape(-1)
    val_labels_cpu = data.y.detach().cpu().index_select(0, val_idx_cpu).long()
    return val_idx_cpu.to(device), val_labels_cpu.to(device)


def validation_task_metrics(checkpoint_metrics: dict[str, Any],
                            val_logits: torch.Tensor,
                            val_labels: torch.Tensor) -> dict[str, float]:
    required = ("val_acc", "val_macro_f1")
    missing = [key for key in required if key not in checkpoint_metrics]
    if missing:
        raise ValueError(f"checkpoint lacks validation metrics: {missing}")
    if any(key.startswith("test_") for key in checkpoint_metrics):
        raise ValueError("checkpoint contains forbidden test metrics")
    return {
        # Accuracy and F1 are the task's saved validation values. The F1 label
        # universe is fixed by the task; re-creating it here could require the
        # held-out test labels. Cross-entropy is an independent val-only measure.
        "val_acc": float(checkpoint_metrics["val_acc"]),
        "val_macro_f1": float(checkpoint_metrics["val_macro_f1"]),
        "val_cross_entropy": float(F.cross_entropy(val_logits, val_labels).detach().cpu()),
    }


@torch.no_grad()
def _stream_angle_statistics(model, diagnostic: dict[str, Any],
                             max_rotation_angle: float) -> dict[str, dict[str, float]]:
    modalities = ("text", "visual")
    if not model.use_crst:
        return {m: {"mean_abs_theta": 0.0, "std_abs_theta": 0.0,
                    "angle_saturation_ratio": 0.0} for m in modalities}

    p_rel = diagnostic["P_rel"]
    row, col = p_rel.indices()
    weight = p_rel.values()
    hidden = {m: diagnostic[f"H0_{m}"] for m in modalities}
    context_stats = {m: model._context_sums(hidden[m], row, col, weight) for m in modalities}
    projections = {
        m: getattr(model, f"relation_encoder_{m}").project(hidden[m]) for m in modalities
    }
    accum = {
        m: {
            "sum": torch.zeros((), dtype=torch.float64, device=weight.device),
            "square_sum": torch.zeros((), dtype=torch.float64, device=weight.device),
            "saturated": torch.zeros((), dtype=torch.float64, device=weight.device),
        }
        for m in modalities
    }
    count = 0
    for begin in range(0, row.numel(), model.edge_chunk_size):
        end = min(begin + model.edge_chunk_size, row.numel())
        erow, ecol, eweight = row[begin:end], col[begin:end], weight[begin:end]
        contexts = {}
        for m in modalities:
            sums, masses = context_stats[m]
            context, _ = model.leave_one_edge_out_context(
                hidden[m], erow, ecol, eweight, sums, masses,
            )
            if model.variant == "v05_no_relation_context":
                context = torch.zeros_like(context)
            contexts[m] = context
        relations = {}
        for m in modalities:
            q, k = projections[m]
            encoder = getattr(model, f"relation_encoder_{m}")
            relations[m] = encoder(q[erow], k[ecol], contexts[m])
        shared = model.shared_relation_encoder(torch.cat((
            relations["text"], relations["visual"],
            (relations["text"] - relations["visual"]).abs(),
        ), dim=-1))
        modality_relations = {
            "text": model.shared_norm_text(
                relations["text"] + model.shared_to_text(shared)),
            "visual": model.shared_norm_visual(
                relations["visual"] + model.shared_to_visual(shared)),
        }
        for m in modalities:
            angle = getattr(model, f"rotation_{m}").angles(modality_relations[m]).abs()
            angle64 = angle.double()
            accum[m]["sum"] += angle64.sum()
            accum[m]["square_sum"] += angle64.square().sum()
            accum[m]["saturated"] += (angle > 0.9 * max_rotation_angle).double().sum()
        count += (end - begin) * (model.hidden_dim // model.rotation_group_size)

    result = {}
    for m in modalities:
        if count:
            mean = accum[m]["sum"] / count
            variance = (accum[m]["square_sum"] / count - mean.square()).clamp_min(0.0)
            saturation = accum[m]["saturated"] / count
            result[m] = {
                "mean_abs_theta": float(mean.cpu()),
                "std_abs_theta": float(variance.sqrt().cpu()),
                "angle_saturation_ratio": float(saturation.cpu()),
            }
        else:
            result[m] = {
                "mean_abs_theta": 0.0, "std_abs_theta": 0.0,
                "angle_saturation_ratio": 0.0,
            }
    return result


def _crst_stats(diagnostic: dict[str, Any], modality: str,
                streamed: dict[str, float]) -> dict[str, float]:
    sampled = diagnostic["rotation_angles"][modality].detach().float().abs()
    p95 = float(torch.quantile(sampled.reshape(-1), 0.95).cpu()) if sampled.numel() else 0.0
    return {
        "mean_abs_theta": streamed["mean_abs_theta"],
        "std_abs_theta": streamed["std_abs_theta"],
        "p95_abs_theta": p95,
        "mean_cos_base_rotated": float(diagnostic["cos_base_rotated"][modality]),
        "norm_preservation_max_abs_error": float(
            diagnostic["norm_preservation_max_abs_error"][modality]
        ),
        "valid_context_edge_ratio": float(diagnostic["valid_context_edge_ratio"][modality]),
        "angle_saturation_ratio": streamed["angle_saturation_ratio"],
    }

def _imci_stats(diagnostic: dict[str, Any], modality: str) -> dict[str, Any]:
    weights = diagnostic["attention_weights"][modality]
    if weights is None or not weights.numel():
        return {
            "mean_hop_attention": [None, None, None],
            "mean_hop_attention_entropy": None,
            "per_hop_attention_std_across_nodes_heads": [None, None, None],
        }
    # Model analyze gives [nodes, heads, query=1, hops=3].
    probs = weights.detach().float().squeeze(-2)
    mean_hop = probs.mean(dim=(0, 1))
    hop_std = probs.std(dim=(0, 1), unbiased=False)
    entropy = diagnostic["hop_attention_entropy"][modality]
    return {
        "mean_hop_attention": mean_hop.cpu().tolist(),
        "mean_hop_attention_entropy": float(entropy.detach().cpu()),
        "per_hop_attention_std_across_nodes_heads": hop_std.cpu().tolist(),
    }


def load_dataset(dataset: str):
    from src.data import load_mag_data

    config = _config(dataset, "cpu", "v05_full")
    return load_mag_data(config, "nc", 42)


def analyze_checkpoint(checkpoint: Path, dataset: str, seed: int, variant: str,
                       device: torch.device, data) -> dict[str, Any]:
    from torch import nn
    from src.models import build_model

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("task") != "nc" or payload.get("protocol_version") != PROTOCOL:
        raise ValueError(f"wrong task/protocol in {checkpoint}")
    if int(payload.get("seed", -1)) != seed:
        raise ValueError(f"checkpoint seed mismatch in {checkpoint}")
    if payload.get("selection") != "best_val_accuracy":
        raise ValueError(f"checkpoint was not selected by validation accuracy: {checkpoint}")
    checkpoint_metrics = payload.get("metrics", {})
    if any(key.startswith("test_") for key in checkpoint_metrics):
        raise ValueError(f"test metrics found in {checkpoint}")

    config = _config(dataset, str(device), variant)
    model = build_model(config, payload["data_info"]).to(device)
    model.load_state_dict(payload["model_state"], strict=True)
    classifier = nn.Linear(model.out_dim, int(data.num_classes)).to(device)
    classifier.load_state_dict(payload["head_state"], strict=True)
    model.eval()
    classifier.eval()

    x = data.x.to(device)
    edge = data.edge_index.to(device)
    val_idx, val_labels = validation_view(data, device)
    with torch.no_grad():
        diagnostic = model.analyze(
            x, edge, sample_seed=42000 + seed + sum(ord(c) for c in dataset),
            max_diagnostic_edges=512, collect_edge_state=True,
            collect_attention=True,
        )
        val_logits = classifier(diagnostic["fused_z"].index_select(0, val_idx))
        task = validation_task_metrics(checkpoint_metrics, val_logits, val_labels)

    rotation_limit = float(config.model.max_rotation_angle)
    streamed_angles = _stream_angle_statistics(model, diagnostic, rotation_limit)
    crst = {m: _crst_stats(diagnostic, m, streamed_angles[m])
            for m in ("text", "visual")}
    imci = {m: _imci_stats(diagnostic, m) for m in ("text", "visual")}
    flat: dict[str, Any] = {
        "dataset": dataset, "seed": int(seed), "variant": variant,
        "best_epoch": int(payload["epoch"]),
        **task,
        "checkpoint": str(checkpoint),
        "test_evaluated": False,
    }
    for modality in ("text", "visual"):
        for name, value in crst[modality].items():
            flat[f"crst_{modality}_{name}"] = value
        imci_values = imci[modality]
        for hop, value in enumerate(imci_values["mean_hop_attention"], 1):
            flat[f"imci_{modality}_mean_hop_attention_{hop}"] = value
        flat[f"imci_{modality}_mean_hop_attention_entropy"] = imci_values[
            "mean_hop_attention_entropy"
        ]
        for hop, value in enumerate(
            imci_values["per_hop_attention_std_across_nodes_heads"], 1
        ):
            flat[f"imci_{modality}_per_hop_attention_std_across_nodes_heads_{hop}"] = value
    flat["crst"] = crst
    flat["imci"] = imci
    flat["diagnostic_edge_sample_count"] = int(diagnostic["edge_indices"].numel())
    flat["physical_p_rel_unchanged"] = (
        torch.equal(diagnostic["P_rel"].indices(),
                    model.backbone._get_operators(edge, int(x.size(0)), x.dtype)[2].indices())
    )
    return flat


def _mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["dataset"], record["variant"])].append(record)
    aggregate = {}
    for (dataset, variant), rows in sorted(grouped.items()):
        numeric = {}
        excluded = {"seed", "best_epoch", "test_evaluated",
                    "dataset", "variant", "checkpoint", "crst", "imci"}
        fields = [key for key, value in rows[0].items()
                  if key not in excluded and isinstance(value, (int, float))
                  and not isinstance(value, bool)]
        for field in fields:
            numeric[field] = _mean_std([float(row[field]) for row in rows
                                        if row.get(field) is not None])
        aggregate.setdefault(dataset, {})[variant] = {
            "num_seeds": len(rows), "metrics": numeric,
        }
    return aggregate


def paired_contrasts(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive matched dataset/seed differences for the paper's 2x2 ablation."""
    definitions = (
        ("crst_only_minus_plain", "v05_crst_only", "v05_plain",
         "CRST contribution without context absorption"),
        ("absorb_only_minus_plain", "v05_absorb_only", "v05_plain",
         "Context absorption contribution with ordinary propagation"),
        ("full_minus_absorb_only", "v05_full", "v05_absorb_only",
         "CRST contribution when context absorption is enabled"),
        ("full_minus_crst_only", "v05_full", "v05_crst_only",
         "Context absorption contribution when CRST is enabled"),
        ("full_minus_plain", "v05_full", "v05_plain",
         "Overall reciprocal structure-attribute interaction effect"),
    )
    indexed = {(row["dataset"], int(row["seed"]), row["variant"]): row
               for row in records}
    output: dict[str, Any] = {}
    for name, left, right, interpretation in definitions:
        by_dataset: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for (dataset, seed, variant), left_row in indexed.items():
            if variant != left:
                continue
            right_row = indexed.get((dataset, seed, right))
            if right_row is None:
                continue
            by_dataset[dataset].append((
                100.0 * (float(left_row["val_acc"]) - float(right_row["val_acc"])),
                float(left_row["val_macro_f1"]) - float(right_row["val_macro_f1"]),
            ))
        pairs = [pair for values in by_dataset.values() for pair in values]
        acc_deltas = [pair[0] for pair in pairs]
        f1_deltas = [pair[1] for pair in pairs]
        per_dataset = {}
        for dataset in sorted(by_dataset):
            values = by_dataset[dataset]
            per_dataset[dataset] = {
                "matched_seed_pairs": len(values),
                "mean_acc_delta_percentage_points": statistics.fmean(v[0] for v in values),
                "mean_macro_f1_delta": statistics.fmean(v[1] for v in values),
                "positive_acc_seed_pairs": sum(v[0] > 0 for v in values),
                "positive_macro_f1_seed_pairs": sum(v[1] > 0 for v in values),
            }
        positive_acc_datasets = [dataset for dataset, values in sorted(by_dataset.items())
                                 if statistics.fmean(v[0] for v in values) > 0]
        positive_f1_datasets = [dataset for dataset, values in sorted(by_dataset.items())
                                if statistics.fmean(v[1] for v in values) > 0]
        output[name] = {
            "left_variant": left, "right_variant": right,
            "interpretation": interpretation,
            "matched_seed_pairs": len(pairs),
            "mean_paired_acc_delta_percentage_points": (
                statistics.fmean(acc_deltas) if acc_deltas else None
            ),
            "mean_paired_macro_f1_delta": statistics.fmean(f1_deltas) if f1_deltas else None,
            "positive_datasets": {
                "acc": positive_acc_datasets, "macro_f1": positive_f1_datasets,
            },
            "positive_seed_pairs": {
                "acc": {"count": sum(value > 0 for value in acc_deltas), "total": len(acc_deltas)},
                "macro_f1": {"count": sum(value > 0 for value in f1_deltas), "total": len(f1_deltas)},
            },
            "per_dataset": per_dataset,
            "inference": "descriptive paired contrasts; no significance claim",
        }
    return output


def _write_reports(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_records(records)
    present = {row["variant"] for row in records}
    ordered_variants = [name for name in MAIN_ABLATION_VARIANTS if name in present]
    ordered_variants.extend(name for name in VARIANTS if name in present and name not in ordered_variants)
    report = {
        "protocol_version": PROTOCOL,
        "datasets": list(dict.fromkeys(row["dataset"] for row in records)),
        "seeds": sorted({row["seed"] for row in records}),
        "variants": ordered_variants,
        "test_evaluated": False, "lp_evaluated": False,
        "records": records, "aggregate": aggregate,
        "paired_contrasts": paired_contrasts(records),
    }
    (output_dir / "risa_v05_main_ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8",
    )
    columns = [
        "dataset", "seed", "variant", "best_epoch", "val_acc",
        "val_macro_f1", "val_cross_entropy",
    ]
    for modality in ("text", "visual"):
        columns.extend([
            f"crst_{modality}_{name}" for name in (
                "mean_abs_theta", "std_abs_theta", "p95_abs_theta",
                "mean_cos_base_rotated", "norm_preservation_max_abs_error",
                "valid_context_edge_ratio", "angle_saturation_ratio",
            )
        ])
        columns.extend(
            f"imci_{modality}_mean_hop_attention_{hop}" for hop in (1, 2, 3)
        )
        columns.append(f"imci_{modality}_mean_hop_attention_entropy")
        columns.extend(
            f"imci_{modality}_per_hop_attention_std_across_nodes_heads_{hop}"
            for hop in (1, 2, 3)
        )
    with (output_dir / "risa_v05_main_ablation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# RISA v0.5 main ablation validation analysis", "",
        f"- Protocol: {PROTOCOL}",
        "- Test and LP evaluation: disabled.",
        "- Val Acc / Macro-F1: read from validation-selected checkpoints.",
        "- Val cross-entropy: post-hoc calculation on val_idx only; it does not select checkpoints.",
        "- CRST p95/std/saturation: deterministic random edge sample; means and norm/context summaries stream over edges.",
        "- IMCI attention summaries aggregate over nodes and heads.", "",
    ]
    for dataset in report["datasets"]:
        lines.extend([f"## {dataset}", ""])
        for variant in report["variants"]:
            rows = grouped_rows(records, dataset, variant)
            if not rows:
                continue
            lines.extend([
                f"### {variant}", "",
                "| Metric | Mean ± population std |",
                "|---|---:|",
            ])
            summary_metrics = ["val_acc", "val_macro_f1", "val_cross_entropy"]
            crst_fields = (
                "mean_abs_theta", "std_abs_theta", "p95_abs_theta",
                "mean_cos_base_rotated", "norm_preservation_max_abs_error",
                "valid_context_edge_ratio", "angle_saturation_ratio",
            )
            for modality in ("text", "visual"):
                summary_metrics.extend(f"crst_{modality}_{field}" for field in crst_fields)
                summary_metrics.append(f"imci_{modality}_mean_hop_attention_entropy")
                summary_metrics.extend(
                    f"imci_{modality}_mean_hop_attention_{hop}" for hop in (1, 2, 3)
                )
                summary_metrics.extend(
                    f"imci_{modality}_per_hop_attention_std_across_nodes_heads_{hop}"
                    for hop in (1, 2, 3)
                )
            for key in summary_metrics:
                values = [float(row[key]) for row in rows if row.get(key) is not None]
                if values:
                    stats = _mean_std(values)
                    lines.append(f"| {key} | {stats['mean']:.6g} ± {stats['std']:.6g} |")
            lines.append("")
    lines.extend(["## Paired contrasts", "", "Descriptive matched dataset/seed differences; no significance claims.", ""])
    lines.extend([
        "| Contrast | Mean Acc delta (pp) | Mean Macro-F1 delta | Positive datasets (Acc / F1) | Positive seed pairs (Acc / F1) |",
        "|---|---:|---:|---|---:|",
    ])
    for name, contrast in report["paired_contrasts"].items():
        acc = contrast["mean_paired_acc_delta_percentage_points"]
        f1 = contrast["mean_paired_macro_f1_delta"]
        acc_text = "N/A" if acc is None else f"{acc:.4f}"
        f1_text = "N/A" if f1 is None else f"{f1:.6f}"
        positive_datasets = contrast["positive_datasets"]
        positive_pairs = contrast["positive_seed_pairs"]
        lines.append(
            f"| {name} | {acc_text} | {f1_text} | "
            f"{', '.join(positive_datasets['acc']) or '—'} / "
            f"{', '.join(positive_datasets['macro_f1']) or '—'} | "
            f"{positive_pairs['acc']['count']}/{positive_pairs['acc']['total']} / "
            f"{positive_pairs['macro_f1']['count']}/{positive_pairs['macro_f1']['total']} |"
        )
    lines.append("")
    (output_dir / "risa_v05_main_ablation.md").write_text("\n".join(lines), encoding="utf-8")


def grouped_rows(records: list[dict[str, Any]], dataset: str,
                 variant: str) -> list[dict[str, Any]]:
    return sorted(
        (row for row in records if row["dataset"] == dataset and row["variant"] == variant),
        key=lambda row: row["seed"],
    )


def checkpoint_path(checkpoint_root: Path, dataset: str, variant: str, seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    run_number = SEEDS.index(seed) + 1
    return checkpoint_root / dataset / variant / f"best_run{run_number}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc analysis of v0.5 formal NC checkpoints")
    parser.add_argument("--datasets", nargs="+", choices=NC_DATASETS, default=list(NC_DATASETS))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS,
                        default=list(MAIN_ABLATION_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--checkpoint-root", type=Path, default=OUTPUT_ROOT / "formal")
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if not args.device.startswith("cuda") or torch.cuda.is_available()
                          else "cpu")
    records = []
    for dataset in args.datasets:
        data = load_dataset(dataset)
        for variant in args.variants:
            for seed in args.seeds:
                checkpoint = checkpoint_path(args.checkpoint_root, dataset, variant, seed)
                record = analyze_checkpoint(checkpoint, dataset, seed, variant, device, data)
                records.append(record)
                print(json.dumps({
                    "dataset": dataset, "seed": seed, "variant": variant,
                    "status": "analyzed", "checkpoint": str(checkpoint),
                    "val_acc": record["val_acc"],
                    "val_macro_f1": record["val_macro_f1"],
                    "val_cross_entropy": record["val_cross_entropy"],
                }), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    _write_reports(records, args.output_dir.resolve())
    print(json.dumps({
        "analyzed_checkpoints": len(records),
        "report_json": str((args.output_dir / "risa_v05_main_ablation.json").resolve()),
        "report_markdown": str((args.output_dir / "risa_v05_main_ablation.md").resolve()),
        "report_csv": str((args.output_dir / "risa_v05_main_ablation.csv").resolve()),
        "test_evaluated": False, "lp_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()

