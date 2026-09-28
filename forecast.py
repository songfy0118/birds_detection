"""Measured train/evaluate workflow using the existing Transformer forecaster.

This is a reference run, not reconstruction of the paper's original split,
coordinate calibration, training procedure or reported metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from models.transformer import TransformerForecaster
from models.heads.student_t import nll_student_t_mixture_2d
from utils.data import TrajJsonlDataset, traj_collate


def grouped_split(dataset, seed=42):
    """Keep source videos together, or tracks for exports with one video."""
    videos = {item["video"] for item in dataset.items}
    by_video = len(videos) >= 3 and all(videos)
    groups = {}
    for index, item in enumerate(dataset.items):
        if not by_video and item["track_id"] < 0:
            raise ValueError("A video or valid track ID is required for a grouped split.")
        key = item["video"] if by_video else (item["video"], item["track_id"])
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Need at least three independent groups for train/val/test.")
    random.Random(seed).shuffle(keys)
    n_val = n_test = max(1, int(len(keys) * 0.15))
    partitions = (keys[n_val + n_test:], keys[:n_val], keys[n_val:n_val + n_test])
    splits = [[i for key in group for i in groups[key]] for group in partitions]
    return dict(zip(("train", "validation", "test"), splits)), ("video" if by_video else "track")


def evaluate(model, loader, device):
    model.eval()
    totals = dict(nll=0.0, ade=0.0, fde=0.0, minade=0.0)
    count = 0
    with torch.no_grad():
        for batch in loader:
            past = batch["past_xy"].to(device)
            target = batch["future_xy"].to(device)
            out = model(past)  # Dataset normalizes once; model normalization is off.
            nll = nll_student_t_mixture_2d(
                out["logits"], out["mu"], out["log_scale"], out["dof"], target
            )
            weights = out["logits"].softmax(-1).unsqueeze(-1)
            mean = (weights * out["mu"]).sum(2)
            error = torch.linalg.vector_norm(mean - target, dim=-1)
            component_error = torch.linalg.vector_norm(out["mu"] - target.unsqueeze(2), dim=-1)
            values = {"nll": nll, "ade": error.mean(), "fde": error[:, -1].mean(),
                      "minade": component_error.mean(1).min(-1).values.mean()}
            size = len(past)
            if not all(torch.isfinite(value).item() for value in values.values()):
                raise ValueError("Non-finite evaluation metric; refusing to save a report.")
            for key, value in values.items():
                totals[key] += float(value) * size
            count += size
    if count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {**{key: value / count for key, value in totals.items()}, "samples": count}


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def validate_splits(dataset, splits, split_unit):
    """Reject corrupted partitions and cross-partition source leakage."""
    if set(splits) != {"train", "validation", "test"} or split_unit not in {"video", "track"}:
        raise ValueError("Invalid checkpoint split metadata.")
    used_rows, used_groups = set(), set()
    for indices in splits.values():
        if not indices or any(type(i) is not int or not 0 <= i < len(dataset) for i in indices):
            raise ValueError("Split indices must be nonempty and within the dataset.")
        rows = set(indices)
        if len(rows) != len(indices) or used_rows & rows:
            raise ValueError("Duplicate or overlapping split indices.")
        groups = {
            dataset.items[i]["video"] if split_unit == "video"
            else (dataset.items[i]["video"], dataset.items[i]["track_id"])
            for i in indices
        }
        if used_groups & groups:
            raise ValueError("Source groups overlap across partitions.")
        used_rows.update(rows)
        used_groups.update(groups)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bird_trajs.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("output/reference_run"))
    parser.add_argument("--epochs", type=positive_int, default=5)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=positive_int, default=2)
    parser.add_argument("--smoke", action="store_true", help="Small software check, not a benchmark.")
    parser.add_argument("--checkpoint", type=Path, help="Evaluate a checkpoint produced by this workflow.")
    args = parser.parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; use --device cpu.")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = TrajJsonlDataset(args.data)
    data_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.checkpoint:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("format") != "bird-reference-v1":
            raise ValueError("Unsupported checkpoint. Legacy weights require their original configuration and preprocessing.")
        if saved["data_sha256"] != data_hash:
            raise ValueError("Dataset differs from checkpoint provenance.")
        config, splits, split_unit = saved["model_config"], saved["splits"], saved["split_unit"]
        smoke = saved["smoke"]
    else:
        splits, split_unit = grouped_split(dataset, args.seed)
        smoke = args.smoke
        if smoke:
            splits = {key: value[:16] for key, value in splits.items()}
        config = dict(
            past_len=dataset.P, future_len=dataset.F, d_model=32 if smoke else 96,
            nhead=4, enc_layers=1 if smoke else 2, dec_layers=1 if smoke else 2,
            dim_ff=64 if smoke else 1024, dropout=0.3, head="mdn_student_t",
            mdn_K=5, student_t_dof_init=10.0, learnable_dof=True, normalize_xy=False,
        )
    if (config["past_len"], config["future_len"]) != (dataset.P, dataset.F):
        raise ValueError("Checkpoint horizons do not match the dataset.")
    if config.get("normalize_xy") is not False:
        raise ValueError("Reference checkpoints must disable model-side normalization.")
    validate_splits(dataset, splits, split_unit)
    model = TransformerForecaster(**config).to(args.device)
    loaders = {
        name: DataLoader(Subset(dataset, indices), batch_size=args.batch_size,
                         shuffle=name == "train", num_workers=0, collate_fn=traj_collate)
        for name, indices in splits.items()
    }
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.checkpoint or args.out / "best.pt"
    history = []
    if not args.checkpoint:
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
        best = float("inf")
        for epoch in range(1, (1 if smoke else args.epochs) + 1):
            model.train()
            for batch in loaders["train"]:
                optimizer.zero_grad(set_to_none=True)
                out = model(batch["past_xy"].to(args.device))
                loss = nll_student_t_mixture_2d(
                    out["logits"], out["mu"], out["log_scale"], out["dof"],
                    batch["future_xy"].to(args.device),
                )
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
            metrics = evaluate(model, loaders["validation"], args.device)
            history.append({"epoch": epoch, **metrics})
            print(json.dumps(history[-1]))
            if metrics["nll"] < best:
                best = metrics["nll"]
                torch.save(dict(
                    format="bird-reference-v1", model=model.state_dict(), model_config=config,
                    data_sha256=data_hash, splits=splits, split_unit=split_unit,
                    seed=args.seed, smoke=smoke, epoch=epoch,
                ), checkpoint_path)
    saved = torch.load(checkpoint_path, map_location=args.device, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    result = dict(
        scope="software smoke check" if smoke else "reference experiment; not paper reproduction",
        data_sha256=data_hash, split_unit=split_unit, seed=saved["seed"], model_config=config,
        parameters=sum(p.numel() for p in model.parameters()),
        coordinate_units="image coordinates normalized to [-1, 1]; not metres",
        metric_definitions={"nll": "mean full-mixture NLL per 2D step",
                            "ade": "mixture-weighted mean trajectory ADE",
                            "fde": "mixture-weighted mean trajectory FDE",
                            "minade": "best component trajectory ADE; oracle uses test targets"},
        split_sizes={key: len(value) for key, value in splits.items()},
        selected_epoch=saved["epoch"], history=history,
        test=evaluate(model, loaders["test"], args.device),
    )
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["test"]))
    print(f"Measured results: {args.out / 'metrics.json'}")


if __name__ == "__main__":
    main()
