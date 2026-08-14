#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full eval-only script for the current best LiTS Stage2 model.

默认评估当前最强配置：
  model        = /root/autodl-tmp/lits_stage2_v33_conservative/best_s2.pth
  train module = train_data_heavy_stage2_v3_1_fix.py
  TTA          = single(4): original + x/y/z flip
  overlap      = 0.5
  tumor bias   = 1.05
  min_vol      = 200

输出：
  1) 控制台 summary
  2) per-case CSV
  3) summary JSON

推荐运行：
  cd /root/autodl-tmp
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
  python eval_stage2_best_full.py

可选环境变量：
  STAGE2_TRAIN_SCRIPT=train_data_heavy_stage2_v3_1_fix
  STAGE2_BEST=/root/autodl-tmp/lits_stage2_v33_conservative/best_s2.pth
  LITS_TTA_MODE=single        # none|single|all
  LITS_OVERLAP=0.5
  LITS_TUMOR_BIAS=1.05
  LITS_MIN_TUMOR_VOL=200
  LITS_NUM_WORKERS=4
"""

import os
# 放在 numpy/torch/monai 前，避免 libgomp 读到非法值。
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import sys
import csv
import json
import time
import importlib
from pathlib import Path
from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from monai.data import PersistentDataset, DataLoader, decollate_batch
from monai.transforms import Compose, EnsureTyped, AsDiscreted
from monai.inferers import sliding_window_inference

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

MODULE_NAME = os.environ.get("STAGE2_TRAIN_SCRIPT", "train_data_heavy_stage2_v3_1_fix").replace(".py", "")
try:
    base = importlib.import_module(MODULE_NAME)
except ModuleNotFoundError as e:
    raise SystemExit(
        f"找不到 {MODULE_NAME}.py。请把本脚本放到训练脚本同目录，"
        f"或设置 STAGE2_TRAIN_SCRIPT=你的训练脚本名。原始错误: {e}"
    )

DEVICE = base.DEVICE
AMP_ENABLED = base.AMP_ENABLED
PATCH_SIZE = base.PATCH_SIZE
SW_BATCH_SIZE = base.SW_BATCH_SIZE

DEFAULT_BEST = "/root/autodl-tmp/lits_stage2_v33_conservative/best_s2.pth"
if not os.path.exists(DEFAULT_BEST):
    DEFAULT_BEST = "/root/autodl-tmp/lits_stage2_optimized/best_s2.pth"

BEST_MODEL = os.path.expanduser(os.environ.get("STAGE2_BEST", DEFAULT_BEST))
TTA_MODE = os.environ.get("LITS_TTA_MODE", "single").lower().strip()
OVERLAP = float(os.environ.get("LITS_OVERLAP", "0.5"))
TUMOR_BIAS = float(os.environ.get("LITS_TUMOR_BIAS", "1.05"))
MIN_TUMOR_VOL = int(os.environ.get("LITS_MIN_TUMOR_VOL", "200"))
NUM_WORKERS = int(os.environ.get("LITS_NUM_WORKERS", "4"))


def get_tta_flips(mode: str) -> List[Tuple[int, ...]]:
    # Tensor shape: B,C,H,W,D；spatial dims are 2,3,4.
    axes = [2, 3, 4]
    if mode in ("none", "0", "false"):
        return [()]
    if mode in ("single", "4"):
        return [(), (2,), (3,), (4,)]
    if mode in ("all", "8"):
        flips: List[Tuple[int, ...]] = [()]
        for r in (1, 2, 3):
            flips.extend(tuple(c) for c in combinations(axes, r))
        return flips
    raise SystemExit("LITS_TTA_MODE 只能是 none / single / all")


TTA_FLIPS = get_tta_flips(TTA_MODE)


def safe_case_name(item: Dict, idx: int) -> str:
    """尽量从 val_files 中提取病例名；提取不到就用 idx。"""
    for key in ("image", "label", "sdf"):
        val = item.get(key)
        if isinstance(val, (str, Path)):
            return Path(val).name
    return f"case_{idx:03d}"


def make_val_loader():
    data = base.build_file_list()
    val_n = min(20, max(1, int(len(data) * 0.15)))
    val_files = data[-val_n:]
    _, _, val_pre_trans = base.get_transforms()
    val_ds = PersistentDataset(
        data=val_files,
        transform=val_pre_trans,
        cache_dir=os.path.join(base.CACHE_DIR, "val"),
    )
    val_workers = max(1, NUM_WORKERS // 2)
    kwargs = dict(
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=AMP_ENABLED,
        persistent_workers=val_workers > 0,
    )
    if val_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(val_ds, **kwargs), val_files


def binary_stats(pred: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
    """Return dice/inter/pred/label/precision/recall for binary tensors."""
    pred = pred.float()
    label = label.float()
    pred_sum = float(pred.sum().item())
    label_sum = float(label.sum().item())
    inter = float((pred * label).sum().item())
    dice = 1.0 if pred_sum + label_sum <= 0 else 2.0 * inter / (pred_sum + label_sum + 1e-8)
    precision = 1.0 if pred_sum <= 0 and label_sum <= 0 else inter / max(pred_sum, 1e-8)
    recall = 1.0 if label_sum <= 0 and pred_sum <= 0 else inter / max(label_sum, 1e-8)
    return {
        "dice": float(dice),
        "inter_vox": inter,
        "pred_vox": pred_sum,
        "label_vox": label_sum,
        "precision": float(precision),
        "recall": float(recall),
    }


def infer_logits_tta(model: torch.nn.Module, v_inputs: torch.Tensor) -> torch.Tensor:
    """Average logits over configured flip TTA."""
    acc = None
    n = 0

    def predictor(x):
        return base.extract_main_output(model(x))

    for dims in TTA_FLIPS:
        x = torch.flip(v_inputs, dims=list(dims)) if dims else v_inputs
        with torch.amp.autocast(device_type=DEVICE.type, enabled=AMP_ENABLED):
            logits = sliding_window_inference(
                inputs=x,
                roi_size=PATCH_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=predictor,
                overlap=OVERLAP,
                mode="gaussian",
            )
        if dims:
            logits = torch.flip(logits, dims=list(dims))
        logits = logits.float()
        acc = logits if acc is None else acc + logits
        n += 1
    return acc / max(1, n)


def aggregate_macro(rows: List[Dict[str, float]], key: str) -> float:
    vals = [float(r[key]) for r in rows]
    return float(np.mean(vals)) if vals else 0.0


def pooled_dice(sum_inter: float, sum_pred: float, sum_label: float) -> float:
    return float(2.0 * sum_inter / (sum_pred + sum_label + 1e-8)) if (sum_pred + sum_label) > 0 else 1.0


def main():
    if not os.path.exists(BEST_MODEL):
        raise SystemExit(f"未找到 best 模型: {BEST_MODEL}")

    print("=" * 88)
    print("LiTS Stage2 best-model full evaluation")
    print("=" * 88)
    print(f"Device: {DEVICE}")
    print(f"Train module: {MODULE_NAME}")
    print(f"Best model: {BEST_MODEL}")
    print(f"Patch: {PATCH_SIZE}, SW_BATCH={SW_BATCH_SIZE}, AMP={AMP_ENABLED}")
    print(f"TTA mode: {TTA_MODE}, flips={TTA_FLIPS}, n={len(TTA_FLIPS)}")
    print(f"Overlap: {OVERLAP}")
    print(f"Tumor logit bias: {TUMOR_BIAS}")
    print(f"Min tumor connected component volume: {MIN_TUMOR_VOL}")
    print("=" * 88)

    val_loader, val_files = make_val_loader()
    first = next(iter(val_loader))
    in_channels = int(first["image"].shape[1] + first["sdf"].shape[1])
    del first

    model = base.get_model(in_channels=in_channels)
    state = base.strip_state_dict(base.safe_torch_load(BEST_MODEL, map_location=DEVICE))
    incompat = model.load_state_dict(state, strict=False)
    print(f"Loaded best. missing={len(incompat.missing_keys)}, unexpected={len(incompat.unexpected_keys)}")
    if incompat.missing_keys or incompat.unexpected_keys:
        print("WARNING: checkpoint keys mismatch detected.")
        print("missing_keys:", incompat.missing_keys[:20])
        print("unexpected_keys:", incompat.unexpected_keys[:20])
    model.eval()

    post_pred = Compose([
        EnsureTyped(keys=["pred"]),
        AsDiscreted(keys=["pred"], argmax=True),
        base.CCAPostProcessingd(keys=["pred"], min_tumor_vol=MIN_TUMOR_VOL),
        AsDiscreted(keys=["pred"], to_onehot=3),
    ])

    rows: List[Dict[str, float]] = []
    tumor_inter = tumor_pred_sum = tumor_label_sum = 0.0
    liver_inter = liver_pred_sum = liver_label_sum = 0.0
    whole_inter = whole_pred_sum = whole_label_sum = 0.0

    t0 = time.time()
    with torch.no_grad():
        for idx, val_data in enumerate(tqdm(val_loader, desc="Evaluating cases")):
            v_images = val_data["image"].to(DEVICE, non_blocking=True)
            v_sdfs = val_data["sdf"].to(DEVICE, non_blocking=True)
            v_labels = val_data["label"].to(DEVICE, non_blocking=True)
            v_inputs = torch.cat([v_images, v_sdfs], dim=1)

            logits = infer_logits_tta(model, v_inputs)
            logits[:, 2:3] += TUMOR_BIAS

            data_list = decollate_batch({"pred": logits})
            pp = post_pred(data_list[0])["pred"].to(DEVICE)  # expected: C,H,W,D one-hot

            tumor_pred = pp[2:3, ...].unsqueeze(0)
            liver_class_pred = pp[1:2, ...].unsqueeze(0)
            whole_liver_pred = (pp[1:3, ...].sum(dim=0, keepdim=True) > 0).float().unsqueeze(0)

            tumor_label = (v_labels == 2).float()
            liver_class_label = (v_labels == 1).float()
            whole_liver_label = (v_labels > 0).float()

            tumor = binary_stats(tumor_pred, tumor_label)
            liver = binary_stats(liver_class_pred, liver_class_label)
            whole = binary_stats(whole_liver_pred, whole_liver_label)

            tumor_inter += tumor["inter_vox"]
            tumor_pred_sum += tumor["pred_vox"]
            tumor_label_sum += tumor["label_vox"]
            liver_inter += liver["inter_vox"]
            liver_pred_sum += liver["pred_vox"]
            liver_label_sum += liver["label_vox"]
            whole_inter += whole["inter_vox"]
            whole_pred_sum += whole["pred_vox"]
            whole_label_sum += whole["label_vox"]

            case_name = safe_case_name(val_files[idx], idx)
            rows.append({
                "case_index": idx,
                "case_name": case_name,
                "tumor_dice": tumor["dice"],
                "tumor_precision": tumor["precision"],
                "tumor_recall": tumor["recall"],
                "tumor_inter_vox": int(round(tumor["inter_vox"])),
                "pred_tumor_vox": int(round(tumor["pred_vox"])),
                "label_tumor_vox": int(round(tumor["label_vox"])),
                "pred_label_tumor_ratio": tumor["pred_vox"] / max(1.0, tumor["label_vox"]),
                "liver_class_dice": liver["dice"],
                "whole_liver_dice": whole["dice"],
                "pred_whole_liver_vox": int(round(whole["pred_vox"])),
                "label_whole_liver_vox": int(round(whole["label_vox"])),
            })

    elapsed = time.time() - t0

    summary = {
        "best_model": BEST_MODEL,
        "train_module": MODULE_NAME,
        "patch_size": list(PATCH_SIZE),
        "sw_batch_size": SW_BATCH_SIZE,
        "tta_mode": TTA_MODE,
        "tta_n": len(TTA_FLIPS),
        "tta_flips": [list(x) for x in TTA_FLIPS],
        "overlap": OVERLAP,
        "tumor_logit_bias": TUMOR_BIAS,
        "min_tumor_vol": MIN_TUMOR_VOL,
        "num_cases": len(rows),
        "elapsed_sec": elapsed,
        "macro_mean_tumor_dice": aggregate_macro(rows, "tumor_dice"),
        "macro_mean_tumor_precision": aggregate_macro(rows, "tumor_precision"),
        "macro_mean_tumor_recall": aggregate_macro(rows, "tumor_recall"),
        "macro_mean_liver_class_dice": aggregate_macro(rows, "liver_class_dice"),
        "macro_mean_whole_liver_dice": aggregate_macro(rows, "whole_liver_dice"),
        "pooled_tumor_dice": pooled_dice(tumor_inter, tumor_pred_sum, tumor_label_sum),
        "pooled_liver_class_dice": pooled_dice(liver_inter, liver_pred_sum, liver_label_sum),
        "pooled_whole_liver_dice": pooled_dice(whole_inter, whole_pred_sum, whole_label_sum),
        "total_pred_tumor_vox": int(round(tumor_pred_sum)),
        "total_label_tumor_vox": int(round(tumor_label_sum)),
        "total_pred_label_tumor_ratio": tumor_pred_sum / max(1.0, tumor_label_sum),
    }

    tag = f"best_tta_{TTA_MODE}_ov_{str(OVERLAP).replace('.', 'p')}_bias_{str(TUMOR_BIAS).replace('.', 'p')}_minvol_{MIN_TUMOR_VOL}"
    out_csv = SCRIPT_DIR / f"stage2_eval_{tag}_per_case.csv"
    out_json = SCRIPT_DIR / f"stage2_eval_{tag}_summary.json"

    if rows:
        fieldnames = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"Cases: {summary['num_cases']}")
    print(f"Macro Tumor Dice:      {summary['macro_mean_tumor_dice']:.4f}")
    print(f"Macro Tumor Precision: {summary['macro_mean_tumor_precision']:.4f}")
    print(f"Macro Tumor Recall:    {summary['macro_mean_tumor_recall']:.4f}")
    print(f"Pooled Tumor Dice:     {summary['pooled_tumor_dice']:.4f}")
    print(f"Tumor voxels: pred={summary['total_pred_tumor_vox']} | label={summary['total_label_tumor_vox']} | ratio={summary['total_pred_label_tumor_ratio']:.3f}")
    print(f"Macro whole-liver Dice: {summary['macro_mean_whole_liver_dice']:.4f}")
    print(f"Macro class-1 liver Dice: {summary['macro_mean_liver_class_dice']:.4f}")
    print(f"Elapsed: {elapsed/60:.1f} min")

    worst = sorted(rows, key=lambda r: r["tumor_dice"])[:5]
    best = sorted(rows, key=lambda r: r["tumor_dice"], reverse=True)[:5]
    print("\nWorst 5 tumor cases:")
    for r in worst:
        print(f"  #{r['case_index']:02d} {r['case_name']} | dice={r['tumor_dice']:.4f} | pred={r['pred_tumor_vox']} | label={r['label_tumor_vox']} | recall={r['tumor_recall']:.4f}")
    print("\nBest 5 tumor cases:")
    for r in best:
        print(f"  #{r['case_index']:02d} {r['case_name']} | dice={r['tumor_dice']:.4f} | pred={r['pred_tumor_vox']} | label={r['label_tumor_vox']} | recall={r['tumor_recall']:.4f}")

    print(f"\nSaved per-case CSV: {out_csv}")
    print(f"Saved summary JSON: {out_json}")


if __name__ == "__main__":
    main()
