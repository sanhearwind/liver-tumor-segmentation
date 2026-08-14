# Two-Stage LiTS Liver/Tumor Segmentation

This repository contains a two-stage 3D medical-image segmentation pipeline
for the LiTS (Liver Tumor Segmentation) task:

1. Stage 1: liver-region localization / segmentation.
2. Stage 2: tumor segmentation within the liver region.
3. Evaluation utilities for reproducible validation runs.

## Repository layout

```text
src/       Training, preprocessing, prediction, and evaluation scripts
configs/   Configuration examples (add local-only configs here as needed)
docs/      Method notes and experiment documentation
outputs/   Kept empty in the public repository; generated results are ignored
```

## Data and model checkpoints

The LiTS dataset, model checkpoints (`*.pth`, `*.pt`, `*.ckpt`), generated
NIfTI volumes, and experiment logs are intentionally **not** included. Obtain
the dataset through its official license and configure local paths through
environment variables or local, untracked files.

The scripts currently use an `~/autodl-tmp/` layout by default. Adjust those
paths before running on another machine.

## Environment

The core dependencies are Python, PyTorch, MONAI, NumPy, SciPy, NiBabel, and
scikit-image. Pin exact versions after confirming the runtime used for the
final experiment, then record them in `requirements.txt`.

## Responsible use

This is a research/portfolio code release, not a clinically validated medical
device. Do not use its predictions for diagnosis or treatment decisions.
