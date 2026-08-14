# LiTS Two-Stage 3D Liver/Tumor Segmentation

Research code for a two-stage 3D segmentation pipeline on the LiTS liver
tumor CT task. The repository is an experiment snapshot rather than a
turnkey Python package: dataset files, cached data, checkpoints, and logs are
kept outside version control.

## Pipeline

```text
LiTS CT + labels
      |
      v
Stage 1: liver/tumor sub-region prediction
      |
      +--> Stage-1 prediction masks --> ROI extraction
      |
      +--> SDF generation
                         |
                         v
Stage 2: CT + SDF input --> 3-class liver/tumor segmentation
                         |
                         v
                  sliding-window evaluation
```

### Stage 1

- 3D MONAI `DynUNet` with two binary output channels: whole liver and tumor.
- NIfTI loading, RAS reorientation, resampling to `(1.0, 1.0, 1.5)`, CT
  intensity scaling, foreground cropping, and label-guided patch sampling.
- Deep supervision with a Tversky + BCE objective.
- Validation uses MONAI Dice metrics and saves the best tumor checkpoint.

### Intermediate processing

- `generate_stage1_predictions.py` produces Stage-1 prediction masks.
- `generate_roi.py` derives a liver bounding box and writes cropped ROI
  volumes, labels, and bounding-box JSON files.
- `generate_sdf.py` converts liver/tumor masks to signed-distance-field (SDF)
  volumes for Stage 2.

### Stage 2

- 3D `DynUNet` with deep supervision and three output classes
  (background/liver/tumor).
- Uses CT together with generated SDF channels.
- Uses `GeneralizedDiceLoss + FocalLoss`, mixed precision on CUDA, gradient
  accumulation, persistent MONAI datasets, and GPU-memory-aware patch sizes.
- Validation applies tumor-volume post-processing and reports tumor, liver,
  and whole-liver Dice/precision/recall statistics.

## Repository layout

```text
src/
  train_stage1.py                 Stage-1 training
  generate_stage1_predictions.py  Stage-1 inference
  generate_roi.py                 ROI extraction
  generate_sdf.py                 SDF generation
  train_stage2.py                 Stage-2 training
  evaluate_stage2.py              Stage-2 evaluation
configs/                          Local configuration notes/examples
docs/                             Experiment documentation
outputs/                          Generated outputs (ignored)
```

## Dataset layout

The scripts expect a local LiTS directory similar to:

```text
LiTs/
  volume-0.nii
  volume-1.nii
  ...
  segmentations/
    segmentation-0.nii
    segmentation-1.nii
    ...
```

The LiTS data is not included. Obtain and use it under the dataset's own
terms. Do not commit patient-derived images or generated NIfTI volumes.

## Environment

The code was written for a Python 3 environment with CUDA-enabled PyTorch
recommended. Main dependencies are:

```text
PyTorch
MONAI
NumPy
SciPy
NiBabel
tqdm
requests
```

Install versions compatible with the target CUDA/PyTorch runtime. 

## Running the pipeline

The scripts currently use an `~/autodl-tmp/` directory layout by default.
Before running, edit the path constants in the scripts or adapt the local
directory layout.

From the repository root:

```bash
python src/train_stage1.py
python src/generate_stage1_predictions.py
python src/generate_sdf.py
python src/train_stage2.py
```

For evaluation, point the evaluator at the training module and checkpoint:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export STAGE2_TRAIN_SCRIPT="train_stage2"
export STAGE2_BEST="/path/to/best_s2.pth"
python src/evaluate_stage2.py
```

On PowerShell, use `$env:PYTHONPATH`, `$env:STAGE2_TRAIN_SCRIPT`, and
`$env:STAGE2_BEST` instead of `export`.

Useful runtime variables include:

| Variable | Purpose |
| --- | --- |
| `STAGE2_BEST` | Stage-2 checkpoint used by evaluation |
| `STAGE2_TRAIN_SCRIPT` | Training module imported by evaluation |
| `LITS_CACHE_DIR` | Persistent Stage-2 cache location |
| `LITS_NUM_WORKERS` | DataLoader worker count |
| `LITS_TTA_MODE` | Evaluation TTA: `none`, `single`, or `all` |
| `LITS_OVERLAP` | Sliding-window overlap |
| `LITS_TUMOR_BIAS` | Tumor-logit calibration bias |
| `LITS_MIN_TUMOR_VOL` | Minimum tumor component volume |
| `AUTODL_WECHAT_TOKEN` | Optional training notification token |

Notification credentials are read from the environment only and must never
be committed.

## Reproducibility notes

- Checkpoints and persistent caches are required for evaluation but are not
  distributed here.
- Stage-1 and Stage-2 scripts contain experiment-specific defaults and local
  paths; review them before launching a new run.
- Set deterministic mode in `train_stage2.py` when reproducibility is more
  important than throughput.
- Record the dataset split, checkpoint, patch size, calibration parameters,
  and dependency versions with each experiment.

## Responsible use

This repository is for research and software-development purposes. It is not
a clinically validated medical device, and its outputs must not be used for
diagnosis or treatment decisions.
