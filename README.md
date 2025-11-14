# Project2_Classification

This repository contains a lightweight image classification training and inference skeleton primarily used for Dogs vs. Cats experiments (and can be adapted to CIFAR-10). It includes configurable models, dataloaders, augmentations, training loop, and inference utilities for producing prediction CSVs suitable for competition submission.

## Implemented features

- Models
  - `SimpleCNNBinary` — small convolutional model for binary tasks
  - `BNCnn` — CNN with BatchNorm and dropout
  - `ResNet18` — ResNet-18 style backbone with convenient builder
  - `ResNeXt` — ResNeXt bottleneck implementation (50/101 builders)
  - `VGG` — configurable VGG variants

- Losses
  - CrossEntropy (multi-class)
  - BCEWithLogits (binary, single output)
  - FocalLoss (optional)

- Data and transforms
  - `DogsVsCatsDataModule` using `torchvision.datasets.ImageFolder` (expects folder layout with class subdirectories for `train/` and `val/`, and images in `test/`)
  - `CIFAR10DataModule` using `torchvision.datasets.CIFAR10` (for CIFAR-10 experiments; config-driven)
  - Config-driven augmentations (`configs/aug/`): `light`, `basic`, `none`

- Training and evaluation
  - `src.core.trainer_base.TrainerBase` — training loop with validation and checkpoint saving
  - Automatic saving of `cfg_effective.json` and best checkpoint (`best_acc.ckpt`) in `experiments/` run folders

- Inference
  - `src.inference.predict_folder` — load a run and predict `datasets/test/`, write `predictions.csv`
  - `src.inference.build_submission` — convert `predictions.csv` to `submission.csv` (id,label)

## Directory overview

Key folders and files:

- `src/` — source code (models, datamodules, training engine, transforms, utils)
- `configs/` — YAML configurations for base settings, tasks, models, augmentations, optimizers
- `data/` — datasets folder (Dogs vs Cats needs manual download; CIFAR-10 auto-downloads)
- `experiments/` — run outputs: `cfg_effective.json`, `checkpoints/`, `predictions.csv`, `submission.csv`

## Quick setup

1. Create (or activate) a Python environment and install requirements:

```bash
python -m pip install -r requirements.txt
```

2. Prepare datasets:

**For Dogs vs. Cats:**
- Download the dataset from [Google Drive](https://drive.google.com/file/d/1q0r6yeHQMS17R3wz-s2FIbMR5DAGZK5v/view)
- Extract to `data/dogs_vs_cats/` with the following structure:

```
data/
  dogs_vs_cats/
    train/
      cat/
      dog/
    val/
      cat/
      dog/
    test/   # images only, no subfolders
```

**For CIFAR-10:**
- No manual download needed! The dataset will be automatically downloaded by torchvision on first run
- It will be cached in `data/cifar10/` (configurable in `configs/task/cifar10.yaml`)

## How to run

All runs are controlled by combining YAML config files. Later `--cfg` entries override earlier ones.

**Train Dogs vs. Cats (example with `cnn_bn`):**

```bash
python -m src.engine.trainer \
  --cfg configs/base.yaml \
  --cfg configs/task/dogs_vs_cats.yaml \
  --cfg configs/model/cnn_bn.yaml \
  --cfg configs/aug/light.yaml \
  --cfg configs/optim/adamw.yaml
```

**Train CIFAR-10 (example with `resnet18`):**

```bash
python -m src.engine.trainer \
  --cfg configs/base.yaml \
  --cfg configs/task/cifar10.yaml \
  --cfg configs/model/resnet18.yaml \
  --cfg configs/aug/light.yaml \
  --cfg configs/optim/adamw.yaml
```

Notes:
- Training will write a new run folder under `experiments/` and save `cfg_effective.json` and `checkpoints/best_acc.ckpt`.
- The trainer constructs the model from the provided config and saves only the model state dict and class names in the checkpoint.
- **Important**: Use `loss_fn: bce` for binary classification (Dogs vs Cats), and `loss_fn: ce` for multi-class (CIFAR-10).

**Predict (Dogs vs. Cats - use a completed run folder):**

```bash
python -m src.inference.predict_folder \
  --run experiments/2025-10-26_14-51-41_dogs_vs_cats_cnn_bn
```

- This command reads the run's `cfg_effective.json` (so you don't need to repeat `--cfg`), builds the model, loads weights from `checkpoints/best_acc.ckpt`, predicts images in `data/<task>/test/`, and writes `predictions.csv` in the run folder.

**Build submission CSV (Dogs vs. Cats):**

```bash
python -m src.inference.build_submission --run experiments/2025-10-26_14-51-41_dogs_vs_cats_cnn_bn
```

- This converts `predictions.csv` into `submission.csv` with two columns: `id,label` where `label` is 0 for cat and 1 for dog.

**Evaluate CIFAR-10 test set:**

```bash
python -m src.engine.evaluator --run experiments/<your_cifar10_run_folder>
```

Example:
```bash
python -m src.engine.evaluator --run experiments/2025-10-27_17-08-44_cifar10_resnet18
```

- This command evaluates the trained CIFAR-10 model on the official test set (10,000 images)
- It reports overall test accuracy and per-class accuracy for all 10 classes
- The evaluator automatically loads the model config and checkpoint from the run folder

## Augmentation options

- `configs/aug/light.yaml` — training-time augmentations (RandomResizedCrop, horizontal flip, Normalize)
- `configs/aug/basic.yaml` — deterministic Resize + Normalize
- `configs/aug/none.yaml` — minimal transforms (Resize + ToTensor)

Choose the augmentation you want by passing the corresponding `--cfg configs/aug/<name>.yaml` during training. The predictor will use the run's saved config automatically.

## Important notes

- **Data paths**: Dogs vs Cats expects data in `data/dogs_vs_cats/`; CIFAR-10 uses `data/cifar10/` (auto-downloaded)
- **Loss function selection**:
  - Use `loss_fn: bce` for **binary classification** (e.g., Dogs vs Cats with 2 classes)
  - Use `loss_fn: ce` for **multi-class classification** (e.g., CIFAR-10 with 10 classes)
  - Check your model config file (`configs/model/*.yaml`) to ensure the correct loss is set
- Label mapping: `ImageFolder` builds classes alphabetically; by default `['cat', 'dog']` → `cat=0`, `dog=1`. This mapping is saved in the run checkpoint under `class_names` and used by inference.
- Model outputs:
  - If `loss_fn: bce` is used during training, the model outputs a single logit per sample (shape `[B,1]`) and inference code converts it with `sigmoid` to obtain probabilities for two classes.
  - If `loss_fn: ce` is used, the model outputs logits for each class (shape `[B,C]`) and inference uses `softmax`.

## Tips

- If you want portable runs, save the `cfg_effective.json` that the trainer already produces — inference uses it automatically.
- For reproducibility, set `seed` in `configs/base.yaml`.


