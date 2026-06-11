"""
train_tl_fl_5fold.py

5-fold cross-validation training of 3D U-Net for TL/FL segmentation
on GT-cropped ImageTBAD dataset.

Input:
  gt_cropped_imagesTr/  -> CT volumes cropped around GT aorta ROI
  gt_cropped_labelsTr/  -> TL/FL labels (0=bg, 1=TL, 2=FL)
  folds.json            -> 5-fold split

Output:
  tl_fl_training_outputs/
    fold_1/
      best_model.pth
      final_model.pth
      summary.txt
      train_loss.npy
      val_dice.npy
      val_tl_dice.npy
      val_fl_dice.npy
      training_curve.png

Usage:
  python train_tl_fl_5fold.py --base_dir C:/imageTBAD --epochs 100
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.inferers import sliding_window_inference
from monai.handlers.utils import from_engine


# ── helpers ──────────────────────────────────────────────────────────────────

def build_file_list(images_dir: Path, labels_dir: Path, names: list) -> list:
    data = []
    for name in names:
        img_path = images_dir / name
        lbl_path = labels_dir / name
        if img_path.exists() and lbl_path.exists():
            data.append({"image": str(img_path), "label": str(lbl_path)})
        else:
            print(f"  [WARN] Missing pair for {name}, skipping.")
    return data


def get_transforms(patch_size, spacing, train=True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=spacing,
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=-200, a_max=800,
                             b_min=0.0,  b_max=1.0,
                             clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"],
                    spatial_size=patch_size,
                    mode="constant"),
        EnsureTyped(keys=["image", "label"]),
    ]

    if train:
        aug = [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=3, neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandZoomd(keys=["image", "label"], prob=0.3,
                      min_zoom=0.9, max_zoom=1.1, mode=["trilinear", "nearest"]),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
        ]
        return Compose(base + aug)
    else:
        return Compose(base)


def save_curve(train_losses, val_dices, tl_dices, fl_dices, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train Loss")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax2.plot(val_dices,  label="Mean Dice (TL+FL)")
    ax2.plot(tl_dices,   label="TL Dice")
    ax2.plot(fl_dices,   label="FL Dice")
    ax2.set_title("Validation Dice")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120)
    plt.close()


# ── training loop ─────────────────────────────────────────────────────────────

def train_fold(fold_idx, train_files, val_files, args, device):
    fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    patch_size = tuple(args.patch_size)
    spacing    = tuple(args.spacing)

    print(f"\n{'='*65}")
    print(f"  FOLD {fold_idx}  |  train={len(train_files)}  val={len(val_files)}")
    print(f"{'='*65}")

    # ── datasets / loaders ──
    # CacheDataset caches preprocessed volumes in RAM → 3-5x faster epochs
    print(f"  Caching training data ({len(train_files)} cases) in RAM...")
    train_ds = CacheDataset(data=train_files,
                            transform=get_transforms(patch_size, spacing, train=True),
                            cache_rate=1.0, num_workers=2)
    print(f"  Caching validation data ({len(val_files)} cases) in RAM...")
    val_ds   = CacheDataset(data=val_files,
                            transform=get_transforms(patch_size, spacing, train=False),
                            cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── model ──
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,          # background, TL, FL
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # ── loss / optimizer / scheduler ──
    loss_fn   = DiceCELoss(to_onehot_y=True, softmax=True, include_background=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    best_mean_dice = -1.0
    best_epoch     = -1
    train_losses, val_dices, tl_dices, fl_dices = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        # ── TRAIN ──
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # Print every step on its own line so progress is always visible
            print(f"  [Fold {fold_idx} | Epoch {epoch:3d}/{args.epochs} "
                  f"| Step {step:3d}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        elapsed = time.time() - t0

        # ── VALIDATE ──
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch["image"].to(device)
                val_labels = val_batch["label"].to(device)

                val_outputs = sliding_window_inference(
                    val_images, patch_size, sw_batch_size=2, predictor=model,
                    overlap=0.5
                )
                # one-hot for metric
                val_pred = torch.argmax(val_outputs, dim=1, keepdim=True)
                # convert to one-hot [B, C, ...]
                val_pred_oh = torch.zeros_like(val_outputs).scatter_(
                    1, val_pred, 1)
                val_lbl_oh  = torch.zeros_like(val_outputs).scatter_(
                    1, val_labels.long(), 1)

                dice_metric(y_pred=val_pred_oh, y=val_lbl_oh)

        # metric shape: [num_classes-1] because include_background=False → [TL, FL]
        per_class = dice_metric.aggregate()          # tensor [2]
        tl_dice   = per_class[0].item()
        fl_dice   = per_class[1].item()
        mean_dice = (tl_dice + fl_dice) / 2.0

        val_dices.append(mean_dice)
        tl_dices.append(tl_dice)
        fl_dices.append(fl_dice)

        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ Fold {fold_idx} | Epoch {epoch:3d}/{args.epochs} DONE "
              f"({elapsed:.1f}s)")
        print(f"  │   Train Loss   : {avg_loss:.4f}")
        print(f"  │   Val Mean Dice: {mean_dice:.4f}")
        print(f"  │   Val TL Dice  : {tl_dice:.4f}")
        print(f"  │   Val FL Dice  : {fl_dice:.4f}")
        print(f"  └─────────────────────────────────────────────────────────────┘", flush=True)

        # ── save best ──
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_epoch     = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_metric":      best_mean_dice,
                "best_epoch":       best_epoch,
                "fold":             fold_idx,
            }, str(fold_dir / "best_model.pth"))
            print(f"  *** Best model saved | Fold {fold_idx} | Epoch {epoch} "
                  f"| Mean Dice {best_mean_dice:.4f} "
                  f"| TL {tl_dice:.4f} | FL {fl_dice:.4f} ***")

    # ── save final model ──
    torch.save({"model_state_dict": model.state_dict()},
               str(fold_dir / "final_model.pth"))

    # ── save curves ──
    np.save(str(fold_dir / "train_loss.npy"),  np.array(train_losses))
    np.save(str(fold_dir / "val_dice.npy"),    np.array(val_dices))
    np.save(str(fold_dir / "val_tl_dice.npy"), np.array(tl_dices))
    np.save(str(fold_dir / "val_fl_dice.npy"), np.array(fl_dices))
    save_curve(train_losses, val_dices, tl_dices, fl_dices,
               fold_dir / "training_curve.png")

    # ── summary ──
    summary = (
        f"Fold {fold_idx} Summary\n"
        f"{'='*40}\n"
        f"Best Mean Dice : {best_mean_dice:.4f}\n"
        f"Best TL Dice   : {tl_dices[best_epoch-1]:.4f}\n"
        f"Best FL Dice   : {fl_dices[best_epoch-1]:.4f}\n"
        f"Best Epoch     : {best_epoch}\n"
        f"Total Epochs   : {args.epochs}\n"
        f"Patch Size     : {patch_size}\n"
        f"Spacing        : {spacing}\n"
        f"Batch Size     : {args.batch_size}\n"
        f"LR             : {args.lr}\n"
    )
    print(f"\n{summary}")
    with open(fold_dir / "summary.txt", "w") as f:
        f.write(summary)

    return best_mean_dice, tl_dices[best_epoch-1], fl_dices[best_epoch-1], best_epoch


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   type=str,   default="C:/imageTBAD")
    parser.add_argument("--output_dir", type=str,   default="C:/imageTBAD/tl_fl_training_outputs")
    parser.add_argument("--folds_json", type=str,   default="C:/imageTBAD/folds.json")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--patch_size", type=int,   nargs=3, default=[128, 128, 128])
    parser.add_argument("--spacing",    type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--fold",       type=int,   default=None,
                        help="Run a single fold only (1-5). Default: run all 5.")
    args = parser.parse_args()

    # ── HARD GPU CHECK ──
    if not torch.cuda.is_available():
        print("\n" + "!"*65)
        print("!  ERROR: CUDA is not available. Training requires a GPU.")
        print("!  Check that PyTorch was installed with CUDA support and")
        print("!  your NVIDIA drivers are working: nvidia-smi")
        print("!"*65)
        sys.exit(1)

    device = torch.device("cuda")
    print(f"\n{'='*65}")
    print(f"  GPU CONFIRMED")
    print(f"{'='*65}")
    print(f"  Device      : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA Ver    : {torch.version.cuda}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"{'='*65}")
    torch.backends.cudnn.benchmark = True  # speed up

    base_dir    = Path(args.base_dir)
    images_dir  = base_dir / "gt_cropped_imagesTr"
    labels_dir  = base_dir / "gt_cropped_labelsTr"

    print(f"\nImages : {images_dir}")
    print(f"Labels : {labels_dir}")
    print(f"Output : {args.output_dir}")
    print(f"Epochs : {args.epochs}  |  Patch: {args.patch_size}  |  Spacing: {args.spacing}")

    # load folds — handle both dict format {"fold_1": {...}} and list format [{...}, ...]
    with open(args.folds_json) as f:
        folds_raw = json.load(f)

    folds = {}
    if isinstance(folds_raw, list):
        for i, fold in enumerate(folds_raw, start=1):
            folds[f"fold_{i}"] = fold
        print(f"  folds.json detected as list -> mapped to fold_1..fold_{len(folds_raw)}")
    elif isinstance(folds_raw, dict):
        # already in dict format, possibly with keys like "fold_1" or "1" or 1
        for k, v in folds_raw.items():
            key = k if isinstance(k, str) and k.startswith("fold_") else f"fold_{k}"
            folds[key] = v
    else:
        raise ValueError(f"Unsupported folds.json format: {type(folds_raw)}")

    # show first val case from fold_1 for sanity
    sample_val = folds["fold_1"]["val"][0] if "val" in folds["fold_1"] else folds["fold_1"]["validation"][0]
    print(f"  Sample val filename in fold_1: {sample_val}")

    fold_indices = [args.fold] if args.fold else list(range(1, 6))

    all_results = {}
    fold_times  = []
    overall_start = time.time()

    for i, fold_idx in enumerate(fold_indices, start=1):
        fold_start = time.time()
        fold_key    = f"fold_{fold_idx}"
        fold_data   = folds[fold_key]
        train_names = fold_data.get("train", fold_data.get("training", []))
        val_names   = fold_data.get("val",   fold_data.get("validation", []))

        train_files = build_file_list(images_dir, labels_dir, train_names)
        val_files   = build_file_list(images_dir, labels_dir, val_names)

        if not train_files or not val_files:
            print(f"[ERROR] Fold {fold_idx}: empty train or val list. "
                  f"Check folds.json names match gt_cropped_imagesTr.")
            continue

        best_mean, best_tl, best_fl, best_ep = train_fold(
            fold_idx, train_files, val_files, args, device)

        all_results[fold_idx] = {
            "best_mean_dice": best_mean,
            "best_tl_dice":   best_tl,
            "best_fl_dice":   best_fl,
            "best_epoch":     best_ep,
        }

        fold_elapsed = time.time() - fold_start
        fold_times.append(fold_elapsed)
        avg_fold_time = sum(fold_times) / len(fold_times)
        remaining_folds = len(fold_indices) - i
        eta_min = (avg_fold_time * remaining_folds) / 60.0

        print(f"\n  >> Fold {fold_idx} took {fold_elapsed/60:.1f} min  "
              f"|  ETA for remaining {remaining_folds} fold(s): {eta_min:.1f} min")

    # ── final summary across folds ──
    if len(all_results) > 1:
        print(f"\n{'='*65}")
        print(f"  5-FOLD FINAL SUMMARY")
        print(f"{'='*65}")
        mean_dices = [v["best_mean_dice"] for v in all_results.values()]
        tl_dices   = [v["best_tl_dice"]   for v in all_results.values()]
        fl_dices   = [v["best_fl_dice"]   for v in all_results.values()]

        for fold_idx, res in all_results.items():
            print(f"  Fold {fold_idx} | Mean {res['best_mean_dice']:.4f} "
                  f"| TL {res['best_tl_dice']:.4f} "
                  f"| FL {res['best_fl_dice']:.4f} "
                  f"| Best Epoch {res['best_epoch']}")

        print(f"\n  Avg Mean Dice : {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}")
        print(f"  Avg TL Dice   : {np.mean(tl_dices):.4f} ± {np.std(tl_dices):.4f}")
        print(f"  Avg FL Dice   : {np.mean(fl_dices):.4f} ± {np.std(fl_dices):.4f}")
        print(f"{'='*65}")

        # save global summary
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "global_summary.txt", "w") as f:
            f.write("5-Fold TL/FL Training Summary\n")
            f.write("="*40 + "\n")
            for fold_idx, res in all_results.items():
                f.write(f"Fold {fold_idx}: Mean={res['best_mean_dice']:.4f} "
                        f"TL={res['best_tl_dice']:.4f} FL={res['best_fl_dice']:.4f} "
                        f"Epoch={res['best_epoch']}\n")
            f.write(f"\nAvg Mean Dice : {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}\n")
            f.write(f"Avg TL Dice   : {np.mean(tl_dices):.4f} ± {np.std(tl_dices):.4f}\n")
            f.write(f"Avg FL Dice   : {np.mean(fl_dices):.4f} ± {np.std(fl_dices):.4f}\n")
        print(f"\n  Global summary saved to {out_dir / 'global_summary.txt'}")

    total_min = (time.time() - overall_start) / 60.0
    print(f"\n  TOTAL TIME: {total_min:.1f} min ({total_min/60:.2f} hours)")


if __name__ == "__main__":
    main()