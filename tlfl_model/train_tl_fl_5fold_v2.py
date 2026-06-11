"""
train_tl_fl_5fold_v2.py

Improved 5-fold training script with fixes addressing the v1 issues:
  - Finer spacing (1.0mm vs 1.5mm) to preserve TL/FL detail
  - Smaller patches (96^3) with more samples per case (num_samples=4)
  - Higher pos:neg ratio (4:1) so model sees TL/FL more often
  - DiceFocalLoss to handle class imbalance better than DiceCE
  - Removed CropForegroundd (already cropped to GT ROI)
  - Stronger but safer augmentation
  - 80 epochs with cosine LR + early stopping
  - Verbose terminal logging per step / per epoch / per fold
  - Hard GPU check, mixed precision (AMP) for speed

Usage:
  # First: run only fold 1 to verify quality, ~25 min
  python train_tl_fl_5fold_v2.py --base_dir C:/imageTBAD --epochs 80 --fold 1

  # If fold 1 numbers are good (TL > 0.75, FL > 0.55), run all folds:
  python train_tl_fl_5fold_v2.py --base_dir C:/imageTBAD --epochs 80
"""

import sys
import json
import time
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandZoomd,
    RandShiftIntensityd,
    RandScaleIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    EnsureTyped,
)
from monai.data import CacheDataset, DataLoader
from monai.inferers import sliding_window_inference


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


def get_transforms(patch_size, spacing, num_samples, train=True):
    """
    Note: NO CropForegroundd here — data is already cropped to GT ROI.
    SpatialPadd ensures patch can always be sampled.
    """
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"],
                 pixdim=spacing,
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"],
                             a_min=-200, a_max=600,
                             b_min=0.0,  b_max=1.0,
                             clip=True),
        SpatialPadd(keys=["image", "label"],
                    spatial_size=patch_size,
                    mode="constant"),
        EnsureTyped(keys=["image", "label"]),
    ]

    if train:
        aug = [
            # Aggressive foreground sampling — pos:neg=4:1, 4 samples per case
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=patch_size,
                pos=4, neg=1,
                num_samples=num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
            RandZoomd(keys=["image", "label"], prob=0.2,
                      min_zoom=0.9, max_zoom=1.1,
                      mode=["trilinear", "nearest"]),
            RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.5),
            RandScaleIntensityd(keys=["image"], factors=0.10, prob=0.5),
            RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01),
            RandGaussianSmoothd(keys=["image"], prob=0.10,
                                sigma_x=(0.5, 1.0),
                                sigma_y=(0.5, 1.0),
                                sigma_z=(0.5, 1.0)),
        ]
        return Compose(base + aug)
    else:
        return Compose(base)


def save_curve(train_losses, val_dices, tl_dices, fl_dices, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train Loss")
    ax1.set_title("Training Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(val_dices,  label="Mean Dice (TL+FL)")
    ax2.plot(tl_dices,   label="TL Dice")
    ax2.plot(fl_dices,   label="FL Dice")
    ax2.set_title("Validation Dice"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout(); plt.savefig(str(out_path), dpi=120); plt.close()


def train_fold(fold_idx, train_files, val_files, args, device):
    fold_dir = Path(args.output_dir) / f"fold_{fold_idx}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    patch_size  = tuple(args.patch_size)
    spacing     = tuple(args.spacing)
    num_samples = args.num_samples

    print(f"\n{'='*70}")
    print(f"  FOLD {fold_idx}  |  train={len(train_files)}  val={len(val_files)}")
    print(f"  Patch {patch_size}  |  Spacing {spacing}  |  Samples/case {num_samples}")
    print(f"{'='*70}")

    print(f"  Caching training data ({len(train_files)} cases) in RAM...")
    train_ds = CacheDataset(data=train_files,
                            transform=get_transforms(patch_size, spacing, num_samples, True),
                            cache_rate=1.0, num_workers=2)
    print(f"  Caching validation data ({len(val_files)} cases) in RAM...")
    val_ds   = CacheDataset(data=val_files,
                            transform=get_transforms(patch_size, spacing, num_samples, False),
                            cache_rate=1.0, num_workers=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=1,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── model ──
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    # ── DiceFocalLoss with class weighting ──
    # gamma=2.0 emphasizes hard (rare/missed) examples
    loss_fn = DiceFocalLoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        gamma=2.0,
        lambda_dice=1.0,
        lambda_focal=1.0,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP for ~30% speed boost on RTX 4080
    scaler = torch.amp.GradScaler("cuda")

    dice_metric = DiceMetric(include_background=False, reduction="mean_batch")

    best_mean_dice = -1.0
    best_epoch     = -1
    epochs_no_improve = 0
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
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss    = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            print(f"  [Fold {fold_idx} | Epoch {epoch:3d}/{args.epochs} "
                  f"| Step {step:3d}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        train_elapsed = time.time() - t0

        # ── VALIDATE ──
        model.eval()
        dice_metric.reset()
        v0 = time.time()

        with torch.no_grad():
            for val_batch in val_loader:
                val_images = val_batch["image"].to(device, non_blocking=True)
                val_labels = val_batch["label"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda"):
                    val_outputs = sliding_window_inference(
                        val_images, patch_size, sw_batch_size=2,
                        predictor=model, overlap=0.5)

                val_pred = torch.argmax(val_outputs, dim=1, keepdim=True)
                val_pred_oh = torch.zeros_like(val_outputs).scatter_(1, val_pred, 1)
                val_lbl_oh  = torch.zeros_like(val_outputs).scatter_(1, val_labels.long(), 1)
                dice_metric(y_pred=val_pred_oh, y=val_lbl_oh)

        per_class = dice_metric.aggregate()
        tl_dice   = per_class[0].item()
        fl_dice   = per_class[1].item()
        mean_dice = (tl_dice + fl_dice) / 2.0
        val_elapsed = time.time() - v0

        val_dices.append(mean_dice); tl_dices.append(tl_dice); fl_dices.append(fl_dice)

        elapsed = train_elapsed + val_elapsed
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  ┌──────────────────────────────────────────────────────────────┐")
        print(f"  │ Fold {fold_idx} | Epoch {epoch:3d}/{args.epochs} DONE "
              f"({elapsed:.1f}s, train={train_elapsed:.1f}s, val={val_elapsed:.1f}s)")
        print(f"  │   LR             : {lr_now:.2e}")
        print(f"  │   Train Loss     : {avg_loss:.4f}")
        print(f"  │   Val Mean Dice  : {mean_dice:.4f}")
        print(f"  │   Val TL Dice    : {tl_dice:.4f}")
        print(f"  │   Val FL Dice    : {fl_dice:.4f}")
        print(f"  └──────────────────────────────────────────────────────────────┘", flush=True)

        # save best
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            best_epoch     = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_metric":      best_mean_dice,
                "best_epoch":       best_epoch,
                "fold":             fold_idx,
                "args":             vars(args),
            }, str(fold_dir / "best_model.pth"))
            print(f"  *** Best model saved | Fold {fold_idx} | Epoch {epoch} "
                  f"| Mean {best_mean_dice:.4f} TL {tl_dice:.4f} FL {fl_dice:.4f} ***", flush=True)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping triggered ({args.patience} epochs without improvement).")
                break

    # save final + curves
    torch.save({"model_state_dict": model.state_dict()},
               str(fold_dir / "final_model.pth"))

    np.save(str(fold_dir / "train_loss.npy"),  np.array(train_losses))
    np.save(str(fold_dir / "val_dice.npy"),    np.array(val_dices))
    np.save(str(fold_dir / "val_tl_dice.npy"), np.array(tl_dices))
    np.save(str(fold_dir / "val_fl_dice.npy"), np.array(fl_dices))
    save_curve(train_losses, val_dices, tl_dices, fl_dices,
               fold_dir / "training_curve.png")

    summary = (
        f"Fold {fold_idx} Summary\n{'='*45}\n"
        f"Best Mean Dice : {best_mean_dice:.4f}\n"
        f"Best TL Dice   : {tl_dices[best_epoch-1]:.4f}\n"
        f"Best FL Dice   : {fl_dices[best_epoch-1]:.4f}\n"
        f"Best Epoch     : {best_epoch}\n"
        f"Total Epochs   : {len(train_losses)}\n"
        f"Patch          : {patch_size}\n"
        f"Spacing        : {spacing}\n"
        f"Num Samples    : {num_samples}\n"
        f"Batch Size     : {args.batch_size}\n"
        f"LR             : {args.lr}\n"
    )
    print(f"\n{summary}")
    with open(fold_dir / "summary.txt", "w") as f:
        f.write(summary)

    return best_mean_dice, tl_dices[best_epoch-1], fl_dices[best_epoch-1], best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",    type=str,   default="C:/imageTBAD")
    parser.add_argument("--output_dir",  type=str,   default="C:/imageTBAD/tl_fl_training_outputs_v2")
    parser.add_argument("--folds_json",  type=str,   default="C:/imageTBAD/folds.json")
    parser.add_argument("--epochs",      type=int,   default=80)
    parser.add_argument("--batch_size",  type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--patch_size",  type=int,   nargs=3, default=[96, 96, 96])
    parser.add_argument("--spacing",     type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--num_samples", type=int,   default=4,
                        help="Patches per case per epoch (default 4)")
    parser.add_argument("--patience",    type=int,   default=20,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--fold",        type=int,   default=None,
                        help="Run a single fold (1..5). Default: all 5.")
    args = parser.parse_args()

    # ── HARD GPU CHECK ──
    if not torch.cuda.is_available():
        print("\n!!! ERROR: CUDA is not available. Training requires a GPU. !!!")
        sys.exit(1)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    print(f"\n{'='*70}")
    print(f"  TL/FL TRAINING v2 — GPU CONFIRMED")
    print(f"{'='*70}")
    print(f"  Device      : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA Ver    : {torch.version.cuda}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Mixed Prec. : ENABLED (AMP)")
    print(f"{'='*70}")

    base_dir   = Path(args.base_dir)
    images_dir = base_dir / "gt_cropped_imagesTr"
    labels_dir = base_dir / "gt_cropped_labelsTr"

    print(f"\n  Images       : {images_dir}")
    print(f"  Labels       : {labels_dir}")
    print(f"  Output       : {args.output_dir}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Patch        : {args.patch_size}")
    print(f"  Spacing      : {args.spacing}")
    print(f"  Num Samples  : {args.num_samples}")
    print(f"  Batch Size   : {args.batch_size}")
    print(f"  LR           : {args.lr}")
    print(f"  Loss         : DiceFocalLoss (gamma=2.0)")
    print(f"  Patience     : {args.patience} epochs")

    with open(args.folds_json) as f:
        folds_raw = json.load(f)

    folds = {}
    if isinstance(folds_raw, list):
        for i, fold in enumerate(folds_raw, start=1):
            folds[f"fold_{i}"] = fold
        print(f"  folds.json   : list with {len(folds_raw)} folds")
    else:
        for k, v in folds_raw.items():
            key = k if isinstance(k, str) and k.startswith("fold_") else f"fold_{k}"
            folds[key] = v

    fold_indices = [args.fold] if args.fold else list(range(1, 6))

    all_results = {}; fold_times = []
    overall_start = time.time()

    for i, fold_idx in enumerate(fold_indices, start=1):
        fold_start = time.time()
        fd          = folds[f"fold_{fold_idx}"]
        train_names = fd.get("train", fd.get("training", []))
        val_names   = fd.get("val",   fd.get("validation", []))

        train_files = build_file_list(images_dir, labels_dir, train_names)
        val_files   = build_file_list(images_dir, labels_dir, val_names)

        if not train_files or not val_files:
            print(f"[ERROR] Fold {fold_idx}: empty file list. Check folds.json.")
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
        avg_t = sum(fold_times) / len(fold_times)
        eta_min = (avg_t * (len(fold_indices) - i)) / 60.0
        print(f"\n  >> Fold {fold_idx} took {fold_elapsed/60:.1f} min  "
              f"|  ETA remaining: {eta_min:.1f} min")

    # global summary
    if len(all_results) > 1:
        mean_dices = [v["best_mean_dice"] for v in all_results.values()]
        tl_dices   = [v["best_tl_dice"]   for v in all_results.values()]
        fl_dices   = [v["best_fl_dice"]   for v in all_results.values()]

        print(f"\n{'='*70}")
        print(f"  5-FOLD FINAL SUMMARY")
        print(f"{'='*70}")
        for fold_idx, res in all_results.items():
            print(f"  Fold {fold_idx} | Mean {res['best_mean_dice']:.4f} "
                  f"| TL {res['best_tl_dice']:.4f} | FL {res['best_fl_dice']:.4f} "
                  f"| Best Epoch {res['best_epoch']}")
        print(f"\n  Avg Mean Dice : {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}")
        print(f"  Avg TL Dice   : {np.mean(tl_dices):.4f} ± {np.std(tl_dices):.4f}")
        print(f"  Avg FL Dice   : {np.mean(fl_dices):.4f} ± {np.std(fl_dices):.4f}")
        print(f"{'='*70}")

        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "global_summary.txt", "w") as f:
            f.write("5-Fold TL/FL Training v2 Summary\n" + "="*45 + "\n")
            for fold_idx, res in all_results.items():
                f.write(f"Fold {fold_idx}: Mean={res['best_mean_dice']:.4f} "
                        f"TL={res['best_tl_dice']:.4f} FL={res['best_fl_dice']:.4f} "
                        f"Epoch={res['best_epoch']}\n")
            f.write(f"\nAvg Mean Dice : {np.mean(mean_dices):.4f} ± {np.std(mean_dices):.4f}\n")
            f.write(f"Avg TL Dice   : {np.mean(tl_dices):.4f} ± {np.std(tl_dices):.4f}\n")
            f.write(f"Avg FL Dice   : {np.mean(fl_dices):.4f} ± {np.std(fl_dices):.4f}\n")

    total_min = (time.time() - overall_start) / 60.0
    print(f"\n  TOTAL TIME: {total_min:.1f} min ({total_min/60:.2f} hours)")


if __name__ == "__main__":
    main()