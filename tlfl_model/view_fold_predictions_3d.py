"""
view_fold_predictions_3d.py

Generates an interactive 3D Plotly HTML viewer showing GT vs prediction
side-by-side using marching-cubes meshes for TL (red) and FL (blue).

Output: HTML file you can open in a browser, rotate, zoom, toggle layers.

Usage:
  python view_fold_predictions_3d.py --base_dir C:/imageTBAD --fold 1 --case_idx 0
  python view_fold_predictions_3d.py --base_dir C:/imageTBAD --fold 1 --case_idx 0 --downsample 2
"""

import argparse
import json
import numpy as np
import torch
import nibabel as nib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes
from pathlib import Path

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd,
    ScaleIntensityRanged, EnsureTyped,
)
from monai.inferers import sliding_window_inference


def load_fold_model(model_path, device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)
    ckpt = torch.load(model_path, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded best_model.pth from epoch {ckpt.get('best_epoch', '?')} "
              f"with metric {ckpt.get('best_metric', 0):.4f}")
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def mesh_from_mask(mask, label_value, color, name, downsample=1, opacity=0.55):
    """Returns a Plotly Mesh3d trace from a binary mask via marching cubes."""
    binary = (mask == label_value).astype(np.uint8)
    if binary.sum() < 50:
        return None  # too small to mesh

    if downsample > 1:
        binary = binary[::downsample, ::downsample, ::downsample]

    try:
        verts, faces, _, _ = marching_cubes(binary, level=0.5)
    except (RuntimeError, ValueError):
        return None

    # NOTE: marching_cubes returns verts in (z, y, x) order
    return go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
        hovertext=name, hoverinfo="text",
    )


def dice_score(pred, gt, label):
    p = (pred == label); g = (gt == label)
    inter = np.logical_and(p, g).sum(); denom = p.sum() + g.sum()
    return float("nan") if denom == 0 else 2.0 * inter / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir",   type=str, default="C:/imageTBAD")
    parser.add_argument("--fold",       type=int, default=1)
    parser.add_argument("--case_idx",   type=int, default=0)
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--spacing",    type=float, nargs=3, default=[1.5, 1.5, 1.5])
    parser.add_argument("--downsample", type=int, default=1,
                        help="Downsample factor for mesh (1=full, 2=half) — bigger = faster, lower quality")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    base_dir   = Path(args.base_dir)
    images_dir = base_dir / "gt_cropped_imagesTr"
    labels_dir = base_dir / "gt_cropped_labelsTr"
    model_path = base_dir / "tl_fl_training_outputs" / f"fold_{args.fold}" / "best_model.pth"
    out_dir    = base_dir / "tl_fl_training_outputs" / f"fold_{args.fold}" / "predictions_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(base_dir / "folds.json") as f:
        folds_raw = json.load(f)
    val_names = folds_raw[args.fold - 1]["val"] if isinstance(folds_raw, list) \
                else folds_raw[f"fold_{args.fold}"]["val"]

    if args.case_idx >= len(val_names):
        print(f"case_idx {args.case_idx} out of range (0..{len(val_names)-1})")
        return

    case_name = val_names[args.case_idx]
    print(f"Fold     : {args.fold}")
    print(f"Case     : {case_name} (val idx {args.case_idx} of {len(val_names)})")

    transform = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=tuple(args.spacing),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=800,
                             b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys=["image", "label"]),
    ])

    data = transform({
        "image": str(images_dir / case_name),
        "label": str(labels_dir / case_name),
    })

    img = data["image"].unsqueeze(0).to(device)
    lbl = data["label"].squeeze().cpu().numpy().astype(np.uint8)

    # predict
    model = load_fold_model(str(model_path), device)
    print("  Running sliding-window inference...")
    with torch.no_grad():
        out = sliding_window_inference(img, tuple(args.patch_size),
                                       sw_batch_size=2, predictor=model,
                                       overlap=0.5)
    pred = torch.argmax(out, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    tl_d = dice_score(pred, lbl, 1)
    fl_d = dice_score(pred, lbl, 2)
    print(f"  TL Dice = {tl_d:.4f}  |  FL Dice = {fl_d:.4f}")
    print(f"  TL voxels  GT={int((lbl==1).sum())}  Pred={int((pred==1).sum())}")
    print(f"  FL voxels  GT={int((lbl==2).sum())}  Pred={int((pred==2).sum())}")

    # build meshes
    print("  Building 3D meshes...")
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"Ground Truth — {case_name}",
            f"Prediction — TL={tl_d:.3f}  FL={fl_d:.3f}",
        ),
    )

    # GT meshes
    gt_tl = mesh_from_mask(lbl, 1, "red",  "TL (GT)",   args.downsample)
    gt_fl = mesh_from_mask(lbl, 2, "blue", "FL (GT)",   args.downsample)
    if gt_tl: fig.add_trace(gt_tl, row=1, col=1)
    if gt_fl: fig.add_trace(gt_fl, row=1, col=1)

    # Pred meshes
    pr_tl = mesh_from_mask(pred, 1, "red",  "TL (Pred)", args.downsample)
    pr_fl = mesh_from_mask(pred, 2, "blue", "FL (Pred)", args.downsample)
    if pr_tl: fig.add_trace(pr_tl, row=1, col=2)
    if pr_fl: fig.add_trace(pr_fl, row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"3D — Fold {args.fold} — {case_name}",
            x=0.5, xanchor="center",
        ),
        height=800, width=1600,
        scene=dict(aspectmode="data",
                   xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
        scene2=dict(aspectmode="data",
                    xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    )

    out_html = out_dir / f"3d_{case_name.replace('.nii.gz','')}.html"
    fig.write_html(str(out_html), include_plotlyjs="cdn")
    print(f"\n  Saved: {out_html}")
    print(f"  Open this file in a browser to interact (rotate, zoom, toggle TL/FL).")


if __name__ == "__main__":
    main()