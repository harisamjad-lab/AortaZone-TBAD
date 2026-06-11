"""
view_avt_validation.py

Visualizes a validation case from AVT training:
- Prediction vs Ground Truth in 2D (axial + coronal slices)
- Prediction vs Ground Truth in 3D (interactive Plotly HTML)
- Per-case Dice score

Usage:
  python view_avt_validation.py --case avt_007
  python view_avt_validation.py --case avt_007 --downsample 2
"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes
from pathlib import Path


def load_nii(path):
    return nib.load(str(path)).get_fdata()


def auto_window(ct):
    """Return (vmin, vmax) for soft-tissue display, handling signed/unsigned encoding."""
    if ct.min() >= 0:
        return (824, 1624)
    return (-200, 600)


def overlay_mask(ct_slice, mask_slice, color, vmin, vmax, alpha=0.55):
    norm = (np.clip(ct_slice, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
    rgb = np.stack([norm]*3, axis=-1)
    rgb[mask_slice > 0] = color
    return rgb


def dice_score(pred, gt):
    p = pred > 0
    g = gt   > 0
    inter = np.logical_and(p, g).sum()
    denom = p.sum() + g.sum()
    return float("nan") if denom == 0 else 2.0 * inter / denom


def mesh_from_mask(mask, color, name, downsample=1, opacity=0.55):
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() < 50:
        return None
    if downsample > 1:
        binary = binary[::downsample, ::downsample, ::downsample]
    try:
        verts, faces, _, _ = marching_cubes(binary, level=0.5)
    except (RuntimeError, ValueError):
        return None
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",       type=str, required=True,
                        help="Case ID, e.g. avt_007")
    parser.add_argument("--results_dir", type=str,
                        default="C:/TBAD_Pipeline/models/aorta_avt/Dataset502_AVT/"
                                "nnUNetTrainer_250epochs__nnUNetPlans__3d_fullres/fold_0/validation")
    parser.add_argument("--raw_dir",    type=str,
                        default="C:/TBAD_Pipeline/data/AVT_nnunet/nnUNet_raw/Dataset502_AVT")
    parser.add_argument("--out_dir",    type=str,
                        default="C:/TBAD_Pipeline/outputs/figures/avt_validation")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Mesh downsample factor (1=full, 2=faster)")
    args = parser.parse_args()

    pred_path = Path(args.results_dir) / f"{args.case}.nii.gz"
    ct_path   = Path(args.raw_dir) / "imagesTr" / f"{args.case}_0000.nii.gz"
    gt_path   = Path(args.raw_dir) / "labelsTr" / f"{args.case}.nii.gz"
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    for p in [pred_path, ct_path, gt_path]:
        if not p.exists():
            print(f"ERROR: missing file {p}")
            return

    print(f"Loading CT  : {ct_path}")
    ct   = load_nii(ct_path)
    print(f"Loading GT  : {gt_path}")
    gt   = load_nii(gt_path).astype(np.uint8)
    print(f"Loading Pred: {pred_path}")
    pred = load_nii(pred_path).astype(np.uint8)

    print(f"\n  CT shape   : {ct.shape}")
    print(f"  GT voxels  : {int((gt>0).sum())}")
    print(f"  Pred voxels: {int((pred>0).sum())}")

    dice = dice_score(pred, gt)
    print(f"  Dice score : {dice:.4f}")

    vmin, vmax = auto_window(ct)
    print(f"  CT window  : [{vmin}, {vmax}]")

    # ── 2D view ──
    has_label = np.where((gt > 0).sum(axis=(1, 2)) > 0)[0]
    z = has_label[len(has_label) // 2] if len(has_label) else ct.shape[0] // 2
    has_label_y = np.where((gt > 0).sum(axis=(0, 2)) > 0)[0]
    y = has_label_y[len(has_label_y) // 2] if len(has_label_y) else ct.shape[1] // 2

    GT_COLOR   = [0.2, 0.4, 1.0]   # blue
    PRED_COLOR = [1.0, 0.2, 0.2]   # red

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"AVT validation: {args.case}  |  Dice = {dice:.4f}",
                 fontsize=14)

    # axial row
    axes[0,0].imshow(ct[z], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0,0].set_title(f"Axial CT (z={z})")
    axes[0,1].imshow(overlay_mask(ct[z], gt[z], GT_COLOR, vmin, vmax))
    axes[0,1].set_title("Axial GT (blue)")
    axes[0,2].imshow(overlay_mask(ct[z], pred[z], PRED_COLOR, vmin, vmax))
    axes[0,2].set_title("Axial Pred (red)")

    # coronal row
    axes[1,0].imshow(ct[:, y, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1,0].set_title(f"Coronal CT (y={y})")
    axes[1,1].imshow(overlay_mask(ct[:, y, :], gt[:, y, :], GT_COLOR, vmin, vmax))
    axes[1,1].set_title("Coronal GT (blue)")
    axes[1,2].imshow(overlay_mask(ct[:, y, :], pred[:, y, :], PRED_COLOR, vmin, vmax))
    axes[1,2].set_title("Coronal Pred (red)")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    out_2d = out_dir / f"{args.case}_2d.png"
    plt.savefig(str(out_2d), dpi=120, bbox_inches="tight")
    print(f"\n  Saved 2D: {out_2d}")
    plt.show()

    # ── 3D view (Plotly side-by-side) ──
    print("\n  Building 3D meshes...")
    fig3d = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(
            f"Ground Truth — {args.case}",
            f"Prediction — Dice {dice:.3f}",
        ),
    )

    gt_mesh   = mesh_from_mask(gt,   "blue", "Aorta (GT)",   args.downsample)
    pred_mesh = mesh_from_mask(pred, "red",  "Aorta (Pred)", args.downsample)

    if gt_mesh:   fig3d.add_trace(gt_mesh,   row=1, col=1)
    if pred_mesh: fig3d.add_trace(pred_mesh, row=1, col=2)

    fig3d.update_layout(
        title=f"3D AVT Validation — {args.case}  |  Dice = {dice:.4f}",
        height=750, width=1400,
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )

    out_3d = out_dir / f"{args.case}_3d.html"
    fig3d.write_html(str(out_3d), include_plotlyjs="cdn")
    print(f"  Saved 3D: {out_3d}")
    print(f"  Open the .html in a browser to interact (rotate, zoom).")


if __name__ == "__main__":
    main()