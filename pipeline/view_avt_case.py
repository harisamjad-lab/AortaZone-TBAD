"""
view_avt_case.py

Properly visualizes AVT cases with auto-windowing for both signed and
unsigned HU encoding.

Usage:
  python view_avt_case.py --case img1
  python view_avt_case.py --case img10
"""

import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path


def read_nrrd(path):
    img  = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(img)


def auto_window(ct):
    """Auto-detect signed vs unsigned and return (vmin, vmax) for soft-tissue window."""
    if ct.min() >= 0:
        # unsigned: shift to standard HU then window
        return (824, 1624)   # equivalent to -200..600 in standard HU
    else:
        return (-200, 600)


def overlay_mask(ct_slice, mask_slice, vmin, vmax):
    norm = (np.clip(ct_slice, vmin, vmax) - vmin) / (vmax - vmin)
    rgb = np.stack([norm]*3, axis=-1)
    rgb[mask_slice > 0] = [1.0, 0.2, 0.2]
    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avt_dir", type=str, default="C:/ATV_DataSet/ATV_DataSet")
    parser.add_argument("--case",    type=str, required=True,
                        help="Case ID, e.g. img1, img10, img11")
    parser.add_argument("--out_dir", type=str,
                        default="C:/TBAD_Pipeline/outputs/figures/avt_inspection")
    args = parser.parse_args()

    case_dir = Path(args.avt_dir) / args.case
    ct_path  = case_dir / f"{args.case}.nrrd"
    seg_path = case_dir / f"{args.case}.seg.nrrd"

    ct  = read_nrrd(ct_path).astype(np.float32)
    seg = read_nrrd(seg_path).astype(np.uint8)

    print(f"Case   : {args.case}")
    print(f"Shape  : {ct.shape}")
    print(f"HU     : min={ct.min():.0f}, max={ct.max():.0f}, mean={ct.mean():.1f}")
    print(f"Encoding: {'unsigned (shifted)' if ct.min() >= 0 else 'signed HU'}")
    vmin, vmax = auto_window(ct)
    print(f"Window : [{vmin}, {vmax}]")
    print(f"Mask   : voxels={int((seg>0).sum())}")

    has_label = np.where((seg > 0).sum(axis=(1, 2)) > 0)[0]
    z = has_label[len(has_label) // 2] if len(has_label) else ct.shape[0] // 2
    has_label_y = np.where((seg > 0).sum(axis=(0, 2)) > 0)[0]
    y = has_label_y[len(has_label_y) // 2] if len(has_label_y) else ct.shape[1] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"AVT case: {args.case}  (HU encoding: {'unsigned' if ct.min()>=0 else 'signed'})", fontsize=13)

    axes[0,0].imshow(ct[z], cmap="gray", vmin=vmin, vmax=vmax)
    axes[0,0].set_title(f"Axial CT (z={z})")
    axes[0,1].imshow(seg[z], cmap="hot")
    axes[0,1].set_title("Axial Seg")
    axes[0,2].imshow(overlay_mask(ct[z], seg[z], vmin, vmax))
    axes[0,2].set_title("Axial Overlay")

    axes[1,0].imshow(ct[:, y, :], cmap="gray", vmin=vmin, vmax=vmax)
    axes[1,0].set_title(f"Coronal CT (y={y})")
    axes[1,1].imshow(seg[:, y, :], cmap="hot")
    axes[1,1].set_title("Coronal Seg")
    axes[1,2].imshow(overlay_mask(ct[:, y, :], seg[:, y, :], vmin, vmax))
    axes[1,2].set_title("Coronal Overlay")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fixed_{args.case}.png"
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()