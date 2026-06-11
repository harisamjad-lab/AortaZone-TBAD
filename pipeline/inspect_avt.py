"""
inspect_avt.py

Reports on AVT dataset structure, image/label properties, and pairing.
Generates a summary table + 3 sample slice overlays.

Usage:
  python inspect_avt.py --avt_dir C:/ATV_DataSet/ATV_DataSet
"""

import argparse
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path
import json


def read_nrrd(path):
    img  = sitk.ReadImage(str(path))
    data = sitk.GetArrayFromImage(img)  # shape: (Z, Y, X) for SimpleITK
    spacing = img.GetSpacing()           # (X, Y, Z) order
    origin  = img.GetOrigin()
    direction = img.GetDirection()
    return data, spacing, origin, direction, img


def inspect_case(case_dir: Path):
    case_id = case_dir.name
    ct_path  = case_dir / f"{case_id}.nrrd"
    seg_path = case_dir / f"{case_id}.seg.nrrd"

    info = {"case": case_id}

    # CT
    if not ct_path.exists():
        info["error"] = f"missing CT: {ct_path}"
        return info, None, None
    ct_data, ct_spacing, _, _, _ = read_nrrd(ct_path)
    info["ct_shape"]   = list(ct_data.shape)
    info["ct_spacing"] = list(np.round(ct_spacing, 3))
    info["ct_min"]     = float(ct_data.min())
    info["ct_max"]     = float(ct_data.max())
    info["ct_mean"]    = float(round(ct_data.mean(), 1))

    # SEG
    if not seg_path.exists():
        info["error"] = f"missing seg: {seg_path}"
        return info, ct_data, None
    seg_data, seg_spacing, _, _, _ = read_nrrd(seg_path)
    info["seg_shape"]   = list(seg_data.shape)
    info["seg_labels"]  = sorted([int(x) for x in np.unique(seg_data)])
    info["seg_voxels"]  = int((seg_data > 0).sum())
    info["shapes_match"] = (list(ct_data.shape) == list(seg_data.shape))

    return info, ct_data, seg_data


def plot_sample(ct, seg, case_id, out_path):
    has_label = np.where((seg > 0).sum(axis=(1, 2)) > 0)[0]
    if len(has_label) == 0:
        print(f"  [WARN] {case_id} has empty seg, skipping plot")
        return
    z = has_label[len(has_label) // 2]
    has_label_y = np.where((seg > 0).sum(axis=(0, 2)) > 0)[0]
    y = has_label_y[len(has_label_y) // 2] if len(has_label_y) else seg.shape[1] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"AVT inspection: {case_id}", fontsize=13)

    # axial
    axes[0,0].imshow(ct[z], cmap="gray", vmin=-200, vmax=600)
    axes[0,0].set_title(f"Axial CT (z={z})")
    axes[0,1].imshow(seg[z], cmap="hot")
    axes[0,1].set_title(f"Axial Seg")

    # axial overlay
    overlay = np.stack([
        np.clip((ct[z] + 200) / 800, 0, 1)
    ]*3, axis=-1)
    overlay[seg[z] > 0] = [1.0, 0.2, 0.2]
    axes[0,2].imshow(overlay)
    axes[0,2].set_title("Axial Overlay")

    # coronal (y axis on SITK convention is shape[1])
    axes[1,0].imshow(ct[:, y, :], cmap="gray", vmin=-200, vmax=600)
    axes[1,0].set_title(f"Coronal CT (y={y})")
    axes[1,1].imshow(seg[:, y, :], cmap="hot")
    axes[1,1].set_title("Coronal Seg")
    overlay2 = np.stack([
        np.clip((ct[:, y, :] + 200) / 800, 0, 1)
    ]*3, axis=-1)
    overlay2[seg[:, y, :] > 0] = [1.0, 0.2, 0.2]
    axes[1,2].imshow(overlay2)
    axes[1,2].set_title("Coronal Overlay")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avt_dir", type=str,
                        default="C:/ATV_DataSet/ATV_DataSet")
    parser.add_argument("--out_dir", type=str,
                        default="C:/TBAD_Pipeline/outputs/figures/avt_inspection")
    parser.add_argument("--n_plots", type=int, default=3,
                        help="Number of cases to plot (default: 3)")
    args = parser.parse_args()

    avt_dir = Path(args.avt_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # find case folders
    case_dirs = sorted([d for d in avt_dir.iterdir()
                        if d.is_dir() and d.name.startswith("img")])
    print(f"\n{'='*70}")
    print(f"  AVT DATASET INSPECTION")
    print(f"{'='*70}")
    print(f"  Path  : {avt_dir}")
    print(f"  Cases : {len(case_dirs)}")
    print(f"{'='*70}\n")

    summary = []
    sampled = 0
    for cd in case_dirs:
        info, ct, seg = inspect_case(cd)
        summary.append(info)

        # print one-line summary
        if "error" in info:
            print(f"  [{info['case']:6s}] ERROR: {info['error']}")
        else:
            print(f"  [{info['case']:6s}] CT shape={tuple(info['ct_shape'])} "
                  f"spacing={info['ct_spacing']} "
                  f"HU=[{info['ct_min']:6.0f},{info['ct_max']:6.0f}] mean={info['ct_mean']:6.1f}")
            print(f"  {'':8s}  Seg labels={info['seg_labels']} "
                  f"voxels={info['seg_voxels']} "
                  f"match={info['shapes_match']}")

        # plot first N cases
        if sampled < args.n_plots and ct is not None and seg is not None:
            plot_sample(ct, seg, info['case'],
                        out_dir / f"sample_{info['case']}.png")
            sampled += 1
            print(f"  {'':8s}  -> saved overlay to sample_{info['case']}.png")

    # save summary JSON
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # aggregate stats
    print(f"\n{'='*70}")
    print(f"  AGGREGATE STATS")
    print(f"{'='*70}")
    valid = [s for s in summary if "error" not in s]
    if valid:
        all_labels = set()
        for s in valid:
            all_labels.update(s["seg_labels"])
        print(f"  Valid cases       : {len(valid)} / {len(summary)}")
        print(f"  Unique seg labels : {sorted(all_labels)}")
        print(f"  Shape mismatches  : {sum(1 for s in valid if not s['shapes_match'])}")
        print(f"  HU range overall  : "
              f"min={min(s['ct_min'] for s in valid):.0f}  "
              f"max={max(s['ct_max'] for s in valid):.0f}")

        shapes = [tuple(s['ct_shape']) for s in valid]
        print(f"  Shape variety     : {len(set(shapes))} unique shapes")
        spacings = [tuple(s['ct_spacing']) for s in valid]
        print(f"  Spacing variety   : {len(set(spacings))} unique spacings")

    print(f"\n  Summary saved to: {out_dir / 'summary.json'}")
    print(f"  Sample plots in : {out_dir}")


if __name__ == "__main__":
    main()