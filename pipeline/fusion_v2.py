"""
fusion_v2.py  (Fix 1 + Fix 2 + Fix 3 + QC + status logic + component cleanup)

Fix 1:
  TL/FL is gated by the aorta mask.

Fix 2:
  Anatomical regions are split by aorta extent, not TL/FL extent.

Fix 3:
  Region tint is visible on aorta-only pixels.

QC:
  Reports how much TL/FL prediction survives aorta gating.
  Adds low-quality warning if >90% of TL/FL is outside the aorta.
  Saves QC fields into JSON.

Status logic:
  Avoids overconfident "TBAD DETECTED" when TL/FL signal is low quality.
  Separates candidate TBAD from final high-confidence detection.

Step 3:
  Connected-component cleanup after aorta gating.
  Removes small disconnected TL/FL islands below min_component_voxels.

Coronal Y fix:
  Picks Y slice with maximum TL/FL voxel count (argmax), not median.
  Ensures coronal panel shows thickest cross-section, not the edge.

Region tint fix:
  Blend strength increased from 0.45 to 0.55 for better visibility.
  Axial panel title now shows which anatomical region the slice falls in.
  Aorta-only mask uses explicit bool cast for safety.

Usage:
  python fusion_v2.py --case case_050 \
    --tlfl_file  C:/TBAD_Pipeline/outputs/predictions/tlfl_case050_fullvol.nii.gz \
    --aorta_file C:/TBAD_Pipeline/outputs/predictions/avt_on_imagetbad_corrected/imagetbad_050.nii.gz
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import label


REGION_COLORS = {
    "upper":  [1.0, 0.6, 0.0],
    "middle": [0.2, 0.6, 1.0],
    "lower":  [0.2, 0.8, 0.3],
}
TL_COLOR = [1.0, 0.15, 0.15]
FL_COLOR = [0.15, 0.15, 1.0]


def load_nii(path):
    nii = nib.load(str(path))
    return nii.get_fdata(), nii.header.get_zooms()


def voxels_to_ml(n, spacing):
    return round(n * spacing[0] * spacing[1] * spacing[2] / 1000.0, 2)


def split_by_aorta(aorta):
    fg = np.where((aorta > 0).sum(axis=(0, 1)) > 0)[0]
    if len(fg) == 0:
        return None

    z_min, z_max = int(fg.min()), int(fg.max())
    z1 = z_min + (z_max - z_min) // 3
    z2 = z_min + 2 * (z_max - z_min) // 3
    return z_min, z1, z2, z_max


def compute_stats(tlfl, z_start, z_end, spacing):
    tl = int((tlfl[:, :, z_start:z_end] == 1).sum())
    fl = int((tlfl[:, :, z_start:z_end] == 2).sum())

    return {
        "TL_present": tl > 100,
        "FL_present": fl > 100,
        "TL_voxels": tl,
        "FL_voxels": fl,
        "TL_volume_ml": voxels_to_ml(tl, spacing),
        "FL_volume_ml": voxels_to_ml(fl, spacing),
    }


def keep_components_above_threshold(mask, min_voxels=200):
    """
    Keep all connected components with size >= min_voxels.
    This removes small islands but does not force a single component.
    """
    mask = mask.astype(bool)

    if mask.sum() == 0:
        return mask

    lbl, n = label(mask)

    if n == 0:
        return mask

    counts = np.bincount(lbl.ravel())
    counts[0] = 0

    keep_ids = np.where(counts >= min_voxels)[0]

    if len(keep_ids) == 0:
        return np.zeros_like(mask, dtype=bool)

    return np.isin(lbl, keep_ids)


def make_2d_figure(ct, tlfl, aorta, bounds, regions_stats, case_id,
                   dissection, report, out_path, vmin, vmax):
    z_min, z1, z2, z_max = bounds

    # Best axial slice: prefer FL; otherwise TL; otherwise middle aorta slice
    fl_counts    = (tlfl == 2).sum(axis=(0, 1))
    tl_counts    = (tlfl == 1).sum(axis=(0, 1))
    aorta_counts = (aorta > 0).sum(axis=(0, 1))

    if fl_counts.max() > 0:
        z = int(np.argmax(fl_counts))
    elif tl_counts.max() > 0:
        z = int(np.argmax(tl_counts))
    elif aorta_counts.max() > 0:
        z = int(np.argmax(aorta_counts))
    else:
        z = ct.shape[2] // 2

    # ── Coronal Y fix ──
    # Pick Y with maximum TL/FL voxel count — thickest cross-section, not edge.
    tlfl_y_profile  = (tlfl > 0).sum(axis=(0, 2))   # shape: (Y,)
    if tlfl_y_profile.max() > 0:
        y = int(np.argmax(tlfl_y_profile))
    else:
        aorta_y_profile = (aorta > 0).sum(axis=(0, 2))
        y = int(np.argmax(aorta_y_profile)) if aorta_y_profile.max() > 0 else ct.shape[1] // 2

    def region_color_for_z(z_idx):
        if z_idx >= z2:
            return REGION_COLORS["upper"]
        if z_idx >= z1:
            return REGION_COLORS["middle"]
        return REGION_COLORS["lower"]

    # ── Region tint fix: strength raised from 0.45 to 0.55 ──
    def blend_region(base_rgb, region_color, strength=0.55):
        return (1.0 - strength) * base_rgb + strength * np.array(region_color)

    def make_overlay(ct_sl, tlfl_sl, aorta_sl, z_idx):
        norm = (np.clip(ct_sl, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
        rgb  = np.stack([norm] * 3, axis=-1)

        rc = region_color_for_z(z_idx)

        # ── aorta_sl cast fix: explicit bool to avoid uint8 comparison edge cases ──
        aorta_only = aorta_sl.astype(bool) & (tlfl_sl == 0)
        rgb[aorta_only] = blend_region(rgb[aorta_only], rc)

        rgb[tlfl_sl == 1] = TL_COLOR
        rgb[tlfl_sl == 2] = FL_COLOR
        return rgb

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    status = report.get("status", "Unknown")
    title  = f"Fusion: {case_id}  |  {status}"

    if "DETECTED" in status:
        title_color = "darkred"
    elif "suspected" in status or "Indeterminate" in status:
        title_color = "darkorange"
    else:
        title_color = "darkgreen"

    fig.suptitle(title, fontsize=13, color=title_color)

    # Panel 0 — raw CT
    axes[0].imshow(ct[:, :, z].T, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Axial CT (z={z})")

    # Panel 1 — TL/FL overlay with region tint
    # ── Axial title now shows which anatomical region the slice falls in ──
    if z >= z2:
        z_region_name = "upper aorta"
    elif z >= z1:
        z_region_name = "middle aorta"
    else:
        z_region_name = "lower aorta"

    axes[1].imshow(
        make_overlay(
            ct[:, :, z].T,
            tlfl[:, :, z].T,
            aorta[:, :, z].T,
            z,
        )
    )
    axes[1].set_title(f"Axial z={z} ({z_region_name}): TL (red) / FL (blue)")

    # Panel 2 — coronal view
    norm_cor = (np.clip(ct[:, y, :], vmin, vmax) - vmin) / (vmax - vmin + 1e-8)
    rgb_cor  = np.stack([norm_cor] * 3, axis=-1)

    for z_idx in range(ct.shape[2]):
        rc       = region_color_for_z(z_idx)
        tlfl_sl  = tlfl[:, y, z_idx]
        aorta_sl = aorta[:, y, z_idx]

        # ── aorta_sl cast fix ──
        aorta_only = aorta_sl.astype(bool) & (tlfl_sl == 0)
        rgb_cor[aorta_only, z_idx] = blend_region(
            rgb_cor[aorta_only, z_idx], rc
        )

        rgb_cor[tlfl_sl == 1, z_idx] = TL_COLOR
        rgb_cor[tlfl_sl == 2, z_idx] = FL_COLOR

    axes[2].imshow(rgb_cor)
    axes[2].set_title("Coronal: TL/FL with anatomical aorta regions")

    patches = [
        mpatches.Patch(color=REGION_COLORS["upper"],  label="Upper aorta"),
        mpatches.Patch(color=REGION_COLORS["middle"], label="Middle aorta"),
        mpatches.Patch(color=REGION_COLORS["lower"],  label="Lower aorta"),
        mpatches.Patch(color=TL_COLOR, label="True lumen (TL)"),
        mpatches.Patch(color=FL_COLOR, label="False lumen (FL)"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"  Saved 2D: {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",     type=str, required=True)
    parser.add_argument("--ct_dir",   type=str, default="C:/imageTBAD/imagesTr")
    parser.add_argument("--tlfl_file",  type=str, required=True)
    parser.add_argument(
        "--aorta_file",
        type=str,
        required=True,
        help="Corrected aorta segmentation in same space as CT",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="C:/TBAD_Pipeline/outputs/reports",
    )
    parser.add_argument(
        "--min_component_voxels",
        type=int,
        default=200,
        help="Remove TL/FL connected components smaller than this after aorta gating.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_dir = Path("C:/TBAD_Pipeline/outputs/figures/fusion")
    fig_dir.mkdir(parents=True, exist_ok=True)

    ct_path = Path(args.ct_dir) / f"{args.case}.nii.gz"

    ct, spacing = load_nii(ct_path)

    tlfl, _ = load_nii(args.tlfl_file)
    tlfl = tlfl.astype(np.uint8)

    aorta, _ = load_nii(args.aorta_file)
    aorta = (aorta > 0).astype(np.uint8)

    if aorta.shape != tlfl.shape:
        raise ValueError(
            f"Shape mismatch: aorta={aorta.shape} tlfl={tlfl.shape}. "
            f"Reorient/resample aorta to CT space first."
        )

    if aorta.shape != ct.shape:
        raise ValueError(
            f"Shape mismatch: ct={ct.shape} aorta={aorta.shape}."
        )

    # ── Fix 1: aorta-gate TL/FL ──
    before_tl = int((tlfl == 1).sum())
    before_fl = int((tlfl == 2).sum())

    tlfl[aorta == 0] = 0

    after_gate_tl = int((tlfl == 1).sum())
    after_gate_fl = int((tlfl == 2).sum())

    before_total   = before_tl + before_fl
    after_gate_total = after_gate_tl + after_gate_fl

    kept_fraction    = after_gate_total / before_total if before_total > 0 else 0.0
    removed_fraction = 1.0 - kept_fraction if before_total > 0 else 0.0
    low_quality_prediction = kept_fraction < 0.10

    print(f"CT    : {ct.shape}  spacing={tuple(round(float(s), 2) for s in spacing)}")
    print(f"Aorta : {int(aorta.sum())} voxels")
    print(f"TL/FL before gate : TL={before_tl}  FL={before_fl}")
    print(
        f"TL/FL after  gate : TL={after_gate_tl}  FL={after_gate_fl}  "
        f"(removed {before_tl - after_gate_tl} TL, {before_fl - after_gate_fl} FL outside aorta)"
    )
    print(f"TL/FL kept inside aorta fraction   : {kept_fraction:.4f}")
    print(f"TL/FL removed outside aorta fraction: {removed_fraction:.4f}")

    if low_quality_prediction:
        print("WARNING: >90% of TL/FL prediction was outside the aorta.")
        print("         TL/FL model output should be treated as LOW QUALITY for this case.")

    # ── Step 3: connected-component cleanup after aorta gating ──
    tl_before_cc = int((tlfl == 1).sum())
    fl_before_cc = int((tlfl == 2).sum())

    tl_clean = keep_components_above_threshold(
        tlfl == 1,
        min_voxels=args.min_component_voxels,
    )
    fl_clean = keep_components_above_threshold(
        tlfl == 2,
        min_voxels=args.min_component_voxels,
    )

    tlfl[:] = 0
    tlfl[tl_clean] = 1
    tlfl[fl_clean] = 2

    tl_after_cc = int((tlfl == 1).sum())
    fl_after_cc = int((tlfl == 2).sum())

    print(
        f"After component cleanup "
        f"(min_component_voxels={args.min_component_voxels}): "
        f"TL {tl_before_cc}->{tl_after_cc}, FL {fl_before_cc}->{fl_after_cc}"
    )

    # Visualization windowing
    vmin = -200 if ct.min() < 0 else 824
    vmax =  600 if ct.min() < 0 else 1624

    # ── Fix 2: split anatomical regions by aorta extent ──
    bounds = split_by_aorta(aorta)

    if bounds is None:
        print("No aorta found — cannot create anatomical regions.")
        return

    z_min, z1, z2, z_max = bounds

    print(f"\nAorta Z extent: {z_min} -> {z_max}")
    print(f"  Upper  : z {z2} -> {z_max}")
    print(f"  Middle : z {z1} -> {z2}")
    print(f"  Lower  : z {z_min} -> {z1}")

    regions_stats = {}

    for rname, zs, ze in [
        ("upper",  z2,   z_max + 1),
        ("middle", z1,   z2),
        ("lower",  z_min, z1),
    ]:
        stats = compute_stats(tlfl, zs, ze, spacing)
        regions_stats[rname] = stats

        print(f"\n  {rname.capitalize()} aorta:")
        print(
            f"    TL: {stats['TL_voxels']} voxels ({stats['TL_volume_ml']} ml) "
            f"{'PRESENT' if stats['TL_present'] else 'absent'}"
        )
        print(
            f"    FL: {stats['FL_voxels']} voxels ({stats['FL_volume_ml']} ml) "
            f"{'PRESENT' if stats['FL_present'] else 'absent'}"
        )

    total_tl   = int((tlfl == 1).sum())
    total_fl   = int((tlfl == 2).sum())
    total_lumen = total_tl + total_fl
    fl_ratio    = total_fl / (total_lumen + 1e-8)

    # Candidate TBAD = enough TL and FL survive inside the aorta
    # QC pass       = enough raw TL/FL prediction was inside the aorta
    candidate_tbad = (
        total_tl > 1000 and
        total_fl > 1000 and
        fl_ratio > 0.05
    )

    qc_pass    = kept_fraction >= 0.10
    dissection = candidate_tbad and qc_pass

    if candidate_tbad and qc_pass:
        status = "TBAD DETECTED"
    elif candidate_tbad and not qc_pass:
        status = "TBAD suspected - LOW QC"
    elif total_tl > 1000 and total_fl <= 1000:
        status = "Indeterminate: TL only / FL insufficient"
        if not qc_pass:
            status += " - LOW QC"
    elif total_lumen > 0:
        status = "Indeterminate: weak TL/FL signal"
        if not qc_pass:
            status += " - LOW QC"
    else:
        status = "No TL/FL detected"

    summary_parts = []

    for rname, stats in regions_stats.items():
        if stats["TL_present"] or stats["FL_present"]:
            parts = []
            if stats["TL_present"]:
                parts.append(f"TL ({stats['TL_volume_ml']} ml)")
            if stats["FL_present"]:
                parts.append(f"FL ({stats['FL_volume_ml']} ml)")
            summary_parts.append(f"{rname.capitalize()}: {', '.join(parts)}")

    summary  = status + ". "
    summary += " | ".join(summary_parts)

    report = {
        "case_id":             args.case,
        "status":              status,
        "dissection_detected": dissection,
        "candidate_tbad":      candidate_tbad,
        "qc_pass":             qc_pass,
        "fl_ratio":            fl_ratio,
        "regions_defined_by":  "aorta_z_extent",
        "regions":             regions_stats,
        "aorta_z_extent": {
            "z_min":            z_min,
            "z_lower_middle":   z1,
            "z_middle_upper":   z2,
            "z_max":            z_max,
        },
        "total_TL_volume_ml":  voxels_to_ml(total_tl, spacing),
        "total_FL_volume_ml":  voxels_to_ml(total_fl, spacing),
        "tlfl_gated_by_aorta": True,
        "tlfl_voxels_removed_by_gate": {
            "TL": before_tl - after_gate_tl,
            "FL": before_fl - after_gate_fl,
        },
        "tlfl_quality_control": {
            "before_gate_total_voxels":      before_total,
            "after_gate_total_voxels":       after_gate_total,
            "kept_inside_aorta_fraction":    kept_fraction,
            "removed_outside_aorta_fraction": removed_fraction,
            "low_quality_prediction":        low_quality_prediction,
        },
        "component_cleanup": {
            "enabled":                       True,
            "min_component_voxels":          args.min_component_voxels,
            "TL_before_cleanup_voxels":      tl_before_cc,
            "FL_before_cleanup_voxels":      fl_before_cc,
            "TL_after_cleanup_voxels":       tl_after_cc,
            "FL_after_cleanup_voxels":       fl_after_cc,
            "TL_removed_by_cleanup_voxels":  tl_before_cc - tl_after_cc,
            "FL_removed_by_cleanup_voxels":  fl_before_cc - fl_after_cc,
        },
        "summary": summary,
    }

    print(f"\n{'=' * 60}")
    print(f"  STATUS: {status}")
    print(f"  Candidate TBAD   : {candidate_tbad}")
    print(f"  dissection_detected: {dissection}")
    print(f"  FL ratio         : {fl_ratio:.4f}")
    print(f"  QC pass          : {qc_pass}")
    print(
        f"  QC: kept_inside_aorta_fraction={kept_fraction:.4f}, "
        f"low_quality_prediction={low_quality_prediction}"
    )
    print(
        f"  Cleanup: TL {tl_before_cc}->{tl_after_cc}, "
        f"FL {fl_before_cc}->{fl_after_cc}"
    )
    print(f"  Summary: {summary}")
    print(f"{'=' * 60}")

    def to_py(obj):
        if isinstance(obj, dict):       return {k: to_py(v) for k, v in obj.items()}
        if isinstance(obj, list):       return [to_py(v) for v in obj]
        if isinstance(obj, np.bool_):   return bool(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating):return round(float(obj), 4)
        if isinstance(obj, float):      return round(obj, 4)
        return obj

    json_path = out_dir / f"{args.case}_report_v2.json"

    with open(json_path, "w") as f:
        json.dump(to_py(report), f, indent=2)

    print(f"\n  Report: {json_path}")

    make_2d_figure(
        ct, tlfl, aorta, bounds, regions_stats,
        args.case, dissection, report,
        fig_dir / f"{args.case}_fusion_v2.png",
        vmin, vmax,
    )


if __name__ == "__main__":
    main()