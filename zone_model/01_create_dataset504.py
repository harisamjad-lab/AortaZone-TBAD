"""
01_create_dataset504.py

Creates Dataset504_AortaZoneV2 from Dataset503_AortaZone.

Channel 0 (_0000): binary mask (0/1) — same as Dataset503
Channel 1 (_0001): normalized Z position (0.0=top of aorta, 1.0=bottom)
                   0 for background voxels

Targets: copied directly from Dataset503 (same 19-zone labels)

Run from: C:\\AortaZone_Project_v2\\
"""

import nibabel as nib
import numpy as np
import shutil
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────
SRC_IMAGES  = Path(r"C:\nnUNet\nnUNet_raw\Dataset503_AortaZone\imagesTr")
SRC_LABELS  = Path(r"C:\nnUNet\nnUNet_raw\Dataset503_AortaZone\labelsTr")
DST_ROOT    = Path(r"C:\nnUNet\nnUNet_raw\Dataset504_AortaZoneV2")
DST_IMAGES  = DST_ROOT / "imagesTr"
DST_LABELS  = DST_ROOT / "labelsTr"

DST_IMAGES.mkdir(parents=True, exist_ok=True)
DST_LABELS.mkdir(parents=True, exist_ok=True)

# ── Process each case ─────────────────────────────────────────────────
cases = sorted(SRC_IMAGES.glob("*_0000.nii.gz"))
print(f"Found {len(cases)} cases in Dataset503")

for ch0_path in cases:
    case_name = ch0_path.name.replace("_0000.nii.gz", "")
    print(f"Processing {case_name}...", end=" ")

    # Load binary mask (channel 0)
    nii = nib.load(str(ch0_path))
    binary = nii.get_fdata().astype(np.float32)

    # Compute normalized Z position (channel 1)
    z_channel = np.zeros_like(binary, dtype=np.float32)
    fg_slices = np.where(binary.sum(axis=(0, 1)) > 0)[0]

    if len(fg_slices) > 1:
        z_min, z_max = fg_slices.min(), fg_slices.max()
        for z in fg_slices:
            norm_z = (z - z_min) / (z_max - z_min)  # 0.0 to 1.0
            z_channel[:, :, z] = binary[:, :, z] * norm_z
    else:
        print("WARNING: no foreground found, skipping z channel")

    # Save channel 0 (copy original)
    shutil.copy(str(ch0_path), str(DST_IMAGES / ch0_path.name))

    # Save channel 1
    ch1_path = DST_IMAGES / ch0_path.name.replace("_0000.nii.gz", "_0001.nii.gz")
    nib.save(nib.Nifti1Image(z_channel, nii.affine, nii.header), str(ch1_path))

    # Verify
    fg = int((binary > 0).sum())
    z_vals = z_channel[binary > 0]
    print(f"fg={fg:,}  z_range=[{z_vals.min():.3f}, {z_vals.max():.3f}]")

# ── Copy labels ───────────────────────────────────────────────────────
print("\nCopying labels...")
for lbl in sorted(SRC_LABELS.glob("*.nii.gz")):
    shutil.copy(str(lbl), str(DST_LABELS / lbl.name))
    print(f"  Copied {lbl.name}")

# ── Write dataset.json ────────────────────────────────────────────────
dataset_json = {
    "channel_names": {
        "0": "binary_mask",
        "1": "normalized_z_position"
    },
    "labels": {
        "background": 0,
        "zone_01_root": 1, "zone_02_ascending": 2, "zone_03_prox_arch": 3,
        "zone_04_mid_arch": 4, "zone_05_dist_arch": 5,
        "zone_06_brachio": 6, "zone_07_r_subclavian": 7, "zone_08_r_carotid": 8,
        "zone_09_desc_thoracic": 9, "zone_10_l_carotid": 10,
        "zone_11_l_subclavian": 11, "zone_12_prox_desc": 12,
        "zone_13_mid_desc": 13, "zone_14_dist_desc": 14,
        "zone_15_celiac": 15, "zone_16_sma": 16, "zone_17_infrarenal": 17,
        "zone_18_r_iliac": 18, "zone_19_l_iliac": 19
    },
    "numTraining": len(cases),
    "file_ending": ".nii.gz",
    "name": "Dataset504_AortaZoneV2",
    "description": "Binary mask + normalized Z position -> 19 anatomical zones"
}

with open(str(DST_ROOT / "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\nDone! Dataset504 created at: {DST_ROOT}")
print(f"Cases: {len(cases)}")
print(f"Channels per case: 2 (_0000 binary, _0001 z-position)")
print(f"\nNext: run 02_verify_dataset504.py")
