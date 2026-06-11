"""
Crops AortaSeg24 labels to top 350mm (removes zones 20-23 / iliacs)
and creates matching binary inputs.
Output: C:\AortaZone_Project\data\aortaseg24_cropped\inputs + targets
"""
import numpy as np
import nibabel as nib
from pathlib import Path

LABEL_DIR  = Path(r"C:\nnUNet\nnUNet_raw\Dataset501_AortaSeg24\labelsTr")
INPUT_DIR  = Path(r"C:\AortaZone_Project\data\aortaseg24_cropped\inputs")
TARGET_DIR = Path(r"C:\AortaZone_Project\data\aortaseg24_cropped\targets")

CROP_MM    = 350.0   # keep top N mm of aorta (thoracic coverage)
REMAP_ABOVE = 19     # zones 20-23 → 0 (excluded)

files = sorted(LABEL_DIR.glob("*.nii.gz"))
print(f"Found {len(files)} cases\n")

skipped = []
for i, lf in enumerate(files):
    nii    = nib.load(str(lf))
    data   = np.asarray(nii.dataobj, dtype=np.uint8)
    zspacing = float(abs(nii.header.get_zooms()[2]))
    z_total  = data.shape[2]
    z_mm     = z_total * zspacing

    # Find topmost foreground slice (superior end)
    fg_slices = np.where(data > 0)[2]
    if len(fg_slices) == 0:
        skipped.append(lf.name)
        continue
    z_top = int(fg_slices.max())

    # Crop window: from z_top downward by CROP_MM
    n_slices  = int(CROP_MM / zspacing)
    z_start   = max(0, z_top - n_slices)
    cropped   = data[:, :, z_start:z_top + 1]

    # Zero out zones above 19
    cropped[cropped > REMAP_ABOVE] = 0

    zones_present = sorted(np.unique(cropped[cropped > 0]).tolist())

    # Save target (cropped 19L)
    new_affine = nii.affine.copy()
    new_affine[2, 3] += z_start * zspacing
    target_nii = nib.Nifti1Image(cropped, new_affine, nii.header)
    target_nii.set_data_dtype(np.uint8)
    nib.save(target_nii, str(TARGET_DIR / lf.name))

    # Save input (binary)
    binary     = (cropped > 0).astype(np.uint8)
    binary_nii = nib.Nifti1Image(binary, new_affine, nii.header)
    binary_nii.set_data_dtype(np.uint8)
    nib.save(binary_nii, str(INPUT_DIR / lf.name))

    print(f"[{i+1:3d}/100] {lf.name:<30} orig={z_total}sl  cropped={cropped.shape[2]}sl  zones={zones_present}")

print(f"\nDone. Skipped: {skipped if skipped else 'none'}")
print(f"Inputs  → {INPUT_DIR}")
print(f"Targets → {TARGET_DIR}")