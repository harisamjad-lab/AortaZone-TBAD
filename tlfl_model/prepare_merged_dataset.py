"""
Merges ImageTBAD (100 cases) + figshare (40 cases) into a single
nnU-Net-ready dataset: Dataset505_TLFLMerged

ImageTBAD: already 1.0mm isotropic, use gt_cropped directly
Figshare:  native spacing, resample to 1.0mm isotropic first

Output: C:\nnUNet\nnUNet_raw\Dataset505_TLFLMerged\
  imagesTr\   -> case_XXX_0000.nii.gz
  labelsTr\   -> case_XXX.nii.gz
  dataset.json
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import zoom
import json, shutil

# Paths
TBAD_IMG = Path(r"C:\imageTBAD\gt_cropped_imagesTr")
TBAD_LBL = Path(r"C:\imageTBAD\gt_cropped_labelsTr")
FIG_IMG  = Path(r"C:\Aortic_dissection_data\dataset_cropped\imagesTr")
FIG_LBL  = Path(r"C:\Aortic_dissection_data\dataset_cropped\labelsTr")
OUT      = Path(r"C:\nnUNet\nnUNet_raw\Dataset505_TLFLMerged")
OUT_IMG  = OUT / "imagesTr"
OUT_LBL  = OUT / "labelsTr"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

TARGET = np.array([1.0, 1.0, 1.0])

def resample_to_1mm(data, spacing, order):
    factors = spacing / TARGET
    new_shape = np.round(np.array(data.shape) * factors).astype(int)
    zooms = new_shape / np.array(data.shape)
    return zoom(data, zooms, order=order)

case_id = 1

# --- ImageTBAD (already 1.0mm, just copy + rename) ---
print("=== ImageTBAD cases ===")
tbad_cases = sorted(TBAD_IMG.glob("*.nii.gz"))
print(f"Found {len(tbad_cases)} ImageTBAD cases")
for p in tbad_cases:
    name = f"case_{case_id:03d}"
    shutil.copy(str(p), str(OUT_IMG / f"{name}_0000.nii.gz"))
    lbl_p = TBAD_LBL / p.name
    if lbl_p.exists():
        # verify labels
        lbl = np.asarray(nib.load(str(lbl_p)).dataobj)
        unique = np.unique(lbl).tolist()
        # keep only TL(1) and FL(2), drop FLT(3) if present
        if 3 in unique:
            lbl[lbl == 3] = 0
        nib.save(nib.Nifti1Image(lbl.astype(np.uint8), nib.load(str(lbl_p)).affine),
                 str(OUT_LBL / f"{name}.nii.gz"))
        print(f"  {name} <- {p.name} | labels={np.unique(lbl).tolist()}")
    case_id += 1

# --- Figshare (resample to 1.0mm) ---
print(f"\n=== Figshare cases ===")
fig_cases = sorted(FIG_IMG.glob("*.nii.gz"))
print(f"Found {len(fig_cases)} figshare cases")
for p in fig_cases:
    name = f"case_{case_id:03d}"
    nii_img = nib.load(str(p))
    nii_lbl = nib.load(str(FIG_LBL / p.name))
    spacing = np.array(nii_img.header.get_zooms()[:3])

    img_data = nii_img.get_fdata().astype(np.float32)
    lbl_data = np.asarray(nii_lbl.dataobj).astype(np.uint8)

    # resample
    img_r = resample_to_1mm(img_data, spacing, order=3)
    lbl_r = resample_to_1mm(lbl_data, spacing, order=0)

    # verify labels ? figshare uses 1=TL, 2=FL only
    unique = np.unique(lbl_r).tolist()

    new_affine = nii_img.affine.copy()
    for i in range(3):
        new_affine[i, i] = np.sign(new_affine[i, i]) * 1.0

    nib.save(nib.Nifti1Image(img_r, new_affine), str(OUT_IMG / f"{name}_0000.nii.gz"))
    nib.save(nib.Nifti1Image(lbl_r.astype(np.uint8), new_affine), str(OUT_LBL / f"{name}.nii.gz"))
    print(f"  {name} <- {p.name} | {img_data.shape}{spacing} -> {img_r.shape}[1.0] labels={unique}")
    case_id += 1

total = case_id - 1
print(f"\nTotal cases: {total}")

# --- dataset.json ---
dataset = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "TrueLumen": 1, "FalseLumen": 2},
    "numTraining": total,
    "file_ending": ".nii.gz",
    "name": "Dataset505_TLFLMerged",
    "description": "ImageTBAD (100) + figshare (40) merged, 1mm isotropic, TL/FL labels"
}
with open(str(OUT / "dataset.json"), "w") as f:
    json.dump(dataset, f, indent=2)
print(f"Saved dataset.json -> {OUT}")
print("\nNext: nnUNetv2_plan_and_preprocess -d 505 -c 3d_fullres")
