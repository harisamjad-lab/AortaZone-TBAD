"""
Sets up Dataset503_AortaZone for nnU-Net ResEnc training.
Input : binary masks  (C:\AortaZone_Project\data\aortaseg24_cropped\inputs)
Target: 19-zone labels(C:\AortaZone_Project\data\aortaseg24_cropped\targets)
Output: C:\nnUNet\nnUNet_raw\Dataset503_AortaZone\
"""
import json
import shutil
import numpy as np
import nibabel as nib
from pathlib import Path

INPUT_DIR  = Path(r"C:\AortaZone_Project\data\aortaseg24_cropped\inputs")
TARGET_DIR = Path(r"C:\AortaZone_Project\data\aortaseg24_cropped\targets")
NNUNET_RAW = Path(r"C:\nnUNet\nnUNet_raw\Dataset503_AortaZone")

IMG_DIR    = NNUNET_RAW / "imagesTr"
LBL_DIR    = NNUNET_RAW / "labelsTr"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

files = sorted(INPUT_DIR.glob("*.nii.gz"))
print(f"Found {len(files)} cases\n")

for i, f in enumerate(files):
    # nnU-Net naming: caseXXX_0000.nii.gz for images, caseXXX.nii.gz for labels
    case_id = f"case{i+1:03d}"

    # Image: binary mask as float32 (nnU-Net expects float images)
    nii  = nib.load(str(f))
    data = np.asarray(nii.dataobj, dtype=np.float32)
    img_nii = nib.Nifti1Image(data, nii.affine, nii.header)
    img_nii.set_data_dtype(np.float32)
    nib.save(img_nii, str(IMG_DIR / f"{case_id}_0000.nii.gz"))

    # Label: copy target directly
    shutil.copy2(str(TARGET_DIR / f.name), str(LBL_DIR / f"{case_id}.nii.gz"))

    print(f"[{i+1:3d}/{len(files)}] {f.name} → {case_id}")

# ── dataset.json ──────────────────────────────────────────────────────────────
# Determine all zone labels present
all_labels = {0: "background"}
for lf in sorted(LBL_DIR.glob("*.nii.gz"))[:5]:
    d = np.asarray(nib.load(str(lf)).dataobj, dtype=np.uint8)
    for z in np.unique(d[d > 0]).tolist():
        all_labels[int(z)] = f"zone_{z}"

dataset_json = {
    "channel_names": {"0": "binary_mask"},
    "labels": {v: k for k, v in all_labels.items()},
    "numTraining": len(files),
    "file_ending": ".nii.gz",
    "name": "Dataset503_AortaZone",
    "description": "Binary aorta mask to 19-zone anatomical labeling for TBAD",
    "reference": "AortaSeg24 cropped + binarized",
    "licence": "see AortaSeg24",
    "release": "1.0",
    "overwrite_image_reader_writer": "NibabelIOWithReorient"
}

with open(str(NNUNET_RAW / "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\ndataset.json saved")
print(f"\nDataset ready at: {NNUNET_RAW}")
print(f"Images : {len(list(IMG_DIR.glob('*.nii.gz')))} files")
print(f"Labels : {len(list(LBL_DIR.glob('*.nii.gz')))} files")
print(f"\nZones in dataset: {sorted(all_labels.keys())}")
print("\nNext: run nnU-Net fingerprint extraction (see instructions below)")
print("="*60)
print("ACTIVATE NNUNET ENV THEN RUN:")
print()
print("  $env:nnUNet_raw          = 'C:\\nnUNet\\nnUNet_raw'")
print("  $env:nnUNet_preprocessed = 'C:\\nnUNet\\nnUNet_preprocessed'")
print("  $env:nnUNet_results      = 'C:\\nnUNet\\nnUNet_results'")
print()
print("  nnUNetv2_plan_and_preprocess -d 503 -c 3d_fullres --verify_dataset_integrity")
print("="*60)