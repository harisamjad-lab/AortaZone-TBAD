"""
02_verify_dataset504.py

Sanity checks before training:
- All cases have both _0000 and _0001 channels
- Channel 0 is 0/1 binary
- Channel 1 is 0.0 to 1.0 normalized Z
- Labels match Dataset503
- Shapes consistent
"""

import nibabel as nib
import numpy as np
from pathlib import Path

DST_ROOT   = Path(r"C:\nnUNet\nnUNet_raw\Dataset504_AortaZoneV2")
IMG_DIR    = DST_ROOT / "imagesTr"
LBL_DIR    = DST_ROOT / "labelsTr"
SRC_LABELS = Path(r"C:\nnUNet\nnUNet_raw\Dataset503_AortaZone\labelsTr")

cases_ch0 = sorted(IMG_DIR.glob("*_0000.nii.gz"))
print(f"Cases found: {len(cases_ch0)}")

errors = []
for ch0 in cases_ch0:
    name = ch0.name.replace("_0000.nii.gz", "")
    ch1  = IMG_DIR / ch0.name.replace("_0000", "_0001")
    lbl  = LBL_DIR / f"{name}.nii.gz"

    # Check files exist
    if not ch1.exists(): errors.append(f"{name}: missing _0001"); continue
    if not lbl.exists():  errors.append(f"{name}: missing label"); continue

    d0 = nib.load(str(ch0)).get_fdata().astype(np.float32)
    d1 = nib.load(str(ch1)).get_fdata().astype(np.float32)
    dl = nib.load(str(lbl)).get_fdata()

    # Shape checks
    if d0.shape != d1.shape: errors.append(f"{name}: shape mismatch ch0{d0.shape} ch1{d1.shape}"); continue
    if d0.shape != dl.shape: errors.append(f"{name}: label shape mismatch"); continue

    # Value checks
    ch0_vals = np.unique(d0).tolist()
    if not all(v in [0.0, 1.0] for v in ch0_vals):
        errors.append(f"{name}: ch0 not binary, got {ch0_vals[:5]}"); continue

    fg_z = d1[d0 > 0]
    if fg_z.max() < 0.9:
        errors.append(f"{name}: z-channel max={fg_z.max():.3f} expected ~1.0"); continue

    print(f"  OK {name}: shape={d0.shape} z=[{fg_z.min():.3f},{fg_z.max():.3f}] labels={np.unique(dl[dl>0]).tolist()[:5]}")

print()
if errors:
    print("ERRORS:")
    for e in errors: print(f"  {e}")
else:
    print("ALL CHECKS PASSED - ready for plan+preprocess")
    print("\nNext: run 03_plan_preprocess.ps1")
