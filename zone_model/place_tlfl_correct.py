"""
Places Dataset505 cropped TL/FL predictions back into full volume space
using the same ROI bounding box used during gt_cropped generation.
Output: C:\TBAD_Pipeline\outputs\predictions\tlfl_505_{CASE}_fullvol.nii.gz
"""
import nibabel as nib
import numpy as np
from pathlib import Path

MARGIN = 20
CASES = ["002","031","050","068","079","085","150"]

ROI_DIR  = Path(r"C:\imageTBAD\roi_labelsTr")
FULL_DIR = Path(r"C:\imageTBAD\imagesTr")
PRED_DIR = Path(r"C:\nnUNet\tlfl_predictions_505")
OUT_DIR  = Path(r"C:\TBAD_Pipeline\outputs\predictions")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for num in CASES:
    case = f"case_{num}"
    
    # Load full volume for shape/affine
    full_nii = nib.load(str(FULL_DIR / f"case_{num}.nii.gz"))
    full_shape = full_nii.shape
    
    # Recover bbox from ROI
    roi = np.asarray(nib.load(str(ROI_DIR / f"case_{num}.nii.gz")).dataobj)
    coords = np.argwhere(roi > 0)
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    s = roi.shape
    z0,y0,x0 = [max(0, int(mn[i])-MARGIN) for i in range(3)]
    z1,y1,x1 = [min(s[i], int(mx[i])+MARGIN) for i in range(3)]
    
    # Load cropped prediction
    pred = np.asarray(nib.load(str(PRED_DIR / f"{case}.nii.gz")).dataobj).astype(np.uint8)
    
    # Place into full volume
    full_pred = np.zeros(full_shape, dtype=np.uint8)
    full_pred[z0:z1, y0:y1, x0:x1] = pred[:z1-z0, :y1-y0, :x1-x0]
    
    out_path = OUT_DIR / f"tlfl_505_case{num}_fullvol.nii.gz"
    nib.save(nib.Nifti1Image(full_pred, full_nii.affine), str(out_path))
    
    tl = int((full_pred==1).sum())
    fl = int((full_pred==2).sum())
    print(f"{case}: placed into {full_shape} | TL={tl:,} FL={fl:,} -> {out_path.name}")

print("\nDone. Now update 06_inference_tbad_v2.py to use tlfl_505_caseXXX_fullvol.nii.gz")
