"""
Re-crops figshare cases (101-140) in Dataset505 using tight label bbox + 20mm margin
to match ImageTBAD gt_cropped sizing (~150-250 per axis).
"""
import nibabel as nib
import numpy as np
from pathlib import Path

OUT_IMG = Path(r"C:\nnUNet\nnUNet_raw\Dataset505_TLFLMerged\imagesTr")
OUT_LBL = Path(r"C:\nnUNet\nnUNet_raw\Dataset505_TLFLMerged\labelsTr")
MARGIN = 20

def tight_crop(img, lbl, margin):
    coords = np.argwhere(lbl > 0)
    if not len(coords):
        return img, lbl
    mn = coords.min(axis=0)
    mx = coords.max(axis=0) + 1
    s = img.shape
    z0,y0,x0 = [max(0, int(mn[i])-margin) for i in range(3)]
    z1,y1,x1 = [min(s[i], int(mx[i])+margin) for i in range(3)]
    return img[z0:z1, y0:y1, x0:x1], lbl[z0:z1, y0:y1, x0:x1]

print("Re-cropping figshare cases 101-140...")
for case_id in range(101, 141):
    name = f"case_{case_id:03d}"
    ip = OUT_IMG / f"{name}_0000.nii.gz"
    lp = OUT_LBL / f"{name}.nii.gz"
    if not ip.exists():
        continue

    nii_i = nib.load(str(ip))
    nii_l = nib.load(str(lp))
    img = nii_i.get_fdata().astype(np.float32)
    lbl = np.asarray(nii_l.dataobj).astype(np.uint8)

    old_shape = img.shape
    img_c, lbl_c = tight_crop(img, lbl, MARGIN)

    nib.save(nib.Nifti1Image(img_c, nii_i.affine), str(ip))
    nib.save(nib.Nifti1Image(lbl_c, nii_l.affine), str(lp))
    print(f"  {name}: {old_shape} -> {img_c.shape} labels={np.unique(lbl_c).tolist()}")

# Final size check
print("\nFinal volume stats:")
vols = []
for p in sorted(OUT_IMG.glob("*.nii.gz")):
    s = nib.load(str(p)).shape
    vols.append(s[0]*s[1]*s[2])
vols = np.array(vols)
print(f"  min={vols.min():,}  max={vols.max():,}  mean={int(vols.mean()):,}  median={int(np.median(vols)):,}")
print(f"  Cases > 10M voxels: {(vols>10_000_000).sum()}")
print(f"  Cases > 20M voxels: {(vols>20_000_000).sum()}")
print("Done.")
