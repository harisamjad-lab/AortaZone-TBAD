"""
06_inference_tbad_v2.py

Prepares 2-channel nnU-Net input for Dataset504 inference on TBAD cases.

Channel 0: binary mask (TL/FL union AVT), plain 0/1 float32
Channel 1: normalized Z position (0.0=top, 1.0=bottom of foreground)

Usage: python 06_inference_tbad_v2.py case085
"""

import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
import sys

CASE     = sys.argv[1] if len(sys.argv) > 1 else "case085"
CASE_NUM = CASE[4:]

TLFL_PATH = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_505_{CASE}_fullvol.nii.gz")
AVT_PATH  = Path(rf"C:\TBAD_Pipeline\outputs\predictions\avt_on_imagetbad_corrected\case_{CASE_NUM}.nii.gz")
OUT_DIR   = Path(r"C:\AortaZone_Project_v2\outputs\predictions") / CASE
OUT_DIR.mkdir(parents=True, exist_ok=True)

MORPH_CLOSE_MM = 5.0
FG_MARGIN      = 20

print(f"\n{'='*50}")
print(f"Processing: {CASE} (Dataset504 v2)")
print(f"{'='*50}")

# Load TL/FL
tlfl_nii = nib.load(str(TLFL_PATH))
tlfl     = np.asarray(tlfl_nii.dataobj, dtype=np.uint8)
spacing  = tlfl_nii.header.get_zooms()[:3]
affine   = tlfl_nii.affine
print(f"TL: {int((tlfl==1).sum()):,}  FL: {int((tlfl==2).sum()):,}")

# Union with AVT
avt    = np.asarray(nib.load(str(AVT_PATH)).dataobj, dtype=np.uint8)
binary = ((tlfl > 0) | (avt > 0)).astype(np.uint8)
print(f"Union voxels: {int(binary.sum()):,}")

# Morphological cleanup
struct        = ndimage.generate_binary_structure(3, 1)
radius        = int(MORPH_CLOSE_MM / float(spacing[0]))
binary_closed = ndimage.binary_closing(binary, structure=ndimage.iterate_structure(struct, radius)).astype(np.uint8)
labeled, n    = ndimage.label(binary_closed)
if n > 1:
    sizes        = ndimage.sum(binary_closed, labeled, range(1, n+1))
    binary_clean = (labeled == np.argmax(sizes)+1).astype(np.uint8)
else:
    binary_clean = binary_closed
for z in range(binary_clean.shape[2]):
    binary_clean[:,:,z] = ndimage.binary_fill_holes(binary_clean[:,:,z])
print(f"Clean mask: {int(binary_clean.sum()):,} voxels")

# Z crop
fg    = np.where(binary_clean > 0)
z_min = max(0, int(fg[2].min()) - FG_MARGIN)
z_max = min(binary_clean.shape[2], int(fg[2].max()) + FG_MARGIN)
binary_cropped = binary_clean[:, :, z_min:z_max]
print(f"Cropped: {binary_clean.shape} -> {binary_cropped.shape}  Z[{z_min}:{z_max}]")

# Save crop info
crop_info = {"x_min":0,"x_max":binary_clean.shape[0],
             "y_min":0,"y_max":binary_clean.shape[1],
             "z_min":z_min,"z_max":z_max,
             "orig_shape":list(binary_clean.shape)}
with open(str(OUT_DIR / "crop_info.json"), "w") as f:
    json.dump(crop_info, f, indent=2)

# Channel 0: binary 0/1
ch0 = binary_cropped.astype(np.float32)

# Channel 1: normalized Z position
ch1 = np.zeros_like(ch0)
fg_slices = np.where(ch0.sum(axis=(0,1)) > 0)[0]
if len(fg_slices) > 1:
    zlo, zhi = fg_slices.min(), fg_slices.max()
    for z in fg_slices:
        norm_z = (z - zlo) / (zhi - zlo)
        ch1[:,:,z] = ch0[:,:,z] * norm_z
print(f"Z channel: [{ch1[ch0>0].min():.3f}, {ch1[ch0>0].max():.3f}]")

# Save both channels
nnunet_in  = OUT_DIR / "nnunet_input"
nnunet_out = OUT_DIR / "nnunet_output"
nnunet_in.mkdir(exist_ok=True)
nnunet_out.mkdir(exist_ok=True)

crop_affine         = affine.copy()
crop_affine[:3, 3] += affine[:3, :3] @ np.array([0, 0, z_min])

for ch, name in [(ch0, "_0000"), (ch1, "_0001")]:
    img = nib.Nifti1Image(ch, crop_affine, tlfl_nii.header)
    img.set_data_dtype(np.float32)
    nib.save(img, str(nnunet_in / f"{CASE}{name}.nii.gz"))
    print(f"Saved {CASE}{name}.nii.gz  unique={np.unique(ch)[:4].tolist()}")

print(f"\nReady. Run inference:")
print(f"  nnUNetv2_predict -i \"{nnunet_in}\" -o \"{nnunet_out}\" -d 504 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 0 -npp 1 -nps 1")
