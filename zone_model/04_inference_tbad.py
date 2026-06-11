"""
04_inference_tbad.py
Pipeline: Model B (TL/FL) + AVT -> Union binary -> Morph -> Z-crop -> Model A -> 19 zones

Input to Model A is union of TL/FL and AVT masks (plain float32 0/1).
AVT adds branch structure; TL/FL ensures full lumen coverage including FL.
"""
import json
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
import sys

CASE = sys.argv[1] if len(sys.argv) > 1 else "case150"
CASE_NUM = CASE[4:]  # "150" from "case150"

TLFL_PATH = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz")
AVT_PATH  = Path(rf"C:\TBAD_Pipeline\outputs\predictions\avt_on_imagetbad_corrected\case_{CASE_NUM}.nii.gz")
OUT_DIR   = Path(r"C:\AortaZone_Project\outputs\predictions") / CASE
OUT_DIR.mkdir(parents=True, exist_ok=True)

MORPH_CLOSE_MM = 5.0
FG_MARGIN      = 20

print(f"\n{'='*50}")
print(f"Processing: {CASE}")
print(f"{'='*50}")

# Step 1: Load TL/FL
print(f"\nStep 1: Loading TL/FL prediction...")
tlfl_nii = nib.load(str(TLFL_PATH))
tlfl     = np.asarray(tlfl_nii.dataobj, dtype=np.uint8)
spacing  = tlfl_nii.header.get_zooms()[:3]
affine   = tlfl_nii.affine
print(f"  Shape: {tlfl.shape}  Spacing: {spacing}")
print(f"  TL: {int(np.sum(tlfl==1)):,}  FL: {int(np.sum(tlfl==2)):,}")

# Step 2: Load AVT and compute union with TL/FL
print(f"\nStep 2: Loading AVT mask and computing union with TL/FL...")
avt_nii = nib.load(str(AVT_PATH))
avt     = np.asarray(avt_nii.dataobj, dtype=np.uint8)
tlfl_binary = (tlfl > 0)
avt_binary  = (avt > 0)
binary = (tlfl_binary | avt_binary).astype(np.uint8)
print(f"  TL/FL voxels: {int(tlfl_binary.sum()):,}")
print(f"  AVT voxels:   {int(avt_binary.sum()):,}")
print(f"  Union voxels: {int(binary.sum()):,}")

# Step 3: Morphological cleanup
print(f"\nStep 3: Morphological cleanup (closing={MORPH_CLOSE_MM}mm)...")
struct  = ndimage.generate_binary_structure(3, 1)
radius  = int(MORPH_CLOSE_MM / float(spacing[0]))
binary_closed = ndimage.binary_closing(
    binary, structure=ndimage.iterate_structure(struct, radius)
).astype(np.uint8)

labeled, n_comp = ndimage.label(binary_closed)
print(f"  Found {n_comp} components ? keeping largest")
if n_comp > 1:
    comp_sizes   = ndimage.sum(binary_closed, labeled, range(1, n_comp+1))
    binary_clean = (labeled == np.argmax(comp_sizes)+1).astype(np.uint8)
else:
    binary_clean = binary_closed

for z in range(binary_clean.shape[2]):
    binary_clean[:, :, z] = ndimage.binary_fill_holes(binary_clean[:, :, z])

clean_n = int(np.sum(binary_clean))
in_n    = int(np.sum(binary))
print(f"  Clean mask: {clean_n:,} voxels (delta {clean_n - in_n:+,})")
nib.save(nib.Nifti1Image(binary_clean.astype(np.uint8), affine, tlfl_nii.header),
         str(OUT_DIR / f"{CASE}_binary_clean.nii.gz"))

# Step 4: Z-only crop
print(f"\nStep 4: Z-axis crop (keeping full 512x512 XY)...")
fg    = np.where(binary_clean > 0)
z_min = max(0, int(fg[2].min()) - FG_MARGIN)
z_max = min(binary_clean.shape[2], int(fg[2].max()) + FG_MARGIN)

binary_cropped = binary_clean[:, :, z_min:z_max]
print(f"  {binary_clean.shape} -> {binary_cropped.shape}  Z[{z_min}:{z_max}]")

crop_info = {
    "x_min": 0, "x_max": binary_clean.shape[0],
    "y_min": 0, "y_max": binary_clean.shape[1],
    "z_min": z_min, "z_max": z_max,
    "orig_shape": list(binary_clean.shape)
}
with open(str(OUT_DIR / "crop_info.json"), "w") as f:
    json.dump(crop_info, f, indent=2)

# Step 5: Save nnU-Net input (plain 0/1 float32, no normalization)
print(f"\nStep 5: Saving nnU-Net input (plain 0/1 float32)...")
nnunet_in_dir  = OUT_DIR / "nnunet_input"
nnunet_out_dir = OUT_DIR / "nnunet_output"
nnunet_in_dir.mkdir(exist_ok=True)
nnunet_out_dir.mkdir(exist_ok=True)

crop_affine         = affine.copy()
crop_affine[:3, 3] += affine[:3, :3] @ np.array([0, 0, z_min])

data_out = binary_cropped.astype(np.float32)
img_nii  = nib.Nifti1Image(data_out, crop_affine, tlfl_nii.header)
img_nii.set_data_dtype(np.float32)
nib.save(img_nii, str(nnunet_in_dir / f"{CASE}_0000.nii.gz"))
print(f"  Saved: {nnunet_in_dir / f'{CASE}_0000.nii.gz'}")
print(f"  Shape: {data_out.shape}  unique: {np.unique(data_out).tolist()}")

print(f"\nReady for inference. Run:")
print(f"  C:\\nnUNet\\venv\\Scripts\\nnUNetv2_predict -i \"{nnunet_in_dir}\" -o \"{nnunet_out_dir}\" -d 503 -c 3d_fullres -p nnUNetResEncUNetMPlans -f 0 -npp 1 -nps 1")
