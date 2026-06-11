import numpy as np
import nibabel as nib
from pathlib import Path

TBAD_DIR    = Path(r"C:\imageTBAD\imagesTr")
SEG24_DIR   = Path(r"C:\nnUNet\nnUNet_raw\Dataset501_AortaSeg24\labelsTr")

print("="*60)
print("TBAD CASES — Z extent (physical mm) and slice count")
print("="*60)
tbad_z_extents = []
for f in sorted(TBAD_DIR.glob("*.nii.gz"))[:20]:  # sample 20
    nii  = nib.load(str(f))
    z_mm = nii.shape[2] * abs(nii.header.get_zooms()[2])
    tbad_z_extents.append(z_mm)
    print(f"{f.name:<30} slices={nii.shape[2]}  z_spacing={nii.header.get_zooms()[2]:.2f}mm  z_extent={z_mm:.1f}mm")

print(f"\nTBAD median Z extent : {np.median(tbad_z_extents):.1f} mm")
print(f"TBAD min/max Z extent: {np.min(tbad_z_extents):.1f} / {np.max(tbad_z_extents):.1f} mm")

print("\n" + "="*60)
print("AortaSeg24 — Z extent and zone Z-coverage")
print("="*60)
seg24_z_extents = []
for f in sorted(SEG24_DIR.glob("*.nii.gz"))[:10]:  # sample 10
    nii  = nib.load(str(f))
    data = np.asarray(nii.dataobj)
    z_mm = nii.shape[2] * abs(nii.header.get_zooms()[2])
    seg24_z_extents.append(z_mm)

    # Z range per zone
    zone_ranges = {}
    for z in range(23, 0, -1):
        zs = np.where(data == z)[2]
        if len(zs):
            zone_ranges[z] = (int(zs.min()), int(zs.max()))

    print(f"\n{f.name}  slices={nii.shape[2]}  z_extent={z_mm:.1f}mm")
    for zone, (zmin, zmax) in sorted(zone_ranges.items()):
        print(f"  zone {zone:2d}: slices {zmin:3d}-{zmax:3d}")

print(f"\nAortaSeg24 median Z extent: {np.median(seg24_z_extents):.1f} mm")
print(f"AortaSeg24 min/max        : {np.min(seg24_z_extents):.1f} / {np.max(seg24_z_extents):.1f} mm")
print("\nDone.")