"""
view_cta_3d.py
3D interactive visualization of ImageTBAD CTA scan with zone overlay.

Usage:
  python scripts\view_cta_3d.py case085
  python scripts\view_cta_3d.py case068 --with_zones
"""

import sys
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from pathlib import Path
from skimage import measure

CASE = sys.argv[1] if len(sys.argv) > 1 else "case085"
WITH_ZONES = "--with_zones" in sys.argv
CASE_NUM = CASE[4:]

CTA_PATH  = Path(rf"C:\imageTBAD\imagesTr\case_{CASE_NUM}.nii.gz")
TLFL_PATH = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz")
ZONE_PATH = Path(rf"C:\AortaZone_Project_v2\outputs\predictions\{CASE}\nnunet_output\{CASE}.nii.gz")

ZONE_NAMES = {
    1:"Aortic Root",2:"Ascending",3:"Prox Arch",4:"Mid Arch",5:"Dist Arch",
    6:"Brachiocephalic",7:"R Subclavian",8:"R Carotid",9:"Desc Thoracic",
    10:"L Carotid",11:"L Subclavian",12:"Prox Desc",13:"Mid Desc",
    14:"Dist Desc",15:"Celiac",16:"SMA",17:"Infrarenal"
}
ZONE_COLORS = {
    1:"#FF4444",2:"#44FF44",3:"#FFD700",4:"#4169E1",5:"#FF8C00",
    6:"#9400D3",7:"#00CED1",8:"#FF69B4",9:"#ADFF2F",10:"#FFB6C1",
    11:"#20B2AA",12:"#BA55D3",13:"#DAA520",14:"#FF6347",15:"#DC143C",
    16:"#00FF7F",17:"#FF1493"
}

print(f"Loading {CASE}...")

# Load CTA
cta_nii = nib.load(str(CTA_PATH))
cta = cta_nii.get_fdata()
spacing = cta_nii.header.get_zooms()[:3]
print(f"  CTA shape: {cta.shape}  spacing: {spacing}")

# Load TL/FL
tlfl = np.asarray(nib.load(str(TLFL_PATH)).dataobj)

fig = go.Figure()

def add_surface(mask, color, name, opacity=0.6, step=4):
    """Extract isosurface from binary mask and add to figure."""
    mask_ds = mask[::step, ::step, ::step]
    if mask_ds.sum() < 100:
        print(f"  Skipping {name} ? too few voxels")
        return
    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask_ds.astype(np.float32), level=0.5,
            spacing=(spacing[0]*step, spacing[1]*step, spacing[2]*step)
        )
        fig.add_trace(go.Mesh3d(
            x=verts[:,0], y=verts[:,1], z=verts[:,2],
            i=faces[:,0], j=faces[:,1], k=faces[:,2],
            color=color, opacity=opacity, name=name,
            showlegend=True, hoverinfo="name"
        ))
        print(f"  Added {name}: {int(mask.sum()):,} voxels")
    except Exception as e:
        print(f"  Failed {name}: {e}")

# CTA volume rendering (bone window for aortic context)
print("Adding CTA context (aortic window HU -200 to 800)...")
aorta_window = np.clip(cta, 100, 500)
aorta_context = (aorta_window > 250).astype(np.uint8)
add_surface(aorta_context, "#CCCCCC", "CTA context", opacity=0.08, step=6)

# TL and FL surfaces
print("Adding TL/FL surfaces...")
add_surface(tlfl==1, "#00BFFF", "True Lumen", opacity=0.55, step=3)
add_surface(tlfl==2, "#FF8C00", "False Lumen", opacity=0.55, step=3)

# Zone surfaces
if WITH_ZONES and ZONE_PATH.exists():
    print("Adding zone surfaces...")
    zones = np.asarray(nib.load(str(ZONE_PATH)).dataobj)
    for z_id in sorted(np.unique(zones[zones>0]).tolist()):
        add_surface(
            zones==z_id,
            ZONE_COLORS.get(z_id, "#FFFFFF"),
            f"Z{z_id} {ZONE_NAMES.get(z_id,'')}",
            opacity=0.7, step=3
        )
elif WITH_ZONES:
    print(f"  Zone file not found: {ZONE_PATH}")

# Layout
title = f"{CASE} ? CTA + TL/FL"
if WITH_ZONES:
    title += " + Zone labels (V2)"

fig.update_layout(
    title=title,
    scene=dict(
        bgcolor="black",
        xaxis=dict(color="white", title="X (mm)"),
        yaxis=dict(color="white", title="Y (mm)"),
        zaxis=dict(color="white", title="Z (mm)"),
        aspectmode="data"
    ),
    paper_bgcolor="black",
    font_color="white",
    legend=dict(bgcolor="black", itemclick="toggle", itemdoubleclick="toggleothers")
)

out = Path(rf"C:\AortaZone_Project\outputs\figures\{CASE}\cta_3d{'_zones' if WITH_ZONES else ''}.html")
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out))
print(f"\nSaved: {out}")
print("Open in browser to explore interactively.")
