"""
visualize_pipeline_steps.py
Shows every intermediate step of the pipeline for one case.

Steps shown:
  1. TL/FL segmentation (raw model output)
  2. AVT outer wall mask
  3. Union binary mask (TL+FL+AVT merged)
  4. Morphological cleanup (after closing + hole fill)
  5. Z-position channel (the V2 novelty)
  6. 2-channel input side by side
  7. Final zone predictions
  8. Zone + TL/FL overlay (clinical result)

Usage: python scripts\visualize_pipeline_steps.py case068
"""

import sys
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from scipy import ndimage

CASE = sys.argv[1] if len(sys.argv) > 1 else "case068"
NUM  = CASE[4:]

# Paths
TLFL_PATH  = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz")
AVT_PATH   = Path(rf"C:\TBAD_Pipeline\outputs\predictions\avt_on_imagetbad_corrected\case_{NUM}.nii.gz")
ZONE_PATH  = Path(rf"C:\AortaZone_Project_v2\outputs\predictions\{CASE}\nnunet_output\{CASE}.nii.gz")
CH0_PATH   = Path(rf"C:\AortaZone_Project_v2\outputs\predictions\{CASE}\nnunet_input\{CASE}_0000.nii.gz")
OUT_DIR    = Path(rf"C:\AortaZone_Project_v2\outputs\figures\{CASE}\pipeline_steps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ZONE_NAMES = {
    1:"Aortic Root",2:"Ascending",3:"Prox Arch",4:"Mid Arch",5:"Dist Arch",
    6:"Brachio",7:"R Subclavian",8:"R Carotid",9:"Desc Thoracic",
    10:"L Carotid",11:"L Subclavian",12:"Prox Desc",13:"Mid Desc",
    14:"Dist Desc",15:"Celiac",16:"SMA",17:"Infrarenal"
}
ZONE_COLORS = {
    1:"#FF4444",2:"#44FF44",3:"#FFD700",4:"#4169E1",5:"#FF8C00",
    6:"#9400D3",7:"#00CED1",8:"#FF69B4",9:"#ADFF2F",10:"#FFB6C1",
    11:"#20B2AA",12:"#BA55D3",13:"#DAA520",14:"#FF6347",15:"#DC143C",
    16:"#00FF7F",17:"#FF1493"
}

def pts(mask, color, name, n=8000, visible=True):
    x, y, z = np.where(mask)
    if not len(x):
        return None
    idx = np.random.choice(len(x), min(n, len(x)), replace=False)
    return go.Scatter3d(
        x=x[idx], y=y[idx], z=z[idx], mode="markers", name=name,
        marker=dict(size=2, color=color, opacity=0.7),
        visible=True if visible else "legendonly"
    )

def save(fig, title, fname):
    fig.update_layout(
        title=dict(text=title, font=dict(color="white")),
        paper_bgcolor="black", font_color="white",
        legend=dict(bgcolor="black", itemclick="toggle", itemdoubleclick="toggleothers"),
        scene=dict(bgcolor="black",
                   xaxis=dict(color="white"),
                   yaxis=dict(color="white"),
                   zaxis=dict(color="white"))
    )
    path = OUT_DIR / fname
    fig.write_html(str(path))
    print(f"  Saved: {path.name}")

print(f"Loading data for {CASE}...")
tlfl = np.asarray(nib.load(str(TLFL_PATH)).dataobj, dtype=np.uint8)
avt  = np.asarray(nib.load(str(AVT_PATH)).dataobj,  dtype=np.uint8)
zones = np.asarray(nib.load(str(ZONE_PATH)).dataobj)
ch0  = nib.load(str(CH0_PATH)).get_fdata().astype(np.float32)

# Recompute intermediate steps
print("Recomputing pipeline intermediates...")
tlfl_bin  = (tlfl > 0).astype(np.uint8)
avt_bin   = (avt  > 0).astype(np.uint8)
union     = (tlfl_bin | avt_bin).astype(np.uint8)

struct = ndimage.generate_binary_structure(3, 1)
closed = ndimage.binary_closing(union, structure=ndimage.iterate_structure(struct, 5)).astype(np.uint8)
labeled, n = ndimage.label(closed)
if n > 1:
    sizes = ndimage.sum(closed, labeled, range(1, n+1))
    clean = (labeled == np.argmax(sizes)+1).astype(np.uint8)
else:
    clean = closed
for z in range(clean.shape[2]):
    clean[:,:,z] = ndimage.binary_fill_holes(clean[:,:,z])

# Z-position channel
z_channel = np.zeros_like(ch0)
fg_slices = np.where(ch0.sum(axis=(0,1)) > 0)[0]
if len(fg_slices) > 1:
    zlo, zhi = fg_slices.min(), fg_slices.max()
    for zi in fg_slices:
        norm = (zi - zlo) / (zhi - zlo)
        z_channel[:,:,zi] = ch0[:,:,zi] * norm

print("\n--- Generating step visualizations ---")

# STEP 1: TL/FL
print("Step 1: TL/FL segmentation...")
fig = go.Figure()
fig.add_trace(pts(tlfl==1, "#00BFFF", "True Lumen"))
fig.add_trace(pts(tlfl==2, "#FF8C00", "False Lumen"))
save(fig, f"{CASE} ? Step 1: TL/FL segmentation (raw model output)", "step1_tlfl.html")

# STEP 2: AVT
print("Step 2: AVT outer wall...")
fig = go.Figure()
fig.add_trace(pts(avt_bin>0, "#AAAAAA", "AVT outer wall"))
save(fig, f"{CASE} ? Step 2: AVT outer aorta mask", "step2_avt.html")

# STEP 3: Union
print("Step 3: Union mask...")
fig = go.Figure()
fig.add_trace(pts(tlfl_bin>0, "#00BFFF", "TL+FL"))
fig.add_trace(pts(avt_bin>0,  "#AAAAAA", "AVT (added branches)", visible=False))
fig.add_trace(pts(union>0,    "#FFFFFF", "Union TL+FL+AVT", visible=False))
save(fig, f"{CASE} ? Step 3: Union binary mask (toggle to compare)", "step3_union.html")

# STEP 4: Morphological cleanup
print("Step 4: Morphological cleanup...")
fig = go.Figure()
fig.add_trace(pts(union>0, "#888888", "Before cleanup", visible=False))
fig.add_trace(pts(clean>0, "#FFFFFF", "After closing + hole fill"))
save(fig, f"{CASE} ? Step 4: Morphological cleanup", "step4_clean.html")

# STEP 5: Z-position channel
print("Step 5: Z-position channel...")
fig = go.Figure()
x, y, z = np.where(z_channel > 0)
if len(x):
    n = min(12000, len(x))
    idx = np.random.choice(len(x), n, replace=False)
    zvals = z_channel[x[idx], y[idx], z[idx]]
    import plotly.express as px
    colors = px.colors.sample_colorscale("RdYlGn_r", zvals.tolist())
    fig.add_trace(go.Scatter3d(
        x=x[idx], y=y[idx], z=z[idx], mode="markers",
        name="Z position (green=top, red=bottom)",
        marker=dict(size=2, color=zvals, colorscale="RdYlGn_r",
                    colorbar=dict(title="Z pos", tickvals=[0,0.5,1],
                                  ticktext=["top","mid","bottom"],
                                  titlefont=dict(color="white"),
                                  tickfont=dict(color="white")),
                    opacity=0.8)
    ))
save(fig, f"{CASE} ? Step 5: Z-position channel (green=arch top, red=infrarenal)", "step5_zchannel.html")

# STEP 6: Both input channels side by side
print("Step 6: 2-channel input...")
fig = go.Figure()
fig.add_trace(pts(ch0>0, "#FFFFFF", "Ch0: binary mask (0/1)"))
# Z channel with color
x2, y2, z2 = np.where(z_channel > 0)
if len(x2):
    n2 = min(8000, len(x2))
    idx2 = np.random.choice(len(x2), n2, replace=False)
    zv2 = z_channel[x2[idx2], y2[idx2], z2[idx2]]
    fig.add_trace(go.Scatter3d(
        x=x2[idx2], y=y2[idx2], z=z2[idx2], mode="markers",
        name="Ch1: Z-position (toggle me)",
        visible="legendonly",
        marker=dict(size=2, color=zv2, colorscale="RdYlGn_r", opacity=0.8)
    ))
save(fig, f"{CASE} ? Step 6: 2-channel model input (toggle channels)", "step6_input_channels.html")

# STEP 7: Zone prediction only
print("Step 7: Zone predictions...")
fig = go.Figure()
for z_id in sorted(np.unique(zones[zones>0]).tolist()):
    t = pts(zones==z_id, ZONE_COLORS.get(z_id,"#FFF"),
            f"Z{z_id} {ZONE_NAMES.get(z_id,'')}", n=5000, visible="legendonly")
    if t: fig.add_trace(t)
save(fig, f"{CASE} ? Step 7: Zone predictions (toggle zones)", "step7_zones.html")

# STEP 8: Full clinical result
print("Step 8: Clinical result (zones + TL/FL)...")
fig = go.Figure()
fig.add_trace(pts(tlfl==1, "#00BFFF", "True Lumen", n=5000))
fig.add_trace(pts(tlfl==2, "#FF8C00", "False Lumen", n=5000))
for z_id in sorted(np.unique(zones[zones>0]).tolist()):
    t = pts(zones==z_id, ZONE_COLORS.get(z_id,"#FFF"),
            f"Z{z_id} {ZONE_NAMES.get(z_id,'')}", n=4000, visible="legendonly")
    if t: fig.add_trace(t)
save(fig, f"{CASE} ? Step 8: Final ? zones + TL/FL (clinical result)", "step8_clinical.html")

print(f"\nAll steps saved to: {OUT_DIR}")
print("\nOpen order for presentation:")
for i, name in enumerate([
    "step1_tlfl.html", "step2_avt.html", "step3_union.html",
    "step4_clean.html", "step5_zchannel.html", "step6_input_channels.html",
    "step7_zones.html", "step8_clinical.html"
], 1):
    print(f"  {i}. {name}")
