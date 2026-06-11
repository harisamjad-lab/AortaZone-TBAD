import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from scipy import ndimage
from pathlib import Path

CASE = "case068"
NUM  = "068"

tlfl = np.asarray(nib.load(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz").dataobj, dtype=np.uint8)
avt  = np.asarray(nib.load(rf"C:\TBAD_Pipeline\outputs\predictions\avt_on_imagetbad_corrected\case_{NUM}.nii.gz").dataobj, dtype=np.uint8)

union = ((tlfl > 0) | (avt > 0)).astype(np.uint8)
struct = ndimage.generate_binary_structure(3, 1)
closed = ndimage.binary_closing(union, structure=ndimage.iterate_structure(struct, 5)).astype(np.uint8)
labeled, n = ndimage.label(closed)
clean = (labeled == np.argmax(ndimage.sum(closed, labeled, range(1, n+1)))+1).astype(np.uint8) if n > 1 else closed
for z in range(clean.shape[2]):
    clean[:,:,z] = ndimage.binary_fill_holes(clean[:,:,z])

z_channel = np.zeros_like(clean, dtype=np.float32)
fg = np.where(clean.sum(axis=(0,1)) > 0)[0]
zlo, zhi = fg.min(), fg.max()
for zi in fg:
    z_channel[:,:,zi] = clean[:,:,zi] * (zi - zlo) / (zhi - zlo)

x, y, z = np.where(z_channel > 0)
n = min(15000, len(x))
idx = np.random.choice(len(x), n, replace=False)
zvals = z_channel[x[idx], y[idx], z[idx]]

fig = go.Figure(go.Scatter3d(
    x=x[idx], y=y[idx], z=z[idx], mode="markers",
    marker=dict(
        size=2, color=zvals, colorscale="RdYlGn_r", opacity=0.85,
        colorbar=dict(
            title=dict(text="Z position"),
            tickvals=[0, 0.5, 1],
            ticktext=["top (arch)", "mid (thoracic)", "bottom (infrarenal)"],
            tickfont=dict(color="white")
        )
    ),
    name="Z position (green=arch top, red=infrarenal)"
))

fig.update_layout(
    title=f"{CASE} ? Z-position channel (V2 novelty)",
    paper_bgcolor="black", font_color="white",
    scene=dict(bgcolor="black",
               xaxis=dict(color="white"),
               yaxis=dict(color="white"),
               zaxis=dict(color="white"))
)

out = Path(rf"C:\AortaZone_Project_v2\outputs\figures\{CASE}\zchannel.html")
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out))
print("Saved:", out)
