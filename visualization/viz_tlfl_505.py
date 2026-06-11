import sys
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

CASE = sys.argv[1] if len(sys.argv) > 1 else "case_002"
NUM = CASE.split("_")[-1]

new_pred = Path(rf"C:\nnUNet\tlfl_predictions_505\{CASE}.nii.gz")
old_pred = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_case{NUM}_fullvol.nii.gz")

print(f"Loading {CASE}...")
new = np.asarray(nib.load(str(new_pred)).dataobj)
print(f"  New: shape={new.shape} labels={np.unique(new).tolist()}")

def pts(mask, color, name, n=8000, visible=True):
    x,y,z = np.where(mask)
    if not len(x): return None
    idx = np.random.choice(len(x), min(n,len(x)), replace=False)
    return go.Scatter3d(x=x[idx],y=y[idx],z=z[idx],mode="markers",
                        name=name,visible=visible,
                        marker=dict(size=2,color=color,opacity=0.7))

fig = go.Figure()
fig.add_trace(pts(new==1, "#00BFFF", "TL (new nnUNet)"))
fig.add_trace(pts(new==2, "#FF8C00", "FL (new nnUNet)"))

if old_pred.exists():
    old = np.asarray(nib.load(str(old_pred)).dataobj)
    fig.add_trace(pts(old==1, "#00FF7F", "TL (old MONAI)", visible="legendonly"))
    fig.add_trace(pts(old==2, "#FF4444", "FL (old MONAI)", visible="legendonly"))
    print(f"  Old MONAI loaded for comparison")

fig.update_layout(
    title=f"{CASE} - TL/FL: new nnUNet vs old MONAI",
    paper_bgcolor="black", font_color="white",
    legend=dict(bgcolor="black", itemclick="toggle", itemdoubleclick="toggleothers"),
    scene=dict(bgcolor="black",
               xaxis=dict(color="white"),
               yaxis=dict(color="white"),
               zaxis=dict(color="white"))
)

out = Path(rf"C:\nnUNet\tlfl_predictions_505\{CASE}_viz.html")
fig.write_html(str(out))
print(f"Saved: {out}")
