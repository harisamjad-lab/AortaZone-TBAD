import sys
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

CASE = sys.argv[1] if len(sys.argv) > 1 else "case068"

ZONE_NAMES = {
    1:"Root",2:"Ascending",3:"PrxArch",4:"MidArch",5:"DistArch",
    6:"Brachio",7:"RSubclav",8:"RCarotid",9:"DescThor",10:"LCarotid",
    11:"LSubclav",12:"PrxDesc",13:"MidDesc",14:"DistDesc",
    15:"Celiac",16:"SMA",17:"Infrarenal"
}
ZONE_COLORS = {
    1:"#FF4444",2:"#44FF44",3:"#FFD700",4:"#4169E1",5:"#FF8C00",
    6:"#9400D3",7:"#00CED1",8:"#FF69B4",9:"#ADFF2F",10:"#FFB6C1",
    11:"#20B2AA",12:"#BA55D3",13:"#DAA520",14:"#FF6347",15:"#DC143C",
    16:"#00FF7F",17:"#FF1493"
}

def pts(mask, color, name, n=5000, visible=True):
    x,y,z = np.where(mask)
    if not len(x): return None
    idx = np.random.choice(len(x), min(n,len(x)), replace=False)
    return go.Scatter3d(x=x[idx],y=y[idx],z=z[idx],mode="markers",
                        name=name, visible=visible,
                        marker=dict(size=2,color=color,opacity=0.7))

old_path = Path(rf"C:\AortaZone_Project_v2\outputs\predictions\{CASE}\nnunet_output\{CASE}.nii.gz")
new_path = Path(rf"C:\AortaZone_Project_v2\outputs\predictions\{CASE}\nnunet_output_505\{CASE}.nii.gz")
tlfl_path = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz")

old = np.asarray(nib.load(str(old_path)).dataobj)
new = np.asarray(nib.load(str(new_path)).dataobj)
tlfl = np.asarray(nib.load(str(tlfl_path)).dataobj)

fig = go.Figure()

# TL/FL always visible
fig.add_trace(pts(tlfl==1, "#00BFFF", "True Lumen"))
fig.add_trace(pts(tlfl==2, "#FF8C00", "False Lumen"))

# Old zone predictions (toggle off by default)
for z_id in sorted(np.unique(old[old>0]).tolist()):
    t = pts(old==z_id, ZONE_COLORS.get(z_id,"#FFF"),
            f"OLD Z{z_id} {ZONE_NAMES.get(z_id,'')}", visible="legendonly")
    if t: fig.add_trace(t)

# New zone predictions (toggle off by default)
for z_id in sorted(np.unique(new[new>0]).tolist()):
    t = pts(new==z_id, ZONE_COLORS.get(z_id,"#FFF"),
            f"NEW Z{z_id} {ZONE_NAMES.get(z_id,'')}", visible="legendonly")
    if t: fig.add_trace(t)

fig.update_layout(
    title=f"{CASE} ? Zone comparison: OLD (MONAI TL/FL) vs NEW (505 TL/FL)",
    paper_bgcolor="black", font_color="white",
    legend=dict(bgcolor="black", itemclick="toggle", itemdoubleclick="toggleothers"),
    scene=dict(bgcolor="black",
               xaxis=dict(color="white"),
               yaxis=dict(color="white"),
               zaxis=dict(color="white"))
)

out = Path(rf"C:\AortaZone_Project_v2\outputs\figures\{CASE}\zone_compare_505.html")
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out))
print(f"Saved: {out}")
