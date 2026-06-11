"""
07_compare_v1_v2.py

Compare Dataset503 (binary only) vs Dataset504 (binary + Z position)
on the same TBAD case. Shows zones side by side in browser.

Usage: python 07_compare_v1_v2.py case085
"""

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import sys

CASE = sys.argv[1] if len(sys.argv) > 1 else "case085"
NUM  = CASE[4:]

tlfl  = np.asarray(nib.load(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz").dataobj)
v1    = np.asarray(nib.load(rf"C:\AortaZone_Project\outputs\predictions\{CASE}\nnunet_output\{CASE}.nii.gz").dataobj)
v2    = np.asarray(nib.load(rf"C:\AortaZone_Project_v2\outputs\predictions\{CASE}\nnunet_output\{CASE}.nii.gz").dataobj)

COLORS = {1:"#FF4444",2:"#44FF44",3:"#FFD700",4:"#4169E1",5:"#FF8C00",
          6:"#9400D3",7:"#00CED1",8:"#FF69B4",9:"#ADFF2F",10:"#FFB6C1",
          11:"#20B2AA",12:"#BA55D3",13:"#DAA520",14:"#FF6347",15:"#DC143C",
          16:"#00FF7F",17:"#FF1493"}
NAMES = {1:"Z1 Root",2:"Z2 Ascending",3:"Z3 PrxArch",4:"Z4 MidArch",5:"Z5 DistArch",
         6:"Z6 Brachio",7:"Z7 RSubclav",8:"Z8 RCarotid",9:"Z9 DescThor",
         10:"Z10 LCarotid",11:"Z11 LSubclav",12:"Z12 PrxDesc",13:"Z13 MidDesc",
         14:"Z14 DistDesc",15:"Z15 Celiac",16:"Z16 SMA",17:"Z17 Infrarenal"}

def pts(arr, n=5000):
    x,y,z = np.where(arr)
    if not len(x): return [],[],[]
    idx = np.random.choice(len(x), min(n,len(x)), replace=False)
    return x[idx].tolist(), y[idx].tolist(), z[idx].tolist()

def make_fig(pred, title):
    fig = go.Figure()
    for lbl, color, name in [(1,"#00BFFF","True Lumen"),(2,"#FF8C00","False Lumen")]:
        x,y,z = pts(tlfl==lbl)
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name=name,
            marker=dict(size=2,color=color,opacity=0.5)))
    for zid in sorted(np.unique(pred[pred>0]).tolist()):
        x,y,z = pts(pred==zid, 4000)
        fig.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",
            name=NAMES.get(zid,f"Z{zid}"), visible="legendonly",
            marker=dict(size=2,color=COLORS.get(zid,"#FFF"),opacity=0.9)))
    fig.update_layout(title=title, paper_bgcolor="black", font_color="white",
        legend=dict(bgcolor="black",itemclick="toggle",itemdoubleclick="toggleothers"),
        scene=dict(bgcolor="black",
                   xaxis=dict(color="white"),yaxis=dict(color="white"),zaxis=dict(color="white")))
    return fig

out = Path(rf"C:\AortaZone_Project_v2\outputs\figures\{CASE}")
out.mkdir(parents=True, exist_ok=True)

make_fig(v1, f"{CASE} — V1: Binary only (Dataset503)").write_html(str(out/f"{CASE}_v1_binary.html"))
make_fig(v2, f"{CASE} — V2: Binary + Z position (Dataset504)").write_html(str(out/f"{CASE}_v2_zreg.html"))

print(f"V1 zones ({len(np.unique(v1[v1>0]))}): {np.unique(v1[v1>0]).tolist()}")
print(f"V2 zones ({len(np.unique(v2[v2>0]))}): {np.unique(v2[v2>0]).tolist()}")
print(f"\nSaved to {out}")
print("Open both HTMLs side by side to compare")
