import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

CASE = "085"

tlfl  = np.asarray(nib.load(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_case{CASE}_fullvol.nii.gz").dataobj)
pred_a = np.asarray(nib.load(rf"C:\nnUNet\predictions_tbad_v2_clean\imagetbad_{CASE}.nii.gz").dataobj)
pred_b = np.asarray(nib.load(rf"C:\AortaZone_Project\outputs\predictions\case{CASE}\nnunet_output\case{CASE}.nii.gz").dataobj)

COLORS_A = {1:"#FF4444",2:"#44FF44",3:"#FFD700",4:"#4169E1",5:"#FF8C00",
            6:"#9400D3",7:"#00CED1",8:"#FF69B4",9:"#ADFF2F",10:"#FFB6C1",
            11:"#20B2AA",12:"#BA55D3",13:"#DAA520",14:"#FF6347",15:"#DC143C",
            16:"#00FF7F",17:"#FF1493",18:"#FF8800",19:"#0088FF"}
COLORS_B = COLORS_A.copy()

NAMES_A = {1:"Z1 Root",2:"Z2 Ascending",3:"Z3 AscMid",4:"Z4 PrxArch",5:"Z5 MidArch",
           6:"Z6 DistArch",7:"Z7 DescThor1",8:"Z8 DescThor2",9:"Z9 DescThor3",
           10:"Z10 DescThor4",11:"Z11 DescThor5",12:"Z12 Celiac",13:"Z13 SMA",
           14:"Z14 RenalL",15:"Z15 RenalR",18:"Z18 Infrarenal",19:"Z19 IliacL"}
NAMES_B = {1:"Z1 Root",2:"Z2 Ascending",3:"Z3 PrxArch",4:"Z4 MidArch",5:"Z5 DistArch",
           6:"Z6 Brachio",7:"Z7 RSubclav",8:"Z8 RCarotid",9:"Z9 DescThor",
           10:"Z10 LCarotid",11:"Z11 LSubclav",12:"Z12 PrxDesc",13:"Z13 MidDesc"}

def pts(arr, n=5000):
    x,y,z = np.where(arr)
    if len(x)==0: return [],[],[]
    idx = np.random.choice(len(x), min(n,len(x)), replace=False)
    return x[idx].tolist(), y[idx].tolist(), z[idx].tolist()

# ?? Figure 1: Model A (CTA?zones) ????????????????????????????????????
fig_a = go.Figure()
x,y,z = pts(tlfl==1); fig_a.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name="True Lumen",marker=dict(size=2,color="#00BFFF",opacity=0.5)))
x,y,z = pts(tlfl==2); fig_a.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name="False Lumen",marker=dict(size=2,color="#FF8C00",opacity=0.5)))
for zid in sorted(np.unique(pred_a[pred_a>0]).tolist()):
    x,y,z = pts(pred_a==zid, 4000)
    fig_a.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name=NAMES_A.get(int(zid),f"Z{zid}"),
        visible="legendonly",marker=dict(size=2,color=COLORS_A.get(int(zid),"#FFF"),opacity=0.85)))
fig_a.update_layout(title=f"case{CASE} ? Model A: CTA ? 23 Zones (nnU-Net Dataset501)",
    paper_bgcolor="black",font_color="white",
    legend=dict(bgcolor="black",itemclick="toggle",itemdoubleclick="toggleothers"),
    scene=dict(bgcolor="black",xaxis=dict(color="white"),yaxis=dict(color="white"),zaxis=dict(color="white")))

# ?? Figure 2: Model B (Binary?zones) ?????????????????????????????????
fig_b = go.Figure()
x,y,z = pts(tlfl==1); fig_b.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name="True Lumen",marker=dict(size=2,color="#00BFFF",opacity=0.5)))
x,y,z = pts(tlfl==2); fig_b.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name="False Lumen",marker=dict(size=2,color="#FF8C00",opacity=0.5)))
for zid in sorted(np.unique(pred_b[pred_b>0]).tolist()):
    x,y,z = pts(pred_b==zid, 4000)
    fig_b.add_trace(go.Scatter3d(x=x,y=y,z=z,mode="markers",name=NAMES_B.get(int(zid),f"Z{zid}"),
        visible="legendonly",marker=dict(size=2,color=COLORS_B.get(int(zid),"#FFF"),opacity=0.85)))
fig_b.update_layout(title=f"case{CASE} ? Model B: Binary Mask ? 19 Zones (nnU-Net Dataset503)",
    paper_bgcolor="black",font_color="white",
    legend=dict(bgcolor="black",itemclick="toggle",itemdoubleclick="toggleothers"),
    scene=dict(bgcolor="black",xaxis=dict(color="white"),yaxis=dict(color="white"),zaxis=dict(color="white")))

out = Path(r"C:\AortaZone_Project\outputs\figures\comparison")
out.mkdir(parents=True, exist_ok=True)
fig_a.write_html(str(out / f"case{CASE}_modelA_CTA.html"))
fig_b.write_html(str(out / f"case{CASE}_modelB_binary.html"))
print("Saved:")
print(f"  {out}/case{CASE}_modelA_CTA.html")
print(f"  {out}/case{CASE}_modelB_binary.html")
print(f"\nModel A zones: {np.unique(pred_a[pred_a>0]).tolist()}")
print(f"Model B zones: {np.unique(pred_b[pred_b>0]).tolist()}")
