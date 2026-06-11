"""
Extracts aorta centerline from union mask, computes arc-length (0=root, 1=distal),
visualizes vs old linear-Z. VALIDATION step before retraining.
Usage: python viz_centerline_arclength.py case068
"""
import sys
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

CASE = sys.argv[1] if len(sys.argv) > 1 else "case068"
NUM = CASE[4:]

TLFL = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{CASE}_fullvol.nii.gz")
AVT = Path(rf"C:\TBAD_Pipeline\outputs\predictions\avt_on_imagetbad_corrected\case_{NUM}.nii.gz")

print(f"Loading {CASE}...")
tlfl = np.asarray(nib.load(str(TLFL)).dataobj, dtype=np.uint8)
avt = np.asarray(nib.load(str(AVT)).dataobj, dtype=np.uint8)

union = ((tlfl > 0) | (avt > 0)).astype(np.uint8)
struct = ndimage.generate_binary_structure(3, 1)
closed = ndimage.binary_closing(union, structure=ndimage.iterate_structure(struct, 5)).astype(np.uint8)
labeled, n = ndimage.label(closed)
if n > 1:
    sizes = ndimage.sum(closed, labeled, range(1, n+1))
    clean = (labeled == np.argmax(sizes)+1).astype(np.uint8)
else:
    clean = closed
for z in range(clean.shape[2]):
    clean[:, :, z] = ndimage.binary_fill_holes(clean[:, :, z])
print(f"Clean mask: {int(clean.sum()):,} voxels")

# OLD linear-Z
z_linear = np.zeros_like(clean, dtype=np.float32)
fg = np.where(clean.sum(axis=(0,1)) > 0)[0]
zlo, zhi = fg.min(), fg.max()
for zi in fg:
    z_linear[:, :, zi] = clean[:, :, zi] * (zi - zlo) / (zhi - zlo)

# NEW centerline arc-length
print("Skeletonizing...")
skel = skeletonize(clean.astype(bool))
skel_coords = np.argwhere(skel)
print(f"Centerline points: {len(skel_coords):,}")

start_idx = int(np.argmax(skel_coords[:, 2]))
tree = cKDTree(skel_coords)
visited = np.zeros(len(skel_coords), dtype=bool)
order = [start_idx]
visited[start_idx] = True
current = start_idx
for _ in range(len(skel_coords) - 1):
    dists, idxs = tree.query(skel_coords[current], k=min(15, len(skel_coords)))
    nxt = None
    for d, ii in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
        if not visited[ii]:
            nxt = int(ii); break
    if nxt is None:
        unvis = np.where(~visited)[0]
        if len(unvis) == 0: break
        dd = np.linalg.norm(skel_coords[unvis] - skel_coords[current], axis=1)
        nxt = int(unvis[np.argmin(dd)])
    order.append(nxt); visited[nxt] = True; current = nxt

ordered = skel_coords[order]
deltas = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
arclen = np.concatenate([[0], np.cumsum(deltas)])
arclen_norm = arclen / arclen.max()

print("Mapping voxels...")
mask_coords = np.argwhere(clean > 0)
ctree = cKDTree(ordered)
_, nearest = ctree.query(mask_coords, k=1)
z_arc = np.zeros_like(clean, dtype=np.float32)
z_arc[mask_coords[:,0], mask_coords[:,1], mask_coords[:,2]] = arclen_norm[nearest]

def sample(coords, vals, n=12000):
    if len(coords) > n:
        idx = np.random.choice(len(coords), n, replace=False)
        return coords[idx], vals[idx]
    return coords, vals

fig = go.Figure()
xl,yl,zl = np.where(z_linear>0); vl = z_linear[xl,yl,zl]
cl,vlv = sample(np.column_stack([xl,yl,zl]), vl)
fig.add_trace(go.Scatter3d(x=cl[:,0],y=cl[:,1],z=cl[:,2],mode="markers",
    name="OLD linear-Z",visible=True,
    marker=dict(size=2,color=vlv,colorscale="RdYlGn_r",opacity=0.8,
                colorbar=dict(title=dict(text="position")))))
xa,ya,za = np.where(z_arc>0); va = z_arc[xa,ya,za]
ca,vav = sample(np.column_stack([xa,ya,za]), va)
fig.add_trace(go.Scatter3d(x=ca[:,0],y=ca[:,1],z=ca[:,2],mode="markers",
    name="NEW arc-length",visible="legendonly",
    marker=dict(size=2,color=vav,colorscale="RdYlGn_r",opacity=0.8)))
fig.add_trace(go.Scatter3d(x=ordered[:,0],y=ordered[:,1],z=ordered[:,2],mode="lines",
    name="centerline",line=dict(color="cyan",width=4)))

fig.update_layout(title=f"{CASE} - OLD linear-Z vs NEW arc-length",
    paper_bgcolor="black",font_color="white",
    legend=dict(bgcolor="black",itemclick="toggle",itemdoubleclick="toggleothers"),
    scene=dict(bgcolor="black",xaxis=dict(color="white"),
               yaxis=dict(color="white"),zaxis=dict(color="white")))

out = Path(rf"C:\AortaZone_Project_v2\outputs\figures\{CASE}\centerline_arclength.html")
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out))
print(f"\nSaved: {out}")
