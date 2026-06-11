import sys
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from pathlib import Path
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
import networkx as nx

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

print("Skeletonizing...")
skel = skeletonize(clean.astype(bool))
coords = np.argwhere(skel)
print(f"Skeleton points: {len(coords):,}")

print("Building graph...")
tree = cKDTree(coords)
pairs = tree.query_pairs(r=1.8)
G = nx.Graph()
G.add_nodes_from(range(len(coords)))
for i, j in pairs:
    G.add_edge(i, j, weight=float(np.linalg.norm(coords[i]-coords[j])))

largest = max(nx.connected_components(G), key=len)
G = G.subgraph(largest).copy()
sub_nodes = list(largest)
print(f"Largest component: {len(sub_nodes):,} nodes")

sub_coords = coords[sub_nodes]
root_node = sub_nodes[int(np.argmax(sub_coords[:, 2]))]

print("Finding main trunk (longest path)...")
lengths = nx.single_source_dijkstra_path_length(G, root_node, weight="weight")
farthest = max(lengths, key=lengths.get)
trunk_nodes = nx.dijkstra_path(G, root_node, farthest, weight="weight")
trunk = coords[trunk_nodes]
print(f"Trunk: {len(trunk)} points, dist={lengths[farthest]:.1f}")

deltas = np.linalg.norm(np.diff(trunk, axis=0), axis=1)
arclen = np.concatenate([[0], np.cumsum(deltas)])
arclen_norm = arclen / arclen.max()

print("Mapping voxels...")
mask_coords = np.argwhere(clean > 0)
ttree = cKDTree(trunk)
_, nearest = ttree.query(mask_coords, k=1)
z_arc = np.zeros_like(clean, dtype=np.float32)
z_arc[mask_coords[:,0], mask_coords[:,1], mask_coords[:,2]] = arclen_norm[nearest]

def sample(coords, vals, n=12000):
    if len(coords) > n:
        idx = np.random.choice(len(coords), n, replace=False)
        return coords[idx], vals[idx]
    return coords, vals

fig = go.Figure()
xa,ya,za = np.where(z_arc>0); va = z_arc[xa,ya,za]
ca,vav = sample(np.column_stack([xa,ya,za]), va)
fig.add_trace(go.Scatter3d(x=ca[:,0],y=ca[:,1],z=ca[:,2],mode="markers",
    name="arc-length",
    marker=dict(size=2,color=vav,colorscale="RdYlGn_r",opacity=0.8,
                colorbar=dict(title=dict(text="root->distal")))))
fig.add_trace(go.Scatter3d(x=trunk[:,0],y=trunk[:,1],z=trunk[:,2],mode="lines",
    name="main trunk",line=dict(color="cyan",width=5)))

fig.update_layout(title=f"{CASE} - Clean arc-length (graph longest-path)",
    paper_bgcolor="black",font_color="white",
    legend=dict(bgcolor="black",itemclick="toggle",itemdoubleclick="toggleothers"),
    scene=dict(bgcolor="black",xaxis=dict(color="white"),
               yaxis=dict(color="white"),zaxis=dict(color="white")))

out = Path(rf"C:\AortaZone_Project_v2\outputs\figures\{CASE}\centerline_clean.html")
out.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(str(out))
print(f"\nSaved: {out}")
