"""
Replaces linear-Z channel (_0001) in Dataset504 training inputs
with centerline arc-length channel for all AortaSeg24 cases.
Output: C:\nnUNet\nnUNet_raw\Dataset506_AortaZoneV3\
"""
import numpy as np
import nibabel as nib
import networkx as nx
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from pathlib import Path
import shutil

SRC = Path(r"C:\nnUNet\nnUNet_raw\Dataset504_AortaZoneV2")
OUT = Path(r"C:\nnUNet\nnUNet_raw\Dataset506_AortaZoneV3")
OUT_IMG = OUT / "imagesTr"
OUT_LBL = OUT / "labelsTr"
OUT_IMG.mkdir(parents=True, exist_ok=True)
OUT_LBL.mkdir(parents=True, exist_ok=True)

def get_arclength_channel(binary_mask):
    struct = ndimage.generate_binary_structure(3,1)
    closed = ndimage.binary_closing(binary_mask, structure=ndimage.iterate_structure(struct,5)).astype(np.uint8)
    labeled, n = ndimage.label(closed)
    if n > 1:
        sizes = ndimage.sum(closed, labeled, range(1,n+1))
        clean = (labeled == np.argmax(sizes)+1).astype(np.uint8)
    else:
        clean = closed
    for z in range(clean.shape[2]):
        clean[:,:,z] = ndimage.binary_fill_holes(clean[:,:,z])

    skel = skeletonize(clean.astype(bool))
    coords = np.argwhere(skel)
    if len(coords) < 10:
        return None, clean

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=1.8)
    G = nx.Graph()
    G.add_nodes_from(range(len(coords)))
    for i,j in pairs:
        G.add_edge(i,j,weight=float(np.linalg.norm(coords[i]-coords[j])))

    largest = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest).copy()
    sub_nodes = list(largest)
    sub_coords = coords[sub_nodes]

    root_node = sub_nodes[int(np.argmax(sub_coords[:,2]))]
    lengths = nx.single_source_dijkstra_path_length(G, root_node, weight="weight")
    farthest = max(lengths, key=lengths.get)
    trunk_nodes = nx.dijkstra_path(G, root_node, farthest, weight="weight")
    trunk = coords[trunk_nodes]

    deltas = np.linalg.norm(np.diff(trunk, axis=0), axis=1)
    arclen = np.concatenate([[0], np.cumsum(deltas)])
    arclen_norm = arclen / arclen.max()

    mask_coords = np.argwhere(clean > 0)
    ttree = cKDTree(trunk)
    _, nearest = ttree.query(mask_coords, k=1)
    z_arc = np.zeros_like(clean, dtype=np.float32)
    z_arc[mask_coords[:,0], mask_coords[:,1], mask_coords[:,2]] = arclen_norm[nearest]
    return z_arc, clean

# Copy labels unchanged
for f in sorted((SRC/"labelsTr").glob("*.nii.gz")):
    shutil.copy(str(f), str(OUT_LBL/f.name))
print(f"Copied {len(list((OUT_LBL).glob('*.nii.gz')))} labels")

# Recompute channels
cases = sorted((SRC/"imagesTr").glob("*_0000.nii.gz"))
print(f"Processing {len(cases)} cases...")
ok, fail = 0, 0
for p in cases:
    name = p.name.replace("_0000.nii.gz","")
    nii = nib.load(str(p))
    binary = (nii.get_fdata() > 0.5).astype(np.uint8)

    z_arc, clean = get_arclength_channel(binary)

    # Copy binary channel unchanged
    shutil.copy(str(p), str(OUT_IMG/p.name))

    if z_arc is None:
        print(f"  WARN {name}: skeleton failed, using linear-Z fallback")
        old_ch1 = SRC/"imagesTr"/(p.name.replace("_0000","_0001"))
        shutil.copy(str(old_ch1), str(OUT_IMG/old_ch1.name))
        fail += 1
    else:
        out_ch1 = OUT_IMG/(p.name.replace("_0000","_0001"))
        nib.save(nib.Nifti1Image(z_arc, nii.affine), str(out_ch1))
        print(f"  OK {name}: arc-length range [{z_arc[clean>0].min():.3f}, {z_arc[clean>0].max():.3f}]")
        ok += 1

import json
ds = json.load(open(str(SRC/"dataset.json")))
ds["name"] = "Dataset506_AortaZoneV3"
ds["description"] = "AortaZone V3: binary + centerline arc-length channel (replaces linear-Z)"
json.dump(ds, open(str(OUT/"dataset.json"),"w"), indent=2)
print(f"\nDone: {ok} arc-length, {fail} fallback")
print(f"Dataset506 ready at: {OUT}")
print("Next: nnUNetv2_plan_and_preprocess -d 506 -c 3d_fullres -pl nnUNetPlannerResEncM")
