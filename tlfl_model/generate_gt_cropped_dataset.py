"""
generate_gt_cropped_dataset.py

Creates a clean cropped dataset for TL/FL training by using
ground-truth ROI labels (roi_labelsTr) to define crop bounding boxes.

Input:
  imagesTr/       -> CT volumes
  labelsTr/       -> TL/FL labels (0=bg, 1=TL, 2=FL, FLT=0)
  roi_labelsTr/   -> binary aorta ROI labels (0=bg, 1=aorta union)

Output:
  gt_cropped_imagesTr/  -> CT cropped around GT aorta ROI
  gt_cropped_labelsTr/  -> TL/FL labels cropped to same bbox

Usage:
  python generate_gt_cropped_dataset.py --base_dir C:/imageTBAD --margin 20
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path


def get_bbox(mask: np.ndarray, margin: int, shape: tuple):
    """
    Returns (z0,z1, y0,y1, x0,x1) bounding box of nonzero region
    with added margin, clamped to volume shape.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        print("  [WARN] ROI mask is empty — using full volume as fallback.")
        return (0, shape[0], 0, shape[1], 0, shape[2])

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1  # exclusive end

    z0 = max(0, int(z0) - margin)
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)
    z1 = min(shape[0], int(z1) + margin)
    y1 = min(shape[1], int(y1) + margin)
    x1 = min(shape[2], int(x1) + margin)

    return (z0, z1, y0, y1, x0, x1)


def process_case(case_name, images_dir, labels_dir, roi_labels_dir,
                 out_images_dir, out_labels_dir, margin):

    ct_path  = images_dir     / case_name
    lbl_path = labels_dir     / case_name
    roi_path = roi_labels_dir / case_name

    # existence checks
    missing = [str(p) for p in [ct_path, lbl_path, roi_path] if not p.exists()]
    if missing:
        print(f"  [SKIP] Missing: {missing}")
        return False

    # load
    ct_nib  = nib.load(str(ct_path))
    lbl_nib = nib.load(str(lbl_path))
    roi_nib = nib.load(str(roi_path))

    ct_data  = ct_nib.get_fdata(dtype=np.float32)
    lbl_data = lbl_nib.get_fdata(dtype=np.float32).astype(np.uint8)
    roi_data = roi_nib.get_fdata(dtype=np.float32).astype(np.uint8)

    if ct_data.shape != roi_data.shape:
        print(f"  [WARN] Shape mismatch CT={ct_data.shape} ROI={roi_data.shape} — skipping.")
        return False

    # compute bbox from GT ROI
    z0, z1, y0, y1, x0, x1 = get_bbox(roi_data, margin, roi_data.shape)

    # crop CT and TL/FL label with same bbox
    ct_crop  = ct_data [z0:z1, y0:y1, x0:x1]
    lbl_crop = lbl_data[z0:z1, y0:y1, x0:x1]

    # verify labels in crop
    unique_lbl = np.unique(lbl_crop).tolist()

    # save with original affine (approximate — voxel positions shift but spacing preserved)
    affine = ct_nib.affine
    nib.save(nib.Nifti1Image(ct_crop,  affine), str(out_images_dir / case_name))
    nib.save(nib.Nifti1Image(lbl_crop, affine), str(out_labels_dir / case_name))

    print(f"  OK  {case_name} | {ct_data.shape} -> {ct_crop.shape} | "
          f"bbox z[{z0}:{z1}] y[{y0}:{y1}] x[{x0}:{x1}] | labels={unique_lbl}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default="C:/imageTBAD",
                        help="Root directory of ImageTBAD dataset")
    parser.add_argument("--margin",   type=int, default=20,
                        help="Voxel margin around ROI bbox (default: 20)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    images_dir     = base_dir / "imagesTr"
    labels_dir     = base_dir / "labelsTr"
    roi_labels_dir = base_dir / "roi_labelsTr"
    out_images_dir = base_dir / "gt_cropped_imagesTr"
    out_labels_dir = base_dir / "gt_cropped_labelsTr"

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    cases = sorted([f.name for f in images_dir.glob("*.nii.gz")])
    print(f"{'='*60}")
    print(f"GT-Based Crop Generator")
    print(f"{'='*60}")
    print(f"Base dir : {base_dir}")
    print(f"Cases    : {len(cases)}")
    print(f"Margin   : {args.margin} voxels")
    print(f"Output CT: {out_images_dir}")
    print(f"Output LB: {out_labels_dir}")
    print(f"{'='*60}\n")

    success = 0
    skipped = 0
    for i, case_name in enumerate(cases):
        print(f"[{i+1:3d}/{len(cases)}] {case_name}")
        ok = process_case(
            case_name,
            images_dir, labels_dir, roi_labels_dir,
            out_images_dir, out_labels_dir,
            args.margin
        )
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\n{'='*60}")
    print(f"DONE: {success} cropped, {skipped} skipped, {len(cases)} total.")
    print(f"GT-cropped dataset ready:")
    print(f"  {out_images_dir}")
    print(f"  {out_labels_dir}")
    print(f"{'='*60}")
    print(f"\nNext step: verify with view_gt_cropped_sample.py, then run train_tl_fl_5fold.py")


if __name__ == "__main__":
    main()