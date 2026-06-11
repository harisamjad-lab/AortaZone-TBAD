import os
import re
import numpy as np
import SimpleITK as sitk

raw_images_dir = "raw_images"
raw_labels_dir = "raw_labels"

tlf_images_dir = "imagesTr"
tlf_labels_dir = "labelsTr"

roi_images_dir = "roi_imagesTr"
roi_labels_dir = "roi_labelsTr"

os.makedirs(tlf_images_dir, exist_ok=True)
os.makedirs(tlf_labels_dir, exist_ok=True)
os.makedirs(roi_images_dir, exist_ok=True)
os.makedirs(roi_labels_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(raw_images_dir) if f.endswith(".nii.gz")])

def get_case_id(fname):
    m = re.match(r"(\d+)_image\.nii\.gz", fname)
    if m is None:
        raise ValueError(f"Unexpected filename: {fname}")
    return int(m.group(1))

for img_file in image_files:
    case_id = get_case_id(img_file)
    lbl_file = f"{case_id}_label.nii.gz"

    img_path = os.path.join(raw_images_dir, img_file)
    lbl_path = os.path.join(raw_labels_dir, lbl_file)

    if not os.path.exists(lbl_path):
        print(f"Skipping case {case_id}: missing label")
        continue

    # read
    img = sitk.ReadImage(img_path)
    lbl = sitk.ReadImage(lbl_path)

    img_np = sitk.GetArrayFromImage(img)
    lbl_np = sitk.GetArrayFromImage(lbl)

    # -----------------------------
    # ROI label: union(TL, FL, FLT)
    # -----------------------------
    roi_np = (lbl_np > 0).astype(np.uint8)

    roi_lbl = sitk.GetImageFromArray(roi_np)
    roi_lbl.CopyInformation(lbl)

    # -----------------------------
    # TL/FL label: ignore FLT
    # Assumed mapping:
    # 0 = bg, 1 = TL, 2 = FL, 3 = FLT
    # -----------------------------
    tlf_np = np.zeros_like(lbl_np, dtype=np.uint8)
    tlf_np[lbl_np == 1] = 1   # TL
    tlf_np[lbl_np == 2] = 2   # FL
    # lbl==3 (FLT) stays 0

    tlf_lbl = sitk.GetImageFromArray(tlf_np)
    tlf_lbl.CopyInformation(lbl)

    # -----------------------------
    # save with unified names
    # -----------------------------
    out_name = f"case_{case_id:03d}.nii.gz"

    sitk.WriteImage(img, os.path.join(tlf_images_dir, out_name))
    sitk.WriteImage(tlf_lbl, os.path.join(tlf_labels_dir, out_name))

    sitk.WriteImage(img, os.path.join(roi_images_dir, out_name))
    sitk.WriteImage(roi_lbl, os.path.join(roi_labels_dir, out_name))

    print(f"Processed case {case_id:03d}")

print("\nDone. Converted datasets created.")