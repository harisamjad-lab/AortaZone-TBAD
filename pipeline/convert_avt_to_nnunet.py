"""
convert_avt_to_nnunet.py

Converts AVT dataset (.nrrd format) to nnU-Net v2 format.

Input structure:
  C:/ATV_DataSet/ATV_DataSet/img1/img1.nrrd          (CT)
  C:/ATV_DataSet/ATV_DataSet/img1/img1.seg.nrrd      (binary aorta mask)
  ... (img1 through img56, 56 cases total)

Output structure (nnU-Net v2):
  nnUNet_raw/Dataset502_AVT/
    imagesTr/
      avt_001_0000.nii.gz
      avt_002_0000.nii.gz
      ...
    labelsTr/
      avt_001.nii.gz
      avt_002.nii.gz
      ...
    dataset.json

Usage:
  python convert_avt_to_nnunet.py
"""

import argparse
import json
import shutil
import SimpleITK as sitk
from pathlib import Path


def convert_one(case_dir: Path, out_image: Path, out_label: Path):
    case_id = case_dir.name
    ct_path  = case_dir / f"{case_id}.nrrd"
    seg_path = case_dir / f"{case_id}.seg.nrrd"

    if not ct_path.exists() or not seg_path.exists():
        return False, f"missing files in {case_dir}"

    # Read CT and save as .nii.gz
    ct = sitk.ReadImage(str(ct_path))
    sitk.WriteImage(ct, str(out_image))

    # Read seg and save as .nii.gz, ensuring binary
    seg = sitk.ReadImage(str(seg_path))
    seg_arr = sitk.GetArrayFromImage(seg)
    seg_arr = (seg_arr > 0).astype("uint8")  # ensure clean binary
    seg_clean = sitk.GetImageFromArray(seg_arr)
    seg_clean.CopyInformation(seg)  # preserve spacing/origin/direction
    sitk.WriteImage(seg_clean, str(out_label))

    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avt_dir",      type=str,
                        default="C:/ATV_DataSet/ATV_DataSet")
    parser.add_argument("--nnunet_raw",   type=str,
                        default="C:/TBAD_Pipeline/data/AVT_nnunet/nnUNet_raw")
    parser.add_argument("--dataset_id",   type=int, default=502)
    parser.add_argument("--dataset_name", type=str, default="AVT")
    args = parser.parse_args()

    avt_dir = Path(args.avt_dir)
    out_dataset = Path(args.nnunet_raw) / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    images_tr   = out_dataset / "imagesTr"
    labels_tr   = out_dataset / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    case_dirs = sorted(
        [d for d in avt_dir.iterdir() if d.is_dir() and d.name.startswith("img")],
        key=lambda d: int(d.name.replace("img", ""))
    )

    print(f"\n{'='*70}")
    print(f"  AVT -> nnU-Net Conversion")
    print(f"{'='*70}")
    print(f"  Source : {avt_dir}")
    print(f"  Target : {out_dataset}")
    print(f"  Cases  : {len(case_dirs)}")
    print(f"{'='*70}\n")

    success = 0; failed = 0
    for i, cd in enumerate(case_dirs, start=1):
        out_id = f"avt_{i:03d}"
        out_image = images_tr / f"{out_id}_0000.nii.gz"
        out_label = labels_tr / f"{out_id}.nii.gz"
        ok, msg = convert_one(cd, out_image, out_label)
        if ok:
            success += 1
            print(f"  [{i:3d}/{len(case_dirs)}] {cd.name:8s} -> {out_id}  OK")
        else:
            failed += 1
            print(f"  [{i:3d}/{len(case_dirs)}] {cd.name:8s} FAIL: {msg}")

    # write dataset.json
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "aorta": 1},
        "numTraining": success,
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": "Aortic Vessel Tree dataset (binary aorta segmentation)",
    }
    with open(out_dataset / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  DONE: {success} converted, {failed} failed")
    print(f"  dataset.json written to {out_dataset / 'dataset.json'}")
    print(f"{'='*70}\n")
    print(f"  Next step: set env vars and run plan_and_preprocess")
    print(f"  $env:nnUNet_raw = '{args.nnunet_raw}'")
    print(f"  $env:nnUNet_preprocessed = 'C:/TBAD_Pipeline/data/AVT_nnunet/nnUNet_preprocessed'")
    print(f"  $env:nnUNet_results = 'C:/TBAD_Pipeline/models/aorta_avt'")
    print(f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} --verify_dataset_integrity")


if __name__ == "__main__":
    main()