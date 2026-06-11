"""
correct_aorta_orientation.py

Corrects AVT aorta masks into ImageTBAD imagesTr orientation.

Important:
Some AVT predictions are saved with an LPS-like affine:
  [[-1, 0, 0, 511],
   [ 0,-1, 0, 511],
   [ 0, 0, 1,   0]]

Those need:
  aorta[::-1, ::-1, :]

But some newer predictions are already saved in identity/imagesTr orientation.
Those must NOT be flipped.

This script checks the affine and only flips when needed.

Example:
  python C:/TBAD_Pipeline/scripts/fusion/correct_aorta_orientation.py --case case_079
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def case_to_num(case):
    return case.split("_")[-1]


def needs_xy_flip(aorta_affine, ct_affine):
    """
    Detect the known failure mode:
    CT affine is identity-like, but aorta affine has negative x/y axes.

    Returns True only when aorta x and y diagonal elements are negative
    relative to CT orientation.
    """
    ax = float(aorta_affine[0, 0])
    ay = float(aorta_affine[1, 1])

    cx = float(ct_affine[0, 0])
    cy = float(ct_affine[1, 1])

    # Known old AVT output: aorta affine x/y are negative while CT x/y are positive.
    if cx > 0 and cy > 0 and ax < 0 and ay < 0:
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True, help="Example: case_002")
    parser.add_argument("--ct_dir", type=str, default="C:/imageTBAD/imagesTr")
    parser.add_argument(
        "--aorta_dir",
        type=str,
        default="C:/TBAD_Pipeline/outputs/predictions/avt_on_imagetbad",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="C:/TBAD_Pipeline/outputs/predictions/avt_on_imagetbad_corrected",
    )
    args = parser.parse_args()

    case = args.case
    case_num = case_to_num(case)

    ct_path = Path(args.ct_dir) / f"{case}.nii.gz"
    aorta_path = Path(args.aorta_dir) / f"imagetbad_{case_num}.nii.gz"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"imagetbad_{case_num}.nii.gz"
    alias_path = out_dir / f"{case}.nii.gz"

    if not ct_path.exists():
        raise FileNotFoundError(f"Missing CT: {ct_path}")
    if not aorta_path.exists():
        raise FileNotFoundError(f"Missing aorta: {aorta_path}")

    ct_nib = nib.load(str(ct_path))
    aorta_nib = nib.load(str(aorta_path))

    ct = ct_nib.get_fdata()
    aorta = aorta_nib.get_fdata()

    print(f"Case: {case}")
    print(f"CT path    : {ct_path}")
    print(f"Aorta path : {aorta_path}")
    print(f"Output     : {out_path}")
    print(f"Alias      : {alias_path}")
    print(f"CT shape   : {ct.shape}")
    print(f"Aorta shape: {aorta.shape}")

    print("\nCT affine:")
    print(ct_nib.affine)

    print("\nOriginal aorta affine:")
    print(aorta_nib.affine)

    if ct.shape != aorta.shape:
        raise ValueError(f"Shape mismatch: CT={ct.shape}, aorta={aorta.shape}")

    do_flip = needs_xy_flip(aorta_nib.affine, ct_nib.affine)

    if do_flip:
        print("\nCorrection decision: FLIP X and Y")
        corrected = aorta[::-1, ::-1, :]
    else:
        print("\nCorrection decision: NO FLIP — aorta already appears aligned")
        corrected = aorta

    corrected = (corrected > 0).astype(np.uint8)

    print(f"\nOriginal aorta voxels : {int((aorta > 0).sum())}")
    print(f"Corrected aorta voxels: {int(corrected.sum())}")

    out_nib = nib.Nifti1Image(corrected, ct_nib.affine, ct_nib.header)
    out_nib.set_data_dtype(np.uint8)

    nib.save(out_nib, str(out_path))
    nib.save(out_nib, str(alias_path))

    print(f"\nSaved corrected aorta: {out_path}")
    print(f"Saved alias          : {alias_path}")


if __name__ == "__main__":
    main()