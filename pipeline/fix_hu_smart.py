"""

fix_hu_smart.py



Auto-detects HU encoding per case and normalizes to standard signed HU

(air ~ -1000, water ~ 0, contrast aorta ~ 200-500).



Detection logic:

  - If min >= 0  : file is unsigned-shifted, subtract 1024

  - If min < 0   : file is already signed, leave alone (but clip extremes)



After fix, all volumes are clipped to [-1024, 3071] to remove outliers

and match standard CT range.



Usage:

  python fix_hu_smart.py --in_dir C:/nnUNet/test_imagetbad

"""



import argparse

import nibabel as nib

import numpy as np

from pathlib import Path





def fix_one(in_path: Path, out_path: Path):

    img = nib.load(str(in_path))

    data = img.get_fdata().astype(np.float32)



    before_min  = float(data.min())

    before_max  = float(data.max())

    before_mean = float(data.mean())



    # ── Detection logic ──

    if before_min >= 0:

        action = "shift -1024 (was unsigned-encoded)"

        data = data - 1024.0

    elif before_min < -1500:

        action = "no shift, but clipping extreme outliers"

    else:

        action = "no shift, already signed HU"



    # ── Always clip to standard CT range ──

    data = np.clip(data, -1024.0, 3071.0)



    after_min  = float(data.min())

    after_max  = float(data.max())

    after_mean = float(data.mean())



    new_img = nib.Nifti1Image(data, img.affine, img.header)

    nib.save(new_img, str(out_path))



    print(f"  {in_path.name}")

    print(f"    Action : {action}")

    print(f"    Before : min={before_min:7.1f}  max={before_max:7.1f}  mean={before_mean:7.1f}")

    print(f"    After  : min={after_min:7.1f}  max={after_max:7.1f}  mean={after_mean:7.1f}")

    print()





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dir",  type=str, default="C:/nnUNet/test_imagetbad")

    parser.add_argument("--out_dir", type=str, default=None)

    args = parser.parse_args()



    in_dir  = Path(args.in_dir)

    out_dir = Path(args.out_dir) if args.out_dir else in_dir

    out_dir.mkdir(parents=True, exist_ok=True)



    files = sorted(in_dir.glob("*.nii.gz"))

    print(f"Found {len(files)} files in {in_dir}\n")

    for f in files:

        fix_one(f, out_dir / f.name)

    print("Done.")





if __name__ == "__main__":

    main()

