"""
predict_tlfl_case.py

Runs the trained TL/FL MONAI U-Net on one gt_cropped ImageTBAD case
and saves a cropped-space TL/FL prediction NIfTI.

Output:
  C:/TBAD_Pipeline/outputs/predictions/tlfl_caseXXX.nii.gz

Example:
  python C:/TBAD_Pipeline/scripts/inference/predict_tlfl_case.py --case case_002
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import nibabel as nib

from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.inferers import sliding_window_inference


def case_to_num(case):
    return case.split("_")[-1]


def load_model(model_path, device):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    ).to(device)

    ckpt = torch.load(str(model_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(
            f"Loaded checkpoint: epoch={ckpt.get('best_epoch', '?')} "
            f"metric={ckpt.get('best_metric', '?')}"
        )
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw model state_dict.")

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=str, required=True, help="Example: case_002")
    parser.add_argument("--base_dir", type=str, default="C:/imageTBAD")
    parser.add_argument(
        "--model_path",
        type=str,
        default="C:/imageTBAD/tl_fl_training_outputs/fold_1/best_model.pth",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="C:/TBAD_Pipeline/outputs/predictions",
    )
    parser.add_argument("--patch_size", type=int, nargs=3, default=[128, 128, 128])
    parser.add_argument("--spacing", type=float, nargs=3, default=[1.5, 1.5, 1.5])
    args = parser.parse_args()

    case = args.case
    case_num = case_to_num(case)

    base_dir = Path(args.base_dir)
    image_path = base_dir / "gt_cropped_imagesTr" / f"{case}.nii.gz"
    model_path = Path(args.model_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"tlfl_case{case_num}.nii.gz"

    if not image_path.exists():
        raise FileNotFoundError(f"Missing cropped CT: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    print(f"Case       : {case}")
    print(f"Image      : {image_path}")
    print(f"Model      : {model_path}")
    print(f"Output     : {out_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    transform = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(
            keys=["image"],
            pixdim=tuple(args.spacing),
            mode=("bilinear"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-200,
            a_max=800,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image"]),
    ])

    data = transform({"image": str(image_path)})
    img = data["image"].unsqueeze(0).to(device)  # [1, 1, X, Y, Z]

    print(f"Input tensor shape after spacing: {tuple(img.shape)}")

    model = load_model(model_path, device)

    with torch.no_grad():
        logits = sliding_window_inference(
            img,
            roi_size=tuple(args.patch_size),
            sw_batch_size=2,
            predictor=model,
            overlap=0.5,
        )

    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    print(f"Prediction shape: {pred.shape}")
    print(f"Background voxels: {int((pred == 0).sum())}")
    print(f"TL voxels        : {int((pred == 1).sum())}")
    print(f"FL voxels        : {int((pred == 2).sum())}")

    # This is a cropped-space prediction at 1.5 mm.
    # place_tlfl_correct.py will assign the correct spacing/origin/direction
    # before resampling to the 1.0 mm cropped CT.
    affine = np.diag([args.spacing[0], args.spacing[1], args.spacing[2], 1.0])
    nib.save(nib.Nifti1Image(pred, affine), str(out_path))

    print(f"\nSaved: {out_path}")
    print("\nNext:")
    print(f"python C:/TBAD_Pipeline/scripts/fusion/place_tlfl_correct.py --case {case}")


if __name__ == "__main__":
    main()