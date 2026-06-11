# 05_train.ps1
# Train Dataset504_AortaZoneV2
# Run AFTER 04_patch_nnunet.py

$env:nnUNet_raw          = "C:\nnUNet\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\nnUNet\nnUNet_preprocessed"
$env:nnUNet_results      = "C:\nnUNet\nnUNet_results"

# IMPORTANT: Run 04_patch_nnunet.py first to apply Windows fixes
# Then run this

C:\nnUNet\venv\Scripts\nnUNetv2_train 504 3d_fullres 0 -p nnUNetResEncUNetMPlans

# Training takes ~6-8 hours
# Monitor progress: check training_log_*.txt in:
# C:\nnUNet\nnUNet_results\Dataset504_AortaZoneV2\nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres\fold_0\
