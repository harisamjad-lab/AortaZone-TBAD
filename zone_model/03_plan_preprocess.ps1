# 03_plan_preprocess.ps1
# Run plan + preprocess for Dataset504
# CRITICAL: must set env vars first

$env:nnUNet_raw          = "C:\nnUNet\nnUNet_raw"
$env:nnUNet_preprocessed = "C:\nnUNet\nnUNet_preprocessed"
$env:nnUNet_results      = "C:\nnUNet\nnUNet_results"

# Step 1: Extract dataset fingerprint
C:\nnUNet\venv\Scripts\nnUNetv2_extract_fingerprint -d 504 -np 1

# Step 2: Plan experiment with ResEncM (same as Dataset503)
C:\nnUNet\venv\Scripts\nnUNetv2_plan_experiment -d 504 -pl nnUNetResEncUNetMPlans

# Step 3: Preprocess
C:\nnUNet\venv\Scripts\nnUNetv2_preprocess -d 504 -plans_name nnUNetResEncUNetMPlans -np 1

Write-Host "Plan+preprocess done. Next: run 04_patch_nnunet.py then 05_train.ps1"
