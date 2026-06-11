"""
04_patch_nnunet.py

Applies required Windows patches to nnUNetTrainer for Dataset504 training.
Same patches as Dataset503 — verified working.

IMPORTANT: Run this BEFORE starting training.
"""

from pathlib import Path

TRAINER_PATH = Path(r"C:\nnUNet\venv\Lib\site-packages\nnunetv2\training\nnUNetTrainer\nnUNetTrainer.py")
PLANNER_PATH = Path(r"C:\nnUNet\venv\Lib\site-packages\nnunetv2\experiment_planning\experiment_planners\default_experiment_planner.py")

# ── Patch 1: nnUNetTrainer.py ─────────────────────────────────────────
print("Patching nnUNetTrainer.py...")
txt = TRAINER_PATH.read_text(encoding="utf-8")
original = txt

# Fix 1: num_processes = 0 (Windows multiprocessing deadlock)
txt = txt.replace("num_processes = max(1,", "num_processes = max(0,")

# Fix 2: torch threads (crash fix)
if "torch.set_num_threads(1)" not in txt:
    txt = txt.replace("import torch\n", "import torch\ntorch.set_num_threads(1)\n")

# Fix 3: epochs
txt = txt.replace("self.num_epochs = 1000", "self.num_epochs = 150")
txt = txt.replace("self.num_epochs = 500",  "self.num_epochs = 150")

# Fix 4: no L-R mirroring (aortic anatomy)
txt = txt.replace(
    "allowed_mirroring_axes = (0, 1, 2)",
    "allowed_mirroring_axes = (1, 2)"
)

TRAINER_PATH.write_text(txt, encoding="utf-8")

# Verify patches
checks = {
    "num_processes=0":           "num_processes = max(0," in txt,
    "torch.set_num_threads":     "torch.set_num_threads(1)" in txt,
    "num_epochs=150":            "self.num_epochs = 150" in txt,
    "mirroring=(1,2)":           "allowed_mirroring_axes = (1, 2)" in txt,
}
print("nnUNetTrainer patches:")
for k, v in checks.items():
    print(f"  {'OK' if v else 'MISSING'} {k}")

# ── Patch 2: default_experiment_planner.py ───────────────────────────
print("\nPatching default_experiment_planner.py...")
txt2 = PLANNER_PATH.read_text(encoding="utf-8")
if "torch.set_num_threads(1)" not in txt2:
    txt2 = txt2.replace("import torch\n", "import torch\ntorch.set_num_threads(1)\n")
    PLANNER_PATH.write_text(txt2, encoding="utf-8")
    print("  OK torch.set_num_threads added")
else:
    print("  OK already patched")

# ── Patch 3: NoNormalization for Dataset504 ───────────────────────────
# After plan_experiment runs, Dataset504 plans.json needs NoNormalization
# for BOTH channels. This script checks and fixes it.
import json, time

plans_path = Path(r"C:\nnUNet\nnUNet_preprocessed\Dataset504_AortaZoneV2\nnUNetResEncUNetMPlans.json")

if plans_path.exists():
    print(f"\nPatching plans.json for NoNormalization...")
    with open(plans_path) as f:
        plans = json.load(f)

    cfg = plans["configurations"]["3d_fullres"]
    cfg["normalization_schemes"] = ["NoNormalization", "NoNormalization"]
    cfg["use_mask_for_norm"] = [False, False]

    with open(plans_path, "w") as f:
        json.dump(plans, f, indent=2)
    print(f"  OK: normalization set to NoNormalization for both channels")
    print(f"  Schemes: {cfg['normalization_schemes']}")
else:
    print(f"\nplans.json not found yet at {plans_path}")
    print("Run 03_plan_preprocess.ps1 first, then re-run this script.")

print("\nAll patches applied. Next: run 05_train.ps1")
