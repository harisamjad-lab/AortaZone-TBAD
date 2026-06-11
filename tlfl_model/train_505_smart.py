import subprocess, json, sys, os
from pathlib import Path

RESULTS = Path(r"C:\nnUNet\nnUNet_results\Dataset505_TLFLMerged\nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres")
TRAINER = r"C:\nnUNet\venv\Scripts\nnUNetv2_train"
TARGET_DICE = 0.80

def get_best_dice(fold):
    summary = RESULTS / f"fold_{fold}" / "validation" / "summary.json"
    if not summary.exists():
        return None
    try:
        data = json.load(open(summary))
        return data["foreground_mean"]["Dice"]
    except:
        return None

env = os.environ.copy()
env["nnUNet_raw"]          = r"C:\nnUNet\nnUNet_raw"
env["nnUNet_preprocessed"] = r"C:\nnUNet\nnUNet_preprocessed"
env["nnUNet_results"]      = r"C:\nnUNet\nnUNet_results"

for fold in range(5):
    print(f"\n{'='*60}")
    print(f"FOLD {fold} ? starting (200 epochs, patch 128x128x128)")
    print(f"{'='*60}")

    subprocess.run([
        TRAINER, "505", "3d_fullres", str(fold),
        "-p", "nnUNetResEncUNetLPlans", "--npz"
    ], env=env)

    dice = get_best_dice(fold)
    if dice is not None:
        print(f"\nFold {fold} best val Dice: {dice:.4f}")
        if dice >= TARGET_DICE:
            print(f"TARGET {TARGET_DICE} REACHED ? stopping.")
            print(f"Use fold {fold} model for inference.")
            sys.exit(0)
        else:
            print(f"Dice {dice:.4f} < {TARGET_DICE} ? moving to fold {fold+1}")
    else:
        print(f"Could not read Dice for fold {fold} ? continuing")

print("\nAll 5 folds done. Best fold did not reach 0.80 ? use ensemble.")
