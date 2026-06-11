import nibabel as nib
import numpy as np
from pathlib import Path

ZONE_NAMES = {
    1:"Aortic Root", 2:"Ascending", 3:"Prox Arch", 4:"Mid Arch", 5:"Dist Arch",
    6:"Brachiocephalic", 7:"R Subclavian", 8:"R Carotid", 9:"Desc Thoracic",
    10:"L Carotid", 11:"L Subclavian", 12:"Prox Desc", 13:"Mid Desc",
    14:"Dist Desc", 15:"Celiac", 16:"SMA", 17:"Infrarenal"
}

CASES = ["case002","case031","case050","case068","case079","case085","case150"]

print(f"{'Case':<10} {'Zone':<5} {'Name':<20} {'TL':>8} {'FL':>8} {'Total':>8} {'FL%':>7}")
print("-" * 70)

summary = {}

for case in CASES:
    zone_path = Path(rf"C:\AortaZone_Project\outputs\predictions\{case}\nnunet_output\{case}.nii.gz")
    tlfl_path = Path(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{case}_fullvol.nii.gz")

    if not zone_path.exists() or not tlfl_path.exists():
        print(f"{case}: files missing"); continue

    zones = np.asarray(nib.load(str(zone_path)).dataobj)
    tlfl  = np.asarray(nib.load(str(tlfl_path)).dataobj)

    if zones.shape != tlfl.shape:
        print(f"{case}: shape mismatch {zones.shape} vs {tlfl.shape}"); continue

    case_results = []
    for z in sorted(np.unique(zones[zones>0]).tolist()):
        zone_mask = (zones == z)
        tl = int((zone_mask & (tlfl == 1)).sum())
        fl = int((zone_mask & (tlfl == 2)).sum())
        total = tl + fl
        fl_pct = (fl / total * 100) if total > 0 else 0
        case_results.append((z, tl, fl, total, fl_pct))
        print(f"{case:<10} Z{z:<4} {ZONE_NAMES.get(z,''):20s} {tl:>8,} {fl:>8,} {total:>8,} {fl_pct:>6.1f}%")

    summary[case] = case_results
    print()

# Per-zone average FL% across all cases
print("=" * 70)
print("AVERAGE FL% PER ZONE ACROSS ALL CASES")
print("=" * 70)
from collections import defaultdict
zone_fl = defaultdict(list)
for case, results in summary.items():
    for z, tl, fl, total, fl_pct in results:
        if total > 100:  # skip tiny zones
            zone_fl[z].append(fl_pct)

for z in sorted(zone_fl.keys()):
    vals = zone_fl[z]
    print(f"Z{z:<4} {ZONE_NAMES.get(z,''):20s}: {np.mean(vals):5.1f}% FL  (n={len(vals)} cases)")
