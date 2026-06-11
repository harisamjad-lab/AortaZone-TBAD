import nibabel as nib, numpy as np
from collections import defaultdict
from pathlib import Path

ZONE_NAMES = {1:"Root",2:"Ascending",3:"PrxArch",4:"MidArch",5:"DistArch",
              6:"Brachio",7:"RSubclav",8:"RCarotid",9:"DescThor",10:"LCarotid",
              11:"LSubclav",12:"PrxDesc",13:"MidDesc",14:"DistDesc",
              15:"Celiac",16:"SMA",17:"Infrarenal"}

zone_fl_old = defaultdict(list)
zone_fl_new = defaultdict(list)

for case in ["case002","case031","case050","case068","case079","case085","case150"]:
    zones = np.asarray(nib.load(
        rf"C:\AortaZone_Project_v2\outputs\predictions\{case}\nnunet_output\{case}.nii.gz").dataobj)
    old_tlfl = np.asarray(nib.load(
        rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{case}_fullvol.nii.gz").dataobj)
    new_tlfl = np.asarray(nib.load(
        rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_505_{case}_fullvol.nii.gz").dataobj)

    for z in np.unique(zones[zones>0]).tolist():
        mask = (zones==z)
        for tlfl, store in [(old_tlfl, zone_fl_old), (new_tlfl, zone_fl_new)]:
            tl = int((mask & (tlfl==1)).sum())
            fl = int((mask & (tlfl==2)).sum())
            total = tl+fl
            if total > 100:
                store[z].append(fl/total*100)

print(f"{'Zone':<6} {'Name':<14} {'Old FL%':>8} {'New FL%':>8} {'Diff':>8}")
print("-"*50)
for z in sorted(set(list(zone_fl_old.keys())+list(zone_fl_new.keys()))):
    old_v = np.mean(zone_fl_old[z]) if z in zone_fl_old else 0
    new_v = np.mean(zone_fl_new[z]) if z in zone_fl_new else 0
    diff = new_v - old_v
    print(f"Z{z:<5} {ZONE_NAMES.get(z,''):14} {old_v:8.1f}% {new_v:8.1f}% {diff:+8.1f}%")
