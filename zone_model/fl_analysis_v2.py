import nibabel as nib, numpy as np
from collections import defaultdict

ZONE_NAMES = {1:"Root",2:"Ascending",3:"PrxArch",4:"MidArch",5:"DistArch",
              6:"Brachio",7:"RSubclav",8:"RCarotid",9:"DescThor",10:"LCarotid",
              11:"LSubclav",12:"PrxDesc",13:"MidDesc",14:"DistDesc",
              15:"Celiac",16:"SMA",17:"Infrarenal"}

zone_fl = defaultdict(list)
for case in ["case002","case031","case050","case068","case079","case085","case150"]:
    zones = np.asarray(nib.load(rf"C:\AortaZone_Project_v2\outputs\predictions\{case}\nnunet_output\{case}.nii.gz").dataobj)
    tlfl  = np.asarray(nib.load(rf"C:\TBAD_Pipeline\outputs\predictions\tlfl_{case}_fullvol.nii.gz").dataobj)
    for z in np.unique(zones[zones>0]).tolist():
        mask = (zones==z)
        tl = int((mask & (tlfl==1)).sum())
        fl = int((mask & (tlfl==2)).sum())
        total = tl+fl
        if total > 100:
            zone_fl[z].append(fl/total*100)

print("V2 Average FL% per zone:")
for z in sorted(zone_fl.keys()):
    print(f"  Z{z:<3} {ZONE_NAMES.get(z,''):12}: {np.mean(zone_fl[z]):5.1f}%  (n={len(zone_fl[z])})")
