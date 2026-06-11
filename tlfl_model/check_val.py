import json
p = r"C:\nnUNet\nnUNet_results\Dataset505_TLFLMerged\nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres\fold_0\validation\summary.json"
d = json.load(open(p))
print("Mean Dice:", d["foreground_mean"]["Dice"])
print("TL Dice:  ", d["mean"]["1"]["Dice"])
print("FL Dice:  ", d["mean"]["2"]["Dice"])
print("\nPer case:")
for entry in d["metric_per_case"]:
    name = entry["reference_file"].split("\\")[-1].replace(".nii.gz","")
    tl = entry["metrics"].get("1", {}).get("Dice", 0)
    fl = entry["metrics"].get("2", {}).get("Dice", 0)
    print(f"  {name}: TL={tl:.4f}  FL={fl:.4f}")
