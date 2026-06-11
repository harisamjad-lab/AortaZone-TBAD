import os
import random
import json

data_dir = "imagesTr"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nii.gz")])

random.seed(42)
random.shuffle(files)

k = 5
folds = []

fold_size = len(files) // k

for i in range(k):
    val = files[i*fold_size:(i+1)*fold_size]
    train = [f for f in files if f not in val]

    folds.append({
        "train": train,
        "val": val
    })

with open("folds.json", "w") as f:
    json.dump(folds, f, indent=4)

print("Folds created and saved to folds.json")