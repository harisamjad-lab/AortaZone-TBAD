# Datasets Used

## Zone Model Training (Dataset504 ? AortaZone V2)
- AortaSeg24: 100 healthy aorta CTA scans with 23-zone labels
- Source: AortaSeg24 challenge (https://aortaseg24.grand-challenge.org/)
- Preprocessing: binary mask + Z-position channel (2-channel input)
- Labels: Z1-Z17 anatomical zones (mapped from 23-zone to 17-zone)

## TL/FL Model Training (Dataset505 ? merged)
- ImageTBAD: 100 TBAD CTA cases with TL/FL/FLT labels
  - Source: Yao et al. 2021, PubMed PMID 34646158
  - Preprocessing: resampled to 1.0mm isotropic, GT ROI crop + 20mm margin
- Figshare (Mayer et al. 2024): 40 TBAD CTA cases with TL/FL labels
  - Source: https://figshare.com/articles/dataset/Aortic_Dissection_Dataset_and_Segmentations/22269091
  - Preprocessing: resampled to 1.0mm isotropic, tight label crop + 20mm margin
- Total: 140 cases, labels [0=bg, 1=TL, 2=FL]

## TBAD Inference Cases (7 cases from ImageTBAD)
- case002, case031, case050, case068, case079, case085, case150
- These cases are NOT in the Dataset505 training split (held out)
- Used for qualitative zone labeling evaluation

## AVT Dataset (outer aorta wall model ? Dataset501)
- AVT: 56 healthy aorta CTA scans
- Source: Radiology AI figshare (https://figshare.com/articles/dataset/ct_segmentation/14912436)
- Used for AVT nnUNet model (pre-trained, not retrained in this project)

## Note
No dataset files are included in this repository.
All datasets must be downloaded separately and placed in the paths
specified in each script before running.
