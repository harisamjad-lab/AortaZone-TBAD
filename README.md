# Automatic Anatomical Zone Labeling for Type B Aortic Dissection

## Overview
End-to-end pipeline for automatically labeling 19 anatomical zones (Z1-Z17) on TBAD CTA scans
without any TBAD zone-labeled training data.

## Key Results
- Zone model validation Dice: 0.809 (Dataset504, V2)
- TL/FL model validation Dice: 0.899 (Dataset505, nnUNet ResEncL)
- Full Z1-Z17 prediction on case068 (best case)
- Per-zone FL involvement ratio matching known TBAD pathophysiology

## Folder Structure
- pipeline/       TBAD CTA preprocessing, TL/FL inference, AVT model integration
- zone_model/     Zone labeling model training, inference, FL analysis
- tlfl_model/     TL/FL segmentation model training (ImageTBAD + figshare merged)
- visualization/  3D interactive HTML visualizations (Plotly)
- configs/        nnUNet plan JSON files
- results/        Validation metrics and zone coverage tables
- docs/           Pipeline diagrams and documentation

## Models
- Zone model:  nnUNet ResEncM, Dataset504, 2-channel (binary + Z-position)
- TL/FL model: nnUNet ResEncL, Dataset505, ImageTBAD(100) + figshare(40) cases
- AVT model:   nnUNet Dataset501, pre-trained AortaSeg24, outer wall + branches, Dice 0.93

## Requirements
- Python 3.10+
- nnUNet v2
- MONAI
- nibabel, numpy, scipy, scikit-image, plotly, networkx

## Datasets
- ImageTBAD: 100 TBAD CTA cases (Yao et al. 2021)
- figshare: 40 TBAD CTA cases (Mayer et al. 2024)
- AortaSeg24: 100 healthy aortas for zone model training

## Citation
If you use this code, please cite:
[paper citation to be added]
