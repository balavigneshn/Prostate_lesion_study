import nibabel as nib
import numpy as np
import pandas as pd

from pathlib import Path
from radiomics import imageoperations

ROOT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study\Dataset"
)

OUTPUT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
    r"\Radiomics_Results\PET_ROI_Intensity_Audit"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

BIN_WIDTHS = [0.03, 0.06, 0.125, 0.25, 0.5, 1.0]

DATASETS = [
    ("imagesTr", "labelsTr_n"),
    ("imagesTs", "labelsTs_n")
]

rows = []

for image_folder, mask_folder in DATASETS:

    image_dir = ROOT / image_folder
    mask_dir = ROOT / mask_folder

    mask_files = sorted(mask_dir.glob("*.nii.gz"))

    print(f"\nProcessing {mask_folder}", flush=True)

    for i, mask_path in enumerate(mask_files, start=1):

        patient_id = mask_path.name.replace(".nii.gz", "")
        pet_path = image_dir / f"{patient_id}_0001.nii.gz"

        print(f"[{i}/{len(mask_files)}] {patient_id}", flush=True)

        if not pet_path.exists():
            continue

        pet_img = nib.load(str(pet_path))
        mask_img = nib.load(str(mask_path))

        if pet_img.shape != mask_img.shape:
            print("Shape mismatch:", patient_id)
            continue

        if not np.allclose(pet_img.affine, mask_img.affine, atol=1e-4):
            print("Affine mismatch:", patient_id)
            continue

        pet = pet_img.get_fdata()
        mask = mask_img.get_fdata()

        labels = np.unique(mask)
        labels = labels[labels > 0]

        for label_value in labels:

            roi_values = pet[mask == label_value]
            roi_values = roi_values[np.isfinite(roi_values)]

            if len(roi_values) == 0:
                continue

            row = {
                "patient_id": patient_id,
                "mask_label": int(label_value),
                "ROI_voxels": len(roi_values)
            }

            for bw in BIN_WIDTHS:

                edges = imageoperations.getBinEdges(
                    roi_values,
                    binWidth=bw
                )

                discretized = np.digitize(
                    roi_values,
                    edges
                )

                row[f"exact_levels_{bw}"] = len(
                    np.unique(discretized)
                )

            rows.append(row)

        # Save progress after each patient
        pd.DataFrame(rows).to_csv(
            OUTPUT / "exact_PyRadiomics_gray_level_counts.csv",
            index=False
        )

df = pd.DataFrame(rows)

df["ROI_size_group"] = np.where(
    df["ROI_voxels"] < 64,
    "Small (<64 voxels)",
    "Large (>=64 voxels)"
)

columns = [f"exact_levels_{bw}" for bw in BIN_WIDTHS]

print("\n========== EXACT PYRADIOMICS RESULTS ==========")

print("\nTotal ROIs:", len(df))
print("Unique patients:", df["patient_id"].nunique())

print("\nOverall median occupied levels:")
print(df[columns].median())

print("\nMedian occupied levels by ROI size:")
print(df.groupby("ROI_size_group")[columns].median())

print("\nMean occupied levels by ROI size:")
print(df.groupby("ROI_size_group")[columns].mean())

print("\nROI counts by size:")
print(df["ROI_size_group"].value_counts())

print("\nSaved:", OUTPUT / "exact_PyRadiomics_gray_level_counts.csv")