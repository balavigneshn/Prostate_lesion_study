import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study")
folder = root / "Radiomics_Results" / "Clinical_Integration" / "Eligible_22_Patients"

radiomics_path = folder / "eligible_22_patients_radiomics.csv"
clinical_path = folder / "eligible_22_patients_clinical.csv"

radiomics = pd.read_csv(radiomics_path)
clinical = pd.read_csv(clinical_path)

# Identify radiomics feature columns
metadata_cols = {
    "patient_id", "roi_label", "roi_voxels", "small_roi", "bin_width"
}
feature_cols = [
    col for col in radiomics.columns
    if col not in metadata_cols
    and col.startswith("original_")
]

# Calculate patient-level mean features
patient_features = (
    radiomics.groupby("patient_id")[feature_cols]
    .mean()
    .reset_index()
)

# Keep one PSA-PFS value per patient
clinical["patient_id"] = pd.to_numeric(
    clinical["patient_id"], errors="raise"
).astype(int)

clinical["PSA PFS"] = pd.to_numeric(
    clinical["PSA PFS"], errors="raise"
)

if clinical["patient_id"].duplicated().any():
    raise ValueError("Duplicate clinical patient IDs detected.")

clinical_outcome = clinical[["patient_id", "PSA PFS"]]

# Merge patient-level radiomics with clinical outcome
dataset = patient_features.merge(
    clinical_outcome,
    on="patient_id",
    how="inner",
    validate="one_to_one"
).sort_values("patient_id")

# Validate and save
assert len(dataset) == 22, f"Expected 22 patients, found {len(dataset)}"
assert dataset["patient_id"].nunique() == 22
assert dataset["PSA PFS"].notna().all()

output_path = folder / "patient_level_radiomics_mean.csv"
dataset.to_csv(output_path, index=False)

print("Patients:", len(dataset))
print("Radiomics features:", len(feature_cols))
print("Total columns:", len(dataset.columns))
print("Missing values:", int(dataset.isna().sum().sum()))
print("\nPSA-PFS distribution:")
print(dataset["PSA PFS"].describe().to_string())
print("\nSaved to:", output_path)
