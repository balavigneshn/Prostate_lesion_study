import pandas as pd
from pathlib import Path

root = Path(r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study")

radiomics_path = root / "Radiomics_Results" / "binWidth_0.25" / "radiomics_binWidth_0.25.csv"
clinical_path = root / "Clinical data" / "GT" / "PSMA PET patients clinical data.xlsx"

output_dir = root / "Radiomics_Results" / "Clinical_Integration" / "Eligible_22_Patients"
output_dir.mkdir(parents=True, exist_ok=True)

# Load source files
radiomics = pd.read_csv(radiomics_path)
clinical = pd.read_excel(clinical_path, sheet_name="工作表1")

# Normalize patient IDs
clinical["patient_id"] = pd.to_numeric(
    clinical["Case no"].astype(str).str.extract(r"Ga68PSMA(\d+)", expand=False),
    errors="coerce"
).astype("Int64")

clinical["PSA PFS"] = pd.to_numeric(clinical["PSA PFS"], errors="coerce")

# Select patients with matching radiomics and numeric PSA-PFS
radiomics_ids = set(radiomics["patient_id"].dropna().astype(int))
eligible = clinical[
    clinical["patient_id"].isin(radiomics_ids)
    & clinical["PSA PFS"].notna()
].copy()

# Ensure one clinical record per patient
if eligible["patient_id"].duplicated().any():
    raise ValueError("Duplicate clinical records found for eligible patients.")

eligible_ids = set(eligible["patient_id"].astype(int))

# Retain all ROI records belonging to eligible patients
filtered_radiomics = radiomics[
    radiomics["patient_id"].isin(eligible_ids)
].copy()

# Save outputs
filtered_radiomics.to_csv(
    output_dir / "eligible_22_patients_radiomics.csv",
    index=False
)

eligible.sort_values("patient_id").to_csv(
    output_dir / "eligible_22_patients_clinical.csv",
    index=False
)

# Verify
print("Eligible patients:", len(eligible))
print("Radiomics ROI records:", len(filtered_radiomics))
print("Unique radiomics patients:", filtered_radiomics["patient_id"].nunique())
print("Unique clinical patients:", eligible["patient_id"].nunique())
print("Missing PSA-PFS values:", eligible["PSA PFS"].isna().sum())

print("\nROIs per patient:")
print(
    filtered_radiomics.groupby("patient_id")["roi_label"]
    .nunique()
    .to_string()
)

print("\nFiles saved to:")
print(output_dir)
