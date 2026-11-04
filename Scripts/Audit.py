from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd

# ============================================================
# 1. PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
)

RADIOMICS_FILE = (
    PROJECT_ROOT
    / "Radiomics_Results"
    / "binWidth_0.25"
    / "radiomics_binWidth_0.25.csv"
)
CLINICAL_FILE = (
    PROJECT_ROOT
    / "Clinical data"
    / "GT"
    / "PSMA PET patients clinical data.xlsx"
)

OUTPUT_DIR = PROJECT_ROOT / "Radiomics_Results" / "Clinical_Integration"

OUTPUT_FILE = OUTPUT_DIR / "Data_Audit_Report.xlsx"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. SETTINGS
# ============================================================

# The column "PSA PFS >" must NOT be used as a numeric outcome.
NUMERIC_OUTCOME_NAME = "PSA PFS"

# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def normalize_column_name(value):
    """Normalize column names for robust matching."""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def find_column(df, candidates):
    """Find a column using case-insensitive normalized names."""
    normalized = {
        normalize_column_name(col): col
        for col in df.columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)
        if key in normalized:
            return normalized[key]

    return None


def normalize_patient_id(value):
    """
    Normalize patient identifiers without matching partial numbers.

    Examples:
        9 -> 9
        9.0 -> 9
        '009' -> 9
        'Ga68PSMA009' -> 9

    Returns None if no usable identifier can be extracted.
    """
    if pd.isna(value):
        return None

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return None
        if float(value).is_integer():
            return str(int(value))
        return None

    text = str(value).strip()

    if not text:
        return None

    # Handle an identifier embedded in a known image naming convention.
    match = re.fullmatch(
        r"(?:Ga68PSMA|PSMA)[\s_-]*0*(\d+)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        return str(int(match.group(1)))

    # Handle numeric strings, including leading zeros.
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        number = float(text)
        if number.is_integer():
            return str(int(number))

    # Do not extract arbitrary numbers from unrelated text.
    return None


def find_patient_id_column(df, source):
    """Identify a likely patient ID column using known conventions."""
    candidates = [
        "Patient_ID",
        "Patient ID",
        "PatientID",
        "Case no",
        "Case No.",
        "Case number",
        "Case_Number",
        "Case",
        "Patient",
        "ID",
        "Case ID",
    ]

    col = find_column(df, candidates)

    if col is not None:
        return col

    # Radiomics files sometimes encode the patient ID in an ROI field.
    if source == "radiomics":
        for candidate in ["ROI_ID", "ROI", "Segment", "ROI_Label"]:
            col = find_column(df, [candidate])
            if col is not None:
                return col

    return None


def extract_patient_id_from_roi(value):
    """Extract patient ID from known ROI naming patterns."""
    if pd.isna(value):
        return None

    text = str(value).strip()

    patterns = [
        r"^0*(\d+)_Segment_\d+$",
        r"^Ga68PSMA0*(\d+)(?:_Segment_\d+)?$",
        r"^PSMA0*(\d+)(?:_Segment_\d+)?$",
    ]

    for pattern in patterns:
        match = re.fullmatch(pattern, text, flags=re.IGNORECASE)
        if match:
            return str(int(match.group(1)))

    return None


def is_numeric_series(series):
    """Determine whether a column contains usable numeric values."""
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().sum() > 0


# ============================================================
# 4. VERIFY INPUT FILES
# ============================================================

for file_path in [RADIOMICS_FILE, CLINICAL_FILE]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

print("=" * 70)
print("PROSTATE PSMA PET RADIOMICS DATA AUDIT")
print("=" * 70)

print(f"Radiomics file: {RADIOMICS_FILE}")
print(f"Clinical file:  {CLINICAL_FILE}")
print(f"Output file:    {OUTPUT_FILE}")

# ============================================================
# 5. LOAD RADIOMICS DATA
# ============================================================

radiomics = pd.read_csv(RADIOMICS_FILE)

print("\nRADIOMICS DATA")
print("-" * 70)
print("Shape:", radiomics.shape)
print("\nColumns:")
for col in radiomics.columns:
    print(" -", repr(col))

if radiomics.empty:
    raise ValueError("The radiomics CSV contains no data rows.")

radiomics.columns = [
    str(col).strip()
    for col in radiomics.columns
]

rad_id_col = find_patient_id_column(radiomics, "radiomics")

if rad_id_col is None:
    raise ValueError(
        "Could not identify a patient ID column in the radiomics CSV. "
        "Inspect the printed columns and update the script."
    )

print("\nDetected radiomics identifier column:", rad_id_col)

# First try direct normalization of the detected identifier column.
radiomics["Patient_ID_Normalized"] = (
    radiomics[rad_id_col].map(normalize_patient_id)
)

# If the identifier column is an ROI name, parse the known ROI pattern.
missing_rad_id = radiomics["Patient_ID_Normalized"].isna()

radiomics.loc[missing_rad_id, "Patient_ID_Normalized"] = (
    radiomics.loc[missing_rad_id, rad_id_col]
    .map(extract_patient_id_from_roi)
)

# Detect an ROI identifier column, if present.
roi_id_col = find_column(
    radiomics,
    ["ROI_ID", "ROI", "Segment", "ROI_Label", "Mask_ID"]
)

# Determine which columns are numeric features.
metadata_candidates = {
    normalize_column_name(x)
    for x in [
        rad_id_col,
        "Patient_ID_Normalized",
        "Patient_ID",
        "ROI_ID",
        "ROI",
        "Segment",
        "ROI_Label",
        "Mask_ID",
        "Image",
        "ImageFile",
        "MaskFile",
        "BinWidth",
        "ExtractionStatus",
        "Status",
    ]
}

numeric_columns = [
    col for col in radiomics.columns
    if normalize_column_name(col) not in metadata_candidates
    and pd.api.types.is_numeric_dtype(radiomics[col])
]

rad_valid_id = radiomics["Patient_ID_Normalized"].notna()

print("Radiomics rows:", len(radiomics))
print(
    "Rows with identifiable patient ID:",
    int(rad_valid_id.sum())
)
print(
    "Rows without identifiable patient ID:",
    int((~rad_valid_id).sum())
)
print(
    "Unique identifiable radiomics patients:",
    radiomics.loc[rad_valid_id, "Patient_ID_Normalized"].nunique()
)
print("Potential numeric feature columns:", len(numeric_columns))

# ROI counts per patient.
roi_counts = (
    radiomics.loc[rad_valid_id]
    .groupby("Patient_ID_Normalized")
    .size()
    .rename("Radiomics_Row_Count")
    .reset_index()
    .rename(columns={"Patient_ID_Normalized": "Patient_ID"})
)

# Per-patient feature missingness.
if numeric_columns:
    feature_missing = (
        radiomics.loc[rad_valid_id]
        .groupby("Patient_ID_Normalized")[numeric_columns]
        .apply(lambda frame: frame.isna().sum().sum())
        .rename("Numeric_Feature_Missing_Cells")
        .reset_index()
        .rename(columns={"Patient_ID_Normalized": "Patient_ID"})
    )
else:
    feature_missing = pd.DataFrame(
        columns=["Patient_ID", "Numeric_Feature_Missing_Cells"]
    )

# Duplicate row checks.
duplicate_rows = radiomics[
    radiomics.duplicated(keep=False)
].copy()

if roi_id_col:
    duplicate_roi_keys = radiomics[
        radiomics.duplicated(
            subset=["Patient_ID_Normalized", roi_id_col],
            keep=False
        )
        & radiomics["Patient_ID_Normalized"].notna()
    ].copy()
else:
    duplicate_roi_keys = pd.DataFrame()

# ============================================================
# 6. LOAD CLINICAL WORKBOOK
# ============================================================

excel_file = pd.ExcelFile(CLINICAL_FILE)

print("\nCLINICAL WORKBOOK")
print("-" * 70)
print("Sheets:", excel_file.sheet_names)

clinical_frames = {}
sheet_summaries = []
clinical_column_inventory = []

for sheet in excel_file.sheet_names:
    df = pd.read_excel(CLINICAL_FILE, sheet_name=sheet)

    df.columns = [str(col).strip() for col in df.columns]

    clinical_frames[sheet] = df

    print(f"\nSheet: {sheet}")
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))

    sheet_summaries.append({
        "Sheet_Name": sheet,
        "Rows": len(df),
        "Columns_Count": len(df.columns),
        "Columns": " | ".join(df.columns),
    })

    for col in df.columns:
        clinical_column_inventory.append({
            "Sheet_Name": sheet,
            "Column_Name": col,
            "Non_Missing_Count": int(df[col].notna().sum()),
            "Missing_Count": int(df[col].isna().sum()),
            "Data_Type": str(df[col].dtype),
        })

# ============================================================
# 7. IDENTIFY CLINICAL ID AND PSA-PFS COLUMNS
# ============================================================

clinical_audits = []
clinical_for_matching = []

for sheet, df in clinical_frames.items():

    id_col = find_patient_id_column(df, "clinical")

    psa_col = find_column(
        df,
        ["PSA PFS"]
    )

    psa_greater_col = find_column(
        df,
        ["PSA PFS >"]
    )

    print(f"\nSheet: {sheet}")
    print("Detected patient ID column:", id_col)
    print("Detected numeric PSA-PFS column:", psa_col)
    print("PSA PFS > column (audit only):", psa_greater_col)

    clinical_audits.append({
        "Sheet_Name": sheet,
        "Detected_ID_Column": id_col,
        "Detected_Numeric_PSA_PFS_Column": psa_col,
        "PSA_PFS_Greater_Than_Column_Audit_Only": psa_greater_col,
        "Rows": len(df),
    })

    if id_col is None:
        print(
            "WARNING: No patient ID column detected in this sheet."
        )
        continue

    temp = df.copy()

    temp["Patient_ID_Normalized"] = temp[id_col].map(
        normalize_patient_id
    )

    # Keep a record of source sheet and row number.
    temp["Source_Sheet"] = sheet
    temp["Source_Excel_Row"] = np.arange(2, len(temp) + 2)

    if psa_col is not None:
        # Use only the exact numeric PSA PFS column.
        # The PSA PFS > column is deliberately excluded.
        temp["PSA_PFS_Numeric_Raw"] = temp[psa_col]
        temp["PSA_PFS_Numeric"] = pd.to_numeric(
            temp[psa_col],
            errors="coerce"
        )
    else:
        temp["PSA_PFS_Numeric_Raw"] = np.nan
        temp["PSA_PFS_Numeric"] = np.nan

    clinical_for_matching.append(temp)

# ============================================================
# 8. COMBINE CLINICAL SHEETS FOR AUDIT
# ============================================================

if clinical_for_matching:
    clinical_all = pd.concat(
        clinical_for_matching,
        ignore_index=True,
        sort=False
    )
else:
    clinical_all = pd.DataFrame()

if not clinical_all.empty:
    clinical_valid_id = clinical_all[
        clinical_all["Patient_ID_Normalized"].notna()
    ].copy()

    clinical_valid_id["Has_Numeric_PSA_PFS"] = (
        clinical_valid_id["PSA_PFS_Numeric"].notna()
    )

    clinical_patient_summary = (
        clinical_valid_id
        .groupby("Patient_ID_Normalized")
        .agg(
            Clinical_Row_Count=("Patient_ID_Normalized", "size"),
            Numeric_PSA_PFS_Count=("PSA_PFS_Numeric", "count"),
        )
        .reset_index()
        .rename(
            columns={"Patient_ID_Normalized": "Patient_ID"}
        )
    )

    clinical_patient_summary["Has_Numeric_PSA_PFS"] = (
        clinical_patient_summary["Numeric_PSA_PFS_Count"] > 0
    )

    clinical_duplicates = clinical_valid_id[
        clinical_valid_id.duplicated(
            subset=["Patient_ID_Normalized"],
            keep=False
        )
    ].copy()

    clinical_ids = set(
        clinical_valid_id["Patient_ID_Normalized"].astype(str)
    )

    numeric_outcome_ids = set(
        clinical_valid_id.loc[
            clinical_valid_id["PSA_PFS_Numeric"].notna(),
            "Patient_ID_Normalized"
        ].astype(str)
    )

else:
    clinical_valid_id = pd.DataFrame()
    clinical_patient_summary = pd.DataFrame()
    clinical_duplicates = pd.DataFrame()
    clinical_ids = set()
    numeric_outcome_ids = set()

# ============================================================
# 9. MATCH RADIOLOGY PATIENTS WITH CLINICAL PATIENTS
# ============================================================

radiomics_ids = set(
    radiomics.loc[
        radiomics["Patient_ID_Normalized"].notna(),
        "Patient_ID_Normalized"
    ].astype(str)
)

matched_ids = radiomics_ids & clinical_ids
radiomics_only_ids = radiomics_ids - clinical_ids
clinical_only_ids = clinical_ids - radiomics_ids
matched_numeric_outcome_ids = radiomics_ids & numeric_outcome_ids

print("\nPATIENT MATCHING")
print("-" * 70)
print("Unique radiomics patients:", len(radiomics_ids))
print("Unique clinical patients:", len(clinical_ids))
print("Patients present in both datasets:", len(matched_ids))
print("Radiomics-only patients:", len(radiomics_only_ids))
print("Clinical-only patients:", len(clinical_only_ids))
print(
    "Radiomics patients with numeric PSA-PFS:",
    len(matched_numeric_outcome_ids)
)

patient_matching = pd.DataFrame({
    "Patient_ID": sorted(
        radiomics_ids | clinical_ids,
        key=lambda x: int(x)
    )
})

patient_matching["Has_Radiomics"] = (
    patient_matching["Patient_ID"].isin(radiomics_ids)
)

patient_matching["Has_Clinical_Record"] = (
    patient_matching["Patient_ID"].isin(clinical_ids)
)

patient_matching["Has_Numeric_PSA_PFS"] = (
    patient_matching["Patient_ID"].isin(numeric_outcome_ids)
)

patient_matching["Has_Radiomics_And_Clinical"] = (
    patient_matching["Patient_ID"].isin(matched_ids)
)

patient_matching["Eligible_For_Numeric_PSA_PFS_Analysis"] = (
    patient_matching["Patient_ID"].isin(
        matched_numeric_outcome_ids
    )
)

patient_matching["Audit_Status"] = np.select(
    [
        patient_matching["Has_Radiomics_And_Clinical"]
        & patient_matching["Has_Numeric_PSA_PFS"],

        patient_matching["Has_Radiomics_And_Clinical"]
        & ~patient_matching["Has_Numeric_PSA_PFS"],

        patient_matching["Has_Radiomics"]
        & ~patient_matching["Has_Clinical_Record"],

        ~patient_matching["Has_Radiomics"]
        & patient_matching["Has_Clinical_Record"],
    ],
    [
        "Radiomics + clinical + numeric PSA-PFS",
        "Matched, but numeric PSA-PFS unavailable",
        "Radiomics only",
        "Clinical only",
    ],
    default="Needs review"
)

# ============================================================
# 10. SUMMARIZE MISSINGNESS
# ============================================================

missingness_rows = []

for col in radiomics.columns:
    missingness_rows.append({
        "Dataset": "Radiomics",
        "Column": col,
        "Total_Rows": len(radiomics),
        "Missing_Count": int(radiomics[col].isna().sum()),
        "Missing_Percent": (
            float(radiomics[col].isna().mean() * 100)
            if len(radiomics) else np.nan
        )
    })

for sheet, df in clinical_frames.items():
    for col in df.columns:
        missingness_rows.append({
            "Dataset": f"Clinical: {sheet}",
            "Column": col,
            "Total_Rows": len(df),
            "Missing_Count": int(df[col].isna().sum()),
            "Missing_Percent": (
                float(df[col].isna().mean() * 100)
                if len(df) else np.nan
            )
        })

missingness = pd.DataFrame(missingness_rows)

# ============================================================
# 11. BUILD AUDIT SUMMARY
# ============================================================

summary_rows = [
    ("Radiomics file", str(RADIOMICS_FILE)),
    ("Clinical workbook", str(CLINICAL_FILE)),
    ("Radiomics rows", len(radiomics)),
    ("Radiomics columns", len(radiomics.columns)),
    ("Detected patient ID column", rad_id_col),
    ("Potential numeric radiomics feature columns", len(numeric_columns)),
    ("Radiomics rows with identifiable patient ID", int(rad_valid_id.sum())),
    ("Radiomics rows without identifiable patient ID", int((~rad_valid_id).sum())),
    ("Unique radiomics patients", len(radiomics_ids)),
    ("Clinical workbook sheets", len(excel_file.sheet_names)),
    ("Unique clinical patients", len(clinical_ids)),
    ("Matched radiomics/clinical patients", len(matched_ids)),
    ("Radiomics-only patients", len(radiomics_only_ids)),
    ("Clinical-only patients", len(clinical_only_ids)),
    ("Matched patients with numeric PSA-PFS", len(matched_numeric_outcome_ids)),
    ("Exact duplicate radiomics rows", len(duplicate_rows)),
    ("Clinical rows with duplicated patient IDs", len(clinical_duplicates)),
    ("Primary outcome column", NUMERIC_OUTCOME_NAME),
    ("PSA PFS > used in primary outcome analysis", "NO"),
]

summary = pd.DataFrame(
    summary_rows,
    columns=["Audit_Item", "Result"]
)

# ============================================================
# 12. WRITE REPORT
# ============================================================

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    summary.to_excel(
        writer,
        sheet_name="Audit_Summary",
        index=False
    )

    pd.DataFrame(sheet_summaries).to_excel(
        writer,
        sheet_name="Clinical_Sheets",
        index=False
    )

    pd.DataFrame(clinical_column_inventory).to_excel(
        writer,
        sheet_name="Clinical_Columns",
        index=False
    )

    pd.DataFrame(clinical_audits).to_excel(
        writer,
        sheet_name="Clinical_ID_Outcome_Audit",
        index=False
    )

    roi_counts.to_excel(
        writer,
        sheet_name="ROIs_Per_Patient",
        index=False
    )

    feature_missing.to_excel(
        writer,
        sheet_name="Feature_Missingness_By_Patient",
        index=False
    )

    patient_matching.to_excel(
        writer,
        sheet_name="Patient_Matching",
        index=False
    )

    missingness.to_excel(
        writer,
        sheet_name="Missingness",
        index=False
    )

    duplicate_rows.to_excel(
        writer,
        sheet_name="Duplicate_Radiomics_Rows",
        index=False
    )

    duplicate_roi_keys.to_excel(
        writer,
        sheet_name="Duplicate_ROI_Keys",
        index=False
    )

    clinical_patient_summary.to_excel(
        writer,
        sheet_name="Clinical_Patient_Summary",
        index=False
    )

    clinical_duplicates.to_excel(
        writer,
        sheet_name="Duplicate_Clinical_IDs",
        index=False
    )

    # Preserve clinical source rows and normalized IDs for audit.
    # The PSA PFS > column is retained only as source data/audit context.
    # It is not used to construct the numeric outcome.
    if not clinical_all.empty:
        clinical_all.to_excel(
            writer,
            sheet_name="Clinical_Raw_Audit",
            index=False
        )

    # Record the exact radiomics column inventory.
    pd.DataFrame({
        "Column_Name": radiomics.columns,
        "Data_Type": [str(radiomics[c].dtype) for c in radiomics.columns],
        "Missing_Count": [int(radiomics[c].isna().sum()) for c in radiomics.columns],
        "Unique_Non_Missing": [int(radiomics[c].nunique(dropna=True)) for c in radiomics.columns],
    }).to_excel(
        writer,
        sheet_name="Radiomics_Columns",
        index=False
    )

# ============================================================
# 13. FINAL TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("AUDIT COMPLETED")
print("=" * 70)

print(f"Radiomics rows: {len(radiomics)}")
print(f"Unique radiomics patients: {len(radiomics_ids)}")
print(f"Unique clinical patients: {len(clinical_ids)}")
print(f"Matched patients: {len(matched_ids)}")
print(f"Matched patients with numeric PSA-PFS: {len(matched_numeric_outcome_ids)}")
print(f"Radiomics-only patients: {len(radiomics_only_ids)}")
print(f"Clinical-only patients: {len(clinical_only_ids)}")
print(f"Exact duplicate radiomics rows: {len(duplicate_rows)}")
print(f"Duplicated clinical patient rows: {len(clinical_duplicates)}")

print("\nReport saved to:")
print(OUTPUT_FILE)

print("\nIMPORTANT:")
print("- PSA PFS > was not used as a numeric outcome.")
print("- Duplicate clinical IDs require review before merging.")
print("- Numeric PSA-PFS units and event/censoring definitions still require confirmation.")
print("- Review the detected patient ID columns before relying on the match counts.")

