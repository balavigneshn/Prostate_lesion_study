from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
)

PATIENT_DIR = (
    ROOT / "Radiomics_Results"
    / "Clinical_Integration"
    / "Patient_Level_Median"
)

CLINICAL_FILE = (
    ROOT / "Radiomics_Results"
    / "Clinical_Integration"
    / "Eligible_22_Patients"
    / "eligible_22_patients_clinical.csv"
)

OUTPUT_DIR = (
    ROOT / "Radiomics_Results"
    / "Clinical_Integration"
    / "Cox_Regression"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
BIN_WIDTHS = {
    "0.125": "patient_level_binWidth_0p125.csv",
    "0.25": "patient_level_binWidth_0p25.csv",
}

FEATURES = [
    "original_glszm_LargeAreaHighGrayLevelEmphasis",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    "original_glcm_Correlation",
]

# --------------------------------------------------
# LOAD CLINICAL DATA
# --------------------------------------------------
clinical = pd.read_csv(CLINICAL_FILE)
clinical.columns = clinical.columns.str.strip()

required = ["patient_id", "PSA PFS", "PSA recurrence"]

missing = [c for c in required if c not in clinical.columns]
if missing:
    raise ValueError(f"Missing clinical columns: {missing}")

clinical["PSA PFS"] = pd.to_numeric(
    clinical["PSA PFS"], errors="coerce"
)

clinical["event"] = (
    clinical["PSA recurrence"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("v")
    .astype(int)
)

clinical = clinical.dropna(
    subset=["patient_id", "PSA PFS"]
).copy()

if clinical["patient_id"].duplicated().any():
    raise ValueError("Duplicate patient IDs in clinical data.")

print("\n========== CLINICAL COHORT ==========")
print("Patients:", len(clinical))
print("Events:", clinical["event"].sum())
print("Censored:", (clinical["event"] == 0).sum())

# --------------------------------------------------
# COX REGRESSION
# --------------------------------------------------
results = []

for width, filename in BIN_WIDTHS.items():

    print(f"\n========== BIN WIDTH {width} ==========")

    feature_file = PATIENT_DIR / filename

    if not feature_file.exists():
        print("File not found:", feature_file)
        continue

    radiomics = pd.read_csv(feature_file)
    radiomics.columns = radiomics.columns.str.strip()

    if "patient_id" not in radiomics.columns:
        print("patient_id column missing. Skipping.")
        continue

    if radiomics["patient_id"].duplicated().any():
        raise ValueError(
            f"Duplicate patient IDs in {filename}"
        )

    data = radiomics.drop(
    columns=["PSA PFS", "event"],
    errors="ignore"
    ).merge(
        clinical[["patient_id", "PSA PFS", "event"]],
        on="patient_id",
        how="inner",
        validate="one_to_one"
    )

    print("Matched patients:", len(data))
    print("Recurrence events:", data["event"].sum())

    for feature in FEATURES:

        if feature not in data.columns:
            print(f"Feature missing: {feature}")
            continue

        model_data = data[
            ["PSA PFS", "event", feature]
        ].copy()

        model_data[feature] = pd.to_numeric(
            model_data[feature], errors="coerce"
        )

        model_data = model_data.replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        if model_data[feature].nunique() < 2:
            print(f"Insufficient variation: {feature}")
            continue

        # Standardize feature: HR per 1 SD increase
        mean = model_data[feature].mean()
        std = model_data[feature].std(ddof=0)

        if std == 0 or not np.isfinite(std):
            continue

        model_data["feature_z"] = (
            model_data[feature] - mean
        ) / std

        model_data = model_data[
            ["PSA PFS", "event", "feature_z"]
        ]

        try:
            cph = CoxPHFitter()

            cph.fit(
                model_data,
                duration_col="PSA PFS",
                event_col="event"
            )

            row = cph.summary.loc["feature_z"]

            results.append({
                "bin_width": width,
                "feature": feature,
                "n": len(model_data),
                "events": int(model_data["event"].sum()),
                "hazard_ratio_per_SD": row["exp(coef)"],
                "CI_lower": row["exp(coef) lower 95%"],
                "CI_upper": row["exp(coef) upper 95%"],
                "p_value": row["p"],
                "concordance_index": cph.concordance_index_,
                "status": "Success"
            })

            print(f"\nFeature: {feature}")
            print(f"HR per SD: {row['exp(coef)']:.4f}")
            print(
                f"95% CI: "
                f"{row['exp(coef) lower 95%']:.4f} – "
                f"{row['exp(coef) upper 95%']:.4f}"
            )
            print(f"P-value: {row['p']:.4f}")
            print(f"C-index: {cph.concordance_index_:.4f}")

        except Exception as error:
            print(f"\nModel failed for {feature}: {error}")

            results.append({
                "bin_width": width,
                "feature": feature,
                "n": len(model_data),
                "events": int(model_data["event"].sum()),
                "status": f"Failed: {error}"
            })

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
results_df = pd.DataFrame(results)

output_file = OUTPUT_DIR / "cox_regression_results.csv"
results_df.to_csv(output_file, index=False)

print("\n========== FINAL RESULTS ==========")
print(results_df.to_string(index=False))
print("\nSaved to:", output_file)
print("\nEND OF ANALYSIS")