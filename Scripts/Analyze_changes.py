from pathlib import Path
import itertools
import numpy as np
import pandas as pd

ROOT = (
    Path(__file__).resolve().parent.parent
    / "Radiomics_Results"
    / "BinWidth_Comparison"
)

FILES = {
    0.03: "radiomics_binWidth_0p03.csv",
    0.06: "radiomics_binWidth_0p06.csv",
    0.125: "radiomics_binWidth_0p125.csv",
    0.25: "radiomics_binWidth_0p25.csv",
    0.5: "radiomics_binWidth_0p5.csv",
    1.0: "radiomics_binWidth_1p0.csv",
}

OUTPUT = ROOT / "Absolute_Changes_All_Widths"
OUTPUT.mkdir(exist_ok=True)

tables = {}

for width, filename in FILES.items():
    path = ROOT / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")

    df = pd.read_csv(path)

    if df["roi_key"].duplicated().any():
        raise ValueError(f"Duplicate roi_key values in {filename}")

    tables[width] = df.set_index("roi_key")

# Use only ROIs present in every bin-width table.
common_rois = set.intersection(
    *(set(df.index) for df in tables.values())
)

common_rois = sorted(common_rois)

if not common_rois:
    raise ValueError("No common ROIs found across all bin widths.")

metadata_cols = {
    "patient_id", "roi_key", "roi_label", "roi_voxels",
    "small_roi", "bin_width"
}

feature_cols = [
    c for c in tables[0.25].columns
    if c not in metadata_cols
    and pd.api.types.is_numeric_dtype(tables[0.25][c])
    and not c.startswith("original_shape")
]

# Verify that all tables contain the same feature columns.
for width, df in tables.items():
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing features at bin width {width}: {sorted(missing)}"
        )

rows = []

for w1, w2 in itertools.combinations(FILES.keys(), 2):
    a = tables[w1].loc[common_rois, feature_cols]
    b = tables[w2].loc[common_rois, feature_cols]

    for group_name, roi_mask in [
        ("Small", tables[0.25].loc[common_rois, "small_roi"].eq(True)),
        ("Large", tables[0.25].loc[common_rois, "small_roi"].eq(False)),
    ]:
        roi_ids = np.array(common_rois)[roi_mask.to_numpy()]

        if len(roi_ids) == 0:
            continue

        for feature in feature_cols:
            x = pd.to_numeric(a.loc[roi_ids, feature], errors="coerce")
            y = pd.to_numeric(b.loc[roi_ids, feature], errors="coerce")

            valid = np.isfinite(x.to_numpy()) & np.isfinite(y.to_numpy())
            x = x.to_numpy()[valid]
            y = y.to_numpy()[valid]

            if len(x) == 0:
                continue

            abs_diff = np.abs(x - y)
            denom = (np.abs(x) + np.abs(y)) / 2

            pct_change = np.full(len(x), np.nan)
            nonzero_denom = denom > 0
            pct_change[nonzero_denom] = (
                100 * abs_diff[nonzero_denom] / denom[nonzero_denom]
            )

            both_zero = (x == 0) & (y == 0)

            feature_class = (
                feature.split("_")[1]
                if feature.startswith("original_") and "_" in feature
                else "other"
            )

            rows.append({
                "bin_width_1": w1,
                "bin_width_2": w2,
                "comparison": f"{w1:g} vs {w2:g}",
                "group": group_name,
                "feature": feature,
                "feature_class": feature_class,
                "n_rois": len(x),
                "median_absolute_difference": np.median(abs_diff),
                "median_symmetric_pct_change": (
                    np.nanmedian(pct_change)
                    if np.isfinite(pct_change).any()
                    else np.nan
                ),
                "both_zero_pct": 100 * both_zero.mean(),
            })

results = pd.DataFrame(rows)

results.to_csv(
    OUTPUT / "feature_level_absolute_changes_all_widths.csv",
    index=False
)

# Summarize feature-level results by class and ROI size.
summary = (
    results.groupby(
        ["comparison", "group", "feature_class"],
        as_index=False
    )
    .agg(
        n_features=("feature", "nunique"),
        median_feature_pct_change=(
            "median_symmetric_pct_change", "median"
        ),
        max_feature_pct_change=(
            "median_symmetric_pct_change", "max"
        ),
        median_absolute_difference_across_features=(
            "median_absolute_difference", "median"
        ),
    )
)

summary.to_csv(
    OUTPUT / "feature_class_absolute_changes_all_widths.csv",
    index=False
)

print(f"Common ROIs: {len(common_rois)}")
print(f"Features analyzed: {len(feature_cols)}")
print(f"Output folder: {OUTPUT}")

print("\n=== FEATURE-CLASS SUMMARY ===")
print(
    summary[
        summary["feature_class"].isin(["glszm", "gldm", "glrlm"])
    ].to_string(index=False)
)

print("\nSaved:")
print("  feature_level_absolute_changes_all_widths.csv")
print("  feature_class_absolute_changes_all_widths.csv")
