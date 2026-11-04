from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
    r"\Radiomics_Results\BinWidth_Comparison"
    r"\All_Texture_Feature_Stability"
)

INPUT = ROOT / "texture_stability_vs_0p25.csv"
OUTPUT = ROOT / "BinWidth_Candidate_Ranking"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

required = {
    "feature",
    "family",
    "comparison_bin_width",
    "spearman_rank_correlation",
    "median_symmetric_percentage_change",
    "median_absolute_difference",
    "matched_nonmissing_rois",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Keep the five non-reference widths.
df = df[df["comparison_bin_width"].astype(str) != "0p25"].copy()

for col in [
    "spearman_rank_correlation",
    "median_symmetric_percentage_change",
    "median_absolute_difference",
    "matched_nonmissing_rois",
]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Summary for each width and feature family.
family_summary = (
    df.groupby(["comparison_bin_width", "family"])
    .agg(
        number_of_features=("feature", "nunique"),
        median_spearman=("spearman_rank_correlation", "median"),
        minimum_spearman=("spearman_rank_correlation", "min"),
        median_symmetric_change=(
            "median_symmetric_percentage_change", "median"
        ),
        median_absolute_difference=(
            "median_absolute_difference", "median"
        ),
        median_matched_rois=("matched_nonmissing_rois", "median"),
    )
    .reset_index()
)

family_summary.to_csv(
    OUTPUT / "candidate_family_summary.csv",
    index=False
)

# One row per feature per candidate width.
# Rank 1 means best among the five candidates for that feature.
df["rank_correlation"] = df.groupby("feature")[
    "spearman_rank_correlation"
].rank(ascending=False, method="min")

df["rank_relative_change"] = df.groupby("feature")[
    "median_symmetric_percentage_change"
].rank(ascending=True, method="min")

# Equal-weight ranking of the two metrics.
# This is a descriptive ranking, not a validated scientific score.
df["combined_rank"] = (
    df["rank_correlation"] + df["rank_relative_change"]
) / 2

df.to_csv(
    OUTPUT / "feature_by_candidate_rankings.csv",
    index=False
)

# Candidate-level overall summary.
candidate_summary = (
    df.groupby("comparison_bin_width")
    .agg(
        number_of_features=("feature", "nunique"),
        median_spearman=("spearman_rank_correlation", "median"),
        minimum_spearman=("spearman_rank_correlation", "min"),
        median_symmetric_change=(
            "median_symmetric_percentage_change", "median"
        ),
        median_absolute_difference=(
            "median_absolute_difference", "median"
        ),
        median_combined_rank=("combined_rank", "median"),
        median_matched_rois=("matched_nonmissing_rois", "median"),
    )
    .reset_index()
)

candidate_summary.to_csv(
    OUTPUT / "candidate_overall_summary.csv",
    index=False
)

print("\n=== OVERALL BIN-WIDTH CANDIDATES ===")
print(candidate_summary.to_string(index=False))

print("\n=== OUTPUT FILES ===")
print(f"Saved to: {OUTPUT}")
print("- candidate_overall_summary.csv")
print("- candidate_family_summary.csv")
print("- feature_by_candidate_rankings.csv")