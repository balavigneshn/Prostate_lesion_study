from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
)

INPUT_FILE = (
    ROOT
    / "Radiomics_Results"
    / "Clinical_Integration"
    / "Eligible_22_Patients"
    / "eligible_22_patients_clinical.csv"
)

OUTPUT_DIR = (
    ROOT
    / "Radiomics_Results"
    / "Clinical_Integration"
    / "Kaplan_Meier"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# LOAD AND PREPARE DATA
# --------------------------------------------------
df = pd.read_csv(INPUT_FILE)

df["PSA PFS"] = pd.to_numeric(df["PSA PFS"], errors="coerce")

df["event"] = (
    df["PSA recurrence"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("v")
    .astype(int)
)

df = df.dropna(subset=["PSA PFS"]).copy()

if (df["PSA PFS"] <= 0).any():
    raise ValueError("PSA PFS contains zero or negative values.")

if not df["event"].isin([0, 1]).all():
    raise ValueError("Invalid event coding.")

# --------------------------------------------------
# COHORT SUMMARY
# --------------------------------------------------
print("\n========== COHORT SUMMARY ==========")
print(f"Total patients: {len(df)}")
print(f"Recurrence events: {df['event'].sum()}")
print(f"Censored patients: {(df['event'] == 0).sum()}")
print(f"Follow-up range: {df['PSA PFS'].min()}–{df['PSA PFS'].max()} months")

# Save prepared survival data
df.to_csv(OUTPUT_DIR / "survival_analysis_dataset.csv", index=False)

# --------------------------------------------------
# KAPLAN-MEIER FIT
# --------------------------------------------------
kmf = KaplanMeierFitter()

kmf.fit(
    durations=df["PSA PFS"],
    event_observed=df["event"],
    label="PSA recurrence-free survival"
)

print("\n========== KAPLAN-MEIER RESULTS ==========")
print(f"Median survival time: {kmf.median_survival_time_} months")

# Survival estimates at selected time points
for month in [6, 12, 18]:
    estimate = kmf.predict(month)
    print(f"Recurrence-free survival at {month} months: {estimate:.3f}")

# Save survival function and confidence intervals
km_results = pd.concat(
    [kmf.survival_function_, kmf.confidence_interval_],
    axis=1
)

km_results.to_csv(OUTPUT_DIR / "kaplan_meier_results.csv")

# --------------------------------------------------
# PLOT
# --------------------------------------------------
plt.figure(figsize=(9, 6))

kmf.plot_survival_function(
    ci_show=True,
    show_censors=True
)

plt.title("Kaplan–Meier Curve: PSA Recurrence-Free Survival")
plt.xlabel("Time from treatment (months)")
plt.ylabel("Probability of remaining recurrence-free")
plt.ylim(0, 1.05)
plt.xlim(left=0)
plt.grid(alpha=0.3)
plt.tight_layout()

plot_path = OUTPUT_DIR / "kaplan_meier_curve.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"\nKaplan–Meier plot saved: {plot_path}")

# --------------------------------------------------
# NUMBER AT RISK
# --------------------------------------------------
print("\n========== NUMBER AT RISK ==========")

for month in [0, 6, 12, 18, 24]:
    at_risk = (df["PSA PFS"] >= month).sum()
    print(f"{month} months: {at_risk}")

print(f"\nAll results saved to: {OUTPUT_DIR}")

print("END OF ANALYSIS")
