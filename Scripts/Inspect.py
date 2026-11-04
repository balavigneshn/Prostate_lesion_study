import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

root = Path(r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study")

folder = (
    root / "Radiomics_Results"
    / "Clinical_Integration"
    / "Patient_Level_Median"
    / "Spearman_Clinical_Association"
)

df = pd.read_csv(folder / "spearman_all_bin_widths.csv")

rho = df.pivot(
    index="feature",
    columns="bin_width",
    values="spearman_rho"
)
rho.columns = rho.columns.astype(str)

widths = ["0.03", "0.06", "0.125", "0.25", "0.5", "1.0"]

print("\n===== CLINICAL ASSOCIATION CONSISTENCY =====")

for i, w1 in enumerate(widths):
    for w2 in widths[i + 1:]:

        pair = rho[[w1, w2]].dropna()

        association_rho, p = spearmanr(
            pair[w1], pair[w2]
        )

        sign_agreement = (
            np.sign(pair[w1]) == np.sign(pair[w2])
        ).mean() * 100

        median_difference = (
            pair[w1] - pair[w2]
        ).abs().median()

        print(
            f"{w1} vs {w2}: "
            f"rho agreement={association_rho:.3f}, "
            f"sign agreement={sign_agreement:.1f}%, "
            f"median |Δrho|={median_difference:.3f}"
        )