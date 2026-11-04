from pathlib import Path
import shutil
import hashlib
import pandas as pd

PROJECT = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
)

ROOT = PROJECT / "Radiomics_Results"
COMPARE = ROOT / "BinWidth_Comparison"

# New, organized structure
DEST = ROOT / "_Organized_Radiomics_Results"

FOLDERS = [
    "01_Original_Extractions/Original_BinWidth_0.25",
    "01_Original_Extractions/Six_BinWidth_Comparison",
    "02_Analysis_Results/Candidate_Ranking",
    "02_Analysis_Results/Texture_Stability",
    "02_Analysis_Results/Gray_Level_Occupancy",
    "03_Quality_Control/Extraction_Logs",
    "03_Quality_Control/Mask_Validation",
    "03_Quality_Control/Patient_File_Matching",
    "04_For_Dr_Yuyi",
    "05_Clinical_Integration",
    "06_Archive/Older_Analysis",
]

for folder in FOLDERS:
    (DEST / folder).mkdir(parents=True, exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# Source path -> destination path
PLAN = {}

# Original extraction folder: preserve the whole folder.
old_dir = ROOT / "binWidth_0.25"
if old_dir.exists():
    for f in old_dir.iterdir():
        if f.is_file():
            PLAN[f] = (
                DEST / "01_Original_Extractions"
                / "Original_BinWidth_0.25" / f.name
            )

# Six primary bin-width CSVs.
for f in COMPARE.glob("radiomics_binWidth_*.csv"):
    PLAN[f] = (
        DEST / "01_Original_Extractions"
        / "Six_BinWidth_Comparison" / f.name
    )

# Combined radiomics data.
for name in ["radiomics_all_bin_widths_long.csv"]:
    f = COMPARE / name
    if f.exists():
        PLAN[f] = (
            DEST / "01_Original_Extractions"
            / "Six_BinWidth_Comparison" / f.name
        )

# Current candidate ranking outputs.
ranking = (
    COMPARE / "All_Texture_Feature_Stability"
    / "BinWidth_Candidate_Ranking"
)
for f in ranking.glob("*.csv"):
    PLAN[f] = DEST / "02_Analysis_Results" / "Candidate_Ranking" / f.name

# Current texture stability outputs.
stability = COMPARE / "All_Texture_Feature_Stability"
for name in [
    "texture_stability_vs_0p25.csv",
    "texture_adjacent_binwidth_stability.csv",
    "texture_stability_family_summary.csv",
    "texture_stability_feature_summary.csv",
    "analysis_run_summary.csv",
]:
    f = stability / name
    if f.exists():
        PLAN[f] = DEST / "02_Analysis_Results" / "Texture_Stability" / f.name

# Occupancy outputs.
for name in [
    "roi_gray_level_occupancy_long.csv",
    "roi_gray_level_occupancy_wide.csv",
]:
    f = COMPARE / name
    if f.exists():
        PLAN[f] = DEST / "02_Analysis_Results" / "Gray_Level_Occupancy" / f.name

# QC outputs.
qc_groups = {
    "Extraction_Logs": [
        "extraction_log_all_bin_widths.csv",
    ],
    "Mask_Validation": [
        "preflight_mask_summary.csv",
    ],
    "Patient_File_Matching": [
        "selected_patient_file_matching.csv",
    ],
}

for folder, names in qc_groups.items():
    for name in names:
        f = COMPARE / name
        if f.exists():
            PLAN[f] = DEST / "03_Quality_Control" / folder / name

# Files to show Dr. Yuyi: copy the summary files.
presentation = {
    ranking / "candidate_overall_summary.csv":
        "01_Candidate_Overall_Summary.csv",
    ranking / "candidate_family_summary.csv":
        "02_Candidate_Family_Summary.csv",
    stability / "texture_stability_feature_summary.csv":
        "03_Feature_Stability_Summary.csv",
    stability / "texture_stability_vs_0p25.csv":
        "04_Stability_vs_Reference_0p25.csv",
}

for src, name in presentation.items():
    if src.exists():
        PLAN[src] = DEST / "04_For_Dr_Yuyi" / name

# Archive the older analysis files.
archive = COMPARE / "Archive_Older_Analysis"
if archive.exists():
    for f in archive.iterdir():
        if f.is_file():
            PLAN[f] = DEST / "06_Archive" / "Older_Analysis" / f.name

# Include any existing clinical integration files.
clinical = ROOT / "Clinical_Integration"
if clinical.exists():
    for f in clinical.rglob("*"):
        if f.is_file():
            PLAN[f] = (
                DEST / "05_Clinical_Integration"
                / f.relative_to(clinical)
            )

# Preview and copy without modifying the originals.
rows = []

for src, dst in PLAN.items():
    status = "READY"

    if not src.exists():
        status = "SOURCE_MISSING"
    elif dst.exists():
        if sha256(src) == sha256(dst):
            status = "ALREADY_COPIED"
        else:
            status = "DESTINATION_CONFLICT"
    else:
        shutil.copy2(src, dst)
        status = "COPIED"

    rows.append({
        "source": str(src),
        "destination": str(dst),
        "status": status,
        "source_bytes": src.stat().st_size if src.exists() else None,
    })

inventory = pd.DataFrame(rows)
inventory.to_csv(
    DEST / "Organization_Manifest.csv",
    index=False,
    encoding="utf-8-sig",
)

print("\nORGANIZATION PREVIEW / COPY COMPLETE")
print(inventory["status"].value_counts().to_string())
print(f"\nOrganized copy: {DEST}")
print(f"Manifest: {DEST / 'Organization_Manifest.csv'}")