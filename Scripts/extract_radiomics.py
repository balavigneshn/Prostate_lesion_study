from pathlib import Path
from datetime import datetime
import csv
import json
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics import featureextractor


# ============================================================
# 1. USER CONFIGURATION
# ============================================================

BASE_DIR = Path(
    r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study"
)

DATASET_DIR = BASE_DIR / "Dataset"
SCRIPT_DIR = BASE_DIR / "Scripts"

# ------------------------------------------------------------
# BIN WIDTH
# Change ONLY this value for each experiment.
# ------------------------------------------------------------

BIN_WIDTH = 0.25

# Suggested experiments:
# BIN_WIDTH = 0.03
# BIN_WIDTH = 0.06
# BIN_WIDTH = 0.125
# BIN_WIDTH = 0.25
# BIN_WIDTH = 0.5
# BIN_WIDTH = 1.0

# ------------------------------------------------------------
# SELECTED CLINICAL COHORT
# ------------------------------------------------------------

PATIENT_IDS = [
    9, 14, 15, 20, 21, 23, 24, 25, 26, 27, 28, 29,
    33, 35, 36, 37, 40, 42, 54, 58, 60, 62, 63, 66,
    67, 68, 69, 71, 73, 74, 76, 77, 82, 84, 96,
    102, 103, 104, 105, 110, 111, 113, 117, 118,
    119, 120, 122
]

# Flag small ROIs for review; do not automatically discard them.
SMALL_ROI_THRESHOLD = 64

# ------------------------------------------------------------
# DATA DIRECTORIES
# ------------------------------------------------------------

PET_DIRS = [
    DATASET_DIR / "imagesTr",
    DATASET_DIR / "imagesTs",
]

MASK_DIRS = [
    DATASET_DIR / "labelsTr_n",
    DATASET_DIR / "labelsTs_n",
]

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

# Make a folder name from the selected bin width.
# Example: 0.25 -> binWidth_0p25

WIDTH_TAG = f"{BIN_WIDTH:g}".replace(".", "p")

OUTPUT_DIR = (
    BASE_DIR / "Radiomics_Results" / f"binWidth_{WIDTH_TAG}"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE = OUTPUT_DIR / "successful_rois.jsonl"

FEATURE_CSV = (
    OUTPUT_DIR / f"radiomics_binWidth_{BIN_WIDTH:g}.csv"
)

LOG_CSV = (
    OUTPUT_DIR / f"extraction_log_binWidth_{BIN_WIDTH:g}.csv"
)

# ============================================================
# 2. PYRADIOMICS CONFIGURATION
# ============================================================

PARAMS = {
    "binWidth": BIN_WIDTH,
    "label": 1,
    "correctMask": True,
    "minimumROIDimensions": 2,
    "normalize": False,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "additionalInfo": True,
}

extractor = featureextractor.RadiomicsFeatureExtractor(
    **PARAMS
)

# Extract original-image features only.
extractor.disableAllImageTypes()
extractor.enableImageTypeByName("Original")

# Enable all available feature classes.
extractor.disableAllFeatures()
extractor.enableAllFeatures()


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def patient_id_from_name(filename):
    """Extract patient ID from a dataset filename."""

    match = re.search(
        r"Ga68PSMA0*(\d+)",
        str(filename),
        re.IGNORECASE,
    )

    if match:
        return int(match.group(1))

    return None


def build_file_index(directories, suffix):
    """
    Index files by patient ID.

    Example PET:
        Ga68PSMA009_0001.nii.gz

    Example mask:
        Ga68PSMA009.nii.gz
    """

    index = {}

    for directory in directories:
        if not directory.exists():
            print(f"WARNING: Directory not found: {directory}")
            continue

        for path in directory.glob("*.nii.gz"):

            if not path.name.lower().endswith(suffix.lower()):
                continue

            patient_id = patient_id_from_name(path.name)

            if patient_id is None:
                continue

            index.setdefault(patient_id, []).append(path)

    return index


def get_single_file(index, patient_id):
    """Return a file only when exactly one match exists."""

    matches = index.get(patient_id, [])

    if len(matches) == 1:
        return matches[0], None

    if len(matches) == 0:
        return None, "File not found"

    return None, (
        "Multiple matching files found: "
        + "; ".join(str(p) for p in matches)
    )


def geometry_matches(image, mask):
    """Check image and mask geometry compatibility."""

    if image.GetSize() != mask.GetSize():
        return False, "Image and mask sizes differ"

    if not np.allclose(
        image.GetSpacing(),
        mask.GetSpacing(),
        rtol=0,
        atol=1e-5,
    ):
        return False, "Image and mask spacing differ"

    if not np.allclose(
        image.GetOrigin(),
        mask.GetOrigin(),
        rtol=0,
        atol=1e-4,
    ):
        return False, "Image and mask origins differ"

    if not np.allclose(
        image.GetDirection(),
        mask.GetDirection(),
        rtol=0,
        atol=1e-5,
    ):
        return False, "Image and mask directions differ"

    return True, "OK"


def write_log(record):
    """Append one extraction-log entry immediately."""

    file_exists = LOG_CSV.exists()

    with LOG_CSV.open(
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "patient_id",
                "roi_label",
                "roi_voxels",
                "status",
                "message",
                "pet_path",
                "mask_path",
                "bin_width",
            ],
        )

        if not file_exists or LOG_CSV.stat().st_size == 0:
            writer.writeheader()

        writer.writerow(record)
        f.flush()


def load_successful_keys():
    """
    Read previous successful ROI checkpoints.

    Failed ROIs are not checkpointed as successful and can
    therefore be retried on a subsequent run.
    """

    completed = set()
    records = []

    if not CHECKPOINT_FILE.exists():
        return completed, records

    with CHECKPOINT_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            key = (
                int(record["patient_id"]),
                int(record["roi_label"]),
            )

            completed.add(key)
            records.append(record)

    return completed, records


def append_checkpoint(record):
    """Save one successful ROI immediately."""

    with CHECKPOINT_FILE.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        )

        f.flush()


def export_feature_table(records):
    """Export checkpoint records to a wide feature table."""

    if not records:
        print("\nNo successful feature records to export.")
        return

    df = pd.DataFrame(records)

    # Ensure stable ordering of identification columns.
    leading_columns = [
        "patient_id",
        "roi_label",
        "roi_voxels",
        "small_roi",
        "bin_width",
    ]

    feature_columns = [
        column
        for column in df.columns
        if column not in leading_columns
    ]

    df = df[leading_columns + feature_columns]

    df.sort_values(
        ["patient_id", "roi_label"],
        inplace=True,
    )

    df.to_csv(
        FEATURE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\nFeature table saved:\n{FEATURE_CSV}")


# ============================================================
# 4. INDEX PET IMAGES AND MASKS
# ============================================================

print("=" * 70)
print("SELECTED-COHORT RADIOMICS EXTRACTION")
print("=" * 70)

print(f"Bin width: {BIN_WIDTH} SUVbw")
print(f"Selected patients: {len(PATIENT_IDS)}")
print(f"Output directory: {OUTPUT_DIR}")

print("\nIndexing PET images and segmentation masks...")

pet_index = build_file_index(
    PET_DIRS,
    "_0001.nii.gz",
)

mask_index = build_file_index(
    MASK_DIRS,
    ".nii.gz",
)

completed_keys, records = load_successful_keys()

print(f"Previously successful ROIs: {len(completed_keys)}")

new_successes = 0
failed_rois = 0
skipped_rois = 0
missing_pet = 0
missing_mask = 0


# ============================================================
# 5. EXTRACT SELECTED PATIENTS
# ============================================================

start_time = datetime.now()

for position, patient_id in enumerate(PATIENT_IDS, start=1):

    print(
        f"\n[{position}/{len(PATIENT_IDS)}] "
        f"Patient Ga68PSMA{patient_id:03d}"
    )

    pet_path, pet_error = get_single_file(
        pet_index,
        patient_id,
    )

    mask_path, mask_error = get_single_file(
        mask_index,
        patient_id,
    )

    if pet_path is None:

        missing_pet += 1

        write_log({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "roi_label": "",
            "roi_voxels": "",
            "status": "MISSING_PET",
            "message": pet_error,
            "pet_path": "",
            "mask_path": str(mask_path) if mask_path else "",
            "bin_width": BIN_WIDTH,
        })

        print(f"  PET unavailable: {pet_error}")
        continue

    if mask_path is None:

        missing_mask += 1

        write_log({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "roi_label": "",
            "roi_voxels": "",
            "status": "MISSING_MASK",
            "message": mask_error,
            "pet_path": str(pet_path),
            "mask_path": "",
            "bin_width": BIN_WIDTH,
        })

        print(f"  Mask unavailable: {mask_error}")
        continue

    try:

        pet_image = sitk.ReadImage(str(pet_path))
        mask_image = sitk.ReadImage(str(mask_path))

    except Exception as exc:

        failed_rois += 1

        write_log({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "roi_label": "",
            "roi_voxels": "",
            "status": "IMAGE_READ_ERROR",
            "message": str(exc),
            "pet_path": str(pet_path),
            "mask_path": str(mask_path),
            "bin_width": BIN_WIDTH,
        })

        print(f"  Image reading failed: {exc}")
        continue

    matches, geometry_message = geometry_matches(
        pet_image,
        mask_image,
    )

    if not matches:

        failed_rois += 1

        write_log({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "roi_label": "",
            "roi_voxels": "",
            "status": "GEOMETRY_MISMATCH",
            "message": geometry_message,
            "pet_path": str(pet_path),
            "mask_path": str(mask_path),
            "bin_width": BIN_WIDTH,
        })

        print(f"  Geometry mismatch: {geometry_message}")
        continue

    mask_array = sitk.GetArrayViewFromImage(mask_image)

    roi_labels = np.unique(mask_array)
    roi_labels = roi_labels[roi_labels != 0]

    if len(roi_labels) == 0:

        skipped_rois += 1

        write_log({
            "timestamp": datetime.now().isoformat(),
            "patient_id": patient_id,
            "roi_label": "",
            "roi_voxels": 0,
            "status": "EMPTY_MASK",
            "message": "Mask contains no nonzero labels",
            "pet_path": str(pet_path),
            "mask_path": str(mask_path),
            "bin_width": BIN_WIDTH,
        })

        print("  Empty mask; skipping.")
        continue

    print(f"  Nonzero mask labels: {len(roi_labels)}")

    for roi_label in roi_labels:

        roi_label = int(roi_label)

        key = (patient_id, roi_label)

        if key in completed_keys:

            skipped_rois += 1
            continue

        roi_voxels = int(
            np.count_nonzero(mask_array == roi_label)
        )

        small_roi = roi_voxels < SMALL_ROI_THRESHOLD

        try:

            # Convert this label into a binary mask.
            # This ensures the extractor always receives label=1.
            binary_array = (
                mask_array == roi_label
            ).astype(np.uint8)

            binary_mask = sitk.GetImageFromArray(
                binary_array
            )

            binary_mask.CopyInformation(mask_image)

            feature_result = extractor.execute(
                pet_image,
                binary_mask,
                label=1,
            )

            # Exclude diagnostics from the feature table.
            # Keep actual original-image radiomics features.
            feature_record = {}

            for name, value in feature_result.items():

                if name.startswith("diagnostics_"):
                    continue

                if hasattr(value, "item"):
                    value = value.item()

                if isinstance(value, (int, float, bool)):
                    feature_record[name] = float(value)

                elif isinstance(value, np.number):
                    feature_record[name] = float(value)

            record = {
                "patient_id": patient_id,
                "roi_label": roi_label,
                "roi_voxels": roi_voxels,
                "small_roi": small_roi,
                "bin_width": BIN_WIDTH,
                **feature_record,
            }

            # Save successful extraction immediately.
            append_checkpoint(record)

            completed_keys.add(key)
            records.append(record)
            new_successes += 1

            write_log({
                "timestamp": datetime.now().isoformat(),
                "patient_id": patient_id,
                "roi_label": roi_label,
                "roi_voxels": roi_voxels,
                "status": "SUCCESS",
                "message": (
                    "Small ROI; review"
                    if small_roi
                    else "OK"
                ),
                "pet_path": str(pet_path),
                "mask_path": str(mask_path),
                "bin_width": BIN_WIDTH,
            })

        except Exception as exc:

            failed_rois += 1

            write_log({
                "timestamp": datetime.now().isoformat(),
                "patient_id": patient_id,
                "roi_label": roi_label,
                "roi_voxels": roi_voxels,
                "status": "FAILED",
                "message": str(exc),
                "pet_path": str(pet_path),
                "mask_path": str(mask_path),
                "bin_width": BIN_WIDTH,
            })

            print(
                f"  Failed label {roi_label}: {exc}"
            )


# ============================================================
# 6. EXPORT RESULTS
# ============================================================

export_feature_table(records)

elapsed_minutes = (
    datetime.now() - start_time
).total_seconds() / 60

print("\n" + "=" * 70)
print("EXTRACTION FINISHED")
print("=" * 70)

print(f"Selected patients: {len(PATIENT_IDS)}")
print(f"Previously saved successful ROIs: "
      f"{len(completed_keys) - new_successes}")
print(f"New successful extractions: {new_successes}")
print(f"Failed ROIs/patients: {failed_rois}")
print(f"Empty masks: {skipped_rois}")
print(f"Missing PET images: {missing_pet}")
print(f"Missing masks: {missing_mask}")
print(f"Elapsed minutes: {elapsed_minutes:.2f}")

print(f"\nFeature table:\n{FEATURE_CSV}")
print(f"\nExtraction log:\n{LOG_CSV}")
print(f"\nCheckpoint:\n{CHECKPOINT_FILE}")