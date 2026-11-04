"""
PSMA PET radiomics bin-width comparison pipeline
================================================

Run inside the balavignesh_nycu environment.

The script:
  1. Finds PET images (_0001.nii.gz) and matching label masks.
  2. Processes the selected patient IDs.
  3. Extracts Original-image PyRadiomics features at six SUVbw bin widths.
  4. Records ROI voxel counts, SUV summaries, and occupied gray levels using
     PyRadiomics' own binning implementation.
  5. Saves per-bin CSVs, a combined long-format table, logs, and a summary.

Important:
  - This reruns ALL bin widths, including 0.25, using the SAME current Python
    environment/settings. The previous CSV is left untouched and is only used
    as a historical reference.
  - The script does not use PSA-PFS to select a bin width.
  - Review the printed file-match summary before interpreting results.
"""

from __future__ import annotations

import json
import re
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, imageoperations


# ----------------------------- USER SETTINGS -----------------------------

ROOT = Path(r"C:\Users\ADMIN\BALAVIGNESH N\Prostate_lesion_study")
DATASET_ROOT = ROOT / "Dataset"

OUTPUT_ROOT = ROOT / "Radiomics_Results" / "BinWidth_Comparison"
BIN_WIDTHS = [0.03, 0.06, 0.125, 0.25, 0.5, 1.0]

# Cohort agreed for the research comparison.
SELECTED_PATIENT_IDS = {
    9, 14, 15, 20, 21, 23, 24, 25, 26, 27, 28, 29, 33, 35, 36, 37,
    40, 42, 54, 58, 60, 62, 63, 66, 67, 68, 69, 71, 73, 74, 76, 77,
    82, 84, 96, 102, 103, 104, 105, 110, 111, 113, 117, 118, 119,
    120, 122
}

# Extraction settings held fixed across all bin widths.
BASE_SETTINGS = {
    "label": 1,  # overridden per ROI label at execute() time
    "correctMask": True,
    "normalize": False,
    "normalizeScale": 1,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline",
    "preCrop": False,
    "padDistance": 5,
    "distances": [1],
    "force2D": False,
    "resegmentRange": None,
    "symmetricalGLCM": True,
    "additionalInfo": True,
    "geometryTolerance": 1e-3,
}

# ------------------------------------------------------------------------


def patient_id_from_name(name: str):
    """Extract numeric patient ID from names such as Ga68PSMA009_0001.nii.gz."""
    match = re.search(r"Ga68PSMA[_-]?(\d+)", name, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else None


def list_nii(folder: Path):
    return sorted(folder.glob("*.nii.gz")) if folder.exists() else []


def build_file_maps():
    image_dirs = [DATASET_ROOT / "imagesTr", DATASET_ROOT / "imagesTs"]
    mask_dirs = [DATASET_ROOT / "labelsTr_n", DATASET_ROOT / "labelsTs_n"]

    pet_map = {}
    mask_map = {}
    duplicate_pet = []
    duplicate_mask = []

    for folder in image_dirs:
        for path in list_nii(folder):
            if path.name.endswith("_0001.nii.gz"):
                pid = patient_id_from_name(path.name)
                if pid is None:
                    continue
                if pid in pet_map:
                    duplicate_pet.append((pid, str(pet_map[pid]), str(path)))
                else:
                    pet_map[pid] = path

    for folder in mask_dirs:
        for path in list_nii(folder):
            # Exclude auxiliary files if present; masks should be Ga68PSMA###.nii.gz.
            if path.name.endswith("_0000.nii.gz") or path.name.endswith("_0001.nii.gz"):
                continue
            pid = patient_id_from_name(path.name)
            if pid is None:
                continue
            if pid in mask_map:
                duplicate_mask.append((pid, str(mask_map[pid]), str(path)))
            else:
                mask_map[pid] = path

    return pet_map, mask_map, duplicate_pet, duplicate_mask


def make_extractor(bin_width: float):
    settings = dict(BASE_SETTINGS)
    settings["binWidth"] = float(bin_width)
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    extractor.disableAllImageTypes()
    extractor.enableImageTypeByName("Original")
    extractor.enableAllFeatures()
    return extractor


def safe_float(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("PSMA PET RADIOMICS BIN-WIDTH COMPARISON")
    print("PyRadiomics version:", radiomics.__version__)
    print("Dataset root:", DATASET_ROOT)
    print("Output root:", OUTPUT_ROOT)
    print("Bin widths:", BIN_WIDTHS)
    print("=" * 78)

    pet_map, mask_map, duplicate_pet, duplicate_mask = build_file_maps()

    print(f"PET images discovered: {len(pet_map)}")
    print(f"Masks discovered: {len(mask_map)}")
    if duplicate_pet:
        print("WARNING: duplicate PET IDs found:")
        for item in duplicate_pet:
            print("  ", item)
    if duplicate_mask:
        print("WARNING: duplicate mask IDs found:")
        for item in duplicate_mask:
            print("  ", item)

    selected = sorted(SELECTED_PATIENT_IDS)
    matched_ids = [pid for pid in selected if pid in pet_map and pid in mask_map]
    missing_pet = [pid for pid in selected if pid not in pet_map]
    missing_mask = [pid for pid in selected if pid not in mask_map]
    print(f"Selected patients: {len(selected)}")
    print(f"Selected IDs with PET + mask: {len(matched_ids)}")
    print("Selected IDs missing PET:", missing_pet)
    print("Selected IDs missing mask:", missing_mask)

    # Save the cohort matching report before extraction.
    matching_rows = []
    for pid in selected:
        matching_rows.append({
            "patient_id": pid,
            "pet_found": pid in pet_map,
            "mask_found": pid in mask_map,
            "pet_path": str(pet_map[pid]) if pid in pet_map else "",
            "mask_path": str(mask_map[pid]) if pid in mask_map else "",
            "status": (
                "matched" if pid in pet_map and pid in mask_map else
                "missing PET" if pid not in pet_map else "missing mask"
            ),
        })
    pd.DataFrame(matching_rows).to_csv(
        OUTPUT_ROOT / "selected_patient_file_matching.csv", index=False
    )

    # Read matched images/masks once; image and mask arrays are also used for
    # the occupied-gray-level calculation at each bin width.
    case_cache = {}
    preflight_rows = []
    for pid in matched_ids:
        try:
            image = sitk.ReadImage(str(pet_map[pid]))
            mask = sitk.ReadImage(str(mask_map[pid]))
            image_arr = sitk.GetArrayFromImage(image)
            mask_arr = sitk.GetArrayFromImage(mask)

            if image_arr.shape != mask_arr.shape:
                raise ValueError(
                    f"Array shape mismatch: PET {image_arr.shape}, mask {mask_arr.shape}"
                )

            labels = sorted(int(x) for x in np.unique(mask_arr) if x > 0)
            case_cache[pid] = {
                "image": image,
                "mask": mask,
                "image_arr": image_arr,
                "mask_arr": mask_arr,
                "labels": labels,
            }
            preflight_rows.append({
                "patient_id": pid,
                "status": "ready" if labels else "empty mask",
                "pet_shape_zyx": str(tuple(image_arr.shape)),
                "mask_shape_zyx": str(tuple(mask_arr.shape)),
                "positive_mask_voxels": int(np.count_nonzero(mask_arr > 0)),
                "positive_labels": ";".join(map(str, labels)),
                "roi_label_count": len(labels),
            })
        except Exception as exc:
            preflight_rows.append({
                "patient_id": pid,
                "status": f"preflight error: {exc}",
                "pet_shape_zyx": "",
                "mask_shape_zyx": "",
                "positive_mask_voxels": np.nan,
                "positive_labels": "",
                "roi_label_count": 0,
            })
            print(f"[PREFLIGHT ERROR] Patient {pid}: {exc}")

    pd.DataFrame(preflight_rows).to_csv(
        OUTPUT_ROOT / "preflight_mask_summary.csv", index=False
    )

    all_feature_rows = []
    all_occupancy_rows = []
    log_rows = []

    for bw in BIN_WIDTHS:
        print(f"\n{'-' * 70}\nExtracting binWidth={bw}\n{'-' * 70}")
        extractor = make_extractor(bw)
        bin_rows = []

        for pid in matched_ids:
            if pid not in case_cache:
                log_rows.append({
                    "bin_width": bw, "patient_id": pid, "roi_label": np.nan,
                    "status": "preflight failed", "message": "See preflight_mask_summary.csv"
                })
                continue

            case = case_cache[pid]
            image = case["image"]
            mask = case["mask"]
            image_arr = case["image_arr"]
            mask_arr = case["mask_arr"]

            for label in case["labels"]:
                roi_bool = mask_arr == label
                roi_voxels = int(np.count_nonzero(roi_bool))
                if roi_voxels == 0:
                    continue

                vals = image_arr[roi_bool].astype(np.float64)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    log_rows.append({
                        "bin_width": bw, "patient_id": pid, "roi_label": label,
                        "status": "failed", "message": "No finite PET values in ROI"
                    })
                    continue

                # Use PyRadiomics' binning function so gray-level occupancy
                # follows the same fixed-bin-width convention as feature extraction.
                try:
                    discretized, _edges = imageoperations.binImage(
                        image_arr, roi_bool, binWidth=float(bw)
                    )
                    occupied_gray_levels = int(
                        np.unique(discretized[roi_bool]).size
                    )
                except Exception as exc:
                    occupied_gray_levels = np.nan
                    log_rows.append({
                        "bin_width": bw, "patient_id": pid, "roi_label": label,
                        "status": "gray-level count warning",
                        "message": f"{type(exc).__name__}: {exc}"
                    })

                roi_key = f"{pid}_Segment_{label}"
                occupancy_row = {
                    "patient_id": pid,
                    "roi_label": label,
                    "roi_key": roi_key,
                    "bin_width": bw,
                    "roi_voxels": roi_voxels,
                    "SUV_min": float(np.min(vals)),
                    "SUV_max": float(np.max(vals)),
                    "SUV_mean": float(np.mean(vals)),
                    "SUV_median": float(np.median(vals)),
                    "SUV_std": float(np.std(vals, ddof=0)),
                    "SUV_range": float(np.max(vals) - np.min(vals)),
                    "occupied_gray_levels": occupied_gray_levels,
                    "theoretical_range_bins_approx": (
                        float(np.floor(np.max(vals) / bw) - np.floor(np.min(vals) / bw) + 1)
                        if bw > 0 else np.nan
                    ),
                    "small_roi": bool(roi_voxels < 64),
                }
                all_occupancy_rows.append(occupancy_row)

                try:
                    result = extractor.execute(image, mask, label=int(label))
                    feature_row = {
                        "patient_id": pid,
                        "roi_label": int(label),
                        "roi_key": roi_key,
                        "roi_voxels": roi_voxels,
                        "small_roi": bool(roi_voxels < 64),
                        "bin_width": float(bw),
                    }
                    for key, value in result.items():
                        if key.startswith("original_"):
                            feature_row[key] = safe_float(value)

                    bin_rows.append(feature_row)
                    all_feature_rows.append(feature_row)
                    log_rows.append({
                        "bin_width": bw, "patient_id": pid, "roi_label": label,
                        "status": "success", "message": ""
                    })
                except Exception as exc:
                    log_rows.append({
                        "bin_width": bw, "patient_id": pid, "roi_label": label,
                        "status": "failed",
                        "message": f"{type(exc).__name__}: {exc}"
                    })
                    print(f"[FAILED] bw={bw}, patient={pid}, label={label}: {exc}")

        bin_df = pd.DataFrame(bin_rows)
        bin_tag = str(bw).replace(".", "p")
        bin_csv = OUTPUT_ROOT / f"radiomics_binWidth_{bin_tag}.csv"
        bin_df.to_csv(bin_csv, index=False)
        print(f"Saved {bin_csv.name}: {len(bin_df)} ROI rows")

    # Combined results and per-ROI gray-level table.
    features_df = pd.DataFrame(all_feature_rows)
    occupancy_df = pd.DataFrame(all_occupancy_rows)
    logs_df = pd.DataFrame(log_rows)

    features_df.to_csv(OUTPUT_ROOT / "radiomics_all_bin_widths_long.csv", index=False)
    occupancy_df.to_csv(OUTPUT_ROOT / "roi_gray_level_occupancy_long.csv", index=False)
    logs_df.to_csv(OUTPUT_ROOT / "extraction_log_all_bin_widths.csv", index=False)

    # Wide occupancy table: one row per ROI and one occupied-level column per bin width.
    if not occupancy_df.empty:
        occ_wide = occupancy_df.pivot_table(
            index=["patient_id", "roi_label", "roi_key", "roi_voxels",
                   "SUV_min", "SUV_max", "SUV_mean", "SUV_median", "SUV_std",
                   "SUV_range", "small_roi"],
            columns="bin_width",
            values="occupied_gray_levels",
            aggfunc="first",
        ).reset_index()
        occ_wide.columns = [
            (f"occupied_gray_levels_bw_{str(c).replace('.', 'p')}" if isinstance(c, float) else c)
            for c in occ_wide.columns
        ]
        occ_wide.to_csv(OUTPUT_ROOT / "roi_gray_level_occupancy_wide.csv", index=False)

    # Feature stability: compare each ROI-feature to its own mean across bins.
    # CV is descriptive; for features near zero, CV can be unstable, so also report
    # standard deviation and normalized range using max(abs(mean), small epsilon).
    metadata_cols = {"patient_id", "roi_label", "roi_key", "roi_voxels", "small_roi", "bin_width"}
    feature_cols = [
        c for c in features_df.columns
        if c not in metadata_cols and c.startswith("original_")
    ]

    stability_rows = []
    if not features_df.empty:
        for (roi_key, feature), grp in features_df.melt(
            id_vars=["patient_id", "roi_label", "roi_key", "bin_width"],
            value_vars=feature_cols,
            var_name="feature",
            value_name="value",
        ).groupby(["roi_key", "feature"], dropna=False):
            vals = pd.to_numeric(grp["value"], errors="coerce").dropna()
            if len(vals) < 2:
                continue
            mean_abs = float(np.mean(np.abs(vals)))
            mean_val = float(np.mean(vals))
            sd = float(np.std(vals, ddof=1))
            stability_rows.append({
                "roi_key": roi_key,
                "feature": feature,
                "n_bin_widths_available": int(len(vals)),
                "mean_across_bin_widths": mean_val,
                "sd_across_bin_widths": sd,
                "range_across_bin_widths": float(vals.max() - vals.min()),
                "normalized_range": float((vals.max() - vals.min()) / max(mean_abs, 1e-12)),
            })

    stability_df = pd.DataFrame(stability_rows)
    stability_df.to_csv(OUTPUT_ROOT / "feature_stability_per_roi.csv", index=False)

    if not stability_df.empty:
        stability_summary = stability_df.groupby("feature").agg(
            roi_feature_pairs=("roi_key", "count"),
            median_normalized_range=("normalized_range", "median"),
            mean_normalized_range=("normalized_range", "mean"),
            median_sd_across_bins=("sd_across_bin_widths", "median"),
        ).reset_index().sort_values("median_normalized_range")
        stability_summary.to_csv(
            OUTPUT_ROOT / "feature_stability_summary.csv", index=False
        )

    # Compact run summary.
    summary = {
        "pyradiomics_version": radiomics.__version__,
        "dataset_root": str(DATASET_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "selected_patient_count": len(selected),
        "matched_pet_mask_count": len(matched_ids),
        "missing_pet_ids": missing_pet,
        "missing_mask_ids": missing_mask,
        "bin_widths": BIN_WIDTHS,
        "feature_rows_total": int(len(features_df)),
        "unique_patients_with_feature_rows": int(features_df["patient_id"].nunique()) if not features_df.empty else 0,
        "unique_rois_with_feature_rows": int(features_df["roi_key"].nunique()) if not features_df.empty else 0,
        "feature_columns_count": int(len(feature_cols)),
        "successful_extractions": int((logs_df["status"] == "success").sum()) if not logs_df.empty else 0,
        "failed_extractions": int((logs_df["status"] == "failed").sum()) if not logs_df.empty else 0,
        "notes": [
            "All bin widths were rerun in the same installed PyRadiomics environment.",
            "The prior binWidth=0.25 CSV was not overwritten.",
            "Occupied gray levels were computed with radiomics.imageoperations.binImage.",
            "Cross-bin stability is descriptive, not proof of test-retest reproducibility.",
            "Small ROIs are flagged; interpret texture features cautiously.",
            "No clinical outcome was used to select bin width.",
        ],
    }
    (OUTPUT_ROOT / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print("EXTRACTION FINISHED")
    print(json.dumps(summary, indent=2))
    print("\nKey output files:")
    for name in [
        "selected_patient_file_matching.csv",
        "preflight_mask_summary.csv",
        "radiomics_all_bin_widths_long.csv",
        "roi_gray_level_occupancy_long.csv",
        "roi_gray_level_occupancy_wide.csv",
        "feature_stability_per_roi.csv",
        "feature_stability_summary.csv",
        "extraction_log_all_bin_widths.csv",
        "run_summary.json",
    ]:
        path = OUTPUT_ROOT / name
        print(f"  {path} {'[created]' if path.exists() else '[not created]'}")


if __name__ == "__main__":
    main()
