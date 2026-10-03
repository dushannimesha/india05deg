#!/usr/bin/env python3
"""Merge and audit annual India hydrology files independently for R1-R10."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_ROOT = ROOT / "data" / "processed"
RESULTS_ROOT = ROOT / "results"

REGIONS = tuple(f"R{index}" for index in range(1, 11))
EXPECTED_YEARS = list(range(1982, 2012))
EXPECTED_MONTHS_PER_REGION_ID = 12 * len(EXPECTED_YEARS)

EXPECTED_COLUMNS = [
    "region_id",
    "longitude",
    "latitude",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
    "date",
    "year",
    "month",
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm",
    "surface_runoff_mm",
    "subsurface_runoff_mm",
    "total_runoff_mm",
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
    "soil_moisture_total_mm",
]

FLUX_STATE_COLUMNS = [
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm",
    "surface_runoff_mm",
    "subsurface_runoff_mm",
    "total_runoff_mm",
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
    "soil_moisture_total_mm",
]

SOIL_LAYER_COLUMNS = [
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
]


def stop(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def extract_year(path: Path) -> int:
    match = re.fullmatch(
        r"India_Hydrology_(\d{4})_0p5deg_M01_12\.csv",
        path.name,
    )
    if match is None:
        stop(f"Cannot extract year from filename: {path.name}")
    return int(match.group(1))


def find_annual_files(region: str) -> list[Path]:
    raw_region_dir = RAW_DIR / region
    files = sorted(
        raw_region_dir.glob("India_Hydrology_*_0p5deg_M01_12.csv"),
        key=extract_year,
    )

    if not files:
        stop(f"No annual India CSV files found in {raw_region_dir}")

    years = [extract_year(path) for path in files]

    missing = sorted(set(EXPECTED_YEARS) - set(years))
    unexpected = sorted(set(years) - set(EXPECTED_YEARS))
    duplicates = sorted(
        year for year in set(years) if years.count(year) > 1
    )

    if missing:
        stop(f"Missing years: {missing}")

    if unexpected:
        stop(f"Unexpected years: {unexpected}")

    if duplicates:
        stop(f"Duplicate years: {duplicates}")

    if len(files) != len(EXPECTED_YEARS):
        stop(
            f"{region}: expected {len(EXPECTED_YEARS)} annual files "
            f"but found {len(files)}"
        )

    return files


def read_and_validate_file(
    path: Path,
    region: str,
    reference_region_ids: set[str] | None,
) -> tuple[pd.DataFrame, dict[str, object], set[str]]:
    file_year = extract_year(path)

    frame = pd.read_csv(path, low_memory=False)

    missing_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in frame.columns
    ]

    if missing_columns:
        stop(f"{path.name} is missing columns: {missing_columns}")

    extra_columns = [
        column for column in frame.columns
        if column not in EXPECTED_COLUMNS
    ]

    if extra_columns:
        print(
            f"WARNING: {path.name} contains extra columns: "
            f"{extra_columns}"
        )

    frame = frame[EXPECTED_COLUMNS].copy()

    frame["region_id"] = frame["region_id"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    numeric_columns = [
        column for column in EXPECTED_COLUMNS
        if column not in {"region_id", "date"}
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    invalid_dates = int(frame["date"].isna().sum())
    if invalid_dates:
        stop(f"{path.name} contains {invalid_dates} invalid dates")

    content_years = sorted(
        frame["year"].dropna().astype(int).unique().tolist()
    )

    if content_years != [file_year]:
        stop(
            f"{path.name}: filename year is {file_year}, "
            f"but content contains {content_years}"
        )

    date_years = sorted(frame["date"].dt.year.unique().tolist())

    if date_years != [file_year]:
        stop(
            f"{path.name}: date column contains years {date_years}"
        )

    unique_months = sorted(
        frame["month"].dropna().astype(int).unique().tolist()
    )

    if unique_months != list(range(1, 13)):
        stop(
            f"{path.name}: expected months 1-12, "
            f"found {unique_months}"
        )

    month_date_mismatch = int(
        (
            frame["month"].astype(int)
            != frame["date"].dt.month.astype(int)
        ).sum()
    )

    if month_date_mismatch:
        stop(
            f"{path.name}: {month_date_mismatch} rows have "
            "inconsistent date and month values"
        )

    duplicate_rows = int(
        frame.duplicated(
            subset=["region_id", "date"],
            keep=False,
        ).sum()
    )

    if duplicate_rows:
        stop(
            f"{region}/{path.name}: {duplicate_rows} duplicated "
            "region_id/date rows"
        )

    region_ids = set(frame["region_id"].unique())

    if reference_region_ids is not None and region_ids != reference_region_ids:
        missing_ids = sorted(reference_region_ids - region_ids)
        new_ids = sorted(region_ids - reference_region_ids)

        stop(
            f"{region}/{path.name}: region coverage differs from the "
            f"first year. Missing IDs={len(missing_ids)}, "
            f"new IDs={len(new_ids)}"
        )

    records_per_region_id = frame.groupby("region_id").size()

    invalid_region_counts = records_per_region_id[
        records_per_region_id != 12
    ]

    if not invalid_region_counts.empty:
        stop(
            f"{region}/{path.name}: {len(invalid_region_counts)} region IDs "
            "do not contain exactly 12 records"
        )

    months_per_region_id = frame.groupby("region_id")["month"].nunique()

    invalid_month_counts = months_per_region_id[
        months_per_region_id != 12
    ]

    if not invalid_month_counts.empty:
        stop(
            f"{region}/{path.name}: {len(invalid_month_counts)} region IDs "
            "do not contain all 12 months"
        )

    missing_values = int(frame.isna().sum().sum())

    negative_counts = {
        column: int((frame[column] < 0).sum())
        for column in FLUX_STATE_COLUMNS
    }

    total_negative_values = int(sum(negative_counts.values()))

    reconstructed_runoff = (
        frame["surface_runoff_mm"]
        + frame["subsurface_runoff_mm"]
    )

    runoff_error = (
        frame["total_runoff_mm"] - reconstructed_runoff
    ).abs()

    runoff_mismatches = int(
        (
            ~np.isclose(
                frame["total_runoff_mm"],
                reconstructed_runoff,
                rtol=1e-5,
                atol=1e-4,
            )
        ).sum()
    )

    reconstructed_soil = frame[SOIL_LAYER_COLUMNS].sum(axis=1)

    soil_error = (
        frame["soil_moisture_total_mm"] - reconstructed_soil
    ).abs()

    soil_mismatches = int(
        (
            ~np.isclose(
                frame["soil_moisture_total_mm"],
                reconstructed_soil,
                rtol=1e-5,
                atol=1e-3,
            )
        ).sum()
    )

    audit = {
        "region": region,
        "file": path.name,
        "year": file_year,
        "rows": int(len(frame)),
        "region_id_count": int(frame["region_id"].nunique()),
        "date_min": frame["date"].min().strftime("%Y-%m-%d"),
        "date_max": frame["date"].max().strftime("%Y-%m-%d"),
        "missing_values": missing_values,
        "negative_values": total_negative_values,
        "duplicate_grid_date_rows": duplicate_rows,
        "runoff_identity_mismatches": runoff_mismatches,
        "runoff_identity_max_error_mm": float(runoff_error.max()),
        "soil_identity_mismatches": soil_mismatches,
        "soil_identity_max_error_mm": float(soil_error.max()),
    }

    return frame, audit, region_ids


def make_variable_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in FLUX_STATE_COLUMNS:
        values = frame[column]

        rows.append(
            {
                "variable": column,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "negative": int((values < 0).sum()),
                "minimum": float(values.min()),
                "q01": float(values.quantile(0.01)),
                "q25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "q75": float(values.quantile(0.75)),
                "q99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
                "standard_deviation": float(values.std()),
            }
        )

    return pd.DataFrame(rows)


def process_region(region: str) -> dict[str, object]:
    processed_dir = PROCESSED_ROOT / region
    results_dir = RESULTS_ROOT / region
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    files = find_annual_files(region)
    frames: list[pd.DataFrame] = []
    annual_audits: list[dict[str, object]] = []
    reference_region_ids: set[str] | None = None

    for index, path in enumerate(files, start=1):
        frame, audit, region_ids = read_and_validate_file(
            path,
            region,
            reference_region_ids,
        )
        if reference_region_ids is None:
            reference_region_ids = region_ids

        frames.append(frame)
        annual_audits.append(audit)
        print(
            f"{region} [{index:02d}/{len(files)}] {path.name} | "
            f"rows={audit['rows']:,} | "
            f"region_ids={audit['region_id_count']} | "
            f"missing={audit['missing_values']} | "
            f"negative={audit['negative_values']}"
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(
        ["region_id", "date"],
        kind="mergesort",
    ).reset_index(drop=True)

    region_id_count = int(merged["region_id"].nunique())
    expected_rows = region_id_count * EXPECTED_MONTHS_PER_REGION_ID
    if len(merged) != expected_rows:
        stop(
            f"{region}: expected {expected_rows:,} rows based on "
            f"{region_id_count} region IDs, found {len(merged):,}"
        )

    duplicate_rows = int(
        merged.duplicated(
            subset=["region_id", "date"],
            keep=False,
        ).sum()
    )
    if duplicate_rows:
        stop(f"{region}: merged data has {duplicate_rows} duplicate region/date rows")

    records_per_region_id = merged.groupby("region_id").size()
    invalid_histories = records_per_region_id[
        records_per_region_id != EXPECTED_MONTHS_PER_REGION_ID
    ]
    if not invalid_histories.empty:
        stop(
            f"{region}: {len(invalid_histories)} region IDs do not have "
            f"{EXPECTED_MONTHS_PER_REGION_ID} monthly records"
        )

    coordinate_columns = [
        "longitude",
        "latitude",
        "lon_min",
        "lon_max",
        "lat_min",
        "lat_max",
    ]
    coordinate_counts = merged.groupby("region_id")[
        coordinate_columns
    ].nunique(dropna=False)
    unstable_coordinates = coordinate_counts.gt(1).any(axis=1)
    if unstable_coordinates.any():
        stop(
            f"{region}: {int(unstable_coordinates.sum())} region IDs "
            "change coordinates or bounds over time"
        )

    runoff_abs_error = (
        merged["total_runoff_mm"]
        - merged["surface_runoff_mm"]
        - merged["subsurface_runoff_mm"]
    ).abs()
    soil_abs_error = (
        merged["soil_moisture_total_mm"]
        - merged[SOIL_LAYER_COLUMNS].sum(axis=1)
    ).abs()

    parquet_path = processed_dir / f"india_{region}_hydrology_monthly_1982_2011.parquet"
    compressed_csv_path = processed_dir / f"india_{region}_hydrology_monthly_1982_2011.csv.gz"
    annual_audit_path = results_dir / "annual_file_audit.csv"
    variable_summary_path = results_dir / "variable_summary.csv"
    coordinate_path = results_dir / "region_coordinates.csv"
    summary_path = results_dir / "audit_summary.json"

    merged.to_parquet(parquet_path, index=False, compression="snappy")
    merged.to_csv(compressed_csv_path, index=False, compression="gzip")
    pd.DataFrame(annual_audits).to_csv(annual_audit_path, index=False)
    make_variable_summary(merged).to_csv(variable_summary_path, index=False)
    (
        merged[["region_id", *coordinate_columns]]
        .drop_duplicates()
        .sort_values(["latitude", "longitude"], ascending=[False, True])
        .reset_index(drop=True)
        .to_csv(coordinate_path, index=False)
    )

    summary = {
        "status": "PASS",
        "region": region,
        "annual_file_count": len(files),
        "year_start": int(merged["year"].min()),
        "year_end": int(merged["year"].max()),
        "region_id_count": region_id_count,
        "months_per_region_id": EXPECTED_MONTHS_PER_REGION_ID,
        "total_rows": int(len(merged)),
        "total_columns": int(len(merged.columns)),
        "missing_values": int(merged.isna().sum().sum()),
        "negative_values": int(
            sum((merged[column] < 0).sum() for column in FLUX_STATE_COLUMNS)
        ),
        "duplicate_region_date_rows": duplicate_rows,
        "longitude_min": float(merged["longitude"].min()),
        "longitude_max": float(merged["longitude"].max()),
        "latitude_min": float(merged["latitude"].min()),
        "latitude_max": float(merged["latitude"].max()),
        "date_start": merged["date"].min().strftime("%Y-%m-%d"),
        "date_end": merged["date"].max().strftime("%Y-%m-%d"),
        "runoff_identity_max_error_mm": float(runoff_abs_error.max()),
        "runoff_identity_mean_error_mm": float(runoff_abs_error.mean()),
        "soil_identity_max_error_mm": float(soil_abs_error.max()),
        "soil_identity_mean_error_mm": float(soil_abs_error.mean()),
        "parquet_output": str(parquet_path),
        "compressed_csv_output": str(compressed_csv_path),
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(
        f"{region} PASS | {region_id_count} region IDs | "
        f"{len(merged):,} records | outputs: {processed_dir}, {results_dir}"
    )
    return summary


def main() -> None:
    print("INDIA 0.5-DEGREE HYDROLOGY — REGIONAL MERGE AND AUDIT")
    summaries = [process_region(region) for region in REGIONS]
    print(
        f"Completed {len(summaries)} regions: "
        f"{sum(item['total_rows'] for item in summaries):,} total regional records."
    )


if __name__ == "__main__":
    main()
