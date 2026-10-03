"""Validate India 0.5-degree cells and prepare complete per-region data."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGIONS = tuple(f"R{index}" for index in range(1, 11))
EXPECTED_YEARS = list(range(1982, 2012))
EXPECTED_MONTHS = 12 * len(EXPECTED_YEARS)
CELL_SIZE_DEG = 0.5

HYDRO_COLUMNS = [
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm",
    "actual_evapotranspiration_mm_raw",
    "surface_runoff_mm",
    "subsurface_runoff_mm",
    "total_runoff_mm",
    "soil_moisture_0_10cm_mm",
    "soil_moisture_10_40cm_mm",
    "soil_moisture_40_100cm_mm",
    "soil_moisture_100_200cm_mm",
    "soil_moisture_total_mm",
]

COORDINATE_COLUMNS = [
    "longitude",
    "latitude",
    "lon_min",
    "lon_max",
    "lat_min",
    "lat_max",
]


def prepare_region(region: str) -> dict[str, object]:
    processed_dir = ROOT / "data" / "processed" / region
    results_dir = ROOT / "results" / region
    processed_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    input_path = (
        processed_dir / f"india_{region}_hydrology_monthly_1982_2011_qc.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(f"QC dataset does not exist: {input_path}")

    frame = pd.read_parquet(input_path)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")

    required_columns = {
        "region_id",
        "date",
        "year",
        "month",
        *COORDINATE_COLUMNS,
        "aet_negative_corrected",
        *HYDRO_COLUMNS,
    }
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{region} dataset is missing columns: {missing_columns}")

    if frame["date"].isna().any():
        raise ValueError(f"{region} dataset contains invalid dates")

    if sorted(frame["year"].astype(int).unique()) != EXPECTED_YEARS:
        raise ValueError(f"{region} dataset does not cover 1982-2011")

    duplicate_count = int(
        frame.duplicated(["region_id", "date"], keep=False).sum()
    )
    if duplicate_count:
        raise ValueError(f"{region} has {duplicate_count} duplicate region/date rows")

    bounds_valid = (
        ((frame["lon_max"] - frame["lon_min"] - CELL_SIZE_DEG).abs() < 1e-8)
        & ((frame["lat_max"] - frame["lat_min"] - CELL_SIZE_DEG).abs() < 1e-8)
    )
    if not bounds_valid.all():
        raise ValueError(f"{region} contains cells that are not 0.5 degrees square")

    coordinates = frame.groupby("region_id", observed=True)[
        COORDINATE_COLUMNS
    ].nunique(dropna=False)
    if coordinates.gt(1).any(axis=None):
        raise ValueError(f"{region} has region IDs with changing coordinates")

    cell_audit = (
        frame.assign(_missing_values=frame[HYDRO_COLUMNS].isna().sum(axis=1))
        .groupby("region_id", as_index=False, observed=True)
        .agg(
            records=("date", "size"),
            unique_months=("date", "nunique"),
            missing_values=("_missing_values", "sum"),
        )
    )
    cell_audit["complete"] = (
        (cell_audit["records"] == EXPECTED_MONTHS)
        & (cell_audit["unique_months"] == EXPECTED_MONTHS)
        & (cell_audit["missing_values"] == 0)
    )
    incomplete = cell_audit.loc[~cell_audit["complete"]].copy()
    incomplete["reason"] = incomplete.apply(
        lambda row: "; ".join(
            reason
            for condition, reason in [
                (row.records != EXPECTED_MONTHS, "incomplete monthly history"),
                (row.missing_values > 0, "missing hydrology values"),
            ]
            if condition
        ),
        axis=1,
    )
    complete_ids = set(cell_audit.loc[cell_audit["complete"], "region_id"])
    if not complete_ids:
        raise ValueError(f"{region} has no complete grid IDs for analysis")

    analysis = frame.loc[frame["region_id"].isin(complete_ids)].copy()
    analysis["native_cell_count"] = 1
    analysis["longitude_mean"] = analysis["longitude"]
    analysis["latitude_mean"] = analysis["latitude"]
    analysis["region_lon_index"] = (analysis["lon_min"] / CELL_SIZE_DEG).round().astype(int)
    analysis["region_lat_index"] = (analysis["lat_min"] / CELL_SIZE_DEG).round().astype(int)
    analysis["region_lon_min"] = analysis["lon_min"]
    analysis["region_lon_max"] = analysis["lon_max"]
    analysis["region_lat_min"] = analysis["lat_min"]
    analysis["region_lat_max"] = analysis["lat_max"]
    analysis["region_center_lon"] = analysis["longitude"]
    analysis["region_center_lat"] = analysis["latitude"]
    analysis["regional_area_weight_sum"] = 1.0
    analysis["aet_corrected_native_record_count"] = analysis[
        "aet_negative_corrected"
    ].astype(int)
    analysis["small_region_lt_4_cells"] = False
    analysis["very_small_region_lt_2_cells"] = False

    inventory = (
        analysis.groupby("region_id", as_index=False, observed=True)
        .agg(
            longitude_mean=("longitude", "first"),
            latitude_mean=("latitude", "first"),
            longitude_min_native=("longitude", "min"),
            longitude_max_native=("longitude", "max"),
            latitude_min_native=("latitude", "min"),
            latitude_max_native=("latitude", "max"),
            region_lon_index=("region_lon_index", "first"),
            region_lat_index=("region_lat_index", "first"),
            region_lon_min=("region_lon_min", "first"),
            region_lon_max=("region_lon_max", "first"),
            region_lat_min=("region_lat_min", "first"),
            region_lat_max=("region_lat_max", "first"),
            region_center_lon=("region_center_lon", "first"),
            region_center_lat=("region_center_lat", "first"),
            corrected_aet_records=("aet_negative_corrected", "sum"),
        )
        .sort_values(["region_lat_index", "region_lon_index"], ascending=[False, True])
        .reset_index(drop=True)
    )
    inventory["native_cell_count"] = 1
    inventory["corrected_aet_records"] = inventory["corrected_aet_records"].astype(int)
    inventory["small_region_lt_4_cells"] = False
    inventory["very_small_region_lt_2_cells"] = False

    grid_lookup = inventory[
        [
            "region_id",
            "longitude_mean",
            "latitude_mean",
            "region_lon_index",
            "region_lat_index",
            "region_lon_min",
            "region_lon_max",
            "region_lat_min",
            "region_lat_max",
            "region_center_lon",
            "region_center_lat",
        ]
    ].rename(
        columns={"longitude_mean": "longitude", "latitude_mean": "latitude"}
    )

    aet_summary = (
        frame.groupby(
            ["region_id", "longitude", "latitude"],
            as_index=False,
            observed=True,
        )
        .agg(
            corrected_aet_records=("aet_negative_corrected", "sum"),
            minimum_raw_aet_mm=("actual_evapotranspiration_mm_raw", "min"),
            mean_raw_aet_mm=("actual_evapotranspiration_mm_raw", "mean"),
        )
        .sort_values(
            ["corrected_aet_records", "latitude", "longitude"],
            ascending=[False, False, True],
        )
    )

    regional_path = (
        processed_dir / f"india_{region}_hydrology_regional_monthly_0p5degree.parquet"
    )
    regional_csv_path = (
        processed_dir / f"india_{region}_hydrology_regional_monthly_0p5degree.csv.gz"
    )
    analysis_path = (
        processed_dir / f"india_{region}_hydrology_monthly_1982_2011_analysis.parquet"
    )
    analysis_csv_path = (
        processed_dir / f"india_{region}_hydrology_monthly_1982_2011_analysis.csv.gz"
    )
    inventory_path = results_dir / "region_inventory_0p5degree.csv"
    grid_lookup_path = results_dir / "native_grid_to_region_0p5degree.csv"
    aet_summary_path = results_dir / "aet_corrections_by_native_grid.csv"
    incomplete_path = results_dir / "incomplete_region_ids.csv"
    summary_path = results_dir / "regionalization_summary_0p5degree.json"

    analysis = analysis.sort_values(["region_id", "date"]).reset_index(drop=True)
    analysis.to_parquet(regional_path, index=False, compression="snappy")
    analysis.to_csv(regional_csv_path, index=False, compression="gzip")
    analysis.to_parquet(analysis_path, index=False, compression="snappy")
    analysis.to_csv(analysis_csv_path, index=False, compression="gzip")
    inventory.to_csv(inventory_path, index=False)
    grid_lookup.to_csv(grid_lookup_path, index=False)
    aet_summary.to_csv(aet_summary_path, index=False)
    incomplete.to_csv(incomplete_path, index=False)

    summary = {
        "status": "PASS",
        "region": region,
        "input_dataset": str(input_path),
        "source_cell_size_degrees": CELL_SIZE_DEG,
        "source_region_id_count": int(len(cell_audit)),
        "analysis_region_id_count": int(len(complete_ids)),
        "excluded_incomplete_region_id_count": int(len(incomplete)),
        "excluded_incomplete_region_ids": incomplete["region_id"].tolist(),
        "excluded_source_rows": int(incomplete["records"].sum()),
        "source_missing_values": int(frame[HYDRO_COLUMNS].isna().sum().sum()),
        "regional_monthly_rows": int(len(analysis)),
        "months_per_region_id": EXPECTED_MONTHS,
        "total_aet_corrected_records": int(frame["aet_negative_corrected"].sum()),
        "regional_monthly_parquet": str(regional_path),
        "regional_monthly_csv": str(regional_csv_path),
        "analysis_dataset": str(analysis_path),
        "analysis_csv": str(analysis_csv_path),
        "region_inventory": str(inventory_path),
        "incomplete_region_ids": str(incomplete_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{region} PASS | complete={len(complete_ids)} | "
        f"excluded={len(incomplete)} | rows={len(analysis):,}"
    )
    return summary


def main() -> None:
    summaries = [prepare_region(region) for region in REGIONS]
    excluded_count = sum(
        int(item["excluded_incomplete_region_id_count"])
        for item in summaries
    )
    print(
        f"Prepared {len(summaries)} independent India regions; "
        f"{excluded_count} incomplete grid IDs are listed in per-region audits."
    )