#!/usr/bin/env python3
"""Compare corrected-AET cells with nearby valid cells in India R1-R10."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

REGIONS = tuple(f"R{index}" for index in range(1, 11))
FIGURE_DIR = ROOT / "results"

N_NEIGHBOURS = 8

VARIABLES = [
    "precipitation_mm",
    "PET_mm",
    "actual_evapotranspiration_mm_raw",
    "total_runoff_mm",
    "soil_moisture_total_mm",
]

PLOT_VARIABLES = [
    (
        "precipitation_mm",
        "Precipitation",
        "mm month$^{-1}$",
    ),
    (
        "actual_evapotranspiration_mm_raw",
        "Actual evapotranspiration",
        "mm month$^{-1}$",
    ),
    (
        "total_runoff_mm",
        "Total runoff",
        "mm month$^{-1}$",
    ),
    (
        "soil_moisture_total_mm",
        "Total soil moisture",
        "mm",
    ),
]


def safe_correlation(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    mask = np.isfinite(first) & np.isfinite(second)

    first_valid = first[mask]
    second_valid = second[mask]

    if len(first_valid) < 3:
        return float("nan")

    if np.std(first_valid) == 0 or np.std(second_valid) == 0:
        return float("nan")

    return float(
        np.corrcoef(first_valid, second_valid)[0, 1]
    )


def calculate_distance_km(
    longitude_1: float,
    latitude_1: float,
    longitude_2: pd.Series,
    latitude_2: pd.Series,
) -> pd.Series:
    """
    Approximate local horizontal distance over India.

    One degree latitude is approximately 111.32 km.
    Longitude distance is adjusted by mean latitude.
    """
    mean_latitude = np.deg2rad(
        (latitude_1 + latitude_2) / 2.0
    )

    delta_x = (
        (longitude_2 - longitude_1)
        * 111.32
        * np.cos(mean_latitude)
    )

    delta_y = (
        latitude_2 - latitude_1
    ) * 111.32

    return np.sqrt(delta_x**2 + delta_y**2)


def find_neighbours(
    grid_lookup: pd.DataFrame,
    suspect_id: str,
    excluded_ids: set[str] | None = None,
) -> pd.DataFrame:
    suspect = grid_lookup.loc[
        grid_lookup["grid_id"] == suspect_id
    ]

    if len(suspect) != 1:
        raise RuntimeError(
            f"Expected one row for {suspect_id}, "
            f"found {len(suspect)}."
        )

    suspect_row = suspect.iloc[0]

    excluded_ids = excluded_ids or set()
    candidates = grid_lookup.loc[
        (grid_lookup["grid_id"] != suspect_id)
        & ~grid_lookup["grid_id"].isin(excluded_ids)
    ].copy()

    candidates["distance_km"] = calculate_distance_km(
        longitude_1=float(suspect_row["longitude"]),
        latitude_1=float(suspect_row["latitude"]),
        longitude_2=candidates["longitude"],
        latitude_2=candidates["latitude"],
    )

    neighbours = (
        candidates
        .sort_values(
            ["distance_km", "grid_id"]
        )
        .head(N_NEIGHBOURS)
        .reset_index(drop=True)
    )

    neighbours.insert(
        0,
        "suspect_grid_id",
        suspect_id,
    )

    neighbours.insert(
        1,
        "neighbour_rank",
        np.arange(1, len(neighbours) + 1),
    )

    return neighbours


def build_neighbour_reference(
    frame: pd.DataFrame,
    neighbour_ids: list[str],
) -> pd.DataFrame:
    neighbour_data = frame.loc[
        frame["grid_id"].isin(neighbour_ids),
        ["date", *VARIABLES],
    ].copy()

    reference = (
        neighbour_data.groupby(
            "date",
            as_index=False,
            observed=True,
        )[VARIABLES]
        .median()
    )

    reference = reference.rename(
        columns={
            variable: f"neighbour_median__{variable}"
            for variable in VARIABLES
        }
    )

    return reference


def calculate_metrics(
    suspect_data: pd.DataFrame,
    reference: pd.DataFrame,
    suspect_id: str,
) -> list[dict[str, object]]:
    combined = suspect_data.merge(
        reference,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    if len(combined) != 360:
        raise RuntimeError(
            f"{suspect_id}: expected 360 matched dates, "
            f"found {len(combined)}."
        )

    rows: list[dict[str, object]] = []

    for variable in VARIABLES:
        suspect_values = combined[variable].to_numpy(
            dtype=float
        )

        reference_values = combined[
            f"neighbour_median__{variable}"
        ].to_numpy(dtype=float)

        difference = suspect_values - reference_values

        rmse = float(
            np.sqrt(np.mean(difference**2))
        )

        reference_mean = float(
            np.mean(reference_values)
        )

        reference_range = float(
            np.max(reference_values)
            - np.min(reference_values)
        )

        normalized_rmse_mean = (
            rmse / abs(reference_mean)
            if abs(reference_mean) > 1e-12
            else float("nan")
        )

        normalized_rmse_range = (
            rmse / reference_range
            if reference_range > 1e-12
            else float("nan")
        )

        rows.append(
            {
                "suspect_grid_id": suspect_id,
                "variable": variable,
                "suspect_mean": float(
                    np.mean(suspect_values)
                ),
                "neighbour_median_mean": reference_mean,
                "mean_bias": float(
                    np.mean(difference)
                ),
                "mean_absolute_error": float(
                    np.mean(np.abs(difference))
                ),
                "rmse": rmse,
                "normalized_rmse_by_neighbour_mean": (
                    normalized_rmse_mean
                ),
                "normalized_rmse_by_neighbour_range": (
                    normalized_rmse_range
                ),
                "pearson_correlation": safe_correlation(
                    suspect_values,
                    reference_values,
                ),
                "suspect_minimum": float(
                    np.min(suspect_values)
                ),
                "suspect_maximum": float(
                    np.max(suspect_values)
                ),
                "neighbour_minimum": float(
                    np.min(reference_values)
                ),
                "neighbour_maximum": float(
                    np.max(reference_values)
                ),
            }
        )

    return rows


def calculate_climatology(
    suspect_data: pd.DataFrame,
    reference: pd.DataFrame,
    suspect_id: str,
) -> pd.DataFrame:
    combined = suspect_data.merge(
        reference,
        on="date",
        how="inner",
        validate="one_to_one",
    )

    combined["month"] = combined["date"].dt.month

    climatology_rows = []

    for month, group in combined.groupby(
        "month",
        observed=True,
    ):
        for variable in VARIABLES:
            climatology_rows.append(
                {
                    "suspect_grid_id": suspect_id,
                    "month": int(month),
                    "variable": variable,
                    "suspect_climatology": float(
                        group[variable].mean()
                    ),
                    "neighbour_median_climatology": float(
                        group[
                            f"neighbour_median__{variable}"
                        ].mean()
                    ),
                    "suspect_standard_deviation": float(
                        group[variable].std()
                    ),
                    "neighbour_standard_deviation": float(
                        group[
                            f"neighbour_median__{variable}"
                        ].std()
                    ),
                }
            )

    return pd.DataFrame(climatology_rows)


def make_figure(
    climatology: pd.DataFrame,
    suspect_lookup: pd.DataFrame,
) -> None:
    suspect_ids = suspect_lookup["grid_id"].tolist()

    figure, axes = plt.subplots(
        nrows=len(PLOT_VARIABLES),
        ncols=len(suspect_ids),
        figsize=(max(10.5, 2.4 * len(suspect_ids)), 10.5),
        sharex=True,
        constrained_layout=True,
    )

    if len(suspect_ids) == 1:
        axes = np.asarray(axes).reshape(
            len(PLOT_VARIABLES),
            1,
        )

    for column_index, suspect_id in enumerate(
        suspect_ids
    ):
        coordinate = suspect_lookup.loc[
            suspect_lookup["grid_id"] == suspect_id
        ].iloc[0]

        for row_index, (
            variable,
            title,
            unit,
        ) in enumerate(PLOT_VARIABLES):
            axis = axes[row_index, column_index]

            subset = climatology.loc[
                (
                    climatology["suspect_grid_id"]
                    == suspect_id
                )
                & (
                    climatology["variable"]
                    == variable
                )
            ].sort_values("month")

            axis.plot(
                subset["month"],
                subset["suspect_climatology"],
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color="black",
                label="Suspect cell",
            )

            axis.plot(
                subset["month"],
                subset[
                    "neighbour_median_climatology"
                ],
                marker="s",
                linewidth=1.2,
                markersize=3.0,
                linestyle="--",
                color="0.45",
                label="Median of 8 neighbours",
            )

            axis.set_xticks(range(1, 13))
            axis.grid(
                linewidth=0.35,
                linestyle=":",
                alpha=0.6,
            )

            if column_index == 0:
                axis.set_ylabel(
                    f"{title}\n({unit})"
                )

            if row_index == 0:
                axis.set_title(
                    f"{suspect_id}\n"
                    f"({coordinate['longitude']:.2f}°E, "
                    f"{coordinate['latitude']:.2f}°N)"
                )

            if row_index == len(PLOT_VARIABLES) - 1:
                axis.set_xlabel("Month")

            if row_index == 0 and column_index == 0:
                axis.legend(
                    frameon=True,
                    fontsize=8,
                    loc="best",
                )

    figure.suptitle(
        "Monthly climatology of corrected-AET cells "
        "and their nearest neighbours",
        fontsize=12,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig03_negative_aet_neighbours.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig03_negative_aet_neighbours.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def process_region(region: str) -> dict[str, object]:
    global FIGURE_DIR

    processed_dir = ROOT / "data" / "processed" / region
    results_dir = ROOT / "results" / region
    FIGURE_DIR = results_dir / "figures"
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    input_path = (
        processed_dir
        / f"india_{region}_hydrology_monthly_1982_2011_qc.parquet"
    )
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    frame = pd.read_parquet(input_path)
    frame["date"] = pd.to_datetime(frame["date"])
    frame["grid_id"] = frame["region_id"].astype(str)

    complete_cell_mask = frame.groupby("grid_id")[VARIABLES].apply(
        lambda values: values.notna().all().all()
    )
    complete_ids = set(complete_cell_mask[complete_cell_mask].index)
    grid_lookup = (
        frame[["grid_id", "longitude", "latitude"]]
        .drop_duplicates()
        .loc[lambda lookup: lookup["grid_id"].isin(complete_ids)]
        .sort_values("grid_id")
        .reset_index(drop=True)
    )

    all_suspect_ids = sorted(
        frame.loc[frame["aet_negative_corrected"], "grid_id"]
        .unique()
        .tolist()
    )
    suspect_ids = [
        suspect_id for suspect_id in all_suspect_ids
        if suspect_id in complete_ids
    ]
    skipped_suspect_ids = sorted(set(all_suspect_ids) - set(suspect_ids))

    if not suspect_ids:
        decision_summary = {
            "status": "PASS",
            "region": region,
            "suspect_cell_count": 0,
            "suspect_region_ids": [],
            "skipped_incomplete_suspect_region_ids": skipped_suspect_ids,
            "neighbours_per_suspect_cell": N_NEIGHBOURS,
            "diagnostic_only": True,
            "cells_removed": 0,
        }
        (results_dir / "negative_aet_neighbour_diagnostic.json").write_text(
            json.dumps(decision_summary, indent=2),
            encoding="utf-8",
        )
        print(f"{region}: no corrected-AET cells; diagnostic skipped.")
        return decision_summary

    print(f"{region}: diagnosing {len(suspect_ids)} corrected-AET cells")
    neighbour_tables = []
    metric_rows = []
    climatology_tables = []
    suspect_summary_rows = []

    for suspect_id in suspect_ids:
        suspect_data = (
            frame.loc[
                frame["grid_id"] == suspect_id,
                [
                    "grid_id",
                    "longitude",
                    "latitude",
                    "date",
                    "aet_negative_corrected",
                    *VARIABLES,
                ],
            ]
            .sort_values("date")
            .reset_index(drop=True)
        )
        neighbours = find_neighbours(
            grid_lookup,
            suspect_id,
            excluded_ids=set(suspect_ids),
        )
        if neighbours.empty:
            skipped_suspect_ids.append(suspect_id)
            continue

        neighbour_tables.append(neighbours)
        neighbour_ids = neighbours["grid_id"].tolist()
        reference = build_neighbour_reference(frame, neighbour_ids)
        metric_rows.extend(
            calculate_metrics(suspect_data, reference, suspect_id)
        )
        climatology_tables.append(
            calculate_climatology(suspect_data, reference, suspect_id)
        )
        suspect_summary_rows.append(
            {
                "region_id": suspect_id,
                "longitude": float(suspect_data["longitude"].iloc[0]),
                "latitude": float(suspect_data["latitude"].iloc[0]),
                "corrected_aet_records": int(
                    suspect_data["aet_negative_corrected"].sum()
                ),
                "corrected_fraction": float(
                    suspect_data["aet_negative_corrected"].mean()
                ),
                "minimum_raw_aet_mm": float(
                    suspect_data["actual_evapotranspiration_mm_raw"].min()
                ),
                "median_total_soil_moisture_mm": float(
                    suspect_data["soil_moisture_total_mm"].median()
                ),
                "median_total_runoff_mm": float(
                    suspect_data["total_runoff_mm"].median()
                ),
            }
        )
        print(f"  {suspect_id}: {int(suspect_data['aet_negative_corrected'].sum())} corrected records")

    neighbour_table = (
        pd.concat(neighbour_tables, ignore_index=True)
        if neighbour_tables
        else pd.DataFrame()
    )
    metrics = pd.DataFrame(metric_rows)
    climatology = (
        pd.concat(climatology_tables, ignore_index=True)
        if climatology_tables
        else pd.DataFrame()
    )
    suspect_summary = pd.DataFrame(suspect_summary_rows)

    neighbour_table = neighbour_table.rename(
        columns={"suspect_grid_id": "suspect_region_id", "grid_id": "neighbour_region_id"}
    )
    metrics = metrics.rename(columns={"suspect_grid_id": "region_id"})
    climatology = climatology.rename(columns={"suspect_grid_id": "region_id"})

    neighbour_table.to_csv(results_dir / "negative_aet_neighbours.csv", index=False)
    metrics.to_csv(results_dir / "negative_aet_neighbour_metrics.csv", index=False)
    climatology.to_csv(results_dir / "negative_aet_monthly_climatology.csv", index=False)
    suspect_summary.to_csv(results_dir / "negative_aet_cell_summary.csv", index=False)

    if not climatology.empty:
        suspect_lookup = grid_lookup.loc[
            grid_lookup["grid_id"].isin(suspect_summary["region_id"])
        ].copy()
        make_figure(climatology.rename(columns={"region_id": "suspect_grid_id"}), suspect_lookup)

    decision_summary = {
        "status": "PASS",
        "region": region,
        "suspect_cell_count": len(suspect_ids) - len(skipped_suspect_ids),
        "suspect_region_ids": suspect_ids,
        "skipped_incomplete_suspect_region_ids": sorted(set(skipped_suspect_ids)),
        "neighbours_per_suspect_cell": N_NEIGHBOURS,
        "diagnostic_only": True,
        "cells_removed": 0,
        "recommendation": "Review per-cell neighbor metrics before any exclusion decision.",
    }
    summary_path = results_dir / "negative_aet_neighbour_diagnostic.json"
    summary_path.write_text(json.dumps(decision_summary, indent=2), encoding="utf-8")
    print(f"{region} PASS | diagnosed={decision_summary['suspect_cell_count']} | outputs={results_dir}")
    return decision_summary


def main() -> None:
    summaries = [process_region(region) for region in REGIONS]
    print(
        f"Completed diagnostics for {len(summaries)} regions; "
        f"{sum(int(summary['suspect_cell_count']) for summary in summaries)} "
        "corrected-AET cells were assessed."
    )


if __name__ == "__main__":
    main()
