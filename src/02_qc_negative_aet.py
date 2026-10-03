#!/usr/bin/env python3
"""Record and correct negative AET values independently for India R1-R10."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REGIONS = tuple(f"R{index}" for index in range(1, 11))

AET_COLUMN = "actual_evapotranspiration_mm"
RAW_AET_COLUMN = "actual_evapotranspiration_mm_raw"
FLAG_COLUMN = "aet_negative_corrected"


def process_region(region: str) -> dict[str, object]:
    processed_dir = ROOT / "data" / "processed" / region
    results_dir = ROOT / "results" / region
    input_path = (
        processed_dir / f"india_{region}_hydrology_monthly_1982_2011.parquet"
    )
    output_parquet = (
        processed_dir
        / f"india_{region}_hydrology_monthly_1982_2011_qc.parquet"
    )
    output_csv = (
        processed_dir
        / f"india_{region}_hydrology_monthly_1982_2011_qc.csv.gz"
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset does not exist: {input_path}"
        )

    frame = pd.read_parquet(input_path)

    negative_mask = frame[AET_COLUMN] < 0
    negative_records = frame.loc[
        negative_mask,
        [
            "region_id",
            "longitude",
            "latitude",
            "date",
            "year",
            "month",
            "precipitation_mm",
            "PET_mm",
            AET_COLUMN,
            "total_runoff_mm",
            "soil_moisture_total_mm",
        ],
    ].copy()

    negative_count = int(negative_mask.sum())
    total_count = int(len(frame))
    negative_percentage = (
        100.0 * negative_count / total_count if total_count else 0.0
    )

    if negative_count == 0:
        minimum_negative = None
        maximum_negative = None
        mean_negative = None
        median_negative = None
    else:
        values = negative_records[AET_COLUMN]
        minimum_negative = float(values.min())
        maximum_negative = float(values.max())
        mean_negative = float(values.mean())
        median_negative = float(values.median())

    # Preserve the unmodified AET values.
    frame[RAW_AET_COLUMN] = frame[AET_COLUMN]

    # Flag records corrected by the physical non-negativity rule.
    frame[FLAG_COLUMN] = negative_mask

    # Apply the correction only to the working AET variable.
    frame[AET_COLUMN] = frame[AET_COLUMN].clip(lower=0.0)

    remaining_negative_count = int(
        (frame[AET_COLUMN] < 0).sum()
    )

    corrected_difference = (
        frame[AET_COLUMN] - frame[RAW_AET_COLUMN]
    )

    negative_records = negative_records.sort_values(
        ["year", "month", "latitude", "longitude"]
    ).reset_index(drop=True)

    by_year = (
        negative_records.groupby("year", as_index=False)
        .agg(
            negative_record_count=(AET_COLUMN, "size"),
            minimum_negative_aet_mm=(AET_COLUMN, "min"),
            mean_negative_aet_mm=(AET_COLUMN, "mean"),
            maximum_negative_aet_mm=(AET_COLUMN, "max"),
        )
        .sort_values("year")
    )

    by_month = (
        negative_records.groupby("month", as_index=False)
        .agg(
            negative_record_count=(AET_COLUMN, "size"),
            minimum_negative_aet_mm=(AET_COLUMN, "min"),
            mean_negative_aet_mm=(AET_COLUMN, "mean"),
            maximum_negative_aet_mm=(AET_COLUMN, "max"),
        )
        .sort_values("month")
    )

    negative_records.to_csv(
        results_dir / "negative_aet_records.csv",
        index=False,
    )

    by_year.to_csv(
        results_dir / "negative_aet_by_year.csv",
        index=False,
    )

    by_month.to_csv(
        results_dir / "negative_aet_by_month.csv",
        index=False,
    )

    frame.to_parquet(
        output_parquet,
        index=False,
        compression="snappy",
    )

    frame.to_csv(
        output_csv,
        index=False,
        compression="gzip",
    )

    summary = {
        "status": (
            "PASS"
            if remaining_negative_count == 0
            else "FAIL"
        ),
        "region": region,
        "input_dataset": str(input_path),
        "output_parquet": str(output_parquet),
        "output_compressed_csv": str(output_csv),
        "total_records": total_count,
        "negative_aet_records_before_correction": negative_count,
        "negative_aet_percentage": negative_percentage,
        "minimum_negative_aet_mm": minimum_negative,
        "maximum_negative_aet_mm": maximum_negative,
        "mean_negative_aet_mm": mean_negative,
        "median_negative_aet_mm": median_negative,
        "remaining_negative_aet_records": (
            remaining_negative_count
        ),
        "maximum_absolute_correction_mm": float(
            corrected_difference.abs().max()
        ),
        "mean_absolute_correction_all_records_mm": float(
            corrected_difference.abs().mean()
        ),
        "correction_rule": (
            "actual_evapotranspiration_mm = "
            "max(actual_evapotranspiration_mm_raw, 0)"
        ),
        "raw_values_preserved_in_column": RAW_AET_COLUMN,
        "corrected_records_flagged_in_column": FLAG_COLUMN,
    }

    summary_path = results_dir / "negative_aet_qc_summary.json"

    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(f"{region} NEGATIVE AET QUALITY CONTROL")
    print(f"Status                         : {summary['status']}")
    print(f"Total records                  : {total_count:,}")
    print(f"Negative AET records           : {negative_count}")
    print(
        f"Percentage of dataset          : "
        f"{negative_percentage:.6f}%"
    )
    print(f"Minimum negative AET           : {minimum_negative}")
    print(f"Mean negative AET              : {mean_negative}")
    print(f"Maximum negative AET           : {maximum_negative}")
    print(
        f"Remaining negative values      : "
        f"{remaining_negative_count}"
    )
    print(
        f"Maximum absolute correction    : "
        f"{summary['maximum_absolute_correction_mm']:.8f} mm"
    )
    print()
    print(f"QC Parquet dataset             : {output_parquet}")
    print(f"QC compressed CSV              : {output_csv}")
    print(
        f"Negative-record table          : "
        f"{results_dir / 'negative_aet_records.csv'}"
    )
    print(
        f"Year summary                   : "
        f"{results_dir / 'negative_aet_by_year.csv'}"
    )
    print(
        f"Month summary                  : "
        f"{results_dir / 'negative_aet_by_month.csv'}"
    )
    print(f"QC summary                     : {summary_path}")

    if remaining_negative_count != 0:
        raise SystemExit(
            f"{region}: negative AET values remain after correction."
        )
    return summary


def main() -> None:
    summaries = [process_region(region) for region in REGIONS]
    corrected_count = sum(
        int(summary["negative_aet_records_before_correction"])
        for summary in summaries
    )
    print(
        f"Completed {len(summaries)} regions; "
        f"{corrected_count:,} negative AET records flagged and corrected."
    )


if __name__ == "__main__":
    main()
