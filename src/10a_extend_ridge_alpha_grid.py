#!/usr/bin/env python3
"""
Extend Stage-10 Ridge alpha selection and evaluate the intercept-only limit.

This loads the existing Stage-10 functions and evaluates geographic CV
without repeating the coefficient bootstrap.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT = (
    ROOT
    / "src"
    / "10_component_lag_depth_attribution.py"
)

TABLE_DIR = ROOT / "results" / "tables"

EXTENDED_ALPHA_GRID = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
    30000.0,
    100000.0,
    300000.0,
    1000000.0,
]


def load_stage10_module():
    specification = (
        importlib.util.spec_from_file_location(
            "stage10_attribution",
            SOURCE_SCRIPT,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not load {SOURCE_SCRIPT}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    specification.loader.exec_module(
        module
    )

    return module


def validate_attribution_data(
    data: pd.DataFrame,
    stage10,
) -> tuple[int, list[int]]:
    required_columns = [
        "region_id",
        "month",
        "spatial_fold_5",
        *stage10.FEATURE_COLUMNS,
        *stage10.TARGET_ORDER,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]
    if missing_columns:
        raise RuntimeError(
            "Attribution dataset is missing required columns: "
            f"{missing_columns}."
        )

    if data.empty:
        raise RuntimeError(
            "Attribution dataset contains no observations."
        )

    if data[required_columns].isna().any().any():
        raise RuntimeError(
            "Attribution dataset contains missing metadata, "
            "features, or target values."
        )

    region_month_counts = data.groupby(
        "region_id",
        observed=True,
    )["month"].agg(
        row_count="size",
        month_count="nunique",
    )
    if (
        (region_month_counts["row_count"] != 12).any()
        or (region_month_counts["month_count"] != 12).any()
        or set(data["month"].unique()) != set(range(1, 13))
    ):
        raise RuntimeError(
            "Each spatial unit must contain 12 unique monthly "
            "observations numbered 1 through 12."
        )

    region_fold_counts = data.groupby(
        "region_id",
        observed=True,
    )["spatial_fold_5"].nunique()
    if (region_fold_counts != 1).any():
        raise RuntimeError(
            "A spatial unit is assigned to multiple geographic folds."
        )

    folds = sorted(
        data["spatial_fold_5"]
        .astype(int)
        .unique()
        .tolist()
    )
    if folds != list(range(stage10.SPATIAL_FOLD_COUNT)):
        raise RuntimeError(
            f"Expected geographic folds "
            f"0-{stage10.SPATIAL_FOLD_COUNT - 1}, found {folds}."
        )

    return int(data["region_id"].nunique()), folds


def intercept_only_validation_rmse(
    data: pd.DataFrame,
    target: str,
    stage10,
) -> float:
    fold_scores = []
    for validation_fold in range(
        stage10.SPATIAL_FOLD_COUNT
    ):
        train = data.loc[
            data["spatial_fold_5"]
            != validation_fold
        ]
        validation = data.loc[
            data["spatial_fold_5"]
            == validation_fold
        ]
        prediction = np.full(
            len(validation),
            train[target].mean(),
            dtype=float,
        )
        fold_scores.append(
            stage10.rmse(
                validation[target].to_numpy(
                    dtype=float
                ),
                prediction,
            )
        )

    return float(
        sum(fold_scores) / len(fold_scores)
    )


def main() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage10 = load_stage10_module()

    stage10.ALPHA_GRID = (
        EXTENDED_ALPHA_GRID
    )

    if not stage10.DATA_PATH.exists():
        raise FileNotFoundError(
            stage10.DATA_PATH
        )

    if (
        not EXTENDED_ALPHA_GRID
        or any(
            not math.isfinite(alpha) or alpha < 0
            for alpha in EXTENDED_ALPHA_GRID
        )
        or EXTENDED_ALPHA_GRID
        != sorted(set(EXTENDED_ALPHA_GRID))
    ):
        raise RuntimeError(
            "Extended Ridge alpha grid must contain unique, "
            "finite, non-negative values in ascending order."
        )

    data = (
        pd.read_parquet(
            stage10.DATA_PATH
        )
        .sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
    )

    region_count, folds = validate_attribution_data(
        data,
        stage10,
    )

    all_fold_tables = []
    all_alpha_tables = []
    decision_rows = []

    print("=" * 78)
    print("EXTENDED RIDGE ALPHA BOUNDARY CHECK")
    print("=" * 78)
    print(
        f"Extended alpha grid: "
        f"{EXTENDED_ALPHA_GRID}"
    )
    print(f"Spatial units      : {region_count}")
    print(f"Region-month rows  : {len(data)}")
    print(f"Spatial folds      : {folds}")
    print()

    for target in stage10.TARGET_ORDER:
        (
            selected_alpha,
            fold_table,
            alpha_table,
        ) = stage10.select_ridge_alpha(
            data,
            target,
        )

        all_fold_tables.append(
            fold_table
        )

        all_alpha_tables.append(
            alpha_table
        )

        best_index = (
            alpha_table[
                "mean_validation_rmse_mm"
            ].idxmin()
        )

        best_row = alpha_table.loc[
            best_index
        ]

        one_standard_error_threshold = float(
            best_row["mean_validation_rmse_mm"]
            + best_row[
                "standard_error_validation_rmse_mm"
            ]
        )

        intercept_rmse = intercept_only_validation_rmse(
            data,
            target,
            stage10,
        )
        intercept_only_selected = bool(
            intercept_rmse
            <= one_standard_error_threshold
        )
        maximum_alpha = max(
            EXTENDED_ALPHA_GRID
        )

        finite_alpha_boundary_hit = bool(
            selected_alpha
            == maximum_alpha
        )
        if intercept_only_selected:
            selected_policy = "intercept_only"
            reported_alpha = float("nan")
            selected_rmse = intercept_rmse
        else:
            selected_policy = "finite_ridge_alpha"
            reported_alpha = float(selected_alpha)
            selected_row = alpha_table.loc[
                alpha_table["alpha"]
                == selected_alpha
            ].iloc[0]
            selected_rmse = float(
                selected_row[
                    "mean_validation_rmse_mm"
                ]
            )
        unresolved_boundary = bool(
            finite_alpha_boundary_hit
            and not intercept_only_selected
        )

        decision_rows.append(
            {
                "target": target,
                "depth_label": (
                    stage10.TARGETS[target]
                ),
                "minimum_mean_rmse_alpha": float(
                    best_row["alpha"]
                ),
                "minimum_mean_validation_rmse_mm": float(
                    best_row[
                        "mean_validation_rmse_mm"
                    ]
                ),
                "one_standard_error_threshold_mm": (
                    one_standard_error_threshold
                ),
                "intercept_only_mean_validation_rmse_mm": (
                    intercept_rmse
                ),
                "intercept_only_within_one_standard_error": (
                    intercept_only_selected
                ),
                "one_standard_error_selection": (
                    selected_policy
                ),
                "selected_one_standard_error_alpha": (
                    reported_alpha
                ),
                "selected_alpha_mean_validation_rmse_mm": (
                    selected_rmse
                ),
                "selected_alpha_at_upper_boundary": (
                    unresolved_boundary
                ),
                "finite_grid_one_standard_error_alpha_at_upper_boundary": (
                    finite_alpha_boundary_hit
                ),
            }
        )

        print(
            f"{stage10.TARGETS[target]:>10s}: "
            f"best-mean alpha="
            f"{float(best_row['alpha']):g}, "
            f"one-SE policy={selected_policy}, "
            f"alpha={reported_alpha:g}, "
            f"finite-grid boundary="
            f"{finite_alpha_boundary_hit}"
        )

    fold_output = pd.concat(
        all_fold_tables,
        ignore_index=True,
    )

    alpha_output = pd.concat(
        all_alpha_tables,
        ignore_index=True,
    )

    decision_output = pd.DataFrame(
        decision_rows
    )

    fold_path = (
        TABLE_DIR
        / "ridge_alpha_extended_cv_folds.csv"
    )

    alpha_path = (
        TABLE_DIR
        / "ridge_alpha_extended_selection.csv"
    )

    decision_path = (
        TABLE_DIR
        / "ridge_alpha_extended_decision.csv"
    )

    metadata_path = (
        TABLE_DIR
        / "ridge_alpha_extended_metadata.json"
    )

    fold_output.to_csv(
        fold_path,
        index=False,
    )

    alpha_output.to_csv(
        alpha_path,
        index=False,
    )

    decision_output.to_csv(
        decision_path,
        index=False,
    )

    boundary_targets = (
        decision_output.loc[
            decision_output[
                "selected_alpha_at_upper_boundary"
            ],
            "target",
        ].tolist()
    )
    intercept_only_targets = (
        decision_output.loc[
            decision_output[
                "one_standard_error_selection"
            ]
            == "intercept_only",
            "target",
        ].tolist()
    )

    metadata = {
        "status": (
            "BOUNDARY_REMAINS"
            if boundary_targets
            else "PASS"
        ),
        "extended_alpha_grid": (
            EXTENDED_ALPHA_GRID
        ),
        "spatial_unit_count": region_count,
        "spatial_unit_month_row_count": int(len(data)),
        "spatial_folds": folds,
        "targets_still_at_upper_boundary": (
            boundary_targets
        ),
        "targets_using_intercept_only_sensitivity": (
            intercept_only_targets
        ),
        "one_standard_error_policy": (
            "Choose intercept-only when its geographic-CV RMSE is "
            "within the best finite Ridge alpha's one-standard-error "
            "threshold; otherwise use the largest eligible finite "
            "Ridge alpha."
        ),
        "decision_table": str(
            decision_path
        ),
        "alpha_selection_table": str(
            alpha_path
        ),
        "fold_table": str(
            fold_path
        ),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Extended alpha decisions:")
    print(
        decision_output.to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.5f}"
            ),
        )
    )

    print()
    print(f"Status         : {metadata['status']}")
    print(f"Decision table : {decision_path}")
    print(f"Metadata       : {metadata_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
