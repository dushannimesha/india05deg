#!/usr/bin/env python3
"""
Finalize component-lag-depth attribution under two Ridge-alpha policies.

Alpha values are read from the extended geographic-CV decision table
created by 10a_extend_ridge_alpha_grid.py. If the one-standard-error
criterion includes the intercept-only limit, that null model is used for
sensitivity and component memory indices are reported as undefined.

Component memory is compared only when the sensitivity policy retains
finite Ridge slopes; it is undefined for intercept-only targets.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

STAGE10_PATH = (
    ROOT
    / "src"
    / "10_component_lag_depth_attribution.py"
)

TABLE_DIR = ROOT / "results" / "tables"
FIGURE_DIR = ROOT / "results" / "figures"
ALPHA_DECISION_PATH = (
    TABLE_DIR / "ridge_alpha_extended_decision.csv"
)
ALPHA_METADATA_PATH = (
    TABLE_DIR / "ridge_alpha_extended_metadata.json"
)

ARCHIVE_DIR = (
    FIGURE_DIR / "archive_stage10"
)

BOOTSTRAP_ITERATIONS = 5_000
BASE_RANDOM_SEED = 20260806

PRIMARY_POLICY = "minimum_geographic_cv_rmse"
SENSITIVITY_POLICY = "one_standard_error"

ALPHA_POLICIES: dict[
    str,
    dict[str, float | None],
] = {}


def load_stage10():
    specification = (
        importlib.util.spec_from_file_location(
            "stage10_attribution",
            STAGE10_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise RuntimeError(
            f"Could not load {STAGE10_PATH}"
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


def load_alpha_policies(
    stage10,
    *,
    expected_spatial_unit_count: int,
    expected_row_count: int,
) -> dict[str, dict[str, float | None]]:
    if not ALPHA_DECISION_PATH.exists():
        raise FileNotFoundError(
            "Run 10a_extend_ridge_alpha_grid.py first; "
            f"missing {ALPHA_DECISION_PATH}."
        )

    if not ALPHA_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Run 10a_extend_ridge_alpha_grid.py first; "
            f"missing {ALPHA_METADATA_PATH}."
        )

    with ALPHA_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as stream:
        alpha_metadata = json.load(stream)

    decision = pd.read_csv(
        ALPHA_DECISION_PATH
    )

    required_columns = {
        "target",
        "minimum_mean_rmse_alpha",
        "selected_one_standard_error_alpha",
        "selected_alpha_at_upper_boundary",
        "one_standard_error_selection",
    }
    missing_columns = required_columns - set(decision.columns)
    if missing_columns:
        raise RuntimeError(
            "Extended-alpha decision table is missing columns: "
            f"{sorted(missing_columns)}."
        )

    expected_targets = set(stage10.TARGET_ORDER)
    if (
        decision["target"].duplicated().any()
        or set(decision["target"]) != expected_targets
    ):
        raise RuntimeError(
            "Extended-alpha decision table must have exactly one "
            "row for each layer target."
        )

    if alpha_metadata.get("status") != "PASS":
        boundary_targets = alpha_metadata.get(
            "targets_still_at_upper_boundary",
            decision.loc[
                decision["selected_alpha_at_upper_boundary"].astype(bool),
                "target",
            ].tolist(),
        )
        raise RuntimeError(
            "Attribution finalization is blocked because the extended "
            "alpha search is unresolved. The finite one-standard-error "
            "alpha remains at the grid boundary and the intercept-only "
            f"model is not an eligible alternative for: "
            f"{boundary_targets}. Extend or revise the alpha search in "
            "10a_extend_ridge_alpha_grid.py, rerun it, and then rerun "
            "this script."
        )

    decision = decision.set_index("target")
    invalid_selections = decision.loc[
        ~decision["one_standard_error_selection"].isin(
            ["finite_ridge_alpha", "intercept_only"]
        )
    ]
    if not invalid_selections.empty:
        raise RuntimeError(
            "Extended-alpha decision table contains an unsupported "
            "one-standard-error policy for targets: "
            f"{invalid_selections.index.tolist()}."
        )

    boundary_flags = decision[
        "selected_alpha_at_upper_boundary"
    ].astype(bool)
    if boundary_flags.any():
        boundary_targets = decision.index[
            boundary_flags
        ].tolist()
        raise RuntimeError(
            "Attribution finalization is blocked because selected "
            "alphas remain at the upper grid boundary for: "
            f"{boundary_targets}."
        )

    if int(
        alpha_metadata.get("spatial_unit_count", -1)
    ) != expected_spatial_unit_count:
        raise RuntimeError(
            "Extended-alpha results were generated for a different "
            "number of spatial units than the current feature dataset."
        )

    if int(
        alpha_metadata.get(
            "spatial_unit_month_row_count",
            -1,
        )
    ) != expected_row_count:
        raise RuntimeError(
            "Extended-alpha results were generated for a different "
            "number of spatial-unit/month rows than the current "
            "feature dataset."
        )

    return {
        PRIMARY_POLICY: {
            target: float(
                decision.loc[
                    target,
                    "minimum_mean_rmse_alpha",
                ]
            )
            for target in stage10.TARGET_ORDER
        },
        SENSITIVITY_POLICY: {
            target: (
                None
                if decision.loc[
                    target,
                    "one_standard_error_selection",
                ]
                == "intercept_only"
                else float(
                    decision.loc[
                        target,
                        "selected_one_standard_error_alpha",
                    ]
                )
            )
            for target in stage10.TARGET_ORDER
        },
    }


def validate_attribution_data(
    data: pd.DataFrame,
    stage10,
) -> int:
    required_columns = [
        "region_id",
        "month",
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

    if data.empty or data[required_columns].isna().any().any():
        raise RuntimeError(
            "Attribution dataset is empty or contains missing "
            "metadata, features, or targets."
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
            "Each spatial unit must have 12 unique monthly "
            "observations numbered 1 through 12."
        )

    return int(data["region_id"].nunique())


def build_intercept_only_coefficient_table(
    target: str,
    stage10,
) -> pd.DataFrame:
    rows = []
    for feature in stage10.FEATURE_COLUMNS:
        match = stage10.FEATURE_PATTERN.match(feature)
        if match:
            component = match.group(1)
            lag = int(match.group(2))
            feature_type = "precipitation_component"
        else:
            component = None
            lag = np.nan
            feature_type = (
                "atmospheric"
                if feature == "PET_mm"
                else "spatial"
            )

        rows.append(
            {
                "target": target,
                "depth_label": stage10.TARGETS[target],
                "selected_alpha": np.nan,
                "feature": feature,
                "feature_type": feature_type,
                "component": component,
                "component_label": (
                    stage10.COMPONENT_LABELS.get(
                        component,
                        "",
                    )
                ),
                "lag_months": lag,
                "standardized_coefficient": 0.0,
                "bootstrap_mean_coefficient": np.nan,
                "bootstrap_median_coefficient": np.nan,
                "bootstrap_q025_coefficient": np.nan,
                "bootstrap_q975_coefficient": np.nan,
                "probability_positive": np.nan,
                "probability_negative": np.nan,
                "stable_positive_95": False,
                "stable_negative_95": False,
                "absolute_standardized_coefficient": 0.0,
            }
        )

    return pd.DataFrame(rows)


def archive_existing_figures() -> None:
    ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filenames = [
        "fig23_component_lag_depth_attribution.png",
        "fig23_component_lag_depth_attribution.pdf",
        "fig24_component_memory_by_depth.png",
        "fig24_component_memory_by_depth.pdf",
    ]

    for filename in filenames:
        source = FIGURE_DIR / filename

        if source.exists():
            destination = (
                ARCHIVE_DIR
                / filename.replace(
                    ".png",
                    "_previous.png",
                ).replace(
                    ".pdf",
                    "_previous.pdf",
                )
            )

            shutil.copy2(
                source,
                destination,
            )


def build_policy_comparison(
    coefficient_table: pd.DataFrame,
    memory_table: pd.DataFrame,
    stage10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficient_rows = []

    for target in stage10.TARGET_ORDER:
        primary = (
            coefficient_table.loc[
                (
                    coefficient_table[
                        "alpha_policy"
                    ]
                    == PRIMARY_POLICY
                )
                & (
                    coefficient_table["target"]
                    == target
                )
                & (
                    coefficient_table[
                        "feature_type"
                    ]
                    == "precipitation_component"
                )
            ]
            .set_index("feature")
            .loc[
                stage10.PRECIPITATION_FEATURES
            ]
        )

        sensitivity = (
            coefficient_table.loc[
                (
                    coefficient_table[
                        "alpha_policy"
                    ]
                    == SENSITIVITY_POLICY
                )
                & (
                    coefficient_table["target"]
                    == target
                )
                & (
                    coefficient_table[
                        "feature_type"
                    ]
                    == "precipitation_component"
                )
            ]
            .set_index("feature")
            .loc[
                stage10.PRECIPITATION_FEATURES
            ]
        )

        primary_values = primary[
            "standardized_coefficient"
        ].to_numpy(dtype=float)

        sensitivity_values = sensitivity[
            "standardized_coefficient"
        ].to_numpy(dtype=float)
        intercept_only_sensitivity = (
            ALPHA_POLICIES[
                SENSITIVITY_POLICY
            ][target]
            is None
        )

        coefficient_rows.append(
            {
                "target": target,
                "depth_label": (
                    stage10.TARGETS[target]
                ),
                "primary_alpha": float(
                    primary[
                        "selected_alpha"
                    ].iloc[0]
                ),
                "sensitivity_alpha": float(
                    sensitivity[
                        "selected_alpha"
                    ].iloc[0]
                ),
                "coefficient_correlation": (
                    float("nan")
                    if intercept_only_sensitivity
                    else float(
                        np.corrcoef(
                            primary_values,
                            sensitivity_values,
                        )[0, 1]
                    )
                ),
                "mean_absolute_coefficient_difference": float(
                    np.mean(
                        np.abs(
                            primary_values
                            - sensitivity_values
                        )
                    )
                ),
                "maximum_absolute_coefficient_difference": float(
                    np.max(
                        np.abs(
                            primary_values
                            - sensitivity_values
                        )
                    )
                ),
                "coefficient_sign_agreement_fraction": (
                    float("nan")
                    if intercept_only_sensitivity
                    else float(
                        np.mean(
                            np.sign(primary_values)
                            == np.sign(
                                sensitivity_values
                            )
                        )
                    )
                ),
                "primary_positive_coefficient_count": int(
                    np.sum(
                        primary_values > 0
                    )
                ),
                "sensitivity_positive_coefficient_count": int(
                    np.sum(
                        sensitivity_values > 0
                    )
                ),
            }
        )

    memory_rows = []

    for target in stage10.TARGET_ORDER:
        for component in stage10.COMPONENTS:
            primary = memory_table.loc[
                (
                    memory_table[
                        "alpha_policy"
                    ]
                    == PRIMARY_POLICY
                )
                & (
                    memory_table["target"]
                    == target
                )
                & (
                    memory_table["component"]
                    == component
                )
            ]

            sensitivity = memory_table.loc[
                (
                    memory_table[
                        "alpha_policy"
                    ]
                    == SENSITIVITY_POLICY
                )
                & (
                    memory_table["target"]
                    == target
                )
                & (
                    memory_table["component"]
                    == component
                )
            ]

            if (
                len(primary) != 1
                or len(sensitivity) != 1
            ):
                raise RuntimeError(
                    f"Missing memory comparison for "
                    f"{target}, {component}."
                )

            primary = primary.iloc[0]
            sensitivity = sensitivity.iloc[0]
            intercept_only_sensitivity = (
                ALPHA_POLICIES[
                    SENSITIVITY_POLICY
                ][target]
                is None
            )

            memory_rows.append(
                {
                    "target": target,
                    "depth_label": (
                        stage10.TARGETS[target]
                    ),
                    "component": component,
                    "component_label": (
                        stage10.COMPONENT_LABELS[
                            component
                        ]
                    ),
                    "primary_alpha": float(
                        ALPHA_POLICIES[
                            PRIMARY_POLICY
                        ][target]
                    ),
                    "sensitivity_alpha": (
                        float("nan")
                        if intercept_only_sensitivity
                        else float(
                            ALPHA_POLICIES[
                                SENSITIVITY_POLICY
                            ][target]
                        )
                    ),
                    "primary_dominant_lag": int(
                        primary[
                            "dominant_lag_by_absolute_effect"
                        ]
                    ),
                    "sensitivity_dominant_lag": (
                        float("nan")
                        if intercept_only_sensitivity
                        else int(
                            sensitivity[
                                "dominant_lag_by_absolute_effect"
                            ]
                        )
                    ),
                    "dominant_lag_agreement": (
                        float("nan")
                        if intercept_only_sensitivity
                        else bool(
                            primary[
                                "dominant_lag_by_absolute_effect"
                            ]
                            == sensitivity[
                                "dominant_lag_by_absolute_effect"
                            ]
                        )
                    ),
                    "primary_memory_index_months": float(
                        primary[
                            "absolute_effect_memory_index_months"
                        ]
                    ),
                    "sensitivity_memory_index_months": float(
                        sensitivity[
                            "absolute_effect_memory_index_months"
                        ]
                    ),
                    "memory_index_difference_months": (
                        float("nan")
                        if intercept_only_sensitivity
                        else float(
                            sensitivity[
                                "absolute_effect_memory_index_months"
                            ]
                            - primary[
                                "absolute_effect_memory_index_months"
                            ]
                        )
                    ),
                    "primary_net_effect": float(
                        primary[
                            "net_standardized_effect_across_lags"
                        ]
                    ),
                    "sensitivity_net_effect": float(
                        sensitivity[
                            "net_standardized_effect_across_lags"
                        ]
                    ),
                }
            )

    return (
        pd.DataFrame(coefficient_rows),
        pd.DataFrame(memory_rows),
    )


def calculate_depth_memory_summary(
    memory_table: pd.DataFrame,
    stage10,
) -> pd.DataFrame:
    summary = (
        memory_table.groupby(
            [
                "alpha_policy",
                "target",
                "depth_label",
            ],
            as_index=False,
            observed=True,
        )
        .agg(
            mean_memory_index_months=(
                "absolute_effect_memory_index_months",
                "mean",
            ),
            median_memory_index_months=(
                "absolute_effect_memory_index_months",
                "median",
            ),
            minimum_memory_index_months=(
                "absolute_effect_memory_index_months",
                "min",
            ),
            maximum_memory_index_months=(
                "absolute_effect_memory_index_months",
                "max",
            ),
        )
    )

    depth_order = {
        target: index
        for index, target
        in enumerate(stage10.TARGET_ORDER)
    }

    summary["depth_order"] = (
        summary["target"]
        .map(depth_order)
    )

    return (
        summary.sort_values(
            [
                "alpha_policy",
                "depth_order",
            ]
        )
        .reset_index(drop=True)
    )


def make_sensitivity_figure(
    coefficient_table: pd.DataFrame,
    depth_memory_summary: pd.DataFrame,
    stage10,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.5, 5.0),
        constrained_layout=True,
    )

    positions = np.arange(
        len(stage10.TARGET_ORDER)
    )

    for policy in [
        PRIMARY_POLICY,
        SENSITIVITY_POLICY,
    ]:
        policy_label = (
            "one-SE (intercept-only where eligible)"
            if policy == SENSITIVITY_POLICY
            else policy
        )
        policy_memory = (
            depth_memory_summary.loc[
                depth_memory_summary[
                    "alpha_policy"
                ]
                == policy
            ]
            .set_index("target")
            .loc[
                stage10.TARGET_ORDER
            ]
        )

        axes[0].plot(
            positions,
            policy_memory[
                "mean_memory_index_months"
            ],
            marker="o",
            linewidth=1.6,
            label=policy_label,
        )

    axes[0].set_xticks(positions)

    axes[0].set_xticklabels(
        [
            stage10.TARGETS[target]
            for target
            in stage10.TARGET_ORDER
        ]
    )

    axes[0].set_xlabel(
        "Soil-moisture depth"
    )

    axes[0].set_ylabel(
        "Mean rainfall-memory index (months)"
    )

    axes[0].set_title(
        "Memory-depth relationship"
    )

    axes[0].legend(
        frameon=True,
        fontsize=8,
    )

    primary = (
        coefficient_table.loc[
            (
                coefficient_table[
                    "alpha_policy"
                ]
                == PRIMARY_POLICY
            )
            & (
                coefficient_table[
                    "feature_type"
                ]
                == "precipitation_component"
            )
        ]
        .sort_values(
            ["target", "feature"]
        )
    )

    sensitivity = (
        coefficient_table.loc[
            (
                coefficient_table[
                    "alpha_policy"
                ]
                == SENSITIVITY_POLICY
            )
            & (
                coefficient_table[
                    "feature_type"
                ]
                == "precipitation_component"
            )
        ]
        .sort_values(
            ["target", "feature"]
        )
    )

    primary_values = primary[
        "standardized_coefficient"
    ].to_numpy(dtype=float)

    sensitivity_values = sensitivity[
        "standardized_coefficient"
    ].to_numpy(dtype=float)

    axes[1].scatter(
        primary_values,
        sensitivity_values,
        s=30,
        edgecolor="black",
        linewidth=0.35,
    )

    minimum = float(
        min(
            primary_values.min(),
            sensitivity_values.min(),
        )
    )

    maximum = float(
        max(
            primary_values.max(),
            sensitivity_values.max(),
        )
    )

    axes[1].plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.0,
    )

    axes[1].set_xlabel(
        "Minimum-CV-alpha coefficient"
    )

    axes[1].set_ylabel(
        "One-SE policy coefficient (0 for intercept-only)"
    )

    axes[1].set_title(
        "Coefficient change under stronger regularization"
    )
    if minimum == maximum:
        axes[1].set_xlim(
            minimum - 0.5,
            maximum + 0.5,
        )
        axes[1].set_ylim(
            minimum - 0.5,
            maximum + 0.5,
        )

    for axis in axes:
        axis.grid(
            linewidth=0.35,
            linestyle=":",
            alpha=0.6,
        )

    figure.suptitle(
        "Regularization sensitivity of "
        "component-lag-depth attribution",
        fontsize=13,
    )

    figure.savefig(
        FIGURE_DIR
        / "fig25_attribution_alpha_sensitivity.png",
        dpi=400,
        bbox_inches="tight",
    )

    figure.savefig(
        FIGURE_DIR
        / "fig25_attribution_alpha_sensitivity.pdf",
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage10 = load_stage10()

    stage10.BOOTSTRAP_ITERATIONS = (
        BOOTSTRAP_ITERATIONS
    )

    if not stage10.DATA_PATH.exists():
        raise FileNotFoundError(stage10.DATA_PATH)

    data = (
        pd.read_parquet(
            stage10.DATA_PATH
        )
        .sort_values(
            ["region_id", "month"]
        )
        .reset_index(drop=True)
    )

    spatial_unit_count = validate_attribution_data(
        data,
        stage10,
    )

    global ALPHA_POLICIES
    ALPHA_POLICIES = load_alpha_policies(
        stage10,
        expected_spatial_unit_count=spatial_unit_count,
        expected_row_count=len(data),
    )

    archive_existing_figures()

    all_coefficient_tables = []
    all_memory_tables = []

    print("=" * 78)
    print("FINAL ATTRIBUTION ALPHA-SENSITIVITY ANALYSIS")
    print("=" * 78)
    print(
        f"Bootstrap iterations per fit: "
        f"{BOOTSTRAP_ITERATIONS}"
    )
    print(f"Spatial units: {spatial_unit_count}")
    print(f"Primary alphas: {ALPHA_POLICIES[PRIMARY_POLICY]}")
    print(
        "Sensitivity alphas: "
        f"{ALPHA_POLICIES[SENSITIVITY_POLICY]}"
    )
    print()

    for policy_index, (
        policy,
        alpha_dictionary,
    ) in enumerate(
        ALPHA_POLICIES.items()
    ):
        print(f"Policy: {policy}")

        policy_coefficient_tables = []

        for target_index, target in enumerate(
            stage10.TARGET_ORDER
        ):
            alpha = float(
                alpha_dictionary[target]
            ) if alpha_dictionary[target] is not None else None

            if alpha is None:
                coefficient_table = (
                    build_intercept_only_coefficient_table(
                        target,
                        stage10,
                    )
                )
                coefficient_table.insert(
                    0,
                    "alpha_policy",
                    policy,
                )
                policy_coefficient_tables.append(
                    coefficient_table
                )
                print(
                    f"  {stage10.TARGETS[target]:>10s}: "
                    "intercept-only; component memory "
                    "indices are undefined"
                )
                continue

            (
                full_coefficients,
                x_scaler,
                y_scaler,
                x_standardized,
                y_standardized,
            ) = stage10.fit_standardized_ridge(
                data,
                target,
                alpha,
            )

            rng = np.random.default_rng(
                BASE_RANDOM_SEED
                + policy_index * 10_000
                + target_index * 1_000
            )

            bootstrap_array = (
                stage10.bootstrap_coefficients(
                    x_standardized=(
                        x_standardized
                    ),
                    y_standardized=(
                        y_standardized
                    ),
                    region_ids=data[
                        "region_id"
                    ].to_numpy(),
                    alpha=alpha,
                    rng=rng,
                )
            )

            coefficient_table = (
                stage10.build_coefficient_table(
                    target=target,
                    selected_alpha=alpha,
                    full_coefficients=(
                        full_coefficients
                    ),
                    bootstrap_coefficients_array=(
                        bootstrap_array
                    ),
                )
            )

            coefficient_table.insert(
                0,
                "alpha_policy",
                policy,
            )

            policy_coefficient_tables.append(
                coefficient_table
            )

            print(
                f"  {stage10.TARGETS[target]:>10s}: "
                f"alpha={alpha:g}"
            )

        policy_coefficients = pd.concat(
            policy_coefficient_tables,
            ignore_index=True,
        )

        policy_memory = (
            stage10.build_memory_table(
                policy_coefficients
            )
        )
        intercept_only_targets = [
            target
            for target, alpha
            in alpha_dictionary.items()
            if alpha is None
        ]
        if intercept_only_targets:
            mask = policy_memory["target"].isin(
                intercept_only_targets
            )
            policy_memory.loc[
                mask,
                "dominant_lag_by_absolute_effect",
            ] = np.nan
            policy_memory.loc[
                mask,
                "dominant_standardized_coefficient",
            ] = np.nan

        policy_memory.insert(
            0,
            "alpha_policy",
            policy,
        )

        all_coefficient_tables.append(
            policy_coefficients
        )

        all_memory_tables.append(
            policy_memory
        )

    coefficient_table = pd.concat(
        all_coefficient_tables,
        ignore_index=True,
    )

    memory_table = pd.concat(
        all_memory_tables,
        ignore_index=True,
    )

    (
        coefficient_comparison,
        memory_comparison,
    ) = build_policy_comparison(
        coefficient_table,
        memory_table,
        stage10,
    )

    depth_memory_summary = (
        calculate_depth_memory_summary(
            memory_table,
            stage10,
        )
    )

    coefficient_path = (
        TABLE_DIR
        / "component_lag_depth_coefficients_alpha_sensitivity.csv"
    )

    memory_path = (
        TABLE_DIR
        / "component_lag_depth_memory_alpha_sensitivity.csv"
    )

    coefficient_comparison_path = (
        TABLE_DIR
        / "attribution_alpha_coefficient_comparison.csv"
    )

    memory_comparison_path = (
        TABLE_DIR
        / "attribution_alpha_memory_comparison.csv"
    )

    depth_summary_path = (
        TABLE_DIR
        / "attribution_alpha_depth_memory_summary.csv"
    )

    coefficient_table.to_csv(
        coefficient_path,
        index=False,
    )

    memory_table.to_csv(
        memory_path,
        index=False,
    )

    coefficient_comparison.to_csv(
        coefficient_comparison_path,
        index=False,
    )

    memory_comparison.to_csv(
        memory_comparison_path,
        index=False,
    )

    depth_memory_summary.to_csv(
        depth_summary_path,
        index=False,
    )

    primary_coefficients = (
        coefficient_table.loc[
            coefficient_table[
                "alpha_policy"
            ]
            == PRIMARY_POLICY
        ]
        .drop(columns="alpha_policy")
        .reset_index(drop=True)
    )

    primary_memory = (
        memory_table.loc[
            memory_table[
                "alpha_policy"
            ]
            == PRIMARY_POLICY
        ]
        .drop(columns="alpha_policy")
        .reset_index(drop=True)
    )

    final_coefficient_path = (
        TABLE_DIR
        / "final_component_lag_depth_coefficients.csv"
    )

    final_memory_path = (
        TABLE_DIR
        / "final_component_lag_depth_memory_indices.csv"
    )

    primary_coefficients.to_csv(
        final_coefficient_path,
        index=False,
    )

    primary_memory.to_csv(
        final_memory_path,
        index=False,
    )

    # Replace the provisional Stage-10 central figures using the
    # finalized minimum-geographic-CV alpha policy.
    stage10.make_coefficient_heatmap(
        primary_coefficients
    )

    stage10.make_memory_figure(
        primary_memory
    )

    make_sensitivity_figure(
        coefficient_table,
        depth_memory_summary,
        stage10,
    )

    monotonic_results: dict[str, bool | None] = {}

    for policy in [
        PRIMARY_POLICY,
        SENSITIVITY_POLICY,
    ]:
        if (
            policy == SENSITIVITY_POLICY
            and any(
                alpha is None
                for alpha in ALPHA_POLICIES[policy].values()
            )
        ):
            monotonic_results[policy] = None
            continue
        values = (
            depth_memory_summary.loc[
                depth_memory_summary[
                    "alpha_policy"
                ]
                == policy
            ]
            .set_index("target")
            .loc[
                stage10.TARGET_ORDER,
                "mean_memory_index_months",
            ]
            .to_numpy(dtype=float)
        )

        monotonic_results[policy] = bool(
            np.all(
                np.diff(values) >= 0.0
            )
        )

    minimum_coefficient_correlation: float | None = float(
        coefficient_comparison[
            "coefficient_correlation"
        ].min()
    )
    if not np.isfinite(minimum_coefficient_correlation):
        minimum_coefficient_correlation = None

    dominant_lag_agreement_fraction: float | None = float(
        memory_comparison[
            "dominant_lag_agreement"
        ].mean()
    )
    if not np.isfinite(dominant_lag_agreement_fraction):
        dominant_lag_agreement_fraction = None

    primary_stable_positive = int(
        primary_coefficients.loc[
            primary_coefficients[
                "feature_type"
            ]
            == "precipitation_component",
            "stable_positive_95",
        ].sum()
    )

    primary_stable_negative = int(
        primary_coefficients.loc[
            primary_coefficients[
                "feature_type"
            ]
            == "precipitation_component",
            "stable_negative_95",
        ].sum()
    )

    metadata = {
        "status": "PASS",
        "spatial_unit_count": spatial_unit_count,
        "spatial_unit_month_row_count": int(len(data)),
        "alpha_decision_source": str(ALPHA_DECISION_PATH),
        "extended_alpha_metadata_source": str(ALPHA_METADATA_PATH),
        "primary_attribution_policy": (
            PRIMARY_POLICY
        ),
        "primary_alphas": (
            ALPHA_POLICIES[
                PRIMARY_POLICY
            ]
        ),
        "regularization_sensitivity_policy": (
            SENSITIVITY_POLICY
        ),
        "sensitivity_alphas": (
            ALPHA_POLICIES[
                SENSITIVITY_POLICY
            ]
        ),
        "sensitivity_interpretation": (
            "Targets marked intercept-only use the null model with no "
            "precipitation-component slopes. Their component memory "
            "indices, dominant lags, and bootstrap uncertainty are "
            "undefined; zero coefficients are shown only to represent "
            "the null model in coefficient outputs."
        ),
        "bootstrap_iterations_per_target_policy": (
            BOOTSTRAP_ITERATIONS
        ),
        "mean_memory_increases_monotonically_with_depth": (
            monotonic_results
        ),
        "minimum_depthwise_coefficient_correlation_between_policies": (
            minimum_coefficient_correlation
        ),
        "dominant_lag_agreement_fraction_between_policies": (
            dominant_lag_agreement_fraction
        ),
        "primary_stable_positive_precipitation_coefficients": (
            primary_stable_positive
        ),
        "primary_stable_negative_precipitation_coefficients": (
            primary_stable_negative
        ),
        "interpretation": (
            "Use the minimum-geographic-CV-alpha results as "
            "the primary common-model attribution. The one-standard-"
            "error sensitivity policy uses the intercept-only model "
            "when it meets the one-standard-error criterion; otherwise "
            "it uses the largest eligible finite Ridge alpha. Memory "
            "indices are not defined for intercept-only targets."
        ),
        "final_coefficient_table": str(
            final_coefficient_path
        ),
        "final_memory_table": str(
            final_memory_path
        ),
        "alpha_sensitivity_coefficients": str(
            coefficient_path
        ),
        "alpha_sensitivity_memory": str(
            memory_path
        ),
        "coefficient_comparison": str(
            coefficient_comparison_path
        ),
        "memory_comparison": str(
            memory_comparison_path
        ),
        "depth_memory_summary": str(
            depth_summary_path
        ),
    }

    metadata_path = (
        TABLE_DIR
        / "final_attribution_alpha_sensitivity_metadata.json"
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("ALPHA-SENSITIVITY SUMMARY")
    print("=" * 78)

    print("Mean memory index by depth:")

    print(
        depth_memory_summary[
            [
                "alpha_policy",
                "depth_label",
                "mean_memory_index_months",
                "median_memory_index_months",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Coefficient agreement:")

    print(
        coefficient_comparison[
            [
                "depth_label",
                "primary_alpha",
                "sensitivity_alpha",
                "coefficient_correlation",
                "mean_absolute_coefficient_difference",
                "maximum_absolute_coefficient_difference",
                "coefficient_sign_agreement_fraction",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print("Memory agreement:")

    print(
        memory_comparison[
            [
                "depth_label",
                "component",
                "primary_dominant_lag",
                "sensitivity_dominant_lag",
                "dominant_lag_agreement",
                "primary_memory_index_months",
                "sensitivity_memory_index_months",
                "memory_index_difference_months",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "Monotonic mean-memory increase:"
    )

    for policy, result in (
        monotonic_results.items()
    ):
        print(f"  {policy}: {result}")

    print()
    print(
        "Minimum coefficient correlation: "
        f"{minimum_coefficient_correlation}"
    )

    print(
        "Dominant-lag agreement fraction: "
        f"{dominant_lag_agreement_fraction}"
    )

    print(
        "Primary stable positive effects: "
        f"{primary_stable_positive}"
    )

    print(
        "Primary stable negative effects: "
        f"{primary_stable_negative}"
    )

    print()
    print(
        f"Final coefficient table : "
        f"{final_coefficient_path}"
    )

    print(
        f"Final memory table      : "
        f"{final_memory_path}"
    )

    print(
        f"Final central heatmap   : "
        f"{FIGURE_DIR / 'fig23_component_lag_depth_attribution.png'}"
    )

    print(
        f"Final memory figure     : "
        f"{FIGURE_DIR / 'fig24_component_memory_by_depth.png'}"
    )

    print(
        f"Alpha-sensitivity figure: "
        f"{FIGURE_DIR / 'fig25_attribution_alpha_sensitivity.png'}"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
