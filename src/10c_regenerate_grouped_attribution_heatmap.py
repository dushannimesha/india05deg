#!/usr/bin/env python3
"""
Regenerate the final component-lag-depth attribution heatmap
as four clearly separated precipitation-component blocks.

No model fitting is performed.
The script reads the finalized Stage-10 coefficient table.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT
    / "results"
    / "tables"
    / "final_component_lag_depth_coefficients.csv"
)

METADATA_INPUT = (
    ROOT
    / "results"
    / "tables"
    / "final_attribution_alpha_sensitivity_metadata.json"
)

FIGURE_DIR = ROOT / "results" / "figures"
PRIMARY_POLICY = "minimum_geographic_cv_rmse"

TARGET_ORDER = [
    "SM_0_10cm_mm",
    "SM_10_40cm_mm",
    "SM_40_100cm_mm",
    "SM_100_200cm_mm",
]

DEPTH_LABELS = {
    "SM_0_10cm_mm": "0–10 cm",
    "SM_10_40cm_mm": "10–40 cm",
    "SM_40_100cm_mm": "40–100 cm",
    "SM_100_200cm_mm": "100–200 cm",
}

COMPONENTS = ["C1", "C2", "C3", "C4"]

COMPONENT_LABELS = {
    "C1": "January-centred eastern",
    "C2": "May–June southwestern",
    "C3": "April–October bimodal",
    "C4": "November-centred northern",
}


def load_final_coefficients() -> pd.DataFrame:
    if not INPUT.exists():
        raise FileNotFoundError(
            "Run 10b_finalize_attribution_alpha_sensitivity.py "
            "successfully before regenerating this figure; "
            f"missing {INPUT}."
        )

    if not METADATA_INPUT.exists():
        raise FileNotFoundError(
            "Final attribution metadata is missing; do not use a "
            "coefficient table without its finalization record. "
            f"Expected {METADATA_INPUT}."
        )

    with METADATA_INPUT.open(
        "r",
        encoding="utf-8",
    ) as stream:
        metadata = json.load(stream)

    if metadata.get("status") != "PASS":
        raise RuntimeError(
            "Final attribution metadata does not report PASS; "
            "successfully run 10a and 10b before regenerating "
            "the final coefficients."
        )

    if metadata.get("primary_attribution_policy") != PRIMARY_POLICY:
        raise RuntimeError(
            "Final attribution metadata uses an unexpected primary "
            "alpha policy."
        )

    data = pd.read_csv(INPUT)
    required_columns = {
        "target",
        "feature",
        "feature_type",
        "component",
        "standardized_coefficient",
        "stable_positive_95",
        "stable_negative_95",
    }
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise RuntimeError(
            "Final coefficient table is missing required columns: "
            f"{sorted(missing_columns)}."
        )

    data = data.loc[
        data["feature_type"]
        == "precipitation_component"
    ].copy()

    if data.empty:
        raise RuntimeError(
            "Final coefficient table has no precipitation-component "
            "rows."
        )

    coefficient_values = pd.to_numeric(
        data["standardized_coefficient"],
        errors="coerce",
    )
    if not np.isfinite(coefficient_values).all():
        raise RuntimeError(
            "Final coefficient table contains non-finite or "
            "non-numeric coefficient values."
        )
    data["standardized_coefficient"] = coefficient_values

    expected_pairs = {
        (target, f"K4_{component}_L{lag}_mm", component)
        for target in TARGET_ORDER
        for component in COMPONENTS
        for lag in range(4)
    }
    actual_pairs = list(
        data[
            ["target", "feature", "component"]
        ].itertuples(
            index=False,
            name=None,
        )
    )

    if len(actual_pairs) != len(expected_pairs):
        raise RuntimeError(
            "Final coefficient table must contain exactly one "
            "coefficient for each target/component/lag combination."
        )

    if len(set(actual_pairs)) != len(actual_pairs):
        raise RuntimeError(
            "Final coefficient table contains duplicate "
            "target/component/lag rows."
        )

    if set(actual_pairs) != expected_pairs:
        missing_pairs = sorted(expected_pairs - set(actual_pairs))
        extra_pairs = sorted(set(actual_pairs) - expected_pairs)
        raise RuntimeError(
            "Final coefficient table does not match the expected "
            f"layer/component/lag grid. Missing={missing_pairs}; "
            f"extra={extra_pairs}."
        )

    return data


def main():
    data = load_final_coefficients()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # One common colour scale for all components.
    max_abs = float(
        np.max(
            np.abs(
                data[
                    "standardized_coefficient"
                ].to_numpy(dtype=float)
            )
        )
    )
    if max_abs == 0.0:
        raise RuntimeError(
            "All precipitation-component coefficients are zero; "
            "a shared heatmap colour scale cannot be constructed."
        )

    # Four separate axes with deliberate whitespace.
    fig = plt.figure(
        figsize=(14.2, 5.6)
    )

    grid = fig.add_gridspec(
        nrows=1,
        ncols=5,
        width_ratios=[
            1.0,
            1.0,
            1.0,
            1.0,
            0.055,
        ],
        wspace=0.14,
        left=0.08,
        right=0.93,
        bottom=0.20,
        top=0.78,
    )

    axes = []

    for component_index, component in enumerate(
        COMPONENTS
    ):
        axis = fig.add_subplot(
            grid[0, component_index]
        )

        axes.append(axis)

        feature_order = [
            f"K4_{component}_L{lag}_mm"
            for lag in range(4)
        ]

        matrix = np.zeros(
            (
                len(TARGET_ORDER),
                4,
            ),
            dtype=float,
        )

        stable = np.zeros_like(
            matrix,
            dtype=bool,
        )

        for depth_index, target in enumerate(
            TARGET_ORDER
        ):
            subset = (
                data[
                    (
                        data["target"]
                        == target
                    )
                    & (
                        data["component"]
                        == component
                    )
                ]
                .set_index("feature")
                .loc[feature_order]
            )

            matrix[depth_index] = (
                subset[
                    "standardized_coefficient"
                ].to_numpy(dtype=float)
            )

            stable[depth_index] = (
                subset[
                    "stable_positive_95"
                ].to_numpy(dtype=bool)
                |
                subset[
                    "stable_negative_95"
                ].to_numpy(dtype=bool)
            )

        image = axis.imshow(
            matrix,
            cmap="RdBu_r",
            vmin=-max_abs,
            vmax=max_abs,
            aspect="auto",
            interpolation="nearest",
        )

        # Strong component title.
        axis.set_title(
            component,
            fontsize=14,
            fontweight="bold",
            pad=26,
        )

        # Small descriptive subtitle.
        axis.text(
            0.5,
            1.035,
            COMPONENT_LABELS[component],
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

        axis.set_xticks(
            np.arange(4)
        )

        axis.set_xticklabels(
            [
                r"$L_0$",
                r"$L_1$",
                r"$L_2$",
                r"$L_3$",
            ],
            fontsize=10,
        )

        axis.set_yticks(
            np.arange(
                len(TARGET_ORDER)
            )
        )

        if component_index == 0:
            axis.set_yticklabels(
                [
                    DEPTH_LABELS[target]
                    for target
                    in TARGET_ORDER
                ],
                fontsize=10,
            )

            axis.set_ylabel(
                "Soil-moisture depth",
                fontsize=11,
                labelpad=10,
            )

        else:
            axis.set_yticklabels([])

        # Cell boundaries make each block cleaner.
        axis.set_xticks(
            np.arange(-0.5, 4, 1),
            minor=True,
        )

        axis.set_yticks(
            np.arange(
                -0.5,
                len(TARGET_ORDER),
                1,
            ),
            minor=True,
        )

        axis.grid(
            which="minor",
            linewidth=0.7,
            alpha=0.35,
        )

        axis.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

        # Stronger outer border for each component.
        for spine in axis.spines.values():
            spine.set_linewidth(1.15)

        # Numbers + significance marker.
        for row in range(
            matrix.shape[0]
        ):
            for column in range(
                matrix.shape[1]
            ):
                value = matrix[
                    row,
                    column,
                ]

                marker = (
                    "*"
                    if stable[row, column]
                    else ""
                )

                # White text only for very dark cells.
                text_color = (
                    "white"
                    if abs(value)
                    > 0.72 * max_abs
                    else "black"
                )

                axis.text(
                    column,
                    row,
                    f"{value:.2f}{marker}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    fontweight="semibold",
                    color=text_color,
                )

    # Single shared colorbar.
    colorbar_axis = fig.add_subplot(
        grid[0, 4]
    )

    colorbar = fig.colorbar(
        image,
        cax=colorbar_axis,
    )

    colorbar.set_label(
        "Standardized coefficient",
        fontsize=10,
        labelpad=9,
    )

    colorbar.ax.tick_params(
        labelsize=9
    )

    # Main title.
    fig.suptitle(
        "Standardized Ridge attribution across "
        "precipitation modes, lags, and soil depths",
        fontsize=14,
        fontweight="semibold",
        y=0.94,
    )

    # Common lag explanation.
    fig.text(
        0.50,
        0.105,
        "Antecedent precipitation lag",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Bootstrap note.
    fig.text(
        0.08,
        0.035,
        "* complete-region bootstrap 95% interval excludes zero",
        ha="left",
        fontsize=8.5,
    )

    png_path = (
        FIGURE_DIR
        / "fig23_component_lag_depth_attribution_grouped.png"
    )

    pdf_path = (
        FIGURE_DIR
        / "fig23_component_lag_depth_attribution_grouped.pdf"
    )

    fig.savefig(
        png_path,
        dpi=500,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("=" * 72)
    print("GROUPED COMPONENT ATTRIBUTION FIGURE")
    print("=" * 72)
    print(f"Input : {INPUT}")
    print(f"PNG   : {png_path}")
    print(f"PDF   : {pdf_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
