"""Hybrid Pipeline tuning + final-evaluation dashboard.

Run:
    streamlit run src/hybrid_comparison.py

or:
    & C:/venvs/IP-Asignment-venv/Scripts/python.exe -m streamlit run src/hybrid_comparison.py

Dashboard rules
- Same visual design as baseline_comparison.py
- No retraining and no overwrite of experiment results
- Hybrid tuning metrics use Validation results
- Hybrid final classification metrics use Test results
- Shows only tuning + final classification metrics
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages


# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HYBRID_ROOT = PROJECT_ROOT / "results" / "hybrid"
EXPORT_ROOT = HYBRID_ROOT / "dashboard_exports"

FINAL_TASKS = {
    "Fruit Type": HYBRID_ROOT / "fruit_type_classification" / "hybrid_fruit_type_classification_test_metrics.json",
    "Ripeness": HYBRID_ROOT / "ripeness_classification" / "hybrid_ripeness_classification_test_metrics.json",
}

# These are checked in order. If no saved tuning file exists, the locked
# tuning results from the completed Hybrid experiment below are displayed.
TUNING_CANDIDATES = {
    "Fruit Type": [
        HYBRID_ROOT / "fruit_type_classification" / "hybrid_fruit_type_tuning_results.json",
        HYBRID_ROOT / "fruit_type_classification" / "hybrid_lightgbm_tuning_results.json",
        HYBRID_ROOT / "fruit_type_classification" / "lightgbm_tuning_results.json",
        HYBRID_ROOT / "fruit_type_classification" / "hybrid_fruit_type_tuning_results.csv",
        HYBRID_ROOT / "fruit_type_classification" / "hybrid_lightgbm_tuning_results.csv",
        HYBRID_ROOT / "fruit_type_classification" / "lightgbm_tuning_results.csv",
    ],
    "Ripeness": [
        HYBRID_ROOT / "ripeness_classification" / "hybrid_ripeness_tuning_results.json",
        HYBRID_ROOT / "ripeness_classification" / "hybrid_catboost_tuning_results.json",
        HYBRID_ROOT / "ripeness_classification" / "catboost_tuning_results.json",
        HYBRID_ROOT / "ripeness_classification" / "hybrid_ripeness_tuning_results.csv",
        HYBRID_ROOT / "ripeness_classification" / "hybrid_catboost_tuning_results.csv",
        HYBRID_ROOT / "ripeness_classification" / "catboost_tuning_results.csv",
    ],
}


# 2. THEME — SAME AS BASELINE DASHBOARD

RED = "#C1121F"
RED_DARK = "#8A0E16"
RED_LIGHT = "#FCE8EA"

BLUE = "#145DA0"
BLUE_DARK = "#0B3C6F"

BLACK = "#111111"
DARK_GRAY = "#333333"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
VERY_LIGHT_GRAY = "#F7F7F7"
PAGE_GRAY = "#D9DEE5"
TABLE_BORDER = "#9CA3AF"
WHITE = "#FFFFFF"

CONFIG_COLORS = {
    "Configuration 1": RED,
    "Configuration 2": BLUE,
}

DISPLAY_METRICS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1_score": "Macro F1",
}

VIEW_DESCRIPTIONS = {
    "Overall": (
        "This page summarizes the selected Hybrid tuning configurations and the final "
        "Test-set classification performance for Fruit Type and Ripeness. Validation "
        "results are used for tuning, while Test results are used only for final evaluation."
    ),
    "Tuning Results": (
        "This page compares Configuration 1 and Configuration 2 using Validation results. "
        "Configuration 2 is selected for both LightGBM and CatBoost because it produces "
        "stronger validation classification performance."
    ),
    "Fruit Type": (
        "This page shows the final Hybrid LightGBM Fruit Type classification results on "
        "the Test set, including classification metrics, Test-set class support and the confusion matrix."
    ),
    "Fruit Ripeness": (
        "This page shows the final Hybrid CatBoost Ripeness classification results on "
        "the Test set, including classification metrics, Test-set class support and the confusion matrix."
    ),
    "Generalization Check": (
        "This page compares the selected Validation results with the final Test results. "
        "A small Validation-to-Test gap suggests similar generalization performance, while "
        "a large drop may indicate overfitting. Underfitting cannot be confirmed without "
        "Training-set performance metrics."
    ),
    "Metric Guide": (
        "This page explains the metrics used for Hybrid classifier tuning and final evaluation."
    ),
}


# 3. LOCKED TUNING RESULTS

LOCKED_TUNING_RESULTS = {
    "Fruit Type": [
        {
            "Configuration": "Configuration 1",
            "Selected": False,
            "n_estimators": 150,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "Validation Accuracy": 0.8275,
            "Validation Macro Precision": 0.8139,
            "Validation Macro Recall": 0.7001,
            "Validation Macro F1": 0.7335,
            "Training Time (s)": 3.1672,
            "Prediction Time (ms/ROI)": 0.0126,
        },
        {
            "Configuration": "Configuration 2",
            "Selected": True,
            "n_estimators": 250,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "Validation Accuracy": 0.8355,
            "Validation Macro Precision": 0.8206,
            "Validation Macro Recall": 0.7100,
            "Validation Macro F1": 0.7436,
            "Training Time (s)": 3.4971,
            "Prediction Time (ms/ROI)": 0.0237,
        },
    ],
    "Ripeness": [
        {
            "Configuration": "Configuration 1",
            "Selected": False,
            "iterations": 150,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "Validation Accuracy": 0.6173,
            "Validation Macro Precision": 0.6422,
            "Validation Macro Recall": 0.5794,
            "Validation Macro F1": 0.5999,
            "Training Time (s)": 2.0849,
            "Prediction Time (ms/ROI)": 0.0005,
        },
        {
            "Configuration": "Configuration 2",
            "Selected": True,
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 3,
            "Validation Accuracy": 0.6416,
            "Validation Macro Precision": 0.6597,
            "Validation Macro Recall": 0.6096,
            "Validation Macro F1": 0.6276,
            "Training Time (s)": 3.2650,
            "Prediction Time (ms/ROI)": 0.0005,
        },
    ],
}


# 4. FILE HELPERS

@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    try:
        with Path(path).open(encoding="utf-8") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def load_final_results() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}

    for task, path in FINAL_TASKS.items():
        value = read_json(str(path))
        if isinstance(value, dict):
            loaded[task] = value

    return loaded


def normalize_tuning_columns(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return table

    aliases = {
        "configuration": "Configuration",
        "config": "Configuration",
        "selected": "Selected",
        "is_selected": "Selected",
        "validation_accuracy": "Validation Accuracy",
        "accuracy": "Validation Accuracy",
        "validation_macro_precision": "Validation Macro Precision",
        "macro_precision": "Validation Macro Precision",
        "validation_macro_recall": "Validation Macro Recall",
        "macro_recall": "Validation Macro Recall",
        "validation_macro_f1": "Validation Macro F1",
        "validation_macro_f1_score": "Validation Macro F1",
        "macro_f1": "Validation Macro F1",
        "macro_f1_score": "Validation Macro F1",
        "training_time_seconds": "Training Time (s)",
        "training_time": "Training Time (s)",
        "prediction_time_ms_per_roi": "Prediction Time (ms/ROI)",
        "prediction_time": "Prediction Time (ms/ROI)",
    }

    rename_map = {}
    for column in table.columns:
        key = str(column).strip().lower()
        if key in aliases:
            rename_map[column] = aliases[key]

    table = table.rename(columns=rename_map).copy()

    if "Configuration" not in table.columns:
        table.insert(
            0,
            "Configuration",
            [f"Configuration {index + 1}" for index in range(len(table))],
        )

    if "Selected" not in table.columns:
        table["Selected"] = False

    if not table["Selected"].astype(bool).any() and "Validation Macro F1" in table.columns:
        scores = pd.to_numeric(table["Validation Macro F1"], errors="coerce")
        if scores.notna().any():
            table.loc[scores.idxmax(), "Selected"] = True

    return table


def normalize_tuning_json(data: Any) -> pd.DataFrame:
    if isinstance(data, list):
        return normalize_tuning_columns(pd.DataFrame(data))

    if not isinstance(data, dict):
        return pd.DataFrame()

    for key in ("configurations", "results", "tuning_results", "validation_results"):
        if isinstance(data.get(key), list):
            return normalize_tuning_columns(pd.DataFrame(data[key]))

    if data and all(isinstance(value, dict) for value in data.values()):
        rows = []
        for name, value in data.items():
            row = {"Configuration": name}
            row.update(value)
            rows.append(row)
        return normalize_tuning_columns(pd.DataFrame(rows))

    return normalize_tuning_columns(pd.DataFrame([data]))


def load_tuning_results() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    loaded: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}

    for task, candidates in TUNING_CANDIDATES.items():
        table = pd.DataFrame()
        source = ""

        for path in candidates:
            if not path.exists():
                continue

            if path.suffix.lower() == ".json":
                table = normalize_tuning_json(read_json(str(path)))
            else:
                table = normalize_tuning_columns(read_csv(str(path)))

            if not table.empty:
                source = str(path.relative_to(PROJECT_ROOT))
                break

        if table.empty:
            table = pd.DataFrame(LOCKED_TUNING_RESULTS[task])
            source = "Locked Hybrid tuning results"

        loaded[task] = table
        sources[task] = source

    return loaded, sources


# 5. RESULT TABLES

def final_summary_table(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for task, metrics in results.items():
        rows.append(
            {
                "Task": task,
                "Pipeline": "Hybrid",
                "Technique": "LightGBM" if task == "Fruit Type" else "CatBoost",
                "Accuracy": float(metrics.get("accuracy", np.nan)),
                "Macro Precision": float(metrics.get("macro_precision", np.nan)),
                "Macro Recall": float(metrics.get("macro_recall", np.nan)),
                "Macro F1": float(metrics.get("macro_f1_score", np.nan)),
                "Training Time (s)": float(metrics.get("training_time_seconds", np.nan)),
                "Prediction Time (ms/ROI)": float(metrics.get("prediction_time_ms_per_roi", np.nan)),
                "Test ROIs": int(metrics.get("samples_evaluated", 0)),
                "Classes Evaluated": int(
                    metrics.get("classes_evaluated", len(metrics.get("class_labels", [])))
                ),
            }
        )

    return pd.DataFrame(rows)


def selected_tuning_summary(tuning: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for task, table in tuning.items():
        selected = table.loc[table["Selected"].astype(bool)]
        if selected.empty:
            selected = table.tail(1)

        row = selected.iloc[0]

        if task == "Fruit Type":
            parameters = (
                f"n_estimators={int(row.get('n_estimators', 250))}, "
                f"learning_rate={row.get('learning_rate', 0.05)}, "
                f"num_leaves={int(row.get('num_leaves', 31))}, "
                f"max_depth={int(row.get('max_depth', -1))}"
            )
        else:
            parameters = (
                f"iterations={int(row.get('iterations', 250))}, "
                f"learning_rate={row.get('learning_rate', 0.05)}, "
                f"depth={int(row.get('depth', 6))}, "
                f"l2_leaf_reg={row.get('l2_leaf_reg', 3)}"
            )

        rows.append(
            {
                "Task": task,
                "Technique": "LightGBM" if task == "Fruit Type" else "CatBoost",
                "Selected Configuration": row.get("Configuration", "Configuration 2"),
                "Selected Parameters": parameters,
                "Validation Accuracy": row.get("Validation Accuracy", np.nan),
                "Validation Macro Precision": row.get("Validation Macro Precision", np.nan),
                "Validation Macro Recall": row.get("Validation Macro Recall", np.nan),
                "Validation Macro F1": row.get("Validation Macro F1", np.nan),
                "Training Time (s)": row.get("Training Time (s)", np.nan),
                "Prediction Time (ms/ROI)": row.get("Prediction Time (ms/ROI)", np.nan),
            }
        )

    return pd.DataFrame(rows)


def support_table(metrics: dict[str, Any]) -> pd.DataFrame:
    support = metrics.get("per_class_support", {})
    return pd.DataFrame({"Class": list(support), "Test Support": list(support.values())})


def per_class_metrics_table(metrics: dict[str, Any]) -> pd.DataFrame:
    """Derive per-class Precision, Recall and F1 from the saved confusion matrix."""
    labels = [
        str(label)
        for label in metrics.get(
            "class_labels",
            [],
        )
    ]

    matrix = np.asarray(
        metrics.get(
            "confusion_matrix",
            [],
        ),
        dtype=float,
    )

    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] != len(labels)
    ):
        return pd.DataFrame()

    rows = []

    for index, label in enumerate(labels):
        true_positive = matrix[index, index]
        false_positive = matrix[:, index].sum() - true_positive
        false_negative = matrix[index, :].sum() - true_positive
        support = matrix[index, :].sum()

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        rows.append(
            {
                "Class": label,
                "Precision": float(precision),
                "Recall": float(recall),
                "F1": float(f1),
                "Support": int(support),
            }
        )

    return pd.DataFrame(rows)


def generalization_table(
    selected_tuning: pd.DataFrame,
    final_test: pd.DataFrame,
) -> pd.DataFrame:
    """Compare selected Validation metrics with final Test metrics."""
    rows = []

    for task in [
        "Fruit Type",
        "Ripeness",
    ]:
        validation_rows = selected_tuning.loc[
            selected_tuning["Task"]
            == task
        ]

        test_rows = final_test.loc[
            final_test["Task"]
            == task
        ]

        if (
            validation_rows.empty
            or test_rows.empty
        ):
            continue

        validation = validation_rows.iloc[0]
        test = test_rows.iloc[0]

        validation_accuracy = float(
            validation.get(
                "Validation Accuracy",
                np.nan,
            )
        )

        test_accuracy = float(
            test.get(
                "Accuracy",
                np.nan,
            )
        )

        validation_f1 = float(
            validation.get(
                "Validation Macro F1",
                np.nan,
            )
        )

        test_f1 = float(
            test.get(
                "Macro F1",
                np.nan,
            )
        )

        accuracy_gap = (
            validation_accuracy
            - test_accuracy
        )

        f1_gap = (
            validation_f1
            - test_f1
        )

        max_gap = max(
            abs(accuracy_gap),
            abs(f1_gap),
        )

        if max_gap <= 0.03:
            status = "Small gap"
            interpretation = "No strong overfitting signal"
        elif max_gap <= 0.08:
            status = "Moderate gap"
            interpretation = "Monitor possible overfitting"
        else:
            status = "Large gap"
            interpretation = "Possible overfitting"

        rows.append(
            {
                "Task": task,
                "Validation Accuracy": validation_accuracy,
                "Test Accuracy": test_accuracy,
                "Accuracy Gap": accuracy_gap,
                "Validation Macro F1": validation_f1,
                "Test Macro F1": test_f1,
                "Macro F1 Gap": f1_gap,
                "Gap Status": status,
                "Interpretation": interpretation,
            }
        )

    return pd.DataFrame(
        rows
    )


# 6. CHARTS

def style_axes(axis: plt.Axes, grid_axis: str = "y") -> None:
    axis.set_facecolor(WHITE)
    axis.tick_params(colors=BLACK)
    axis.xaxis.label.set_color(BLACK)
    axis.yaxis.label.set_color(BLACK)
    axis.title.set_color(BLACK)

    for spine in axis.spines.values():
        spine.set_color("#CFCFCF")

    axis.grid(
        axis=grid_axis,
        color="#D6D6D6",
        alpha=0.35,
        linewidth=0.8,
    )
    axis.set_axisbelow(True)


def tuning_score_chart(table: pd.DataFrame, task: str) -> plt.Figure:
    metrics = [
        "Validation Accuracy",
        "Validation Macro Precision",
        "Validation Macro Recall",
        "Validation Macro F1",
    ]

    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    figure.patch.set_facecolor(WHITE)

    x = np.arange(len(metrics))
    width = 0.34

    for index, config in enumerate(["Configuration 1", "Configuration 2"]):
        rows = table.loc[table["Configuration"].astype(str) == config]
        if rows.empty:
            continue

        row = rows.iloc[0]
        values = [float(row.get(metric, np.nan)) for metric in metrics]
        offset = -width / 2 if index == 0 else width / 2

        bars = axis.bar(
            x + offset,
            values,
            width,
            label=config,
            color=CONFIG_COLORS[config],
        )

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            axis.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=BLACK,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(
        ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"],
        rotation=12,
        ha="right",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Validation score")
    axis.set_title(f"{task} — Hybrid Tuning Performance", fontweight="bold")
    axis.legend(frameon=False)
    style_axes(axis)
    figure.tight_layout()
    return figure


def tuning_time_chart(table: pd.DataFrame, task: str, metric: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(6.4, 4.1))
    figure.patch.set_facecolor(WHITE)

    labels = []
    values = []
    colors = []

    for config in ["Configuration 1", "Configuration 2"]:
        rows = table.loc[table["Configuration"].astype(str) == config]
        if rows.empty:
            continue

        value = float(rows.iloc[0].get(metric, np.nan))
        labels.append(config)
        values.append(value)
        colors.append(CONFIG_COLORS[config])

    bars = axis.bar(labels, values, color=colors, width=0.56)

    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.4g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BLACK,
        )

    axis.set_title(f"{task} — {metric}", fontweight="bold")
    axis.set_ylabel(metric)
    axis.text(
        0.5,
        -0.16,
        "Lower is better",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color=MID_GRAY,
    )
    style_axes(axis)
    figure.tight_layout()
    return figure


def final_score_chart(metrics: dict[str, Any], task: str) -> plt.Figure:
    labels = list(DISPLAY_METRICS.values())
    values = [float(metrics.get(key, np.nan)) for key in DISPLAY_METRICS]

    figure, axis = plt.subplots(figsize=(7.5, 4.25))
    figure.patch.set_facecolor(WHITE)

    bars = axis.bar(labels, values, color=[RED, BLUE, BLUE, RED])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Test score")
    axis.set_title(
        f"Hybrid Pipeline — {task} Final Test Metrics",
        loc="left",
        fontweight="bold",
    )
    axis.tick_params(axis="x", rotation=14)
    style_axes(axis)

    for bar, value in zip(bars, values):
        if not np.isnan(value):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                color=BLACK,
            )

    figure.tight_layout()
    return figure


def confusion_figure(metrics: dict[str, Any], task: str) -> plt.Figure:
    labels = [str(label) for label in metrics.get("class_labels", [])]
    matrix = np.asarray(metrics.get("confusion_matrix", []), dtype=float)

    if matrix.ndim != 2 or not labels:
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.axis("off")
        axis.text(0.5, 0.5, "No confusion matrix available", ha="center", va="center")
        return figure

    figure_size = max(6.5, min(12, len(labels) * 0.72))
    figure, axis = plt.subplots(figsize=(figure_size, figure_size * 0.78))
    figure.patch.set_facecolor(WHITE)

    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Hybrid Pipeline — {task} Confusion Matrix", loc="left", fontweight="bold")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{int(matrix[row, column])}",
                ha="center",
                va="center",
                color=WHITE if matrix[row, column] > threshold else BLACK,
                fontsize=8,
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="ROI count")
    figure.tight_layout()
    return figure


def class_support_bar(metrics: dict[str, Any], title: str) -> plt.Figure:
    support = metrics.get("per_class_support", {})
    labels = list(support.keys())
    values = [support[label] for label in labels]

    figure, axis = plt.subplots(figsize=(7.2, max(4.2, len(labels) * 0.38)))
    figure.patch.set_facecolor(WHITE)

    if not values:
        axis.axis("off")
        axis.text(0.5, 0.5, "No class support data available", ha="center", va="center")
        return figure

    positions = np.arange(len(labels))
    bars = axis.barh(positions, values, color=BLUE)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.invert_yaxis()
    axis.set_xlabel("Number of test ROIs")
    axis.set_title(title, fontweight="bold")

    for bar, value in zip(bars, values):
        axis.annotate(
            f"{int(value):,}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=BLACK,
        )

    style_axes(axis, grid_axis="x")
    figure.tight_layout()
    return figure


# 7. DISPLAY HELPERS

def show_description(text: str) -> None:
    st.markdown(
        f'<div class="description-card">{text}</div>',
        unsafe_allow_html=True,
    )


def show_chart(figure: plt.Figure) -> None:
    with st.container(border=True):
        st.pyplot(figure, clear_figure=True, use_container_width=True)


def show_table(table: pd.DataFrame, formats: dict[str, str] | None = None) -> None:
    if table.empty:
        st.info("No saved results are available for this section.")
        return

    styled = table.style.format(formats or {}, na_rep="Not available").hide(axis="index")
    st.markdown(
        '<div class="comparison-table">' + styled.to_html() + "</div>",
        unsafe_allow_html=True,
    )


def selected_row(table: pd.DataFrame) -> pd.Series:
    selected = table.loc[table["Selected"].astype(bool)]
    return selected.iloc[0] if not selected.empty else table.iloc[-1]


def short_configuration_name(value: Any) -> str:
    """Return a short configuration label that fits cleanly inside st.metric cards."""
    text = str(value)

    if text.lower().startswith("configuration"):
        suffix = text[len("configuration"):].strip()
        return f"Config {suffix}" if suffix else "Config"

    return text


def final_metric_comparison_chart(
    final_table: pd.DataFrame,
) -> plt.Figure:
    """Compare Fruit Type and Ripeness final Test classification scores."""
    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]

    figure, axis = plt.subplots(figsize=(8.4, 4.8))
    figure.patch.set_facecolor(WHITE)

    x = np.arange(len(metrics))
    width = 0.34

    task_colors = {
        "Fruit Type": RED,
        "Ripeness": BLUE,
    }

    for index, task in enumerate(["Fruit Type", "Ripeness"]):
        rows = final_table.loc[final_table["Task"] == task]

        if rows.empty:
            continue

        row = rows.iloc[0]
        values = [
            float(row.get(metric, np.nan))
            for metric in metrics
        ]

        offset = -width / 2 if index == 0 else width / 2

        bars = axis.bar(
            x + offset,
            values,
            width,
            label=task,
            color=task_colors[task],
            alpha=0.95,
        )

        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue

            axis.annotate(
                f"{value:.3f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=BLACK,
            )

    axis.set_xticks(x)
    axis.set_xticklabels(
        metrics,
        rotation=12,
        ha="right",
    )
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Test score")
    axis.set_title(
        "Hybrid Final Classification Metric Comparison",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )
    axis.legend(frameon=False)

    style_axes(axis)
    figure.tight_layout()
    return figure


def final_classification_radar(
    metrics: dict[str, Any],
    task: str,
) -> plt.Figure:
    """Radar chart for one Hybrid final classification task."""
    labels = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]

    values = [
        float(metrics.get("accuracy", 0.0)),
        float(metrics.get("macro_precision", 0.0)),
        float(metrics.get("macro_recall", 0.0)),
        float(metrics.get("macro_f1_score", 0.0)),
    ]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()

    values += values[:1]
    angles += angles[:1]

    figure, axis = plt.subplots(
        figsize=(6.3, 5.3),
        subplot_kw={"polar": True},
    )
    figure.patch.set_facecolor(WHITE)
    axis.set_facecolor(WHITE)

    axis.plot(
        angles,
        values,
        color=BLUE if task == "Ripeness" else RED,
        linewidth=2.2,
        label=f"Hybrid {task}",
    )

    axis.fill(
        angles,
        values,
        color=BLUE if task == "Ripeness" else RED,
        alpha=0.12,
    )

    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(
        labels,
        color=BLACK,
        fontsize=9,
    )

    axis.set_ylim(0, 1)
    axis.set_yticks(
        [0.2, 0.4, 0.6, 0.8, 1.0]
    )
    axis.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        color=MID_GRAY,
        fontsize=8,
    )

    axis.grid(
        color=LIGHT_GRAY,
        alpha=0.9,
    )

    axis.set_title(
        f"Hybrid {task} Classification Radar",
        color=BLACK,
        fontweight="bold",
        pad=24,
    )

    axis.legend(
        loc="upper right",
        bbox_to_anchor=(1.20, 1.12),
        frameon=False,
    )

    figure.tight_layout()
    return figure


def final_efficiency_chart(
    final_table: pd.DataFrame,
    metric: str,
    title: str,
) -> plt.Figure:
    """Compare Fruit Type and Ripeness final efficiency metrics."""
    figure, axis = plt.subplots(figsize=(6.2, 4.1))
    figure.patch.set_facecolor(WHITE)

    tasks = []
    values = []
    colors = []

    for task, color in [
        ("Fruit Type", RED),
        ("Ripeness", BLUE),
    ]:
        rows = final_table.loc[final_table["Task"] == task]

        if rows.empty:
            continue

        value = pd.to_numeric(
            pd.Series([rows.iloc[0].get(metric, np.nan)]),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):
            continue

        tasks.append(task)
        values.append(float(value))
        colors.append(color)

    bars = axis.bar(
        tasks,
        values,
        color=colors,
        width=0.56,
        alpha=0.95,
    )

    for bar, value in zip(bars, values):
        axis.annotate(
            f"{value:.4g}",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BLACK,
        )

    axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        color=BLACK,
    )
    axis.set_ylabel(metric)

    axis.text(
        0.5,
        -0.16,
        "Lower is better",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color=MID_GRAY,
    )

    style_axes(axis)
    figure.tight_layout()
    return figure


def task_efficiency_chart(
    metrics: dict[str, Any],
    task: str,
) -> plt.Figure:
    """Show final training and prediction time for one task on separate normalized bars."""
    training_time = float(
        metrics.get("training_time_seconds", np.nan)
    )
    prediction_time = float(
        metrics.get("prediction_time_ms_per_roi", np.nan)
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(8.0, 3.9),
    )
    figure.patch.set_facecolor(WHITE)

    values = [
        ("Training Time", training_time, "seconds"),
        ("Prediction Time", prediction_time, "ms/ROI"),
    ]

    for axis, (label, value, unit) in zip(axes, values):
        axis.bar(
            [label],
            [value],
            color=RED if task == "Fruit Type" else BLUE,
            width=0.52,
        )

        if not np.isnan(value):
            axis.text(
                0,
                value,
                f"{value:.4g} {unit}",
                ha="center",
                va="bottom",
                fontsize=9,
                color=BLACK,
            )

        axis.set_title(
            f"{task} {label}",
            fontweight="bold",
        )
        axis.set_ylabel(unit)
        style_axes(axis)

    figure.suptitle(
        f"Hybrid {task} Efficiency",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )

    figure.tight_layout()
    return figure



def validation_test_gap_chart(
    generalization: pd.DataFrame,
    metric: str,
) -> plt.Figure:
    """Compare selected Validation score with final Test score."""
    if metric == "Accuracy":
        validation_column = "Validation Accuracy"
        test_column = "Test Accuracy"
        title = "Validation vs Test Accuracy"
    else:
        validation_column = "Validation Macro F1"
        test_column = "Test Macro F1"
        title = "Validation vs Test Macro F1"

    figure, axis = plt.subplots(
        figsize=(7.6, 4.5)
    )
    figure.patch.set_facecolor(
        WHITE
    )

    x = np.arange(
        len(
            generalization
        )
    )
    width = 0.34

    validation_values = pd.to_numeric(
        generalization[
            validation_column
        ],
        errors="coerce",
    ).to_numpy()

    test_values = pd.to_numeric(
        generalization[
            test_column
        ],
        errors="coerce",
    ).to_numpy()

    validation_bars = axis.bar(
        x - width / 2,
        validation_values,
        width,
        label="Validation",
        color=RED,
        alpha=0.95,
    )

    test_bars = axis.bar(
        x + width / 2,
        test_values,
        width,
        label="Test",
        color=BLUE,
        alpha=0.95,
    )

    for bars, values in [
        (
            validation_bars,
            validation_values,
        ),
        (
            test_bars,
            test_values,
        ),
    ]:
        for bar, value in zip(
            bars,
            values,
        ):
            if np.isnan(
                value
            ):
                continue

            axis.annotate(
                f"{value:.3f}",
                (
                    bar.get_x()
                    + bar.get_width()
                    / 2,
                    bar.get_height(),
                ),
                xytext=(
                    0,
                    4,
                ),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                color=BLACK,
            )

    axis.set_xticks(
        x
    )
    axis.set_xticklabels(
        generalization[
            "Task"
        ]
    )
    axis.set_ylim(
        0,
        1.05,
    )
    axis.set_ylabel(
        "Score"
    )
    axis.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )
    axis.legend(
        frameon=False
    )

    style_axes(
        axis
    )
    figure.tight_layout()

    return figure


def generalization_gap_chart(
    generalization: pd.DataFrame,
) -> plt.Figure:
    """Show Validation minus Test gaps for Accuracy and Macro F1."""
    figure, axis = plt.subplots(
        figsize=(7.6, 4.5)
    )
    figure.patch.set_facecolor(
        WHITE
    )

    x = np.arange(
        len(
            generalization
        )
    )
    width = 0.34

    accuracy_gap = pd.to_numeric(
        generalization[
            "Accuracy Gap"
        ],
        errors="coerce",
    ).to_numpy()

    f1_gap = pd.to_numeric(
        generalization[
            "Macro F1 Gap"
        ],
        errors="coerce",
    ).to_numpy()

    accuracy_bars = axis.bar(
        x - width / 2,
        accuracy_gap,
        width,
        label="Accuracy gap",
        color=RED,
        alpha=0.95,
    )

    f1_bars = axis.bar(
        x + width / 2,
        f1_gap,
        width,
        label="Macro F1 gap",
        color=BLUE,
        alpha=0.95,
    )

    for bars, values in [
        (
            accuracy_bars,
            accuracy_gap,
        ),
        (
            f1_bars,
            f1_gap,
        ),
    ]:
        for bar, value in zip(
            bars,
            values,
        ):
            if np.isnan(
                value
            ):
                continue

            axis.annotate(
                f"{value:.3f}",
                (
                    bar.get_x()
                    + bar.get_width()
                    / 2,
                    bar.get_height(),
                ),
                xytext=(
                    0,
                    4
                    if value >= 0
                    else -12,
                ),
                textcoords="offset points",
                ha="center",
                va=(
                    "bottom"
                    if value >= 0
                    else "top"
                ),
                fontsize=8,
                color=BLACK,
            )

    axis.axhline(
        0,
        color=BLACK,
        linewidth=1,
    )

    axis.axhline(
        0.03,
        color=LIGHT_GRAY,
        linewidth=1,
        linestyle="--",
    )

    axis.set_xticks(
        x
    )
    axis.set_xticklabels(
        generalization[
            "Task"
        ]
    )

    axis.set_ylabel(
        "Validation − Test"
    )

    axis.set_title(
        "Generalization Gap",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )

    axis.legend(
        frameon=False
    )

    axis.text(
        0.5,
        -0.16,
        "A small gap indicates similar Validation and Test performance",
        transform=axis.transAxes,
        ha="center",
        fontsize=8,
        color=MID_GRAY,
    )

    style_axes(
        axis
    )
    figure.tight_layout()

    return figure


def precision_recall_by_class_chart(
    metrics: dict[str, Any],
    task: str,
) -> plt.Figure:
    """Line graph of per-class Precision and Recall.

    This is a class-by-class Precision/Recall line graph derived from the
    saved confusion matrix. It is not a threshold-based Precision-Recall
    curve because the saved final metrics do not contain prediction
    probabilities for every Test ROI.
    """
    table = per_class_metrics_table(
        metrics
    )

    if table.empty:
        figure, axis = plt.subplots(
            figsize=(
                7,
                4,
            )
        )

        axis.axis(
            "off"
        )

        axis.text(
            0.5,
            0.5,
            "No confusion-matrix data available",
            ha="center",
            va="center",
            color=BLACK,
        )

        return figure

    figure_width = max(
        8.5,
        len(
            table
        )
        * 0.82,
    )

    figure, axis = plt.subplots(
        figsize=(
            figure_width,
            4.9,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    x = np.arange(
        len(
            table
        )
    )

    precision_values = pd.to_numeric(
        table[
            "Precision"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    recall_values = pd.to_numeric(
        table[
            "Recall"
        ],
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    axis.plot(
        x,
        precision_values,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Precision",
        color=RED,
    )

    axis.plot(
        x,
        recall_values,
        marker="o",
        linewidth=2.2,
        markersize=6,
        label="Recall",
        color=BLUE,
    )

    for position, value in zip(
        x,
        precision_values,
    ):
        if np.isnan(
            value
        ):
            continue

        axis.annotate(
            f"{value:.2f}",
            (
                position,
                value,
            ),
            xytext=(
                0,
                8,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=RED_DARK,
        )

    for position, value in zip(
        x,
        recall_values,
    ):
        if np.isnan(
            value
        ):
            continue

        axis.annotate(
            f"{value:.2f}",
            (
                position,
                value,
            ),
            xytext=(
                0,
                -14,
            ),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color=BLUE_DARK,
        )

    axis.set_xticks(
        x
    )

    axis.set_xticklabels(
        table[
            "Class"
        ],
        rotation=45,
        ha="right",
    )

    axis.set_ylim(
        0,
        1.08,
    )

    axis.set_ylabel(
        "Score"
    )

    axis.set_xlabel(
        "Class"
    )

    axis.set_title(
        f"Hybrid {task} — Precision and Recall by Class",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )

    axis.legend(
        frameon=False,
        loc="lower left",
    )

    axis.grid(
        axis="both",
        color="#D6D6D6",
        alpha=0.35,
        linewidth=0.8,
    )

    axis.set_axisbelow(
        True
    )

    for spine in axis.spines.values():
        spine.set_color(
            "#CFCFCF"
        )

    figure.tight_layout()

    return figure


def f1_by_class_chart(
    metrics: dict[str, Any],
    task: str,
) -> plt.Figure:
    """Per-class F1 chart derived from the confusion matrix."""
    table = per_class_metrics_table(
        metrics
    )

    if table.empty:
        figure, axis = plt.subplots(
            figsize=(7, 4)
        )
        axis.axis(
            "off"
        )
        axis.text(
            0.5,
            0.5,
            "No confusion-matrix data available",
            ha="center",
            va="center",
            color=BLACK,
        )
        return figure

    ordered = table.sort_values(
        "F1",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(
            7.5,
            max(
                4.4,
                len(
                    ordered
                )
                * 0.42,
            ),
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    bars = axis.barh(
        ordered[
            "Class"
        ],
        ordered[
            "F1"
        ],
        color=BLUE,
        alpha=0.95,
    )

    axis.set_xlim(
        0,
        1.05,
    )
    axis.set_xlabel(
        "F1 score"
    )

    axis.set_title(
        f"Hybrid {task} — Per-Class F1",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )

    for bar, value in zip(
        bars,
        ordered[
            "F1"
        ],
    ):
        axis.annotate(
            f"{value:.3f}",
            (
                bar.get_width(),
                bar.get_y()
                + bar.get_height()
                / 2,
            ),
            xytext=(
                5,
                0,
            ),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=BLACK,
        )

    style_axes(
        axis,
        grid_axis="x",
    )
    figure.tight_layout()

    return figure


# 8. EXPORT

def export_files(
    tuning_results: dict[str, pd.DataFrame],
    final_table: pd.DataFrame,
    final_results: dict[str, dict[str, Any]],
) -> tuple[list[Path], Path]:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)

    csv_paths = []

    final_csv = EXPORT_ROOT / "hybrid_final_test_summary.csv"
    final_table.to_csv(final_csv, index=False)
    csv_paths.append(final_csv)

    for task, table in tuning_results.items():
        name = "fruit_type" if task == "Fruit Type" else "ripeness"
        path = EXPORT_ROOT / f"hybrid_{name}_tuning_summary.csv"
        table.to_csv(path, index=False)
        csv_paths.append(path)

    pdf_path = EXPORT_ROOT / "hybrid_tuning_and_final_evaluation_report.pdf"
    figures = []

    for task, table in tuning_results.items():
        figures.append(tuning_score_chart(table, task))

    if final_results:
        final_export_table = final_summary_table(
            final_results
        )

        figures.append(
            final_metric_comparison_chart(
                final_export_table
            )
        )

        figures.append(
            final_efficiency_chart(
                final_export_table,
                "Training Time (s)",
                "Hybrid Final Training Time",
            )
        )

        figures.append(
            final_efficiency_chart(
                final_export_table,
                "Prediction Time (ms/ROI)",
                "Hybrid Final Prediction Time",
            )
        )

        export_selected = selected_tuning_summary(
            tuning_results
        )

        export_generalization = generalization_table(
            export_selected,
            final_export_table,
        )

        if not export_generalization.empty:
            figures.append(
                validation_test_gap_chart(
                    export_generalization,
                    "Accuracy",
                )
            )

            figures.append(
                validation_test_gap_chart(
                    export_generalization,
                    "Macro F1",
                )
            )

            figures.append(
                generalization_gap_chart(
                    export_generalization
                )
            )

    for task, metrics in final_results.items():
        figures.append(
            final_score_chart(
                metrics,
                task,
            )
        )

        figures.append(
            final_classification_radar(
                metrics,
                task,
            )
        )

        figures.append(
            task_efficiency_chart(
                metrics,
                task,
            )
        )

        figures.append(
            precision_recall_by_class_chart(
                metrics,
                task,
            )
        )

        figures.append(
            f1_by_class_chart(
                metrics,
                task,
            )
        )

        figures.append(
            confusion_figure(
                metrics,
                task,
            )
        )

        figures.append(
            class_support_bar(
                metrics,
                f"Hybrid {task} Test Support",
            )
        )

    try:
        with PdfPages(pdf_path) as pdf:
            metadata = pdf.infodict()
            metadata["Title"] = "Hybrid Pipeline Tuning and Final Evaluation"
            metadata["Author"] = "Smart Fruit Image Analysis System"

            for figure in figures:
                pdf.savefig(figure, bbox_inches="tight", facecolor=WHITE)
    finally:
        for figure in figures:
            plt.close(figure)

    return csv_paths, pdf_path


# 9. STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Hybrid Pipeline Evaluation Dashboard",
    page_icon="🍎",
    layout="wide",
)


# 10. CSS — SAME BASELINE DESIGN

st.markdown(
    f"""
    <style>
        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"] {{
            background: {PAGE_GRAY} !important;
            color: {BLACK} !important;
            font-family: Arial, Helvetica, sans-serif;
        }}

        [data-testid="stHeader"] {{
            background: {PAGE_GRAY} !important;
        }}

        [data-testid="stSidebar"] {{
            background: {BLACK} !important;
            border-right: 1px solid #2B2B2B !important;
        }}

        [data-testid="stSidebar"] * {{
            color: {WHITE} !important;
        }}

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{
            color: #D1D5DB !important;
            opacity: 1 !important;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label {{
            border-radius: 8px;
            padding: 0.28rem 0.4rem;
        }}

        [data-testid="stSidebar"] [role="radiogroup"] label:hover {{
            background: #262626 !important;
        }}

        .block-container {{
            max-width: 1450px;
            padding-top: 1.7rem;
            padding-bottom: 3rem;
        }}

        h1 {{
            color: {BLACK} !important;
            font-weight: 800 !important;
            letter-spacing: -0.02em;
        }}

        h2 {{
            color: {BLACK} !important;
            font-weight: 750 !important;
        }}

        h3 {{
            color: {BLUE_DARK} !important;
            font-weight: 700 !important;
        }}

        h1,
        h2,
        h3 {{
            overflow-wrap: anywhere;
        }}

        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {{
            color: {MID_GRAY} !important;
            opacity: 1 !important;
        }}

        [data-testid="stMetric"] {{
            background: {WHITE} !important;
            border: 1.5px solid #B8BEC7 !important;
            border-top: 4px solid {BLACK} !important;
            border-radius: 10px;
            padding: 0.85rem 1rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
        }}

        [data-testid="stMetricLabel"] p {{
            color: {DARK_GRAY} !important;
            font-weight: 700;
        }}

        [data-testid="stMetricValue"] {{
            color: {BLACK} !important;
            font-weight: 800;
            font-size: clamp(1.20rem, 2.4vw, 2.05rem) !important;
            line-height: 1.15 !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }}

        [data-testid="stMetricValue"] > div {{
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
            word-break: break-word !important;
        }}

        [data-testid="stMetric"] {{
            min-height: 132px;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {WHITE} !important;
            border: 1px solid #C9CED6 !important;
            border-radius: 10px;
            padding: 0.8rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }}

        div.stButton > button,
        [data-testid="stDownloadButton"] > button,
        div.stDownloadButton > button {{
            background: {RED} !important;
            color: {WHITE} !important;
            border: 2px solid {RED} !important;
            border-radius: 8px;
            font-weight: 700;
        }}

        div.stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover,
        div.stDownloadButton > button:hover {{
            background: {WHITE} !important;
            color: {RED} !important;
            border-color: {RED} !important;
        }}

        .description-card {{
            background: {WHITE};
            color: {BLACK};
            border: 1px solid #C9CED6;
            border-left: 5px solid {BLUE};
            padding: 0.9rem 1rem;
            margin: 0.3rem 0 1.1rem 0;
            border-radius: 6px;
            line-height: 1.55;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        }}

        .winner-strip {{
            background: {WHITE};
            color: {BLACK};
            border: 1px solid #C9CED6;
            border-left: 6px solid {RED};
            padding: 0.8rem 1rem;
            margin: 0.6rem 0 1rem 0;
            border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        }}

        .winner-strip strong {{
            color: {RED_DARK} !important;
        }}

        .comparison-table {{
            overflow-x: auto;
            margin: 0.45rem 0 1.2rem;
            border: 2px solid {TABLE_BORDER};
            border-radius: 8px;
            background: {WHITE};
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }}

        .comparison-table table {{
            width: 100%;
            border-collapse: collapse;
            background: {WHITE};
            font-family: Arial, Helvetica, sans-serif;
            border: 1px solid {TABLE_BORDER};
        }}

        .comparison-table th {{
            background: {BLACK} !important;
            color: {WHITE} !important;
            padding: 0.72rem;
            text-align: left;
            white-space: nowrap;
            border: 1px solid {TABLE_BORDER} !important;
        }}

        .comparison-table td {{
            color: {BLACK} !important;
            padding: 0.66rem 0.72rem;
            border: 1px solid {TABLE_BORDER} !important;
            background: {WHITE} !important;
        }}

        .comparison-table tr:nth-child(even) td {{
            background: {VERY_LIGHT_GRAY} !important;
        }}

        .comparison-table tbody tr:hover td {{
            background: {RED_LIGHT} !important;
            color: {BLACK} !important;
            border-color: {RED} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# 11. LOAD DATA

title_column, export_column = st.columns([5, 1.2], vertical_alignment="center")

with title_column:
    st.title("Hybrid Pipeline Evaluation Dashboard")

with export_column:
    export_placeholder = st.empty()

st.caption(
    "Hybrid only | Tuning uses Validation results | "
    "Final classification uses Test results | No retraining"
)

final_results = load_final_results()
tuning_results, tuning_sources = load_tuning_results()

if not final_results:
    expected = "\n".join(
        f"- `{path.relative_to(PROJECT_ROOT)}`"
        for path in FINAL_TASKS.values()
    )
    st.error(
        "No Hybrid final-test metrics were found. Run the Hybrid notebook first.\n\n"
        f"Expected files:\n{expected}"
    )
    st.stop()

final_table = final_summary_table(final_results)
selected_table = selected_tuning_summary(tuning_results)
generalization_results = generalization_table(
    selected_table,
    final_table,
)


# 12. SIDEBAR

with st.sidebar:
    st.header("Dashboard")

    view = st.radio(
        "Choose View",
        [
            "Overall",
            "Tuning Results",
            "Fruit Type",
            "Fruit Ripeness",
            "Generalization Check",
            "Metric Guide",
        ],
    )

    st.divider()

    st.caption(
        "Saved experiment results only. "
        "No model retraining is performed."
    )

    if st.button("Refresh Saved Results"):
        st.cache_data.clear()
        st.rerun()

show_description(VIEW_DESCRIPTIONS[view])


# 13. OVERALL

if view == "Overall":
    st.header("Hybrid Pipeline Summary")

    fruit = final_results.get("Fruit Type", {})
    ripe = final_results.get("Ripeness", {})

    cards = st.columns(4)
    cards[0].metric("Fruit Type Accuracy", f"{float(fruit.get('accuracy', np.nan)):.4f}")
    cards[1].metric("Fruit Type Macro F1", f"{float(fruit.get('macro_f1_score', np.nan)):.4f}")
    cards[2].metric("Ripeness Accuracy", f"{float(ripe.get('accuracy', np.nan)):.4f}")
    cards[3].metric("Ripeness Macro F1", f"{float(ripe.get('macro_f1_score', np.nan)):.4f}")

    st.subheader("Selected Hybrid Tuning Configuration")
    show_table(
        selected_table,
        {
            "Validation Accuracy": "{:.4f}",
            "Validation Macro Precision": "{:.4f}",
            "Validation Macro Recall": "{:.4f}",
            "Validation Macro F1": "{:.4f}",
            "Training Time (s)": "{:.4f}",
            "Prediction Time (ms/ROI)": "{:.4f}",
        },
    )

    st.subheader("Final Test Summary")
    show_table(
        final_table,
        {
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Training Time (s)": "{:.4f}",
            "Prediction Time (ms/ROI)": "{:.4f}",
        },
    )

    st.subheader("Final Classification Metric Comparison")

    show_chart(
        final_metric_comparison_chart(
            final_table
        )
    )

    st.subheader("Final Test Score Charts")

    c1, c2 = st.columns(2)

    with c1:
        if "Fruit Type" in final_results:
            show_chart(
                final_score_chart(
                    final_results["Fruit Type"],
                    "Fruit Type",
                )
            )

    with c2:
        if "Ripeness" in final_results:
            show_chart(
                final_score_chart(
                    final_results["Ripeness"],
                    "Ripeness",
                )
            )

    st.subheader("Classification Shape Comparison")

    radar_left, radar_right = st.columns(2)

    with radar_left:
        if "Fruit Type" in final_results:
            show_chart(
                final_classification_radar(
                    final_results["Fruit Type"],
                    "Fruit Type",
                )
            )

    with radar_right:
        if "Ripeness" in final_results:
            show_chart(
                final_classification_radar(
                    final_results["Ripeness"],
                    "Ripeness",
                )
            )

    st.subheader("Generalization Check")

    generalization_left, generalization_right = st.columns(
        2
    )

    with generalization_left:
        show_chart(
            validation_test_gap_chart(
                generalization_results,
                "Accuracy",
            )
        )

    with generalization_right:
        show_chart(
            validation_test_gap_chart(
                generalization_results,
                "Macro F1",
            )
        )

    show_table(
        generalization_results,
        {
            "Validation Accuracy": "{:.4f}",
            "Test Accuracy": "{:.4f}",
            "Accuracy Gap": "{:.4f}",
            "Validation Macro F1": "{:.4f}",
            "Test Macro F1": "{:.4f}",
            "Macro F1 Gap": "{:.4f}",
        },
    )

    st.info(
        "This is a Validation-to-Test generalization check. "
        "A small gap means the Test result is close to the selected Validation result. "
        "It can reveal a possible overfitting signal, but true underfitting diagnosis "
        "requires Training-set performance metrics."
    )

    st.subheader("Efficiency Comparison")

    efficiency_left, efficiency_right = st.columns(2)

    with efficiency_left:
        show_chart(
            final_efficiency_chart(
                final_table,
                "Training Time (s)",
                "Hybrid Final Training Time",
            )
        )

    with efficiency_right:
        show_chart(
            final_efficiency_chart(
                final_table,
                "Prediction Time (ms/ROI)",
                "Hybrid Final Prediction Time",
            )
        )

    with export_placeholder.container():
        if st.button("Export CSV and PDF", key="export_hybrid_top", use_container_width=True):
            csv_paths, pdf_path = export_files(tuning_results, final_table, final_results)
            st.success(
                "Saved "
                + ", ".join(path.name for path in csv_paths)
                + f" and {pdf_path.name} to {EXPORT_ROOT.relative_to(PROJECT_ROOT)}."
            )


# 14. TUNING RESULTS

elif view == "Tuning Results":
    st.header("Hybrid Hyperparameter Tuning")

    for task in ["Fruit Type", "Ripeness"]:
        table = tuning_results[task]
        selected = selected_row(table)
        technique = "LightGBM" if task == "Fruit Type" else "CatBoost"

        st.subheader(f"{task} — {technique}")

        cards = st.columns(4)

        selected_configuration = str(
            selected.get(
                "Configuration",
                "Configuration 2",
            )
        )

        cards[0].metric(
            "Selected Configuration",
            short_configuration_name(
                selected_configuration
            ),
            help=f"Full selection: {selected_configuration}",
        )

        cards[1].metric(
            "Validation Accuracy",
            f"{float(selected.get('Validation Accuracy', np.nan)):.4f}",
        )

        cards[2].metric(
            "Validation Macro F1",
            f"{float(selected.get('Validation Macro F1', np.nan)):.4f}",
        )

        cards[3].metric(
            "Training Time",
            f"{float(selected.get('Training Time (s)', np.nan)):.4f} s",
        )

        st.markdown(
            f"""
            <div class="winner-strip">
                <strong>{selected.get('Configuration', 'Configuration 2')}</strong>
                was selected because it achieved the stronger Validation classification
                performance for the Hybrid {task} classifier.
            </div>
            """,
            unsafe_allow_html=True,
        )

        formats = {
            "learning_rate": "{:.4f}",
            "Validation Accuracy": "{:.4f}",
            "Validation Macro Precision": "{:.4f}",
            "Validation Macro Recall": "{:.4f}",
            "Validation Macro F1": "{:.4f}",
            "Training Time (s)": "{:.4f}",
            "Prediction Time (ms/ROI)": "{:.4f}",
        }

        show_table(
            table,
            {key: value for key, value in formats.items() if key in table.columns},
        )

        show_chart(tuning_score_chart(table, task))

        t1, t2 = st.columns(2)
        with t1:
            show_chart(tuning_time_chart(table, task, "Training Time (s)"))
        with t2:
            show_chart(tuning_time_chart(table, task, "Prediction Time (ms/ROI)"))

        st.caption(f"Source: {tuning_sources[task]}")
        st.divider()


# 15. FINAL CLASSIFICATION

elif view in {"Fruit Type", "Fruit Ripeness"}:
    task = "Fruit Type" if view == "Fruit Type" else "Ripeness"
    technique = "LightGBM" if task == "Fruit Type" else "CatBoost"
    metrics = final_results.get(task)

    if not metrics:
        st.error(f"No saved Hybrid {task} final-test metrics are available.")
        st.stop()

    st.header(f"{task} Classification — {technique}")

    cards = st.columns(4)
    for column, (key, label) in zip(cards, DISPLAY_METRICS.items()):
        column.metric(label, f"{float(metrics.get(key, np.nan)):.4f}")

    efficiency = st.columns(4)
    efficiency[0].metric("Training Time", f"{float(metrics.get('training_time_seconds', np.nan)):.4f} s")
    efficiency[1].metric("Prediction Time", f"{float(metrics.get('prediction_time_ms_per_roi', np.nan)):.4f} ms/ROI")
    efficiency[2].metric("Test ROIs", f"{int(metrics.get('samples_evaluated', 0)):,}")
    efficiency[3].metric(
        "Classes Evaluated",
        int(metrics.get("classes_evaluated", len(metrics.get("class_labels", [])))),
    )

    st.subheader("Classification Performance")

    chart_column, radar_column = st.columns((1, 1))

    with chart_column:
        show_chart(
            final_score_chart(
                metrics,
                task,
            )
        )

    with radar_column:
        show_chart(
            final_classification_radar(
                metrics,
                task,
            )
        )

    st.subheader("Efficiency")

    show_chart(
        task_efficiency_chart(
            metrics,
            task,
        )
    )

    st.subheader("Per-Class Precision and Recall Line Graph")

    show_chart(
        precision_recall_by_class_chart(
            metrics,
            task,
        )
    )

    st.subheader("Per-Class F1")

    show_chart(
        f1_by_class_chart(
            metrics,
            task,
        )
    )

    per_class_table = per_class_metrics_table(
        metrics
    )

    show_table(
        per_class_table,
        {
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1": "{:.4f}",
        },
    )

    st.subheader("Confusion Matrix")

    show_chart(
        confusion_figure(
            metrics,
            task,
        )
    )

    st.subheader("Test-Set Class Support")

    show_table(
        support_table(
            metrics
        )
    )

    show_chart(
        class_support_bar(
            metrics,
            f"Hybrid {task} Test Support",
        )
    )


# 16. GENERALIZATION CHECK

elif view == "Generalization Check":
    st.header(
        "Hybrid Generalization Check"
    )

    st.info(
        "This page compares the selected Validation result with the final Test result. "
        "It is useful for checking whether performance drops after tuning. "
        "A true underfitting check would also require Training-set scores."
    )

    if generalization_results.empty:
        st.warning(
            "Generalization results could not be constructed from the saved metrics."
        )

    else:
        status_cards = st.columns(
            len(
                generalization_results
            )
        )

        for column, (_, row) in zip(
            status_cards,
            generalization_results.iterrows(),
        ):
            column.metric(
                f"{row['Task']} Gap Status",
                row[
                    "Gap Status"
                ],
                help=row[
                    "Interpretation"
                ],
            )

        show_table(
            generalization_results,
            {
                "Validation Accuracy": "{:.4f}",
                "Test Accuracy": "{:.4f}",
                "Accuracy Gap": "{:.4f}",
                "Validation Macro F1": "{:.4f}",
                "Test Macro F1": "{:.4f}",
                "Macro F1 Gap": "{:.4f}",
            },
        )

        c1, c2 = st.columns(
            2
        )

        with c1:
            show_chart(
                validation_test_gap_chart(
                    generalization_results,
                    "Accuracy",
                )
            )

        with c2:
            show_chart(
                validation_test_gap_chart(
                    generalization_results,
                    "Macro F1",
                )
            )

        show_chart(
            generalization_gap_chart(
                generalization_results
            )
        )

    st.subheader(
        "Per-Class Precision / Recall Line Graph"
    )

    for task in [
        "Fruit Type",
        "Ripeness",
    ]:
        metrics = final_results.get(
            task
        )

        if not metrics:
            continue

        st.markdown(
            f"### {task}"
        )

        chart_left, chart_right = st.columns(
            2
        )

        with chart_left:
            show_chart(
                precision_recall_by_class_chart(
                    metrics,
                    task,
                )
            )

        with chart_right:
            show_chart(
                f1_by_class_chart(
                    metrics,
                    task,
                )
            )

        show_table(
            per_class_metrics_table(
                metrics
            ),
            {
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
            },
        )

        st.divider()


# 17. METRIC GUIDE

else:
    st.header("Hybrid Performance Metric Guide")

    guide = pd.DataFrame(
        [
            {"Evaluation": "Tuning — Validation", "Metric": "Accuracy", "Interpretation": "Higher is better"},
            {"Evaluation": "Tuning — Validation", "Metric": "Macro Precision", "Interpretation": "Higher is better"},
            {"Evaluation": "Tuning — Validation", "Metric": "Macro Recall", "Interpretation": "Higher is better"},
            {"Evaluation": "Tuning — Validation", "Metric": "Macro F1", "Interpretation": "Higher is better"},
            {"Evaluation": "Tuning / Final", "Metric": "Training Time", "Interpretation": "Lower is better"},
            {"Evaluation": "Tuning / Final", "Metric": "Prediction Time (ms/ROI)", "Interpretation": "Lower is better"},
            {"Evaluation": "Final — Test", "Metric": "Accuracy", "Interpretation": "Higher is better"},
            {"Evaluation": "Final — Test", "Metric": "Macro Precision", "Interpretation": "Higher is better"},
            {"Evaluation": "Final — Test", "Metric": "Macro Recall", "Interpretation": "Higher is better"},
            {"Evaluation": "Final — Test", "Metric": "Macro F1", "Interpretation": "Higher is better"},
            {"Evaluation": "Final — Test", "Metric": "Confusion Matrix", "Interpretation": "Shows class-level prediction errors"},
            {"Evaluation": "Final — Test", "Metric": "Test ROIs", "Interpretation": "Number of evaluated fruit ROIs"},
            {"Evaluation": "Generalization", "Metric": "Validation–Test Accuracy Gap", "Interpretation": "Smaller absolute gap is preferred"},
            {"Evaluation": "Generalization", "Metric": "Validation–Test Macro F1 Gap", "Interpretation": "Smaller absolute gap is preferred"},
            {"Evaluation": "Class-Level Test", "Metric": "Per-Class Precision", "Interpretation": "Higher is better; shown as a class-by-class line graph"},
            {"Evaluation": "Class-Level Test", "Metric": "Per-Class Recall", "Interpretation": "Higher is better; shown as a class-by-class line graph"},
            {"Evaluation": "Class-Level Test", "Metric": "Per-Class F1", "Interpretation": "Higher is better"},
        ]
    )

    show_table(guide)

    st.info(
        "This Hybrid dashboard intentionally excludes preprocessing, segmentation, morphology, "
        "detection, counting and feature-extraction comparisons. Those remain in the Benchmark "
        "A vs Benchmark B dashboard."
    )
