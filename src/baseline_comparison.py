"""Benchmark A vs Benchmark B performance comparison dashboard.

Run:
    streamlit run src/baseline_comparison.py

or:
    & C:/venvs/IP-Asignment-venv/Scripts/python.exe -m streamlit run src/baseline_comparison.py

Dashboard rules
- Reads saved experiment results only
- Does not retrain or overwrite models
- Classification metrics use Test results
- Upstream image processing metrics use Validation results
- Benchmark A is red
- Benchmark B is blue
- Main interface uses black, red, blue and white
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages

# 1. PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
REPORT_ROOT = RESULTS_ROOT / "benchmark_comparison"

# 2. THEME

RED = "#C1121F"
RED_DARK = "#8A0E16"
RED_LIGHT = "#FCE8EA"

BLUE = "#145DA0"
BLUE_DARK = "#0B3C6F"
BLUE_LIGHT = "#E8F1FA"

BLACK = "#111111"
DARK_GRAY = "#333333"
MID_GRAY = "#6B7280"
LIGHT_GRAY = "#E5E7EB"
VERY_LIGHT_GRAY = "#F7F7F7"
PAGE_GRAY = "#D9DEE5"
TABLE_BORDER = "#9CA3AF"
WHITE = "#FFFFFF"

BENCHMARK_COLORS = {
    "Benchmark A": RED,
    "Benchmark B": BLUE,
}

# 2A. BENCHMARK ALGORITHM CONFIGURATION

BENCHMARK_ALGORITHM_TABLE = pd.DataFrame(
    [
        {
            "Stage": "Preprocessing",
            "Benchmark A": "Median Filter 3×3",
            "Benchmark B": "Wiener Filter 3×3",
        },
        {
            "Stage": "Segmentation",
            "Benchmark A": "Otsu Thresholding",
            "Benchmark B": "Global Thresholding, threshold 128",
        },
        {
            "Stage": "Morphological Processing",
            "Benchmark A": "Morphological Opening, ellipse 3×3",
            "Benchmark B": "Morphological Closing, ellipse 3×3",
        },
        {
            "Stage": "Fruit Detection",
            "Benchmark A": "External Contours, minimum area 10,000",
            "Benchmark B": (
                "Connected Component Analysis, 8-connectivity, "
                "minimum area 7,000"
            ),
        },
        {
            "Stage": "Fruit Counting",
            "Benchmark A": "Valid Contour Counting",
            "Benchmark B": "Connected Component Counting",
        },
        {
            "Stage": "Feature Extraction",
            "Benchmark A": "RGB Colour Histogram, 8 bins/channel",
            "Benchmark B": "Uniform LBP, radius 3, 8 points",
        },
        {
            "Stage": "Fruit Type Classification",
            "Benchmark A": "LightGBM",
            "Benchmark B": "KNN",
        },
        {
            "Stage": "Ripeness Classification",
            "Benchmark A": "CatBoost",
            "Benchmark B": "Decision Tree",
        },
    ]
)

# 3. STAGE DESCRIPTIONS

STAGE_DESCRIPTIONS = {
    "3. Preprocessing": (
        "Preprocessing improves the input image before segmentation. "
        "Benchmark A uses a Median Filter 3×3, while Benchmark B uses a Wiener Filter 3×3. "
        "BRISQUE, NIQE and PIQE are better when lower, SSIM is better when higher, "
        "and processing time is better when lower."
    ),
    "4. Segmentation": (
        "Segmentation separates the fruit region from the background. "
        "Benchmark A uses Otsu Thresholding, while Benchmark B uses Global Thresholding "
        "with threshold 128. The comparison focuses on boundary edge alignment, "
        "region uniformity, foreground background contrast and processing time."
    ),
    "5. Morphological Processing": (
        "Morphological processing refines the segmented mask. "
        "Benchmark A uses Morphological Opening with an ellipse 3×3 kernel, while "
        "Benchmark B uses Morphological Closing with an ellipse 3×3 kernel. "
        "Higher mask quality scores are preferred and lower processing time is preferred."
    ),
    "6. Fruit Detection": (
        "Fruit detection identifies valid fruit objects after mask refinement. "
        "Benchmark A uses External Contour Detection with minimum area 10,000. "
        "Benchmark B uses Connected Component Analysis with 8-connectivity and "
        "minimum area 7,000. Higher detection accuracy and lower processing time are preferred."
    ),
    "7. Fruit Counting": (
        "Fruit counting estimates the number of fruit objects in each image. "
        "Benchmark A uses Valid Contour Counting, while Benchmark B uses Connected "
        "Component Counting. Higher Exact Count Accuracy is preferred, while lower "
        "MAE, MAPE and processing time are preferred."
    ),
    "10. Feature Extraction": (
        "Feature extraction converts each fruit ROI into a numerical feature vector. "
        "Benchmark A uses an RGB Colour Histogram with 8 bins per channel. "
        "Benchmark B uses Uniform LBP with radius 3 and 8 points. Higher Fisher and "
        "Silhouette scores indicate stronger separation, while lower extraction time "
        "and smaller feature size improve efficiency."
    ),
    "11. Fruit Type Classification": (
        "Fruit type classification predicts the fruit category for each detected ROI. "
        "Benchmark A uses LightGBM, while Benchmark B uses KNN. Accuracy, Macro Precision, "
        "Macro Recall and Macro F1 measure predictive quality, while training and "
        "prediction time measure efficiency."
    ),
    "12. Ripeness Classification": (
        "Ripeness classification predicts the ripeness level for each detected fruit ROI. "
        "Benchmark A uses CatBoost, while Benchmark B uses a Decision Tree. "
        "The models are compared using Accuracy, Macro Precision, Macro Recall, Macro F1, "
        "training time, prediction time and the confusion matrix."
    ),
}

VIEW_DESCRIPTIONS = {
    "Overall": (
        "This page summarizes the most important results from Benchmark A and Benchmark B. "
        "It combines fruit counting, fruit type classification, ripeness classification "
        "and stage level winners to support the final Hybrid Pipeline selection."
    ),
    "Stage Evaluation": (
        "This page compares every evaluated pipeline stage using the saved experimental "
        "results. Each stage shows its description, performance table, charts, stage winner "
        "and best primary score."
    ),
    "Fruit Count": (
        "This page focuses on fruit detection and counting performance. "
        "Higher detection and exact count accuracy are preferred, while lower counting "
        "error and processing time are preferred."
    ),
    "Fruit Type": (
        "This page compares the fruit type classifiers using the final Test set. "
        "It includes overall classification metrics, per-class Precision and Recall line graphs, "
        "class support, confusion matrices and a radar chart."
    ),
    "Fruit Ripeness": (
        "This page compares the ripeness classifiers using the final Test set. "
        "It includes overall classification metrics, per-class Precision and Recall line graphs, "
        "class support, confusion matrices and a radar chart."
    ),
    "Metric Guide": (
        "This page explains which performance metrics are associated with each pipeline stage "
        "and whether a higher or lower value indicates better performance."
    ),
}

# 4. PERFORMANCE METRIC GUIDE

STAGE_METRIC_GUIDE = pd.DataFrame(
    [
        {
            "Stage": "1. Data Extraction / Dataset Preparation",
            "Technique": "Dataset loading, cleaning and split preparation",
            "Performance Metrics": (
                "Total images, class distribution, missing labels, invalid labels and duplicate count"
            ),
            "Interpretation": "Dataset quality and completeness checks",
        },
        {
            "Stage": "2. Initial Image Standardization",
            "Technique": "Letterbox Resizing",
            "Performance Metrics": (
                "Dimension Compliance, Aspect Ratio Preservation and Processing Time"
            ),
            "Interpretation": (
                "Higher compliance and preservation are better. Lower Processing Time is better"
            ),
        },
        {
            "Stage": "3. Preprocessing",
            "Technique": "A: Median Filter 3×3 | B: Wiener Filter 3×3",
            "Performance Metrics": "BRISQUE, NIQE, PIQE, SSIM and Processing Time",
            "Interpretation": (
                "Lower BRISQUE, NIQE and PIQE are better. Higher SSIM is better. "
                "Lower Processing Time is better"
            ),
        },
        {
            "Stage": "4. Segmentation",
            "Technique": "A: Otsu Thresholding | B: Global Thresholding, threshold 128",
            "Performance Metrics": (
                "Boundary Edge Alignment, Region Uniformity, "
                "Foreground Background Contrast and Processing Time"
            ),
            "Interpretation": (
                "Higher quality scores are better. Lower Processing Time is better"
            ),
        },
        {
            "Stage": "5. Morphological Processing",
            "Technique": "A: Opening, ellipse 3×3 | B: Closing, ellipse 3×3",
            "Performance Metrics": (
                "Boundary Edge Alignment, Region Uniformity, "
                "Foreground Background Contrast and Processing Time"
            ),
            "Interpretation": (
                "Higher quality scores are better. Lower Processing Time is better"
            ),
        },
        {
            "Stage": "6. Fruit Detection",
            "Technique": "A: External Contours, min area 10,000 | B: CCA, 8-connectivity, min area 7,000",
            "Performance Metrics": "Detection Accuracy and Processing Time",
            "Interpretation": (
                "Higher Detection Accuracy is better. Lower Processing Time is better"
            ),
        },
        {
            "Stage": "7. Fruit Counting",
            "Technique": "A: Valid Contour Counting | B: Connected Component Counting",
            "Performance Metrics": "Exact Count Accuracy, MAE, MAPE and Processing Time",
            "Interpretation": (
                "Higher Exact Count Accuracy is better. Lower MAE, MAPE and Processing Time are better"
            ),
        },
        {
            "Stage": "8. ROI Extraction",
            "Technique": "Bounding Box Cropping",
            "Performance Metrics": "ROI Detection Rate, ROI Coverage or IoU and Processing Time",
            "Interpretation": (
                "Higher ROI quality is better. Lower Processing Time is better"
            ),
        },
        {
            "Stage": "9. ROI Standardization",
            "Technique": "Image Resizing",
            "Performance Metrics": (
                "Dimension Compliance, Aspect Ratio Preservation and Processing Time"
            ),
            "Interpretation": (
                "Higher compliance and preservation are better. Lower Processing Time is better"
            ),
        },
        {
            "Stage": "10. Feature Extraction",
            "Technique": "A: RGB Histogram, 8 bins/channel | B: Uniform LBP, radius 3, 8 points",
            "Performance Metrics": (
                "Fisher Score, Silhouette Score, Feature Extraction Time and Feature Vector Size"
            ),
            "Interpretation": (
                "Higher Fisher and Silhouette scores are better. "
                "Lower time and Feature Vector Size are better"
            ),
        },
        {
            "Stage": "11. Fruit Type Classification",
            "Technique": "A: LightGBM | B: KNN",
            "Performance Metrics": (
                "Accuracy, Macro Precision, Macro Recall, Macro F1, "
                "Per-Class Precision and Recall, Confusion Matrix, "
                "Training Time and Prediction Time"
            ),
            "Interpretation": (
                "Higher classification scores are better. "
                "Lower Training Time and Prediction Time are better"
            ),
        },
        {
            "Stage": "12. Ripeness Classification",
            "Technique": "A: CatBoost | B: Decision Tree",
            "Performance Metrics": (
                "Accuracy, Macro Precision, Macro Recall, Macro F1, "
                "Per-Class Precision and Recall, Confusion Matrix, "
                "Training Time and Prediction Time"
            ),
            "Interpretation": (
                "Higher classification scores are better. "
                "Lower Training Time and Prediction Time are better"
            ),
        },
    ]
)

# 5. FILE HELPERS

def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()

def find_column(table: pd.DataFrame, *keywords: str) -> str | None:
    if table.empty:
        return None

    for column in table.columns:
        text = str(column).lower()

        if all(keyword.lower() in text for keyword in keywords):
            return column

    return None

def numeric_series(table: pd.DataFrame, *keywords: str) -> pd.Series:
    column = find_column(table, *keywords)

    if column is None:
        return pd.Series(dtype=float)

    return pd.to_numeric(
        table[column],
        errors="coerce",
    ).dropna()

def first_metric(table: pd.DataFrame, *keywords: str) -> float:
    values = numeric_series(table, *keywords)

    if values.empty:
        return np.nan

    return float(values.iloc[0])

# 6. LOAD SAVED RESULTS

@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, Any]:
    benchmark_paths = {
        "Benchmark A": {
            "root": RESULTS_ROOT / "pipeline_a",
            "fruit_metrics": (
                "fruit_type_classification/lightgbm_fruit_type_test_metrics.json"
            ),
            "ripeness_metrics": (
                "ripeness_classification/catboost_ripeness_test_metrics.json"
            ),
            "detection": "detection/external_contour_validation_results.csv",
            "counting": "counting/fruit_counting_validation_results.csv",
            "segmentation": "segmentation/otsu_summary.csv",
            "morphology": "morphology/morphological_opening_tuning_results.csv",
            "preprocessing": "preprocessing/median_filter_tuning_results.csv",
            "feature_config": "feature_extraction/best_feature_extraction_config.json",
        },
        "Benchmark B": {
            "root": RESULTS_ROOT / "pipeline_b",
            "fruit_metrics": (
                "fruit_type_classification/knn_fruit_type_test_metrics.json"
            ),
            "ripeness_metrics": (
                "ripeness_classification/decision_tree_ripeness_test_metrics.json"
            ),
            "detection": "detection/cca_validation_results.csv",
            "counting": "counting/fruit_counting_validation_results.csv",
            "segmentation": "segmentation/global_summary.csv",
            "morphology": "morphology/morphological_closing_tuning_results.csv",
            "preprocessing": "preprocessing/wiener_filter_tuning_results.csv",
            "feature_config": "feature_extraction/best_feature_extraction_config.json",
        },
    }

    loaded: dict[str, Any] = {}

    for benchmark, paths in benchmark_paths.items():
        root = paths["root"]

        loaded[benchmark] = {
            "fruit_metrics": read_json(root / paths["fruit_metrics"]),
            "ripeness_metrics": read_json(root / paths["ripeness_metrics"]),
            "detection": read_csv(root / paths["detection"]),
            "counting": read_csv(root / paths["counting"]),
            "segmentation": read_csv(root / paths["segmentation"]),
            "morphology": read_csv(root / paths["morphology"]),
            "preprocessing": read_csv(root / paths["preprocessing"]),
            "feature_config": read_json(root / paths["feature_config"]),
        }

    return loaded

# 7. RESULT TABLE BUILDERS

def classification_table(
    data: dict[str, Any],
    task: str,
) -> pd.DataFrame:
    metric_key = (
        "fruit_metrics"
        if task == "Fruit Type"
        else "ripeness_metrics"
    )

    rows = []

    for benchmark, values in data.items():
        metrics = values.get(metric_key)

        if not metrics:
            continue

        technique = (
            "LightGBM"
            if task == "Fruit Type" and benchmark == "Benchmark A"
            else "KNN"
            if task == "Fruit Type" and benchmark == "Benchmark B"
            else "CatBoost"
            if task != "Fruit Type" and benchmark == "Benchmark A"
            else "Decision Tree"
        )

        rows.append(
            {
                "Benchmark": benchmark,
                "Technique": technique,
                "Accuracy": metrics.get("accuracy", np.nan),
                "Macro Precision": metrics.get("macro_precision", np.nan),
                "Macro Recall": metrics.get("macro_recall", np.nan),
                "Macro F1": metrics.get("macro_f1_score", np.nan),
                "Training Time (s)": metrics.get(
                    "training_time_seconds",
                    np.nan,
                ),
                "Prediction Time (ms/ROI)": metrics.get(
                    "prediction_time_ms_per_roi",
                    np.nan,
                ),
                "Test ROIs": metrics.get(
                    "samples_evaluated",
                    np.nan,
                ),
            }
        )

    return pd.DataFrame(rows)

def feature_metrics(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    config = config or {}
    winner = config.get("winner_metrics", {})

    return {
        "method": config.get(
            "method",
            config.get(
                "feature_method",
                "Not available",
            ),
        ),
        "fisher": winner.get(
            "fisher_score",
            winner.get(
                "mean_fisher_score",
                np.nan,
            ),
        ),
        "silhouette": winner.get(
            "silhouette_score",
            winner.get(
                "mean_silhouette_score",
                np.nan,
            ),
        ),
        "time": winner.get(
            "feature_extraction_time_ms_per_roi",
            winner.get(
                "extraction_time_ms_per_roi",
                np.nan,
            ),
        ),
        "size": winner.get(
            "feature_vector_size",
            winner.get(
                "vector_size",
                np.nan,
            ),
        ),
    }

def pipeline_summary_table(
    data: dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for benchmark, values in data.items():
        detection = values["detection"]
        counting = values["counting"]
        segmentation = values["segmentation"]
        preprocessing = values["preprocessing"]

        feat = feature_metrics(
            values["feature_config"]
        )

        detection_success = numeric_series(
            detection,
            "detection_success",
        )

        detection_time = numeric_series(
            detection,
            "detection",
            "time",
        )

        exact = numeric_series(
            counting,
            "exact_count_correct",
        )

        absolute_error = numeric_series(
            counting,
            "absolute_error",
        )

        rows.append(
            {
                "Benchmark": benchmark,
                "Preprocessing Time (ms)": first_metric(
                    preprocessing,
                    "time",
                ),
                "Segmentation Time (ms)": first_metric(
                    segmentation,
                    "time",
                ),
                "Detection Accuracy": (
                    float(detection_success.mean())
                    if not detection_success.empty
                    else np.nan
                ),
                "Detection Time (ms)": (
                    float(detection_time.mean())
                    if not detection_time.empty
                    else np.nan
                ),
                "Exact Count Accuracy": (
                    float(exact.mean())
                    if not exact.empty
                    else np.nan
                ),
                "Counting MAE": (
                    float(absolute_error.mean())
                    if not absolute_error.empty
                    else np.nan
                ),
                "Feature Time (ms/ROI)": feat["time"],
                "Feature Size": feat["size"],
                "Feature Method": feat["method"],
            }
        )

    return pd.DataFrame(rows)

def stage_evaluation_tables(
    data: dict[str, Any],
    fruit_scores: pd.DataFrame,
    ripeness_scores: pd.DataFrame,
) -> list[tuple[str, pd.DataFrame]]:
    techniques = {
        "Benchmark A": {
            "preprocessing": "Median Filter 3×3",
            "segmentation": "Otsu Thresholding",
            "morphology": "Morphological Opening, ellipse 3×3",
            "detection": "External Contours, minimum area 10,000",
            "counting": "Valid Contour Counting",
        },
        "Benchmark B": {
            "preprocessing": "Wiener Filter 3×3",
            "segmentation": "Global Thresholding, threshold 128",
            "morphology": "Morphological Closing, ellipse 3×3",
            "detection": (
                "Connected Component Analysis, 8-connectivity, "
                "minimum area 7,000"
            ),
            "counting": "Connected Component Counting",
        },
    }

    preprocessing_rows = []
    segmentation_rows = []
    morphology_rows = []
    detection_rows = []
    counting_rows = []
    feature_rows = []

    for benchmark, values in data.items():
        names = techniques[benchmark]

        preprocessing = values["preprocessing"]
        segmentation = values["segmentation"]
        morphology = values["morphology"]
        detection = values["detection"]
        counting = values["counting"]

        feat = feature_metrics(
            values["feature_config"]
        )

        preprocessing_rows.append(
            {
                "Benchmark": benchmark,
                "Technique": names["preprocessing"],
                "BRISQUE": first_metric(
                    preprocessing,
                    "brisque",
                ),
                "NIQE": first_metric(
                    preprocessing,
                    "niqe",
                ),
                "PIQE": first_metric(
                    preprocessing,
                    "piqe",
                ),
                "SSIM": first_metric(
                    preprocessing,
                    "ssim",
                ),
                "Time (ms)": first_metric(
                    preprocessing,
                    "time",
                ),
            }
        )

        segmentation_rows.append(
            {
                "Benchmark": benchmark,
                "Technique": names["segmentation"],
                "Boundary Edge Alignment": first_metric(
                    segmentation,
                    "boundary",
                    "edge",
                ),
                "Region Uniformity": first_metric(
                    segmentation,
                    "region",
                    "uniformity",
                ),
                "Foreground Background Contrast": first_metric(
                    segmentation,
                    "contrast",
                ),
                "Time (ms)": first_metric(
                    segmentation,
                    "time",
                ),
            }
        )

        morphology_rows.append(
            {
                "Benchmark": benchmark,
                "Technique": names["morphology"],
                "Boundary Edge Alignment": first_metric(
                    morphology,
                    "boundary",
                    "edge",
                ),
                "Region Uniformity": first_metric(
                    morphology,
                    "region",
                    "uniformity",
                ),
                "Foreground Background Contrast": first_metric(
                    morphology,
                    "contrast",
                ),
                "Time (ms)": first_metric(
                    morphology,
                    "time",
                ),
            }
        )

        detection_accuracy = numeric_series(
            detection,
            "detection_success",
        )

        detection_time = numeric_series(
            detection,
            "detection",
            "time",
        )

        detection_rows.append(
            {
                "Benchmark": benchmark,
                "Technique": names["detection"],
                "Detection Accuracy": (
                    float(detection_accuracy.mean())
                    if not detection_accuracy.empty
                    else np.nan
                ),
                "Time (ms)": (
                    float(detection_time.mean())
                    if not detection_time.empty
                    else np.nan
                ),
            }
        )

        ground_truth = numeric_series(
            counting,
            "ground_truth_count",
        )

        if ground_truth.empty:
            ground_truth = numeric_series(
                counting,
                "ground",
                "truth",
            )

        absolute_error = numeric_series(
            counting,
            "absolute_error",
        )

        if absolute_error.empty:
            absolute_error = numeric_series(
                counting,
                "absolute",
                "error",
            )

        exact = numeric_series(
            counting,
            "exact_count_correct",
        )

        if exact.empty:
            exact = numeric_series(
                counting,
                "exact",
                "count",
            )

        counting_time = numeric_series(
            counting,
            "counting",
            "time",
        )

        if counting_time.empty:
            counting_time = numeric_series(
                counting,
                "time",
            )

        if (
            not ground_truth.empty
            and not absolute_error.empty
            and len(ground_truth) == len(absolute_error)
        ):
            gt = ground_truth.reset_index(drop=True)
            err = absolute_error.reset_index(drop=True)
            valid = gt != 0

            mape = (
                float((err[valid] / gt[valid]).mean())
                if valid.any()
                else np.nan
            )
        else:
            mape = np.nan

        counting_rows.append(
            {
                "Benchmark": benchmark,
                "Technique": names["counting"],
                "Exact Count Accuracy": (
                    float(exact.mean())
                    if not exact.empty
                    else np.nan
                ),
                "MAE": (
                    float(absolute_error.mean())
                    if not absolute_error.empty
                    else np.nan
                ),
                "MAPE": mape,
                "Time (ms)": (
                    float(counting_time.mean())
                    if not counting_time.empty
                    else np.nan
                ),
            }
        )

        feature_method_name = (
            "RGB Colour Histogram, 8 bins/channel"
            if benchmark == "Benchmark A"
            else "Uniform LBP, radius 3, 8 points"
        )

        feature_rows.append(
            {
                "Benchmark": benchmark,
                "Technique": feature_method_name,
                "Fisher Score": feat["fisher"],
                "Silhouette Score": feat["silhouette"],
                "Time (ms/ROI)": feat["time"],
                "Feature Size": feat["size"],
            }
        )

    return [
        (
            "3. Preprocessing",
            pd.DataFrame(preprocessing_rows),
        ),
        (
            "4. Segmentation",
            pd.DataFrame(segmentation_rows),
        ),
        (
            "5. Morphological Processing",
            pd.DataFrame(morphology_rows),
        ),
        (
            "6. Fruit Detection",
            pd.DataFrame(detection_rows),
        ),
        (
            "7. Fruit Counting",
            pd.DataFrame(counting_rows),
        ),
        (
            "10. Feature Extraction",
            pd.DataFrame(feature_rows),
        ),
        (
            "11. Fruit Type Classification",
            fruit_scores.copy(),
        ),
        (
            "12. Ripeness Classification",
            ripeness_scores.copy(),
        ),
    ]

# 8. STAGE SCORING

METRIC_DIRECTIONS = {
    "3. Preprocessing": {
        "BRISQUE": False,
        "NIQE": False,
        "PIQE": False,
        "SSIM": True,
        "Time (ms)": False,
    },
    "4. Segmentation": {
        "Boundary Edge Alignment": True,
        "Region Uniformity": True,
        "Foreground Background Contrast": True,
        "Time (ms)": False,
    },
    "5. Morphological Processing": {
        "Boundary Edge Alignment": True,
        "Region Uniformity": True,
        "Foreground Background Contrast": True,
        "Time (ms)": False,
    },
    "6. Fruit Detection": {
        "Detection Accuracy": True,
        "Time (ms)": False,
    },
    "7. Fruit Counting": {
        "Exact Count Accuracy": True,
        "MAE": False,
        "MAPE": False,
        "Time (ms)": False,
    },
    "10. Feature Extraction": {
        "Fisher Score": True,
        "Silhouette Score": True,
        "Time (ms/ROI)": False,
        "Feature Size": False,
    },
    "11. Fruit Type Classification": {
        "Accuracy": True,
        "Macro Precision": True,
        "Macro Recall": True,
        "Macro F1": True,
        "Training Time (s)": False,
        "Prediction Time (ms/ROI)": False,
    },
    "12. Ripeness Classification": {
        "Accuracy": True,
        "Macro Precision": True,
        "Macro Recall": True,
        "Macro F1": True,
        "Training Time (s)": False,
        "Prediction Time (ms/ROI)": False,
    },
}

PRIMARY_STAGE_METRIC = {
    "3. Preprocessing": (
        "SSIM",
        True,
    ),
    "4. Segmentation": (
        "Boundary Edge Alignment",
        True,
    ),
    "5. Morphological Processing": (
        "Boundary Edge Alignment",
        True,
    ),
    "6. Fruit Detection": (
        "Detection Accuracy",
        True,
    ),
    "7. Fruit Counting": (
        "Exact Count Accuracy",
        True,
    ),
    "10. Feature Extraction": (
        "Fisher Score",
        True,
    ),
    "11. Fruit Type Classification": (
        "Macro F1",
        True,
    ),
    "12. Ripeness Classification": (
        "Macro F1",
        True,
    ),
}

def score_stage(
    table: pd.DataFrame,
    metric_directions: dict[str, bool],
) -> tuple[float, float, str, int]:
    if table.empty or "Benchmark" not in table or len(table) < 2:
        return 0.0, 0.0, "Not available", 0

    a_rows = table.loc[
        table["Benchmark"] == "Benchmark A"
    ]

    b_rows = table.loc[
        table["Benchmark"] == "Benchmark B"
    ]

    if a_rows.empty or b_rows.empty:
        return 0.0, 0.0, "Not available", 0

    a = a_rows.iloc[0]
    b = b_rows.iloc[0]

    score_a = 0.0
    score_b = 0.0
    compared = 0

    for metric, higher_is_better in metric_directions.items():
        first = pd.to_numeric(
            pd.Series(
                [a.get(metric, np.nan)]
            ),
            errors="coerce",
        ).iloc[0]

        second = pd.to_numeric(
            pd.Series(
                [b.get(metric, np.nan)]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(first) or pd.isna(second):
            continue

        compared += 1

        if np.isclose(first, second):
            score_a += 0.5
            score_b += 0.5

        elif (
            first > second
        ) == higher_is_better:
            score_a += 1.0

        else:
            score_b += 1.0

    if compared == 0:
        winner = "Not available"

    elif np.isclose(
        score_a,
        score_b,
    ):
        winner = "Tie"

    else:
        winner = (
            "Benchmark A"
            if score_a > score_b
            else "Benchmark B"
        )

    return (
        score_a,
        score_b,
        winner,
        compared,
    )

def stage_scorecard(
    stage_tables: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    rows = []

    for stage_name, table in stage_tables:
        directions = METRIC_DIRECTIONS.get(
            stage_name,
            {},
        )

        score_a, score_b, winner, metric_count = score_stage(
            table,
            directions,
        )

        rows.append(
            {
                "Stage": stage_name,
                "Benchmark A Score": score_a,
                "Benchmark B Score": score_b,
                "Metrics Compared": metric_count,
                "Winner": winner,
            }
        )

    return pd.DataFrame(rows)

def best_primary_metric(
    stage_name: str,
    table: pd.DataFrame,
) -> tuple[str, str, float, str]:
    """Return primary metric name, best benchmark, value and direction text."""
    if stage_name not in PRIMARY_STAGE_METRIC:
        return (
            "Not available",
            "Not available",
            np.nan,
            "",
        )

    metric, higher_is_better = PRIMARY_STAGE_METRIC[
        stage_name
    ]

    if table.empty or metric not in table:
        return (
            metric,
            "Not available",
            np.nan,
            "",
        )

    candidates = []

    for benchmark in [
        "Benchmark A",
        "Benchmark B",
    ]:
        row = table.loc[
            table["Benchmark"] == benchmark
        ]

        if row.empty:
            continue

        value = pd.to_numeric(
            pd.Series(
                [row.iloc[0].get(metric, np.nan)]
            ),
            errors="coerce",
        ).iloc[0]

        if not pd.isna(value):
            candidates.append(
                (
                    benchmark,
                    float(value),
                )
            )

    if not candidates:
        return (
            metric,
            "Not available",
            np.nan,
            "",
        )

    best = (
        max(
            candidates,
            key=lambda item: item[1],
        )
        if higher_is_better
        else min(
            candidates,
            key=lambda item: item[1],
        )
    )

    direction_text = (
        "Higher is better"
        if higher_is_better
        else "Lower is better"
    )

    return (
        metric,
        best[0],
        best[1],
        direction_text,
    )

# 9. MATPLOTLIB CHART STYLE

def style_axes(
    axis: plt.Axes,
    grid_axis: str = "y",
) -> None:
    axis.set_facecolor(
        WHITE
    )

    axis.tick_params(
        colors=BLACK
    )

    axis.xaxis.label.set_color(
        BLACK
    )

    axis.yaxis.label.set_color(
        BLACK
    )

    axis.title.set_color(
        BLACK
    )

    for spine in axis.spines.values():
        spine.set_color(
            "#CFCFCF"
        )

    axis.grid(
        axis=grid_axis,
        color="#D6D6D6",
        alpha=0.35,
        linewidth=0.8,
    )

def grouped_bar(
    table: pd.DataFrame,
    metrics: list[str],
    title: str,
    y_label: str = "Score",
    zero_to_one: bool = False,
) -> plt.Figure:
    figure, axis = plt.subplots(
        figsize=(
            7.6,
            4.5,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    x = np.arange(
        len(metrics)
    )

    width = 0.34

    for benchmark_index, benchmark in enumerate(
        [
            "Benchmark A",
            "Benchmark B",
        ]
    ):
        rows = table.loc[
            table["Benchmark"] == benchmark
        ]

        if rows.empty:
            continue

        row = rows.iloc[0]

        values = [
            pd.to_numeric(
                pd.Series(
                    [row.get(metric, np.nan)]
                ),
                errors="coerce",
            ).iloc[0]
            for metric in metrics
        ]

        offset = (
            -width / 2
            if benchmark_index == 0
            else width / 2
        )

        bars = axis.bar(
            x + offset,
            values,
            width,
            label=benchmark,
            color=BENCHMARK_COLORS[
                benchmark
            ],
            alpha=0.95,
        )

        for bar, value in zip(
            bars,
            values,
        ):
            if pd.isna(
                value
            ):
                continue

            axis.annotate(
                f"{value:.3f}",
                (
                    bar.get_x()
                    + bar.get_width() / 2,
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
        metrics,
        rotation=15,
        ha="right",
    )

    axis.set_ylabel(
        y_label
    )

    axis.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        color=BLACK,
        pad=12,
    )

    if zero_to_one:
        axis.set_ylim(
            0,
            1.08,
        )

    axis.legend(
        frameon=False
    )

    style_axes(
        axis
    )

    figure.tight_layout(
        rect=[
            0.02,
            0.06,
            0.98,
            0.91,
        ],
        pad=1.8,
    )

    return figure

def single_metric_chart(
    table: pd.DataFrame,
    metric: str,
    title: str,
    lower_is_better: bool = False,
) -> plt.Figure:
    figure, axis = plt.subplots(
        figsize=(
            5.8,
            3.9,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    benchmarks = []
    values = []
    colors = []

    for benchmark in [
        "Benchmark A",
        "Benchmark B",
    ]:
        rows = table.loc[
            table["Benchmark"] == benchmark
        ]

        if rows.empty:
            continue

        value = pd.to_numeric(
            pd.Series(
                [
                    rows.iloc[0].get(
                        metric,
                        np.nan,
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(
            value
        ):
            continue

        benchmarks.append(
            benchmark
        )

        values.append(
            float(value)
        )

        colors.append(
            BENCHMARK_COLORS[
                benchmark
            ]
        )

    bars = axis.bar(
        benchmarks,
        values,
        color=colors,
        width=0.58,
        alpha=0.95,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.annotate(
            f"{value:.4g}",
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(
                0,
                5,
            ),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=BLACK,
        )

    axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        color=BLACK,
    )

    axis.set_ylabel(
        metric
    )

    direction = (
        "Lower is better"
        if lower_is_better
        else "Higher is better"
    )

    axis.text(
        0.5,
        -0.18,
        direction,
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

def scorecard_chart(
    scorecard: pd.DataFrame,
) -> plt.Figure:
    figure, axis = plt.subplots(
        figsize=(
            9.5,
            max(
                4.8,
                len(scorecard) * 0.58,
            ),
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    y = np.arange(
        len(scorecard)
    )

    height = 0.34

    axis.barh(
        y - height / 2,
        scorecard[
            "Benchmark A Score"
        ],
        height,
        label="Benchmark A",
        color=RED,
    )

    axis.barh(
        y + height / 2,
        scorecard[
            "Benchmark B Score"
        ],
        height,
        label="Benchmark B",
        color=BLUE,
    )

    axis.set_yticks(
        y
    )

    axis.set_yticklabels(
        scorecard[
            "Stage"
        ]
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Metric points"
    )

    axis.set_title(
        "Per Stage Comparison Score",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )

    axis.legend(
        frameon=False
    )

    style_axes(
        axis,
        grid_axis="x",
    )

    figure.tight_layout()

    return figure

def stage_wins_pie(
    scorecard: pd.DataFrame,
) -> plt.Figure:
    counts = (
        scorecard[
            "Winner"
        ]
        .value_counts()
    )

    labels = []
    values = []
    colors = []

    mapping = [
        (
            "Benchmark A",
            RED,
        ),
        (
            "Benchmark B",
            BLUE,
        ),
        (
            "Tie",
            BLACK,
        ),
    ]

    for label, color in mapping:
        value = int(
            counts.get(
                label,
                0,
            )
        )

        if value > 0:
            labels.append(
                label
            )
            values.append(
                value
            )
            colors.append(
                color
            )

    figure, axis = plt.subplots(
        figsize=(
            5.8,
            4.4,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    if not values:
        axis.axis(
            "off"
        )

        axis.text(
            0.5,
            0.5,
            "No stage winner data available",
            ha="center",
            va="center",
            color=BLACK,
        )

        return figure

    wedges, _, autotexts = axis.pie(
        values,
        colors=colors,
        startangle=90,
        autopct=lambda percent: (
            f"{percent:.0f}%"
        ),
        wedgeprops={
            "edgecolor": WHITE,
            "linewidth": 2,
        },
        textprops={
            "color": BLACK,
            "fontsize": 9,
        },
    )

    for autotext in autotexts:
        autotext.set_color(
            WHITE
        )

        autotext.set_fontweight(
            "bold"
        )

    axis.legend(
        wedges,
        [
            f"{label}  {value} stage(s)"
            for label, value in zip(
                labels,
                values,
            )
        ],
        loc="center left",
        bbox_to_anchor=(
            0.9,
            0.5,
        ),
        frameon=False,
    )

    axis.set_title(
        "Stage Winners",
        color=BLACK,
        fontweight="bold",
    )

    figure.tight_layout()

    return figure

def classification_radar(
    table: pd.DataFrame,
    title: str,
) -> plt.Figure:
    metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
    ]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(metrics),
        endpoint=False,
    ).tolist()

    angles += angles[:1]

    figure, axis = plt.subplots(
        figsize=(
            6.3,
            5.3,
        ),
        subplot_kw={
            "polar": True
        },
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis.set_facecolor(
        WHITE
    )

    for benchmark in [
        "Benchmark A",
        "Benchmark B",
    ]:
        rows = table.loc[
            table[
                "Benchmark"
            ] == benchmark
        ]

        if rows.empty:
            continue

        row = rows.iloc[0]

        values = [
            pd.to_numeric(
                pd.Series(
                    [
                        row.get(
                            metric,
                            np.nan,
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]
            for metric in metrics
        ]

        if all(
            pd.isna(value)
            for value in values
        ):
            continue

        values = [
            0.0
            if pd.isna(value)
            else float(value)
            for value in values
        ]

        values += values[:1]

        axis.plot(
            angles,
            values,
            color=BENCHMARK_COLORS[
                benchmark
            ],
            linewidth=2.2,
            label=benchmark,
        )

        axis.fill(
            angles,
            values,
            color=BENCHMARK_COLORS[
                benchmark
            ],
            alpha=0.12,
        )

    axis.set_xticks(
        angles[:-1]
    )

    axis.set_xticklabels(
        metrics,
        color=BLACK,
        fontsize=9,
    )

    axis.set_ylim(
        0,
        1,
    )

    axis.set_yticks(
        [
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ]
    )

    axis.set_yticklabels(
        [
            "0.2",
            "0.4",
            "0.6",
            "0.8",
            "1.0",
        ],
        color=MID_GRAY,
        fontsize=8,
    )

    axis.grid(
        color=LIGHT_GRAY,
        alpha=0.9,
    )

    axis.set_title(
        title,
        color=BLACK,
        fontweight="bold",
        pad=24,
    )

    axis.legend(
        loc="upper right",
        bbox_to_anchor=(
            1.2,
            1.12,
        ),
        frameon=False,
    )

    figure.tight_layout()

    return figure

def class_support_bar(
    metrics: dict[str, Any],
    title: str,
) -> plt.Figure:
    support = metrics.get(
        "per_class_support",
        {},
    )

    labels = list(
        support.keys()
    )

    values = [
        support[
            label
        ]
        for label in labels
    ]

    figure, axis = plt.subplots(
        figsize=(
            7.2,
            max(
                4.2,
                len(labels) * 0.38,
            ),
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    if not values:
        axis.axis(
            "off"
        )

        axis.text(
            0.5,
            0.5,
            "No class support data available",
            ha="center",
            va="center",
            color=BLACK,
        )

        return figure

    positions = np.arange(
        len(labels)
    )

    support_color = (
        RED
        if "Benchmark A" in title
        else BLUE
        if "Benchmark B" in title
        else DARK_GRAY
    )

    bars = axis.barh(
        positions,
        values,
        color=support_color,
        alpha=0.92,
    )

    axis.set_yticks(
        positions
    )

    axis.set_yticklabels(
        labels
    )

    axis.invert_yaxis()

    axis.set_xlabel(
        "Number of test ROIs"
    )

    axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        color=BLACK,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        axis.annotate(
            f"{int(value):,}",
            (
                bar.get_width(),
                bar.get_y()
                + bar.get_height() / 2,
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


def per_class_metrics_table(
    metrics: dict[str, Any],
) -> pd.DataFrame:
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

    for index, label in enumerate(
        labels
    ):
        true_positive = matrix[
            index,
            index,
        ]

        false_positive = (
            matrix[
                :,
                index,
            ].sum()
            - true_positive
        )

        false_negative = (
            matrix[
                index,
                :,
            ].sum()
            - true_positive
        )

        support = matrix[
            index,
            :,
        ].sum()

        precision_denominator = (
            true_positive
            + false_positive
        )

        recall_denominator = (
            true_positive
            + false_negative
        )

        precision = (
            true_positive
            / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive
            / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        rows.append(
            {
                "Class": label,
                "Precision": float(
                    precision
                ),
                "Recall": float(
                    recall
                ),
                "F1": float(
                    f1
                ),
                "Support": int(
                    support
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def precision_recall_by_class_chart(
    metrics: dict[str, Any],
    title: str,
    benchmark: str,
) -> plt.Figure:
    """Line graph of per-class Precision and Recall."""
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

    # Keep the same red/blue visual language as the Hybrid dashboard:
    # red = Precision, blue = Recall.
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
        title,
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


def confusion_figure(
    metrics: dict[str, Any],
    title: str,
) -> plt.Figure:
    labels = metrics.get(
        "class_labels",
        [],
    )

    matrix = np.asarray(
        metrics.get(
            "confusion_matrix",
            [],
        )
    )

    if (
        matrix.ndim != 2
        or not labels
    ):
        figure, axis = plt.subplots(
            figsize=(
                6,
                4,
            )
        )

        axis.axis(
            "off"
        )

        axis.text(
            0.5,
            0.5,
            "No confusion matrix available",
            ha="center",
            va="center",
            color=BLACK,
        )

        return figure

    size = max(
        5.4,
        min(
            8.5,
            len(labels) * 0.67,
        ),
    )

    figure, axis = plt.subplots(
        figsize=(
            size,
            size * 0.9,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    image = axis.imshow(
        matrix,
        cmap="Blues",
    )

    figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
        label="ROI count",
    )

    axis.set_title(
        title,
        fontsize=12,
        fontweight="bold",
        color=BLACK,
    )

    axis.set_xlabel(
        "Predicted label"
    )

    axis.set_ylabel(
        "True label"
    )

    axis.set_xticks(
        range(
            len(labels)
        )
    )

    axis.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        range(
            len(labels)
        )
    )

    axis.set_yticklabels(
        labels
    )

    if (
        matrix.size
        and len(labels) <= 12
    ):
        threshold = (
            matrix.max() / 2
            if matrix.max()
            else 0
        )

        for row, column in np.ndindex(
            matrix.shape
        ):
            axis.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column,
                    ]
                ),
                ha="center",
                va="center",
                fontsize=7.5,
                color=(
                    WHITE
                    if matrix[
                        row,
                        column,
                    ] > threshold
                    else BLACK
                ),
            )

    figure.tight_layout()

    return figure

# 10. DISPLAY HELPERS

def show_description(
    text: str,
) -> None:
    st.markdown(
        f"""
        <div class="description-card">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_chart(
    figure: plt.Figure,
) -> None:
    with st.container(
        border=True
    ):
        st.pyplot(
            figure,
            clear_figure=True,
        )

def show_table(
    table: pd.DataFrame,
    formats: dict[str, str] | None = None,
) -> None:
    if table.empty:
        st.info(
            "No saved results are available for this section."
        )

        return

    formats = formats or {}

    styled = (
        table.style
        .format(
            formats,
            na_rep="Not available",
        )
        .hide(
            axis="index"
        )
    )

    st.markdown(
        (
            '<div class="comparison-table">'
            f"{styled.to_html()}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

def format_score(
    value: float,
    decimals: int = 4,
) -> str:
    if pd.isna(
        value
    ):
        return "N/A"

    return f"{value:.{decimals}f}"

def show_stage_winner(
    stage_name: str,
    table: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> None:
    score_row = scorecard.loc[
        scorecard[
            "Stage"
        ] == stage_name
    ]

    if score_row.empty:
        return

    result = score_row.iloc[0]

    (
        primary_metric,
        primary_winner,
        primary_value,
        direction_text,
    ) = best_primary_metric(
        stage_name,
        table,
    )

    columns = st.columns(
        4
    )

    columns[0].metric(
        "Stage Winner",
        result[
            "Winner"
        ],
    )

    columns[1].metric(
        "Benchmark A Stage Score",
        f"{result['Benchmark A Score']:.1f}",
    )

    columns[2].metric(
        "Benchmark B Stage Score",
        f"{result['Benchmark B Score']:.1f}",
    )

    best_value_text = (
        format_score(
            primary_value,
            4,
        )
        if not pd.isna(
            primary_value
        )
        else "N/A"
    )

    columns[3].metric(
        f"Best {primary_metric}",
        best_value_text,
        help=(
            f"{primary_winner}. {direction_text}"
            if primary_winner != "Not available"
            else "No saved primary metric available"
        ),
    )

    if (
        primary_winner
        != "Not available"
    ):
        st.markdown(
            f"""
            <div class="winner-strip">
                <strong>{primary_winner}</strong> has the best
                <strong>{primary_metric}</strong> result for this stage at
                <strong>{best_value_text}</strong>.
                {direction_text}.
            </div>
            """,
            unsafe_allow_html=True,
        )

# 11. EXPORT

def export_csv(
    filename: str,
    sections: list[tuple[str, pd.DataFrame]],
) -> Path:
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = []

    for section, table in sections:
        if table.empty:
            continue

        copy = table.copy()

        copy.insert(
            0,
            "Section",
            section,
        )

        frames.append(
            copy
        )

    path = REPORT_ROOT / filename

    if frames:
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        ).to_csv(
            path,
            index=False,
        )

    else:
        pd.DataFrame(
            {
                "Message": [
                    "No data available"
                ]
            }
        ).to_csv(
            path,
            index=False,
        )

    return path

def export_pdf(
    filename: str,
    charts: list[plt.Figure],
) -> Path:
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = REPORT_ROOT / filename

    try:
        with PdfPages(
            path
        ) as pdf:
            for figure in charts:
                pdf.savefig(
                    figure,
                    bbox_inches="tight",
                )

    finally:
        for figure in charts:
            plt.close(
                figure
            )

    return path

# 11A. COMPLETE PDF REPORT

def report_value(
    value: Any,
    decimals: int = 4,
) -> str:
    """Format values for the exported PDF report."""
    if value is None:
        return "Not available"

    try:
        if pd.isna(value):
            return "Not available"
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        return f"{float(value):.{decimals}f}"

    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return f"{int(value):,}"

    return str(value)

def add_report_footer(
    figure: plt.Figure,
    page_number: int | None = None,
) -> None:
    """Add a consistent footer to a PDF report page."""
    footer = "Benchmark Comparison Report"

    if page_number is not None:
        footer = (
            f"{footer} | Page {page_number}"
        )

    figure.text(
        0.98,
        0.018,
        footer,
        ha="right",
        va="bottom",
        fontsize=8,
        color=MID_GRAY,
    )

def report_cover_figure(
    overall_winner: str,
    total_a: float,
    total_b: float,
    metrics_compared: int,
) -> plt.Figure:
    """Create the first page of the PDF report."""
    figure = plt.figure(
        figsize=(
            8.27,
            11.69,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis = figure.add_axes(
        [
            0,
            0,
            1,
            1,
        ]
    )

    axis.axis(
        "off"
    )

    axis.add_patch(
        plt.Rectangle(
            (
                0,
                0.88,
            ),
            1,
            0.12,
            transform=axis.transAxes,
            color=BLACK,
        )
    )

    axis.add_patch(
        plt.Rectangle(
            (
                0,
                0.872,
            ),
            0.50,
            0.008,
            transform=axis.transAxes,
            color=RED,
        )
    )

    axis.add_patch(
        plt.Rectangle(
            (
                0.50,
                0.872,
            ),
            0.50,
            0.008,
            transform=axis.transAxes,
            color=BLUE,
        )
    )

    axis.text(
        0.07,
        0.78,
        "Benchmark Comparison Report",
        fontsize=28,
        fontweight="bold",
        color=BLACK,
        va="top",
    )

    axis.text(
        0.07,
        0.72,
        "Benchmark A vs Benchmark B",
        fontsize=15,
        fontweight="bold",
        color=DARK_GRAY,
        va="top",
    )

    description = (
        "This report summarizes the saved experimental results used to compare "
        "Benchmark A and Benchmark B across the evaluated image processing and "
        "classification stages. Classification results use the Test set, while "
        "upstream image processing results use Validation results."
    )

    axis.text(
        0.07,
        0.64,
        textwrap.fill(
            description,
            width=78,
        ),
        fontsize=11,
        color=DARK_GRAY,
        va="top",
        linespacing=1.55,
    )

    axis.text(
        0.07,
        0.49,
        "Overall Stage Winner",
        fontsize=11,
        fontweight="bold",
        color=MID_GRAY,
    )

    axis.text(
        0.07,
        0.445,
        overall_winner,
        fontsize=27,
        fontweight="bold",
        color=(
            RED
            if overall_winner == "Benchmark A"
            else BLUE
            if overall_winner == "Benchmark B"
            else BLACK
        ),
    )

    card_specs = [
        (
            "Benchmark A Points",
            f"{total_a:.1f}",
            RED,
        ),
        (
            "Benchmark B Points",
            f"{total_b:.1f}",
            BLUE,
        ),
        (
            "Metrics Compared",
            str(metrics_compared),
            BLACK,
        ),
    ]

    start_x = 0.07
    card_width = 0.26
    gap = 0.035

    for index, (
        label,
        value,
        accent,
    ) in enumerate(
        card_specs
    ):
        x = (
            start_x
            + index
            * (
                card_width
                + gap
            )
        )

        axis.add_patch(
            plt.Rectangle(
                (
                    x,
                    0.29,
                ),
                card_width,
                0.105,
                transform=axis.transAxes,
                facecolor=VERY_LIGHT_GRAY,
                edgecolor=LIGHT_GRAY,
                linewidth=1.2,
            )
        )

        axis.add_patch(
            plt.Rectangle(
                (
                    x,
                    0.385,
                ),
                card_width,
                0.010,
                transform=axis.transAxes,
                facecolor=accent,
                edgecolor=accent,
            )
        )

        axis.text(
            x + 0.015,
            0.357,
            label,
            fontsize=9,
            fontweight="bold",
            color=DARK_GRAY,
        )

        axis.text(
            x + 0.015,
            0.312,
            value,
            fontsize=19,
            fontweight="bold",
            color=BLACK,
        )

    axis.text(
        0.07,
        0.19,
        "Report Contents",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
    )

    contents = (
        "Benchmark algorithm configuration\n"
        "Overall prediction performance\n"
        "Stage comparison scorecard\n"
        "Individual stage descriptions and result tables\n"
        "Stage winner and strongest primary metric\n"
        "Performance charts and efficiency charts\n"
        "Fruit type and ripeness confusion matrices\n"
        "Class support distributions\n"
        "Radar charts and stage winner distribution\n"
        "Final Hybrid Pipeline selection summary"
    )

    axis.text(
        0.09,
        0.155,
        contents,
        fontsize=10,
        color=DARK_GRAY,
        va="top",
        linespacing=1.55,
    )

    figure.tight_layout()

    return figure

def report_table_figure(
    title: str,
    subtitle: str,
    table: pd.DataFrame,
    formats: dict[str, str] | None = None,
) -> plt.Figure:
    """Create one PDF page containing a title, description and table."""
    formats = formats or {}

    display = table.copy()

    for column in display.columns:
        if column in formats:
            format_string = formats[column]

            display[column] = display[column].map(
                lambda value: (
                    "Not available"
                    if pd.isna(value)
                    else format_string.format(value)
                )
            )
        else:
            display[column] = display[column].map(
                lambda value: report_value(
                    value
                )
            )

    figure, axis = plt.subplots(
        figsize=(
            11.69,
            8.27,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis.axis(
        "off"
    )

    axis.text(
        0.03,
        0.95,
        title,
        transform=axis.transAxes,
        fontsize=20,
        fontweight="bold",
        color=BLACK,
        va="top",
    )

    axis.text(
        0.03,
        0.885,
        textwrap.fill(
            subtitle,
            width=135,
        ),
        transform=axis.transAxes,
        fontsize=9.5,
        color=DARK_GRAY,
        va="top",
        linespacing=1.4,
    )

    wrapped_columns = [
        textwrap.fill(
            str(column),
            width=20,
        )
        for column in display.columns
    ]

    cell_values = []

    for row in display.astype(
        str
    ).values:
        wrapped_row = []

        for value in row:
            wrapped_row.append(
                textwrap.fill(
                    value,
                    width=24,
                )
            )

        cell_values.append(
            wrapped_row
        )

    table_height = min(
        0.62,
        0.12
        + 0.065
        * max(
            1,
            len(display),
        ),
    )

    rendered = axis.table(
        cellText=cell_values,
        colLabels=wrapped_columns,
        cellLoc="left",
        colLoc="left",
        bbox=[
            0.03,
            0.14,
            0.94,
            table_height,
        ],
    )

    rendered.auto_set_font_size(
        False
    )

    rendered.set_fontsize(
        8
    )

    for (
        row_index,
        column_index,
    ), cell in rendered.get_celld().items():
        cell.set_edgecolor(
            "#9CA3AF"
        )

        cell.set_linewidth(
            0.8
        )

        if row_index == 0:
            cell.set_facecolor(
                BLACK
            )

            cell.get_text().set_color(
                WHITE
            )

            cell.get_text().set_fontweight(
                "bold"
            )

        else:
            cell.set_facecolor(
                WHITE
                if row_index % 2
                else VERY_LIGHT_GRAY
            )

            cell.get_text().set_color(
                BLACK
            )

            if (
                "Benchmark"
                in display.columns
                and column_index == 0
            ):
                benchmark_text = str(
                    display.iloc[
                        row_index - 1,
                        0,
                    ]
                )

                if benchmark_text == "Benchmark A":
                    cell.get_text().set_color(
                        RED_DARK
                    )

                    cell.get_text().set_fontweight(
                        "bold"
                    )

                elif benchmark_text == "Benchmark B":
                    cell.get_text().set_color(
                        BLUE_DARK
                    )

                    cell.get_text().set_fontweight(
                        "bold"
                    )

    return figure

def report_stage_summary_figure(
    stage_name: str,
    stage_table: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> plt.Figure:
    """Create a stage summary page with description, winner and table."""
    score_row = scorecard.loc[
        scorecard[
            "Stage"
        ] == stage_name
    ]

    if score_row.empty:
        result = None
    else:
        result = score_row.iloc[
            0
        ]

    (
        primary_metric,
        primary_winner,
        primary_value,
        direction_text,
    ) = best_primary_metric(
        stage_name,
        stage_table,
    )

    figure, axis = plt.subplots(
        figsize=(
            11.69,
            8.27,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis.axis(
        "off"
    )

    axis.text(
        0.035,
        0.95,
        stage_name,
        transform=axis.transAxes,
        fontsize=21,
        fontweight="bold",
        color=BLACK,
        va="top",
    )

    description = STAGE_DESCRIPTIONS.get(
        stage_name,
        "This stage compares Benchmark A and Benchmark B using the saved experimental metrics.",
    )

    axis.text(
        0.035,
        0.88,
        textwrap.fill(
            description,
            width=138,
        ),
        transform=axis.transAxes,
        fontsize=9.5,
        color=DARK_GRAY,
        va="top",
        linespacing=1.4,
    )

    if result is not None:
        winner = result[
            "Winner"
        ]

        winner_color = (
            RED
            if winner == "Benchmark A"
            else BLUE
            if winner == "Benchmark B"
            else BLACK
        )

        summary_cards = [
            (
                "Stage Winner",
                winner,
                winner_color,
            ),
            (
                "Benchmark A Score",
                f"{result['Benchmark A Score']:.1f}",
                RED,
            ),
            (
                "Benchmark B Score",
                f"{result['Benchmark B Score']:.1f}",
                BLUE,
            ),
            (
                f"Best {primary_metric}",
                report_value(
                    primary_value,
                    4,
                ),
                winner_color,
            ),
        ]

        card_width = 0.215
        gap = 0.016

        for index, (
            label,
            value,
            accent,
        ) in enumerate(
            summary_cards
        ):
            x = (
                0.035
                + index
                * (
                    card_width
                    + gap
                )
            )

            axis.add_patch(
                plt.Rectangle(
                    (
                        x,
                        0.635,
                    ),
                    card_width,
                    0.115,
                    transform=axis.transAxes,
                    facecolor=VERY_LIGHT_GRAY,
                    edgecolor="#AEB5BF",
                    linewidth=1.0,
                )
            )

            axis.add_patch(
                plt.Rectangle(
                    (
                        x,
                        0.742,
                    ),
                    card_width,
                    0.008,
                    transform=axis.transAxes,
                    facecolor=accent,
                    edgecolor=accent,
                )
            )

            axis.text(
                x + 0.012,
                0.711,
                textwrap.fill(
                    label,
                    width=24,
                ),
                transform=axis.transAxes,
                fontsize=8,
                fontweight="bold",
                color=DARK_GRAY,
                va="top",
            )

            axis.text(
                x + 0.012,
                0.655,
                str(
                    value
                ),
                transform=axis.transAxes,
                fontsize=14,
                fontweight="bold",
                color=BLACK,
                va="bottom",
            )

        if primary_winner != "Not available":
            winner_text = (
                f"{winner} is the overall winner for this stage based on "
                f"{int(result['Metrics Compared'])} available performance metrics. "
                f"The strongest primary result is {primary_metric}, achieved by "
                f"{primary_winner} at {report_value(primary_value, 4)}. "
                f"{direction_text}."
            )

            axis.text(
                0.035,
                0.585,
                textwrap.fill(
                    winner_text,
                    width=140,
                ),
                transform=axis.transAxes,
                fontsize=9,
                color=DARK_GRAY,
                va="top",
                linespacing=1.35,
            )

    display = stage_table.copy()

    for column in display.columns:
        display[column] = display[column].map(
            lambda value: report_value(
                value,
                4,
            )
        )

    wrapped_columns = [
        textwrap.fill(
            str(column),
            width=18,
        )
        for column in display.columns
    ]

    cell_values = [
        [
            textwrap.fill(
                str(value),
                width=20,
            )
            for value in row
        ]
        for row in display.astype(
            str
        ).values
    ]

    rendered = axis.table(
        cellText=cell_values,
        colLabels=wrapped_columns,
        cellLoc="left",
        colLoc="left",
        bbox=[
            0.035,
            0.12,
            0.93,
            0.36,
        ],
    )

    rendered.auto_set_font_size(
        False
    )

    rendered.set_fontsize(
        7.5
    )

    for (
        row_index,
        column_index,
    ), cell in rendered.get_celld().items():
        cell.set_edgecolor(
            "#9CA3AF"
        )

        cell.set_linewidth(
            0.75
        )

        if row_index == 0:
            cell.set_facecolor(
                BLACK
            )

            cell.get_text().set_color(
                WHITE
            )

            cell.get_text().set_fontweight(
                "bold"
            )

        else:
            cell.set_facecolor(
                WHITE
                if row_index % 2
                else VERY_LIGHT_GRAY
            )

            cell.get_text().set_color(
                BLACK
            )

            if column_index == 0:
                benchmark_text = str(
                    display.iloc[
                        row_index - 1,
                        0,
                    ]
                )

                if benchmark_text == "Benchmark A":
                    cell.get_text().set_color(
                        RED_DARK
                    )

                    cell.get_text().set_fontweight(
                        "bold"
                    )

                elif benchmark_text == "Benchmark B":
                    cell.get_text().set_color(
                        BLUE_DARK
                    )

                    cell.get_text().set_fontweight(
                        "bold"
                    )

    return figure

def stage_report_charts(
    stage_name: str,
    table: pd.DataFrame,
) -> list[plt.Figure]:
    """Return all useful charts for one stage in the PDF report."""
    charts: list[plt.Figure] = []

    if stage_name == "3. Preprocessing":
        charts.extend(
            [
                grouped_bar(
                    table,
                    [
                        "BRISQUE",
                        "NIQE",
                        "PIQE",
                    ],
                    "Preprocessing No Reference Image Quality",
                    y_label="Score",
                ),
                single_metric_chart(
                    table,
                    "SSIM",
                    "Preprocessing Structural Similarity",
                    lower_is_better=False,
                ),
                single_metric_chart(
                    table,
                    "Time (ms)",
                    "Preprocessing Time",
                    lower_is_better=True,
                ),
            ]
        )

    elif stage_name in {
        "4. Segmentation",
        "5. Morphological Processing",
    }:
        charts.extend(
            [
                grouped_bar(
                    table,
                    [
                        "Boundary Edge Alignment",
                        "Region Uniformity",
                    ],
                    f"{stage_name} Mask Quality",
                    y_label="Score",
                ),
                single_metric_chart(
                    table,
                    "Foreground Background Contrast",
                    f"{stage_name} Foreground Background Contrast",
                    lower_is_better=False,
                ),
                single_metric_chart(
                    table,
                    "Time (ms)",
                    f"{stage_name} Processing Time",
                    lower_is_better=True,
                ),
            ]
        )

    elif stage_name == "6. Fruit Detection":
        charts.extend(
            [
                single_metric_chart(
                    table,
                    "Detection Accuracy",
                    "Fruit Detection Accuracy",
                    lower_is_better=False,
                ),
                single_metric_chart(
                    table,
                    "Time (ms)",
                    "Fruit Detection Time",
                    lower_is_better=True,
                ),
            ]
        )

    elif stage_name == "7. Fruit Counting":
        charts.extend(
            [
                single_metric_chart(
                    table,
                    "Exact Count Accuracy",
                    "Fruit Counting Exact Count Accuracy",
                    lower_is_better=False,
                ),
                single_metric_chart(
                    table,
                    "MAE",
                    "Fruit Counting Mean Absolute Error",
                    lower_is_better=True,
                ),
                single_metric_chart(
                    table,
                    "MAPE",
                    "Fruit Counting Mean Absolute Percentage Error",
                    lower_is_better=True,
                ),
                single_metric_chart(
                    table,
                    "Time (ms)",
                    "Fruit Counting Time",
                    lower_is_better=True,
                ),
            ]
        )

    elif stage_name == "10. Feature Extraction":
        charts.extend(
            [
                single_metric_chart(
                    table,
                    "Fisher Score",
                    "Feature Extraction Fisher Score",
                    lower_is_better=False,
                ),
                single_metric_chart(
                    table,
                    "Silhouette Score",
                    "Feature Extraction Silhouette Score",
                    lower_is_better=False,
                ),
                single_metric_chart(
                    table,
                    "Time (ms/ROI)",
                    "Feature Extraction Time",
                    lower_is_better=True,
                ),
                single_metric_chart(
                    table,
                    "Feature Size",
                    "Feature Vector Size",
                    lower_is_better=True,
                ),
            ]
        )

    elif stage_name in {
        "11. Fruit Type Classification",
        "12. Ripeness Classification",
    }:
        charts.extend(
            [
                grouped_bar(
                    table,
                    [
                        "Accuracy",
                        "Macro Precision",
                        "Macro Recall",
                        "Macro F1",
                    ],
                    f"{stage_name} Performance",
                    zero_to_one=True,
                ),
                classification_radar(
                    table,
                    f"{stage_name} Radar",
                ),
                single_metric_chart(
                    table,
                    "Training Time (s)",
                    f"{stage_name} Training Time",
                    lower_is_better=True,
                ),
                single_metric_chart(
                    table,
                    "Prediction Time (ms/ROI)",
                    f"{stage_name} Prediction Time",
                    lower_is_better=True,
                ),
            ]
        )

    return charts

def report_hybrid_summary_figure(
    scorecard: pd.DataFrame,
    overall_winner: str,
    total_a: float,
    total_b: float,
) -> plt.Figure:
    """Create the final Hybrid Pipeline selection summary page."""
    figure, axis = plt.subplots(
        figsize=(
            11.69,
            8.27,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    axis.axis(
        "off"
    )

    axis.text(
        0.04,
        0.94,
        "Final Hybrid Pipeline Selection Summary",
        transform=axis.transAxes,
        fontsize=22,
        fontweight="bold",
        color=BLACK,
        va="top",
    )

    explanation = (
        "The Hybrid Pipeline can be assembled using the winning technique from each "
        "evaluated stage. The stage score is based only on saved metrics that are "
        "available for both benchmarks. Missing metrics are not assigned artificial values."
    )

    axis.text(
        0.04,
        0.86,
        textwrap.fill(
            explanation,
            width=135,
        ),
        transform=axis.transAxes,
        fontsize=10,
        color=DARK_GRAY,
        va="top",
        linespacing=1.45,
    )

    winner_color = (
        RED
        if overall_winner == "Benchmark A"
        else BLUE
        if overall_winner == "Benchmark B"
        else BLACK
    )

    axis.add_patch(
        plt.Rectangle(
            (
                0.04,
                0.69,
            ),
            0.92,
            0.10,
            transform=axis.transAxes,
            facecolor=VERY_LIGHT_GRAY,
            edgecolor=winner_color,
            linewidth=2,
        )
    )

    axis.text(
        0.065,
        0.755,
        "Overall Stage Winner",
        transform=axis.transAxes,
        fontsize=9,
        fontweight="bold",
        color=MID_GRAY,
    )

    axis.text(
        0.065,
        0.705,
        overall_winner,
        transform=axis.transAxes,
        fontsize=21,
        fontweight="bold",
        color=winner_color,
    )

    axis.text(
        0.47,
        0.733,
        f"Benchmark A Points  {total_a:.1f}",
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=RED_DARK,
    )

    axis.text(
        0.71,
        0.733,
        f"Benchmark B Points  {total_b:.1f}",
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        color=BLUE_DARK,
    )

    hybrid_table = scorecard[
        [
            "Stage",
            "Winner",
            "Benchmark A Score",
            "Benchmark B Score",
            "Metrics Compared",
        ]
    ].copy()

    hybrid_table[
        "Benchmark A Score"
    ] = hybrid_table[
        "Benchmark A Score"
    ].map(
        lambda value: f"{value:.1f}"
    )

    hybrid_table[
        "Benchmark B Score"
    ] = hybrid_table[
        "Benchmark B Score"
    ].map(
        lambda value: f"{value:.1f}"
    )

    wrapped_columns = [
        textwrap.fill(
            str(column),
            width=20,
        )
        for column in hybrid_table.columns
    ]

    cell_values = [
        [
            textwrap.fill(
                str(value),
                width=27,
            )
            for value in row
        ]
        for row in hybrid_table.astype(
            str
        ).values
    ]

    rendered = axis.table(
        cellText=cell_values,
        colLabels=wrapped_columns,
        cellLoc="left",
        colLoc="left",
        bbox=[
            0.04,
            0.12,
            0.92,
            0.48,
        ],
    )

    rendered.auto_set_font_size(
        False
    )

    rendered.set_fontsize(
        7.6
    )

    for (
        row_index,
        column_index,
    ), cell in rendered.get_celld().items():
        cell.set_edgecolor(
            "#9CA3AF"
        )

        cell.set_linewidth(
            0.75
        )

        if row_index == 0:
            cell.set_facecolor(
                BLACK
            )

            cell.get_text().set_color(
                WHITE
            )

            cell.get_text().set_fontweight(
                "bold"
            )

        else:
            cell.set_facecolor(
                WHITE
                if row_index % 2
                else VERY_LIGHT_GRAY
            )

            cell.get_text().set_color(
                BLACK
            )

            if column_index == 1:
                winner = str(
                    hybrid_table.iloc[
                        row_index - 1,
                        column_index,
                    ]
                )

                if winner == "Benchmark A":
                    cell.get_text().set_color(
                        RED_DARK
                    )

                    cell.get_text().set_fontweight(
                        "bold"
                    )

                elif winner == "Benchmark B":
                    cell.get_text().set_color(
                        BLUE_DARK
                    )

                    cell.get_text().set_fontweight(
                        "bold"
                    )

    return figure

def export_full_pdf_report(
    filename: str,
    data: dict[str, Any],
    final_table: pd.DataFrame,
    pipeline_scores: pd.DataFrame,
    fruit_scores: pd.DataFrame,
    ripeness_scores: pd.DataFrame,
    stage_tables: list[tuple[str, pd.DataFrame]],
    scorecard: pd.DataFrame,
) -> Path:
    """Export a complete report with text, tables, winners and diagrams."""
    REPORT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = REPORT_ROOT / filename

    total_a = float(
        scorecard[
            "Benchmark A Score"
        ].sum()
    )

    total_b = float(
        scorecard[
            "Benchmark B Score"
        ].sum()
    )

    metrics_compared = int(
        scorecard[
            "Metrics Compared"
        ].sum()
    )

    if np.isclose(
        total_a,
        total_b,
    ):
        overall_winner = "Tie"
    else:
        overall_winner = (
            "Benchmark A"
            if total_a > total_b
            else "Benchmark B"
        )

    figures: list[plt.Figure] = []

    figures.append(
        report_cover_figure(
            overall_winner,
            total_a,
            total_b,
            metrics_compared,
        )
    )

    figures.append(
        report_table_figure(
            "Benchmark Algorithm Configuration",
            (
                "This table lists the exact algorithms and locked parameters used "
                "in Benchmark A and Benchmark B for every major evaluated stage."
            ),
            BENCHMARK_ALGORITHM_TABLE,
        )
    )

    figures.append(
        report_table_figure(
            "Overall Prediction Performance",
            (
                "This table compares the final fruit count, fruit type and ripeness "
                "prediction accuracy for Benchmark A and Benchmark B."
            ),
            final_table,
            {
                "Fruit Count Accuracy": "{:.4f}",
                "Fruit Type Accuracy": "{:.4f}",
                "Ripeness Accuracy": "{:.4f}",
            },
        )
    )

    figures.append(
        grouped_bar(
            final_table,
            [
                "Fruit Count Accuracy",
                "Fruit Type Accuracy",
                "Ripeness Accuracy",
            ],
            "Overall Final Prediction Accuracy",
            zero_to_one=True,
        )
    )

    figures.append(
        report_table_figure(
            "Overall Stage Scorecard",
            (
                "Each available metric gives one comparison point to the better benchmark. "
                "Higher is better for quality and accuracy metrics. Lower is better for "
                "error and processing time metrics."
            ),
            scorecard,
            {
                "Benchmark A Score": "{:.1f}",
                "Benchmark B Score": "{:.1f}",
            },
        )
    )

    figures.append(
        scorecard_chart(
            scorecard
        )
    )

    figures.append(
        stage_wins_pie(
            scorecard
        )
    )

    for stage_name, stage_table in stage_tables:
        figures.append(
            report_stage_summary_figure(
                stage_name,
                stage_table,
                scorecard,
            )
        )

        figures.extend(
            stage_report_charts(
                stage_name,
                stage_table,
            )
        )

        if stage_name == "11. Fruit Type Classification":
            for benchmark in [
                "Benchmark A",
                "Benchmark B",
            ]:
                metrics = data[
                    benchmark
                ].get(
                    "fruit_metrics"
                )

                if metrics:
                    figures.append(
                        precision_recall_by_class_chart(
                            metrics,
                            f"{benchmark} Fruit Type — Precision and Recall by Class",
                            benchmark,
                        )
                    )

                    figures.append(
                        class_support_bar(
                            metrics,
                            f"{benchmark} Fruit Type Test Support",
                        )
                    )

                    figures.append(
                        confusion_figure(
                            metrics,
                            f"{benchmark} Fruit Type Confusion Matrix",
                        )
                    )

        if stage_name == "12. Ripeness Classification":
            for benchmark in [
                "Benchmark A",
                "Benchmark B",
            ]:
                metrics = data[
                    benchmark
                ].get(
                    "ripeness_metrics"
                )

                if metrics:
                    figures.append(
                        precision_recall_by_class_chart(
                            metrics,
                            f"{benchmark} Ripeness — Precision and Recall by Class",
                            benchmark,
                        )
                    )

                    figures.append(
                        class_support_bar(
                            metrics,
                            f"{benchmark} Ripeness Test Support",
                        )
                    )

                    figures.append(
                        confusion_figure(
                            metrics,
                            f"{benchmark} Ripeness Confusion Matrix",
                        )
                    )

    figures.append(
        report_table_figure(
            "Fruit Type Classification Test Results",
            (
                "Fruit type classification is evaluated using Accuracy, Macro Precision, "
                "Macro Recall, Macro F1, Training Time and Prediction Time."
            ),
            fruit_scores,
            {
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Training Time (s)": "{:.3f}",
                "Prediction Time (ms/ROI)": "{:.4f}",
            },
        )
    )

    figures.append(
        report_table_figure(
            "Ripeness Classification Test Results",
            (
                "Ripeness classification is evaluated using Accuracy, Macro Precision, "
                "Macro Recall, Macro F1, Training Time and Prediction Time."
            ),
            ripeness_scores,
            {
                "Accuracy": "{:.4f}",
                "Macro Precision": "{:.4f}",
                "Macro Recall": "{:.4f}",
                "Macro F1": "{:.4f}",
                "Training Time (s)": "{:.3f}",
                "Prediction Time (ms/ROI)": "{:.4f}",
            },
        )
    )

    figures.append(
        classification_radar(
            fruit_scores,
            "Fruit Type Classification Radar",
        )
    )

    figures.append(
        classification_radar(
            ripeness_scores,
            "Ripeness Classification Radar",
        )
    )

    figures.append(
        report_hybrid_summary_figure(
            scorecard,
            overall_winner,
            total_a,
            total_b,
        )
    )

    try:
        with PdfPages(
            path
        ) as pdf:
            metadata = pdf.infodict()

            metadata[
                "Title"
            ] = "Benchmark Comparison Report"

            metadata[
                "Subject"
            ] = (
                "Benchmark A and Benchmark B performance comparison"
            )

            metadata[
                "Author"
            ] = "Smart Fruit Image Analysis System"

            for page_number, figure in enumerate(
                figures,
                start=1,
            ):
                add_report_footer(
                    figure,
                    page_number,
                )

                pdf.savefig(
                    figure,
                    bbox_inches="tight",
                    facecolor=WHITE,
                )

    finally:
        for figure in figures:
            plt.close(
                figure
            )

    return path

# 12. STREAMLIT CONFIGURATION

st.set_page_config(
    page_title="Benchmark Comparison Dashboard",
    page_icon="📊",
    layout="wide",
)

# 13. STREAMLIT CSS

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

        [data-testid="stSidebar"] [role="radiogroup"] label:hover * {{
            color: {WHITE} !important;
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
            border-bottom: none !important;
            padding-bottom: 0.20rem;
        }}

        h3 {{
            color: {BLUE_DARK} !important;
            font-weight: 700 !important;
        }}

        p,
        label,
        span {{
            color: {BLACK};
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
            transition: 0.18s ease;
        }}

        [data-testid="stMetric"]:hover {{
            background: {WHITE} !important;
            border: 1px solid #C9CED6 !important;
            border-top: 4px solid {BLACK} !important;
            transform: none !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07) !important;
        }}

        [data-testid="stMetric"]:hover * {{
            color: {BLACK} !important;
        }}

        [data-testid="stMetricLabel"] p {{
            color: {DARK_GRAY} !important;
            font-weight: 700;
        }}

        [data-testid="stMetricValue"] {{
            color: {BLACK} !important;
            font-weight: 800;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {WHITE} !important;
            border: 1px solid #C9CED6 !important;
            border-radius: 10px;
            padding: 0.8rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }}

        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            border-color: #AEB5BF !important;
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        }}

        [data-testid="stColumn"]:last-child div.stButton {{
            padding-top: 0.15rem;
        }}

        [data-testid="stColumn"]:last-child div.stButton > button {{
            min-height: 2.80rem;
            width: 100%;
        }}

        div.stButton > button,
        [data-testid="stDownloadButton"] > button,
        div.stDownloadButton > button {{
            background: {RED} !important;
            color: {WHITE} !important;
            border: 2px solid {RED} !important;
            border-radius: 8px;
            font-weight: 700;
            transition: 0.18s ease;
        }}

        div.stButton > button p,
        div.stButton > button span,
        [data-testid="stDownloadButton"] > button p,
        [data-testid="stDownloadButton"] > button span,
        div.stDownloadButton > button p,
        div.stDownloadButton > button span {{
            color: {WHITE} !important;
        }}

        div.stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover,
        div.stDownloadButton > button:hover {{
            background: {WHITE} !important;
            color: {RED} !important;
            border-color: {RED} !important;
            box-shadow: 0 4px 10px rgba(193, 18, 31, 0.12);
        }}

        div.stButton > button:hover p,
        div.stButton > button:hover span,
        [data-testid="stDownloadButton"] > button:hover p,
        [data-testid="stDownloadButton"] > button:hover span,
        div.stDownloadButton > button:hover p,
        div.stDownloadButton > button:hover span {{
            color: {RED} !important;
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

        [data-testid="stAlert"] {{
            border-radius: 8px;
        }}

        div[role="radiogroup"] label:hover {{
            color: {RED_DARK} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# 14. LOAD DATA

title_column, top_export_column = st.columns(
    [
        5,
        1.20,
    ],
    vertical_alignment="center",
)

with title_column:
    st.title(
        "Benchmark Comparison Dashboard"
    )

with top_export_column:
    top_export_placeholder = st.empty()

st.caption(
    "Benchmark A vs Benchmark B | Classification uses Test results | "
    "Upstream image processing uses Validation results"
)

data = load_dashboard_data()

fruit_scores = classification_table(
    data,
    "Fruit Type",
)

ripeness_scores = classification_table(
    data,
    "Ripeness",
)

pipeline_scores = pipeline_summary_table(
    data
)

if (
    fruit_scores.empty
    and ripeness_scores.empty
):
    st.error(
        "No saved classification results were found under "
        "results/pipeline_a or results/pipeline_b."
    )

    st.stop()

stage_tables = stage_evaluation_tables(
    data,
    fruit_scores,
    ripeness_scores,
)

scorecard = stage_scorecard(
    stage_tables
)

# 15. SIDEBAR

with st.sidebar:
    st.header(
        "Dashboard"
    )

    view = st.radio(
        "Choose View",
        [
            "Overall",
            "Stage Evaluation",
            "Fruit Count",
            "Fruit Type",
            "Fruit Ripeness",
            "Metric Guide",
        ],
    )

    st.divider()

    st.caption(
        "Saved results only. "
        "No model retraining is performed."
    )

    if st.button(
        "Refresh Saved Results"
    ):
        st.cache_data.clear()
        st.rerun()

# 16. PAGE DESCRIPTION

show_description(
    VIEW_DESCRIPTIONS[
        view
    ]
)

# 17. OVERALL VIEW

if view == "Overall":
    st.header(
        "Overall Benchmark Comparison"
    )

    total_a = float(
        scorecard[
            "Benchmark A Score"
        ].sum()
    )

    total_b = float(
        scorecard[
            "Benchmark B Score"
        ].sum()
    )

    if np.isclose(
        total_a,
        total_b,
    ):
        overall_winner = "Tie"

    else:
        overall_winner = (
            "Benchmark A"
            if total_a > total_b
            else "Benchmark B"
        )

    top_cards = st.columns(
        4
    )

    top_cards[0].metric(
        "Overall Stage Winner",
        overall_winner,
    )

    top_cards[1].metric(
        "Benchmark A Points",
        f"{total_a:.1f}",
    )

    top_cards[2].metric(
        "Benchmark B Points",
        f"{total_b:.1f}",
    )

    top_cards[3].metric(
        "Metrics Compared",
        int(
            scorecard[
                "Metrics Compared"
            ].sum()
        ),
    )

    st.subheader(
        "Benchmark Algorithms"
    )

    show_description(
        "The table below lists the exact algorithm selected for every major stage "
        "in Benchmark A and Benchmark B, including the locked parameters used in "
        "the saved experiments."
    )

    show_table(
        BENCHMARK_ALGORITHM_TABLE
    )

    final_rows = []

    for benchmark in [
        "Benchmark A",
        "Benchmark B",
    ]:
        fruit_row = fruit_scores.loc[
            fruit_scores[
                "Benchmark"
            ] == benchmark
        ]

        ripe_row = ripeness_scores.loc[
            ripeness_scores[
                "Benchmark"
            ] == benchmark
        ]

        pipeline_row = pipeline_scores.loc[
            pipeline_scores[
                "Benchmark"
            ] == benchmark
        ]

        final_rows.append(
            {
                "Benchmark": benchmark,
                "Fruit Count Accuracy": (
                    pipeline_row.iloc[0][
                        "Exact Count Accuracy"
                    ]
                    if not pipeline_row.empty
                    else np.nan
                ),
                "Fruit Type Accuracy": (
                    fruit_row.iloc[0][
                        "Accuracy"
                    ]
                    if not fruit_row.empty
                    else np.nan
                ),
                "Ripeness Accuracy": (
                    ripe_row.iloc[0][
                        "Accuracy"
                    ]
                    if not ripe_row.empty
                    else np.nan
                ),
            }
        )

    final_table = pd.DataFrame(
        final_rows
    )

    st.subheader(
        "Final Prediction Performance"
    )

    show_table(
        final_table,
        {
            "Fruit Count Accuracy": "{:.4f}",
            "Fruit Type Accuracy": "{:.4f}",
            "Ripeness Accuracy": "{:.4f}",
        },
    )

    show_chart(
        grouped_bar(
            final_table,
            [
                "Fruit Count Accuracy",
                "Fruit Type Accuracy",
                "Ripeness Accuracy",
            ],
            "Final Prediction Accuracy",
            zero_to_one=True,
        )
    )

    st.subheader(
        "Stage Winners"
    )

    left_chart, right_chart = st.columns(
        2
    )

    with left_chart:
        show_chart(
            scorecard_chart(
                scorecard
            )
        )

    with right_chart:
        show_chart(
            stage_wins_pie(
                scorecard
            )
        )

    show_table(
        scorecard,
        {
            "Benchmark A Score": "{:.1f}",
            "Benchmark B Score": "{:.1f}",
        },
    )

    st.subheader(
        "Classification Shape Comparison"
    )

    radar_left, radar_right = st.columns(
        2
    )

    with radar_left:
        show_chart(
            classification_radar(
                fruit_scores,
                "Fruit Type Classification Radar",
            )
        )

    with radar_right:
        show_chart(
            classification_radar(
                ripeness_scores,
                "Ripeness Classification Radar",
            )
        )

    st.subheader(
        "Fruit Type Classification"
    )

    show_table(
        fruit_scores,
        {
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Training Time (s)": "{:.3f}",
            "Prediction Time (ms/ROI)": "{:.4f}",
        },
    )

    st.subheader(
        "Ripeness Classification"
    )

    show_table(
        ripeness_scores,
        {
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Training Time (s)": "{:.3f}",
            "Prediction Time (ms/ROI)": "{:.4f}",
        },
    )

    if view == "Overall":
        with top_export_placeholder.container():
            if st.button(
                "Export CSV and PDF",
                key="export_overall_top",
                use_container_width=True,
            ):
                csv_path = export_csv(
                    "benchmark_comparison_overall.csv",
                    [
                        (
                            "Benchmark Algorithm Configuration",
                            BENCHMARK_ALGORITHM_TABLE,
                        ),
                        (
                            "Final Prediction Performance",
                            final_table,
                        ),
                        (
                            "Stage Scorecard",
                            scorecard,
                        ),
                        (
                            "Fruit Type Classification",
                            fruit_scores,
                        ),
                        (
                            "Ripeness Classification",
                            ripeness_scores,
                        ),
                    ],
                )

                pdf_path = export_full_pdf_report(
                    "benchmark_comparison_full_report.pdf",
                    data,
                    final_table,
                    pipeline_scores,
                    fruit_scores,
                    ripeness_scores,
                    stage_tables,
                    scorecard,
                )

                st.success(
                    f"Saved {csv_path.name} and the complete PDF report "
                    f"{pdf_path.name} to {REPORT_ROOT.relative_to(PROJECT_ROOT)}."
                )

# 18. STAGE EVALUATION VIEW

elif view == "Stage Evaluation":
    st.header(
        "Pipeline Stage Evaluation"
    )

    for stage_name, table in stage_tables:
        st.subheader(
            stage_name
        )

        show_description(
            STAGE_DESCRIPTIONS.get(
                stage_name,
                "This stage compares Benchmark A and Benchmark B using the saved experimental metrics.",
            )
        )

        show_stage_winner(
            stage_name,
            table,
            scorecard,
        )

        formats = {
            column: "{:.4f}"
            for column in table.columns
            if column not in {
                "Benchmark",
                "Technique",
                "Test ROIs",
            }
        }

        show_table(
            table,
            formats,
        )

        if stage_name == "3. Preprocessing":
            c1, c2, c3 = st.columns(
                3
            )

            with c1:
                show_chart(
                    grouped_bar(
                        table,
                        [
                            "BRISQUE",
                            "NIQE",
                            "PIQE",
                        ],
                        "No Reference Image Quality",
                        y_label="Score",
                    )
                )

            with c2:
                show_chart(
                    single_metric_chart(
                        table,
                        "SSIM",
                        "Structural Similarity",
                        lower_is_better=False,
                    )
                )

            with c3:
                show_chart(
                    single_metric_chart(
                        table,
                        "Time (ms)",
                        "Preprocessing Time",
                        lower_is_better=True,
                    )
                )

        elif stage_name in {
            "4. Segmentation",
            "5. Morphological Processing",
        }:
            c1, c2 = st.columns(
                2
            )

            with c1:
                show_chart(
                    grouped_bar(
                        table,
                        [
                            "Boundary Edge Alignment",
                            "Region Uniformity",
                        ],
                        "Mask Quality",
                        y_label="Score",
                    )
                )

            with c2:
                show_chart(
                    single_metric_chart(
                        table,
                        "Foreground Background Contrast",
                        "Foreground Background Contrast",
                        lower_is_better=False,
                    )
                )

            show_chart(
                single_metric_chart(
                    table,
                    "Time (ms)",
                    "Processing Time",
                    lower_is_better=True,
                )
            )

        elif stage_name == "6. Fruit Detection":
            c1, c2 = st.columns(
                2
            )

            with c1:
                show_chart(
                    single_metric_chart(
                        table,
                        "Detection Accuracy",
                        "Fruit Detection Accuracy",
                        lower_is_better=False,
                    )
                )

            with c2:
                show_chart(
                    single_metric_chart(
                        table,
                        "Time (ms)",
                        "Detection Time",
                        lower_is_better=True,
                    )
                )

        elif stage_name == "7. Fruit Counting":
            c1, c2 = st.columns(
                2
            )

            with c1:
                show_chart(
                    single_metric_chart(
                        table,
                        "Exact Count Accuracy",
                        "Exact Count Accuracy",
                        lower_is_better=False,
                    )
                )

            with c2:
                show_chart(
                    single_metric_chart(
                        table,
                        "MAE",
                        "Mean Absolute Error",
                        lower_is_better=True,
                    )
                )

            c3, c4 = st.columns(
                2
            )

            with c3:
                show_chart(
                    single_metric_chart(
                        table,
                        "MAPE",
                        "Mean Absolute Percentage Error",
                        lower_is_better=True,
                    )
                )

            with c4:
                show_chart(
                    single_metric_chart(
                        table,
                        "Time (ms)",
                        "Counting Time",
                        lower_is_better=True,
                    )
                )

        elif stage_name == "10. Feature Extraction":
            c1, c2 = st.columns(
                2
            )

            with c1:
                show_chart(
                    single_metric_chart(
                        table,
                        "Fisher Score",
                        "Fisher Score",
                        lower_is_better=False,
                    )
                )

            with c2:
                show_chart(
                    single_metric_chart(
                        table,
                        "Silhouette Score",
                        "Silhouette Score",
                        lower_is_better=False,
                    )
                )

            c3, c4 = st.columns(
                2
            )

            with c3:
                show_chart(
                    single_metric_chart(
                        table,
                        "Time (ms/ROI)",
                        "Feature Extraction Time",
                        lower_is_better=True,
                    )
                )

            with c4:
                show_chart(
                    single_metric_chart(
                        table,
                        "Feature Size",
                        "Feature Vector Size",
                        lower_is_better=True,
                    )
                )

        elif stage_name in {
            "11. Fruit Type Classification",
            "12. Ripeness Classification",
        }:
            show_chart(
                grouped_bar(
                    table,
                    [
                        "Accuracy",
                        "Macro Precision",
                        "Macro Recall",
                        "Macro F1",
                    ],
                    f"{stage_name} Performance",
                    zero_to_one=True,
                )
            )

            radar_title = (
                "Fruit Type Classification Radar"
                if stage_name
                == "11. Fruit Type Classification"
                else "Ripeness Classification Radar"
            )

            show_chart(
                classification_radar(
                    table,
                    radar_title,
                )
            )

            c1, c2 = st.columns(
                2
            )

            with c1:
                show_chart(
                    single_metric_chart(
                        table,
                        "Training Time (s)",
                        "Training Time",
                        lower_is_better=True,
                    )
                )

            with c2:
                show_chart(
                    single_metric_chart(
                        table,
                        "Prediction Time (ms/ROI)",
                        "Prediction Time",
                        lower_is_better=True,
                    )
                )

        st.divider()

    st.header(
        "Final Hybrid Selection Score"
    )

    total_a = scorecard[
        "Benchmark A Score"
    ].sum()

    total_b = scorecard[
        "Benchmark B Score"
    ].sum()

    if np.isclose(
        total_a,
        total_b,
    ):
        overall_winner = "Tie"

    else:
        overall_winner = (
            "Benchmark A"
            if total_a > total_b
            else "Benchmark B"
        )

    final_cards = st.columns(
        3
    )

    final_cards[0].metric(
        "Benchmark A",
        f"{total_a:.1f}",
    )

    final_cards[1].metric(
        "Benchmark B",
        f"{total_b:.1f}",
    )

    final_cards[2].metric(
        "Overall Winner",
        overall_winner,
    )

    show_table(
        scorecard,
        {
            "Benchmark A Score": "{:.1f}",
            "Benchmark B Score": "{:.1f}",
        },
    )

    stage_chart_left, stage_chart_right = st.columns(
        2
    )

    with stage_chart_left:
        show_chart(
            scorecard_chart(
                scorecard
            )
        )

    with stage_chart_right:
        show_chart(
            stage_wins_pie(
                scorecard
            )
        )

# 19. FRUIT COUNT VIEW

elif view == "Fruit Count":
    st.header(
        "Fruit Detection and Counting"
    )

    count_table = pipeline_scores[
        [
            "Benchmark",
            "Detection Accuracy",
            "Detection Time (ms)",
            "Exact Count Accuracy",
            "Counting MAE",
        ]
    ].copy()

    show_table(
        count_table,
        {
            "Detection Accuracy": "{:.4f}",
            "Detection Time (ms)": "{:.4f}",
            "Exact Count Accuracy": "{:.4f}",
            "Counting MAE": "{:.4f}",
        },
    )

    c1, c2 = st.columns(
        2
    )

    with c1:
        show_chart(
            grouped_bar(
                count_table,
                [
                    "Detection Accuracy",
                    "Exact Count Accuracy",
                ],
                "Detection and Counting Accuracy",
                zero_to_one=True,
            )
        )

    with c2:
        show_chart(
            single_metric_chart(
                count_table,
                "Counting MAE",
                "Counting Error",
                lower_is_better=True,
            )
        )

    show_chart(
        single_metric_chart(
            count_table,
            "Detection Time (ms)",
            "Detection Processing Time",
            lower_is_better=True,
        )
    )

# 20. CLASSIFICATION VIEW

elif view in {
    "Fruit Type",
    "Fruit Ripeness",
}:
    is_fruit = (
        view == "Fruit Type"
    )

    scores = (
        fruit_scores
        if is_fruit
        else ripeness_scores
    )

    metric_key = (
        "fruit_metrics"
        if is_fruit
        else "ripeness_metrics"
    )

    stage_name = (
        "11. Fruit Type Classification"
        if is_fruit
        else "12. Ripeness Classification"
    )

    st.header(
        f"{view} Classification"
    )

    show_stage_winner(
        stage_name,
        scores,
        scorecard,
    )

    show_table(
        scores,
        {
            "Accuracy": "{:.4f}",
            "Macro Precision": "{:.4f}",
            "Macro Recall": "{:.4f}",
            "Macro F1": "{:.4f}",
            "Training Time (s)": "{:.3f}",
            "Prediction Time (ms/ROI)": "{:.4f}",
        },
    )

    chart_left, chart_right = st.columns(
        2
    )

    with chart_left:
        show_chart(
            grouped_bar(
                scores,
                [
                    "Accuracy",
                    "Macro Precision",
                    "Macro Recall",
                    "Macro F1",
                ],
                f"{view} Classification Metrics",
                zero_to_one=True,
            )
        )

    with chart_right:
        show_chart(
            classification_radar(
                scores,
                f"{view} Classification Radar",
            )
        )

    st.subheader(
        "Per-Class Precision and Recall Line Graph"
    )

    precision_recall_columns = st.columns(
        2
    )

    for column, benchmark in zip(
        precision_recall_columns,
        [
            "Benchmark A",
            "Benchmark B",
        ],
    ):
        with column:
            metrics = data[
                benchmark
            ].get(
                metric_key
            )

            if metrics:
                show_chart(
                    precision_recall_by_class_chart(
                        metrics,
                        f"{benchmark} {view} — Precision and Recall by Class",
                        benchmark,
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

            else:
                st.info(
                    f"No {benchmark} Precision/Recall data available."
                )

    st.subheader(
        "Class Support"
    )

    support_columns = st.columns(
        2
    )

    for column, benchmark in zip(
        support_columns,
        [
            "Benchmark A",
            "Benchmark B",
        ],
    ):
        with column:
            metrics = data[
                benchmark
            ].get(
                metric_key
            )

            if metrics:
                show_chart(
                    class_support_bar(
                        metrics,
                        f"{benchmark} {view} Test Support",
                    )
                )

            else:
                st.info(
                    f"No {benchmark} support data available."
                )

    st.subheader(
        "Confusion Matrices"
    )

    matrix_columns = st.columns(
        2
    )

    for column, benchmark in zip(
        matrix_columns,
        [
            "Benchmark A",
            "Benchmark B",
        ],
    ):
        with column:
            metrics = data[
                benchmark
            ].get(
                metric_key
            )

            if metrics:
                show_chart(
                    confusion_figure(
                        metrics,
                        f"{benchmark} {view}",
                    )
                )

            else:
                st.info(
                    f"No {benchmark} confusion matrix available."
                )

# 21. METRIC GUIDE VIEW

else:
    st.header(
        "Performance Metric Guide"
    )

    st.subheader(
        "Benchmark Algorithm Configuration"
    )

    show_table(
        BENCHMARK_ALGORITHM_TABLE
    )

    st.subheader(
        "Stage Performance Metrics"
    )

    show_table(
        STAGE_METRIC_GUIDE
    )

    st.info(
        "A stage is only compared when a corresponding saved experimental result is available. "
        "The dashboard does not create artificial scores for stages that were not separately evaluated."
    )
