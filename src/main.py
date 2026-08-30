"""Interactive Hybrid fruit detection and classification application.

Run with:
    streamlit run src/main.py

The interface uses the trained Hybrid models only.
It does not retrain models or overwrite experiment results.
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# 2. HYBRID PIPELINE IMPORTS

from classification.fruit_type_a import predict_lightgbm
from classification.ripeness_a import predict_catboost
from feature_extraction.feature_extraction_b import extract_lbp
from feature_extraction.hsv_colour_moments import extract_hsv_colour_moments
from morphology.morphology_a import apply_morphological_opening
from preprocessing.preprocessing_a import preprocess_image
from segmentation.segmentation_a import segment_image
from shared.initial_image_standardization import letterbox_resize
from shared.roi_extraction import extract_fruit_rois
from shared.roi_standardization import standardize_roi, standardize_roi_mask


# 3. MODEL PATHS

HYBRID_ROOT = PROJECT_ROOT / "results" / "hybrid"

FRUIT_MODEL_PATH = (
    HYBRID_ROOT
    / "fruit_type_classification"
    / "hybrid_fruit_type_classification_model.joblib"
)

RIPENESS_MODEL_PATH = (
    HYBRID_ROOT
    / "ripeness_classification"
    / "hybrid_ripeness_classification_model.joblib"
)


# 4. HYBRID PIPELINE PARAMETERS

TARGET_SIZE = (512, 512)
MIN_CONTOUR_AREA = 10_000

RED_BGR = (31, 18, 193)
WHITE_BGR = (255, 255, 255)

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


# 5. MODEL LOADING

@st.cache_resource(show_spinner="Loading Hybrid classifiers…")
def load_models() -> tuple[Any, Any]:
    """Load saved Hybrid fruit-type and ripeness classifiers."""
    fruit_model = joblib.load(FRUIT_MODEL_PATH)
    ripeness_model = joblib.load(RIPENESS_MODEL_PATH)

    return fruit_model, ripeness_model


# 6. IMAGE HELPERS

def decode_upload(uploaded_file: Any) -> np.ndarray:
    """Decode a Streamlit uploaded image into BGR OpenCV format."""
    data = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        data,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            "The uploaded file is not a readable image."
        )

    return image


def hybrid_feature(
    roi: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Extract the final Hybrid feature vector: Uniform LBP + HSV Colour Moments."""
    standardized_roi, _, _ = standardize_roi(
        roi,
        target_size=TARGET_SIZE,
    )

    standardized_mask = standardize_roi_mask(
        mask,
        target_size=TARGET_SIZE,
    )

    lbp = extract_lbp(
        standardized_roi,
        standardized_mask,
        radius=3,
    )

    hsv = extract_hsv_colour_moments(
        roi,
        mask,
    )

    return np.hstack(
        (
            lbp,
            hsv,
        )
    ).astype(
        np.float32,
        copy=False,
    )


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
) -> None:
    """Draw a readable label above a detected fruit bounding box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.48
    thickness = 1

    (width, height), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness,
    )

    top = max(
        0,
        y - height - baseline - 8,
    )

    cv2.rectangle(
        image,
        (x, top),
        (
            x + width + 8,
            y,
        ),
        RED_BGR,
        thickness=cv2.FILLED,
    )

    cv2.putText(
        image,
        text,
        (
            x + 4,
            y - baseline - 3,
        ),
        font,
        scale,
        WHITE_BGR,
        thickness,
        cv2.LINE_AA,
    )


# 7. HYBRID PROCESSING

def process_image(
    image: np.ndarray,
    fruit_model: Any,
    ripeness_model: Any,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    dict[str, float],
]:
    """Run one uploaded image through the complete Hybrid pipeline."""
    started = perf_counter()

    # --------------------------------------------------------
    # Stage 1: Initial image standardisation
    # --------------------------------------------------------

    standardized = letterbox_resize(
        image,
        TARGET_SIZE,
    )

    # --------------------------------------------------------
    # Stage 2: Preprocessing
    # Median Filter 3 × 3
    # --------------------------------------------------------

    filtered, preprocessing_info = preprocess_image(
        standardized,
        kernel_size=3,
    )

    # --------------------------------------------------------
    # Stage 3: Segmentation
    # Global Thresholding 128 in the Hybrid implementation
    # --------------------------------------------------------

    mask, segmentation_info, _ = segment_image(
        filtered
    )

    # --------------------------------------------------------
    # Stage 4: Morphological Opening
    # Ellipse 3 × 3
    # --------------------------------------------------------

    opened_mask, morphology_info, _ = apply_morphological_opening(
        mask,
        kernel_size=3,
    )

    # --------------------------------------------------------
    # Stage 5: External contour detection + ROI extraction
    # --------------------------------------------------------

    rois, _, detection_time = extract_fruit_rois(
        standardized,
        opened_mask,
        min_contour_area=MIN_CONTOUR_AREA,
    )

    if not rois:
        return (
            standardized,
            pd.DataFrame(),
            {
                "preprocessing_seconds": preprocessing_info[
                    "processing_time_seconds"
                ],
                "segmentation_seconds": segmentation_info[
                    "processing_time_seconds"
                ],
                "morphology_seconds": morphology_info[
                    "processing_time_seconds"
                ],
                "detection_seconds": detection_time,
                "classification_seconds": 0.0,
                "total_seconds": perf_counter() - started,
            },
        )

    # --------------------------------------------------------
    # Stage 6: Hybrid feature extraction
    # Uniform LBP + HSV Colour Moments
    # --------------------------------------------------------

    features = np.vstack(
        [
            hybrid_feature(
                item["roi"],
                item["roi_mask"],
            )
            for item in rois
        ]
    )

    # --------------------------------------------------------
    # Stage 7: Fruit Type classification
    # LightGBM
    # --------------------------------------------------------

    fruit_predictions, fruit_time = predict_lightgbm(
        fruit_model,
        features,
    )

    # --------------------------------------------------------
    # Stage 8: Ripeness classification
    # CatBoost
    # --------------------------------------------------------

    ripeness_predictions, ripeness_time = predict_catboost(
        ripeness_model,
        features,
    )

    # --------------------------------------------------------
    # Stage 9: Annotation and result table
    # --------------------------------------------------------

    annotated = standardized.copy()

    rows: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for index, (
        item,
        fruit,
        ripeness,
    ) in enumerate(
        zip(
            rois,
            fruit_predictions,
            ripeness_predictions,
        ),
        start=1,
    ):
        x, y, width, height = item[
            "bounding_box"
        ]

        label = (
            f"{fruit} — {ripeness}"
        )

        cv2.rectangle(
            annotated,
            (
                x,
                y,
            ),
            (
                x + width,
                y + height,
            ),
            RED_BGR,
            2,
        )

        draw_label(
            annotated,
            label,
            x,
            y,
        )

        rows.append(
            {
                "Detected fruit": f"Fruit {index}",
                "Fruit type": str(
                    fruit
                ),
                "Ripeness": str(
                    ripeness
                ),
                "Bounding box (x, y, w, h)": (
                    f"({x}, {y}, {width}, {height})"
                ),
                "ROI size (w × h)": (
                    f"{width} × {height}"
                ),
                "Object area (px²)": round(
                    float(
                        item["area"]
                    ),
                    1,
                ),
            }
        )

    # --------------------------------------------------------
    # Stage timing
    # --------------------------------------------------------

    timings = {
        "preprocessing_seconds": preprocessing_info[
            "processing_time_seconds"
        ],
        "segmentation_seconds": segmentation_info[
            "processing_time_seconds"
        ],
        "morphology_seconds": morphology_info[
            "processing_time_seconds"
        ],
        "detection_seconds": detection_time,
        "classification_seconds": (
            fruit_time
            + ripeness_time
        ),
        "total_seconds": (
            perf_counter()
            - started
        ),
    }

    return (
        annotated,
        pd.DataFrame(
            rows
        ),
        timings,
    )


# 8. DASHBOARD DATA HELPERS

def build_timing_table(
    timings: dict[str, float],
) -> pd.DataFrame:
    """Build a stage timing table for the processed image."""
    return pd.DataFrame(
        {
            "Stage": [
                "Preprocessing",
                "Segmentation",
                "Morphology",
                "Detection",
                "Classification",
                "Total",
            ],
            "Time (ms)": [
                timings[
                    "preprocessing_seconds"
                ]
                * 1000,
                timings[
                    "segmentation_seconds"
                ]
                * 1000,
                timings[
                    "morphology_seconds"
                ]
                * 1000,
                timings[
                    "detection_seconds"
                ]
                * 1000,
                timings[
                    "classification_seconds"
                ]
                * 1000,
                timings[
                    "total_seconds"
                ]
                * 1000,
            ],
        }
    )


def stage_timing_chart(
    timing_table: pd.DataFrame,
) -> plt.Figure:
    """Visualise processing time for each Hybrid stage."""
    stage_rows = timing_table.loc[
        timing_table["Stage"]
        != "Total"
    ].copy()

    figure, axis = plt.subplots(
        figsize=(
            8.2,
            4.5,
        )
    )

    figure.patch.set_facecolor(
        WHITE
    )

    bars = axis.bar(
        stage_rows[
            "Stage"
        ],
        stage_rows[
            "Time (ms)"
        ],
        color=BLUE,
        alpha=0.95,
    )

    axis.set_title(
        "Hybrid Pipeline Stage Processing Time",
        fontsize=13,
        fontweight="bold",
        color=BLACK,
        pad=12,
    )

    axis.set_ylabel(
        "Time (ms)"
    )

    axis.tick_params(
        axis="x",
        rotation=12,
    )

    axis.grid(
        axis="y",
        color=LIGHT_GRAY,
        alpha=0.7,
        linewidth=0.8,
    )

    axis.set_axisbelow(
        True
    )

    for spine in axis.spines.values():
        spine.set_color(
            LIGHT_GRAY
        )

    for bar, value in zip(
        bars,
        stage_rows[
            "Time (ms)"
        ],
    ):
        axis.annotate(
            f"{value:.2f}",
            (
                bar.get_x()
                + bar.get_width()
                / 2,
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
            color=BLACK,
        )

    figure.tight_layout()

    return figure


def fruit_summary_table(
    results: pd.DataFrame,
) -> pd.DataFrame:
    """Return only the user-facing classification result fields."""
    if results.empty:
        return pd.DataFrame()

    return results[
        [
            "Detected fruit",
            "Fruit type",
            "Ripeness",
        ]
    ].copy()


# 9. STREAMLIT PAGE CONFIGURATION

st.set_page_config(
    page_title="Hybrid Fruit Analysis",
    page_icon="🍎",
    layout="wide",
)


# 10. UI THEME

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

        [data-testid="stSidebar"]
        [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"]
        [data-testid="stCaptionContainer"] p {{
            color: #D1D5DB !important;
            opacity: 1 !important;
        }}

        .block-container {{
            max-width: 1500px;
            padding-top: 1.6rem;
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
            min-height: 120px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.07);
        }}

        [data-testid="stMetricLabel"] p {{
            color: {DARK_GRAY} !important;
            font-weight: 700 !important;
        }}

        [data-testid="stMetricValue"] {{
            color: {BLACK} !important;
            font-weight: 800 !important;
            font-size: clamp(
                1.3rem,
                2.3vw,
                2.1rem
            ) !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {WHITE} !important;
            border: 1px solid #C9CED6 !important;
            border-radius: 10px;
            padding: 0.9rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }}

        .hero-card {{
            background: {WHITE};
            border: 1px solid #C9CED6;
            border-left: 6px solid {RED};
            border-radius: 10px;
            padding: 1.1rem 1.2rem;
            margin: 0.4rem 0 1.2rem 0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
        }}

        .hero-card-title {{
            font-size: 1.05rem;
            font-weight: 800;
            color: {BLACK};
            margin-bottom: 0.25rem;
        }}

        .hero-card-text {{
            color: {DARK_GRAY};
            line-height: 1.5;
            font-size: 0.94rem;
        }}

        .pipeline-strip {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.6rem 0 1.2rem 0;
        }}

        .pipeline-step {{
            background: {WHITE};
            color: {BLACK};
            border: 1px solid #B8BEC7;
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 700;
            box-shadow: 0 2px 7px rgba(0, 0, 0, 0.04);
        }}

        .result-banner {{
            background: {WHITE};
            border: 1px solid #C9CED6;
            border-left: 6px solid {BLUE};
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
        }}

        .result-banner strong {{
            color: {BLUE_DARK};
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
            border-color: {RED} !important;
        }}

        [data-testid="stFileUploader"] {{
            background: {WHITE};
            border-radius: 10px;
        }}

        div.stButton > button,
        [data-testid="stDownloadButton"] > button {{
            background: {RED} !important;
            color: {WHITE} !important;
            border: 2px solid {RED} !important;
            border-radius: 8px;
            font-weight: 700;
        }}

        div.stButton > button:hover,
        [data-testid="stDownloadButton"] > button:hover {{
            background: {WHITE} !important;
            color: {RED} !important;
            border-color: {RED} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


# 11. HEADER

st.title(
    "Hybrid Fruit Detection & Classification"
)

st.caption(
    "Smart Fruit Image Analysis System | "
    "Detection • Counting • Fruit Type • Ripeness"
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-card-title">
            Hybrid Pipeline
        </div>
        <div class="hero-card-text">
            Upload a fruit image and the system will standardise the image,
            detect fruit regions, count detected fruit objects, classify the
            fruit type and predict its ripeness level using the trained Hybrid models.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="pipeline-strip">
        <span class="pipeline-step">512×512 Letterbox</span>
        <span class="pipeline-step">Median Filter</span>
        <span class="pipeline-step">Global Threshold</span>
        <span class="pipeline-step">Opening</span>
        <span class="pipeline-step">External Contours</span>
        <span class="pipeline-step">LBP + HSV</span>
        <span class="pipeline-step">LightGBM</span>
        <span class="pipeline-step">CatBoost</span>
    </div>
    """,
    unsafe_allow_html=True,
)


# 12. SIDEBAR

with st.sidebar:
    st.header(
        "Fruit Analysis"
    )

    st.caption(
        "Upload one image and run the saved Hybrid pipeline."
    )

    st.divider()

    st.markdown(
        "**Hybrid configuration**"
    )

    st.write(
        "Standardisation: 512 × 512"
    )

    st.write(
        "Detection: External Contours"
    )

    st.write(
        "Features: Uniform LBP + HSV"
    )

    st.write(
        "Fruit Type: LightGBM"
    )

    st.write(
        "Ripeness: CatBoost"
    )

    st.divider()

    st.caption(
        "The application performs inference only. "
        "It does not retrain the saved models."
    )


# 13. MODEL AVAILABILITY CHECK

missing_models = [
    path
    for path in (
        FRUIT_MODEL_PATH,
        RIPENESS_MODEL_PATH,
    )
    if not path.exists()
]

if missing_models:
    st.warning(
        "Hybrid models are not available yet. "
        "Run the Hybrid notebook until training completes, "
        "then return here."
    )

    st.code(
        "\n".join(
            str(
                path.relative_to(
                    PROJECT_ROOT
                )
            )
            for path in missing_models
        )
    )

    st.stop()


# 14. IMAGE UPLOAD

st.subheader(
    "Upload Fruit Image"
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "bmp",
        "webp",
    ],
    help=(
        "Supported formats: JPG, JPEG, PNG, BMP and WEBP."
    ),
)

if uploaded_file is None:
    st.info(
        "Upload a fruit image to start the Hybrid analysis."
    )

    st.stop()


# 15. RUN HYBRID PIPELINE

try:
    original = decode_upload(
        uploaded_file
    )

    fruit_model, ripeness_model = load_models()

    with st.spinner(
        "Processing image through the Hybrid pipeline…"
    ):
        annotated, results, timings = process_image(
            original,
            fruit_model,
            ripeness_model,
        )

except Exception as error:
    st.error(
        f"Unable to process the uploaded image: {error}"
    )

    st.stop()


# 16. MAIN RESULT OVERVIEW

st.divider()

st.header(
    "Analysis Result"
)

if results.empty:
    result_message = (
        "No valid fruit object was detected above the configured "
        f"minimum contour area of {MIN_CONTOUR_AREA:,} px²."
    )
else:
    result_message = (
        f"The Hybrid pipeline detected "
        f"<strong>{len(results)} fruit object(s)</strong> "
        "and completed fruit-type and ripeness classification."
    )

st.markdown(
    f"""
    <div class="result-banner">
        {result_message}
    </div>
    """,
    unsafe_allow_html=True,
)

image_column, summary_column = st.columns(
    (
        1.45,
        1,
    ),
    gap="large",
)

with image_column:
    with st.container(
        border=True
    ):
        st.subheader(
            "Annotated Fruit Image"
        )

        st.image(
            cv2.cvtColor(
                annotated,
                cv2.COLOR_BGR2RGB,
            ),
            use_container_width=True,
        )

with summary_column:
    st.subheader(
        "Processing Summary"
    )

    summary_cards = st.columns(
        2
    )

    summary_cards[0].metric(
        "Fruit Count",
        len(
            results
        ),
    )

    summary_cards[1].metric(
        "Processing Time",
        f"{timings['total_seconds']:.3f} s",
    )

    status_cards = st.columns(
        2
    )

    status_cards[0].metric(
        "Pipeline",
        "Hybrid",
    )

    status_cards[1].metric(
        "Classification",
        (
            "Completed"
            if not results.empty
            else "No fruit"
        ),
    )

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-card-title">
                Detection Summary
            </div>
            <div class="hero-card-text">
                <strong>Detected objects:</strong> {len(results)}<br>
                <strong>Minimum contour area:</strong> {MIN_CONTOUR_AREA:,} px²<br>
                <strong>Input standardisation:</strong> 512 × 512 letterbox<br>
                <strong>Feature representation:</strong> Uniform LBP + HSV Colour Moments
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# 17. CLASSIFICATION RESULTS

st.subheader(
    "Classification Results"
)

if results.empty:
    st.warning(
        "No fruit was detected above the configured minimum object area."
    )

else:
    classification_summary = fruit_summary_table(
        results
    )

    styled_summary = (
        classification_summary.style
        .hide(
            axis="index"
        )
    )

    st.markdown(
        '<div class="comparison-table">'
        + styled_summary.to_html()
        + "</div>",
        unsafe_allow_html=True,
    )

    if len(
        results
    ) == 1:
        fruit_row = results.iloc[
            0
        ]

        result_cards = st.columns(
            3
        )

        result_cards[0].metric(
            "Detected Fruit",
            fruit_row[
                "Detected fruit"
            ],
        )

        result_cards[1].metric(
            "Fruit Type",
            fruit_row[
                "Fruit type"
            ],
        )

        result_cards[2].metric(
            "Ripeness",
            fruit_row[
                "Ripeness"
            ],
        )


# 18. OBJECT PROPERTIES

with st.expander(
    "View Object Properties",
    expanded=False,
):
    if results.empty:
        st.info(
            "No detected fruit object properties are available."
        )

    else:
        styled_properties = (
            results.style
            .format(
                {
                    "Object area (px²)": "{:.1f}",
                }
            )
            .hide(
                axis="index"
            )
        )

        st.markdown(
            '<div class="comparison-table">'
            + styled_properties.to_html()
            + "</div>",
            unsafe_allow_html=True,
        )


# 19. STAGE TIMING

st.subheader(
    "Stage Processing Time"
)

timing_table = build_timing_table(
    timings
)

timing_chart_column, timing_table_column = st.columns(
    (
        1.35,
        1,
    ),
    gap="large",
)

with timing_chart_column:
    with st.container(
        border=True
    ):
        timing_figure = stage_timing_chart(
            timing_table
        )

        st.pyplot(
            timing_figure,
            clear_figure=True,
            use_container_width=True,
        )

with timing_table_column:
    styled_timing = (
        timing_table.style
        .format(
            {
                "Time (ms)": "{:.2f}",
            }
        )
        .hide(
            axis="index"
        )
    )

    st.markdown(
        '<div class="comparison-table">'
        + styled_timing.to_html()
        + "</div>",
        unsafe_allow_html=True,
    )


# 20. FINAL STATUS

st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-card-title">
            Analysis Completed
        </div>
        <div class="hero-card-text">
            The uploaded image has been processed through the saved Hybrid pipeline.
            The displayed predictions are inference results from the trained
            LightGBM fruit-type classifier and CatBoost ripeness classifier.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
