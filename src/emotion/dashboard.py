# import os
# import sys

import pandas as pd
import streamlit as st

# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    DerivativesGetter,
    LinearInterpolator,
    MinusOneToOneNormalizer,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)
from src.emotion.analysis.statistics.plot_emotions_over_time import (
    plot_smoothed_emotions_over_time,
)
from src.emotion.analysis.statistics.plot_emotions_statistics import (
    plot_max_emotion_distribution,
)
from src.emotion.analysis.statistics.plot_gaze_statistics import plot_gaze_statistics
from src.emotion.analysis.statistics.plot_motion_over_time import (
    plot_smoothed_motion_over_time,
)
from src.emotion.analysis.statistics.plot_motion_statistics import (
    plot_point_derivatives,
)
from src.emotion.utils.constants import IDENTITY_DIR


def run_app(emo_window: int = 150):
    # Set the title of the app and add a custom favicon
    st.set_page_config(
        page_title="Visual Feature Analytics Dashboard",
        page_icon=":bar_chart:",
        layout="wide",
    )

    # Use a custom theme and add some spacing
    custom_css = """
    <style>
        .stApp {
            background-color: #fff;
        }
        .stButton button {
            background-color: #268bd2;
            color: #fff;
            font-weight: bold;
        }
        .stTextArea textarea {
            background-color: #fff;
            color: #268bd2;
            font-weight: bold;
            border-color: #268bd2;
            border-radius: 0;
        }
    </style>
    """
    st.write(custom_css, unsafe_allow_html=True)
    st.write("<br>", unsafe_allow_html=True)

    # Add content to the app
    st.title("Analytics Dashboard for the Visual PERMA Features")
    st.write("<hr>", unsafe_allow_html=True)

    # Read the data from the csv file
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    # Define the data pre-processing pipelines
    emo_time_preprocessing_pipeline = [
        LinearInterpolator(),
        RollingAverageSmoother(
            window_size=emo_window,
            cols=[
                "Angry",
                "Disgust",
                "Happy",
                "Sad",
                "Surprise",
                "Fear",
                "Neutral",
            ],
        ),
    ]

    emo_stat_preprocessing_pipeline = [LinearInterpolator()]

    motion_time_preprocessing_pipeline = [
        LinearInterpolator(),
        DerivativesGetter(),
        RollingAverageSmoother(window_size=5, cols=["Derivatives"]),
        ZeroToOneNormalizer(cols=["Derivatives"]),
    ]

    motion_stat_preprocessing_pipeline = [
        LinearInterpolator(),
        DerivativesGetter(negatives=True),
        RollingAverageSmoother(window_size=5, cols=["Derivatives"]),
        MinusOneToOneNormalizer(cols=["Derivatives"]),
    ]

    preprocessing_pipelines = {
        "emo_stat": emo_stat_preprocessing_pipeline,
        "emo_time": emo_time_preprocessing_pipeline,
        "motion_time": motion_time_preprocessing_pipeline,
        "motion_stat": motion_stat_preprocessing_pipeline,
    }

    # Pre-process the data and store in a dictionary
    preprocessed_data = {}
    for name, pipeline in preprocessing_pipelines.items():
        preprocessor = DataPreprocessor(pipeline)
        preprocessed_data[name] = preprocessor.preprocess_data(df)

    # Define a list of functions that generate the figures
    fig_funcs = [
        lambda: plot_smoothed_emotions_over_time(
            preprocessed_data["emo_time"], plot=False
        ),
        lambda: plot_max_emotion_distribution(
            preprocessed_data["emo_stat"], plot=False
        ),
        lambda: plot_smoothed_motion_over_time(
            preprocessed_data["motion_time"], plot=False
        ),
        lambda: plot_point_derivatives(
            preprocessed_data["motion_stat"], 250, plot=False
        ),
        lambda: plot_gaze_statistics(df, plot=False),
    ]

    # Define a list of explanation strings
    explanations = [
        "These plots show time series of the 6 Ekman Emotions + Neutral Emotion for the different team members.",
        "These plots show categorical distributions of the maximum emotions expressed by each team member.",
        "These plots show time series of the point derivatives and thus velocities of the team members face center point",
        "These plots show kernel density estimations of the face center point velocities for the different team members.",
        "This is plot show the inbound to outbound gaze directions between the different team members",
    ]

    # Create a grid of columns to display the figures
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, _ = st.columns(2)

    cols = [col1, col2, col3, col4, col5]

    # Loop over the columns in the correct order and display the figures
    for i, (col, explanation) in enumerate(zip(cols, explanations)):
        fig = fig_funcs[i]()
        col.pyplot(fig)
        # col.write(f"Figure {i+1}: {fig.get_axes()[0].get_title()}")
        col.markdown(
            f"<h4>Figure {i+1}: {fig._suptitle.get_text()}</h4>",
            unsafe_allow_html=True,
        )
        col.write(explanation)
        col.markdown("<br>", unsafe_allow_html=True)
        col.markdown("---")


if __name__ == "__main__":
    run_app(emo_window=500)
