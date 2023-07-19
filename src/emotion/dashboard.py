import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.emotion.analysis.data_preprocessing import (
    DataPreprocessor,
    LinearInterpolator,
    MinusOneToOneNormalizer,
    RollingAverageSmoother,
    ZeroToOneNormalizer,
)
from src.emotion.analysis.feature_generator import VelocityGenerator
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
from src.emotion.utils.constants import DATA_DIR_OUTPUT

# import sys


# grandparent_folder = os.path.abspath(
#     os.path.join(
#         os.path.dirname(os.path.abspath(__file__)),
#         os.pardir,
#         os.pardir,
#     )
# )
# sys.path.append(grandparent_folder)


def run_app(filename: str, emo_window: int = 150):
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
    st.title("Analytics Dashboard for the PERMA Prediction from Visual Features")
    st.write("<hr>", unsafe_allow_html=True)

    # Read the data from the csv file
    df = pd.read_csv(
        DATA_DIR_OUTPUT / (filename + "/analysis_results/" + filename + ".csv")
    )

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
        VelocityGenerator(),
        RollingAverageSmoother(window_size=5, cols=["Velocity"]),
        ZeroToOneNormalizer(cols=["Velocity"]),
    ]

    motion_stat_preprocessing_pipeline = [
        LinearInterpolator(),
        VelocityGenerator(negatives=True),
        RollingAverageSmoother(window_size=5, cols=["Velocity"]),
        MinusOneToOneNormalizer(cols=["Velocity"]),
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
            preprocessed_data["emo_time"], filename, plot=False
        ),
        lambda: plot_max_emotion_distribution(
            preprocessed_data["emo_stat"], filename, plot=False
        ),
        lambda: plot_smoothed_motion_over_time(
            preprocessed_data["motion_time"], filename, plot=False
        ),
        lambda: plot_point_derivatives(
            preprocessed_data["motion_stat"], filename, 250, plot=False
        ),
        lambda: plot_gaze_statistics(df, filename, plot=False),
    ]

    # Define a list of explanation strings
    explanations = [
        "These plots show time series of the 6 Ekman Emotions + Neutral Emotion for the different team members.",
        "These plots show categorical distributions of the maximum emotions expressed by each team member.",
        "These plots show time series of the point derivatives and thus velocities of the team members face center point",
        "These plots show kernel density estimations of the face center point velocities for the different team members.",
        "This is plot show the inbound to outbound gaze directions between the different team members",
    ]

    # Create a list of columns
    cols = []

    fig_dir = DATA_DIR_OUTPUT / (filename + "/prediction_results/")
    fig_dir = str(fig_dir)  # glob needs a string path

    # Define the order in which you want to process the files
    fig_files_order = ["perma_radar_*.png", "perma_bar_*.png"]

    for fig_file_pattern in fig_files_order:
        # Use glob to get a list of files matching the pattern
        matching_files = glob.glob(os.path.join(fig_dir, fig_file_pattern))

        for fig_path in sorted(matching_files):
            # Read the image file and create a figure
            img = mpimg.imread(fig_path)
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.axis("off")  # Hide axis

            # Dynamically create a new column for each figure
            new_col = st.columns(1)[0]
            new_col.pyplot(fig)
            new_col.markdown("---")

            # Add the new column to the list
            cols.append(new_col)

    # Now that all .png figures are rendered, create the rest of the columns for your plots
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    col5, _ = st.columns(2)

    cols = [col1, col2, col3, col4, col5]

    # Your existing code
    for i, (col, explanation) in enumerate(zip(cols, explanations)):
        fig = fig_funcs[i]()
        col.pyplot(fig)
        col.markdown(
            f"<h4>Figure {i+1}: {fig._suptitle.get_text()}</h4>",
            unsafe_allow_html=True,
        )
        col.write(explanation)
        col.markdown("<br>", unsafe_allow_html=True)
        col.markdown("---")


if __name__ == "__main__":
    emo_window = int(os.getenv("EMO_WINDOW", 500))
    filename = os.getenv("FILE")
    # Rest of your code
    run_app(filename=str(filename), emo_window=emo_window)
