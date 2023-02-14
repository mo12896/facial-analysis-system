from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_smoothed_motion_over_time(w_size: int = 5):
    # read the csv file into a pandas DataFrame
    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    # group the data by ClassID and Frame
    grouped = df.groupby("ClassID")

    fig = plt.figure(figsize=(10, 15), tight_layout=True)

    for i, (person_id, group) in enumerate(grouped):

        # calculate the center points in x and y position for each group
        x_center = (group["XMax"] + group["XMin"]) * 0.5
        y_center = (group["YMax"] + group["YMin"]) * 0.5

        # calculate the ratio of change in x to change in y
        dx, dy = x_center.diff(), y_center.diff()
        derivatives = abs(dx / dy)

        # Compute the smoothed derivative signal using a moving average
        window_size = w_size
        smoothed_derivatives = derivatives.rolling(window_size).mean()

        ax = fig.add_subplot(2, 2, i + 1)
        # plot the smoothed derivatives over time
        plt.plot(smoothed_derivatives.index.values, smoothed_derivatives.values)
        ax.set_ylim(0, 200)
        ax.set_title(f"Center derivatives w.r.t. time for {person_id}")
        ax.set_xlabel("Frames")
        ax.set_ylabel("Derivatives")

    plt.show()
    fig.savefig(IDENTITY_DIR / "point_derivatives_over_time.png")


if __name__ == "__main__":
    # If window size is 1, no smoothing is applied
    plot_smoothed_motion_over_time(5)
