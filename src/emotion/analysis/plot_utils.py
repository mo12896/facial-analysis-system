import matplotlib.pyplot as plt
import numpy as np


def plot_time_series(X: np.ndarray, labels: list, upper_limit: int = 1) -> None:
    # to set the plot size
    plt.figure(figsize=(16, 8), dpi=150)

    for i in range(len(labels)):
        plt.plot(X[i, :], label=labels[i])

    # adding Label to the x-axis
    plt.ylim(0, upper_limit)
    plt.xlabel("Frames")
    plt.legend()
    plt.show()
