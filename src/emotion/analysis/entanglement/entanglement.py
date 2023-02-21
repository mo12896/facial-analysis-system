from enum import Enum

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist, squareform
from tslearn.metrics import cdist_dtw


class SimilarityMetric(Enum):
    euclidean = "euclidean"
    correlation = "correlation"
    dtw = "dtw"


def plot_entanglement_graph(
    X: np.ndarray,
    metric: SimilarityMetric,
    person_ids: list,
    plot_category: str,
    save_fig: bool = False,
) -> None:

    plt.title(f"{plot_category.capitalize()} Entanglement")

    # Compute pairwise distances between all nodes using Euclidean distance

    if metric.value == "dtw":
        dist_matrix = cdist_dtw(X)
    else:
        distances = pdist(X, metric=metric.value)

        # Convert condensed distance matrix to square distance matrix
        dist_matrix = squareform(distances)

    # Create a graph from the distance matrix
    G = nx.from_numpy_array(dist_matrix)

    # Compute the average value of each row in X
    avg_values = np.mean(X, axis=1)

    # Create a dictionary that maps node index to normalized average value
    node_sizes = {i: 5000 * avg_values[i] for i in range(len(person_ids))}

    # Compute the average distance of each node to all other nodes
    avg_distances = np.mean(dist_matrix, axis=1)

    # Normalize the average distances to the range [0, 1]
    norm_avg_distances = avg_distances / np.max(avg_distances)

    # Draw the graph with node positions using the kamada_kawai_layout (docs)
    pos = nx.drawing.layout.kamada_kawai_layout(G, dim=2)
    # nx.draw(G, pos)
    # nx.draw(G, pos, node_size=list(node_sizes.values()))
    nx.draw(
        G,
        pos,
        node_size=list(node_sizes.values()),
        node_color=norm_avg_distances,
        cmap=plt.cm.get_cmap("coolwarm"),
    )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.get_cmap("coolwarm"), norm=plt.Normalize(vmin=0, vmax=1)
    )
    sm._A = []
    plt.colorbar(sm)

    # # Add labels to the nodes
    labels = {i: class_id for i, class_id in enumerate(person_ids)}
    nx.draw_networkx_labels(G, pos, labels)

    # Show the plot
    if save_fig:
        plt.savefig(f"{plot_category}_entanglement_with_{metric}.png")

    plt.show()
