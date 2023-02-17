from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

IDENTITY_DIR = Path("/home/moritz/Workspace/masterthesis/data/identities")


def plot_gaze_network_graph(df: pd.DataFrame, node_size: int = 2000):
    # Create an empty graph
    G = nx.DiGraph()

    # Add nodes from ClassID column
    G.add_nodes_from(df["ClassID"].unique())

    # Initialize dictionaries to keep track of the number of inbounding and outbounding edges for each node
    inbound_edges = {}
    outbound_edges = {}

    # Iterate over each row and add edges for the GazeDetections
    for i, row in df.iterrows():
        # Convert from string to list of strings
        gaze_detections = eval(row["GazeDetections"])
        if len(gaze_detections) > 0:
            for detection in gaze_detections:
                # Increment the weight of the edge if it already exists
                if G.has_edge(row["ClassID"], detection):
                    G[row["ClassID"]][detection]["weight"] += 1
                else:
                    G.add_edge(row["ClassID"], detection, weight=1)

                # Increment the count of outbound edges for the current node
                if row["ClassID"] in outbound_edges:
                    outbound_edges[row["ClassID"]] += 1
                else:
                    outbound_edges[row["ClassID"]] = 1

                # Increment the count of inbound edges for the target node
                if detection in inbound_edges:
                    inbound_edges[detection] += 1
                else:
                    inbound_edges[detection] = 1

    # Invert the weights of the edges
    for u, v, attr in G.edges(data=True):
        attr["weight"] = 1 / attr["weight"]

    # Draw the graph
    pos = nx.kamada_kawai_layout(G)

    _, ax = plt.subplots()

    # Draw the inbounding and outbounding edges with opposing curves
    for edge in G.edges:
        u, v = edge
        weight = G[u][v]["weight"]
        if inbound_edges[v] > 1:
            # Inbounding edge from multiple nodes
            edge_color = "black"
            edge_label = str(weight)
            connectionstyle = "arc3, rad = 0.2"

            # Draw the curved edge
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=(1 / weight) * 0.05,
                edge_color=edge_color,
                connectionstyle=connectionstyle,
                arrowsize=25,
                node_size=node_size,
                ax=ax,
            )

            # Add the edge label at the position of the curved edge
            edge_label = str(int(1 / weight))
            center_x = (pos[u][0] + pos[v][0]) / 2
            center_y = (pos[u][1] + pos[v][1]) / 2
            ha = "left" if pos[v][0] > pos[u][0] else "right"
            va = "center"
            offset = 0.05
            if ha == "right":
                center_x += offset
            else:
                center_x -= offset
            text = ax.annotate(
                edge_label,
                xy=(center_x, center_y),
                color="black",
                ha=ha,
                va=va,
            )
            text.set_path_effects(
                [path_effects.withStroke(linewidth=1, foreground="white")]
            )

    # Draw the nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        cmap=plt.cm.get_cmap("coolwarm"),
        node_size=node_size,
        label={node: node for node in G.nodes()},
        ax=ax,
    )

    # Draw the edge labels
    nx.draw_networkx_labels(
        G, pos, font_size=10, labels={node: node for node in G.nodes()}, ax=ax
    )

    # Show the plot
    plt.show()


if __name__ == "__main__":

    df = pd.read_csv(IDENTITY_DIR / "identities.csv")

    df["GazeCount"] = df["GazeDetections"].apply(lambda x: len(eval(x)))

    # df = preprocess_data(df)

    plot_gaze_network_graph(df)
