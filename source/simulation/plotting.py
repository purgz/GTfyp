import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from mpl_toolkits.mplot3d import axes3d, Axes3D
from itertools import combinations
import pandas as pd
import scienceplots

# python-ternary
import ternary

from matplotlib.animation import FuncAnimation

"""
SciencePlots library - ref in paper
"""
#plt.style.use(['science','no-latex'])
plt.style.use(["science"])
mpl.rcParams["text.usetex"] = True
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


# Plots pyramid edges
def plot_ax(ax):
    verts = [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, 0.28867513, 0.81649658],
    ]
    lines = combinations(verts, 2)
    for x in lines:
        line = np.transpose(np.array(x))
        ax.plot3D(line[0], line[1], line[2], c="0")


# tranform from "barycentric" composition space to cartesian coordinates
def get_cartesian_array_from_barycentric(b):
    verts = [
        [0, 0, 0],
        [1, 0, 0],
        [0.5, np.sqrt(3) / 2, 0],
        [0.5, 0.28867513, 0.81649658],
    ]

    # create transformation array vis https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    t = np.transpose(np.array(verts))
    t_array = np.array([t.dot(x) for x in b])  # apply transform to all points

    return t_array


def label_points(ax):  # create labels of each vertices of the simplex
    a = np.array([1, 0, 0, 0])  # Barycentric coordinates of vertices (A or c1)
    b = np.array([0, 1, 0, 0])  # Barycentric coordinates of vertices (B or c2)
    c = np.array([0, 0, 1, 0])  # Barycentric coordinates of vertices (C or c3)
    d = np.array([0, 0, 0, 1])  # Barycentric coordinates of vertices (D or c3)
    labels = ["R", "P", "S", "L"]
    cartesian_points = get_cartesian_array_from_barycentric([a, b, c, d])
    for point, label in zip(cartesian_points, labels):
        if "a" in label:
            ax.text(point[0], point[1] - 0.075, point[2], label, size=16)
        elif "b" in label:
            ax.text(point[0] + 0.02, point[1] - 0.02, point[2], label, size=16)
        else:
            ax.text(point[0], point[1], point[2], label, size=16)


# Pass in dataframe with columns R, P, S, L
# 3d ternary takes 4 dimensional game
def plot_3d_ternary(df, ax, colour="b"):

    bary_arr = df.values

    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, 0.28867513, 0.81649658]]
    )

    cartesian_points = get_cartesian_array_from_barycentric(bary_arr)
    # ax.scatter(cartesian_points[:,0],cartesian_points[:,1],cartesian_points[:,2],c=c)

    ax.plot(
        cartesian_points[:, 0],
        cartesian_points[:, 1],
        cartesian_points[:, 2],
        color=colour,
        linewidth=1.2,
        alpha=0.7,
    )


def add_edge_labels(ax, numEdgeLabels=10):  # Add ratio labels along each edge
    verts = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0], [0.5, 0.28867513, 0.81649658]]
    )

    # Define edges of the pyramid
    edges = [
        (verts[0], verts[1]),  # A-B
        (verts[2], verts[0]),  # C-A
        (verts[0], verts[3]),  # A-D
        (verts[1], verts[2]),  # B-C
        (verts[1], verts[3]),  # B-D
        (verts[2], verts[3]),  # C-D
    ]

    ticks = np.linspace(0, 1, numEdgeLabels)[1:-1]  # 0.0 to 1.0 in 5 steps
    tick_labels = [f"{t:.1f}" for t in ticks]

    for start, end in edges:

        for t, label in zip(ticks, tick_labels):
            # label ticks along edges
            pos = (1 - t) * start + t * end
            ax.text(
                pos[0],
                pos[1],
                pos[2] + 0.02,
                label,
                size=10,
                ha="center",
                color="gray",
                weight="bold",
            )


def add_grid_lines(ax, numEdgeLabels=10):  # Add ternary-style grid lines to ABC face
    a = np.array([0, 0, 0])
    b = np.array([1, 0, 0])
    c = np.array([0.5, np.sqrt(3) / 2, 0])

    ticks = np.linspace(0, 1, numEdgeLabels)[
        1:-1
    ]  # Exclude 0 and 1 to avoid drawing edges twice

    # At each of the ticks, draw lines to opposite edge as in ternary plots
    # Lines parallel to AB
    for t in ticks:
        p1 = (1 - t) * c + t * b
        p2 = (1 - t) * c + t * a
        line = np.array([p1, p2]).T
        ax.plot3D(
            line[0], line[1], line[2], color="lightgray", linewidth=0.8, alpha=0.8
        )

    # Lines parallel to BC
    for t in ticks:
        p1 = (1 - t) * a + t * c
        p2 = (1 - t) * a + t * b
        line = np.array([p1, p2]).T
        ax.plot3D(
            line[0], line[1], line[2], color="lightgray", linewidth=0.8, alpha=0.8
        )

    # Lines parallel to AC
    for t in ticks:
        p1 = (1 - t) * b + t * c
        p2 = (1 - t) * b + t * a
        line = np.array([p1, p2]).T
        ax.plot3D(
            line[0], line[1], line[2], color="lightgray", linewidth=0.8, alpha=0.8
        )


# Utility functions


# Method for plotting multile experiments results
def quaternary_plot(
    dfs, numPerRow=2, labels=["Local update", "Moran Process"], colors=["b", "g"]
):

    numPlots = len(dfs)

    # Add random colors to colors if too few are provided
    if len(colors) < numPlots:
        np.random.seed(42)  # For reproducibility
        colors += [
            np.random.rand(
                3,
            )
            for _ in range(numPlots - len(colors))
        ]

    while len(labels) < numPlots:
        labels.append("No label")

    fig = plt.figure()

    rows = numPlots // numPerRow + (numPlots % numPerRow)
    cols = numPerRow

    for i, df in enumerate(dfs):

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        plot_ax(ax)
        label_points(ax)
        plot_3d_ternary(df, ax, colour=colors[i])
        add_edge_labels(ax)
        add_grid_lines(ax)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(labels[i], pad=20)
        ax.plot([], [], color=colors[i], label="Trajectory")
        ax.legend(loc="upper right", fontsize=10)

        # Want to plot a single point in ax with given coords
        # ax.scatter(0.5, 0.28867513, 0.81649658, color=colors[i], s=50, label="Single Point")
        deterministic_point = get_cartesian_array_from_barycentric(
            np.array([[2 / 9, 2 / 9, 2 / 9, 3 / 9]])
        )
        ax.scatter(
            deterministic_point[0, 0],
            deterministic_point[0, 1],
            deterministic_point[0, 2],
            color=colors[i],
            s=50,
            label="Deterministic Point",
        )

    plt.show()



def matrix_to_latex(matrix):
    rows = []
    for row in matrix:
        res = " & ".join(f"{x}" for x in row)
        rows.append(res)

    result = " \\\\ ".join(rows)
    return r"$\begin{bmatrix} " + result + r" \end{bmatrix}$"


def point_cloud(dfs, matrix=None):

    fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    plot_ax(ax)
    label_points(ax)
    add_edge_labels(ax)
    add_grid_lines(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1, 1, 1])

    if matrix is not None:
        test = matrix_to_latex(matrix)

        ax.text2D(0.7,0.7, test, transform=ax.transAxes)
        
    colors = ["b", "g"]

    total_frames = 0

    all_trajectories = []

    for i, df in enumerate(dfs):

        frames = sorted(df["frame"].unique())
        total_frames = max(total_frames, len(frames))

        carts = get_cartesian_array_from_barycentric(
            df[["c1", "c2", "c3", "c4"]].to_numpy()
        )

        print(df[["c1", "c2", "c3", "c4"]].values)

        print(carts)

        df["x"], df["y"], df["z"] = carts[:, 0], carts[:, 1], carts[:, 2]

        first_frame = df[df["frame"] == frames[0]]
        scatter = ax.scatter(
            first_frame["x"].values,
            first_frame["y"].values,
            first_frame["z"].values,
            s=10,
            alpha=1,
            color=colors[i]
        )

        all_trajectories.append((df, frames, scatter))

    def update(frame):
        for df, frames, scatter in all_trajectories:
            if frame >= len(frames):
                continue
            sub = df[df["frame"] == frame]
            scatter._offsets3d = (sub["x"].values, sub["y"].values, sub["z"].values)

        return [s[2] for s in all_trajectories]

    ani = FuncAnimation(
        fig, update, frames=total_frames, interval=0.5, blit=False, repeat=True
    )

    # Todo - create a proper file writer instead of gif for better quality. - gives the balls transparency - alpha parameter for visbility with multiple overlaps.
    #ani.save("./results/animations/ani.gif")
    plt.show()


# 2d ternary plot
def ternary_plot(traj):

    fig, axes = plt.subplots(1, 1, figsize=(12, 6))

    tax = ternary.TernaryAxesSubplot(ax=axes, scale=1.0)
    tax.boundary()
    tax.gridlines(color="gray", multiple=0.1)
    tax.plot(traj.to_numpy(), linewidth=1, label="3x3 game trajectory")
    tax.right_corner_label("A", fontsize=12)
    tax.top_corner_label("B", fontsize=12)
    tax.left_corner_label("C", fontsize=12)
    tax.legend()

    plt.show()


# Normalisation here is for the stochastic simulations since they need to be divided by the population size
def game_2d_plot(
    dfs, norm, N, sameAxis=True, labels=["Local update", "Moran process"], t_eval=None
):

    fig = plt.figure()

    for i, df in enumerate(dfs):
        if norm[i]:
            time_norm = np.arange(len(df)) / N
            plt.plot(time_norm, df.values, label=labels[i])
        else:

            plt.plot(t_eval, df.values, label=labels[i])

    plt.legend()

    plt.show()


# Method for plotting higher strategy, e.g 4 strategy games trajectory on a 2d graph.
def high_dim_2d_plot(
    file_paths,
    Ns: list[int],
    labels=["Moran Process", "Local update"],
    norm=[True],
    t_eval=None,
    data_res=50,
):

    fig = plt.figure()

    for i in range(len(file_paths)):

        data = pd.read_csv(file_paths[i], comment="#")

        r = data.iloc[:, 0]
        p = data.iloc[:, 1]
        s = data.iloc[:, 2]
        a = data.iloc[:, 3]

        for j in range(len(data.columns)):
            if norm[i]:
                # Need to remultiply by data res to fix scaling
                time_norm = (np.arange(len(data)) * data_res) / Ns[i]
                plt.plot(time_norm, data.iloc[:, j])
            else:
                plt.plot(t_eval, data.iloc[:, j])

    plt.xlabel("T")
    plt.ylabel("R,P,S,A")
    plt.show()


# Plotting drift reversal.
def drift_plot_H(
    file_paths: list[str],
    labels=["Moran process", "Local update"],
    xlabel: str = "N",
    column=None,
):

    for i in range(len(file_paths)):

        data = pd.read_csv(file_paths[i], comment="#")

        x_vals = data.iloc[:, 0]

        num_processes = len(data.columns) - 1

        markers = ["o", "s", "v"]

        if column is not None:
            plt.plot(
                x_vals, data.iloc[:, column + 1], label=labels[i], marker=markers[i]
            )
        else:
            for i in range(num_processes):
                plt.plot(x_vals, data.iloc[:, i + 1], label=labels[i], marker=markers[i])

    test = r"$\begin{bmatrix} 0 & -s & 1 & 0.2 \\ 1 & 0 & -s & 0.2 \\ -s & 1 & 0 & 0.2 \\ 0.1 & 0.1 & 0.1 & 0 \end{bmatrix}$"

    # plt.text(0.7,0.7, test, transform=plt.gca().transAxes)

    # plt.annotate("Drift reversal", xy=(200, -2.2e-5), xytext=(175, 0.002), arrowprops=dict(arrowstyle="->", color="black"))

    plt.xlabel(xlabel)
    plt.ylabel(r"$\langle \Delta H \rangle N$")
    # plt.ylabel("delta H") # switch if latex not installed.
    plt.legend()
    plt.show()


def w_ensemble_plot(filePath: str, log=True, x_label: str = "w"):
    data = pd.read_csv(filePath, comment="#")

    ws = data.iloc[:, 0]

    plt.plot(ws, data.iloc[:, 1], marker="s")
    if log:
        plt.yscale("log", base=10)

        plt.yticks([100, 1000])

    plt.xlabel(x_label)
    plt.ylabel("Nc")
    plt.show()
