"""
Import the sim and plotting as a pip module and show some basic usage.
"""


"""
works with import:


pip install git+https://github.com/purgz/GTfyp.git

Below is an example script for the prisoners dilemma.

"""
import numpy as np

import simulation

import replicator

import pandas as pd

from simulation.games import Games


def main():
    print("Sucessful import")

    # Simple 3x3 RPS game
    matrix = Games.PRISONERS_DILEMMA

    pop_size = 200
    iterations = 5000
    w = 0.4

    print("Running Moran simulation...")

    delta_H, delta_rps, traj1, _ = simulation.moran_batch_sim(
        pop_size=pop_size,
        iterations=200 * 50,
        w=w,
        initial_dist=np.array([0.9,0.1]),
        simulations=1,
        matrix=matrix,
        traj=True,
        initial_rand=False,
    )

    delta_H, delta_rps, traj2, _ = simulation.moran_batch_sim(
        pop_size=20000,
        iterations=20000 * 50,
        w=w,
        initial_dist=np.array([0.9,0.1]),
        simulations=1,
        matrix=matrix,
        traj=True,
        initial_rand=False,
    )

    delta_H, delta_rps, traj3, _ = simulation.moran_batch_sim(
        pop_size=100000,
        iterations=100000 * 50,
        w=w,
        initial_dist=np.array([0.9,0.1]),
        simulations=1,
        matrix=matrix,
        traj=True,
        initial_rand=False,
    )

    df = pd.DataFrame({"C": traj1[0], "D": traj1[1]})
    df2 = pd.DataFrame({"C": traj2[0], "D": traj2[1]})
    df3 = pd.DataFrame({"C": traj3[0], "D": traj3[1]})

    simulation.game_2d_plot([df["D"], df2["D"], df3["D"]], N=[200,20000,100000], norm=[True,True,True], labels=["N=200","N=20000","N=100000"])

    """simulation.game_2d_plot([df["D"]],  N=200,    norm=[True], labels=["N=200"])
    simulation.game_2d_plot([df2["D"]], N=20000,  norm=[True], labels=["N=20000"])
    simulation.game_2d_plot([df3["D"]], N=100000, norm=[True], labels=["N=100000"])"""


if __name__ == "__main__":
    main()