from source import simulation

import argparse

import pandas as pd
import numpy as np

"""
Main entry point for using standalone simulation 


Hav cmd line interface to run simulations and plot results.
"""

pdArray = np.array([[3, 0],
           [5, 1]])


# Need this because of multiprocessing - idk why
if  __name__ == "__main__":

    mResults, lResults = simulation.runSimulationPool(matrix=pdArray, popSize=100, simulations=1, initialDist=[0.5,0.5], iterations=100000, w=0.4)


    """ 
    df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})

    df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})
    """

    df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1]})

    df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1]})




    print(df_RPS_LU.tail())
    print(df_RPS_MO.tail())

    simulation.quaternaryPlot([df_RPS_LU, df_RPS_MO])