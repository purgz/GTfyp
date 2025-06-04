from source import simulation

import argparse

import pandas as pd

"""
Main entry point for using standalone simulation 


Hav cmd line interface to run simulations and plot results.
"""

# Need this because of multiprocessing - idk why
if  __name__ == "__main__":

    mResults, lResults = simulation.runSimulationPool()


    df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})

    df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})


    simulation.quaternaryPlot([df_RPS_MO, df_RPS_LU])