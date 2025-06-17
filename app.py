from source import simulation

import argparse

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from source import replicator

"""
Main entry point for using standalone simulation 


Hav cmd line interface to run simulations and plot results.
"""

pdArray = np.array([[3, 0],
           [5, 1]])


# Need this because of multiprocessing - idk why
if  __name__ == "__main__":

    
    deltaM = []
    deltaL = []



    """
        TODO - make this an actual loop 

        Make a data storage system for simulations since they take so long to run :)

        Can see how parameters effect the critical population size 

        Probably a csv file is satisfactory for this 

        would be - parameters , pop size , drift results
        can then create a plotting function to plot these results on demand.


        would like to create a single method that does the following:
        - derives the replicator equations and outputs fixed points
        - numerically integrates the replicator equations for plotting
        - runs simulation suite with given parameter ranges
        - displays 3d plot.

    """
    """mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=100,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=120,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=140,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=160,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=180,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=300,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=500,simulations=300,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=1000,simulations=1000,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 20000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)"""

    #mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(matrix=pdArray, popSize=100, simulations=1, initialDist=[0.5,0.5], iterations=100000, w=0.4, H=1)

    #RPS - large pop


    test = replicator.testNumericalIntegration()

    print(test.tail())


    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=5000,simulations=1,H=3, initialDist=[0.7,0.1, 0.1, 0.1], w=0.4, iterations = 1000000)


    df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})

    df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})
    

    """
    df_RPS_MO = pd.DataFrame({"C": mResults[0], "D": mResults[1]})

    df_RPS_LU = pd.DataFrame({"C": lResults[0], "D": lResults[1]})
    """


    plt.plot(deltaM, label="moran")
    plt.plot(deltaL, label="local")

    plt.legend()

    plt.show()


    print(df_RPS_LU.tail())
    print(df_RPS_MO.tail())

    simulation.quaternaryPlot([df_RPS_LU, df_RPS_MO, test], numPerRow=3, labels=["LU", "MO", "Numerical"], colors=["r","b","g"])