from source import simulation
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from source import replicator

"""
Few things to do :

make sure the numerical code is correctly adjusting to new matrices since its hard coded for PD
e.g the delta Pi needs to be calculated correctly.

nice graph for hawk dove [2,10],[0,-5] - matchees pd results

also need to generalize the adjusted dynamics since theyre also hard coded.
"""


def pdExample():

  # Example running prisoners dilemma example.
  N = 200
  w = 0.9
  iterations = 7000

  # Standard prisoners dilemma payoff matrix
  pdArray = np.array([[3, 0],
        [5, 1]])

  # Code for running prisoners dilemma simulation and numerical trajectories - then plotting all of them
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(matrix=pdArray, popSize=N, simulations=1, initialDist=[0.9,0.1], iterations=iterations, w=w, H=1)

  # Collect results into dataframe
  df_PD_MO = pd.DataFrame({"C": mResults[0], "D": mResults[1]})
  df_PD_LU = pd.DataFrame({"C": lResults[0], "D": lResults[1]})

  # Get numerical trajectory for prisoners dilemma.
  test = replicator.pdNumerical(pdArray, w)

  # Get numerical trajectory for adjusted dynamics - moran process
  adjusted = replicator.pdNumericalAdjusted()

  # Plot the 4 trajectories on the same graph.
  # This method also normalizes the numerical solutions timeframe.
  simulation.Game2dPlot([df_PD_LU.get("D"), df_PD_MO.get("D"), test.get("D"), adjusted.get("D")], N=N, labels=["LU", "MO", "NUMERICAL", "ADJUSTED"], norm=[True, True, False, False])


def rpsExample():

  N = 5000
  w = 0.7
  iterations = 100000
  
  rpsArray = np.array([[0, -1 , 1], [1, 0, -1], [-1, 1, 0]])

  mResults, lResults, dm, dl = simulation.runSimulationPool(matrix=rpsArray, popSize=N, simulations=1, initialDist=[0.5,0.25,0.25], iterations=iterations,w=w, H=1)

  df_RPS_MO = pd.DataFrame({"R": mResults[0], "P": mResults[1], "S": mResults[2]})
  
  df_RPS_LO = pd.DataFrame({"R": lResults[0], "P": lResults[1], "S": lResults[2]})

  simulation.ternaryPlot(df_RPS_MO)

# Need this because of multiprocessing
if  __name__ == "__main__":

  pdExample()
  rpsExample()
    
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


      - want to be able to give augRPS a payoff matrix - or just parameter values, and then be given back the replicators and numerical integration of it in a standalone method.

  """
  """mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=100,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=120,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=140,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=160,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=180,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=300,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=500,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=1000,simulations=50,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.3, iterations = 100000)
  deltaM.append(deltaMoran)
  deltaL.append(deltaLocal)"""


  #RPS - large pop


  test = replicator.numericalTrajectory()

  print(test.tail())


  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=1000,simulations=1,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.4, iterations = 100000)


  df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})

  df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})
  

  plt.plot(deltaM, label="moran")
  plt.plot(deltaL, label="local")

  plt.legend()

  plt.show()


  print(df_RPS_LU.tail())
  print(df_RPS_MO.tail())

  simulation.quaternaryPlot([df_RPS_LU, df_RPS_MO, test], numPerRow=3, labels=["LU", "MO", "Numerical"], colors=["r","b","g"])
  
