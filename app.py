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



def pdExample(popsize=200):

  # Example running prisoners dilemma example.
  N = popsize
  w = 0.9
  iterations = 7000
  initialDist = [0.9,0.1]

  # Standard prisoners dilemma payoff matrix
  pdArray = np.array([[3, 0],
        [5, 1]])

  # Code for running prisoners dilemma simulation and numerical trajectories - then plotting all of them
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(matrix=pdArray, popSize=N, simulations=1, initialDist=[0.9,0.1], iterations=iterations, w=w, H=1)

  # Collect results into dataframe
  df_PD_MO = pd.DataFrame({"C": mResults[0], "D": mResults[1]})
  df_PD_LU = pd.DataFrame({"C": lResults[0], "D": lResults[1]})

  # Get numerical trajectory for prisoners dilemma.
  test = replicator.pdNumerical(pdArray, w, initialDist)

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


"""
Testing benchmarks
without changes - 99s runtime
"""
def runPopulationEnsemble(populationSizes):

  # Run a large batch with different parameters
  # Would like to test for popsizes, W value, and different payoff matrix values - starting with popsize here
  # Perhaps a config could be nice to have one single method to test for all.

  # Add the rest of the simulation options as arguments
  deltaM = []
  deltaL = []

  for i in range(len(populationSizes)):
    print("population ", populationSizes[i])
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=populationSizes[i],simulations=100,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.4, iterations = 100000)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)

  
  plt.plot(deltaM, label="moran")
  plt.plot(deltaL, label="local")
  plt.legend()
  plt.show()





# Need this because of multiprocessing
if  __name__ == "__main__":

  #RPS - large pop
  print("Running main")

  parser = argparse.ArgumentParser()
  """
  CMD Arguments:
  Game presets:
  standard prisoners dilemma: -pd
  """
  parser.add_argument("-preset", 
                      choices=["pd","rps","arps"],
                      help="Use a preset game matrix; OPTIONS = pd, rps, arps")
  parser.add_argument("-N", type=int, help="Specify population size to be used in simulation")
  args = parser.parse_args()

  if args.preset:
    print("Preset ", args.preset, " has been selected")
  




  
  test = replicator.numericalTrajectory()
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=10000,simulations=100,H=3, initialDist=[0.7,0.2, 0.1, 0], w=0.3, iterations = 700000)
  df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})
  df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})
  print(df_RPS_LU.tail())
  print(df_RPS_MO.tail())
  
  simulation.quaternaryPlot([df_RPS_LU, df_RPS_MO, test], numPerRow=3, labels=["LU", "MO", "Numerical"], colors=["r","b","g"])
  #simulation.quaternaryPlot([df_RPS_MO], numPerRow=1, labels=["MO"], colors=["r"])
  


  if args.preset:
    if args.preset == "pd":
      # add check for args.N
      print("Running prisoners dilemma preset: [[3,0],[5,1]]")
      pdExample(popsize=int(args.N))

  #runPopulationEnsemble([50,100, 150, 200,250,300])

 # pdExample()
  #rpsExample()
    


  
