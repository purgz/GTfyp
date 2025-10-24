from source import simulation
from source import replicator

from source.simulation import Games

from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import scienceplots
import plotly.express as px

plt.style.use(['science','no-latex'])

"""
Few things to do :

make sure the numerical code is correctly adjusting to new matrices since its hard coded for PD
e.g the delta Pi needs to be calculated correctly.

nice graph for hawk dove [2,10],[0,-5] - matchees pd results

also need to generalize the adjusted dynamics since theyre also hard coded.
"""


"""
Working on generalizing so this will essentially work for ANY 2x2 symmetric game.
"""


def pdExample(popsize=1000, iterations = 10000, w=0.9, initialDist = [0.9,0.1]):

  # Example running prisoners dilemma example.
  N = popsize
  initialDist = [0.9,0.1]

  # Standard prisoners dilemma payoff matrix
  
  pdArray = Games.PRISONERS_DILEMMA

  # Code for running prisoners dilemma simulation and numerical trajectories - then plotting all of them
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(
    matrix=pdArray, 
    popSize=N, 
    simulations=100, 
    initialDist=[0.9,0.1], 
    iterations=iterations, 
    w=w, 
    H=1, 
    data_res=1)

  # Collect results into dataframe
  df_PD_MO = pd.DataFrame({"C": mResults[0], "D": mResults[1]})
  df_PD_LU = pd.DataFrame({"C": lResults[0], "D": lResults[1]})

  # Get numerical trajectory for prisoners dilemma.
  test = replicator.pdNumerical(pdArray, w, initialDist)

  # Get numerical trajectory for adjusted dynamics - moran process
  adjusted, t_eval = replicator.pdNumericalAdjusted()

  # Plot the 4 trajectories on the same graph.
  # This method also normalizes the numerical solutions timeframe.
  simulation.game2dPlot([df_PD_MO.get("D"), df_PD_LU.get("D"), adjusted.get("D"),  test.get("D")],
                        N=N, 
                        labels=["MO", "LO", "Adjusted", "Standard"],
                        norm=[True, True, False, False],
                        t_eval=t_eval)


def rpsExample(N=10000, iterations = 1000000):

  w = 0.5

  #rpsArray = np.array([[0, -1.2 , 1], [1, 0, -1.2], [-1.2, 1, 0]])

  rpsArray = np.array([[0, -1 , 1], [1, 0, -1], [-1, 1, 0]])
  
  mResults, lResults, dm, dl = simulation.runSimulationPool(matrix=rpsArray, popSize=N, simulations=100, initialDist=[0.5,0.25,0.25], iterations=iterations,w=w, H=1)

  df_RPS_MO = pd.DataFrame({"R": mResults[0], "P": mResults[1], "S": mResults[2]})
  
  df_RPS_LO = pd.DataFrame({"R": lResults[0], "P": lResults[1], "S": lResults[2]})

  #fig = px.line_ternary(df_RPS_MO, a="R", b="P", c="S", title="RPS Moran Process Trajectory", labels={"R":"Rock", "P":"Paper", "S":"Scissors"})
  #fig2 = px.line_ternary(df_RPS_LO, a="R", b="P", c="S", title="RPS LOCAL Process Trajectory", labels={"R":"Rock", "P":"Paper", "S":"Scissors"})

  #fig.show()
  #fig2.show()

  simulation.ternaryPlot(df_RPS_MO)


"""
Testing benchmarks
without changes - 99s runtime
"""
def runPopulationEnsemble(populationSizes, fileOutputPath="", plotDelta=False):



  start = time.time()


  # Add arguments here to customize the ensemble !

  # Run a large batch with different parameters
  # Would like to test for popsizes, W value, and different payoff matrix values - starting with popsize here
  # Perhaps a config could be nice to have one single method to test for all.

  # Add the rest of the simulation options as arguments
  deltaM = []
  deltaL = []

  for i in tqdm(range(len(populationSizes)), position=0, leave=True):
    #print("population ", populationSizes[i])
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=populationSizes[i],simulations=1000,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.4, iterations = 100000, data_res=500)
    deltaM.append(deltaMoran)
    deltaL.append(deltaLocal)


  end = time.time()
  
  print("Time taken to run population test ensemble")
  print(end - start)

  # Combine into a single file for csv saving.
  df_deltaResults = pd.DataFrame(np.column_stack((populationSizes,deltaM, deltaL)), columns=["popsizes","deltaMoran", "deltaLocal"])

  deltaH_Write(df_deltaResults, filePath=fileOutputPath
               , args = ["w: 0.4" , "simulations: 1", "iterations 100000"]
               , optionalComments="Testing rewrite with drift files")


  if plotDelta:

    plt.plot(df_deltaResults["popsizes"], df_deltaResults["deltaMoran"], marker="o",label="moran")
    plt.plot(df_deltaResults["popsizes"], df_deltaResults["deltaLocal"], marker="s", label="local")
    plt.xlabel("N")
    plt.ylabel("delta H")
    plt.legend()
    plt.show()


def searchCriticalPopsize(w=0.4):

  # Binary search for critical popsize where drift reversal occurs.
  # Hardcoded for initial deltaM being positive.

  # Searches for a sign change with tolerance of 1 in popsize.
  criticalN = None
  iteration = 0

  low = 440
  high = 455
  max_iterations = 20

  
  prevSign = None

  while low <= high and iteration < max_iterations:
    iteration += 1
    mid = (low + high) // 2
    print("Testing popsize: ", mid)
    mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=mid,simulations=3000,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=w, iterations = 100000)

    if deltaMoran > 0:
      sign = 1
    elif deltaMoran < 0:
      sign = -1
    else:
      sign = 0

    # Checks for a sign change
    if prevSign is not None and (sign * prevSign) < 0:
      # Sign change
     
      if high - low > 1:
        if sign > 0:
          low = mid
        else:
          high = mid
        continue      
      criticalN = mid
      break

    if sign >= 0:
      low = mid + 1
    else:
      high = mid - 1
    
    prevSign = sign
  
  if criticalN is not None:
    print("Critical popsize found: ", criticalN)
  else:
    print("Critical popsize not found in range.")

  return criticalN








# Find critical popsize for a range over W values - very long run time simulation so needs to write to file output.
def criticalPopsizeEnsemble():

  # I think this function will need to periodically write to file since it will take so long to run
  # Periodically write to file, allow for restart at a later time.

  # Example of what this function might look like, we can have a similar one for testing different parameter values in the matrix.
  ws = np.linspace(0.1, 0.9, 9)

  Ns = []

  for w in ws:
    criticalN = searchCriticalPopsize(w=w)
    Ns.append((w,criticalN))

  print("Critical Ns ", Ns)
  


def arpsExample(N = 500, iterations = 100000):
  moranResults, localResults, dMoran, dLocal = simulation.runSimulationPool(popSize=N,simulations=10, 
                                                                            iterations=iterations,
                                                                            H=3,
                                                                            initialDist=[0.5,0.2,0.2,0.1],
                                                                            w=0.4, data_res=50)
  
  
  df_RPS_MO = pd.DataFrame({"c1": moranResults[0], "c2": moranResults[1], "c3": moranResults[2], "c4": moranResults[3]})
  df_RPS_LU = pd.DataFrame({"c1": localResults[0], "c2": localResults[1], "c3": localResults[2], "c4": localResults[3]})


  trajectoryWrite(df_RPS_MO, "./results/moran" + str(N) + "_" + str(iterations) + ".csv", args = [N, iterations, 0.2], 
                  optionalComments= "Testing the file writing for trajectories.")

  trajectoryWrite(df_RPS_LU, "./results/local" + str(N) + "_" + str(iterations) + ".csv", args = [N, iterations, 0.2]),

  simulation.quaternaryPlot([df_RPS_MO, df_RPS_LU],labels=["Moran", "Local"])



def trajectoryWrite(df, filePath, args=[], optionalComments=None):

  # Construct and add comments
  with open(filePath, "w") as f:

    if optionalComments:
      f.write("# " + optionalComments + "\n")

    f.write("# Arguments used\n# args=")
    for arg in args:
      f.write(str(arg) + " ")
    f.write("\n")

  df.to_csv(filePath, mode = "a", index=False)


def deltaH_Write(df, filePath, args=[], optionalComments=None):

  with open(filePath, "w") as f:
    f.write("# Drift (delta H) plot results")
    if optionalComments:
      f.write("# " + optionalComments + "\n")

    f.write("# Arguments used\n# args=")
    for arg in args:
      f.write(str(arg) + " ")
    f.write("\n")

  df.to_csv(filePath, mode = "a", index=False)

  


# Need this because of multiprocessing
if  __name__ == "__main__":


  #RPS - large pop
  print("Running main")
  #pdExample()
  #rpsExample()
  
  #searchCriticalPopsize()
  #runPopulationEnsemble(range(100,700, 100), fileOutputPath="./results/tqdmdrifttest.csv", plotDelta=True)

  #simulation.driftPlotH("./results/drift.csv", labels=["Moran, Local"])

  
  test, t_eval = replicator.numericalTrajectory(interactionProcess="Moran")
  trajectoryWrite(test, "./results/moranNumerical.csv")


  #filePaths = ["./results/moran100000_15000000.csv", "./results/moranNumerical.csv"]
  filePaths = ["./results/moran100000_15000000.csv", "./results/moranNumerical.csv"]
  norms = [True, False]

  simulation.highDim2dplot(filePaths, [100000, None], norm=norms, t_eval=t_eval)
  

  """
  df_MO = pd.read_csv("./results/moran400_100000.csv")

  print("Attempting file read")
  simulation.quaternaryPlot([df_MO], labels=["Moran"])
  exit()
  """


  #test = replicator.numericalTrajectory()
  """
  # As pop size gets very large - closely tracks the analytic solution
  mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=1000,simulations=1000,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.2, iterations = 1000000, data_res=100)
  #mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=540,simulations=100,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=0.2, iterations = 100000)
  
  #mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=20000,simulations=100,H=3, initialDist=[0.5,0.25, 0.25, 0], w=0.5, iterations = 1000000)
  #mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(popSize=60000,simulations=1,H=3, initialDist=[0.7,0.1, 0.1, 0.1], w=0.6, iterations = 20000000)

  df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})
  df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})
 

  print("Delta Moran: ", deltaMoran)
  print("Delta Local: ", deltaLocal)
  
  #simulation.quaternaryPlot([df_RPS_LU, df_RPS_MO, test], numPerRow=3, labels=["LU", "MO", "Numerical"], colors=["r","b","g"])
  simulation.quaternaryPlot([df_RPS_MO, df_RPS_LU], numPerRow=2, labels=["MO", "LU"], colors=["r", "g"])
  """

  parser = argparse.ArgumentParser()
  """
  CMD Arguments:
  Game presets:
  standard prisoners dilemma: -pd
  """


  subParsers = parser.add_subparsers(dest="preset")

  # 2x2 game
  pd_parser = subParsers.add_parser("pd")
  pd_parser.add_argument("-N", type = int, default=1000)
  pd_parser.add_argument("-iterations", type=int, default=1000000)

  # 3x3 game
  rps_parser = subParsers.add_parser("rps")
  rps_parser.add_argument("-N", type=int, default=10000)
  rps_parser.add_argument("-iterations", type=int, default=1000000)

  arps_parser = subParsers.add_parser("arps")
  arps_parser.add_argument("-N", type=int, default=500)
  arps_parser.add_argument("-iterations", type=int, default=100000)


  args = parser.parse_args()


  if args.preset:
    print("Preset ", args.preset, " has been selected")
  
  if args.preset:
    if args.preset == "pd":
      # add check for args.N
      print("Running prisoners dilemma preset: [[3,0],[5,1]]")
      pdExample(popsize=args.N, iterations=args.iterations)  
    elif args.preset == "rps":
      print("Running rock paper scissors preset : [0,-1,1],[1,0,-1],[-1,1,0]")
      rpsExample(N = args.N, iterations=args.iterations)
    elif args.preset == "arps":
      print("Running augmented rps: " + str(Games.AUGMENTED_RPS))
      arpsExample(N = args.N, iterations=args.iterations)
      
    


  
