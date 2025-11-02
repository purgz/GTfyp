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

from multiprocessing import Pool


plt.style.use(['science','no-latex'])

"""
Note - begun using my linux desktop instead and performace roughly 2x when using the multiprocess simulation

- Windows sucks
- Run sims on linux
"""


def pdExample(popsize : int = 1000, iterations : int = 10000, w : float =0.9, initialDist : list[float] = [0.9,0.1]) -> None:

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


def rpsExample(N : int =10000, iterations : int = 1000000) -> None:

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
For popualtion drift test and critical popsize search, data res is very high so we dont waste cpu on memory accesses - 10 - 20% speedup
As the trajectory is not needed for the final results - only care about accurate delta H values.
Delta H is calculated before the dimension reduction
"""
def runPopulationEnsemble(populationSizes : list[int] , fileOutputPath : str ="", plotDelta : bool = True, simulations : int = 10000000) -> None:


  
  start = time.time()
  
  # Add the rest of the simulation options as arguments
  driftHs = []
  driftRpss = []

 
  # Simulation for each popsize.
  for i in tqdm(range(len(populationSizes)), position=0, leave=True):


    driftH, driftRps, _ = simulation.moran_batch_drift(populationSizes[i], 2, 0.45, simulations, basicRps, np.array([0.25,0.25,0.25,0.25]))
    driftHs.append(driftH)
    driftRpss.append(driftRps)
 
  end = time.time()
  
  print("Time taken to run population test ensemble")
  print(end - start)


  # Combine into a single file for csv saving.
  df_deltaResults = pd.DataFrame(np.column_stack((populationSizes,driftHs, driftRpss)), columns=["popsizes","deltaH", "deltaRps"])

  deltaH_Write(df_deltaResults, filePath=fileOutputPath
               , args = ["w: 0.45" , "simulations: 1000000", "iterations 2", "matrix=Standard rps, 0.2, 0.1"]
               , optionalComments="Large average delta H experiment with randomizes starting point in the RPS plane")


  if plotDelta:

    plt.plot(df_deltaResults["popsizes"], df_deltaResults["deltaH"], marker="o",label="H_4")
    plt.plot(df_deltaResults["popsizes"], df_deltaResults["deltaRps"], marker="s",label="H_RPS")
    plt.xlabel("N")
    plt.ylabel("delta H")
    plt.legend()
    plt.show()


def searchCriticalPopsize(w : float = 0.4) -> int:

  # Binary search for critical popsize where drift reversal occurs.
  # Hardcoded for initial deltaM being positive.

  # Searches for a sign change with tolerance of 1 in popsize.
  criticalN = None
  iteration = 0

  low = 100
  high = 3000
  max_iterations = 20
  mid = 0

  prevSign = None

  while low <= high and iteration < max_iterations:
    iteration += 1
    mid = (low + high) // 2
    print("Testing popsize: ", mid)
    """  
        mResults, lResults, deltaMoran, deltaLocal = simulation.runSimulationPool(
          popSize=mid,simulations=5000,H=3, initialDist=[0.25,0.25, 0.25, 0.25], w=w, iterations = 100000,
                                                                                  pool=pool, data_res=5000)
    """

    deltaMoran, _ , _= simulation.moran_batch_drift(mid, 2, w, 20000000, Games.AUGMENTED_RPS, np.array([0.25,0.25,0.25,0.25]))


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

  if criticalN is None:
    criticalN = mid  # best estimate
  if criticalN is not None:
    print("Critical popsize found: ", criticalN)
  else:
    print("Critical popsize not found in range.")
  return criticalN








# Find critical popsize for a range over W values - very long run time simulation so needs to write to file output.
def criticalPopsizeEnsemble() -> None:

  # I think this function will need to periodically write to file since it will take so long to run
  # Periodically write to file, allow for restart at a later time.

  fileOutputPath = "./results/criticalN_w.csv"

  # Example of what this function might look like, we can have a similar one for testing different parameter values in the matrix.
  ws = np.linspace(0.1, 0.5, 15)

  Ns = []

  for w in tqdm(ws, position=0, leave=True):
    criticalN = searchCriticalPopsize(w=w)
    Ns.append(criticalN)

  print("Critical Ns ", Ns)

  df = pd.DataFrame({"W": ws, "CriticalN": Ns})

  deltaH_Write(df, filePath=fileOutputPath
               , args = [f"W range {ws[0]}" , "4000", "iterations 100000", "matrix=Standard rps, 0.2, 0.1"]
               , optionalComments="Critical population size search for varied w.")





# Vary the values of alpha and beta and s(rps param) in the 4x4 game
"""
Two graphs needed;
  - delta H against different values for params
  - critical N for different params
  - can reuse search critical popsize method here.
"""
def alphaBetaEnsemble(w : int) -> None:

  defaultMatrix = np.copy(Games.AUGMENTED_RPS)

  # maybe can reuse the above method with a little refactoring.
  pass
  


def arpsExample(N : int  = 500, iterations : int = 100000) -> None:
  moranResults, localResults, dMoran, dLocal = simulation.runSimulationPool(popSize=N,simulations=10, 
                                                                            iterations=iterations,
                                                                            H=3,
                                                                            initialDist=[0.5,0.25, 0.25,0],
                                                                            w=0.2, data_res=50)
  
  
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

  # 4x4 game
  arps_parser = subParsers.add_parser("arps")
  arps_parser.add_argument("-N", type=int, default=500)
  arps_parser.add_argument("-iterations", type=int, default=100000)

  # Other options
  #experimentParser = parser.add_subparsers(dest="experiment")
  # Add args for critical W finding, and ensembles for population and W.


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



  # Below is testig code - remove at some point
  basicRps = np.array([[0,   -0.8,   1,       5],
                      [1,    0,   -0.8,       5],
                      [-0.8,   1,   0,        5],
                      [2, 2, 2, 0]])
  
  basicRps = Games.AUGMENTED_RPS
  """
  deltamoran, deltaRps, mResults = simulation.moran_batch_drift(20000, 1000000, 0.45, 1, basicRps, np.array([0.25,0.25,0.25,0.25]), traj=True)
  df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})
  
  print("DELTA RPS ", deltaRps)
  print("DELTA MORAN ", deltamoran)

  trajectoryWrite(df_RPS_MO, "./results/moranTest.csv")

  test, t_eval = replicator.numericalTrajectory(interactionProcess="Moran")
  trajectoryWrite(test, "./results/moranNumerical.csv")


  #filePaths = ["./results/moran100000_15000000.csv", "./results/moranNumerical.csv"]
  filePaths = ["./results/moranTest.csv", "./results/moranNumerical.csv"]
  norms = [True, False]

  simulation.quaternaryPlot([df_RPS_MO], numPerRow=1, labels=["Moran"])

  simulation.highDim2dplot(filePaths, [20000, None], norm=norms, t_eval=t_eval, data_res=1)
  """

  # maybe these functions should return file name - and autogerenrate one if one isnt given.
  #criticalPopsizeEnsemble()
  simulation.wEnsemblePlot("./results/criticalN_w.csv")


  deltaLocal, deltaRps, lResults = simulation.local_batch_drift(20000, 3000000, 0.45, 1, basicRps, traj=True, initialDist=np.array([0.5,0.2,0.2,0.1]))
  df_RPS_LO = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})
  trajectoryWrite(df_RPS_LO, "./results/localTest.csv")

  test, t_eval = replicator.numericalTrajectory(interactionProcess="Local", w=0.45)
  trajectoryWrite(test, "./results/localNumerical.csv")

  filePaths = ["./results/localTest.csv", "./results/localNumerical.csv"]
  norms = [True, False]

  simulation.highDim2dplot(filePaths, [20000, None], norm=norms, t_eval=t_eval, data_res=1)

  """
    
    runPopulationEnsemble(range(50,700,10), 
                          fileOutputPath="./results/population_ensemble.csv", 
                          plotDelta=True,
                          )"""

 
 
  #simulation.driftPlotH("./results/population_ensemble_w_0.4.csv", labels=["Moran", "Local"])




  
 
  

  


      
    


  
