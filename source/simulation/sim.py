import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations
from multiprocessing import Pool




#If running this file directly uncomment following
#from plotting import quaternaryPlot
from .plotting import quaternaryPlot



# Here the loner > 0, therefore some central equivilibrium should be present - drift away from this would be towards pure RPS
basicRps = np.array([[0,   -1,   1,       0.2],
                    [1,    0,   -1,       0.2],
                    [-1,   1,   0,        0.2],
                    [0.1, 0.1, 0.1, 0]])



"""basicRps = np.array([[0,   -1,   1,       0],
                    [1,    0,   -1,       0],
                    [-1,   1,   0,        0],
                    [0, 0, 0, 0]])
"""
"""basicRps = np.array([[1,     2.35,    0,          0.1],
                    [0,      1,       2.35,       0.1],
                    [2.35,   0,       1,          0.1],
                    [1.1,    1.1,     1.1,        0]])
"""



def payoffAgainstPop(population,matrix, popSize):
    payoffs = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        payoffs[i] = sum((population[j]) * matrix[i][j] for j in range(matrix.shape[0]))
    return payoffs / (popSize-1)



"""
Paper coevolutionary dynamics in large but finite populations

'An individidual of type j is chosen for repoduction with probabiiltiy i_j * Pi_j / (N * phi)
where phi = average payoff.
"""

def moranSelection(payoffs, avg, population, popSize, numStrategies=4):
    probs = np.zeros(numStrategies)
    for i in range(numStrategies):

        probs[i] = (population[i] * payoffs[i]) / (popSize * avg)

    return probs


def localUpdate(matrix, popSize, initialDist = [0.1, 0.1, 0.1, 0.7], iterations = 100000, w=0.4):
    
    population = np.random.multinomial(popSize, initialDist)

    numStrategies = matrix.shape[0]
    
    results = np.zeros((numStrategies, iterations))

    individuals = np.repeat(np.arange(numStrategies), population)

    for i in range(iterations):
        

        ind1, ind2 = np.random.choice(popSize, size=2, replace=False)
     
        p1, p2 = individuals[ind1], individuals[ind2]

        payoffs = payoffAgainstPop(population, matrix, popSize)
    
        #deltaPi = np.max(payoffs) - np.min(payoffs)

        deltaPi = np.max(matrix) - np.min(matrix)
        
        p = 1/2 + (w/2) * ((payoffs[p2] - payoffs[p1]) / deltaPi)

        # With this probability switch p1 to p2
        if (np.random.rand() < p):
            population[p1] -= 1
            population[p2] += 1
            # Update the individuals when an update occurs.
            individuals[ind1] = p2
  
        for j in range(numStrategies):
            results[j][i] = population[j] / popSize

    # Return normalized RPSL distribution
    return results
          


def moranSimulation(matrix, popSize, initialDist = [0.1, 0.1, 0.1, 0.7], iterations = 100000, w=0.4):
    # Population represented just as their frequency of strategies for efficiency,
    # I think individual agents in simple dynamics unneccessary overhead
    population = np.random.multinomial(popSize, initialDist)

    numStrategies = matrix.shape[0]
    
    results = np.zeros((numStrategies, iterations))

    for i in range(iterations):
        # Death: uniform random
        killed = random.choices(range(numStrategies), weights=population)[0]
        # Birth: fitness-proportional
        # P = reproductive fitness in moran process 1 - w + w * Pi
        p = 1 - w + w * payoffAgainstPop(population, matrix, popSize)
        avg = np.sum(p * population) / popSize
        probs = moranSelection(p, avg, population, popSize, matrix.shape[0])

        chosen = random.choices(range(numStrategies), weights=probs)[0]
    
        population[chosen] += 1
        population[killed] -= 1

        for j in range(numStrategies):
            results[j][i] = population[j] / popSize

    # Return normalized RPSL distribution
    return results




"""
popSize = 100
simulations = 1
"""



def singleSim(matrix, popSize, initialDist, iterations, w, H):
    # Add other interaction processs here
    moranResult = moranSimulation(matrix, popSize, initialDist, iterations,w)
    localResult = localUpdate(matrix, popSize, initialDist, iterations,w)

    delta_L_moran = np.mean(np.diff(moranResult[H]))
    delta_L_local = np.mean(np.diff(localResult[H]))

    # Lyapunov function? doesnt seem to work  
    #delta_L_moran = np.mean(np.diff(-np.prod(moranResult, axis=0)))
    #delta_L_local = np.mean(np.diff(-np.prod(localResult, axis=0)))

    return moranResult, localResult, delta_L_moran, delta_L_local

# Method for api to call
def runSimulationPool(matrix=basicRps, popSize=100, simulations=100, initialDist=[0.1, 0.1, 0.1, 0.7], iterations=100000, w=0.4, H=3):
    # Runs multiprocessing simulations for moran and local update process

    # H parameter decides which strategy will be focussed for the drift analysis

    deltaMoran = []
    deltaLocal = []
    mResults = []
    lResults = []

    numStrategies = matrix.shape[0]
    
    args = [(matrix, popSize, initialDist, iterations, w, H) for _ in range(simulations)]

    print("Running simulation pool")
    print("Strategies: ", numStrategies, " Population size: ", popSize, " Simulations: ", simulations, " Iterations: ", iterations, "w: ", w, " Initial distribution: ", initialDist)

    with Pool() as pool:
        # Starmap to allow passing of arguments to each simulation
        results = pool.starmap(singleSim, args)

    for i, (moranResult, localResult, delta_L_moran, delta_L_local) in enumerate(results):
        deltaMoran.append(delta_L_moran)
        deltaLocal.append(delta_L_local)

        if i == 0:
            mResults = np.array(moranResult)
            lResults = np.array(localResult)
        else:
            mResults += np.array(moranResult)
            lResults += np.array(localResult)

    print(np.mean(deltaMoran), " Moran drift")
    print(np.mean(deltaLocal), " local drift")

    mResults /= simulations
    lResults /= simulations

    return mResults, lResults, np.mean(deltaMoran), np.mean(deltaLocal)



"""
# Running the file directly no longer works due to changed packacge structure, run via app.py
# Multiprocessing magic
if __name__ == '__main__':
    with Pool() as pool:
        results = pool.map(singleSim, range(simulations))

    for i, (moranResult, localResult, delta_L_moran, delta_L_local) in enumerate(results):
        deltaMoran.append(delta_L_moran)
        deltaLocal.append(delta_L_local)

        if i == 0:
            mResults = np.array(moranResult)
            lResults = np.array(localResult)
        else:
            mResults += np.array(moranResult)
            lResults += np.array(localResult)

            
    print(np.mean(deltaMoran), " Moran drift")
    print(np.mean(deltaLocal), " local drift")

    mResults /= simulations
    lResults /= simulations


    df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})

    df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})


    # Plot multiple results
    quaternaryPlot([df_RPS_LU, df_RPS_MO, df_RPS_MO, df_RPS_MO], labels=["Local update", "Moran process"], numPerRow=3, colors=['g', 'b'])

"""