import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations
from multiprocessing import Pool




from .plotting import quaternaryPlot




# Here the loner > 0, therefore some central equivilibrium should be present - drift away from this would be towards pure RPS
basicRps = np.array([[0,   -1,   1,       0.2],
                    [1,    0,   -1,       0.2],
                    [-1,   1,   0,        0.2],
                    [0.1, 0.1, 0.1, 0]])


"""basicRps = np.array([[1,     2.35,    0,          0.1],
                    [0,      1,       2.35,       0.1],
                    [2.35,   0,       1,          0.1],
                    [1.1,    1.1,     1.1,        0]])
"""



def payoffAgainstPop(population):
    payoffs = np.zeros(4)
    for i in range(4):
        payoffs[i] = sum(population[j] * basicRps[i][j] for j in range(4))
    return payoffs / (popSize - 1)


"""
Paper coevolutionary dynamics in large but finite populations

'An individidual of type j is chosen for repoduction with probabiiltiy i_j * Pi_j / (N * phi)
where phi = average payoff.
"""

def moranSelection(payoffs, avg, population):
    probs = np.zeros(4)
    for i in range(4):

        probs[i] = (population[i] * payoffs[i]) / (popSize * avg)

    return probs


def localUpdate(matrix, N, initialDist = [0.25, 0.25, 0.25, 0.25], iterations = 100000, w=0.5):

    population = np.random.multinomial(popSize, initialDist)

    R = np.zeros(iterations)
    P = np.zeros(iterations)
    S = np.zeros(iterations)
    L = np.zeros(iterations)


    for i in range(iterations):
        # Using this with no replacement fixed my drift issue !!!
        p1, p2 = np.random.choice([0,1,2,3], size=2, p=population/popSize, replace=False)
        
        payoffs = payoffAgainstPop(population)
        deltaPi = np.max(payoffs) - np.min(payoffs)
        #deltaPi = 2

        p = 1/2 + (w/2) * ((payoffs[p2] - payoffs[p1]) / deltaPi)


        # With this probability switch p1 to p2
        if (random.random() < p):
            population[p1] -= 1
            population[p2] += 1

        
        R[i] = population[0]
        P[i] = population[1]
        S[i] = population[2]
        L[i] = population[3]      

    # Return normalized RPSL distribution
    return R / popSize, P / popSize , S / popSize, L / popSize
       

def moranSimulation(matrix, N, initialDist = [0.25, 0.25, 0.25, 0.25], iterations = 100000, w=0.5):
    # Population represented just as their frequency of strategies for efficiency,
    # I think individual agents in simple dynamics unneccessary overhead
    population = np.random.multinomial(popSize, initialDist)

    
    R = np.zeros(iterations)
    P = np.zeros(iterations)
    S = np.zeros(iterations)
    L = np.zeros(iterations)

    for i in range(iterations):
        # Death: uniform random
        killed = random.choices([0, 1, 2, 3], weights=population)[0]
        # Birth: fitness-proportional
        # P = reproductive fitness in moran process 1 - w + w * Pi
        p = 1 - w + w * payoffAgainstPop(population)
        avg = np.sum(p * population) / popSize
        probs = moranSelection(p, avg, population)

        chosen = random.choices([0, 1, 2, 3], weights=probs)[0]
    
        population[chosen] += 1
        population[killed] -= 1

        # Can just use 1 var instead, with list of lists, but mauybe slower?
        R[i] = population[0]
        P[i] = population[1]
        S[i] = population[2]
        L[i] = population[3]

    # Return normalized RPSL distribution
    return R / popSize, P / popSize , S / popSize, L / popSize





popSize = 5000
simulations = 1
deltaMoran = []
deltaLocal = []

mResults = []
lResults = []


def singleSim(_):
    # Add other interaction processs here
    moranResult = moranSimulation(basicRps, 100, iterations = 100000)
    localResult = localUpdate(basicRps, 100, iterations = 100000)
    delta_L_moran = np.mean(np.diff(moranResult[3]))
    delta_L_local = np.mean(np.diff(localResult[3]))

    return moranResult, localResult, delta_L_moran, delta_L_local

# Method for api to call
def runSimulationPool():
    # Runs multiprocessing simulations for moran and local update process
    print("Running pool")
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

    return mResults, lResults




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

    """for i in range(simulations):
        #print(i)
        if i % 10 == 0:
            print(i)
        moranResult = moranSimulation(basicRps, 100, iterations = 10000)
        localResult = localUpdate(basicRps, 100, iterations = 10000)

        # Compute delta L for each simulation
        delta_L_moran = np.mean(np.diff(moranResult[3]))
        delta_L_local = np.mean(np.diff(localResult[3]))
        # Add to delta L history
        deltaMoran.append(delta_L_moran)
        deltaLocal.append(delta_L_local)

        if i == 0:
            mResults = np.array(moranResult)
            lResults = np.array(localResult)
        else:
            mResults += np.array(moranResult)
            lResults += np.array(localResult)
    """

    df_RPS_MO = pd.DataFrame({"c1": mResults[0], "c2": mResults[1], "c3": mResults[2], "c4": mResults[3]})

    df_RPS_LU = pd.DataFrame({"c1": lResults[0], "c2": lResults[1], "c3": lResults[2], "c4": lResults[3]})


    # Plot multiple results
    quaternaryPlot([df_RPS_LU, df_RPS_MO, df_RPS_MO, df_RPS_MO], labels=["Local update", "Moran process"], numPerRow=3, colors=['g', 'b'])

