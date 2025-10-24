import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations
from multiprocessing import Pool
from tqdm import tqdm

from numba import njit, prange


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



"""def payoffAgainstPop(population,matrix, popSize):
  payoffs = np.zeros(matrix.shape[0])
  for i in range(matrix.shape[0]):
      payoffs[i] = sum((population[j]) * matrix[i][j] for j in range(matrix.shape[0]))
  return payoffs / (popSize-1)
"""

@njit(cache=True, inline="always")
def payoffAgainstPop(population, matrix, popSize):
    payoffs = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        total = 0.0
        for j in range(matrix.shape[0]):
            if i == j:
                # Exclude self-interaction
                total += (population[j] - 1) * matrix[i][j]
            else:
                total += population[j] * matrix[i][j]
        payoffs[i] = total
    return payoffs / (popSize - 1)



"""def payoffAgainstPop(population, matrix, popSize):
  return (matrix @ population) / (popSize - 1)
"""

"""
Paper coevolutionary dynamics in large but finite populations

'An individidual of type j is chosen for repoduction with probabiiltiy i_j * Pi_j / (N * phi)
where phi = average payoff.
"""

@njit(cache=True, inline="always")
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

    deltaPi = np.max(matrix) - np.min(matrix)
        
    for i in range(iterations):
        
        ind1, ind2 = np.random.choice(popSize, size=2, replace=False)
     
        p1, p2 = individuals[ind1], individuals[ind2]

        payoffs = payoffAgainstPop(population, matrix, popSize)
    
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
          
@njit(cache=True)
def localUpdate_numba(matrix, popSize, population, iterations=100000, w=0.4):
    numStrategies = matrix.shape[0]
    results = np.zeros((numStrategies, iterations))
    
    individuals = np.empty(popSize, dtype=np.int64)

    # Build individuals array from population
    idx = 0
    for s in range(numStrategies):
        for _ in range(population[s]):
            individuals[idx] = s
            idx += 1

    deltaPi = np.max(matrix) - np.min(matrix)

    for i in range(iterations):
        # Pick two distinct individuals
        ind1 = np.random.randint(popSize)
        ind2 = ind1
        while ind2 == ind1:
            ind2 = np.random.randint(popSize)

        p1 = individuals[ind1]
        p2 = individuals[ind2]

        payoffs = payoffAgainstPop(population, matrix, popSize)

        p = 0.5 + 0.5 * w * ((payoffs[p2] - payoffs[p1]) / deltaPi)

        if np.random.rand() < p:
            population[p1] -= 1
            population[p2] += 1
            individuals[ind1] = p2

        for j in range(numStrategies):
            results[j, i] = population[j] / popSize

  
    return results


@njit
def moranSimulation(matrix, popSize,population, initialDist = [0.1, 0.1, 0.1, 0.7], iterations = 100000, w=0.3):
    # Population represented just as their frequency of strategies for efficiency,
    # I think individual agents in simple dynamics unneccessary overhead
    #population = np.random.multinomial(popSize, initialDist)

    numStrategies = matrix.shape[0]
    
    results = np.zeros((numStrategies, iterations))

    for i in range(iterations):
        # Death: uniform random
        #killed = random.choices(range(numStrategies), weights=population)[0]
        killed = weighted_choice(population)
        # Birth: fitness-proportional
        # P = reproductive fitness in moran process 1 - w + w * Pi
        p = 1 - w + w * payoffAgainstPop(population, matrix, popSize)
        
        #avg = np.sum(p * population) / popSize
        avg = 0.0
        for j in range(numStrategies):
            avg += p[j] * population[j]
        avg /= float(popSize)

        probs = moranSelection(p, avg, population, popSize, matrix.shape[0])

        #chosen = random.choices(range(numStrategies), weights=probs)[0]
        chosen = weighted_choice(probs)
  

        population[chosen] += 1
        population[killed] -= 1

       
        for j in range(numStrategies):
            results[j][i] = population[j] / popSize

    # Return normalized RPSL distribution
    return results

"""
@njit(cache=True, inline="always")
def weighted_choice(weights):

    choices = [i for i in range(len(weights))]
    
    return np.searchsorted(np.cumsum(weights), np.random.random(), side="right")
"""

@njit(inline='always')
def weighted_choice(weights):
    # Unnormalized weights allowed
    total = np.sum(weights)
    r = np.random.rand() * total
    acc = 0.0
    for i in range(weights.size):
        acc += weights[i]
        if acc >= r:
            return i
    return weights.size - 1



@njit(cache=True)
def moranSimulation_numba(matrix, popSize, population, iterations=100000, w=0.3):
    numStrategies = matrix.shape[0]
    results = np.zeros((numStrategies, iterations))
    

    for i in range(iterations):

        
        killed = weighted_choice(population / popSize)


        payoffs = payoffAgainstPop(population, matrix, popSize)
        p = 1 - w + w * payoffs
        
        avg = np.sum(p * population) / popSize

        probs_birth = moranSelection(p, avg, population, popSize, matrix.shape[0])
        
        chosen = weighted_choice(probs_birth)

    
        population[chosen] += 1
        population[killed] -= 1

        for j in range(numStrategies):
          results[j, i] = population[j] / popSize


    return results


@njit(cache=True)
def fermiSim_numba(matrix, popSize, population, iterations=100000, w=0.3):
    numStrategies = matrix.shape[0]
    results = np.zeros((numStrategies, iterations))
    individuals = np.empty(popSize, dtype=np.int64)

    # Build individuals array from population
    idx = 0
    for s in range(numStrategies):
        for _ in range(population[s]):
            individuals[idx] = s
            idx += 1

    deltaPi = np.max(matrix) - np.min(matrix)

    for i in range(iterations):
        # Pick two distinct individuals
        ind1 = np.random.randint(popSize)
        ind2 = ind1
        while ind2 == ind1:
            ind2 = np.random.randint(popSize)

        p1 = individuals[ind1]
        p2 = individuals[ind2]

        payoffs = payoffAgainstPop(population, matrix, popSize)

        p = 1 / (1 + np.exp(-w * (payoffs[p2] - payoffs[p1])))

        if np.random.rand() < p:
            population[p1] -= 1
            population[p2] += 1
            individuals[ind1] = p2

        for j in range(numStrategies):
            results[j, i] = population[j] / popSize

    return results



"""
popSize = 100
simulations = 1
"""


def reseed():
    """
    Ensure independent randomness for each simulation run.
    Uses process ID and OS entropy to avoid collisions across processes.
    """
    seed = (os.getpid() * int.from_bytes(os.urandom(4), "little")) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)




def singleSim(matrix, popSize, initialDist, iterations, w, H, data_res, processes):
    
    reseed()
    
    population = np.random.multinomial(popSize, initialDist)
    population2 = np.random.multinomial(popSize, initialDist)

    results = []

    if "Moran" in processes:
      moranResult = moranSimulation_numba(matrix, popSize, population.copy(), iterations,w)
      delta_L_moran = np.mean(np.diff(-(moranResult[H] * (1 - moranResult[H]))))
      moranResult = moranResult[:, ::data_res]
      results.append((moranResult, delta_L_moran))
    if "Local" in processes:
      localResult = localUpdate_numba(matrix, popSize, population2.copy(), iterations,w)
      delta_L_local = np.mean(np.diff(-(localResult[H] * (1 - localResult[H]))))
      localResult = localResult[:, ::data_res]
      results.append((localResult, delta_L_local))
    if "Fermi" in processes:
      fermiResult = fermiSim_numba(matrix, popSize, population.copy(),iterations, w)
      delta_L_fermi = np.mean(np.diff(fermiResult[H]))
      fermiResult = fermiResult[:, ::data_res]
      results.append((fermiResult, delta_L_fermi))


    return results


def simHelper(args):
    return singleSim(*args)

# Method for api to call
def runSimulationPool(matrix=basicRps, popSize=100,
                       simulations=100, 
                       initialDist=[0.1, 0.1, 0.1, 0.7],
                       iterations=100000, w=0.4, H=3, data_res = 1,
                       processes=["Moran", "Local"]):
    # Runs multiprocessing simulations for moran and local update process

    # H parameter decides which strategy will be focussed for the drift analysis
    print("Processes " , processes)

    deltaMoran = []
    deltaLocal = []
    mResults = []
    lResults = []

    numStrategies = matrix.shape[0]

    """
    Testing with random initial conditions
    """

    """
    # Prepare initial dist
    fixed = initialDist[3]
    args = []
    for i in range(simulations):
        remaining = 1 - fixed
        random_simplex = np.random.rand(numStrategies - 1)
        random_simplex /= np.sum(random_simplex)
        random_simplex *= remaining
        initial = np.append(random_simplex, fixed)
        args.append((matrix, popSize, initial, iterations, w, H, data_res, processes))
    """
    
    args = [(matrix, popSize, initialDist, iterations, w, H, data_res, processes) for _ in range(simulations)]

    #print("Running simulation pool")
    #print("Strategies: ", numStrategies, " Population size: ", popSize, " Simulations: ", simulations, " Iterations: ", iterations, "w: ", w, " Initial distribution: ", initialDist)

    # Warm up numba - prevent threads recompiling each time.
    _ = moranSimulation_numba(basicRps, 10, np.array([2,3,4,1]), iterations=10, w=0.3)
    _ = localUpdate_numba(basicRps, 10, np.array([2,3,4,1]), iterations=10, w=0.3)

    with Pool() as pool:
       
       # Imap unordered allows usage of tqdm for progress bar.
       # Progres bar pauses at first because of numba warming up.

        for i , (results) in tqdm(
            enumerate(pool.imap_unordered(simHelper, args)), total=simulations, position=1, leave=False
            ):

            moranResult = results[0][0]
            delta_L_moran = results[0][1]
            deltaMoran.append(delta_L_moran)
            
            localResult = results[1][0]
            delta_L_local = results[1][1]
            
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
>>> from cProfile import Profile
>>> from pstats import SortKey, Stats

>>> def fib(n):
...     return n if n < 2 else fib(n - 2) + fib(n - 1)
...

>>> with Profile() as profile:
...     print(f"{fib(35) = }")
...     (
...         Stats(profile)
...         .strip_dirs()
...         .sort_stats(SortKey.CALLS)
...         .print_stats()
...     )

"""



# Running the file directly no longer works due to changed packacge structure, run via app.py
if __name__ == '__main__':
    """
    from cProfile import Profile
    from pstats import SortKey, Stats

    with Profile() as profile:
        print(f"{singleSim(basicRps,2000,[0.1,0.1,0.1,0.7], 1000000, 0.3,3)}")
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CALLS)
            .print_stats()
        )
    """
    
    moran, local, a, b = singleSim(
                          basicRps,
                          popSize=1000,
                          initialDist=[0.1,0.1,0.1,0.7], 
                          iterations=1000000, w=0.3,H=3, data_res=1)
    

    
    df_RPS_MO = pd.DataFrame({"c1": moran[0], "c2": moran[1], "c3": moran[2], "c4": moran[3]})

    quaternaryPlot(dfs=[df_RPS_MO],labels="Moran process")
    

    """
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
