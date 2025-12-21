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


# If running this file directly uncomment following

# from plotting import quaternary_plot
from .plotting import quaternary_plot


# Here the loner > 0, therefore some central equivilibrium should be present - drift away from this would be towards pure RPS
basic_rps = np.array(
    [[0, -0.8, 1, 0.2], 
     [1, 0, -0.8, 0.2], 
     [-0.8, 1, 0, 0.2], 
     [0.2, 0.2, 0.2, 0]]
)


@njit(inline="always")
def payoff_against_pop(population, matrix, pop_size):
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
    return payoffs / (pop_size - 1)


"""
Paper coevolutionary dynamics in large but finite populations

'An individidual of type j is chosen for repoduction with probabiiltiy i_j * Pi_j / (N * phi)
where phi = average payoff.
"""


@njit(inline="always")
def moran_selection(payoffs, avg, population, pop_size, numStrategies=4):
    probs = np.zeros(numStrategies)
    for i in range(numStrategies):

        probs[i] = (population[i] * payoffs[i]) / (pop_size * avg)

    return probs


@njit(inline="always")
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



def reseed():
    seed = (os.getpid() * int.from_bytes(os.urandom(4), "little")) % (2**32 - 1)
    np.random.seed(seed)
    random.seed(seed)

@njit(parallel=True, cache=True)
def fermi_batch_sim(
    pop_size,
    iterations,
    w,
    simulations,
    matrix=basic_rps,
    initial_dist=np.array([0.25, 0.25, 0.25, 0.25]),
    traj=False,
    point_cloud=False,
    initial_rand=True
):

    n = matrix.shape[0]
    deltas = np.zeros(simulations)
    deltas_rps = np.zeros(simulations)
    all_results = np.zeros((n, iterations))

    # help with weird floating point errors
    initial_dist = initial_dist / np.sum(initial_dist)

    sample_rate = 5000
    num_frames = iterations // sample_rate
    all_traj = np.zeros((simulations, n, num_frames))


    for s in prange(simulations):

        if initial_rand:
            initial = np.random.exponential(1,n)
            initial /= np.sum(initial)
        else:
            initial = initial_dist

        population = np.random.multinomial(pop_size, initial)

        deltaPi = np.max(matrix) - np.min(matrix)

        results = np.zeros((n, iterations))

        individuals = np.empty(pop_size, dtype=np.int64)

        # Build individuals array from population
        idx = 0
        for i in range(n):
            for _ in range(population[i]):
                individuals[idx] = i
                idx += 1

        for i in range(iterations):
            ind1 = np.random.randint(pop_size)
            ind2 = ind1
            while ind2 == ind1:
                ind2 = np.random.randint(pop_size)

            p1 = individuals[ind1]
            p2 = individuals[ind2]

            payoffs = payoff_against_pop(population, matrix, pop_size)

            p = 1 / (1 + np.exp(-w * (payoffs[p2] - payoffs[p1])))

            if np.random.rand() < p:
                population[p1] -= 1
                population[p2] += 1
                individuals[ind1] = p2

            for j in range(n):
                results[j, i] = population[j] / pop_size

        H = n - 1 if n == 4 else 0
        if iterations >= 2:
            delta_H = (-(results[H, 1] * (1 - results[H, 1]))) - (
                -(results[H, 0] * (1 - results[H, 0]))
            )

        deltas[s] = delta_H

        if n >= 3:
            H_before = -(results[0, 0] * results[1, 0] * results[2, 0])
            H_after = -(results[0, 1] * results[1, 1] * results[2, 1])
            deltas_rps[s] = H_after - H_before

        if traj:
            all_results += results

        if point_cloud:
            all_traj[s, :, :] = results[:, ::sample_rate]

    mean_delta_H = np.mean(deltas)
    mean_delta_rps = np.mean(deltas_rps)

    return mean_delta_H, mean_delta_rps, all_results / simulations, all_traj

@njit(parallel=True, cache=True)
def local_batch_sim(
    pop_size,
    iterations,
    w,
    simulations,
    matrix=basic_rps,
    initial_dist=np.array([0.25, 0.25, 0.25, 0.25]),
    traj=False,
    point_cloud=False,
    initial_rand=True
):

    n = matrix.shape[0]
    deltas = np.zeros(simulations)
    deltas_rps = np.zeros(simulations)
    all_results = np.zeros((n, iterations))

    # help with weird floating point errors
    initial_dist = initial_dist / np.sum(initial_dist)


    sample_rate = 10000
    num_frames = iterations // sample_rate

    all_traj = np.zeros((simulations, n, num_frames))

  
    for s in prange(simulations):
        # Randomize in the simplex
        """fixed = initial_dist[3]
        remaining = 1 - fixed
        random_simplex = np.random.rand(n - 1)
        random_simplex /= np.sum(random_simplex)
        random_simplex *= remaining
        initial = np.append(random_simplex, fixed)
        """
        if initial_rand:
            initial = np.random.exponential(1,n)
            initial /= np.sum(initial)
     
        else:
            initial = initial_dist

        population = np.random.multinomial(pop_size, initial)

        deltaPi = np.max(matrix) - np.min(matrix)

        results = np.zeros((n, iterations))

        individuals = np.empty(pop_size, dtype=np.int64)

        # Build individuals array from population
        idx = 0
        for i in range(n):
            for _ in range(population[i]):
                individuals[idx] = i
                idx += 1

        for i in range(iterations):
            ind1 = np.random.randint(pop_size)
            ind2 = ind1
            while ind2 == ind1:
                ind2 = np.random.randint(pop_size)

            p1 = individuals[ind1]
            p2 = individuals[ind2]

            payoffs = payoff_against_pop(population, matrix, pop_size)
            p = 0.5 + 0.5 * w * ((payoffs[p2] - payoffs[p1]) / deltaPi)

            if np.random.rand() < p:
                population[p1] -= 1
                population[p2] += 1
                individuals[ind1] = p2

            for j in range(n):
                results[j, i] = population[j] / pop_size

        H = n - 1 if n == 4 else 0
        if iterations >= 2:
            delta_H = (-(results[H, 1] * (1 - results[H, 1]))) - (
                -(results[H, 0] * (1 - results[H, 0]))
            )

        deltas[s] = delta_H

        if n >= 3:
            H_before = -(results[0, 0] * results[1, 0] * results[2, 0])
            H_after = -(results[0, 1] * results[1, 1] * results[2, 1])
            deltas_rps[s] = H_after - H_before

        if traj:
            all_results += results

        if point_cloud:
            all_traj[s, :, :] = results[:, ::sample_rate]

    mean_delta_H = np.mean(deltas)
    mean_delta_rps = np.mean(deltas_rps)

    return mean_delta_H, mean_delta_rps, all_results / simulations, all_traj


@njit(parallel=True, cache=True)
def moran_batch_sim(
    pop_size,
    iterations,
    w,
    simulations,
    matrix=basic_rps,
    initial_dist=np.array([0.25, 0.25, 0.25, 0.25]),
    traj=False,
    point_cloud=False,
    initial_rand=True,
    fixed_point=None
):
    n = matrix.shape[0]
    deltas = np.zeros(simulations)
    deltas_rps = np.zeros(simulations)
    all_results = np.zeros((n, iterations))

    initial_dist = initial_dist / np.sum(initial_dist)

    sample_rate = 1000
    num_frames = iterations // sample_rate

    all_traj = np.zeros((simulations, n, num_frames))


    #  Need to be able to pass the fixedd point for given matrix - bias initial slightly

    for s in prange(simulations):

        
        """# Randomize in the simplex
        fixed = initial_dist[3]
        remaining = 1 - fixed
        random_simplex = np.random.rand(n - 1)
        random_simplex /= np.sum(random_simplex)
        random_simplex *= remaining
        initial = np.append(random_simplex, fixed)
        """
        # Sample distribution options.
        if initial_rand:
           
            if fixed_point is not None:
              # If we have a fixd point bias around it.
              x_star_vec = fixed_point
              scale = 0.2
              alpha_centered = x_star_vec * scale  + 1e-7 # scale >> 1 sharpens around fixed point
              initial = np.random.dirichlet(alpha_centered) # dirchlet centered around the fixed point.
              initial = np.clip(initial, 1e-7, 1-1e-7)
              initial /= np.sum(initial)
            else:
              initial = np.random.exponential(1,n)
         
              initial /= np.sum(initial)
        else:
            # For non random (single trajectory plottting) just use the provided dist.
            initial = initial_dist

        population = np.random.multinomial(pop_size, initial)

        results = np.zeros((n, iterations + 1))


        for i in range(n):
            results[i, 0] = population[i] / pop_size

        for i in range(iterations):
            killed = weighted_choice(population / pop_size)
            payoffs = payoff_against_pop(population, matrix, pop_size)
            p = 1.0 - w + w * payoffs
            avg = np.sum(p * population) / pop_size
            probs_birth = moran_selection(p, avg, population, pop_size, n)
            chosen = weighted_choice(probs_birth)

            population[chosen] += 1
            population[killed] -= 1

            for j in range(n):
                results[j, i+1] = population[j] / pop_size


        H = n - 1 if n == 4 else 0
        if iterations >= 1:
            delta_H = (-(results[H, 1] * (1 - results[H, 1]))) - (
                -(results[H, 0] * (1 - results[H, 0]))
            )

        deltas[s] = delta_H

        if n >= 3:
            H_before = -(results[0, 0] * results[1, 0] * results[2, 0])
            H_after = -(results[0, 1] * results[1, 1] * results[2, 1])
            deltas_rps[s] = H_after - H_before

        if traj:
            all_results += results

        if point_cloud:
            all_traj[s, :, :] = results[:, 1::sample_rate]

    mean_delta_H = np.mean(deltas)
    mean_delta_rps = np.mean(deltas_rps)

    return mean_delta_H, mean_delta_rps, all_results / simulations, all_traj



# Running the file directly no longer works due to changed packacge structure, run via app.py
if __name__ == "__main__":
    """
    from cProfile import Profile
    from pstats import SortKey, Stats

    with Profile() as profile:
        print(f"{singleSim(basic_rps,20000,[0.5,0.2,0.2,0.1], 1000000, 0.2,3, data_res=50, processes="Moran")}")
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CALLS)
            .print_stats()
        )
    """
