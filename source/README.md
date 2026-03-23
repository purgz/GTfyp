# GTfyp Simulation Library

This directory contains the **pip-installable simulation library** extracted from the GTfyp project.

It is designed so users can install the repository and directly import:

```python
import simulation
import replicator
```

without needing the rest of the GTfyp application code.

---

## Contents

### simulation/
Finite-population evolutionary game simulations and plotting utilities.

Features:
- Interaction processes:
  - Moran process
  - Local update process
  - Fermi process
- Arbitrary N×N payoff matrices
- Trajectory output for time-series analysis
- Plotting helpers for:
  - 2D games (2 strategies)
  - Ternary plots (3 strategies)
  - Quaternary / simplex plots (4 strategies)
  - Animated point clouds for 4 strategy game
- Drift observable from simulated results


## Functions


### Simulation (Finite Population Dynamics)

Three functions moran_batch_sim, local_batch_sim, fermi_batch_sim, runs fast agent based simulations of the three
processes, can generate trajectories, or lots of trajectories (point clouds) with initial random or fixed initial distribution.

Uses numba preprocessing and simulations run in parallel for fast computation.

#### `moran_batch_sim(pop_size, iterations, matrix, w, initial_dist, simulations=1, traj=False, initial_rand=True, point_cloud=False)`
Runs batch simulations of the Moran process.

**Arguments:**
- `pop_size` *(int)* — Population size \( N \)
- `iterations` *(int)* — Total number of update steps
- `matrix` *(np.ndarray)* — Payoff matrix (N×N)
- `w` *(float)* — Selection strength
- `initial_dist` *(np.ndarray)* — Initial strategy distribution
- `simulations` *(int)* — Number of independent runs
- `traj` *(bool)* — Return trajectory data
- `initial_rand` *(bool)* — Random initial distribution if True
- `point_cloud` *(bool)* — Return full state distribution over time

**Returns:**
- `delta_H`, `delta_RPS`, trajectory data, (optional) point cloud

---

#### `local_batch_sim(pop_size, iterations, matrix, w, initial_dist, ...)`
Simulates the **local update (pairwise comparison)** process.

**Arguments:** Same as `moran_batch_sim`

**Notes:**
- Uses linear payoff comparison
- Depends on payoff difference normalization

---

#### `fermi_batch_sim(pop_size, iterations, matrix, w, initial_dist, ...)`
Simulates the **Fermi process** using a logistic update rule.

**Arguments:** Same as `moran_batch_sim`



### Plotting Utilities

#### `game_2d_plot(dfs, norm, N, sameAxis=True, labels=None, t_eval=None)`
Plots time evolution for 2-strategy systems.

**Arguments:**
- `dfs` *(list)* — List of pandas Series/DataFrames
- `norm` *(list[bool])* — Whether to normalize time by \( N \)
- `N` *(int or list)* — Population size(s)
- `labels` *(list)* — Plot labels
- `t_eval` *(array)* — Time axis (for deterministic trajectories)

---

#### `ternary_plot(dfs)`
Plots trajectories on a **2D simplex** (3 strategies).

**Arguments:**
- `dfs` *(list[pd.DataFrame])* — Each DataFrame contains strategy fractions

---

#### `quaternary_plot(dfs, numPerRow=2, labels=None, colors=None)`
Plots 4-strategy trajectories in a **3D simplex (tetrahedron)**.

**Arguments:**
- `dfs` *(list[pd.DataFrame])* — Trajectory data
- `numPerRow` *(int)* — Layout control
- `labels` *(list)* — Titles for each subplot
- `colors` *(list)* — Line colors

---

#### `quaternary_plot_same_axis(dfs, labels=None, colors=None)`
Overlays multiple trajectories on the **same 3D simplex**.

---

#### `high_dim_2d_plot(file_paths, Ns, labels=None, norm=None, t_eval=None, data_res=50, dfs=None)`
Projects higher-dimensional dynamics onto a **2D time series**.

**Arguments:**
- `file_paths` *(list[str])* — CSV inputs (optional)
- `dfs` *(list[pd.DataFrame])* — Direct data input
- `Ns` *(list[int])* — Population sizes
- `norm` *(list[bool])* — Normalize time axis
- `t_eval` *(array)* — Deterministic time values
- `data_res` *(int)* — Scaling factor for time

---

#### `point_cloud(dfs, matrix=None, repeat=False, save_file=None)`
Creates an **animated point cloud** of trajectories in the 4-strategy simplex.

**Arguments:**
- `dfs` *(list[pd.DataFrame])* — Simulation data with frame indexing
- `matrix` *(np.ndarray)* — Optional payoff matrix (displayed on plot)
- `repeat` *(bool)* — Loop animation
- `save_file` *(str)* — Output video file path

---

### Drift and Analysis Plots

#### `drift_plot_H(file_paths, labels=None, xlabel="N", column=None)`
Plots drift observable \( \langle \Delta H \rangle \).

**Arguments:**
- `file_paths` *(list[str])* — CSV data sources
- `labels` *(list)* — Plot labels
- `xlabel` *(str)* — X-axis label
- `column` *(int)* — Specific column to plot

---

#### `w_ensemble_plot(filePath, log=True, x_label="w")`
Plots drift or critical population size vs selection strength.

---

#### `drift_cases_plot()`
Plots example drift regimes for different parameter configurations.

---

#### `drift_cases_plot_pub(savepath=None)`
Publication-quality version of drift regime plot.

---

#### `drift_cases_plot_diagonal(savepath=None)`
Log–log comparison of:
- \( N/N_{SD} \)
- \( N/N_{RPS} \)

Shows regions of:
- Boundary attraction
- Interior coexistence
- Mixed regimes

---


### replicator/
Deterministic replicator dynamics

Features:
- Fixed-point and equilibrium calculations
- Numerical replicator dynamics
- Numerical values for the observation variable ΔH under different interaction processes  
  (see full analytical derivations in the project report)

**Full derivations and theory:**  
The complete mathematical derivations for ΔH, replicator dynamics, and critical population sizes
are provided in the LaTeX report:


[report/appendix.pdf](../report/appendix.pdf)



## Functions (replicator)

```python
numerical_trajectory_from_fokker_planck(
    matrix,
    interaction_process="Moran",
    w=0.2,
    initial_dist=[...],
    time_span=150
)
```
Main function for getting replicator solution trajectories.
Takes the payoff matrix as a numpy array and returns solved trajectory in same format as simulation functions.


---

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/purgz/GTfyp.git
```
---

## Quickstart Example

### 2×2 Prisoner’s Dilemma (Moran process)

See more complex examples in /example_code folder
[example_code](../example_code)


```python
"""
Import the sim and plotting as a pip module and show some basic usage.

pip install git+https://github.com/purgz/GTfyp.git

Below is an example script for the prisoners dilemma.
"""
import numpy as np
import simulation
import replicator
import pandas as pd
from simulation.games import Games

def main():
    print("Sucessful import")

    # Simple 3x3 RPS game
    matrix = Games.PRISONERS_DILEMMA
    w = 0.4

    print("Running Moran simulation...")

    delta_H, delta_rps, traj1, _ = simulation.moran_batch_sim(
        pop_size=200,
        iterations=200 * 50,
        w=w,
        initial_dist=np.array([0.9,0.1]),
        simulations=1,
        matrix=matrix,
        traj=True,
        initial_rand=False,
    )

    delta_H, delta_rps, traj2, _ = simulation.moran_batch_sim(
        pop_size=20000,
        iterations=20000 * 50,
        w=w,
        initial_dist=np.array([0.9,0.1]),
        simulations=1,
        matrix=matrix,
        traj=True,
        initial_rand=False,
    )

    delta_H, delta_rps, traj3, _ = simulation.moran_batch_sim(
        pop_size=100000,
        iterations=100000 * 50,
        w=w,
        initial_dist=np.array([0.9,0.1]),
        simulations=1,
        matrix=matrix,
        traj=True,
        initial_rand=False,
    )

    df = pd.DataFrame({"C": traj1[0], "D": traj1[1]})
    df2 = pd.DataFrame({"C": traj2[0], "D": traj2[1]})
    df3 = pd.DataFrame({"C": traj3[0], "D": traj3[1]})

    simulation.game_2d_plot([df["D"], df2["D"], df3["D"]],
                            N=[200,20000,100000],
                            norm=[True,True,True],
                            labels=["N=200","N=20000","N=100000"])

if __name__ == "__main__":
    main()
```

---

## Notes

### Matrix size must match initial_dist
The length of `initial_dist` must equal the number of strategies.

Examples:
- 2×2 game → `[0.5, 0.5]`
- 3×3 game → `[0.33, 0.33, 0.34]`
- 4×4 game → `[0.25, 0.25, 0.25, 0.25]`

If the dimensions do not match, NumPy/Numba will raise a runtime error.

---

### Plotting functions expect pandas objects
Some plotting helpers access `.values` internally.

Correct usage:
```python
simulation.game_2d_plot([df["D"]], N=pop_size, norm=[True])
```


---

## Dependencies

- numpy
- pandas
- matplotlib
- numba
- tqdm
- scienceplots
- python-ternary
