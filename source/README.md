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


[latex_doc/out/main.pdf](../latex_doc/out/main.pdf)

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
