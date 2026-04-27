"""
demo.py  —  Live demonstration of Combined RPS-SD Evolutionary Game Dynamics
=============================================================================
Organised as clearly labelled, independently runnable demo functions.

  • Section 1 : 2×2 Snowdrift warm-up
  • Section 2 : 3×3 RPS warm-up (ternary plot)
  • Section 3 : 4×4 deterministic trajectories, all three processes
  • Section 4 : 4×4 neutral (zero-sum RPS) combined game
  • Section 5 : 4×4 simulation vs. replicator  (large-N & small-N comparison)
  • Section 6 : Drift reversal demo (Fig. 10 in report)
  • Section 7 : Point-cloud animations — column cloud & disk cloud
                (computed with small parameters for live running, saved to CSV)
  • Section 8 : Large-N interior coexistence  (deterministic + optional sim)

Usage
-----
  python demo.py              # runs full sequence via main()
  python demo.py --section 3  # run a single numbered section

Parameters are deliberately small for live demo speed.  The comments next
to each simulation call note how to scale up for publication-quality output.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simulation
import replicator
from simulation.games import Games


plt.style.use(["science", "no-latex"])


# ── Output directory for CSV saves ───────────────────────────────────────────
RESULTS_DIR = "./demo_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Global selection strength ─────────────────────────────────────────────────
W = 0.45


# ═════════════════════════════════════════════════════════════════════════════
# Shared payoff matrices  (eq. 3 in report)
# ═════════════════════════════════════════════════════════════════════════════

# Attracting combined game  (c = -0.8 > -1  →  interior stable fixed pt)
# Fixed point: (2/7, 2/7, 2/7, 1/7)
RPS_SD_ATTRACT = np.array([
    [ 0.0, -0.8,  1.0,  0.2],
    [ 1.0,  0.0, -0.8,  0.2],
    [-0.8,  1.0,  0.0,  0.2],
    [ 0.1,  0.1,  0.1,  0.0],
])

# Neutral combined game  (c = -1  →  neutrally stable RPS orbits in RPS plane)
# Fixed point: (2/9, 2/9, 2/9, 1/3)
RPS_SD_NEUTRAL = np.array([
    [ 0.0, -1.0,  1.0,  0.2],
    [ 1.0,  0.0, -1.0,  0.2],
    [-1.0,  1.0,  0.0,  0.2],
    [ 0.1,  0.1,  0.1,  0.0],
])

# SD-reversal case:  N_RPS < N_SD  →  column-shaped point cloud  (Fig 11)
# With β=0.3, γ=0.14, c=-0.2, N_RPS ≈ 89, N_SD ≈ 1390
RPS_SD_COLUMN = np.array([
    [ 0.0, -0.2,  1.0,  0.14],
    [ 1.0,  0.0, -0.2,  0.14],
    [-0.2,  1.0,  0.0,  0.14],
    [ 0.3,  0.3,  0.3,  0.0 ],
])

# RPS-reversal case:  N_SD < N_RPS  →  disk-shaped point cloud  (Fig 12)
# With β=0.24, γ=0.4, c=-0.8, N_SD ≈ 96, N_RPS ≈ 1400
RPS_SD_DISK = np.array([
    [ 0.0, -0.8,  1.0,  0.4 ],
    [ 1.0,  0.0, -0.8,  0.4 ],
    [-0.8,  1.0,  0.0,  0.4 ],
    [0.24, 0.24, 0.24,  0.0 ],
])


# ═════════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═════════════════════════════════════════════════════════════════════════════

def _save_trajectory(df: pd.DataFrame, path: str) -> None:
    """Save a trajectory DataFrame to CSV."""
    df.to_csv(path, index=False)
    print(f"  Saved  → {path}")


def _load_trajectory(path: str) -> pd.DataFrame:
    """Load a trajectory CSV produced by _save_trajectory."""
    return pd.read_csv(path, comment="#")


def _trajectories_to_anim(trajectories):

    sims, n, frames = trajectories.shape
    transposed = trajectories.transpose(0,2,1)

    df = pd.DataFrame(
        transposed.reshape(-1, n),  # shape (sims*frames, n)
        columns=[f"c{i+1}" for i in range(n)],
    )

    df["sim"] = np.repeat(np.arange(sims), frames)
    df["frame"] = np.tile(np.arange(frames), sims)

    return df


def _section_header(title: str) -> None:
    width = 72
    print("\n" + "═" * width)
    print(f"  {title}")
    print("═" * width)


# ═════════════════════════════════════════════════════════════════════════════
# Section 1 — 2×2 Snowdrift game
# ═════════════════════════════════════════════════════════════════════════════

def demo_2x2_snowdrift() -> None:
    """
    Snowdrift game — convergence to the interior mixed-strategy Nash
    equilibrium q* = 1/2 (Fig 3 in report).

    Shows:
      • Moran process simulation  (N = 10 000)
      • Local update simulation   (N = 10 000)
      • Standard replicator dynamics
      • Adjusted (Moran) replicator dynamics
    All starting from ≈ 100 % defectors.
    """
    _section_header("Demo 1 — 2×2 Snowdrift game")
    matrix = Games.SNOWDRIFT
    w      = 0.9
    init   = np.array([0.01, 0.99])   # start near all-defectors
    N      = 10_000
    iters  = N * 35                   # 35 time steps (normalised)

    print("  Running Moran simulation …")
    _, _, m_traj, _ = simulation.moran_batch_sim(
        pop_size=N, iterations=iters, w=w, matrix=matrix,
        simulations=1, initial_dist=init,
        traj=True, initial_rand=False,
    )

    print("  Running Local-update simulation …")
    _, _, l_traj, _ = simulation.local_batch_sim(
        pop_size=N, iterations=iters, w=w, matrix=matrix,
        simulations=1, initial_dist=init,
        traj=True, initial_rand=False,
    )

    df_mo = pd.DataFrame({"C": m_traj[0], "D": m_traj[1]})
    df_lu = pd.DataFrame({"C": l_traj[0], "D": l_traj[1]})

    num_std         = replicator.pdNumerical(matrix, w, list(init))
    num_adj, t_eval = replicator.pd_adjusted(matrix, w, list(init))

    # Plot defector fraction for all four trajectories
    simulation.game_2d_plot(
        [df_mo["D"], df_lu["D"], num_adj["D"], num_std["D"]],
        N=N,
        labels=["MO", "LO", "Adjusted", "Standard"],
        norm=[True, True, False, False],
        t_eval=t_eval,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 2 — 3×3 Rock-Paper-Scissors  (ternary plot)
# ═════════════════════════════════════════════════════════════════════════════

def demo_3x3_rps() -> None:
    """
    Two RPS scenarios shown on a ternary simplex plot (Figs 1 & 2):
      (a) Standard zero-sum  c = -1  →  neutral closed orbits
      (b) Attracting         c = -0.8 → spiral inward to (1/3, 1/3, 1/3)

    Each scenario shows:
      • Large-N simulation  (N = 50 000)  — closely follows replicator
      • Small-N simulation  (N = 2 000)   — shows stochastic deviations
      • Numerical deterministic trajectory
    """
    _section_header("Demo 2 — 3×3 RPS  (ternary plot)")
    w     = 0.3
    init3 = np.array([0.6, 0.2, 0.2])

    for label, c_val in [("zero-sum (c = -1)", -1.0), ("attracting (c = -0.8)", -0.8)]:
        print(f"\n  ── RPS {label} ──")
        rps3 = np.array([[0, c_val, 1], [1, 0, c_val], [c_val, 1, 0]])

        # Pad to 4×4 for the Fokker-Planck numerical integrator
        rps4 = np.block([
            [rps3,            np.zeros((3, 1))],
            [np.zeros((1, 3)), np.zeros((1, 1))],
        ])

        for N, label_n in [(50_000, "N=50000"), (2_000, "N=2000")]:
            print(f"    Running Moran simulation {label_n} …")
            _, _, traj, _ = simulation.moran_batch_sim(
                pop_size=N, iterations=N * 100, w=w, matrix=rps3,
                simulations=1, initial_dist=init3,
                traj=True, initial_rand=False,
            )
            dfs_to_plot = [pd.DataFrame(
                {"R": traj[0], "P": traj[1], "S": traj[2]}
            )]

        print("    Computing deterministic trajectory …")
        num, _ = replicator.numerical_trajectory_from_fokker_planck(
            rps4, interaction_process="Moran",
            w=w, initial_dist=list(init3), time_span=100,
        )
        num_rps = num[["c1", "c2", "c3"]].rename(
            columns={"c1": "R", "c2": "P", "c3": "S"}
        )

        # Re-run both N values properly for the plot
        trajs = []
        for N in [50_000, 2_000]:
            _, _, traj, _ = simulation.moran_batch_sim(
                pop_size=N, iterations=N * 100, w=w, matrix=rps3,
                simulations=1, initial_dist=init3,
                traj=True, initial_rand=False,
            )
            trajs.append(pd.DataFrame(
                {"R": traj[0], "P": traj[1], "S": traj[2]}
            ))

        simulation.ternary_plot([num_rps, trajs[0], trajs[1]])


# ═════════════════════════════════════════════════════════════════════════════
# Section 3 — 4×4 deterministic trajectories  (all three processes)
# ═════════════════════════════════════════════════════════════════════════════

def demo_4x4_deterministic() -> None:
    """
    Numerical (deterministic, N → ∞) trajectories for the attracting
    combined game under all three microscopic processes:
      • Moran process         (adjusted replicator, eq. 16)
      • Local update process  (standard replicator scaled by w, eq. 17)
      • Fermi process         (pairwise-difference form, eq. 18)

    Saves trajectories to CSV so they can be reloaded without re-integrating.
    """
    _section_header("Demo 3 — 4×4 deterministic trajectories (three processes)")
    matrix = RPS_SD_ATTRACT
    w      = 0.45
    init   = [0.5, 0.2, 0.2]   # initial (x, y, z);  q = 1 - x - y - z

    trajectories = {}
    for process in ("Moran", "Local", "Fermi"):
        print(f"  Integrating {process} replicator …")
        df, t_eval = replicator.numerical_trajectory_from_fokker_planck(
            matrix, interaction_process=process,
            w=w, initial_dist=init,
        )
        trajectories[process] = df
        path = os.path.join(RESULTS_DIR, f"numerical_{process.lower()}.csv")
        _save_trajectory(df, path)

    # 3-D simplex — all three on the same axis
    simulation.quaternary_plot_same_axis(
        list(trajectories.values()),
        labels=list(trajectories.keys()),
        colors=["b", "g", "r"],
    )

    # 2-D time series of the + (snowdrift) strategy fraction
    simulation.high_dim_2d_plot(
        None,
        Ns=[None, None, None],
        norm=[False, False, False],
        dfs=list(trajectories.values()),
        labels=list(trajectories.keys()),
        t_eval=t_eval,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 4 — 4×4 neutral combined game  (zero-sum RPS in RPS plane)
# ═════════════════════════════════════════════════════════════════════════════

def demo_4x4_neutral() -> None:
    """
    Combined game with standard zero-sum RPS  (c = -1, Fig 4 in report).

    The trajectory:
      (1) converges to the RPS plane along the SD axis  (eigenvalue −1/15)
      (2) then forms neutral closed orbits at a fixed distance from (2/9, 2/9, 2/9, 1/3)
          (purely imaginary eigenvalue pair)

    Uses a large-N Moran simulation and the Moran adjusted replicator.
    """
    _section_header("Demo 4 — 4×4 neutral combined game  (c = -1)")
    matrix = RPS_SD_NEUTRAL
    w      = 0.45
    init4  = np.array([0.5, 0.2, 0.2, 0.1])
    N      = 50_000          # large N → tracks deterministic trajectory
    iters  = N * 150         # ~150 normalised time steps

    print("  Integrating Moran replicator …")
    num_mo, t_eval = replicator.numerical_trajectory_from_fokker_planck(
        matrix, interaction_process="Moran",
        w=w, initial_dist=[0.5, 0.2, 0.2],
    )

    print(f"  Running Moran simulation  N={N} …")
    _, _, traj, _ = simulation.moran_batch_sim(
        pop_size=N, iterations=iters, w=w, matrix=matrix,
        simulations=1, initial_dist=init4,
        traj=True, initial_rand=False,
    )
    df_sim = pd.DataFrame({
        "R": traj[0], "P": traj[1], "S": traj[2], "L": traj[3],
    })

    simulation.quaternary_plot_same_axis(
        [num_mo, df_sim],
        labels=["Numerical (Moran)", f"Moran N={N}"],
        colors=["b", "g"],
    )

    simulation.high_dim_2d_plot(
        None,
        Ns=[None, N],
        norm=[False, True],
        dfs=[num_mo, df_sim],
        data_res=1,
        labels=["Replicator", f"Moran N={N}"],
        t_eval=t_eval,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 5 — 4×4 simulation vs. replicator  (large N & small N)
# ═════════════════════════════════════════════════════════════════════════════

def demo_4x4_sim_vs_replicator() -> None:
    """
    Demonstrates the two-regime behaviour of the attracting combined game:

      Large N = 100 000 : simulation tightly tracks the replicator
      Small N = 200     : averaged over many realisations; drift reversal

    Adds CSV caching:
      • Loads saved trajectories if available
      • Otherwise computes and saves them for future runs
    """
    _section_header("Demo 5 — 4×4 simulation vs. replicator (large N & small N)")

    matrix  = RPS_SD_ATTRACT
    w       = 0.45
    init4   = np.array([0.5, 0.2, 0.2, 0.1])
    N_large = 50_000
    N_small = 200
    sims_lo = 5000

    # Cache file paths
    num_path   = os.path.join(RESULTS_DIR, "section5_numerical.csv")
    large_path = os.path.join(RESULTS_DIR, f"section5_moran_large_N{N_large}.csv")
    small_path = os.path.join(
        RESULTS_DIR,
        f"section5_moran_small_N{N_small}_sims{sims_lo}.csv"
    )

    # ── Deterministic replicator (cached) ──────────────────────────────
    if os.path.exists(num_path):
        print(f"  Loading cached numerical trajectory → {num_path}")
        num_mo = _load_trajectory(num_path)

        # still needed for plotting time axis
        t_eval = np.linspace(0, 150, len(num_mo))
    else:
        print("  Integrating Moran replicator …")
        num_mo, t_eval = replicator.numerical_trajectory_from_fokker_planck(
            matrix,
            interaction_process="Moran",
            w=w,
            initial_dist=[0.5, 0.2, 0.2],
        )
        _save_trajectory(num_mo, num_path)

    # ── Large-N simulation (cached) ────────────────────────────────────
    if os.path.exists(large_path):
        print(f"  Loading cached large-N simulation → {large_path}")
        df_large = _load_trajectory(large_path)
    else:
        print(f"  Running Moran simulation N={N_large} …")
        _, _, traj_large, _ = simulation.moran_batch_sim(
            pop_size=N_large,
            iterations=N_large * 150,
            w=w,
            matrix=matrix,
            simulations=20,
            initial_dist=init4,
            traj=True,
            initial_rand=False,
        )

        df_large = pd.DataFrame({
            "R": traj_large[0],
            "P": traj_large[1],
            "S": traj_large[2],
            "L": traj_large[3],
        })
        _save_trajectory(df_large, large_path)

    # ── Small-N multi-run simulation (cached) ──────────────────────────
    if os.path.exists(small_path):
        print(f"  Loading cached small-N simulation → {small_path}")
        df_small = _load_trajectory(small_path)
    else:
        print(
            f"  Running Moran simulation N={N_small} ×{sims_lo} realisations …"
        )
        _, _, traj_small, _ = simulation.moran_batch_sim(
            pop_size=N_small,
            iterations=N_small * 150,
            w=w,
            matrix=matrix,
            simulations=sims_lo,
            initial_dist=init4,
            traj=True,
            initial_rand=False,
        )

        df_small = pd.DataFrame({
            "R": traj_small[0],
            "P": traj_small[1],
            "S": traj_small[2],
            "L": traj_small[3],
        })
        _save_trajectory(df_small, small_path)

    # ── Plot: simplex (numerical vs large-N sim) ───────────────────────
    simulation.quaternary_plot_same_axis(
        [num_mo, df_large],
        labels=["Numerical (N → ∞)", f"Moran N={N_large}"],
        colors=["b", "g"],
    )

    # ── Plot: 2D time series comparison ────────────────────────────────
    simulation.high_dim_2d_plot(
        None,
        Ns=[None, N_large, N_small],
        norm=[False, True, True],
        data_res=1,
        dfs=[num_mo, df_large, df_small],
        labels=[
            "Replicator",
            f"N={N_large}",
            f"N={N_small} (drift reversal, ×{sims_lo})",
        ],
        t_eval=t_eval,
    )

# ═════════════════════════════════════════════════════════════════════════════
# Section 6 — Drift reversal  (Fig 10 equivalent)
# ═════════════════════════════════════════════════════════════════════════════

def demo_drift_reversal() -> None:
    """
    Drift reversal demo (Fig 10 in report):
      • Large N (= 100 000) — follows deterministic replicator trajectory
      • Small N (= 200), averaged over many realisations — + fraction
        moves *away* from the replicator fixed point towards zero.

    Shows Moran and Local-update processes side by side.
    All trajectories start at the uniform interior distribution (1/4, 1/4, 1/4, 1/4).
    """
    _section_header("Demo 6 — Drift reversal  (Fig 10 in report)")
    matrix  = RPS_SD_ATTRACT
    w       = 0.45
    init4   = np.array([0.3, 0.3, 0.3, 0.1])
    N_large = 100000
    N_small = 200
    sims_lo = 1000   # increase to 1000 for Fig 10 quality

    print("  Integrating Moran / Local replicators …")
    num_mo, t_eval = replicator.numerical_trajectory_from_fokker_planck(
        matrix, interaction_process="Moran",
        w=w, initial_dist=[0.3, 0.3, 0.3],
    )
    num_lu, _ = replicator.numerical_trajectory_from_fokker_planck(
        matrix, interaction_process="Local",
        w=w, initial_dist=[0.3, 0.3, 0.3],
    )

    dfs, labels, Ns, norms = [num_mo, num_lu], ["Replicator MO", "Replicator LU"], [None, None], [False, False]

    for process_name, sim_func in [("Moran", simulation.moran_batch_sim),
                                    ("Local", simulation.local_batch_sim)]:
        print(f"  Running {process_name}  N={N_large} …")
        _, _, traj, _ = sim_func(
            pop_size=N_large, iterations=N_large * 150, w=w, matrix=matrix,
            simulations=5, initial_dist=init4,
            traj=True, initial_rand=False,
        )
        df = pd.DataFrame({"R": traj[0], "P": traj[1], "S": traj[2], "L": traj[3]})
        dfs.append(df)
        labels.append(f"{process_name}  N={N_large}")
        Ns.append(N_large)
        norms.append(True)

        print(f"  Running {process_name}  N={N_small}  ×{sims_lo} realisations …")
        _, _, traj_s, _ = sim_func(
            pop_size=N_small, iterations=N_small * 150, w=w, matrix=matrix,
            simulations=sims_lo, initial_dist=init4,
            traj=True, initial_rand=False,
        )
        df_s = pd.DataFrame({"R": traj_s[0], "P": traj_s[1], "S": traj_s[2], "L": traj_s[3]})
        dfs.append(df_s)
        labels.append(f"{process_name}  N={N_small}  (drift reversal)")
        Ns.append(N_small)
        norms.append(True)

    simulation.high_dim_2d_plot(
        None,
        Ns=Ns,data_res = 1, norm=norms,
        dfs=dfs, labels=labels,
        t_eval=t_eval,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Section 7 — Point-cloud animations  (column & disk drift-reversal cases)
# ═════════════════════════════════════════════════════════════════════════════

def demo_point_cloud_column(force_recompute: bool = False) -> None:
    """
    SD-reversal case (Fig 11 / lower-left of Fig 13):
      N_RPS < N < N_SD  →  trajectories fill a column-shaped cloud
                            (stable in RPS, unstable in SD direction).

    Parameters used: β=0.30, γ=0.14, c=−0.2, N=300, w=0.45
    Critical sizes for this matrix: N_RPS ≈ 89, N_SD ≈ 1390.

    Data is saved to CSV after generation so subsequent runs simply reload it.
    Set force_recompute=True to regenerate from scratch.
    """
    _section_header("Demo 7a — Point cloud: column  (SD reversal)")
    path = os.path.join(RESULTS_DIR, "cloud_column.csv")

    if os.path.exists(path) and not force_recompute:
        print(f"  Reloading saved cloud from {path}")
        anim = _load_trajectory(path)
    else:
        pop_size   = 300      # N_RPS (≈89) < 300 < N_SD (≈1390)
        iterations = 200_000  # scale up to 500_000 for cleaner shape
        num_points = 300      # number of random initial conditions
        sims       = 20       # short average per point to suppress noise

        print(f"  Computing cloud  (N={pop_size}, {num_points} points) …")
        cloud = simulation.local_batch_cloud(
            pop_size, iterations=iterations, num_points=num_points,
            w=W, simulations=sims, matrix=RPS_SD_COLUMN,
        )
        anim = _trajectories_to_anim(cloud)
        _save_trajectory(anim, path)

    simulation.point_cloud([anim], matrix=None)


def demo_point_cloud_disk(force_recompute: bool = False) -> None:
    """
    RPS-reversal case (Fig 12 / lower-right of Fig 13):
      N_SD < N < N_RPS  →  trajectories fill a disk-shaped cloud
                            (stable in SD, unstable in RPS direction).

    Parameters used: β=0.24, γ=0.40, c=−0.8, N=600, w=0.45
    Critical sizes for this matrix: N_SD ≈ 96, N_RPS ≈ 1400.

    Data is saved to CSV after generation so subsequent runs simply reload it.
    Set force_recompute=True to regenerate from scratch.
    """
    _section_header("Demo 7b — Point cloud: disk  (RPS reversal)")
    path = os.path.join(RESULTS_DIR, "cloud_disk.csv")

    if os.path.exists(path) and not force_recompute:
        print(f"  Reloading saved cloud from {path}")
        anim = _load_trajectory(path)
    else:
        pop_size   = 600      # N_SD (≈96) < 600 < N_RPS (≈1400)
        iterations = 400_000  # scale up to 1_000_000 for cleaner disk
        num_points = 300
        sims       = 20

        print(f"  Computing cloud  (N={pop_size}, {num_points} points) …")
        cloud = simulation.local_batch_cloud(
            pop_size, iterations=iterations, num_points=num_points,
            w=W, simulations=sims, matrix=RPS_SD_DISK,
        )
        anim = _trajectories_to_anim(cloud)
        _save_trajectory(anim, path)

    simulation.point_cloud([anim], matrix=None)


def demo_point_clouds_from_saved() -> None:
    """
    Reload previously saved column and disk CSVs and display both.
    Run demo_point_cloud_column() and demo_point_cloud_disk() first.
    """
    _section_header("Demo 7c — Point clouds: reload from saved CSVs")
    col_path  = os.path.join(RESULTS_DIR, "cloud_column.csv")
    disk_path = os.path.join(RESULTS_DIR, "cloud_disk.csv")

    missing = [p for p in (col_path, disk_path) if not os.path.exists(p)]
    if missing:
        print("  Missing CSVs:", missing)
        print("  Run demo_point_cloud_column() and demo_point_cloud_disk() first.")
        return

    col_anim  = _load_trajectory(col_path)
    disk_anim = _load_trajectory(disk_path)

    print("  Column cloud (SD reversal):")
    simulation.point_cloud([col_anim],  matrix=RPS_SD_COLUMN)
    print("  Disk cloud   (RPS reversal):")
    simulation.point_cloud([disk_anim], matrix=RPS_SD_DISK)




# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

SECTIONS = {
    1: ("2×2 Snowdrift game",                  demo_2x2_snowdrift),
    2: ("3×3 RPS  (ternary plot)",              demo_3x3_rps),
    3: ("4×4 deterministic, three processes",   demo_4x4_deterministic),
    4: ("4×4 neutral combined game",            demo_4x4_neutral),
    5: ("4×4 simulation vs. replicator",        demo_4x4_sim_vs_replicator),
    6: ("Drift reversal  (Fig 10)",             demo_drift_reversal),
    7: ("Point-cloud animations",               lambda: (demo_point_cloud_column(), demo_point_cloud_disk()))
}


def main(section: int | None = None) -> None:
    """
    Run the full demo sequence, or a single numbered section.

    Quick live-demo guide
    ─────────────────────
    For fastest results comment out the heavy sections (6, 7) and run only:
      • Section 1  — 2×2 warm-up         (~5 s)
      • Section 2  — 3×3 ternary plot     (~15 s)
      • Section 3  — 4×4 deterministic    (~10 s, no simulation)
      • Section 4  — 4×4 neutral simplex  (~1 min, N=50k)
      • Section 5  — sim vs. replicator   (~30 s, small N)
    """
    if section is not None:
        if section not in SECTIONS:
            print(f"Unknown section {section}. Choose from {list(SECTIONS.keys())}")
            return
        title, fn = SECTIONS[section]
        print(f"\nRunning section {section}: {title}")
        fn()
        return

    for num, (title, fn) in SECTIONS.items():
        fn()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EGT combined game demo")
    parser.add_argument(
        "--section", type=int, default=None,
        help="Run a single demo section (1-8). Omit to run all.",
    )
    args = parser.parse_args()
    main(section=args.section)