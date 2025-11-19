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
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


plt.style.use(['science'])


#plt.style.use(["science", "no-latex"])


# Comment out if latex is not correctly installed

plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def pd_example(
    pop_size: int = 1000,
    iterations: int = 10000,
    w: float = 0.9,
    initial_dist: list[float] = [0.9, 0.1],
) -> None:

    # Example running prisoners dilemma example.
    N = pop_size

    # Standard prisoners dilemma payoff matrix

    pd_array = Games.PRISONERS_DILEMMA

    initial_dist = np.array([0.9, 0.1])

    iterations = pop_size * 35

    _, _, m_results,_ = simulation.moran_batch_sim(
        pop_size=pop_size,
        iterations=iterations,
        w=w,
        matrix=pd_array,
        simulations=1,
        initial_dist=initial_dist,
        traj=True,
        initial_rand=False
    )
    _, _, l_results,_ = simulation.local_batch_sim(
        pop_size=pop_size,
        iterations=iterations,
        w=w,
        matrix=pd_array,
        simulations=1,
        initial_dist=initial_dist,
        traj=True,
        initial_rand=False
    )

    print("Simulations complete")

    # Collect results into dataframe
    df_PD_MO = pd.DataFrame({"C": m_results[0], "D": m_results[1]})
    df_PD_LU = pd.DataFrame({"C": l_results[0], "D": l_results[1]})

    # Get numerical trajectory for prisoners dilemma.
    test = replicator.pdNumerical(pd_array, w, initial_dist)

    # Get numerical trajectory for adjusted dynamics - moran process
    adjusted, t_eval = replicator.pdNumericalAdjusted()

    # Plot the 4 trajectories on the same graph.
    # This method also normalizes the numerical solutions timeframe.
    simulation.game_2d_plot(
        [df_PD_MO.get("D"), df_PD_LU.get("D"), adjusted.get("D"), test.get("D")],
        N=N,
        labels=["MO", "LO", "Adjusted", "Standard"],
        norm=[True, True, False, False],
        t_eval=t_eval,
    )


def rps_example(N: int = 10000, iterations: int = 1000000) -> None:

    w = 0.5

    # rps_array = np.array([[0, -1.2 , 1], [1, 0, -1.2], [-1.2, 1, 0]])

    rps_array = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])

    _, _, m_results, _ = simulation.moran_batch_sim(
        pop_size=N,
        iterations=iterations,
        w=w,
        matrix=rps_array,
        simulations=1,
        initial_dist=np.array([0.5, 0.25, 0.25]),
        traj=True,
        initial_rand=False
    )
    _, _, l_results,_ = simulation.local_batch_sim(
        pop_size=N,
        iterations=iterations,
        w=w,
        matrix=rps_array,
        simulations=1,
        initial_dist=np.array([0.5, 0.25, 0.25]),
        traj=True,
        initial_rand=False
    )

    df_RPS_MO = pd.DataFrame({"R": m_results[0], "P": m_results[1], "S": m_results[2]})

    df_RPS_LO = pd.DataFrame({"R": l_results[0], "P": l_results[1], "S": l_results[2]})

    # fig = px.line_ternary(df_RPS_MO, a="R", b="P", c="S", title="RPS Moran Process Trajectory", labels={"R":"Rock", "P":"Paper", "S":"Scissors"})
    # fig2 = px.line_ternary(df_RPS_LO, a="R", b="P", c="S", title="RPS LOCAL Process Trajectory", labels={"R":"Rock", "P":"Paper", "S":"Scissors"})

    # fig.show()
    # fig2.show()

    simulation.ternary_plot(df_RPS_MO)
    simulation.ternary_plot(df_RPS_LO)


"""
For popualtion drift test and critical pop_size search, data res is very high so we dont waste cpu on memory accesses - 10 - 20% speedup
As the trajectory is not needed for the final results - only care about accurate delta H values.
Delta H is calculated before the dimension reduction
"""


def run_population_ensemble(
    pop_sizes: list[int],
    w: float = 0.45,
    file_output_path: str = "",
    plot_delta: bool = True,
    simulations: int = 20000000,
    process="MORAN",
    matrix=Games.AUGMENTED_RPS,
) -> None:

    start = time.time()

    # Add the rest of the simulation options as arguments
    drift_SD = []
    drift_rpss = []

    # Simulation for each pop_size.
    for i in tqdm(range(len(pop_sizes)), position=0, leave=True):

        match process:
            case "MORAN":
                drift_H, drift_rps, _, _ = simulation.moran_batch_sim(
                    pop_sizes[i],
                    2,
                    w,
                    simulations,
                    matrix,
                    np.array([0.25, 0.25, 0.25, 0.25]),
                )
                drift_SD.append(drift_H * pop_sizes[i])
                drift_rpss.append(drift_rps * pop_sizes[i])
            case "LOCAL":
                drift_H, drift_rps, _,_ = simulation.local_batch_sim(
                    pop_sizes[i],
                    2,
                    w,
                    simulations,
                    matrix,
                    np.array([0.25, 0.25, 0.25, 0.25]),
                )
                drift_SD.append(drift_H * pop_sizes[i])
                drift_rpss.append(drift_rps * pop_sizes[i])

    end = time.time()

    print("Time taken to run population test ensemble ", process)
    print(end - start)

    # Combine into a single file for csv saving.
    df_deltaResults = pd.DataFrame(
        np.column_stack((pop_sizes, drift_SD, drift_rpss)),
        columns=["pop_sizes", "delta_H", "deltaRps"],
    )

    delta_H_Write(
        df_deltaResults,
        filePath=file_output_path,
        args=[
            "w: 0.45",
            "simulations: 1000000",
            "iterations 2",
            "matrix=Standard rps, 0.2, 0.1",
        ],
        optionalComments="Large average delta H experiment with randomizes starting point in the RPS plane",
    )

    if plot_delta:

        plt.plot(
            df_deltaResults["pop_sizes"],
            df_deltaResults["delta_H"],
            marker="o",
            label="H_4",
        )
        plt.plot(
            df_deltaResults["pop_sizes"],
            df_deltaResults["deltaRps"],
            marker="s",
            label="H_RPS",
        )

        plt.plot(df_deltaResults["pop_sizes"], [0 for i in range(len(pop_sizes))])

        plt.xlabel("N")
        plt.ylabel("delta H")
        plt.legend()
        plt.show()


def search_critical_pop_size(w: float = 0.4, matrix=None, low=0, high=2000) -> int:

    # Binary search for critical pop_size where drift reversal occurs.
    # Hardcoded for initial deltaM being positive.

    # Searches for a sign change with tolerance of 1 in pop_size.
    critical_N = None
    iteration = 0

    max_iterations = 20
    mid = 0

    matrix = matrix

    prev_sign = None

    if matrix is None:
        matrix = Games.AUGMENTED_RPS

    while low <= high and iteration < max_iterations:
        iteration += 1
        mid = (low + high) // 2
        print("Testing pop_size: ", mid)

        delta_moran, _, _, _ = simulation.moran_batch_sim(
            mid, 2, w, 2000000, matrix, np.array([0.25, 0.25, 0.25, 0.25])
        )

        if delta_moran > 0:
            sign = 1
        elif delta_moran < 0:
            sign = -1
        else:
            sign = 0

        # Checks for a sign change
        if prev_sign is not None and (sign * prev_sign) < 0:
            # Sign change

            if high - low > 1:
                if sign > 0:
                    low = mid
                else:
                    high = mid
                continue
            critical_N = mid
            break

        if sign >= 0:
            low = mid + 1
        else:
            high = mid - 1

        prev_sign = sign

    if critical_N is None:
        critical_N = mid  # best estimate
    if critical_N is not None:
        print("Critical pop_size found: ", critical_N)
    else:
        print("Critical pop_size not found in range.")
    return critical_N


# Find critical pop_size for a range over W values - very long run time simulation so needs to write to file output.
def critical_pop_size_ensemble(
    file_output_path: str, option="W_TEST", matrices=None, defaultW=0.45
) -> None:

    Ns = []

    match option:
        case "W_TEST":
            ws = np.linspace(0.1, 0.5, 15)

            for w in tqdm(ws, position=0, leave=True):
                critical_N = search_critical_pop_size(w=w)
                Ns.append(critical_N)

            print("Critical Ns ", Ns)

            df = pd.DataFrame({"W": ws, "critical_N": Ns})

            delta_H_Write(
                df,
                filePath=file_output_path,
                args=[
                    f"W range {ws[0]}",
                    "4000",
                    "iterations 100000",
                    "matrix=Standard rps, 0.2, 0.1",
                ],
                optionalComments="Critical population size search for varied w.",
            )
            return
        case "MATRIX_TEST" | "MATRIX_TEST_G":
            if matrices is None:
                raise TypeError("Missing matrix values for testing")

            betas = []
            for matrix in tqdm(matrices, position=0, leave=True):
                print(matrix)
                if option == "MATRIX_TEST_G":
                    betas.append(matrix[0][3])
                else:
                    betas.append(matrix[3][0])
                critical_N = search_critical_pop_size(w=defaultW, matrix=matrix)
                Ns.append(critical_N)

            df = pd.DataFrame({"beta_gamma": betas, "critical_N": Ns})

            delta_H_Write(
                df, filePath=file_output_path, optionalComments="Matrix ensemble"
            )


# Vary the values of alpha and beta and s(rps param) in the 4x4 game
"""
Two graphs needed;
  - delta H against different values for params
  - critical N for different params
  - can reuse search critical pop_size method here.
"""


def matrix_param_ensemble(
    file_output_path: str,
    betas: list[float],
    gamma: float = 0.8,
    w: int = 0.45,
    pop_size: int = 500,
    simulations: int = 10000000,
    plot_delta: bool = False,
) -> None:

    s = 1

    alpha = 0
    beta = 0.5

    start = time.time()

    # Add the rest of the simulation options as arguments
    drift_SD = []
    drift_rpss = []

    # Simulation for each pop_size.
    for i in tqdm(range(len(betas)), position=0, leave=True):

        matrix = np.array(
            [
                [0, -s, 1, gamma],
                [1, 0, -s, gamma],
                [-s, 1, 0, gamma],
                [betas[i], betas[i], betas[i], 0],
            ]
        )

        drift_H, drift_rps, _, _ = simulation.moran_batch_sim(
            pop_size=pop_size,
            iterations=2,
            w=w,
            simulations=simulations,
            matrix=matrix,
            initial_dist=np.array([0.25, 0.25, 0.25, 0.25]),
        )
        drift_SD.append(drift_H)
        drift_rpss.append(drift_rps)

    end = time.time()

    print("Time taken to run population test ensemble")
    print(end - start)

    # Combine into a single file for csv saving.
    df_deltaResults = pd.DataFrame(
        np.column_stack((betas, drift_SD, drift_rpss)),
        columns=["betas", "delta_H", "deltaRps"],
    )

    delta_H_Write(
        df_deltaResults,
        filePath=file_output_path,
        args=["w: 0.45", "simulations: 1000000", "iterations 2", "matrix=Variable"],
        optionalComments="Delta H experiement with different values for betas\n#"
        + str(betas),
    )

    if plot_delta:

        plt.plot(
            df_deltaResults["betas"], df_deltaResults["delta_H"], marker="o", label="H_4"
        )
        plt.plot(
            df_deltaResults["betas"],
            df_deltaResults["deltaRps"],
            marker="s",
            label="H_RPS",
        )
        plt.xlabel("beta")
        plt.ylabel("delta H")
        plt.legend()
        plt.show()


def arps_example(N: int = 500, iterations: int = 100000) -> None:
    moranResults, local_results, dMoran, dLocal = simulation.runSimulationPool(
        pop_size=N,
        simulations=10,
        iterations=iterations,
        H=3,
        initial_dist=[0.5, 0.25, 0.25, 0],
        w=0.2,
        data_res=50,
    )

    df_RPS_MO = pd.DataFrame(
        {
            "c1": moranResults[0],
            "c2": moranResults[1],
            "c3": moranResults[2],
            "c4": moranResults[3],
        }
    )
    df_RPS_LU = pd.DataFrame(
        {
            "c1": local_results[0],
            "c2": local_results[1],
            "c3": local_results[2],
            "c4": local_results[3],
        }
    )

    write_trajectory(
        df_RPS_MO,
        "./results/moran" + str(N) + "_" + str(iterations) + ".csv",
        args=[N, iterations, 0.2],
        optionalComments="Testing the file writing for trajectories.",
    )

    write_trajectory(
        df_RPS_LU,
        "./results/local" + str(N) + "_" + str(iterations) + ".csv",
        args=[N, iterations, 0.2],
    ),

    simulation.quaternary_plot([df_RPS_MO, df_RPS_LU], labels=["Moran", "Local"])


def write_trajectory(df, filePath, args=[], optionalComments=None):

    # Construct and add comments
    with open(filePath, "w") as f:

        if optionalComments:
            f.write("# " + optionalComments + "\n")

        f.write("# Arguments used\n# args=")
        for arg in args:
            f.write(str(arg) + " ")
        f.write("\n")

    df.to_csv(filePath, mode="a", index=False)


def delta_H_Write(df, filePath, args=[], optionalComments=None):

    with open(filePath, "w") as f:
        f.write("# Drift (delta H) plot results")
        if optionalComments:
            f.write("# " + optionalComments + "\n")

        f.write("# Arguments used\n# args=")
        for arg in args:
            f.write(str(arg) + " ")
        f.write("\n")

    df.to_csv(filePath, mode="a", index=False)


# run ensembles from the command line and output to a file
# Can include the option to plot as a sub argument.
def crit_N_search_parser():
    pass


def delta_H_parser():
    pass


# Helper function to run matrix parameter tests.
def get_four_player_matrices(betas=None, gammas=None, gamma=0.2, beta=0.1):

    matrices = []
    s = 1

    if betas is None and gammas is None:
        raise TypeError("Provide betas or gammas")

    # Produce beta or gamma variations.
    if betas is not None:
        for b in betas:
            matrix = np.array(
                [[0, -s, 1, gamma], [1, 0, -s, gamma], [-s, 1, 0, gamma], [b, b, b, 0]]
            )
            matrices.append(matrix)
    elif gammas is not None:
        for g in gammas:
            matrix = np.array(
                [[0, -s, 1, g], [1, 0, -s, g], [-s, 1, 0, g], [beta, beta, beta, 0]]
            )
            matrices.append(matrix)

    return matrices



def trajectories_to_anim(trajectories):

    sims, n, frames = trajectories.shape
    transposed = trajectories.transpose(0,2,1)

    df = pd.DataFrame(
        transposed.reshape(-1, n),  # shape (sims*frames, n)
        columns=[f"c{i+1}" for i in range(n)],
    )

    df["sim"] = np.repeat(np.arange(sims), frames)
    df["frame"] = np.tile(np.arange(frames), sims)

    return df


def point_cloud_animation(matrix=Games.AUGMENTED_RPS, pop_size=800, iterations=100000, w=0.45, num_points=300 , file_output_path=None):
    
    _, _, _, all_traj = simulation.local_batch_sim(
        pop_size, iterations, w, num_points, point_cloud=True, matrix=matrix
    )


    df = trajectories_to_anim(all_traj)


    if file_output_path is not None:
        df.to_csv(file_output_path, index=False)

    simulation.point_cloud(df, matrix=matrix)




# Need this because of multiprocessing
if __name__ == "__main__":

    logger.info("Running main")

    parser = argparse.ArgumentParser()
    """
  CMD Arguments:
  Game presets:
  standard prisoners dilemma: -pd
  """

    sub_parsers = parser.add_subparsers(dest="preset")

    # 2x2 game
    pd_parser = sub_parsers.add_parser("pd")
    pd_parser.add_argument("-N", type=int, default=1000)
    pd_parser.add_argument("-iterations", type=int, default=1000000)

    # 3x3 game
    rps_parser = sub_parsers.add_parser("rps")
    rps_parser.add_argument("-N", type=int, default=10000)
    rps_parser.add_argument("-iterations", type=int, default=1000000)

    # 4x4 game
    arps_parser = sub_parsers.add_parser("arps")
    arps_parser.add_argument("-N", type=int, default=500)
    arps_parser.add_argument("-iterations", type=int, default=100000)

    # Other options
    # experimentParser = parser.add_sub_parsers(dest="experiment")
    # Add args for critical W finding, and ensembles for population and W.

    args = parser.parse_args()

    if args.preset:
        print("Preset ", args.preset, " has been selected")

    if args.preset:
        if args.preset == "pd":
            # add check for args.N
            print("Running prisoners dilemma preset: [[3,0],[5,1]]")
            pd_example(pop_size=args.N, iterations=args.iterations)
        elif args.preset == "rps":
            print("Running rock paper scissors preset : [0,-1,1],[1,0,-1],[-1,1,0]")
            rps_example(N=args.N, iterations=args.iterations)
        elif args.preset == "arps":
            print("Running augmented rps: " + str(Games.AUGMENTED_RPS))
            arps_example(N=args.N, iterations=args.iterations)

    # delta_moran, deltaRps, m_results,_ = simulation.moran_batch_sim(20000, 1000000, 0.45, 1, Games.AUGMENTED_RPS, np.array([0.5,0.2,0.2,0.1]), traj=True)

    # df_RPS_MO = pd.DataFrame({"c1": m_results[0][::1], "c2": m_results[1][::1], "c3": m_results[2][::1], "c4": m_results[3][::1]})

    # simulation.quaternary_plot([df_RPS_MO], numPerRow=1, labels=["Moran"])

    """
  basic_rps = np.array([[0,   -0.6,   1,       0.5],
                    [1,    0,   -0.6,       0.5],
                    [-0.6,   1,   0,        0.5],
                    [0.25, 0.25, 0.25, 0]])
  """

    """
  
  Cant see the second case where they end along the vertical axis for low pop,
  issue is that rps tends to sprial outwards

  Either we get fixation at central point, or RPS drifts outwards and SD fixes
  or both drift and we end in the bottom corners.
  dont see the case where rps fixes at center and sd doesnt drift as SD has lower crit N for drift to occur than rps.
  (in the games tested.)


  potentially an example of microbes
  RPS in e coli
  invader cheaters taking advantage of public good microbes
  producers make resources that otherss can benefit from,
  but proudcers have to pay a cost to make these.
  rps, 0.1,
  0.1, 0 cheaters get nothign against themselvs
  """

    # Double reversal??????
    # _,_,_, all_traj = simulation.moran_batch_sim(10000, 3000000, 0.45, 300, point_cloud=True)

    # _,_,_, all_traj = simulation.moran_batch_sim(200, 3000000, 0.45, 300, point_cloud=True)

   
    
    basic_rps = np.array(
    [[0, 1, 1,     0.1], 
     [1, 0, 1,     0.1], 
     [1, 1, 0,     0.1], 
     [0.7, 0.7, 0.7, 0]]
    )
    # 5000, 300, 100
    # RPS and SD different critical sizes where the drift occurs.
    # Large population  lots of iterations converges, interior
    # Small pop - SD drifts first end up with the rod
    # 90,000, 10,000,000
    # 1000, 100,000

    
    """run_population_ensemble(range(10,100,5), 
                        file_output_path="./results/rod_example_delta.csv", 
                        plot_delta=True,
                        process="LOCAL",
                        simulations=20000000,
                        w=0.45,
                        matrix=basic_rps
                        )"""
                          
  

    
    """_, _, _, all_traj = simulation.moran_batch_sim(
        50000, 20000000, 0.45, 200, point_cloud=True, matrix=basic_rps
    )"""

    """
    _, _, _, all_traj = simulation.moran_batch_sim(
        1000, 50000, 0.45, 10000, point_cloud=True, matrix=basic_rps
    )"""

    """_, _, _, all_traj = simulation.moran_batch_sim(
        1000, 200000, 0.45, 3000, point_cloud=True, matrix=basic_rps
    )"""

    #point_cloud_animation(pop_size=20000, iterations=15000000, w=0.45, num_points=300, matrix=basic_rps)

    all_traj = np.zeros((1000, 4, 500))
    for i in range(1000):

  
        initial = np.random.exponential(1,4)
        initial /= np.sum(initial)
     
  

        a, t_eval = replicator.numericalTrajectory(interactionProcess="Moran", w=0.45, initial_dist=initial[:3], matrix=basic_rps)
        a = a.to_numpy().T
        all_traj[i, :, :] = a[:, ::10]

    df = trajectories_to_anim(all_traj)

    simulation.point_cloud(df)

    point_cloud_animation(pop_size=800, iterations=100000, w=0.45, num_points=300, matrix=basic_rps)


    

    # Below is testig code - remove at some point
    basic_rps = np.array(
        [[0, -0.3, 1, 0.3], [1, 0, -0.3, 0.3], [-0.3, 1, 0, 0.3], [0.2, 0.2, 0.2, 0]]
    )

    # basic_rps = Games.AUGMENTED_RPS
    """
  delta_moran, deltaRps, m_results,_ = simulation.moran_batch_sim(40000, 6000000, 0.45, 1, basic_rps, np.array([0.5,0.2,0.2,0.1]), traj=True)
  
  df_RPS_MO = pd.DataFrame({"c1": m_results[0][::1], "c2": m_results[1][::1], "c3": m_results[2][::1], "c4": m_results[3][::1]})
  
  print("DELTA RPS ", deltaRps)
  print("DELTA MORAN ", delta_moran)

  write_trajectory(df_RPS_MO, "./results/moranTest.csv")

  test, t_eval = replicator.numericalTrajectory(interactionProcess="Moran", w=0.45)
  write_trajectory(test, "./results/moranNumerical.csv")


  #file_paths = ["./results/moran100000_15000000.csv", "./results/moranNumerical.csv"]
  file_paths = ["./results/moranTest.csv", "./results/moranNumerical.csv"]
  norms = [True, False]

  #simulation.quaternary_plot([df_RPS_MO], numPerRow=1, labels=["Moran"])

  simulation.high_dim_2d_plot(file_paths, [40000, None], norm=norms, t_eval=t_eval, data_res=1)
  
  """
    """
  run_population_ensemble(range(50,400,10), 
                        file_output_path="./results/population_ensemble_MORAN_new_matrix.csv", 
                        plot_delta=True,
                        process="MORAN",
                        matrix=basic_rps,
                        simulations=1000000,
                        w=0.45
                        )
                        
  
  simulation.drift_plot_H(["./results/population_ensemble_MORAN_new_matrix.csv"], labels=[r"$\Delta H_{SD}$", r"$\Delta H_{rps}$"])
  """

    """run_population_ensemble(range(50,400,10), 
                        file_output_path="./results/population_ensemble_MORAN.csv", 
                        plot_delta=True,
                        process="MORAN"
                        )
  

  run_population_ensemble(range(50,400,10), 
                        file_output_path="./results/population_ensemble_LOCAL.csv", 
                        plot_delta=True,
                        process="LOCAL"
                        )"""

    simulation.drift_plot_H(
        ["./results/population_ensemble_MORAN.csv"],
        labels=[r"$\Delta H_{SD}$", r"$\Delta H_{rps}$"],
    )

    simulation.drift_plot_H(
        [
            "./results/population_ensemble_LOCAL.csv",
            "./results/population_ensemble_MORAN.csv",
        ],
        labels=["Local", "Moran"],
        column=0,
    )

    # maybe these functions should return file name - and autogerenrate one if one isnt given.
    # critical_pop_size_ensemble("./results/critical_N_w_2.csv")
    simulation.w_ensemble_plot("./results/critical_N_w_2.csv", log=True)

    """betas = np.linspace(0.1, 1, 10)
  matrices = get_four_player_matrices(betas=betas, gamma=0.5)
  critical_pop_size_ensemble("./results/critical_N_matrix_betas.csv", option="MATRIX_TEST",
                          defaultW=0.45, matrices=matrices)"""

    simulation.w_ensemble_plot(
        "./results/critical_N_matrix_betas.csv", log=True, x_label=r"$\beta$"
    )

    """gammas = np.linspace(0, 0.6, 6)
  matrices = get_four_player_matrices(gammas=gammas, beta=0.2)
  critical_pop_size_ensemble("./results/critical_N_matrix_gammas.csv", option="MATRIX_TEST_G",
                          defaultW=0.45, matrices=matrices)"""

    simulation.w_ensemble_plot(
        "./results/critical_N_matrix_gammas.csv", log=True, x_label=r"$\gamma$"
    )
    """
    matrix_param_ensemble("./results/parameterTest_200.csv", np.linspace(0, 1, 20),pop_size=200,w=0.45, plot_delta=True)

    matrix_param_ensemble("./results/parameterTest_400.csv", np.linspace(0, 1, 20),pop_size=400,w=0.45, plot_delta=True)

    matrix_param_ensemble("./results/parameterTest_600.csv", np.linspace(0, 1, 20),pop_size=600,w=0.45, plot_delta=True)


    matrix_param_ensemble("./results/parameterTest_0.2.csv", np.linspace(0, 1, 20),gamma=0.2,pop_size=200,w=0.45, plot_delta=True)

    matrix_param_ensemble("./results/parameterTest_0.6.csv", np.linspace(0, 1, 20), gamma=0.6,pop_size=200,w=0.45, plot_delta=True)

    matrix_param_ensemble("./results/parameterTest_1.csv", np.linspace(0, 1, 20), gamma=1,pop_size=200,w=0.45, plot_delta=True)

  """

    simulation.drift_plot_H(
        [
            "./results/parameterTest_0.2.csv",
            "./results/parameterTest_0.6.csv",
            "./results/parameterTest_1.csv",
        ],
        xlabel="beta",
        labels=["0.2", "0.6", "1"],
        column=0,
    )

    """
  delta_local, deltaRps, l_results = simulation.local_batch_sim(20000, 3000000, 0.45, 1, Games.AUGMENTED_RPS, traj=True, initial_dist=np.array([0.5,0.2,0.2,0.1]))
  df_RPS_LO = pd.DataFrame({"c1": l_results[0], "c2": l_results[1], "c3": l_results[2], "c4": l_results[3]})
  write_trajectory(df_RPS_LO, "./results/localTest.csv")
  """

    test, t_eval = replicator.numericalTrajectory(interactionProcess="Local", w=0.45)
    write_trajectory(test, "./results/localNumerical.csv")

    file_paths = ["./results/localTest.csv", "./results/localNumerical.csv"]
    norms = [True, False]

    simulation.high_dim_2d_plot(
        file_paths, [20000, None], norm=norms, t_eval=t_eval, data_res=1
    )