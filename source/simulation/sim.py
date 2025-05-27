


import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations

from multiprocessing import Pool




"""
Notes:

Local update doesnt seem to be working as expected, continue to work on



Moran process seems to be working - correct dynamics for RPS I think.


Add some averaging for each simulation run, modulise the plotting code.



Todo :

Need to extract graph plotting code into a separate module

Maybe also remove the simulation methods and code so theyre isolated from the simulation running code - easier to navigate

Create method to process resutlts of each sim

"""

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


def localUpdate(matrix, N, initialDist = [0.25, 0.25, 0.25, 0.25], iterations = 100000, w=0.3):

    population = np.random.multinomial(popSize, initialDist)

    R = np.zeros(iterations)
    P = np.zeros(iterations)
    S = np.zeros(iterations)
    L = np.zeros(iterations)


    for i in range(iterations):
        # Using this with no replacement fixed my drift issue !!!
        p1, p2 = np.random.choice([0,1,2,3], size=2, p=population/popSize, replace=False)
        
        payoffs = payoffAgainstPop(population)
        #deltaPi = np.max(payoffs) - np.min(payoffs)
        deltaPi = 2

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
       

def moranSimulation(matrix, N, initialDist = [0.25, 0.25, 0.25, 0.25], iterations = 100000, w=0.3):
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



#popSize = 50


"""print("Moran update process sim....")
moranResult = moranSimulation(basicRps, 4000)

df_RPS_MO = pd.DataFrame({"c1": moranResult[0], "c2": moranResult[1], "c3": moranResult[2], "c4": moranResult[3]})
print(df_RPS_MO.tail())

print("Local update process sim....")
localResult = localUpdate(basicRps, 3000)
df_RPS_LU = pd.DataFrame({"c1": localResult[0], "c2": localResult[1], "c3": localResult[2], "c4": localResult[3]})
print(df_RPS_LU.tail())
"""


#delta_L_moran = np.mean(np.diff(moranResult[3]))
#delta_L_local = np.mean(np.diff(localResult[3]))

#print("Moran: ", delta_L_moran, " | ", " Local Update: ", delta_L_local)


# I think im starting to see drift reversal now.
# W = 0.2, popsize = 100, iterations = 100000
# Equal mix strategies, when N = 1000 - positive drift to central equilibrium
# When n = 100, negative drift.

# However for some reason there is drifti n the local update process, even though it should be neutral.

popSize = 100
simulations = 100
deltaMoran = []
deltaLocal = []

mResults = []
lResults = []


def singleSim(_):
    moranResult = moranSimulation(basicRps, 100, iterations = 100000)
    localResult = localUpdate(basicRps, 100, iterations = 100000)
    delta_L_moran = np.mean(np.diff(moranResult[3]))
    delta_L_local = np.mean(np.diff(localResult[3]))

    return moranResult, localResult, delta_L_moran, delta_L_local

# Method for api to call
def runSimulationPool():
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

"""
Todo

- Remove plotting code from sim.py

- improve readability of this file

- maybe separate multiprocess and simulation running code from interation processes.

- add command line arguments,

- add custom matrix parameter values.

- add 2d plots (ternary) to plotting

"""


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




    # Resolution for markers on quaternary plot.
    numEdgeLabels = 10


    def plot_ax(ax):               #plot tetrahedral outline
        verts=[[0,0,0],
        [1,0,0],
        [0.5,np.sqrt(3)/2,0],
        [0.5,0.28867513, 0.81649658]]
        lines=combinations(verts,2)
        for x in lines:
            line=np.transpose(np.array(x))
            ax.plot3D(line[0],line[1],line[2],c='0')

    def label_points(ax):  #create labels of each vertices of the simplex
        a=(np.array([1,0,0,0])) # Barycentric coordinates of vertices (A or c1)
        b=(np.array([0,1,0,0])) # Barycentric coordinates of vertices (B or c2)
        c=(np.array([0,0,1,0])) # Barycentric coordinates of vertices (C or c3)
        d=(np.array([0,0,0,1])) # Barycentric coordinates of vertices (D or c3)
        labels=['R','P','S','L']
        cartesian_points=get_cartesian_array_from_barycentric([a,b,c,d])
        for point,label in zip(cartesian_points,labels):
            if 'a' in label:
                ax.text(point[0],point[1]-0.075,point[2], label, size=16)
            elif 'b' in label:
                ax.text(point[0]+0.02,point[1]-0.02,point[2], label, size=16)
            else:
                ax.text(point[0],point[1],point[2], label, size=16)

    def get_cartesian_array_from_barycentric(b):      #tranform from "barycentric" composition space to cartesian coordinates
        verts=[[0,0,0],
            [1,0,0],
            [0.5,np.sqrt(3)/2,0],
            [0.5,0.28867513, 0.81649658]]

        #create transformation array vis https://en.wikipedia.org/wiki/Barycentric_coordinate_system
        t = np.transpose(np.array(verts))        
        t_array=np.array([t.dot(x) for x in b]) #apply transform to all points

        return t_array

    def plot_3d_tern(df, ax , colour="b"): #use function "get_cartesian_array_from_barycentric" to plot the scatter points
    #args are b=dataframe to plot and c=scatter point color
        bary_arr=df.values

        verts = np.array([[0,0,0], [1,0,0], [0.5,np.sqrt(3)/2,0], [0.5,0.28867513, 0.81649658]])

        cartesian_points=get_cartesian_array_from_barycentric(bary_arr)
        #ax.scatter(cartesian_points[:,0],cartesian_points[:,1],cartesian_points[:,2],c=c)

        ax.plot(cartesian_points[:,0], cartesian_points[:,1], cartesian_points[:,2], 
                color=colour, linewidth=1.2, alpha=0.7)


    def add_edge_labels(ax):  # Add ratio labels along each edge
        verts = np.array([[0,0,0], [1,0,0], [0.5,np.sqrt(3)/2,0], [0.5,0.28867513, 0.81649658]])


        edges = [
            (verts[0], verts[1]),  # A-B
            (verts[2], verts[0]),  # C-A
            (verts[0], verts[3]),  # A-D
            (verts[1], verts[2]),  # B-C
            (verts[1], verts[3]),  # B-D
            (verts[2], verts[3])   # C-D
        ]
        
        ticks = np.linspace(0, 1, numEdgeLabels)[1:-1]  # 0.0 to 1.0 in 5 steps
        tick_labels = [f"{t:.1f}" for t in ticks]

        for start, end in edges:

            for t, label in zip(ticks, tick_labels):
                # Linear interpolation for tick position
            
                pos = (1-t) * start + t * end
                ax.text(pos[0], pos[1], pos[2] + 0.02, 
                        label, size=10, ha='center', color='gray', weight='bold')

    def add_grid_lines(ax):  # Add ternary-style grid lines to ABC face
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0.5, np.sqrt(3)/2, 0])
        
        ticks = np.linspace(0, 1, numEdgeLabels)[1:-1]  # Exclude 0 and 1 to avoid drawing edges twice

        # Lines parallel to AB
        for t in ticks:
            p1 = (1-t) * c + t * b
            p2 = (1-t) * c + t * a
            line = np.array([p1, p2]).T
            ax.plot3D(line[0], line[1], line[2], color='lightgray', linewidth=0.8, alpha=0.8)

        # Lines parallel to BC
        for t in ticks:
            p1 = (1-t) * a + t * c
            p2 = (1-t) * a + t * b
            line = np.array([p1, p2]).T
            ax.plot3D(line[0], line[1], line[2], color='lightgray', linewidth=0.8, alpha=0.8)

        # Lines parallel to AC
        for t in ticks:
            p1 = (1-t) * b + t * c
            p2 = (1-t) * b + t * a
            line = np.array([p1, p2]).T
            ax.plot3D(line[0], line[1], line[2], color='lightgray', linewidth=0.8, alpha=0.8)




    fig = plt.figure()
    ax = fig.add_subplot(121, projection="3d")
    plot_ax(ax) #call function to draw tetrahedral outline
    label_points(ax) #label the vertices
    plot_3d_tern(df_RPS_LU, ax,'g') #...
    add_edge_labels(ax)
    add_grid_lines(ax)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect([1,1,1])
    ax.set_title("Local update process", pad=20)
    ax.plot([], [], color="green", label="Trajectory (LU)")  #
    ax.legend(loc="upper right", fontsize=10)

    ax2 = fig.add_subplot(122, projection="3d")
    plot_ax(ax2) #call function to draw tetrahedral outline
    label_points(ax2) #label the vertices
    plot_3d_tern(df_RPS_MO, ax2,'b') #...
    add_edge_labels(ax2)
    add_grid_lines(ax2)
    ax2.grid(False)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_zticks([])
    ax2.set_box_aspect([1,1,1])
    ax2.set_title("Moran process", pad = 20)
    ax2.plot([], [], color="b", label="Trajectory (MO)")  #
    ax2.legend(loc="upper right", fontsize=10)


    #ax.view_init(elev=20, azim=120)




    plt.show()
