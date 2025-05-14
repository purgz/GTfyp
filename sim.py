import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations


basicRps = np.array([[0,   -0.5,   1,       0.1],
                    [1,    0,   -0.5,       0.1],
                    [-0.5,   1,   0,       0.1],
                    [-1, -1, -1, 0]])


basicRps = np.array([[1,   0,   2,       0.1],
                    [2,    1,   0,       0.1],
                    [0,   2,   1,       0.1],
                    [-1, -1, -1, 0]])


popSize = 200
w = 0.6



# Average payoff formula, Paper 1
def payoff(population, w=w):
    payoffs = np.zeros(4)
    for i in range(4):
        payoffs[i] = sum(population[j] * basicRps[i][j] for j in range(4))
    return 1 - w + w * (payoffs / (popSize - 1))



"""
Paper coevolutionary dynamics in large but finite populations

'An individidual of type j is chosen for repoduction with probabiiltiy i_j * Pi_j / (N * phi)
where phi = average payoff.
"""

def moranSelection(payoffs, avg, population, w=w):
    probs = np.zeros(4)
    for i in range(4):

        probs[i] = (population[i] * payoffs[i]) / (popSize * avg)

    return probs


def localUpdate(matrix, N, initialDist = [0.5, 0.25, 0.24, 0.01], iterations = 100000, w=0.6):

    population = np.random.multinomial(popSize, initialDist)

    R = np.zeros(iterations)
    P = np.zeros(iterations)
    S = np.zeros(iterations)
    L = np.zeros(iterations)

    # Maximal payoff difference
    deltaPi = basicRps.max(axis=1).max() - basicRps.min(axis=1).min()

    for i in range(iterations):
        #p1, p2 = np.random.choice([0,1,2,3], size=2, p=population/popSize)
        
        p1 = random.choices([0, 1, 2, 3], weights=population)[0]
        
        p2 = random.choices([0, 1, 2, 3], weights=population)[0]

        p = 1/2 + (w/2) * ((basicRps[p2][p1] - basicRps[p1][p2]) / deltaPi)

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
       

def moranSimulation(matrix, N, initialDist = [0.5, 0.25, 0.24, 0.01], iterations = 100000, w=0.6):
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
        p = payoff(population)
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



moranResult = moranSimulation(basicRps, 4000)

#localResult = localUpdate(basicRps, 3000)

result = moranResult

df_RPS = pd.DataFrame({"c1": result[0], "c2": result[1], "c3": result[2], "c4": result[3]})

print(df_RPS.tail())




numEdgeLabels = 10


def plot_ax():               #plot tetrahedral outline
    verts=[[0,0,0],
     [1,0,0],
     [0.5,np.sqrt(3)/2,0],
     [0.5,0.28867513, 0.81649658]]
    lines=combinations(verts,2)
    for x in lines:
        line=np.transpose(np.array(x))
        ax.plot3D(line[0],line[1],line[2],c='0')

def label_points():  #create labels of each vertices of the simplex
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

def plot_3d_tern(df,c='1', colour="b"): #use function "get_cartesian_array_from_barycentric" to plot the scatter points
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
ax = fig.add_subplot(111, projection="3d")

plot_ax() #call function to draw tetrahedral outline

label_points() #label the vertices


plot_3d_tern(df_RPS,'g') #...


add_edge_labels(ax)
add_grid_lines(ax)

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect([1,1,1])

#ax.view_init(elev=20, azim=120)




plt.show()