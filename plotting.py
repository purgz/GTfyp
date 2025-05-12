import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations
import pandas as pd


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


#Create Dataset 1
np.random.seed(123)
c1=np.random.normal(8,2.5,33)
c2=np.random.normal(8,2.5,33)
c3=np.random.normal(8,2.5,33)
c4=[100-x for x in c1+c2+c3]   #make sur ecomponents sum to 100

#df unecessary but that is the format of my real data
df1=pd.DataFrame(data=[c1,c2,c3,c4],index=['c1','c2','c3','c4']).T
df1=df1/100



#Create Dataset 2
np.random.seed(1234)
c1=np.random.normal(16,2.5,33)
c2=np.random.normal(16,2.5,33)
c3=np.random.normal(16,2.5,33)
c4=[100-x for x in c1+c2+c3]

df2=pd.DataFrame(data=[c1,c2,c3,c4],index=['c1','c2','c3','c4']).T
df2=df2/100



#Create Dataset 3
np.random.seed(12345)
c1=np.random.normal(33,2.5,33)
c2=np.random.normal(33,2.5,33)
c3=np.random.normal(33,2.5,33)
c4=[100-x for x in c1+c2+c3]

df3=pd.DataFrame(data=[c1,c2,c3,c4],index=['c1','c2','c3','c4']).T
df3=df3/100

print(df3.head())

# c4 = L,  c1 = R, c2 = P, c3 = S






fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

plot_ax() #call function to draw tetrahedral outline

label_points() #label the vertices

#plot_3d_tern(df1,'b') #call function to plot df1

#plot_3d_tern(df2,'r') #...plot df2

#plot_3d_tern(df3,'g') #...





#########

def generate_4d_spiral(n_points=100, n_turns=10):
    theta = np.linspace(0, n_turns * np.pi, n_points)  # Control the number of turns
    c4 = np.linspace(0,10, n_points)
    r = np.linspace(0.2, 1.0, n_points)  # Grow to full simplex size

    # Generate positive spiral components
    c1 = 0.4 + 0.3 * np.sin(theta)
    c2 = 0.4 + 0.3 * np.cos(theta)
    c3 = 0.2 + 0.2 * np.sin(2 * theta)

    # Normalize the first 3 components to ensure each point sums to 1
    total = c1 + c2 + c3 + c4
    c1 /= total
    c2 /= total
    c3 /= total
    c4 /= total
    
    # Combine into a 4D dataframe
    return pd.DataFrame({"c1": c1, "c2": c2, "c3": c3, "c4": c4})

# Generate the 4D spiral data
df_spiral = generate_4d_spiral()

# Print the first few rows to verify
print(df_spiral.head())

plot_3d_tern(df_spiral,'g') #...


add_edge_labels(ax)
add_grid_lines(ax)

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.set_box_aspect([1,1,1])

#ax.view_init(elev=20, azim=120)




plt.show()