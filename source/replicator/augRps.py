import sympy as sp
from sympy import latex

from sympy.utilities.lambdify import lambdify
import numpy as np
from scipy.integrate import solve_ivp


import pandas as pd



# This file contains the RPS replicator derivation for the 3x3 standard game

a, c, b, gamma, beta = sp.symbols('a c b gamma beta')



A = sp.Matrix([[a, c, b, gamma], 
               [b, a, c, gamma], 
               [c, b, a, gamma], 
               [a + beta, a + beta, a + beta, 0]])


# NxN matrix - N-1 replicator equations
# x - fraction of rock, y = fraction of paper, z = fraction of scissors
x = sp.symbols('x')
y = sp.symbols('y')
z = sp.symbols('z')
q = 1 - x - y - z

payoffR = a * x + y * c + b * z + gamma * q
payoffP = b * x + a * y + c * z + gamma * q
payoffS = c * x + b * y + a * z + gamma * q 
payoffL = (a + beta) * x + (a + beta) * y + (a + beta) * z

# Pairwise replicator equations 
x_dot = (x * y) * (payoffR - payoffP) + (x * z) * (payoffR - payoffS) + (x * q) * (payoffR - payoffL)
y_dot = (y * x) * (payoffP - payoffR) + (y * z) * (payoffP - payoffS) + (y * q) * (payoffP - payoffL)
z_dot = (z * x) * (payoffS - payoffR) + (z * y) * (payoffS - payoffP) + (z * q) * (payoffS - payoffL)


x_dot_sub = x_dot.subs({a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1})
y_dot_sub = y_dot.subs({a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1})
z_dot_sub = z_dot.subs({a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1})
fixed_points = sp.solve([x_dot_sub, y_dot_sub, z_dot_sub], (x, y, z), dict=True)
print("Fixed points for augmented RPS game: ", fixed_points)



"""
NUMERICAL integration below - reformat this into methods.
Also reformat the rest of this into methods that can be called independently.
Add name = main so that you can run this file directly too but not have it call whenever a method is called from here.
"""
t = sp.symbols("t")
f = lambdify((t, x , y , z), [x_dot_sub, y_dot_sub, z_dot_sub], modules="numpy")

def replicatorSystem(t, vars):
    x,y,z = vars
    dxdt,dydt,dzdt = f(t,x,y,z)
    return [dxdt, dydt, dzdt]


x0 = [0.7,0.1, 0.1]
t_span=(0,1000)
t_eval=np.linspace(*t_span, 50000)

sol = solve_ivp(replicatorSystem, t_span, x0, t_eval=t_eval)


x_vals = sol.y[0]
y_vals = sol.y[1]
z_vals = sol.y[2]
w_vals = 1 - x_vals - y_vals - z_vals

df_RPS_MO = pd.DataFrame({
    "c1": x_vals,
    "c2": y_vals,
    "c3": z_vals,
    "c4": w_vals
})

def testNumericalIntegration():
    return df_RPS_MO


F_x = sp.diff(x_dot, x)
F_y = sp.diff(x_dot, y)
F_z = sp.diff(x_dot, z)

G_x = sp.diff(y_dot, x)
G_y = sp.diff(y_dot, y)
G_z = sp.diff(y_dot, z)

P_x = sp.diff(z_dot, x)
P_y = sp.diff(z_dot, y)
P_z = sp.diff(z_dot, z)

# 3x3 Jacobian Matrix
J = sp.Matrix([[F_x, F_y, F_z],
               [G_x, G_y, G_z],
               [P_x, P_y, P_z]])



eigenvalues = J.eigenvals()



standardConfig = {a: 0, b: 1, c: -1, gamma: 0.2, beta: 0.1}



eigenvalues_sub = {eig.subs(standardConfig) for eig in eigenvalues}


results = {eig.subs({x: 2/9, y: 2/9, z: 2/9}) for eig in eigenvalues_sub}

print(latex(results))

#print(latex(eigenvalues_sub))


